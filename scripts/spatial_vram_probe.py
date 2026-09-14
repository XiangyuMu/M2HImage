from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader

from conditions import choose_dtype, get_resolution, load_yaml, seed_everything
from dataset import PairedWarmupDataset
from train_paired import (
    HairSupervisedWarmupFlowModel,
    build_optimizer,
    configure_runtime,
    load_components,
)


def run_case(
    base_cfg: dict[str, Any],
    *,
    hair_stride: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    cfg['model']['spatial_conditions']['garment_reference']['stride'] = 1
    cfg['model']['spatial_conditions']['hair']['reference']['stride'] = int(
        hair_stride
    )
    cfg['model']['lora_rank'] = int(rank)
    cfg['model']['gradient_checkpointing']['enabled'] = True
    cfg['model']['gradient_checkpointing']['transformer_block_ratio'] = 1.0
    configure_runtime(cfg, world_size=3, all_gpus_train=False)
    cfg['_runtime']['grad_accum'] = 1
    transformer = controlnet = vae = adapter = pulid = model = optimizer = None
    stage = 'load'
    try:
        transformer, controlnet, vae, adapter, pulid, notes = load_components(cfg, device, dtype)
        model = HairSupervisedWarmupFlowModel(
            transformer, controlnet, adapter, pulid, vae, cfg
        ).train()
        optimizer = build_optimizer(model, cfg)
        dataset_cfg = json.loads(json.dumps(cfg))
        cache_samples = (
            Path(cfg['data']['root']) / cfg['data']['cache_dir'] / 'samples'
        )
        available = sorted(cache_samples.glob('*.npz'))
        if not available:
            raise RuntimeError('spatial VRAM probe needs at least one cache sample')
        with np.load(available[0], allow_pickle=False) as cached:
            has_hair_latent = 'hair_ref_latents' in cached.files
        if not has_hair_latent:
            dataset_cfg['model']['spatial_conditions']['hair']['reference'][
                'enabled'
            ] = False
        dataset = PairedWarmupDataset(
            dataset_cfg, 'train', require_coverage=False
        )
        dataset.ids = [sample_id for sample_id in dataset.ids if dataset.sample_path(sample_id).exists()]
        if not dataset.ids:
            raise RuntimeError('spatial VRAM probe needs at least one cache sample')
        batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
        if not has_hair_latent:
            # Shape-only pre-cache probe: a garment latent is a valid packed tensor with the
            # same memory/layout cost as the future hair latent. No metric claim uses it.
            batch['hair_ref_latents'] = batch['garment_ref_latents'].clone()
            batch['hair_ref_empty'] = torch.zeros(1, dtype=torch.float32)
        batch['tau_override'] = torch.full((1,), 0.5, dtype=torch.float32)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        stage = 'forward'
        loss, metrics = model(
            batch,
            decode_trigger=True,
            hair_loss_accum_scale=1.0,
        )
        stage = 'backward'
        loss.backward()
        stage = 'optimizer'
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        return {
            'hair_stride': int(hair_stride),
            'rank': int(rank),
            'image_tokens': int(model.image_token_count),
            'garment_reference_tokens': int(
                float(metrics['garment_reference_tokens'])
            ),
            'hair_reference_tokens': int(
                float(metrics['hair_reference_tokens'])
            ),
            'sequence_tokens': int(
                model.image_token_count
                + float(metrics['garment_reference_tokens'])
                + float(metrics['hair_reference_tokens'])
            ),
            'synthetic_hair_reference': not has_hair_latent,
            'status': 'ok',
            'failed_stage': '',
            'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
            'step_sec': time.perf_counter() - started,
            'loss': float(loss.detach().float().cpu()),
            'sample_id': dataset.ids[0],
            'load_notes': notes,
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower()
        if not is_oom:
            raise
        return {
            'hair_stride': int(hair_stride),
            'rank': int(rank),
            'status': 'oom',
            'failed_stage': stage,
            'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
            'step_sec': 0.0,
            'error': str(exc).split('\n')[0],
        }
    finally:
        del optimizer, model, transformer, controlnet, vae, adapter, pulid
        gc.collect()
        torch.cuda.empty_cache()


def write_report(path: Path, cfg: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    width, height = get_resolution(cfg['data']['resolution'])
    lines = [
        f'# Spatial Conditioning VRAM Report {width}x{height}',
        '',
        '- measured path: paired flow, one frozen pose ControlNet, FLUX over [image | garment-reference | hair-reference], PuLID, differentiable VAE/DINO hair loss, and joint backward/optimizer',
        '- garment reference remains full resolution; only the hair reference may use stride 2',
        '- ControlNet runs on image tokens only; its residual is zero-padded over both reference segments',
        '- safety gate: torch.cuda.max_memory_allocated() <= 44 GiB on A6000 48G',
        '- fallback order: hair stride 1 -> hair stride 2 -> full checkpointing -> rank 8',
        '',
        '| hair stride | rank | image tokens | garment ref | hair ref | total tokens | status | failed stage | peak GiB | step sec |',
        '|---:|---:|---:|---:|---:|---:|---|---|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['hair_stride']} | {row['rank']} | {row.get('image_tokens', 3072)} | "
            f"{row.get('garment_reference_tokens', 3072)} | "
            f"{row.get('hair_reference_tokens', 0)} | {row.get('sequence_tokens', 0)} | "
            f"{row['status']} | {row.get('failed_stage', '')} | {row.get('peak_gib', 0.0):.2f} | "
            f"{row.get('step_sec', 0.0):.2f} |"
        )
    feasible = [row for row in rows if row['status'] == 'ok' and row['peak_gib'] <= 44.0]
    lines.append('')
    if feasible:
        selected = feasible[0]
        grad_accum = 6
        estimate = selected['step_sec'] * grad_accum * int(cfg['training']['total_steps']) / 3600
        lines.extend([
            f"Conclusion: use garment stride=1, hair stride={selected['hair_stride']}, LoRA rank={selected['rank']}; "
            f"peak={selected['peak_gib']:.2f} GiB.",
            f'Conservative 3-GPU 4400-step estimate: {estimate:.2f} h '
            f'(probe step x grad_accum={grad_accum}; data loading/watcher excluded).',
        ])
    else:
        lines.append('Conclusion: BLOCKED; no tested row is under the 44 GiB safety gate.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Probe the complete spatial paired-flow training step.')
    parser.add_argument('--config', default='configs/spatial_warmup.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--report', default=None)
    parser.add_argument(
        '--hair-strides',
        type=int,
        nargs='+',
        default=None,
        help='Explicit hair-reference strides to measure; defaults to the configured stride.',
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg['experiment']['seed']))
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = choose_dtype(cfg['model']['precision'])
    width, height = get_resolution(cfg['data']['resolution'])
    report = Path(args.report) if args.report else (
        Path(cfg['data']['root']) / 'phase1' / f'vram_report_spatial_{width}x{height}.md'
    )
    strides = args.hair_strides or [
        int(cfg['model']['spatial_conditions']['hair']['reference']['stride'])
    ]
    rows = [
        run_case(cfg, hair_stride=stride, rank=16, device=device, dtype=dtype)
        for stride in dict.fromkeys(strides)
    ]
    if not any(row['status'] == 'ok' and row['peak_gib'] <= 44.0 for row in rows):
        rows.append(run_case(cfg, hair_stride=2, rank=8, device=device, dtype=dtype))
    write_report(report, cfg, rows)
    print(report)
    for row in rows:
        print(json.dumps({key: value for key, value in row.items() if key != 'load_notes'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
