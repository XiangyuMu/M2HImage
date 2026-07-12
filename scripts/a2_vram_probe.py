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

import numpy as np
import torch

from conditions import choose_dtype, get_resolution, load_yaml, seed_everything
from train_paired import (
    DifferentialFlowModel,
    build_optimizer,
    configure_runtime,
    load_checkpoint,
    load_components,
)


def prompt_shapes(cfg: dict[str, Any]) -> tuple[int, int, int]:
    cache = Path(cfg['data']['root']) / cfg['data']['cache_dir'] / 'text' / 'prompt.npz'
    with np.load(cache, mmap_mode='r') as row:
        prompt = row['prompt_embeds']
        pooled = row['pooled_prompt_embeds']
        return int(prompt.shape[0]), int(prompt.shape[1]), int(pooled.shape[-1])


def dummy_batch(cfg: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    width, height = get_resolution(cfg['data']['resolution'])
    tokens = (height // 16) * (width // 16)
    text_tokens, text_dim, pooled_dim = prompt_shapes(cfg)
    generator = torch.Generator(device=device).manual_seed(int(cfg['experiment']['seed']) + 71)
    ones = torch.ones(1, tokens, device=device, dtype=torch.float32)
    return {
        'target_latents': torch.randn(1, tokens, 64, device=device, dtype=dtype, generator=generator),
        'pose_latents': torch.randn(1, tokens, 64, device=device, dtype=dtype, generator=generator),
        'prompt_embeds': torch.randn(1, text_tokens, text_dim, device=device, dtype=dtype, generator=generator),
        'pooled_prompt_embeds': torch.randn(1, pooled_dim, device=device, dtype=dtype, generator=generator),
        'pulid_id_embed': torch.randn(1, 32, 2048, device=device, dtype=dtype, generator=generator),
        'appearance': torch.randn(1, 1024, device=device, dtype=dtype, generator=generator),
        'garment': torch.randn(1, 64, 1024, device=device, dtype=dtype, generator=generator),
        'head_pose': torch.randn(1, 7, device=device, dtype=dtype, generator=generator),
        'head_pose_is_null': torch.zeros(1, device=device),
        'cf_j_pulid_id_embed': torch.randn(1, 32, 2048, device=device, dtype=dtype, generator=generator),
        'cf_k_pulid_id_embed': torch.randn(1, 32, 2048, device=device, dtype=dtype, generator=generator),
        'cf_j_appearance': torch.randn(1, 1024, device=device, dtype=dtype, generator=generator),
        'cf_k_appearance': torch.randn(1, 1024, device=device, dtype=dtype, generator=generator),
        'delta_arc_jk': torch.full((1,), 0.7, device=device),
        'cloth_safe_z': ones,
        'body_bg_z': ones,
        'face_z': ones,
        'tau_override': torch.full((1,), 0.5, device=device),
    }


def run_case(
    base_cfg: dict[str, Any],
    rank: int,
    diff_every: int,
    train_step: int,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    cfg['model']['lora_rank'] = rank
    cfg['model']['gradient_checkpointing']['enabled'] = True
    cfg['model']['gradient_checkpointing']['transformer_block_ratio'] = 1.0
    cfg['training']['differential']['enabled'] = True
    cfg['training']['differential']['diff_every'] = diff_every
    cfg['training']['differential']['hinge_g_resolved'] = float(
        cfg['training']['differential'].get('smoke_hinge_g', 1.0)
    )
    configure_runtime(cfg, world_size=1, all_gpus_train=True)
    cfg['_runtime']['grad_accum'] = 1
    stage = 'load_components'
    transformer = controlnet = vae = adapter = pulid = model = optimizer = None
    setup_started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        transformer, controlnet, vae, adapter, pulid, notes = load_components(cfg, device, dtype)
        model = DifferentialFlowModel(transformer, controlnet, adapter, pulid, cfg).train()
        optimizer = build_optimizer(model, cfg)
        resume = Path(cfg['training']['resume'])
        loaded_resume = resume.exists() and rank == int(base_cfg['model']['lora_rank'])
        if loaded_resume:
            load_checkpoint(resume, model, optimizer=optimizer)
        batch = dummy_batch(cfg, device, dtype)
        torch.cuda.synchronize(device)
        setup_sec = time.perf_counter() - setup_started
        torch.cuda.reset_peak_memory_stats(device)
        stage = 'forward'
        step_started = time.perf_counter()
        loss, metrics = model(batch, train_step=train_step)
        stage = 'backward'
        loss.backward()
        stage = 'optimizer_step'
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        return {
            'label': label,
            'rank': rank,
            'diff_every': diff_every,
            'train_step': train_step,
            'status': 'ok',
            'failed_stage': '',
            'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
            'step_sec': time.perf_counter() - step_started,
            'setup_sec': setup_sec,
            'controlnet_forwards': float(metrics['controlnet_forward_count']),
            'transformer_forwards': float(metrics['transformer_forward_count']),
            'loss_total': float(loss.detach().float().cpu()),
            'resume_state_loaded': loaded_resume,
            'load_notes': notes,
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if not isinstance(exc, torch.cuda.OutOfMemoryError) and 'out of memory' not in str(exc).lower():
            raise
        torch.cuda.synchronize(device)
        return {
            'label': label,
            'rank': rank,
            'diff_every': diff_every,
            'train_step': train_step,
            'status': 'oom',
            'failed_stage': stage,
            'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
            'step_sec': 0.0,
            'setup_sec': time.perf_counter() - setup_started,
            'error': str(exc).split('\n')[0],
        }
    finally:
        del optimizer, model, transformer, controlnet, vae, adapter, pulid
        gc.collect()
        torch.cuda.empty_cache()


def write_report(path: Path, cfg: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    width, height = get_resolution(cfg['data']['resolution'])
    lines = [
        f'# A2 Differential VRAM Report {width}x{height}',
        '',
        f'- packed tokens: {(height // 16) * (width // 16)}',
        '- full differential micro-step: one frozen ControlNet forward + paired/CF-j/CF-k transformer forwards + joint backward',
        '- measurement: torch.cuda.max_memory_allocated(); feasible safety target is <=44 GiB on A6000 48G',
        '- diff_every=2 reduces average compute but cannot reduce the peak of a differential micro-step; both full and skipped steps are reported.',
        '- rank-8 fallback rows measure memory with a fresh rank-8 LoRA; if selected, the rank-16 B2-prime LoRA must be projected once and shared by both A2/B2-cont.',
        '',
        '| case | rank | diff_every | CN forwards | transformer forwards | status | failed stage | peak GiB | step sec |',
        '|---|---:|---:|---:|---:|---|---|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['rank']} | {row['diff_every']} | "
            f"{row.get('controlnet_forwards', 0):.0f} | {row.get('transformer_forwards', 0):.0f} | "
            f"{row['status']} | {row.get('failed_stage', '')} | {row.get('peak_gib', 0.0):.2f} | "
            f"{row.get('step_sec', 0.0):.2f} |"
        )
    full_rows = [
        row for row in rows
        if row['status'] == 'ok' and row.get('transformer_forwards') == 3.0 and row['peak_gib'] <= 44.0
    ]
    lines.append('')
    if not full_rows:
        lines.append('Conclusion: no tested full differential step meets the 44 GiB safety target.')
    else:
        selected = full_rows[0]
        pair = next(
            (
                row for row in rows
                if row['status'] == 'ok'
                and row['rank'] == selected['rank']
                and row.get('transformer_forwards') == 1.0
            ),
            None,
        )
        diff_fraction = 0.6 / max(1, selected['diff_every'])
        pair_sec = pair['step_sec'] if pair else selected['step_sec'] / 3.0
        weighted_micro_sec = diff_fraction * selected['step_sec'] + (1.0 - diff_fraction) * pair_sec
        grad_accum = int(cfg.get('_runtime', {}).get('grad_accum', 6))
        estimated_hours = weighted_micro_sec * grad_accum * 4000 / 3600
        lines.extend([
            f"Conclusion: use rank={selected['rank']}, diff_every={selected['diff_every']}; "
            f"full-step peak={selected['peak_gib']:.2f} GiB.",
            f'Conservative 4000-step estimate: {estimated_hours:.2f} h '
            f'(tau hit 60%, grad_accum={grad_accum}, probe includes an optimizer step per micro-step).',
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='A2 full differential training-step VRAM probe.')
    parser.add_argument('--config', default='configs/a2_diff.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--report', default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    configure_runtime(cfg, world_size=3, all_gpus_train=False)
    seed_everything(int(cfg['experiment']['seed']))
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = choose_dtype(cfg['model']['precision'])
    width, height = get_resolution(cfg['data']['resolution'])
    report = (
        Path(args.report)
        if args.report
        else Path(cfg['data']['root']) / 'phase1' / f'vram_report_diff_{width}x{height}.md'
    )
    rows = [
        run_case(cfg, rank=16, diff_every=1, train_step=0, label='rank16 full diff', device=device, dtype=dtype),
        run_case(cfg, rank=16, diff_every=2, train_step=1, label='rank16 diff_every2 skipped', device=device, dtype=dtype),
    ]
    if not (rows[0]['status'] == 'ok' and rows[0]['peak_gib'] <= 44.0):
        rows.append(
            run_case(cfg, rank=16, diff_every=2, train_step=0, label='rank16 diff_every2 full', device=device, dtype=dtype)
        )
        rows.append(
            run_case(cfg, rank=8, diff_every=2, train_step=0, label='rank8 diff_every2 full', device=device, dtype=dtype)
        )
        rows.append(
            run_case(cfg, rank=8, diff_every=2, train_step=1, label='rank8 diff_every2 skipped', device=device, dtype=dtype)
        )
    write_report(report, cfg, rows)
    print(report)
    for row in rows:
        print(json.dumps({key: value for key, value in row.items() if key != 'load_notes'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
