from __future__ import annotations

import argparse
import gc
import sys
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from conditions import choose_dtype, get_resolution, load_yaml, seed_everything
from train_paired import WarmupFlowModel, build_optimizer, configure_runtime, load_components


def _maybe_prompt_shapes(cfg: dict[str, Any]) -> tuple[int, int, int]:
    cache = Path(cfg['data']['root']) / cfg['data']['cache_dir'] / 'text' / 'prompt.npz'
    if cache.exists():
        import numpy as np
        with np.load(cache, mmap_mode='r') as row:
            prompt = row['prompt_embeds']
            pooled = row['pooled_prompt_embeds']
            return int(prompt.shape[0]), int(prompt.shape[1]), int(pooled.shape[-1])
    return 512, 4096, 768


def make_dummy_batch(cfg: dict[str, Any], batch_size: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    width, height = get_resolution(cfg['data']['resolution'])
    tokens = (height // 16) * (width // 16)
    text_tokens, text_dim, pooled_dim = _maybe_prompt_shapes(cfg)
    gen = torch.Generator(device=device).manual_seed(int(cfg['experiment'].get('seed', 0)) + 17)
    return {
        'target_latents': torch.randn(batch_size, tokens, 64, device=device, dtype=dtype, generator=gen),
        'pose_latents': torch.randn(batch_size, tokens, 64, device=device, dtype=dtype, generator=gen),
        'prompt_embeds': torch.randn(batch_size, text_tokens, text_dim, device=device, dtype=dtype, generator=gen),
        'pooled_prompt_embeds': torch.randn(batch_size, pooled_dim, device=device, dtype=dtype, generator=gen),
        'pulid_id_embed': torch.randn(batch_size, 32, 2048, device=device, dtype=dtype, generator=gen),
        'appearance': torch.randn(batch_size, 1024, device=device, dtype=dtype, generator=gen),
        'garment': torch.randn(batch_size, int(cfg['model']['identity_adapter'].get('garment_grid_max_tokens', 64)), 1024, device=device, dtype=dtype, generator=gen),
        'head_pose': torch.randn(batch_size, 7, device=device, dtype=dtype, generator=gen),
        'head_pose_is_null': torch.zeros(batch_size, device=device, dtype=torch.float32),
    }


def run_case(cfg: dict[str, Any], rank: int, forwards: int, micro_batch: int, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))
    cfg['model']['lora_rank'] = int(rank)
    cfg['training']['micro_batch'] = int(micro_batch)
    cfg['training']['preferred_micro_batch'] = int(micro_batch)
    configure_runtime(cfg, world_size=1, all_gpus_train=True)
    cfg['_runtime']['grad_accum'] = 1
    stage = 'load_components'
    started = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    transformer = controlnet = vae = adapter = pulid = None
    try:
        transformer, controlnet, vae, adapter, pulid, notes = load_components(cfg, device, dtype)
        model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg).to(device=device, dtype=dtype).train()
        optimizer = build_optimizer(model, cfg)
        batch = make_dummy_batch(cfg, micro_batch, device, dtype)
        stage = f'{forwards}x_forward'
        losses = []
        for _ in range(int(forwards)):
            loss, _metrics = model(batch)
            losses.append(loss)
        total = torch.stack(losses).mean()
        stage = 'backward'
        total.backward()
        stage = 'optimizer_step'
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        elapsed = time.time() - started
        return {
            'rank': rank,
            'forwards': forwards,
            'micro_batch': micro_batch,
            'status': 'ok',
            'failed_stage': '',
            'peak_gib': peak,
            'elapsed_sec': elapsed,
            'load_notes': notes,
        }
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return {
            'rank': rank,
            'forwards': forwards,
            'micro_batch': micro_batch,
            'status': 'oom',
            'failed_stage': stage,
            'peak_gib': peak,
            'elapsed_sec': time.time() - started,
            'error': str(exc).split('\n')[0],
        }
    finally:
        del transformer, controlnet, vae, adapter, pulid
        gc.collect()
        torch.cuda.empty_cache()


def write_report(path: Path, cfg: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    width, height = get_resolution(cfg['data']['resolution'])
    lines = [
        f'# Phase 1 VRAM Report {width}x{height}',
        '',
        f'- resolution: `{width}x{height}`',
        f'- token_count: `{(height // 16) * (width // 16)}`',
        '- model: FLUX.1-dev + InstantX Union ControlNet + frozen PuLID-FLUX + transformer LoRA',
        '- measurement: `torch.cuda.max_memory_allocated()` after forward/backward/optimizer step',
        '',
        '| rank | forwards | micro_batch | status | failed_stage | peak GiB | 48G feasible | elapsed sec |',
        '|---:|---:|---:|---|---|---:|---|---:|',
    ]
    feasible_rows = []
    for row in rows:
        peak = row.get('peak_gib', 0.0)
        feasible = row['status'] == 'ok' and peak <= 44.0
        if feasible:
            feasible_rows.append(row)
        lines.append(
            f"| {row['rank']} | {row['forwards']} | {row['micro_batch']} | {row['status']} | {row.get('failed_stage','')} | {peak:.2f} | {'yes' if feasible else 'no'} | {row.get('elapsed_sec',0.0):.1f} |"
        )
    lines.append('')
    if feasible_rows:
        best = feasible_rows[0]
        lines.append(f"Conclusion: 48G feasible with rank={best['rank']}, forwards={best['forwards']}, micro_batch={best['micro_batch']} (peak {best['peak_gib']:.2f} GiB).")
    else:
        lines.append('Conclusion: no tested configuration fit the 44 GiB safety target; reduce rank, share differential computation, reduce decode frequency, LoRA ControlNet, or train at proportional lower resolution.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Native-resolution Phase 1 training VRAM stress test.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--forwards', type=int, default=3)
    parser.add_argument('--micro-batch', type=int, default=1)
    parser.add_argument('--fallback-rank', type=int, default=8)
    parser.add_argument('--report', default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg['experiment'].get('seed', 0)))
    device = torch.device(args.device)
    dtype = choose_dtype(cfg['model']['precision'])
    width, height = get_resolution(cfg['data']['resolution'])
    report = Path(args.report) if args.report else Path(cfg['data']['root']) / 'phase1' / f'vram_report_{width}x{height}.md'
    rows = []
    rows.append(run_case(cfg, args.rank, args.forwards, args.micro_batch, device, dtype))
    if not (rows[-1]['status'] == 'ok' and rows[-1]['peak_gib'] <= 44.0) and args.fallback_rank and args.fallback_rank != args.rank:
        rows.append(run_case(cfg, args.fallback_rank, args.forwards, args.micro_batch, device, dtype))
    write_report(report, cfg, rows)
    print(report)
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
