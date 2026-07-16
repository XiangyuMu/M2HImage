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
from torch.utils.data import DataLoader

from conditions import choose_dtype, get_resolution, load_yaml, seed_everything
from dataset import PairedWarmupDataset
from train_paired import (
    DirectedDifferentialFlowModel,
    build_optimizer,
    configure_runtime,
    load_checkpoint,
    load_components,
    load_directed_identity_components,
)


def run_case(
    base_cfg: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    latent_scale: float,
    decode_freq: int,
    label: str,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    cfg['training']['differential']['hinge_g_resolved'] = float(
        cfg['training']['differential'].get('smoke_hinge_g', 1.0)
    )
    cfg['training']['differential']['decode']['latent_scale'] = float(latent_scale)
    cfg['training']['differential']['decode']['freq'] = int(decode_freq)
    cfg['model']['gradient_checkpointing']['enabled'] = True
    cfg['model']['gradient_checkpointing']['transformer_block_ratio'] = 1.0
    configure_runtime(cfg, world_size=3, all_gpus_train=False)
    cfg['_runtime']['grad_accum'] = 1
    stage = 'load_components'
    transformer = controlnet = vae = adapter = pulid = recognizer = detector = model = optimizer = None
    torch.cuda.empty_cache()
    try:
        transformer, controlnet, vae, adapter, pulid, notes = load_components(cfg, device, dtype)
        recognizer, detector, identity_notes = load_directed_identity_components(cfg, device)
        notes.update(identity_notes)
        model = DirectedDifferentialFlowModel(
            transformer,
            controlnet,
            adapter,
            pulid,
            vae,
            recognizer,
            detector,
            cfg,
        )
        optimizer = build_optimizer(model, cfg)
        load_checkpoint(Path(cfg['training']['resume']), model, optimizer=optimizer)
        dataset = PairedWarmupDataset(cfg, 'train', require_coverage=True)
        batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
        batch['tau_override'] = torch.full((1,), 0.5, dtype=torch.float32)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        stage = 'forward'
        loss, metrics = model(
            batch,
            train_step=0,
            decode_trigger=True,
            identity_loss_accum_scale=1.0,
        )
        if float(metrics['id_loss_attempt_count']) < 1.0:
            raise RuntimeError('A4 probe did not enter the decode identity branch')
        if float(metrics['id_loss_skip_count']) > 0.0:
            raise RuntimeError('A4 probe face detector skipped the decoded branch; choose a healthy cached sample')
        stage = 'backward'
        loss.backward()
        stage = 'optimizer_step'
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        return {
            'label': label,
            'status': 'ok',
            'failed_stage': '',
            'latent_scale': latent_scale,
            'decode_freq': decode_freq,
            'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
            'step_sec': time.perf_counter() - started,
            'loss_total': float(loss.detach().float().cpu()),
            'loss_id_dir': float(metrics['loss_id_dir']),
            'loss_id_abs': float(metrics['loss_id_abs']),
            'sim_gap': float(metrics['sim_gap']),
            'id_decode_seconds': float(metrics['id_decode_seconds']),
            'sample_id': list(batch['sample_id']),
            'load_notes': notes,
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower()
        if not is_oom and 'probe' not in str(exc).lower() and 'detector skipped' not in str(exc).lower():
            raise
        return {
            'label': label,
            'status': 'oom' if is_oom else 'invalid',
            'failed_stage': stage,
            'latent_scale': latent_scale,
            'decode_freq': decode_freq,
            'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
            'step_sec': 0.0,
            'error': str(exc).split('\n')[0],
        }
    finally:
        del optimizer, model, recognizer, detector, transformer, controlnet, vae, adapter, pulid
        gc.collect()
        torch.cuda.empty_cache()


def write_report(path: Path, cfg: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    width, height = get_resolution(cfg['data']['resolution'])
    lines = [
        f'# A4 Directed Identity VRAM Report {width}x{height}',
        '',
        f'- packed tokens: {(height // 16) * (width // 16)}',
        '- measured path: 1x frozen ControlNet + 3x FLUX transformer + 1x in-graph VAE decode + no-grad SCRFD geometry + frozen Glint360K ArcFace + joint backward + optimizer step',
        '- measurement: `torch.cuda.max_memory_allocated()`; A6000 safety threshold is <=44 GiB',
        '- `decode_freq` changes average throughput, not the peak of a triggered decode step.',
        '',
        '| case | latent scale | decode freq | status | failed stage | peak GiB | step sec | sim_gap |',
        '|---|---:|---:|---|---|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['latent_scale']:.2f} | {row['decode_freq']} | {row['status']} | "
            f"{row.get('failed_stage', '')} | {row.get('peak_gib', 0.0):.2f} | "
            f"{row.get('step_sec', 0.0):.2f} | {row.get('sim_gap', 0.0):.4f} |"
        )
    feasible = [row for row in rows if row['status'] == 'ok' and row['peak_gib'] <= 44.0]
    lines.append('')
    if feasible:
        selected = feasible[0]
        lines.append(
            f"Conclusion: use latent_scale={selected['latent_scale']:.2f}, decode_freq={selected['decode_freq']}; "
            f"triggered-step peak={selected['peak_gib']:.2f} GiB."
        )
    else:
        lines.append('Conclusion: BLOCKED; no tested A4 triggered step met the 44 GiB safety threshold.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Probe the complete A4 directed-identity training step.')
    parser.add_argument('--config', default='configs/a4_directed.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--report', default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg['experiment']['seed']))
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = choose_dtype(cfg['model']['precision'])
    width, height = get_resolution(cfg['data']['resolution'])
    report = Path(args.report) if args.report else (
        Path(cfg['data']['root']) / 'phase1' / f'vram_report_a4_{width}x{height}.md'
    )
    rows = [run_case(cfg, device, dtype, 1.0, 3, 'full-resolution decode')]
    if not (rows[0]['status'] == 'ok' and rows[0]['peak_gib'] <= 44.0):
        rows.append(run_case(cfg, device, dtype, 0.5, 3, 'half-resolution decode'))
    if not any(row['status'] == 'ok' and row['peak_gib'] <= 44.0 for row in rows):
        rows.append(run_case(cfg, device, dtype, 0.5, 4, 'half decode, lower frequency'))
    write_report(report, cfg, rows)
    print(report)
    for row in rows:
        print(json.dumps({key: value for key, value in row.items() if key != 'load_notes'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
