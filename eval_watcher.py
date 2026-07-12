from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from conditions import (
    arcface_embedding_from_path, choose_dtype, find_one, get_resolution, load_yaml, seed_everything, unpack_latents,
)
from dataset import PairedWarmupDataset
from train_paired import WarmupFlowModel, load_checkpoint, load_components


def decode_tokens(vae, tokens: torch.Tensor, resolution) -> Image.Image:
    latents = unpack_latents(tokens, resolution)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0][0]
    arr = ((image.float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype('uint8')
    return Image.fromarray(arr)


def generate(model: WarmupFlowModel, batch: dict, steps: int, seed: int, device, dtype) -> torch.Tensor:
    z = torch.randn(1, (model.height // 16) * (model.width // 16), 64, device=device, dtype=dtype, generator=torch.Generator(device=device).manual_seed(seed))
    # Deterministic Euler solver for the trained flow field: integrate from tau=1 to tau=0.
    for i in range(steps):
        tau = torch.full((1,), 1.0 - i / steps, device=device, dtype=dtype)
        model_timestep = tau
        local = batch
        prompt = local['prompt_embeds'].to(device=device, dtype=dtype).unsqueeze(0) if local['prompt_embeds'].ndim == 2 else local['prompt_embeds'].to(device=device, dtype=dtype)
        pooled = local['pooled_prompt_embeds'].to(device=device, dtype=dtype).unsqueeze(0) if local['pooled_prompt_embeds'].ndim == 1 else local['pooled_prompt_embeds'].to(device=device, dtype=dtype)
        cond_tokens = model._condition_tokens(
            prompt,
            local['appearance'].to(device=device, dtype=dtype).unsqueeze(0),
            local['garment'].to(device=device, dtype=dtype).unsqueeze(0),
            local['head_pose'].to(device=device, dtype=dtype).unsqueeze(0),
        )
        from conditions import make_image_ids
        img_ids = make_image_ids(model.width, model.height, device, dtype)
        with torch.no_grad():
            cn_samples = model._controlnet_forward(
                z,
                model_timestep,
                prompt,
                pooled,
                local['pose_latents'].to(device=device, dtype=dtype).unsqueeze(0),
                img_ids,
            )
            v = model._transformer_forward(
                z,
                model_timestep,
                cond_tokens,
                local['pulid_id_embed'].to(device=device, dtype=dtype).unsqueeze(0),
                cn_samples,
                pooled=pooled,
                img_ids=img_ids,
            )
        z = z - (1.0 / steps) * v
    return z


def make_panel(root: Path, sample_id: str, generated: Image.Image, resolution, swap: Image.Image | None = None, swap_id: str | None = None) -> Image.Image:
    width, height = get_resolution(resolution)
    parts = [
        Image.open(find_one(root / 'images/mannequin', sample_id)).convert('RGB').resize((width, height)),
        Image.open(find_one(root / 'dwpose/without_head/mannequin', sample_id)).convert('RGB').resize((width, height)),
        generated.resize((width, height)),
    ]
    labels = ['m_i', 'pose', 'generated c_i']
    if swap is not None:
        parts.append(swap.resize((width, height)))
        labels.append(f'swap c_{swap_id}')
    parts.append(Image.open(find_one(root / 'images/human', sample_id)).convert('RGB').resize((width, height)))
    labels.append('h_i')
    canvas = Image.new('RGB', (width * len(parts), height + 24), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, image in enumerate(parts):
        canvas.paste(image, (i * width, 24))
        draw.text((i * width + 8, 6), labels[i], fill=(0, 0, 0))
    return canvas


def swap_identity(batch: dict, donor: dict) -> dict:
    out = dict(batch)
    out['pulid_id_embed'] = donor['pulid_id_embed']
    out['appearance'] = donor['appearance']
    return out


def embedding_for_image(path: Path, cfg: dict, device_index: int) -> np.ndarray | None:
    try:
        return arcface_embedding_from_path(
            path,
            helper_python=cfg['cache'].get('arcface_helper_python'),
            helper_script=cfg['cache'].get('arcface_helper_script'),
            model_root=cfg['cache'].get('arcface_model_root', '/data/muxiangyu/modelLibrary/insightface'),
            device_id=device_index,
        )
    except Exception:
        return None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))



def gate_history(path: Path, gate_init: float) -> tuple[dict | None, dict[str, dict[str, float]]]:
    keys = ('appearance_gate', 'garment_gate', 'head_pose_gate')
    stats = {
        key: {'min': gate_init, 'max': gate_init, 'max_abs_deviation': 0.0}
        for key in keys
    }
    if not path.exists():
        return None, stats
    last = None
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not all(key in row for key in keys):
                continue
            last = row
            for key in keys:
                value = float(row[key])
                stats[key]['min'] = min(stats[key]['min'], value)
                stats[key]['max'] = max(stats[key]['max'], value)
                stats[key]['max_abs_deviation'] = max(
                    stats[key]['max_abs_deviation'],
                    abs(value - gate_init),
                )
    return last, stats


def differential_collapse_risk(path: Path, cfg: dict) -> tuple[bool, dict]:
    differential = cfg.get('training', {}).get('differential', {})
    if not differential.get('enabled', False):
        return False, {'status': 'disabled'}
    window = int(differential.get('collapse_window_steps', 500))
    drop_threshold = float(differential.get('collapse_drop_fraction', 0.30))
    calibration_steps = int(differential.get('calibration_steps', 200))
    rows: list[tuple[int, float]] = []
    if path.exists():
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    run_step = int(row['run_step'])
                    value = float(row['face_diff_norm'])
                    active = float(row.get('diff_active_ratio', 0.0))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                if run_step > calibration_steps and active > 0.0 and np.isfinite(value) and value > 0.0:
                    rows.append((run_step, value))
    if not rows:
        return False, {'status': 'insufficient', 'reason': 'no active differential log rows'}
    max_step = max(step for step, _ in rows)

    def mean_between(start: int, end: int) -> float | None:
        values = [value for step, value in rows if start < step <= end]
        return float(np.mean(values)) if values else None

    baseline = mean_between(calibration_steps, calibration_steps + window)
    latest = mean_between(max_step - window, max_step)
    previous = mean_between(max_step - 2 * window, max_step - window)
    details = {
        'status': 'ok',
        'window_steps': window,
        'drop_threshold': drop_threshold,
        'baseline_mean': baseline,
        'previous_mean': previous,
        'latest_mean': latest,
        'latest_run_step': max_step,
    }
    if baseline is None or previous is None or latest is None:
        details['status'] = 'insufficient'
        return False, details
    risk = (
        latest < baseline * (1.0 - drop_threshold)
        and previous < baseline * (1.0 - 0.8 * drop_threshold)
        and latest <= previous * 1.05
    )
    details['risk'] = risk
    details['drop_fraction'] = 1.0 - latest / max(baseline, 1e-8)
    return risk, details


def run_once(config_path: str, ckpt: Path, device: str) -> bool:
    cfg = load_yaml(config_path)
    seed_everything(int(cfg['experiment']['seed']))
    torch_device = torch.device(device)
    device_index = torch_device.index if torch_device.type == 'cuda' and torch_device.index is not None else 0
    dtype = choose_dtype(cfg['model']['precision'])
    transformer, controlnet, vae, adapter, pulid, _ = load_components(cfg, torch_device, dtype)
    if vae is None:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(cfg['model']['base'], subfolder='vae', torch_dtype=dtype, local_files_only=True).to(torch_device)
    model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg)
    load_checkpoint(ckpt, model)
    model.eval()
    ds = PairedWarmupDataset(cfg, 'val', require_coverage=True)
    ids = ds.ids[: int(cfg['eval']['fixed_val_count'])]
    swap_count = min(int(cfg['eval'].get('identity_swap_count', 0)), len(ids))
    experiment_dir = Path(cfg['data']['root']) / 'phase1' / cfg['experiment']['id']
    out = experiment_dir / 'warmup_vis' / ckpt.name
    out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg['data']['root'])
    generated_paths: list[Path] = []
    swap_rows = []
    for i, sid in enumerate(ids):
        batch = ds[ds.ids.index(sid)]
        tokens = generate(model, batch, int(cfg['eval']['generate_steps']), seed=1000 + i, device=torch_device, dtype=dtype)
        image = decode_tokens(vae, tokens, cfg['data']['resolution'])
        gen_path = out / f'{sid}_generated.png'
        image.save(gen_path)
        generated_paths.append(gen_path)
        swap_image = None
        swap_id = None
        swap_path = None
        if i < swap_count:
            swap_id = ids[(i + 1) % len(ids)]
            donor = ds[ds.ids.index(swap_id)]
            swap_batch = swap_identity(batch, donor)
            swap_tokens = generate(model, swap_batch, int(cfg['eval']['generate_steps']), seed=2000 + i, device=torch_device, dtype=dtype)
            swap_image = decode_tokens(vae, swap_tokens, cfg['data']['resolution'])
            swap_path = out / f'{sid}_swap_{swap_id}.png'
            swap_image.save(swap_path)
            generated_paths.append(swap_path)
        make_panel(root, sid, image, cfg['data']['resolution'], swap=swap_image, swap_id=swap_id).save(out / f'{sid}.png')
        if swap_path is not None:
            swap_rows.append({'sample_id': sid, 'swap_id': swap_id, 'paired_path': gen_path, 'swap_path': swap_path})

    embeddings: dict[Path, np.ndarray | None] = {path: embedding_for_image(path, cfg, device_index) for path in generated_paths}
    detected = sum(emb is not None for emb in embeddings.values())
    face_rate = detected / max(1, len(generated_paths))
    swap_results = []
    adapter_not_responding = False
    for row in swap_rows:
        a = embeddings.get(row['paired_path'])
        b = embeddings.get(row['swap_path'])
        cos = cosine(a, b) if a is not None and b is not None else None
        if cos is not None and cos >= 0.85:
            adapter_not_responding = True
        swap_results.append({**row, 'cosine': cos})

    warnings = []
    stop_training = False
    if face_rate < 0.95:
        warnings.append('⚠ STOP-TRAINING: face detection rate below 95%')
        stop_training = True
    if adapter_not_responding:
        warnings.append('⚠ STOP-TRAINING: adapter not responding; swap ArcFace cosine >= 0.85')
        stop_training = True
    gate_init = float(cfg['model'].get('identity_adapter', {}).get('gate_init', 0.1))
    gate_move_threshold = float(cfg['eval'].get('gate_move_threshold', 1e-3))
    gate_row, gate_stats = gate_history(experiment_dir / 'logs' / 'train.jsonl', gate_init)
    gates_moved = gate_row and all(
        gate_stats[key]['max_abs_deviation'] > gate_move_threshold
        for key in ('appearance_gate', 'garment_gate', 'head_pose_gate')
    )
    if not gates_moved:
        warnings.append(
            f'⚠ STOP-TRAINING: condition gates have not moved from init={gate_init} '
            f'by more than {gate_move_threshold}'
        )
        stop_training = True
    collapse_risk, collapse_details = differential_collapse_risk(
        experiment_dir / 'logs' / 'train.jsonl', cfg
    )
    if collapse_risk:
        warnings.insert(
            0,
            '⚠ COLLAPSE-RISK: face-region differential norm declined by more than 30% '
            'across sustained 500-step windows',
        )
    report = ['# Warmup Watcher Report', '']
    report.extend(warnings or ['status: automatic checks passed thresholds'])
    report.extend(['', f'checkpoint: `{ckpt}`', '', '## Automatic Checks', ''])
    report.append(f'face detection rate: {detected}/{len(generated_paths)} = {face_rate:.2%}')
    report.append(f'gate latest row: {gate_row if gate_row else "N/A"}')
    report.append(f'gate history movement: {gate_stats}')
    report.append(f'differential collapse monitor: {collapse_details}')
    report.append('')
    report.append('| sample | swap_id | ArcFace cos(generated c_i, swap c_j) | status |')
    report.append('|---|---|---:|---|')
    for row in swap_results:
        cos = row['cosine']
        status = 'N/A detection failed' if cos is None else ('adapter not responding' if cos >= 0.85 else 'responding')
        report.append(f"| {row['sample_id']} | {row['swap_id']} | {'N/A' if cos is None else f'{cos:.4f}'} | {status} |")
    report.extend(['', '## Manual Checklist', '', 'Inspect plastic feel, garment fidelity, pose following, and identity swap response in the five-column panels.'])
    (out / 'watcher_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    if stop_training:
        marker = experiment_dir / 'STOP_TRAINING'
        payload = {
            'checkpoint': str(ckpt),
            'warnings': warnings,
            'face_detection_rate': face_rate,
            'gate_row': gate_row,
            'gate_history_movement': gate_stats,
            'swap_results': [
                {
                    'sample_id': row['sample_id'],
                    'swap_id': row['swap_id'],
                    'cosine': row['cosine'],
                }
                for row in swap_results
            ],
        }
        tmp = marker.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(marker)
    return stop_training


def main() -> None:
    parser = argparse.ArgumentParser(description='GPU3 checkpoint watcher for Phase 1 warmup.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--ckpt-dir', required=False)
    parser.add_argument('--ckpt', default=None, help='Run one specific checkpoint directory and exit.')
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.ckpt:
        run_once(args.config, Path(args.ckpt), args.device)
        return
    if not args.ckpt_dir:
        raise SystemExit('--ckpt-dir is required unless --ckpt is provided')
    seen = set()
    while True:
        ready = sorted(
            Path(args.ckpt_dir).glob('*/READY'),
            key=lambda marker: (marker.parent.name == 'final', marker.parent.name),
        )
        for marker in ready:
            ckpt = marker.parent
            if str(ckpt) in seen:
                continue
            run_once(args.config, ckpt, args.device)
            seen.add(str(ckpt))
            if ckpt.name == 'final':
                return
        if args.once:
            break
        time.sleep(load_yaml(args.config)['eval']['watcher_poll_seconds'])


if __name__ == '__main__':
    main()
