from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from conditions import (
    arcface_embedding_from_path, choose_dtype, find_one, get_resolution, load_yaml, seed_everything, unpack_latents,
)
from dataset import PairedWarmupDataset
from manual_review_gate import review_status, write_approval
from probe_response_track import (
    compute_response_snapshot,
    finalize_snapshot,
    run_watcher_metrics,
)
from spatial_conditions import face_hair_appearance_crop, load_fashn_labels
from train_paired import WarmupFlowModel, load_checkpoint, load_components
from watcher_protocol import watcher_eval_set_from_config


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
        hair_inputs = model._hair_inputs(local, device, dtype)
        cond_tokens = model._condition_tokens(
            prompt,
            local['appearance'].to(device=device, dtype=dtype).unsqueeze(0),
            local['garment'].to(device=device, dtype=dtype).unsqueeze(0),
            local['head_pose'].to(device=device, dtype=dtype).unsqueeze(0),
            *hair_inputs,
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
                garment_ref_latents=local.get('garment_ref_latents'),
                hair_ref_latents=local.get('hair_ref_latents'),
            )
        z = z - (1.0 / steps) * v
    return z


def make_panel(
    root: Path,
    sample_id: str,
    generated: Image.Image,
    resolution,
    identity_variants: list[tuple[str, Image.Image]] | None = None,
    pose_folder: str = 'dwpose/without_head/mannequin',
) -> Image.Image:
    width, height = get_resolution(resolution)
    parts = [
        Image.open(find_one(root / 'images/mannequin', sample_id)).convert('RGB').resize((width, height)),
        Image.open(find_one(root / pose_folder, sample_id)).convert('RGB').resize((width, height)),
        generated.resize((width, height)),
    ]
    labels = ['m_i', 'pose', 'generated c_i']
    for label, variant in identity_variants or []:
        parts.append(variant.resize((width, height)))
        labels.append(label)
    parts.append(Image.open(find_one(root / 'images/human', sample_id)).convert('RGB').resize((width, height)))
    labels.append('h_i')
    canvas = Image.new('RGB', (width * len(parts), height + 24), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, image in enumerate(parts):
        canvas.paste(image, (i * width, 24))
        draw.text((i * width + 8, 6), labels[i], fill=(0, 0, 0))
    return canvas


def _overlay_hair_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    image = image.convert('RGB')
    if mask.shape != (image.height, image.width):
        mask = np.asarray(
            Image.fromarray((mask > 0).astype(np.uint8) * 255, mode='L').resize(
                image.size, Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        ) > 0
    else:
        mask = mask > 0
    array = np.asarray(image, dtype=np.float32).copy()
    tint = np.zeros_like(array)
    tint[..., 0] = 255.0
    array[mask] = 0.65 * array[mask] + 0.35 * tint[mask]
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode='RGB')


def make_hair_review_panel(
    root: Path,
    sample_id: str,
    generated: Image.Image,
    resolution,
) -> Image.Image:
    width, height = get_resolution(resolution)
    labels = load_fashn_labels(root, sample_id)
    hair_mask = labels == 2
    reference_crop, _ = face_hair_appearance_crop(root, sample_id)
    human = Image.open(find_one(root / 'images/human', sample_id)).convert('RGB')
    mannequin = Image.open(
        find_one(root / 'images/mannequin', sample_id)
    ).convert('RGB')
    parts = [
        mannequin.resize((width, height), Image.Resampling.BICUBIC),
        reference_crop.resize((width, height), Image.Resampling.BICUBIC),
        _overlay_hair_mask(generated.resize((width, height)), hair_mask),
        _overlay_hair_mask(human.resize((width, height)), hair_mask),
    ]
    labels_text = [
        'mannequin',
        'reference head crop',
        'generated + target hair mask',
        'h_i + hair mask',
    ]
    canvas = Image.new('RGB', (width * 4, height + 24), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, part in enumerate(parts):
        canvas.paste(part, (index * width, 24))
        draw.text((index * width + 8, 6), labels_text[index], fill=(0, 0, 0))
    return canvas


def visible_hair_val_ids(
    dataset: PairedWarmupDataset,
    root: Path,
    count: int,
    min_fraction: float,
) -> list[str]:
    rows = []
    for sample_id in dataset.ids:
        labels = load_fashn_labels(root, sample_id)
        fraction = float((labels == 2).mean())
        if fraction >= min_fraction:
            rows.append((sample_id, fraction))
    rows.sort(key=lambda row: (-row[1], row[0]))
    if len(rows) < count:
        raise RuntimeError(
            f'watcher requires {count} visible-hair val samples at area >= '
            f'{min_fraction:.2%}, found {len(rows)}'
        )
    return [sample_id for sample_id, _ in rows[:count]]


def swap_identity(batch: dict, donor: dict) -> dict:
    out = dict(batch)
    out['pulid_id_embed'] = donor['pulid_id_embed']
    out['appearance'] = donor['appearance']
    for key in ('hair_ref_tokens', 'hair_ref_positions', 'hair_ref_mask'):
        if key in donor:
            out[key] = donor[key]
    for key in ('hair_semantic_tokens', 'hair_semantic_mask'):
        if key in donor:
            out[key] = donor[key]
    if 'hair_ref_latents' in donor:
        out['hair_ref_latents'] = donor['hair_ref_latents']
        out['hair_ref_empty'] = donor.get('hair_ref_empty', torch.tensor(0.0))
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



def gate_history(path: Path, gate_init: float, keys: tuple[str, ...]) -> tuple[dict | None, dict[str, dict[str, float]]]:
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


def directed_identity_history(path: Path, output_plot: Path) -> dict:
    rows = []
    if path.exists():
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    attempts = float(row.get('id_loss_attempt_count', 0.0))
                    triggered = float(row.get('id_loss_triggered', 0.0))
                    sim_gap = float(row.get('sim_gap', 0.0))
                    skips = float(row.get('id_loss_skip_count', 0.0))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if triggered > 0.0 or attempts > 0.0:
                    rows.append({
                        'step': int(row.get('step', 0)),
                        'sim_gap': sim_gap,
                        'attempts': attempts,
                        'skips': skips,
                        'loss_id_dir': float(row.get('loss_id_dir', 0.0)),
                        'loss_id_abs': float(row.get('loss_id_abs', 0.0)),
                    })
    attempts = sum(row['attempts'] for row in rows)
    skips = sum(row['skips'] for row in rows)
    summary = {
        'status': 'insufficient' if not rows else 'ok',
        'triggered_log_rows': len(rows),
        'attempts': attempts,
        'skips': skips,
        'skip_rate': skips / attempts if attempts > 0.0 else None,
        'sim_gap_latest': rows[-1]['sim_gap'] if rows else None,
        'sim_gap_mean': float(np.mean([row['sim_gap'] for row in rows])) if rows else None,
        'sim_gap_recent_slope': (
            float(np.polyfit(
                np.asarray([row['step'] for row in rows[-10:]], dtype=np.float64),
                np.asarray([row['sim_gap'] for row in rows[-10:]], dtype=np.float64),
                1,
            )[0] * 500.0)
            if len(rows) >= 2
            else None
        ),
        'loss_id_dir_latest': rows[-1]['loss_id_dir'] if rows else None,
        'loss_id_abs_latest': rows[-1]['loss_id_abs'] if rows else None,
    }
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        output_plot.parent.mkdir(parents=True, exist_ok=True)
        steps = [row['step'] for row in rows]
        gaps = [row['sim_gap'] for row in rows]
        cumulative_skip = []
        running_attempts = 0.0
        running_skips = 0.0
        for row in rows:
            running_attempts += row['attempts']
            running_skips += row['skips']
            cumulative_skip.append(running_skips / max(running_attempts, 1.0))
        figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(steps, gaps, linewidth=1.5)
        axes[0].set_ylabel('sim_gap')
        axes[0].grid(alpha=0.25)
        axes[1].plot(steps, cumulative_skip, linewidth=1.5, color='tab:red')
        axes[1].axhline(0.5, linestyle='--', linewidth=1.0, color='black')
        axes[1].set_ylabel('cumulative skip rate')
        axes[1].set_xlabel('optimizer step')
        axes[1].grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_plot, dpi=150)
        plt.close(figure)
        summary['plot'] = str(output_plot)
    return summary


def hair_loss_history(path: Path) -> dict:
    count_fields = (
        'hair_loss_attempt_count',
        'hair_loss_skip_count',
        'hair_schedule_sample_count',
        'hair_schedule_hit_count',
        'hair_skip_decode_freq_count',
        'hair_skip_tau_window_count',
        'hair_skip_area_count',
        'hair_skip_target_invalid_count',
        'hair_skip_parsing_failure_count',
    )
    totals = {field: 0.0 for field in count_fields}
    latest: dict | None = None
    if path.exists():
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not any(field in row for field in count_fields):
                    continue
                latest = row
                for field in count_fields:
                    try:
                        totals[field] += float(row.get(field, 0.0))
                    except (TypeError, ValueError):
                        pass
    attempts = totals['hair_loss_attempt_count']
    schedule = totals['hair_schedule_sample_count']
    result = {
        'status': 'ok' if latest is not None else 'insufficient',
        **totals,
        'hair_loss_skip_rate': (
            totals['hair_loss_skip_count'] / attempts if attempts > 0.0 else None
        ),
        'schedule_hit_rate': (
            totals['hair_schedule_hit_count'] / schedule if schedule > 0.0 else None
        ),
        'reason_rates': {
            'decode_freq': (
                totals['hair_skip_decode_freq_count'] / schedule
                if schedule > 0.0 else None
            ),
            'tau_window': (
                totals['hair_skip_tau_window_count'] / schedule
                if schedule > 0.0 else None
            ),
            'area': (
                totals['hair_skip_area_count'] / attempts
                if attempts > 0.0 else None
            ),
            'target_invalid': (
                totals['hair_skip_target_invalid_count'] / attempts
                if attempts > 0.0 else None
            ),
            'parsing_failure': (
                totals['hair_skip_parsing_failure_count'] / attempts
                if attempts > 0.0 else None
            ),
        },
    }
    if latest is not None:
        result['latest_step'] = int(latest.get('step', 0))
        result['hair_cosine_rolling'] = latest.get('hair_cosine_rolling')
        result['hair_skip_rate_rolling'] = latest.get('hair_skip_rate_rolling')
    return result


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
    checkpoint_step = load_checkpoint(ckpt, model)
    completed_run_steps = checkpoint_step - int(model.run_origin_step or 0)
    model.eval()
    ds = PairedWarmupDataset(cfg, 'val', require_coverage=True)
    root = Path(cfg['data']['root'])
    track_cfg = cfg.get('eval', {}).get('response_track', {})
    fixed_protocol = watcher_eval_set_from_config(cfg)
    ids = list(fixed_protocol['sample_ids'])
    missing_ids = sorted(set(ids) - set(ds.ids))
    if missing_ids:
        raise RuntimeError(
            f'frozen watcher ids are absent from the val dataset: {missing_ids}'
        )
    spatial_cfg = cfg.get('model', {}).get('spatial_conditions', {})
    head_control_enabled = bool(spatial_cfg.get('head_control', {}).get('enabled', False))
    spatial_hair_enabled = bool(spatial_cfg.get('hair', {}).get('enabled', False))
    pose_folder = 'dwpose/with_head/mannequin' if head_control_enabled else 'dwpose/without_head/mannequin'
    swap_count = min(int(cfg['eval'].get('identity_swap_count', 0)), len(ids))
    differential_cfg = cfg.get('training', {}).get('differential', {})
    directed_enabled = bool(
        differential_cfg.get('enabled', False)
        and differential_cfg.get('identity_loss', {}).get('enabled', False)
    )
    experiment_dir = Path(cfg['data']['root']) / 'phase1' / cfg['experiment']['id']
    out = experiment_dir / 'warmup_vis' / ckpt.name
    out.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []
    trend_rows: list[dict] = []
    swap_rows = []
    for i, sid in enumerate(ids):
        batch = ds[ds.ids.index(sid)]
        tokens = generate(model, batch, int(cfg['eval']['generate_steps']), seed=1000 + i, device=torch_device, dtype=dtype)
        image = decode_tokens(vae, tokens, cfg['data']['resolution'])
        gen_path = out / f'{sid}_generated.png'
        image.save(gen_path)
        generated_paths.append(gen_path)
        from eval_b2 import garment_type

        trend_rows.append({
            'mid': sid,
            'jid': sid,
            'seed': 1000 + i,
            'garment_type': garment_type(root, sid),
            'path': gen_path,
        })
        variants: list[tuple[str, Image.Image]] = []
        variant_paths: list[tuple[str, Path]] = []
        if i < swap_count:
            donor_offsets = (1, 2) if directed_enabled else (1,)
            for variant_index, offset in enumerate(donor_offsets):
                donor_id = ids[(i + offset) % len(ids)]
                donor = ds[ds.ids.index(donor_id)]
                swap_batch = swap_identity(batch, donor)
                swap_tokens = generate(
                    model,
                    swap_batch,
                    int(cfg['eval']['generate_steps']),
                    seed=1000 + i,
                    device=torch_device,
                    dtype=dtype,
                )
                swap_image = decode_tokens(vae, swap_tokens, cfg['data']['resolution'])
                role = ('j', 'k')[variant_index] if directed_enabled else 'swap'
                swap_path = out / f'{sid}_{role}_{donor_id}.png'
                swap_image.save(swap_path)
                generated_paths.append(swap_path)
                if spatial_hair_enabled:
                    donor_image = Image.open(find_one(root / 'images/human', donor_id)).convert('RGB')
                    variants.append((f'reference c_{role}:{donor_id}', donor_image))
                variants.append((f'generated c_{role}:{donor_id}', swap_image))
                variant_paths.append((donor_id, swap_path))
        make_panel(
            root, sid, image, cfg['data']['resolution'],
            identity_variants=variants, pose_folder=pose_folder,
        ).save(out / f'{sid}.png')
        make_hair_review_panel(
            root,
            sid,
            image,
            cfg['data']['resolution'],
        ).save(out / f'{sid}_hair_review.png')
        for donor_id, swap_path in variant_paths:
            swap_rows.append({'sample_id': sid, 'swap_id': donor_id, 'paired_path': gen_path, 'swap_path': swap_path})
        if directed_enabled and len(variant_paths) == 2:
            swap_rows.append({
                'sample_id': sid,
                'swap_id': f'{variant_paths[0][0]} vs {variant_paths[1][0]}',
                'paired_path': variant_paths[0][1],
                'swap_path': variant_paths[1][1],
                'comparison': 'j_vs_k',
            })

    response_track_enabled = bool(
        cfg.get('eval', {}).get('response_track', {}).get('enabled', False)
    )
    response_snapshot = None
    response_error = None
    if response_track_enabled:
        try:
            response_snapshot = compute_response_snapshot(
                cfg,
                model,
                ds,
                ckpt,
                checkpoint_step,
                torch_device,
                dtype,
            )
        except Exception as exc:  # noqa: BLE001
            response_error = str(exc)
    gate_keys = tuple(adapter.gate_values())
    # Trend metrics run only after the large FLUX/ControlNet/VAE graph is released.
    # This keeps GPU3 below its memory ceiling while retaining a single watcher process.
    del model, transformer, controlnet, pulid, adapter, vae
    gc.collect()
    if torch_device.type == 'cuda':
        torch.cuda.empty_cache()

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
        warnings.append('⚠ face detection rate below 95%')
        if not response_track_enabled:
            stop_training = True
    if adapter_not_responding:
        warnings.append('⚠ adapter not responding; swap ArcFace cosine >= 0.85')
        if not response_track_enabled:
            stop_training = True
    gate_init = float(cfg['model'].get('identity_adapter', {}).get('gate_init', 0.1))
    gate_move_threshold = float(cfg['eval'].get('gate_move_threshold', 1e-3))
    gate_row, gate_stats = gate_history(experiment_dir / 'logs' / 'train.jsonl', gate_init, gate_keys)
    gates_moved = gate_row and all(
        gate_stats[key]['max_abs_deviation'] > gate_move_threshold
        for key in gate_keys
    )
    if not gates_moved and not response_track_enabled:
        warnings.append(
            f'⚠ STOP-TRAINING: condition gates have not moved from init={gate_init} '
            f'by more than {gate_move_threshold}'
        )
        stop_training = True
    watcher_metrics = {}
    response_track = None
    if response_track_enabled and response_snapshot is not None:
        try:
            metric_count = int(
                cfg['eval']['response_track'].get('metric_sample_count', len(ids))
            )
            if metric_count != len(ids):
                raise RuntimeError(
                    'watcher metric_sample_count must equal the 16-sample frozen protocol; '
                    f'got {metric_count}'
                )
            watcher_metrics = run_watcher_metrics(
                cfg,
                trend_rows[:metric_count],
                out / 'metrics_v2_trend',
                device,
            )
        except Exception as exc:  # noqa: BLE001
            watcher_metrics = {'status': 'failed', 'error': str(exc)}
        valid_swap_cosines = [
            float(row['cosine'])
            for row in swap_results
            if row.get('cosine') is not None
        ]
        response_track = finalize_snapshot(
            cfg,
            response_snapshot,
            watcher_metrics=watcher_metrics,
            guards={
                'face_detection_rate': face_rate,
                'swap_cosine_mean': (
                    float(np.mean(valid_swap_cosines))
                    if valid_swap_cosines else None
                ),
                'swap_cosine_max': (
                    float(np.max(valid_swap_cosines))
                    if valid_swap_cosines else None
                ),
            },
        )
        trajectory = response_track.get('trajectory', {})
        if trajectory.get('stop'):
            if trajectory.get('status') == 'hair_plateau':
                warnings.insert(
                    0,
                    f'⚠ HAIR-PLATEAU: Hair-DINO is flat and Hair-LAB is not improving at step {checkpoint_step}',
                )
            else:
                warnings.insert(
                    0,
                    '⚠ STOP-TRAINING: a fixed-set garment/head/identity guard regressed',
                )
            warnings.extend(
                f"next: {value}"
                for value in trajectory.get('recommendations', [])
            )
            stop_training = True
    elif response_track_enabled:
        watcher_metrics = {
            'status': 'failed',
            'error': response_error or 'response snapshot unavailable',
        }
        hard_stop_step = int(
            cfg['eval']['response_track'].get('hair_hard_stop_step', 3000)
        )
        warnings.append(
            f'response track unavailable: {watcher_metrics["error"]}'
        )
        if checkpoint_step == hard_stop_step:
            warnings.insert(0, '⚠ HAIR-PLATEAU: response track is unavailable at the hard gate')
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
    directed_history = {'status': 'disabled'}
    hair_history = hair_loss_history(experiment_dir / 'logs' / 'train.jsonl')
    hair_max_skip = float(
        cfg.get('training', {}).get('hair_loss', {}).get('max_skip_rate', 0.15)
    )
    if (
        hair_history.get('hair_loss_skip_rate') is not None
        and hair_history['hair_loss_skip_rate'] > hair_max_skip
    ):
        warnings.insert(
            0,
            f"⚠ HAIR-LOSS: decode-attempt skip rate {hair_history['hair_loss_skip_rate']:.2%} exceeds {hair_max_skip:.0%}; reasons={hair_history.get('reason_rates')}",
        )
    if directed_enabled:
        directed_history = directed_identity_history(
            experiment_dir / 'logs' / 'train.jsonl',
            out / 'directed_identity_curves.png',
        )
        max_skip = float(differential_cfg.get('identity_loss', {}).get('max_skip_rate', 0.5))
        if directed_history.get('skip_rate') is not None and directed_history['skip_rate'] > max_skip:
            warnings.insert(
                0,
                f"⚠ A4 ID-LOSS: decode face-detection skip rate {directed_history['skip_rate']:.2%} exceeds {max_skip:.0%}",
            )
    automatic_stop = stop_training
    manual_review = review_status(cfg, experiment_dir, completed_run_steps)
    if manual_review['due'] and not manual_review['approved']:
        warnings.insert(
            0,
            '⚠ STOP-TRAINING: mandatory step-500 human review is pending',
        )
        stop_training = True
    report = ['# Warmup Watcher Report', '']
    report.extend(warnings or ['status: automatic checks passed thresholds'])
    report.extend(['', f'checkpoint: `{ckpt}`', '', '## Automatic Checks', ''])
    report.append(f'face detection rate: {detected}/{len(generated_paths)} = {face_rate:.2%}')
    report.append(f'gate latest row: {gate_row if gate_row else "N/A"}')
    report.append(f'gate history movement: {gate_stats}')
    report.append(f'differential collapse monitor: {collapse_details}')
    report.append(f'directed identity training: {directed_history}')
    report.append(f'hair supervision schedule: {hair_history}')
    report.append(f'mandatory human review gate: {manual_review}')
    if response_track_enabled:
        report.extend([
            '',
            '## Spatial Response Trajectory',
            '',
            f'response-track status: {response_track.get("trajectory") if response_track else watcher_metrics}',
        ])
        if response_track:
            response = response_track['response']
            slopes = response_track.get('trajectory', {}).get(
                'slopes_per_500_steps', {}
            )
            report.extend([
                f'checkpoint gates: {response_track.get("gates", {})}',
                f'garment union concentration (descriptive only): {response.get("garment_union_concentration", "N/A")}',
                f'hair-ref swap / identity response: {response.get("hair_relative_response", "N/A")}',
                f'hair-ref swap concentration: {response.get("hair_ref_swap_concentration", "N/A")}',
                f'Garment-DINO median to mannequin: {watcher_metrics.get("garment_dino", "N/A")}',
                f'Garment-HF-LPIPS median to mannequin (lower): {watcher_metrics.get("garment_hf_lpips", "N/A")}',
                f'Garment-Gradient-Sim median to mannequin (higher): {watcher_metrics.get("garment_gradient_sim", "N/A")}',
                f'Hair-DINO median to reference: {watcher_metrics.get("hair_dino", "N/A")}',
                f'Hair LAB median distance to reference: {watcher_metrics.get("hair_lab_distance", "N/A")}',
                f'head-five-point median distance: {watcher_metrics.get("head5_distance", "N/A")}',
                f'body median distance: {watcher_metrics.get("body_distance", "N/A")}',
                f'recent slopes per 500 steps: {slopes}',
                f'cumulative curves: `{response_track.get("cumulative", {}).get("plot", "N/A")}`',
                '',
                'Training-side hair loss and watcher Hair-DINO use independent code paths but the same DINOv2 weights. '
                'Human review and LAB color distance are required corroboration.',
            ])
    if directed_history.get('plot'):
        report.append('directed identity curves: `directed_identity_curves.png`')
    report.append('')
    report.append('| sample | swap_id | ArcFace cos(generated c_i, swap c_j) | status |')
    report.append('|---|---|---:|---|')
    for row in swap_results:
        cos = row['cosine']
        status = 'N/A detection failed' if cos is None else ('adapter not responding' if cos >= 0.85 else 'responding')
        report.append(f"| {row['sample_id']} | {row['swap_id']} | {'N/A' if cos is None else f'{cos:.4f}'} | {status} |")
    report.extend([
        '',
        '## Fixed-Sample Visual Checklist',
        '',
        '- [ ] Garment is converging toward the same physical item as the source mannequin.',
        '- [ ] Hair is converging toward the displayed identity reference without importing its clothing.',
        '- [ ] Head landmarks and facing continue to follow the with-head mannequin control.',
    ])
    if manual_review.get('enabled'):
        report.extend([
            '',
            'After all three checks pass, approve and clear the manual pause with:',
            f'`python eval_watcher.py --config {config_path} --approve-manual-review --reviewer <name>`',
        ])
    elif response_track_enabled:
        report.extend([
            '',
            'Garment response concentration is descriptive only and never stops training.',
            f"Configured checkpoints {cfg['eval']['response_track'].get('trend_gate_steps', [])} track Hair-DINO, Hair-LAB, Garment-HF-LPIPS, hair-ref concentration, and visual fidelity.",
            'Garment/head/body/face/identity use the immutable 16-sample protocol and stop only after repeated failures; one bad point is warning-only.',
            f"The configured hard step {cfg['eval']['response_track'].get('hair_hard_stop_step')} stops only when Hair-DINO remains flat and Hair-LAB also fails to decline, unless a preserved-path guard regresses.",
        ])
    (out / 'watcher_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    if stop_training:
        marker = experiment_dir / 'STOP_TRAINING'
        payload = {
            'reason': (
                'hair_response_plateau'
                if response_track and response_track.get('trajectory', {}).get('status') == 'hair_plateau'
                else 'preserved_path_regression'
                if response_track and response_track.get('trajectory', {}).get('stop')
                else 'watcher_checks_failed' if automatic_stop else 'manual_review_pending'
            ),
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
            'manual_review': manual_review,
            'response_track': response_track,
            'watcher_metrics': watcher_metrics,
            'hair_supervision_schedule': hair_history,
        }
        tmp = marker.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(marker)
    return stop_training


def approve_manual_review(config_path: str, reviewer: str) -> dict:
    cfg = load_yaml(config_path)
    experiment_dir = Path(cfg['experiment']['output_root']) / cfg['experiment']['id']
    stop_marker = experiment_dir / 'STOP_TRAINING'
    if not stop_marker.exists():
        raise RuntimeError(
            'manual review cannot be approved before training reaches its review pause'
        )
    try:
        stop_payload = json.loads(stop_marker.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'unreadable STOP_TRAINING marker: {exc}') from exc
    if stop_payload.get('reason') != 'manual_review_pending':
        raise RuntimeError(
            'STOP_TRAINING contains automatic failures; resolve them before any manual approval'
        )
    marker_review = stop_payload.get('manual_review', {})
    required_step = int(cfg.get('eval', {}).get('manual_review_step', 0) or 0)
    if (
        not marker_review.get('due')
        or int(marker_review.get('completed_run_steps', -1)) < required_step
    ):
        raise RuntimeError('STOP_TRAINING marker does not prove that the configured review step was reached')
    approval_path = write_approval(cfg, experiment_dir, reviewer)
    stop_marker.unlink()
    result = {
        'approval_path': str(approval_path),
        'cleared_manual_stop': True,
        'reviewed_checkpoint': stop_payload.get('checkpoint'),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='GPU3 checkpoint watcher for Phase 1 warmup.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--ckpt-dir', required=False)
    parser.add_argument('--ckpt', default=None, help='Run one specific checkpoint directory and exit.')
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--approve-manual-review', action='store_true')
    parser.add_argument('--reviewer', default=None)
    args = parser.parse_args()
    if args.approve_manual_review:
        if not args.reviewer:
            raise SystemExit('--reviewer is required with --approve-manual-review')
        approve_manual_review(args.config, args.reviewer)
        return
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
            # A fresh process per checkpoint guarantees that FLUX, ControlNet, VAE,
            # and PuLID CUDA allocations are released before the next evaluation.
            # Reusing one process leaked references through wrapped block forwards
            # and eventually OOMed the final watcher after long training runs.
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    '--config',
                    args.config,
                    '--ckpt',
                    str(ckpt),
                    '--device',
                    args.device,
                ],
                check=True,
            )
            seen.add(str(ckpt))
            if ckpt.name == 'final':
                return
        if args.once:
            break
        time.sleep(load_yaml(args.config)['eval']['watcher_poll_seconds'])


if __name__ == '__main__':
    main()
