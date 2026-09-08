from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from conditions import find_one, get_resolution
from interp_common import identity_image_name, read_json, resolve_root_path, write_csv, write_json
from metrics.common import cosine, safe_mean, safe_median
from metrics.garment_sim import (
    cloth_safe_mask,
    dino_region_feature,
    load_dino_model,
    lpips_tensor,
    read_rgb,
)

# Evaluation-only identity recognition is deliberately imported only in this
# metric runner. Generation and interpolation selection never access it.
from metrics.heldout_id import (  # noqa: E402
    RetinaFaceAligner,
    UnifaceAdaFaceRecognizer,
    resize_face_crop_rgb,
)


IMAGE_FIELDS = (
    'checkpoint', 'mid', 'jid', 'kid', 't', 'seed', 'path', 'status', 'error',
    'identity_status', 'identity_error', 'sim_to_j', 'sim_to_k', 'det_conf', 'face_size_px',
    'garment_status', 'garment_error', 'face_window_status', 'face_window_error',
)
PATH_FIELDS = (
    'checkpoint', 'mid', 'jid', 'kid', 'distance_jk', 'garment_type', 'status', 'error',
    'path_efficiency', 'jump_ratio', 'face_lpips_direct', 'face_lpips_path_length',
    'monotonic_violations', 'monotonic_transitions', 'monotonic_violation_rate',
    'sim_j_endpoint_swing', 'sim_k_endpoint_swing', 'identity_swing',
    'garment_pairwise_mean', 'garment_max_drift_from_t0', 'garment_normalized_drift',
    'valid_identity_frames', 'valid_garment_frames', 'valid_face_window_frames',
)


def normalized_garment_drift(max_drift: float, sim_j_endpoint_swing: float) -> float:
    """Normalize garment drift by the preregistered sim-to-j endpoint change."""
    return float(max_drift) / max(float(sim_j_endpoint_swing), 1e-6)


def _heldout_components(cfg: dict[str, Any], device: torch.device):
    mcfg = cfg['metrics']['heldout_id']
    recognizer_name = str(mcfg.get('recognizer', '')).lower()
    if recognizer_name not in {'uniface_adaface_ir101', 'uniface_adaface_ir_101'}:
        raise RuntimeError(
            'identity interpolation runner requires the configured held-out UniFace AdaFace IR-101; '
            f'got {recognizer_name!r}'
        )
    checkpoint = Path(mcfg['checkpoint'])
    if not checkpoint.exists():
        raise FileNotFoundError(f'held-out recognizer checkpoint missing: {checkpoint}')
    recognizer = UnifaceAdaFaceRecognizer(
        cache_dir=mcfg.get('cache_dir', checkpoint.parent),
        checkpoint=checkpoint,
        device=device,
    )
    device_index = int(str(device).split(':')[-1]) if str(device).startswith('cuda') else -1
    detector = RetinaFaceAligner(
        model_root=mcfg['detector_model_root'],
        device_id=int(mcfg.get('detector_device_id', device_index)),
        det_size=int(mcfg.get('det_size', 640)),
    )
    return recognizer, detector


def _reference_embedding(
    root: Path,
    sample_id: str,
    recognizer,
    detector,
    cfg: dict[str, Any],
) -> np.ndarray:
    mcfg = cfg['metrics']['heldout_id']
    path = find_one(root / 'derived/face_crops/human', sample_id)
    try:
        aligned, _, _ = detector.align(
            path,
            expand=float(mcfg.get('ref_expand', 1.1)),
            min_crop=int(mcfg.get('min_crop_px', 256)),
        )
    except RuntimeError as exc:
        if 'found no face' not in str(exc):
            raise
        aligned = resize_face_crop_rgb(path)
    return recognizer.embed_aligned_rgb(aligned)


def _binary_region_mask(root: Path, mid: str, key: str, resolution) -> np.ndarray:
    path = root / 'derived/region_masks' / f'{mid}.npz'
    if not path.exists():
        raise FileNotFoundError(f'region mask missing: {path}')
    data = np.load(path)
    if key not in data.files:
        raise KeyError(f'{path} has no {key!r}')
    values = np.asarray(data[key])
    if values.dtype == np.bool_ or float(values.max(initial=0.0)) <= 1.0:
        mask = (values > 0.5).astype(np.uint8) * 255
    else:
        mask = (values > 127).astype(np.uint8) * 255
    width, height = get_resolution(resolution)
    return (
        np.asarray(Image.fromarray(mask, mode='L').resize((width, height), Image.Resampling.NEAREST)) > 127
    ).astype(np.uint8)


def _fixed_face_crop(image: np.ndarray, mask: np.ndarray, pad: int, size: int) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError('id_strong mask is empty')
    x0 = max(0, int(xs.min()) - int(pad))
    x1 = min(mask.shape[1], int(xs.max()) + int(pad) + 1)
    y0 = max(0, int(ys.min()) - int(pad))
    y1 = min(mask.shape[0], int(ys.max()) + int(pad) + 1)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        raise RuntimeError('fixed id_strong bbox produced an empty crop')
    return np.asarray(Image.fromarray(crop).resize((int(size), int(size)), Image.Resampling.BICUBIC))


@torch.no_grad()
def _lpips_distance(loss_fn, first: np.ndarray, second: np.ndarray, device: torch.device) -> float:
    return float(loss_fn(lpips_tensor(first, device), lpips_tensor(second, device)).cpu().item())


def _make_strips(
    selection: dict[str, Any],
    gen_root: Path,
    out_dir: Path,
    count: int,
    ts: list[float],
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    thumb_w, thumb_h = 192, 256
    label_h, row_gap = 26, 8
    for row in selection['rows'][: int(count)]:
        mid, jid, kid = str(row['mid']), str(row['jid']), str(row['kid'])
        canvas = Image.new(
            'RGB',
            (thumb_w * len(ts), (thumb_h + label_h) * 2 + row_gap),
            'white',
        )
        draw = ImageDraw.Draw(canvas)
        for row_index, checkpoint in enumerate(('a4', 'b2cont')):
            y = row_index * (thumb_h + label_h + row_gap)
            for index, t in enumerate(ts):
                path = gen_root / checkpoint / identity_image_name(mid, jid, kid, t, int(row['seed']))
                if path.exists():
                    image = Image.open(path).convert('RGB').resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                    canvas.paste(image, (index * thumb_w, y + label_h))
                draw.text((index * thumb_w + 5, y + 6), f'{checkpoint} t={t:.2f}', fill='black')
        output = out_dir / f'{mid}.jpg'
        canvas.save(output, quality=92)
        outputs.append(str(output))
    return outputs


def run_identity_interp_metrics(
    cfg: dict[str, Any],
    selection_path: str | Path,
    gen_root: str | Path,
    out_dir: str | Path,
    device: str = 'cuda:0',
) -> dict[str, Any]:
    root = Path(cfg['data']['root'])
    identity_cfg = cfg['inference_interpolation']['identity']
    selection = read_json(selection_path)
    gen_root = Path(gen_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = [float(value) for value in identity_cfg['ts']]
    if ts != sorted(ts) or ts[0] != 0.0 or ts[-1] != 1.0:
        raise RuntimeError(f'identity interpolation t grid must be ordered and include endpoints, got {ts}')
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith('cuda') else 'cpu')
    recognizer, detector = _heldout_components(cfg, torch_device)
    dino_model = load_dino_model(cfg, torch_device)
    import lpips

    loss_fn = lpips.LPIPS(net='alex').to(torch_device).eval().requires_grad_(False)
    dino_cfg = cfg['metrics']['garment']
    dino_size = int(dino_cfg.get('dino_image_size', 518))
    mask_out_value = float(dino_cfg.get('mask_out_value', 0.5))
    resolution = cfg['data']['resolution']
    pad = int(identity_cfg.get('face_window_pad', 16))
    crop_size = int(identity_cfg.get('face_crop_size', 256))
    epsilon = float(identity_cfg.get('monotonic_epsilon', 0.005))

    requested_ids = sorted({str(row[key]) for row in selection['rows'] for key in ('jid', 'kid')})
    references: dict[str, np.ndarray] = {}
    reference_errors: dict[str, str] = {}
    for sample_id in tqdm(requested_ids, desc='held-out identity references'):
        try:
            references[sample_id] = _reference_embedding(root, sample_id, recognizer, detector, cfg)
        except Exception as exc:  # noqa: BLE001
            reference_errors[sample_id] = f'{type(exc).__name__}: {exc}'

    cloth_masks: dict[str, np.ndarray] = {}
    face_masks: dict[str, np.ndarray] = {}
    image_rows: list[dict[str, Any]] = []
    image_cache: dict[tuple[str, str, float], dict[str, Any]] = {}
    total = len(selection['rows']) * 2 * len(ts)
    progress = tqdm(total=total, desc='identity interpolation metrics')
    for selection_row in selection['rows']:
        mid, jid, kid = str(selection_row['mid']), str(selection_row['jid']), str(selection_row['kid'])
        if mid not in cloth_masks:
            cloth_masks[mid] = cloth_safe_mask(root, mid, resolution)
            face_masks[mid] = _binary_region_mask(root, mid, 'id_strong', resolution)
        for checkpoint in ('a4', 'b2cont'):
            for t in ts:
                seed = int(selection_row['seed'])
                path = gen_root / checkpoint / identity_image_name(mid, jid, kid, t, seed)
                row: dict[str, Any] = {
                    'checkpoint': checkpoint,
                    'mid': mid,
                    'jid': jid,
                    'kid': kid,
                    't': t,
                    'seed': seed,
                    'path': str(path),
                    'status': 'ok',
                    'error': '',
                    'identity_status': 'pending',
                    'identity_error': '',
                    'garment_status': 'pending',
                    'garment_error': '',
                    'face_window_status': 'pending',
                    'face_window_error': '',
                }
                payload: dict[str, Any] = {}
                if not path.exists():
                    row.update(status='failed', error='generated image missing')
                    image_rows.append(row)
                    progress.update(1)
                    continue
                try:
                    image = read_rgb(path, resolution)
                except Exception as exc:  # noqa: BLE001
                    row.update(status='failed', error=f'image read: {type(exc).__name__}: {exc}')
                    image_rows.append(row)
                    progress.update(1)
                    continue
                try:
                    payload['face_crop'] = _fixed_face_crop(image, face_masks[mid], pad, crop_size)
                    row['face_window_status'] = 'ok'
                except Exception as exc:  # noqa: BLE001
                    row['face_window_status'] = 'failed'
                    row['face_window_error'] = f'{type(exc).__name__}: {exc}'
                try:
                    payload['garment_feature'] = dino_region_feature(
                        dino_model,
                        image,
                        cloth_masks[mid],
                        torch_device,
                        dino_size,
                        mask_out_value,
                    )
                    row['garment_status'] = 'ok'
                except Exception as exc:  # noqa: BLE001
                    row['garment_status'] = 'failed'
                    row['garment_error'] = f'{type(exc).__name__}: {exc}'
                try:
                    if jid in reference_errors or kid in reference_errors:
                        raise RuntimeError(
                            f'reference failure j={reference_errors.get(jid)} k={reference_errors.get(kid)}'
                        )
                    aligned, face_size, det_conf = detector.align(
                        path,
                        expand=float(cfg['metrics']['heldout_id'].get('gen_expand', 1.3)),
                        min_crop=int(cfg['metrics']['heldout_id'].get('min_crop_px', 256)),
                    )
                    embedding = recognizer.embed_aligned_rgb(aligned)
                    row.update(
                        identity_status='ok',
                        sim_to_j=cosine(embedding, references[jid]),
                        sim_to_k=cosine(embedding, references[kid]),
                        det_conf=det_conf,
                        face_size_px=face_size,
                    )
                except Exception as exc:  # noqa: BLE001
                    row['identity_status'] = 'failed'
                    row['identity_error'] = f'{type(exc).__name__}: {exc}'
                failures = [
                    name for name in ('identity', 'garment', 'face_window')
                    if row[f'{name}_status'] != 'ok'
                ]
                if failures:
                    row['status'] = 'partial'
                    row['error'] = ','.join(failures)
                image_rows.append(row)
                image_cache[(checkpoint, mid, t)] = {**payload, **row}
                progress.update(1)
    progress.close()

    path_rows: list[dict[str, Any]] = []
    for selection_row in selection['rows']:
        mid, jid, kid = str(selection_row['mid']), str(selection_row['jid']), str(selection_row['kid'])
        for checkpoint in ('a4', 'b2cont'):
            frames = [image_cache.get((checkpoint, mid, t), {}) for t in ts]
            output: dict[str, Any] = {
                'checkpoint': checkpoint,
                'mid': mid,
                'jid': jid,
                'kid': kid,
                'distance_jk': selection_row['distance_jk'],
                'garment_type': selection_row['garment_type'],
                'status': 'ok',
                'error': '',
                'valid_identity_frames': sum(frame.get('identity_status') == 'ok' for frame in frames),
                'valid_garment_frames': sum(frame.get('garment_status') == 'ok' for frame in frames),
                'valid_face_window_frames': sum(frame.get('face_window_status') == 'ok' for frame in frames),
            }
            errors = []
            if all('face_crop' in frame for frame in frames):
                steps = [
                    _lpips_distance(loss_fn, frames[index]['face_crop'], frames[index + 1]['face_crop'], torch_device)
                    for index in range(len(frames) - 1)
                ]
                direct = _lpips_distance(loss_fn, frames[0]['face_crop'], frames[-1]['face_crop'], torch_device)
                path_length = float(sum(steps))
                output.update(
                    path_efficiency=float(direct / max(path_length, 1e-12)),
                    jump_ratio=float(max(steps) / max(float(np.mean(steps)), 1e-12)),
                    face_lpips_direct=direct,
                    face_lpips_path_length=path_length,
                )
            else:
                errors.append('face LPIPS incomplete')
            if all(frame.get('identity_status') == 'ok' for frame in frames):
                sim_j = np.asarray([float(frame['sim_to_j']) for frame in frames], dtype=np.float64)
                sim_k = np.asarray([float(frame['sim_to_k']) for frame in frames], dtype=np.float64)
                violations = int(np.sum(np.diff(sim_j) > epsilon) + np.sum(np.diff(sim_k) < -epsilon))
                transitions = 2 * (len(ts) - 1)
                swing_j = float(abs(sim_j[-1] - sim_j[0]))
                swing_k = float(abs(sim_k[-1] - sim_k[0]))
                output.update(
                    monotonic_violations=violations,
                    monotonic_transitions=transitions,
                    monotonic_violation_rate=violations / transitions,
                    sim_j_endpoint_swing=swing_j,
                    sim_k_endpoint_swing=swing_k,
                    # The preregistered normalization denominator is explicitly
                    # the sim_to_j endpoint change, not an average over j and k.
                    identity_swing=swing_j,
                )
            else:
                errors.append('held-out identity path incomplete')
            if all('garment_feature' in frame for frame in frames):
                features = [frame['garment_feature'] for frame in frames]
                pairwise = [cosine(features[a], features[b]) for a, b in combinations(range(len(features)), 2)]
                drifts = [1.0 - cosine(features[0], feature) for feature in features[1:]]
                max_drift = float(max(drifts, default=0.0))
                output.update(
                    garment_pairwise_mean=float(np.mean(pairwise)),
                    garment_max_drift_from_t0=max_drift,
                )
                if 'identity_swing' in output:
                    output['garment_normalized_drift'] = normalized_garment_drift(
                        max_drift,
                        float(output['identity_swing']),
                    )
                else:
                    errors.append('normalized garment drift lacks identity swing')
            else:
                errors.append('DINO garment path incomplete')
            if errors:
                output['status'] = 'partial'
                output['error'] = '; '.join(errors)
            path_rows.append(output)

    write_csv(out_dir / 'identity_interp_per_image.csv', image_rows, list(IMAGE_FIELDS))
    write_csv(out_dir / 'identity_interp_per_path.csv', path_rows, list(PATH_FIELDS))
    strips = _make_strips(
        selection,
        gen_root,
        resolve_root_path(root, identity_cfg['strip_dir']),
        int(identity_cfg.get('strip_count', 6)),
        ts,
    )
    by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in path_rows:
        by_checkpoint[row['checkpoint']].append(row)
    metrics = (
        'path_efficiency', 'jump_ratio', 'monotonic_violation_rate', 'identity_swing',
        'garment_pairwise_mean', 'garment_max_drift_from_t0', 'garment_normalized_drift',
    )
    summary = {
        'status': 'ok',
        'paths': len(path_rows),
        'images': len(image_rows),
        'image_failures': sum(row['status'] == 'failed' for row in image_rows),
        'image_partials': sum(row['status'] == 'partial' for row in image_rows),
        'heldout_recognizer': 'UniFace AdaFace IR-101 (evaluation only)',
        'face_smoothness_window': 'fixed bbox of source-mid id_strong mask; no per-frame detector geometry',
        'garment_feature': 'same cloth_safe-mask DINOv2 region feature as official GarmentSim',
        'branches': {
            checkpoint: {
                metric: {
                    'mean': safe_mean([row.get(metric) for row in rows]),
                    'median': safe_median([row.get(metric) for row in rows]),
                }
                for metric in metrics
            }
            for checkpoint, rows in by_checkpoint.items()
        },
        'csv_per_image': str(out_dir / 'identity_interp_per_image.csv'),
        'csv_per_path': str(out_dir / 'identity_interp_per_path.csv'),
        'strips': strips,
    }
    write_json(out_dir / 'identity_interp_summary.json', summary)
    return summary


def main() -> None:
    import argparse

    from conditions import load_yaml

    parser = argparse.ArgumentParser(description='Identity-condition interpolation path metrics.')
    parser.add_argument('--config', default='configs/interpolation.yaml')
    parser.add_argument('--selection', default=None)
    parser.add_argument('--gen-root', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    identity_cfg = cfg['inference_interpolation']['identity']
    summary = run_identity_interp_metrics(
        cfg,
        resolve_root_path(root, args.selection or identity_cfg['selection']),
        resolve_root_path(root, args.gen_root or identity_cfg['output_root']),
        resolve_root_path(root, args.out_dir or identity_cfg['metrics_dir']),
        device=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
