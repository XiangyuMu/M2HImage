from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from conditions import find_one, load_yaml


B2_NAME_RE = re.compile(r'^(?P<mid>.+)__id(?P<jid>.+)__seed(?P<seed>\d+)\.png$')
LOWER_BODY_KEYS = (8, 9, 10, 11, 12, 13, 14)

PARSING_LABELS = {'top': 3, 'dress': 4, 'skirt': 5, 'pants': 6}
_PARSING_COUNTS_CACHE: dict[str, dict[str, dict[int, int]]] = {}


def safe_mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return mean(vals) if vals else None


def safe_median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return median(vals) if vals else None


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return 'n/a'
    return f'{float(value):.{digits}f}'


def load_attrs(root: Path, sample_id: str) -> dict[str, Any]:
    path = root / 'derived/id_attributes' / f'{sample_id}.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def garment_type(root: Path, sample_id: str) -> str:
    attrs = load_attrs(root, sample_id)
    attr_type = attrs.get('garment_type', attrs.get('cloth_type'))
    if attr_type:
        return str(attr_type)
    return parsing_garment_type(root, sample_id)


def load_parsing_counts(root: Path, image_type: str = 'mannequin') -> dict[str, dict[int, int]]:
    key = f'{root}:{image_type}'
    if key in _PARSING_COUNTS_CACHE:
        return _PARSING_COUNTS_CACHE[key]
    manifest = root / 'human_parsing/fashn/metadata/fashn_human_parser_manifest.jsonl'
    counts: dict[str, dict[int, int]] = {}
    if manifest.exists():
        with manifest.open('r', encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                if row.get('image_type') != image_type or row.get('status') != 'ok':
                    continue
                counts[row['id']] = {int(k): int(v) for k, v in row.get('class_pixel_counts', {}).items()}
    _PARSING_COUNTS_CACHE[key] = counts
    return counts


def classify_parsing_counts(counts: dict[int, int]) -> str:
    top = int(counts.get(PARSING_LABELS['top'], 0))
    dress = int(counts.get(PARSING_LABELS['dress'], 0))
    skirt = int(counts.get(PARSING_LABELS['skirt'], 0))
    pants = int(counts.get(PARSING_LABELS['pants'], 0))
    garment = top + dress + skirt + pants
    if garment <= 0:
        return 'unknown'
    ratios = {'top': top / garment, 'dress': dress / garment, 'skirt': skirt / garment, 'pants': pants / garment}
    if ratios['dress'] >= 0.25:
        return 'dress'
    if ratios['skirt'] >= 0.18:
        return 'skirt'
    if ratios['pants'] >= 0.20:
        return 'pants'
    if ratios['top'] >= 0.20:
        return 'top'
    return max(ratios, key=ratios.get)


def parsing_garment_type(root: Path, sample_id: str) -> str:
    counts = load_parsing_counts(root, image_type='mannequin').get(sample_id)
    if counts is None:
        counts = load_parsing_counts(root, image_type='human').get(sample_id, {})
    return classify_parsing_counts(counts)


def is_skirt_or_dress(root: Path, sample_id: str) -> bool:
    return garment_type(root, sample_id).lower() in {'skirt', 'dress'}


def cache_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def cache_sanity(cfg: dict[str, Any], subset: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg['data']['root'])
    cache = root / cfg['data']['cache_dir'] / 'samples'
    deltas = []
    garment_scores = []
    by_m: dict[str, list[str]] = defaultdict(list)
    for row in subset['pairs']:
        mid, jid = row['mannequin_id'], row['identity_id']
        m_path, j_path = cache / f'{mid}.npz', cache / f'{jid}.npz'
        if not m_path.exists() or not j_path.exists():
            continue
        m = np.load(m_path)
        j = np.load(j_path)
        deltas.append(cache_cosine(m['identity'], j['identity']) - cache_cosine(m['identity'], m['identity']))
        by_m[mid].append(jid)
    for ids in by_m.values():
        feats = [np.load(cache / f'{sid}.npz')['garment'] for sid in ids if (cache / f'{sid}.npz').exists()]
        for a, b in combinations(feats, 2):
            garment_scores.append(cache_cosine(a, b))
    return {
        'delta_id_cache_proxy_mean': safe_mean(deltas),
        'delta_id_cache_proxy_median': safe_median(deltas),
        'same_mannequin_garment_token_cosine_mean': safe_mean(garment_scores),
        'same_mannequin_garment_token_pair_count': len(garment_scores),
    }


def expected_b2_images(cfg: dict[str, Any], subset: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(cfg['data']['root'])
    gen_dir = root / cfg['eval']['b2_output_dir']
    rows = []
    for row in subset['pairs']:
        mid = row['mannequin_id']
        jid = row['identity_id']
        for seed in row.get('seeds', cfg['eval']['b2_seeds']):
            path = gen_dir / f'{mid}__id{jid}__seed{seed}.png'
            rows.append({'mid': mid, 'jid': jid, 'seed': int(seed), 'path': path})
    return rows


def read_rgb(path: Path, size: int | None = None) -> np.ndarray:
    image = Image.open(path).convert('RGB')
    if size is not None and image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def read_bgr(path: Path, size: int | None = None) -> np.ndarray:
    rgb = read_rgb(path, size=size)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def mask_bbox(mask: np.ndarray, pad: int = 16) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def cloth_safe_mask(root: Path, sample_id: str, size: int) -> np.ndarray:
    mask_path = find_one(root / 'clothes_bySAM/masks/human', sample_id)
    mask = Image.open(mask_path).convert('L').resize((size, size), Image.Resampling.NEAREST)
    arr = (np.asarray(mask) > 127).astype(np.uint8)
    kernel = np.ones((9, 9), dtype=np.uint8)
    safe = cv2.erode(arr, kernel, iterations=1)
    if safe.sum() < 128:
        safe = arr
    return safe


def masked_crop(image: np.ndarray, mask: np.ndarray, crop_size: int = 256) -> np.ndarray:
    x0, y0, x1, y1 = mask_bbox(mask)
    crop = image[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1].astype(bool)
    if crop.size == 0:
        crop = image
        crop_mask = np.ones(image.shape[:2], dtype=bool)
    white = np.full_like(crop, 255)
    crop = np.where(crop_mask[..., None], crop, white)
    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


def torch_lpips_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = image.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_garment_proxy(cfg: dict[str, Any], rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    from skimage.metrics import structural_similarity
    import lpips

    root = Path(cfg['data']['root'])
    size = int(cfg['data']['resolution'])
    loss_fn = lpips.LPIPS(net='alex').to(device).eval()
    by_mid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row['path'].exists():
            by_mid[row['mid']].append(row)

    lpips_vals: list[float] = []
    ssim_vals: list[float] = []
    pair_count = 0
    with torch.no_grad():
        for mid, items in sorted(by_mid.items()):
            safe = cloth_safe_mask(root, mid, size)
            crops: dict[Path, np.ndarray] = {}
            for item in items:
                crops[item['path']] = masked_crop(read_rgb(item['path'], size=size), safe)
            for a, b in combinations(items, 2):
                if a['jid'] == b['jid']:
                    continue
                ia = crops[a['path']]
                ib = crops[b['path']]
                ssim_vals.append(float(structural_similarity(ia, ib, channel_axis=2, data_range=255)))
                ta = torch_lpips_tensor(ia, device)
                tb = torch_lpips_tensor(ib, device)
                lpips_vals.append(float(loss_fn(ta, tb).detach().float().cpu().item()))
                pair_count += 1

    return {
        'kind': 'local_proxy_lpips_ssim_not_dino',
        'pair_count': pair_count,
        'lpips_mean': safe_mean(lpips_vals),
        'lpips_median': safe_median(lpips_vals),
        'ssim_mean': safe_mean(ssim_vals),
        'ssim_median': safe_median(ssim_vals),
    }


def load_dwpose_detector():
    dwpose_root = Path('/data/muxiangyu/pythonPrograms/StableAnimator/DWPose')
    if str(dwpose_root) not in sys.path:
        sys.path.insert(0, str(dwpose_root))
    from dwpose_utils.dwpose_detector import DWposeDetectorAligned

    return DWposeDetectorAligned()


def body_from_pose(pose: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bodies = pose.get('bodies', {})
    candidate = np.asarray(bodies.get('candidate', []), dtype=np.float32)
    subset = np.asarray(bodies.get('subset', []), dtype=np.float32)
    score = np.asarray(bodies.get('score', []), dtype=np.float32)
    body = np.full((18, 2), np.nan, dtype=np.float32)
    scores = np.zeros(18, dtype=np.float32)
    if candidate.size == 0 or subset.size == 0 or score.size == 0:
        return body, scores
    if subset.ndim == 1:
        subset = subset[None, :]
    if score.ndim == 1:
        score = score[None, :]
    best = int(np.nanargmax(np.where(score > 0.3, score, 0.0).sum(axis=1)))
    for key in range(min(18, subset.shape[1])):
        idx = int(subset[best, key])
        if idx >= 0 and idx < candidate.shape[0]:
            body[key] = candidate[idx, :2]
            scores[key] = score[best, key]
    return body, scores


def pose_distance(
    a: np.ndarray,
    sa: np.ndarray,
    b: np.ndarray,
    sb: np.ndarray,
    keys: tuple[int, ...] | None = None,
    min_common: int = 4,
) -> float:
    if keys is None:
        keys = tuple(range(18))
    idx = np.asarray(keys, dtype=np.int64)
    common = (sa[idx] > 0.3) & (sb[idx] > 0.3) & np.isfinite(a[idx]).all(axis=1) & np.isfinite(b[idx]).all(axis=1)
    if int(common.sum()) < min_common:
        return float('nan')
    return float(np.linalg.norm(a[idx][common] - b[idx][common], axis=1).mean())


def pose_cache_key(kind: str, sample_id: str | None = None, path: Path | None = None) -> str:
    if path is not None:
        return f'{kind}:{path.name}'
    return f'{kind}:{sample_id}'


def pose_to_json(body: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    return {'body': body.tolist(), 'scores': scores.tolist()}


def pose_from_json(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(payload['body'], dtype=np.float32), np.asarray(payload['scores'], dtype=np.float32)


def compute_pose_proxy(cfg: dict[str, Any], subset: dict[str, Any], rows: list[dict[str, Any]], cache_path: Path) -> dict[str, Any]:
    root = Path(cfg['data']['root'])
    size = int(cfg['data']['resolution'])
    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding='utf-8'))
    detector = load_dwpose_detector()

    def get_pose(key: str, image_path: Path) -> tuple[np.ndarray, np.ndarray]:
        if key in cache:
            return pose_from_json(cache[key])
        image = read_bgr(image_path, size=size)
        body, scores = body_from_pose(detector(image))
        cache[key] = pose_to_json(body, scores)
        if len(cache) % 50 == 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding='utf-8')
        return body, scores

    source_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mid in subset['mannequins']:
        m_path = find_one(root / 'images/mannequin', mid)
        source_poses[mid] = get_pose(pose_cache_key('source', mid), m_path)

    gen_poses: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    drift_vals: list[float] = []
    lower_vals: list[float] = []
    skirt_lower_vals: list[float] = []
    detection_success = 0
    for row in rows:
        if not row['path'].exists():
            continue
        body, scores = get_pose(pose_cache_key('gen', path=row['path']), row['path'])
        gen_poses[row['path']] = (body, scores)
        if (scores > 0.3).sum() >= 4:
            detection_success += 1
        src_body, src_scores = source_poses[row['mid']]
        drift = pose_distance(body, scores, src_body, src_scores)
        lower = pose_distance(body, scores, src_body, src_scores, keys=LOWER_BODY_KEYS, min_common=3)
        if math.isfinite(drift):
            drift_vals.append(drift)
        if math.isfinite(lower):
            lower_vals.append(lower)
            if is_skirt_or_dress(root, row['mid']):
                skirt_lower_vals.append(lower)

    by_mid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row['path'] in gen_poses:
            by_mid[row['mid']].append(row)
    cross_pose_vals: list[float] = []
    for items in by_mid.values():
        for a, b in combinations(items, 2):
            if a['jid'] == b['jid']:
                continue
            ba, sa = gen_poses[a['path']]
            bb, sb = gen_poses[b['path']]
            dist = pose_distance(ba, sa, bb, sb)
            if math.isfinite(dist):
                cross_pose_vals.append(dist)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding='utf-8')
    return {
        'kind': 'dwpose_generated_vs_mannequin',
        'source_count': len(source_poses),
        'generated_count': len(gen_poses),
        'generated_detection_success': detection_success,
        'generated_detection_rate': detection_success / max(1, len(rows)),
        'drift_mean': safe_mean(drift_vals),
        'drift_median': safe_median(drift_vals),
        'lower_body_drift_mean': safe_mean(lower_vals),
        'lower_body_drift_median': safe_median(lower_vals),
        'skirt_dress_lower_body_drift_mean': safe_mean(skirt_lower_vals),
        'skirt_dress_lower_body_drift_median': safe_median(skirt_lower_vals),
        'cross_identity_pose_variance_mean': safe_mean(cross_pose_vals),
        'cross_identity_pose_variance_median': safe_median(cross_pose_vals),
        'cross_identity_pair_count': len(cross_pose_vals),
    }


def laplacian_blur(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_face_realism_proxy(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    from insightface.app import FaceAnalysis

    root = Path(cfg['data']['root'])
    model_root = Path(cfg['cache']['arcface_model_root'])
    app = FaceAnalysis(name='antelopev2', root=str(model_root), allowed_modules=['detection'], providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    confs: list[float] = []
    area_ratios: list[float] = []
    blur_vals: list[float] = []
    detected = 0
    for row in rows:
        if not row['path'].exists():
            continue
        image = read_bgr(row['path'])
        faces = app.get(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if faces:
            faces = sorted(faces, key=lambda f: float(getattr(f, 'det_score', 0.0)), reverse=True)
            face = faces[0]
            detected += 1
            confs.append(float(face.det_score))
            x0, y0, x1, y1 = [int(round(v)) for v in face.bbox]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
            area_ratios.append(float(max(0, x1 - x0) * max(0, y1 - y0) / max(1, image.shape[0] * image.shape[1])))
            crop = gray[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else gray
            blur_vals.append(laplacian_blur(crop))
        else:
            blur_vals.append(laplacian_blur(gray))
    return {
        'kind': 'insightface_detector_proxy_not_identity_recognizer',
        'image_count': len(rows),
        'detected_count': detected,
        'detection_rate': detected / max(1, len(rows)),
        'det_conf_mean': safe_mean(confs),
        'det_conf_median': safe_median(confs),
        'face_area_ratio_mean': safe_mean(area_ratios),
        'face_area_ratio_median': safe_median(area_ratios),
        'laplacian_blur_mean': safe_mean(blur_vals),
        'laplacian_blur_median': safe_median(blur_vals),
    }


def write_report(cfg: dict[str, Any], subset: dict[str, Any], metrics: dict[str, Any]) -> Path:
    root = Path(cfg['data']['root'])
    report_path = root / cfg['eval']['report_path']
    cache = metrics['cache_sanity']
    garment = metrics['garment_proxy']
    pose = metrics['pose_proxy']
    face = metrics['face_realism_proxy']
    expected = metrics['expected_images']
    actual = metrics['actual_images']
    lines = [
        '# B2 Adapter-only Baseline Report',
        '',
        f"subset: {len(subset['mannequins'])} mannequins / {len(subset['identity_pool'])} identity pool / {len(subset['pairs'])} pairs",
        f"garment type counts: {metrics.get('garment_type_counts', {})}",
        f"seeds per pair: {cfg['eval']['b2_seeds']}",
        f'expected generated images: {expected}',
        f'actual generated images: {actual}',
        f"generation status: {'complete' if actual == expected else 'incomplete'}",
        f"local metrics runtime: {fmt(metrics.get('runtime_seconds'), 1)} sec",
        '',
        '## Cache-space Sanity Proxies',
        '',
        'These are not official B2 metrics; they only verify that the frozen subset/cache is wired correctly.',
        f"DeltaID cache proxy mean={fmt(cache['delta_id_cache_proxy_mean'])}, median={fmt(cache['delta_id_cache_proxy_median'])}",
        f"Same-mannequin cross-identity garment-token cosine mean={fmt(cache['same_mannequin_garment_token_cosine_mean'])}",
        '',
        '## Generated-image Metrics',
        '',
        '### DeltaID',
        '',
        'Status: BLOCKED for the official metric. Held-out AdaFace/CurricularFace weights were not found locally; InsightFace antelopev2/glintr100 is treated as the training identity backbone and is not used for official DeltaID.',
        '',
        '### GarmentSim',
        '',
        'Status: LOCAL PROXY ONLY. Official cloth_safe DINO is not wired locally; this run reports cloth-mask LPIPS/SSIM across identities for the same mannequin.',
        f"pairs: {garment['pair_count']}",
        f"LPIPS mean={fmt(garment['lpips_mean'])}, median={fmt(garment['lpips_median'])} (lower is better)",
        f"SSIM mean={fmt(garment['ssim_mean'])}, median={fmt(garment['ssim_median'])} (higher is better)",
        '',
        '### Pose Drift',
        '',
        'Status: COMPLETED with local StableAnimator DWPose.',
        f"generated DWPose detection: {pose['generated_detection_success']}/{pose['generated_count']} ({fmt(pose['generated_detection_rate'] * 100, 2)}%)",
        f"generated-vs-mannequin body drift mean={fmt(pose['drift_mean'])}, median={fmt(pose['drift_median'])}",
        f"lower-body drift mean={fmt(pose['lower_body_drift_mean'])}, median={fmt(pose['lower_body_drift_median'])}",
        f"skirt/dress lower-body drift mean={fmt(pose['skirt_dress_lower_body_drift_mean'])}, median={fmt(pose['skirt_dress_lower_body_drift_median'])}",
        f"cross-identity pose variance mean={fmt(pose['cross_identity_pose_variance_mean'])}, median={fmt(pose['cross_identity_pose_variance_median'])}, pairs={pose['cross_identity_pair_count']}",
        '',
        '### Face Realism',
        '',
        'Status: LOCAL DETECTOR PROXY. This is detector confidence/area/blur, not identity recognition.',
        f"detection: {face['detected_count']}/{face['image_count']} ({fmt(face['detection_rate'] * 100, 2)}%)",
        f"detector confidence mean={fmt(face['det_conf_mean'])}, median={fmt(face['det_conf_median'])}",
        f"face area ratio mean={fmt(face['face_area_ratio_mean'])}, median={fmt(face['face_area_ratio_median'])}",
        f"Laplacian blur mean={fmt(face['laplacian_blur_mean'], 2)}, median={fmt(face['laplacian_blur_median'], 2)}",
        '',
        '### Head Pose MAE',
        '',
        'Status: BLOCKED. Source 6DRepNet JSON exists, but no local generated-image 6DRepNet runner/weights were found.',
        '',
        '## B2 Readout',
        '',
        'B2 generation is complete and local generated-image proxies are recorded. The decisive adapter-only baseline number is still blocked on a held-out identity recognizer; install AdaFace or CurricularFace locally before using DeltaID as the A2 gate.',
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute local generated-image B2 metrics and refresh eval/b2_report.md.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--subset', default=None)
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--skip-garment', action='store_true')
    parser.add_argument('--skip-pose', action='store_true')
    parser.add_argument('--skip-face', action='store_true')
    args = parser.parse_args()

    start = time.time()
    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    subset_path = Path(args.subset) if args.subset else root / cfg['data']['cf_subset']
    subset = json.loads(subset_path.read_text(encoding='utf-8'))
    rows = expected_b2_images(cfg, subset)
    missing = [str(row['path']) for row in rows if not row['path'].exists()]
    if missing:
        preview = '\n'.join(missing[:20])
        raise RuntimeError(f'B2 generation is incomplete: missing {len(missing)} images, first entries:\n{preview}')

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    garment_types = subset.get('garment_types') or {sid: garment_type(root, sid) for sid in subset['mannequins']}
    metrics: dict[str, Any] = {
        'expected_images': len(rows),
        'actual_images': len(rows) - len(missing),
        'subset': str(subset_path),
        'b2_output_dir': str(root / cfg['eval']['b2_output_dir']),
        'garment_type_counts': dict(Counter(garment_types.values())),
        'cache_sanity': cache_sanity(cfg, subset),
    }
    if args.skip_garment:
        metrics['garment_proxy'] = {'pair_count': 0, 'lpips_mean': None, 'lpips_median': None, 'ssim_mean': None, 'ssim_median': None}
    else:
        metrics['garment_proxy'] = compute_garment_proxy(cfg, rows, device)
    if args.skip_pose:
        metrics['pose_proxy'] = {
            'generated_count': len(rows), 'generated_detection_success': 0, 'generated_detection_rate': 0.0,
            'drift_mean': None, 'drift_median': None, 'lower_body_drift_mean': None, 'lower_body_drift_median': None,
            'skirt_dress_lower_body_drift_mean': None, 'skirt_dress_lower_body_drift_median': None,
            'cross_identity_pose_variance_mean': None, 'cross_identity_pose_variance_median': None, 'cross_identity_pair_count': 0,
        }
    else:
        pose_cache = root / 'eval' / 'b2_pose_cache.json'
        metrics['pose_proxy'] = compute_pose_proxy(cfg, subset, rows, pose_cache)
    if args.skip_face:
        metrics['face_realism_proxy'] = {
            'image_count': len(rows), 'detected_count': 0, 'detection_rate': 0.0, 'det_conf_mean': None,
            'det_conf_median': None, 'face_area_ratio_mean': None, 'face_area_ratio_median': None,
            'laplacian_blur_mean': None, 'laplacian_blur_median': None,
        }
    else:
        metrics['face_realism_proxy'] = compute_face_realism_proxy(cfg, rows)

    metrics['runtime_seconds'] = time.time() - start
    out_json = root / 'eval' / 'b2_local_metrics.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report_path = write_report(cfg, subset, metrics)
    print(f'wrote metrics={out_json}')
    print(f'wrote report={report_path}')


if __name__ == '__main__':
    main()
