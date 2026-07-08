from __future__ import annotations

import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from conditions import find_one
from metrics.common import cosine, expected_rows, plot_histogram, safe_mean, safe_median, sha256_short, write_csv, write_json


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _metric_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get('metrics', {}).get('garment', {})


def load_dino_model(cfg: dict[str, Any], device: torch.device):
    mcfg = _metric_cfg(cfg)
    repo_root = Path(mcfg.get('dino_repo_root', '/data/muxiangyu/pythonPrograms/GSVTON'))
    checkpoint = Path(mcfg.get('dino_checkpoint', '/data/muxiangyu/pythonPrograms/GSVTON/pretrained/dinov2_vitb14_pretrain.pth'))
    if not repo_root.exists():
        raise FileNotFoundError(f'DINOv2 repo_root not found: {repo_root}')
    if not checkpoint.exists():
        raise FileNotFoundError(f'DINOv2 checkpoint not found: {checkpoint}')
    sys.path.insert(0, str(repo_root))
    try:
        from src.dinov2.models.vision_transformer import load_base_state, vit_with_map

        model = vit_with_map()
        load_base_state(model, str(checkpoint))
    finally:
        try:
            sys.path.remove(str(repo_root))
        except ValueError:
            pass
    model.eval().requires_grad_(False).to(device)
    return model


def cloth_safe_mask(root: Path, sample_id: str, size: int) -> np.ndarray:
    mask_path = root / 'derived/region_masks' / f'{sample_id}.npz'
    if not mask_path.exists():
        raise FileNotFoundError(f'cloth_safe region mask missing: {mask_path}')
    data = np.load(mask_path)
    if 'cloth_safe' not in data.files:
        raise KeyError(f'{mask_path} has no cloth_safe key')
    mask = (np.asarray(data['cloth_safe']) > 127).astype(np.uint8) * 255
    pil = Image.fromarray(mask, mode='L').resize((size, size), Image.Resampling.NEAREST)
    arr = (np.asarray(pil) > 127).astype(np.uint8)
    if arr.sum() < 64:
        raise RuntimeError(f'cloth_safe mask too small for {sample_id}')
    return arr


def read_rgb(path: str | Path, size: int) -> np.ndarray:
    image = Image.open(path).convert('RGB')
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def masked_image(image: np.ndarray, mask: np.ndarray, mask_out_value: float = 0.5) -> np.ndarray:
    bg = int(round(float(mask_out_value) * 255.0))
    out = np.full_like(image, bg, dtype=np.uint8)
    out[mask.astype(bool)] = image[mask.astype(bool)]
    return out


def to_dino_tensor(image: np.ndarray, dino_size: int, device: torch.device) -> torch.Tensor:
    pil = Image.fromarray(image).resize((dino_size, dino_size), Image.Resampling.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.to(device)


def patch_mask(mask: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    h, w = grid_hw
    pil = Image.fromarray((mask.astype(np.uint8) * 255), mode='L').resize((w, h), Image.Resampling.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return (arr.reshape(-1) > 0.25)


@torch.no_grad()
def dino_region_feature(model, image: np.ndarray, mask: np.ndarray, device: torch.device, dino_size: int, mask_out_value: float) -> np.ndarray:
    masked = masked_image(image, mask, mask_out_value=mask_out_value)
    tensor = to_dino_tensor(masked, dino_size=dino_size, device=device)
    patches = model(tensor)
    if isinstance(patches, (tuple, list)):
        patches = patches[0]
    patches = patches.float()[0]
    n, dim = patches.shape
    side = int(round(math.sqrt(n)))
    if side * side != n:
        selected = torch.ones(n, dtype=torch.bool, device=patches.device)
    else:
        pmask = torch.from_numpy(patch_mask(mask, (side, side))).to(device=patches.device)
        selected = pmask if int(pmask.sum()) > 0 else torch.ones(n, dtype=torch.bool, device=patches.device)
    feat = patches[selected].mean(dim=0)
    feat = torch.nn.functional.normalize(feat, dim=0)
    return feat.detach().cpu().numpy().astype('float32')


def mask_bbox(mask: np.ndarray, pad: int = 16) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


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


def lpips_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = image.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_lpips_ssim(rows_by_mid: dict[str, list[dict[str, Any]]], masks: dict[str, np.ndarray], size: int, device: torch.device) -> dict[str, Any]:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        structural_similarity = None
    try:
        import lpips

        loss_fn = lpips.LPIPS(net='alex').to(device).eval()
    except Exception:
        loss_fn = None
    lpips_vals: list[float] = []
    ssim_vals: list[float] = []
    pair_count = 0
    with torch.no_grad():
        for mid, items in rows_by_mid.items():
            crops = {row['path']: masked_crop(read_rgb(row['path'], size=size), masks[mid]) for row in items}
            for a, b in combinations(items, 2):
                ia = crops[a['path']]
                ib = crops[b['path']]
                if structural_similarity is not None:
                    ssim_vals.append(float(structural_similarity(ia, ib, channel_axis=2, data_range=255)))
                if loss_fn is not None:
                    lpips_vals.append(float(loss_fn(lpips_tensor(ia, device), lpips_tensor(ib, device)).detach().cpu().item()))
                pair_count += 1
    return {
        'pair_count': pair_count,
        'lpips_mean': safe_mean(lpips_vals),
        'lpips_median': safe_median(lpips_vals),
        'ssim_mean': safe_mean(ssim_vals),
        'ssim_median': safe_median(ssim_vals),
        'lpips_available': loss_fn is not None,
        'ssim_available': structural_similarity is not None,
    }


def run_garment_sim(
    cfg: dict[str, Any],
    subset: dict[str, Any],
    gen_dir: str | Path,
    out_dir: str | Path,
    device: str = 'cuda:0',
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mcfg = _metric_cfg(cfg)
    root = Path(cfg['data']['root'])
    size = int(cfg['data']['resolution'])
    dino_size = int(mcfg.get('dino_image_size', 518))
    mask_out_value = float(mcfg.get('mask_out_value', 0.5))
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith('cuda') else 'cpu')
    model = load_dino_model(cfg, torch_device)

    rows = [row for row in expected_rows(cfg, subset, gen_dir) if row['path'].exists()]
    by_mid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mid[row['mid']].append(row)

    masks: dict[str, np.ndarray] = {}
    features: dict[str, np.ndarray] = {}
    feature_rows: list[dict[str, Any]] = []
    for mid, items in tqdm(sorted(by_mid.items()), desc='DINO garment features'):
        masks[mid] = cloth_safe_mask(root, mid, size)
        for row in items:
            try:
                image = read_rgb(row['path'], size=size)
                feat = dino_region_feature(model, image, masks[mid], torch_device, dino_size, mask_out_value)
                key = str(row['path'])
                features[key] = feat
                feature_rows.append({**row, 'status': 'ok', 'error': '', 'path': str(row['path'])})
            except Exception as exc:  # noqa: BLE001
                feature_rows.append({**row, 'status': 'failed', 'error': str(exc), 'path': str(row['path'])})

    pair_rows: list[dict[str, Any]] = []
    per_mid_means: list[float] = []
    for mid, items in sorted(by_mid.items()):
        vals = []
        for a, b in combinations(items, 2):
            fa = features.get(str(a['path']))
            fb = features.get(str(b['path']))
            if fa is None or fb is None:
                continue
            sim = cosine(fa, fb)
            vals.append(sim)
            pair_rows.append({
                'mid': mid,
                'path_a': str(a['path']),
                'path_b': str(b['path']),
                'jid_a': a['jid'],
                'jid_b': b['jid'],
                'seed_a': a['seed'],
                'seed_b': b['seed'],
                'dino_cosine': sim,
            })
        if vals:
            per_mid_means.append(float(np.mean(vals)))

    source_rows: list[dict[str, Any]] = []
    source_vals: list[float] = []
    source_features: dict[str, np.ndarray] = {}
    for mid, items in tqdm(sorted(by_mid.items()), desc='DINO source garment'):
        try:
            source_path = find_one(root / 'images/human', mid)
            source_image = read_rgb(source_path, size=size)
            source_features[mid] = dino_region_feature(model, source_image, masks[mid], torch_device, dino_size, mask_out_value)
            for row in items:
                feat = features.get(str(row['path']))
                if feat is None:
                    continue
                sim = cosine(feat, source_features[mid])
                source_vals.append(sim)
                source_rows.append({
                    'mid': mid,
                    'jid': row['jid'],
                    'seed': row['seed'],
                    'path': str(row['path']),
                    'source_path': str(source_path),
                    'dino_source_cosine': sim,
                })
        except Exception as exc:  # noqa: BLE001
            source_rows.append({'mid': mid, 'status': 'failed', 'error': str(exc)})

    aux = compute_lpips_ssim(by_mid, masks, size, torch_device)
    write_csv(out_dir / 'garment_features.csv', feature_rows, ['mid', 'jid', 'seed', 'garment_type', 'theta_source', 'path', 'status', 'error'])
    write_csv(out_dir / 'garment_pairwise_dino.csv', pair_rows, ['mid', 'path_a', 'path_b', 'jid_a', 'jid_b', 'seed_a', 'seed_b', 'dino_cosine'])
    write_csv(out_dir / 'garment_source_dino.csv', source_rows, ['mid', 'jid', 'seed', 'path', 'source_path', 'dino_source_cosine', 'status', 'error'])
    plot_histogram(out_dir / 'garment_dino_pairwise_hist.png', [row['dino_cosine'] for row in pair_rows], 'Garment DINO Cross-identity Similarity', 'DINO cosine')
    plot_histogram(out_dir / 'garment_dino_source_hist.png', [row['dino_source_cosine'] for row in source_rows if 'dino_source_cosine' in row], 'Generated vs Source Garment DINO', 'DINO cosine')

    summary = {
        'status': 'ok',
        'dino': 'GSVTON vendored DINOv2 ViT-B/14',
        'dino_repo_root': str(mcfg.get('dino_repo_root', '/data/muxiangyu/pythonPrograms/GSVTON')),
        'dino_checkpoint': str(mcfg.get('dino_checkpoint', '/data/muxiangyu/pythonPrograms/GSVTON/pretrained/dinov2_vitb14_pretrain.pth')),
        'dino_checkpoint_hash': sha256_short(mcfg.get('dino_checkpoint', '/data/muxiangyu/pythonPrograms/GSVTON/pretrained/dinov2_vitb14_pretrain.pth')),
        'mask_projection': 'source mid derived/region_masks/{mid}.npz cloth_safe resized to generated 512 frame; mask outside set to gray before DINO',
        'dino_image_size': dino_size,
        'mask_out_value': mask_out_value,
        'generated_images': len(rows),
        'feature_failures': len([row for row in feature_rows if row.get('status') != 'ok']),
        'pair_count': len(pair_rows),
        'cross_identity_group_mean_mean': safe_mean(per_mid_means),
        'cross_identity_group_mean_median': safe_median(per_mid_means),
        'pairwise_mean': safe_mean([row['dino_cosine'] for row in pair_rows]),
        'pairwise_median': safe_median([row['dino_cosine'] for row in pair_rows]),
        'source_similarity_mean': safe_mean(source_vals),
        'source_similarity_median': safe_median(source_vals),
        'lpips_mean': aux['lpips_mean'],
        'lpips_median': aux['lpips_median'],
        'ssim_mean': aux['ssim_mean'],
        'ssim_median': aux['ssim_median'],
        'lpips_available': aux['lpips_available'],
        'ssim_available': aux['ssim_available'],
        'csv_pairwise': str(out_dir / 'garment_pairwise_dino.csv'),
        'csv_source': str(out_dir / 'garment_source_dino.csv'),
        'hist_pairwise': str(out_dir / 'garment_dino_pairwise_hist.png'),
        'hist_source': str(out_dir / 'garment_dino_source_hist.png'),
    }
    write_json(out_dir / 'garment_summary.json', summary)
    return summary

