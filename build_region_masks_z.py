from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from conditions import find_one, get_resolution, load_yaml, read_ids


SOURCE_KEYS = {
    'cloth_safe_z': 'cloth_safe',
    'body_bg_z': 'body_bg',
    'face_z': 'id_strong',
}


def pool_token_mask(mask: np.ndarray, width: int, height: int) -> tuple[np.ndarray, bool]:
    resized = tuple(mask.shape) != (height, width)
    if resized:
        mask = np.asarray(
            Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST)
        )
    values = mask.astype(np.float32)
    if values.max(initial=0.0) > 1.0:
        values /= 255.0
    token = values.reshape(height // 16, 16, width // 16, 16).mean(axis=(1, 3))
    return token.reshape(-1).astype(np.float16), resized


def overlay_mask(image: Image.Image, token_mask: np.ndarray, width: int, height: int, color: tuple[int, int, int]) -> Image.Image:
    grid = token_mask.reshape(height // 16, width // 16).astype(np.float32)
    alpha = Image.fromarray(np.uint8(np.clip(grid, 0.0, 1.0) * 150)).resize(
        (width, height), Image.Resampling.NEAREST
    )
    layer = Image.new('RGB', (width, height), color)
    return Image.composite(layer, image, alpha)


def debug_panel(root: Path, sample_id: str, payload: dict[str, np.ndarray], width: int, height: int) -> Image.Image:
    image = Image.open(find_one(root / 'images/human', sample_id)).convert('RGB').resize(
        (width, height), Image.Resampling.BICUBIC
    )
    panels = [
        ('image', image),
        ('cloth_safe_z', overlay_mask(image, payload['cloth_safe_z'], width, height, (255, 40, 40))),
        ('body_bg_z', overlay_mask(image, payload['body_bg_z'], width, height, (40, 180, 255))),
        ('face_z', overlay_mask(image, payload['face_z'], width, height, (40, 255, 100))),
    ]
    label_h = 32
    canvas = Image.new('RGB', (width * len(panels), height + label_h), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        x = index * width
        canvas.paste(panel, (x, label_h))
        draw.text((x + 8, 8), label, fill='black')
    return canvas


def build_one(
    root: str,
    sample_id: str,
    output_dir: str,
    width: int,
    height: int,
    overwrite: bool,
    write_debug: bool,
) -> dict[str, Any]:
    root_path = Path(root)
    output_path = Path(output_dir) / f'{sample_id}.npz'
    if output_path.exists() and not overwrite:
        return {'id': sample_id, 'status': 'existing', 'resized': False}
    source_path = root_path / 'derived/region_masks' / f'{sample_id}.npz'
    if not source_path.exists():
        raise FileNotFoundError(f'missing source region mask: {source_path}')
    source = np.load(source_path)
    missing = [key for key in SOURCE_KEYS.values() if key not in source.files]
    if missing:
        raise KeyError(f'{source_path} missing keys: {missing}')
    payload: dict[str, np.ndarray] = {}
    resized_any = False
    for output_key, source_key in SOURCE_KEYS.items():
        payload[output_key], resized = pool_token_mask(np.asarray(source[source_key]), width, height)
        resized_any = resized_any or resized
    expected = (height // 16) * (width // 16)
    for key, value in payload.items():
        if value.shape != (expected,):
            raise RuntimeError(f'{sample_id} {key} shape={value.shape}, expected {(expected,)}')
        if not np.isfinite(value).all() or value.min(initial=0.0) < 0.0 or value.max(initial=1.0) > 1.0:
            raise RuntimeError(f'{sample_id} {key} contains invalid mask values')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix('.npz.tmp')
    with tmp.open('wb') as handle:
        np.savez(handle, **payload)
    tmp.replace(output_path)
    if write_debug:
        debug_dir = output_path.parent / 'debug'
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_panel(root_path, sample_id, payload, width, height).save(debug_dir / f'{sample_id}.png')
    return {'id': sample_id, 'status': 'written', 'resized': resized_any}


def main() -> None:
    parser = argparse.ArgumentParser(description='Build packed-token region masks for A2 differential losses.')
    parser.add_argument('--config', default='configs/a2_diff.yaml')
    parser.add_argument('--split', default='train')
    parser.add_argument('--workers', type=int, default=min(24, os.cpu_count() or 8))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--debug-count', type=int, default=20)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    width, height = get_resolution(cfg['data']['resolution'])
    output_dir = root / cfg['data'].get('region_masks_z_dir', 'derived/region_masks_z')
    output_dir.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    for split in args.split.split(','):
        split = split.strip()
        if split:
            ids.extend(read_ids(root / cfg['data'][f'{split}_split']))
    ids = sorted(set(ids))
    if args.limit is not None:
        ids = ids[: args.limit]
    debug_ids = set(ids[: max(0, args.debug_count)])
    failures: list[dict[str, str]] = []
    resized = 0
    written = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                build_one,
                str(root),
                sample_id,
                str(output_dir),
                width,
                height,
                args.overwrite,
                sample_id in debug_ids,
            ): sample_id
            for sample_id in ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc='region masks z'):
            sample_id = futures[future]
            try:
                result = future.result()
                written += int(result['status'] == 'written')
                resized += int(result['resized'])
            except Exception as exc:  # noqa: BLE001
                failures.append({'id': sample_id, 'error': str(exc)})

    manifest = {
        'config': args.config,
        'source': 'derived/region_masks/{id}.npz',
        'output': str(output_dir),
        'resolution': {'width': width, 'height': height},
        'token_grid': {'height': height // 16, 'width': width // 16, 'tokens': (height // 16) * (width // 16)},
        'mapping': SOURCE_KEYS,
        'dtype': 'float16',
        'pooling': 'non-overlapping 16x16 image-pixel average pooling in packed-token row-major order',
        'requested': len(ids),
        'written': written,
        'resized_sources': resized,
        'failures': failures,
        'debug_dir': str(output_dir / 'debug'),
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    if failures:
        raise RuntimeError(f'region mask build failed for {len(failures)} samples; see {output_dir / "manifest.json"}')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'failures'}, indent=2))


if __name__ == '__main__':
    main()
