"""B-C0/C1/C2 exploratory copy/fusion controls; fixed input-only evaluation ROI.

No learned model, no target H, and no method-selected scoring masks.
Seam statistics below are diagnostic, not calibrated perceptual defect labels.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image


def mask_regions(mask, radius=8):
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    interior = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    outer = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return interior, outer & ~interior, ~outer


def feather_copy(source, base, mask, erosion=4, feather=8):
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip((dist - erosion) / max(1, feather), 0, 1)[:, :, None]
    return np.rint(alpha * source.astype(float) + (1-alpha) * base).clip(0, 255).astype(np.uint8)


def poisson_copy(source, base, mask, erosion=4):
    safe = cv2.erode(mask.astype(np.uint8), np.ones((2*erosion+1, 2*erosion+1), np.uint8))
    yy, xx = np.nonzero(safe)
    if len(xx) == 0:
        raise ValueError('empty clone mask')
    center = ((int(xx.min()) + int(xx.max()) + 1)//2, (int(yy.min()) + int(yy.max()) + 1)//2)
    src, dst = cv2.cvtColor(source, cv2.COLOR_RGB2BGR), cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    out = cv2.seamlessClone(src, dst, safe*255, center, cv2.NORMAL_CLONE)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def region_mse(source, generated, region):
    if not region.any():
        return None
    return float(np.mean(((source.astype(float)-generated.astype(float))/255.)[region]**2))


def metrics(source, base, generated, mask):
    interior, band, exterior = mask_regions(mask)
    hf = lambda im: im.astype(np.float32)/255 - cv2.GaussianBlur(im.astype(np.float32)/255, (0, 0), 2)
    dhf = (hf(source)-hf(generated))**2
    magnitude = lambda im: np.sqrt(sum(cv2.Sobel(im.astype(np.float32)/255, cv2.CV_32F, dx, dy, ksize=3)**2 for dx, dy in ((1,0),(0,1))))
    generated_edge, source_edge = magnitude(generated), magnitude(source)
    changed = (np.abs(generated.astype(float)-base) > 2).any(axis=2)
    return {'garment_mse': region_mse(source, generated, mask.astype(bool)),
            'interior_mse': region_mse(source, generated, interior),
            'interior_hf_mse': float(dhf[interior].mean()) if interior.any() else None,
            'boundary_gradient_difference_to_source': float(np.abs(generated_edge-source_edge)[band].mean()) if band.any() else None,
            'outside_allowed_edit_fraction': float(changed[exterior].mean()) if exterior.any() else None,
            'interior_area': int(interior.sum()), 'boundary_area': int(band.sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    audit = json.loads((args.probe/'audit.json').read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    names = ('base', 'feather_e4_f8', 'feather_e8_f16', 'poisson_e4')
    for method in names[1:]:
        (args.out/method).mkdir()
    rows = []
    start = time.monotonic()
    with (args.out/'per_image.jsonl').open('x') as handle:
        for pair in audit['pairs']:
            mid, jid, seed = pair['mid'], pair['jid'], pair['seed']
            name = f'{mid}__id{jid}__seed{seed}.png'
            source = np.asarray(Image.open(args.root/'images/mannequin'/f'{mid}.png').convert('RGB'))
            base = np.asarray(Image.open(args.probe/'outputs'/name).convert('RGB'))
            mask = (np.asarray(Image.open(args.probe/'masks'/f'{mid}.png')) > 127).astype(np.uint8)
            for method in names:
                row = {**pair, 'method': method, 'status': 'ok'}
                try:
                    if method == 'base':
                        generated = base
                    elif method == 'feather_e4_f8':
                        generated = feather_copy(source, base, mask, 4, 8)
                    elif method == 'feather_e8_f16':
                        generated = feather_copy(source, base, mask, 8, 16)
                    else:
                        generated = poisson_copy(source, base, mask)
                    if method != 'base':
                        Image.fromarray(generated).save(args.out/method/name)
                    row.update(metrics(source, base, generated, mask))
                except Exception as exc:
                    row.update(status='failed', error=str(exc))
                rows.append(row)
                handle.write(json.dumps(row, allow_nan=False)+'\n')
                handle.flush()
            print(json.dumps({'mid': mid, 'pairs_completed': len(rows)//len(names)}), flush=True)
    summary = {'formal_result': False, 'pairs': len(audit['pairs']),
               'scope': 'copy and identity-aligned Poisson only; learned warp NOT yet implemented',
               'metric_warning': 'Fixed input ROI; seam proxy not a calibrated defect rate. No identity/pose/DINO assessment yet.',
               'seconds': time.monotonic()-start, 'gpu_hours': 0, 'methods': {}}
    fields = ('garment_mse', 'interior_mse', 'interior_hf_mse', 'boundary_gradient_difference_to_source', 'outside_allowed_edit_fraction')
    for method in names:
        selected = [r for r in rows if r['method']==method]
        summary['methods'][method] = {'total': len(selected), 'failed': sum(r['status']!='ok' for r in selected)}
        for field in fields:
            vals = [r[field] for r in selected if r.get(field) is not None]
            summary['methods'][method][field] = {'n': len(vals), 'mean': float(np.mean(vals)) if vals else None}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
