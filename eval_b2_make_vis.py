from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from conditions import find_one, load_yaml


PANEL_W = 160
PANEL_H = 220
LABEL_H = 22
PAD = 6


def fit_image(path: Path, size: tuple[int, int] = (PANEL_W, PANEL_H - LABEL_H)) -> Image.Image:
    image = Image.open(path).convert('RGB')
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', size, (245, 245, 245))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def make_cell(path: Path, label: str, font: ImageFont.ImageFont) -> Image.Image:
    cell = Image.new('RGB', (PANEL_W, PANEL_H), (255, 255, 255))
    draw = ImageDraw.Draw(cell)
    draw.rectangle([0, 0, PANEL_W - 1, PANEL_H - 1], outline=(190, 190, 190))
    draw.text((6, 4), label[:28], fill=(20, 20, 20), font=font)
    cell.paste(fit_image(path), (0, LABEL_H))
    return cell


def paste_row(sheet: Image.Image, cells: list[Image.Image], y: int) -> None:
    x = PAD
    for cell in cells:
        sheet.paste(cell, (x, y))
        x += PANEL_W + PAD


def save_sheet(rows: list[list[Image.Image]], path: Path) -> None:
    if not rows:
        raise RuntimeError('no rows to render')
    width = PAD + len(rows[0]) * (PANEL_W + PAD)
    height = PAD + len(rows) * (PANEL_H + PAD)
    sheet = Image.new('RGB', (width, height), (232, 232, 232))
    y = PAD
    for row in rows:
        paste_row(sheet, row, y)
        y += PANEL_H + PAD
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def gen_path(root: Path, cfg: dict, mid: str, jid: str, seed: int) -> Path:
    return root / cfg['eval']['b2_output_dir'] / f'{mid}__id{jid}__seed{seed}.png'


def build_manual_sheet(cfg: dict, subset: dict, out_dir: Path) -> Path:
    root = Path(cfg['data']['root'])
    font = ImageFont.load_default()
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in subset['pairs']:
        by_type[row.get('garment_type', subset.get('garment_types', {}).get(row['mannequin_id'], 'unknown'))].append(row)
    selected = []
    for gtype in ['dress', 'skirt', 'pants', 'top', 'unknown']:
        selected.extend(by_type.get(gtype, [])[:5])
    selected = selected[:20]
    rows = []
    for row in selected:
        mid = row['mannequin_id']
        jid = row['identity_id']
        gtype = row.get('garment_type', subset.get('garment_types', {}).get(mid, 'unknown'))
        cells = [
            make_cell(find_one(root / 'images/mannequin', mid), f'm {mid} {gtype}', font),
            make_cell(find_one(root / 'dwpose/without_head/mannequin', mid), 'pose', font),
            make_cell(find_one(root / 'derived/face_crops/human', jid), f'id {jid}', font),
            make_cell(gen_path(root, cfg, mid, jid, 0), 'gen seed0', font),
            make_cell(gen_path(root, cfg, mid, jid, 1), 'gen seed1', font),
        ]
        rows.append(cells)
    out = out_dir / 'manual_20_contact_sheet.png'
    save_sheet(rows, out)
    return out


def build_identity_spotcheck(cfg: dict, subset: dict, out_dir: Path) -> Path:
    root = Path(cfg['data']['root'])
    font = ImageFont.load_default()
    by_mid: dict[str, list[dict]] = defaultdict(list)
    for row in subset['pairs']:
        by_mid[row['mannequin_id']].append(row)
    rows = []
    for mid, items in list(by_mid.items())[:10]:
        unique = []
        seen = set()
        for item in items:
            jid = item['identity_id']
            if jid not in seen:
                unique.append(item)
                seen.add(jid)
            if len(unique) >= 2:
                break
        if len(unique) < 2:
            continue
        a, b = unique[0], unique[1]
        gtype = subset.get('garment_types', {}).get(mid, a.get('garment_type', 'unknown'))
        cells = [
            make_cell(find_one(root / 'images/mannequin', mid), f'm {mid} {gtype}', font),
            make_cell(find_one(root / 'derived/face_crops/human', a['identity_id']), f'id {a["identity_id"]}', font),
            make_cell(gen_path(root, cfg, mid, a['identity_id'], 0), 'gen A', font),
            make_cell(find_one(root / 'derived/face_crops/human', b['identity_id']), f'id {b["identity_id"]}', font),
            make_cell(gen_path(root, cfg, mid, b['identity_id'], 0), 'gen B', font),
        ]
        rows.append(cells)
    out = out_dir / 'identity_spotcheck_10pairs.png'
    save_sheet(rows, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Build B2 visualization contact sheets for the frozen subset.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--subset', default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    subset_path = Path(args.subset) if args.subset else root / cfg['data']['cf_subset']
    subset = json.loads(subset_path.read_text(encoding='utf-8'))
    out_dir = root / 'eval/b2_vis'
    manual = build_manual_sheet(cfg, subset, out_dir)
    spot = build_identity_spotcheck(cfg, subset, out_dir)
    print(f'wrote {manual}')
    print(f'wrote {spot}')


if __name__ == '__main__':
    main()
