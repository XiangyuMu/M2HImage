from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from qual_eval_common import (
    ALL_RUNS,
    build_panel_context,
    panel_manifest,
    resolve_path,
    write_json,
)


FONT_PATH = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
COLORS = {
    'background': '#f5f6f8',
    'surface': '#ffffff',
    'border': '#aeb4bd',
    'muted': '#555d68',
    'header': '#e9edf2',
    'run_label': '#dfe5ec',
    'text': '#15191e',
}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = FONT_PATH.with_name('DejaVuSans-Bold.ttf') if bold else FONT_PATH
    try:
        return ImageFont.truetype(str(path), int(size))
    except OSError:
        return ImageFont.load_default()


def text_box(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=3)
    return right - left, bottom - top


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    text_font: ImageFont.ImageFont,
    fill: str = COLORS['text'],
) -> None:
    x0, y0, x1, y1 = box
    width, height = text_box(draw, text, text_font)
    draw.multiline_text(
        (x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2),
        text,
        font=text_font,
        fill=fill,
        align='center',
        spacing=3,
    )


def open_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert('RGB').copy()


def contained(image: Image.Image, size: tuple[int, int], background: str = COLORS['surface']) -> Image.Image:
    width, height = map(int, size)
    output = Image.new('RGB', (width, height), background)
    scaled = image.copy()
    scaled.thumbnail((width, height), Image.Resampling.LANCZOS)
    output.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
    return output


def draw_border(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rectangle(box, outline=COLORS['border'], width=1)


def make_run_panel(
    context: dict[str, Any],
    row: dict[str, Any],
    run_names: list[str],
    output: Path,
    title_suffix: str,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        return
    panel_cfg = context['qcfg']['panels']
    cell_w = int(panel_cfg['generated_cell']['width'])
    generated_h = int(panel_cfg['generated_cell']['height'])
    title_h, header_h, metric_h, gap = 58, 205, 54, 8
    row_h = generated_h + metric_h + gap
    width = cell_w * 5
    height = title_h + header_h + row_h * len(run_names)
    canvas = Image.new('RGB', (width, height), COLORS['background'])
    draw = ImageDraw.Draw(canvas)
    title = (
        f"mid {row['mid']} | {row['garment_type']} | {', '.join(row['tags'])}"
        f" | {title_suffix}"
    )
    draw.rectangle((0, 0, width, title_h), fill=COLORS['header'])
    draw_centered(draw, (0, 0, width, title_h), title, font(19, bold=True))

    source = context['sources'][row['mid']]
    header_y = title_h
    header_images = [(source['mannequin'], f"mannequin m_i\nID {row['mid']}")]
    header_images.extend(
        (source['faces'][jid], f"identity c_j{index + 1}\nID {jid}")
        for index, jid in enumerate(row['identity_ids'])
    )
    for column, (path, label) in enumerate(header_images):
        x = column * cell_w
        draw.rectangle((x, header_y, x + cell_w, header_y + header_h), fill=COLORS['surface'])
        image_box_h = header_h - 46
        image = contained(open_rgb(path), (cell_w - 18, image_box_h - 8))
        canvas.paste(image, (x + 9, header_y + 4))
        draw_centered(
            draw,
            (x + 4, header_y + image_box_h, x + cell_w - 4, header_y + header_h),
            label,
            font(13, bold=True),
        )
        draw_border(draw, (x, header_y, x + cell_w - 1, header_y + header_h - 1))

    for run_index, run_name in enumerate(run_names):
        run = context['runs'][run_name]
        y = title_h + header_h + run_index * row_h
        draw.rectangle((0, y, cell_w, y + row_h), fill=COLORS['run_label'])
        group_metric = float(run['garment'][row['mid']])
        draw_centered(
            draw,
            (8, y + 8, cell_w - 8, y + row_h - 8),
            f"{run['label']}\nGarmentSim\n{group_metric:.4f}",
            font(18, bold=True),
        )
        draw_border(draw, (0, y, cell_w - 1, y + row_h - 1))
        for identity_index, jid in enumerate(row['identity_ids']):
            column = identity_index + 1
            x = column * cell_w
            image_path = source['generated'][(run_name, jid)]
            image = contained(open_rgb(image_path), (cell_w, generated_h))
            canvas.paste(image, (x, y))
            metric = run['delta'][(row['mid'], jid, int(context['qcfg']['seed_for_panels']))]
            draw.rectangle((x, y + generated_h, x + cell_w, y + row_h), fill=COLORS['surface'])
            draw_centered(
                draw,
                (x + 4, y + generated_h, x + cell_w - 4, y + row_h - gap),
                f"sim_target {float(metric['sim_target']):.4f}\nmid GarmentSim {group_metric:.4f}",
                font(13),
                fill=COLORS['muted'],
            )
            draw_border(draw, (x, y, x + cell_w - 1, y + row_h - 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format='PNG', optimize=True)


def cloth_bbox(mask_path: Path, expected_size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    data = np.load(mask_path)
    if 'cloth_safe' not in data.files:
        raise KeyError(f'{mask_path} has no cloth_safe channel')
    raw = np.asarray(data['cloth_safe'])
    binary = raw > (0.5 if float(raw.max(initial=0.0)) <= 1.0 else 127.0)
    mask = Image.fromarray((binary.astype(np.uint8) * 255), mode='L')
    if mask.size != expected_size:
        mask = mask.resize(expected_size, Image.Resampling.NEAREST)
    bbox = mask.getbbox()
    if bbox is None:
        raise RuntimeError(f'empty cloth_safe mask: {mask_path}')
    return np.asarray(mask) > 127, bbox


def make_cloth_zoom(
    context: dict[str, Any],
    row: dict[str, Any],
    output: Path,
    overwrite: bool,
) -> float:
    should_write = overwrite or not output.exists()
    panel_cfg = context['qcfg']['panels']
    crop_w = int(panel_cfg['cloth_crop_cell']['width'])
    crop_h = int(panel_cfg['cloth_crop_cell']['height'])
    heat_w = int(panel_cfg['heatmap_cell']['width'])
    heat_h = int(panel_cfg['heatmap_cell']['height'])
    run_names = list(panel_cfg['main_runs'])
    source = context['sources'][row['mid']]
    first_image = open_rgb(source['generated'][(run_names[0], row['identity_ids'][0])])
    mask, bbox = cloth_bbox(source['mask'], first_image.size)
    left, top, right, bottom = bbox
    mask_crop = mask[top:bottom, left:right].astype(np.float32)

    crops: dict[str, list[Image.Image]] = {}
    heatmaps: dict[str, list[tuple[tuple[int, int], np.ndarray]]] = {}
    all_differences: list[np.ndarray] = []
    pairs = list(combinations(range(len(row['identity_ids'])), 2))
    for run_name in run_names:
        run_crops = [
            open_rgb(source['generated'][(run_name, jid)]).crop(bbox)
            for jid in row['identity_ids']
        ]
        gray = [np.asarray(image.convert('L'), dtype=np.float32) for image in run_crops]
        run_heatmaps = []
        for first, second in pairs:
            difference = np.abs(gray[first] - gray[second]) * mask_crop
            run_heatmaps.append(((first, second), difference))
            all_differences.append(difference[mask_crop > 0])
        crops[run_name] = run_crops
        heatmaps[run_name] = run_heatmaps
    nonempty = [values for values in all_differences if values.size]
    shared_scale = max(1.0, float(np.quantile(np.concatenate(nonempty), 0.99)))

    label_w, title_h, crop_label_h, heat_label_h, colorbar_h = 160, 58, 26, 24, 54
    crop_area_h = crop_label_h + crop_h
    heat_area_h = heat_label_h + heat_h
    block_h = crop_area_h + heat_area_h + 14
    content_w = max(crop_w * 4, heat_w * 6)
    width = label_w + content_w
    height = title_h + block_h * len(run_names) + colorbar_h
    canvas = Image.new('RGB', (width, height), COLORS['background'])
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, title_h), fill=COLORS['header'])
    draw_centered(
        draw,
        (0, 0, width, title_h),
        f"cloth_safe bbox | mid {row['mid']} | shared grayscale abs-difference p99={shared_scale:.1f}",
        font(18, bold=True),
    )
    for run_index, run_name in enumerate(run_names):
        run = context['runs'][run_name]
        y = title_h + run_index * block_h
        draw.rectangle((0, y, label_w, y + block_h - 1), fill=COLORS['run_label'])
        draw_centered(draw, (6, y, label_w - 6, y + block_h), run['label'], font(17, bold=True))
        draw_border(draw, (0, y, label_w - 1, y + block_h - 1))
        for index, image in enumerate(crops[run_name]):
            x = label_w + index * crop_w
            draw_centered(
                draw,
                (x, y, x + crop_w, y + crop_label_h),
                f"c_j{index + 1}",
                font(12, bold=True),
            )
            canvas.paste(contained(image, (crop_w, crop_h)), (x, y + crop_label_h))
            draw_border(draw, (x, y, x + crop_w - 1, y + crop_area_h - 1))
        heat_y = y + crop_area_h + 8
        for index, ((first, second), difference) in enumerate(heatmaps[run_name]):
            x = label_w + index * heat_w
            draw_centered(
                draw,
                (x, heat_y, x + heat_w, heat_y + heat_label_h),
                f"j{first + 1}-j{second + 1}",
                font(11, bold=True),
            )
            scaled = np.clip(difference / shared_scale * 255.0, 0.0, 255.0).astype(np.uint8)
            heat_image = Image.fromarray(scaled, mode='L').convert('RGB')
            canvas.paste(contained(heat_image, (heat_w, heat_h), '#000000'), (x, heat_y + heat_label_h))
            draw_border(draw, (x, heat_y, x + heat_w - 1, heat_y + heat_area_h - 1))

    color_y = height - colorbar_h + 8
    bar_w = min(420, width - label_w - 40)
    gradient = np.tile(np.arange(256, dtype=np.uint8), (22, 1))
    bar = Image.fromarray(gradient, mode='L').resize((bar_w, 22), Image.Resampling.BILINEAR).convert('RGB')
    bar_x = label_w + (content_w - bar_w) // 2
    canvas.paste(bar, (bar_x, color_y))
    draw.text((bar_x, color_y + 26), '0', font=font(11), fill=COLORS['text'])
    end_label = f'{shared_scale:.1f}+'
    end_width, _ = text_box(draw, end_label, font(11))
    draw.text((bar_x + bar_w - end_width, color_y + 26), end_label, font=font(11), fill=COLORS['text'])
    draw_centered(
        draw,
        (bar_x, color_y + 24, bar_x + bar_w, color_y + 50),
        '|Delta grayscale| inside cloth_safe',
        font(11),
    )
    if should_write:
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, format='PNG', optimize=True)
    return shared_scale


def make_overview(
    context: dict[str, Any],
    manifest: dict[str, Any],
    output: Path,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        return
    panel_cfg = context['qcfg']['panels']
    run_names = list(panel_cfg['overview_runs'])
    cell_w = int(panel_cfg['overview_cell']['width'])
    cell_h = int(panel_cfg['overview_cell']['height'])
    label_w, title_h, header_h, row_gap = 190, 56, 34, 4
    width = label_w + cell_w * len(run_names)
    row_h = cell_h + row_gap
    height = title_h + header_h + row_h * len(manifest['rows'])
    canvas = Image.new('RGB', (width, height), COLORS['background'])
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, title_h), fill=COLORS['header'])
    draw_centered(draw, (0, 0, width, title_h), '16-mid overview | identity j1 | seed 0', font(19, bold=True))
    draw.rectangle((0, title_h, label_w, title_h + header_h), fill=COLORS['run_label'])
    draw_centered(draw, (0, title_h, label_w, title_h + header_h), 'mid / type / tag', font(12, bold=True))
    for run_index, run_name in enumerate(run_names):
        x = label_w + run_index * cell_w
        draw.rectangle((x, title_h, x + cell_w, title_h + header_h), fill=COLORS['run_label'])
        draw_centered(
            draw,
            (x, title_h, x + cell_w, title_h + header_h),
            context['runs'][run_name]['label'],
            font(12, bold=True),
        )
    for row_index, row in enumerate(manifest['rows']):
        y = title_h + header_h + row_index * row_h
        draw.rectangle((0, y, label_w, y + cell_h), fill=COLORS['surface'])
        draw_centered(
            draw,
            (6, y, label_w - 6, y + cell_h),
            f"{row['mid']}\n{row['garment_type']}\n{', '.join(row['tags'])}",
            font(11, bold=True),
        )
        jid = row['identity_ids'][0]
        for run_index, run_name in enumerate(run_names):
            x = label_w + run_index * cell_w
            image = open_rgb(context['sources'][row['mid']]['generated'][(run_name, jid)])
            canvas.paste(contained(image, (cell_w, cell_h)), (x, y))
            draw_border(draw, (x, y, x + cell_w - 1, y + cell_h - 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format='PNG', optimize=True)


def write_panel_index(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        '# Qualitative Panel Index',
        '',
        f"- Selection seed: `{manifest['protocol']['seed']}`",
        f"- Image seed: `{manifest['protocol']['image_seed']}`",
        f"- Subset: `{manifest['protocol']['subset']}`",
        f"- Subset hash: `{manifest['protocol']['subset_hash']}`",
        f"- Rule: {manifest['protocol']['selection']}",
        f"- Overlap handling: {manifest['protocol']['overlap_note']}",
        '- Metrics are read from existing CSV files; no image or feature metric is recomputed.',
        '- Main paper recommendation: `overview_grid.png`, one worst-garment `panel_*.png`, and its `cloth_zoom_*.png`.',
        '',
        '## Forced extremes',
        '',
        f"- Worst garment: `{manifest['diagnostics']['global_worst_garment']}`",
        f"- Best identity: `{manifest['diagnostics']['global_best_identity']}`",
        '',
        '## Selected mids',
        '',
        '| # | mid | type | tags | identities | B2 GSim / sim | a=.75 GSim / sim | A4 GSim / sim | A2 GSim / sim | A4-B2 GSim | A4-B2 DeltaID | files |',
        '|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in manifest['rows']:
        metrics = row['runs']
        files = row['outputs']
        links = '<br>'.join(
            f"[{label}]({files[key]})"
            for key, label in (
                ('panel', 'panel'), ('cloth_zoom', 'cloth'), ('appendix_panel', 'appendix')
            )
        )
        lines.append(
            f"| {row['order'] + 1} | `{row['mid']}` | {row['garment_type']} | "
            f"{', '.join(row['tags'])} | {', '.join(row['identity_ids'])} | "
            f"{metrics['b2cont']['garment_sim']:.4f} / {metrics['b2cont']['sim_target_seed0_mean']:.4f} | "
            f"{metrics['alpha075']['garment_sim']:.4f} / {metrics['alpha075']['sim_target_seed0_mean']:.4f} | "
            f"{metrics['a4']['garment_sim']:.4f} / {metrics['a4']['sim_target_seed0_mean']:.4f} | "
            f"{metrics['a2']['garment_sim']:.4f} / {metrics['a2']['sim_target_seed0_mean']:.4f} | "
            f"{row['a4_minus_b2cont_garment']:+.4f} | {row['a4_minus_b2cont_deltaid']:+.4f} | {links} |"
        )
    lines.extend([
        '',
        f"Overview: [{Path(manifest['outputs']['overview']).name}]({manifest['outputs']['overview']})",
        '',
        'Each cell label is `held-out sim_target / per-mid GarmentSim`. GarmentSim is the existing mean of `garment_pairwise_dino.csv` for that mid and includes the frozen evaluation seeds used by the official runner.',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Build deterministic qualitative panels from existing images only.')
    parser.add_argument('--config', default='configs/qualitative.yaml')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    context = build_panel_context(args.config)
    manifest = panel_manifest(context)
    root = context['root']
    qcfg = context['qcfg']
    output_dir = resolve_path(root, qcfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_mid = {row['mid']: row for row in manifest['rows']}
    for mid in context['selected']:
        row = rows_by_mid[mid]
        panel = output_dir / f'panel_{mid}.png'
        appendix = output_dir / f'appendix_panel_{mid}.png'
        cloth = output_dir / f'cloth_zoom_{mid}.png'
        make_run_panel(
            context,
            row,
            list(qcfg['panels']['main_runs']),
            panel,
            'main comparison',
            args.overwrite,
        )
        make_run_panel(
            context,
            row,
            list(qcfg['panels']['appendix_runs']),
            appendix,
            'appendix with A2 reference row',
            args.overwrite,
        )
        heat_scale = make_cloth_zoom(context, row, cloth, args.overwrite)
        row['cloth_heatmap_shared_p99'] = heat_scale
        row['outputs'] = {
            'panel': str(panel),
            'cloth_zoom': str(cloth),
            'appendix_panel': str(appendix),
        }
    overview = output_dir / 'overview_grid.png'
    make_overview(context, manifest, overview, args.overwrite)
    manifest['outputs'] = {'overview': str(overview)}
    selection_json = resolve_path(root, qcfg['selection_json'])
    index_md = resolve_path(root, qcfg['panel_index'])
    write_json(selection_json, manifest)
    write_panel_index(manifest, index_md)
    print(
        f"built {len(manifest['rows'])} main panels, {len(manifest['rows'])} cloth zooms, "
        f"{len(manifest['rows'])} appendix panels, and overview under {output_dir}"
    )


if __name__ == '__main__':
    main()
