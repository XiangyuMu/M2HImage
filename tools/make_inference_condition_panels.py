from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conditions import find_one, get_resolution, load_yaml
from spatial_conditions import (
    face_hair_appearance_crop,
    masked_garment_image,
    masked_hair_image,
)


FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def contained(
    image: Image.Image,
    size: tuple[int, int],
    *,
    fill: str | tuple[int, int, int] = "#f1f3f5",
) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", size, fill)
    source = image.convert("RGB")
    scale = min(width / source.width, height / source.height)
    scaled_size = (
        max(1, int(round(source.width * scale))),
        max(1, int(round(source.height * scale))),
    )
    scaled = source.resize(scaled_size, Image.Resampling.LANCZOS)
    canvas.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
    return canvas


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    text_font: ImageFont.ImageFont,
    *,
    fill: str = "#17191c",
    spacing: int = 3,
) -> None:
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox(
        (0, 0), value, font=text_font, spacing=spacing, align="center"
    )
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.multiline_text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        value,
        font=text_font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def mask_bbox(mask: np.ndarray, pad: int = 8) -> tuple[int, int, int, int]:
    ys, xs = np.where(np.asarray(mask) > 0.05)
    if len(xs) == 0:
        return 0, 0, int(mask.shape[1]), int(mask.shape[0])
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(int(mask.shape[1]), int(xs.max()) + pad + 1),
        min(int(mask.shape[0]), int(ys.max()) + pad + 1),
    )


def mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> Image.Image:
    height, width = mask.shape
    source = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    array = np.asarray(source, dtype=np.float32)
    weight = np.asarray(mask, dtype=np.float32).clip(0.0, 1.0)[..., None]
    tinted = array * 0.72 + np.asarray(color, dtype=np.float32) * 0.28
    composed = array * (0.42 + 0.58 * weight) + tinted * weight * 0.42
    composed /= 1.0 + 0.42 * weight
    result = Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8), mode="RGB")

    binary = Image.fromarray((mask > 0.05).astype(np.uint8) * 255, mode="L")
    inset_w = max(80, width // 4)
    inset_h = max(108, height // 4)
    inset = contained(binary.convert("RGB"), (inset_w, inset_h), fill="black")
    draw = ImageDraw.Draw(inset)
    draw.rectangle((0, 0, inset_w - 1, inset_h - 1), outline="white", width=3)
    result.paste(inset, (width - inset_w - 10, height - inset_h - 10))
    return result


def condition_with_zoom(image: Image.Image, mask: np.ndarray) -> Image.Image:
    height, width = mask.shape
    full = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    if float(np.asarray(mask).mean()) <= 0.0:
        return full
    crop = full.crop(mask_bbox(mask, pad=max(8, width // 64)))
    inset_w = max(112, width // 3)
    inset_h = max(112, height // 3)
    inset = contained(crop, (inset_w, inset_h), fill="#e3e5e7")
    draw = ImageDraw.Draw(inset)
    draw.rectangle((0, 0, inset_w - 1, inset_h - 1), outline="#ffffff", width=4)
    draw.rectangle((5, 5, 62, 29), fill="#17191ccc")
    draw.text((11, 7), "zoom", font=font(14, bold=True), fill="white")
    full.paste(inset, (width - inset_w - 10, height - inset_h - 10))
    return full


def array_info(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    numeric = np.asarray(array, dtype=np.float32)
    finite = numeric[np.isfinite(numeric)]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "l2": float(np.linalg.norm(finite)) if finite.size else None,
    }


def read_cache(cache_path: Path, keys: tuple[str, ...]) -> dict[str, Any]:
    if not cache_path.is_file():
        raise FileNotFoundError(f"missing cache sample: {cache_path}")
    result: dict[str, Any] = {}
    with np.load(cache_path, allow_pickle=False) as payload:
        missing = [key for key in keys if key not in payload.files]
        if missing:
            raise KeyError(f"{cache_path} is missing cache keys: {missing}")
        for key in keys:
            result[key] = np.asarray(payload[key]).copy()
    return result


def shape_text(info: dict[str, Any]) -> str:
    return "x".join(str(value) for value in info["shape"])


def draw_metadata(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    record: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = box
    draw.rectangle(box, fill="#e7eaee", outline="#9ba4ae", width=2)
    mid_cache = record["cache"]["mid"]
    jid_cache = record["cache"]["jid"]
    pose_values = ", ".join(f"{value:+.3f}" for value in record["head_pose_token"])
    model_cfg = cfg["model"]
    prompt = str(cfg["data"]["prompt"])
    left_lines = [
        "ACTIVE ROUTING",
        f"mid -> pose ControlNet + garment reference + head-pose token",
        f"jid -> PuLID identity + CLIP appearance + hair reference",
        f"prompt: {prompt}",
        (
            f"spatial cache: pose={shape_text(mid_cache['pose_latents'])}; "
            f"garment={shape_text(mid_cache['garment_ref_latents'])}; "
            f"hair={shape_text(jid_cache['hair_ref_latents'])} -> {record['hair_tokens_used']} tokens"
        ),
    ]
    right_lines = [
        "NUMERIC / MODEL CONDITIONS",
        f"head_pose[7] = [{pose_values}]",
        (
            f"PuLID={shape_text(jid_cache['pulid_id_embed'])}; "
            f"appearance={shape_text(jid_cache['appearance'])}; "
            f"hair_empty={record['hair_ref_empty']}"
        ),
        (
            f"ControlNet mode={model_cfg['control_mode']} scale={model_cfg['controlnet_scale']}; "
            f"PuLID weight={model_cfg['pulid']['id_weight']}"
        ),
        "inactive: legacy garment CLIP tokens; legacy hair token route",
    ]
    column_gap = 24
    column_w = (right - left - column_gap - 36) // 2
    starts = (left + 18, left + 18 + column_w + column_gap)
    body_font = font(15)
    heading_font = font(15, bold=True)
    line_h = 27
    for column, lines in enumerate((left_lines, right_lines)):
        y = top + 13
        for index, line in enumerate(lines):
            current_font = heading_font if index == 0 else body_font
            wrapped = textwrap.wrap(line, width=92) or [""]
            for piece in wrapped:
                draw.text((starts[column], y), piece, font=current_font, fill="#20242a")
                y += line_h


def make_panel(
    record: dict[str, Any],
    images: dict[str, Image.Image],
    output: Path,
    cfg: dict[str, Any],
) -> None:
    cell_size = (384, 512)
    margin, gap = 18, 8
    title_h, label_h, row_gap, metadata_h = 72, 48, 12, 190
    width = margin * 2 + cell_size[0] * 5 + gap * 4
    rows_h = 2 * (label_h + cell_size[1]) + row_gap
    height = title_h + margin + rows_h + margin + metadata_h + margin
    canvas = Image.new("RGB", (width, height), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, title_h), fill="#dbe1e7")
    draw_centered(
        draw,
        (0, 0, width, title_h),
        (
            f"Fine-grained inference conditions  |  sample {record['order']:02d}  |  "
            f"mid={record['mid']}  jid={record['jid']}  seed={record['seed']}"
        ),
        font(24, bold=True),
    )

    rows = (
        (
            ("mannequin", f"Mannequin source\nm_i={record['mid']}"),
            ("pose", "Body/head pose\nControlNet input"),
            ("garment_overlay", "Garment mask\neroded mask + inset"),
            ("garment_ref", "Masked garment\nexact pre-VAE input"),
            ("generated", f"Generated output\nseed={record['seed']}"),
        ),
        (
            ("identity", f"Target identity\nc_j={record['jid']}"),
            ("hair_overlay", "Hair mask\nFASHN label 2 + inset"),
            ("hair_ref", "Reference hair cutout\nexact pre-VAE input"),
            ("appearance", "Appearance input\nface+hair -> CLIP"),
            ("pulid_face", "PuLID identity input\nface crop -> embed"),
        ),
    )
    y = title_h + margin
    row_colors = ("#bf6b35", "#27877f")
    for row_index, columns in enumerate(rows):
        for column_index, (key, label) in enumerate(columns):
            x = margin + column_index * (cell_size[0] + gap)
            draw.rectangle(
                (x, y, x + cell_size[0], y + label_h),
                fill="#ffffff",
                outline=row_colors[row_index],
                width=2,
            )
            draw_centered(
                draw,
                (x, y, x + cell_size[0], y + label_h),
                label,
                font(16, bold=True),
            )
            cell_y = y + label_h
            canvas.paste(contained(images[key], cell_size), (x, cell_y))
            draw.rectangle(
                (x, cell_y, x + cell_size[0] - 1, cell_y + cell_size[1] - 1),
                outline="#8c959f",
                width=2,
            )
        y += label_h + cell_size[1] + row_gap

    metadata_top = title_h + margin + rows_h + margin
    draw_metadata(
        canvas,
        (margin, metadata_top, width - margin, metadata_top + metadata_h),
        record,
        cfg,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def make_overview(rows: list[dict[str, Any]], output: Path) -> None:
    tile_size = (180, 240)
    title_h, label_h, gap, margin = 58, 30, 14, 16
    strip_w = tile_size[0] * 4
    strip_h = label_h + tile_size[1]
    columns = 2
    grid_rows = math.ceil(len(rows) / columns)
    width = margin * 2 + strip_w * columns + gap * (columns - 1)
    height = title_h + margin * 2 + strip_h * grid_rows + gap * (grid_rows - 1)
    canvas = Image.new("RGB", (width, height), "#eef0f2")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, title_h), fill="#d8dfe6")
    draw_centered(
        draw,
        (0, 0, width, title_h),
        "Random 10 active spatial conditions: pose | garment reference | hair reference | output",
        font(18, bold=True),
    )
    for index, row in enumerate(rows):
        grid_x = index % columns
        grid_y = index // columns
        left = margin + grid_x * (strip_w + gap)
        top = title_h + margin + grid_y * (strip_h + gap)
        draw.rectangle((left, top, left + strip_w, top + label_h), fill="white")
        draw_centered(
            draw,
            (left, top, left + strip_w, top + label_h),
            f"#{row['order']:02d}  mid={row['mid']}  jid={row['jid']}  seed={row['seed']}",
            font(13, bold=True),
        )
        for column, key in enumerate(("pose", "garment_ref", "hair_ref", "generated")):
            x = left + column * tile_size[0]
            image = contained(row["overview_images"][key], tile_size)
            canvas.paste(image, (x, top + label_h))
            draw.rectangle(
                (x, top + label_h, x + tile_size[0] - 1, top + strip_h - 1),
                outline="#9aa3ac",
                width=1,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def config_manifest_warning(cache_dir: Path, config_values: dict[str, Any]) -> str | None:
    manifest_path = cache_dir.parent / "manifest.json"
    if not manifest_path.is_file():
        return f"cache manifest missing: {manifest_path}"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cached = payload.get("reference_preprocessing", {})
    comparable = {
        "neutral_gray": int(cached.get("neutral_gray", -1)),
        "mask_erosion_px": int(cached.get("mask_erosion_px", -1)),
        "hair_ref_min_area_fraction": float(
            cached.get("hair_ref_min_area_fraction", -1.0)
        ),
    }
    if comparable != config_values:
        return (
            "cache manifest reference_preprocessing is stale: "
            f"manifest={comparable}, final resolved config={config_values}. "
            "Panels use the final run's resolved config, which is authoritative for the "
            "quality-repair cache rebuild."
        )
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize every active image/identity condition for a fixed sample manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/spatial_quality_repair_random10/manifest.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/"
            "phase1_spatial_quality_repair_r16_6400_768x1024/resolved_config.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/spatial_quality_repair_random10/conditions"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    selection = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = selection.get("samples", [])
    if not rows:
        raise RuntimeError(f"sample manifest contains no samples: {args.manifest}")

    root = Path(cfg["data"]["root"])
    width, height = get_resolution(cfg["data"]["resolution"])
    cache_dir = root / cfg["data"]["cache_dir"] / "samples"
    spatial_cfg = cfg["model"]["spatial_conditions"]
    pose_folder = root / spatial_cfg["head_control"]["real_source"]
    ref_gray = int(cfg["cache"].get("reference_neutral_gray", 127))
    erosion = int(cfg["cache"].get("reference_mask_erosion_px", 0))
    hair_min = float(cfg["cache"].get("hair_ref_min_area_fraction", 0.01))
    appearance_gray = int(cfg["cache"].get("neutral_gray", 127))
    appearance_pad = float(cfg["cache"].get("appearance_bbox_pad_fraction", 0.08))
    hair_label = int(cfg["cache"].get("hair_label", 2))
    hair_stride = int(spatial_cfg["hair"]["reference"]["stride"])
    hair_tokens_used = math.ceil((height // 16) / hair_stride) * math.ceil(
        (width // 16) / hair_stride
    )

    prepared: list[dict[str, Any]] = []
    for source_row in rows:
        mid = str(source_row["mid"])
        jid = str(source_row["jid"])
        generated_path = Path(str(source_row["generated"]))
        mannequin_path = find_one(root / "images/mannequin", mid)
        identity_path = find_one(root / "images/human", jid)
        pose_path = find_one(pose_folder, mid)
        pulid_face_path = find_one(root / "derived/face_crops/human", jid)
        if not generated_path.is_file():
            raise FileNotFoundError(f"missing generated output: {generated_path}")

        mid_values = read_cache(
            cache_dir / f"{mid}.npz",
            ("pose_latents", "garment_ref_latents", "head_pose"),
        )
        jid_values = read_cache(
            cache_dir / f"{jid}.npz",
            ("pulid_id_embed", "appearance", "hair_ref_latents", "hair_ref_empty"),
        )
        mannequin = open_rgb(mannequin_path)
        identity = open_rgb(identity_path)
        garment_ref, garment_mask = masked_garment_image(
            root,
            mid,
            cfg["data"]["resolution"],
            neutral_gray=ref_gray,
            erosion_px=erosion,
        )
        hair_ref, hair_mask, hair_empty_rebuilt = masked_hair_image(
            root,
            jid,
            cfg["data"]["resolution"],
            neutral_gray=ref_gray,
            hair_label=hair_label,
            min_area_fraction=hair_min,
            erosion_px=erosion,
        )
        appearance, _ = face_hair_appearance_crop(
            root,
            jid,
            neutral_gray=appearance_gray,
            bbox_pad_fraction=appearance_pad,
        )
        cached_empty = int(np.asarray(jid_values["hair_ref_empty"]).reshape(-1)[0])
        if cached_empty != int(hair_empty_rebuilt):
            raise RuntimeError(
                f"hair_ref_empty mismatch for jid={jid}: cache={cached_empty}, "
                f"reconstructed={int(hair_empty_rebuilt)}"
            )

        visual_images = {
            "mannequin": mannequin,
            "pose": open_rgb(pose_path),
            "garment_overlay": mask_overlay(mannequin, garment_mask, (255, 137, 46)),
            "garment_ref": condition_with_zoom(garment_ref, garment_mask),
            "generated": open_rgb(generated_path),
            "identity": identity,
            "hair_overlay": mask_overlay(identity, hair_mask, (18, 190, 181)),
            "hair_ref": condition_with_zoom(hair_ref, hair_mask),
            "appearance": appearance,
            "pulid_face": open_rgb(pulid_face_path),
        }
        record: dict[str, Any] = {
            "order": int(source_row["order"]),
            "mid": mid,
            "jid": jid,
            "seed": int(source_row["seed"]),
            "paths": {
                "mannequin": str(mannequin_path),
                "identity": str(identity_path),
                "pose_control": str(pose_path),
                "garment_mask": str(find_one(root / "clothes_bySAM/masks/human", mid)),
                "hair_parsing": str(
                    find_one(root / "human_parsing/fashn/masks/human", jid)
                ),
                "pulid_face_crop": str(pulid_face_path),
                "generated": str(generated_path),
                "mid_cache": str(cache_dir / f"{mid}.npz"),
                "jid_cache": str(cache_dir / f"{jid}.npz"),
            },
            "cache": {
                "mid": {key: array_info(value) for key, value in mid_values.items()},
                "jid": {key: array_info(value) for key, value in jid_values.items()},
            },
            "head_pose_token": [
                float(value) for value in np.asarray(mid_values["head_pose"]).reshape(-1)
            ],
            "garment_mask_fraction": float(garment_mask.mean()),
            "hair_mask_fraction": float(hair_mask.mean()),
            "hair_ref_empty": cached_empty,
            "hair_tokens_used": hair_tokens_used,
            "overview_images": {
                "pose": visual_images["pose"],
                "garment_ref": visual_images["garment_ref"],
                "hair_ref": visual_images["hair_ref"],
                "generated": visual_images["generated"],
            },
        }
        panel_path = args.output_dir / (
            f"conditions_{record['order']:02d}__mid{mid}__jid{jid}__seed{record['seed']}.png"
        )
        record["panel"] = str(panel_path.resolve())
        record["_images"] = visual_images
        prepared.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for record in prepared:
        make_panel(record, record.pop("_images"), Path(record["panel"]), cfg)

    overview_path = args.output_dir / "overview_conditions_random10.png"
    make_overview(prepared, overview_path)
    for record in prepared:
        record.pop("overview_images", None)

    config_values = {
        "neutral_gray": ref_gray,
        "mask_erosion_px": erosion,
        "hair_ref_min_area_fraction": hair_min,
    }
    provenance_warning = config_manifest_warning(cache_dir, config_values)
    output_manifest = {
        "source_selection_manifest": str(args.manifest.resolve()),
        "resolved_config": str(args.config.resolve()),
        "selection_seed": selection.get("selection_seed"),
        "selected_count": len(prepared),
        "resolution": {"width": width, "height": height},
        "routing": {
            "from_mid": [
                "pose_latents",
                "garment_ref_latents",
                "head_pose",
            ],
            "from_jid": [
                "pulid_id_embed",
                "appearance",
                "hair_ref_latents",
            ],
            "inactive": ["legacy garment CLIP tokens", "legacy hair token route"],
        },
        "reference_preprocessing": config_values,
        "appearance_preprocessing": {
            "neutral_gray": appearance_gray,
            "labels": [1, 2],
            "meaning": "face+hair only, then pooled CLIP",
        },
        "pose_control": {
            "folder": str(pose_folder),
            "control_mode": cfg["model"]["control_mode"],
            "conditioning_scale": cfg["model"]["controlnet_scale"],
        },
        "hair_reference_tokens_used": hair_tokens_used,
        "overview": str(overview_path.resolve()),
        "provenance_warning": provenance_warning,
        "samples": prepared,
    }
    manifest_path = args.output_dir / "manifest_conditions.json"
    manifest_path.write_text(
        json.dumps(output_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Fine-grained Inference Conditions: Random 10",
        "",
        f"- Source selection: `{args.manifest.resolve()}`",
        f"- Final resolved config: `{args.config.resolve()}`",
        f"- Selection seed: `{selection.get('selection_seed')}` (the original ten samples are unchanged)",
        f"- Overview: [overview_conditions_random10.png]({overview_path.name})",
        f"- Machine-readable provenance: [manifest_conditions.json]({manifest_path.name})",
        "",
        "## What Each Panel Shows",
        "",
        "Top row (mannequin-side): mannequin source, with-head DWPose ControlNet image, exact eroded garment mask, exact masked garment image before VAE encoding, and the existing generated output.",
        "",
        "Bottom row (identity-side): target person, exact FASHN hair mask, exact masked hair image before VAE encoding, exact face+hair-only crop before pooled CLIP, and the face crop used to build the PuLID embedding.",
        "",
        "The bottom metadata band reports the actual cached tensor shapes and numeric head-pose token. Latent tensors and embeddings are not RGB images; their deterministic source images are shown instead.",
        "",
        "`region_masks_z` is not an inference condition and is intentionally absent. Legacy garment CLIP tokens and legacy hair tokens are disabled in this run.",
        "",
        "## Preprocessing",
        "",
        f"- Garment/hair reference background: RGB `{ref_gray}`; mask erosion: `{erosion}px`.",
        f"- Hair label: FASHN `{hair_label}`; empty threshold: `{hair_min:.3%}`; model stride: `{hair_stride}` (`{hair_tokens_used}` tokens).",
        f"- Appearance crop background: RGB `{appearance_gray}`; retained labels: face=1 and hair=2.",
        f"- ControlNet: mode `{cfg['model']['control_mode']}`, scale `{cfg['model']['controlnet_scale']}`.",
        "",
    ]
    if provenance_warning:
        lines.extend(
            [
                "## Provenance Warning",
                "",
                provenance_warning,
                "",
            ]
        )
    lines.extend(
        [
            "## Samples",
            "",
            "| # | mid | jid | seed | garment area | hair area | panel |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for record in prepared:
        panel_name = Path(record["panel"]).name
        lines.append(
            f"| {record['order']} | `{record['mid']}` | `{record['jid']}` | "
            f"{record['seed']} | {record['garment_mask_fraction']:.2%} | "
            f"{record['hair_mask_fraction']:.2%} | [{panel_name}]({panel_name}) |"
        )
    readme_path = args.output_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "panels": len(prepared),
                "overview": str(overview_path.resolve()),
                "manifest": str(manifest_path.resolve()),
                "readme": str(readme_path.resolve()),
                "provenance_warning": provenance_warning,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
