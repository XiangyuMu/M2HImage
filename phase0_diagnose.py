#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


REGION_KEYS = ("id_strong", "id_weak", "cloth", "cloth_safe", "edge", "body_bg")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
DEFAULT_PROMPT = (
    "a photorealistic human wearing the same garment, same body pose, "
    "natural skin and hair, high quality"
)


@dataclass(frozen=True)
class SampleInfo:
    sample_id: str
    garment_type: str
    yaw_bucket: str
    yaw_abs: float | None
    yaw: float | None
    pitch: float | None
    roll: float | None
    status: str
    cloth_safe_ratio: float
    edge_ratio: float
    id_strong_ratio: float


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_one(folder: Path, sample_id: str) -> Path:
    for ext in IMAGE_EXTS:
        path = folder / f"{sample_id}{ext}"
        if path.exists():
            return path
    matches = sorted(folder.glob(f"{sample_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"missing file for id={sample_id} under {folder}")


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_l(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def load_label_metadata(root: Path) -> tuple[dict[int, str], dict[str, int]]:
    path = root / "human_parsing/fashn/metadata/labels.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    labels = {int(key): str(value) for key, value in payload["labels"].items()}
    name_to_id = {value: key for key, value in labels.items()}
    return labels, name_to_id


def print_probe(root: Path, labels: dict[int, str]) -> None:
    print("=== Step 0 probe: labels.json ===")
    for label_id in sorted(labels):
        print(f"{label_id:>2}: {labels[label_id]}")

    probe_id = first_probe_id(root)
    print(f"=== Step 0 probe: sample {probe_id} ===")
    npz_path = root / "derived/region_masks" / f"{probe_id}.npz"
    with np.load(npz_path) as npz:
        print(f"region npz: {npz_path}")
        print(f"keys: {list(npz.files)}")
        for key in npz.files:
            arr = npz[key]
            values = np.unique(arr)
            preview = values[:12].tolist()
            print(
                f"{key}: shape={arr.shape} dtype={arr.dtype} "
                f"min={arr.min()} max={arr.max()} unique_head={preview}"
            )

    pose_path = root / "derived/head_pose_6drepnet/human" / f"{probe_id}.json"
    with pose_path.open("r", encoding="utf-8") as handle:
        pose = json.load(handle)
    print(f"head pose json: {pose_path}")
    print(json.dumps(pose, indent=2, ensure_ascii=False))


def first_probe_id(root: Path) -> str:
    for rel in ("splits/test.txt", "metadata/sample_ids.txt"):
        path = root / rel
        if path.exists():
            for sample_id in read_lines(path):
                if (root / "derived/region_masks" / f"{sample_id}.npz").exists():
                    return sample_id
    matches = sorted((root / "derived/region_masks").glob("*.npz"))
    if not matches:
        raise FileNotFoundError("no region mask npz files found")
    return matches[0].stem


def load_region_masks(root: Path, sample_id: str) -> dict[str, np.ndarray]:
    path = root / "derived/region_masks" / f"{sample_id}.npz"
    with np.load(path) as npz:
        missing = [key for key in REGION_KEYS if key not in npz.files]
        if missing:
            raise KeyError(f"{path} missing keys: {missing}; found={list(npz.files)}")
        return {key: np.asarray(npz[key]) > 127 for key in REGION_KEYS}


def load_head_pose(root: Path, sample_id: str) -> dict[str, Any]:
    path = root / "derived/head_pose_6drepnet/human" / f"{sample_id}.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_parsing(root: Path, split: str, sample_id: str) -> np.ndarray:
    path = find_one(root / "human_parsing/fashn/masks" / split, sample_id)
    return np.asarray(load_l(path), dtype=np.uint8)


def bool_to_l(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def mask_ratio(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else 0.0


def label_mask(label_map: np.ndarray, name_to_id: dict[str, int], names: Iterable[str]) -> np.ndarray:
    ids = [name_to_id[name] for name in names if name in name_to_id]
    if not ids:
        return np.zeros(label_map.shape, dtype=bool)
    return np.isin(label_map, ids)


def classify_garment(label_map: np.ndarray, name_to_id: dict[str, int]) -> str:
    areas = {
        "top": int(label_mask(label_map, name_to_id, ("top",)).sum()),
        "dress": int(label_mask(label_map, name_to_id, ("dress",)).sum()),
        "pants": int(label_mask(label_map, name_to_id, ("pants",)).sum()),
        "skirt": int(label_mask(label_map, name_to_id, ("skirt",)).sum()),
    }
    total = max(1, label_map.size)
    if areas["dress"] / total > 0.01 and areas["dress"] >= max(areas["top"], areas["pants"], areas["skirt"]) * 0.5:
        return "dress"
    if areas["pants"] / total > 0.015:
        return "pants"
    if areas["skirt"] / total > 0.01:
        return "skirt"
    if areas["top"] / total > 0.01:
        return "top"
    return "other"


def yaw_bucket(yaw_abs: float | None) -> str:
    if yaw_abs is None:
        return "unknown"
    if yaw_abs < 15:
        return "front"
    if yaw_abs < 35:
        return "three_quarter"
    return "side"


def get_sample_info(root: Path, sample_id: str, name_to_id: dict[str, int]) -> SampleInfo:
    parsing = load_parsing(root, "human", sample_id)
    regions = load_region_masks(root, sample_id)
    pose = load_head_pose(root, sample_id)
    status = str(pose.get("status", "unknown"))
    yaw = as_float(pose.get("yaw"))
    pitch = as_float(pose.get("pitch"))
    roll = as_float(pose.get("roll"))
    yaw_abs = abs(yaw) if yaw is not None and status == "ok" else None
    return SampleInfo(
        sample_id=sample_id,
        garment_type=classify_garment(parsing, name_to_id),
        yaw_bucket=yaw_bucket(yaw_abs),
        yaw_abs=yaw_abs,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        status=status,
        cloth_safe_ratio=mask_ratio(regions["cloth_safe"]),
        edge_ratio=mask_ratio(regions["edge"]),
        id_strong_ratio=mask_ratio(regions["id_strong"]),
    )


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def valid_for_phase0(root: Path, sample_id: str) -> bool:
    required = [
        root / "derived/region_masks" / f"{sample_id}.npz",
        root / "derived/head_pose_6drepnet/human" / f"{sample_id}.json",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        find_one(root / "images/human", sample_id)
        find_one(root / "images/mannequin", sample_id)
        find_one(root / "human_parsing/fashn/masks/human", sample_id)
        find_one(root / "human_parsing/fashn/masks/mannequin", sample_id)
    except FileNotFoundError:
        return False
    return True


def select_sample_ids(
    root: Path,
    out_dir: Path,
    n: int,
    seed: int,
    name_to_id: dict[str, int],
    *,
    resample: bool,
) -> tuple[list[str], dict[str, SampleInfo]]:
    sample_path = out_dir / "sample_ids.txt"
    manifest_path = out_dir / "sample_manifest.json"
    if sample_path.exists() and not resample:
        sample_ids = read_lines(sample_path)
        infos = {}
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for row in payload:
                infos[row["sample_id"]] = SampleInfo(**row)
        else:
            for sample_id in sample_ids:
                infos[sample_id] = get_sample_info(root, sample_id, name_to_id)
        print(f"Using existing sample list: {sample_path}")
        return sample_ids, infos

    test_path = root / "splits/test.txt"
    if not test_path.exists():
        raise FileNotFoundError(f"missing split file: {test_path}")
    candidates = [sample_id for sample_id in read_lines(test_path) if valid_for_phase0(root, sample_id)]
    if len(candidates) < n:
        raise RuntimeError(f"only {len(candidates)} valid test samples found, need n={n}")

    infos = {sample_id: get_sample_info(root, sample_id, name_to_id) for sample_id in candidates}
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[str]] = {}
    for sample_id, info in infos.items():
        groups.setdefault((info.garment_type, info.yaw_bucket), []).append(sample_id)
    for values in groups.values():
        rng.shuffle(values)

    garment_order = ("top", "dress", "pants", "skirt", "other")
    yaw_order = ("front", "side", "three_quarter", "unknown")
    keys = [(g, y) for g in garment_order for y in yaw_order if groups.get((g, y))]

    selected: list[str] = []
    while keys and len(selected) < n:
        progressed = False
        for key in list(keys):
            bucket = groups[key]
            while bucket and bucket[0] in selected:
                bucket.pop(0)
            if not bucket:
                keys.remove(key)
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) == n:
                break
        if not progressed:
            break

    if len(selected) < n:
        remaining = [sample_id for sample_id in candidates if sample_id not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    selected_infos = {sample_id: infos[sample_id] for sample_id in selected}
    ensure_dir(out_dir)
    sample_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump([info.__dict__ for info in selected_infos.values()], handle, indent=2)

    print(f"Wrote selected ids: {sample_path}")
    for sample_id in selected:
        info = selected_infos[sample_id]
        yaw_text = "NA" if info.yaw is None else f"{info.yaw:.1f}"
        print(f"  {sample_id}: garment={info.garment_type} yaw={yaw_text} bucket={info.yaw_bucket}")
    return selected, selected_infos


def resize_array_nearest(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = bool_to_l(mask).resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def resize_to_height(image: Image.Image, height: int, *, is_mask: bool = False) -> Image.Image:
    width = max(1, round(image.width * height / image.height))
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    return image.resize((width, height), resample)


def resize_for_generation(image: Image.Image, resolution: int, *, is_mask: bool = False) -> Image.Image:
    if resolution <= 0:
        return image
    scale = resolution / max(image.width, image.height)
    width = max(64, int(round(image.width * scale / 16) * 16))
    height = max(64, int(round(image.height * scale / 16) * 16))
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    return image.resize((width, height), resample)


def soft_alpha(mask: np.ndarray, radius: int) -> np.ndarray:
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)
    image = bool_to_l(mask).filter(ImageFilter.GaussianBlur(radius=max(1, radius)))
    return np.asarray(image, dtype=np.float32) / 255.0


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expanded_head_prior(head_core: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    bbox = bbox_from_mask(head_core)
    if bbox is None:
        return np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    ex0 = max(0, int(round(x0 - 0.45 * bw)))
    ex1 = min(width, int(round(x1 + 0.45 * bw)))
    ey0 = max(0, int(round(y0 - 0.95 * bh)))
    ey1 = min(height, int(round(y1 + 0.45 * bh)))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((ex0, ey0, ex1, ey1), fill=255)
    dilated = bool_to_l(head_core).filter(ImageFilter.MaxFilter(25))
    return (np.asarray(mask) > 127) | (np.asarray(dilated) > 127)


def build_source_canvas(
    mannequin: Image.Image,
    human_parsing: np.ndarray,
    mannequin_parsing: np.ndarray,
    regions: dict[str, np.ndarray],
    name_to_id: dict[str, int],
    *,
    blur_sigma: float,
) -> tuple[Image.Image, dict[str, np.ndarray]]:
    size = mannequin.size
    width, height = size
    region_resized = {
        key: resize_array_nearest(value, size) if value.shape != (height, width) else value
        for key, value in regions.items()
    }
    parse_resized = (
        np.asarray(Image.fromarray(human_parsing).resize(size, Image.Resampling.NEAREST))
        if human_parsing.shape != (height, width)
        else human_parsing
    )
    man_parse_resized = (
        np.asarray(Image.fromarray(mannequin_parsing).resize(size, Image.Resampling.NEAREST))
        if mannequin_parsing.shape != (height, width)
        else mannequin_parsing
    )

    parsing_head = label_mask(parse_resized, name_to_id, ("face", "hair"))
    mannequin_head = label_mask(man_parse_resized, name_to_id, ("face", "hair"))
    head_core = parsing_head | mannequin_head | region_resized["id_strong"]
    if not head_core.any():
        head_core = region_resized["id_strong"] | parsing_head
    head_prior = expanded_head_prior(head_core, size)

    # In the probed M2H_Final_v2 masks, body_bg is a background-like mask
    # (0/255, roughly matching parsing background). The foreground is its inverse.
    person = ~region_resized["body_bg"]
    person |= region_resized["id_strong"]
    person |= region_resized["id_weak"]
    person |= region_resized["cloth"]
    person |= region_resized["cloth_safe"]
    person |= region_resized["edge"]
    person |= head_prior
    body = person & ~head_prior
    background = ~person

    source = np.asarray(mannequin.convert("RGB"), dtype=np.float32)
    lowpass = np.asarray(
        mannequin.convert("RGB").filter(ImageFilter.GaussianBlur(radius=max(0.1, blur_sigma))),
        dtype=np.float32,
    )

    canvas = source.copy()
    body_alpha = soft_alpha(body, radius=max(2, int(round(blur_sigma / 2))))
    canvas = canvas * (1.0 - body_alpha[..., None]) + lowpass * body_alpha[..., None]

    # The head area is deliberately not copied from the mannequin head contour.
    # A soft, upward-expanded oval gives the fill model a weak location prior
    # without preserving fake-head edges or facial texture.
    head_alpha = soft_alpha(head_prior, radius=max(4, int(round(blur_sigma))))
    bg_pixels = source[region_resized["body_bg"]]
    if len(bg_pixels):
        neutral = np.median(bg_pixels, axis=0)
    else:
        neutral = np.array([220, 220, 220])
    neutral = 0.9 * neutral + 0.1 * np.array([205, 190, 175])
    neutral_img = np.empty_like(lowpass)
    neutral_img[:, :] = neutral
    placeholder = neutral_img
    canvas = canvas * (1.0 - head_alpha[..., None]) + placeholder * head_alpha[..., None]

    canvas[background] = source[background]
    debug_masks = {
        "person": person,
        "body": body,
        "head_core": head_core,
        "head_prior": head_prior,
        "background": background,
    }
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB"), debug_masks


def build_fill_mask(regions: dict[str, np.ndarray], size: tuple[int, int], *, dilate_radius: int) -> Image.Image:
    width, height = size
    region_resized = {
        key: resize_array_nearest(value, size) if value.shape != (height, width) else value
        for key, value in regions.items()
    }
    mask = ~region_resized["body_bg"]
    for key in ("id_strong", "id_weak", "cloth", "cloth_safe", "edge"):
        mask |= region_resized[key]
    image = bool_to_l(mask)
    if dilate_radius > 0:
        image = image.filter(ImageFilter.MaxFilter(dilate_radius * 2 + 1))
    return image.point(lambda value: 255 if value > 127 else 0)


def overlay_masks(base: Image.Image, masks: dict[str, np.ndarray]) -> Image.Image:
    colors = {
        "body_bg": (120, 120, 120, 55),
        "cloth_safe": (0, 220, 80, 95),
        "edge": (0, 80, 255, 115),
        "id_weak": (255, 230, 0, 100),
        "id_strong": (255, 0, 0, 125),
    }
    result = base.convert("RGBA")
    width, height = base.size
    for key in ("body_bg", "cloth_safe", "edge", "id_weak", "id_strong"):
        mask = masks[key]
        if mask.shape != (height, width):
            mask = resize_array_nearest(mask, (width, height))
        rgba = Image.new("RGBA", base.size, colors[key])
        alpha = bool_to_l(mask).point(lambda value, a=colors[key][3]: a if value > 127 else 0)
        rgba.putalpha(alpha)
        result = Image.alpha_composite(result, rgba)
    return result.convert("RGB")


def rotation_matrix_6drepnet(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y = math.radians(yaw)
    p = math.radians(pitch)
    r = math.radians(roll)
    rx = np.array(
        [[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]],
        dtype=np.float64,
    )
    ry = np.array(
        [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]],
        dtype=np.float64,
    )
    rz = np.array(
        [[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def draw_head_axes(base: Image.Image, pose: dict[str, Any]) -> Image.Image:
    image = base.copy()
    draw = ImageDraw.Draw(image)
    status = str(pose.get("status", "unknown"))
    bbox = pose.get("bbox")
    yaw = as_float(pose.get("yaw"))
    pitch = as_float(pose.get("pitch"))
    roll = as_float(pose.get("roll"))
    if not isinstance(bbox, list) or len(bbox) < 4 or yaw is None or pitch is None or roll is None:
        draw.text((10, 10), f"head pose missing/status={status}", fill=(255, 0, 0))
        return image
    x, y, w, h = [float(value) for value in bbox[:4]]
    center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)
    length = max(35.0, min(140.0, max(w, h) * 1.25))
    rot = rotation_matrix_6drepnet(yaw=yaw, pitch=pitch, roll=roll)
    axes = {
        "X": (rot[:2, 0], (255, 0, 0)),
        "Y": (rot[:2, 1], (0, 220, 0)),
        "Z": (rot[:2, 2], (0, 80, 255)),
    }
    draw.rectangle((x, y, x + w, y + h), outline=(255, 255, 255), width=2)
    draw.rectangle((x, y, x + w, y + h), outline=(0, 0, 0), width=1)
    for label, (direction, color) in axes.items():
        end = center + direction * length
        draw.line((center[0], center[1], end[0], end[1]), fill=color, width=5)
        draw.text((end[0] + 3, end[1] + 3), label, fill=color)
    draw.text(
        (10, 10),
        f"yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f} status={status}",
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    return image


def font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def titled_panel(image: Image.Image, title: str, height: int) -> Image.Image:
    body = resize_to_height(image, height)
    title_h = 34
    panel = Image.new("RGB", (body.width, body.height + title_h), (245, 245, 245))
    panel.paste(body, (0, title_h))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, title_h), fill=(30, 30, 30))
    draw.text((8, 7), title, fill=(255, 255, 255), font=font())
    return panel


def hstack_panels(panels: list[Image.Image]) -> Image.Image:
    height = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def make_grid(images: list[Image.Image], cols: int, pad: int = 8, bg: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), bg)
    widths = [image.width for image in images]
    heights = [image.height for image in images]
    cell_w = max(widths)
    cell_h = max(heights)
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * cell_w + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad), bg)
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        x = pad + col * (cell_w + pad) + (cell_w - image.width) // 2
        y = pad + row * (cell_h + pad) + (cell_h - image.height) // 2
        grid.paste(image, (x, y))
    return grid


def write_error(out_dir: Path, task: str, sample_id: str | None, exc: BaseException) -> None:
    ensure_dir(out_dir)
    with (out_dir / "errors.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_text()}] task={task} id={sample_id or '-'} error={exc}\n")
        handle.write(traceback.format_exc())
        handle.write("\n")


def preprocess_one(
    root: Path,
    out_dir: Path,
    sample_id: str,
    name_to_id: dict[str, int],
    *,
    blur_sigma: float,
    panel_height: int,
    overwrite: bool,
) -> Image.Image:
    preprocess_dir = out_dir / "preprocess"
    source_dir = out_dir / "source_canvas"
    fill_mask_dir = out_dir / "fill_masks"
    ensure_dir(preprocess_dir)
    ensure_dir(source_dir)
    ensure_dir(fill_mask_dir)

    panel_path = preprocess_dir / f"{sample_id}.png"
    if panel_path.exists() and not overwrite:
        return load_rgb(panel_path)

    mannequin = load_rgb(find_one(root / "images/mannequin", sample_id))
    human = load_rgb(find_one(root / "images/human", sample_id))
    regions = load_region_masks(root, sample_id)
    human_parsing = load_parsing(root, "human", sample_id)
    mannequin_parsing = load_parsing(root, "mannequin", sample_id)
    pose = load_head_pose(root, sample_id)

    source_canvas, _ = build_source_canvas(
        mannequin, human_parsing, mannequin_parsing, regions, name_to_id, blur_sigma=blur_sigma
    )
    overlay = overlay_masks(human, regions)
    axes = draw_head_axes(human, pose)
    fill_mask = build_fill_mask(regions, mannequin.size, dilate_radius=5)

    source_canvas.save(source_dir / f"{sample_id}.png")
    fill_mask.save(fill_mask_dir / f"{sample_id}.png")

    panels = [
        titled_panel(mannequin, "m_i", panel_height),
        titled_panel(human, "h_i ref", panel_height),
        titled_panel(source_canvas, "canvas", panel_height),
        titled_panel(overlay, "masks", panel_height),
        titled_panel(axes, "pose", panel_height),
    ]
    output = hstack_panels(panels)
    output.save(panel_path)
    return output


def write_preprocess_checklist(out_dir: Path, sample_ids: list[str], infos: dict[str, SampleInfo]) -> None:
    path = out_dir / "preprocess_checklist.md"
    lines = [
        "# Phase 0 Preprocess Checklist",
        "",
        "Inspect `preprocess_grid.png` first, then open individual panels under `preprocess/`.",
        "",
        "Global checks:",
        "- Source canvas: no sharp mannequin plastic face/head contour remains.",
        "- Source canvas: background is preserved from `m_i`; body area is low-pass only.",
        "- Source canvas: head prior is soft, upward-expanded, and leaves room for hair.",
        "- Mask overlay: `cloth_safe` avoids identity, hairline, neck/shoulder identity zones.",
        "- Mask overlay: `edge` is a plausible garment boundary band, not random speckle.",
        "- Mask overlay: `id_strong` covers face plus hair.",
        "- Head pose: axes follow 6DRepNet convention `Rz(roll)@Ry(yaw)@Rx(pitch)` with image y-down.",
        "",
        "| id | garment | yaw | bucket | cloth_safe | edge | canvas ok | masks ok | axes ok | notes |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for sample_id in sample_ids:
        info = infos[sample_id]
        yaw = "NA" if info.yaw is None else f"{info.yaw:.1f}"
        lines.append(
            f"| {sample_id} | {info.garment_type} | {yaw} | {info.yaw_bucket} | "
            f"{info.cloth_safe_ratio:.3f} | {info.edge_ratio:.3f} | [ ] | [ ] | [ ] | |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote checklist: {path}")


def run_preprocess(args: argparse.Namespace, root: Path, out_dir: Path, name_to_id: dict[str, int]) -> None:
    sample_ids, infos = select_sample_ids(
        root, out_dir, args.n, args.seed, name_to_id, resample=args.resample
    )
    panels = []
    for sample_id in sample_ids:
        try:
            print(f"[preprocess] {sample_id}")
            panel = preprocess_one(
                root,
                out_dir,
                sample_id,
                name_to_id,
                blur_sigma=args.blur_sigma,
                panel_height=args.resolution,
                overwrite=args.overwrite,
            )
            panels.append(resize_to_height(panel, max(160, args.resolution // 2)))
        except Exception as exc:  # noqa: BLE001
            print(f"[preprocess] failed {sample_id}: {exc}", file=sys.stderr)
            write_error(out_dir, "preprocess", sample_id, exc)

    if panels:
        grid = make_grid(panels, cols=1)
        grid.save(out_dir / "preprocess_grid.png")
        print(f"Wrote preprocess grid: {out_dir / 'preprocess_grid.png'}")
    write_preprocess_checklist(out_dir, sample_ids, infos)


def require_fill_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from diffusers import FluxFillPipeline
    except ImportError as exc:
        raise RuntimeError(
            "FLUX Fill requires a recent diffusers build with FluxFillPipeline. "
            "Install/upgrade dependencies, for example: "
            "pip install -U diffusers transformers accelerate sentencepiece protobuf bitsandbytes"
        ) from exc
    return torch, FluxFillPipeline, None


def load_flux_fill_pipeline(args: argparse.Namespace) -> Any:
    torch, FluxFillPipeline, _ = require_fill_dependencies()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs: dict[str, Any] = {"torch_dtype": dtype}

    if args.load_4bit or args.load_8bit:
        try:
            from diffusers import FluxTransformer2DModel
            from transformers import BitsAndBytesConfig, T5EncoderModel
        except ImportError as exc:
            raise RuntimeError(
                "--load-4bit/--load-8bit requires transformers bitsandbytes integration and "
                "a diffusers version exposing FluxTransformer2DModel."
            ) from exc
        if args.load_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        transformer = FluxTransformer2DModel.from_pretrained(
            args.model_id,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=dtype,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            args.model_id,
            subfolder="text_encoder_2",
            quantization_config=quant_config,
            torch_dtype=dtype,
        )
        kwargs["transformer"] = transformer
        kwargs["text_encoder_2"] = text_encoder_2

    pipe = FluxFillPipeline.from_pretrained(args.model_id, **kwargs)
    if torch.cuda.is_available():
        if args.sequential_offload and hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        elif hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
    else:
        pipe.to("cpu")
    return pipe


def run_pipe_once(
    pipe: Any,
    args: argparse.Namespace,
    image: Image.Image,
    mask: Image.Image,
    seed: int,
    resolution: int,
) -> Image.Image:
    torch = sys.modules["torch"]
    gen_image = resize_for_generation(image, resolution)
    gen_mask = resize_for_generation(mask, resolution, is_mask=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)
    return pipe(
        prompt=args.prompt,
        image=gen_image,
        mask_image=gen_mask,
        height=gen_image.height,
        width=gen_image.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def ensure_source_and_mask(
    root: Path,
    out_dir: Path,
    sample_id: str,
    name_to_id: dict[str, int],
    args: argparse.Namespace,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    source_path = out_dir / "source_canvas" / f"{sample_id}.png"
    mask_path = out_dir / "fill_masks" / f"{sample_id}.png"
    if not source_path.exists() or not mask_path.exists():
        preprocess_one(
            root,
            out_dir,
            sample_id,
            name_to_id,
            blur_sigma=args.blur_sigma,
            panel_height=args.resolution,
            overwrite=False,
        )
    mannequin = load_rgb(find_one(root / "images/mannequin", sample_id))
    source = load_rgb(source_path)
    mask = load_l(mask_path)
    return mannequin, source, mask


def fill_modes(args: argparse.Namespace) -> list[str]:
    if args.fill_inputs == "both":
        return ["source_canvas", "mannequin"]
    return [args.fill_inputs]


def run_fill_one(
    pipe: Any,
    root: Path,
    out_dir: Path,
    sample_id: str,
    name_to_id: dict[str, int],
    args: argparse.Namespace,
) -> Image.Image:
    fill_dir = out_dir / "fill_sanity"
    output_dir = fill_dir / "outputs"
    ensure_dir(fill_dir)
    ensure_dir(output_dir)
    grid_path = fill_dir / f"{sample_id}.png"
    if grid_path.exists() and not args.overwrite:
        return load_rgb(grid_path)

    mannequin, source, mask = ensure_source_and_mask(root, out_dir, sample_id, name_to_id, args)
    panels = [
        titled_panel(mannequin, "m_i", args.resolution),
        titled_panel(source, "canvas", args.resolution),
    ]

    seeds = [args.seed + i for i in range(args.num_seeds)]
    for mode in fill_modes(args):
        init_image = source if mode == "source_canvas" else mannequin
        for seed in seeds:
            output_path = output_dir / f"{sample_id}_{mode}_seed{seed}.png"
            if output_path.exists() and not args.overwrite:
                result = load_rgb(output_path)
            else:
                try:
                    result = run_pipe_once(pipe, args, init_image, mask, seed, args.resolution)
                except RuntimeError as exc:
                    if not is_oom(exc):
                        raise
                    torch = sys.modules["torch"]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(
                        f"[fill] OOM for {sample_id} mode={mode} seed={seed}; "
                        "retrying at resolution=512 with offload/batch=1",
                        file=sys.stderr,
                    )
                    result = run_pipe_once(pipe, args, init_image, mask, seed, min(args.resolution, 512))
                result.save(output_path)
            short_mode = "canvas" if mode == "source_canvas" else "raw"
            panels.append(titled_panel(result, f"{short_mode} s{seed}", args.resolution))

    grid = hstack_panels(panels)
    grid.save(grid_path)
    return grid


def load_scores(path: Path) -> dict[str, dict[str, str]]:
    scores: dict[str, dict[str, str]] = {}
    if not path.exists():
        return scores
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = row.get("id") or row.get("sample_id")
            if sample_id:
                scores[sample_id] = row
    return scores


def score_value(row: dict[str, str], key: str) -> int | None:
    value = (row.get(key) or "").strip()
    if value in {"0", "1"}:
        return int(value)
    return None


def write_scores_template(out_dir: Path, sample_ids: list[str]) -> Path:
    path = out_dir / "fill_sanity_scores_template.csv"
    if path.exists():
        try:
            if set(load_scores(path)) == set(sample_ids):
                return path
        except Exception:
            pass
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "redraws_person", "structure_ok", "catastrophic", "notes"],
        )
        writer.writeheader()
        for sample_id in sample_ids:
            writer.writerow(
                {
                    "id": sample_id,
                    "redraws_person": "",
                    "structure_ok": "",
                    "catastrophic": "",
                    "notes": "",
                }
            )
    return path


def traffic_light(scores: dict[str, dict[str, str]], sample_ids: list[str]) -> tuple[str, str]:
    rows = [scores[sample_id] for sample_id in sample_ids if sample_id in scores]
    complete = [
        row
        for row in rows
        if score_value(row, "redraws_person") is not None
        and score_value(row, "structure_ok") is not None
        and score_value(row, "catastrophic") is not None
    ]
    if not complete:
        return "PENDING_MANUAL_REVIEW", "Fill outputs exist, but 0/1 sanity marks have not been filled yet."

    redraw = sum(score_value(row, "redraws_person") or 0 for row in complete) / len(complete)
    structure = sum(score_value(row, "structure_ok") or 0 for row in complete) / len(complete)
    catastrophic = sum(score_value(row, "catastrophic") or 0 for row in complete) / len(complete)
    # Require a clear majority for GREEN. A bare 9/16 or 10/16 pass rate is
    # better treated as mixed behavior for this pre-training gate.
    if redraw >= 0.65 and structure >= 0.65 and catastrophic <= 0.25:
        return "GREEN", (
            f"Most marked samples redraw the person and keep rough structure "
            f"(redraw={redraw:.2f}, structure={structure:.2f}, catastrophic={catastrophic:.2f})."
        )
    if catastrophic > 0.5 or redraw <= 0.35:
        return "RED", (
            f"Marked samples are commonly catastrophic or stuck to mannequin pixels "
            f"(redraw={redraw:.2f}, structure={structure:.2f}, catastrophic={catastrophic:.2f})."
        )
    return "YELLOW", (
        f"Mixed zero-shot behavior "
        f"(redraw={redraw:.2f}, structure={structure:.2f}, catastrophic={catastrophic:.2f})."
    )


def write_fill_report(out_dir: Path, sample_ids: list[str], infos: dict[str, SampleInfo]) -> None:
    scores_path = out_dir / "fill_sanity_scores.csv"
    template_path = write_scores_template(out_dir, sample_ids)
    scores = load_scores(scores_path)
    light, conclusion = traffic_light(scores, sample_ids)

    icon = {"GREEN": "\U0001F7E2", "YELLOW": "\U0001F7E1", "RED": "\U0001F534"}.get(light, "")
    lines = [
        "# Phase 0 FLUX.1 Fill-dev Zero-shot Sanity Report",
        "",
        "Scope: this report only checks whether large full-person repainting catastrophically fails.",
        "Do not judge identity fidelity, garment sharpness, face realism, or overall generation quality here.",
        "",
        f"Conclusion: {icon} {light}",
        "",
        conclusion,
        "",
        "Scoring keys:",
        "- `redraws_person`: 1 if the output becomes a real human instead of preserving mannequin/plastic pixels.",
        "- `structure_ok`: 1 if garment/body pose structure roughly follows `m_i`; 0 for major structure collapse.",
        "- `catastrophic`: 1 for severe full-person failure, multi-limb/body chaos, or inability to escape mannequin pixels.",
        "",
        f"Fill in manual scores at `{scores_path.name}` or copy from `{template_path.name}` and rerun `--task fill` to refresh this report.",
        "",
        "Traffic-light policy:",
        "- GREEN: a clear majority of samples have redraws_person=1 and structure_ok=1, with few catastrophic cases. Go to paired warmup.",
        "- YELLOW: mixed failures or mannequin-pixel retention. Keep FLUX Fill, but tune canvas/mask strategy.",
        "- RED: widespread catastrophic failures or mannequin retention. Change canvas/mask/prompt strategy; do not switch backbone in this phase.",
        "",
        "If YELLOW/RED, suggested next steps:",
        "- Use a stricter full-person mask from inverse `body_bg`, then dilate garment/limb/head boundaries.",
        "- Make the head placeholder more neutral and remove any sharp fake-head contour from the source canvas.",
        "- Increase mask coverage around hairline, neck, shoulder, hands, feet, and garment edges.",
        "- Compare `source_canvas` versus raw `m_i` columns to decide whether canvas construction or mask coverage is the failure source.",
        "",
        "| id | garment | yaw | bucket | redraws_person | structure_ok | catastrophic | notes |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for sample_id in sample_ids:
        info = infos[sample_id]
        yaw = "NA" if info.yaw is None else f"{info.yaw:.1f}"
        row = scores.get(sample_id, {})
        redraw = row.get("redraws_person", "")
        structure = row.get("structure_ok", "")
        catastrophic = row.get("catastrophic", "")
        notes = row.get("notes", "")
        lines.append(
            f"| {sample_id} | {info.garment_type} | {yaw} | {info.yaw_bucket} | "
            f"{redraw} | {structure} | {catastrophic} | {notes} |"
        )
    path = out_dir / "fill_sanity_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote fill sanity report: {path}")


def write_fill_blocked_report(
    out_dir: Path,
    sample_ids: list[str],
    infos: dict[str, SampleInfo],
    exc: BaseException,
) -> None:
    write_scores_template(out_dir, sample_ids)
    lines = [
        "# Phase 0 FLUX.1 Fill-dev Zero-shot Sanity Report",
        "",
        "Conclusion: BLOCKED",
        "",
        "FLUX.1 Fill-dev did not run because the pipeline could not be loaded.",
        "No zero-shot quality or collapse judgment was made.",
        "",
        "Observed blocker:",
        "",
        "```text",
        str(exc),
        "```",
        "",
        "Common fixes:",
        "- Make sure the Hugging Face token has access to `black-forest-labs/FLUX.1-Fill-dev`.",
        "- For fine-grained tokens, enable access to public gated repositories.",
        "- Keep proxy variables pointed at the active local proxy port, currently `127.0.0.1:7890` on this host.",
        "- Rerun `--task fill` after the model can be downloaded or is present in the local HF cache.",
        "",
        "Reminder: after Fill runs, judge only catastrophic full-person repainting failures or mannequin-pixel retention.",
        "Identity mismatch, blurry clothing, and imperfect faces are expected for this zero-shot check.",
        "",
        "| id | garment | yaw | bucket | redraws_person | structure_ok | catastrophic | notes |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for sample_id in sample_ids:
        info = infos[sample_id]
        yaw = "NA" if info.yaw is None else f"{info.yaw:.1f}"
        lines.append(
            f"| {sample_id} | {info.garment_type} | {yaw} | {info.yaw_bucket} |  |  |  | not run |"
        )
    path = out_dir / "fill_sanity_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote blocked fill sanity report: {path}")


def run_fill(args: argparse.Namespace, root: Path, out_dir: Path, name_to_id: dict[str, int]) -> None:
    sample_ids, infos = select_sample_ids(
        root, out_dir, args.n, args.seed, name_to_id, resample=False
    )
    if args.max_fill_samples is not None:
        sample_ids = sample_ids[: args.max_fill_samples]
        print(f"[fill] limiting this run to first {len(sample_ids)} samples")
    try:
        pipe = load_flux_fill_pipeline(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[fill] pipeline load failed: {exc}", file=sys.stderr)
        write_error(out_dir, "fill_load_pipeline", None, exc)
        write_fill_blocked_report(out_dir, sample_ids, infos, exc)
        return
    grids = []
    for sample_id in sample_ids:
        try:
            print(f"[fill] {sample_id}")
            grid = run_fill_one(pipe, root, out_dir, sample_id, name_to_id, args)
            grids.append(resize_to_height(grid, max(160, args.resolution // 2)))
        except Exception as exc:  # noqa: BLE001
            print(f"[fill] failed {sample_id}: {exc}", file=sys.stderr)
            write_error(out_dir, "fill", sample_id, exc)
    if grids:
        grid = make_grid(grids, cols=1)
        path = out_dir / "fill_sanity_grid.png"
        grid.save(path)
        print(f"Wrote fill sanity grid: {path}")
    write_fill_report(out_dir, sample_ids, infos)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M2H_Final_v2 Phase 0 preprocessing visualization and FLUX Fill sanity check."
    )
    parser.add_argument("--root", type=Path, required=True, help="Path to M2H_Final_v2.")
    parser.add_argument("--task", choices=("preprocess", "fill", "all"), default="all")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <root>/phase0.")
    parser.add_argument("--resample", action="store_true", help="Regenerate phase0/sample_ids.txt.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-sample outputs.")
    parser.add_argument("--blur-sigma", type=float, default=8.0)
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=30.0)
    parser.add_argument("--fill-inputs", choices=("source_canvas", "mannequin", "both"), default="both")
    parser.add_argument(
        "--max-fill-samples",
        type=int,
        default=None,
        help="Run fill on only the first K selected ids without changing phase0/sample_ids.txt.",
    )
    parser.add_argument("--load-4bit", action="store_true", help="Quantize FLUX transformer and T5 encoder to 4-bit.")
    parser.add_argument("--load-8bit", action="store_true", help="Quantize FLUX transformer and T5 encoder to 8-bit.")
    parser.add_argument(
        "--sequential-offload",
        action="store_true",
        help="Use sequential CPU offload instead of model CPU offload. Slower, lower VRAM.",
    )
    args = parser.parse_args()
    if args.load_4bit and args.load_8bit:
        parser.error("--load-4bit and --load-8bit are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir else root / "phase0")
    ensure_dir(out_dir)

    labels, name_to_id = load_label_metadata(root)
    print_probe(root, labels)

    if args.task in {"preprocess", "all"}:
        run_preprocess(args, root, out_dir, name_to_id)
    if args.task in {"fill", "all"}:
        run_fill(args, root, out_dir, name_to_id)

    print(f"Phase 0 outputs: {out_dir}")


if __name__ == "__main__":
    main()
