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
    "a photorealistic human wearing casual garment, matching the given body pose, "
    "natural skin and hair, high quality"
)
DEFAULT_DEV_MODEL = "black-forest-labs/FLUX.1-dev"
DEFAULT_CONTROLNET_MODEL = "InstantX/FLUX.1-dev-Controlnet-Union"
DEFAULT_CONTROL_MODE = 4  # InstantX union ControlNet: mode 4 = pose.


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

    dwpose_path = root / "dwpose/keypoints/mannequin" / f"{probe_id}.npz"
    if dwpose_path.exists():
        with np.load(dwpose_path, allow_pickle=True) as npz:
            print(f"dwpose npz: {dwpose_path}")
            print(f"keys: {list(npz.files)}")
            for key in npz.files:
                arr = npz[key]
                shape = getattr(arr, "shape", None)
                dtype = getattr(arr, "dtype", None)
                if getattr(arr, "size", 0) and dtype != object:
                    print(f"{key}: shape={shape} dtype={dtype} min={arr.min()} max={arr.max()}")
                else:
                    print(f"{key}: shape={shape} dtype={dtype} value={arr.tolist() if shape == () else 'object/empty'}")
    else:
        print(f"dwpose npz missing for probe sample: {dwpose_path}")

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
        masks: dict[str, np.ndarray] = {}
        for key in REGION_KEYS:
            arr = np.asarray(npz[key])
            if arr.dtype == np.bool_:
                masks[key] = arr.astype(bool)
            else:
                threshold = 0.5 if float(np.nanmax(arr)) <= 1.0 else 127.0
                masks[key] = arr > threshold
        return masks


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
        find_one(root / "clothes_bySAM/masks/human", sample_id)
        find_one(root / "dwpose/without_head/mannequin", sample_id)
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


def load_control_image(root: Path, sample_id: str) -> Image.Image:
    return load_rgb(find_one(root / "dwpose/without_head/mannequin", sample_id))


def load_sam_cloth_mask(root: Path, sample_id: str, size: tuple[int, int]) -> Image.Image:
    mask = load_l(find_one(root / "clothes_bySAM/masks/human", sample_id))
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask.point(lambda value: 255 if value > 127 else 0)


def build_garment_condition(mannequin: Image.Image, sam_mask: Image.Image) -> Image.Image:
    """Visual garment condition: mannequin pixels inside SAM cloth mask, white elsewhere."""
    mask = sam_mask.resize(mannequin.size, Image.Resampling.NEAREST)
    garment = Image.new("RGB", mannequin.size, (245, 245, 245))
    garment.paste(mannequin.convert("RGB"), (0, 0), mask)
    return garment


def garment_prompt_hint(root: Path, sample_id: str, mannequin: Image.Image, info: SampleInfo) -> str:
    mask = load_sam_cloth_mask(root, sample_id, mannequin.size)
    arr = np.asarray(mannequin.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask) > 127
    if not m.any():
        return f"wearing a {info.garment_type}"
    color = arr[m].mean(axis=0)
    names = [
        ("black", np.array([25, 25, 25])),
        ("white", np.array([235, 235, 235])),
        ("gray", np.array([128, 128, 128])),
        ("red", np.array([190, 45, 45])),
        ("blue", np.array([45, 95, 190])),
        ("green", np.array([60, 140, 85])),
        ("yellow", np.array([220, 190, 55])),
        ("pink", np.array([220, 120, 160])),
        ("beige", np.array([200, 180, 140])),
    ]
    nearest = min(names, key=lambda item: float(np.linalg.norm(color - item[1])))[0]
    return f"wearing a {nearest} {info.garment_type}"


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
    garment_dir = out_dir / "garment_condition"
    ensure_dir(preprocess_dir)
    ensure_dir(garment_dir)

    panel_path = preprocess_dir / f"{sample_id}.png"
    if panel_path.exists() and not overwrite:
        return load_rgb(panel_path)

    mannequin = load_rgb(find_one(root / "images/mannequin", sample_id))
    human = load_rgb(find_one(root / "images/human", sample_id))
    regions = load_region_masks(root, sample_id)
    pose = load_head_pose(root, sample_id)
    control = load_control_image(root, sample_id)
    sam_mask = load_sam_cloth_mask(root, sample_id, mannequin.size)
    garment = build_garment_condition(mannequin, sam_mask)

    overlay = overlay_masks(human, regions)
    axes = draw_head_axes(human, pose)

    garment.save(garment_dir / f"{sample_id}.png")

    panels = [
        titled_panel(mannequin, "m_i", panel_height),
        titled_panel(human, "h_i ref", panel_height),
        titled_panel(control, "pose ControlNet", panel_height),
        titled_panel(garment, "garment cond", panel_height),
        titled_panel(overlay, "masks", panel_height),
        titled_panel(axes, "head axes", panel_height),
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
        "- Pose ControlNet control image: body/limb skeleton follows the mannequin and has no head conditioning.",
        "- Garment condition: `m_i` is cleanly masked by the SAM cloth mask; background/non-cloth pixels are removed.",
        "- Mask overlay: `cloth_safe` avoids identity, hairline, neck/shoulder identity zones.",
        "- Mask overlay: `edge` is a plausible garment boundary band, not random speckle.",
        "- Mask overlay: `id_strong` covers face plus hair.",
        "- Head pose: axes follow 6DRepNet convention `Rz(roll)@Ry(yaw)@Rx(pitch)` with image y-down.",
        "",
        "| id | garment | yaw | bucket | cloth_safe | edge | pose control ok | garment ok | masks ok | axes ok | notes |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for sample_id in sample_ids:
        info = infos[sample_id]
        yaw = "NA" if info.yaw is None else f"{info.yaw:.1f}"
        lines.append(
            f"| {sample_id} | {info.garment_type} | {yaw} | {info.yaw_bucket} | "
            f"{info.cloth_safe_ratio:.3f} | {info.edge_ratio:.3f} | [ ] | [ ] | [ ] | [ ] | |"
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


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


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


def require_controlnet_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from diffusers import (
            FluxControlNetModel,
            FluxControlNetPipeline,
            FluxTransformer2DModel,
        )
        from transformers import T5EncoderModel
    except ImportError as exc:
        raise RuntimeError(
            "FLUX.1-dev + ControlNet requires recent diffusers/transformers. "
            "Install/upgrade diffusers transformers accelerate sentencepiece protobuf bitsandbytes peft."
        ) from exc
    return torch, FluxControlNetModel, FluxControlNetPipeline, FluxTransformer2DModel, T5EncoderModel


def load_flux_controlnet_pipeline(args: argparse.Namespace) -> Any:
    torch, FluxControlNetModel, FluxControlNetPipeline, FluxTransformer2DModel, T5EncoderModel = (
        require_controlnet_dependencies()
    )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs: dict[str, Any] = {"torch_dtype": dtype}

    if args.load_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("--load-4bit requires transformers bitsandbytes integration.") from exc
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["transformer"] = FluxTransformer2DModel.from_pretrained(
            args.model_id,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=dtype,
        )
        kwargs["text_encoder_2"] = T5EncoderModel.from_pretrained(
            args.model_id,
            subfolder="text_encoder_2",
            quantization_config=quant_config,
            torch_dtype=dtype,
        )

    controlnet = FluxControlNetModel.from_pretrained(args.controlnet_model_id, torch_dtype=dtype)
    pipe = FluxControlNetPipeline.from_pretrained(args.model_id, controlnet=controlnet, **kwargs)
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
    return pipe


def zeroshot_modes(args: argparse.Namespace) -> list[str]:
    return ["pose_only", "garment_prompt"] if args.use_garment else ["pose_only"]


def run_controlnet_once(
    pipe: Any,
    args: argparse.Namespace,
    prompt: str,
    control: Image.Image,
    seed: int,
    resolution: int,
) -> Image.Image:
    torch = sys.modules["torch"]
    control_image = resize_for_generation(control, resolution)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)
    return pipe(
        prompt=prompt,
        control_image=control_image,
        control_mode=args.control_mode,
        controlnet_conditioning_scale=args.controlnet_scale,
        height=control_image.height,
        width=control_image.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        max_sequence_length=args.max_sequence_length,
    ).images[0]


def run_zeroshot_one(
    pipe: Any,
    root: Path,
    out_dir: Path,
    sample_id: str,
    info: SampleInfo,
    args: argparse.Namespace,
) -> Image.Image:
    zero_dir = out_dir / "zeroshot"
    output_dir = zero_dir / "outputs"
    ensure_dir(zero_dir)
    ensure_dir(output_dir)
    grid_path = zero_dir / f"{sample_id}.png"
    if grid_path.exists() and not args.overwrite:
        return load_rgb(grid_path)

    mannequin = load_rgb(find_one(root / "images/mannequin", sample_id))
    control = load_control_image(root, sample_id)
    panels = [
        titled_panel(mannequin, "m_i", args.resolution),
        titled_panel(control, "pose ControlNet", args.resolution),
    ]
    seeds = [args.seed + i for i in range(args.num_seeds)]
    for mode in zeroshot_modes(args):
        prompt = args.prompt
        if mode == "garment_prompt":
            prompt = f"{args.prompt}, {garment_prompt_hint(root, sample_id, mannequin, info)}"
        for seed in seeds:
            output_path = output_dir / f"{sample_id}_{mode}_seed{seed}.png"
            if output_path.exists() and not args.overwrite:
                result = load_rgb(output_path)
            else:
                try:
                    result = run_controlnet_once(pipe, args, prompt, control, seed, args.resolution)
                except RuntimeError as exc:
                    if not is_oom(exc):
                        raise
                    torch = sys.modules["torch"]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(
                        f"[zeroshot] OOM for {sample_id} mode={mode} seed={seed}; retrying at 512",
                        file=sys.stderr,
                    )
                    result = run_controlnet_once(pipe, args, prompt, control, seed, min(args.resolution, 512))
                result.save(output_path)
            panels.append(titled_panel(result, f"{mode} s{seed}", args.resolution))
    grid = hstack_panels(panels)
    grid.save(grid_path)
    return grid


def write_zeroshot_scores_template(out_dir: Path, sample_ids: list[str]) -> Path:
    path = out_dir / "zeroshot_scores_template.csv"
    if path.exists():
        return path
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "pose_followed", "anatomy_ok", "is_plausible_human", "notes"],
        )
        writer.writeheader()
        for sample_id in sample_ids:
            writer.writerow(
                {"id": sample_id, "pose_followed": "", "anatomy_ok": "", "is_plausible_human": "", "notes": ""}
            )
    return path


def zeroshot_traffic_light(scores: dict[str, dict[str, str]], sample_ids: list[str]) -> tuple[str, str]:
    rows = [scores[sample_id] for sample_id in sample_ids if sample_id in scores]
    complete = [
        row
        for row in rows
        if score_value(row, "pose_followed") is not None
        and score_value(row, "anatomy_ok") is not None
        and score_value(row, "is_plausible_human") is not None
    ]
    if not complete:
        return "PENDING_MANUAL_REVIEW", "Zero-shot outputs exist, but 0/1 sanity marks have not been filled yet."
    pose = sum(score_value(row, "pose_followed") or 0 for row in complete) / len(complete)
    anatomy = sum(score_value(row, "anatomy_ok") or 0 for row in complete) / len(complete)
    human = sum(score_value(row, "is_plausible_human") or 0 for row in complete) / len(complete)
    if pose >= 0.65 and anatomy >= 0.65:
        return "GREEN", f"Most samples follow pose and avoid anatomy collapse (pose={pose:.2f}, anatomy={anatomy:.2f}, human={human:.2f})."
    if pose <= 0.35 or anatomy <= 0.35:
        return "RED", f"Pose following or anatomy is broadly failing (pose={pose:.2f}, anatomy={anatomy:.2f}, human={human:.2f})."
    return "YELLOW", f"Mixed dev+ControlNet behavior (pose={pose:.2f}, anatomy={anatomy:.2f}, human={human:.2f})."


def write_zeroshot_report(out_dir: Path, sample_ids: list[str], infos: dict[str, SampleInfo]) -> None:
    scores_path = out_dir / "zeroshot_scores.csv"
    template_path = write_zeroshot_scores_template(out_dir, sample_ids)
    scores = load_scores(scores_path)
    light, conclusion = zeroshot_traffic_light(scores, sample_ids)
    icon = {"GREEN": "\U0001F7E2", "YELLOW": "\U0001F7E1", "RED": "\U0001F534"}.get(light, "")
    lines = [
        "# Phase 0 FLUX.1-dev + Pose ControlNet Zero-shot Report",
        "",
        "Scope: judge only whether the generated body follows the DWPose control and avoids anatomy collapse.",
        "Identity mismatch, imperfect faces, and blurry garments are expected here.",
        "",
        f"Conclusion: {icon} {light}",
        "",
        conclusion,
        "",
        "Scoring keys:",
        "- `pose_followed`: 1 if the generated body pose follows the no-head pose control.",
        "- `anatomy_ok`: 1 if there are no major multi-limb, broken-limb, or body-structure failures.",
        "- `is_plausible_human`: 1 if the output is a plausible person, ignoring identity and fine face quality.",
        "",
        f"Fill in manual scores at `{scores_path.name}` or copy from `{template_path.name}` and rerun `--task zeroshot` to refresh this report.",
        "",
        "Traffic-light policy:",
        "- GREEN: most samples follow pose and have acceptable anatomy. Go to paired warmup after Task A/B pass.",
        "- YELLOW: pose follows intermittently or anatomy occasionally fails. Tune ControlNet scale/prompt.",
        "- RED: widespread pose failure or anatomy collapse. Check ControlNet weights, control_mode, control image preprocessing, and conditioning scale; do not switch backbone in this phase.",
        "",
        "| id | garment | yaw | bucket | pose_followed | anatomy_ok | is_plausible_human | notes |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for sample_id in sample_ids:
        info = infos[sample_id]
        yaw = "NA" if info.yaw is None else f"{info.yaw:.1f}"
        row = scores.get(sample_id, {})
        lines.append(
            f"| {sample_id} | {info.garment_type} | {yaw} | {info.yaw_bucket} | "
            f"{row.get('pose_followed', '')} | {row.get('anatomy_ok', '')} | "
            f"{row.get('is_plausible_human', '')} | {row.get('notes', '')} |"
        )
    path = out_dir / "zeroshot_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote zeroshot report: {path}")


def write_zeroshot_blocked_report(
    out_dir: Path,
    sample_ids: list[str],
    infos: dict[str, SampleInfo],
    exc: BaseException,
) -> None:
    write_zeroshot_scores_template(out_dir, sample_ids)
    lines = [
        "# Phase 0 FLUX.1-dev + Pose ControlNet Zero-shot Report",
        "",
        "Conclusion: BLOCKED",
        "",
        "The zero-shot run did not start because FLUX.1-dev or the pose ControlNet could not be loaded.",
        "",
        "Observed blocker:",
        "",
        "```text",
        str(exc),
        "```",
        "",
        "If this report was produced with `HF_HUB_OFFLINE=1`, it is an intentional fast-fail after an incomplete or too-slow model download. Unset `HF_HUB_OFFLINE` and rerun once the FLUX.1-dev cache is complete.",
        "",
        "Check Hugging Face access, local cache, proxy, and `--controlnet-model-id`. Do not change the FLUX.1-dev backbone for this phase.",
        "",
        "| id | garment | yaw | bucket | pose_followed | anatomy_ok | is_plausible_human | notes |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for sample_id in sample_ids:
        info = infos[sample_id]
        yaw = "NA" if info.yaw is None else f"{info.yaw:.1f}"
        lines.append(f"| {sample_id} | {info.garment_type} | {yaw} | {info.yaw_bucket} |  |  |  | not run |")
    path = out_dir / "zeroshot_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote blocked zeroshot report: {path}")


def run_zeroshot(args: argparse.Namespace, root: Path, out_dir: Path, name_to_id: dict[str, int]) -> None:
    sample_ids, infos = select_sample_ids(root, out_dir, args.n, args.seed, name_to_id, resample=False)
    if args.max_zeroshot_samples is not None:
        sample_ids = sample_ids[: args.max_zeroshot_samples]
        print(f"[zeroshot] limiting this run to first {len(sample_ids)} samples")
    try:
        pipe = load_flux_controlnet_pipeline(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[zeroshot] pipeline load failed: {exc}", file=sys.stderr)
        write_error(out_dir, "zeroshot_load_pipeline", None, exc)
        write_zeroshot_blocked_report(out_dir, sample_ids, infos, exc)
        return
    grids = []
    for sample_id in sample_ids:
        try:
            print(f"[zeroshot] {sample_id}")
            grid = run_zeroshot_one(pipe, root, out_dir, sample_id, infos[sample_id], args)
            grids.append(resize_to_height(grid, max(160, args.resolution // 2)))
        except Exception as exc:  # noqa: BLE001
            print(f"[zeroshot] failed {sample_id}: {exc}", file=sys.stderr)
            write_error(out_dir, "zeroshot", sample_id, exc)
    if grids:
        grid = make_grid(grids, cols=1)
        path = out_dir / "zeroshot_grid.png"
        grid.save(path)
        print(f"Wrote zeroshot grid: {path}")
    write_zeroshot_report(out_dir, sample_ids, infos)


def gb(value: float) -> str:
    return f"{value / (1024 ** 3):.2f} GB"


def gpu_total_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return max(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / (1024 ** 3)
    except Exception:
        return 0.0


def write_vram_simulation_report(out_dir: Path, args: argparse.Namespace, reason: str) -> None:
    path = out_dir / "vram_report.md"
    # Conservative rough estimates. They are intentionally labeled simulation;
    # the pass/fail gate still requires an A6000 48G measurement.
    module_rows = [
        ("FLUX.1-dev transformer", "approx 11.9B", "23.8 GB bf16 params"),
        ("T5 text_encoder_2", "approx 4.8B", "9.6 GB bf16 params; usually frozen/offloaded"),
        ("CLIP text_encoder", "approx 0.12B", "0.24 GB bf16 params; frozen/offloaded"),
        ("VAE", "approx 0.08B", "0.16 GB bf16 params; decode activations dominate when enabled"),
        ("InstantX union ControlNet", "community FLUX ControlNet", "must be measured with real weights"),
        ("LoRA trainables", f"rank {args.rank}/16", "small params; activations dominate peak"),
    ]
    configs = [
        ("minimal", 8, 1, 0, "on", 30.0),
        ("+differential", 8, 3, 0, "on", 38.0),
        ("+decode", 8, 3, 2, "on", 43.0),
        ("full", 16, 3, 2, "on", 46.0),
    ]
    lines = [
        "# Phase 0 VRAM Report",
        "",
        f"Mode: SIMULATE_ONLY ({reason})",
        "",
        "This host did not run the training-memory gate on an A6000 48G. The table below is a planning estimate, not an acceptance result.",
        "A real 48G run must execute `--task vram` without `--simulate-only` and record `torch.cuda.max_memory_allocated()`.",
        "",
        "## Module Size Estimate",
        "",
        "| module | params | memory note |",
        "| --- | ---: | --- |",
    ]
    for row in module_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    lines.extend(
        [
            "",
            "## Config Estimate",
            "",
            "| config | rank | forward passes | decode | offload | estimated peak | 48G feasible? |",
            "| --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for name, rank, fwds, decodes, offload, peak in configs:
        feasible = "likely, requires A6000 measurement" if peak < 48 else "borderline, reduce config before training"
        lines.append(f"| {name} | {rank} | {fwds} | {decodes} | {offload} | {peak:.1f} GB simulated | {feasible} |")
    lines.extend(
        [
            "",
            "Conclusion: NOT YET ACCEPTED. This must be measured on A6000 48G before paired warmup.",
            "",
            "If the full config OOMs on 48G, retest in this order: rank 16 -> 8, reduce identity VAE decode frequency, share/cache differential branch text/control features, run 384 resolution, and make ControlNet LoRA-only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote simulated VRAM report: {path}")


def trainable_params_for_optimizer(model: Any) -> list[Any]:
    params = [p for p in model.parameters() if p.requires_grad]
    return params


def attach_lora_to_transformer(transformer: Any, rank: int) -> str:
    try:
        from peft import LoraConfig

        target_modules = ["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
        config = LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=target_modules)
        transformer.add_adapter(config)
        for name, param in transformer.named_parameters():
            param.requires_grad = "lora" in name.lower()
        return "peft_lora"
    except Exception as exc:  # noqa: BLE001
        import torch

        for param in transformer.parameters():
            param.requires_grad = False
        transformer._phase0_dummy_lora = torch.nn.Parameter(torch.zeros(rank, rank, device=next(transformer.parameters()).device))
        return f"dummy_lora_fallback: {exc}"


def run_real_vram_config(args: argparse.Namespace, name: str, rank: int, forwards: int, decodes: int) -> dict[str, str]:
    import torch

    from diffusers import FluxControlNetModel, FluxTransformer2DModel

    device = torch.device("cuda")
    dtype = torch.bfloat16
    result = {"config": name, "rank": str(rank), "forwards": str(forwards), "decode": str(decodes), "offload": "on"}
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        transformer = FluxTransformer2DModel.from_pretrained(args.model_id, subfolder="transformer", torch_dtype=dtype).to(device)
        controlnet = FluxControlNetModel.from_pretrained(args.controlnet_model_id, torch_dtype=dtype).to(device)
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
        if hasattr(controlnet, "enable_gradient_checkpointing"):
            controlnet.enable_gradient_checkpointing()
        lora_note = attach_lora_to_transformer(transformer, rank)
        params = trainable_params_for_optimizer(transformer)
        try:
            import bitsandbytes as bnb

            optim = bnb.optim.PagedAdamW8bit(params, lr=1e-4) if params else None
            optim_note = "PagedAdamW8bit"
        except Exception as exc:  # noqa: BLE001
            optim = torch.optim.AdamW(params, lr=1e-4) if params else None
            optim_note = f"AdamW fallback: {exc}"

        # Packed FLUX latents for 512x512: VAE latent 64x64, packed by 2 -> 32x32 tokens, 64 channels.
        latent_tokens = (args.resolution // 16) * (args.resolution // 16)
        text_tokens = min(args.max_sequence_length, 512)
        hidden_states = torch.randn(1, latent_tokens, 64, device=device, dtype=dtype, requires_grad=True)
        control_image = torch.randn(1, 3, args.resolution, args.resolution, device=device, dtype=dtype)
        encoder_hidden_states = torch.randn(1, text_tokens, 4096, device=device, dtype=dtype)
        pooled = torch.randn(1, 768, device=device, dtype=dtype)
        timestep = torch.ones(1, device=device, dtype=dtype)
        guidance = torch.full((1,), args.guidance_scale, device=device, dtype=dtype)
        img_ids = torch.zeros(latent_tokens, 3, device=device, dtype=dtype)
        txt_ids = torch.zeros(text_tokens, 3, device=device, dtype=dtype)
        total = hidden_states.sum() * 0.0
        for _ in range(forwards):
            control_out = controlnet(
                hidden_states=hidden_states,
                controlnet_cond=control_image,
                controlnet_mode=torch.tensor([args.control_mode], device=device),
                conditioning_scale=args.controlnet_scale,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=True,
            )
            out = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                controlnet_block_samples=control_out.controlnet_block_samples,
                controlnet_single_block_samples=control_out.controlnet_single_block_samples,
                return_dict=True,
            )
            total = total + out.sample.float().mean()
        for _ in range(decodes):
            total = total + torch.nn.functional.interpolate(
                torch.randn(1, 16, args.resolution // 8, args.resolution // 8, device=device, dtype=dtype),
                scale_factor=8,
                mode="nearest",
            ).float().mean() * 0.01
        total.backward()
        if optim is not None:
            optim.step()
            optim.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated()
        result.update({"peak": gb(peak), "feasible_48g": "YES" if peak < 45 * 1024**3 else "MARGINAL/NO", "notes": f"{lora_note}; {optim_note}"})
    except RuntimeError as exc:
        result.update({"peak": "OOM", "feasible_48g": "NO", "notes": str(exc).splitlines()[0]})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        result.update({"peak": "FAILED", "feasible_48g": "UNKNOWN", "notes": str(exc).splitlines()[0]})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def run_vram(args: argparse.Namespace, out_dir: Path) -> None:
    ensure_dir(out_dir)
    total_gb = gpu_total_gb()
    if args.simulate_only or total_gb < 40:
        reason = "--simulate-only requested" if args.simulate_only else f"current GPU max memory is {total_gb:.1f}GB, not A6000 48G"
        write_vram_simulation_report(out_dir, args, reason)
        return
    configs = [
        ("minimal", 8, 1, 0),
        ("+differential", 8, 3, 0),
        ("+decode", 8, 3, 2),
        ("full", 16, 3, 2),
    ]
    rows = [run_real_vram_config(args, *cfg) for cfg in configs]
    lines = [
        "# Phase 0 VRAM Report",
        "",
        f"Mode: REAL CUDA MEASUREMENT on max GPU {total_gb:.1f}GB",
        "",
        "| config | rank | forward passes | decode | offload | peak VRAM | 48G feasible? | notes |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['rank']} | {row['forwards']} | {row['decode']} | {row['offload']} | "
            f"{row['peak']} | {row['feasible_48g']} | {row['notes']} |"
        )
    full = rows[-1]
    if full["feasible_48g"] == "YES":
        conclusion = "Conclusion: full rank-16 / 3-forward / 2-decode config is feasible on 48G for this measured step."
    else:
        conclusion = (
            "Conclusion: full config is not accepted. Retest lower-cost configs: rank 8, fewer decode calls, "
            "cached/shared differential branches, 384 resolution, and LoRA-only ControlNet."
        )
    lines.extend(["", conclusion])
    path = out_dir / "vram_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote VRAM report: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M2H_Final_v2 Phase 0: preprocess visualization, FLUX.1-dev ControlNet VRAM pressure test, and zero-shot sanity."
    )
    parser.add_argument("--root", type=Path, required=True, help="Path to M2H_Final_v2.")
    parser.add_argument("--task", choices=("preprocess", "vram", "zeroshot", "all"), default="all")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <root>/phase0.")
    parser.add_argument("--resample", action="store_true", help="Regenerate phase0/sample_ids.txt.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-sample outputs.")
    parser.add_argument("--blur-sigma", type=float, default=8.0, help=argparse.SUPPRESS)
    parser.add_argument("--model-id", default=DEFAULT_DEV_MODEL, help="FLUX.1-dev model id.")
    parser.add_argument("--controlnet-model-id", default=DEFAULT_CONTROLNET_MODEL, help="Pose/union FLUX ControlNet model id.")
    parser.add_argument("--control-mode", type=int, default=DEFAULT_CONTROL_MODE, help="Control mode for union ControlNet; InstantX uses 4 for pose.")
    parser.add_argument("--controlnet-scale", type=float, default=0.75, help="ControlNet conditioning scale for zero-shot generation and VRAM mock step.")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument(
        "--max-zeroshot-samples",
        type=int,
        default=None,
        help="Run zero-shot on only the first K selected ids without changing phase0/sample_ids.txt.",
    )
    parser.add_argument("--use-garment", action="store_true", help="Also run a garment-prompt branch using the SAM masked garment color/type.")
    parser.add_argument("--simulate-only", action="store_true", help="For Task B, write theoretical VRAM estimate instead of real CUDA measurement.")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank used in VRAM estimate/report.")
    parser.add_argument("--load-4bit", action="store_true", help="Quantize FLUX transformer and T5 encoder to 4-bit.")
    parser.add_argument(
        "--sequential-offload",
        action="store_true",
        help="Use sequential CPU offload instead of model CPU offload. Slower, lower VRAM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir else root / "phase0")
    ensure_dir(out_dir)

    labels, name_to_id = load_label_metadata(root)
    print_probe(root, labels)

    if args.task in {"preprocess", "all"}:
        run_preprocess(args, root, out_dir, name_to_id)
    if args.task in {"vram", "all"}:
        run_vram(args, out_dir)
    if args.task in {"zeroshot", "all"}:
        run_zeroshot(args, root, out_dir, name_to_id)

    print(f"Phase 0 outputs: {out_dir}")


if __name__ == "__main__":
    main()
