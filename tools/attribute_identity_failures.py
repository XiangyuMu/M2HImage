from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.stats import pointbiserialr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conditions import find_one, get_resolution, load_yaml  # noqa: E402


FACE_LABEL = 1
HAIR_LABEL = 2
GLASSES_LABEL = 11
ARM_LABELS = (12, 13)
JEWELRY_LABEL = 17
GROUPS = ("low", "middle", "high")
GROUP_LABELS = {
    "low": "< P25 (plus detector failures)",
    "middle": "P25-P75",
    "high": "> P75",
}
FAILURE_LABELS = (
    "face_occluded_by_hair",
    "face_undetected_or_lowconf",
    "hair_overgrown",
    "accessory_leak",
    "sleeve_mismatch",
)
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            str(FONT_BOLD if bold else FONT_REGULAR), size
        )
    except OSError:
        return ImageFont.load_default()


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing DWPose predictions: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {str(row["key"]): row for row in rows}


def open_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        value = ImageOps.exif_transpose(image).convert("RGB")
        if value.size != size:
            value = value.resize(size, Image.Resampling.BICUBIC)
        return np.asarray(value, dtype=np.uint8)


def load_labels(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        value = image.convert("L")
        if value.size != size:
            value = value.resize(size, Image.Resampling.NEAREST)
        return np.asarray(value, dtype=np.uint8)


def mask_bbox(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if not len(xs):
        return None
    height, width = mask.shape
    return (
        max(0, int(xs.min()) - int(pad)),
        max(0, int(ys.min()) - int(pad)),
        min(width, int(xs.max()) + int(pad) + 1),
        min(height, int(ys.max()) + int(pad) + 1),
    )


def expected_face_zone(
    labels: np.ndarray,
    pose: dict[str, Any] | None,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """Build a geometric face zone in which hair pixels can be measured.

    FASHN is a single-label parser, so face and hair masks cannot overlap by
    definition. The zone uses DWPose nose/eyes/ears and falls back to an expanded
    parsing-face ellipse. This makes hair intrusion observable without changing
    the cached parsing protocol.
    """
    height, width = labels.shape
    face = labels == FACE_LABEL
    parsed_bbox = mask_bbox(face)
    cx = cy = rx = ry = None
    if pose and pose.get("status") == "ok":
        points = np.asarray(pose.get("body", []), dtype=np.float32)
        scores = np.asarray(pose.get("body_scores", []), dtype=np.float32)
        if points.shape == (18, 2) and scores.shape == (18,):
            points_px = points * np.asarray([width, height], dtype=np.float32)
            valid = scores >= float(score_threshold)
            nose = points_px[0] if valid[0] else None
            eyes = [points_px[index] for index in (14, 15) if valid[index]]
            ears = [points_px[index] for index in (16, 17) if valid[index]]
            center_points = eyes + ([nose] if nose is not None else [])
            if center_points:
                center = np.mean(np.stack(center_points), axis=0)
                cx = float(center[0])
                if nose is not None:
                    cy = float(nose[1])
                else:
                    cy = float(center[1] + 0.35 * max(12.0, width * 0.04))
                widths = []
                if len(ears) == 2:
                    widths.append(float(np.linalg.norm(ears[0] - ears[1])))
                if len(eyes) == 2:
                    widths.append(float(np.linalg.norm(eyes[0] - eyes[1])) * 2.35)
                if widths:
                    rx = 0.5 * max(widths)
                    ry = rx * 1.25
    if parsed_bbox is not None:
        x0, y0, x1, y1 = parsed_bbox
        parsed_rx = max(8.0, (x1 - x0) * 0.62)
        parsed_ry = max(10.0, (y1 - y0) * 0.64)
        if cx is None:
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
            rx, ry = parsed_rx, parsed_ry
        else:
            rx = max(float(rx or 0.0), parsed_rx)
            ry = max(float(ry or 0.0), parsed_ry)
    if cx is None or cy is None or rx is None or ry is None:
        cx, cy = width * 0.5, height * 0.14
        rx, ry = width * 0.07, height * 0.075
    rx = float(np.clip(rx, 12.0, width * 0.18))
    ry = float(np.clip(ry, 16.0, height * 0.16))
    cy += 0.12 * ry
    zone = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        zone,
        (int(round(cx)), int(round(cy))),
        (int(round(rx)), int(round(ry))),
        0.0,
        0.0,
        360.0,
        1,
        thickness=-1,
    )
    return zone


def forearm_zone(
    body: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
    threshold: float = 0.3,
) -> np.ndarray:
    body = np.asarray(body, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    zone = np.zeros((height, width), dtype=np.uint8)
    if body.shape != (18, 2) or scores.shape != (18,):
        return zone
    points = body * np.asarray([width, height], dtype=np.float32)
    if scores[2] >= threshold and scores[5] >= threshold:
        shoulder_width = float(np.linalg.norm(points[2] - points[5]))
    else:
        shoulder_width = width * 0.20
    thickness = max(10, int(round(shoulder_width * 0.15)))
    for elbow, wrist in ((3, 4), (6, 7)):
        if scores[elbow] < threshold or scores[wrist] < threshold:
            continue
        start = tuple(np.rint(points[elbow]).astype(int).tolist())
        end = tuple(np.rint(points[wrist]).astype(int).tolist())
        cv2.line(zone, start, end, 1, thickness=thickness)
        cv2.circle(zone, start, max(1, thickness // 2), 1, thickness=-1)
        cv2.circle(zone, end, max(1, thickness // 2), 1, thickness=-1)
    return zone


def visible_forearm_fraction(labels: np.ndarray, zone: np.ndarray) -> float | None:
    denominator = int(zone.sum())
    if denominator == 0:
        return None
    skin = np.isin(labels, np.asarray(ARM_LABELS, dtype=np.uint8))
    return float(np.logical_and(skin, zone.astype(bool)).sum() / denominator)


def expanded_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    scale: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    side = max(x1 - x0, y1 - y0) * float(scale)
    return (
        max(0, int(round(cx - side * 0.5))),
        max(0, int(round(cy - side * 0.5))),
        min(width, int(round(cx + side * 0.5))),
        min(height, int(round(cy + side * 0.5))),
    )


def ear_zone_from_labels(labels: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    bbox = mask_bbox(labels == FACE_LABEL)
    if bbox is None:
        return np.zeros_like(labels, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    fw, fh = max(1, x1 - x0), max(1, y1 - y0)
    zone = np.zeros_like(labels, dtype=np.uint8)
    yy0 = max(0, int(round(y0 + 0.15 * fh)))
    yy1 = min(height, int(round(y1 + 0.30 * fh)))
    left0 = max(0, int(round(x0 - 0.45 * fw)))
    left1 = min(width, int(round(x0 + 0.18 * fw)))
    right0 = max(0, int(round(x1 - 0.18 * fw)))
    right1 = min(width, int(round(x1 + 0.45 * fw)))
    zone[yy0:yy1, left0:left1] = 1
    zone[yy0:yy1, right0:right1] = 1
    return zone


def accessory_candidate(
    rgb: np.ndarray,
    labels: np.ndarray,
    zone: np.ndarray,
) -> tuple[np.ndarray, float]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    explicit = np.isin(labels, (JEWELRY_LABEL, GLASSES_LABEL))
    saturated = (hsv[..., 1] >= 105) & (hsv[..., 2] >= 70)
    saturated &= ~np.isin(labels, (FACE_LABEL, HAIR_LABEL, 3, 4, 5, 6))
    candidate = (explicit | saturated) & zone.astype(bool)
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    filtered = np.zeros_like(candidate, dtype=np.uint8)
    zone_area = max(1, int(zone.sum()))
    max_area = max(24, int(round(zone_area * 0.20)))
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if 3 <= area <= max_area:
            filtered[components == index] = 1
    pixels = hsv[filtered.astype(bool)]
    histogram = np.zeros(24, dtype=np.float32)
    if len(pixels):
        histogram, _ = np.histogram(
            pixels[:, 0], bins=24, range=(0, 180), weights=pixels[:, 1]
        )
        histogram = histogram.astype(np.float32, copy=False)
        norm = float(np.linalg.norm(histogram))
        if norm > 0:
            histogram /= norm
    return histogram, float(filtered.sum() / zone_area)


def accessory_similarity(
    reference_rgb: np.ndarray,
    reference_labels: np.ndarray,
    generated_rgb: np.ndarray,
    generated_labels: np.ndarray,
) -> tuple[float | None, float, float]:
    ref_hist, ref_fraction = accessory_candidate(
        reference_rgb, reference_labels, ear_zone_from_labels(reference_labels)
    )
    gen_hist, gen_fraction = accessory_candidate(
        generated_rgb, generated_labels, ear_zone_from_labels(generated_labels)
    )
    if not np.any(ref_hist) or not np.any(gen_hist):
        return None, ref_fraction, gen_fraction
    return float(np.dot(ref_hist, gen_hist)), ref_fraction, gen_fraction


def parse_key(row: dict[str, Any]) -> str:
    return f"{row['mid']}__id{row['jid']}__seed{int(row['seed'])}"


def classify_row(
    row: dict[str, str],
    *,
    root: Path,
    parsing_dir: Path,
    poses: dict[str, dict[str, Any]],
    size: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    width, height = size
    mid, jid, seed = str(row["mid"]), str(row["jid"]), int(row["seed"])
    key = f"{mid}__id{jid}__seed{seed}"
    generated_path = Path(row["path"])
    generated_parsing_path = parsing_dir / f"{generated_path.stem}.png"
    mannequin_path = find_one(root / "images/mannequin", mid)
    reference_path = find_one(root / "images/human", jid)
    mannequin_labels_path = find_one(
        root / "human_parsing/fashn/masks/mannequin", mid
    )
    reference_labels_path = find_one(root / "human_parsing/fashn/masks/human", jid)
    required = (
        generated_path,
        generated_parsing_path,
        mannequin_path,
        reference_path,
        mannequin_labels_path,
        reference_labels_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{key} missing inputs: {missing}")

    generated = open_rgb(generated_path, size)
    reference = open_rgb(reference_path, size)
    generated_labels = load_labels(generated_parsing_path, size)
    reference_labels = load_labels(reference_labels_path, size)
    mannequin_labels = load_labels(mannequin_labels_path, size)
    pose = poses.get(key)
    face_zone = expected_face_zone(generated_labels, pose)
    face_zone_area = max(1, int(face_zone.sum()))
    hair_in_face = float(
        np.logical_and(generated_labels == HAIR_LABEL, face_zone.astype(bool)).sum()
        / face_zone_area
    )
    face_visible = float(
        np.logical_and(generated_labels == FACE_LABEL, face_zone.astype(bool)).sum()
        / face_zone_area
    )
    generated_hair_fraction = float((generated_labels == HAIR_LABEL).mean())
    reference_hair_fraction = float((reference_labels == HAIR_LABEL).mean())
    hair_ratio = (
        generated_hair_fraction / reference_hair_fraction
        if reference_hair_fraction > 1e-8
        else None
    )
    det_conf = finite_float(row.get("det_conf"))
    face_lowconf = row.get("status") != "ok" or det_conf is None or det_conf < args.low_conf_threshold

    accessory_score, ref_accessory, gen_accessory = accessory_similarity(
        reference, reference_labels, generated, generated_labels
    )
    accessory_leak = bool(
        accessory_score is not None
        and accessory_score >= args.accessory_similarity_threshold
        and ref_accessory >= args.accessory_min_fraction
        and gen_accessory >= args.accessory_min_fraction
    )

    source_pose_path = root / "dwpose/keypoints/mannequin" / f"{mid}.npz"
    if not source_pose_path.is_file():
        raise FileNotFoundError(f"missing source keypoints: {source_pose_path}")
    with np.load(source_pose_path, allow_pickle=False) as source_pose:
        body = np.asarray(source_pose["body"], dtype=np.float32)
        body_scores = np.asarray(source_pose["body_scores"], dtype=np.float32)
    arm_zone = forearm_zone(body, body_scores, width, height)
    source_forearm = visible_forearm_fraction(mannequin_labels, arm_zone)
    generated_forearm = visible_forearm_fraction(generated_labels, arm_zone)
    sleeve_difference = (
        abs(generated_forearm - source_forearm)
        if source_forearm is not None and generated_forearm is not None
        else None
    )

    face_bbox = mask_bbox(face_zone)
    if face_bbox is None:
        face_bbox = (int(width * 0.4), 0, int(width * 0.6), int(height * 0.28))
    crop_bbox = expanded_bbox(face_bbox, width, height, scale=1.45)
    flags = {
        "face_occluded_by_hair": hair_in_face > args.hair_face_threshold,
        "face_undetected_or_lowconf": face_lowconf,
        "hair_overgrown": hair_ratio is not None and hair_ratio > args.hair_overgrown_threshold,
        "accessory_leak": accessory_leak,
        "sleeve_mismatch": sleeve_difference is not None and sleeve_difference > args.sleeve_threshold,
    }
    return {
        "mid": mid,
        "jid": jid,
        "seed": seed,
        "key": key,
        "garment_type": row.get("garment_type", "unknown"),
        "generated_path": str(generated_path),
        "mannequin_path": str(mannequin_path),
        "reference_path": str(reference_path),
        "generated_parsing_path": str(generated_parsing_path),
        "identity_status": row.get("status", ""),
        "identity_error": row.get("error", ""),
        "sim_target": finite_float(row.get("sim_target")),
        "sim_source": finite_float(row.get("sim_source")),
        "delta_id": finite_float(row.get("delta_id")),
        "det_conf": det_conf,
        "face_size_px": finite_float(row.get("face_size_px")),
        "face_hair_fraction": hair_in_face,
        "face_visible_fraction": face_visible,
        "generated_hair_fraction": generated_hair_fraction,
        "reference_hair_fraction": reference_hair_fraction,
        "hair_area_ratio": hair_ratio,
        "accessory_similarity": accessory_score,
        "reference_accessory_fraction": ref_accessory,
        "generated_accessory_fraction": gen_accessory,
        "source_forearm_skin_ratio": source_forearm,
        "generated_forearm_skin_ratio": generated_forearm,
        "forearm_skin_ratio_difference": sleeve_difference,
        "face_crop_bbox": list(crop_bbox),
        **{name: int(value) for name, value in flags.items()},
    }


def assign_groups(rows: list[dict[str, Any]]) -> tuple[float, float]:
    values = np.asarray(
        [row["sim_target"] for row in rows if row["sim_target"] is not None],
        dtype=np.float64,
    )
    if not len(values):
        raise RuntimeError("identity CSV contains no valid sim_target values")
    p25, p75 = (float(value) for value in np.quantile(values, (0.25, 0.75)))
    for row in rows:
        value = row["sim_target"]
        if value is None or value < p25:
            row["quantile_group"] = "low"
        elif value > p75:
            row["quantile_group"] = "high"
        else:
            row["quantile_group"] = "middle"
    return p25, p75


def rate_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for label in FAILURE_LABELS:
        grouped: dict[str, tuple[int, int, float]] = {}
        for group in GROUPS:
            selected = [row for row in rows if row["quantile_group"] == group]
            count = sum(int(row[label]) for row in selected)
            grouped[group] = (count, len(selected), count / max(1, len(selected)))
        low_rate = grouped["low"][2]
        high_rate = grouped["high"][2]
        enrichment = (
            low_rate / high_rate
            if high_rate > 0
            else (math.inf if low_rate > 0 else 1.0)
        )
        valid = [row for row in rows if row["sim_target"] is not None]
        flags = np.asarray([int(row[label]) for row in valid], dtype=np.int64)
        scores = np.asarray([float(row["sim_target"]) for row in valid], dtype=np.float64)
        correlation = p_value = None
        if len(np.unique(flags)) == 2:
            test = pointbiserialr(flags, scores)
            correlation = float(test.statistic)
            p_value = float(test.pvalue)
        result.append(
            {
                "label": label,
                "low_count": grouped["low"][0],
                "low_total": grouped["low"][1],
                "low_rate": low_rate,
                "middle_count": grouped["middle"][0],
                "middle_total": grouped["middle"][1],
                "middle_rate": grouped["middle"][2],
                "high_count": grouped["high"][0],
                "high_total": grouped["high"][1],
                "high_rate": high_rate,
                "low_over_high": enrichment,
                "point_biserial_r": correlation,
                "point_biserial_p": p_value,
            }
        )
    return result


def contained(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, "#f0f1f3")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def build_group_panel(
    rows: list[dict[str, Any]],
    group: str,
    output: Path,
    count: int,
    seed: int,
) -> list[str]:
    selected = [row for row in rows if row["quantile_group"] == group]
    rng = random.Random(int(seed) + GROUPS.index(group) * 1009)
    chosen = rng.sample(selected, min(count, len(selected)))
    tile_size = (240, 320)
    row_label_h, title_h = 42, 54
    canvas = Image.new(
        "RGB",
        (tile_size[0] * 4, title_h + (row_label_h + tile_size[1]) * len(chosen)),
        "#eceff2",
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, title_h), fill="#d9e0e7")
    draw.text(
        (16, 16),
        f"sim_target group: {GROUP_LABELS[group]} | deterministic sample n={len(chosen)}",
        font=font(18, bold=True),
        fill="#16191d",
    )
    headers = ("mannequin", "reference person", "generated", "generated face crop")
    for index, row in enumerate(chosen):
        y = title_h + index * (row_label_h + tile_size[1])
        tags = [name for name in FAILURE_LABELS if int(row[name])]
        sim = "NA" if row["sim_target"] is None else f"{row['sim_target']:.4f}"
        label = (
            f"{row['key']}  sim={sim}  tags={','.join(tags) if tags else 'none'}"
        )
        draw.rectangle((0, y, canvas.width, y + row_label_h), fill="white")
        draw.text((10, y + 7), label, font=font(13, bold=True), fill="#202328")
        mannequin = Image.open(row["mannequin_path"]).convert("RGB")
        reference = Image.open(row["reference_path"]).convert("RGB")
        generated = Image.open(row["generated_path"]).convert("RGB")
        face_crop = generated.crop(tuple(row["face_crop_bbox"]))
        for column, (image, header) in enumerate(
            zip((mannequin, reference, generated, face_crop), headers, strict=True)
        ):
            x = column * tile_size[0]
            tile = contained(image, tile_size)
            canvas.paste(tile, (x, y + row_label_h))
            draw.rectangle(
                (x, y + row_label_h, x + tile_size[0] - 1, y + row_label_h + 24),
                fill="#000000a0",
            )
            draw.text(
                (x + 7, y + row_label_h + 5),
                header,
                font=font(12, bold=True),
                fill="white",
            )
            draw.rectangle(
                (x, y + row_label_h, x + tile_size[0] - 1, y + row_label_h + tile_size[1] - 1),
                outline="#8d969f",
                width=1,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return [row["key"] for row in chosen]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{float(value):.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribute identity failures using frozen spatial-run metrics and parsing."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/spatial_metrics_v2.yaml"))
    parser.add_argument("--run", default="spatial_quality_repair")
    parser.add_argument("--metrics-dir", type=Path, default=None)
    parser.add_argument("--identity-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/id_failure_attrib"))
    parser.add_argument("--expected-count", type=int, default=400)
    parser.add_argument("--samples-per-group", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--hair-face-threshold", type=float, default=0.25)
    parser.add_argument("--hair-overgrown-threshold", type=float, default=1.5)
    parser.add_argument("--low-conf-threshold", type=float, default=0.6)
    parser.add_argument("--sleeve-threshold", type=float, default=0.15)
    parser.add_argument("--accessory-similarity-threshold", type=float, default=0.60)
    parser.add_argument("--accessory-min-fraction", type=float, default=0.002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    root = Path(cfg["data"]["root"])
    output_root = root / cfg["metrics_v2"]["output_root"]
    metrics_dir = args.metrics_dir or output_root / args.run
    identity_csv = args.identity_csv or metrics_dir / "identity/deltaid_per_image.csv"
    parsing_dir = metrics_dir / "parsing_masks"
    pose_path = metrics_dir / "dwpose_predictions.jsonl"
    rows_raw = read_csv(identity_csv)
    if len(rows_raw) != args.expected_count:
        raise RuntimeError(
            f"identity row count={len(rows_raw)}, expected={args.expected_count}: {identity_csv}"
        )
    keys = [parse_key(row) for row in rows_raw]
    if len(set(keys)) != len(keys):
        raise RuntimeError("identity CSV contains duplicate (mid,jid,seed) rows")
    poses = load_jsonl(pose_path)
    width, height = get_resolution(cfg["data"]["resolution"])
    rows = [
        classify_row(
            row,
            root=root,
            parsing_dir=parsing_dir,
            poses=poses,
            size=(width, height),
            args=args,
        )
        for row in rows_raw
    ]
    p25, p75 = assign_groups(rows)
    rates = rate_table(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, list[str]] = {}
    for group in GROUPS:
        selected[group] = build_group_panel(
            rows,
            group,
            args.output_dir / f"sim_{group}_grid.png",
            args.samples_per_group,
            args.seed,
        )

    rate_by_label = {row["label"]: row for row in rates}
    hair_dominant_labels = [
        label
        for label in ("face_occluded_by_hair", "hair_overgrown")
        if rate_by_label[label]["low_rate"] > 0
        and rate_by_label[label]["low_rate"] >= 2.0 * rate_by_label[label]["high_rate"]
    ]
    verdict = "HAIR-LEAK-DOMINANT" if hair_dominant_labels else "CAPACITY-OR-TRAINING"
    sleeve_rate = float(np.mean([row["sleeve_mismatch"] for row in rows]))
    summary = {
        "run": args.run,
        "identity_csv": str(identity_csv),
        "metrics_dir": str(metrics_dir),
        "count": len(rows),
        "valid_sim_target": sum(row["sim_target"] is not None for row in rows),
        "p25": p25,
        "p75": p75,
        "thresholds": {
            "hair_face": args.hair_face_threshold,
            "hair_overgrown": args.hair_overgrown_threshold,
            "low_conf": args.low_conf_threshold,
            "sleeve": args.sleeve_threshold,
            "accessory_similarity": args.accessory_similarity_threshold,
            "accessory_min_fraction": args.accessory_min_fraction,
        },
        "group_counts": {
            group: sum(row["quantile_group"] == group for row in rows)
            for group in GROUPS
        },
        "rates": rates,
        "sleeve_mismatch_rate": sleeve_rate,
        "hair_dominant_labels": hair_dominant_labels,
        "verdict": verdict,
        "selection_seed": args.seed,
        "selected_panel_keys": selected,
    }

    fields = [
        "mid", "jid", "seed", "key", "garment_type", "quantile_group",
        "generated_path", "identity_status", "identity_error", "sim_target",
        "sim_source", "delta_id", "det_conf", "face_size_px",
        "face_hair_fraction", "face_visible_fraction", "generated_hair_fraction",
        "reference_hair_fraction", "hair_area_ratio", "accessory_similarity",
        "reference_accessory_fraction", "generated_accessory_fraction",
        "source_forearm_skin_ratio", "generated_forearm_skin_ratio",
        "forearm_skin_ratio_difference", *FAILURE_LABELS,
    ]
    write_csv(args.output_dir / "per_image.csv", rows, fields)
    write_csv(
        args.output_dir / "cross_table.csv",
        rates,
        [
            "label", "low_count", "low_total", "low_rate", "middle_count",
            "middle_total", "middle_rate", "high_count", "high_total", "high_rate",
            "low_over_high", "point_biserial_r", "point_biserial_p",
        ],
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Spatial sim_target Failure Attribution",
        "",
        f"**Conclusion: {verdict}.**",
        f"**sleeve_mismatch overall rate: {sleeve_rate:.2%} ({sum(row['sleeve_mismatch'] for row in rows)}/{len(rows)}).**",
        "",
        f"Run: `{args.run}`; images: `{len(rows)}`; valid sim_target: `{summary['valid_sim_target']}`.",
        f"Quantiles over valid rows: P25=`{p25:.6f}`, P75=`{p75:.6f}`. Detector failures are assigned to the low group.",
        f"Deterministic panel seed: `{args.seed}`.",
        "",
        "## Failure Rates By sim_target Group",
        "",
        "| label | low | middle | high | low/high | point-biserial r | p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rates:
        lines.append(
            f"| `{row['label']}` | {row['low_rate']:.2%} ({row['low_count']}/{row['low_total']}) | "
            f"{row['middle_rate']:.2%} ({row['middle_count']}/{row['middle_total']}) | "
            f"{row['high_rate']:.2%} ({row['high_count']}/{row['high_total']}) | "
            f"{fmt(row['low_over_high'], 2)} | {fmt(row['point_biserial_r'])} | {fmt(row['point_biserial_p'], 6)} |"
        )
    lines.extend(
        [
            "",
            "The preregistered hair-dominant rule is low-group rate >= 2x high-group rate for either `face_occluded_by_hair` or `hair_overgrown`.",
            f"Triggered hair labels: `{hair_dominant_labels or 'none'}`.",
            "",
            "## Deterministic Qualitative Grids",
            "",
            "Each row is `[mannequin | reference person | generated | enlarged generated-face crop]`.",
            "",
            "- [low sim_target](sim_low_grid.png)",
            "- [middle sim_target](sim_middle_grid.png)",
            "- [high sim_target](sim_high_grid.png)",
            "",
            "## Definitions And Limitations",
            "",
            f"- `face_occluded_by_hair`: hair occupies more than {args.hair_face_threshold:.0%} of a DWPose/parsing-derived geometric face zone. FASHN labels are mutually exclusive, so direct face-mask/hair-mask intersection would be identically zero.",
            f"- `face_undetected_or_lowconf`: held-out runner detection failed or RetinaFace confidence is below {args.low_conf_threshold:.2f}.",
            f"- `hair_overgrown`: generated/reference full-frame hair-area ratio exceeds {args.hair_overgrown_threshold:.2f}.",
            "- `accessory_leak` is explicitly heuristic: small jewelry/glasses or high-saturation components in reference/generated ear bands must have a similar hue histogram. It is not treated as a causal detector.",
            f"- `sleeve_mismatch`: the absolute change in parsed arms/hands occupancy inside source-DWPose elbow-to-wrist capsules exceeds {args.sleeve_threshold:.2f}.",
            "- Existing generated parsing, DWPose predictions, and held-out identity CSV are reused read-only; no image generation or training occurs.",
            "",
            "Machine-readable outputs: [per_image.csv](per_image.csv), [cross_table.csv](cross_table.csv), [summary.json](summary.json).",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
