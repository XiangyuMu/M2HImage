from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from conditions import find_one, load_yaml, read_ids
from eval_b2 import garment_type
from synth_head_keypoints import HEAD_EDGES, _theta, synth_head_keypoints


# The synthetic template uses image-left to image-right ordering. In OpenPose-18,
# right-eye/right-ear (14/16) appear on image-left for a frontal subject.
REAL_HEAD_INDICES = np.asarray([0, 14, 15, 16, 17], dtype=np.int64)


def select_ids(root: Path, split: Path, per_type: int = 5) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in sorted(read_ids(root / split)):
        kind = garment_type(root, sample_id)
        if kind in {"top", "dress", "pants", "skirt"}:
            groups[kind].append(sample_id)
    ordered = ("top", "dress", "pants", "skirt")
    missing = {kind: len(groups[kind]) for kind in ordered if len(groups[kind]) < per_type}
    if missing:
        raise RuntimeError(f"insufficient head-audit strata: {missing}")
    return [
        groups[kind][rank]
        for rank in range(per_type)
        for kind in ordered
    ]


def geometry(
    root: Path,
    sample_id: str,
    width: int,
    height: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    payload = np.load(root / "dwpose/keypoints/mannequin" / f"{sample_id}.npz")
    body = np.asarray(payload["body"], dtype=np.float32)
    scores = np.asarray(payload["body_scores"], dtype=np.float32)
    scale_xy = np.asarray([width, height], dtype=np.float32)
    neck = body[1] * scale_xy
    shoulder_width = (
        abs(float(body[2, 0] - body[5, 0])) * width
        if scores[2] >= 0.3 and scores[5] >= 0.3
        else 0.22 * width
    )
    actual = body[REAL_HEAD_INDICES] * scale_xy
    actual_scores = scores[REAL_HEAD_INDICES]
    return neck, shoulder_width, actual, actual_scores


def synthetic(
    theta: dict[str, float],
    neck: np.ndarray,
    shoulder_width: float,
    shoulder_scale: float,
    nose_offset_scale: float,
) -> np.ndarray:
    scale = max(24.0, shoulder_scale * shoulder_width)
    return synth_head_keypoints(
        theta, neck, scale, nose_offset_scale=nose_offset_scale
    )


def point_error(
    predicted: np.ndarray,
    actual: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
) -> float:
    valid = scores >= 0.3
    if not valid.any():
        return math.nan
    return float(
        np.linalg.norm(predicted[valid] - actual[valid], axis=1).mean()
        / np.hypot(width, height)
    )


def grid_fit(
    samples: list[dict[str, Any]],
    width: int,
    height: int,
) -> dict[str, float]:
    best = {
        "error": math.inf,
        "shoulder_scale": 0.48,
        "nose_offset_scale": 1.62,
    }
    for shoulder_scale in np.linspace(0.30, 0.65, 36):
        for nose_offset in np.linspace(0.90, 2.20, 53):
            errors = []
            for row in samples:
                predicted = synthetic(
                    row["theta"],
                    row["neck"],
                    row["shoulder_width"],
                    float(shoulder_scale),
                    float(nose_offset),
                )
                error = point_error(
                    predicted,
                    row["actual"],
                    row["scores"],
                    width,
                    height,
                )
                if math.isfinite(error):
                    errors.append(error)
            median = float(np.median(errors))
            if median < best["error"]:
                best = {
                    "error": median,
                    "shoulder_scale": float(shoulder_scale),
                    "nose_offset_scale": float(nose_offset),
                }
    return best


def draw_overlay(
    image: Image.Image,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for left, right in HEAD_EDGES:
        draw.line(
            tuple(actual[left]) + tuple(actual[right]),
            fill=(50, 230, 80),
            width=4,
        )
        draw.line(
            tuple(predicted[left]) + tuple(predicted[right]),
            fill=(255, 70, 70),
            width=3,
        )
    for points, color in ((actual, (50, 230, 80)), (predicted, (255, 70, 70))):
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    draw.rectangle((0, 0, 460, 28), fill="white")
    draw.text((8, 7), "green=real DWPose, red=synthetic theta template", fill="black")
    return output


def synthetic_ratio_history(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        rows = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    ratio = float(row["head_control_synthetic_ratio"])
                    step = int(row["step"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                rows.append((step, ratio))
        result.append(
            {
                "path": str(path),
                "count": len(rows),
                "first_step": rows[0][0] if rows else None,
                "last_step": rows[-1][0] if rows else None,
                "mean": float(np.mean([value for _, value in rows])) if rows else None,
                "near_step2000": [
                    {"step": step, "ratio": value}
                    for step, value in rows
                    if 1900 <= step <= 2050
                ][-8:],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit synthetic theta head landmarks against real mannequin DWPose."
    )
    parser.add_argument(
        "--config",
        default="configs/spatial_warmup_resume_hair_incontext.yaml",
    )
    parser.add_argument(
        "--output", default="artifacts/synth_head_kp_audit"
    )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    root = Path(cfg["data"]["root"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    width = int(cfg["data"]["resolution"]["width"])
    height = int(cfg["data"]["resolution"]["height"])
    ids = select_ids(
        root, Path(cfg["data"]["val_split"]), per_type=max(1, args.limit // 4)
    )[: args.limit]
    head_cfg = cfg["model"]["spatial_conditions"]["head_control"]
    current_shoulder = float(head_cfg.get("shoulder_scale", 0.48))
    current_nose = float(head_cfg.get("nose_offset_scale", 1.62))
    samples = []
    for sample_id in ids:
        neck, shoulder_width, actual, scores = geometry(
            root, sample_id, width, height
        )
        samples.append(
            {
                "sample_id": sample_id,
                "garment_type": garment_type(root, sample_id),
                "theta": _theta(root, sample_id),
                "neck": neck,
                "shoulder_width": shoulder_width,
                "actual": actual,
                "scores": scores,
            }
        )
    fit = grid_fit(samples, width, height)
    csv_rows = []
    for row in samples:
        current = synthetic(
            row["theta"], row["neck"], row["shoulder_width"],
            current_shoulder, current_nose,
        )
        fitted = synthetic(
            row["theta"], row["neck"], row["shoulder_width"],
            fit["shoulder_scale"], fit["nose_offset_scale"],
        )
        valid = row["scores"] >= 0.3
        center_delta = (
            current[valid].mean(axis=0) - row["actual"][valid].mean(axis=0)
            if valid.any()
            else np.asarray([math.nan, math.nan])
        )
        current_error = point_error(
            current, row["actual"], row["scores"], width, height
        )
        fitted_error = point_error(
            fitted, row["actual"], row["scores"], width, height
        )
        csv_rows.append(
            {
                "sample_id": row["sample_id"],
                "garment_type": row["garment_type"],
                "current_error": current_error,
                "fitted_error": fitted_error,
                "center_dx_px": float(center_delta[0]),
                "center_dy_px": float(center_delta[1]),
                "shoulder_width_px": float(row["shoulder_width"]),
            }
        )
        image = Image.open(
            find_one(root / "images/mannequin", row["sample_id"])
        ).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        draw_overlay(image, row["actual"], current).save(
            output / f"{row['sample_id']}_overlay.png"
        )
    with (output / "per_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    current_errors = np.asarray(
        [row["current_error"] for row in csv_rows], dtype=np.float64
    )
    fitted_errors = np.asarray(
        [row["fitted_error"] for row in csv_rows], dtype=np.float64
    )
    improvement = 1.0 - np.median(fitted_errors) / np.median(current_errors)
    systematic_bias = bool(
        np.median(current_errors) > 0.015 and improvement >= 0.15
    )
    run_root = Path(cfg["experiment"]["output_root"])
    history_paths = [
        run_root
        / "phase1_spatial_hair_region_resume_r16_4400_768x1024/logs/train.jsonl",
        run_root
        / "phase1_spatial_hair_incontext_resume_v2_r16_4400_768x1024/logs/train.jsonl",
    ]
    history = synthetic_ratio_history(history_paths)
    result = {
        "sample_ids": ids,
        "current": {
            "shoulder_scale": current_shoulder,
            "nose_offset_scale": current_nose,
            "median_error": float(np.median(current_errors)),
            "mean_error": float(np.mean(current_errors)),
            "median_center_dx_px": float(
                np.median([row["center_dx_px"] for row in csv_rows])
            ),
            "median_center_dy_px": float(
                np.median([row["center_dy_px"] for row in csv_rows])
            ),
        },
        "fitted": {
            **fit,
            "median_sample_error": float(np.median(fitted_errors)),
            "relative_median_improvement": float(improvement),
        },
        "synthetic_ratio_history": history,
        "verdict": "SYSTEMATIC-BIAS" if systematic_bias else "NO-SYSTEMATIC-BIAS",
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Synthetic Head-Keypoint Audit",
        "",
        f"Conclusion: **{result['verdict']}**.",
        "",
        (
            f"Current template median normalized point error: "
            f"`{np.median(current_errors):.6f}`."
        ),
        (
            f"Grid-fitted median error: `{np.median(fitted_errors):.6f}` "
            f"({improvement:.1%} relative improvement)."
        ),
        (
            f"Current shoulder/nose scales: `{current_shoulder:.3f}` / "
            f"`{current_nose:.3f}`; fitted: `{fit['shoulder_scale']:.3f}` / "
            f"`{fit['nose_offset_scale']:.3f}`."
        ),
        (
            f"Median center bias: dx="
            f"`{result['current']['median_center_dx_px']:.2f}px`, dy="
            f"`{result['current']['median_center_dy_px']:.2f}px`."
        ),
        "",
        "## Synthetic-Control History",
        "",
    ]
    for row in history:
        lines.append(
            f"- `{row['path']}`: rows={row['count']}, mean ratio="
            f"`{row['mean']}`, near step-2000={row['near_step2000']}"
        )
    lines.extend([
        "",
        "## Visual Overlays",
        "",
        "Green is the real mannequin DWPose head; red is the synthesized theta template.",
    ])
    for sample_id in ids:
        lines.append(f"- [{sample_id}]({sample_id}_overlay.png)")
    (output / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
