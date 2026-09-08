from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from conditions import find_one, get_resolution, read_ids


HEAD_EDGES = ((1, 0), (0, 2), (3, 1), (2, 4))
HEAD_COLORS = ((255, 80, 80), (255, 180, 60), (80, 220, 255), (180, 100, 255))


def synth_head_keypoints(
    theta: Mapping[str, float] | Sequence[float],
    neck_kp: Sequence[float],
    scale: float,
    nose_offset_scale: float = 1.62,
) -> np.ndarray:
    """Rotate a five-point head template and anchor it above the OpenPose neck."""
    if isinstance(theta, Mapping):
        yaw = float(theta.get("yaw", 0.0))
        pitch = float(theta.get("pitch", 0.0))
        roll = float(theta.get("roll", 0.0))
    else:
        yaw, pitch, roll = (float(value) for value in theta)
    yaw_r, pitch_r, roll_r = map(math.radians, (yaw, pitch, roll))
    # nose, left eye, right eye, left ear, right ear in a head-local frame
    points = np.asarray(
        [[0.0, 0.02], [-0.22, -0.18], [0.22, -0.18], [-0.48, -0.04], [0.48, -0.04]],
        dtype=np.float32,
    )
    points[:, 0] *= max(0.20, math.cos(yaw_r))
    points[0, 0] += 0.24 * math.sin(yaw_r)
    points[:, 1] += 0.12 * math.sin(pitch_r)
    rotation = np.asarray(
        [[math.cos(roll_r), -math.sin(roll_r)], [math.sin(roll_r), math.cos(roll_r)]],
        dtype=np.float32,
    )
    points = points @ rotation.T
    # The template origin is the nose, not the head center. On the mannequin controls,
    # a 999-sample calibration puts (neck_y - nose_y) / scale at median 1.604.
    # The local nose has y=+0.02, so an anchor offset of 1.62 matches that median.
    anchor = np.asarray(neck_kp, dtype=np.float32) - np.asarray(
        [0.0, float(nose_offset_scale) * float(scale)], dtype=np.float32
    )
    return points * float(scale) + anchor


def _pose_geometry(
    root: Path,
    sample_id: str,
    width: int,
    height: int,
    shoulder_scale: float = 0.48,
) -> tuple[np.ndarray, float]:
    path = root / "dwpose/keypoints/mannequin" / f"{sample_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"with-head mannequin keypoints missing: {path}")
    payload = np.load(path)
    body = np.asarray(payload["body"], dtype=np.float32)
    scores = np.asarray(payload["body_scores"], dtype=np.float32)
    if body.shape != (18, 2) or scores.shape != (18,):
        raise RuntimeError(f"unexpected DWPose body shape for {sample_id}: {body.shape}/{scores.shape}")
    neck = body[1] * np.asarray([width, height], dtype=np.float32)
    if scores[2] >= 0.3 and scores[5] >= 0.3:
        shoulder_width = abs(float(body[2, 0] - body[5, 0])) * width
    else:
        shoulder_width = 0.22 * width
    return neck, max(24.0, float(shoulder_scale) * shoulder_width)


def _theta(root: Path, sample_id: str) -> dict[str, float]:
    path = root / "derived/head_pose_6drepnet/human" / f"{sample_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("status", "unknown")) != "ok":
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    return {key: float(payload.get(key, 0.0)) for key in ("yaw", "pitch", "roll")}


def draw_head_keypoints(image: Image.Image, points: np.ndarray, radius: int = 4) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for edge, color in zip(HEAD_EDGES, HEAD_COLORS, strict=True):
        left, right = points[edge[0]], points[edge[1]]
        draw.line((float(left[0]), float(left[1]), float(right[0]), float(right[1])), fill=color, width=3)
    for point in points:
        x, y = float(point[0]), float(point[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 255, 255))
    return output


def synth_head_control_image(
    root: str | Path,
    sample_id: str,
    resolution,
    nose_offset_scale: float = 1.62,
    shoulder_scale: float = 0.48,
) -> Image.Image:
    root = Path(root)
    width, height = get_resolution(resolution)
    base = Image.open(find_one(root / "dwpose/without_head/mannequin", sample_id)).convert("RGB")
    if base.size != (width, height):
        base = base.resize((width, height), Image.Resampling.BICUBIC)
    neck, scale = _pose_geometry(
        root, sample_id, width, height, shoulder_scale=shoulder_scale
    )
    points = synth_head_keypoints(_theta(root, sample_id), neck, scale, nose_offset_scale)
    return draw_head_keypoints(base, points)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render synthetic five-point head controls for visual audit.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="splits/val.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--shoulder-scale", type=float, default=0.48)
    parser.add_argument("--nose-offset-scale", type=float, default=1.62)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for sample_id in read_ids(root / args.split)[: args.limit]:
        synth_head_control_image(
            root,
            sample_id,
            (args.width, args.height),
            shoulder_scale=args.shoulder_scale,
            nose_offset_scale=args.nose_offset_scale,
        ).save(output / f"{sample_id}.png")


if __name__ == "__main__":
    main()
