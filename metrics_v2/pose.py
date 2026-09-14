from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from metrics.common import plot_histogram
from metrics_v2.common import metric_summary, write_csv, write_json


FIELDS = [
    "mid",
    "jid",
    "seed",
    "garment_type",
    "generated_path",
    "status",
    "error",
    "body_distance_to_mannequin",
    "head5_distance_to_mannequin",
    "body_valid_points",
    "head5_valid_points",
]


def normalized_keypoint_distance(
    generated: np.ndarray,
    generated_scores: np.ndarray,
    target: np.ndarray,
    target_scores: np.ndarray,
    indices: list[int],
    width: int,
    height: int,
    threshold: float,
) -> tuple[float | None, int]:
    valid = np.asarray(
        [index for index in indices if generated_scores[index] >= threshold and target_scores[index] >= threshold],
        dtype=np.int64,
    )
    if valid.size == 0:
        return None, 0
    scale = np.asarray([width, height], dtype=np.float32)
    distances_px = np.linalg.norm((generated[valid] - target[valid]) * scale, axis=1)
    diagonal = float(np.hypot(width, height))
    return float(distances_px.mean() / diagonal), int(valid.size)


def _prediction_key(row: dict[str, Any]) -> str:
    return f"{row['mid']}__id{row['jid']}__seed{int(row['seed'])}"


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["key"]): row for row in rows}


def _run_helper(cfg: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path, pose_device: str) -> Path:
    pcfg = cfg["metrics_v2"]["pose"]
    manifest = out_dir / "dwpose_manifest.jsonl"
    predictions = out_dir / "dwpose_predictions.jsonl"
    existing = _load_jsonl(predictions)
    if len(existing) == len(rows) and all(_prediction_key(row) in existing for row in rows):
        return predictions
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"key": _prediction_key(row), "path": str(row["path"])}) + "\n")
    script = Path(__file__).with_name("dwpose_server.py")
    command = [
        str(pcfg["helper_python"]),
        str(script),
        "--manifest",
        str(manifest),
        "--output",
        str(predictions),
        "--repo-root",
        str(pcfg["repo_root"]),
        "--detector",
        str(pcfg["detector_checkpoint"]),
        "--pose",
        str(pcfg["pose_checkpoint"]),
        "--provider",
        str(pcfg.get("provider", "CUDAExecutionProvider")),
    ]
    env = os.environ.copy()
    if str(pose_device).startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = str(pose_device).split(":", 1)[1]
    subprocess.run(command, check=True, env=env)
    return predictions


def run_pose_metrics(
    cfg: dict[str, Any], rows: list[dict[str, Any]], out_dir: str | Path, pose_device: str
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["data"]["root"])
    pcfg = cfg["metrics_v2"]["pose"]
    threshold = float(pcfg.get("score_threshold", 0.3))
    body_indices = [int(value) for value in pcfg["body_indices"]]
    head_indices = [int(value) for value in pcfg["head_indices"]]
    predictions = _load_jsonl(_run_helper(cfg, rows, out_dir, pose_device))
    csv_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="DWPose distance to mannequin"):
        output = {
            "mid": row["mid"],
            "jid": row["jid"],
            "seed": row["seed"],
            "garment_type": row.get("garment_type", "unknown"),
            "generated_path": str(row["path"]),
            "status": "ok",
            "error": "",
            "body_distance_to_mannequin": "",
            "head5_distance_to_mannequin": "",
            "body_valid_points": 0,
            "head5_valid_points": 0,
        }
        try:
            prediction = predictions[_prediction_key(row)]
            if prediction.get("status") != "ok":
                raise RuntimeError(prediction.get("error", "DWPose failed"))
            target_path = root / "dwpose/keypoints/mannequin" / f"{row['mid']}.npz"
            if not target_path.exists():
                raise FileNotFoundError(f"missing mannequin keypoints: {target_path}")
            target = np.load(target_path)
            generated_body = np.asarray(prediction["body"], dtype=np.float32)
            generated_scores = np.asarray(prediction["body_scores"], dtype=np.float32)
            target_body = np.asarray(target["body"], dtype=np.float32)
            target_scores = np.asarray(target["body_scores"], dtype=np.float32)
            width = int(prediction["width"])
            height = int(prediction["height"])
            body_distance, body_count = normalized_keypoint_distance(
                generated_body, generated_scores, target_body, target_scores, body_indices, width, height, threshold
            )
            head_distance, head_count = normalized_keypoint_distance(
                generated_body, generated_scores, target_body, target_scores, head_indices, width, height, threshold
            )
            if body_distance is None and head_distance is None:
                raise RuntimeError("no mutually confident body or head keypoints")
            output.update(
                {
                    "body_distance_to_mannequin": body_distance if body_distance is not None else "",
                    "head5_distance_to_mannequin": head_distance if head_distance is not None else "",
                    "body_valid_points": body_count,
                    "head5_valid_points": head_count,
                }
            )
        except Exception as exc:  # noqa: BLE001
            output["status"] = "failed"
            output["error"] = str(exc)
        csv_rows.append(output)
    write_csv(out_dir / "pose_per_image.csv", csv_rows, FIELDS)
    body_valid = [row for row in csv_rows if row["body_distance_to_mannequin"] != ""]
    head_valid = [row for row in csv_rows if row["head5_distance_to_mannequin"] != ""]
    summary = {
        "status": "ok" if body_valid or head_valid else "failed",
        "measures_against": "with-head DWPose keypoints from dwpose/keypoints/mannequin/{mid}.npz",
        "normalization": "mean pixel L2 divided by image diagonal; confidence >= 0.3 on both sides",
        "body_indices": body_indices,
        "head5_indices": head_indices,
        "count": len(csv_rows),
        "failed": sum(row["status"] == "failed" for row in csv_rows),
        "body_distance": metric_summary(row["body_distance_to_mannequin"] for row in body_valid),
        "head5_distance": metric_summary(row["head5_distance_to_mannequin"] for row in head_valid),
        "csv": str(out_dir / "pose_per_image.csv"),
    }
    write_json(out_dir / "pose_summary.json", summary)
    plot_histogram(
        out_dir / "head5_distance_hist.png",
        [float(row["head5_distance_to_mannequin"]) for row in head_valid],
        "Head five-point distance to mannequin",
        "diagonal-normalized distance (lower is better)",
    )
    return summary
