from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, rankdata, wilcoxon

from conditions import load_yaml


KEY_FIELDS = ("mid", "jid", "seed")


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["mid"]), str(row["jid"]), int(row["seed"])


def metric_bundle(metrics_dir: str | Path) -> dict[str, float]:
    path = Path(metrics_dir)
    garment = read_json(path / "garment_summary.json")
    hair = read_json(path / "hair_summary.json")
    pose = read_json(path / "pose_summary.json")
    identity = read_json(path / "identity_summary.json")
    distribution = read_json(path / "distribution_summary.json")
    return {
        "garment_dino": float(garment["garment_dino"]["mean"]),
        "garment_hf_lpips": float(garment["garment_hf_lpips"]["mean"]),
        "print_hf_lpips": (
            float(garment["print_region_hf_lpips"]["mean"])
            if garment["print_region_hf_lpips"]["mean"] is not None
            else float("nan")
        ),
        "hair_dino": float(hair["hair_dino"]["mean"]),
        "hair_lab": float(hair["hair_lab_distance"]["mean"]),
        "head5": float(pose["head5_distance"]["mean"]),
        "body": float(pose["body_distance"]["mean"]),
        "sim_target": float(identity["sim_target_mean"]),
        "delta_id": float(identity["mean"]),
        "face_detection_rate": float(identity["face_detection_rate"]),
        "det_conf": float(identity.get("det_conf_mean", float("nan"))),
        "fid": float(distribution["fid"]),
        "kid": float(distribution["kid_mean"]),
    }



def legacy_garment_bundle(metrics_dir: str | Path) -> dict[str, float]:
    summary = read_json(Path(metrics_dir) / "garment_summary.json")
    return {
        "garment_sim": float(summary["cross_identity_group_mean_mean"]),
        "garment_source_dino": float(summary["source_similarity_mean"]),
    }


def train_log_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "logs/train.jsonl"

def paired_wilcoxon(
    csv_a: str | Path,
    csv_b: str | Path,
    field: str,
    alternative: str,
) -> dict[str, float | int]:
    left = {row_key(row): row for row in read_csv(csv_a)}
    right = {row_key(row): row for row in read_csv(csv_b)}
    keys = sorted(set(left) & set(right))
    pairs = []
    for key in keys:
        a = finite(left[key].get(field))
        b = finite(right[key].get(field))
        if a is not None and b is not None:
            pairs.append((a, b))
    if not pairs:
        raise RuntimeError(f"no paired finite rows for {field}: {csv_a} vs {csv_b}")
    a = np.asarray([row[0] for row in pairs], dtype=np.float64)
    b = np.asarray([row[1] for row in pairs], dtype=np.float64)
    diff = a - b
    nonzero = diff != 0.0
    if not np.any(nonzero):
        p_value = 1.0
        effect = 0.0
    else:
        p_value = float(
            wilcoxon(a, b, alternative=alternative, zero_method="wilcox").pvalue
        )
        ranks = rankdata(np.abs(diff[nonzero]))
        signed = np.sign(diff[nonzero])
        effect = float(np.sum(ranks * signed) / np.sum(ranks))
    return {
        "n": int(len(a)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_delta": float(diff.mean()),
        "median_delta": float(np.median(diff)),
        "p": p_value,
        "rank_biserial": effect,
    }


def attribution_rows(path: str | Path) -> dict[tuple[str, str, int], dict[str, str]]:
    return {row_key(row): row for row in read_csv(path)}


def attribution_rate(path: str | Path, label: str) -> float:
    rows = list(attribution_rows(path).values())
    if not rows:
        raise RuntimeError(f"empty attribution CSV: {path}")
    return float(np.mean([int(row[label]) for row in rows]))


def paired_mcnemar_less(
    control_csv: str | Path,
    experiment_csv: str | Path,
    label: str,
) -> dict[str, float | int]:
    control = attribution_rows(control_csv)
    experiment = attribution_rows(experiment_csv)
    keys = sorted(set(control) & set(experiment))
    improved = sum(
        int(control[key][label]) == 1 and int(experiment[key][label]) == 0
        for key in keys
    )
    worsened = sum(
        int(control[key][label]) == 0 and int(experiment[key][label]) == 1
        for key in keys
    )
    discordant = improved + worsened
    p_value = (
        float(binomtest(improved, discordant, p=0.5, alternative="greater").pvalue)
        if discordant
        else 1.0
    )
    return {
        "n": len(keys),
        "improved": int(improved),
        "worsened": int(worsened),
        "p": p_value,
    }


def gate_shift(train_jsonl: str | Path, field: str = "hair_gate") -> dict[str, float]:
    rows = []
    path = Path(train_jsonl)
    if not path.is_file():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = finite(row.get(field))
        if value is not None:
            rows.append((int(row.get("step", row.get("global_step", len(rows)))), value))
    if not rows:
        raise RuntimeError(f"{field} never logged in {path}")
    return {
        "first_step": float(rows[0][0]),
        "last_step": float(rows[-1][0]),
        "initial": float(rows[0][1]),
        "final": float(rows[-1][1]),
        "shift": float(abs(rows[-1][1] - rows[0][1])),
    }


def fairness_signature(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    launch = read_json(run_dir / "launch.json")
    cfg = load_yaml(run_dir / "resolved_config.yaml")
    runtime = cfg["_runtime"]
    return {
        "resume_trainable_hash": launch.get("resume_trainable_hash"),
        "train_ids_hash": launch.get("train_ids_hash"),
        "resume_sampler_state": launch.get("resume_sampler_state"),
        "seed": cfg["experiment"]["seed"],
        "global_batch": runtime["global_batch"],
        "effective_lr": runtime["effective_lr"],
        "baseline_lr": cfg["training"]["baseline_lr"],
        "rank": cfg["model"]["lora_rank"],
        "resume_step": runtime["resume_step"],
        "target_step": runtime["target_step"],
        "executed_steps": int(runtime["target_step"]) - int(runtime["resume_step"]),
    }


def assert_fair_pair(left_dir: str | Path, right_dir: str | Path) -> dict[str, Any]:
    left = fairness_signature(left_dir)
    right = fairness_signature(right_dir)
    differences = {
        key: {"left": left[key], "right": right[key]}
        for key in left
        if left[key] != right[key]
    }
    if differences:
        raise RuntimeError(
            "fairness fields differ: " + json.dumps(differences, ensure_ascii=False)
        )
    return left


def watcher_bundle(path: str | Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    row = read_json(path)
    metrics = row["watcher_metrics"]
    return {
        "garment_dino_median": float(metrics["garment_dino"]),
        "hair_dino_median": float(metrics["hair_dino"]),
        "head5_median": float(metrics["head5_distance"]),
        "body_median": float(metrics["body_distance"]),
        "face_detection_rate": float(row["guards"]["face_detection_rate"]),
        "swap_cosine_median": float(row["guards"]["swap_cosine_median"]),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"
