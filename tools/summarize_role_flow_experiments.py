#!/usr/bin/env python3
"""Summarize the A/B/C role-flow experiments without inventing missing results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENT_ROOT = Path("/data/muxiangyu/experiments/M2H_Final_v2_clean_v1")
DEFAULT_CONFIGS = {
    "A": Path("configs/role_selective/A_timestep_routed.yaml"),
    "B": Path("configs/role_selective/B_garment_weighted.yaml"),
    "C": Path("configs/role_selective/C_async_flow.yaml"),
}
DEFAULT_RUN_IDS = {
    "A": "A_timestep_routed",
    "B": "B_garment_weighted",
    "C": "C_async_flow",
}
METRIC_DIRECTIONS = {
    "id_cosine": "higher",
    "tar_at_1e-3": "higher",
    "garment_dino": "higher",
    "garment_iou": "higher",
    "pose_pck": "higher",
    "bg_ssim": "higher",
    "bg_lpips": "lower",
    "fid": "lower",
}
CSV_FIELDS = (
    "method",
    "status",
    "experiment_id",
    "description",
    "config",
    "config_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_ready",
    "training_status",
    "training_step",
    "target_steps",
    "seconds_per_step",
    "peak_gib",
    "last_loss_total",
    "eval_status",
    "eval_split",
    "evaluated",
    "ok",
    "failed",
    "manifest_sha256",
    *METRIC_DIRECTIONS.keys(),
    "pending_reasons",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    value = Path(path)
    if not value.exists():
        return "missing"
    if value.is_file():
        return sha256_file(value)
    digest = hashlib.sha256()
    for child in sorted(item for item in value.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(value)).encode("utf-8"))
        digest.update(sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        from conditions import load_yaml

        payload = load_yaml(path)
    return dict(payload or {})


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _load_set_metrics(eval_summary_path: Path) -> tuple[dict[str, Any], Path]:
    path = eval_summary_path.with_name("set_metrics.json")
    if not path.is_file():
        return {}, path
    return _read_json(path), path


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _last_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    last: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = dict(json.loads(line))
    return last


def _extract_seconds_per_step(status: dict[str, Any], last_train: dict[str, Any]) -> float | None:
    for key in ("seconds_per_step", "sec_per_step", "step_seconds", "avg_step_seconds"):
        value = _finite(status.get(key, last_train.get(key)))
        if value is not None:
            return value
    duration = _finite(status.get("duration_seconds", status.get("elapsed_seconds")))
    steps = _finite(status.get("run_step", status.get("step")))
    if duration is not None and steps and steps > 0:
        return duration / steps
    return None


def _metric_template() -> dict[str, dict[str, Any]]:
    return {
        name: {"mean": None, "median": None, "count": 0, "failed": 0, "direction": direction}
        for name, direction in METRIC_DIRECTIONS.items()
    }


def _summarize_one(
    method: str,
    config_path: Path,
    run_dir: Path,
    eval_summary_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    pending_reasons: list[str] = []
    config: dict[str, Any] = {}
    if config_path.is_file():
        config = _load_yaml(config_path)
    else:
        pending_reasons.append("missing config")
    experiment_cfg = dict(config.get("experiment", {}))
    method_cfg = dict(config.get("experiment_method", {}))
    training_cfg = dict(config.get("training", {}))
    model_cfg = dict(config.get("model", {}))
    data_cfg = dict(config.get("data", {}))

    status_path = run_dir / "training_status.json"
    train_path = run_dir / "logs" / "train.jsonl"
    training_status = _read_json(status_path) if status_path.is_file() else {}
    if not training_status:
        pending_reasons.append("missing training status")
    last_train = _last_jsonl(train_path)
    checkpoint_ready = (checkpoint_path / "READY").is_file() if checkpoint_path.is_dir() else checkpoint_path.is_file()
    if not checkpoint_ready:
        pending_reasons.append("missing checkpoint READY")

    metrics = _metric_template()
    eval_summary: dict[str, Any] = {}
    set_metrics: dict[str, Any] = {}
    set_metrics_path = eval_summary_path.with_name("set_metrics.json")
    if eval_summary_path.is_file():
        eval_summary = _read_json(eval_summary_path)
        for name in METRIC_DIRECTIONS:
            if name == "fid":
                continue
            item = dict(eval_summary.get("metrics", {}).get(name, {}))
            metrics[name].update(
                {
                    "mean": _finite(item.get("mean")),
                    "median": _finite(item.get("median")),
                    "count": int(item.get("count", 0) or 0),
                    "failed": int(item.get("failed", 0) or 0),
                }
            )
        set_metrics, set_metrics_path = _load_set_metrics(eval_summary_path)
        fid_item = dict(set_metrics.get("fid", {}))
        if fid_item:
            metrics["fid"].update(
                {
                    "mean": _finite(fid_item.get("value")),
                    "median": _finite(fid_item.get("value")),
                    "count": int(fid_item.get("generated_count", 0) or 0),
                    "failed": 0 if _finite(fid_item.get("value")) is not None else 1,
                }
            )
        elif "fid" in eval_summary.get("metrics", {}):
            item = dict(eval_summary["metrics"]["fid"])
            metrics["fid"].update(
                {
                    "mean": _finite(item.get("mean")),
                    "median": _finite(item.get("median")),
                    "count": int(item.get("count", 0) or 0),
                    "failed": int(item.get("failed", 0) or 0),
                }
            )
        if not set_metrics:
            pending_reasons.append("missing v2 set metrics")
    else:
        pending_reasons.append("missing eval summary")

    sample_counts = dict(eval_summary.get("sample_counts", {}))
    training = {
        "status": training_status.get("status"),
        "step": training_status.get("step", last_train.get("step")),
        "run_step": training_status.get("run_step", last_train.get("run_step")),
        "target_steps": training_status.get("target_steps", training_cfg.get("total_steps")),
        "seconds_per_step": _extract_seconds_per_step(training_status, last_train),
        "peak_gib": _finite(last_train.get("peak_gib", training_status.get("peak_gib"))),
        "last_loss_total": _finite(last_train.get("loss_total")),
        "last_loss_pair": _finite(last_train.get("loss_pair")),
    }
    status = "complete" if not pending_reasons and eval_summary.get("status") in {"ok", "complete", "completed_with_failures"} else "pending"
    return {
        "method": method,
        "status": status,
        "pending_reasons": pending_reasons,
        "experiment_id": experiment_cfg.get("id", DEFAULT_RUN_IDS.get(method, method)),
        "description": method_cfg.get("description", ""),
        "structure": {
            "method_name": method_cfg.get("name", method),
            "description": method_cfg.get("description", ""),
            "region_schedule": method_cfg.get("region_schedule"),
            "paired_region_weighting": training_cfg.get("paired_region_weighting"),
            "base_model": model_cfg.get("base"),
            "controlnet": model_cfg.get("controlnet"),
            "lora_rank": model_cfg.get("lora_rank"),
            "identity_adapter": model_cfg.get("identity_adapter", {}),
            "spatial_conditions": model_cfg.get("spatial_conditions", {}),
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path) if config_path.is_file() else "missing"},
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_path(checkpoint_path), "ready": checkpoint_ready},
        "training": training,
        "data": {
            "train_split": data_cfg.get("train_split"),
            "val_split": data_cfg.get("val_split"),
            "test_split": data_cfg.get("test_split"),
            "cache_dir": data_cfg.get("cache_dir"),
        },
        "eval": {
            "summary_path": str(eval_summary_path),
            "status": eval_summary.get("status"),
            "split": eval_summary.get("split"),
            "metric_split": eval_summary.get("protocol", {}).get("metric_split", eval_summary.get("split")),
            "calibration_split": eval_summary.get("calibration_split"),
            "sample_counts": sample_counts,
            "manifest": eval_summary.get("manifest"),
            "manifest_sha256": eval_summary.get("manifest_sha256"),
            "checkpoint_sha256_from_eval": eval_summary.get("checkpoint_sha256"),
            "set_metrics_path": str(set_metrics_path),
            "set_metrics": set_metrics,
        },
        "metrics": metrics,
    }


def _rank_metrics(experiments: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        ready = []
        pending = []
        for method, experiment in experiments.items():
            value = _finite(experiment["metrics"][metric]["mean"])
            row = {"method": method, "value": value, "status": experiment["status"]}
            if value is None:
                pending.append(row)
            else:
                ready.append(row)
        ready.sort(key=lambda row: row["value"], reverse=direction == "higher")
        rankings[metric] = [{"rank": index + 1, **row, "direction": direction} for index, row in enumerate(ready)]
        rankings[metric].extend({"rank": None, **row, "direction": direction} for row in sorted(pending, key=lambda row: row["method"]))
    return rankings


def _csv_rows(experiments: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in sorted(experiments):
        item = experiments[method]
        sample_counts = item["eval"]["sample_counts"]
        row = {
            "method": method,
            "status": item["status"],
            "experiment_id": item["experiment_id"],
            "description": item["description"],
            "config": item["config"]["path"],
            "config_sha256": item["config"]["sha256"],
            "checkpoint": item["checkpoint"]["path"],
            "checkpoint_sha256": item["checkpoint"]["sha256"],
            "checkpoint_ready": item["checkpoint"]["ready"],
            "training_status": item["training"]["status"],
            "training_step": item["training"]["step"],
            "target_steps": item["training"]["target_steps"],
            "seconds_per_step": item["training"]["seconds_per_step"],
            "peak_gib": item["training"]["peak_gib"],
            "last_loss_total": item["training"]["last_loss_total"],
            "eval_status": item["eval"]["status"],
            "eval_split": item["eval"]["split"],
            "evaluated": sample_counts.get("evaluated"),
            "ok": sample_counts.get("ok"),
            "failed": sample_counts.get("failed"),
            "manifest_sha256": item["eval"].get("manifest_sha256"),
            "pending_reasons": "; ".join(item["pending_reasons"]),
        }
        for metric in METRIC_DIRECTIONS:
            row[metric] = item["metrics"][metric]["mean"]
        rows.append(row)
    return rows


def _fmt(value: Any) -> str:
    number = _finite(value)
    if number is None:
        return "pending"
    return f"{number:.6g}"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Role-Flow A/B/C Experiment Summary",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Manifest: `{payload['manifest']['path']}`",
        f"- Manifest SHA-256: `{payload['manifest']['sha256']}`",
        "",
        "## Metric Directions",
        "",
        "| Metric | Direction |",
        "| --- | --- |",
    ]
    for metric, direction in METRIC_DIRECTIONS.items():
        lines.append(f"| `{metric}` | {direction} |")
    lines.extend(
        [
            "",
            "## Experiment Status",
            "",
            "| Method | Status | Step | Target | sec/step | Peak GiB | Checkpoint | Pending reasons |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for method, item in payload["experiments"].items():
        training = item["training"]
        reasons = "; ".join(item["pending_reasons"]) or "-"
        ready = "ready" if item["checkpoint"]["ready"] else "pending"
        lines.append(
            f"| {method} | {item['status']} | {_fmt(training['step'])} | {_fmt(training['target_steps'])} | "
            f"{_fmt(training['seconds_per_step'])} | {_fmt(training['peak_gib'])} | {ready} | {reasons} |"
        )
    lines.extend(["", "## Metrics", "", "| Method | " + " | ".join(f"`{name}`" for name in METRIC_DIRECTIONS) + " |", "| --- | " + " | ".join("---:" for _ in METRIC_DIRECTIONS) + " |"])
    for method, item in payload["experiments"].items():
        values = " | ".join(_fmt(item["metrics"][name]["mean"]) for name in METRIC_DIRECTIONS)
        lines.append(f"| {method} | {values} |")
    lines.extend(["", "## Per-Metric Ranking", ""])
    for metric, rows in payload["metric_rankings"].items():
        direction = METRIC_DIRECTIONS[metric]
        parts = [
            f"{row['rank']}. {row['method']} ({_fmt(row['value'])})"
            if row["rank"] is not None
            else f"{row['method']} (pending)"
            for row in rows
        ]
        ranking = ", ".join(parts)
        lines.append(f"- `{metric}` ({direction}): {ranking}")
    lines.extend(["", "## Model Structure Differences", ""])
    for method, item in payload["experiments"].items():
        structure = item["structure"]
        lines.append(f"- {method}: {structure.get('description') or 'pending'}")
        if structure.get("paired_region_weighting"):
            lines.append(f"  - paired_region_weighting: `{json.dumps(structure['paired_region_weighting'], sort_keys=True)}`")
        if structure.get("region_schedule"):
            lines.append(f"  - region_schedule: `{json.dumps(structure['region_schedule'], sort_keys=True)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "role_flow_abc_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "role_flow_abc_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_csv_rows(payload["experiments"]))
    _write_markdown(output_dir / "role_flow_abc_summary.md", payload)


def summarize_experiments(
    *,
    configs: dict[str, Path],
    run_dirs: dict[str, Path],
    eval_summaries: dict[str, Path],
    checkpoints: dict[str, Path],
    manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    experiments = {
        method: _summarize_one(method, configs[method], run_dirs[method], eval_summaries[method], checkpoints[method])
        for method in sorted(configs)
    }
    payload = {
        "schema_version": "m2h-role-flow-abc-summary-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric_directions": METRIC_DIRECTIONS,
        "manifest": {
            "path": str(manifest),
            "sha256": sha256_file(manifest) if manifest.is_file() else "missing",
        },
        "experiments": experiments,
        "metric_rankings": _rank_metrics(experiments),
        "outputs": {
            "json": str(output_dir / "role_flow_abc_summary.json"),
            "csv": str(output_dir / "role_flow_abc_summary.csv"),
            "markdown": str(output_dir / "role_flow_abc_summary.md"),
        },
    }
    _write_outputs(payload, output_dir)
    return payload


def _parse_mapping(values: list[str], defaults: dict[str, Path]) -> dict[str, Path]:
    result = dict(defaults)
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected METHOD=PATH, got {value!r}")
        method, path = value.split("=", 1)
        result[method.strip()] = Path(path).expanduser()
    return result


def _default_paths(experiment_root: Path) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    role_root = experiment_root / "role_flow"
    run_dirs = {method: role_root / "runs" / run_id for method, run_id in DEFAULT_RUN_IDS.items()}
    eval_summaries = {method: role_root / "eval" / run_id / "metrics" / "summary.json" for method, run_id in DEFAULT_RUN_IDS.items()}
    checkpoints = {method: run_dirs[method] / "checkpoints" / "final" for method in DEFAULT_RUN_IDS}
    return run_dirs, eval_summaries, checkpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", action="append", default=[], help="Override config path as METHOD=PATH. Can be repeated.")
    parser.add_argument("--run-dir", action="append", default=[], help="Override run directory as METHOD=PATH. Can be repeated.")
    parser.add_argument("--eval-summary", action="append", default=[], help="Override eval summary as METHOD=PATH. Can be repeated.")
    parser.add_argument("--checkpoint", action="append", default=[], help="Override checkpoint path as METHOD=PATH. Can be repeated.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_root = args.experiment_root.expanduser()
    run_dirs, eval_summaries, checkpoints = _default_paths(experiment_root)
    configs = _parse_mapping(args.config, DEFAULT_CONFIGS)
    run_dirs = _parse_mapping(args.run_dir, run_dirs)
    eval_summaries = _parse_mapping(args.eval_summary, eval_summaries)
    checkpoints = _parse_mapping(args.checkpoint, checkpoints)
    methods = sorted(set(configs) | set(run_dirs) | set(eval_summaries) | set(checkpoints))
    for mapping in (configs, run_dirs, eval_summaries, checkpoints):
        for method in methods:
            mapping.setdefault(method, Path("__missing__"))
    output_dir = args.output_dir or experiment_root / "role_flow" / "reports" / "abc_summary"
    manifest = args.manifest or experiment_root / "role_flow" / "eval_manifest.json"
    payload = summarize_experiments(
        configs=configs,
        run_dirs=run_dirs,
        eval_summaries=eval_summaries,
        checkpoints=checkpoints,
        manifest=manifest,
        output_dir=output_dir,
    )
    print(json.dumps({"outputs": payload["outputs"], "statuses": {key: value["status"] for key, value in payload["experiments"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
