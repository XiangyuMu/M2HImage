from __future__ import annotations

import json
from pathlib import Path

from tools.summarize_role_flow_experiments import METRIC_DIRECTIONS, summarize_experiments


def _write_config(path: Path, exp_id: str, method: str, description: str, extra_training: str = "") -> None:
    path.write_text(
        f"""
experiment:
  id: {exp_id}
experiment_method:
  name: {method}
  description: {description}
training:
  total_steps: 4400
  baseline_lr: 5.0e-05
  baseline_global_batch: 16
{extra_training}model:
  base: /models/flux
  controlnet: /models/controlnet
  lora_rank: 16
data:
  train_split: splits/train.txt
  val_split: splits/val.txt
  test_split: splits/test.txt
""".lstrip(),
        encoding="utf-8",
    )


def _write_eval_summary(path: Path, method: str, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "method": method,
                "split": "test",
                "manifest": "/tmp/eval_manifest.json",
                "manifest_sha256": "manifest-hash",
                "checkpoint": "/tmp/checkpoint",
                "checkpoint_sha256": "checkpoint-hash-from-eval",
                "sample_counts": {"evaluated": 10, "ok": 10, "failed": 0, "calibration": 4},
                "metrics": {name: {"mean": value, "median": value, "count": 10, "failed": 0} for name, value in metrics.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_summarize_experiments_marks_missing_eval_pending_and_ranks_by_direction(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    runs = tmp_path / "runs"
    evals = tmp_path / "eval"
    output = tmp_path / "out"
    config_a = configs / "A.yaml"
    config_b = configs / "B.yaml"
    config_c = configs / "C.yaml"
    _write_config(config_a, "A_timestep_routed", "A", "timestep routed adapters")
    _write_config(config_b, "B_garment_weighted", "B", "A plus garment loss", "  paired_region_weighting:\n    enabled: true\n    w_cloth: 2.5\n")
    _write_config(config_c, "C_async_flow", "C", "async flow path")
    manifest = tmp_path / "eval_manifest.json"
    manifest.write_text('{"rows": []}\n', encoding="utf-8")
    for exp_id in ("A_timestep_routed", "B_garment_weighted", "C_async_flow"):
        run = runs / exp_id
        (run / "logs").mkdir(parents=True)
        (run / "training_status.json").write_text(json.dumps({"status": "complete", "step": 4400, "target_steps": 4400}), encoding="utf-8")
        (run / "logs" / "train.jsonl").write_text('{"step": 4400, "loss_total": 0.2, "peak_gib": 31.5}\n', encoding="utf-8")
        (run / "checkpoints" / "final").mkdir(parents=True)
        (run / "checkpoints" / "final" / "READY").write_text("ready\n", encoding="utf-8")
    metric_names = tuple(METRIC_DIRECTIONS)
    _write_eval_summary(evals / "A_timestep_routed" / "metrics" / "summary.json", "A", {name: 0.5 for name in metric_names})
    values_b = {name: 0.6 for name in metric_names}
    values_b["bg_lpips"] = 0.4
    values_b["fid"] = 12.0
    _write_eval_summary(evals / "B_garment_weighted" / "metrics" / "summary.json", "B", values_b)

    payload = summarize_experiments(
        configs={"A": config_a, "B": config_b, "C": config_c},
        run_dirs={"A": runs / "A_timestep_routed", "B": runs / "B_garment_weighted", "C": runs / "C_async_flow"},
        eval_summaries={"A": evals / "A_timestep_routed" / "metrics" / "summary.json", "B": evals / "B_garment_weighted" / "metrics" / "summary.json", "C": evals / "C_async_flow" / "metrics" / "summary.json"},
        checkpoints={"A": runs / "A_timestep_routed" / "checkpoints" / "final", "B": runs / "B_garment_weighted" / "checkpoints" / "final", "C": runs / "C_async_flow" / "checkpoints" / "final"},
        manifest=manifest,
        output_dir=output,
    )

    assert payload["experiments"]["C"]["status"] == "pending"
    assert payload["metric_rankings"]["id_cosine"][0]["method"] == "B"
    assert payload["metric_rankings"]["bg_lpips"][0]["method"] == "B"
    assert payload["metric_rankings"]["fid"][0]["method"] == "A"
    assert payload["experiments"]["A"]["training"]["last_loss_total"] == 0.2
    assert (output / "role_flow_abc_summary.json").is_file()
    assert (output / "role_flow_abc_summary.csv").is_file()
    markdown = (output / "role_flow_abc_summary.md").read_text(encoding="utf-8")
    assert "pending" in markdown
    assert "higher" in markdown
    assert "lower" in markdown


def test_missing_config_is_reported_without_fake_values(tmp_path: Path) -> None:
    output = tmp_path / "out"
    payload = summarize_experiments(
        configs={"A": tmp_path / "missing.yaml"},
        run_dirs={"A": tmp_path / "missing_run"},
        eval_summaries={"A": tmp_path / "missing_summary.json"},
        checkpoints={"A": tmp_path / "missing_checkpoint"},
        manifest=tmp_path / "missing_manifest.json",
        output_dir=output,
    )

    experiment = payload["experiments"]["A"]
    assert experiment["status"] == "pending"
    assert experiment["metrics"]["id_cosine"]["mean"] is None
    assert "missing config" in experiment["pending_reasons"]


def test_v2_set_metrics_supplies_fid_and_complete_status(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config, "A_timestep_routed", "A", "v2")
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "training_status.json").write_text(json.dumps({"status": "complete", "step": 4400}), encoding="utf-8")
    (run / "checkpoints" / "final").mkdir(parents=True)
    (run / "checkpoints" / "final" / "READY").write_text("ready\n", encoding="utf-8")
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    summary = {
        "status": "complete",
        "protocol": {"metric_split": "test"},
        "sample_counts": {"generated": 580, "evaluated": 400, "evaluated_test": 400, "val": 180, "test": 400},
        "metrics": {"id_cosine": {"mean": 0.8, "median": 0.8, "count": 400, "failed": 0}},
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (metrics_dir / "set_metrics.json").write_text(
        json.dumps({"scope": "test", "fid": {"value": 11.25, "generated_count": 400}}),
        encoding="utf-8",
    )
    payload = summarize_experiments(
        configs={"A": config},
        run_dirs={"A": run},
        eval_summaries={"A": metrics_dir / "summary.json"},
        checkpoints={"A": run / "checkpoints" / "final"},
        manifest=tmp_path / "missing_manifest.json",
        output_dir=tmp_path / "out",
    )
    experiment = payload["experiments"]["A"]
    assert experiment["status"] == "complete"
    assert experiment["metrics"]["fid"]["mean"] == 11.25
    assert experiment["eval"]["metric_split"] == "test"
