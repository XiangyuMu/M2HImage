from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paired_eval_common import (
    assert_fair_pair,
    attribution_rate,
    fmt,
    gate_shift,
    metric_bundle,
    paired_mcnemar_less,
    paired_wilcoxon,
    train_log_path,
    watcher_bundle,
)


DEFAULT_ROOT = Path("/data/muxiangyu/datasets/M2HImage/M2H_Final_v2")


def metric_csv(metrics_dir: Path, relative: str) -> Path:
    path = metrics_dir / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preregistered spatial-hair A/B gate report."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--space-run-dir",
        type=Path,
        default=DEFAULT_ROOT / "phase1/phase1_spatial_hair_ab_space_r16_8400_768x1024",
    )
    parser.add_argument(
        "--semantic-run-dir",
        type=Path,
        default=DEFAULT_ROOT / "phase1/phase1_spatial_hair_ab_semantic_r16_8400_768x1024",
    )
    parser.add_argument(
        "--space-metrics",
        type=Path,
        default=DEFAULT_ROOT / "eval/metrics_v2/spatial_hair_ab_space",
    )
    parser.add_argument(
        "--semantic-metrics",
        type=Path,
        default=DEFAULT_ROOT / "eval/metrics_v2/spatial_hair_ab_semantic",
    )
    parser.add_argument(
        "--current-metrics",
        type=Path,
        default=DEFAULT_ROOT / "eval/metrics_v2/spatial_quality_repair",
    )
    parser.add_argument(
        "--space-attribution",
        type=Path,
        default=Path("artifacts/id_failure_attrib/hair_ab_space/per_image.csv"),
    )
    parser.add_argument(
        "--semantic-attribution",
        type=Path,
        default=Path("artifacts/id_failure_attrib/hair_ab_semantic/per_image.csv"),
    )
    parser.add_argument(
        "--current-attribution",
        type=Path,
        default=Path("artifacts/id_failure_attrib/per_image.csv"),
    )
    parser.add_argument(
        "--space-watcher",
        type=Path,
        default=Path("artifacts/hair_ab_fixed_set/space/step8400.json"),
    )
    parser.add_argument(
        "--semantic-watcher",
        type=Path,
        default=Path("artifacts/hair_ab_fixed_set/semantic/step8400.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "eval/spatial_hair_ab_report.md",
    )
    parser.add_argument(
        "--decision-json",
        type=Path,
        default=DEFAULT_ROOT / "eval/spatial_hair_ab_decision.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fairness = assert_fair_pair(args.space_run_dir, args.semantic_run_dir)
    current = metric_bundle(args.current_metrics)
    space = metric_bundle(args.space_metrics)
    semantic = metric_bundle(args.semantic_metrics)

    identity = paired_wilcoxon(
        metric_csv(args.semantic_metrics, "identity/deltaid_per_image.csv"),
        metric_csv(args.space_metrics, "identity/deltaid_per_image.csv"),
        "sim_target",
        "greater",
    )
    hair = paired_wilcoxon(
        metric_csv(args.semantic_metrics, "hair_per_image.csv"),
        metric_csv(args.space_metrics, "hair_per_image.csv"),
        "hair_dino_to_reference",
        "greater",
    )
    garment_regression = paired_wilcoxon(
        metric_csv(args.semantic_metrics, "garment_per_image.csv"),
        metric_csv(args.space_metrics, "garment_per_image.csv"),
        "garment_dino_to_mannequin",
        "less",
    )
    head_regression = paired_wilcoxon(
        metric_csv(args.semantic_metrics, "pose_per_image.csv"),
        metric_csv(args.space_metrics, "pose_per_image.csv"),
        "head5_distance_to_mannequin",
        "greater",
    )
    occlusion = paired_mcnemar_less(
        args.space_attribution,
        args.semantic_attribution,
        "face_occluded_by_hair",
    )
    rates = {
        "current": attribution_rate(
            args.current_attribution, "face_occluded_by_hair"
        ),
        "space": attribution_rate(
            args.space_attribution, "face_occluded_by_hair"
        ),
        "semantic": attribution_rate(
            args.semantic_attribution, "face_occluded_by_hair"
        ),
    }
    gate = gate_shift(train_log_path(args.semantic_run_dir))
    watcher = {
        "space": watcher_bundle(args.space_watcher),
        "semantic": watcher_bundle(args.semantic_watcher),
    }

    checks = {
        "hair_dino_retained": semantic["hair_dino"] >= space["hair_dino"] - 0.02,
        "face_occlusion_significantly_lower": (
            rates["semantic"] < rates["space"] and occlusion["p"] < 0.05
        ),
        "sim_target_gain": semantic["sim_target"] >= space["sim_target"] + 0.02,
        "garment_dino_not_regressed": (
            semantic["garment_dino"] >= space["garment_dino"] - 0.005
        ),
        "head5_not_regressed": semantic["head5"] <= space["head5"] + 0.002,
    }
    semantic_pass = all(checks.values())
    semantic_failed = (
        gate["shift"] < 0.01 and semantic["hair_dino"] < 0.60
    )
    if semantic_pass:
        verdict = "B-SEMANTIC-PASS"
        winner = "semantic"
        conclusion = (
            "DINO dense hair tokens preserve hair while reducing face spill and "
            "recovering identity; use the semantic branch for Part C."
        )
    elif semantic_failed:
        verdict = "SEMANTIC-HAIR-FAILED"
        winner = "space"
        conclusion = (
            "The token route failed its preregistered gate; retain spatial hair "
            "and record low-frequency blurred spatial reference as fallback only."
        )
    else:
        verdict = "B-SEMANTIC-NO-PASS"
        winner = "space"
        conclusion = (
            "The semantic branch did not satisfy every preregistered condition; "
            "the spatial control wins without an extra rescue run."
        )

    winner_run = args.semantic_run_dir if winner == "semantic" else args.space_run_dir
    winner_metrics = args.semantic_metrics if winner == "semantic" else args.space_metrics
    decision = {
        "verdict": verdict,
        "winner": winner,
        "winner_run_dir": str(winner_run),
        "winner_checkpoint": str(winner_run / "checkpoints/final"),
        "winner_metrics": str(winner_metrics),
        "winner_config": (
            "configs/spatial_hair_ab_semantic.yaml"
            if winner == "semantic"
            else "configs/spatial_hair_ab_space.yaml"
        ),
        "fairness": fairness,
        "checks": checks,
        "metrics": {
            "current": current,
            "space": space,
            "semantic": semantic,
        },
        "tests": {
            "identity": identity,
            "hair": hair,
            "garment_regression": garment_regression,
            "head_regression": head_regression,
            "face_occlusion": occlusion,
        },
        "face_occlusion_rates": rates,
        "semantic_hair_gate": gate,
        "watcher": watcher,
    }
    write_json(args.decision_json, decision)

    columns = (("current", current), ("B-space", space), ("B-semantic", semantic))
    lines = [
        "# Spatial Hair Conditioning A/B Gate",
        "",
        f"**Conclusion: {verdict}. {conclusion}**",
        (
            f"**Hair-DINO: space {space['hair_dino']:.4f}, semantic "
            f"{semantic['hair_dino']:.4f}; sim_target: space "
            f"{space['sim_target']:.4f}, semantic {semantic['sim_target']:.4f}; "
            f"face_occluded: space {rates['space']:.2%}, semantic "
            f"{rates['semantic']:.2%}.**"
        ),
        "",
        "## Full-system metrics",
        "",
        "| metric | current checkpoint | B-space | B-semantic |",
        "|---|---:|---:|---:|",
    ]
    metric_rows = (
        ("Hair-DINO (higher)", "hair_dino"),
        ("sim_target (higher)", "sim_target"),
        ("Garment-DINO (higher)", "garment_dino"),
        ("Print-HF-LPIPS (lower)", "print_hf_lpips"),
        ("head5 distance (lower)", "head5"),
        ("body distance (lower)", "body"),
        ("face detection rate", "face_detection_rate"),
        ("FID (lower)", "fid"),
    )
    for label, key in metric_rows:
        lines.append(
            f"| {label} | {fmt(columns[0][1][key])} | "
            f"{fmt(columns[1][1][key])} | {fmt(columns[2][1][key])} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered checks",
            "",
            "| check | result |",
            "|---|---|",
        ]
    )
    for name, passed in checks.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            (
                "Paired sim_target Wilcoxon greater: "
                f"delta={identity['mean_delta']:+.4f}, p={identity['p']:.6g}, "
                f"rank-biserial={identity['rank_biserial']:+.4f}, n={identity['n']}."
            ),
            (
                "Paired face-occlusion McNemar/binomial: "
                f"improved={occlusion['improved']}, worsened={occlusion['worsened']}, "
                f"p={occlusion['p']:.6g}."
            ),
            (
                "Semantic hair_gate movement over logged training: "
                f"{gate['initial']:.6f} -> {gate['final']:.6f}, "
                f"absolute shift={gate['shift']:.6f}."
            ),
            "",
            "## Fairness",
            "",
            (
                "The branches share resume_trainable_hash, train_ids_hash, sampler "
                "state, seed, global batch, effective LR, rank, resume step, target "
                "step, and executed-step count."
            ),
            "",
            "Fairness signature:",
            "",
            json.dumps(fairness, indent=2, ensure_ascii=False),
            "",
            "Machine-readable decision:",
            str(args.decision_json),
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
