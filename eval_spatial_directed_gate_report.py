from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paired_eval_common import (
    assert_fair_pair,
    fmt,
    legacy_garment_bundle,
    metric_bundle,
    paired_wilcoxon,
    read_json,
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
        description="Preregistered new-spatial-system directed identity gate."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--hair-decision",
        type=Path,
        default=DEFAULT_ROOT / "eval/spatial_hair_ab_decision.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "eval/spatial_directed_gate_report.md",
    )
    parser.add_argument(
        "--decision-json",
        type=Path,
        default=DEFAULT_ROOT / "eval/spatial_directed_gate_decision.json",
    )
    return parser.parse_args()


def branch_paths(root: Path, winner: str) -> dict[str, Path]:
    if winner not in {"space", "semantic"}:
        raise ValueError(f"unknown Part B winner: {winner}")
    run_prefix = f"phase1_spatial_c_{{kind}}_{winner}_r16_12400_768x1024"
    metric_prefix = f"spatial_c_{{kind}}_{winner}"
    return {
        "directed_run": root / "phase1" / run_prefix.format(kind="a4prime"),
        "control_run": root / "phase1" / run_prefix.format(kind="cont"),
        "directed_metrics": root / "eval/metrics_v2" / metric_prefix.format(kind="a4prime"),
        "control_metrics": root / "eval/metrics_v2" / metric_prefix.format(kind="cont"),
        "directed_legacy": root / "eval" / f"spatial_c_a4prime_{winner}_legacy_metrics",
        "control_legacy": root / "eval" / f"spatial_c_cont_{winner}_legacy_metrics",
    }


def main() -> None:
    args = parse_args()
    hair_decision = read_json(args.hair_decision)
    winner = str(hair_decision["winner"])
    paths = branch_paths(args.root, winner)
    fairness = assert_fair_pair(paths["directed_run"], paths["control_run"])
    directed = metric_bundle(paths["directed_metrics"])
    control = metric_bundle(paths["control_metrics"])
    new_directed_legacy = legacy_garment_bundle(paths["directed_legacy"])
    new_control_legacy = legacy_garment_bundle(paths["control_legacy"])

    identity = paired_wilcoxon(
        metric_csv(paths["directed_metrics"], "identity/deltaid_per_image.csv"),
        metric_csv(paths["control_metrics"], "identity/deltaid_per_image.csv"),
        "sim_target",
        "greater",
    )
    delta_id = paired_wilcoxon(
        metric_csv(paths["directed_metrics"], "identity/deltaid_per_image.csv"),
        metric_csv(paths["control_metrics"], "identity/deltaid_per_image.csv"),
        "delta_id",
        "greater",
    )
    garment_regression = paired_wilcoxon(
        metric_csv(paths["directed_metrics"], "garment_per_image.csv"),
        metric_csv(paths["control_metrics"], "garment_per_image.csv"),
        "garment_dino_to_mannequin",
        "less",
    )
    head_regression = paired_wilcoxon(
        metric_csv(paths["directed_metrics"], "pose_per_image.csv"),
        metric_csv(paths["control_metrics"], "pose_per_image.csv"),
        "head5_distance_to_mannequin",
        "greater",
    )
    print_regression = paired_wilcoxon(
        metric_csv(paths["directed_metrics"], "garment_per_image.csv"),
        metric_csv(paths["control_metrics"], "garment_per_image.csv"),
        "print_region_hf_lpips",
        "greater",
    )

    old_a4_metrics = args.root / "eval/metrics_v2/a4"
    old_cont_metrics = args.root / "eval/metrics_v2/b2cont"
    old_a4 = metric_bundle(old_a4_metrics)
    old_cont = metric_bundle(old_cont_metrics)
    old_a4_legacy = legacy_garment_bundle(args.root / "eval/a4_metrics")
    old_cont_legacy = legacy_garment_bundle(args.root / "eval/b2cont_metrics")
    old_identity_gain = old_a4["sim_target"] - old_cont["sim_target"]
    old_garment_cost = (
        old_a4_legacy["garment_sim"] - old_cont_legacy["garment_sim"]
    )
    new_identity_gain = directed["sim_target"] - control["sim_target"]
    new_garment_dino_cost = directed["garment_dino"] - control["garment_dino"]
    new_garment_sim_cost = (
        new_directed_legacy["garment_sim"] - new_control_legacy["garment_sim"]
    )

    checks = {
        "identity_gain_at_least_0.03": new_identity_gain >= 0.03,
        "identity_greater_p_below_0.05": identity["p"] < 0.05,
        "garment_dino_not_below_tolerance": (
            directed["garment_dino"] >= control["garment_dino"] - 0.005
        ),
        "head5_not_above_tolerance": directed["head5"] <= control["head5"] + 0.002,
        "face_detection_at_least_99pct": directed["face_detection_rate"] >= 0.99,
        "print_hf_lpips_not_worse": (
            directed["print_hf_lpips"] <= control["print_hf_lpips"]
            and print_regression["p"] >= 0.05
        ),
    }
    passed = all(checks.values())
    verdict = "PASS" if passed else "FAIL"
    conclusion = (
        "identity-directed counterfactual training reproduces on the spatial "
        "system and satisfies all non-regression guards."
        if passed
        else "identity-directed counterfactual training does not clear the "
        "preregistered new-system gate; no post-hoc rescue is admitted."
    )
    decision = {
        "verdict": verdict,
        "winner_hair_branch": winner,
        "checks": checks,
        "fairness": fairness,
        "metrics": {"C-A4prime": directed, "C-cont": control},
        "tests": {
            "sim_target_greater": identity,
            "delta_id_greater": delta_id,
            "garment_dino_regression": garment_regression,
            "head5_regression": head_regression,
            "print_hf_lpips_regression": print_regression,
        },
        "cross_system": {
            "old": {
                "sim_target_gain": old_identity_gain,
                "garment_sim_cost": old_garment_cost,
            },
            "new": {
                "sim_target_gain": new_identity_gain,
                "garment_dino_cost": new_garment_dino_cost,
                "garment_sim_cost": new_garment_sim_cost,
            },
        },
        "paths": {key: str(value) for key, value in paths.items()},
    }
    write_json(args.decision_json, decision)

    lines = [
        "# New Spatial System Directed Identity Gate",
        "",
        (
            f"**Conclusion: {verdict}. {conclusion}**"
        ),
        (
            f"**sim_target gain: {new_identity_gain:+.4f} "
            f"(greater p={identity['p']:.6g}); Garment-DINO cost: "
            f"{new_garment_dino_cost:+.4f}; new-system GarmentSim cost: "
            f"{new_garment_sim_cost:+.4f}.**"
        ),
        (
            f"**Cross-system: old (+{old_identity_gain:.4f} identity, "
            f"{old_garment_cost:+.4f} garment) vs new "
            f"({new_identity_gain:+.4f} identity, "
            f"{new_garment_sim_cost:+.4f} garment).**"
        ),
        "",
        f"Part B winner used as common start: {winner}.",
        "",
        "## C-A4prime vs C-cont",
        "",
        "| metric | C-A4prime | C-cont | directed - control |",
        "|---|---:|---:|---:|",
    ]
    metric_rows = (
        ("sim_target", "sim_target"),
        ("DeltaID", "delta_id"),
        ("Garment-DINO", "garment_dino"),
        ("GarmentSim", None),
        ("Print-HF-LPIPS", "print_hf_lpips"),
        ("Hair-DINO", "hair_dino"),
        ("head5", "head5"),
        ("face detection", "face_detection_rate"),
        ("FID", "fid"),
    )
    for label, key in metric_rows:
        if key is None:
            left = new_directed_legacy["garment_sim"]
            right = new_control_legacy["garment_sim"]
        else:
            left = directed[key]
            right = control[key]
        lines.append(
            f"| {label} | {fmt(left)} | {fmt(right)} | {fmt(left - right)} |"
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
    for name, value in checks.items():
        lines.append(f"| {name} | {'PASS' if value else 'FAIL'} |")
    lines.extend(
        [
            "",
            (
                "Paired held-out sim_target Wilcoxon greater: "
                f"n={identity['n']}, mean delta={identity['mean_delta']:+.4f}, "
                f"p={identity['p']:.6g}, "
                f"rank-biserial={identity['rank_biserial']:+.4f}."
            ),
            (
                "Paired Garment-DINO degradation test: "
                f"mean delta={garment_regression['mean_delta']:+.4f}, "
                f"less-side p={garment_regression['p']:.6g}."
            ),
            (
                "Paired Print-HF-LPIPS degradation test: "
                f"mean delta={print_regression['mean_delta']:+.4f}, "
                f"greater-side p={print_regression['p']:.6g}."
            ),
            "",
            "## Fairness",
            "",
            (
                "Both branches share resume hash, train IDs, sampler state, seed, "
                "global batch, effective LR, rank, and exactly 4000 executed steps."
            ),
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
