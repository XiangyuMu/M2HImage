from __future__ import annotations

from pathlib import Path
from typing import Any

from metrics.common import fmt
from metrics_v2.common import paired_test, provenance, read_csv, read_json, resolve_path, write_json


def _summary(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {"status": "not_run", "reason": f"missing {path}"}


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def write_run_report(
    cfg: dict[str, Any], run_name: str, out_dir: str | Path, legacy_metrics_dir: str | Path
) -> Path:
    out_dir = Path(out_dir)
    garment = _summary(out_dir / "garment_summary.json")
    hair = _summary(out_dir / "hair_summary.json")
    pose = _summary(out_dir / "pose_summary.json")
    identity = _summary(out_dir / "identity_summary.json")
    distribution = _summary(out_dir / "distribution_summary.json")
    panels = _summary(out_dir / "panels_summary.json")
    legacy = _summary(Path(legacy_metrics_dir) / "garment_summary.json")
    prov = provenance(cfg)
    top_line = (
        f"Current {run_name} four quality gates: Garment-DINO-to-mannequin="
        f"{fmt(_nested(garment, 'garment_dino', 'mean'))}; Garment-HF-LPIPS="
        f"{fmt(_nested(garment, 'garment_hf_lpips', 'mean'))}; Hair-DINO-to-reference="
        f"{fmt(_nested(hair, 'hair_dino', 'mean'))}; head-5-point-distance="
        f"{fmt(_nested(pose, 'head5_distance', 'mean'))}."
    )
    lines = [
        f"# M2H Metrics v2: {run_name}",
        "",
        f"**{top_line}**",
        "",
        "## System Layer",
        "",
        "| Metric | Measured against | Mean | Median | P10 / auxiliary | Valid |",
        "|---|---|---:|---:|---:|---:|",
        f"| Garment-DINO (higher) | source mannequin garment | {fmt(_nested(garment, 'garment_dino', 'mean'))} | {fmt(_nested(garment, 'garment_dino', 'median'))} | {fmt(_nested(garment, 'garment_dino', 'p10'))} | {_nested(garment, 'garment_dino', 'count') or 0} |",
        f"| Garment-LPIPS (lower) | source mannequin garment bbox | {fmt(_nested(garment, 'garment_lpips', 'mean'))} | {fmt(_nested(garment, 'garment_lpips', 'median'))} | {fmt(_nested(garment, 'garment_lpips', 'p10'))} | {_nested(garment, 'garment_lpips', 'count') or 0} |",
        f"| Garment-HF-LPIPS (lower) | source mannequin high-pass garment bbox | {fmt(_nested(garment, 'garment_hf_lpips', 'mean'))} | {fmt(_nested(garment, 'garment_hf_lpips', 'median'))} | {fmt(_nested(garment, 'garment_hf_lpips', 'p10'))} | {_nested(garment, 'garment_hf_lpips', 'count') or 0} |",
        f"| Garment-Gradient-Sim (higher) | source mannequin Sobel garment map | {fmt(_nested(garment, 'garment_gradient_sim', 'mean'))} | {fmt(_nested(garment, 'garment_gradient_sim', 'median'))} | {fmt(_nested(garment, 'garment_gradient_sim', 'p10'))} | {_nested(garment, 'garment_gradient_sim', 'count') or 0} |",
        f"| Print-region HF-LPIPS (lower) | source high-gradient garment candidate | {fmt(_nested(garment, 'print_region_hf_lpips', 'mean'))} | {fmt(_nested(garment, 'print_region_hf_lpips', 'median'))} | {fmt(_nested(garment, 'print_region_hf_lpips', 'p10'))} | {_nested(garment, 'print_region_hf_lpips', 'count') or 0} |",
        f"| DWPose body distance (lower) | mannequin body keypoints | {fmt(_nested(pose, 'body_distance', 'mean'))} | {fmt(_nested(pose, 'body_distance', 'median'))} | {fmt(_nested(pose, 'body_distance', 'p10'))} | {_nested(pose, 'body_distance', 'count') or 0} |",
        f"| DWPose head-5 distance (lower) | mannequin nose/eyes/ears | {fmt(_nested(pose, 'head5_distance', 'mean'))} | {fmt(_nested(pose, 'head5_distance', 'median'))} | {fmt(_nested(pose, 'head5_distance', 'p10'))} | {_nested(pose, 'head5_distance', 'count') or 0} |",
        f"| held-out sim_target (higher) | reference person jid | {fmt(identity.get('sim_target_mean'))} | n/a | DeltaID {fmt(identity.get('mean'))} | {identity.get('count', 0)} |",
        f"| held-out sim_source | original person mid | {fmt(identity.get('sim_source_mean'))} | n/a | face det {fmt(identity.get('face_detection_rate'))} | {identity.get('count', 0)} |",
        f"| Hair-DINO (higher) | reference person hair | {fmt(_nested(hair, 'hair_dino', 'mean'))} | {fmt(_nested(hair, 'hair_dino', 'median'))} | {fmt(_nested(hair, 'hair_dino', 'p10'))} | {_nested(hair, 'hair_dino', 'count') or 0} |",
        f"| Hair LAB distance (lower) | reference person hair color | {fmt(_nested(hair, 'hair_lab_distance', 'mean'))} | {fmt(_nested(hair, 'hair_lab_distance', 'median'))} | no-hair {hair.get('no_hair', 0)} | {_nested(hair, 'hair_lab_distance', 'count') or 0} |",
        f"| FID (lower) | human test split | {fmt(distribution.get('fid'))} | n/a | KID {fmt(distribution.get('kid_mean'), 6)} | run-level |",
        "",
        f"Generated garment mask fallback rate: {fmt(garment.get('fallback_mask_rate'))} ({garment.get('fallback_mask_count', 0)} images).",
        f"Face detector mean confidence: {fmt(identity.get('det_conf_mean'))}; detection rate: {fmt(identity.get('face_detection_rate'))}.",
        "",
        "## Provenance",
        "",
        f"- FASHN SegFormer weights: `{prov['fashn_hash']}`; labels: `{prov['fashn_labels_hash']}`",
        f"- DINOv2 weights: `{prov['dino_hash']}`",
        f"- held-out AdaFace IR-101 weights: `{prov['adaface_hash']}`",
        f"- DWPose detector / pose weights: `{prov['dwpose_detector_hash']}` / `{prov['dwpose_pose_hash']}`",
        "",
        "## Mechanism Layer",
        "",
        "GarmentSim below is the frozen old-runner cross-identity consistency metric. It is reported for context and is not a system-level gate.",
        "",
        f"- Cross-identity GarmentSim: {fmt(legacy.get('cross_identity_group_mean_mean'))}",
        "",
        "## Qualitative Masks",
        "",
        "Green is garment; blue is hair. These overlays are the required trust check for the two parsing-dependent metrics.",
        "",
        f"![mask overlays]({Path(panels.get('contact_sheet', 'qualitative_mask_overlays.png')).name})",
        "",
        f"Panel index: `{panels.get('index_csv', out_dir / 'panel_index.csv')}`",
        "",
        "## Print / Logo Audit",
        "",
        "The source and generated crops below are selected mechanically from dense high-gradient connected components inside the source garment mask.",
        "",
        f"![print and logo crops]({Path(garment.get('print_zoom_grid', out_dir / 'print_zoom_grid.png')).name})",
        "",
        "## Protocol Notes",
        "",
        "- Generated garment masks use FASHN parsing; failures fall back to the source mannequin SAM cloth mask and are marked per image.",
        "- Garment-HF-LPIPS applies `image - GaussianBlur(sigma=2)` after the masked garment bbox crop; it is the high-frequency gate missing from the original suite.",
        "- Garment-Gradient-Sim compares full-resolution Sobel magnitude maps under each side's garment mask.",
        "- Hair regions below 1% of frame area are marked `no_hair` and excluded rather than imputed.",
        "- DWPose distances require confidence >=0.3 on both generated and mannequin points and are divided by image diagonal.",
        "- RetinaFace is shared only for detection/alignment; held-out AdaFace IR-101 performs recognition.",
        "- FID/KID compares this generated run with the frozen human test split.",
    ]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_dir / "report_bundle.json", {"run": run_name, "top_line": top_line, "report": str(report_path)})
    return report_path


def _identity_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(out_dir / "identity/deltaid_per_image.csv")
    for row in rows:
        row["face_detected"] = 1.0 if row.get("status") == "ok" else 0.0
    return rows


def write_compare_report(
    cfg: dict[str, Any],
    a4_dir: str | Path,
    b2_dir: str | Path,
    output_path: str | Path,
    write_baseline: bool = True,
    left_name: str = "a4",
    right_name: str = "b2cont",
) -> Path:
    a4_dir, b2_dir = Path(a4_dir), Path(b2_dir)
    datasets = {
        "garment": (read_csv(a4_dir / "garment_per_image.csv"), read_csv(b2_dir / "garment_per_image.csv")),
        "pose": (read_csv(a4_dir / "pose_per_image.csv"), read_csv(b2_dir / "pose_per_image.csv")),
        "hair": (read_csv(a4_dir / "hair_per_image.csv"), read_csv(b2_dir / "hair_per_image.csv")),
        "identity": (_identity_rows(a4_dir), _identity_rows(b2_dir)),
    }
    definitions = [
        ("Garment-DINO to mannequin (higher)", "garment", "garment_dino_to_mannequin"),
        ("Garment-LPIPS to mannequin (lower)", "garment", "garment_lpips_to_mannequin"),
        ("Garment-HF-LPIPS to mannequin (lower)", "garment", "garment_hf_lpips_to_mannequin"),
        ("Garment-Gradient-Sim to mannequin (higher)", "garment", "garment_gradient_sim_to_mannequin"),
        ("Print-region HF-LPIPS to mannequin (lower)", "garment", "print_region_hf_lpips"),
        ("DWPose body distance to mannequin (lower)", "pose", "body_distance_to_mannequin"),
        ("DWPose head-5 distance to mannequin (lower)", "pose", "head5_distance_to_mannequin"),
        ("held-out sim_target to reference (higher)", "identity", "sim_target"),
        ("held-out sim_source to original", "identity", "sim_source"),
        ("held-out DeltaID (higher)", "identity", "delta_id"),
        ("Hair-DINO to reference (higher)", "hair", "hair_dino_to_reference"),
        ("Hair LAB distance to reference (lower)", "hair", "hair_lab_distance"),
        ("Face detection rate (higher)", "identity", "face_detected"),
        ("Face detection confidence (higher)", "identity", "det_conf"),
    ]
    tests: dict[str, Any] = {}
    table_rows = []
    for label, dataset, field in definitions:
        result = paired_test(*datasets[dataset], field)
        tests[field] = result
        table_rows.append(
            f"| {label} | {result['count']} | {fmt(result['a_mean'])} | {fmt(result['b_mean'])} | "
            f"{fmt(result['mean_diff'])} | {fmt(result['p_two_sided'], 6)} |"
        )
    a4_dist = _summary(a4_dir / "distribution_summary.json")
    b2_dist = _summary(b2_dir / "distribution_summary.json")
    root = Path(cfg["data"]["root"])
    left_cfg = cfg["metrics_v2"]["runs"][left_name]
    right_cfg = cfg["metrics_v2"]["runs"][right_name]
    left_label = str(left_cfg.get("label", left_name))
    right_label = str(right_cfg.get("label", right_name))
    a4_legacy = _summary(resolve_path(root, left_cfg["legacy_metrics_dir"]) / "garment_summary.json")
    b2_legacy = _summary(resolve_path(root, right_cfg["legacy_metrics_dir"]) / "garment_summary.json")
    garment = _summary(a4_dir / "garment_summary.json")
    hair = _summary(a4_dir / "hair_summary.json")
    pose = _summary(a4_dir / "pose_summary.json")
    top_line = (
        f"Current {left_label} four quality gates: Garment-DINO-to-mannequin={fmt(_nested(garment, 'garment_dino', 'mean'))}; "
        f"Garment-HF-LPIPS={fmt(_nested(garment, 'garment_hf_lpips', 'mean'))}; "
        f"Hair-DINO-to-reference={fmt(_nested(hair, 'hair_dino', 'mean'))}; "
        f"head-5-point-distance={fmt(_nested(pose, 'head5_distance', 'mean'))}."
    )
    def delta(a: Any, b: Any) -> float | None:
        return float(a) - float(b) if a is not None and b is not None else None

    lines = [
        f"# M2H Metrics v2: {left_label} vs {right_label}",
        "",
        f"**{top_line}**",
        "",
        f"All per-image tests are paired by `(mid, jid, seed)` and use two-sided Wilcoxon signed-rank. Diff is {left_label} minus {right_label}.",
        "",
        "## System Layer Paired Comparison",
        "",
        f"| Metric | Paired N | {left_label} | {right_label} | {left_label} - {right_label} | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        "FID/KID are set-level metrics and therefore have no valid per-image Wilcoxon test.",
        "",
        f"| Distribution metric | {left_label} | {right_label} | {left_label} - {right_label} |",
        "|---|---:|---:|---:|",
        f"| FID (lower) | {fmt(a4_dist.get('fid'))} | {fmt(b2_dist.get('fid'))} | {fmt(delta(a4_dist.get('fid'), b2_dist.get('fid')))} |",
        f"| KID mean (lower) | {fmt(a4_dist.get('kid_mean'), 6)} | {fmt(b2_dist.get('kid_mean'), 6)} | {fmt(delta(a4_dist.get('kid_mean'), b2_dist.get('kid_mean')), 6)} |",
        "",
        "## Mechanism Layer Reference",
        "",
        f"| Metric | {left_label} | {right_label} | {left_label} - {right_label} |",
        "|---|---:|---:|---:|",
        f"| Frozen old-runner GarmentSim | {fmt(a4_legacy.get('cross_identity_group_mean_mean'))} | {fmt(b2_legacy.get('cross_identity_group_mean_mean'))} | {fmt(delta(a4_legacy.get('cross_identity_group_mean_mean'), b2_legacy.get('cross_identity_group_mean_mean')))} |",
        "",
        "This mechanism-layer row is quoted from existing outputs and is not part of the new system gate.",
        "",
        "## Qualitative Mask Audit",
        "",
        f"- {left_label}: [overlay sheet]({a4_dir / 'qualitative_mask_overlays.png'})",
        f"- {right_label}: [overlay sheet]({b2_dir / 'qualitative_mask_overlays.png'})",
        "",
        "Review these overlays before interpreting Garment-DINO or Hair-DINO; mask quality is part of metric validity.",
        "",
        "## Print / Logo Readability Audit",
        "",
        f"- {left_label}: [print zoom grid]({a4_dir / 'print_zoom_grid.png'})",
        f"- {right_label}: [print zoom grid]({b2_dir / 'print_zoom_grid.png'})",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output_path.with_suffix(".json"), {"top_line": top_line, "paired_tests": tests})
    if write_baseline:
        docs_path = Path(__file__).resolve().parents[1] / "docs/results/metrics_v2_baseline.md"
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(
            "# M2H Metrics v2 Repair-before Baseline\n\n"
            f"**{top_line}**\n\n"
            "This baseline evaluates frozen A4 and B2-cont generations before system-level input fixes. "
            f"The full paired report is `{output_path}`.\n",
            encoding="utf-8",
        )
    return output_path
