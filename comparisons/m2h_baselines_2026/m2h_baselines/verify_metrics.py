from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> str:
    return f"{row['mid']}__id{row['jid']}__seed{int(row['seed'])}"


def verify_metrics_outputs(
    run_dir: Path,
    canonical_dir: Path,
    profile: str,
    *,
    allow_quality_failures: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    quality_warnings: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    def quality_check(condition: bool, message: str) -> None:
        if not condition:
            quality_warnings.append(message)

    expected_keys = {path.stem for path in canonical_dir.glob("*.png")}
    check(bool(expected_keys), f"no canonical PNG files under {canonical_dir}")
    check(len(expected_keys) == 4 if profile == "smoke" else True, f"smoke expected 4 canonical images, got {len(expected_keys)}")

    parsing = read_json(run_dir / "parsing_summary.json")
    check(int(parsing.get("expected", -1)) == len(expected_keys), "parsing expected count mismatch")
    check(int(parsing.get("cached", 0)) + int(parsing.get("computed", 0)) == len(expected_keys), "parsing coverage mismatch")
    check(not parsing.get("failed"), f"parsing failures: {parsing.get('failed')}")
    mask_keys = {path.stem for path in (run_dir / "parsing_masks").glob("*.png")}
    check(mask_keys == expected_keys, "parsing mask keys do not match canonical images")

    summary_rules = {
        "garment_summary.json": ("garment", True),
        "hair_summary.json": ("hair", True),
        "pose_summary.json": ("pose", True),
        "identity_summary.json": ("identity", True),
    }
    summaries: dict[str, dict[str, Any]] = {}
    for filename, (label, require_all) in summary_rules.items():
        summary = read_json(run_dir / filename)
        summaries[label] = summary
        quality_check(summary.get("status") == "ok", f"{label} summary status is {summary.get('status')!r}")
        if require_all:
            quality_check(int(summary.get("count", -1)) == len(expected_keys), f"{label} valid count mismatch")
        quality_check(int(summary.get("failed", 0)) == 0, f"{label} has failed rows")
    quality_check(int(summaries["hair"].get("no_hair", 0)) == 0, "hair contains no_hair rows")
    quality_check(
        float(summaries["identity"].get("face_detection_rate", 0.0)) == 1.0,
        "identity face detection rate is not 1.0",
    )

    csv_files = {
        "garment": run_dir / "garment_per_image.csv",
        "hair": run_dir / "hair_per_image.csv",
        "pose": run_dir / "pose_per_image.csv",
        "identity": run_dir / "identity" / "deltaid_per_image.csv",
    }
    for label, path in csv_files.items():
        rows = read_csv(path)
        check(len(rows) == len(expected_keys), f"{label} CSV row count mismatch")
        check({row_key(row) for row in rows} == expected_keys, f"{label} CSV keys do not match canonical images")
        bad = [row_key(row) for row in rows if row.get("status") != "ok"]
        quality_check(not bad, f"{label} has non-ok rows: {bad}")

    panels = read_json(run_dir / "panels_summary.json")
    contact_sheet = Path(str(panels.get("contact_sheet", "")))
    index_csv = Path(str(panels.get("index_csv", "")))
    check(int(panels.get("case_count", 0)) > 0, "panels contain no cases")
    check(contact_sheet.is_file() and contact_sheet.stat().st_size > 0, "panel contact sheet missing or empty")
    check(index_csv.is_file() and index_csv.stat().st_size > 0, "panel index missing or empty")

    report_bundle = read_json(run_dir / "report_bundle.json")
    report = Path(str(report_bundle.get("report", "")))
    check(report.is_file() and report.stat().st_size > 0, "report is missing or empty")
    if profile == "smoke":
        check(not (run_dir / "distribution_summary.json").exists(), "smoke must skip distribution/FID/KID")

    if quality_warnings and not allow_quality_failures:
        errors.extend(quality_warnings)

    payload = {
        "status": "ok" if not errors else "failed",
        "quality_status": "warning" if quality_warnings else "ok",
        "allow_quality_failures": allow_quality_failures,
        "profile": profile,
        "expected_count": len(expected_keys),
        "canonical_keys": sorted(expected_keys),
        "checks": {
            "parsing": parsing.get("expected"),
            "garment": summaries["garment"].get("count"),
            "hair": summaries["hair"].get("count"),
            "pose": summaries["pose"].get("count"),
            "identity": summaries["identity"].get("count"),
            "panels": panels.get("case_count"),
            "report": str(report),
            "distribution_skipped": profile == "smoke",
        },
        "errors": errors,
        "quality_warnings": quality_warnings,
    }
    output = run_dir / "metrics_verification.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("Metrics-v2 verification failed:\n- " + "\n- ".join(errors))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly verify one baseline Metrics-v2 output directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--canonical-dir", required=True)
    parser.add_argument(
        "--allow-quality-failures",
        action="store_true",
        help="Exit successfully when all artifacts are complete but generated-image quality checks fail.",
    )
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    payload = verify_metrics_outputs(
        Path(args.run_dir),
        Path(args.canonical_dir),
        args.profile,
        allow_quality_failures=args.allow_quality_failures,
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
