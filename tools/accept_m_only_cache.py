#!/usr/bin/env python3
"""Deterministic acceptance checks for the formal mannequin-only cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_EXPERIMENT = "/data/muxiangyu/experiments/M2H_Final_v2_clean_v1"
DEFAULT_REPORT = "/data/muxiangyu/datasets/M2HImage/M2H_Final_v2_clean_v1/dataset_cleaning_report"
SPLITS = ("train", "val", "test")
REQUIRED = (
    "mannequin_garment_mask",
    "mannequin_garment_image",
    "mannequin_parsing",
    "mannequin_pose_image",
    "mannequin_dwpose_body",
    "mannequin_dwpose_body_scores",
    "mannequin_dwpose_hands",
    "mannequin_dwpose_hand_scores",
    "mannequin_dwpose_face",
    "mannequin_dwpose_face_scores",
    "mannequin_dwpose_num_people",
    "mannequin_dwpose_selected_person_index",
    "mannequin_dwpose_height",
    "mannequin_dwpose_width",
    "mannequin_dwpose_score_threshold",
    "mannequin_region_cloth_safe_z",
    "mannequin_region_body_bg_z",
    "mannequin_region_face_z",
    "mannequin_region_hair_z",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_from_manifest(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [str(row["id"]) for row in csv.DictReader(handle) if row.get("id")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = Path(args.experiment_root) / "cache_m_only"
    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    source_roles = Counter()
    source_paths = Counter()
    field_counts = Counter()
    expected_shapes = {
        "mannequin_garment_mask": (1024, 768),
        "mannequin_garment_image": (1024, 768, 3),
        "mannequin_parsing": (1024, 768),
        "mannequin_pose_image": (1024, 768, 3),
        "mannequin_dwpose_body": (18, 2),
        "mannequin_dwpose_body_scores": (18,),
        "mannequin_dwpose_hands": (42, 2),
        "mannequin_dwpose_hand_scores": (42,),
        "mannequin_dwpose_face": (68, 2),
        "mannequin_dwpose_face_scores": (68,),
        "mannequin_region_cloth_safe_z": (3072,),
        "mannequin_region_body_bg_z": (3072,),
        "mannequin_region_face_z": (3072,),
        "mannequin_region_hair_z": (3072,),
    }
    for split in SPLITS:
        manifest_path = root / split / "manifest.csv"
        expected_ids = sorted(ids_from_manifest(manifest_path)) if manifest_path.is_file() else []
        files = sorted((root / split / "samples").glob("*.npz"))
        actual_ids = sorted(path.stem for path in files)
        if expected_ids != actual_ids:
            errors.append({"split": split, "error": "manifest_cache_id_mismatch",
                           "manifest_count": len(expected_ids), "cache_count": len(actual_ids)})
        for path in files:
            item = {"split": split, "id": path.stem, "path": str(path), "status": "ok"}
            try:
                with np.load(path, allow_pickle=False) as payload:
                    names = tuple(payload.files)
                    missing = [name for name in REQUIRED if name not in names]
                    extra = [name for name in names if name not in REQUIRED]
                    if missing or extra:
                        raise ValueError(f"fields:missing={missing},extra={extra}")
                    for name in names:
                        array = np.asarray(payload[name])
                        field_counts[name] += 1
                        if array.size == 0:
                            raise ValueError(f"empty:{name}")
                        if array.dtype.kind in "fc" and not np.isfinite(array).all():
                            raise ValueError(f"nan_or_inf:{name}")
                        shape = expected_shapes.get(name)
                        if shape and tuple(array.shape) != shape:
                            raise ValueError(f"shape:{name}:{tuple(array.shape)}!={shape}")
                        if name.startswith("mannequin_region_") and (
                            float(array.min()) < -1e-4 or float(array.max()) > 1.0001
                        ):
                            raise ValueError(f"region_range:{name}")
                    item["field_count"] = len(names)
                    item["sha256"] = sha256(path)
            except Exception as exc:  # noqa: BLE001
                item["status"] = "failed"
                item["error"] = str(exc)
                errors.append(item)
            rows.append(item)

    provenance = report / "cache_field_provenance.jsonl"
    provenance_errors = 0
    if provenance.is_file():
        with provenance.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                source_roles[record.get("source_role", "")] += 1
                source_paths[record.get("source_path", "")] += 1
                source = str(record.get("source_path", ""))
                path_parts = Path(source).parts
                if record.get("source_role") != "mannequin" or "human" in path_parts:
                    provenance_errors += 1
    else:
        provenance_errors = 1
    result = {
        "version": "m-only-cache-acceptance-v1",
        "cache_root": str(root),
        "formal_splits": list(SPLITS),
        "samples_checked": len(rows),
        "sample_failures": len(errors),
        "provenance_records": sum(source_roles.values()),
        "provenance_errors": provenance_errors,
        "required_field_count": len(REQUIRED),
        "field_counts": dict(sorted(field_counts.items())),
        "source_role_counts": dict(source_roles),
        "unique_source_paths": len(source_paths),
        "status": "passed" if not errors and provenance_errors == 0 else "failed",
        "errors": errors[:1000],
    }
    (report / "m_only_cache_acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (report / "m_only_cache_acceptance.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["split", "id", "path", "status", "field_count", "sha256", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: result[key] for key in (
        "samples_checked", "sample_failures", "provenance_records",
        "provenance_errors", "status",
    )}, ensure_ascii=False))
    return 1 if args.strict and result["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
