#!/usr/bin/env python3
"""Repair non-finite DWPose keypoints in a clean overlay.

The source dataset is immutable.  Repairs are written to a mirror overlay,
then atomically renamed into place.  Missing DWPose coordinates are represented
as (0, 0) with zero confidence so downstream confidence gates ignore them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET = "/data/muxiangyu/datasets/M2HImage/M2H_Final_v2_clean_v1"
DEFAULT_SOURCE = "/data/muxiangyu/datasets/M2HImage/M2H_Final_v2"
DEFAULT_OVERLAY = "asset_overlay_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def repair_npz(source: Path, destination: Path) -> dict[str, Any]:
    fields: dict[str, np.ndarray] = {}
    repaired_fields: list[str] = []
    repaired_values = 0
    repaired_points = 0
    with np.load(source, allow_pickle=False) as archive:
        for name in archive.files:
            array = np.asarray(archive[name]).copy()
            if not np.issubdtype(array.dtype, np.inexact):
                fields[name] = array
                continue
            bad = ~np.isfinite(array)
            if not bad.any():
                fields[name] = array
                continue
            repaired_fields.append(name)
            repaired_values += int(bad.sum())
            if name in {"body", "hands", "face"} and array.ndim == 2 and array.shape[-1] == 2:
                invalid_points = bad.any(axis=1)
                array[invalid_points] = 0.0
                repaired_points += int(invalid_points.sum())
                score_name = {
                    "body": "body_scores",
                    "hands": "hand_scores",
                    "face": "face_scores",
                }[name]
                if score_name in archive.files:
                    scores = np.asarray(archive[score_name]).copy()
                    if scores.ndim == 1 and len(scores) == len(invalid_points):
                        scores[invalid_points] = 0.0
                        fields[score_name] = scores
                fields[name] = array
            else:
                # Non-coordinate metadata or score arrays: NaN/Inf carries no
                # usable value, and zero is the neutral invalid sentinel.
                array[bad] = 0.0
                fields[name] = array
    atomic_save_npz(destination, fields)
    return {
        "repaired_fields": repaired_fields,
        "repaired_values": repaired_values,
        "repaired_points": repaired_points,
        "source_sha256": sha256(source),
        "new_sha256": sha256(destination),
        "source_path": str(source),
        "new_path": str(destination),
    }


def iter_bad_paths(report: Path, source_root: Path) -> list[Path]:
    paths: set[Path] = set()
    with report.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "nan_or_inf":
                continue
            path = Path(row["path"])
            try:
                path.relative_to(source_root)
            except ValueError:
                continue
            paths.add(path)
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE)
    parser.add_argument("--overlay-root", default="")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")

    dataset_root = Path(args.dataset_root).resolve()
    source_root = Path(args.source_root).resolve()
    report_root = dataset_root / "dataset_cleaning_report"
    failure_csv = report_root / "full_readability_failures.csv"
    overlay = Path(args.overlay_root) if args.overlay_root else dataset_root / DEFAULT_OVERLAY
    overlay = overlay.resolve()
    paths = iter_bad_paths(failure_csv, source_root)
    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for source in paths:
        relative = source.relative_to(source_root)
        destination = overlay / relative
        row: dict[str, Any] = {
            "status": "planned" if args.dry_run else "ok",
            "asset_type": "dwpose_keypoints",
            "source_path": str(source),
            "new_path": str(destination),
            "relative_path": relative.as_posix(),
            "created_at": now,
            "repair_method": "invalid_coordinate_to_zero_and_score_to_zero",
            "repaired_fields": "",
            "repaired_values": 0,
            "repaired_points": 0,
            "source_sha256": sha256(source),
            "new_sha256": "",
            "error": "",
        }
        try:
            if not args.dry_run:
                result = repair_npz(source, destination)
                row.update(result)
            else:
                with np.load(source, allow_pickle=False) as archive:
                    row["repaired_fields"] = ",".join(
                        name
                        for name in archive.files
                        if np.issubdtype(np.asarray(archive[name]).dtype, np.inexact)
                        and np.any(~np.isfinite(archive[name]))
                    )
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}:{exc}"
        row["repaired_fields"] = (
            json.dumps(row["repaired_fields"], ensure_ascii=False)
            if isinstance(row["repaired_fields"], list)
            else row["repaired_fields"]
        )
        rows.append(row)

    if not args.dry_run:
        report_root.mkdir(parents=True, exist_ok=True)
        manifest = report_root / "npz_repair_manifest.csv"
        fields = [
            "status", "asset_type", "relative_path", "source_path", "new_path",
            "source_sha256", "new_sha256", "repaired_fields", "repaired_values",
            "repaired_points", "repair_method", "created_at", "error",
        ]
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "schema_version": "m2h_npz_repair_v1",
            "source_root": str(source_root),
            "dataset_root": str(dataset_root),
            "overlay_root": str(overlay),
            "input_failure_rows": len(paths),
            "repaired_assets": sum(row["status"] == "ok" for row in rows),
            "failed_assets": sum(row["status"] == "failed" for row in rows),
            "repaired_values": sum(int(row["repaired_values"]) for row in rows),
            "repaired_points": sum(int(row["repaired_points"]) for row in rows),
            "repair_semantics": "nonfinite coordinate point -> (0,0), matching score -> 0; all other nonfinite float values -> 0",
            "manifest": str(manifest),
        }
        (report_root / "npz_repair_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        summary = {
            "input_failure_rows": len(paths),
            "planned_assets": len(rows),
            "overlay_root": str(overlay),
            "dry_run": True,
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if args.strict and any(row["status"] == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
