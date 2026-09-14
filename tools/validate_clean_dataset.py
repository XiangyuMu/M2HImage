#!/usr/bin/env python3
"""Validate a cleaned M2H dataset without modifying its inputs.

The validator is intentionally manifest driven.  The clean dataset currently
stores references to the immutable source dataset, while quarantine entries
may be hard links below ``quarantine``.  Both forms are handled here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import imghdr
import json
import math
import os
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "/data/muxiangyu/datasets/M2HImage/M2H_Final_v2_clean_v1"
DEFAULT_EXPERIMENT = "/data/muxiangyu/experiments/M2H_Final_v2_clean_v1"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
NPZ_EXTENSIONS = {".npz"}
DERIVED_DIRS = (
    "derived",
    "dwpose",
    "human_parsing",
    "parsing",
    "clothes_bySAM",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted({line.strip() for line in path.read_text().splitlines() if line.strip()})


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = csv.DictReader(fh)
        if not rows.fieldnames or "id" not in rows.fieldnames:
            raise ValueError(f"manifest has no id column: {path}")
        out: dict[str, dict[str, str]] = {}
        for row in rows:
            sid = str(row.get("id", "")).strip()
            if sid:
                out[sid] = {str(k): str(v or "") for k, v in row.items()}
    return out


def resolve_source_path(raw: str, source_root: Path, clean_root: Path, sid: str) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    candidates = [p]
    # A manifest can have been generated on a different host.
    marker = "/M2H_Final_v2/"
    if marker in raw:
        candidates.append(source_root / raw.split(marker, 1)[1])
    candidates.append(clean_root / raw.lstrip("/"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return p


def find_by_id(root: Path, sid: str, suffixes: Iterable[str] | None = None) -> list[Path]:
    """Find exact stem matches in a bounded dataset tree."""
    suffixes = set(suffixes or ())
    found: list[Path] = []
    if not root.exists():
        return found
    # rglob is deterministic after sorting and avoids shell-dependent globbing.
    for path in root.rglob(f"{sid}.*"):
        if path.is_file() and (not suffixes or path.suffix.lower() in suffixes):
            found.append(path)
    return sorted(found)


def quarantine_candidates(clean_root: Path, sid: str) -> list[Path]:
    found: list[Path] = []
    for qroot in (clean_root / "quarantine",):
        qdir = qroot / "exact_cross_split" / sid
        if qdir.exists():
            found.extend(p for p in qdir.rglob("*") if p.is_file())
        qdir = qroot / "high_risk_dhash" / sid
        if qdir.exists():
            found.extend(p for p in qdir.rglob("*") if p.is_file())
    return sorted(set(found))


def image_info(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path),
        "kind": "image",
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "readable": False,
        "format": None,
        "width": None,
        "height": None,
        "channels": None,
        "mode": None,
        "error": "",
    }
    if not path.exists():
        item["error"] = "missing"
        return item
    if item["size_bytes"] == 0:
        item["error"] = "zero_file"
        return item
    try:
        # Pillow gives exact dimensions/channels.  It is optional so the
        # validator remains useful in minimal server environments.
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            item["format"] = im.format
            item["width"], item["height"] = im.size
            item["mode"] = im.mode
            item["channels"] = len(im.getbands())
        item["readable"] = bool(item["width"] and item["height"])
        if not item["readable"]:
            item["error"] = "zero_dimension"
        return item
    except ImportError:
        # imghdr only checks the signature, so explicitly mark dimensions as
        # unavailable instead of claiming a complete image validation.
        kind = imghdr.what(path)
        item["format"] = kind
        item["readable"] = kind is not None
        item["error"] = "" if kind else "decode_failed"
        item["dependency_note"] = "Pillow unavailable; dimensions/channels not checked"
        return item
    except Exception as exc:
        item["error"] = f"decode_failed:{type(exc).__name__}:{exc}"
        return item


def npy_header(raw: bytes) -> tuple[dict[str, Any], int]:
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("invalid_npy_magic")
    major, minor = raw[6], raw[7]
    if major == 1:
        hlen = struct.unpack("<H", raw[8:10])[0]
        start = 10
    elif major in (2, 3):
        hlen = struct.unpack("<I", raw[8:12])[0]
        start = 12
    else:
        raise ValueError(f"unsupported_npy_version:{major}.{minor}")
    header = raw[start : start + hlen].decode("latin1").strip()
    # NPY headers are Python literals with only these three fields.  Using
    # literal_eval avoids executing arbitrary code from a corrupt archive.
    import ast

    return ast.literal_eval(header), start + hlen


def scan_npz(path: Path, use_numpy: bool) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path),
        "kind": "npz",
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "readable": False,
        "fields": [],
        "nan_count": 0,
        "inf_count": 0,
        "error": "",
    }
    if not path.exists():
        item["error"] = "missing"
        return item
    if item["size_bytes"] == 0:
        item["error"] = "zero_file"
        return item
    try:
        with zipfile.ZipFile(path) as archive:
            names = sorted(n for n in archive.namelist() if n.endswith(".npy"))
            if not names:
                raise ValueError("empty_npz")
            for name in names:
                header, _ = npy_header(archive.read(name)[:4096])
                shape = tuple(int(x) for x in header["shape"])
                field = {
                    "name": name[:-4],
                    "shape": list(shape),
                    "dtype": str(header["descr"]),
                    "fortran_order": bool(header["fortran_order"]),
                    "empty": any(dim == 0 for dim in shape),
                }
                item["fields"].append(field)
            item["readable"] = True
            if any(f["empty"] for f in item["fields"]):
                item["error"] = "empty_array"
            if use_numpy:
                import numpy as np  # type: ignore

                with np.load(path, allow_pickle=False) as data:
                    for name in sorted(data.files):
                        arr = data[name]
                        if np.issubdtype(arr.dtype, np.inexact):
                            item["nan_count"] += int(np.isnan(arr).sum())
                            item["inf_count"] += int(np.isinf(arr).sum())
                if item["nan_count"] or item["inf_count"]:
                    item["error"] = "nan_or_inf"
            else:
                item["dependency_note"] = "NumPy unavailable; NaN/Inf values not checked"
    except Exception as exc:
        item["error"] = f"npz_read_failed:{type(exc).__name__}:{exc}"
    return item


def required_paths(
    row: dict[str, str],
    source_root: Path,
    clean_root: Path,
    sid: str,
    asset_index: dict[str, list[Path]],
) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    human = resolve_source_path(row.get("human_path", ""), source_root, clean_root, sid)
    mannequin = resolve_source_path(row.get("mannequin_path", ""), source_root, clean_root, sid)
    if human:
        pairs.append(("human_image", human))
    if mannequin:
        pairs.append(("mannequin_image", mannequin))
    # These are the derived assets used by the existing M2H training pipeline.
    for path in asset_index.get(sid, []):
        try:
            label = path.relative_to(source_root).as_posix()
        except ValueError:
            label = path.relative_to(clean_root).as_posix()
        pairs.append((f"derived:{label}", path))
    # Include all files physically present in the quarantine bundle, including
    # files whose basename is not the sample id.
    for path in quarantine_candidates(clean_root, sid):
        pairs.append(("quarantine:" + path.relative_to(clean_root).as_posix(), path))
    return sorted(set(pairs), key=lambda x: (x[0], str(x[1])))


def record_status(info: dict[str, Any]) -> str:
    if not info["exists"]:
        return "missing"
    if not info["readable"]:
        return "unreadable"
    if info.get("error") in {"nan_or_inf", "empty_array"}:
        return info["error"]
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    requested_workers = args.num_workers  # deterministic single-process ordering is intentional

    clean_root = Path(args.dataset_root).resolve()
    source_root = Path("/data/muxiangyu/datasets/M2HImage/M2H_Final_v2")
    report_root = clean_root / "dataset_cleaning_report"
    manifest_path = Path(args.split_manifest) if args.split_manifest else report_root / "person_disjoint_manifest.csv"
    if manifest_path.is_dir():
        manifest_path = manifest_path / "person_disjoint_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest = load_manifest(manifest_path)
    split_dir = clean_root / "splits_person_disjoint"
    split_ids = {
        split: read_ids(split_dir / f"{split}.txt")
        for split in ("train", "val", "test", "quarantine")
    }
    all_ids = sorted(set().union(*split_ids.values()))
    missing_manifest = sorted(set(all_ids) - set(manifest))

    try:
        import numpy  # type: ignore  # noqa: F401

        use_numpy = True
    except ImportError:
        use_numpy = False

    # Build the basename index once.  Calling rglob separately for every id
    # makes a 40k-sample validation needlessly expensive on network storage.
    asset_index: dict[str, list[Path]] = {}
    wanted_ids = set(all_ids)
    for base in (source_root / "derived", source_root / "dwpose", clean_root / "quarantine"):
        if not base.exists():
            continue
        for dirpath, _, filenames in os.walk(base):
            for filename in filenames:
                suffix = Path(filename).suffix.lower()
                if Path(filename).stem not in wanted_ids:
                    continue
                if suffix not in IMAGE_EXTENSIONS | NPZ_EXTENSIONS | {".npy", ".json"}:
                    continue
                path = Path(dirpath) / filename
                asset_index.setdefault(path.stem, []).append(path)
    for paths in asset_index.values():
        paths.sort()

    records: list[dict[str, Any]] = []
    seen_assets: set[str] = set()
    for split in ("train", "val", "test", "quarantine"):
        for sid in split_ids[split]:
            row = manifest.get(sid, {"id": sid})
            for label, path in required_paths(row, source_root, clean_root, sid, asset_index):
                key = str(path)
                if key in seen_assets:
                    continue
                seen_assets.add(key)
                suffix = path.suffix.lower()
                if suffix in IMAGE_EXTENSIONS:
                    info = image_info(path)
                elif suffix in NPZ_EXTENSIONS:
                    info = scan_npz(path, use_numpy)
                else:
                    continue
                info.update({"id": sid, "split": split, "asset": label, "status": record_status(info)})
                records.append(info)

    # Capture image/NPZ files physically present under the clean root, even if
    # they are not represented by a manifest row (a useful leakage check).
    for path in sorted(clean_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS | NPZ_EXTENSIONS:
            continue
        if str(path) in seen_assets:
            continue
        seen_assets.add(str(path))
        info = image_info(path) if path.suffix.lower() in IMAGE_EXTENSIONS else scan_npz(path, use_numpy)
        info.update({"id": "", "split": "unindexed", "asset": "unindexed", "status": record_status(info)})
        records.append(info)

    records.sort(key=lambda x: (x.get("split", ""), x.get("id", ""), x["path"]))
    failures = [r for r in records if r["status"] != "ok"]
    dependency_notes = sorted(
        {r["dependency_note"] for r in records if r.get("dependency_note")}
    )
    summary: dict[str, Any] = {
        "schema_version": "m2h_clean_validation_v1",
        "created_at_epoch": int(time.time()),
        "dataset_root": str(clean_root),
        "source_root": str(source_root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split_counts": {k: len(v) for k, v in split_ids.items()},
        "indexed_id_count": len(all_ids),
        "missing_manifest_ids": missing_manifest,
        "asset_count": len(records),
        "failure_count": len(failures),
        "status_counts": {
            status: sum(r["status"] == status for r in records)
            for status in sorted({r["status"] for r in records} | {"ok"})
        },
        "image_count": sum(r["kind"] == "image" for r in records),
        "npz_count": sum(r["kind"] == "npz" for r in records),
        "nan_or_inf_files": sum(r["status"] == "nan_or_inf" for r in records),
        "shape_or_empty_files": sum(r["status"] == "empty_array" for r in records),
        "dependency_notes": dependency_notes,
        "num_workers_requested": requested_workers,
        "dry_run": bool(args.dry_run),
    }
    out_report = report_root / "full_readability_report.json"
    out_fail = report_root / "full_readability_failures.csv"
    out_summary = report_root / "validation_summary.json"
    if not args.dry_run:
        report_root.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps({"summary": summary, "records": records}, indent=2, sort_keys=True) + "\n")
        fields = ["id", "split", "asset", "kind", "path", "status", "error", "size_bytes", "format", "width", "height", "channels", "mode", "nan_count", "inf_count", "fields"]
        with out_fail.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in failures:
                writer.writerow(
                    {
                        k: json.dumps(row.get(k, []), sort_keys=True)
                        if k == "fields"
                        else row.get(k, "")
                        for k in fields
                    }
                )
        out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.strict and (failures or missing_manifest or not use_numpy):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
