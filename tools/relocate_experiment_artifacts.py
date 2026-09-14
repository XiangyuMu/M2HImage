#!/usr/bin/env python3
"""Register legacy training artifacts under the experiment root.

The source dataset remains intact. Files are hard-linked into the experiment
tree, so this operation is reversible and does not create a second data copy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(rel: Path) -> str | None:
    parts = rel.parts
    if "eval" in parts:
        return "eval"
    if "checkpoints" in parts or "phase1" in parts and any("checkpoint" in x for x in parts):
        return "checkpoints"
    if "cache_" in str(rel) or "cache" in parts:
        return "cache_m_only"
    if "logs" in parts or rel.suffix in {".log", ".out", ".err"}:
        return "logs"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/data/muxiangyu/datasets/M2HImage/M2H_Final_v2_clean_v1")
    ap.add_argument("--experiment-root", default="/data/muxiangyu/experiments/M2H_Final_v2_clean_v1")
    ap.add_argument("--split-manifest", default=None)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from-manifest", default=None,
                    help="Execute links from an existing dry-run manifest without re-hashing sources.")
    args = ap.parse_args()
    source = Path(args.dataset_root)
    exp = Path(args.experiment_root)
    # Legacy artifacts are under the read-only original root referenced by the clean manifest.
    original = source.parent / "M2H_Final_v2"
    roots = [original / "phase1", original / "eval", original / "derived/logs"]
    rows = []
    if args.from_manifest:
        with Path(args.from_manifest).open(newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["status"] = "linked"
            source_path = Path(row["old_path"])
            destination = Path(row["new_path"])
            if not source_path.is_file():
                row["status"] = "missing_source"
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                row["status"] = "already_exists" if digest(destination) == row["sha256"] else "conflict"
                continue
            try:
                os.link(source_path, destination)
            except OSError:
                row["status"] = "link_failed"
        exp.mkdir(parents=True, exist_ok=True)
        for d in ("cache_m_only", "checkpoints", "eval", "logs", "manifests"):
            (exp / d).mkdir(exist_ok=True)
        target = exp / "manifests/artifact_relocation_manifest.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="") as f:
            fields = ["old_path", "new_path", "sha256", "size_bytes", "migration_time", "mode", "status", "split_semantics"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        failures = sum(r["status"] in {"conflict", "link_failed", "missing_source"} for r in rows)
        print(f"planned_or_linked={len(rows)} failures={failures} dry_run={args.dry_run}")
        return 2 if args.strict and failures else 0
    seen = set()
    stamp = datetime.now(timezone.utc).isoformat()
    for base in roots:
        if not base.is_dir():
            continue
        for p in sorted(x for x in base.rglob("*") if x.is_file()):
            kind = classify(p.relative_to(original))
            if kind is None:
                continue
            if p in seen:
                continue
            seen.add(p)
            rel = p.relative_to(original)
            destination = exp / kind / "legacy_split" / rel
            row = {"old_path": str(p), "new_path": str(destination), "sha256": digest(p),
                   "size_bytes": p.stat().st_size, "migration_time": stamp,
                   "mode": "hardlink", "status": "planned" if args.dry_run else "linked",
                   "split_semantics": "legacy_split"}
            rows.append(row)
            if not args.dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if digest(destination) != row["sha256"]:
                        row["status"] = "conflict"
                    continue
                try:
                    os.link(p, destination)
                except OSError:
                    row["status"] = "link_failed"
    if not args.dry_run:
        exp.mkdir(parents=True, exist_ok=True)
        for d in ("cache_m_only", "checkpoints", "eval", "logs", "manifests"):
            (exp / d).mkdir(exist_ok=True)
    target = exp / "manifests/artifact_relocation_manifest.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as f:
        fields = ["old_path", "new_path", "sha256", "size_bytes", "migration_time", "mode", "status", "split_semantics"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    failures = sum(r["status"] in {"conflict", "link_failed"} for r in rows)
    print(f"planned_or_linked={len(rows)} failures={failures} dry_run={args.dry_run}")
    return 2 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
