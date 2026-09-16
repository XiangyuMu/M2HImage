#!/usr/bin/env python3
"""Build a frozen, provenance-rich evaluation manifest for role-flow runs.

The clean dataset contains split and identity-cluster assignments, while the
images remain in the read-only original dataset.  This tool joins those two
facts and emits explicit absolute source paths.  It never copies or modifies
images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EXPERIMENT_ROOT = Path("/data/muxiangyu/experiments/M2H_Final_v2_clean_v1")
DEFAULT_DATASET_ROOT = Path("/data/muxiangyu/datasets/M2HImage/M2H_Final_v2_clean_v1")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_person_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"person manifest is empty: {path}")
    required = {"id", "source_dataset", "source_key", "human_path", "mannequin_path", "clean_split", "person_cluster_id"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"person manifest missing columns: {sorted(missing)}")
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"person manifest has duplicate ids: {path}")
    return result


def load_pair_source(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs", payload) if isinstance(payload, dict) else payload
    if not isinstance(pairs, list):
        raise ValueError(f"pair source must contain a list or pairs field: {path}")
    return [dict(row) for row in pairs]


def balanced_ids(ids: Iterable[str], count: int) -> list[str]:
    """Deterministically take IDs in sorted order, with no random state."""
    return sorted(ids)[:count]


def fallback_pairs(ids: list[str], mannequin_count: int, identity_count: int, seeds: list[int]) -> list[dict[str, Any]]:
    mannequins = balanced_ids(ids, mannequin_count)
    identities = balanced_ids(ids, identity_count)
    pairs: list[dict[str, Any]] = []
    for mid in mannequins:
        choices = [jid for jid in identities if jid != mid]
        for jid in choices:
            pairs.append({"mannequin_id": mid, "identity_id": jid, "theta_source": mid, "seeds": list(seeds)})
    return pairs


def _check_source_path(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} source image is missing: {path}")


def build_manifest(
    *,
    dataset_root: Path,
    person_manifest_path: Path,
    split_dir: Path,
    output: Path,
    pair_source: Path | None = None,
    val_pair_source: Path | None = None,
    test_mannequin_count: int = 50,
    test_identity_count: int = 20,
    val_mannequin_count: int = 10,
    val_identity_count: int = 10,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    person = load_person_manifest(person_manifest_path)
    split_files = {name: split_dir / f"{name}.txt" for name in ("train", "val", "test", "quarantine")}
    split_ids = {name: read_ids(path) for name, path in split_files.items()}
    quarantine = set(split_ids["quarantine"]) | {sid for sid, row in person.items() if row.get("clean_split") == "quarantine"}
    seeds = [0, 1] if seeds is None else list(seeds)

    test_source_pairs = load_pair_source(pair_source)
    val_source_pairs = load_pair_source(val_pair_source)
    if not test_source_pairs:
        test_source_pairs = fallback_pairs(split_ids["test"], test_mannequin_count, test_identity_count, seeds)
    if not val_source_pairs:
        val_source_pairs = fallback_pairs(split_ids["val"], val_mannequin_count, val_identity_count, seeds)

    source_pairs = [("val", val_source_pairs), ("test", test_source_pairs)]
    rows: list[dict[str, Any]] = []
    for split, pairs in source_pairs:
        allowed = set(split_ids[split]) - quarantine
        for pair_index, pair in enumerate(pairs):
            mid = str(pair.get("mannequin_id", pair.get("mid", "")))
            jid = str(pair.get("identity_id", pair.get("jid", "")))
            if not mid or not jid:
                raise ValueError(f"pair {split}/{pair_index} lacks mannequin_id or identity_id")
            if mid not in allowed or jid not in allowed:
                raise ValueError(f"pair {split}/{pair_index} is outside clean {split}: {mid}, {jid}")
            if mid not in person or jid not in person:
                raise ValueError(f"pair {split}/{pair_index} is absent from person manifest: {mid}, {jid}")
            mrow, jrow = person[mid], person[jid]
            if mrow.get("clean_split") != split or jrow.get("clean_split") != split:
                raise ValueError(f"pair {split}/{pair_index} has inconsistent clean_split")
            mannequin_path = Path(mrow["mannequin_path"]).expanduser().resolve()
            human_path = Path(jrow["human_path"]).expanduser().resolve()
            _check_source_path(mannequin_path, f"mannequin {mid}")
            _check_source_path(human_path, f"human {jid}")
            row_seeds = pair.get("seeds", seeds)
            for seed in row_seeds:
                rows.append({
                    "mid": mid,
                    "jid": jid,
                    "seed": int(seed),
                    "split": split,
                    "pair_index": pair_index,
                    "theta_source": str(pair.get("theta_source", mid)),
                    "garment_type": str(pair.get("garment_type", "unknown")),
                    "mannequin_path": str(mannequin_path),
                    "human_path": str(human_path),
                    "mannequin_sha256": sha256_file(mannequin_path),
                    "human_sha256": sha256_file(human_path),
                    "mannequin_source_dataset": mrow.get("source_dataset", ""),
                    "identity_source_dataset": jrow.get("source_dataset", ""),
                    "mannequin_source_key": mrow.get("source_key", ""),
                    "identity_source_key": jrow.get("source_key", ""),
                    "mannequin_person_cluster_id": mrow.get("person_cluster_id", ""),
                    "identity_person_cluster_id": jrow.get("person_cluster_id", ""),
                    "mannequin_clean_split": mrow.get("clean_split", ""),
                    "identity_clean_split": jrow.get("clean_split", ""),
                })
    rows.sort(key=lambda row: (row["split"], row["mid"], row["jid"], row["seed"]))
    keys = [(row["split"], row["mid"], row["jid"], row["seed"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("evaluation manifest contains duplicate split/mid/jid/seed keys")

    input_hashes = {
        "person_manifest": sha256_file(person_manifest_path),
        "split_files": {name: sha256_file(path) for name, path in split_files.items()},
    }
    if pair_source and pair_source.exists():
        input_hashes["test_pair_source"] = sha256_file(pair_source)
    if val_pair_source and val_pair_source.exists():
        input_hashes["val_pair_source"] = sha256_file(val_pair_source)
    provenance = {
        "schema_version": "m2h-role-flow-eval-v1",
        "dataset_root": str(dataset_root.resolve()),
        "read_only_source_root": "/data/muxiangyu/datasets/M2HImage/M2H_Final_v2",
        "person_manifest": str(person_manifest_path.resolve()),
        "split_dir": str(split_dir.resolve()),
        "input_sha256": input_hashes,
        "selection": {
            "test_pair_source": str(pair_source.resolve()) if pair_source and pair_source.exists() else "deterministic_sorted_fallback",
            "val_pair_source": str(val_pair_source.resolve()) if val_pair_source and val_pair_source.exists() else "deterministic_sorted_fallback",
            "seeds": seeds,
            "quarantine_count": len(quarantine),
        },
    }
    payload = {
        "schema_version": "m2h-role-flow-eval-v1",
        "provenance": provenance,
        "rows": rows,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--person-manifest", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pair-source", type=Path, default=None, help="Frozen test pair source, such as role_test_subset.json")
    parser.add_argument("--val-pair-source", type=Path, default=None)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--test-mannequin-count", type=int, default=50)
    parser.add_argument("--test-identity-count", type=int, default=20)
    parser.add_argument("--val-mannequin-count", type=int, default=10)
    parser.add_argument("--val-identity-count", type=int, default=10)
    parser.add_argument("--seeds", default="0,1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = args.dataset_root / "dataset_cleaning_report"
    split_dir = args.split_dir or args.dataset_root / "splits_person_disjoint_final"
    person_manifest = args.person_manifest or report / "person_disjoint_manifest.csv"
    output = args.output or args.experiment_root / "role_flow" / "eval_manifest.json"
    pair_source = args.pair_source
    if pair_source is None:
        candidate = args.experiment_root / "role_flow" / "eval" / "role_test_subset.json"
        pair_source = candidate if candidate.exists() else None
    payload = build_manifest(
        dataset_root=args.dataset_root,
        person_manifest_path=person_manifest,
        split_dir=split_dir,
        output=output,
        pair_source=pair_source,
        val_pair_source=args.val_pair_source,
        test_mannequin_count=args.test_mannequin_count,
        test_identity_count=args.test_identity_count,
        val_mannequin_count=args.val_mannequin_count,
        val_identity_count=args.val_identity_count,
        seeds=[int(item) for item in args.seeds.split(",") if item.strip()],
    )
    counts = defaultdict(int)
    for row in payload["rows"]:
        counts[row["split"]] += 1
    print(json.dumps({"output": str(output), "content_sha256": payload["content_sha256"], "rows_by_split": dict(counts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
