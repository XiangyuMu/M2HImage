from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


UNIT_SPECS = (
    ("face", "identity_morphology", "face_morphology"),
    ("hair", "hair_appearance_geometry", "reliable_local_hair"),
    ("head_pose", "global_geometry_pose", "global_head_pose"),
    ("garment", "garment_appearance", "garment_design"),
    ("drape", "garment_drape", "garment_gross_drape"),
    ("scene", "scene", "background_scene"),
    ("hair_contact", "occlusion", "hair_face_contact"),
)

ANNOTATION_FIELDS = (
    "annotator_id",
    "image_id",
    "unit_id",
    "feature_family",
    "attribute",
    "support_label",
    "occlusion_order",
    "owner_observable",
    "unit_weight",
    "canvas_width",
    "canvas_height",
    "boundary_polygon_json",
    "artifacts",
    "notes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def near_duplicate_cluster(source_key: str) -> str:
    head, separator, tail = source_key.rpartition("_")
    return head if separator and tail.isdigit() else source_key


def select_rows(rows: list[dict[str, str]], count: int, seed: int, split: str) -> list[dict[str, str]]:
    eligible = [
        row
        for row in rows
        if row.get("split") == split
        and row.get("action") == "keep"
        and Path(row.get("human_path", "")).is_file()
        and Path(row.get("mannequin_path", "")).is_file()
    ]
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        by_source[row["source_dataset"]].append(row)
    if len(by_source) < 2:
        raise RuntimeError("the G2 pack requires both frozen synthetic dataset sources")

    quotas = {source: count // len(by_source) for source in sorted(by_source)}
    for source in sorted(by_source)[: count % len(by_source)]:
        quotas[source] += 1

    selected: list[dict[str, str]] = []
    used_clusters: set[str] = set()
    for source in sorted(by_source):
        candidates = sorted(by_source[source], key=lambda row: stable_rank(seed, row["id"]))
        for row in candidates:
            cluster = near_duplicate_cluster(row["source_key"])
            if cluster in used_clusters:
                continue
            selected.append(row)
            used_clusters.add(cluster)
            if sum(value["source_dataset"] == source for value in selected) == quotas[source]:
                break
    if len(selected) != count:
        raise RuntimeError(f"could select only {len(selected)} of {count} requested samples")
    return sorted(selected, key=lambda row: stable_rank(seed, row["id"]))


def write_csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_pack(dataset_root: Path, output_dir: Path, count: int, seed: int, split: str) -> dict[str, Any]:
    manifest_path = dataset_root / "metadata" / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_rows(rows, count=count, seed=seed, split=split)

    manifest_rows = []
    for rank, row in enumerate(selected):
        human_path = Path(row["human_path"])
        mannequin_path = Path(row["mannequin_path"])
        manifest_rows.append({
            "selection_rank": rank,
            "selection_seed": seed,
            "image_id": row["id"],
            "source_dataset": row["source_dataset"],
            "source_key": row["source_key"],
            "near_duplicate_cluster": near_duplicate_cluster(row["source_key"]),
            "split": row["split"],
            "human_path": str(human_path),
            "human_sha256": sha256_file(human_path),
            "mannequin_path": str(mannequin_path),
            "mannequin_sha256": sha256_file(mannequin_path),
        })
    manifest_fields = list(manifest_rows[0])
    selection_path = output_dir / "selection_manifest.csv"
    write_csv(selection_path, manifest_fields, manifest_rows)

    blank_rows = []
    for row in manifest_rows:
        for suffix, family, attribute in UNIT_SPECS:
            blank_rows.append({
                "annotator_id": "",
                "image_id": row["image_id"],
                "unit_id": f"{row['image_id']}:{suffix}",
                "feature_family": family,
                "attribute": attribute,
                "support_label": "",
                "occlusion_order": "",
                "owner_observable": "",
                "unit_weight": "",
                "canvas_width": "768",
                "canvas_height": "1024",
                "boundary_polygon_json": "",
                "artifacts": "",
                "notes": "",
            })
    annotation_paths = []
    for index in range(1, 4):
        path = output_dir / f"annotations_rater_{index:02d}.csv"
        rows_for_rater = [dict(row, annotator_id=f"rater_{index:02d}") for row in blank_rows]
        write_csv(path, list(ANNOTATION_FIELDS), rows_for_rater)
        annotation_paths.append(path)

    adjudication_path = output_dir / "adjudication.csv"
    adjudication_fields = list(ANNOTATION_FIELDS) + ["adjudicator_id", "adjudication_reason"]
    write_csv(adjudication_path, adjudication_fields, blank_rows)

    metadata = {
        "schema_version": 1,
        "purpose": "G2 ownership ontology and hair-policy gate",
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "sample_count": count,
        "unit_count_per_annotator": len(blank_rows),
        "selection_seed": seed,
        "split": split,
        "annotation_files": [str(path) for path in annotation_paths],
        "adjudication_file": str(adjudication_path),
        "image_copy_policy": "paths_and_hashes_only",
    }
    metadata_path = output_dir / "pack_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic, no-image-copy G2 annotation pack.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30, choices=range(20, 51))
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_pack(args.dataset_root, args.output_dir, args.count, args.seed, args.split)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
