from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from complete_2x2_blocks import manifest_sha256, validate_complete_blocks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster(source_key: str) -> str:
    head, separator, tail = source_key.rpartition("_")
    return head if separator and tail.isdigit() else source_key


def build(dataset_root: Path, output: Path, block_count: int = 20, seed: int = 20260906) -> dict[str, object]:
    if not 20 <= block_count <= 50:
        raise ValueError("block_count must be in [20, 50]")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest_path = dataset_root / "metadata" / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "val" and row.get("action") == "keep"]
    sources = sorted({row["source_dataset"] for row in rows})
    if len(sources) < 2:
        raise ValueError("both frozen synthetic sources are required")
    per_source_blocks = {source: block_count // len(sources) for source in sources}
    for source in sources[: block_count % len(sources)]:
        per_source_blocks[source] += 1

    output_rows: list[dict[str, object]] = []
    used_ids: set[str] = set()
    used_clusters: set[str] = set()
    block_index = 0
    for source in sources:
        all_candidates = sorted((row for row in rows if row["source_dataset"] == source), key=lambda row: row["id"])
        candidates = []
        source_clusters: set[str] = set()
        for row in all_candidates:
            row_cluster = cluster(row["source_key"])
            if row_cluster in source_clusters:
                continue
            source_clusters.add(row_cluster)
            candidates.append(row)
        needed = per_source_blocks[source] * 4
        if len(candidates) < needed:
            raise ValueError(f"not enough {source} validation rows for {per_source_blocks[source]} blocks")
        candidates = candidates[:needed]
        for local_index in range(per_source_blocks[source]):
            identity_rows = candidates[local_index * 4: local_index * 4 + 2]
            mannequin_rows = candidates[local_index * 4 + 2: local_index * 4 + 4]
            block_id = f"g3pilot_{block_index:03d}"
            block_seed = int(seed + block_index)
            noise_hash = hashlib.sha256(f"{block_id}:{block_seed}".encode()).hexdigest()
            for row in identity_rows + mannequin_rows:
                if row["id"] in used_ids or cluster(row["source_key"]) in used_clusters:
                    raise ValueError("selection unexpectedly reuses an image or near-duplicate cluster")
                used_ids.add(row["id"])
                used_clusters.add(cluster(row["source_key"]))
            for identity_index, identity_row in enumerate(identity_rows):
                for mannequin_index, mannequin_row in enumerate(mannequin_rows):
                    output_rows.append({
                        "block_id": block_id,
                        "cell_id": f"I{identity_index}M{mannequin_index}",
                        "identity_index": identity_index,
                        "mannequin_index": mannequin_index,
                        "identity_id": identity_row["id"],
                        "mannequin_id": mannequin_row["id"],
                        "noise_seed": block_seed,
                        "noise_hash": noise_hash,
                        "split": "dev",
                        "source_dataset": source,
                        "intervention_type": "garment",
                        "garment_sku": f"{source}:{mannequin_row['source_key']}",
                        "context_id": f"{source}:g3pilot_context_{block_index:03d}",
                        "input_identity_path": identity_row["human_path"],
                        "input_mannequin_path": mannequin_row["mannequin_path"],
                        "source_split": identity_row["split"],
                        "identity_source_key": identity_row["source_key"],
                        "mannequin_source_key": mannequin_row["source_key"],
                        "identity_near_duplicate_cluster": cluster(identity_row["source_key"]),
                        "mannequin_near_duplicate_cluster": cluster(mannequin_row["source_key"]),
                    })
            block_index += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    validation = validate_complete_blocks(output_rows, expected_split="dev")
    metadata = {
        "schema_version": 1,
        "status": "prepared_not_run",
        "purpose": "G3 engineering manifest; no generation started",
        "planning_authority": [
            ".omx/plans/prd-cvpr-source-attributed-m2h.md",
            ".omx/plans/test-spec-cvpr-source-attributed-m2h.md",
        ],
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "selection_seed": seed,
        "block_count": block_count,
        "cell_count": len(output_rows),
        "source_counts": {source: per_source_blocks[source] for source in sources},
        "unique_source_images": len(used_ids),
        "unique_near_duplicate_clusters": len(used_clusters),
        "validation": validation,
        "manifest_sha256": manifest_sha256(output_rows),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    print(json.dumps(build(args.dataset_root, args.output, args.block_count, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
