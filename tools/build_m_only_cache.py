#!/usr/bin/env python3
"""Build a lightweight mannequin-only condition cache.

The clean dataset is intentionally a manifest/quarantine overlay over the
read-only source dataset. This builder therefore resolves source files from
the person-disjoint manifest and writes only mannequin-derived arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_DATASET = "/data/muxiangyu/datasets/M2HImage/M2H_Final_v2_clean_v1"
DEFAULT_EXPERIMENT = "/data/muxiangyu/experiments/M2H_Final_v2_clean_v1"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
# Keep this aligned with metrics_v2.parsing.DEFAULT_GARMENT_LABELS. Labels
# 8/9/11/16/17 are accessories, face-adjacent, or body pixels and must not
# become clothing conditions merely because the parser predicted them.
GARMENT_LABELS = (3, 4, 5, 6, 7, 10)
HAIR_LABEL = 2
FACE_LABEL = 1
BODY_LABELS = (12, 13, 14, 15)
TOKEN_H, TOKEN_W, POOL = 64, 48, 16
_SHA_CACHE: dict[str, str] = {}
_SHA_LOCK = threading.Lock()


def sha256(path: Path) -> str:
    cache_key = str(path)
    with _SHA_LOCK:
        cached = _SHA_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    with _SHA_LOCK:
        _SHA_CACHE[cache_key] = value
    return value


def find_one(directory: Path, sample_id: str) -> Path:
    for ext in IMAGE_EXTS:
        candidate = directory / f"{sample_id}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(directory.glob(f"{sample_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"missing {sample_id} under {directory}")


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "id" not in rows.fieldnames:
            raise ValueError(f"manifest lacks id column: {path}")
        result = {}
        for row in rows:
            sample_id = str(row["id"]).strip()
            if sample_id:
                result[sample_id] = {str(k): str(v or "") for k, v in row.items()}
    return result


def region_tokens(parsing_array: np.ndarray) -> dict[str, np.ndarray]:
    """Derive all region conditions from the mannequin parsing map only."""
    height, width = parsing_array.shape
    target_h, target_w = TOKEN_H * POOL, TOKEN_W * POOL
    if (height, width) != (target_h, target_w):
        parsing_array = np.asarray(
            Image.fromarray(parsing_array, mode="L").resize(
                (target_w, target_h), Image.Resampling.NEAREST
            ), dtype=np.uint8,
        )
    garment = np.isin(parsing_array, GARMENT_LABELS)
    face = parsing_array == FACE_LABEL
    hair = parsing_array == HAIR_LABEL
    body = np.isin(parsing_array, BODY_LABELS)
    # body_bg is the non-garment, non-face/hair region used by the spatial
    # condition; every mask is pooled in the same row-major 16x16 grid.
    body_bg = body | ((parsing_array == 0) & ~garment)
    masks = {
        "cloth_safe_z": garment,
        "body_bg_z": body_bg,
        "face_z": face,
        "hair_z": hair,
    }
    return {
        key: value.reshape(TOKEN_H, POOL, TOKEN_W, POOL).mean(axis=(1, 3)).reshape(-1).astype(np.float16)
        for key, value in masks.items()
    }


def source_path(row: dict[str, str], field: str, fallback: Path) -> Path:
    value = row.get(field, "")
    return Path(value) if value else fallback


def source_record(
    field_name: str,
    source_role: str,
    path: Path,
    generator: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "field_name": field_name,
        "source_role": source_role,
        "source_path": str(path),
        "source_sha256": sha256(path),
        "generator": generator,
        "generator_version": "m-only-cache-v1",
        "preprocess_config": config,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_sample(
    sample_id: str,
    row: dict[str, str],
    source_root: Path,
    overlay_root: Path,
    out_path: Path,
    provenance_path: Path,
    strict: bool,
) -> dict[str, Any]:
    mannequin = source_path(row, "mannequin_path", find_one(source_root / "images/mannequin", sample_id))
    parsing = find_one(source_root / "human_parsing/fashn/masks/mannequin", sample_id)
    pose = find_one(source_root / "dwpose/with_head/mannequin", sample_id)
    keypoints_source = source_root / "dwpose/keypoints/mannequin" / f"{sample_id}.npz"
    keypoints_overlay = overlay_root / "dwpose/keypoints/mannequin" / f"{sample_id}.npz"
    keypoints = keypoints_overlay if keypoints_overlay.exists() else keypoints_source
    required = [mannequin, parsing, pose, keypoints]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{sample_id}: missing {missing}")

    with Image.open(mannequin) as image_handle:
        image = image_handle.convert("RGB")
        image_array = np.asarray(image, dtype=np.uint8)
    with Image.open(parsing) as parsing_handle:
        parsing_array = np.asarray(parsing_handle.convert("L"), dtype=np.uint8)
    if parsing_array.shape != image_array.shape[:2]:
        parsing_array = np.asarray(
            Image.fromarray(parsing_array, mode="L").resize(
                (image_array.shape[1], image_array.shape[0]), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
    garment_mask = np.isin(parsing_array, GARMENT_LABELS).astype(np.uint8) * 255
    neutral = np.full_like(image_array, 127, dtype=np.uint8)
    garment_image = np.where(garment_mask[..., None] > 0, image_array, neutral)

    with np.load(keypoints, allow_pickle=False) as keypoint_payload:
        keypoint_arrays = {
            f"mannequin_dwpose_{key}": np.asarray(keypoint_payload[key])
            for key in keypoint_payload.files
        }
    region_arrays = {
        f"mannequin_region_{key}": value for key, value in region_tokens(parsing_array).items()
    }
    payload: dict[str, np.ndarray] = {
        "mannequin_garment_mask": garment_mask,
        "mannequin_garment_image": garment_image,
        "mannequin_parsing": parsing_array,
        "mannequin_pose_image": np.asarray(Image.open(pose).convert("RGB"), dtype=np.uint8),
        **keypoint_arrays,
        **region_arrays,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".npz.tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp_path.replace(out_path)

    config = {
        "garment_labels": list(GARMENT_LABELS),
        "neutral_gray": 127,
        "pose_source": "dwpose/with_head/mannequin",
        "allow_human_inputs": False,
    }
    records = []
    field_sources = {
        "mannequin_garment_mask": parsing,
        "mannequin_garment_image": mannequin,
        "mannequin_parsing": parsing,
        "mannequin_pose_image": pose,
    }
    for field_name, path in field_sources.items():
        records.append(source_record(field_name, "mannequin", path, "build_m_only_cache", config))
    for field_name in keypoint_arrays:
        generator = "copy_npz_field_repaired_overlay" if keypoints == keypoints_overlay else "copy_npz_field"
        records.append(source_record(field_name, "mannequin", keypoints, generator, config))
    for field_name in region_arrays:
        records.append(source_record(field_name, "mannequin", parsing, "derive_region_tokens_from_parsing", config))
    with provenance_path.open("a", encoding="utf-8") as handle:
        for record in records:
            record.update({"sample_id": sample_id, "output_path": str(out_path)})
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "id": sample_id,
        "status": "ok",
        "cache_path": str(out_path),
        "field_count": len(payload),
        "source_role": "mannequin",
        "output_sha256": sha256(out_path),
        "shape": [int(image_array.shape[1]), int(image_array.shape[0])],
    }


def build_one_safe(
    sample_id: str,
    row: dict[str, str],
    source_root: Path,
    overlay_root: Path,
    split_out: Path,
    provenance_path: Path,
    strict: bool,
) -> dict[str, Any]:
    try:
        return build_sample(
            sample_id,
            row,
            source_root,
            overlay_root,
            split_out / f"{sample_id}.npz",
            provenance_path,
            strict,
        )
    except Exception as exc:  # noqa: BLE001
        return {"id": sample_id, "status": "failed", "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="limit each split for smoke tests")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    dataset_root = Path(args.dataset_root)
    experiment_root = Path(args.experiment_root)
    report_root = dataset_root / "dataset_cleaning_report"
    manifest_path = Path(args.split_manifest) if args.split_manifest else report_root / "person_disjoint_manifest.csv"
    manifest = read_manifest(manifest_path)
    source_root = Path("/data/muxiangyu/datasets/M2HImage/M2H_Final_v2")
    overlay_root = dataset_root / "asset_overlay_v1"
    split_root = dataset_root / "splits_person_disjoint"
    if not source_root.exists():
        raise FileNotFoundError(f"source dataset is unavailable: {source_root}")
    cache_root = experiment_root / "cache_m_only"
    provenance_path = report_root / "cache_field_provenance.jsonl"
    summary_path = cache_root / "cache_build_summary.json"
    if not args.dry_run:
        cache_root.mkdir(parents=True, exist_ok=True)
        report_root.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text("", encoding="utf-8")

    summary: dict[str, Any] = {
        "version": "m-only-cache-v1",
        "dataset_root": str(dataset_root),
        "source_root": str(source_root),
        "experiment_root": str(experiment_root),
        "manifest": str(manifest_path),
        "splits": {},
        "num_workers_requested": args.num_workers,
        "dry_run": args.dry_run,
        "human_inputs_allowed": False,
    }
    failures: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        split_file = split_root / f"{split}.txt"
        ids = read_ids(split_file)
        if args.limit is not None:
            ids = ids[: args.limit]
        rows = []
        split_out = cache_root / split / "samples"
        split_manifest = cache_root / split / "manifest.csv"
        if args.dry_run:
            for sample_id in ids:
                row = manifest.get(sample_id, {})
                expected = source_path(
                    row, "mannequin_path",
                    find_one(source_root / "images/mannequin", sample_id),
                )
                rows.append({"id": sample_id, "status": "would_build", "mannequin_path": str(expected)})
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {
                    executor.submit(
                        build_one_safe, sample_id, manifest.get(sample_id, {}),
                        source_root, overlay_root, split_out, provenance_path, args.strict,
                    ): sample_id
                    for sample_id in ids
                }
                for future in as_completed(futures):
                    result = future.result()
                    rows.append(result)
                    if result.get("status") == "failed":
                        failures.append(result)
                        if args.strict:
                            raise RuntimeError(f"{result['id']}: {result['error']}")
            rows.sort(key=lambda item: str(item.get("id", "")))
        if not args.dry_run:
            split_manifest.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = sorted({key for row in rows for key in row})
            with split_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        summary["splits"][split] = {
            "requested": len(ids),
            "built": sum(row.get("status") == "ok" for row in rows),
            "failed": sum(row.get("status") == "failed" for row in rows),
            "manifest": str(split_manifest),
        }
    summary["failures"] = failures
    summary["total_requested"] = sum(item["requested"] for item in summary["splits"].values())
    summary["total_built"] = sum(item["built"] for item in summary["splits"].values())
    summary["total_failed"] = sum(item["failed"] for item in summary["splits"].values())
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
