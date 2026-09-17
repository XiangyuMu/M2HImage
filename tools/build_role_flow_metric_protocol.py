#!/usr/bin/env python3
"""Build frozen metric protocol artifacts for role-flow evaluation.

The artifacts are deliberately separate from generated-image evaluation:

* ``fid_reference_manifest.json`` freezes the final test human image
  population used as the FID real distribution.
* ``tar_calibration.json`` freezes the final val identity calibration protocol
  from held-out AdaFace embeddings of human face crops.

The script is manifest-driven, deterministic, and fail-closed. It does not
modify dataset files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from conditions import find_one, load_yaml


DEFAULT_DATASET_ROOT = Path("/data/muxiangyu/datasets/M2HImage/M2H_Final_v2_clean_v1")
DEFAULT_EXPERIMENT_ROOT = Path("/data/muxiangyu/experiments/M2H_Final_v2_clean_v1")
DEFAULT_CONFIG = Path("configs/role_selective/A_timestep_routed.yaml")
SCHEMA_VERSION = "m2h-role-flow-metric-protocol-v1"


class FaceCropEmbedder(Protocol):
    metadata: dict[str, Any]

    def embed(self, face_crop_path: Path) -> np.ndarray:
        ...


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    row: dict[str, str]

    @property
    def cluster_id(self) -> str:
        return str(self.row.get("person_cluster_id", ""))


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
    if not path.is_file():
        raise FileNotFoundError(f"split file is missing: {path}")
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"split file has duplicate ids: {path}")
    return ids


def load_person_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"person manifest is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"person manifest is empty: {path}")
    required = {"id", "human_path", "clean_split", "person_cluster_id"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"person manifest missing columns: {sorted(missing)}")
    output = {str(row["id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"person manifest has duplicate ids: {path}")
    return output


def load_split_records(
    *,
    split_name: str,
    split_dir: Path,
    person_manifest: dict[str, dict[str, str]],
) -> list[SampleRecord]:
    ids = sorted(read_ids(split_dir / f"{split_name}.txt"))
    quarantine_ids = set(read_ids(split_dir / "quarantine.txt")) if (split_dir / "quarantine.txt").is_file() else set()
    records: list[SampleRecord] = []
    missing: list[str] = []
    quarantined: list[str] = []
    for sample_id in ids:
        row = person_manifest.get(sample_id)
        if row is None:
            missing.append(sample_id)
            continue
        if sample_id in quarantine_ids or row.get("clean_split") == "quarantine":
            quarantined.append(sample_id)
            continue
        records.append(SampleRecord(sample_id=sample_id, row=row))
    errors = []
    if missing:
        errors.append(f"missing in person manifest: {missing[:10]}")
    if quarantined:
        errors.append(f"formal split contains quarantine ids: {quarantined[:10]}")
    if errors:
        raise ValueError(f"{split_name} split failed validation; " + "; ".join(errors))
    if not records:
        raise ValueError(f"{split_name} split has no formal records")
    return records


def resolve_manifest_path(path_value: str, dataset_root: Path, asset_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    for root in (asset_root, dataset_root):
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (dataset_root / path).resolve()


def split_file_hashes(split_dir: Path) -> dict[str, str]:
    names = ("train", "val", "test", "quarantine")
    return {name: sha256_file(split_dir / f"{name}.txt") for name in names if (split_dir / f"{name}.txt").is_file()}


def split_file_counts(split_dir: Path) -> dict[str, int]:
    names = ("train", "val", "test", "quarantine")
    return {name: len(read_ids(split_dir / f"{name}.txt")) for name in names if (split_dir / f"{name}.txt").is_file()}


def input_hashes(person_manifest_path: Path, split_dir: Path, config_path: Path | None) -> dict[str, Any]:
    hashes: dict[str, Any] = {
        "person_manifest": sha256_file(person_manifest_path),
        "split_files": split_file_hashes(split_dir),
        "split_counts": split_file_counts(split_dir),
    }
    if config_path is not None:
        hashes["config"] = sha256_file(config_path)
    return hashes


def build_fid_reference_manifest(
    *,
    dataset_root: Path,
    asset_root: Path,
    person_manifest_path: Path,
    split_dir: Path,
    config_path: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    person_manifest = load_person_manifest(person_manifest_path)
    records = load_split_records(split_name="test", split_dir=split_dir, person_manifest=person_manifest)
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for record in records:
        human_path = resolve_manifest_path(record.row["human_path"], dataset_root, asset_root)
        if not human_path.is_file():
            raise FileNotFoundError(f"test human image is missing for id={record.sample_id}: {human_path}")
        resolved = str(human_path)
        if resolved in seen_paths:
            raise ValueError(f"duplicate FID human image path: {resolved}")
        seen_paths.add(resolved)
        rows.append({
            "id": record.sample_id,
            "split": "test",
            "human_path": resolved,
            "human_sha256": sha256_file(human_path),
            "person_cluster_id": record.cluster_id,
            "manifest_clean_split": record.row.get("clean_split", ""),
            "source_dataset": record.row.get("source_dataset", ""),
            "source_key": record.row.get("source_key", ""),
        })
    rows.sort(key=lambda row: row["id"])
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "fid_reference_manifest",
        "dataset_root": str(dataset_root.resolve()),
        "asset_root": str(asset_root.resolve()),
        "person_manifest": str(person_manifest_path.resolve()),
        "split_dir": str(split_dir.resolve()),
        "config": str(config_path.resolve()) if config_path is not None else None,
        "input_sha256": input_hashes(person_manifest_path, split_dir, config_path),
        "split_authority": "splits_person_disjoint_final files; manifest clean_split is provenance only",
        "split": "test",
        "count": len(rows),
        "rows": rows,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def calibration_threshold(impostor_scores: np.ndarray, far: float) -> tuple[float, float]:
    values = np.asarray(impostor_scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("TAR calibration requires at least one impostor pair")
    if not 0.0 <= float(far) <= 1.0:
        raise ValueError(f"far must be in [0, 1], got {far}")
    allowed_false_accepts = int(np.floor(float(far) * values.size))
    if allowed_false_accepts >= values.size:
        threshold = float(np.nextafter(np.min(values), -np.inf))
        empirical_far = float(np.mean(values >= threshold))
        return threshold, empirical_far
    sorted_desc = np.sort(values)[::-1]
    boundary = float(sorted_desc[allowed_false_accepts])
    threshold = float(np.nextafter(boundary, np.inf))
    empirical_far = float(np.mean(values >= threshold))
    return threshold, empirical_far


def impostor_pairs(records: list[SampleRecord]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left, right in itertools.combinations(range(len(records)), 2):
        left_cluster = records[left].cluster_id
        right_cluster = records[right].cluster_id
        if left_cluster and right_cluster and left_cluster == right_cluster:
            continue
        pairs.append((left, right))
    return pairs


def impostor_pair_count(records: list[SampleRecord]) -> int:
    total = len(records) * (len(records) - 1) // 2
    cluster_counts: dict[str, int] = {}
    for record in records:
        if not record.cluster_id:
            continue
        cluster_counts[record.cluster_id] = cluster_counts.get(record.cluster_id, 0) + 1
    same_cluster = sum(count * (count - 1) // 2 for count in cluster_counts.values())
    return total - same_cluster


def compute_impostor_scores(embeddings: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={embeddings.shape}")
    if not pairs:
        return np.asarray([], dtype=np.float32)
    scores = np.empty((len(pairs),), dtype=np.float32)
    for index, (left, right) in enumerate(pairs):
        scores[index] = float(np.dot(embeddings[left], embeddings[right]))
    return scores


def compute_impostor_scores_for_records(
    embeddings: np.ndarray,
    records: list[SampleRecord],
) -> tuple[np.ndarray, int]:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={embeddings.shape}")
    if embeddings.shape[0] != len(records):
        raise ValueError(f"embedding row count {embeddings.shape[0]} does not match record count {len(records)}")
    if len(records) < 2:
        return np.asarray([], dtype=np.float32), 0

    similarities = np.matmul(embeddings.astype(np.float32, copy=False), embeddings.astype(np.float32, copy=False).T)
    upper_triangle = np.triu(np.ones((len(records), len(records)), dtype=bool), k=1)
    cluster_ids = np.asarray([record.cluster_id for record in records], dtype=object)
    nonempty_cluster = cluster_ids != ""
    same_cluster = (cluster_ids[:, None] == cluster_ids[None, :]) & nonempty_cluster[:, None] & nonempty_cluster[None, :]
    valid_pairs = upper_triangle & ~same_cluster
    scores = similarities[valid_pairs].astype(np.float32, copy=False)
    excluded_same_cluster_pair_count = int(upper_triangle.sum() - valid_pairs.sum())
    return scores, excluded_same_cluster_pair_count


class HeldoutAdaFaceEmbedder:
    def __init__(self, cfg: dict[str, Any], device: str):
        from metrics.heldout_id import AdaFaceRecognizer, RetinaFaceAligner, UnifaceAdaFaceRecognizer
        import torch

        mcfg = cfg.get("metrics", {}).get("heldout_id", {})
        checkpoint = Path(str(mcfg.get("checkpoint", "")))
        if not checkpoint.is_file():
            raise RuntimeError(f"held-out AdaFace checkpoint is unavailable: {checkpoint}")
        torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
        providers = mcfg.get("recognizer_providers")
        recognizer_name = str(mcfg.get("recognizer", "")).lower()
        if recognizer_name.startswith("uniface"):
            self.recognizer = UnifaceAdaFaceRecognizer(
                cache_dir=mcfg.get("cache_dir", checkpoint.parent),
                checkpoint=checkpoint,
                device=torch_device,
                providers=[str(item) for item in providers] if providers else None,
            )
        else:
            self.recognizer = AdaFaceRecognizer(
                Path(str(mcfg.get("repo", ""))),
                checkpoint,
                torch_device,
                architecture=mcfg.get("architecture", "ir_101"),
            )
        self.detector = RetinaFaceAligner(
            model_root=mcfg.get("detector_model_root", cfg.get("cache", {}).get("arcface_model_root", "")),
            device_id=int(mcfg.get("detector_device_id", 0 if str(torch_device).startswith("cuda") else -1)),
            det_size=int(mcfg.get("det_size", 640)),
            providers=mcfg.get("detector_providers"),
        )
        self.mcfg = dict(mcfg)
        self.metadata = {
            "recognizer": mcfg.get("recognizer", "adaface_ir101"),
            "repo": str(mcfg.get("repo", "")),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "architecture": mcfg.get("architecture", "ir_101"),
            "detector": "RetinaFaceAligner",
            "detector_model_root": str(mcfg.get("detector_model_root", cfg.get("cache", {}).get("arcface_model_root", ""))),
            "det_size": int(mcfg.get("det_size", 640)),
            "preprocess": {
                "ref_expand": float(mcfg.get("ref_expand", 1.1)),
                "min_crop_px": int(mcfg.get("min_crop_px", 256)),
                "fallback": "resize_face_crop_rgb_when_no_face",
                "aligned_size": 112,
                "normalize": "l2",
            },
        }

    def embed(self, face_crop_path: Path) -> np.ndarray:
        from metrics.heldout_id import resize_face_crop_rgb

        try:
            aligned, _, _ = self.detector.align(
                face_crop_path,
                expand=float(self.mcfg.get("ref_expand", 1.1)),
                min_crop=int(self.mcfg.get("min_crop_px", 256)),
            )
        except RuntimeError as exc:
            if "found no face" not in str(exc):
                raise
            aligned = resize_face_crop_rgb(face_crop_path)
        embedding = np.asarray(self.recognizer.embed_aligned_rgb(aligned), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if norm <= 0.0 or not np.isfinite(norm):
            raise ValueError(f"invalid embedding norm for {face_crop_path}: {norm}")
        return embedding / norm


def face_crop_path(asset_root: Path, sample_id: str) -> Path:
    return find_one(asset_root / "derived" / "face_crops" / "human", sample_id).resolve()


def build_tar_calibration(
    *,
    dataset_root: Path,
    asset_root: Path,
    person_manifest_path: Path,
    split_dir: Path,
    config_path: Path,
    output_path: Path,
    embedding_npz_path: Path,
    embedder: FaceCropEmbedder,
    far: float = 1e-3,
) -> dict[str, Any]:
    person_manifest = load_person_manifest(person_manifest_path)
    records = load_split_records(split_name="val", split_dir=split_dir, person_manifest=person_manifest)
    failures: list[dict[str, str]] = []
    valid_records: list[SampleRecord] = []
    face_paths: list[Path] = []
    embeddings: list[np.ndarray] = []
    for record in records:
        try:
            crop_path = face_crop_path(asset_root, record.sample_id)
            if not crop_path.is_file():
                raise FileNotFoundError(crop_path)
            embedding = np.asarray(embedder.embed(crop_path), dtype=np.float32).reshape(-1)
            if embedding.size != 512:
                raise ValueError(f"expected 512-D embedding, got shape={embedding.shape}")
            if not np.all(np.isfinite(embedding)):
                raise ValueError("embedding contains NaN or Inf")
            valid_records.append(record)
            face_paths.append(crop_path)
            embeddings.append(embedding)
        except Exception as exc:  # noqa: BLE001
            failures.append({"id": record.sample_id, "error": str(exc)})
    if failures:
        blocked_payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact": "tar_calibration",
            "status": "blocked",
            "reason": "face_crop_embedding_failed",
            "dataset_root": str(dataset_root.resolve()),
            "asset_root": str(asset_root.resolve()),
            "person_manifest": str(person_manifest_path.resolve()),
            "split_dir": str(split_dir.resolve()),
            "config": str(config_path.resolve()),
            "input_sha256": input_hashes(person_manifest_path, split_dir, config_path),
            "split_authority": "splits_person_disjoint_final files; manifest clean_split is provenance only",
            "split": "val",
            "far_target": float(far),
            "val_id_count": len(records),
            "failure_count": len(failures),
            "failures": failures,
            "model": embedder.metadata,
        }
        blocked_payload["content_sha256"] = canonical_sha256(blocked_payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(blocked_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"TAR calibration failed for {len(failures)} val IDs; first={failures[0]}")
    embedding_matrix = np.stack(embeddings, axis=0).astype(np.float32)
    scores, excluded_same_cluster_pair_count = compute_impostor_scores_for_records(embedding_matrix, valid_records)
    threshold, empirical_far = calibration_threshold(scores, far)

    embedding_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        embedding_npz_path,
        ids=np.asarray([record.sample_id for record in valid_records]),
        person_cluster_ids=np.asarray([record.cluster_id for record in valid_records]),
        face_crop_paths=np.asarray([str(path) for path in face_paths]),
        embeddings=embedding_matrix,
    )
    embedding_npz_sha = sha256_file(embedding_npz_path)
    rows = [
        {
            "id": record.sample_id,
            "split": "val",
            "person_cluster_id": record.cluster_id,
            "face_crop_path": str(path),
            "face_crop_sha256": sha256_file(path),
            "manifest_clean_split": record.row.get("clean_split", ""),
            "source_dataset": record.row.get("source_dataset", ""),
            "source_key": record.row.get("source_key", ""),
        }
        for record, path in zip(valid_records, face_paths, strict=True)
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "tar_calibration",
        "dataset_root": str(dataset_root.resolve()),
        "asset_root": str(asset_root.resolve()),
        "person_manifest": str(person_manifest_path.resolve()),
        "split_dir": str(split_dir.resolve()),
        "config": str(config_path.resolve()),
        "input_sha256": input_hashes(person_manifest_path, split_dir, config_path),
        "split_authority": "splits_person_disjoint_final files; manifest clean_split is provenance only",
        "split": "val",
        "far_target": float(far),
        "threshold_method": "conservative_order_statistic_nextafter; accepts score >= threshold with empirical FAR <= target even under ties",
        "allowed_false_accept_count": int(np.floor(float(far) * scores.size)),
        "threshold": threshold,
        "empirical_far": empirical_far,
        "val_id_count": len(valid_records),
        "impostor_pair_count": int(scores.size),
        "excluded_same_cluster_pair_count": excluded_same_cluster_pair_count,
        "embedding_npz": str(embedding_npz_path.resolve()),
        "embedding_npz_sha256": embedding_npz_sha,
        "embedding_shape": list(embedding_matrix.shape),
        "embedding_dtype": str(embedding_matrix.dtype),
        "model": embedder.metadata,
        "failures": failures,
        "rows": rows,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--person-manifest", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fid-output", type=Path, default=None)
    parser.add_argument("--tar-output", type=Path, default=None)
    parser.add_argument("--embedding-npz", type=Path, default=None)
    parser.add_argument("--mode", choices=("all", "fid", "tar"), default="all")
    parser.add_argument("--fid-only", action="store_true", help="Alias for --mode fid")
    parser.add_argument("--tar-only", action="store_true", help="Alias for --mode tar")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--far", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    experiment_root = args.experiment_root.resolve()
    output_dir = (args.output_dir or experiment_root / "role_flow" / "metric_protocol").resolve()
    person_manifest = (args.person_manifest or dataset_root / "dataset_cleaning_report" / "person_disjoint_manifest.csv").resolve()
    split_dir = (args.split_dir or dataset_root / "splits_person_disjoint_final").resolve()
    asset_root = (args.asset_root or dataset_root).resolve()
    config_path = args.config.resolve() if args.config else None
    mode = "fid" if args.fid_only else "tar" if args.tar_only else args.mode
    results: dict[str, Any] = {}

    if mode in {"all", "fid"}:
        fid_output = (args.fid_output or output_dir / "fid_reference_manifest.json").resolve()
        fid_payload = build_fid_reference_manifest(
            dataset_root=dataset_root,
            asset_root=asset_root,
            person_manifest_path=person_manifest,
            split_dir=split_dir,
            config_path=config_path,
            output_path=fid_output,
        )
        results["fid_reference_manifest"] = {
            "path": str(fid_output),
            "count": fid_payload["count"],
            "content_sha256": fid_payload["content_sha256"],
        }

    if mode in {"all", "tar"}:
        if config_path is None or not config_path.is_file():
            raise FileNotFoundError(f"TAR calibration requires a config file: {config_path}")
        cfg = load_yaml(config_path)
        embedder = HeldoutAdaFaceEmbedder(cfg, args.device)
        tar_output = (args.tar_output or output_dir / "tar_calibration.json").resolve()
        embedding_npz = (args.embedding_npz or output_dir / "val_face_embeddings.npz").resolve()
        tar_payload = build_tar_calibration(
            dataset_root=dataset_root,
            asset_root=asset_root,
            person_manifest_path=person_manifest,
            split_dir=split_dir,
            config_path=config_path,
            output_path=tar_output,
            embedding_npz_path=embedding_npz,
            embedder=embedder,
            far=args.far,
        )
        results["tar_calibration"] = {
            "path": str(tar_output),
            "val_id_count": tar_payload["val_id_count"],
            "impostor_pair_count": tar_payload["impostor_pair_count"],
            "threshold": tar_payload["threshold"],
            "empirical_far": tar_payload["empirical_far"],
            "content_sha256": tar_payload["content_sha256"],
        }

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
