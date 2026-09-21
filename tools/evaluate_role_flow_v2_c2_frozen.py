#!/usr/bin/env python3
"""Evaluate role-flow A/B/C outputs under the frozen protocol v2 contract.

The protocol gate is intentionally stricter than the metric implementations:
all generated files, hashes, frozen TAR calibration, and the frozen full-test
FID reference population are verified before any model-backed metric can load.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA_VERSION = "m2h-role-flow-eval-v2"
EXPECTED_EVAL_COUNT = 580
EXPECTED_SPLIT_COUNTS = {"val": 180, "test": 400}
EXPECTED_FID_REFERENCE_COUNT = 1971
DEFAULT_GARMENT_LABELS = (3, 4, 5, 6, 7, 10)
BACKGROUND_EROSION_KERNEL = 11
MIN_GARMENT_AREA_FRACTION = 0.005
METRIC_NAMES = (
    "id_cosine",
    "tar_at_1e-3",
    "garment_dino",
    "garment_iou",
    "pose_pck",
    "bg_ssim",
    "bg_lpips",
)
PAIR_FIELDS = (
    "split",
    "mid",
    "jid",
    "seed",
    "generated_path",
    "human_path",
    "mannequin_path",
    "identity_face_detected",
    "identity_status",
    "identity_error",
    *METRIC_NAMES,
    "status",
    "error",
)


@dataclass(frozen=True)
class ProtocolContext:
    config_path: Path
    manifest_path: Path
    generated_dir: Path
    generation_provenance_path: Path
    checkpoint_path: Path
    asset_root: Path
    identity_threshold_path: Path
    fid_reference_manifest_path: Path
    output_dir: Path
    rows: list[dict[str, Any]]
    manifest_payload: dict[str, Any]
    generation_provenance: dict[str, Any]
    identity_threshold_payload: dict[str, Any]
    identity_threshold: float
    fid_reference_payload: dict[str, Any]
    fid_reference_count: int
    split_counts: dict[str, int]
    image_size: tuple[int, int]
    input_hashes: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_path(path: str | Path) -> str:
    value = Path(path)
    if value.is_file():
        return sha256_file(value)
    if not value.is_dir():
        raise FileNotFoundError(f"checkpoint path is missing: {value}")
    files: list[dict[str, Any]] = []
    for item in sorted(candidate for candidate in value.rglob("*") if candidate.is_file()):
        files.append({
            "path": item.relative_to(value).as_posix(),
            "size": item.stat().st_size,
            "sha256": sha256_file(item),
        })
    if not files:
        raise ValueError(f"checkpoint directory has no files: {value}")
    return canonical_sha256(files)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def expected_filename(row: dict[str, Any]) -> str:
    return f"{row['mid']}__id{row['jid']}__seed{int(row['seed'])}.png"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    from conditions import load_yaml

    return load_yaml(path)


def _config_image_size(config_path: Path) -> tuple[int, int]:
    cfg = _load_yaml(config_path)
    resolution = cfg.get("data", {}).get("resolution", {})
    if isinstance(resolution, int):
        return int(resolution), int(resolution)
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        return int(resolution[0]), int(resolution[1])
    return int(resolution.get("width", 768)), int(resolution.get("height", 1024))


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"manifest must contain rows: {path}")
    required = {"split", "mid", "jid", "seed", "human_path", "mannequin_path"}
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"manifest row {index} is not an object")
        missing = required - set(row)
        if missing:
            raise ValueError(f"manifest row {index} missing {sorted(missing)}")
        split = str(row["split"])
        if split not in {"val", "test"}:
            raise ValueError(f"manifest row {index} has unsupported split={split!r}")
        normalized.append({**row, "split": split, "mid": str(row["mid"]), "jid": str(row["jid"]), "seed": int(row["seed"])})
    return {**payload, "rows": normalized}


def _check_output_empty(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty for protocol v2: {output_dir}")


def _assert_hash(name: str, actual: str, expected: str | None) -> None:
    if expected and actual != expected:
        raise ValueError(f"{name} sha256 mismatch: expected {expected}, got {actual}")


def _verify_image(path: Path, expected_size: tuple[int, int]) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"unreadable generated image: {path}: {exc}") from exc
    if mode != "RGB" or (width, height) != expected_size:
        raise ValueError(f"generated image expected RGB {expected_size[0]}x{expected_size[1]}: {path} got {mode} {width}x{height}")
    return {"path": str(path), "filename": path.name, "sha256": sha256_file(path), "width": width, "height": height, "mode": mode}


def _selected_row_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"split": row["split"], "mid": row["mid"], "jid": row["jid"], "seed": int(row["seed"])} for row in rows]


def _validate_eval_manifest(rows: list[dict[str, Any]]) -> dict[str, int]:
    if len(rows) != EXPECTED_EVAL_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_EVAL_COUNT} eval rows, got {len(rows)}")
    counts = {split: sum(1 for row in rows if row["split"] == split) for split in ("val", "test")}
    expected_counts = dict(EXPECTED_SPLIT_COUNTS)
    if counts != expected_counts:
        raise ValueError(f"expected split counts {expected_counts}, got {counts}")
    keys = [(row["split"], row["mid"], row["jid"], int(row["seed"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("manifest contains duplicate split/mid/jid/seed keys")
    filenames = [expected_filename(row) for row in rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError("manifest rows produce duplicate generated filenames")
    return counts


def load_identity_threshold(path: str | Path) -> dict[str, Any]:
    payload = _read_json(Path(path))
    if payload.get("artifact") != "tar_calibration":
        raise ValueError(f"identity threshold must be a tar_calibration artifact: {path}")
    far = float(payload.get("far_target", 1e-3))
    if not math.isclose(far, 1e-3, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"identity threshold FAR must be 1e-3, got {far}")
    threshold = float(payload["threshold"])
    if not math.isfinite(threshold):
        raise ValueError(f"identity threshold is not finite: {threshold}")
    return payload


def _validate_fid_reference(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("artifact") != "fid_reference_manifest":
        raise ValueError(f"FID reference must be fid_reference_manifest: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"FID reference has no rows: {path}")
    count = int(payload.get("count", len(rows)))
    if count != EXPECTED_FID_REFERENCE_COUNT or len(rows) != EXPECTED_FID_REFERENCE_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_FID_REFERENCE_COUNT} FID reference rows, got count={count} rows={len(rows)}")
    paths: set[str] = set()
    hashes: set[str] = set()
    for index, row in enumerate(rows):
        human_path = Path(str(row.get("human_path", "")))
        if str(row.get("split", "test")) != "test":
            raise ValueError(f"FID reference row {index} is not test split")
        if not human_path.is_file():
            raise FileNotFoundError(f"FID reference image is missing: {human_path}")
        digest = sha256_file(human_path)
        _assert_hash("FID reference human", digest, row.get("human_sha256"))
        resolved = str(human_path.resolve())
        if resolved in paths:
            raise ValueError(f"duplicate FID reference path: {resolved}")
        if digest in hashes:
            raise ValueError(f"duplicate FID reference sha256: {digest}")
        paths.add(resolved)
        hashes.add(digest)
    return payload


def _validate_generation_provenance(
    *,
    provenance: dict[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    manifest_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    generated_dir: Path,
    image_size: tuple[int, int],
) -> dict[str, str]:
    if provenance.get("schema_version") != "m2h-role-flow-generation-v2":
        raise ValueError("generation provenance schema must be m2h-role-flow-generation-v2")
    if provenance.get("status") != "complete":
        raise ValueError(f"generation provenance status must be complete, got {provenance.get('status')!r}")
    if provenance.get("failures"):
        raise ValueError("generation provenance contains failures")
    inputs = provenance.get("inputs", {})
    input_hashes = {
        "config": sha256_file(config_path),
        "checkpoint": sha256_path(checkpoint_path),
        "manifest": sha256_file(manifest_path),
        "manifest_content": str(manifest_payload.get("content_sha256", "")),
    }
    _assert_hash("config", input_hashes["config"], inputs.get("config", {}).get("sha256"))
    _assert_hash("checkpoint", input_hashes["checkpoint"], inputs.get("checkpoint", {}).get("sha256"))
    _assert_hash("manifest", input_hashes["manifest"], inputs.get("manifest", {}).get("sha256"))
    provenance_manifest_content = inputs.get("manifest", {}).get("content_sha256")
    if provenance_manifest_content and manifest_payload.get("content_sha256") and provenance_manifest_content != manifest_payload["content_sha256"]:
        raise ValueError("manifest content sha256 mismatch")

    expected = [expected_filename(row) for row in rows]
    selection = provenance.get("selection", {})
    if int(selection.get("selected_row_count", -1)) != len(rows):
        raise ValueError("generation selected_row_count mismatch")
    if list(selection.get("expected_filenames", [])) != expected:
        raise ValueError("generation expected_filenames mismatch")
    if list(selection.get("row_keys", [])) != _selected_row_keys(rows):
        raise ValueError("generation row_keys mismatch")

    actual_pngs = sorted(path.name for path in generated_dir.glob("*.png"))
    if actual_pngs != sorted(expected):
        missing = sorted(set(expected) - set(actual_pngs))
        extra = sorted(set(actual_pngs) - set(expected))
        raise ValueError(f"generated directory must contain exact frozen PNG set; missing={missing[:3]} extra={extra[:3]}")

    image_records = {str(item.get("filename")): item for item in provenance.get("images", []) if isinstance(item, dict)}
    if set(image_records) != set(expected):
        raise ValueError("generation image metadata set mismatch")
    for filename in expected:
        image_path = generated_dir / filename
        record = _verify_image(image_path, image_size)
        stored = image_records[filename]
        if stored.get("readable") is not True:
            raise ValueError(f"generated provenance marks unreadable image: {filename}")
        _assert_hash("generated image", record["sha256"], stored.get("sha256"))
        if int(stored.get("width", -1)) != image_size[0] or int(stored.get("height", -1)) != image_size[1] or stored.get("mode") != "RGB":
            raise ValueError(f"generated image metadata expected RGB {image_size[0]}x{image_size[1]}: {filename}")
    return input_hashes


def preflight_protocol(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    generated_dir: str | Path,
    generation_provenance_path: str | Path,
    checkpoint_path: str | Path,
    asset_root: str | Path,
    identity_threshold_path: str | Path,
    fid_reference_manifest_path: str | Path,
    output_dir: str | Path,
) -> ProtocolContext:
    config = Path(config_path).resolve()
    manifest = Path(manifest_path).resolve()
    generated = Path(generated_dir).resolve()
    provenance_path = Path(generation_provenance_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    asset = Path(asset_root).resolve()
    threshold = Path(identity_threshold_path).resolve()
    fid_reference = Path(fid_reference_manifest_path).resolve()
    out = Path(output_dir).resolve()

    _check_output_empty(out)
    for required in (config, manifest, provenance_path, threshold, fid_reference):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not generated.is_dir():
        raise FileNotFoundError(generated)
    if not asset.is_dir():
        raise FileNotFoundError(asset)

    manifest_payload = _load_manifest(manifest)
    rows = list(manifest_payload["rows"])
    split_counts = _validate_eval_manifest(rows)
    image_size = _config_image_size(config)
    generation_provenance = _read_json(provenance_path)
    input_hashes = _validate_generation_provenance(
        provenance=generation_provenance,
        config_path=config,
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        manifest_payload=manifest_payload,
        rows=rows,
        generated_dir=generated,
        image_size=image_size,
    )
    threshold_payload = load_identity_threshold(threshold)
    fid_payload = _validate_fid_reference(fid_reference)
    return ProtocolContext(
        config_path=config,
        manifest_path=manifest,
        generated_dir=generated,
        generation_provenance_path=provenance_path,
        checkpoint_path=checkpoint,
        asset_root=asset,
        identity_threshold_path=threshold,
        fid_reference_manifest_path=fid_reference,
        output_dir=out,
        rows=rows,
        manifest_payload=manifest_payload,
        generation_provenance=generation_provenance,
        identity_threshold_payload=threshold_payload,
        identity_threshold=float(threshold_payload["threshold"]),
        fid_reference_payload=fid_payload,
        fid_reference_count=len(fid_payload["rows"]),
        split_counts=split_counts,
        image_size=image_size,
        input_hashes=input_hashes,
    )


def _find_image(folder: Path, sample_id: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = folder / f"{sample_id}{suffix}"
        if candidate.is_file():
            return candidate
    matches = sorted(folder.glob(f"{sample_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"missing image id={sample_id} under {folder}")


def _resize_nearest(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8))
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def garment_mask_from_labels(labels: np.ndarray, garment_labels: Iterable[int] = DEFAULT_GARMENT_LABELS) -> np.ndarray:
    return np.isin(np.asarray(labels, dtype=np.uint8), np.asarray(tuple(garment_labels), dtype=np.uint8))


def load_source_garment_mask(asset_root: str | Path, mid: str, size: tuple[int, int] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    label_path = _find_image(Path(asset_root) / "human_parsing" / "fashn" / "masks" / "mannequin", str(mid))
    with Image.open(label_path) as image:
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        labels = np.asarray(image.convert("L"), dtype=np.uint8)
    mask = garment_mask_from_labels(labels)
    return mask, {
        "source_role": "mannequin",
        "source_path": str(label_path),
        "source_sha256": sha256_file(label_path),
        "generator": "FASHN SegFormer-B4 parser",
        "garment_labels": list(DEFAULT_GARMENT_LABELS),
    }


def load_label_map(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        return np.asarray(image.convert("L"), dtype=np.uint8)


def generated_label_path(out_dir: str | Path, generated_path: str | Path) -> Path:
    from metrics_v2.parsing import parsing_path

    return parsing_path(out_dir, generated_path)


def garment_iou_from_masks(generated_mask: np.ndarray, source_mask: np.ndarray, min_area_fraction: float = MIN_GARMENT_AREA_FRACTION) -> dict[str, Any]:
    generated_bool = np.asarray(generated_mask, dtype=bool)
    source_bool = np.asarray(source_mask, dtype=bool)
    if generated_bool.shape != source_bool.shape:
        raise ValueError(f"mask shape mismatch: generated={generated_bool.shape} source={source_bool.shape}")
    if float(generated_bool.mean()) < float(min_area_fraction):
        return {"garment_iou": None, "status": "generated_garment_invalid"}
    union = np.logical_or(generated_bool, source_bool)
    if not bool(union.any()):
        return {"garment_iou": None, "status": "empty_union"}
    intersection = np.logical_and(generated_bool, source_bool)
    return {"garment_iou": float(intersection.sum() / union.sum()), "status": "ok"}


def common_background_mask(source_labels: np.ndarray, generated_labels: np.ndarray, erosion_kernel: int = BACKGROUND_EROSION_KERNEL) -> np.ndarray:
    source = np.asarray(source_labels, dtype=np.uint8)
    generated = np.asarray(generated_labels, dtype=np.uint8)
    if source.shape != generated.shape:
        raise ValueError(f"label shape mismatch: source={source.shape} generated={generated.shape}")
    background = np.logical_not((source != 0) | (generated != 0)).astype(np.uint8)
    kernel = max(1, int(erosion_kernel))
    if kernel % 2 == 0:
        kernel += 1
    try:
        import cv2

        eroded = cv2.erode(background, np.ones((kernel, kernel), dtype=np.uint8), iterations=1)
        return eroded.astype(bool)
    except Exception:  # noqa: BLE001
        pad = kernel // 2
        padded = np.pad(background, pad, mode="constant", constant_values=0)
        out = np.zeros_like(background, dtype=bool)
        for y in range(background.shape[0]):
            for x in range(background.shape[1]):
                out[y, x] = bool(np.all(padded[y : y + kernel, x : x + kernel]))
        return out


def _masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    if not bool(mask.any()):
        raise ValueError("empty background mask")
    left = np.asarray(a, dtype=np.uint8).copy()
    right = np.asarray(b, dtype=np.uint8).copy()
    for array in (left, right):
        if bool(mask.any()):
            fill = np.mean(array[mask], axis=0)
        else:
            fill = np.asarray([127, 127, 127], dtype=np.float32)
        array[~mask] = np.asarray(fill, dtype=np.uint8)
    _score, score_map = structural_similarity(
        left,
        right,
        channel_axis=2,
        data_range=255,
        full=True,
    )
    score_map = np.asarray(score_map, dtype=np.float32)
    if score_map.ndim == 3:
        score_map = score_map.mean(axis=2)
    if score_map.shape != mask.shape:
        raise ValueError(f"SSIM map shape mismatch: map={score_map.shape} mask={mask.shape}")
    return float(score_map[mask].mean())


def _masked_spatial_lpips(a: np.ndarray, b: np.ndarray, mask: np.ndarray, model: Any, device: str) -> float:
    import torch

    if not bool(mask.any()):
        raise ValueError("empty background mask")
    ta = torch.from_numpy(a.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
    tb = torch.from_numpy(b.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        value = model(ta, tb)
    spatial = value.detach().float()
    if spatial.ndim == 0 or spatial.numel() == 1:
        return float(spatial.item())
    if spatial.ndim == 4:
        spatial = spatial[0, 0]
    elif spatial.ndim == 3:
        spatial = spatial[0]
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(spatial.device)
    if tuple(spatial.shape[-2:]) != mask.shape:
        spatial = torch.nn.functional.interpolate(spatial.view(1, 1, *spatial.shape[-2:]), size=mask.shape, mode="bilinear", align_corners=False)[0, 0]
    return float((spatial * mask_tensor).sum().item() / max(float(mask_tensor.sum().item()), 1.0))


def _normalise_cfg(config_path: Path, asset_root: Path) -> dict[str, Any]:
    from tools.evaluate_role_flow import _normalise_cfg as normalise

    cfg = normalise(_load_yaml(config_path))
    cfg["data"] = {**cfg.get("data", {}), "root": str(asset_root), "asset_root": str(asset_root)}
    return cfg


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["mid"]), str(row["jid"]), int(row["seed"])


def _run_identity_metrics(ctx: ProtocolContext, cfg: dict[str, Any], device: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    from tools.evaluate_role_flow import _embed_rows

    # ``_embed_rows`` consumes the normalized row field ``generated_path``;
    # protocol preflight rows intentionally contain only source paths, so add
    # the frozen generated path here without mutating the protocol manifest.
    embed_rows = [
        dict(row, generated_path=str(ctx.generated_dir / expected_filename(row)))
        for row in ctx.rows
    ]
    generated, refs, failures = _embed_rows(cfg, embed_rows, device)
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in ctx.rows:
        key = _row_key(row)
        generated_path = str(ctx.generated_dir / expected_filename(row))
        if key in failures:
            error = failures.get(key, "identity embedding missing")
            if "RetinaFace found no face" in error:
                output[key] = {
                    "generated_path": generated_path,
                    "id_cosine": -1.0,
                    "tar_at_1e-3": 0.0,
                    "identity_face_detected": False,
                    "identity_status": "no_face",
                    "identity_error": error,
                    "status": "no_face",
                    "error": error,
                }
                continue
            output[key] = {
                "generated_path": generated_path,
                "identity_face_detected": False,
                "identity_status": "failed",
                "identity_error": error,
                "status": "failed",
                "error": error,
            }
            continue
        if key not in generated or row["jid"] not in refs:
            error = "identity embedding missing"
            output[key] = {
                "generated_path": generated_path,
                "identity_face_detected": False,
                "identity_status": "failed",
                "identity_error": error,
                "status": "failed",
                "error": error,
            }
            continue
        score = float(np.dot(generated[key], refs[row["jid"]]))
        output[key] = {
            "generated_path": generated_path,
            "id_cosine": score,
            "tar_at_1e-3": float(score >= ctx.identity_threshold),
            "identity_face_detected": True,
            "identity_status": "ok",
            "identity_error": "",
            "status": "ok",
            "error": "",
        }
    return output


def _run_garment_metrics(ctx: ProtocolContext, cfg: dict[str, Any], device: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    from metrics_v2.common import read_rgb
    from metrics_v2.features import RegionFeatureExtractor, cosine
    from metrics_v2.parsing import build_generated_parsing

    parse_rows = [{"mid": row["mid"], "jid": row["jid"], "seed": row["seed"], "path": ctx.generated_dir / expected_filename(row)} for row in ctx.rows]
    parsing_report = build_generated_parsing(cfg, parse_rows, ctx.output_dir, device)
    if parsing_report.get("failed"):
        raise RuntimeError(f"generated FASHN parsing failed: {parsing_report['failed'][:3]}")
    extractor = RegionFeatureExtractor(cfg, device)
    source_features: dict[str, np.ndarray] = {}
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in ctx.rows:
        key = _row_key(row)
        generated_path = str(ctx.generated_dir / expected_filename(row))
        try:
            source_mask, source_record = load_source_garment_mask(ctx.asset_root, row["mid"], ctx.image_size)
            labels = load_label_map(generated_label_path(ctx.output_dir, ctx.generated_dir / expected_filename(row)), ctx.image_size)
            generated_mask = garment_mask_from_labels(labels)
            iou = garment_iou_from_masks(generated_mask, source_mask)
            if iou["status"] != "ok":
                output[key] = {
                    **iou,
                    "generated_path": generated_path,
                    "status": iou["status"],
                    "source_mask": source_mask,
                    "generated_mask": generated_mask,
                    "source_record": source_record,
                }
                continue
            mannequin = read_rgb(row["mannequin_path"], ctx.image_size)
            generated = read_rgb(ctx.generated_dir / expected_filename(row), ctx.image_size)
            if row["mid"] not in source_features:
                source_features[row["mid"]] = extractor.dino_feature(mannequin, source_mask.astype(np.uint8))
            generated_feature = extractor.dino_feature(generated, generated_mask.astype(np.uint8))
            output[key] = {
                "generated_path": generated_path,
                "garment_dino": cosine(generated_feature, source_features[row["mid"]]),
                "garment_iou": iou["garment_iou"],
                "source_mask": source_mask,
                "generated_mask": generated_mask,
                "source_record": source_record,
                "status": "ok",
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            output[key] = {"generated_path": generated_path, "status": "failed", "error": str(exc)}
    return output


def _run_background_metrics(ctx: ProtocolContext, cfg: dict[str, Any], device: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    import lpips

    model = lpips.LPIPS(net="alex", spatial=True).eval().requires_grad_(False).to(device)
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in ctx.rows:
        key = _row_key(row)
        generated_path = str(ctx.generated_dir / expected_filename(row))
        try:
            source_label_path = _find_image(ctx.asset_root / "human_parsing" / "fashn" / "masks" / "mannequin", row["mid"])
            generated_parse_path = generated_label_path(ctx.output_dir, ctx.generated_dir / expected_filename(row))
            source_labels = load_label_map(source_label_path, ctx.image_size)
            generated_labels = load_label_map(generated_parse_path, ctx.image_size)
            background = common_background_mask(source_labels, generated_labels)
            if float(background.mean()) <= 0.01:
                raise ValueError("common background fraction <= 0.01")
            generated = np.asarray(Image.open(ctx.generated_dir / expected_filename(row)).convert("RGB"), dtype=np.uint8)
            mannequin = np.asarray(Image.open(row["mannequin_path"]).convert("RGB"), dtype=np.uint8)
            if (generated.shape[1], generated.shape[0]) != ctx.image_size:
                generated = np.asarray(Image.fromarray(generated).resize(ctx.image_size, Image.Resampling.BICUBIC), dtype=np.uint8)
            if (mannequin.shape[1], mannequin.shape[0]) != ctx.image_size:
                mannequin = np.asarray(Image.fromarray(mannequin).resize(ctx.image_size, Image.Resampling.BICUBIC), dtype=np.uint8)
            output[key] = {
                "generated_path": generated_path,
                "bg_ssim": _masked_ssim(generated, mannequin, background),
                "bg_lpips": _masked_spatial_lpips(generated, mannequin, background, model, device),
                "background_fraction": float(background.mean()),
                "status": "ok",
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            output[key] = {"generated_path": generated_path, "status": "failed", "error": str(exc)}
    del model
    return output


def _run_pose_metrics(ctx: ProtocolContext, cfg: dict[str, Any], device: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    from tools.evaluate_role_flow import _pose_pck

    rows = [dict(row, generated_path=str(ctx.generated_dir / expected_filename(row))) for row in ctx.rows]
    (ctx.output_dir / "pose").mkdir(parents=True, exist_ok=True)
    output = _pose_pck(cfg, rows, ctx.output_dir, device)
    for row in ctx.rows:
        key = _row_key(row)
        output.setdefault(key, {})
        output[key].setdefault("generated_path", str(ctx.generated_dir / expected_filename(row)))
    return output


def _run_fid(ctx: ProtocolContext, cfg: dict[str, Any], device: str) -> float:
    import torch_fidelity

    real_dir = ctx.output_dir / "fid_real_refs"
    generated_dir = ctx.output_dir / "fid_generated"
    real_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(ctx.fid_reference_payload["rows"]):
        source = Path(row["human_path"])
        target = real_dir / f"{index:06d}{source.suffix.lower()}"
        if not target.exists():
            target.symlink_to(source.resolve())
    for index, row in enumerate(ctx.rows):
        source = ctx.generated_dir / expected_filename(row)
        target = generated_dir / f"{index:06d}{source.suffix.lower()}"
        if not target.exists():
            target.symlink_to(source.resolve())
    metrics = torch_fidelity.calculate_metrics(
        input1=str(generated_dir),
        input2=str(real_dir),
        cuda=str(device).startswith("cuda"),
        isc=False,
        fid=True,
        kid=False,
        feature_layer_fid="2048",
        rng_seed=20260817,
        verbose=True,
    )
    return float(metrics["frechet_inception_distance"])


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _summary(values: Iterable[Any]) -> dict[str, Any]:
    clean = [number for value in values if (number := _finite(value)) is not None]
    return {
        "mean": float(np.mean(clean)) if clean else None,
        "median": float(np.median(clean)) if clean else None,
        "count": len(clean),
        "failed": 0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    fields = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _metric_failures(item: dict[str, Any], *, metric_name: str, required_fields: tuple[str, ...], generated_path: str) -> list[str]:
    if not isinstance(item, dict):
        return [f"{metric_name}: missing metric row"]
    failures: list[str] = []
    if item.get("generated_path") != generated_path:
        failures.append(f"{metric_name}: generated_path missing or mismatched")
    missing = [field for field in required_fields if field not in item]
    if missing:
        failures.append(f"{metric_name}: missing fields {','.join(missing)}")
    status = str(item.get("status", "ok"))
    if status in {"", "ok"}:
        return failures
    if metric_name == "identity" and status == "no_face":
        return failures
    failures.append(f"{metric_name}: {str(item.get('error') or status)}")
    return failures


def evaluate_v2(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    generated_dir: str | Path,
    generation_provenance_path: str | Path,
    checkpoint_path: str | Path,
    asset_root: str | Path,
    identity_threshold_path: str | Path,
    fid_reference_manifest_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    strict: bool = True,
) -> dict[str, Any]:
    ctx = preflight_protocol(
        config_path=config_path,
        manifest_path=manifest_path,
        generated_dir=generated_dir,
        generation_provenance_path=generation_provenance_path,
        checkpoint_path=checkpoint_path,
        asset_root=asset_root,
        identity_threshold_path=identity_threshold_path,
        fid_reference_manifest_path=fid_reference_manifest_path,
        output_dir=output_dir,
    )
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _normalise_cfg(ctx.config_path, ctx.asset_root)
    metric_maps = [
        ("identity", ("id_cosine", "tar_at_1e-3"), _run_identity_metrics(ctx, cfg, device)),
        ("garment", ("garment_dino", "garment_iou"), _run_garment_metrics(ctx, cfg, device)),
        ("background", ("bg_ssim", "bg_lpips"), _run_background_metrics(ctx, cfg, device)),
        ("pose", ("pose_pck",), _run_pose_metrics(ctx, cfg, device)),
    ]
    pair_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in ctx.rows:
        key = _row_key(row)
        record = {
            "split": row["split"],
            "mid": row["mid"],
            "jid": row["jid"],
            "seed": int(row["seed"]),
            "generated_path": str(ctx.generated_dir / expected_filename(row)),
            "human_path": row["human_path"],
            "mannequin_path": row["mannequin_path"],
        }
        errors: list[str] = []
        generated_path = str(ctx.generated_dir / expected_filename(row))
        for metric_name, required_fields, metric in metric_maps:
            item = metric.get(key)
            if metric_name == "identity" and isinstance(item, dict):
                for name in ("identity_face_detected", "identity_status", "identity_error"):
                    if name in item:
                        record[name] = item[name]
            for name in METRIC_NAMES:
                if isinstance(item, dict) and name in item:
                    record[name] = item[name]
            errors.extend(_metric_failures(item, metric_name=metric_name, required_fields=required_fields, generated_path=generated_path))
        record["status"] = "failed" if errors else "ok"
        record["error"] = " | ".join(errors)
        if errors:
            failures.append(record)
        pair_rows.append(record)
    if strict and failures:
        _write_csv(ctx.output_dir / "per_pair_metrics.csv", pair_rows, PAIR_FIELDS)
        _write_csv(ctx.output_dir / "failures.csv", failures, PAIR_FIELDS)
        raise RuntimeError(f"metric failures under strict protocol v2: {len(failures)}")

    fid_value = _run_fid(ctx, cfg, device)
    set_metrics = {"fid": {"value": fid_value, "reference_count": ctx.fid_reference_count, "generated_count": len(ctx.rows)}}
    metric_summary = {}
    for name in METRIC_NAMES:
        metric_summary[name] = _summary([row.get(name) for row in pair_rows])
        metric_summary[name]["failed"] = sum(_finite(row.get(name)) is None for row in pair_rows)
    identity_status_counts: dict[str, int] = {}
    for row in pair_rows:
        value = str(row.get("identity_status") or "missing")
        identity_status_counts[value] = identity_status_counts.get(value, 0) + 1

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if not failures else "completed_with_failures",
        "created_at": utc_now(),
        "sample_counts": {"evaluated": len(pair_rows), "failed": len(failures), "ok": len(pair_rows) - len(failures), **ctx.split_counts},
        "metrics": metric_summary,
        "set_metrics": set_metrics,
        "identity_detection": {
            "face_detected": identity_status_counts.get("ok", 0),
            "no_face": identity_status_counts.get("no_face", 0),
            "failed": identity_status_counts.get("failed", 0),
            "status_counts": identity_status_counts,
            "no_face_policy": "id_cosine=-1.0; tar_at_1e-3=0.0",
        },
        "protocol": {
            "eval_count": EXPECTED_EVAL_COUNT,
            "split_counts": ctx.split_counts,
            "image_size": {"width": ctx.image_size[0], "height": ctx.image_size[1]},
            "source_garment": "mannequin FASHN labels",
            "generated_garment_fallback": "disabled",
            "background": {"foreground": "full FASHN non-background union", "erosion_kernel": BACKGROUND_EROSION_KERNEL, "lpips": "spatial masked"},
            "tar": {"far_target": 1e-3, "threshold": ctx.identity_threshold, "calibration_content_sha256": ctx.identity_threshold_payload.get("content_sha256")},
            "fid": {"reference_count": ctx.fid_reference_count, "reference_content_sha256": ctx.fid_reference_payload.get("content_sha256"), "scope": "set_only"},
        },
        "inputs": {
            "config": {"path": str(ctx.config_path), "sha256": ctx.input_hashes["config"]},
            "checkpoint": {"path": str(ctx.checkpoint_path), "sha256": ctx.input_hashes["checkpoint"]},
            "manifest": {
                "path": str(ctx.manifest_path),
                "sha256": ctx.input_hashes["manifest"],
                "content_sha256": ctx.input_hashes.get("manifest_content"),
            },
            "generation_provenance": {"path": str(ctx.generation_provenance_path), "sha256": sha256_file(ctx.generation_provenance_path)},
            "identity_threshold": {"path": str(ctx.identity_threshold_path), "sha256": sha256_file(ctx.identity_threshold_path)},
            "fid_reference_manifest": {"path": str(ctx.fid_reference_manifest_path), "sha256": sha256_file(ctx.fid_reference_manifest_path)},
        },
        "outputs": {
            "per_pair_metrics": str(ctx.output_dir / "per_pair_metrics.csv"),
            "failures": str(ctx.output_dir / "failures.csv"),
            "summary": str(ctx.output_dir / "summary.json"),
            "set_metrics": str(ctx.output_dir / "set_metrics.json"),
        },
    }
    _write_csv(ctx.output_dir / "per_pair_metrics.csv", pair_rows, PAIR_FIELDS)
    _write_csv(ctx.output_dir / "failures.csv", failures, PAIR_FIELDS)
    (ctx.output_dir / "set_metrics.json").write_text(json.dumps(set_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ctx.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "status": summary["status"],
        "created_at": summary["created_at"],
        "input_sha256": summary["inputs"],
        "output_sha256": {
            "per_pair_metrics": sha256_file(ctx.output_dir / "per_pair_metrics.csv"),
            "failures": sha256_file(ctx.output_dir / "failures.csv"),
            "summary": sha256_file(ctx.output_dir / "summary.json"),
            "set_metrics": sha256_file(ctx.output_dir / "set_metrics.json"),
        },
    }
    provenance["content_sha256"] = canonical_sha256(provenance)
    (ctx.output_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ctx.output_dir / "READY").write_text("ready\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="config_path", type=Path, required=True)
    parser.add_argument("--manifest", dest="manifest_path", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--generation-provenance", dest="generation_provenance_path", type=Path, default=None)
    parser.add_argument("--checkpoint", dest="checkpoint_path", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--identity-threshold", dest="identity_threshold_path", type=Path, required=True)
    parser.add_argument("--fid-reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generation_provenance = args.generation_provenance_path or args.generated_dir / "generation_provenance.json"
    summary = evaluate_v2(
        config_path=args.config_path,
        manifest_path=args.manifest_path,
        generated_dir=args.generated_dir,
        generation_provenance_path=generation_provenance,
        checkpoint_path=args.checkpoint_path,
        asset_root=args.asset_root,
        identity_threshold_path=args.identity_threshold_path,
        fid_reference_manifest_path=args.fid_reference_manifest,
        output_dir=args.output_dir,
        device=args.device,
        strict=not args.no_strict,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
