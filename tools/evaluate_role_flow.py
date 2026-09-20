"""Evaluate role-selective FLUX runs with one frozen, explicit pair manifest.

The runner intentionally fails closed when an official metric dependency or
weight is unavailable.  It writes one row per (mid, jid, seed), a machine
readable summary, and a provenance record containing all input hashes.

Example:
    python tools/evaluate_role_flow.py \
      --config configs/role_selective/A_timestep_routed.yaml \
      --manifest /path/to/role_eval_manifest.json \
      --generated-dir /path/to/A/generated \
      --output-dir /path/to/A/metrics \
      --split test --calibration-split val --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


METRIC_NAMES = (
    "id_cosine",
    "tar_at_1e-3",
    "garment_dino",
    "garment_iou",
    "pose_pck",
    "bg_ssim",
    "bg_lpips",
    "fid",
)
ROW_FIELDS = (
    "mid",
    "jid",
    "seed",
    "split",
    "generated_path",
    "mannequin_path",
    "human_path",
    *METRIC_NAMES,
    "status",
    "error",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash a file or directory without reading metadata-dependent timestamps."""
    value = Path(path)
    if value.is_file():
        return sha256_file(value)
    if not value.is_dir():
        return "missing"
    digest = hashlib.sha256()
    for child in sorted(item for item in value.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(value)).encode("utf-8"))
        digest.update(sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[Any]) -> float | None:
    clean = [number for value in values if (number := _finite(value)) is not None]
    return float(np.mean(clean)) if clean else None


def _summary(values: Iterable[Any], failed: int = 0) -> dict[str, Any]:
    clean = [number for value in values if (number := _finite(value)) is not None]
    return {
        "mean": float(np.mean(clean)) if clean else None,
        "median": float(np.median(clean)) if clean else None,
        "count": len(clean),
        "failed": int(failed),
    }


def _read_manifest_payload(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if value.suffix.lower() == ".csv":
        with value.open("r", encoding="utf-8", newline="") as handle:
            return {"rows": [dict(row) for row in csv.DictReader(handle)], "provenance": {}}
    payload = json.loads(value.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"rows": payload, "provenance": {}}
    if isinstance(payload, dict):
        rows = payload.get("rows", payload.get("pairs"))
        if isinstance(rows, list):
            return {**payload, "rows": rows}
    raise ValueError(f"evaluation manifest must contain a list or rows field: {value}")


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    return [dict(row) for row in _read_manifest_payload(path)["rows"]]


def resolve_asset_root(manifest: str | Path, dataset_root: str | Path, asset_root: str | Path | None = None) -> Path:
    if asset_root is not None and str(asset_root).strip():
        return Path(asset_root).expanduser()
    provenance = _read_manifest_payload(manifest).get("provenance", {})
    if isinstance(provenance, dict):
        source_root = str(provenance.get("read_only_source_root", "")).strip()
        if source_root:
            return Path(source_root).expanduser()
    return Path(dataset_root).expanduser()


def _asset_root(cfg: dict[str, Any]) -> Path:
    data = cfg.get("data", {})
    return Path(data.get("asset_root") or data["root"]).expanduser()


def _cfg_with_data_root(cfg: dict[str, Any], root: str | Path) -> dict[str, Any]:
    updated = dict(cfg)
    updated["data"] = {**cfg.get("data", {}), "root": str(root)}
    return updated


def _face_crop_path(cfg: dict[str, Any], sample_id: str) -> Path:
    return _find_image(_asset_root(cfg) / "derived/face_crops/human", sample_id)


def _dwpose_target_path(cfg: dict[str, Any], mid: str) -> Path:
    return _asset_root(cfg) / "dwpose/keypoints/mannequin" / f"{mid}.npz"


def _normalise_row(row: dict[str, Any], generated_dir: Path, dataset_root: Path) -> dict[str, Any]:
    mid = str(row.get("mid", row.get("mannequin_id", "")))
    jid = str(row.get("jid", row.get("identity_id", "")))
    if not mid or not jid:
        raise ValueError(f"manifest row lacks mid/jid: {row}")
    seed = int(row.get("seed", 0))
    generated_value = str(row.get("generated_path", "")).strip()
    generated = Path(generated_value) if generated_value else generated_dir / f"{mid}__id{jid}__seed{seed}.png"
    if not generated.is_absolute():
        generated = generated_dir / generated
    mannequin_value = str(row.get("mannequin_path", "")).strip()
    mannequin = Path(mannequin_value) if mannequin_value else _find_image(dataset_root / "images/mannequin", mid)
    if not mannequin.is_absolute():
        mannequin = dataset_root / mannequin
    human_value = str(row.get("human_path", "")).strip()
    human = Path(human_value) if human_value else _find_image(dataset_root / "images/human", jid)
    if not human.is_absolute():
        human = dataset_root / human
    return {
        "mid": mid,
        "jid": jid,
        "seed": seed,
        "split": str(row.get("split", "")),
        "generated_path": str(generated),
        "mannequin_path": str(mannequin),
        "human_path": str(human),
    }


def _find_image(folder: Path, sample_id: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = folder / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(folder.glob(f"{sample_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"missing image id={sample_id} under {folder}")


def load_eval_rows(
    manifest: str | Path,
    generated_dir: str | Path,
    dataset_root: str | Path,
    split: str,
) -> list[dict[str, Any]]:
    rows = [
        _normalise_row(row, Path(generated_dir), Path(dataset_root))
        for row in _read_rows(manifest)
        if not split or str(row.get("split", split)) == split
    ]
    rows.sort(key=lambda row: (row["mid"], row["jid"], row["seed"]))
    keys = [(row["mid"], row["jid"], row["seed"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"manifest contains duplicate pair keys in split={split}")
    if not rows:
        raise ValueError(f"evaluation manifest has no rows for split={split}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _load_cfg(path: Path) -> dict[str, Any]:
    from conditions import load_yaml

    return load_yaml(path)


def _normalise_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Adapt role-selective configs to the existing metrics_v2 interfaces."""
    if "metrics_v2" in cfg:
        return cfg
    metrics = cfg.get("metrics", {})
    garment = metrics.get("garment", {})
    return {
        **cfg,
        "metrics_v2": {
            "version": "role-flow-1",
            "dino": {
                "repo_root": garment.get("dino_repo_root", "/data/muxiangyu/pythonPrograms/GSVTON"),
                "checkpoint": garment.get("dino_checkpoint", ""),
                "image_size": garment.get("dino_image_size", 518),
                "mask_out_value": garment.get("mask_out_value", 0.5),
            },
            "parsing": {
                "model_dir": "/data/muxiangyu/pythonPrograms/M2HImage/models/hf/fashn-ai/fashn-human-parser",
                "labels_json": "human_parsing/fashn/metadata/labels.json",
                "input_width": 384,
                "input_height": 576,
                "batch_size": 4,
                "garment_labels": [3, 4, 5, 6, 7, 10],
                "min_garment_area_fraction": 0.005,
            },
            "pose": {
                "helper_python": "/home/muxiangyu/miniconda3/envs/StableAnimator/bin/python",
                "repo_root": "/data/muxiangyu/pythonPrograms/StableAnimator/DWPose",
                "detector_checkpoint": "/data/muxiangyu/pythonPrograms/StableAnimator/checkpoints/DWPose/yolox_l.onnx",
                "pose_checkpoint": "/data/muxiangyu/pythonPrograms/StableAnimator/checkpoints/DWPose/dw-ll_ucoco_384.onnx",
                "provider": "CUDAExecutionProvider",
                "score_threshold": 0.3,
                "body_indices": list(range(1, 14)),
                "head_indices": [0, 14, 15, 16, 17],
            },
            "identity": {"expected_adaface_hash": "unused-by-role-runner"},
            "distribution": {"rng_seed": 20260817, "feature_layer_fid": 2048},
        },
    }


def _load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return (np.asarray(image, dtype=np.uint8) > 127).astype(np.uint8)


def _source_garment_mask(cfg: dict[str, Any], mid: str, size: tuple[int, int]) -> np.ndarray:
    from metrics_v2.parsing import source_garment_mask

    return source_garment_mask(_cfg_with_data_root(cfg, _asset_root(cfg)), mid, size)


def _run_garment_and_parsing(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    out_dir: Path,
    device: str,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    from metrics_v2.common import image_size, read_rgb
    from metrics_v2.features import RegionFeatureExtractor, cosine
    from metrics_v2.parsing import build_generated_parsing, load_generated_masks

    build_generated_parsing(cfg, [{"mid": row["mid"], "jid": row["jid"], "seed": row["seed"], "path": Path(row["generated_path"])} for row in rows], out_dir, device)
    size = image_size(cfg)
    extractor = RegionFeatureExtractor(cfg, device)
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    source_features: dict[str, np.ndarray] = {}
    for row in rows:
        key = (row["mid"], row["jid"], row["seed"])
        try:
            generated = read_rgb(row["generated_path"], size)
            mannequin = read_rgb(row["mannequin_path"], size)
            source_cfg = _cfg_with_data_root(cfg, _asset_root(cfg))
            generated_mask, _, _ = load_generated_masks(source_cfg, out_dir, {"mid": row["mid"], "path": Path(row["generated_path"])})
            source_mask = _source_garment_mask(cfg, row["mid"], size)
            if row["mid"] not in source_features:
                source_features[row["mid"]] = extractor.dino_feature(mannequin, source_mask)
            dino = cosine(extractor.dino_feature(generated, generated_mask), source_features[row["mid"]])
            intersection = np.logical_and(generated_mask > 0, source_mask > 0).sum()
            union = np.logical_or(generated_mask > 0, source_mask > 0).sum()
            output[key] = {
                "garment_dino": dino,
                "garment_iou": float(intersection / union) if union else None,
                "generated_mask": generated_mask,
                "source_mask": source_mask,
                "status": "ok",
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            output[key] = {"status": "failed", "error": str(exc)}
    return output


def _load_adaface(cfg: dict[str, Any], device: str) -> tuple[Any, Any]:
    from metrics.heldout_id import AdaFaceRecognizer, RetinaFaceAligner, UnifaceAdaFaceRecognizer

    mcfg = cfg.get("metrics", {}).get("heldout_id", {})
    checkpoint = Path(str(mcfg.get("checkpoint", "")))
    if not checkpoint.is_file():
        raise RuntimeError(f"held-out AdaFace checkpoint is unavailable: {checkpoint}")
    providers = mcfg.get("recognizer_providers")
    import torch

    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    if str(mcfg.get("recognizer", "")).lower().startswith("uniface"):
        recognizer = UnifaceAdaFaceRecognizer(
            cache_dir=mcfg.get("cache_dir", checkpoint.parent),
            checkpoint=checkpoint,
            device=torch_device,
            providers=[str(item) for item in providers] if providers else None,
        )
    else:
        repo = Path(str(mcfg.get("repo", "")))
        recognizer = AdaFaceRecognizer(repo, checkpoint, torch_device, architecture=mcfg.get("architecture", "ir_101"))
    detector = RetinaFaceAligner(
        model_root=mcfg.get("detector_model_root", cfg.get("cache", {}).get("arcface_model_root", "")),
        device_id=int(mcfg.get("detector_device_id", 0 if str(torch_device).startswith("cuda") else -1)),
        det_size=int(mcfg.get("det_size", 640)),
        providers=mcfg.get("detector_providers"),
    )
    return recognizer, detector


def _embed_rows(cfg: dict[str, Any], rows: list[dict[str, Any]], device: str) -> tuple[dict[tuple[str, str, int], np.ndarray], dict[str, np.ndarray], dict[tuple[str, str, int], str]]:
    from metrics.heldout_id import resize_face_crop_rgb

    recognizer, detector = _load_adaface(cfg, device)
    mcfg = cfg["metrics"]["heldout_id"]
    generated: dict[tuple[str, str, int], np.ndarray] = {}
    refs: dict[str, np.ndarray] = {}
    failures: dict[tuple[str, str, int], str] = {}

    def ref_embedding(sample_id: str) -> np.ndarray:
        if sample_id in refs:
            return refs[sample_id]
        face_path = _face_crop_path(cfg, sample_id)
        try:
            aligned, _, _ = detector.align(face_path, expand=float(mcfg.get("ref_expand", 1.1)), min_crop=int(mcfg.get("min_crop_px", 256)))
        except RuntimeError as exc:
            if "found no face" not in str(exc):
                raise
            aligned = resize_face_crop_rgb(face_path)
        refs[sample_id] = recognizer.embed_aligned_rgb(aligned)
        return refs[sample_id]

    for row in rows:
        key = (row["mid"], row["jid"], row["seed"])
        try:
            aligned, _, _ = detector.align(row["generated_path"], expand=float(mcfg.get("gen_expand", 1.3)), min_crop=int(mcfg.get("min_crop_px", 256)))
            generated[key] = recognizer.embed_aligned_rgb(aligned)
            ref_embedding(row["jid"])
            ref_embedding(row["mid"])
        except Exception as exc:  # noqa: BLE001
            failures[key] = str(exc)
    return generated, refs, failures


def calibrate_tar(genuine: Iterable[float], impostors: Iterable[float], far: float = 1e-3) -> dict[str, Any]:
    genuine_values = np.asarray([float(value) for value in genuine], dtype=np.float64)
    impostor_values = np.asarray([float(value) for value in impostors], dtype=np.float64)
    if genuine_values.size == 0 or impostor_values.size == 0:
        raise ValueError("TAR calibration requires non-empty genuine and impostor scores")
    threshold = float(np.quantile(impostor_values, 1.0 - float(far), method="higher"))
    return {
        "far_target": float(far),
        "threshold": threshold,
        "impostor_count": int(impostor_values.size),
        "genuine_count": int(genuine_values.size),
        "tar": float(np.mean(genuine_values >= threshold)),
        "empirical_far": float(np.mean(impostor_values >= threshold)),
    }


def _identity_metrics(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    device: str,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    generated, refs, failures = _embed_rows(cfg, rows + calibration_rows, device)
    calibration_ref_ids = {
        str(row["jid"]) for row in calibration_rows
    } | {
        str(row["mid"]) for row in calibration_rows
    }
    genuine_val: list[float] = []
    impostors: list[float] = []
    for row in calibration_rows:
        key = (row["mid"], row["jid"], row["seed"])
        if key not in generated:
            continue
        genuine_val.append(float(np.dot(generated[key], refs[row["jid"]])))
        for ref_id, embedding in sorted(refs.items()):
            if ref_id not in calibration_ref_ids:
                continue
            if ref_id != row["jid"]:
                impostors.append(float(np.dot(generated[key], embedding)))
    calibration = calibrate_tar(genuine_val, impostors)
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    threshold = float(calibration["threshold"])
    for row in rows:
        key = (row["mid"], row["jid"], row["seed"])
        if key in failures or key not in generated:
            output[key] = {"status": "failed", "error": failures.get(key, "identity embedding missing")}
            continue
        score = float(np.dot(generated[key], refs[row["jid"]]))
        output[key] = {"id_cosine": score, "tar_at_1e-3": float(score >= threshold), "status": "ok", "error": ""}
    return output, calibration


def _background_metrics(
    rows: list[dict[str, Any]],
    garment: dict[tuple[str, str, int], dict[str, Any]],
    out_dir: Path,
    device: str,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    from skimage.metrics import structural_similarity
    import torch
    import lpips

    model = lpips.LPIPS(net="alex").eval().requires_grad_(False).to(device)
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["mid"], row["jid"], row["seed"])
        base = garment.get(key, {})
        try:
            generated = np.asarray(Image.open(row["generated_path"]).convert("RGB"), dtype=np.uint8)
            mannequin = np.asarray(Image.open(row["mannequin_path"]).convert("RGB"), dtype=np.uint8)
            if generated.shape != mannequin.shape:
                mannequin = np.asarray(Image.fromarray(mannequin).resize((generated.shape[1], generated.shape[0]), Image.Resampling.BICUBIC))
            mask_a = np.asarray(base["generated_mask"], dtype=bool)
            mask_b = np.asarray(base["source_mask"], dtype=bool)
            if mask_a.shape != generated.shape[:2]:
                mask_a = np.asarray(Image.fromarray(mask_a.astype(np.uint8) * 255).resize((generated.shape[1], generated.shape[0]), Image.Resampling.NEAREST)) > 127
            if mask_b.shape != generated.shape[:2]:
                mask_b = np.asarray(Image.fromarray(mask_b.astype(np.uint8) * 255).resize((generated.shape[1], generated.shape[0]), Image.Resampling.NEAREST)) > 127
            common = ~(mask_a | mask_b)
            fraction = float(common.mean())
            if fraction <= 0.01:
                raise ValueError("common background fraction <= 0.01")
            a = generated.copy()
            b = mannequin.copy()
            a[~common] = 127
            b[~common] = 127
            ssim = float(structural_similarity(a, b, channel_axis=2, data_range=255))
            ta = torch.from_numpy(a.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
            tb = torch.from_numpy(b.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.inference_mode():
                lp = float(model(ta, tb).item())
            result[key] = {"bg_ssim": ssim, "bg_lpips": lp, "status": "ok", "error": ""}
        except Exception as exc:  # noqa: BLE001
            result[key] = {"status": "failed", "error": str(exc)}
    del model
    return result


def _pose_pck(cfg: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path, device: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    from metrics_v2.pose import _run_helper, _prediction_key
    from metrics_v2.common import image_size

    helper_rows = [{"mid": row["mid"], "jid": row["jid"], "seed": row["seed"], "path": Path(row["generated_path"])} for row in rows]
    predictions: dict[str, dict[str, Any]] = {}
    prediction_path = _run_helper(cfg, helper_rows, out_dir / "pose", device)
    for line in prediction_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            predictions[str(item["key"])] = item
    pcfg = cfg["metrics_v2"]["pose"]
    indices = [int(value) for value in pcfg.get("body_indices", list(range(1, 14)))]
    threshold = float(pcfg.get("score_threshold", 0.3))
    pck_threshold = 0.05
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["mid"], row["jid"], row["seed"])
        try:
            pred = predictions[_prediction_key({"mid": row["mid"], "jid": row["jid"], "seed": row["seed"]})]
            if pred.get("status") != "ok":
                raise RuntimeError(pred.get("error", "DWPose failed"))
            target_path = _dwpose_target_path(cfg, row["mid"])
            with np.load(target_path) as target:
                target_body = np.asarray(target["body"], dtype=np.float32)
                target_scores = np.asarray(target["body_scores"], dtype=np.float32)
            body = np.asarray(pred["body"], dtype=np.float32)
            scores = np.asarray(pred["body_scores"], dtype=np.float32)
            width, height = int(pred["width"]), int(pred["height"])
            scale = np.asarray([width, height], dtype=np.float32)
            valid = [index for index in indices if scores[index] >= threshold and target_scores[index] >= threshold]
            if not valid:
                raise RuntimeError("no mutually confident body keypoints")
            distances = np.linalg.norm((body[valid] - target_body[valid]) * scale, axis=1) / float(np.hypot(width, height))
            result[key] = {"pose_pck": float(np.mean(distances <= pck_threshold)), "status": "ok", "error": ""}
        except Exception as exc:  # noqa: BLE001
            result[key] = {"status": "failed", "error": str(exc)}
    return result


def _fid(cfg: dict[str, Any], rows: list[dict[str, Any]], generated_dir: Path, out_dir: Path, device: str) -> float:
    import torch_fidelity

    real_dir = out_dir / "fid_real_refs"
    real_dir.mkdir(parents=True, exist_ok=True)
    generated_stage = out_dir / "fid_generated"
    generated_stage.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        source = Path(row["generated_path"])
        if not source.is_file():
            raise FileNotFoundError(f"missing FID generated image: {source}")
        target = generated_stage / f"{index:06d}{source.suffix.lower()}"
        if not target.exists():
            target.symlink_to(source.resolve())
    seen: set[str] = set()
    for row in rows:
        if row["jid"] in seen:
            continue
        source = Path(row["human_path"])
        if not source.is_file():
            raise FileNotFoundError(f"missing FID real reference: {source}")
        target = real_dir / f"{len(seen):06d}{source.suffix.lower()}"
        target.symlink_to(source.resolve())
        seen.add(row["jid"])
    metrics = torch_fidelity.calculate_metrics(
        input1=str(generated_stage),
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


def evaluate(
    *,
    config_path: str | Path,
    manifest: str | Path,
    generated_dir: str | Path,
    output_dir: str | Path,
    split: str = "test",
    calibration_split: str = "val",
    device: str = "cuda:0",
    checkpoint: str | Path | None = None,
    asset_root: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _normalise_cfg(_load_cfg(config_path))
    dataset_root = Path(cfg["data"]["root"])
    asset_root_path = resolve_asset_root(manifest, dataset_root, asset_root)
    cfg["data"] = {**cfg.get("data", {}), "asset_root": str(asset_root_path)}
    rows = load_eval_rows(manifest, generated_dir, dataset_root, split)
    calibration_rows = load_eval_rows(manifest, generated_dir, dataset_root, calibration_split)
    pair_results = {(row["mid"], row["jid"], row["seed"]): dict(row) for row in rows}
    failures: list[dict[str, Any]] = []
    garment = _run_garment_and_parsing(cfg, rows, output_dir, device)
    identity, calibration = _identity_metrics(cfg, rows, calibration_rows, device)
    background = _background_metrics(rows, garment, output_dir, device)
    pose = _pose_pck(cfg, rows, output_dir, device)
    metric_maps = (garment, identity, background, pose)
    for key, row in pair_results.items():
        row.update({name: None for name in METRIC_NAMES})
        errors: list[str] = []
        for metric_map in metric_maps:
            item = metric_map.get(key, {})
            for name in METRIC_NAMES:
                if name in item:
                    row[name] = item[name]
            if item.get("status") == "failed":
                errors.append(str(item.get("error", "metric failed")))
        row["status"] = "failed" if errors else "ok"
        row["error"] = " | ".join(errors)
        if errors:
            failures.append(row)
    fid_value = _fid(cfg, rows, Path(generated_dir), output_dir, device)
    for row in pair_results.values():
        row["fid"] = fid_value
    rows_out = sorted(pair_results.values(), key=lambda row: (row["mid"], row["jid"], row["seed"]))
    _write_csv(output_dir / "per_pair_metrics.csv", rows_out, ROW_FIELDS)
    _write_csv(output_dir / "failures.csv", failures, ROW_FIELDS)
    summaries: dict[str, Any] = {}
    for name in METRIC_NAMES:
        values = [row.get(name) for row in rows_out]
        summaries[name] = _summary(values, sum(_finite(row.get(name)) is None for row in rows_out))
    summaries["fid"]["count"] = len(rows_out) if rows_out else 0
    summaries["fid"]["failed"] = 0
    summaries["fid"]["mean"] = fid_value
    summaries["fid"]["median"] = fid_value
    checkpoint_path = Path(checkpoint) if checkpoint else Path(cfg.get("training", {}).get("resume") or "")
    summary = {
        "status": "ok" if not failures else "completed_with_failures",
        "experiment": cfg.get("experiment", {}).get("id", config_path.stem),
        "method": cfg.get("experiment_method", {}).get("name"),
        "split": split,
        "calibration_split": calibration_split,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint_path) if str(checkpoint_path) else None,
        "checkpoint_sha256": sha256_path(checkpoint_path) if str(checkpoint_path) else "unspecified",
        "dataset_root": str(dataset_root),
        "asset_root": str(asset_root_path),
        "sample_counts": {"evaluated": len(rows_out), "ok": len(rows_out) - len(failures), "failed": len(failures), "calibration": len(calibration_rows)},
        "calibration": calibration,
        "metrics": summaries,
        "outputs": {"per_pair": str(output_dir / "per_pair_metrics.csv"), "failures": str(output_dir / "failures.csv"), "summary": str(output_dir / "summary.json")},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--asset-root", default=None, help="Root for read-only source-derived assets such as face crops, DWPose, parsing, and masks. Defaults to manifest provenance.read_only_source_root, then config data.root.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    summary = evaluate(
        config_path=args.config,
        manifest=args.manifest,
        generated_dir=args.generated_dir,
        output_dir=args.output_dir,
        split=args.split,
        calibration_split=args.calibration_split,
        device=args.device,
        checkpoint=args.checkpoint,
        asset_root=args.asset_root,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
