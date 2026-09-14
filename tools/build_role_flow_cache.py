#!/usr/bin/env python3
"""Bridge clean mannequin conditions into a FLUX training cache.

The source dataset is read-only. Existing human/reference lineage fields are
copied from the legacy Phase-1 cache, while pose and garment latent fields are
encoded from the clean mannequin-only cache with one resident FLUX VAE.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET = "/data/muxiangyu/datasets/M2HImage/M2H_Final_v2_clean_v1"
DEFAULT_EXPERIMENT = "/data/muxiangyu/experiments/M2H_Final_v2_clean_v1"
DEFAULT_BASE = "/data/muxiangyu/experiments/M2H_Final_v2_clean_v1/cache_m_only/legacy_split/phase1/cache_768x1024"
DEFAULT_M_ONLY = "/data/muxiangyu/experiments/M2H_Final_v2_clean_v1/cache_m_only"
DEFAULT_VAE = "/tmp/m2h_models/FLUX.1-dev"
COPY_KEYS = ("target_latents", "pulid_id_embed", "appearance")
REQUIRED_KEYS = (*COPY_KEYS, "pose_latents", "garment_ref_latents", "garment_grid", "head_pose")
REGION_NAMES = ("cloth_safe_z", "body_bg_z", "face_z", "hair_z")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ids(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def finite_shape(value: np.ndarray) -> bool:
    return value.size > 0 and bool(np.isfinite(value).all())


def head_pose_from_keypoints(payload: dict[str, np.ndarray]) -> np.ndarray:
    """Derive a deterministic 6D head pose token from repaired DWPose points."""
    body = np.asarray(payload.get("mannequin_dwpose_body", np.empty((0, 2))), dtype=np.float32)
    scores = np.asarray(payload.get("mannequin_dwpose_body_scores", np.empty((0,))), dtype=np.float32)
    if body.shape[0] < 6:
        return np.zeros(7, dtype=np.float32)
    if scores.size >= 6 and float(np.min(scores[[0, 1, 2, 5]])) < 0.1:
        return np.zeros(7, dtype=np.float32)
    nose, neck, rsho, lsho = body[0], body[1], body[2], body[5]
    shoulder = float(np.linalg.norm(lsho - rsho))
    if not np.isfinite(shoulder) or shoulder < 1e-4:
        return np.zeros(7, dtype=np.float32)
    yaw = float(np.arctan2(float(nose[0] - neck[0]), 0.35 * shoulder))
    pitch = float(np.arctan2(float(neck[1] - nose[1]), shoulder))
    roll = float(np.arctan2(float(lsho[1] - rsho[1]), float(lsho[0] - rsho[0])))
    return np.asarray(
        [np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch), np.sin(roll), np.cos(roll), 1.0],
        dtype=np.float32,
    )


def encode(vae: Any, image: np.ndarray, width: int, height: int, device: Any, dtype: Any) -> np.ndarray:
    import torch
    from PIL import Image
    from conditions import pack_latents, pil_to_tensor

    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    tensor = pil_to_tensor(pil, {"width": width, "height": height}).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.inference_mode():
        latent = vae.encode(tensor).latent_dist.mean
        latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
        packed = pack_latents(latent)[0].float().cpu().numpy()
    return packed.astype(np.float16, copy=False)


def provenance(field: str, role: str, source: Path | str, generator: str, note: str = "") -> dict[str, Any]:
    path = Path(source)
    return {
        "field_name": field,
        "source_role": role,
        "source_path": str(path),
        "source_sha256": digest(path) if path.exists() and path.is_file() else None,
        "generator": generator,
        "generator_version": "role-flow-cache-v1",
        "preprocess_config": {"resolution": {"width": 768, "height": 1024}, "note": note},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET)
    ap.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT)
    ap.add_argument("--base-cache", default=DEFAULT_BASE)
    ap.add_argument("--m-only-cache", default=DEFAULT_M_ONLY)
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--split-manifest", default=None)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--smoke-ids", nargs="*", default=None)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-split", choices=("train", "val", "test"), default=None)
    ap.add_argument("--ids-file", default=None)
    args = ap.parse_args()
    if args.num_workers != 1:
        raise ValueError("VAE cache bridge uses one resident process; use --num-workers 1")
    dataset = Path(args.dataset_root)
    experiment = Path(args.experiment_root)
    base = Path(args.base_cache)
    m_only = Path(args.m_only_cache)
    out_root = experiment / "role_flow" / "cache_768x1024"
    region_root = out_root / "region_masks_z"
    report = dataset / "dataset_cleaning_report"
    split_root = dataset / "splits_person_disjoint_final"
    if args.split_manifest:
        manifest_path = Path(args.split_manifest)
    else:
        manifest_path = report / "person_disjoint_manifest.csv"
    manifest_rows = {}
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                manifest_rows[str(row["id"])] = row
    split_ids: dict[str, list[str]] = {s: ids(split_root / f"{s}.txt") for s in ("train", "val", "test")}
    if args.only_split:
        split_ids = {args.only_split: split_ids[args.only_split]}
    if args.ids_file:
        wanted = {x.strip() for x in Path(args.ids_file).read_text().splitlines() if x.strip()}
        split_ids = {s: [x for x in v if x in wanted] for s, v in split_ids.items()}
    elif args.smoke_ids:
        wanted = set(args.smoke_ids)
        split_ids = {s: [x for x in v if x in wanted] for s, v in split_ids.items()}
    elif args.limit:
        split_ids = {s: v[: args.limit] for s, v in split_ids.items()}
    if args.dry_run:
        print(json.dumps({"output": str(out_root), "splits": {k: len(v) for k, v in split_ids.items()}}, indent=2))
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    region_root.mkdir(parents=True, exist_ok=True)
    prov_path = report / "role_flow_cache_field_provenance.jsonl"
    manifest_out = out_root / "manifest.csv"
    state_path = out_root / "build_state.json"
    report.mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers import AutoencoderKL
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from conditions import choose_dtype

    device = torch.device(args.device)
    dtype = torch.bfloat16
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae", torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=False)
    vae.eval().requires_grad_(False).to(device)
    rows: list[dict[str, Any]] = []
    all_prov: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for split, split_list in split_ids.items():
        split_out = out_root / split / "samples"
        split_out.mkdir(parents=True, exist_ok=True)
        for sid in sorted(split_list):
            out_path = split_out / f"{sid}.npz"
            mask_path = region_root / f"{sid}.npz"
            if out_path.exists() and mask_path.exists():
                try:
                    with np.load(out_path, allow_pickle=False) as cached:
                        absent = [k for k in REQUIRED_KEYS if k not in cached.files]
                        invalid = [k for k in REQUIRED_KEYS if k in cached.files and not finite_shape(np.asarray(cached[k]))]
                    with np.load(mask_path, allow_pickle=False) as masks:
                        absent_masks = [k for k in REGION_NAMES if k not in masks.files]
                        invalid_masks = [k for k in REGION_NAMES if k in masks.files and not finite_shape(np.asarray(masks[k]))]
                    if not absent and not invalid and not absent_masks and not invalid_masks:
                        rows.append({"id": sid, "split": split, "status": "resumed", "cache_path": str(out_path), "region_mask_path": str(mask_path), "fields": len(REQUIRED_KEYS), "cache_sha256": digest(out_path)})
                        continue
                except Exception:
                    pass
            try:
                with np.load(m_only / split / "samples" / f"{sid}.npz", allow_pickle=False) as m:
                    m_payload = {k: np.asarray(m[k]) for k in m.files}
                with np.load(base / "samples" / f"{sid}.npz", allow_pickle=False) as old:
                    missing = [k for k in COPY_KEYS if k not in old.files]
                    if missing:
                        raise KeyError(f"legacy cache missing {missing}")
                    payload = {k: np.asarray(old[k]) for k in COPY_KEYS}
                for k in COPY_KEYS:
                    if not finite_shape(payload[k]):
                        raise ValueError(f"{k} non-finite or empty")
                    all_prov.append({"sample_id": sid, "output_path": str(out_path), **provenance(k, "human_or_reference", base / "samples" / f"{sid}.npz", "copy_legacy_field", "allowed legacy H/reference lineage")})

                pose_img = m_payload["mannequin_pose_image"]
                garment_img = m_payload["mannequin_garment_image"]
                payload["pose_latents"] = encode(vae, pose_img, 768, 1024, device, dtype)
                payload["garment_ref_latents"] = encode(vae, garment_img, 768, 1024, device, dtype)
                all_prov.extend([
                    {"sample_id": sid, "output_path": str(out_path), **provenance("pose_latents", "mannequin", m_only / split / "samples" / f"{sid}.npz", "FLUX_VAE_encode", "mannequin_pose_image only")},
                    {"sample_id": sid, "output_path": str(out_path), **provenance("garment_ref_latents", "mannequin", m_only / split / "samples" / f"{sid}.npz", "FLUX_VAE_encode", "mannequin_garment_image only")},
                ])
                # The legacy garment-grid route is disabled by the role-flow configs.
                payload["garment_grid"] = np.zeros((64, 1024), dtype=np.float32)
                all_prov.append({"sample_id": sid, "output_path": str(out_path), **provenance("garment_grid", "none", m_only / split / "samples" / f"{sid}.npz", "explicit_zero_inactive_route", "legacy garment-grid route disabled; never consumed")})
                payload["head_pose"] = head_pose_from_keypoints(m_payload)
                all_prov.append({"sample_id": sid, "output_path": str(out_path), **provenance("head_pose", "mannequin", m_only / split / "samples" / f"{sid}.npz", "derive_head_pose_from_repaired_dwpose", "indices nose=0 neck=1 shoulders=2,5")})

                for key in REQUIRED_KEYS:
                    if key not in payload or not finite_shape(np.asarray(payload[key])):
                        raise ValueError(f"invalid required field {key}")
                tmp = out_path.with_suffix(".npz.tmp")
                with tmp.open("wb") as f:
                    np.savez_compressed(f, **payload)
                os.replace(tmp, out_path)
                masks = {name: m_payload[f"mannequin_region_{name}"].astype(np.float16, copy=False) for name in REGION_NAMES}
                mask_tmp = mask_path.with_suffix(".npz.tmp")
                with mask_tmp.open("wb") as f:
                    np.savez_compressed(f, **masks)
                os.replace(mask_tmp, mask_path)
                rows.append({"id": sid, "split": split, "status": "ok", "cache_path": str(out_path), "region_mask_path": str(mask_path), "fields": len(payload), "cache_sha256": digest(out_path)})
            except Exception as exc:
                failures.append({"id": sid, "split": split, "status": "failed", "error": str(exc)})
                rows.append(failures[-1])
                if args.strict:
                    raise

    with prov_path.open("w", encoding="utf-8") as f:
        for item in sorted(all_prov, key=lambda x: (x["sample_id"], x["field_name"])):
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    with manifest_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "status", "cache_path", "region_mask_path", "fields", "cache_sha256", "error"])
        writer.writeheader()
        for row in sorted(rows, key=lambda x: (x["split"], x["id"])):
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
    summary = {
        "version": "role-flow-cache-v1",
        "dataset_root": str(dataset), "base_cache": str(base), "m_only_cache": str(m_only),
        "output_root": str(out_root), "region_masks_z": str(region_root), "vae": str(Path(args.vae)),
        "splits": {s: {"requested": len(v), "built": sum(r["split"] == s and r["status"] == "ok" for r in rows), "failed": sum(r["split"] == s and r["status"] == "failed" for r in rows)} for s, v in split_ids.items()},
        "required_keys": list(REQUIRED_KEYS), "copied_keys": list(COPY_KEYS), "failures": failures,
        "legacy_garment_grid": {"enabled": False, "placeholder": "explicit zeros retained for dataset schema only"},
        "head_pose_source": "mannequin repaired DWPose keypoints",
    }
    state_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
