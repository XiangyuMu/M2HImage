from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from conditions import choose_dtype, load_yaml, make_image_ids, seed_everything
from dataset import PairedWarmupDataset
from eval_b2 import garment_type
from spatial_conditions import load_fashn_labels
from train_paired import WarmupFlowModel, load_checkpoint, load_components
from watcher_protocol import watcher_eval_set_from_config


TRACK_FIELDS = (
    "step",
    "cloth_concentration",
    "cloth_energy_share",
    "cloth_area_share",
    "garment_union_concentration",
    "hair_relative_response",
    "hair_swap_relative_response",
    "hair_off_relative_response",
    "hair_ref_swap_concentration",
    "garment_dino",
    "garment_dino_mean",
    "garment_dino_by_type",
    "garment_hf_lpips",
    "garment_hf_lpips_mean",
    "garment_gradient_sim",
    "hair_dino",
    "hair_dino_mean",
    "hair_lab_distance",
    "head5_distance",
    "body_distance",
    "face_detection_rate",
    "swap_cosine_mean",
    "swap_cosine_max",
    "appearance_gate",
    "hair_gate",
    "head_pose_gate",
)


def _prompt(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = batch["prompt_embeds"].to(device=device, dtype=dtype)
    pooled = batch["pooled_prompt_embeds"].to(device=device, dtype=dtype)
    if prompt.ndim == 2:
        prompt = prompt.unsqueeze(0)
    if pooled.ndim == 1:
        pooled = pooled.unsqueeze(0)
    return prompt, pooled


def _condition_tokens(
    model: WarmupFlowModel,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    *,
    appearance_source: dict[str, Any] | None = None,
    hair_source: dict[str, Any] | None = None,
    hair_off: bool = False,
) -> torch.Tensor:
    appearance_source = appearance_source or batch
    hair_source = hair_source or batch
    prompt, _ = _prompt(batch, device, dtype)
    if model.hair_enabled:
        hair_inputs = model._hair_inputs(hair_source, device, dtype)
        if hair_off:
            hair_inputs = tuple(
                None if value is None else torch.zeros_like(value)
                for value in hair_inputs
            )
    else:
        hair_inputs = (None, None, None)
    return model._condition_tokens(
        prompt,
        appearance_source["appearance"].to(device=device, dtype=dtype).unsqueeze(0),
        batch["garment"].to(device=device, dtype=dtype).unsqueeze(0),
        batch["head_pose"].to(device=device, dtype=dtype).unsqueeze(0),
        *hair_inputs,
    )


def _hair_area_fraction(root: Path, sample_id: str) -> float:
    labels = load_fashn_labels(root, sample_id)
    return float((labels == 2).mean())


def _select_probe_indices(
    dataset: PairedWarmupDataset,
    root: Path,
    count: int,
    visible_hair_min_fraction: float = 0.0,
) -> tuple[list[int], dict[str, str]]:
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    types: dict[str, str] = {}
    for index, sample_id in enumerate(dataset.ids):
        if _hair_area_fraction(root, sample_id) < visible_hair_min_fraction:
            continue
        kind = garment_type(root, sample_id)
        types[sample_id] = kind
        groups[kind].append((sample_id, index))
    for rows in groups.values():
        rows.sort()
    selected: list[int] = []
    offsets = {kind: 0 for kind in groups}
    kinds = sorted(groups)
    while len(selected) < min(count, len(dataset)):
        progressed = False
        for kind in kinds:
            rows = groups[kind]
            offset = offsets[kind]
            if offset >= len(rows):
                continue
            # Alternate from the beginning and end so six samples span each stratum.
            position = offset // 2 if offset % 2 == 0 else len(rows) - 1 - offset // 2
            selected.append(rows[position][1])
            offsets[kind] += 1
            progressed = True
            if len(selected) >= min(count, len(dataset)):
                break
        if not progressed:
            break
    if len(selected) < min(count, len(dataset)):
        raise RuntimeError(f"could select only {len(selected)} of {count} response probes")
    return selected, types


def _same_type_donor_index(
    dataset: PairedWarmupDataset,
    root: Path,
    source_index: int,
    source_type: str,
    visible_hair_min_fraction: float,
    allowed_ids: set[str] | None = None,
) -> int:
    candidates = [
        index
        for index, sample_id in enumerate(dataset.ids)
        if index != source_index
        and (allowed_ids is None or sample_id in allowed_ids)
        and garment_type(root, sample_id) == source_type
        and _hair_area_fraction(root, sample_id) >= visible_hair_min_fraction
    ]
    if not candidates:
        raise RuntimeError(
            f'no same-type visible-hair donor for {dataset.ids[source_index]} '
            f'(type={source_type})'
        )
    candidates.sort(key=lambda index: dataset.ids[index])
    source_id = dataset.ids[source_index]
    later = [index for index in candidates if dataset.ids[index] > source_id]
    return later[0] if later else candidates[0]


def _load_masks(root: Path, sample_id: str, token_count: int) -> dict[str, np.ndarray]:
    path = root / "derived/region_masks_z" / f"{sample_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"response-track mask is missing: {path}")
    with np.load(path, allow_pickle=False) as row:
        mapping = {
            "cloth": "cloth_safe_z",
            "face": "face_z",
            "body_bg": "body_bg_z",
            "hair": "hair_z",
        }
        missing = [source for source in mapping.values() if source not in row.files]
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        masks = {
            name: np.asarray(row[source], dtype=np.float32).clip(0.0, 1.0)
            for name, source in mapping.items()
        }
    for name, mask in masks.items():
        if mask.shape != (token_count,):
            raise RuntimeError(f"{path}: {name} shape={mask.shape}, expected={(token_count,)}")
    return masks


def _response_rms(diff: torch.Tensor) -> float:
    return float(diff.detach().float().square().mean().sqrt().cpu())


def _energy_stats(diff: torch.Tensor, masks: dict[str, np.ndarray]) -> dict[str, float]:
    energy = diff.detach().float().square().mean(dim=-1)[0]
    total = energy.sum().clamp_min(1e-12)
    result: dict[str, float] = {}
    for name, mask_array in masks.items():
        mask = torch.from_numpy(mask_array).to(device=energy.device, dtype=torch.float32)
        area_share = mask.mean()
        energy_share = (energy * mask).sum() / total
        result[f"{name}_area_share"] = float(area_share.cpu())
        result[f"{name}_energy_share"] = float(energy_share.cpu())
        result[f"{name}_concentration"] = float(
            (energy_share / area_share.clamp_min(1e-8)).cpu()
        )
    return result


@torch.inference_mode()
def compute_response_snapshot(
    cfg: dict[str, Any],
    model: WarmupFlowModel,
    dataset: PairedWarmupDataset,
    checkpoint: Path,
    checkpoint_step: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    track_cfg = cfg.get("eval", {}).get("response_track", {})
    sample_count = int(track_cfg.get("sample_count", 6))
    taus = [float(value) for value in track_cfg.get("taus", (0.3, 0.5, 0.7))]
    seed = int(track_cfg.get("seed", 20260817))
    root = Path(cfg["data"]["root"])
    visible_hair_min = float(
        track_cfg.get("visible_hair_min_fraction", 0.01)
    )
    protocol = watcher_eval_set_from_config(cfg)
    protocol_ids = [str(value) for value in protocol["sample_ids"]]
    if sample_count != len(protocol_ids):
        raise RuntimeError(
            "response_track.sample_count must equal the frozen 16-sample protocol; "
            f"got {sample_count}"
        )
    missing_ids = sorted(set(protocol_ids) - set(dataset.ids))
    if missing_ids:
        raise RuntimeError(
            f"frozen response-track ids are absent from val dataset: {missing_ids}"
        )
    indices = [dataset.ids.index(sample_id) for sample_id in protocol_ids]
    garment_types = {
        str(row["id"]): str(row["garment_type"])
        for row in protocol["samples"]
    }
    rows: list[dict[str, Any]] = []
    for order, index in enumerate(indices):
        batch = dataset[index]
        source_type = garment_types[str(batch["sample_id"])]
        donor_index = _same_type_donor_index(
            dataset,
            root,
            index,
            source_type,
            visible_hair_min,
            allowed_ids=set(protocol_ids),
        )
        donor = dataset[donor_index]
        sample_id = str(batch["sample_id"])
        donor_id = str(donor["sample_id"])
        prompt, pooled = _prompt(batch, device, dtype)
        own_tokens = _condition_tokens(model, batch, device, dtype)
        garment_tokens = own_tokens
        hair_swap_tokens = _condition_tokens(
            model, batch, device, dtype, hair_source=donor
        )
        hair_off_tokens = _condition_tokens(
            model, batch, device, dtype, hair_off=True
        )
        identity_swap_tokens = _condition_tokens(
            model,
            batch,
            device,
            dtype,
            appearance_source=donor,
            hair_source=donor,
        )
        masks = _load_masks(root, sample_id, model.image_token_count)
        donor_masks = _load_masks(root, donor_id, model.image_token_count)
        garment_masks = dict(masks)
        garment_masks["cloth"] = np.maximum(
            masks["cloth"], donor_masks["cloth"]
        )
        hair_masks = dict(masks)
        hair_masks["hair"] = np.maximum(masks["hair"], donor_masks["hair"])
        generator = torch.Generator(device=device).manual_seed(seed + order)
        z = torch.randn(
            1,
            model.image_token_count,
            64,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        pose = batch["pose_latents"].to(device=device, dtype=dtype).unsqueeze(0)
        own_pulid = batch["pulid_id_embed"].to(device=device, dtype=dtype).unsqueeze(0)
        donor_pulid = donor["pulid_id_embed"].to(device=device, dtype=dtype).unsqueeze(0)
        image_ids = make_image_ids(model.width, model.height, device, dtype)
        for tau_value in taus:
            tau = torch.tensor([tau_value], device=device, dtype=dtype)
            # Identity is absent from ControlNet, so this residual is shared exactly.
            control = model._controlnet_forward(
                z, tau, prompt, pooled, pose, image_ids
            )

            def forward(
                tokens: torch.Tensor,
                pulid_embed: torch.Tensor,
                garment_ref: torch.Tensor,
                hair_ref: torch.Tensor | None,
            ) -> torch.Tensor:
                return model._transformer_forward(
                    z,
                    tau,
                    tokens,
                    pulid_embed,
                    control,
                    pooled=pooled,
                    img_ids=image_ids,
                    garment_ref_latents=garment_ref,
                    hair_ref_latents=hair_ref,
                )

            own_hair_ref = batch.get("hair_ref_latents")
            donor_hair_ref = donor.get("hair_ref_latents")
            baseline = forward(
                own_tokens,
                own_pulid,
                batch["garment_ref_latents"],
                own_hair_ref,
            )
            if model.hair_reference_enabled:
                if own_hair_ref is None or donor_hair_ref is None:
                    raise RuntimeError(
                        'hair-reference response probe requires both source and donor latents'
                    )
                hair_swap_args = (own_tokens, donor_hair_ref)
                hair_off_args = (own_tokens, torch.zeros_like(own_hair_ref))
                identity_hair_ref = donor_hair_ref
            else:
                hair_swap_args = (hair_swap_tokens, None)
                hair_off_args = (hair_off_tokens, None)
                identity_hair_ref = None
            variants = {
                "garment_swap": forward(
                    garment_tokens,
                    own_pulid,
                    donor["garment_ref_latents"],
                    own_hair_ref,
                ),
                "hair_swap": forward(
                    hair_swap_args[0],
                    own_pulid,
                    batch["garment_ref_latents"],
                    hair_swap_args[1],
                ),
                "hair_off": forward(
                    hair_off_args[0],
                    own_pulid,
                    batch["garment_ref_latents"],
                    hair_off_args[1],
                ),
                "identity_swap": forward(
                    identity_swap_tokens,
                    donor_pulid,
                    batch["garment_ref_latents"],
                    identity_hair_ref,
                ),
            }
            diffs = {name: value - baseline for name, value in variants.items()}
            identity_rms = _response_rms(diffs["identity_swap"])
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "donor_id": donor_id,
                "garment_type": garment_types[sample_id],
                "tau": tau_value,
                "identity_response_rms": identity_rms,
                "garment_response_rms": _response_rms(diffs["garment_swap"]),
                "hair_swap_response_rms": _response_rms(diffs["hair_swap"]),
                "hair_off_response_rms": _response_rms(diffs["hair_off"]),
            }
            row.update(_energy_stats(diffs["garment_swap"], garment_masks))
            row["garment_union_concentration"] = row["cloth_concentration"]
            hair_energy = _energy_stats(diffs["hair_swap"], hair_masks)
            row["hair_ref_swap_concentration"] = hair_energy[
                "hair_concentration"
            ]
            row["garment_fraction_of_identity"] = row["garment_response_rms"] / max(identity_rms, 1e-8)
            row["hair_swap_fraction_of_identity"] = row["hair_swap_response_rms"] / max(identity_rms, 1e-8)
            row["hair_off_fraction_of_identity"] = row["hair_off_response_rms"] / max(identity_rms, 1e-8)
            row["hair_relative_response"] = row[
                "hair_swap_fraction_of_identity"
            ]
            rows.append(row)

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, float]:
        keys = [
            "identity_response_rms",
            "garment_response_rms",
            "garment_fraction_of_identity",
            "cloth_area_share",
            "cloth_energy_share",
            "cloth_concentration",
            "garment_union_concentration",
            "face_area_share",
            "face_energy_share",
            "face_concentration",
            "body_bg_area_share",
            "body_bg_energy_share",
            "body_bg_concentration",
            "hair_swap_fraction_of_identity",
            "hair_off_fraction_of_identity",
            "hair_relative_response",
            "hair_ref_swap_concentration",
        ]
        return {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in keys
        }

    summary = aggregate(rows)
    by_tau = {
        str(tau): aggregate([row for row in rows if row["tau"] == tau])
        for tau in taus
    }
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(checkpoint_step),
        "protocol": {
            "samples": len(indices),
            "protocol_version": protocol["protocol_version"],
            "protocol_hash": protocol["protocol_hash"],
            "protocol_path": protocol["path"],
            "sample_ids": [dataset.ids[index] for index in indices],
            "garment_types": {
                dataset.ids[index]: garment_types[dataset.ids[index]]
                for index in indices
            },
            "taus": taus,
            "seed": seed,
            "noise": "fixed Gaussian z per sample, shared by all interventions and tau probes",
            "controlnet": "one identity-independent pose ControlNet call reused by all branches per sample/tau",
            "concentration": "soft-mask energy share divided by soft-mask area share; 1 means uniform energy",
            "garment_donor": "same garment_type; concentration uses source/donor cloth-mask union and is descriptive only",
            "hair_relative_response": "hair-reference swap RMS normalized by complete identity swap RMS",
            "hair_ref_swap_concentration": "hair-reference swap energy concentration over source/donor hair-mask union",
        },
        "response": summary,
        "response_by_tau": by_tau,
        "gates": model.adapter.gate_values(),
        "rows": rows,
    }


def run_watcher_metrics(
    training_cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    track_cfg = training_cfg.get("eval", {}).get("response_track", {})
    metrics_cfg = load_yaml(track_cfg["metrics_config"])
    metrics_cfg["data"]["root"] = training_cfg["data"]["root"]
    metrics_cfg["data"]["resolution"] = training_cfg["data"]["resolution"]
    metrics_cfg = copy.deepcopy(metrics_cfg)
    metrics_cfg["metrics_v2"]["pose"]["provider"] = str(
        track_cfg.get("pose_provider", "CPUExecutionProvider")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "sample_count": len(rows),
        "feature_space_limitation": (
            "training hair loss and watcher Hair-DINO use independent code paths but the same "
            "DINOv2 checkpoint; human review and LAB color distance remain required corroboration"
        ),
    }
    from metrics_v2.parsing import build_generated_parsing

    try:
        result["parsing"] = build_generated_parsing(
            metrics_cfg, rows, output_dir, device
        )
    except Exception as exc:  # noqa: BLE001
        result["parsing"] = {"status": "failed", "error": str(exc)}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        from metrics_v2.garment import run_garment_metrics

        garment = run_garment_metrics(metrics_cfg, rows, output_dir, device)
        result["garment"] = garment
        result["garment_dino"] = garment.get("garment_dino", {}).get("median")
        result["garment_dino_mean"] = garment.get("garment_dino", {}).get("mean")
        result["garment_dino_by_type"] = {
            str(kind): summary.get("median")
            for kind, summary in garment.get("garment_dino_by_type", {}).items()
        }
        result["garment_hf_lpips"] = garment.get("garment_hf_lpips", {}).get("median")
        result["garment_hf_lpips_mean"] = garment.get("garment_hf_lpips", {}).get("mean")
        result["garment_gradient_sim"] = garment.get("garment_gradient_sim", {}).get("median")
    except Exception as exc:  # noqa: BLE001
        result["garment"] = {"status": "failed", "error": str(exc)}
        result["garment_dino"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        from metrics_v2.hair import run_hair_metrics

        hair = run_hair_metrics(metrics_cfg, rows, output_dir, device)
        result["hair"] = hair
        result["hair_dino"] = hair.get("hair_dino", {}).get("median")
        result["hair_dino_mean"] = hair.get("hair_dino", {}).get("mean")
        result["hair_lab_distance"] = hair.get(
            "hair_lab_distance", {}
        ).get("median")
        result["hair_lab_distance_mean"] = hair.get(
            "hair_lab_distance", {}
        ).get("mean")
    except Exception as exc:  # noqa: BLE001
        result["hair"] = {"status": "failed", "error": str(exc)}
        result["hair_dino"] = None
        result["hair_lab_distance"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        from metrics_v2.pose import run_pose_metrics

        pose = run_pose_metrics(metrics_cfg, rows, output_dir, "cpu")
        result["pose"] = pose
        result["head5_distance"] = pose.get("head5_distance", {}).get("median")
        result["head5_distance_mean"] = pose.get("head5_distance", {}).get("mean")
        result["body_distance"] = pose.get("body_distance", {}).get("median")
        result["body_distance_mean"] = pose.get("body_distance", {}).get("mean")
    except Exception as exc:  # noqa: BLE001
        result["pose"] = {"status": "failed", "error": str(exc)}
        result["head5_distance"] = None
    return result


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def flatten_track(payload: dict[str, Any]) -> dict[str, Any]:
    response = payload.get("response", {})
    metrics = payload.get("watcher_metrics", {})
    guards = payload.get("guards", {})
    gates = payload.get("gates", {})
    garment_summary = metrics.get("garment", {}).get("garment_dino", {})
    garment_hf_summary = metrics.get("garment", {}).get("garment_hf_lpips", {})
    garment_gradient_summary = metrics.get("garment", {}).get("garment_gradient_sim", {})
    garment_by_type = metrics.get(
        "garment_dino_by_type",
        {
            str(kind): summary.get("median")
            for kind, summary in metrics.get("garment", {}).get(
                "garment_dino_by_type", {}
            ).items()
        },
    )
    hair_summary = metrics.get("hair", {}).get("hair_dino", {})
    hair_lab_summary = metrics.get("hair", {}).get("hair_lab_distance", {})
    pose_summary = metrics.get("pose", {})
    protocol = payload.get("protocol", {})
    return {
        "step": int(payload["checkpoint_step"]),
        "protocol_hash": protocol.get("protocol_hash", protocol.get("hash")),
        "cloth_concentration": response.get("cloth_concentration"),
        "cloth_energy_share": response.get("cloth_energy_share"),
        "cloth_area_share": response.get("cloth_area_share"),
        "garment_union_concentration": response.get(
            "garment_union_concentration", response.get("cloth_concentration")
        ),
        "hair_relative_response": response.get(
            "hair_swap_fraction_of_identity",
            response.get("hair_relative_response"),
        ),
        "hair_swap_relative_response": response.get("hair_swap_fraction_of_identity"),
        "hair_off_relative_response": response.get("hair_off_fraction_of_identity"),
        "hair_ref_swap_concentration": response.get(
            "hair_ref_swap_concentration"
        ),
        "garment_dino": garment_summary.get(
            "median", metrics.get("garment_dino")
        ),
        "garment_dino_mean": metrics.get(
            "garment_dino_mean", garment_summary.get("mean")
        ),
        "garment_dino_by_type": garment_by_type,
        "garment_hf_lpips": garment_hf_summary.get(
            "median", metrics.get("garment_hf_lpips")
        ),
        "garment_hf_lpips_mean": metrics.get(
            "garment_hf_lpips_mean",
            garment_hf_summary.get("mean"),
        ),
        "garment_gradient_sim": garment_gradient_summary.get(
            "median", metrics.get("garment_gradient_sim")
        ),
        "hair_dino": hair_summary.get(
            "median", metrics.get("hair_dino")
        ),
        "hair_dino_mean": metrics.get(
            "hair_dino_mean", hair_summary.get("mean")
        ),
        "hair_lab_distance": hair_lab_summary.get(
            "median", metrics.get("hair_lab_distance")
        ),
        "head5_distance": pose_summary.get("head5_distance", {}).get(
            "median", metrics.get("head5_distance")
        ),
        "body_distance": pose_summary.get("body_distance", {}).get(
            "median", metrics.get("body_distance")
        ),
        "face_detection_rate": guards.get("face_detection_rate"),
        "swap_cosine_mean": guards.get("swap_cosine_mean"),
        "swap_cosine_max": guards.get("swap_cosine_max"),
        "appearance_gate": gates.get("appearance_gate"),
        "hair_gate": gates.get("hair_gate"),
        "head_pose_gate": gates.get("head_pose_gate"),
    }


def _recent_slope(history: list[dict[str, Any]], field: str) -> float | None:
    points = [
        (float(row["step"]) / 500.0, value)
        for row in history
        if (value := _finite(row.get(field))) is not None
    ][-3:]
    if len(points) < 2:
        return None
    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def _consecutive_negative_intervals(
    history: list[dict[str, Any]], field: str, count: int
) -> bool:
    values = [
        value
        for row in history
        if (value := _finite(row.get(field))) is not None
    ]
    count = max(1, int(count))
    if len(values) < count + 1:
        return False
    recent = values[-(count + 1):]
    return all(right - left < 0.0 for left, right in zip(recent, recent[1:]))


def _consecutive_positive_intervals(
    history: list[dict[str, Any]], field: str, count: int
) -> bool:
    values = [
        value
        for row in history
        if (value := _finite(row.get(field))) is not None
    ]
    count = max(1, int(count))
    if len(values) < count + 1:
        return False
    recent = values[-(count + 1):]
    return all(right - left > 0.0 for left, right in zip(recent, recent[1:]))


def _consecutive_failed_check(
    history: list[dict[str, Any]],
    check,
    count: int,
) -> bool:
    count = max(1, int(count))
    if len(history) < count:
        return False
    return all(not bool(check(row)) for row in history[-count:])


def _type_medians(row: dict[str, Any]) -> dict[str, float]:
    value = row.get("garment_dino_by_type", {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    return {
        str(kind): finite
        for kind, raw in value.items()
        if (finite := _finite(raw)) is not None
    }


def _load_history(
    output_dir: Path,
    baseline_snapshot: str | Path | None = None,
    historical_snapshots: list[str | Path] | None = None,
    expected_protocol_hash: str | None = None,
    required_steps: list[int] | None = None,
) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    configured = ([baseline_snapshot] if baseline_snapshot else []) + list(
        historical_snapshots or []
    )
    for configured_path in configured:
        baseline_path = Path(configured_path)
        if not baseline_path.is_absolute():
            baseline_path = Path.cwd() / baseline_path
        if not baseline_path.exists():
            raise FileNotFoundError(
                f'response-track historical snapshot is missing: {baseline_path}; '
                'run tools/rebaseline_watcher.py before appending a new point'
            )
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        flattened = flatten_track(payload)
        if (
            expected_protocol_hash
            and flattened.get("protocol_hash") != expected_protocol_hash
        ):
            raise RuntimeError(
                f"response-track protocol mismatch at {baseline_path}: "
                f"stored={flattened.get('protocol_hash')} "
                f"expected={expected_protocol_hash}; historical checkpoints "
                "must be re-evaluated before writing a new point"
            )
        by_step[int(payload["checkpoint_step"])] = flattened
    for path in sorted(output_dir.glob("step*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            flattened = flatten_track(payload)
            if (
                expected_protocol_hash
                and flattened.get("protocol_hash") != expected_protocol_hash
            ):
                raise RuntimeError(
                    f"response-track protocol mismatch at {path}: "
                    f"stored={flattened.get('protocol_hash')} "
                    f"expected={expected_protocol_hash}; refusing mixed history"
                )
            by_step[int(payload["checkpoint_step"])] = flattened
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    missing_steps = sorted(set(required_steps or []) - set(by_step))
    if missing_steps:
        raise RuntimeError(
            "response-track refuses to append because fixed-protocol historical "
            f"backfill is missing steps {missing_steps}"
        )
    return [by_step[step] for step in sorted(by_step)]


def _write_history(output_dir: Path, history: list[dict[str, Any]]) -> Path:
    csv_path = output_dir / "response_track.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["protocol_hash", *TRACK_FIELDS]
        )
        writer.writeheader()
        writer.writerows(
            {
                **row,
                "garment_dino_by_type": json.dumps(
                    row.get("garment_dino_by_type", {}), sort_keys=True
                ),
            }
            for row in history
        )
    if not history:
        return output_dir / "response_track_curves.png"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in history]
    figure, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    series = (
        (0, "garment_union_concentration", "garment union concentration (descriptive)", "tab:blue"),
        (1, "hair_relative_response", "hair-ref swap / identity response", "tab:purple"),
        (1, "hair_ref_swap_concentration", "hair-ref swap concentration", "tab:red"),
        (2, "garment_dino", "Garment-DINO", "tab:green"),
        (2, "garment_hf_lpips", "Garment-HF-LPIPS (lower)", "tab:cyan"),
        (2, "hair_dino", "Hair-DINO", "tab:orange"),
        (3, "hair_lab_distance", "Hair LAB distance", "tab:brown"),
        (4, "appearance_gate", "appearance gate", "tab:blue"),
        (4, "hair_gate", "hair gate", "tab:purple"),
        (4, "head_pose_gate", "head-pose gate", "tab:green"),
    )
    for axis_index, field, label, color in series:
        points = [(step, _finite(row.get(field))) for step, row in zip(steps, history, strict=True)]
        points = [(step, value) for step, value in points if value is not None]
        if points:
            axes[axis_index].plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=label,
                color=color,
            )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, loc="best")
    axes[-1].set_xlabel("checkpoint step")
    figure.tight_layout()
    plot_path = output_dir / "response_track_curves.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return plot_path


def trajectory_decision(
    cfg: dict[str, Any], history: list[dict[str, Any]]
) -> dict[str, Any]:
    track_cfg = cfg.get("eval", {}).get("response_track", {})
    if not history:
        return {"status": "insufficient", "stop": False}
    latest = history[-1]
    baseline = history[0]
    step = int(latest["step"])
    slopes = {
        field: _recent_slope(history, field)
        for field in (
            "cloth_concentration",
            "hair_relative_response",
            "hair_ref_swap_concentration",
            "garment_dino",
            "hair_dino",
            "hair_lab_distance",
            "garment_hf_lpips",
        )
    }
    hair = _finite(latest.get("hair_relative_response"))
    hair_concentration = _finite(latest.get("hair_ref_swap_concentration"))
    garment_dino = _finite(latest.get("garment_dino"))
    head5 = _finite(latest.get("head5_distance"))
    body = _finite(latest.get("body_distance"))
    face_rate = _finite(latest.get("face_detection_rate"))
    swap_max = _finite(latest.get("swap_cosine_max"))
    baseline_swap_max = _finite(baseline.get("swap_cosine_max"))
    trend_intervals = int(track_cfg.get("regression_intervals_to_stop", 2))
    garment_declining = _consecutive_negative_intervals(
        history,
        "garment_dino",
        trend_intervals,
    )
    head_worsening = _consecutive_positive_intervals(
        history, "head5_distance", trend_intervals
    )
    body_worsening = _consecutive_positive_intervals(
        history, "body_distance", trend_intervals
    )
    garment_tolerance = float(track_cfg.get("garment_median_tolerance", 0.02))
    head_tolerance = float(track_cfg.get("head5_median_tolerance", 0.005))
    body_tolerance = float(track_cfg.get("body_median_tolerance", 0.005))
    garment_absolute_min = float(track_cfg.get("garment_absolute_min", -math.inf))
    head_absolute_max = float(track_cfg.get("head5_absolute_max", math.inf))
    body_absolute_max = float(track_cfg.get("body_absolute_max", math.inf))
    baseline_garment = _finite(baseline.get("garment_dino"))
    baseline_head5 = _finite(baseline.get("head5_distance"))
    baseline_body = _finite(baseline.get("body_distance"))
    baseline_types = _type_medians(baseline)
    latest_types = _type_medians(latest)
    missing_types = sorted(set(baseline_types) - set(latest_types))
    type_failures = sorted(
        kind
        for kind, base_value in baseline_types.items()
        if kind in latest_types
        and latest_types[kind] < base_value - garment_tolerance
    )
    def garment_row_ok(row: dict[str, Any]) -> bool:
        value = _finite(row.get("garment_dino"))
        kinds = _type_medians(row)
        return bool(
            value is not None
            and baseline_garment is not None
            and value >= max(
                garment_absolute_min,
                baseline_garment - garment_tolerance,
            )
            and not (set(baseline_types) - set(kinds))
            and all(
                kinds[kind] >= base_value - garment_tolerance
                for kind, base_value in baseline_types.items()
                if kind in kinds
            )
        )

    def head_row_ok(row: dict[str, Any]) -> bool:
        value = _finite(row.get("head5_distance"))
        return bool(
            value is not None
            and baseline_head5 is not None
            and value <= min(
                head_absolute_max,
                baseline_head5 + head_tolerance,
            )
        )

    def body_row_ok(row: dict[str, Any]) -> bool:
        value = _finite(row.get("body_distance"))
        return bool(
            value is not None
            and baseline_body is not None
            and value <= min(
                body_absolute_max,
                baseline_body + body_tolerance,
            )
        )

    garment_threshold_ok = garment_row_ok(latest)
    head_threshold_ok = head_row_ok(latest)
    body_threshold_ok = body_row_ok(latest)
    swap_limit = float(track_cfg.get("max_swap_cosine", 0.85))
    if baseline_swap_max is not None:
        swap_limit = min(
            swap_limit,
            baseline_swap_max
            + float(track_cfg.get("swap_regression_tolerance", 0.05)),
        )
    checks = {
        # Garment concentration is deliberately descriptive only. Cross-type donors made
        # area-normalized energy approximately one even when Garment-DINO and visual fidelity
        # were correct, so proxy/result conflicts are resolved in favor of result metrics.
        "garment_result_guard": (
            garment_threshold_ok and not garment_declining
        ),
        "hair_response_rising": (
            hair is not None
            and (slopes["hair_relative_response"] or 0.0) > 0.0
        ),
        "hair_ref_localized": (
            hair_concentration is not None
            and hair_concentration
            >= float(track_cfg.get("hair_concentration_min", 1.10))
        ),
        "hair_dino_rising": (
            slopes["hair_dino"] is not None
            and slopes["hair_dino"]
            > float(track_cfg.get("dino_positive_slope", 0.0))
        ),
        "hair_lab_falling": (
            slopes["hair_lab_distance"] is not None
            and slopes["hair_lab_distance"] < 0.0
        ),
        "head5_guard": head_threshold_ok and not head_worsening,
        "body_guard": body_threshold_ok and not body_worsening,
        "face_detection_guard": (
            face_rate is not None
            and face_rate >= float(track_cfg.get("min_face_detection_rate", 1.0))
        ),
        "identity_swap_guard": (
            swap_max is not None
            and swap_max <= swap_limit
        ),
    }
    decision: dict[str, Any] = {
        "status": "monitoring",
        "stop": False,
        "step": step,
        "baseline_step": int(baseline["step"]),
        "checks": checks,
        "slopes_per_500_steps": slopes,
        "quality_observations": {
            "garment_hf_lpips_latest": _finite(
                latest.get("garment_hf_lpips")
            ),
            "garment_hf_lpips_slope": slopes.get("garment_hf_lpips"),
            "garment_hf_improving": bool(
                slopes.get("garment_hf_lpips") is not None
                and slopes["garment_hf_lpips"] < 0.0
            ),
        },
        "garment_concentration_policy": "descriptive_only_not_a_stop_condition",
        "swap_cosine_limit": swap_limit,
        "fixed_median_policy": {
            "garment_global_baseline": baseline_garment,
            "garment_global_latest": garment_dino,
            "garment_type_baselines": baseline_types,
            "garment_type_latest": latest_types,
            "garment_type_failures": type_failures,
            "garment_missing_types": missing_types,
            "garment_tolerance": garment_tolerance,
            "garment_absolute_min": garment_absolute_min,
            "head5_baseline": baseline_head5,
            "head5_latest": head5,
            "head5_tolerance": head_tolerance,
            "head5_absolute_max": head_absolute_max,
            "body_baseline": baseline_body,
            "body_latest": body,
            "body_tolerance": body_tolerance,
            "body_absolute_max": body_absolute_max,
            "required_consecutive_worse_intervals": trend_intervals,
        },
    }
    trend_steps = {int(value) for value in track_cfg.get("trend_gate_steps", (1000, 1500))}
    if step in trend_steps:
        decision["status"] = "approved_trend" if all(checks.values()) else "trend_not_yet_approved"
        decision["auto_approval"] = bool(all(checks.values()))
    if len(history) == 1 and step == int(baseline["step"]):
        # The new hair-reference segment is intentionally untrained at step 2000.
        # Record its zero-step impact, but apply preserved-path guards only after
        # resumed optimization has had a checkpoint interval to fit the new route.
        decision.update({
            "status": "baseline_recorded",
            "stop": False,
            "baseline_guard_observation": {
                name: value
                for name, value in checks.items()
                if name in (
                    "garment_result_guard",
                    "head5_guard",
                    "body_guard",
                    "face_detection_guard",
                    "identity_swap_guard",
                )
            },
        })
        return decision
    guard_failures = []
    guard_warnings = []
    failed_checkpoints_to_stop = int(
        track_cfg.get("regression_checkpoints_to_stop", 2)
    )
    if not garment_threshold_ok:
        guard_warnings.append("garment fixed-set median below baseline tolerance")
        if garment_declining or _consecutive_failed_check(
            history, garment_row_ok, failed_checkpoints_to_stop
        ):
            guard_failures.append("garment_result_guard")
    if not head_threshold_ok:
        guard_warnings.append("head5 fixed-set median above baseline tolerance")
        if head_worsening or _consecutive_failed_check(
            history, head_row_ok, failed_checkpoints_to_stop
        ):
            guard_failures.append("head5_guard")
    if not body_threshold_ok:
        guard_warnings.append("body fixed-set median above baseline tolerance")
        if body_worsening or _consecutive_failed_check(
            history, body_row_ok, failed_checkpoints_to_stop
        ):
            guard_failures.append("body_guard")
    face_check = lambda row: (
        (value := _finite(row.get("face_detection_rate"))) is not None
        and value >= float(track_cfg.get("min_face_detection_rate", 1.0))
    )
    swap_check = lambda row: (
        (value := _finite(row.get("swap_cosine_max"))) is not None
        and value <= swap_limit
    )
    for name, row_check in (
        ("face_detection_guard", face_check),
        ("identity_swap_guard", swap_check),
    ):
        if not checks[name]:
            guard_warnings.append(f"{name} failed at one checkpoint")
            if _consecutive_failed_check(
                history, row_check, failed_checkpoints_to_stop
            ):
                guard_failures.append(name)
    decision["single_point_warnings"] = guard_warnings
    marker = (
        Path(cfg["experiment"]["output_root"])
        / cfg["experiment"]["id"]
        / str(track_cfg.get("visual_regression_marker", "VISUAL_REGRESSION"))
    )
    if marker.exists():
        guard_failures.append("human_visual_regression_marker")
    if guard_failures:
        decision.update({
            "status": "guard_regression",
            "stop": True,
            "plateau_reasons": guard_failures,
            "recommendations": [
                "restore the step-2000 garment/head/identity path before continuing hair work"
            ],
        })
        return decision

    hard_stop_step = int(track_cfg.get("hair_hard_stop_step", 3000))
    if step == hard_stop_step:
        hair_dino_flat = (
            slopes["hair_dino"] is None
            or slopes["hair_dino"]
            <= float(track_cfg.get("hair_dino_flat_slope", 0.001))
        )
        hair_lab_not_falling = (
            slopes["hair_lab_distance"] is None
            or slopes["hair_lab_distance"]
            >= float(track_cfg.get("hair_lab_flat_slope", -0.01))
        )
        if hair_dino_flat and hair_lab_not_falling:
            decision.update({
                "status": "hair_plateau",
                "stop": True,
                "plateau_reasons": [
                    "Hair-DINO remains flat and Hair-LAB distance is not declining"
                ],
                "recommendations": [
                    "use full-resolution hair reference tokens",
                    "raise lambda_hair",
                    "raise paired w_hair from 2 to 4",
                ],
            })
        else:
            decision["status"] = "passed_hair_gate"
    elif step > hard_stop_step:
        decision["status"] = "post_hair_gate_monitoring"
    return decision


def _snapshot_protocol_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    protocol = payload.get("protocol", {})
    return protocol.get("protocol_hash", protocol.get("hash"))


def _auto_backfill_history(
    track_cfg: dict[str, Any],
    required_steps: list[int],
    expected_protocol_hash: str,
) -> None:
    if not bool(track_cfg.get("auto_backfill", False)):
        return
    output = Path(
        track_cfg.get(
            "rebaseline_output_dir", "artifacts/rebaseline_fixed_set"
        )
    )
    checkpoints = {
        int(step): Path(path)
        for step, path in track_cfg.get(
            "historical_checkpoints", {}
        ).items()
    }
    config_path = str(
        track_cfg.get(
            "auto_backfill_config",
            "configs/spatial_warmup_resume_hair_incontext.yaml",
        )
    )
    changed = False
    for step in required_steps:
        snapshot = output / f"step{step}.json"
        if _snapshot_protocol_hash(snapshot) == expected_protocol_hash:
            continue
        checkpoint = checkpoints.get(step)
        if checkpoint is None or not (checkpoint / "READY").exists():
            raise RuntimeError(
                f"cannot auto-backfill watcher step {step}: checkpoint is "
                f"missing or not READY ({checkpoint})"
            )
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "tools/rebaseline_watcher.py"),
            "--config", config_path,
            "--ckpt", str(checkpoint),
            "--device", str(track_cfg.get("backfill_device", "cuda:0")),
            "--output", str(output),
            "--overwrite-images",
        ]
        subprocess.run(command, check=True)
        if _snapshot_protocol_hash(snapshot) != expected_protocol_hash:
            raise RuntimeError(
                f"auto-backfill step {step} completed without the expected "
                f"protocol hash {expected_protocol_hash}"
            )
        changed = True
    if changed:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "tools/rebaseline_watcher.py"),
                "--config", config_path,
                "--output", str(output),
                "--report-only",
            ],
            check=True,
        )


def finalize_snapshot(
    cfg: dict[str, Any],
    snapshot: dict[str, Any],
    watcher_metrics: dict[str, Any] | None = None,
    guards: dict[str, Any] | None = None,
) -> dict[str, Any]:
    track_cfg = cfg.get("eval", {}).get("response_track", {})
    output_dir = Path(
        track_cfg.get(
            "output_dir", "artifacts/response_track"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot["watcher_metrics"] = watcher_metrics or {}
    snapshot["guards"] = guards or {}
    fixed_protocol = watcher_eval_set_from_config(cfg)
    snapshot_protocol_hash = snapshot.get("protocol", {}).get(
        "protocol_hash"
    )
    if snapshot_protocol_hash != fixed_protocol["protocol_hash"]:
        raise RuntimeError(
            "current response snapshot was measured on a different sample "
            f"protocol: stored={snapshot_protocol_hash}, "
            f"expected={fixed_protocol['protocol_hash']}"
        )
    current_step = int(snapshot["checkpoint_step"])
    required_steps = [
        int(value)
        for value in track_cfg.get("required_history_steps", [])
        if int(value) < current_step
    ]
    historical_snapshots = list(track_cfg.get("historical_snapshots", []))
    # A protocol change invalidates every historical point. Backfill is normally
    # prepared before training; the watcher never mixes protocols or silently
    # appends a point when any required checkpoint is missing.
    _auto_backfill_history(
        track_cfg, required_steps, fixed_protocol["protocol_hash"]
    )
    history = _load_history(
        output_dir,
        track_cfg.get("baseline_snapshot"),
        historical_snapshots=historical_snapshots,
        expected_protocol_hash=fixed_protocol["protocol_hash"],
        required_steps=required_steps,
    )
    path = output_dir / f"step{int(snapshot['checkpoint_step'])}.json"
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    history = _load_history(
        output_dir,
        track_cfg.get("baseline_snapshot"),
        historical_snapshots=historical_snapshots,
        expected_protocol_hash=fixed_protocol["protocol_hash"],
        required_steps=required_steps,
    )
    plot_path = _write_history(output_dir, history)
    decision = trajectory_decision(cfg, history)
    snapshot["trajectory"] = decision
    snapshot["cumulative"] = {
        "csv": str(output_dir / "response_track.csv"),
        "plot": str(plot_path),
        "points": len(history),
    }
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if decision.get("auto_approval"):
        approval = output_dir / f"trajectory_approval_step{int(snapshot['checkpoint_step'])}.json"
        approval.write_text(
            json.dumps(
                {
                    "checkpoint": snapshot["checkpoint"],
                    "checkpoint_step": snapshot["checkpoint_step"],
                    "decision": decision,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return snapshot


def run_cli(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    seed_everything(int(cfg["experiment"]["seed"]))
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("the 6144-token response probe requires CUDA")
    torch.cuda.set_device(device)
    dtype = choose_dtype(cfg["model"]["precision"])
    dataset = PairedWarmupDataset(cfg, "val", require_coverage=True)
    transformer, controlnet, _, adapter, pulid, _ = load_components(
        cfg, device, dtype
    )
    model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg)
    step = load_checkpoint(Path(args.ckpt), model)
    model.eval()
    snapshot = compute_response_snapshot(
        cfg, model, dataset, Path(args.ckpt), step, device, dtype
    )
    return finalize_snapshot(cfg, snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track spatial-condition response at every 500-step checkpoint"
    )
    parser.add_argument("--config", default="configs/spatial_warmup_resume_hair.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    result = run_cli(args)
    print(
        json.dumps(
            {
                "step": result["checkpoint_step"],
                "response": result["response"],
                "trajectory": result["trajectory"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
