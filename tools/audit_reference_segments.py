from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conditions import (
    choose_dtype,
    load_yaml,
    make_image_ids,
    seed_everything,
    unpack_latents,
)
from dataset import PairedWarmupDataset
from spatial_conditions import (
    downsample_reference_tokens,
    image_mask_to_token_weights,
    make_hair_reference_image_ids,
    make_reference_image_ids,
    masked_garment_image,
    masked_hair_image,
    pad_control_samples_for_reference,
)
from train_paired import WarmupFlowModel, load_checkpoint, load_components
from watcher_protocol import watcher_eval_set_from_config


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tensor_range(ids: torch.Tensor) -> dict[str, Any]:
    value = ids.detach().float().cpu()
    return {
        "count": int(value.shape[0]),
        "batch_min": float(value[:, 0].min()),
        "batch_max": float(value[:, 0].max()),
        "y_min": float(value[:, 1].min()),
        "y_max": float(value[:, 1].max()),
        "x_min": float(value[:, 2].min()),
        "x_max": float(value[:, 2].max()),
    }


def _coordinate_set(ids: torch.Tensor) -> set[tuple[float, float]]:
    values = ids.detach().float().cpu().numpy()
    return {(float(row[1]), float(row[2])) for row in values}


def audit_position_ids(
    model: WarmupFlowModel,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    image_ids = make_image_ids(model.width, model.height, device, dtype)
    garment_ids = make_reference_image_ids(
        model.width,
        model.height,
        model.reference_x_offset,
        device,
        dtype,
        stride=model.reference_stride,
    )
    hair_ids = make_hair_reference_image_ids(
        model.width,
        model.height,
        model.hair_reference_y_offset,
        device,
        dtype,
        stride=model.hair_reference_stride,
    )
    sets = {
        "image": _coordinate_set(image_ids),
        "garment_reference": _coordinate_set(garment_ids),
        "hair_reference": _coordinate_set(hair_ids),
    }
    overlaps = {
        "image_garment": len(sets["image"] & sets["garment_reference"]),
        "image_hair": len(sets["image"] & sets["hair_reference"]),
        "garment_hair": len(sets["garment_reference"] & sets["hair_reference"]),
    }
    if any(overlaps.values()):
        suggested_x = float(max(x for _, x in sets["image"]) + 16.0)
        suggested_y = float(
            max(y for name in ("image", "garment_reference") for y, _ in sets[name])
            + 16.0
        )
        suggestion = {
            "garment_x_offset": suggested_x,
            "hair_y_offset": suggested_y,
        }
    else:
        suggestion = None
    return {
        "segments": {
            "image": _tensor_range(image_ids),
            "garment_reference": _tensor_range(garment_ids),
            "hair_reference": _tensor_range(hair_ids),
        },
        "overlaps": overlaps,
        "disjoint": not any(overlaps.values()),
        "suggested_offsets_if_overlapping": suggestion,
    }


def _decode_packed(
    vae: torch.nn.Module,
    tokens: torch.Tensor,
    width: int,
    height: int,
) -> np.ndarray:
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    value = tokens.to(device=device, dtype=dtype)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    latents = unpack_latents(value, width, height)
    latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    decoded = vae.decode(latents, return_dict=False)[0]
    image = (
        ((decoded[0].detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .permute(1, 2, 0)
        .byte()
        .cpu()
        .numpy()
    )
    return image


def _edge_leak_stats(
    intended: np.ndarray,
    decoded: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    binary = np.asarray(mask, dtype=np.float32) >= 0.5
    kernel = np.ones((5, 5), dtype=np.uint8)
    dilated = cv2.dilate(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
    eroded = cv2.erode(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
    outer = dilated & ~binary
    inner = binary & ~eroded
    ring = dilated & ~eroded
    far_outside = ~dilated

    # Infer the actual neutral canvas value from pixels outside the mask. This
    # must not be hard-coded to the historical 127 canvas: quality-repair runs
    # use the calibrated VAE fixed point (227) to match mannequin backgrounds.
    outside_pixels = intended[~binary]
    neutral = (
        np.median(outside_pixels.astype(np.float32), axis=0)
        if len(outside_pixels)
        else np.asarray([127.0, 127.0, 127.0], dtype=np.float32)
    )
    intended_delta = np.max(
        np.abs(intended.astype(np.float32) - neutral[None, None, :]), axis=2
    )
    # Compare against the intended masked canvas, not an independently encoded
    # all-gray image. A uniform gray canvas is far outside the FLUX VAE training
    # distribution and can decode with a severe color cast even when reference
    # canvases reconstruct correctly, which creates a false 100% leak signal.
    decoded_delta = np.max(
        np.abs(decoded.astype(np.float32) - intended.astype(np.float32)), axis=2
    )

    def ratio(values: np.ndarray, region: np.ndarray) -> float:
        if not bool(region.any()):
            return 0.0
        return float((values[region] > float(threshold)).mean())

    return {
        "mask_area_fraction": float(binary.mean()),
        "intended_outer_ring_non_gray_ratio": ratio(intended_delta, outer),
        "decoded_inner_ring_changed_ratio": ratio(decoded_delta, inner),
        "decoded_outer_ring_changed_ratio": ratio(decoded_delta, outer),
        "decoded_full_2px_ring_changed_ratio": ratio(decoded_delta, ring),
        "decoded_far_outside_changed_ratio": ratio(decoded_delta, far_outside),
        "decoded_far_outside_mean_abs_delta": float(
            np.abs(decoded.astype(np.float32) - intended.astype(np.float32))[
                far_outside
            ].mean()
            if bool(far_outside.any())
            else 0.0
        ),
    }


def _background_color_stats(
    root: Path,
    sample_ids: list[str],
    width: int,
    height: int,
) -> dict[str, list[float]]:
    pixels: list[np.ndarray] = []
    for sample_id in sample_ids:
        image_path = next(
            path
            for path in (root / "images/mannequin").iterdir()
            if path.stem == sample_id
        )
        parsing_path = next(
            path
            for path in (root / "human_parsing/fashn/masks/mannequin").iterdir()
            if path.stem == sample_id
        )
        image = Image.open(image_path).convert("RGB").resize(
            (width, height), Image.Resampling.BICUBIC
        )
        labels = Image.open(parsing_path).convert("L").resize(
            (width, height), Image.Resampling.NEAREST
        )
        array = np.asarray(image, dtype=np.uint8)
        background = np.asarray(labels, dtype=np.uint8) == 0
        if bool(background.any()):
            pixels.append(array[background])
    if not pixels:
        raise RuntimeError("could not collect mannequin parsing background pixels")
    values = np.concatenate(pixels, axis=0).astype(np.float32)
    return {
        "mean_rgb": values.mean(axis=0).tolist(),
        "median_rgb": np.median(values, axis=0).tolist(),
        "p10_rgb": np.percentile(values, 10, axis=0).tolist(),
        "p90_rgb": np.percentile(values, 90, axis=0).tolist(),
    }


def _caption(image: Image.Image, label: str, height: int = 34) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    canvas.paste(image, (0, height))
    ImageDraw.Draw(canvas).text((8, 9), label, fill="black")
    return canvas


def _reference_panel(
    output: Path,
    sample_id: str,
    garment_intended: Image.Image,
    garment_decoded: np.ndarray,
    hair_intended: Image.Image,
    hair_decoded: np.ndarray,
) -> None:
    width = 288
    items = [
        _caption(garment_intended.resize((width, 384)), "garment intended"),
        _caption(Image.fromarray(garment_decoded).resize((width, 384)), "garment VAE decode"),
        _caption(hair_intended.resize((width, 384)), "hair intended"),
        _caption(Image.fromarray(hair_decoded).resize((width, 384)), "hair VAE decode"),
    ]
    panel = Image.new("RGB", (sum(item.width for item in items), max(item.height for item in items)), "white")
    x = 0
    for item in items:
        panel.paste(item, (x, 0))
        x += item.width
    panel.save(output / f"{sample_id}_reference_decode.png")


def _segment_rms(
    samples: list[torch.Tensor] | None,
    start: int,
    end: int,
) -> float:
    if not samples or end <= start:
        return 0.0
    values = [sample[:, start:end].detach().float().square().mean() for sample in samples]
    return float(torch.stack(values).mean().sqrt().cpu())


def _control_residual_row(
    model: WarmupFlowModel,
    batch: dict[str, Any],
    sample_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    prompt = batch["prompt_embeds"].to(device=device, dtype=dtype)
    pooled = batch["pooled_prompt_embeds"].to(device=device, dtype=dtype)
    if prompt.ndim == 2:
        prompt = prompt.unsqueeze(0)
    if pooled.ndim == 1:
        pooled = pooled.unsqueeze(0)
    generator = torch.Generator(device=device).manual_seed(20260820 + sample_index)
    z_tau = torch.randn(
        1,
        model.image_token_count,
        64,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    tau = torch.full((1,), 0.5, device=device, dtype=dtype)
    img_ids = make_image_ids(model.width, model.height, device, dtype)
    control = model._controlnet_forward(
        z_tau,
        tau,
        prompt,
        pooled,
        batch["pose_latents"],
        img_ids,
    )
    garment_count = model.image_token_count // (model.reference_stride**2)
    hair_count = model.image_token_count // (model.hair_reference_stride**2)
    total_reference = garment_count + hair_count
    padded = (
        pad_control_samples_for_reference(control[0], total_reference),
        pad_control_samples_for_reference(control[1], total_reference),
    )
    image_end = model.image_token_count
    garment_end = image_end + garment_count
    hair_end = garment_end + hair_count
    return {
        "sample_id": str(batch["sample_id"]),
        "image_block_rms": _segment_rms(padded[0], 0, image_end),
        "garment_block_rms": _segment_rms(padded[0], image_end, garment_end),
        "hair_block_rms": _segment_rms(padded[0], garment_end, hair_end),
        "image_single_rms": _segment_rms(padded[1], 0, image_end),
        "garment_single_rms": _segment_rms(padded[1], image_end, garment_end),
        "hair_single_rms": _segment_rms(padded[1], garment_end, hair_end),
        "padded_token_count": hair_end,
    }


def _attention_regions(root: Path, sample_id: str, width: int, height: int) -> dict[str, torch.Tensor]:
    path = root / "derived/region_masks_z" / f"{sample_id}.npz"
    with np.load(path, allow_pickle=False) as payload:
        cloth = np.asarray(payload["cloth_safe_z"], dtype=np.float32)
        face = np.asarray(payload["face_z"], dtype=np.float32)
        hair = np.asarray(payload["hair_z"], dtype=np.float32)
    labels_path = root / "human_parsing/fashn/masks/human" / f"{sample_id}.png"
    labels = np.asarray(Image.open(labels_path).convert("L"), dtype=np.uint8)
    background = image_mask_to_token_weights(
        (labels == 0).astype(np.float32), width, height
    )[0, :, 0]
    return {
        "cloth": torch.from_numpy(cloth),
        "face": torch.from_numpy(face * (1.0 - hair)),
        "hair": torch.from_numpy(hair),
        "background": background,
    }


class SegmentAttentionRecorder:
    """Read-only Flux processor wrapper that aggregates exact segment attention mass."""

    def __init__(
        self,
        base_processor: Any,
        layer: str,
        image_tokens: int,
        garment_tokens: int,
        hair_tokens: int,
        query_chunk: int,
        rows: list[dict[str, Any]],
    ) -> None:
        self.base = base_processor
        self.layer = layer
        self.image_tokens = int(image_tokens)
        self.garment_tokens = int(garment_tokens)
        self.hair_tokens = int(hair_tokens)
        self.query_chunk = int(query_chunk)
        self.rows = rows
        self.metadata: dict[str, Any] | None = None
        self.regions: dict[str, torch.Tensor] = {}

    def set_capture(self, metadata: dict[str, Any], regions: dict[str, torch.Tensor]) -> None:
        self.metadata = dict(metadata)
        self.regions = regions

    @torch.no_grad()
    def _capture(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        image_rotary_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        if self.metadata is None or encoder_hidden_states is None:
            return
        from diffusers.models.embeddings import apply_rotary_emb
        from diffusers.models.transformers.transformer_flux import _get_qkv_projections

        query, key, _value, encoder_query, encoder_key, _encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        encoder_query = attn.norm_added_q(
            encoder_query.unflatten(-1, (attn.heads, -1))
        )
        encoder_key = attn.norm_added_k(
            encoder_key.unflatten(-1, (attn.heads, -1))
        )
        query = torch.cat((encoder_query, query), dim=1)
        key = torch.cat((encoder_key, key), dim=1)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        text_tokens = int(encoder_hidden_states.shape[1])
        expected_hidden = self.image_tokens + self.garment_tokens + self.hair_tokens
        if int(hidden_states.shape[1]) != expected_hidden:
            raise RuntimeError(
                f"{self.layer}: hidden tokens={hidden_states.shape[1]}, expected={expected_hidden}"
            )
        destinations = {
            "text": (0, text_tokens),
            "self_image": (text_tokens, text_tokens + self.image_tokens),
            "garment_reference": (
                text_tokens + self.image_tokens,
                text_tokens + self.image_tokens + self.garment_tokens,
            ),
            "hair_reference": (
                text_tokens + self.image_tokens + self.garment_tokens,
                text_tokens + expected_hidden,
            ),
        }
        image_query = query[
            :, text_tokens : text_tokens + self.image_tokens
        ].permute(0, 2, 1, 3)
        keys = key.permute(0, 2, 1, 3)
        key_t = keys.transpose(-1, -2)
        numerators: dict[tuple[str, str], float] = defaultdict(float)
        denominators: dict[str, float] = defaultdict(float)
        for start in range(0, self.image_tokens, self.query_chunk):
            end = min(self.image_tokens, start + self.query_chunk)
            scale = float(image_query.shape[-1]) ** -0.5
            scores = torch.matmul(image_query[:, :, start:end], key_t) * scale
            probs = scores.float().softmax(dim=-1).mean(dim=1)[0]
            for region, weights_cpu in self.regions.items():
                weights = weights_cpu[start:end].to(
                    device=probs.device, dtype=torch.float32
                ).clamp(0.0, 1.0)
                denominator = float(weights.sum().cpu())
                denominators[region] += denominator
                if denominator <= 0.0:
                    continue
                for destination, (left, right) in destinations.items():
                    mass = probs[:, left:right].sum(dim=1)
                    numerators[(region, destination)] += float(
                        (mass * weights).sum().cpu()
                    )
            del scores, probs
        for region in self.regions:
            denominator = max(denominators[region], 1e-12)
            row = {
                **self.metadata,
                "layer": self.layer,
                "region": region,
            }
            for destination in destinations:
                row[f"attention_to_{destination}"] = (
                    numerators[(region, destination)] / denominator
                )
            row["attention_sum"] = sum(
                float(row[f"attention_to_{destination}"])
                for destination in destinations
            )
            self.rows.append(row)

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Any:
        self._capture(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        return self.base(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )


def _install_attention_recorders(
    model: WarmupFlowModel,
    layer_indices: list[int],
    rows: list[dict[str, Any]],
    query_chunk: int,
) -> tuple[list[SegmentAttentionRecorder], list[tuple[Any, Any]]]:
    garment_count = model.image_token_count // (model.reference_stride**2)
    hair_count = model.image_token_count // (model.hair_reference_stride**2)
    recorders = []
    originals = []
    blocks = list(model.transformer.transformer_blocks)
    for index in layer_indices:
        if index < 0 or index >= len(blocks):
            raise IndexError(f"attention layer {index} is outside [0,{len(blocks) - 1}]")
        attention = blocks[index].attn
        original = attention.processor
        recorder = SegmentAttentionRecorder(
            original,
            f"transformer_blocks.{index}",
            model.image_token_count,
            garment_count,
            hair_count,
            query_chunk,
            rows,
        )
        attention.set_processor(recorder)
        originals.append((attention, original))
        recorders.append(recorder)
    return recorders, originals


def _restore_attention_processors(originals: list[tuple[Any, Any]]) -> None:
    for attention, processor in originals:
        attention.set_processor(processor)


@torch.inference_mode()
def _run_attention_probe(
    cfg: dict[str, Any],
    model: WarmupFlowModel,
    dataset: PairedWarmupDataset,
    sample_ids: list[str],
    device: torch.device,
    dtype: torch.dtype,
    taus: list[float],
    layers: list[int],
    query_chunk: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recorders, originals = _install_attention_recorders(
        model, layers, rows, query_chunk
    )
    root = Path(cfg["data"]["root"])
    try:
        for sample_offset, sample_id in enumerate(sample_ids):
            batch = dataset[dataset.ids.index(sample_id)]
            prompt = batch["prompt_embeds"].to(device=device, dtype=dtype)
            pooled = batch["pooled_prompt_embeds"].to(device=device, dtype=dtype)
            if prompt.ndim == 2:
                prompt = prompt.unsqueeze(0)
            if pooled.ndim == 1:
                pooled = pooled.unsqueeze(0)
            cond = model._condition_tokens(
                prompt,
                batch["appearance"].to(device=device, dtype=dtype).unsqueeze(0),
                batch["garment"].to(device=device, dtype=dtype).unsqueeze(0),
                batch["head_pose"].to(device=device, dtype=dtype).unsqueeze(0),
            )
            generator = torch.Generator(device=device).manual_seed(20260820 + sample_offset)
            z_tau = torch.randn(
                1,
                model.image_token_count,
                64,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            img_ids = make_image_ids(model.width, model.height, device, dtype)
            regions = _attention_regions(
                root, sample_id, model.width, model.height
            )
            for tau_value in taus:
                tau = torch.full((1,), float(tau_value), device=device, dtype=dtype)
                control = model._controlnet_forward(
                    z_tau,
                    tau,
                    prompt,
                    pooled,
                    batch["pose_latents"],
                    img_ids,
                )
                metadata = {"sample_id": sample_id, "tau": float(tau_value)}
                for recorder in recorders:
                    recorder.set_capture(metadata, regions)
                model._transformer_forward(
                    z_tau,
                    tau,
                    cond,
                    batch["pulid_id_embed"],
                    control,
                    pooled=pooled,
                    img_ids=img_ids,
                    garment_ref_latents=batch["garment_ref_latents"],
                    hair_ref_latents=batch["hair_ref_latents"],
                )
    finally:
        _restore_attention_processors(originals)
    return rows


def _attention_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["region"])].append(row)
    summary = {}
    for region, values in sorted(grouped.items()):
        summary[region] = {
            key: float(np.mean([float(row[key]) for row in values]))
            for key in (
                "attention_to_text",
                "attention_to_self_image",
                "attention_to_garment_reference",
                "attention_to_hair_reference",
                "attention_sum",
            )
        }
    return summary


def _report(
    output: Path,
    checkpoint: Path,
    position: dict[str, Any],
    control_rows: list[dict[str, Any]],
    content_rows: list[dict[str, Any]],
    attention_rows: list[dict[str, Any]],
    background_color: dict[str, list[float]],
    reference_neutral_gray: int,
    far_leak_threshold: float,
    background_attention_threshold: float,
    background_enrichment_threshold: float,
    neutral_gray_tolerance: float,
) -> dict[str, Any]:
    max_control_reference = max(
        (
            max(
                float(row["garment_block_rms"]),
                float(row["hair_block_rms"]),
                float(row["garment_single_rms"]),
                float(row["hair_single_rms"]),
            )
            for row in control_rows
        ),
        default=0.0,
    )
    max_far_leak = max(
        (
            float(row["decoded_far_outside_changed_ratio"])
            for row in content_rows
        ),
        default=0.0,
    )
    attention = _attention_summary(attention_rows)
    background_reference = (
        float(attention.get("background", {}).get("attention_to_garment_reference", 0.0))
        + float(attention.get("background", {}).get("attention_to_hair_reference", 0.0))
    )
    background_median = np.asarray(
        background_color["median_rgb"], dtype=np.float64
    )
    neutral_gray_delta = float(
        np.max(np.abs(background_median - float(reference_neutral_gray)))
    )
    foreground_reference_values = []
    for region in ("cloth", "face", "hair"):
        values = attention.get(region)
        if values:
            foreground_reference_values.append(
                float(values.get("attention_to_garment_reference", 0.0))
                + float(values.get("attention_to_hair_reference", 0.0))
            )
    foreground_reference = float(
        np.mean(foreground_reference_values)
    ) if foreground_reference_values else 0.0
    background_enrichment = background_reference / max(
        foreground_reference, 1e-12
    )
    background_attention_risk = bool(
        background_reference > background_attention_threshold
        and (
            neutral_gray_delta > neutral_gray_tolerance
            or background_enrichment > background_enrichment_threshold
        )
    )
    reasons = []
    if not position["disjoint"]:
        reasons.append("reference position IDs overlap")
    if max_control_reference > 1e-8:
        reasons.append(
            f"ControlNet reference residual RMS is non-zero ({max_control_reference:.3e})"
        )
    if max_far_leak > far_leak_threshold:
        reasons.append(
            f"decoded reference changes far outside mask ({max_far_leak:.3f} > {far_leak_threshold:.3f})"
        )
    if background_attention_risk:
        reasons.append(
            "background reference attention is both high and leakage-prone "
            f"(mass={background_reference:.3f}, enrichment={background_enrichment:.3f}, "
            f"neutral_delta={neutral_gray_delta:.1f})"
        )
    conclusion = "LEAK-FOUND" if reasons else "NO-LEAK"
    payload = {
        "checkpoint": str(checkpoint),
        "position_ids": position,
        "controlnet": {
            "rows": control_rows,
            "max_reference_rms": max_control_reference,
        },
        "reference_content": {
            "rows": content_rows,
            "max_far_outside_changed_ratio": max_far_leak,
        },
        "attention": {
            "rows": attention_rows,
            "summary": attention,
            "background_total_reference_attention": background_reference,
            "foreground_total_reference_attention": foreground_reference,
            "background_reference_enrichment": background_enrichment,
            "background_attention_risk": background_attention_risk,
        },
        "mannequin_background_color": background_color,
        "reference_neutral_gray": int(reference_neutral_gray),
        "reference_neutral_gray_max_abs_delta": neutral_gray_delta,
        "thresholds": {
            "far_outside_changed_ratio": far_leak_threshold,
            "background_reference_attention": background_attention_threshold,
            "background_reference_enrichment": background_enrichment_threshold,
            "neutral_gray_max_abs_delta": neutral_gray_tolerance,
        },
        "conclusion": conclusion,
        "reasons": reasons,
    }
    (output / "audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Reference Segment Audit",
        "",
        f"**Conclusion: {conclusion}.**",
        f"Checkpoint: `{checkpoint}`",
        "",
        "## Position IDs",
        "",
        f"- disjoint: `{position['disjoint']}`",
        f"- overlaps: `{position['overlaps']}`",
    ]
    for name, values in position["segments"].items():
        lines.append(
            f"- {name}: N={values['count']}, y=[{values['y_min']:.0f},{values['y_max']:.0f}], "
            f"x=[{values['x_min']:.0f},{values['x_max']:.0f}]"
        )
    lines.extend(
        [
            "",
            "## ControlNet Residuals",
            "",
            f"Maximum RMS on either reference segment: `{max_control_reference:.3e}`.",
            "The required invariant is exact zero; ControlNet may affect only the first 3072 image tokens.",
            "",
            "## Reference Content",
            "",
            f"Maximum decoded far-outside changed-pixel ratio: `{max_far_leak:.4f}`.",
            "The changed-pixel test compares each VAE decode against its intended masked canvas. An independently encoded all-gray canvas is not used because it is an out-of-distribution VAE input and produced a severe false color cast in the audit smoke test.",
            "",
            "| sample | route | area | outer 2px changed | far outside changed | far outside MAD |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in content_rows:
        lines.append(
            f"| {row['sample_id']} | {row['route']} | {float(row['mask_area_fraction']):.4f} | "
            f"{float(row['decoded_outer_ring_changed_ratio']):.4f} | "
            f"{float(row['decoded_far_outside_changed_ratio']):.4f} | "
            f"{float(row['decoded_far_outside_mean_abs_delta']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Neutral Background Audit",
            "",
            f"- configured reference gray: `RGB({reference_neutral_gray},{reference_neutral_gray},{reference_neutral_gray})`",
            f"- mannequin parsing-background median: `{[round(value, 2) for value in background_color['median_rgb']]}`",
            f"- mannequin parsing-background mean: `{[round(value, 2) for value in background_color['mean_rgb']]}`",
            f"- maximum channel difference from background median: `{neutral_gray_delta:.2f}`",
            "- interpretation: "
            + (
                "the reference canvas is materially mismatched to the training-image background."
                if neutral_gray_delta > 20.0
                else "the reference canvas is aligned with the training-image background statistic."
            ),
            "",
            "## Cross-Segment Attention",
            "",
            "Values are exact softmax mass averaged over heads, selected layers, samples, and tau values.",
            "",
            "| image query region | text | image/self | garment ref | hair ref | sum |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for region, values in attention.items():
        lines.append(
            f"| {region} | {values['attention_to_text']:.4f} | "
            f"{values['attention_to_self_image']:.4f} | "
            f"{values['attention_to_garment_reference']:.4f} | "
            f"{values['attention_to_hair_reference']:.4f} | "
            f"{values['attention_sum']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Background total reference attention: `{background_reference:.4f}`.",
            f"Foreground-region mean reference attention: `{foreground_reference:.4f}`.",
            f"Background / foreground attention enrichment: `{background_enrichment:.4f}`.",
            "Raw reference mass is diagnostic only because it scales with the 3840 appended keys; "
            "it becomes a leakage reason only when the neutral canvas is mismatched or background "
            "queries are enriched relative to cloth/face/hair queries.",
            "",
            "## Decision",
            "",
            f"- conclusion: **{conclusion}**",
            f"- reasons: `{reasons or ['none']}`",
            "- remediation order when leakage is present: erode the reference mask by 2-3 pixels and rebuild only reference keys; then adjust position offsets; finally calibrate neutral gray to the training-background statistic.",
            "- decoded panels: `*_reference_decode.png`; manually verify garment limbs/head contamination and hair-only support before accepting the automatic conclusion.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    garment_cfg = cfg.setdefault("model", {}).setdefault(
        "spatial_conditions", {}
    ).setdefault("garment_reference", {})
    hair_ref_cfg = cfg["model"]["spatial_conditions"].setdefault(
        "hair", {}
    ).setdefault("reference", {})
    if args.garment_x_offset is not None:
        garment_cfg["x_offset"] = float(args.garment_x_offset)
    if args.hair_y_offset is not None:
        hair_ref_cfg["y_offset"] = float(args.hair_y_offset)
    seed_everything(int(cfg["experiment"]["seed"]))
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("reference attention audit requires a CUDA device")
    torch.cuda.set_device(device)
    dtype = choose_dtype(cfg["model"]["precision"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        Path(cfg["experiment"]["output_root"])
        / cfg["experiment"]["id"]
        / "checkpoints/final"
    )
    dataset = PairedWarmupDataset(cfg, "val", require_coverage=True)
    protocol = watcher_eval_set_from_config(cfg)
    sample_ids = [str(value) for value in protocol["sample_ids"][: args.samples]]
    missing = sorted(set(sample_ids) - set(dataset.ids))
    if missing:
        raise RuntimeError(f"audit samples are absent from val cache: {missing}")

    transformer, controlnet, vae, adapter, pulid, _notes = load_components(
        cfg, device, dtype
    )
    if vae is None:
        raise RuntimeError("reference decode audit requires model.load_vae_in_train=true")
    model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg)
    load_checkpoint(checkpoint, model)
    model.eval()
    position = audit_position_ids(model, device, dtype)

    root = Path(cfg["data"]["root"])
    background_color = _background_color_stats(
        root, sample_ids, model.width, model.height
    )
    content_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(sample_ids):
        batch = dataset[dataset.ids.index(sample_id)]
        control_rows.append(
            _control_residual_row(
                model, batch, sample_index, device, dtype
            )
        )
        garment_intended, garment_mask = masked_garment_image(
            root,
            sample_id,
            cfg["data"]["resolution"],
            neutral_gray=int(
                cfg.get("cache", {}).get(
                    "reference_neutral_gray",
                    cfg.get("cache", {}).get("neutral_gray", 127),
                )
            ),
            erosion_px=int(
                cfg.get("cache", {}).get("reference_mask_erosion_px", 0)
            ),
        )
        hair_intended, hair_mask, _empty = masked_hair_image(
            root,
            sample_id,
            cfg["data"]["resolution"],
            neutral_gray=int(
                cfg.get("cache", {}).get(
                    "reference_neutral_gray",
                    cfg.get("cache", {}).get("neutral_gray", 127),
                )
            ),
            hair_label=int(cfg.get("cache", {}).get("hair_label", 2)),
            min_area_fraction=0.0,
            erosion_px=int(
                cfg.get("cache", {}).get("reference_mask_erosion_px", 0)
            ),
        )
        garment_decoded = _decode_packed(
            vae, batch["garment_ref_latents"], model.width, model.height
        )
        hair_decoded = _decode_packed(
            vae, batch["hair_ref_latents"], model.width, model.height
        )
        garment_decoded_path = output / f"{sample_id}_garment_decoded.png"
        hair_decoded_path = output / f"{sample_id}_hair_decoded.png"
        Image.fromarray(garment_decoded).save(garment_decoded_path)
        Image.fromarray(hair_decoded).save(hair_decoded_path)
        _reference_panel(
            output,
            sample_id,
            garment_intended,
            garment_decoded,
            hair_intended,
            hair_decoded,
        )
        for route, intended, decoded, mask in (
            (
                "garment",
                np.asarray(garment_intended),
                garment_decoded,
                garment_mask,
            ),
            ("hair", np.asarray(hair_intended), hair_decoded, hair_mask),
        ):
            content_rows.append(
                {
                    "sample_id": sample_id,
                    "route": route,
                    **_edge_leak_stats(
                        intended,
                        decoded,
                        mask,
                        args.non_gray_threshold,
                    ),
                }
            )

    attention_rows = _run_attention_probe(
        cfg,
        model,
        dataset,
        sample_ids[: args.attention_samples],
        device,
        dtype,
        [float(value) for value in args.taus],
        [int(value) for value in args.layers],
        args.query_chunk,
    )
    _write_csv(output / "controlnet_segments.csv", control_rows)
    _write_csv(output / "reference_content.csv", content_rows)
    _write_csv(output / "attention_segments.csv", attention_rows)
    payload = _report(
        output,
        checkpoint,
        position,
        control_rows,
        content_rows,
        attention_rows,
        background_color,
        int(
            cfg.get("cache", {}).get(
                "reference_neutral_gray",
                cfg.get("cache", {}).get("neutral_gray", 127),
            )
        ),
        args.far_leak_threshold,
        args.background_attention_threshold,
        args.background_enrichment_threshold,
        args.neutral_gray_tolerance,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only audit of FLUX image/garment/hair reference segments"
    )
    parser.add_argument(
        "--config", default="configs/spatial_warmup_resume_hair_incontext.yaml"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="artifacts/ref_audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--attention-samples", type=int, default=2)
    parser.add_argument("--taus", type=float, nargs="+", default=(0.3, 0.5, 0.7))
    parser.add_argument("--layers", type=int, nargs="+", default=(0, 9, 18))
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument("--non-gray-threshold", type=float, default=12.0)
    parser.add_argument("--far-leak-threshold", type=float, default=0.02)
    parser.add_argument("--background-attention-threshold", type=float, default=0.15)
    parser.add_argument("--background-enrichment-threshold", type=float, default=1.25)
    parser.add_argument("--neutral-gray-tolerance", type=float, default=20.0)
    parser.add_argument("--garment-x-offset", type=float, default=None)
    parser.add_argument("--hair-y-offset", type=float, default=None)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({
        "conclusion": payload["conclusion"],
        "reasons": payload["reasons"],
        "background_total_reference_attention": payload["attention"]["background_total_reference_attention"],
        "background_reference_enrichment": payload["attention"]["background_reference_enrichment"],
        "background_attention_risk": payload["attention"]["background_attention_risk"],
        "reference_neutral_gray_max_abs_delta": payload["reference_neutral_gray_max_abs_delta"],
        "max_control_reference_rms": payload["controlnet"]["max_reference_rms"],
        "max_far_outside_changed_ratio": payload["reference_content"]["max_far_outside_changed_ratio"],
    }, indent=2))


if __name__ == "__main__":
    main()
