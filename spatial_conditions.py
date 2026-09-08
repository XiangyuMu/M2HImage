from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from conditions import find_one, get_resolution, mask_bbox, pack_latents, pil_to_tensor


NEUTRAL_GRAY = 127


def erode_reference_mask(mask: np.ndarray, erosion_px: int = 0) -> np.ndarray:
    value = np.asarray(mask, dtype=np.float32).clip(0.0, 1.0)
    radius = int(erosion_px)
    if radius < 0:
        raise ValueError(f"reference mask erosion must be >=0, got {erosion_px}")
    if radius == 0:
        return value
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    eroded = cv2.erode(value, kernel, iterations=1)
    return np.asarray(eroded, dtype=np.float32).clip(0.0, 1.0)


def load_cloth_mask(root: str | Path, sample_id: str, resolution: Any) -> np.ndarray:
    """Load the source garment mask at model resolution as a float mask in [0, 1]."""
    root = Path(root)
    width, height = get_resolution(resolution)
    path = find_one(root / "clothes_bySAM/masks/human", str(sample_id))
    mask = Image.open(path).convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(mask, dtype=np.float32) / 255.0).clip(0.0, 1.0)


def masked_garment_image(
    root: str | Path,
    sample_id: str,
    resolution: Any,
    neutral_gray: int = NEUTRAL_GRAY,
    erosion_px: int = 0,
) -> tuple[Image.Image, np.ndarray]:
    """Keep the mannequin garment in place and neutralize all pixels outside its mask."""
    root = Path(root)
    width, height = get_resolution(resolution)
    mannequin = Image.open(find_one(root / "images/mannequin", str(sample_id))).convert("RGB")
    if mannequin.size != (width, height):
        mannequin = mannequin.resize((width, height), Image.Resampling.BICUBIC)
    mask = erode_reference_mask(
        load_cloth_mask(root, sample_id, resolution), erosion_px
    )
    image = np.asarray(mannequin, dtype=np.float32)
    composed = image * mask[..., None] + float(neutral_gray) * (1.0 - mask[..., None])
    return Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8), mode="RGB"), mask


def masked_hair_image(
    root: str | Path,
    sample_id: str,
    resolution: Any,
    neutral_gray: int = NEUTRAL_GRAY,
    hair_label: int = 2,
    min_area_fraction: float = 0.01,
    erosion_px: int = 0,
) -> tuple[Image.Image, np.ndarray, bool]:
    """Keep only FASHN hair pixels in their original canvas coordinates."""
    root = Path(root)
    width, height = get_resolution(resolution)
    human = Image.open(find_one(root / "images/human", str(sample_id))).convert("RGB")
    if human.size != (width, height):
        human = human.resize((width, height), Image.Resampling.BICUBIC)
    labels = load_fashn_labels(root, sample_id)
    if labels.shape != (height, width):
        labels = np.asarray(
            Image.fromarray(labels, mode="L").resize(
                (width, height), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
    mask = erode_reference_mask(
        (labels == int(hair_label)).astype(np.float32), erosion_px
    )
    empty = float(mask.mean()) < float(min_area_fraction)
    if empty:
        mask.fill(0.0)
    image = np.asarray(human, dtype=np.float32)
    composed = image * mask[..., None] + float(neutral_gray) * (1.0 - mask[..., None])
    return (
        Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8), mode="RGB"),
        mask,
        empty,
    )


def image_mask_to_token_weights(mask: np.ndarray, width: int, height: int) -> torch.Tensor:
    """Average-pool an image-space mask onto FLUX's packed 16x16-pixel token grid."""
    array = np.asarray(mask, dtype=np.float32)
    if array.shape != (height, width):
        resized = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L")
        array = np.asarray(
            resized.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32
        ) / 255.0
    tensor = torch.from_numpy(array).view(1, 1, height, width)
    pooled = F.avg_pool2d(tensor, kernel_size=16, stride=16)
    expected = (height // 16) * (width // 16)
    weights = pooled.flatten(2).transpose(1, 2).contiguous()
    if weights.shape != (1, expected, 1):
        raise RuntimeError(f"token mask shape={tuple(weights.shape)}, expected={(1, expected, 1)}")
    return weights.clamp_(0.0, 1.0)


@torch.no_grad()
def encode_packed_condition(vae, image: Image.Image, resolution: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Encode a deterministic VAE posterior mean using the validated FLUX latent convention."""
    tensor = pil_to_tensor(image, resolution).unsqueeze(0).to(device=device, dtype=dtype)
    latents = vae.encode(tensor).latent_dist.mean
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pack_latents(latents)


def mask_control_samples(
    samples: Sequence[torch.Tensor] | None,
    token_mask: torch.Tensor | None,
) -> list[torch.Tensor] | None:
    if samples is None:
        return None
    if token_mask is None:
        return list(samples)
    masked: list[torch.Tensor] = []
    for sample in samples:
        mask = token_mask.to(device=sample.device, dtype=sample.dtype)
        if sample.ndim != 3 or sample.shape[1] != mask.shape[1]:
            raise RuntimeError(
                f"ControlNet residual shape={tuple(sample.shape)} cannot use token mask={tuple(mask.shape)}"
            )
        masked.append(sample * mask)
    return masked


def add_control_samples(
    accumulated: Sequence[torch.Tensor] | None,
    current: Sequence[torch.Tensor] | None,
) -> list[torch.Tensor] | None:
    if accumulated is None:
        return None if current is None else list(current)
    if current is None:
        return list(accumulated)
    if len(accumulated) != len(current):
        raise RuntimeError(f"ControlNet residual list mismatch: {len(accumulated)} vs {len(current)}")
    return [left + right for left, right in zip(accumulated, current, strict=True)]


def make_reference_image_ids(
    width: int,
    height: int,
    x_offset: float,
    device: torch.device,
    dtype: torch.dtype,
    stride: int = 1,
) -> torch.Tensor:
    """Create spatial IDs for an in-context reference canvas displaced along x."""
    if stride < 1:
        raise ValueError("reference stride must be >= 1")
    rows = torch.arange(0, height // 16, stride, device=device, dtype=dtype)
    cols = torch.arange(0, width // 16, stride, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack((torch.zeros_like(gy), gy, gx + float(x_offset)), dim=-1).reshape(-1, 3)


def make_hair_reference_image_ids(
    width: int,
    height: int,
    y_offset: float,
    device: torch.device,
    dtype: torch.dtype,
    stride: int = 1,
) -> torch.Tensor:
    """Create IDs for a hair-reference canvas displaced along y, separate from garment."""
    if stride < 1:
        raise ValueError("hair reference stride must be >= 1")
    rows = torch.arange(0, height // 16, stride, device=device, dtype=dtype)
    cols = torch.arange(0, width // 16, stride, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack(
        (torch.zeros_like(gy), gy + float(y_offset), gx), dim=-1
    ).reshape(-1, 3)


def downsample_reference_tokens(
    tokens: torch.Tensor,
    width: int,
    height: int,
    stride: int = 1,
) -> torch.Tensor:
    """Spatially subsample packed FLUX tokens without changing their channel packing."""
    if stride < 1:
        raise ValueError("reference stride must be >= 1")
    squeeze = tokens.ndim == 2
    if squeeze:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 3:
        raise RuntimeError(f"reference tokens must be [B,N,C] or [N,C], got {tuple(tokens.shape)}")
    grid_h, grid_w = int(height) // 16, int(width) // 16
    expected = grid_h * grid_w
    if tokens.shape[1] != expected:
        raise RuntimeError(
            f"reference token count={tokens.shape[1]}, expected={expected} for {width}x{height}"
        )
    sampled = tokens.view(tokens.shape[0], grid_h, grid_w, tokens.shape[2])[:, ::stride, ::stride]
    sampled = sampled.reshape(tokens.shape[0], -1, tokens.shape[2]).contiguous()
    return sampled[0] if squeeze else sampled


def pad_control_samples_for_reference(
    samples: Sequence[torch.Tensor] | None,
    reference_token_count: int,
) -> list[torch.Tensor] | None:
    """Append zero residuals so ControlNet affects only the generated-image token segment."""
    if samples is None:
        return None
    if reference_token_count < 0:
        raise ValueError("reference_token_count must be non-negative")
    if reference_token_count == 0:
        return list(samples)
    padded = []
    for sample in samples:
        if sample.ndim != 3:
            raise RuntimeError(f"ControlNet residual must be [B,N,C], got {tuple(sample.shape)}")
        zeros = sample.new_zeros(sample.shape[0], int(reference_token_count), sample.shape[2])
        padded.append(torch.cat((sample, zeros), dim=1))
    return padded


def load_fashn_labels(root: str | Path, sample_id: str) -> np.ndarray:
    root = Path(root)
    path = find_one(root / "human_parsing/fashn/masks/human", str(sample_id))
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def face_hair_appearance_crop(
    root: str | Path,
    sample_id: str,
    neutral_gray: int = NEUTRAL_GRAY,
    bbox_pad_fraction: float = 0.08,
) -> tuple[Image.Image, np.ndarray]:
    """Return a face+hair-only crop; clothing, neck, and background are neutralized."""
    root = Path(root)
    image = Image.open(find_one(root / "images/human", str(sample_id))).convert("RGB")
    labels = load_fashn_labels(root, sample_id)
    if labels.shape != (image.height, image.width):
        labels = np.asarray(
            Image.fromarray(labels, mode="L").resize(image.size, Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    keep = np.isin(labels, (1, 2)).astype(np.uint8)
    if int(keep.sum()) < 64:
        raise RuntimeError(f"face+hair parsing is empty or too small for {sample_id}")
    pad = max(4, int(round(max(image.size) * float(bbox_pad_fraction))))
    bbox = mask_bbox(Image.fromarray(keep * 255, mode="L"), pad=pad)
    array = np.asarray(image, dtype=np.uint8)
    composed = np.full_like(array, int(neutral_gray), dtype=np.uint8)
    selected = keep.astype(bool)
    composed[selected] = array[selected]
    crop = Image.fromarray(composed, mode="RGB").crop(bbox)
    return crop, keep


def _square_image_and_mask(
    image: Image.Image,
    mask: np.ndarray,
    fill: int = NEUTRAL_GRAY,
) -> tuple[Image.Image, Image.Image]:
    image = image.convert("RGB")
    mask_image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
    side = max(image.size)
    x = (side - image.width) // 2
    y = (side - image.height) // 2
    image_square = Image.new("RGB", (side, side), int(fill))
    mask_square = Image.new("L", (side, side), 0)
    image_square.paste(image, (x, y))
    mask_square.paste(mask_image, (x, y))
    return image_square, mask_square


@torch.no_grad()
def dino_hair_dense_tokens(
    model,
    image: Image.Image,
    hair_mask: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    image_size: int = 518,
    max_tokens: int = 64,
    neutral_gray: int = NEUTRAL_GRAY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract fixed-size masked DINOv2 patch tokens plus normalized 2D positions."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    image_square, mask_square = _square_image_and_mask(image, hair_mask, fill=neutral_gray)
    image_square = image_square.resize((image_size, image_size), Image.Resampling.BICUBIC)
    mask_square = mask_square.resize((image_size, image_size), Image.Resampling.NEAREST)
    array = np.asarray(image_square, dtype=np.float32) / 255.0
    mask_array = np.asarray(mask_square, dtype=np.float32) / 255.0
    neutral = float(neutral_gray) / 255.0
    array = array * mask_array[..., None] + neutral * (1.0 - mask_array[..., None])
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    tensor = ((tensor - mean) / std).to(device=device, dtype=dtype)
    output = model(tensor)
    if isinstance(output, (tuple, list)):
        output = output[0]
    patches = output[0].float()
    count, dim = patches.shape
    side = int(round(count ** 0.5))
    if side * side != count:
        raise RuntimeError(f"DINO output is not a square patch grid: shape={tuple(patches.shape)}")
    patch_mask = torch.from_numpy(mask_array).view(1, 1, image_size, image_size)
    patch_mask = F.interpolate(patch_mask, size=(side, side), mode="area")[0, 0]
    selected = torch.nonzero(patch_mask.reshape(-1) > 0.05, as_tuple=False).flatten()
    if selected.numel() > max_tokens:
        positions = torch.linspace(0, selected.numel() - 1, max_tokens).round().long()
        selected = selected[positions]
    valid_count = int(selected.numel())
    rows = torch.div(selected, side, rounding_mode="floor")
    cols = selected.remainder(side)
    coords = torch.stack(
        (
            rows.float() / max(side - 1, 1) * 2.0 - 1.0,
            cols.float() / max(side - 1, 1) * 2.0 - 1.0,
        ),
        dim=1,
    )
    values = patches[selected]
    token_out = torch.zeros(max_tokens, dim, dtype=torch.float32)
    position_out = torch.zeros(max_tokens, 2, dtype=torch.float32)
    valid_out = torch.zeros(max_tokens, dtype=torch.float32)
    token_out[:valid_count] = values.cpu()
    position_out[:valid_count] = coords.cpu()
    valid_out[:valid_count] = 1.0
    return (
        token_out.numpy().astype(np.float16),
        position_out.numpy().astype(np.float16),
        valid_out.numpy().astype(np.float16),
    )


def _crop_hair_bbox_and_pad_square(
    image: Image.Image,
    hair_mask: np.ndarray,
    fill: int = NEUTRAL_GRAY,
) -> tuple[Image.Image, Image.Image]:
    """Crop to the parsed hair bbox, then square-pad without anisotropic resize."""
    image = image.convert("RGB")
    mask = np.asarray(hair_mask) > 0
    if mask.shape != (image.height, image.width):
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                image.size, Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        ) > 0
    ys, xs = np.where(mask)
    if not len(xs):
        return Image.new("RGB", (1, 1), int(fill)), Image.new("L", (1, 1), 0)
    bbox = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )
    return _square_image_and_mask(
        image.crop(bbox),
        mask[bbox[1] : bbox[3], bbox[0] : bbox[2]],
        fill=fill,
    )


@torch.no_grad()
def dino_hair_semantic_tokens(
    model,
    image: Image.Image,
    hair_mask: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    image_size: int = 518,
    max_tokens: int = 32,
    neutral_gray: int = NEUTRAL_GRAY,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pose-agnostic hair descriptors from a bbox crop.

    The crop is padded to square before a uniform resize. Tokens are reduced by
    a deterministic 2D stride and carry no image-grid position IDs: hair should
    communicate color, curl, length, and volume, not transplant reference pose.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    image_square, mask_square = _crop_hair_bbox_and_pad_square(
        image, hair_mask, fill=neutral_gray
    )
    image_square = image_square.resize((image_size, image_size), Image.Resampling.BICUBIC)
    mask_square = mask_square.resize((image_size, image_size), Image.Resampling.NEAREST)
    array = np.asarray(image_square, dtype=np.float32) / 255.0
    mask_array = np.asarray(mask_square, dtype=np.float32) / 255.0
    neutral = float(neutral_gray) / 255.0
    array = array * mask_array[..., None] + neutral * (1.0 - mask_array[..., None])
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    tensor = ((tensor - mean) / std).to(device=device, dtype=dtype)
    output = model(tensor)
    if isinstance(output, (tuple, list)):
        output = output[0]
    patches = output[0].float()
    count, dim = patches.shape
    side = int(round(count ** 0.5))
    if side * side != count:
        raise RuntimeError(f"DINO output is not a square patch grid: shape={tuple(patches.shape)}")
    patch_mask = torch.from_numpy(mask_array).view(1, 1, image_size, image_size)
    patch_mask = F.interpolate(patch_mask, size=(side, side), mode="area")[0, 0]
    selected = torch.nonzero(patch_mask.reshape(-1) > 0.05, as_tuple=False).flatten()
    if selected.numel() > max_tokens:
        rows = torch.div(selected, side, rounding_mode="floor")
        cols = selected.remainder(side)
        stride = max(1, int(np.ceil(np.sqrt(selected.numel() / max_tokens))))
        strided = selected[(rows.remainder(stride) == 0) & (cols.remainder(stride) == 0)]
        if strided.numel():
            selected = strided
    if selected.numel() > max_tokens:
        keep = torch.linspace(0, selected.numel() - 1, max_tokens).round().long()
        selected = selected[keep]
    valid_count = int(selected.numel())
    token_out = torch.zeros(max_tokens, dim, dtype=torch.float32)
    valid_out = torch.zeros(max_tokens, dtype=torch.float32)
    token_out[:valid_count] = patches[selected].cpu()
    valid_out[:valid_count] = 1.0
    return (
        token_out.numpy().astype(np.float16),
        valid_out.numpy().astype(np.float16),
    )
