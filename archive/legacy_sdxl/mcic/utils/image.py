from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class CanvasTransform:
    original_width: int
    original_height: int
    target_width: int
    target_height: int
    scaled_width: int
    scaled_height: int
    offset_x: int
    offset_y: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def compute_canvas_transform(image: Image.Image, height: int, width: int) -> CanvasTransform:
    original_width, original_height = image.size
    scale = min(width / original_width, height / original_height)
    scaled_width = max(1, round(original_width * scale))
    scaled_height = max(1, round(original_height * scale))
    return CanvasTransform(
        original_width,
        original_height,
        width,
        height,
        scaled_width,
        scaled_height,
        (width - scaled_width) // 2,
        (height - scaled_height) // 2,
    )


def fit_to_canvas(
    image: Image.Image,
    transform: CanvasTransform,
    *,
    is_mask: bool = False,
    pad_value: int = 255,
) -> Image.Image:
    mode = "L" if is_mask else "RGB"
    image = image.convert(mode)
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
    resized = image.resize((transform.scaled_width, transform.scaled_height), resample)
    canvas = Image.new(mode, (transform.target_width, transform.target_height), 0 if is_mask else pad_value)
    canvas.paste(resized, (transform.offset_x, transform.offset_y))
    return canvas


def mask_or(*masks: Image.Image) -> Image.Image:
    array = np.maximum.reduce([np.asarray(mask.convert("L")) for mask in masks])
    return Image.fromarray(array.astype(np.uint8), mode="L")


def mask_subtract(mask: Image.Image, remove: Image.Image) -> Image.Image:
    base = np.asarray(mask.convert("L")) > 127
    excluded = np.asarray(remove.convert("L")) > 127
    return Image.fromarray(((base & ~excluded) * 255).astype(np.uint8), mode="L")


def dilate(mask: Image.Image, radius: int) -> Image.Image:
    size = radius * 2 + 1
    return mask.convert("L").filter(ImageFilter.MaxFilter(size))


def erode(mask: Image.Image, radius: int) -> Image.Image:
    size = radius * 2 + 1
    return mask.convert("L").filter(ImageFilter.MinFilter(size))


def mask_bbox(mask: Image.Image) -> tuple[int, int, int, int] | None:
    return mask.convert("L").point(lambda value: 255 if value > 127 else 0).getbbox()


def tensor_from_image(image: Image.Image):
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def tensor_from_mask(mask: Image.Image):
    import torch

    array = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)
