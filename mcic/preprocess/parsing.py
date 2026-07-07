from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from mcic.utils.image import dilate, erode, mask_or, mask_subtract


def masks_from_labels(label_map: Image.Image, label_ids: dict[str, list[int]]) -> dict[str, Image.Image]:
    labels = np.asarray(label_map)
    result: dict[str, Image.Image] = {}
    for name, ids in label_ids.items():
        array = np.isin(labels, ids).astype(np.uint8) * 255
        result[name] = Image.fromarray(array, mode="L")
    return result


def heuristic_masks(image: Image.Image) -> dict[str, Image.Image]:
    """Creates coarse masks only for pipeline verification without a parser model."""
    width, height = image.size
    masks = {key: Image.new("L", image.size, 0) for key in ("face", "hair", "cloth", "person")}
    draws = {key: ImageDraw.Draw(value) for key, value in masks.items()}
    body = (round(width * 0.22), round(height * 0.08), round(width * 0.78), round(height * 0.98))
    face = (round(width * 0.38), round(height * 0.08), round(width * 0.62), round(height * 0.28))
    hair = (round(width * 0.35), round(height * 0.04), round(width * 0.65), round(height * 0.20))
    cloth = (round(width * 0.26), round(height * 0.30), round(width * 0.74), round(height * 0.82))
    draws["person"].ellipse(body, fill=255)
    draws["face"].ellipse(face, fill=255)
    draws["hair"].ellipse(hair, fill=255)
    draws["cloth"].rectangle(cloth, fill=255)
    return masks


def load_base_masks(image: Image.Image, sample_id: str, config: dict[str, Any]) -> dict[str, Image.Image]:
    preprocess = config["preprocess"]
    if preprocess.get("parsing_backend", "heuristic") == "label_maps":
        folder = Path(preprocess["parsing_dir"])
        candidate = folder / f"{sample_id}.png"
        if not candidate.exists():
            raise FileNotFoundError(f"Missing parsing label map: {candidate}")
        return masks_from_labels(Image.open(candidate), preprocess["label_ids"])
    return heuristic_masks(image)


def compose_training_masks(base: dict[str, Image.Image], config: dict[str, Any]) -> dict[str, Image.Image]:
    radius = int(config["preprocess"].get("boundary_dilate_radius", 7))
    erosion = int(config["preprocess"].get("cloth_erode_radius", 3))
    identity = mask_or(base["face"], base["hair"])
    expanded_identity = dilate(identity, radius)
    cloth_safe = erode(mask_subtract(base["cloth"], expanded_identity), erosion)
    editable = mask_subtract(base["person"], cloth_safe)
    ambiguous = mask_subtract(expanded_identity, erode(identity, radius))
    return {
        "cf_mask": identity,
        "cloth_safe_mask": cloth_safe,
        "paired_mask": editable,
        "ambiguous_mask": ambiguous,
        **base,
    }
