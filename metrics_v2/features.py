from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from metrics.garment_sim import dino_region_feature, load_dino_model, mask_bbox


class RegionFeatureExtractor:
    def __init__(self, cfg: dict[str, Any], device: str):
        self.device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
        dcfg = cfg["metrics_v2"]["dino"]
        proxy_cfg = {
            "metrics": {
                "garment": {
                    "dino_repo_root": dcfg["repo_root"],
                    "dino_checkpoint": dcfg["checkpoint"],
                }
            }
        }
        self.dino = load_dino_model(proxy_cfg, self.device)
        self.dino_size = int(dcfg.get("image_size", 518))
        self.mask_out_value = float(dcfg.get("mask_out_value", 0.5))
        self.lpips = None

    def dino_feature(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return dino_region_feature(
            self.dino,
            image,
            mask,
            self.device,
            self.dino_size,
            self.mask_out_value,
        )

    def _lpips_model(self):
        if self.lpips is None:
            import lpips

            self.lpips = lpips.LPIPS(net="alex").eval().requires_grad_(False).to(self.device)
        return self.lpips

    @torch.inference_mode()
    def lpips_distance(self, image_a: np.ndarray, mask_a: np.ndarray, image_b: np.ndarray, mask_b: np.ndarray) -> float:
        crop_a = masked_bbox_crop(image_a, mask_a)
        crop_b = masked_bbox_crop(image_b, mask_b)
        tensor_a = torch.from_numpy(crop_a.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor_b = torch.from_numpy(crop_b.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return float(self._lpips_model()(tensor_a, tensor_b).item())

    @torch.inference_mode()
    def high_frequency_lpips_distance(
        self,
        image_a: np.ndarray,
        mask_a: np.ndarray,
        image_b: np.ndarray,
        mask_b: np.ndarray,
        *,
        sigma: float = 2.0,
        bbox: tuple[int, int, int, int] | None = None,
        output_size: int = 256,
    ) -> float:
        if bbox is None:
            crop_a = masked_bbox_crop(image_a, mask_a, output_size=output_size)
            crop_b = masked_bbox_crop(image_b, mask_b, output_size=output_size)
        else:
            crop_a = masked_fixed_bbox_crop(
                image_a, mask_a, bbox, output_size=output_size
            )
            crop_b = masked_fixed_bbox_crop(
                image_b, mask_b, bbox, output_size=output_size
            )
        residual_a = gaussian_high_pass(crop_a, sigma=sigma)
        residual_b = gaussian_high_pass(crop_b, sigma=sigma)
        tensor_a = torch.from_numpy(
            np.clip(residual_a / 127.5, -1.0, 1.0)
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor_b = torch.from_numpy(
            np.clip(residual_b / 127.5, -1.0, 1.0)
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return float(self._lpips_model()(tensor_a, tensor_b).item())


def masked_bbox_crop(image: np.ndarray, mask: np.ndarray, output_size: int = 256, pad: int = 8) -> np.ndarray:
    x0, y0, x1, y1 = mask_bbox(mask, pad=pad)
    crop = image[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1].astype(bool)
    if crop.size == 0 or not crop_mask.any():
        raise ValueError("empty region mask")
    masked = np.full_like(crop, 128)
    masked[crop_mask] = crop[crop_mask]
    return cv2.resize(masked, (output_size, output_size), interpolation=cv2.INTER_AREA)


def masked_fixed_bbox_crop(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    output_size: int = 256,
) -> np.ndarray:
    x0, y0, x1, y1 = (int(value) for value in bbox)
    height, width = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid fixed bbox {bbox} for image {width}x{height}")
    crop = image[y0:y1, x0:x1].copy()
    crop_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    if not bool(crop_mask.any()):
        raise ValueError("fixed bbox has no selected mask pixels")
    masked = np.full_like(crop, 128)
    masked[crop_mask] = crop[crop_mask]
    return cv2.resize(
        masked, (output_size, output_size), interpolation=cv2.INTER_AREA
    )


def gaussian_high_pass(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    if float(sigma) <= 0.0:
        raise ValueError(f"high-pass sigma must be positive, got {sigma}")
    blurred = cv2.GaussianBlur(
        value,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT101,
    )
    return value - blurred


def sobel_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    value = gray.astype(np.float32) / 255.0
    grad_x = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


def masked_gradient_cosine(
    image_a: np.ndarray,
    mask_a: np.ndarray,
    image_b: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    grad_a = sobel_gradient_magnitude(image_a) * np.asarray(mask_a, dtype=np.float32)
    grad_b = sobel_gradient_magnitude(image_b) * np.asarray(mask_b, dtype=np.float32)
    return cosine(grad_a, grad_b)


def detect_print_region(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    gradient_percentile: float = 90.0,
    density_kernel: int = 25,
    density_threshold: float = 0.15,
    min_component_fraction: float = 0.0002,
    max_component_fraction: float = 0.05,
    pad: int = 12,
) -> tuple[tuple[int, int, int, int], float]:
    """Find a dense high-gradient garment component as a print/logo candidate."""
    selected = np.asarray(mask, dtype=np.uint8) > 0
    if not bool(selected.any()):
        raise ValueError("cannot detect print region from an empty garment mask")
    interior = cv2.erode(
        selected.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1
    ).astype(bool)
    if not bool(interior.any()):
        interior = selected
    gradient = sobel_gradient_magnitude(image)
    threshold = float(np.percentile(gradient[interior], float(gradient_percentile)))
    seeds = ((gradient >= threshold) & interior).astype(np.float32)
    kernel = max(3, int(density_kernel) | 1)
    density = cv2.boxFilter(
        seeds,
        cv2.CV_32F,
        (kernel, kernel),
        normalize=True,
        borderType=cv2.BORDER_CONSTANT,
    )
    dense = ((density >= float(density_threshold)) & interior).astype(np.uint8)
    dense = cv2.morphologyEx(
        dense,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
        iterations=1,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dense, 8)
    frame_area = float(selected.size)
    candidates: list[tuple[float, int, int, int, int]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        fraction = float(area) / frame_area
        if fraction < float(min_component_fraction) or fraction > float(max_component_fraction):
            continue
        component = labels == index
        score = float((gradient * density)[component].sum())
        candidates.append((score, x, y, x + width, y + height))
    if not candidates:
        x0, y0, x1, y1 = mask_bbox(seeds > 0, pad=0)
        score = float((gradient * seeds).sum())
    else:
        score, x0, y0, x1, y1 = max(candidates, key=lambda item: item[0])
    height, width = selected.shape
    bbox = (
        max(0, int(x0) - int(pad)),
        max(0, int(y0) - int(pad)),
        min(width, int(x1) + int(pad)),
        min(height, int(y1) + int(pad)),
    )
    return bbox, float(score)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def masked_lab_mean(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = mask.astype(bool)
    if not selected.any():
        raise ValueError("empty mask for LAB mean")
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[selected].mean(axis=0)


def load_feature_cache(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    keys = [str(value) for value in data["keys"].tolist()]
    values = np.asarray(data["values"], dtype=np.float32)
    return dict(zip(keys, values, strict=True))


def save_feature_cache(path: str | Path, features: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(features)
    values = np.stack([features[key] for key in keys]) if keys else np.empty((0, 0), dtype=np.float32)
    np.savez_compressed(path, keys=np.asarray(keys), values=values.astype(np.float16))
