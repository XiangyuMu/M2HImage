from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from metrics_v2.common import find_image, image_size


DEFAULT_GARMENT_LABELS = (3, 4, 5, 6, 7, 10)
HAIR_LABEL = 2


def label_id(cfg: dict[str, Any], name: str) -> int:
    root = Path(cfg["data"]["root"])
    configured = Path(cfg["metrics_v2"]["parsing"]["labels_json"])
    path = configured if configured.is_absolute() else root / configured
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels", payload)
    matches = [int(index) for index, label in labels.items() if str(label).lower() == str(name).lower()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one '{name}' label in {path}, found {matches}")
    return matches[0]


class FashnParser:
    """FASHN SegFormer-B4 with the model's production resize convention."""

    def __init__(self, model_dir: str | Path, device: str = "cuda:0", input_size: tuple[int, int] = (384, 576)):
        from transformers import SegformerForSemanticSegmentation

        self.device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
        self.model = SegformerForSemanticSegmentation.from_pretrained(str(model_dir), local_files_only=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.input_size = tuple(int(value) for value in input_size)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _tensor(self, images: list[np.ndarray]) -> torch.Tensor:
        width, height = self.input_size
        arrays = [cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) for image in images]
        batch = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float().to(self.device) / 255.0
        return (batch - self.mean) / self.std

    @torch.inference_mode()
    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if not images:
            return []
        original_sizes = [(image.shape[0], image.shape[1]) for image in images]
        logits = self.model(pixel_values=self._tensor(images)).logits.float()
        outputs = []
        for index, size in enumerate(original_sizes):
            upsampled = torch.nn.functional.interpolate(
                logits[index : index + 1], size=size, mode="bilinear", align_corners=False
            )
            outputs.append(upsampled.argmax(dim=1)[0].byte().cpu().numpy())
        return outputs


def parsing_path(out_dir: str | Path, image_path: str | Path) -> Path:
    return Path(out_dir) / "parsing_masks" / f"{Path(image_path).stem}.png"


def build_generated_parsing(
    cfg: dict[str, Any], rows: list[dict[str, Any]], out_dir: str | Path, device: str
) -> dict[str, Any]:
    pcfg = cfg["metrics_v2"]["parsing"]
    out_dir = Path(out_dir)
    mask_dir = out_dir / "parsing_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    pending = [row for row in rows if not parsing_path(out_dir, row["path"]).exists()]
    parser = None
    failed: list[dict[str, str]] = []
    batch_size = int(pcfg.get("batch_size", 4))
    for offset in tqdm(range(0, len(pending), batch_size), desc="FASHN generated parsing"):
        chunk = pending[offset : offset + batch_size]
        readable: list[tuple[dict[str, Any], np.ndarray]] = []
        for row in chunk:
            try:
                readable.append((row, np.asarray(Image.open(row["path"]).convert("RGB"), dtype=np.uint8)))
            except Exception as exc:  # noqa: BLE001
                failed.append({"path": str(row["path"]), "error": str(exc)})
        if not readable:
            continue
        if parser is None:
            parser = FashnParser(
                pcfg["model_dir"],
                device=device,
                input_size=(int(pcfg.get("input_width", 384)), int(pcfg.get("input_height", 576))),
            )
        try:
            predictions = parser.predict([item[1] for item in readable])
            for (row, _), prediction in zip(readable, predictions, strict=True):
                Image.fromarray(prediction, mode="L").save(parsing_path(out_dir, row["path"]))
        except Exception as exc:  # noqa: BLE001
            for row, _ in readable:
                failed.append({"path": str(row["path"]), "error": str(exc)})
    return {
        "expected": len(rows),
        "cached": len(rows) - len(pending),
        "computed": len(pending) - len(failed),
        "failed": failed,
        "mask_dir": str(mask_dir),
    }


def labels_mask(label_map: np.ndarray, labels: tuple[int, ...] | list[int]) -> np.ndarray:
    return np.isin(np.asarray(label_map), np.asarray(labels)).astype(np.uint8)


def load_generated_masks(
    cfg: dict[str, Any], out_dir: str | Path, row: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, str]:
    width, height = image_size(cfg)
    path = parsing_path(out_dir, row["path"])
    garment_labels = tuple(int(value) for value in cfg["metrics_v2"]["parsing"].get("garment_labels", DEFAULT_GARMENT_LABELS))
    minimum = float(cfg["metrics_v2"]["parsing"].get("min_garment_area_fraction", 0.005))
    source = "fashn_generated"
    if path.exists():
        labels = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        garment = labels_mask(labels, garment_labels)
        hair = labels_mask(labels, (label_id(cfg, "hair"),))
    else:
        garment = np.zeros((height, width), dtype=np.uint8)
        hair = np.zeros((height, width), dtype=np.uint8)
    if float(garment.mean()) < minimum:
        root = Path(cfg["data"]["root"])
        garment = load_binary_mask(find_image(root, "clothes_bySAM/masks/human", str(row["mid"])), (width, height))
        source = "fallback_projected_mannequin_cloth"
    return garment, hair, source


def load_binary_mask(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return (np.asarray(image, dtype=np.uint8) > 127).astype(np.uint8)


def source_garment_mask(cfg: dict[str, Any], mid: str, size: tuple[int, int] | None = None) -> np.ndarray:
    root = Path(cfg["data"]["root"])
    return load_binary_mask(find_image(root, "clothes_bySAM/masks/human", mid), size or image_size(cfg))


def reference_hair_mask(cfg: dict[str, Any], jid: str, size: tuple[int, int] | None = None) -> np.ndarray:
    root = Path(cfg["data"]["root"])
    path = find_image(root, "human_parsing/fashn/masks/human", jid)
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return labels_mask(np.asarray(image, dtype=np.uint8), (label_id(cfg, "hair"),))
