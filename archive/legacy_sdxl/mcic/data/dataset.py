from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from mcic.utils.image import compute_canvas_transform, fit_to_canvas, tensor_from_image, tensor_from_mask


class MCICDataset:
    def __init__(self, config: dict[str, Any], split: str = "train") -> None:
        root = Path(config["data"]["root"])
        metadata_path = root / config["data"].get("cache_dir", "cache_mcic") / "metadata.jsonl"
        with metadata_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.rows = [row for row in rows if row["split"] == split and row["face_quality_pass"]]
        self.config = config
        self.height = config["image"]["height"]
        self.width = config["image"]["width"]
        self.pad_value = config["image"].get("pad_value", 255)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import torch

        row = self.rows[index]
        mannequin = Image.open(row["mannequin_path"]).convert("RGB")
        human = Image.open(row["human_path"]).convert("RGB")
        transform = compute_canvas_transform(mannequin, self.height, self.width)
        mannequin = fit_to_canvas(mannequin, transform, pad_value=self.pad_value)
        human = fit_to_canvas(human, transform, pad_value=self.pad_value)
        masks = {}
        for key in ("paired_mask", "cf_mask", "cloth_safe_mask"):
            masks[key] = fit_to_canvas(Image.open(row[f"{key}_path"]), transform, is_mask=True)
        identity = torch.from_numpy(np.load(row["identity_embedding_path"])).float()
        return {
            "index": index,
            "sample_id": row["sample_id"],
            "mannequin": tensor_from_image(mannequin),
            "human": tensor_from_image(human),
            "paired_mask": tensor_from_mask(masks["paired_mask"]),
            "cf_mask": tensor_from_mask(masks["cf_mask"]),
            "cloth_safe_mask": tensor_from_mask(masks["cloth_safe_mask"]),
            "identity_embedding": identity,
            "face_box": torch.tensor(row["face_box"], dtype=torch.float32),
            "transform": transform.to_dict(),
        }

    def sample_negative_indices(self, indices):
        import torch

        if len(self) < 2:
            raise ValueError("Counterfactual training requires at least two training samples")
        shift = torch.randint(1, len(self), indices.shape, device=indices.device)
        return (indices + shift) % len(self)

    def identity_embeddings(self, indices):
        import numpy as np
        import torch

        embeddings = [np.load(self.rows[int(i)]["identity_embedding_path"]) for i in indices.cpu()]
        return torch.from_numpy(np.stack(embeddings)).float().to(indices.device)
