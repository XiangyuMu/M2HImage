from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .spec import IMAGE_FIELDS


EXPECTED_SIZE = {"low": (512, 512), "native": (768, 1024)}


def load_manifest(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_manifest(manifest_path: Path, prepared_root: Path) -> dict[str, object]:
    records = load_manifest(manifest_path)
    errors: list[str] = []
    mask_fractions: list[float] = []
    for record in records:
        resolution = str(record["resolution"])
        for field in IMAGE_FIELDS:
            path = prepared_root / str(record[field])
            if not path.exists():
                errors.append(f"{record['key']}: missing {field}: {path}")
                continue
            with Image.open(path) as image:
                if field not in {"identity_card", "face", "garment"} and image.size != EXPECTED_SIZE[resolution]:
                    errors.append(f"{record['key']}: {field} size={image.size}")
                if field == "replace_mask":
                    values = np.asarray(image.convert("L"), dtype=np.uint8)
                    unique = set(np.unique(values).tolist())
                    if not unique.issubset({0, 255}):
                        errors.append(f"{record['key']}: replace mask values={sorted(unique)[:10]}")
                    mask_fractions.append(float((values > 0).mean()))
    report = {
        "manifest": str(manifest_path),
        "records": len(records),
        "errors": errors,
        "replace_mask_fraction_mean": float(np.mean(mask_fractions)) if mask_fractions else None,
        "status": "pass" if not errors else "fail",
    }
    if errors:
        raise RuntimeError(json.dumps(report, indent=2, ensure_ascii=False))
    return report

