from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from metrics.common import plot_histogram
from metrics_v2.common import find_image, image_size, metric_summary, read_rgb, write_csv, write_json
from metrics_v2.features import RegionFeatureExtractor, cosine, masked_lab_mean
from metrics_v2.parsing import load_generated_masks, reference_hair_mask


FIELDS = [
    "mid",
    "jid",
    "seed",
    "garment_type",
    "generated_path",
    "reference_path",
    "status",
    "error",
    "generated_hair_fraction",
    "reference_hair_fraction",
    "hair_dino_to_reference",
    "hair_lab_distance",
]


def run_hair_metrics(
    cfg: dict[str, Any], rows: list[dict[str, Any]], out_dir: str | Path, device: str
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    root = Path(cfg["data"]["root"])
    size = image_size(cfg)
    minimum = float(cfg["metrics_v2"]["parsing"].get("min_hair_area_fraction", 0.01))
    extractor = RegionFeatureExtractor(cfg, device)
    reference_features: dict[str, np.ndarray] = {}
    reference_payload: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {}
    csv_rows: list[dict[str, Any]] = []

    def reference(jid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        if jid not in reference_payload:
            path = find_image(root, "images/human", jid)
            image = read_rgb(path, size)
            mask = reference_hair_mask(cfg, jid, size)
            if float(mask.mean()) < minimum:
                raise RuntimeError(f"reference hair area below {minimum:.3f}")
            feature = extractor.dino_feature(image, mask)
            reference_features[jid] = feature
            reference_payload[jid] = (image, mask, masked_lab_mean(image, mask), str(path))
        return reference_payload[jid]

    for row in tqdm(rows, desc="Hair-to-reference metrics"):
        output = {
            "mid": row["mid"],
            "jid": row["jid"],
            "seed": row["seed"],
            "garment_type": row.get("garment_type", "unknown"),
            "generated_path": str(row["path"]),
            "reference_path": "",
            "status": "ok",
            "error": "",
            "generated_hair_fraction": "",
            "reference_hair_fraction": "",
            "hair_dino_to_reference": "",
            "hair_lab_distance": "",
        }
        try:
            generated = read_rgb(row["path"], size)
            _, generated_hair, _ = load_generated_masks(cfg, out_dir, row)
            generated_fraction = float(generated_hair.mean())
            output["generated_hair_fraction"] = generated_fraction
            ref_image, ref_mask, ref_lab, ref_path = reference(str(row["jid"]))
            output["reference_path"] = ref_path
            output["reference_hair_fraction"] = float(ref_mask.mean())
            if generated_fraction < minimum:
                output["status"] = "no_hair"
                output["error"] = f"generated hair area below {minimum:.3f}"
            else:
                feature = extractor.dino_feature(generated, generated_hair)
                output["hair_dino_to_reference"] = cosine(feature, reference_features[str(row["jid"])])
                output["hair_lab_distance"] = float(
                    np.linalg.norm(masked_lab_mean(generated, generated_hair) - ref_lab)
                )
        except Exception as exc:  # noqa: BLE001
            output["status"] = "no_hair" if "hair area below" in str(exc) else "failed"
            output["error"] = str(exc)
        csv_rows.append(output)
    write_csv(out_dir / "hair_per_image.csv", csv_rows, FIELDS)
    valid = [row for row in csv_rows if row["status"] == "ok"]
    summary = {
        "status": "ok" if valid else "failed",
        "measures_against": "reference person images/human/{jid} using FASHN hair label 2",
        "minimum_hair_area_fraction": minimum,
        "count": len(valid),
        "no_hair": sum(row["status"] == "no_hair" for row in csv_rows),
        "failed": sum(row["status"] == "failed" for row in csv_rows),
        "hair_dino": metric_summary(row["hair_dino_to_reference"] for row in valid),
        "hair_lab_distance": metric_summary(row["hair_lab_distance"] for row in valid),
        "csv": str(out_dir / "hair_per_image.csv"),
    }
    write_json(out_dir / "hair_summary.json", summary)
    plot_histogram(
        out_dir / "hair_dino_to_reference_hist.png",
        [float(row["hair_dino_to_reference"]) for row in valid],
        "Hair DINO to reference identity",
        "DINO cosine (higher is better)",
    )
    return summary
