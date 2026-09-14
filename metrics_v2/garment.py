from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from metrics.common import plot_histogram
from metrics_v2.common import find_image, grouped_metric, image_size, metric_summary, read_rgb, write_csv, write_json
from metrics_v2.features import (
    RegionFeatureExtractor,
    cosine,
    detect_print_region,
    load_feature_cache,
    masked_gradient_cosine,
    save_feature_cache,
)
from metrics_v2.parsing import load_generated_masks, source_garment_mask


FIELDS = [
    "mid",
    "jid",
    "seed",
    "garment_type",
    "generated_path",
    "mannequin_path",
    "status",
    "error",
    "mask_source",
    "generated_mask_fraction",
    "mannequin_mask_fraction",
    "garment_dino_to_mannequin",
    "garment_lpips_to_mannequin",
    "garment_hf_lpips_to_mannequin",
    "garment_gradient_sim_to_mannequin",
    "print_bbox",
    "print_region_score",
    "print_region_hf_lpips",
    "print_status",
    "print_error",
]


def _bbox_text(bbox: tuple[int, int, int, int]) -> str:
    return ",".join(str(int(value)) for value in bbox)


def _parse_bbox(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(item) for item in str(value).split(","))
    if len(parts) != 4:
        raise ValueError(f"invalid bbox {value!r}")
    return parts


def _fit_tile(image: np.ndarray, size: int) -> Image.Image:
    value = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    value.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (238, 238, 238))
    canvas.paste(value, ((size - value.width) // 2, (size - value.height) // 2))
    return canvas


def _build_print_zoom_grid(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    case_count: int,
    tile_size: int,
) -> tuple[Path, list[dict[str, Any]]]:
    candidates = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("print_status") == "ok"
    ]
    candidates.sort(
        key=lambda row: (
            -float(row["print_region_score"]),
            str(row["mid"]),
            str(row["jid"]),
            int(row["seed"]),
        )
    )
    selected: list[dict[str, Any]] = []
    seen_mids: set[str] = set()
    for row in candidates:
        if str(row["mid"]) in seen_mids:
            continue
        selected.append(row)
        seen_mids.add(str(row["mid"]))
        if len(selected) >= int(case_count):
            break
    case_width = int(tile_size) * 2
    label_height = 36
    columns = min(4, max(1, len(selected)))
    row_count = max(1, (len(selected) + columns - 1) // columns)
    canvas = Image.new(
        "RGB",
        (columns * case_width, row_count * (int(tile_size) + label_height)),
        "white",
    )
    index_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        bbox = _parse_bbox(str(row["print_bbox"]))
        generated = np.asarray(Image.open(row["generated_path"]).convert("RGB"))
        mannequin = np.asarray(Image.open(row["mannequin_path"]).convert("RGB"))
        x0, y0, x1, y1 = bbox
        source_crop = mannequin[y0:y1, x0:x1]
        generated_crop = generated[y0:y1, x0:x1]
        pair = Image.new("RGB", (case_width, int(tile_size) + label_height), "white")
        pair.paste(_fit_tile(source_crop, int(tile_size)), (0, label_height))
        pair.paste(_fit_tile(generated_crop, int(tile_size)), (int(tile_size), label_height))
        ImageDraw.Draw(pair).text(
            (6, 5),
            f"{row['mid']} / {row['jid']} / s{row['seed']}  source | generated",
            fill="black",
        )
        column = index % columns
        panel_row = index // columns
        canvas.paste(pair, (column * case_width, panel_row * (int(tile_size) + label_height)))
        index_rows.append(
            {
                "mid": row["mid"],
                "jid": row["jid"],
                "seed": row["seed"],
                "garment_type": row["garment_type"],
                "print_bbox": row["print_bbox"],
                "print_region_score": row["print_region_score"],
                "print_region_hf_lpips": row["print_region_hf_lpips"],
            }
        )
    path = out_dir / "print_zoom_grid.png"
    canvas.save(path)
    write_csv(
        out_dir / "print_zoom_index.csv",
        index_rows,
        [
            "mid",
            "jid",
            "seed",
            "garment_type",
            "print_bbox",
            "print_region_score",
            "print_region_hf_lpips",
        ],
    )
    return path, index_rows


def run_garment_metrics(
    cfg: dict[str, Any], rows: list[dict[str, Any]], out_dir: str | Path, device: str
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["data"]["root"])
    size = image_size(cfg)
    extractor = RegionFeatureExtractor(cfg, device)
    hcfg = cfg["metrics_v2"].get("garment_hf", {})
    hf_sigma = float(hcfg.get("sigma", 2.0))
    crop_size = int(hcfg.get("crop_size", 256))
    source_cache_path = out_dir / "garment_source_features.npz"
    source_features = load_feature_cache(source_cache_path)
    csv_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="Garment-to-mannequin metrics"):
        output = {
            "mid": row["mid"],
            "jid": row["jid"],
            "seed": row["seed"],
            "garment_type": row.get("garment_type", "unknown"),
            "generated_path": str(row["path"]),
            "mannequin_path": "",
            "status": "ok",
            "error": "",
            "mask_source": "",
            "generated_mask_fraction": "",
            "mannequin_mask_fraction": "",
            "garment_dino_to_mannequin": "",
            "garment_lpips_to_mannequin": "",
            "garment_hf_lpips_to_mannequin": "",
            "garment_gradient_sim_to_mannequin": "",
            "print_bbox": "",
            "print_region_score": "",
            "print_region_hf_lpips": "",
            "print_status": "not_run",
            "print_error": "",
        }
        try:
            mannequin_path = find_image(root, "images/mannequin", str(row["mid"]))
            output["mannequin_path"] = str(mannequin_path)
            generated = read_rgb(row["path"], size)
            mannequin = read_rgb(mannequin_path, size)
            generated_mask, _, mask_source = load_generated_masks(cfg, out_dir, row)
            mannequin_mask = source_garment_mask(cfg, str(row["mid"]), size)
            output["mask_source"] = mask_source
            output["generated_mask_fraction"] = float(generated_mask.mean())
            output["mannequin_mask_fraction"] = float(mannequin_mask.mean())
            mid = str(row["mid"])
            if mid not in source_features:
                source_features[mid] = extractor.dino_feature(mannequin, mannequin_mask)
            generated_feature = extractor.dino_feature(generated, generated_mask)
            output["garment_dino_to_mannequin"] = cosine(generated_feature, source_features[mid])
            output["garment_lpips_to_mannequin"] = extractor.lpips_distance(
                generated, generated_mask, mannequin, mannequin_mask
            )
            output["garment_hf_lpips_to_mannequin"] = (
                extractor.high_frequency_lpips_distance(
                    generated,
                    generated_mask,
                    mannequin,
                    mannequin_mask,
                    sigma=hf_sigma,
                    output_size=crop_size,
                )
            )
            output["garment_gradient_sim_to_mannequin"] = masked_gradient_cosine(
                generated, generated_mask, mannequin, mannequin_mask
            )
            try:
                print_bbox, print_score = detect_print_region(
                    mannequin,
                    mannequin_mask,
                    gradient_percentile=float(
                        hcfg.get("print_gradient_percentile", 90.0)
                    ),
                    density_kernel=int(hcfg.get("print_density_kernel", 25)),
                    density_threshold=float(
                        hcfg.get("print_density_threshold", 0.15)
                    ),
                    min_component_fraction=float(
                        hcfg.get("print_min_component_fraction", 0.0002)
                    ),
                    max_component_fraction=float(
                        hcfg.get("print_max_component_fraction", 0.05)
                    ),
                    pad=int(hcfg.get("print_bbox_pad", 12)),
                )
                output["print_bbox"] = _bbox_text(print_bbox)
                output["print_region_score"] = print_score
                output["print_region_hf_lpips"] = (
                    extractor.high_frequency_lpips_distance(
                        generated,
                        generated_mask,
                        mannequin,
                        mannequin_mask,
                        sigma=hf_sigma,
                        bbox=print_bbox,
                        output_size=crop_size,
                    )
                )
                output["print_status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                output["print_status"] = "failed"
                output["print_error"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            output["status"] = "failed"
            output["error"] = str(exc)
        csv_rows.append(output)
    save_feature_cache(source_cache_path, source_features)
    write_csv(out_dir / "garment_per_image.csv", csv_rows, FIELDS)
    valid = [row for row in csv_rows if row["status"] == "ok"]
    dino = metric_summary(row["garment_dino_to_mannequin"] for row in valid)
    lpips = metric_summary(row["garment_lpips_to_mannequin"] for row in valid)
    hf_lpips = metric_summary(
        row["garment_hf_lpips_to_mannequin"] for row in valid
    )
    gradient_sim = metric_summary(
        row["garment_gradient_sim_to_mannequin"] for row in valid
    )
    print_valid = [row for row in valid if row["print_status"] == "ok"]
    print_hf_lpips = metric_summary(
        row["print_region_hf_lpips"] for row in print_valid
    )
    print_grid, print_index = _build_print_zoom_grid(
        out_dir,
        csv_rows,
        case_count=int(hcfg.get("print_panel_cases", 16)),
        tile_size=int(hcfg.get("print_panel_tile_size", 180)),
    )
    fallback_count = sum(row["mask_source"].startswith("fallback") for row in valid)
    summary = {
        "status": "ok" if valid else "failed",
        "measures_against": "source mannequin image images/mannequin/{mid} using its SAM cloth mask",
        "generated_mask_protocol": "FASHN parsing garment union; fallback is projected mannequin SAM cloth mask",
        "count": len(valid),
        "failed": len(csv_rows) - len(valid),
        "fallback_mask_count": fallback_count,
        "fallback_mask_rate": fallback_count / len(valid) if valid else None,
        "garment_dino": dino,
        "garment_lpips": lpips,
        "garment_hf_lpips": hf_lpips,
        "garment_gradient_sim": gradient_sim,
        "print_region_hf_lpips": print_hf_lpips,
        "print_region_failed": len(valid) - len(print_valid),
        "print_zoom_grid": str(print_grid),
        "print_zoom_count": len(print_index),
        "garment_dino_by_type": grouped_metric(valid, "garment_type", "garment_dino_to_mannequin"),
        "garment_lpips_by_type": grouped_metric(valid, "garment_type", "garment_lpips_to_mannequin"),
        "garment_hf_lpips_by_type": grouped_metric(
            valid, "garment_type", "garment_hf_lpips_to_mannequin"
        ),
        "garment_gradient_sim_by_type": grouped_metric(
            valid, "garment_type", "garment_gradient_sim_to_mannequin"
        ),
        "print_region_hf_lpips_by_type": grouped_metric(
            print_valid, "garment_type", "print_region_hf_lpips"
        ),
        "csv": str(out_dir / "garment_per_image.csv"),
    }
    write_json(out_dir / "garment_summary.json", summary)
    plot_histogram(
        out_dir / "garment_dino_to_mannequin_hist.png",
        [float(row["garment_dino_to_mannequin"]) for row in valid],
        "Garment DINO to source mannequin",
        "DINO cosine (higher is better)",
    )
    plot_histogram(
        out_dir / "garment_hf_lpips_to_mannequin_hist.png",
        [float(row["garment_hf_lpips_to_mannequin"]) for row in valid],
        "Garment high-frequency LPIPS to source mannequin",
        "HF-LPIPS (lower is better)",
    )
    return summary
