from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.stats import wilcoxon

from conditions import find_one, get_resolution, read_ids
from metrics.common import expected_rows, percentile, safe_mean, safe_median, safe_values, sha256_short


KEY_FIELDS = ("mid", "jid", "seed")


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["mid"]), str(row["jid"]), int(row["seed"])


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def metric_summary(values: Iterable[Any]) -> dict[str, Any]:
    vals = safe_values(list(values))
    return {
        "count": len(vals),
        "mean": safe_mean(vals),
        "median": safe_median(vals),
        "p10": percentile(vals, 10),
    }


def grouped_metric(rows: list[dict[str, Any]], group_key: str, metric_key: str) -> dict[str, Any]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key, "unknown"))].append(row.get(metric_key))
    return {key: metric_summary(values) for key, values in sorted(groups.items())}


def paired_test(
    a_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
    value_key: str,
) -> dict[str, Any]:
    a_map = {row_key(row): finite_float(row.get(value_key)) for row in a_rows}
    b_map = {row_key(row): finite_float(row.get(value_key)) for row in b_rows}
    keys = sorted(key for key in a_map.keys() & b_map.keys() if a_map[key] is not None and b_map[key] is not None)
    av = np.asarray([a_map[key] for key in keys], dtype=np.float64)
    bv = np.asarray([b_map[key] for key in keys], dtype=np.float64)
    if len(keys) == 0:
        return {"count": 0, "a_mean": None, "b_mean": None, "mean_diff": None, "p_two_sided": None}
    diff = av - bv
    if np.allclose(diff, 0.0):
        p_value = 1.0
    else:
        try:
            p_value = float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
        except ValueError:
            p_value = None
    return {
        "count": len(keys),
        "a_mean": float(av.mean()),
        "b_mean": float(bv.mean()),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "p_two_sided": p_value,
    }


def subset_for_mids(subset: dict[str, Any], mids: list[str]) -> dict[str, Any]:
    keep = set(mids)
    result = dict(subset)
    result["mannequins"] = [mid for mid in subset.get("mannequins", []) if mid in keep]
    result["pairs"] = [pair for pair in subset["pairs"] if str(pair["mannequin_id"]) in keep]
    result["garment_types"] = {key: value for key, value in subset.get("garment_types", {}).items() if key in keep}
    counts: dict[str, int] = defaultdict(int)
    for mid in result["mannequins"]:
        counts[result["garment_types"].get(mid, "unknown")] += 1
    result["garment_type_counts"] = dict(counts)
    return result


def validate_generated(cfg: dict[str, Any], subset: dict[str, Any], gen_dir: Path) -> list[dict[str, Any]]:
    rows = expected_rows(cfg, subset, gen_dir)
    missing = [str(row["path"]) for row in rows if not row["path"].exists()]
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} generated images under {gen_dir}:\n{preview}")
    return rows


def image_size(cfg: dict[str, Any]) -> tuple[int, int]:
    return get_resolution(cfg["data"]["resolution"])


def read_rgb(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def find_image(root: Path, rel_folder: str, sample_id: str) -> Path:
    return find_one(root / rel_folder, sample_id)


def deterministic_panel_cases(subset: dict[str, Any], seed: int, mid_count: int = 8, identities_per_mid: int = 2) -> list[tuple[str, str, int]]:
    rng = random.Random(int(seed))
    mids = sorted({str(pair["mannequin_id"]) for pair in subset["pairs"]})
    selected = sorted(rng.sample(mids, min(int(mid_count), len(mids))))
    by_mid: dict[str, list[str]] = defaultdict(list)
    for pair in subset["pairs"]:
        by_mid[str(pair["mannequin_id"])].append(str(pair["identity_id"]))
    cases: list[tuple[str, str, int]] = []
    for mid in selected:
        identities = sorted(set(by_mid[mid]))
        chosen = sorted(rng.sample(identities, min(int(identities_per_mid), len(identities))))
        cases.extend((mid, jid, 0) for jid in chosen)
    return cases


def test_split_images(cfg: dict[str, Any]) -> list[Path]:
    root = Path(cfg["data"]["root"])
    split = resolve_path(root, cfg["data"].get("test_split", "splits/test.txt"))
    return [find_image(root, "images/human", sample_id) for sample_id in read_ids(split)]


def provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    vcfg = cfg["metrics_v2"]
    return {
        "suite_version": str(vcfg.get("version", "2.1.0")),
        "fashn_model": str(vcfg["parsing"]["model_dir"]),
        "fashn_hash": sha256_short(Path(vcfg["parsing"]["model_dir"]) / "model.safetensors"),
        "fashn_labels": str(resolve_path(Path(cfg["data"]["root"]), vcfg["parsing"]["labels_json"])),
        "fashn_labels_hash": sha256_short(resolve_path(Path(cfg["data"]["root"]), vcfg["parsing"]["labels_json"])),
        "dino_checkpoint": str(vcfg["dino"]["checkpoint"]),
        "dino_hash": sha256_short(vcfg["dino"]["checkpoint"]),
        "adaface_checkpoint": str(cfg["metrics"]["heldout_id"]["checkpoint"]),
        "adaface_hash": sha256_short(cfg["metrics"]["heldout_id"]["checkpoint"]),
        "dwpose_detector": str(vcfg["pose"]["detector_checkpoint"]),
        "dwpose_detector_hash": sha256_short(vcfg["pose"]["detector_checkpoint"]),
        "dwpose_pose": str(vcfg["pose"]["pose_checkpoint"]),
        "dwpose_pose_hash": sha256_short(vcfg["pose"]["pose_checkpoint"]),
    }
