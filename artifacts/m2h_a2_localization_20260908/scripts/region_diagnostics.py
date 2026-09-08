"""Bounded source-mask clothing region diagnostics for M2H A2/B2/A4 rows.

This script is descriptive only. It partitions the fixed source M clothing mask
without using outputs, then computes source fidelity and same-arm two-reference
output-output stability metrics in those fixed regions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


MASK_CONFIG = {
    "version": "m2h_a2_region_diagnostics_20260908",
    "mask_source": "cache/masks/{mid}.png",
    "source_image": "root/images/mannequin/{mid}.png",
    "erode_base_radius_px": 8,
    "erode_base_width_px": 768,
    "high_texture_quantile": 0.75,
    "high_texture_definition": "top 25% source-gradient pixels within fixed M mask",
    "high_texture_tie_policy": "gradient >= quantile threshold and gradient > 0; ties can exceed 25%",
    "gradient": "native-resolution source RGB luma central-difference magnitude",
    "output_selection": "none; all regions are source-mask fixed",
}

REGIONS = ("garment", "interior", "boundary", "hightexture", "lowtexture")
EXPECTED_BRANCHES = ("a2_inputOnly_freshI", "b2_inputOnly_freshI", "a4_inputOnly_freshI")
EXPECTED_TAU = 0
EXPECTED_K = 1
EXPECTED_SEED = 0


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("mid"),
        row.get("jid"),
        row.get("branch"),
        row.get("tau"),
        row.get("k"),
        row.get("seed"),
    )


def pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("mid"), row.get("branch"), row.get("tau"), row.get("k"), row.get("seed"))


def load_rows(path: Path, expected_branches: tuple[str, ...] = EXPECTED_BRANCHES) -> list[dict[str, Any]]:
    return validate_rows([json.loads(line) for line in path.read_text().splitlines() if line.strip()], expected_branches)


def validate_rows(rows: list[dict[str, Any]], expected_branches: tuple[str, ...] = EXPECTED_BRANCHES) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    duplicate_keys: list[tuple[Any, ...]] = []
    for row in rows:
        key = row_key(row)
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)
        missing = [name for name in ("mid", "jid", "branch", "tau", "k", "seed", "path") if name not in row]
        if missing:
            raise ValueError(f"row missing required fields {missing}: {row}")
    if duplicate_keys:
        raise ValueError(f"duplicate diagnostic rows: {duplicate_keys[:5]}")
    bad_fixed = [
        row_key(row)
        for row in rows
        if row["tau"] != EXPECTED_TAU or row["k"] != EXPECTED_K or row["seed"] != EXPECTED_SEED
    ]
    if bad_fixed:
        raise ValueError(f"expected only tau0/k1/seed0 rows, found: {bad_fixed[:8]}")

    observed_branches = {str(row["branch"]) for row in rows}
    expected_branch_set = set(expected_branches)
    if observed_branches != expected_branch_set:
        raise ValueError(f"expected exact branches {sorted(expected_branch_set)}, found {sorted(observed_branches)}")

    by_pair: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[pair_key(row)].append(row)
    bad_pairs = {
        key: [row.get("jid") for row in group]
        for key, group in by_pair.items()
        if len(group) != 2 or len({row.get("jid") for row in group}) != 2
    }
    if bad_pairs:
        preview = list(bad_pairs.items())[:8]
        raise ValueError(f"missing or invalid 2-reference pair keys: {preview}")

    branch_pair_sets: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for row in rows:
        branch_pair_sets[str(row["branch"])].add((row["mid"], row["tau"], row["k"], row["seed"]))
    reference_branch = expected_branches[0]
    reference_pairs = branch_pair_sets[reference_branch]
    bad_branch_sets = {
        branch: {
            "missing": sorted(reference_pairs - branch_pair_sets[branch])[:8],
            "extra": sorted(branch_pair_sets[branch] - reference_pairs)[:8],
        }
        for branch in expected_branches
        if branch_pair_sets[branch] != reference_pairs
    }
    if bad_branch_sets:
        raise ValueError(f"all observed arm pair sets must be identical: {bad_branch_sets}")
    return rows


def filter_limit_mids(rows: list[dict[str, Any]], limit_mids: int | None) -> list[dict[str, Any]]:
    if limit_mids is None:
        return rows
    if limit_mids <= 0:
        raise ValueError("--limit-mids must be positive")
    selected = set(sorted({str(row["mid"]) for row in rows})[:limit_mids])
    return [row for row in rows if str(row["mid"]) in selected]


def read_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    arr = np.asarray(Image.open(path).convert("RGB"))
    if arr.ndim != 3 or arr.shape[2] != 3 or min(arr.shape[:2]) <= 0:
        raise ValueError(f"invalid RGB image dimensions for {path}: {arr.shape}")
    return arr


def read_mask(path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    mask = np.asarray(Image.open(path).convert("L")) > 127
    if mask.shape != expected_hw:
        raise ValueError(f"mask/source dimension mismatch for {path}: {mask.shape} vs {expected_hw}")
    return mask


def erode_radius_for_shape(shape: tuple[int, int]) -> int:
    width = shape[1]
    return max(1, int(round(MASK_CONFIG["erode_base_radius_px"] * width / MASK_CONFIG["erode_base_width_px"])))


def binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if radius <= 0:
        return mask.copy()
    size = 2 * radius + 1
    padded = np.pad(mask.astype(np.uint8), radius, mode="constant", constant_values=0)
    image = Image.fromarray(padded * 255, mode="L")
    eroded = image.filter(ImageFilter.MinFilter(size=size))
    cropped = np.asarray(eroded)[radius:-radius, radius:-radius]
    return cropped > 127


def source_gradient_luma(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gy, gx = np.gradient(luma)
    return np.sqrt(gx * gx + gy * gy)


def partition_source_mask(source_rgb: np.ndarray, mask: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if source_rgb.shape[:2] != mask.shape:
        raise ValueError(f"source/mask shape mismatch: {source_rgb.shape[:2]} vs {mask.shape}")
    radius = erode_radius_for_shape(mask.shape)
    interior = binary_erode(mask, radius)
    boundary = mask & ~interior
    gradient = source_gradient_luma(source_rgb)
    mask_values = gradient[mask]
    if mask_values.size:
        threshold = float(np.quantile(mask_values, MASK_CONFIG["high_texture_quantile"]))
        positive_values = mask_values[mask_values > 0]
        hightexture = mask & (gradient >= threshold) & (gradient > 0)
    else:
        threshold = None
        positive_values = mask_values
        hightexture = np.zeros_like(mask, dtype=bool)
    lowtexture = mask & ~hightexture
    regions = {
        "garment": mask,
        "interior": interior,
        "boundary": boundary,
        "hightexture": hightexture,
        "lowtexture": lowtexture,
    }
    support = {name: int(value.sum()) for name, value in regions.items()}
    total = int(mask.size)
    qualification = {
        "image_hw": list(mask.shape),
        "mask_pixels": support["garment"],
        "mask_fraction": support["garment"] / total if total else 0.0,
        "erode_radius_px": radius,
        "texture_threshold": threshold,
        "positive_gradient_pixels": int(positive_values.size),
        "high_texture_ties_can_exceed_25pct": True,
        "region_support": support,
        "region_eligible": {name: count > 0 for name, count in support.items()},
    }
    return regions, qualification


def masked_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) == 0:
        return None
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)) / 255.0
    return float(diff[mask].mean())


def masked_gradient_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) == 0:
        return None
    return float(np.abs(source_gradient_luma(a) - source_gradient_luma(b))[mask].mean())


class RegionMetricExtractor:
    def __init__(self, repo: Path, device: str):
        sys.path.insert(0, str(repo))
        from conditions import load_yaml
        from metrics_v2.features import RegionFeatureExtractor, cosine

        self.cfg = load_yaml(repo / "configs/metrics_v2.yaml")
        self.model = RegionFeatureExtractor(self.cfg, device)
        if str(device).startswith('cuda') and self.model.device.type != 'cuda':
            raise RuntimeError('unexpected CPU fallback for regional metrics')
        self.cosine = cosine
        self.dino_checkpoint = Path(self.cfg["metrics_v2"]["dino"]["checkpoint"])
        self._dino_cache: dict[tuple[int, int], Any] = {}

    def dino_feature(self, image: np.ndarray, mask: np.ndarray) -> Any:
        key = (id(image), id(mask))
        if key not in self._dino_cache:
            self._dino_cache[key] = self.model.dino_feature(image, mask.astype(np.uint8))
        return self._dino_cache[key]

    def dino_similarity(self, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
        if int(mask.sum()) == 0:
            return None
        return float(self.cosine(self.dino_feature(a, mask), self.dino_feature(b, mask)))

    def hf_lpips_distance(self, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
        if int(mask.sum()) == 0:
            return None
        mask_u8 = mask.astype(np.uint8)
        return float(self.model.high_frequency_lpips_distance(a, mask_u8, b, mask_u8, sigma=2, output_size=256))


def resolve_root(cache: Path, root_arg: Path | None) -> Path:
    if root_arg is not None:
        return root_arg
    audit_path = cache / "audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(f"--root omitted and cache audit not found: {audit_path}")
    audit = json.loads(audit_path.read_text())
    return Path(audit["config"]["data"]["root"])


def summarize(rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[row["metric_status"]] += 1
    group_counts: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['branch']}__t{row['tau']}__k{row['k']}"
        entry = group_counts.setdefault(key, {"n": 0, "status_counts": defaultdict(int)})
        entry["n"] += 1
        entry["status_counts"][row["metric_status"]] += 1
    for entry in group_counts.values():
        entry["status_counts"] = dict(entry["status_counts"])
    return {
        "per_image_region_rows": len(rows),
        "cross_ref_region_rows": len(pair_rows),
        "status_counts": dict(status_counts),
        "groups": group_counts,
        "interpretation": "Descriptive fixed source-mask diagnostics only; no naturalness or causal success claims.",
    }


def run(args: argparse.Namespace) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    expected_branches = tuple(item.strip() for item in args.expected_branches.split(",") if item.strip())
    input_rows = load_rows(args.rows, expected_branches)
    rows = filter_limit_mids(input_rows, args.limit_mids)
    rows = validate_rows(rows, expected_branches)
    root = resolve_root(args.cache, args.root)
    extractor = RegionMetricExtractor(args.repo, args.device)
    source_cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]] = {}
    image_cache: dict[Path, np.ndarray] = {}
    per_image: list[dict[str, Any]] = []
    pair_metrics: list[dict[str, Any]] = []
    started = time.monotonic()
    resource = {
        "max_hours": args.max_hours,
        "started_monotonic": started,
        "budget_exceeded": False,
        "gpu_metrics_launched": True,
    }

    metadata = {
        "formal_result": False,
        "script_sha256": sha256_path(Path(__file__)),
        "rows_sha256": sha256_path(args.rows),
        "mask_config": MASK_CONFIG,
        "mask_config_sha256": stable_json_hash(MASK_CONFIG),
        "dino_checkpoint_sha256": sha256_path(extractor.dino_checkpoint) if extractor.dino_checkpoint.exists() else None,
        "hf_lpips": "metrics_v2.features.RegionFeatureExtractor.high_frequency_lpips_distance sigma=2 output_size=256",
        "native_resolution_corollary": "region_gradient_mae is computed before extractor resizing/highpass to flag sparse-mask edge confounds",
        "missing_outputs": "recorded per row as metric_status=output_failed; not silently dropped",
        "input_rows": len(input_rows),
        "selected_rows": len(rows),
        "limit_mids": args.limit_mids,
        "expected_branches": expected_branches,
        "resource_json": "resources.json",
    }
    (out / "mask_config.json").write_text(json.dumps(metadata["mask_config"], indent=2) + "\n")

    def check_budget() -> None:
        elapsed_hours = (time.monotonic() - started) / 3600.0
        resource["elapsed_hours"] = elapsed_hours
        if elapsed_hours > args.max_hours:
            resource["budget_exceeded"] = True
            raise TimeoutError(f"max-hours budget exceeded: {elapsed_hours:.4f} > {args.max_hours}")

    def source_entry(mid: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
        if mid not in source_cache:
            source = read_rgb(root / "images" / "mannequin" / f"{mid}.png")
            mask = read_mask(args.cache / "masks" / f"{mid}.png", source.shape[:2])
            regions, qualification = partition_source_mask(source, mask)
            qualification['source_image_sha256'] = sha256_path(root / 'images/mannequin' / f'{mid}.png')
            qualification['source_mask_sha256'] = sha256_path(args.cache / 'masks' / f'{mid}.png')
            qualification['region_mask_sha256'] = {
                name: hashlib.sha256(np.packbits(value).tobytes()).hexdigest()
                for name, value in regions.items()
            }
            source_cache[mid] = (source, regions, qualification)
        return source_cache[mid]

    def output_image(path_value: str) -> np.ndarray:
        path = Path(path_value)
        if path not in image_cache:
            image_cache[path] = read_rgb(path)
        return image_cache[path]

    with (out / "per_image.jsonl").open("x") as handle:
        for row in rows:
            base = {name: row[name] for name in ("mid", "jid", "seed", "branch")}
            base.update(tau=row["tau"], k=row["k"], path=row["path"])
            try:
                check_budget()
                source, regions, qualification = source_entry(str(row["mid"]))
                image = output_image(str(row["path"]))
                if image.shape != source.shape:
                    raise ValueError(f"output/source dimension mismatch: {image.shape} vs {source.shape}")
            except Exception as exc:
                try:
                    source, regions, qualification = source_entry(str(row["mid"]))
                except Exception:
                    regions = {name: np.zeros((0, 0), dtype=bool) for name in REGIONS}
                    qualification = None
                image = None
                error = str(exc)
            else:
                error = None
            for region_name in REGIONS:
                mask = regions[region_name]
                value = dict(base)
                value.update(
                    region=region_name,
                    metric_status="ok" if error is None else "output_failed",
                    eligible=bool(mask.sum()),
                    support=int(mask.sum()),
                    source_mask=qualification,
                    region_dino=None if error is not None else extractor.dino_similarity(source, image, mask),
                    region_hf_lpips=None if error is not None else extractor.hf_lpips_distance(source, image, mask),
                    region_pixel_mae=None if error is not None else masked_mae(source, image, mask),
                    region_gradient_mae=None if error is not None else masked_gradient_mae(source, image, mask),
                )
                if error is not None:
                    value["error"] = error
                per_image.append(value)
                handle.write(json.dumps(value, allow_nan=False) + "\n")
            handle.flush()

    by_pair: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[pair_key(row)].append(row)
    with (out / "cross_ref.jsonl").open("x") as handle:
        for key, group in sorted(by_pair.items()):
            left, right = sorted(group, key=lambda item: str(item["jid"]))
            base = {
                "mid": key[0],
                "branch": key[1],
                "tau": key[2],
                "k": key[3],
                "seed": key[4],
                "jid_left": left["jid"],
                "jid_right": right["jid"],
                "path_left": left["path"],
                "path_right": right["path"],
            }
            try:
                check_budget()
                _source, regions, qualification = source_entry(str(key[0]))
                left_image = output_image(str(left["path"]))
                right_image = output_image(str(right["path"]))
                if left_image.shape != right_image.shape:
                    raise ValueError(f"pair output dimension mismatch: {left_image.shape} vs {right_image.shape}")
            except Exception as exc:
                try:
                    _source, regions, qualification = source_entry(str(key[0]))
                except Exception:
                    regions = {name: np.zeros((0, 0), dtype=bool) for name in REGIONS}
                    qualification = None
                left_image = None
                right_image = None
                error = str(exc)
            else:
                error = None
            for region_name in REGIONS:
                mask = regions[region_name]
                value = dict(base)
                value.update(
                    region=region_name,
                    metric_status="ok" if error is None else "pair_failed",
                    eligible=bool(mask.sum()),
                    support=int(mask.sum()),
                    source_mask=qualification,
                    cross_ref_hf_lpips=None if error is not None else extractor.hf_lpips_distance(left_image, right_image, mask),
                    cross_ref_pixel_mae=None if error is not None else masked_mae(left_image, right_image, mask),
                    cross_ref_gradient_mae=None if error is not None else masked_gradient_mae(left_image, right_image, mask),
                )
                if error is not None:
                    value["error"] = error
                pair_metrics.append(value)
                handle.write(json.dumps(value, allow_nan=False) + "\n")
            handle.flush()

    result = {
        "metadata": metadata,
        "source_audit": {mid: value[2] for mid, value in source_cache.items()},
        "seconds": time.monotonic() - started,
        "summary": summarize(per_image, pair_metrics),
    }
    resource["elapsed_hours"] = result["seconds"] / 3600.0
    resource["completed_per_image_region_rows"] = len(per_image)
    resource["completed_cross_ref_region_rows"] = len(pair_metrics)
    (out / "resources.json").write_text(json.dumps(resource, indent=2, allow_nan=False) + "\n")
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (out / "READY").write_text("region diagnostics complete; failures recorded in jsonl outputs\n")
    print(json.dumps({"rows": len(per_image), "pairs": len(pair_metrics), "out": str(out)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True, help="Combined A2/B2/A4 rows.jsonl with generated image paths.")
    parser.add_argument("--repo", type=Path, required=True, help="M2HImage repo containing metrics_v2.")
    parser.add_argument("--cache", type=Path, required=True, help="Fresh cache containing fixed source M masks and audit.json.")
    parser.add_argument("--root", type=Path, default=None, help="Dataset root; defaults to cache audit config data root.")
    parser.add_argument("--out", type=Path, required=True, help="New empty output directory.")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--limit-mids", type=int, default=None, help="Smoke mode: first N mids by sorted source id, retaining all arms/refs.")
    parser.add_argument("--max-hours", type=float, default=2.0, help="Wall-clock guard for bounded GPU metric work.")
    parser.add_argument("--expected-branches", default=",".join(EXPECTED_BRANCHES), help="Comma-separated exact arm set expected in rows.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
