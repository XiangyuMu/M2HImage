from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import torch

from conditions import load_yaml
from metrics_v2.common import read_json, resolve_path, subset_for_mids, validate_generated, write_json
from metrics_v2.distribution import run_distribution_metrics
from metrics_v2.garment import run_garment_metrics
from metrics_v2.hair import run_hair_metrics
from metrics_v2.identity import run_identity_metrics
from metrics_v2.panels import build_panels
from metrics_v2.parsing import build_generated_parsing
from metrics_v2.pose import run_pose_metrics
from metrics_v2.report import write_compare_report, write_run_report


ALL_STAGES = ("parsing", "garment", "hair", "pose", "identity", "distribution", "panels", "report")


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one(
    cfg: dict[str, Any],
    subset: dict[str, Any],
    run_name: str,
    stages: list[str],
    device: str,
    pose_device: str,
    output_root: Path,
) -> Path:
    root = Path(cfg["data"]["root"])
    run_cfg = cfg["metrics_v2"]["runs"][run_name]
    gen_dir = resolve_path(root, run_cfg["gen_dir"])
    legacy_dir = resolve_path(root, run_cfg["legacy_metrics_dir"])
    out_dir = output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = validate_generated(cfg, subset, gen_dir)
    if {"parsing", "garment", "hair", "panels"} & set(stages):
        write_json(out_dir / "parsing_summary.json", build_generated_parsing(cfg, rows, out_dir, device))
        cleanup_cuda()
    if "garment" in stages:
        run_garment_metrics(cfg, rows, out_dir, device)
        cleanup_cuda()
    if "hair" in stages:
        run_hair_metrics(cfg, rows, out_dir, device)
        cleanup_cuda()
    if "pose" in stages:
        run_pose_metrics(cfg, rows, out_dir, pose_device)
    if "identity" in stages:
        run_identity_metrics(cfg, subset, gen_dir, out_dir, device)
        cleanup_cuda()
    if "distribution" in stages:
        run_distribution_metrics(cfg, gen_dir, out_dir, device)
        cleanup_cuda()
    if "panels" in stages:
        build_panels(cfg, subset, rows, out_dir, run_name)
    if "report" in stages:
        write_run_report(cfg, run_name, out_dir, legacy_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="M2H system-level metrics v2; reads existing generations only.")
    parser.add_argument("--config", default="configs/metrics_v2.yaml")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--run", default="all", help="Configured run name, or all default runs.")
    parser.add_argument("--compare", nargs=2, metavar=("LEFT_RUN", "RIGHT_RUN"))
    parser.add_argument("--metrics", default="all", help="Comma-separated stages or all.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pose-device", default="cuda:1")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--smoke", action="store_true", help="Use first two sorted mannequin IDs and skip FID/KID.")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = Path(cfg["data"]["root"])
    subset = read_json(resolve_path(root, args.subset or cfg["data"]["cf_subset"]))
    if args.smoke:
        subset = subset_for_mids(subset, sorted(str(value) for value in subset["mannequins"])[:2])
    output_default = "eval/metrics_v2_smoke" if args.smoke else cfg["metrics_v2"]["output_root"]
    output_root = resolve_path(root, args.output_root or output_default)
    configured_runs = cfg["metrics_v2"]["runs"]
    default_runs = [
        str(value)
        for value in cfg["metrics_v2"].get("default_runs", ["a4", "b2cont"])
    ]
    runs = default_runs if args.run == "all" else [str(args.run)]
    unknown_runs = sorted(set(runs) - set(configured_runs))
    if unknown_runs:
        raise ValueError(
            f"unknown metrics_v2 runs: {unknown_runs}; configured={sorted(configured_runs)}"
        )
    compare_names = tuple(args.compare) if args.compare else (
        tuple(cfg["metrics_v2"].get("compare_runs", ["a4", "b2cont"]))
        if args.run == "all"
        else None
    )
    if compare_names and (len(compare_names) != 2 or set(compare_names) - set(configured_runs)):
        raise ValueError(f"invalid --compare runs {compare_names}; configured={sorted(configured_runs)}")
    if args.metrics == "all":
        stages = list(ALL_STAGES)
    else:
        stages = [value.strip() for value in args.metrics.split(",") if value.strip()]
        unknown = sorted(set(stages) - set(ALL_STAGES))
        if unknown:
            raise ValueError(f"unknown metrics_v2 stages: {unknown}")
    if args.smoke and "distribution" in stages:
        stages.remove("distribution")

    output_dirs: dict[str, Path] = {}
    if not args.compare_only:
        for run_name in runs:
            output_dirs[run_name] = run_one(cfg, subset, run_name, stages, args.device, args.pose_device, output_root)
    if args.compare_only and not compare_names:
        raise ValueError("--compare-only requires --compare LEFT_RUN RIGHT_RUN")
    if compare_names and not args.smoke and (args.compare_only or "report" in stages):
        left_name, right_name = compare_names
        default_compare = (left_name, right_name) == ("a4", "b2cont")
        compare_filename = (
            "compare.md"
            if default_compare
            else f"compare_{left_name}_vs_{right_name}.md"
        )
        write_compare_report(
            cfg,
            output_dirs.get(left_name, output_root / left_name),
            output_dirs.get(right_name, output_root / right_name),
            output_root / compare_filename,
            write_baseline=default_compare,
            left_name=left_name,
            right_name=right_name,
        )


if __name__ == "__main__":
    main()
