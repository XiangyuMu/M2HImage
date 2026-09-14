from __future__ import annotations

from pathlib import Path
from typing import Any

from metrics_v2.common import test_split_images, write_json


def _stage_real_test(cfg: dict[str, Any], out_dir: Path) -> Path:
    stage = out_dir / "fid_real_test"
    stage.mkdir(parents=True, exist_ok=True)
    images = test_split_images(cfg)
    for index, source in enumerate(images):
        target = stage / f"{index:05d}{source.suffix.lower()}"
        if not target.exists():
            target.symlink_to(source.resolve())
    return stage


def run_distribution_metrics(
    cfg: dict[str, Any], gen_dir: str | Path, out_dir: str | Path, device: str
) -> dict[str, Any]:
    import torch_fidelity

    out_dir = Path(out_dir)
    real_dir = _stage_real_test(cfg, out_dir)
    dcfg = cfg["metrics_v2"]["distribution"]
    use_cuda = str(device).startswith("cuda")
    metrics = torch_fidelity.calculate_metrics(
        input1=str(gen_dir),
        input2=str(real_dir),
        cuda=use_cuda,
        isc=False,
        fid=True,
        kid=True,
        kid_subsets=int(dcfg.get("kid_subsets", 50)),
        kid_subset_size=int(dcfg.get("kid_subset_size", 400)),
        feature_layer_fid=str(dcfg.get("feature_layer_fid", 2048)),
        rng_seed=int(dcfg.get("rng_seed", cfg["metrics_v2"].get("seed", 2020))),
        verbose=True,
    )
    summary = {
        "status": "ok",
        "measures_against": "images/human IDs from frozen splits/test.txt",
        "generated_dir": str(gen_dir),
        "real_test_dir": str(real_dir),
        "real_count": len(test_split_images(cfg)),
        "fid": float(metrics["frechet_inception_distance"]),
        "kid_mean": float(metrics["kernel_inception_distance_mean"]),
        "kid_std": float(metrics["kernel_inception_distance_std"]),
        "resize_protocol": "torch-fidelity InceptionV3 canonical preprocessing (299x299 internally)",
    }
    write_json(out_dir / "distribution_summary.json", summary)
    return summary
