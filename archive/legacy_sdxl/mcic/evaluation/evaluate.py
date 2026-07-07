from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mcic.inference.generate import generate_image
from mcic.models.identity import offline_identity_embedding, reference_face_box
from mcic.utils.config import add_config_args, load_config, write_json
from mcic.utils.image import compute_canvas_transform, fit_to_canvas


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-8))


def masked_l1(prediction: Image.Image, target: Image.Image, mask: Image.Image) -> float:
    pred = np.asarray(prediction).astype(np.float32) / 255
    truth = np.asarray(target).astype(np.float32) / 255
    weights = np.asarray(mask).astype(np.float32) / 255
    denominator = max(weights.sum() * 3, 1.0)
    return float((np.abs(pred - truth) * weights[..., None]).sum() / denominator)


def perceptual_metrics(prediction: Image.Image, target: Image.Image, mask: Image.Image, lpips_model=None) -> dict[str, float]:
    from skimage.metrics import structural_similarity

    pred = np.asarray(prediction).astype(np.float32) / 255
    truth = np.asarray(target).astype(np.float32) / 255
    weights = (np.asarray(mask).astype(np.float32) / 255)[..., None]
    masked_pred = pred * weights
    masked_truth = truth * weights
    output = {
        "ssim": float(structural_similarity(pred, truth, channel_axis=2, data_range=1.0)),
        "cloth_ssim": float(structural_similarity(masked_pred, masked_truth, channel_axis=2, data_range=1.0)),
    }
    if lpips_model is not None:
        import torch

        to_tensor = lambda value: torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        with torch.no_grad():
            output["lpips"] = float(lpips_model(to_tensor(pred), to_tensor(truth)).item())
            output["cloth_lpips"] = float(lpips_model(to_tensor(masked_pred), to_tensor(masked_truth)).item())
    return output


def generated_identity(image: Image.Image, config: dict):
    try:
        box = reference_face_box(image, config)
        return offline_identity_embedding(image, box, config)
    except RuntimeError:
        return None


def evaluate(config: dict, checkpoint: str) -> dict:
    root = Path(config["data"]["root"])
    metadata = root / config["data"].get("cache_dir", "cache_mcic") / "metadata.jsonl"
    with metadata.open("r", encoding="utf-8") as handle:
        rows = [row for line in handle if (row := json.loads(line)).get("split") == "test" and row["face_quality_pass"]]
    rows = rows[: int(config.get("inference", {}).get("max_samples", 100))]
    if not rows:
        raise RuntimeError("No test metadata found; run preprocessing and ensure the split has test samples.")
    if len(rows) < 2:
        raise RuntimeError("Counterfactual evaluation requires at least two valid test identities.")
    output = Path(config["experiment"].get("output_root", "outputs")) / "evaluation"
    samples = output / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    try:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").eval()
    except ImportError:
        lpips_model = None
    paired_scores, cf_scores = [], []
    for index, row in enumerate(rows):
        man_input = Image.open(row["mannequin_path"]).convert("RGB")
        transform = compute_canvas_transform(man_input, config["image"]["height"], config["image"]["width"])
        target = fit_to_canvas(Image.open(row["human_path"]).convert("RGB"), transform)
        cloth = fit_to_canvas(Image.open(row["cloth_safe_mask_path"]).convert("L"), transform, is_mask=True)
        paired, _, _ = generate_image(config, checkpoint, row["mannequin_path"], row["human_path"], row["paired_mask_path"])
        paired.save(samples / f"paired_{row['sample_id'].replace('/', '__')}.png")
        paired_face = generated_identity(paired, config)
        paired_scores.append(
            {
                "sample_id": row["sample_id"],
                "cloth_l1": masked_l1(paired, target, cloth),
                "face_detected": paired_face is not None,
                **perceptual_metrics(paired, target, cloth, lpips_model),
            }
        )
        target_identity = rows[(index + 1) % len(rows)] if len(rows) > 1 else row
        counterfactual, _, _ = generate_image(
            config, checkpoint, row["mannequin_path"], target_identity["human_path"], row["cf_mask_path"]
        )
        counterfactual.save(samples / f"cf_{row['sample_id'].replace('/', '__')}.png")
        generated_id = generated_identity(counterfactual, config)
        source_id = np.load(row["identity_embedding_path"])
        target_id = np.load(target_identity["identity_embedding_path"])
        score = {
            "sample_id": row["sample_id"],
            "target_sample_id": target_identity["sample_id"],
            "face_detected": generated_id is not None,
            "cloth_l1": masked_l1(counterfactual, target, cloth),
            **perceptual_metrics(counterfactual, target, cloth, lpips_model),
        }
        if generated_id is not None:
            score["sim_target"] = cosine(generated_id, target_id)
            score["sim_source"] = cosine(generated_id, source_id)
            score["delta_id"] = score["sim_target"] - score["sim_source"]
        cf_scores.append(score)
    def mean(name, scores):
        values = [row[name] for row in scores if name in row]
        return float(np.mean(values)) if values else None
    metrics = {
        "resolution": [config["image"]["height"], config["image"]["width"]],
        "sample_count": len(rows),
        "paired": {
            "cloth_l1_mean": float(np.mean([row["cloth_l1"] for row in paired_scores])),
            "face_detect_rate": float(np.mean([row["face_detected"] for row in paired_scores])),
            "ssim_mean": mean("ssim", paired_scores),
            "cloth_ssim_mean": mean("cloth_ssim", paired_scores),
            "lpips_mean": mean("lpips", paired_scores),
            "cloth_lpips_mean": mean("cloth_lpips", paired_scores),
            "per_sample": paired_scores,
        },
        "counterfactual": {
            "cloth_l1_mean": float(np.mean([row["cloth_l1"] for row in cf_scores])),
            "face_detect_rate": float(np.mean([row["face_detected"] for row in cf_scores])),
            "sim_target_mean": mean("sim_target", cf_scores),
            "sim_source_mean": mean("sim_source", cf_scores),
            "delta_id_mean": mean("delta_id", cf_scores),
            "ssim_mean": mean("ssim", cf_scores),
            "cloth_ssim_mean": mean("cloth_ssim", cf_scores),
            "lpips_mean": mean("lpips", cf_scores),
            "cloth_lpips_mean": mean("cloth_lpips", cf_scores),
            "per_sample": cf_scores,
        },
    }
    write_json(output / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paired and counterfactual MCIC output.")
    add_config_args(parser)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    metrics = evaluate(load_config(args.config, args.overrides), args.checkpoint)
    print(json.dumps({key: value for key, value in metrics.items() if key != "counterfactual"}, indent=2))
    delta_id = metrics["counterfactual"]["delta_id_mean"]
    print(f"Counterfactual Delta ID: {delta_id:.4f}" if delta_id is not None else "No valid generated faces for Delta ID.")


if __name__ == "__main__":
    main()
