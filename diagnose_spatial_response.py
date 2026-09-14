from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from conditions import choose_dtype, load_yaml, make_image_ids, seed_everything
from dataset import PairedWarmupDataset
from train_paired import WarmupFlowModel, load_checkpoint, load_components


def _batch_prompt(batch: dict, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = batch["prompt_embeds"].to(device=device, dtype=dtype)
    pooled = batch["pooled_prompt_embeds"].to(device=device, dtype=dtype)
    return (
        prompt.unsqueeze(0) if prompt.ndim == 2 else prompt,
        pooled.unsqueeze(0) if pooled.ndim == 1 else pooled,
    )


def _condition_tokens(
    model: WarmupFlowModel,
    batch: dict,
    device: torch.device,
    dtype: torch.dtype,
    *,
    appearance_source: dict | None = None,
    hair_source: dict | None = None,
    hair_off: bool = False,
) -> torch.Tensor:
    appearance_source = appearance_source or batch
    hair_source = hair_source or batch
    if model.hair_enabled:
        hair_tokens = hair_source["hair_ref_tokens"].to(device=device, dtype=dtype)
        hair_positions = hair_source["hair_ref_positions"].to(device=device, dtype=dtype)
        hair_mask = hair_source["hair_ref_mask"].to(device=device, dtype=dtype)
        if hair_off:
            hair_tokens = torch.zeros_like(hair_tokens)
            hair_positions = torch.zeros_like(hair_positions)
            hair_mask = torch.zeros_like(hair_mask)
        hair_inputs = (hair_tokens, hair_positions, hair_mask)
    else:
        hair_inputs = (None, None, None)
    prompt, _ = _batch_prompt(batch, device, dtype)
    return model._condition_tokens(
        prompt,
        appearance_source["appearance"].to(device=device, dtype=dtype).unsqueeze(0),
        batch["garment"].to(device=device, dtype=dtype).unsqueeze(0),
        batch["head_pose"].to(device=device, dtype=dtype).unsqueeze(0),
        *hair_inputs,
    )


def _masked_rms(value: torch.Tensor, mask: np.ndarray | None = None) -> float:
    square = value.detach().float().square().mean(dim=-1)[0]
    if mask is None:
        return float(square.mean().sqrt().cpu())
    weights = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(square.device).clamp(0.0, 1.0)
    return float(((square * weights).sum() / weights.sum().clamp_min(1e-6)).sqrt().cpu())


def _regions(root: Path, sample_id: str, token_count: int) -> dict[str, np.ndarray | None]:
    path = root / "derived/region_masks_z" / f"{sample_id}.npz"
    if not path.exists():
        image_path = root / "derived/region_masks" / f"{sample_id}.npz"
        if not image_path.exists():
            return {"all": None}
        with np.load(image_path, allow_pickle=False) as row:
            result = {"all": None}
            for output_key, source_key in (
                ("cloth", "cloth_safe"),
                ("face", "id_strong"),
                ("body_bg", "body_bg"),
            ):
                image_mask = np.asarray(row[source_key], dtype=np.float32) / 255.0
                height, width = image_mask.shape
                if height % 16 or width % 16:
                    raise RuntimeError(
                        f"{image_path}: mask shape={image_mask.shape} is not divisible by 16"
                    )
                result[output_key] = image_mask.reshape(
                    height // 16, 16, width // 16, 16
                ).mean(axis=(1, 3)).reshape(-1)
    else:
        with np.load(path, allow_pickle=False) as row:
            result = {
                "all": None,
                "cloth": np.asarray(row["cloth_safe_z"], dtype=np.float32),
                "face": np.asarray(row["face_z"], dtype=np.float32),
                "body_bg": np.asarray(row["body_bg_z"], dtype=np.float32),
            }
    for name, mask in result.items():
        if mask is not None and mask.shape != (token_count,):
            raise RuntimeError(f"{path}: {name} shape={mask.shape}, expected={(token_count,)}")
    return result


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    cfg = load_yaml(args.config)
    seed_everything(int(cfg["experiment"]["seed"]))
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The 6144-token response probe requires a CUDA device")
    torch.cuda.set_device(device)
    dtype = choose_dtype(cfg["model"]["precision"])
    dataset = PairedWarmupDataset(cfg, "val", require_coverage=False)
    count = min(int(args.samples), len(dataset))
    if count < 2:
        raise RuntimeError("At least two validation samples are required")

    transformer, controlnet, _, adapter, pulid, notes = load_components(cfg, device, dtype)
    model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg)
    step = load_checkpoint(Path(args.ckpt), model)
    model.eval()
    root = Path(cfg["data"]["root"])
    rows: list[dict] = []

    for index in range(count):
        batch = dataset[index]
        donor = dataset[(index + 1) % count]
        sample_id = str(batch["sample_id"])
        donor_id = str(donor["sample_id"])
        prompt, pooled = _batch_prompt(batch, device, dtype)
        own_tokens = _condition_tokens(model, batch, device, dtype)
        hair_swap_tokens = _condition_tokens(
            model, batch, device, dtype, hair_source=donor
        )
        hair_off_tokens = _condition_tokens(model, batch, device, dtype, hair_off=True)
        identity_swap_tokens = _condition_tokens(
            model,
            batch,
            device,
            dtype,
            appearance_source=donor,
            hair_source=donor,
        )
        img_ids = make_image_ids(model.width, model.height, device, dtype)
        masks = _regions(root, sample_id, model.image_token_count)
        generator = torch.Generator(device=device).manual_seed(int(args.seed) + index)
        z = torch.randn(
            1,
            model.image_token_count,
            64,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        pose = batch["pose_latents"].to(device=device, dtype=dtype).unsqueeze(0)
        pulid_own = batch["pulid_id_embed"].to(device=device, dtype=dtype).unsqueeze(0)
        pulid_donor = donor["pulid_id_embed"].to(device=device, dtype=dtype).unsqueeze(0)
        own_ref = batch["garment_ref_latents"]
        donor_ref = donor["garment_ref_latents"]

        for tau_value in args.taus:
            tau = torch.tensor([float(tau_value)], device=device, dtype=dtype)
            cn = model._controlnet_forward(z, tau, prompt, pooled, pose, img_ids)

            def forward(tokens: torch.Tensor, pulid_embed: torch.Tensor, garment_ref: torch.Tensor) -> torch.Tensor:
                return model._transformer_forward(
                    z,
                    tau,
                    tokens,
                    pulid_embed,
                    cn,
                    pooled=pooled,
                    img_ids=img_ids,
                    garment_ref_latents=garment_ref,
                )

            baseline = forward(own_tokens, pulid_own, own_ref)
            variants = {
                "garment_swap": forward(own_tokens, pulid_own, donor_ref),
                "hair_swap": forward(hair_swap_tokens, pulid_own, own_ref),
                "hair_off": forward(hair_off_tokens, pulid_own, own_ref),
                "identity_swap": forward(identity_swap_tokens, pulid_donor, own_ref),
            }
            base_rms = _masked_rms(baseline)
            for variant, output in variants.items():
                diff = output - baseline
                row = {
                    "sample_id": sample_id,
                    "donor_id": donor_id,
                    "tau": float(tau_value),
                    "variant": variant,
                    "base_velocity_rms": base_rms,
                }
                for region, mask in masks.items():
                    response = _masked_rms(diff, mask)
                    row[f"{region}_response_rms"] = response
                    row[f"{region}_relative_response"] = response / max(base_rms, 1e-8)
                rows.append(row)

    summary: dict[str, dict[str, float]] = {}
    for variant in sorted({row["variant"] for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            key: mean(float(row[key]) for row in selected)
            for key in selected[0]
            if key.endswith("relative_response")
        }
    identity_all = summary["identity_swap"]["all_relative_response"]
    for variant, values in summary.items():
        values["fraction_of_identity_response"] = (
            values["all_relative_response"] / max(identity_all, 1e-8)
        )

    payload = {
        "checkpoint": str(args.ckpt),
        "checkpoint_step": step,
        "samples": count,
        "sample_ids": dataset.ids[:count],
        "taus": [float(value) for value in args.taus],
        "load_notes": notes,
        "summary": summary,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = [
        "# Step-500 Spatial-Condition Response Probe",
        "",
        f"- checkpoint: `{args.ckpt}`",
        f"- samples: {count}",
        f"- taus: {payload['taus']}",
        "- metric: RMS velocity change divided by baseline velocity RMS; identity swap is the response reference.",
        "",
        "| intervention | all | cloth | face | body/bg | fraction of identity response |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ("garment_swap", "hair_swap", "hair_off", "identity_swap"):
        values = summary[variant]
        report.append(
            f"| {variant} | {values.get('all_relative_response', float('nan')):.6f} | "
            f"{values.get('cloth_relative_response', float('nan')):.6f} | "
            f"{values.get('face_relative_response', float('nan')):.6f} | "
            f"{values.get('body_bg_relative_response', float('nan')):.6f} | "
            f"{values['fraction_of_identity_response']:.3f} |"
        )
    report.extend([
        "",
        "Interpretation: a near-zero swap response means the route is functionally ignored; a material",
        "response with poor decoded fidelity means the route is connected but the flow objective has not",
        "learned the required high-frequency correspondence yet.",
    ])
    output.with_suffix(".md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe garment/hair route response without training")
    parser.add_argument("--config", default="configs/spatial_warmup.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--taus", type=float, nargs="+", default=(0.8, 0.5, 0.2))
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output", default="artifacts/spatial_step500_response.json")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
