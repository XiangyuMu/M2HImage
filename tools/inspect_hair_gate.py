from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import default_collate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conditions import choose_dtype, load_yaml, seed_everything
from dataset import PairedWarmupDataset
from train_paired import (
    WarmupFlowModel,
    build_optimizer,
    configure_runtime,
    load_checkpoint,
    load_components,
)


def _parameter_group(optimizer, parameter: torch.nn.Parameter) -> dict[str, Any] | None:
    for index, group in enumerate(optimizer.param_groups):
        if any(candidate is parameter for candidate in group["params"]):
            return {
                "index": index,
                "name": str(group.get("group_name", "unnamed")),
                "lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
            }
    return None


def _tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach().float()
    return {
        "shape": list(detached.shape),
        "l2": float(detached.norm().cpu()),
        "rms": float(detached.square().mean().sqrt().cpu()),
        "nonzero_fraction": float((detached != 0).float().mean().cpu()),
        "finite": bool(torch.isfinite(detached).all().cpu()),
    }


def _grad_stats(parameter: torch.nn.Parameter) -> dict[str, Any]:
    if parameter.grad is None:
        return {"is_none": True, "value": None, "abs": None}
    grad = parameter.grad.detach().float()
    value = float(grad.reshape(-1)[0].cpu()) if grad.numel() == 1 else None
    return {
        "is_none": False,
        "value": value,
        "abs": float(grad.abs().max().cpu()),
        "l2": float(grad.norm().cpu()),
        "finite": bool(torch.isfinite(grad).all().cpu()),
    }


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    seed_everything(int(cfg["experiment"]["seed"]))
    configure_runtime(cfg, world_size=int(args.world_size), all_gpus_train=False)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("hair-gate dry-run requires a CUDA device")
    torch.cuda.set_device(device)
    dtype = choose_dtype(cfg["model"]["precision"])

    dataset = PairedWarmupDataset(cfg, "train", require_coverage=True)
    transformer, controlnet, vae, adapter, pulid, load_notes = load_components(
        cfg, device, dtype
    )
    model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg).train()
    optimizer = build_optimizer(model, cfg)
    checkpoint_step = load_checkpoint(Path(args.checkpoint), model, optimizer=optimizer)

    sample_index = int(args.sample_index) % len(dataset)
    batch = default_collate([dataset[sample_index]])
    batch["tau_override"] = torch.tensor([float(args.tau)], dtype=torch.float32)

    prompt = batch["prompt_embeds"].to(device=device, dtype=dtype)
    if prompt.ndim == 2:
        prompt = prompt.unsqueeze(0)
    appearance = batch["appearance"].to(device=device, dtype=dtype)
    garment = batch["garment"].to(device=device, dtype=dtype)
    head_pose = batch["head_pose"].to(device=device, dtype=dtype)
    hair_inputs = model._hair_inputs(batch, device, dtype)
    adapter_tokens = model.adapter(
        appearance,
        garment,
        head_pose,
        hair_ref_tokens=hair_inputs[0],
        hair_ref_positions=hair_inputs[1],
        hair_ref_mask=hair_inputs[2],
    )
    prompt_count = int(prompt.shape[1])
    hair_start_adapter = int(model.adapter.appearance_tokens)
    if model.adapter.use_legacy_garment_tokens:
        hair_start_adapter += int(model.adapter.max_garment_tokens)
    hair_stop_adapter = hair_start_adapter + int(model.adapter.hair_tokens)
    hair_slice = adapter_tokens[:, hair_start_adapter:hair_stop_adapter]
    cond_tokens = torch.cat((prompt, adapter_tokens), dim=1)
    hair_start_cond = prompt_count + hair_start_adapter
    hair_stop_cond = prompt_count + hair_stop_adapter

    raw_hair = model.adapter.hair_proj(hair_inputs[0]) + model.adapter.hair_pos_proj(
        hair_inputs[1]
    )
    raw_hair = raw_hair * hair_inputs[2].unsqueeze(-1)
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = model(batch)
    loss.backward()

    hair_gate = model.adapter.hair_gate
    appearance_gate = model.adapter.appearance_gate
    hair_grad = _grad_stats(hair_gate)
    appearance_grad = _grad_stats(appearance_gate)
    hair_group = _parameter_group(optimizer, hair_gate)
    appearance_group = _parameter_group(optimizer, appearance_gate)

    reasons: list[str] = []
    if not hair_gate.requires_grad:
        reasons.append("hair_gate.requires_grad is false")
    if hair_group is None:
        reasons.append("hair_gate is absent from optimizer param_groups")
    if hair_grad["is_none"]:
        reasons.append("hair_gate.grad is None after the real-batch backward")
    if _tensor_stats(hair_slice)["nonzero_fraction"] == 0.0:
        reasons.append("the post-gate hair token slice is identically zero")
    gate_order = str(model.adapter.launch_note().get("gate_order", "unknown"))
    if gate_order != "post_layernorm":
        reasons.append(f"gate order is {gate_order}, expected post_layernorm")

    if reasons:
        conclusion = "BUG-FIXABLE"
    elif hair_grad["abs"] is not None and float(hair_grad["abs"]) < float(
        args.route_dead_grad_threshold
    ):
        conclusion = "ROUTE-DEAD"
        reasons.append(
            "wiring is live but |hair_gate.grad| is below "
            f"{float(args.route_dead_grad_threshold):.3g}"
        )
    else:
        conclusion = "INCONCLUSIVE"
        reasons.append("wiring and gradient are live; one dry-run cannot explain the flat gate")

    payload = {
        "conclusion": conclusion,
        "reasons": reasons,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint_step,
        "config": str(Path(args.config).resolve()),
        "sample_id": str(batch["sample_id"][0]),
        "tau": float(args.tau),
        "loss": float(loss.detach().float().cpu()),
        "hair_gate": {
            "value": float(hair_gate.detach().float().cpu()),
            "requires_grad": bool(hair_gate.requires_grad),
            "optimizer_group": hair_group,
            "grad": hair_grad,
        },
        "appearance_gate_control": {
            "value": float(appearance_gate.detach().float().cpu()),
            "requires_grad": bool(appearance_gate.requires_grad),
            "optimizer_group": appearance_group,
            "grad": appearance_grad,
        },
        "gradient_ratio_hair_to_appearance": (
            None
            if hair_grad["abs"] is None
            or appearance_grad["abs"] is None
            or float(appearance_grad["abs"]) == 0.0
            else float(hair_grad["abs"]) / float(appearance_grad["abs"])
        ),
        "hair_route": {
            "gate_order": gate_order,
            "raw_projected_tokens": _tensor_stats(raw_hair),
            "post_gate_tokens": _tensor_stats(hair_slice),
            "valid_token_count": float(hair_inputs[2].detach().float().sum().cpu()),
            "adapter_slice": [hair_start_adapter, hair_stop_adapter],
            "encoder_condition_slice": [hair_start_cond, hair_stop_cond],
            "encoder_condition_slice_stats": _tensor_stats(
                cond_tokens[:, hair_start_cond:hair_stop_cond]
            ),
            "note": (
                "Hair tokens occupy encoder_hidden_states, not FLUX image hidden_states; "
                "the reported condition slice is the actual transformer input route."
            ),
        },
        "metrics": {
            key: float(value.detach().float().mean().cpu())
            for key, value in metrics.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        },
        "load_notes": load_notes,
        "peak_gib": float(torch.cuda.max_memory_allocated(device) / 1024**3),
        "read_only": True,
    }
    if not math.isfinite(payload["loss"]):
        raise RuntimeError("hair-gate dry-run produced a non-finite loss")
    return payload


def write_report(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hair = payload["hair_gate"]
    appearance = payload["appearance_gate_control"]
    route = payload["hair_route"]
    lines = [
        "# Hair Gate Step-2000 Inspection",
        "",
        f"**Conclusion: `{payload['conclusion']}`.** " + "; ".join(payload["reasons"]),
        "",
        f"- Checkpoint: `{payload['checkpoint']}` (step {payload['checkpoint_step']})",
        f"- Real batch sample: `{payload['sample_id']}`, tau={payload['tau']}",
        f"- Hair gate: value={hair['value']:.9f}, requires_grad={hair['requires_grad']}, "
        f"group={hair['optimizer_group']}, grad={hair['grad']}",
        f"- Appearance control: value={appearance['value']:.9f}, grad={appearance['grad']}",
        f"- Hair/appearance max-gradient ratio: `{payload['gradient_ratio_hair_to_appearance']}`",
        f"- Projected hair tokens: `{route['raw_projected_tokens']}`",
        f"- Post-gate hair tokens: `{route['post_gate_tokens']}`",
        f"- Transformer encoder condition slice: `{route['encoder_condition_slice']}`; "
        f"stats=`{route['encoder_condition_slice_stats']}`",
        f"- Gate order: `{route['gate_order']}`",
        f"- Peak allocated VRAM: {payload['peak_gib']:.3f} GiB",
        "",
        "The checkpoint and optimizer were loaded read-only. The dry-run called backward but did not call optimizer.step().",
        "",
        "```json",
        json.dumps(payload, indent=2, ensure_ascii=False),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only step-2000 hair-gate wiring inspection")
    parser.add_argument("--config", default="configs/spatial_warmup_resume_hair.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--route-dead-grad-threshold", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        default="docs/results/hair_gate_step2000_diagnosis.md",
    )
    args = parser.parse_args()
    payload = inspect(args)
    write_report(payload, Path(args.output))
    print(json.dumps({
        "conclusion": payload["conclusion"],
        "reasons": payload["reasons"],
        "hair_gate": payload["hair_gate"],
        "appearance_gate_control": payload["appearance_gate_control"],
        "output": args.output,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
