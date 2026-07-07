#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEFAULT_LOCAL_MODEL = Path("/data/muxiangyu/pythonPrograms/FLUX/modelsLib/Flux_1_dev")
DEFAULT_CONTROLNET_ID = "InstantX/FLUX.1-dev-Controlnet-Union"
DEFAULT_CONTROL_MODE = 4
PLACEHOLDER_CONTROLNET_CONFIG = {
    "_class_name": "FluxControlNetModel",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": True,
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 5,
    "num_mode": 10,
    "num_single_layers": 10,
    "patch_size": 1,
    "pooled_projection_dim": 768,
}

@dataclass
class VramConfig:
    name: str
    rank: int
    forwards: int
    decodes: int
    resolution: int = 512
    shared_diff: bool = False

@dataclass
class VramResult:
    name: str
    rank: int
    forwards: int
    decodes: int
    resolution: int
    offload: str
    peak_bytes: int | None
    feasible: str
    status: str
    failed_step: str
    notes: str

def gb(value: int | float) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"

def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def resolve_model_id(model_id: str, local_model: Path) -> str:
    if model_id == DEFAULT_MODEL_ID and local_model.exists():
        return str(local_model)
    return model_id

def load_transformer(model_id: str, dtype: torch.dtype, device: torch.device):
    from diffusers import FluxTransformer2DModel
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype, local_files_only=Path(model_id).exists()
    )
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    transformer.to(device=device, dtype=dtype)
    transformer.train()
    return transformer

def attach_lora(transformer: torch.nn.Module, rank: int) -> str:
    from peft import LoraConfig
    for param in transformer.parameters():
        param.requires_grad_(False)
    target_modules = ["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
    config = LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=target_modules)
    transformer.add_adapter(config)
    for name, param in transformer.named_parameters():
        param.requires_grad_("lora" in name.lower())
    return f"LoRA rank={rank}, trainable={count_trainable_params(transformer):,} params"

def make_optimizer(params: list[torch.nn.Parameter]) -> tuple[Any, str]:
    try:
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(params, lr=1e-4), "PagedAdamW8bit instantiated; no optimizer.step()"
    except Exception as exc:
        return torch.optim.AdamW(params, lr=1e-4), f"AdamW fallback instantiated; no optimizer.step(); bitsandbytes error={exc}"

def load_controlnet(controlnet_id: str, dtype: torch.dtype):
    from diffusers import FluxControlNetModel
    notes = []
    try:
        controlnet = FluxControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype, local_files_only=True)
        notes.append(f"real ControlNet loaded from {controlnet_id}")
    except Exception as exc:
        try:
            from accelerate import init_empty_weights
            with init_empty_weights():
                controlnet = FluxControlNetModel.from_config(PLACEHOLDER_CONTROLNET_CONFIG)
            controlnet.to_empty(device="cpu")
            controlnet.to(dtype=dtype)
            for param in controlnet.parameters():
                param.data.zero_()
        except Exception:
            controlnet = FluxControlNetModel.from_config(PLACEHOLDER_CONTROLNET_CONFIG)
            controlnet.to(dtype=dtype)
        notes.append(
            "ControlNet load failed; using same-architecture InstantX Union placeholder "
            "(num_layers=5,num_single_layers=10,num_mode=10), 待换真实权重; "
            f"load_error={str(exc).splitlines()[0]}"
        )
    if hasattr(controlnet, "enable_gradient_checkpointing"):
        controlnet.enable_gradient_checkpointing()
    for param in controlnet.parameters():
        param.requires_grad_(False)
    controlnet.to(dtype=dtype)
    controlnet.eval()
    notes.append(f"controlnet_params={count_params(controlnet):,}")
    return controlnet, "; ".join(notes)

def load_vae(model_id: str, dtype: torch.dtype):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype, local_files_only=Path(model_id).exists())
    for param in vae.parameters():
        param.requires_grad_(False)
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    vae.eval()
    return vae

def make_ids(resolution: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    packed_h = resolution // 16
    packed_w = resolution // 16
    rows = torch.arange(packed_h, device=device, dtype=dtype)
    cols = torch.arange(packed_w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    img_ids = torch.stack((torch.zeros_like(grid_y), grid_y, grid_x), dim=-1).reshape(-1, 3)
    txt_ids = torch.zeros(512, 3, device=device, dtype=dtype)
    return img_ids, txt_ids

def make_inputs(resolution: int, dtype: torch.dtype, device: torch.device, branch: int) -> dict[str, torch.Tensor]:
    latent_tokens = (resolution // 16) * (resolution // 16)
    generator = torch.Generator(device=device).manual_seed(20260706 + branch)
    hidden_states = torch.randn(1, latent_tokens, 64, device=device, dtype=dtype, generator=generator)
    control_cond = torch.randn(1, latent_tokens, 64, device=device, dtype=dtype, generator=generator)
    encoder_hidden_states = torch.randn(1, 512, 4096, device=device, dtype=dtype, generator=generator)
    pooled = torch.randn(1, 768, device=device, dtype=dtype, generator=generator)
    timestep = torch.full((1,), 0.5, device=device, dtype=dtype)
    guidance = torch.full((1,), 3.5, device=device, dtype=dtype)
    img_ids, txt_ids = make_ids(resolution, device, dtype)
    return {"hidden_states": hidden_states, "control_cond": control_cond, "encoder_hidden_states": encoder_hidden_states, "pooled_projections": pooled, "timestep": timestep, "guidance": guidance, "img_ids": img_ids, "txt_ids": txt_ids}

def unpack_flux_latents(tokens: torch.Tensor, resolution: int) -> torch.Tensor:
    batch = tokens.shape[0]
    packed_h = resolution // 16
    packed_w = resolution // 16
    return tokens.reshape(batch, packed_h, packed_w, 16, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(batch, 16, packed_h * 2, packed_w * 2)

def run_controlnet(controlnet, inputs: dict[str, torch.Tensor], control_mode: int, scale: float, device: torch.device):
    controlnet.to(device)
    with torch.no_grad():
        output = controlnet(
            hidden_states=inputs["hidden_states"], controlnet_cond=inputs["control_cond"],
            controlnet_mode=torch.tensor([[control_mode]], device=device, dtype=torch.long), conditioning_scale=scale,
            encoder_hidden_states=inputs["encoder_hidden_states"], pooled_projections=inputs["pooled_projections"],
            timestep=inputs["timestep"], img_ids=inputs["img_ids"], txt_ids=inputs["txt_ids"], guidance=inputs["guidance"], return_dict=True,
        )
    return output.controlnet_block_samples, output.controlnet_single_block_samples

def run_transformer(transformer, inputs: dict[str, torch.Tensor], block_samples, single_block_samples):
    return transformer(
        hidden_states=inputs["hidden_states"], encoder_hidden_states=inputs["encoder_hidden_states"],
        pooled_projections=inputs["pooled_projections"], timestep=inputs["timestep"], img_ids=inputs["img_ids"], txt_ids=inputs["txt_ids"], guidance=inputs["guidance"],
        controlnet_block_samples=block_samples, controlnet_single_block_samples=single_block_samples, return_dict=True,
    ).sample

def run_one_config(cfg: VramConfig, args: argparse.Namespace, model_id: str, device: torch.device) -> VramResult:
    dtype = torch.bfloat16
    offload_note = "on (frozen ControlNet CPU-offloaded between forwards; VAE moved to GPU for decode and kept through backward)"
    failed_step = "setup"
    transformer = controlnet = vae = optimizer = None
    try:
        cleanup_cuda()
        torch.cuda.set_device(device)
        failed_step = "load transformer"
        transformer = load_transformer(model_id, dtype, device)
        transformer_note = f"transformer_params={count_params(transformer):,}"
        failed_step = "attach LoRA"
        lora_note = attach_lora(transformer, cfg.rank)
        failed_step = "build optimizer"
        trainable = [p for p in transformer.parameters() if p.requires_grad]
        optimizer, optimizer_note = make_optimizer(trainable)
        failed_step = "load controlnet"
        controlnet, controlnet_note = load_controlnet(args.controlnet_model_id, dtype)
        failed_step = "load vae"
        if cfg.decodes:
            vae = load_vae(model_id, dtype)
            vae_note = f"vae_params={count_params(vae):,}"
        else:
            vae_note = "vae_not_loaded"
        cleanup_cuda()
        torch.cuda.reset_peak_memory_stats(device)
        failed_step = "forward/backward"
        total_loss = None
        decoded_count = 0
        cached_control = None
        last_tokens = None
        decode_tokens = []
        for branch in range(cfg.forwards):
            inputs = make_inputs(cfg.resolution, dtype, device, branch)
            if cfg.shared_diff and branch > 1 and cached_control is not None:
                block_samples, single_block_samples = cached_control
            else:
                control_inputs = inputs if not (cfg.shared_diff and branch > 0) else make_inputs(cfg.resolution, dtype, device, 1)
                block_samples, single_block_samples = run_controlnet(controlnet, control_inputs, args.control_mode, args.controlnet_scale, device)
                if cfg.shared_diff and branch >= 1:
                    cached_control = (block_samples, single_block_samples)
            controlnet.to("cpu")
            cleanup_cuda()
            tokens = run_transformer(transformer, inputs, block_samples, single_block_samples)
            last_tokens = tokens
            loss_part = tokens.float().square().mean()
            total_loss = loss_part if total_loss is None else total_loss + loss_part
            if vae is not None and len(decode_tokens) < cfg.decodes:
                decode_tokens.append(tokens)
            failed_step = "forward/backward"
        if vae is not None and last_tokens is not None:
            vae.to(device)
            while len(decode_tokens) < cfg.decodes:
                decode_tokens.append(last_tokens)
            for token_for_decode in decode_tokens[: cfg.decodes]:
                failed_step = f"vae decode {decoded_count + 1}"
                decoded = vae.decode(unpack_flux_latents(token_for_decode, cfg.resolution), return_dict=False)[0]
                total_loss = total_loss + decoded.float().abs().mean() * 0.01
                decoded_count += 1
                cleanup_cuda()
        failed_step = "loss.backward"
        assert total_loss is not None
        total_loss.backward()
        peak = torch.cuda.max_memory_allocated(device)
        feasible = "YES" if peak < args.feasible_gib * 1024**3 else "NO"
        notes = "; ".join([transformer_note, controlnet_note, vae_note, lora_note, optimizer_note])
        return VramResult(cfg.name, cfg.rank, cfg.forwards, cfg.decodes, cfg.resolution, offload_note, int(peak), feasible, "OK", "-", notes)
    except RuntimeError as exc:
        peak = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else None
        status = "OOM" if "out of memory" in str(exc).lower() else "FAILED"
        return VramResult(cfg.name, cfg.rank, cfg.forwards, cfg.decodes, cfg.resolution, offload_note, int(peak) if peak is not None else None, "NO", status, failed_step, str(exc).splitlines()[0])
    except Exception as exc:
        peak = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else None
        return VramResult(cfg.name, cfg.rank, cfg.forwards, cfg.decodes, cfg.resolution, offload_note, int(peak) if peak is not None else None, "NO", "FAILED", failed_step, f"{str(exc).splitlines()[0]} | traceback: {traceback.format_exc(limit=2).strip()}")
    finally:
        try:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
        del optimizer, vae, controlnet, transformer
        cleanup_cuda()

def result_peak_text(result: VramResult) -> str:
    if result.peak_bytes is None:
        return "-"
    return gb(result.peak_bytes) if result.status == "OK" else f"{gb(result.peak_bytes)} before {result.status}"

def write_report(path: Path, args: argparse.Namespace, model_id: str, results: list[VramResult], total_gib: float) -> None:
    primary_names = {"最小", "+差分", "+decode", "完整"}
    full = next((row for row in results if row.name == "完整"), None)
    feasible_rows = [row for row in results if row.status == "OK" and row.feasible == "YES"]
    fallback_rows = [row for row in feasible_rows if row.name not in primary_names]
    if full is not None and full.status == "OK" and full.feasible == "YES":
        conclusion = f"结论：完整配置在 48G A6000 上可跑。本次单步峰值 {gb(full.peak_bytes or 0)}，低于 {args.feasible_gib:.1f} GiB 判定线。"
    elif fallback_rows:
        best = fallback_rows[0]
        conclusion = f"结论：完整配置未通过 48G 判定；已重测降配，首个可行配置为 `{best.name}`，峰值 {gb(best.peak_bytes or 0)}。"
    elif feasible_rows:
        best = feasible_rows[-1]
        conclusion = f"结论：完整配置未通过 48G 判定；当前主表中可行的最高配置为 `{best.name}`，峰值 {gb(best.peak_bytes or 0)}。"
    else:
        conclusion = "结论：本轮没有找到 48G 可行配置，需要继续降分辨率或减少前向/decode。"
    lines = [
        "# Phase 0 VRAM Training-step Report", "", f"Generated: {now_text()}", "",
        "Mode: REAL CUDA TRAINING-STEP MEASUREMENT, not inference.",
        f"GPU: {torch.cuda.get_device_name(args.device)}; total={total_gib:.2f} GiB; feasible threshold={args.feasible_gib:.1f} GiB.",
        f"Dataset root: `{args.root}`", f"FLUX model requested: `{args.model_id}`; loaded: `{model_id}`", f"ControlNet requested: `{args.controlnet_model_id}`",
        "Batch=1, resolution per row, gradient_accumulation=16 simulated as one micro-step because accumulation does not increase peak VRAM.",
        "The measured step instantiates PagedAdamW8bit, runs transformer forward(s), optional VAE decode(s), and `loss.backward()`; it intentionally does not call `optimizer.step()`.", "",
        "| 配置 | rank | 前向次数 | decode | 分辨率 | offload | 峰值显存 | 48G 是否可行 | 状态 | 失败步骤/备注 |",
        "|---|---:|---:|---:|---:|---|---:|---|---|---|",
    ]
    for row in results:
        lines.append(f"| {row.name} | {row.rank} | {row.forwards} | {row.decodes} | {row.resolution} | {row.offload} | {result_peak_text(row)} | {row.feasible} | {row.status} | {row.failed_step}: {row.notes} |")
    lines.extend(["", conclusion, "", "降配建议：优先降 LoRA rank 16->8；让差分分支 v_j/v_k 共享 ControlNet/text 部分计算；降低 identity VAE decode 频率；ControlNet 也改 LoRA 或冻结并更激进 CPU offload；仍不够再降到 384。", "", "Notes:", "- If the ControlNet row says `待换真实权重`, this run used the InstantX Union FluxControlNet architecture as a parameter/activation placeholder because local real ControlNet weights were unavailable.", "- `torch.cuda.max_memory_allocated()` was reset after model/optimizer setup, so the peak includes resident model weights plus the measured train-step allocations, but excludes one-time checkpoint loading spikes."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 real CUDA training-step VRAM pressure test.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--local-model", type=Path, default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--controlnet-model-id", default=DEFAULT_CONTROLNET_ID)
    parser.add_argument("--control-mode", type=int, default=DEFAULT_CONTROL_MODE)
    parser.add_argument("--controlnet-scale", type=float, default=0.75)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--feasible-gib", type=float, default=45.0)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real VRAM pressure testing.")
    torch.cuda.set_device(args.device)
    torch.manual_seed(20260706)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    root = args.root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else root / "phase0"
    ensure_dir(out_dir)
    model_id = resolve_model_id(args.model_id, args.local_model.expanduser().resolve())
    total_gib = torch.cuda.get_device_properties(args.device).total_memory / 1024**3
    primary = [VramConfig("最小", 8, 1, 0, 512), VramConfig("+差分", 8, 3, 0, 512), VramConfig("+decode", 8, 3, 2, 512), VramConfig("完整", 16, 3, 2, 512)]
    fallback = [VramConfig("降配-rank8完整", 8, 3, 2, 512), VramConfig("降配-decode1", 8, 3, 1, 512), VramConfig("降配-共享差分", 8, 2, 1, 512, shared_diff=True), VramConfig("降配-384", 8, 2, 1, 384, shared_diff=True)]
    results: list[VramResult] = []
    report_path = out_dir / "vram_report.md"
    for cfg in primary:
        print(f"[vram] running {cfg}", flush=True)
        result = run_one_config(cfg, args, model_id, torch.device(f"cuda:{args.device}"))
        print(f"[vram] {cfg.name}: {result.status} peak={result_peak_text(result)}", flush=True)
        results.append(result)
        write_report(report_path, args, model_id, results, total_gib)
    full = next(row for row in results if row.name == "完整")
    if not (full.status == "OK" and full.feasible == "YES"):
        for cfg in fallback:
            if any(row.status == "OK" and row.feasible == "YES" and row.name not in {"最小", "+差分", "+decode", "完整"} for row in results):
                break
            print(f"[vram] running fallback {cfg}", flush=True)
            result = run_one_config(cfg, args, model_id, torch.device(f"cuda:{args.device}"))
            print(f"[vram] {cfg.name}: {result.status} peak={result_peak_text(result)}", flush=True)
            results.append(result)
            write_report(report_path, args, model_id, results, total_gib)
    write_report(report_path, args, model_id, results, total_gib)
    print(f"Wrote VRAM report: {report_path}")

if __name__ == "__main__":
    main()
