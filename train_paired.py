from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from conditions import (
    FluxConditionAdapter, assert_real_controlnet, atomic_torch_save, choose_dtype, get_resolution, load_yaml, make_image_ids,
    make_text_ids, save_yaml, seed_everything, sha256_file, short_hash_path,
)
from dataset import PairedWarmupDataset, ResumeDistributedSampler
from pulid_flux import PuLIDFluxAdapter


class WarmupFlowModel(torch.nn.Module):
    def __init__(self, transformer, controlnet, adapter: FluxConditionAdapter, pulid: PuLIDFluxAdapter, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.transformer = transformer
        self.controlnet = controlnet
        self.adapter = adapter
        self.pulid = pulid
        self.cfg = cfg
        self.width, self.height = get_resolution(cfg['data']['resolution'])
        self.control_mode = int(cfg['model']['control_mode'])
        self.controlnet_scale = float(cfg['model']['controlnet_scale'])
        self.guidance_scale = float(cfg.get('eval', {}).get('guidance_scale', 3.5))
        self.run_origin_step: int | None = None

    def set_run_origin_step(self, step: int) -> None:
        if self.run_origin_step is None:
            self.run_origin_step = int(step)

    def _prepare_flow(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        dtype = next(self.transformer.parameters()).dtype
        device = next(self.transformer.parameters()).device
        z0 = batch['target_latents'].to(device=device, dtype=dtype)
        z1 = torch.randn_like(z0)
        tau_override = batch.get('tau_override')
        if tau_override is None:
            tau = torch.rand(z0.shape[0], device=device, dtype=dtype)
        else:
            tau = tau_override.to(device=device, dtype=dtype).reshape(-1)
            if tau.shape[0] == 1 and z0.shape[0] > 1:
                tau = tau.expand(z0.shape[0])
            if tau.shape[0] != z0.shape[0]:
                raise RuntimeError(f'tau_override batch={tau.shape[0]}, expected {z0.shape[0]}')
        z_tau = (1.0 - tau.view(-1, 1, 1)) * z0 + tau.view(-1, 1, 1) * z1
        target_v = z1 - z0
        prompt = batch['prompt_embeds'].to(device=device, dtype=dtype)
        pooled = batch['pooled_prompt_embeds'].to(device=device, dtype=dtype)
        if prompt.ndim == 2:
            prompt = prompt.unsqueeze(0).expand(z0.shape[0], -1, -1)
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(0).expand(z0.shape[0], -1)
        img_ids = make_image_ids(self.width, self.height, device, dtype)
        return {
            'z0': z0,
            'z1': z1,
            'z_tau': z_tau,
            'target_v': target_v,
            'tau': tau,
            'prompt': prompt,
            'pooled': pooled,
            'img_ids': img_ids,
            'device': device,
            'dtype': dtype,
        }

    def _condition_tokens(
        self,
        prompt: torch.Tensor,
        appearance: torch.Tensor,
        garment: torch.Tensor,
        head_pose: torch.Tensor,
    ) -> torch.Tensor:
        adapter_tokens = self.adapter(appearance, garment, head_pose)
        return torch.cat([prompt, adapter_tokens], dim=1)

    def _controlnet_forward(
        self,
        z_tau: torch.Tensor,
        tau: torch.Tensor,
        prompt: torch.Tensor,
        pooled: torch.Tensor,
        pose_latents: torch.Tensor,
        img_ids: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        device, dtype = z_tau.device, z_tau.dtype
        with torch.no_grad():
            cn = self.controlnet(
                hidden_states=z_tau,
                controlnet_cond=pose_latents.to(device=device, dtype=dtype),
                controlnet_mode=torch.full((z_tau.shape[0], 1), self.control_mode, device=device, dtype=torch.long),
                conditioning_scale=self.controlnet_scale,
                encoder_hidden_states=prompt,
                pooled_projections=pooled,
                timestep=tau,
                img_ids=img_ids,
                txt_ids=make_text_ids(prompt.shape[1], device, dtype),
                guidance=torch.full((z_tau.shape[0],), self.guidance_scale, device=device, dtype=dtype),
                return_dict=True,
            )
        return cn.controlnet_block_samples, cn.controlnet_single_block_samples

    def _transformer_forward(
        self,
        z_tau: torch.Tensor,
        tau: torch.Tensor,
        cond_tokens: torch.Tensor,
        pulid_embed: torch.Tensor,
        cn_samples: tuple[list[torch.Tensor], list[torch.Tensor]],
        *,
        pooled: torch.Tensor,
        img_ids: torch.Tensor,
    ) -> torch.Tensor:
        device, dtype = z_tau.device, z_tau.dtype
        self.pulid.set_context(
            pulid_embed.to(device=device, dtype=dtype),
            float(self.cfg.get('model', {}).get('pulid', {}).get('id_weight', 1.0)),
        )
        pulid_context = self.pulid.context_kwargs()
        try:
            return self.transformer(
                hidden_states=z_tau,
                encoder_hidden_states=cond_tokens,
                pooled_projections=pooled,
                timestep=tau,
                img_ids=img_ids,
                txt_ids=make_text_ids(cond_tokens.shape[1], device, dtype),
                guidance=torch.full((z_tau.shape[0],), self.guidance_scale, device=device, dtype=dtype),
                joint_attention_kwargs=pulid_context,
                controlnet_block_samples=cn_samples[0],
                controlnet_single_block_samples=cn_samples[1],
                return_dict=True,
            ).sample
        finally:
            # The explicit context tensor is captured by non-reentrant checkpointing,
            # so clearing the mutable fallback cannot mix i/j/k during backward recompute.
            self.pulid.clear_context()

    def _base_metrics(self, batch: dict[str, torch.Tensor], flow: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = flow['device']
        head_null = batch.get('head_pose_is_null')
        head_null_ratio = (
            head_null.float().mean().detach()
            if head_null is not None
            else torch.zeros((), device=device, dtype=torch.float32)
        )
        metrics = {
            'head_pose_null_ratio': head_null_ratio,
            'tau_mean': flow['tau'].detach().float().mean(),
            'z1_mean': flow['z1'].detach().float().mean(),
            'controlnet_forward_count': torch.ones((), device=device, dtype=torch.float32),
        }
        metrics.update({
            name: torch.tensor(value, device=device, dtype=torch.float32)
            for name, value in self.adapter.gate_values().items()
        })
        return metrics

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        train_step: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        flow = self._prepare_flow(batch)
        dtype, device = flow['dtype'], flow['device']
        cond_tokens = self._condition_tokens(
            flow['prompt'],
            batch['appearance'].to(device=device, dtype=dtype),
            batch['garment'].to(device=device, dtype=dtype),
            batch['head_pose'].to(device=device, dtype=dtype),
        )
        cn_samples = self._controlnet_forward(
            flow['z_tau'],
            flow['tau'],
            flow['prompt'],
            flow['pooled'],
            batch['pose_latents'],
            flow['img_ids'],
        )
        pred = self._transformer_forward(
            flow['z_tau'],
            flow['tau'],
            cond_tokens,
            batch['pulid_id_embed'],
            cn_samples,
            pooled=flow['pooled'],
            img_ids=flow['img_ids'],
        )
        loss = F.mse_loss(pred.float(), flow['target_v'].float())
        metrics = self._base_metrics(batch, flow)
        metrics.update({
            'loss_total': loss.detach(),
            'loss_pair': loss.detach(),
            'transformer_forward_count': torch.ones((), device=device, dtype=torch.float32),
        })
        return loss, metrics


class DifferentialFlowModel(WarmupFlowModel):
    def __init__(self, transformer, controlnet, adapter: FluxConditionAdapter, pulid: PuLIDFluxAdapter, cfg: dict[str, Any]) -> None:
        super().__init__(transformer, controlnet, adapter, pulid, cfg)
        self.diff_cfg = cfg['training']['differential']
        resolved = self.diff_cfg.get('hinge_g_resolved', self.diff_cfg.get('hinge_g'))
        self.hinge_g = None if resolved is None else float(resolved)

    def set_hinge_g(self, value: float) -> None:
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError(f'calibrated hinge g must be positive and finite, got {value}')
        self.hinge_g = float(value)
        self.diff_cfg['hinge_g_resolved'] = self.hinge_g

    def differential_state_dict(self) -> dict[str, Any]:
        return {
            'hinge_g': self.hinge_g,
            'run_origin_step': self.run_origin_step,
        }

    def load_differential_state(self, state: dict[str, Any]) -> None:
        if state.get('hinge_g') is not None:
            self.set_hinge_g(float(state['hinge_g']))
        if state.get('run_origin_step') is not None:
            self.run_origin_step = int(state['run_origin_step'])

    @staticmethod
    def _masked_l1_per_sample(diff: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = diff.float().abs()
        weights = mask.to(device=diff.device, dtype=torch.float32).clamp(0.0, 1.0)
        numerator = (values * weights.unsqueeze(-1)).sum(dim=(1, 2))
        denominator = weights.sum(dim=1).clamp_min(1e-6) * values.shape[-1]
        return numerator / denominator

    @staticmethod
    def _active_mean(values: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        weights = active.to(dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        train_step: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        train_step = int(train_step or 0)
        flow = self._prepare_flow(batch)
        dtype, device = flow['dtype'], flow['device']
        garment = batch['garment'].to(device=device, dtype=dtype)
        head_pose = batch['head_pose'].to(device=device, dtype=dtype)
        paired_tokens = self._condition_tokens(
            flow['prompt'],
            batch['appearance'].to(device=device, dtype=dtype),
            garment,
            head_pose,
        )
        # ControlNet is identity-independent in A2: it sees pose, prompt, z_tau, and tau only.
        # Reusing these samples is invalid if a future method injects identity into ControlNet.
        cn_samples = self._controlnet_forward(
            flow['z_tau'],
            flow['tau'],
            flow['prompt'],
            flow['pooled'],
            batch['pose_latents'],
            flow['img_ids'],
        )
        pred_i = self._transformer_forward(
            flow['z_tau'],
            flow['tau'],
            paired_tokens,
            batch['pulid_id_embed'],
            cn_samples,
            pooled=flow['pooled'],
            img_ids=flow['img_ids'],
        )
        loss_pair = F.mse_loss(pred_i.float(), flow['target_v'].float())
        zero = torch.zeros((), device=device, dtype=torch.float32)
        metrics = self._base_metrics(batch, flow)
        tau_min = float(self.diff_cfg.get('tau_min', 0.2))
        tau_max = float(self.diff_cfg.get('tau_max', 0.8))
        diff_every = max(1, int(self.diff_cfg.get('diff_every', 1)))
        scheduled = train_step % diff_every == 0
        active = (flow['tau'].float() >= tau_min) & (flow['tau'].float() <= tau_max)
        if not scheduled:
            active = torch.zeros_like(active)

        loss_teach = zero
        loss_inv = zero
        loss_hinge = zero
        face_diff_mean = zero
        hinge_active_rate = zero
        calibration_ratios = torch.empty((0,), device=device, dtype=torch.float32)
        transformer_count = 1.0
        if bool(active.any()):
            tokens_j = self._condition_tokens(
                flow['prompt'],
                batch['cf_j_appearance'].to(device=device, dtype=dtype),
                garment,
                head_pose,
            )
            tokens_k = self._condition_tokens(
                flow['prompt'],
                batch['cf_k_appearance'].to(device=device, dtype=dtype),
                garment,
                head_pose,
            )
            pred_j = self._transformer_forward(
                flow['z_tau'],
                flow['tau'],
                tokens_j,
                batch['cf_j_pulid_id_embed'],
                cn_samples,
                pooled=flow['pooled'],
                img_ids=flow['img_ids'],
            )
            pred_k = self._transformer_forward(
                flow['z_tau'],
                flow['tau'],
                tokens_k,
                batch['cf_k_pulid_id_embed'],
                cn_samples,
                pooled=flow['pooled'],
                img_ids=flow['img_ids'],
            )
            transformer_count = 3.0
            tau_view = flow['tau'].float().view(-1, 1, 1)
            z_hat_j = flow['z_tau'].float() - tau_view * pred_j.float()
            z_hat_k = flow['z_tau'].float() - tau_view * pred_k.float()
            teach_j = self._masked_l1_per_sample(
                z_hat_j - flow['z0'].float(), batch['cloth_safe_z']
            )
            teach_k = self._masked_l1_per_sample(
                z_hat_k - flow['z0'].float(), batch['cloth_safe_z']
            )
            teach_per = 0.5 * (teach_j + teach_k)
            inv_per = self._masked_l1_per_sample(z_hat_j - z_hat_k, batch['body_bg_z'])
            face_per = self._masked_l1_per_sample(z_hat_j - z_hat_k, batch['face_z'])
            delta_arc = batch['delta_arc_jk'].to(device=device, dtype=torch.float32).reshape(-1)
            loss_teach = self._active_mean(teach_per, active)
            loss_inv = self._active_mean(inv_per, active)
            face_diff_mean = self._active_mean(face_per, active)
            valid_ratio = active & (delta_arc > 1e-6)
            calibration_ratios = (face_per[valid_ratio] / delta_arc[valid_ratio]).detach()
            if self.hinge_g is not None:
                margin = float(self.hinge_g) * delta_arc
                hinge_per = F.relu(margin - face_per)
                loss_hinge = self._active_mean(hinge_per, active)
                hinge_active_rate = self._active_mean((hinge_per > 0).float(), active)

        calibrating = self.hinge_g is None
        lambda_teach = float(self.diff_cfg.get('lambda_teach', 0.5))
        lambda_inv = float(self.diff_cfg.get('lambda_inv', 0.2))
        lambda_hinge = 0.0 if calibrating else float(self.diff_cfg.get('lambda_hinge', 0.05))
        total = (
            loss_pair
            + lambda_teach * loss_teach
            + lambda_inv * loss_inv
            + lambda_hinge * loss_hinge
        )
        metrics.update({
            'loss_total': total.detach(),
            'loss_pair': loss_pair.detach(),
            'loss_teach': loss_teach.detach(),
            'loss_inv': loss_inv.detach(),
            'loss_hinge': loss_hinge.detach(),
            'hinge_active_rate': hinge_active_rate.detach(),
            'face_diff_norm': face_diff_mean.detach(),
            'diff_active_ratio': active.float().mean().detach(),
            'hinge_calibrating': torch.tensor(float(calibrating), device=device),
            'hinge_g': torch.tensor(float(self.hinge_g or 0.0), device=device),
            'transformer_forward_count': torch.tensor(transformer_count, device=device),
            'calibration_ratios': calibration_ratios,
        })
        return total, metrics


def setup_dist() -> tuple[int, int, int]:
    if 'RANK' not in os.environ:
        return 0, 1, int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl')
    return int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def attach_lora(transformer, rank: int) -> str:
    from peft import LoraConfig
    for param in transformer.parameters():
        param.requires_grad_(False)
    target_modules = ['to_q', 'to_k', 'to_v', 'to_out.0', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out']
    transformer.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights='gaussian', target_modules=target_modules))
    for name, param in transformer.named_parameters():
        param.requires_grad_('lora' in name.lower())
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    return f'LoRA rank={rank}, trainable={trainable:,}'


def load_components(cfg: dict[str, Any], device: torch.device, dtype: torch.dtype):
    from diffusers import AutoencoderKL, FluxControlNetModel, FluxTransformer2DModel
    transformer = FluxTransformer2DModel.from_pretrained(cfg['model']['base'], subfolder='transformer', torch_dtype=dtype, local_files_only=True)
    if cfg['model'].get('gradient_checkpointing', {}).get('enabled', True):
        transformer.enable_gradient_checkpointing()
    transformer.to(device=device, dtype=dtype).train()
    lora_note = attach_lora(transformer, int(cfg['model']['lora_rank']))
    control_info = assert_real_controlnet(cfg['model']['controlnet'])
    controlnet = FluxControlNetModel.from_pretrained(cfg['model']['controlnet'], torch_dtype=dtype, local_files_only=True)
    controlnet.requires_grad_(False).to(device=device, dtype=dtype).eval()
    vae = None
    if cfg['model'].get('load_vae_in_train', False):
        vae = AutoencoderKL.from_pretrained(cfg['model']['base'], subfolder='vae', torch_dtype=dtype, local_files_only=True)
        vae.requires_grad_(False).to(device=device, dtype=dtype).eval()
    adapter = FluxConditionAdapter(cfg['model']['identity_adapter']).to_compute(device=device, dtype=dtype)
    pulid = PuLIDFluxAdapter(cfg['model']['pulid'], device=device, dtype=dtype)
    pulid.attach_to_transformer(transformer)
    pulid_delta_l2 = pulid.self_check(device=device, dtype=dtype)
    pulid_transformer_l2 = pulid.transformer_self_check(transformer, device=device, dtype=dtype)
    pulid_context_switch_l2 = float(getattr(pulid, '_last_transformer_context_switch_l2', 0.0))
    threshold = float(cfg['model']['pulid'].get('self_check_min_l2', 1.0))
    if pulid_delta_l2 <= threshold:
        raise RuntimeError(f'PuLID CA startup self-check failed: delta_l2={pulid_delta_l2:.6f} <= {threshold}')
    if pulid_transformer_l2 <= threshold:
        raise RuntimeError(
            'PuLID transformer hook self-check failed: '
            f'delta_l2={pulid_transformer_l2:.6f} <= {threshold}, '
            f'hook_calls={getattr(pulid, "_last_transformer_hook_calls", "na")}, '
            f'delta_norm={getattr(pulid, "_last_transformer_delta_norm", "na")}, '
            f'out0_norm={getattr(pulid, "_last_transformer_out0_norm", "na")}, '
            f'out1_norm={getattr(pulid, "_last_transformer_out1_norm", "na")}'
        )
    if pulid_context_switch_l2 <= threshold:
        raise RuntimeError(
            'PuLID i/j context-switch self-check failed: '
            f'delta_l2={pulid_context_switch_l2:.6f} <= {threshold}'
        )
    return transformer, controlnet, vae, adapter, pulid, {
        'controlnet': control_info,
        'lora': lora_note,
        'adapter': adapter.launch_note(),
        'pulid': pulid.launch_note(),
        'pulid_ca_self_check_l2': pulid_delta_l2,
        'pulid_transformer_self_check_l2': pulid_transformer_l2,
        'pulid_context_switch_self_check_l2': pulid_context_switch_l2,
    }


def build_optimizer(module: torch.nn.Module, cfg: dict[str, Any]):
    lr = float(cfg['_runtime']['effective_lr'])
    adapter_cfg = cfg['model'].get('identity_adapter', {})
    adapter_mult = float(adapter_cfg.get('condition_adapter_lr_mult', 1.0))
    gate_mult = float(adapter_cfg.get('gate_lr_mult', 10.0))
    lora_params = []
    adapter_params = []
    gate_params = []
    gate_suffixes = ('appearance_gate', 'garment_gate', 'pose_gate')
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        is_adapter = '.adapter.' in name or name.startswith('adapter.') or name.startswith('module.adapter.')
        if is_adapter and name.endswith(gate_suffixes):
            gate_params.append(param)
        elif is_adapter:
            adapter_params.append(param)
        else:
            lora_params.append(param)
    groups = []
    if lora_params:
        groups.append({'params': lora_params, 'lr': lr, 'group_name': 'transformer_lora'})
    if adapter_params:
        groups.append({'params': adapter_params, 'lr': lr * adapter_mult, 'group_name': 'condition_adapter'})
    if gate_params:
        groups.append({
            'params': gate_params,
            'lr': lr * gate_mult,
            'weight_decay': 0.0,
            'group_name': 'condition_gates_fp32',
        })
    if cfg['training'].get('optimizer') == 'paged_adamw8bit':
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(groups)
    return torch.optim.AdamW(groups)


def sync_stop_requested(marker: Path, rank: int, device: torch.device) -> bool:
    requested = 1 if rank == 0 and marker.exists() else 0
    flag = torch.tensor([requested], device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.broadcast(flag, src=0)
    return bool(flag.item())


def recompute_batch_runtime(cfg: dict[str, Any], world_size: int, micro: int) -> None:
    accum = math.ceil(int(cfg['training']['baseline_global_batch']) / (world_size * micro))
    global_batch = world_size * micro * accum
    cfg['_runtime']['micro_batch'] = micro
    cfg['_runtime']['grad_accum'] = accum
    cfg['_runtime']['global_batch'] = global_batch
    cfg['_runtime']['effective_lr'] = float(cfg['training']['baseline_lr']) * global_batch / int(cfg['training']['baseline_global_batch'])


def make_loader(dataset, sampler, cfg: dict[str, Any]) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(cfg['_runtime']['micro_batch']),
        sampler=sampler,
        num_workers=int(cfg['training']['num_workers_per_rank']),
        pin_memory=bool(cfg['training']['pin_memory']),
        persistent_workers=bool(cfg['training']['persistent_workers']),
        prefetch_factor=int(cfg['training']['prefetch_factor']),
        drop_last=True,
    )


def sync_probe_ok(ok: bool, device: torch.device) -> bool:
    if not dist.is_initialized():
        return ok
    flag = torch.tensor(1 if ok else 0, device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def maybe_probe_micro_batch(
    model: WarmupFlowModel,
    dataset: PairedWarmupDataset,
    cfg: dict[str, Any],
    world_size: int,
    rank: int,
    device: torch.device,
) -> None:
    if not cfg['training'].get('auto_micro_batch_probe', False):
        return
    preferred = int(cfg['training'].get('preferred_micro_batch', cfg['_runtime']['micro_batch']))
    current = int(cfg['_runtime']['micro_batch'])
    if preferred <= current:
        return
    probe_loader = DataLoader(dataset, batch_size=preferred, shuffle=False, num_workers=0, drop_last=True)
    try:
        batch = next(iter(probe_loader))
    except StopIteration:
        if rank == 0:
            print('[rank0] micro-batch probe skipped: not enough cached samples on this rank', flush=True)
        return
    old = dict(cfg['_runtime'])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    ok = True
    err = ''
    peak = 0.0
    try:
        loss, _ = model(batch)
        loss.backward()
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        ok = peak <= 44.0
        model.zero_grad(set_to_none=True)
    except torch.cuda.OutOfMemoryError as exc:
        ok = False
        err = str(exc).split('\n')[0]
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        if 'out of memory' not in str(exc).lower():
            raise
        ok = False
        err = str(exc).split('\n')[0]
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    ok = sync_probe_ok(ok, device)
    if ok:
        recompute_batch_runtime(cfg, world_size, preferred)
        if rank == 0:
            print(f'[rank0] micro-batch probe accepted: micro={preferred}, peak_gib={peak:.2f}, runtime={cfg["_runtime"]}', flush=True)
    else:
        cfg['_runtime'] = old
        if rank == 0:
            reason = f'peak_gib={peak:.2f} > 44.0' if peak else err
            print(f'[rank0] micro-batch probe fallback: keep micro={current}; reason={reason}', flush=True)


def unwrap_model(module: torch.nn.Module) -> WarmupFlowModel:
    return module.module if hasattr(module, 'module') else module


def accumulate_scalar_metrics(
    totals: dict[str, float],
    counts: dict[str, int],
    metrics: dict[str, torch.Tensor],
) -> None:
    for name, value in metrics.items():
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            continue
        scalar = float(value.detach().float().cpu())
        totals[name] = totals.get(name, 0.0) + scalar
        counts[name] = counts.get(name, 0) + 1


def averaged_metrics(totals: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    return {name: totals[name] / max(1, counts[name]) for name in totals}


def calibrate_hinge_g(local_ratios: list[float], device: torch.device) -> tuple[float, int]:
    gathered: list[list[float] | None]
    if dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_ratios)
        values = [value for rows in gathered if rows for value in rows]
    else:
        values = list(local_ratios)
    if not values:
        raise RuntimeError('hinge g calibration collected no valid face_diff/d_arc ratios')
    result = torch.tensor(
        [float(np.quantile(np.asarray(values, dtype=np.float64), 0.25)), float(len(values))],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.broadcast(result, src=0)
    return float(result[0].item()), int(result[1].item())


def save_checkpoint(path: Path, model: WarmupFlowModel, optimizer, step: int, sampler: ResumeDistributedSampler, cfg: dict[str, Any]) -> None:
    from peft import get_peft_model_state_dict
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        'step': step,
        'adapter': model.adapter.state_dict(),
        'transformer_lora': get_peft_model_state_dict(model.transformer),
        'optimizer': optimizer.state_dict(),
        'sampler': sampler.state_dict(),
        'config': cfg,
        'continuation_origin_step': model.run_origin_step,
    }
    if hasattr(model, 'differential_state_dict'):
        payload['differential_state'] = model.differential_state_dict()
    atomic_torch_save(payload, path / 'trainable.pt')
    (path / 'READY').write_text(str(step), encoding='utf-8')


def load_checkpoint(path: Path, model: WarmupFlowModel, optimizer=None, sampler=None) -> int:
    from peft import set_peft_model_state_dict
    payload = torch.load(path / 'trainable.pt', map_location='cpu')
    adapter_state = payload['adapter']
    current = model.adapter.state_dict()
    compatible = {k: v for k, v in adapter_state.items() if k in current and tuple(current[k].shape) == tuple(v.shape)}
    dropped = sorted(set(adapter_state) - set(compatible))
    missing, unexpected = model.adapter.load_state_dict(compatible, strict=False)
    if dropped or missing or unexpected:
        print(f'[warn] adapter checkpoint loaded partially from {path}: dropped={len(dropped)}, missing={len(missing)}, unexpected={len(unexpected)}', flush=True)
    set_peft_model_state_dict(model.transformer, payload['transformer_lora'])
    if optimizer is not None and 'optimizer' in payload:
        optimizer.load_state_dict(payload['optimizer'])
    if sampler is not None and 'sampler' in payload:
        sampler.load_state_dict(payload['sampler'])
    if 'differential_state' in payload and hasattr(model, 'load_differential_state'):
        model.load_differential_state(payload['differential_state'])
    if payload.get('continuation_origin_step') is not None:
        model.run_origin_step = int(payload['continuation_origin_step'])
    return int(payload.get('step', 0))


def configure_runtime(cfg: dict[str, Any], world_size: int, all_gpus_train: bool) -> None:
    micro = int(cfg['training']['micro_batch'])
    if cfg['training'].get('grad_accum') == 'auto':
        accum = math.ceil(int(cfg['training']['baseline_global_batch']) / (world_size * micro))
    else:
        accum = int(cfg['training']['grad_accum'])
    global_batch = world_size * micro * accum
    effective_lr = float(cfg['training']['baseline_lr']) * global_batch / int(cfg['training']['baseline_global_batch'])
    cfg['_runtime'] = {
        'world_size': world_size,
        'micro_batch': micro,
        'grad_accum': accum,
        'global_batch': global_batch,
        'effective_lr': effective_lr,
        'all_gpus_train': bool(all_gpus_train),
        'arch_note': 'Default is 3-card DDP + GPU3 watcher. Same A6000 cards and Phase0 30.4GiB single-card peak make DDP simpler/faster than FSDP/DeepSpeed; frozen modules have requires_grad=False so DDP does not sync them.',
    }


def train(args: argparse.Namespace) -> None:
    rank, world_size, local_rank = setup_dist()
    cfg = load_yaml(args.config)
    all_gpus_train = bool(args.all_gpus_train)
    if not all_gpus_train and world_size != 3 and not args.dev_single_gpu:
        raise RuntimeError(f'default Phase1 launch expects 3 training ranks, leaving GPU3 for watcher; got world_size={world_size}. Use --all-gpus-train for 4-rank training, or --dev-single-gpu for smoke only.')
    if all_gpus_train and world_size != 4:
        raise RuntimeError(f'--all-gpus-train expects world_size=4, got {world_size}')
    if args.dev_single_gpu and world_size != 1:
        raise RuntimeError('--dev-single-gpu is only for one-process smoke runs')
    configure_runtime(cfg, world_size, all_gpus_train)
    if args.override_output_id:
        cfg['experiment']['id'] = str(args.override_output_id)
        cfg['experiment']['wandb_run_id'] = str(args.override_output_id)
    if args.resume:
        cfg['training']['resume'] = str(args.resume)
    if args.override_total_steps is not None:
        cfg['training']['total_steps'] = int(args.override_total_steps)
        if 'additional_steps' in cfg['training']:
            cfg['training']['additional_steps'] = int(args.override_total_steps)
    if args.smoke_steps > 0:
        cfg['training']['total_steps'] = int(args.smoke_steps)
        cfg['_runtime']['grad_accum'] = 1
        cfg['_runtime']['global_batch'] = world_size * int(cfg['_runtime']['micro_batch'])
        cfg['_runtime']['effective_lr'] = float(cfg['training']['baseline_lr']) * cfg['_runtime']['global_batch'] / int(cfg['training']['baseline_global_batch'])
        differential_cfg = cfg.get('training', {}).get('differential', {})
        if differential_cfg.get('enabled', False) and differential_cfg.get('hinge_g_resolved') is None:
            differential_cfg['hinge_g_resolved'] = float(differential_cfg.get('smoke_hinge_g', 1.0))
    seed_everything(int(cfg['experiment']['seed']) + rank)
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = choose_dtype(cfg['model']['precision'])

    dataset = PairedWarmupDataset(cfg, 'train', require_coverage=bool(cfg['cache'].get('require_coverage', True)) and not args.allow_partial_cache)
    if args.allow_partial_cache:
        dataset.ids = [sid for sid in dataset.ids if dataset.sample_path(sid).exists()]
        if not dataset.ids:
            raise RuntimeError('allow_partial_cache requested but no cached train samples were found')
        if rank == 0:
            print(f'[rank0] allow_partial_cache: using {len(dataset.ids)} cached train samples for smoke only', flush=True)
    sampler = ResumeDistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=int(cfg['experiment']['seed']))
    transformer, controlnet, vae, adapter, pulid, load_notes = load_components(cfg, device, dtype)
    differential_enabled = bool(cfg.get('training', {}).get('differential', {}).get('enabled', False))
    model_cls = DifferentialFlowModel if differential_enabled else WarmupFlowModel
    model = model_cls(transformer, controlnet, adapter, pulid, cfg)
    if not args.smoke_steps:
        maybe_probe_micro_batch(model, dataset, cfg, world_size, rank, device)
    loader = make_loader(dataset, sampler, cfg)
    if cfg['model'].get('compile', False):
        try:
            model.transformer = torch.compile(model.transformer)
        except Exception as exc:  # noqa: BLE001
            if rank == 0:
                print(f'[warn] torch.compile disabled after failure: {exc}', flush=True)
    # DDP is intentionally used instead of FSDP/DeepSpeed: each A6000 fits the full Phase1 model,
    # cards are homogeneous, and frozen ControlNet/VAE/encoders have requires_grad=False so they are not placed in gradient buckets.
    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False) if world_size > 1 else model
    optimizer = build_optimizer(ddp, cfg)

    output = Path(cfg['experiment']['output_root']) / cfg['experiment']['id']
    ckpt_dir = output / 'checkpoints'
    log_dir = output / 'logs'
    stop_marker = output / 'STOP_TRAINING'
    start_step = 0
    if cfg['training'].get('resume'):
        start_step = load_checkpoint(Path(cfg['training']['resume']), ddp.module if hasattr(ddp, 'module') else ddp, optimizer, sampler)
    core_model = unwrap_model(ddp)
    core_model.set_run_origin_step(start_step)
    run_origin_step = int(core_model.run_origin_step if core_model.run_origin_step is not None else start_step)
    if args.smoke_steps > 0:
        target_step = start_step + int(args.smoke_steps)
    elif cfg['training'].get('additional_steps') is not None:
        target_step = run_origin_step + int(cfg['training']['additional_steps'])
    else:
        target_step = int(cfg['training']['total_steps'])
    if target_step < start_step:
        raise RuntimeError(
            f'target step {target_step} is behind resume step {start_step}; '
            'use training.additional_steps for continuation runs'
        )
    cfg['_runtime'].update({
        'resume_step': start_step,
        'run_origin_step': run_origin_step,
        'target_step': target_step,
        'continuation_steps': target_step - run_origin_step,
        'differential_enabled': differential_enabled,
    })
    resume_path = Path(cfg['training']['resume']) if cfg['training'].get('resume') else None
    resume_hash = (
        sha256_file(resume_path / 'trainable.pt')[:16]
        if resume_path is not None and (resume_path / 'trainable.pt').exists()
        else None
    )
    train_ids_hash = hashlib.sha256(
        ('\n'.join(dataset.ids) + '\n').encode('utf-8')
    ).hexdigest()[:16]
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        save_yaml(output / 'resolved_config.yaml', cfg)
        (output / 'launch.json').write_text(json.dumps({
            'load_notes': load_notes,
            'base_hash': short_hash_path(cfg['model']['base']),
            'rank0_device': torch.cuda.get_device_name(local_rank),
            'resume_checkpoint': str(resume_path) if resume_path else None,
            'resume_trainable_hash': resume_hash,
            'resume_sampler_state': sampler.state_dict(),
            'train_ids_hash': train_ids_hash,
            'train_sample_count': len(dataset.ids),
            'excluded_train_ids': dataset.excluded_ids,
            'run_origin_step': run_origin_step,
            'target_step': target_step,
            'fairness_note': 'A2 and B2-cont must use the same resume checkpoint/hash, seed, sampler state, global batch, LR, and continuation step count.',
        }, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'[rank0] runtime={cfg["_runtime"]}', flush=True)
        print(f'[rank0] controlnet={load_notes["controlnet"]}', flush=True)
        print('[rank0] optimizer_groups=' + json.dumps([
            {
                'name': group.get('group_name', 'unnamed'),
                'lr': group['lr'],
                'weight_decay': group.get('weight_decay', 'default'),
                'params': sum(param.numel() for param in group['params']),
                'dtypes': sorted({str(param.dtype) for param in group['params']}),
            }
            for group in optimizer.param_groups
        ]), flush=True)
    if dist.is_initialized():
        dist.barrier()

    if stop_marker.exists():
        raise RuntimeError(f'stale STOP_TRAINING marker exists: {stop_marker}; inspect/remove it before resuming')

    global_step = start_step
    accum = int(cfg['_runtime']['grad_accum'])
    benchmark_start = None
    benchmark_done = False
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    stopped_by_watcher = False
    metric_totals: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    calibration_ratios: list[float] = []
    calibration_steps = int(
        cfg.get('training', {}).get('differential', {}).get('calibration_steps', 0)
    )
    while global_step < target_step and not stopped_by_watcher:
        sampler.set_epoch(global_step // max(1, len(loader)))
        for batch_idx, batch in enumerate(loader):
            if benchmark_start is None:
                torch.cuda.reset_peak_memory_stats(device)
                benchmark_start = time.perf_counter()
            run_step = global_step - run_origin_step
            if args.smoke_steps > 0 and differential_enabled and run_step == 0:
                batch = dict(batch)
                batch_size = int(batch['target_latents'].shape[0])
                batch['tau_override'] = torch.full((batch_size,), 0.5, dtype=torch.float32)
            loss, metrics = ddp(batch, train_step=run_step)
            accumulate_scalar_metrics(metric_totals, metric_counts, metrics)
            ratios = metrics.get('calibration_ratios')
            if isinstance(ratios, torch.Tensor) and ratios.numel():
                calibration_ratios.extend(ratios.detach().float().cpu().tolist())
            (loss / accum).backward()
            micro_step += 1
            if micro_step % accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in ddp.parameters() if p.requires_grad], float(cfg['training']['max_grad_norm']))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            completed_run_steps = global_step - run_origin_step
            core_model = unwrap_model(ddp)
            if (
                isinstance(core_model, DifferentialFlowModel)
                and core_model.hinge_g is None
                and completed_run_steps >= calibration_steps
            ):
                hinge_g, calibration_count = calibrate_hinge_g(calibration_ratios, device)
                core_model.set_hinge_g(hinge_g)
                cfg['training']['differential']['hinge_g_resolved'] = hinge_g
                if rank == 0:
                    save_yaml(output / 'resolved_config.yaml', cfg)
                    (log_dir / 'hinge_calibration.json').write_text(json.dumps({
                        'completed_run_steps': completed_run_steps,
                        'samples': calibration_count,
                        'quantile': 0.25,
                        'definition': 'Q25(face_diff_norm / d_arc_jk), yielding about 25% initial hinge activation',
                        'hinge_g': hinge_g,
                    }, indent=2), encoding='utf-8')
                    print(
                        f'[rank0] hinge calibration complete: g={hinge_g:.6f}, samples={calibration_count}',
                        flush=True,
                    )
            step_metrics = averaged_metrics(metric_totals, metric_counts)
            metric_totals.clear()
            metric_counts.clear()
            if rank == 0 and (
                global_step % int(cfg['training']['log_every']) == 0
                or completed_run_steps <= 3
            ):
                log_dir.mkdir(parents=True, exist_ok=True)
                row = {
                    'step': global_step,
                    'run_step': completed_run_steps,
                    'lr': cfg['_runtime']['effective_lr'],
                    'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
                    'sample_ids': list(batch.get('sample_id', [])),
                }
                for name in (
                    'loss_total', 'loss_pair', 'loss_teach', 'loss_inv', 'loss_hinge',
                    'hinge_active_rate', 'face_diff_norm', 'diff_active_ratio',
                    'hinge_calibrating', 'hinge_g', 'head_pose_null_ratio',
                    'tau_mean', 'z1_mean', 'controlnet_forward_count',
                    'transformer_forward_count', 'appearance_gate', 'garment_gate',
                    'head_pose_gate',
                ):
                    if name in step_metrics:
                        row[name] = step_metrics[name]
                if isinstance(core_model, DifferentialFlowModel) and core_model.hinge_g is not None:
                    row['hinge_g'] = float(core_model.hinge_g)
                with (log_dir / 'train.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(f'[rank0] {row}', flush=True)
            if not benchmark_done and completed_run_steps >= int(cfg['training']['benchmark_steps']):
                elapsed = time.perf_counter() - benchmark_start
                imgs = int(cfg['_runtime']['global_batch']) * int(cfg['training']['benchmark_steps'])
                per_step = elapsed / int(cfg['training']['benchmark_steps'])
                bench = {
                    'steps': int(cfg['training']['benchmark_steps']),
                    'seconds': elapsed,
                    'seconds_per_optimizer_step': per_step,
                    'img_per_sec': imgs / elapsed,
                    'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
                    'estimated_continuation_hours': per_step * int(cfg['_runtime']['continuation_steps']) / 3600,
                    'target_step': target_step,
                }
                if rank == 0:
                    (log_dir / 'benchmark.json').write_text(json.dumps(bench, indent=2), encoding='utf-8')
                    print(f'[rank0] benchmark={bench}', flush=True)
                benchmark_done = True
            if rank == 0 and global_step % int(cfg['training']['checkpoint_every']) == 0:
                save_checkpoint(ckpt_dir / f'step-{global_step:06d}', core_model, optimizer, global_step, sampler, cfg)
            if sync_stop_requested(stop_marker, rank, device):
                stopped_by_watcher = True
                if rank == 0:
                    print(f'[rank0] watcher requested stop at step={global_step}: {stop_marker}', flush=True)
                break
            if global_step >= target_step:
                break
    if rank == 0 and cfg['training'].get('save_final', True):
        save_checkpoint(ckpt_dir / 'final', unwrap_model(ddp), optimizer, global_step, sampler, cfg)
        (output / 'training_status.json').write_text(json.dumps({
            'status': 'stopped_by_watcher' if stopped_by_watcher else 'complete',
            'step': global_step,
            'run_step': global_step - run_origin_step,
            'target_steps': target_step,
            'stop_marker': str(stop_marker) if stopped_by_watcher else None,
        }, indent=2), encoding='utf-8')
    cleanup_dist()
    if stopped_by_watcher:
        raise SystemExit(3)


def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 1 paired-flow warmup training for MA-RA-CDT B2 baseline.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--all-gpus-train', action='store_true')
    parser.add_argument('--dev-single-gpu', action='store_true', help='Smoke-test only: bypass 3-rank default and run one process without DDP.')
    parser.add_argument('--allow-partial-cache', action='store_true', help='Smoke-test only: restrict train IDs to cached samples instead of requiring 100% coverage.')
    parser.add_argument('--smoke-steps', type=int, default=0, help='Smoke-test only: override total_steps and grad_accum for a short run.')
    parser.add_argument('--override-total-steps', type=int, default=None, help='Run a shorter real training job without changing grad_accum, used by speed_bench.py.')
    parser.add_argument('--override-output-id', default=None, help='Override experiment.id for isolated smoke or recovery runs.')
    parser.add_argument('--resume', default=None, help='Resume trainable weights, optimizer, step, and sampler from a checkpoint directory.')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
