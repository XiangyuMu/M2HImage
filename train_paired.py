from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

from conditions import (
    FluxConditionAdapter, assert_real_controlnet, atomic_torch_save, choose_dtype, get_resolution, load_yaml, make_image_ids,
    make_text_ids, save_yaml, seed_everything, sha256_file, short_hash_path, unpack_latents,
)
from dataset import PairedWarmupDataset, ResumeDistributedSampler
from manual_review_gate import review_status
from pulid_flux import PuLIDFluxAdapter
from spatial_conditions import (
    downsample_reference_tokens,
    make_hair_reference_image_ids,
    make_reference_image_ids,
    pad_control_samples_for_reference,
)
from train_recognizer import (
    RetinaFaceGeometryDetector,
    TrainArcFaceRecognizer,
    differentiable_face_align,
    similarity_matrices,
)


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
        spatial = cfg.get('model', {}).get('spatial_conditions', {})
        garment_cfg = spatial.get('garment_reference', {})
        self.garment_reference_enabled = bool(garment_cfg.get('enabled', False))
        self.reference_stride = int(garment_cfg.get('stride', 1))
        self.reference_x_offset = float(garment_cfg.get('x_offset', 64.0))
        hair_cfg = spatial.get('hair', {})
        self.spatial_hair_enabled = bool(hair_cfg.get('enabled', False))
        self.hair_enabled = bool(
            cfg.get('model', {}).get('identity_adapter', {}).get(
                'use_hair_tokens', False
            )
        )
        self.hair_token_mode = str(
            cfg.get('model', {}).get('identity_adapter', {}).get(
                'hair_token_mode', 'dense_with_position'
            )
        ).lower()
        hair_reference_cfg = hair_cfg.get('reference', {})
        self.hair_reference_enabled = bool(
            hair_reference_cfg.get('enabled', False)
        )
        self.hair_reference_stride = int(hair_reference_cfg.get('stride', 1))
        self.hair_reference_y_offset = float(
            hair_reference_cfg.get('y_offset', 64.0)
        )
        self.pair_region_cfg = cfg.get('training', {}).get(
            'paired_region_weighting', {}
        )
        self.pair_region_weighting_enabled = bool(
            self.pair_region_cfg.get('enabled', False)
        )
        self.experiment_cfg = cfg.get('experiment_method', {})
        self.experiment_name = str(self.experiment_cfg.get('name', 'baseline')).lower()
        self.async_flow_enabled = self.experiment_name in {'c', 'async_flow', 'async'}
        if self.reference_stride < 1:
            raise ValueError(f'garment reference stride must be >=1, got {self.reference_stride}')
        if self.hair_reference_stride < 1:
            raise ValueError(
                f'hair reference stride must be >=1, got {self.hair_reference_stride}'
            )
        if self.hair_reference_enabled and not self.spatial_hair_enabled:
            raise RuntimeError(
                'hair in-context reference requires spatial_conditions.hair.enabled'
            )
        self.image_token_count = (self.height // 16) * (self.width // 16)

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
        tau_image = tau.view(-1, 1, 1)
        if self.async_flow_enabled:
            required = ('cloth_safe_z', 'hair_z', 'face_z')
            missing = [key for key in required if key not in batch]
            if missing:
                raise RuntimeError(f'async flow requires token masks {missing}')
            cloth = batch['cloth_safe_z'].to(device=device, dtype=dtype).clamp(0.0, 1.0)
            hair = batch['hair_z'].to(device=device, dtype=dtype).clamp(0.0, 1.0)
            face = batch['face_z'].to(device=device, dtype=dtype).clamp(0.0, 1.0)
            if cloth.ndim != 2 or cloth.shape[:2] != z0.shape[:2]:
                raise RuntimeError(
                    f'async flow mask shape={tuple(cloth.shape)} must match latent tokens={tuple(z0.shape[:2])}'
                )
            identity = torch.maximum(face, hair)
            garment = cloth
            background = (1.0 - torch.maximum(garment, identity)).clamp(0.0, 1.0)
            gamma = self.experiment_cfg.get('region_gamma', {})
            gamma_bg = float(gamma.get('background', 2.0))
            gamma_garment = float(gamma.get('garment', 2.0))
            gamma_boundary = float(gamma.get('boundary', 1.0))
            gamma_identity = float(gamma.get('identity', 0.7))
            if min(gamma_bg, gamma_garment, gamma_boundary, gamma_identity) <= 0.0:
                raise ValueError('async flow region gamma values must be positive')
            # Boundary receives an intermediate schedule where cloth and identity overlap.
            overlap = (garment * identity).clamp(0.0, 1.0)
            garment_interior = (garment - overlap).clamp(0.0, 1.0)
            identity_only = (identity - overlap).clamp(0.0, 1.0)
            beta = (
                background * tau[:, None].pow(gamma_bg)
                + garment_interior * tau[:, None].pow(gamma_garment)
                + overlap * tau[:, None].pow(gamma_boundary)
                + identity_only * tau[:, None].pow(gamma_identity)
            ).unsqueeze(-1)
            dbeta = (
                background * gamma_bg * tau[:, None].clamp_min(1e-6).pow(gamma_bg - 1.0)
                + garment_interior * gamma_garment * tau[:, None].clamp_min(1e-6).pow(gamma_garment - 1.0)
                + overlap * gamma_boundary * tau[:, None].clamp_min(1e-6).pow(gamma_boundary - 1.0)
                + identity_only * gamma_identity * tau[:, None].clamp_min(1e-6).pow(gamma_identity - 1.0)
            ).unsqueeze(-1)
            z_tau = (1.0 - beta) * z0 + beta * z1
            target_v = dbeta * (z1 - z0)
        else:
            z_tau = (1.0 - tau_image) * z0 + tau_image * z1
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
            'async_flow': torch.tensor(float(self.async_flow_enabled), device=device),
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
        hair_ref_tokens: torch.Tensor | None = None,
        hair_ref_positions: torch.Tensor | None = None,
        hair_ref_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.hair_enabled:
            adapter_tokens = self.adapter(
                appearance,
                garment,
                head_pose,
                hair_ref_tokens=hair_ref_tokens,
                hair_ref_positions=hair_ref_positions,
                hair_ref_mask=hair_ref_mask,
                route_scales=self._route_scales(tau),
            )
        else:
            adapter_tokens = self.adapter(
                appearance,
                garment,
                head_pose,
                route_scales=self._route_scales(tau),
            )
        return torch.cat([prompt, adapter_tokens], dim=1)

    def _route_scales(self, tau: torch.Tensor | None) -> dict[str, torch.Tensor] | None:
        if self.experiment_name not in {'a', 'b', 'c', 'async_flow', 'async'} or tau is None:
            return None
        # Early flow steps preserve mannequin geometry/garment; late steps resolve identity.
        t = tau.float().clamp(0.0, 1.0)
        identity = 0.65 + 0.70 * (1.0 - t)
        garment = 1.20 - 0.25 * (1.0 - t)
        pose = 1.15 - 0.15 * (1.0 - t)
        return {'appearance': identity, 'garment': garment, 'pose': pose}

    def _hair_inputs(
        self,
        batch: dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        prefix: str = '',
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.hair_enabled:
            return None, None, None
        if self.hair_token_mode == 'semantic_dense':
            token_key = f'{prefix}hair_semantic_tokens'
            mask_key = f'{prefix}hair_semantic_mask'
            missing = [key for key in (token_key, mask_key) if key not in batch]
            if missing:
                raise RuntimeError(f'semantic hair route enabled but batch is missing {missing}')
            return (
                batch[token_key].to(device=device, dtype=dtype),
                None,
                batch[mask_key].to(device=device, dtype=dtype),
            )
        keys = tuple(f'{prefix}{name}' for name in ('hair_ref_tokens', 'hair_ref_positions', 'hair_ref_mask'))
        missing = [key for key in keys if key not in batch]
        if missing:
            raise RuntimeError(f'hair route enabled but batch is missing {missing}')
        return tuple(batch[key].to(device=device, dtype=dtype) for key in keys)

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
        garment_ref_latents: torch.Tensor | None = None,
        hair_ref_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device, dtype = z_tau.device, z_tau.dtype
        image_token_count = z_tau.shape[1]
        if image_token_count != self.image_token_count:
            raise RuntimeError(
                f'image token count={image_token_count}, expected={self.image_token_count}'
            )
        hidden_states = z_tau
        transformer_img_ids = img_ids
        transformer_cn = cn_samples
        garment_reference_token_count = 0
        hair_reference_token_count = 0
        if self.garment_reference_enabled:
            if garment_ref_latents is None:
                raise RuntimeError('garment reference route enabled but garment_ref_latents is missing')
            reference = garment_ref_latents.to(device=device, dtype=dtype)
            if reference.ndim == 2:
                reference = reference.unsqueeze(0)
            if reference.shape[0] != z_tau.shape[0] or reference.shape[2] != z_tau.shape[2]:
                raise RuntimeError(
                    f'garment reference shape={tuple(reference.shape)} is incompatible with '
                    f'image tokens={tuple(z_tau.shape)}'
                )
            reference = downsample_reference_tokens(
                reference, self.width, self.height, self.reference_stride
            )
            reference_ids = make_reference_image_ids(
                self.width,
                self.height,
                self.reference_x_offset,
                device,
                dtype,
                stride=self.reference_stride,
            )
            garment_reference_token_count = reference.shape[1]
            if reference_ids.shape[0] != garment_reference_token_count:
                raise RuntimeError(
                    f'reference IDs={reference_ids.shape[0]} and tokens={garment_reference_token_count} differ'
                )
            hidden_states = torch.cat((z_tau, reference), dim=1)
            transformer_img_ids = torch.cat((img_ids, reference_ids), dim=0)
        if self.hair_reference_enabled:
            if hair_ref_latents is None:
                raise RuntimeError(
                    'hair in-context route enabled but hair_ref_latents is missing'
                )
            hair_reference = hair_ref_latents.to(device=device, dtype=dtype)
            if hair_reference.ndim == 2:
                hair_reference = hair_reference.unsqueeze(0)
            if (
                hair_reference.shape[0] != z_tau.shape[0]
                or hair_reference.shape[2] != z_tau.shape[2]
            ):
                raise RuntimeError(
                    f'hair reference shape={tuple(hair_reference.shape)} is incompatible '
                    f'with image tokens={tuple(z_tau.shape)}'
                )
            hair_reference = downsample_reference_tokens(
                hair_reference,
                self.width,
                self.height,
                self.hair_reference_stride,
            )
            hair_ids = make_hair_reference_image_ids(
                self.width,
                self.height,
                self.hair_reference_y_offset,
                device,
                dtype,
                stride=self.hair_reference_stride,
            )
            hair_reference_token_count = hair_reference.shape[1]
            if hair_ids.shape[0] != hair_reference_token_count:
                raise RuntimeError(
                    f'hair reference IDs={hair_ids.shape[0]} and '
                    f'tokens={hair_reference_token_count} differ'
                )
            hidden_states = torch.cat((hidden_states, hair_reference), dim=1)
            transformer_img_ids = torch.cat((transformer_img_ids, hair_ids), dim=0)
        total_reference_tokens = (
            garment_reference_token_count + hair_reference_token_count
        )
        if total_reference_tokens:
            # ControlNet remains pose-only and is evaluated on the generated image.
            # Residuals are zero over both appended condition-only reference segments.
            transformer_cn = (
                pad_control_samples_for_reference(cn_samples[0], total_reference_tokens),
                pad_control_samples_for_reference(cn_samples[1], total_reference_tokens),
            )
        self.pulid.set_context(
            pulid_embed.to(device=device, dtype=dtype),
            float(self.cfg.get('model', {}).get('pulid', {}).get('id_weight', 1.0)),
        )
        pulid_context = self.pulid.context_kwargs()
        try:
            sample = self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=cond_tokens,
                pooled_projections=pooled,
                timestep=tau,
                img_ids=transformer_img_ids,
                txt_ids=make_text_ids(cond_tokens.shape[1], device, dtype),
                guidance=torch.full((z_tau.shape[0],), self.guidance_scale, device=device, dtype=dtype),
                joint_attention_kwargs=pulid_context,
                controlnet_block_samples=transformer_cn[0],
                controlnet_single_block_samples=transformer_cn[1],
                return_dict=True,
            ).sample
            if sample.shape[1] != image_token_count + total_reference_tokens:
                raise RuntimeError(
                    f'transformer output tokens={sample.shape[1]}, '
                    f'expected={image_token_count + total_reference_tokens}'
                )
            # Reference tokens are conditions only; flow matching and Euler updates use image tokens.
            return sample[:, :image_token_count]
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
        head_synthetic = batch.get('head_control_is_synthetic')
        head_synthetic_ratio = (
            head_synthetic.float().mean().detach()
            if head_synthetic is not None
            else torch.zeros((), device=device, dtype=torch.float32)
        )
        reference_tokens = self.image_token_count // (self.reference_stride ** 2) if self.garment_reference_enabled else 0
        hair_reference_tokens = (
            self.image_token_count // (self.hair_reference_stride ** 2)
            if self.hair_reference_enabled else 0
        )
        metrics = {
            'head_pose_null_ratio': head_null_ratio,
            'head_control_synthetic_ratio': head_synthetic_ratio,
            'garment_reference_tokens': torch.tensor(reference_tokens, device=device, dtype=torch.float32),
            'hair_reference_tokens': torch.tensor(
                hair_reference_tokens, device=device, dtype=torch.float32
            ),
            'tau_mean': flow['tau'].detach().float().mean(),
            'z1_mean': flow['z1'].detach().float().mean(),
            'controlnet_forward_count': torch.ones((), device=device, dtype=torch.float32),
        }
        metrics.update({
            name: torch.tensor(value, device=device, dtype=torch.float32)
            for name, value in self.adapter.gate_values().items()
        })
        return metrics

    @staticmethod
    def _masked_mse(
        squared_error: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = mask.to(
            device=squared_error.device, dtype=torch.float32
        ).clamp(0.0, 1.0)
        numerator = (squared_error.float() * weights.unsqueeze(-1)).sum()
        denominator = weights.sum().clamp_min(1e-6) * squared_error.shape[-1]
        return numerator / denominator

    def _paired_flow_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        squared = (pred.float() - target.float()).square()
        unweighted = squared.mean()
        if not self.pair_region_weighting_enabled:
            return unweighted, {'loss_pair_unweighted': unweighted.detach()}
        required = ('cloth_safe_z', 'hair_z', 'face_z')
        missing = [key for key in required if key not in batch]
        if missing:
            raise RuntimeError(
                f'paired region weighting requires token masks {missing}'
            )
        cloth = batch['cloth_safe_z'].to(
            device=squared.device, dtype=torch.float32
        ).clamp(0.0, 1.0)
        hair = batch['hair_z'].to(
            device=squared.device, dtype=torch.float32
        ).clamp(0.0, 1.0)
        face = batch['face_z'].to(
            device=squared.device, dtype=torch.float32
        ).clamp(0.0, 1.0)
        face_only = face * (1.0 - hair)
        weight = (
            1.0
            + (float(self.pair_region_cfg.get('w_cloth', 2.0)) - 1.0) * cloth
            + (float(self.pair_region_cfg.get('w_hair', 2.0)) - 1.0) * hair
            + (float(self.pair_region_cfg.get('w_face', 1.0)) - 1.0) * face_only
        )
        weight = weight / weight.mean(dim=1, keepdim=True).clamp_min(1e-6)
        loss = (squared * weight.unsqueeze(-1)).mean()

        occupied = torch.maximum(torch.maximum(cloth, hair), face)
        other = (1.0 - occupied).clamp(0.0, 1.0)
        metrics = {
            'loss_pair_unweighted': unweighted.detach(),
            'mse_cloth_safe': self._masked_mse(squared, cloth).detach(),
            'mse_hair': self._masked_mse(squared, hair).detach(),
            'mse_face': self._masked_mse(squared, face_only).detach(),
            'mse_other': self._masked_mse(squared, other).detach(),
            'pair_weight_mean': weight.mean().detach(),
            'pair_weight_max': weight.max().detach(),
        }
        return loss, metrics

    def _extra_paired_loss(
        self,
        batch: dict[str, torch.Tensor],
        flow: dict[str, torch.Tensor],
        pred: torch.Tensor,
        decode_trigger: bool,
        hair_loss_accum_scale: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del batch, pred, decode_trigger, hair_loss_accum_scale
        return (
            torch.zeros((), device=flow['device'], dtype=torch.float32),
            {},
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        train_step: int | None = None,
        decode_trigger: bool = False,
        identity_loss_accum_scale: float = 1.0,
        hair_loss_accum_scale: float = 1.0,
        optimizer_boundary: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del train_step, identity_loss_accum_scale, optimizer_boundary
        flow = self._prepare_flow(batch)
        dtype, device = flow['dtype'], flow['device']
        hair_inputs = self._hair_inputs(batch, device, dtype)
        cond_tokens = self._condition_tokens(
            flow['prompt'],
            batch['appearance'].to(device=device, dtype=dtype),
            batch['garment'].to(device=device, dtype=dtype),
            batch['head_pose'].to(device=device, dtype=dtype),
            *hair_inputs,
            tau=flow['tau'],
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
            garment_ref_latents=batch.get('garment_ref_latents'),
            hair_ref_latents=batch.get('hair_ref_latents'),
        )
        loss_pair, pair_metrics = self._paired_flow_loss(
            pred, flow['target_v'], batch
        )
        extra_loss, extra_metrics = self._extra_paired_loss(
            batch,
            flow,
            pred,
            decode_trigger,
            hair_loss_accum_scale,
        )
        loss = loss_pair + extra_loss
        metrics = self._base_metrics(batch, flow)
        metrics.update({
            'loss_total': loss.detach(),
            'loss_pair': loss_pair.detach(),
            'transformer_forward_count': torch.ones((), device=device, dtype=torch.float32),
        })
        metrics.update(pair_metrics)
        metrics.update(extra_metrics)
        return loss, metrics


class HairSupervisedWarmupFlowModel(WarmupFlowModel):
    """Paired warmup with sparse in-graph VAE/DINO hair supervision."""

    def __init__(
        self,
        transformer,
        controlnet,
        adapter: FluxConditionAdapter,
        pulid: PuLIDFluxAdapter,
        vae: torch.nn.Module,
        cfg: dict[str, Any],
    ) -> None:
        super().__init__(transformer, controlnet, adapter, pulid, cfg)
        if vae is None:
            raise RuntimeError('enabled paired hair loss requires a loaded VAE')
        if not (
            self.hair_reference_enabled
            or (self.hair_enabled and self.hair_token_mode == 'semantic_dense')
        ):
            raise RuntimeError(
                'paired hair loss requires spatial or semantic hair conditioning'
            )
        from hair_supervision import DifferentiableHairDinoLoss

        self.vae = vae.requires_grad_(False).eval()
        self.hair_loss_cfg = cfg['training']['hair_loss']
        self.hair_supervisor = DifferentiableHairDinoLoss(
            self.hair_loss_cfg,
            next(transformer.parameters()).device,
        )

    def _decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        latents = unpack_latents(tokens, self.width, self.height)
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor

        def decode_fn(value: torch.Tensor) -> torch.Tensor:
            return self.vae.decode(value, return_dict=False)[0]

        if bool(self.hair_loss_cfg.get('gradient_checkpointing', True)):
            return checkpoint(decode_fn, latents, use_reentrant=False)
        return decode_fn(latents)

    def _extra_paired_loss(
        self,
        batch: dict[str, torch.Tensor],
        flow: dict[str, torch.Tensor],
        pred: torch.Tensor,
        decode_trigger: bool,
        hair_loss_accum_scale: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = flow['device']
        zero = torch.zeros((), device=device, dtype=torch.float32)
        defaults = {
            'loss_hair': zero,
            'loss_hair_dino': zero,
            'hair_cosine': zero,
            'loss_hair_sum': zero,
            'hair_cosine_sum': zero,
            'hair_valid_count': zero,
            'hair_loss_attempt_count': zero,
            'hair_loss_skip_count': zero,
            'hair_loss_triggered': zero,
            'hair_decode_seconds': zero,
        }
        if not decode_trigger:
            return zero, defaults
        tau_min = float(self.hair_loss_cfg.get('tau_min', 0.35))
        tau_max = float(self.hair_loss_cfg.get('tau_max', 0.7))
        active = (
            (flow['tau'].float() >= tau_min)
            & (flow['tau'].float() <= tau_max)
        )
        indices = torch.nonzero(active, as_tuple=False).flatten()
        if not len(indices):
            return zero, defaults
        required = ('hair_ref_tokens', 'hair_ref_mask', 'hair_z')
        missing = [key for key in required if key not in batch]
        if missing:
            raise RuntimeError(f'enabled paired hair loss is missing {missing}')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        tau = flow['tau'].float().view(-1, 1, 1)
        z_hat = flow['z_tau'].float() - tau * pred.float()
        decoded = self._decode_tokens(z_hat.index_select(0, indices))
        hair_loss, hair_metrics = self.hair_supervisor(
            decoded,
            batch['hair_ref_tokens'].to(device=device).index_select(0, indices),
            batch['hair_ref_mask'].to(device=device).index_select(0, indices),
            batch['hair_z'].to(device=device).index_select(0, indices),
        )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        weighted = (
            float(self.hair_loss_cfg.get('lambda_hair', 0.1))
            * hair_loss
            * float(hair_loss_accum_scale)
        )
        metrics = {
            **defaults,
            **hair_metrics,
            'loss_hair': hair_loss.detach(),
            'hair_loss_triggered': torch.ones(
                (), device=device, dtype=torch.float32
            ),
            'hair_decode_seconds': torch.tensor(
                time.perf_counter() - started,
                device=device,
                dtype=torch.float32,
            ),
        }
        return weighted, metrics


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

    def _identity_metric_defaults(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {}

    def _decode_schedule_metrics(
        self,
        flow: dict[str, torch.Tensor],
        train_step: int,
        optimizer_boundary: bool,
    ) -> dict[str, torch.Tensor]:
        del flow, train_step, optimizer_boundary
        return {}

    def _extra_differential_loss(
        self,
        batch: dict[str, torch.Tensor],
        flow: dict[str, torch.Tensor],
        z_hat_j: torch.Tensor,
        z_hat_k: torch.Tensor,
        active: torch.Tensor,
        train_step: int,
        decode_trigger: bool,
        identity_loss_accum_scale: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return torch.zeros((), device=flow['device'], dtype=torch.float32), {}

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        train_step: int | None = None,
        decode_trigger: bool = False,
        identity_loss_accum_scale: float = 1.0,
        hair_loss_accum_scale: float = 1.0,
        optimizer_boundary: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del hair_loss_accum_scale
        train_step = int(train_step or 0)
        flow = self._prepare_flow(batch)
        dtype, device = flow['dtype'], flow['device']
        garment = batch['garment'].to(device=device, dtype=dtype)
        head_pose = batch['head_pose'].to(device=device, dtype=dtype)
        paired_hair = self._hair_inputs(batch, device, dtype)
        paired_tokens = self._condition_tokens(
            flow['prompt'],
            batch['appearance'].to(device=device, dtype=dtype),
            garment,
            head_pose,
            *paired_hair,
            tau=flow['tau'],
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
            garment_ref_latents=batch.get('garment_ref_latents'),
            hair_ref_latents=batch.get('hair_ref_latents'),
        )
        loss_pair, pair_metrics = self._paired_flow_loss(
            pred_i, flow['target_v'], batch
        )
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
        extra_loss = zero
        extra_metrics = self._identity_metric_defaults(device)
        transformer_count = 1.0
        if bool(active.any()):
            hair_j = self._hair_inputs(batch, device, dtype, prefix='cf_j_')
            hair_k = self._hair_inputs(batch, device, dtype, prefix='cf_k_')
            tokens_j = self._condition_tokens(
                flow['prompt'],
                batch['cf_j_appearance'].to(device=device, dtype=dtype),
                garment,
                head_pose,
                *hair_j,
                tau=flow['tau'],
            )
            tokens_k = self._condition_tokens(
                flow['prompt'],
                batch['cf_k_appearance'].to(device=device, dtype=dtype),
                garment,
                head_pose,
                *hair_k,
                tau=flow['tau'],
            )
            pred_j = self._transformer_forward(
                flow['z_tau'],
                flow['tau'],
                tokens_j,
                batch['cf_j_pulid_id_embed'],
                cn_samples,
                pooled=flow['pooled'],
                img_ids=flow['img_ids'],
                garment_ref_latents=batch.get('garment_ref_latents'),
                hair_ref_latents=batch.get('cf_j_hair_ref_latents'),
            )
            pred_k = self._transformer_forward(
                flow['z_tau'],
                flow['tau'],
                tokens_k,
                batch['cf_k_pulid_id_embed'],
                cn_samples,
                pooled=flow['pooled'],
                img_ids=flow['img_ids'],
                garment_ref_latents=batch.get('garment_ref_latents'),
                hair_ref_latents=batch.get('cf_k_hair_ref_latents'),
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
            extra_loss, computed_extra_metrics = self._extra_differential_loss(
                batch,
                flow,
                z_hat_j,
                z_hat_k,
                active,
                train_step,
                decode_trigger,
                identity_loss_accum_scale,
            )
            extra_metrics.update(computed_extra_metrics)

        calibrating = self.hinge_g is None
        lambda_teach = float(self.diff_cfg.get('lambda_teach', 0.5))
        lambda_inv = float(self.diff_cfg.get('lambda_inv', 0.2))
        lambda_hinge = 0.0 if calibrating else float(self.diff_cfg.get('lambda_hinge', 0.05))
        total = (
            loss_pair
            + lambda_teach * loss_teach
            + lambda_inv * loss_inv
            + lambda_hinge * loss_hinge
            + extra_loss
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
        metrics.update(pair_metrics)
        metrics.update(extra_metrics)
        metrics.update(
            self._decode_schedule_metrics(
                flow, train_step, optimizer_boundary
            )
        )
        return total, metrics


class DirectedDifferentialFlowModel(DifferentialFlowModel):
    """A4 adds a differentiable identity objective without changing A2 losses.

    Held-out AdaFace is evaluation-only and must never be imported here. Identity
    references, semi-hard distances, and this loss all use frozen Glint360K ArcFace.
    """

    def __init__(
        self,
        transformer,
        controlnet,
        adapter: FluxConditionAdapter,
        pulid: PuLIDFluxAdapter,
        vae: torch.nn.Module,
        train_recognizer: TrainArcFaceRecognizer,
        face_detector: RetinaFaceGeometryDetector,
        cfg: dict[str, Any],
    ) -> None:
        super().__init__(transformer, controlnet, adapter, pulid, cfg)
        if vae is None:
            raise RuntimeError('A4 directed identity loss requires model.load_vae_in_train=true')
        self.vae = vae
        self.train_recognizer = train_recognizer
        self.face_detector = face_detector
        self.decode_cfg = self.diff_cfg['decode']
        self.identity_cfg = self.diff_cfg['identity_loss']
        if not self.decode_cfg.get('enabled', False) or not self.identity_cfg.get('enabled', False):
            raise RuntimeError('DirectedDifferentialFlowModel requires decode.enabled and identity_loss.enabled')
        self.vae.requires_grad_(False).eval()
        self.train_recognizer.requires_grad_(False).eval()
        self.hair_loss_cfg = cfg.get('training', {}).get('hair_loss', {})
        self.hair_supervisor = None
        if self.hair_loss_cfg.get('enabled', False):
            if not (
                self.hair_reference_enabled
                or (self.hair_enabled and self.hair_token_mode == 'semantic_dense')
            ):
                raise RuntimeError(
                    'hair loss requires spatial or semantic hair conditioning'
                )
            from hair_supervision import DifferentiableHairDinoLoss

            self.hair_supervisor = DifferentiableHairDinoLoss(self.hair_loss_cfg, next(transformer.parameters()).device)

    def _identity_metric_defaults(self, device: torch.device) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return {
            'loss_id_dir': zero,
            'loss_id_abs': zero,
            'sim_gap': zero,
            'id_loss_attempt_count': zero,
            'id_loss_skip_count': zero,
            'id_loss_triggered': zero,
            'id_decode_seconds': zero,
            'id_decode_branch': zero,
            'loss_hair_dino': zero,
            'hair_cosine': zero,
            'loss_hair_sum': zero,
            'hair_cosine_sum': zero,
            'hair_valid_count': zero,
            'hair_loss_attempt_count': zero,
            'hair_loss_skip_count': zero,
            'hair_loss_triggered': zero,
            'hair_decode_seconds': zero,
            'hair_schedule_sample_count': zero,
            'hair_schedule_hit_count': zero,
            'hair_skip_decode_freq_count': zero,
            'hair_skip_tau_window_count': zero,
            'hair_skip_area_count': zero,
            'hair_skip_target_invalid_count': zero,
            'hair_skip_parsing_failure_count': zero,
        }

    def _decode_schedule_metrics(
        self,
        flow: dict[str, torch.Tensor],
        train_step: int,
        optimizer_boundary: bool,
    ) -> dict[str, torch.Tensor]:
        device = flow['device']
        zero = torch.zeros((), device=device, dtype=torch.float32)
        if self.hair_supervisor is None or not optimizer_boundary:
            return {
                'hair_schedule_sample_count': zero,
                'hair_schedule_hit_count': zero,
                'hair_skip_decode_freq_count': zero,
                'hair_skip_tau_window_count': zero,
            }
        freq = max(1, int(self.hair_loss_cfg.get('decode_freq', 3)))
        scheduled = int(train_step) % freq == 0
        batch_size = int(flow['tau'].shape[0])
        decode_miss = 0 if scheduled else batch_size
        tau_min = float(self.hair_loss_cfg.get('tau_min', 0.3))
        tau_max = float(self.hair_loss_cfg.get('tau_max', 0.75))
        tau_miss = (
            ((flow['tau'].float() < tau_min) | (flow['tau'].float() > tau_max))
            .float()
            .sum()
            if scheduled
            else zero
        )
        return {
            'hair_schedule_sample_count': torch.tensor(
                float(batch_size), device=device, dtype=torch.float32
            ),
            'hair_schedule_hit_count': torch.tensor(
                float(batch_size if scheduled else 0),
                device=device,
                dtype=torch.float32,
            ),
            'hair_skip_decode_freq_count': torch.tensor(
                float(decode_miss), device=device, dtype=torch.float32
            ),
            'hair_skip_tau_window_count': tau_miss,
        }

    def _decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        latents = unpack_latents(tokens, self.width, self.height)
        latent_scale = float(self.decode_cfg.get('latent_scale', 1.0))
        if not 0.0 < latent_scale <= 1.0:
            raise RuntimeError(f'decode latent_scale must be in (0,1], got {latent_scale}')
        if latent_scale < 1.0:
            target_h = max(1, int(round(latents.shape[-2] * latent_scale)))
            target_w = max(1, int(round(latents.shape[-1] * latent_scale)))
            latents = F.interpolate(latents, size=(target_h, target_w), mode='bilinear', align_corners=False)
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        def decode_fn(value: torch.Tensor) -> torch.Tensor:
            return self.vae.decode(value, return_dict=False)[0]

        if bool(self.decode_cfg.get('gradient_checkpointing', True)):
            return checkpoint(decode_fn, latents, use_reentrant=False)
        return decode_fn(latents)

    def _decode_identity_branch(
        self,
        tokens: torch.Tensor,
        positive_refs: torch.Tensor,
        negative_refs: torch.Tensor,
        identity_selector: torch.Tensor,
        hair_selector: torch.Tensor,
        target_hair_tokens: torch.Tensor | None = None,
        target_hair_mask: torch.Tensor | None = None,
        projected_hair_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = tokens.device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        decoded = self._decode_tokens(tokens)
        zero = decoded.sum() * 0.0
        hair_loss = zero
        hair_metrics = {
            'loss_hair_dino': zero.detach().float(),
            'hair_cosine': zero.detach().float(),
            'loss_hair_sum': zero.detach().float(),
            'hair_cosine_sum': zero.detach().float(),
            'hair_valid_count': zero.detach().float(),
            'hair_loss_attempt_count': torch.zeros((), device=device),
            'hair_loss_skip_count': torch.zeros((), device=device),
            'hair_skip_area_count': torch.zeros((), device=device),
            'hair_skip_target_invalid_count': torch.zeros((), device=device),
            'hair_skip_parsing_failure_count': torch.zeros((), device=device),
        }
        hair_indices = torch.nonzero(
            hair_selector.to(device=device, dtype=torch.bool), as_tuple=False
        ).flatten()
        if self.hair_supervisor is not None and len(hair_indices):
            if (
                target_hair_tokens is None
                or target_hair_mask is None
                or projected_hair_mask is None
            ):
                raise RuntimeError(
                    'enabled hair loss requires branch DINO targets and projected hair_z'
                )
            hair_loss, hair_metrics = self.hair_supervisor(
                decoded.index_select(0, hair_indices),
                target_hair_tokens.index_select(0, hair_indices),
                target_hair_mask.index_select(0, hair_indices),
                projected_hair_mask.index_select(0, hair_indices),
            )
        weighted_hair = float(
            self.hair_loss_cfg.get(
                'lambda_hair', self.hair_loss_cfg.get('lambda', 0.05)
            )
        ) * hair_loss
        identity_indices = torch.nonzero(
            identity_selector.to(device=device, dtype=torch.bool), as_tuple=False
        ).flatten()
        identity_decoded = decoded.index_select(0, identity_indices)
        geometries = (
            self.face_detector.detect_tensor_batch(identity_decoded)
            if len(identity_indices)
            else []
        )
        min_conf = float(self.identity_cfg.get('min_detection_confidence', 0.5))
        valid_indices = [
            index
            for index, geometry in enumerate(geometries)
            if geometry is not None and geometry.confidence >= min_conf
        ]
        attempt_count = len(geometries)
        skip_count = attempt_count - len(valid_indices)
        if not valid_indices:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            return weighted_hair, {
                'loss_id_dir': zero.detach().float(),
                'loss_id_abs': zero.detach().float(),
                'sim_gap': zero.detach().float(),
                'id_loss_attempt_count': torch.tensor(float(attempt_count), device=device),
                'id_loss_skip_count': torch.tensor(float(skip_count), device=device),
                'id_decode_seconds': torch.tensor(time.perf_counter() - started, device=device),
                **hair_metrics,
            }
        valid_local = torch.as_tensor(
            valid_indices, device=device, dtype=torch.long
        )
        valid = identity_indices.index_select(0, valid_local)
        matrices = similarity_matrices([geometries[index] for index in valid_indices])
        aligned = differentiable_face_align(
            identity_decoded.index_select(0, valid_local).clamp(-1.0, 1.0),
            matrices,
            image_size=int(self.train_recognizer.input_size),
        )
        generated = self.train_recognizer(aligned)
        positive = F.normalize(positive_refs.index_select(0, valid).to(device=device, dtype=torch.float32), dim=1)
        negative = F.normalize(negative_refs.index_select(0, valid).to(device=device, dtype=torch.float32), dim=1)
        sim_positive = (generated * positive).sum(dim=1)
        sim_negative = (generated * negative).sum(dim=1)
        sim_gap = sim_positive - sim_negative
        margin = float(self.identity_cfg.get('margin', 0.1))
        loss_dir = F.softplus(sim_negative - sim_positive + margin).mean()
        loss_abs = (1.0 - sim_positive).mean()
        weighted = (
            float(self.identity_cfg.get('lambda_dir', 0.1)) * loss_dir
            + float(self.identity_cfg.get('lambda_abs', 0.05)) * loss_abs
        )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return weighted + weighted_hair, {
            'loss_id_dir': loss_dir.detach(),
            'loss_id_abs': loss_abs.detach(),
            'sim_gap': sim_gap.detach().mean(),
            'id_loss_attempt_count': torch.tensor(float(attempt_count), device=device),
            'id_loss_skip_count': torch.tensor(float(skip_count), device=device),
            'id_decode_seconds': torch.tensor(time.perf_counter() - started, device=device),
            **hair_metrics,
        }

    def _extra_differential_loss(
        self,
        batch: dict[str, torch.Tensor],
        flow: dict[str, torch.Tensor],
        z_hat_j: torch.Tensor,
        z_hat_k: torch.Tensor,
        active: torch.Tensor,
        train_step: int,
        decode_trigger: bool,
        identity_loss_accum_scale: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = flow['device']
        defaults = self._identity_metric_defaults(device)
        id_freq = max(1, int(self.decode_cfg.get('freq', 3)))
        hair_freq = max(1, int(self.hair_loss_cfg.get('decode_freq', 3)))
        id_scheduled = int(train_step) % id_freq == 0
        hair_scheduled = (
            self.hair_supervisor is not None
            and int(train_step) % hair_freq == 0
        )
        id_tau_min = float(self.decode_cfg.get('tau_min', 0.35))
        id_tau_max = float(self.decode_cfg.get('tau_max', 0.7))
        hair_tau_min = float(self.hair_loss_cfg.get('tau_min', 0.3))
        hair_tau_max = float(self.hair_loss_cfg.get('tau_max', 0.75))
        identity_active = (
            active
            & (flow['tau'].float() >= id_tau_min)
            & (flow['tau'].float() <= id_tau_max)
            if id_scheduled
            else torch.zeros_like(active)
        )
        hair_active = (
            active
            & (flow['tau'].float() >= hair_tau_min)
            & (flow['tau'].float() <= hair_tau_max)
            if hair_scheduled
            else torch.zeros_like(active)
        )
        decode_active = identity_active | hair_active
        if not decode_trigger:
            return torch.zeros((), device=device, dtype=torch.float32), defaults
        indices = torch.nonzero(decode_active, as_tuple=False).flatten()
        if not len(indices):
            return torch.zeros((), device=device, dtype=torch.float32), defaults
        refs_j = batch['cf_j_train_embed'].to(device=device, dtype=torch.float32)
        refs_k = batch['cf_k_train_embed'].to(device=device, dtype=torch.float32)
        hair_targets_j = batch.get('cf_j_hair_ref_tokens')
        hair_masks_j = batch.get('cf_j_hair_ref_mask')
        hair_targets_k = batch.get('cf_k_hair_ref_tokens')
        hair_masks_k = batch.get('cf_k_hair_ref_mask')
        projected_hair = batch.get('hair_z')
        if bool(hair_active.any()) and any(
            value is None
            for value in (
                hair_targets_j,
                hair_masks_j,
                hair_targets_k,
                hair_masks_k,
                projected_hair,
            )
        ):
            raise RuntimeError(
                'enabled hair loss requires cached CF-j/k DINO targets and anchor hair_z'
            )
        decode_both = bool(self.decode_cfg.get('both', False))
        branch_period = min(id_freq, hair_freq)
        branches = (
            (0, 1)
            if decode_both
            else ((int(train_step) // branch_period) % 2,)
        )
        identity_selector = identity_active.index_select(0, indices)
        hair_selector = hair_active.index_select(0, indices)
        losses = []
        rows = []
        for branch in branches:
            if branch == 0:
                tokens, positive, negative = z_hat_j, refs_j, refs_k
                target_hair, target_hair_mask = hair_targets_j, hair_masks_j
            else:
                tokens, positive, negative = z_hat_k, refs_k, refs_j
                target_hair, target_hair_mask = hair_targets_k, hair_masks_k
            loss, row = self._decode_identity_branch(
                tokens.index_select(0, indices),
                positive.index_select(0, indices),
                negative.index_select(0, indices),
                identity_selector,
                hair_selector,
                None if target_hair is None else target_hair.to(device=device).index_select(0, indices),
                None if target_hair_mask is None else target_hair_mask.to(device=device).index_select(0, indices),
                None if projected_hair is None else projected_hair.to(device=device).index_select(0, indices),
            )
            losses.append(loss)
            rows.append(row)
        combined = torch.stack(losses).mean() * float(identity_loss_accum_scale)
        metrics = {
            key: torch.stack([row[key].float() for row in rows]).mean()
            for key in (
                'loss_id_dir',
                'loss_id_abs',
                'sim_gap',
                'id_decode_seconds',
                'loss_hair_dino',
                'hair_cosine',
            )
        }
        metrics['id_loss_attempt_count'] = torch.stack(
            [row['id_loss_attempt_count'].float() for row in rows]
        ).sum()
        metrics['id_loss_skip_count'] = torch.stack(
            [row['id_loss_skip_count'].float() for row in rows]
        ).sum()
        metrics['hair_loss_attempt_count'] = torch.stack(
            [row['hair_loss_attempt_count'].float() for row in rows]
        ).sum()
        metrics['hair_loss_skip_count'] = torch.stack(
            [row['hair_loss_skip_count'].float() for row in rows]
        ).sum()
        for key in (
            'loss_hair_sum',
            'hair_cosine_sum',
            'hair_valid_count',
            'hair_skip_area_count',
            'hair_skip_target_invalid_count',
            'hair_skip_parsing_failure_count',
        ):
            metrics[key] = torch.stack(
                [row[key].float() for row in rows]
            ).sum()
        metrics['id_loss_triggered'] = torch.tensor(
            float(bool(identity_active.any())), device=device, dtype=torch.float32
        )
        metrics['hair_loss_triggered'] = torch.tensor(
            float(bool(hair_active.any())), device=device, dtype=torch.float32
        )
        metrics['hair_decode_seconds'] = (
            metrics['id_decode_seconds']
            if bool(hair_active.any())
            else torch.zeros((), device=device, dtype=torch.float32)
        )
        metrics['id_decode_branch'] = torch.tensor(
            0.5 if decode_both else float(branches[0]), device=device, dtype=torch.float32
        )
        return combined, metrics


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
    hair_loss_enabled = bool(
        cfg.get('training', {}).get('hair_loss', {}).get('enabled', False)
    )
    if cfg['model'].get('load_vae_in_train', False) or hair_loss_enabled:
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
        'vae_in_train': bool(vae is not None),
    }


def load_directed_identity_components(
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[TrainArcFaceRecognizer, RetinaFaceGeometryDetector, dict[str, Any]]:
    recognizer_cfg = cfg.get('model', {}).get('train_recognizer')
    if not isinstance(recognizer_cfg, dict):
        raise RuntimeError('A4 requires model.train_recognizer with a real Glint360K ArcFace checkpoint')
    recognizer = TrainArcFaceRecognizer(recognizer_cfg, device)
    detector_device = int(recognizer_cfg.get('detector_device_id', -1))
    detector = RetinaFaceGeometryDetector(recognizer_cfg, device_id=detector_device)
    return recognizer, detector, {
        'train_recognizer': recognizer.launch_note(),
        'face_detector': {
            'name': recognizer_cfg.get('detector_name', 'antelopev2'),
            'model_root': recognizer_cfg.get('detector_model_root'),
            'device_id': detector_device,
            'role': 'no-grad geometry only',
        },
    }


def build_optimizer(module: torch.nn.Module, cfg: dict[str, Any]):
    lr = float(cfg['_runtime']['effective_lr'])
    adapter_cfg = cfg['model'].get('identity_adapter', {})
    adapter_mult = float(adapter_cfg.get('condition_adapter_lr_mult', 1.0))
    gate_mult = float(adapter_cfg.get('gate_lr_mult', 10.0))
    lora_params = []
    adapter_params = []
    gate_params = []
    gate_suffixes = ('appearance_gate', 'garment_gate', 'hair_gate', 'pose_gate')
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


def assert_gate_optimizer_lr(optimizer, cfg: dict[str, Any]) -> None:
    """Fail if checkpoint restore erased the dedicated condition-gate LR."""
    groups = {
        str(group.get('group_name', '')): group
        for group in optimizer.param_groups
    }
    gate_group = groups.get('condition_gates_fp32')
    if gate_group is None:
        raise RuntimeError(
            'trainable appearance/hair/head-pose gates are missing their optimizer group'
        )
    main_group = groups.get('transformer_lora') or groups.get('condition_adapter')
    if main_group is None:
        raise RuntimeError('cannot identify the main optimizer LR for gate validation')
    expected = float(
        cfg.get('model', {}).get('identity_adapter', {}).get('gate_lr_mult', 10.0)
    )
    ratio = float(gate_group['lr']) / max(float(main_group['lr']), 1e-20)
    if not math.isclose(ratio, expected, rel_tol=1e-6, abs_tol=1e-9):
        raise RuntimeError(
            f'condition gate LR ratio={ratio:.6g}, expected={expected:.6g}; '
            'optimizer resume would violate the gate-speed protocol'
        )


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


def summarize_sampling_window(
    local_distances: list[float],
    local_relaxations: list[int],
) -> dict[str, Any]:
    if dist.is_initialized():
        gathered: list[dict[str, list[float] | list[int]] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, {
            'distances': local_distances,
            'relaxations': local_relaxations,
        })
        distances = [float(value) for row in gathered if row for value in row['distances']]
        relaxations = [int(value) for row in gathered if row for value in row['relaxations']]
    else:
        distances = [float(value) for value in local_distances]
        relaxations = [int(value) for value in local_relaxations]
    if not distances:
        return {'count': 0}
    array = np.asarray(distances, dtype=np.float64)
    counts = Counter(relaxations)
    return {
        'count': int(len(array)),
        'd_jk_mean': float(np.mean(array)),
        'd_jk_p25': float(np.quantile(array, 0.25)),
        'd_jk_p75': float(np.quantile(array, 0.75)),
        'relax_strict': int(counts.get(0, 0)),
        'relax_age': int(counts.get(1, 0)),
        'relax_skin': int(counts.get(2, 0)),
        'distances': [float(value) for value in distances],
    }


def reduce_sparse_metric_sums(
    totals: dict[str, float],
    device: torch.device,
) -> dict[str, float]:
    names = (
        'loss_id_dir',
        'loss_id_abs',
        'sim_gap',
        'id_decode_seconds',
        'id_decode_branch',
        'id_loss_attempt_count',
        'id_loss_skip_count',
        'id_loss_triggered',
        'loss_hair_dino',
        'loss_hair',
        'hair_cosine',
        'loss_hair_sum',
        'hair_cosine_sum',
        'hair_valid_count',
        'hair_decode_seconds',
        'hair_loss_triggered',
        'hair_loss_attempt_count',
        'hair_loss_skip_count',
        'hair_schedule_sample_count',
        'hair_schedule_hit_count',
        'hair_skip_decode_freq_count',
        'hair_skip_tau_window_count',
        'hair_skip_area_count',
        'hair_skip_target_invalid_count',
        'hair_skip_parsing_failure_count',
    )
    values = torch.tensor(
        [float(totals.get(name, 0.0)) for name in names],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {name: float(value) for name, value in zip(names, values.cpu().tolist(), strict=True)}


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
    differential_cfg = cfg.get('training', {}).get('differential', {})
    differential_enabled = bool(differential_cfg.get('enabled', False))
    hair_loss_cfg = cfg.get('training', {}).get('hair_loss', {})
    hair_loss_enabled = bool(hair_loss_cfg.get('enabled', False))
    directed_enabled = bool(
        differential_enabled
        and differential_cfg.get('decode', {}).get('enabled', False)
        and differential_cfg.get('identity_loss', {}).get('enabled', False)
    )
    paired_hair_loss_enabled = bool(hair_loss_enabled and not differential_enabled)
    if hair_loss_enabled and differential_enabled and not directed_enabled:
        raise RuntimeError(
            'hair_loss with a differential run requires the existing directed decode path; '
            'the spatial paired continuation must keep differential.enabled=false'
        )
    if directed_enabled:
        train_recognizer, face_detector, identity_notes = load_directed_identity_components(cfg, device)
        load_notes.update(identity_notes)
        model = DirectedDifferentialFlowModel(
            transformer,
            controlnet,
            adapter,
            pulid,
            vae,
            train_recognizer,
            face_detector,
            cfg,
        )
    elif differential_enabled:
        model = DifferentialFlowModel(transformer, controlnet, adapter, pulid, cfg)
    elif paired_hair_loss_enabled:
        model = HairSupervisedWarmupFlowModel(
            transformer,
            controlnet,
            adapter,
            pulid,
            vae,
            cfg,
        )
    else:
        model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg)
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
    assert_gate_optimizer_lr(optimizer, cfg)
    core_model = unwrap_model(ddp)
    core_model.set_run_origin_step(start_step)
    if isinstance(core_model, DifferentialFlowModel) and core_model.hinge_g is not None:
        # A resumed calibrated run must persist the actual scalar, not the
        # unresolved null from the source YAML.
        cfg['training']['differential']['hinge_g_resolved'] = float(
            core_model.hinge_g)
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
        'directed_identity_enabled': directed_enabled,
        'paired_hair_loss_enabled': paired_hair_loss_enabled,
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
    identity_bank_path = (
        Path(cfg['data']['root']) / cfg['data']['identity_bank']
        if differential_enabled and cfg['data'].get('identity_bank')
        else None
    )
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
            'differential_sampling': differential_cfg.get('sampling', 'random'),
            'hinge_g_resolved': differential_cfg.get('hinge_g_resolved'),
            'identity_bank': {
                'path': str(identity_bank_path) if identity_bank_path else None,
                'hash': sha256_file(identity_bank_path)[:16] if identity_bank_path and identity_bank_path.exists() else None,
                'semihard_pool': differential_cfg.get('semihard_pool'),
            },
            'fairness_note': 'A4, A2, and B2-cont must use the same resume checkpoint/hash, seed, sampler state, train IDs, global batch, LR, rank, and continuation step count.',
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
    sampling_log_every = max(1, int(differential_cfg.get('sampling_log_every', 100)))
    sampling_distances: list[float] = []
    sampling_relaxations: list[int] = []
    sampling_stats_for_log: dict[str, Any] | None = None
    hair_metric_window: deque[dict[str, float]] = deque(
        maxlen=max(1, int(hair_loss_cfg.get('log_window_steps', 100)))
    )
    while global_step < target_step and not stopped_by_watcher:
        sampler.set_epoch(global_step // max(1, len(loader)))
        for batch_idx, batch in enumerate(loader):
            if benchmark_start is None:
                torch.cuda.reset_peak_memory_stats(device)
                benchmark_start = time.perf_counter()
            run_step = global_step - run_origin_step
            if args.smoke_steps > 0 and (directed_enabled or paired_hair_loss_enabled):
                batch = dict(batch)
                batch_size = int(batch['target_latents'].shape[0])
                batch['tau_override'] = torch.full((batch_size,), 0.5, dtype=torch.float32)
            decode_cfg = differential_cfg.get('decode', {})
            decode_freq = max(1, int(decode_cfg.get('freq', 3)))
            optimizer_boundary = micro_step % accum == accum - 1
            directed_decode_trigger = bool(
                directed_enabled
                and run_step % decode_freq == 0
                and optimizer_boundary
            )
            hair_decode_freq = max(1, int(hair_loss_cfg.get('decode_freq', 3)))
            directed_hair_decode_trigger = bool(
                directed_enabled
                and hair_loss_enabled
                and run_step % hair_decode_freq == 0
                and optimizer_boundary
            )
            paired_hair_decode_trigger = bool(
                paired_hair_loss_enabled
                and run_step % hair_decode_freq == 0
                and optimizer_boundary
            )
            decode_trigger = (
                directed_decode_trigger
                or directed_hair_decode_trigger
                or paired_hair_decode_trigger
            )
            identity_loss_accum_scale = (
                float(accum)
                if directed_enabled
                and differential_cfg.get('identity_loss', {}).get('compensate_grad_accum', True)
                else 1.0
            )
            hair_loss_accum_scale = (
                float(accum)
                if paired_hair_loss_enabled
                and hair_loss_cfg.get('compensate_grad_accum', True)
                else 1.0
            )
            loss, metrics = ddp(
                batch,
                train_step=run_step,
                decode_trigger=decode_trigger,
                identity_loss_accum_scale=identity_loss_accum_scale,
                hair_loss_accum_scale=hair_loss_accum_scale,
                optimizer_boundary=optimizer_boundary,
            )
            accumulate_scalar_metrics(metric_totals, metric_counts, metrics)
            if differential_enabled and 'delta_arc_jk' in batch:
                sampling_distances.extend(batch['delta_arc_jk'].detach().float().cpu().reshape(-1).tolist())
                sampling_relaxations.extend(
                    batch.get('cf_sampling_relaxation', torch.zeros_like(batch['delta_arc_jk']))
                    .detach().long().cpu().reshape(-1).tolist()
                )
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
            launch_completed_steps = global_step - start_step
            core_model = unwrap_model(ddp)
            if (
                isinstance(core_model, DifferentialFlowModel)
                and core_model.hinge_g is None
                and (
                    launch_completed_steps
                    if differential_cfg.get('calibration_from_launch', False)
                    else completed_run_steps
                ) >= calibration_steps
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
            sparse_sums = (
                reduce_sparse_metric_sums(metric_totals, device)
                if directed_enabled or paired_hair_loss_enabled
                else {
                    'loss_id_dir': 0.0,
                    'loss_id_abs': 0.0,
                    'sim_gap': 0.0,
                    'id_decode_seconds': 0.0,
                    'id_decode_branch': 0.0,
                    'id_loss_attempt_count': 0.0,
                    'id_loss_skip_count': 0.0,
                    'id_loss_triggered': 0.0,
                    'loss_hair_dino': 0.0,
                    'loss_hair': 0.0,
                    'hair_cosine': 0.0,
                    'loss_hair_sum': 0.0,
                    'hair_cosine_sum': 0.0,
                    'hair_valid_count': 0.0,
                    'hair_decode_seconds': 0.0,
                    'hair_loss_triggered': 0.0,
                    'hair_loss_attempt_count': 0.0,
                    'hair_loss_skip_count': 0.0,
                    'hair_schedule_sample_count': 0.0,
                    'hair_schedule_hit_count': 0.0,
                    'hair_skip_decode_freq_count': 0.0,
                    'hair_skip_tau_window_count': 0.0,
                    'hair_skip_area_count': 0.0,
                    'hair_skip_target_invalid_count': 0.0,
                    'hair_skip_parsing_failure_count': 0.0,
                }
            )
            id_attempt_sum = sparse_sums['id_loss_attempt_count']
            id_skip_sum = sparse_sums['id_loss_skip_count']
            id_trigger_sum = sparse_sums['id_loss_triggered']
            hair_attempt_sum = sparse_sums['hair_loss_attempt_count']
            hair_skip_sum = sparse_sums['hair_loss_skip_count']
            hair_valid_sum = sparse_sums['hair_valid_count']
            hair_trigger_sum = sparse_sums['hair_loss_triggered']
            hair_schedule_sum = sparse_sums['hair_schedule_sample_count']
            hair_schedule_hit_sum = sparse_sums['hair_schedule_hit_count']
            step_metrics = averaged_metrics(metric_totals, metric_counts)
            if id_trigger_sum > 0.0:
                for name in (
                    'loss_id_dir',
                    'loss_id_abs',
                    'sim_gap',
                    'id_decode_seconds',
                    'id_decode_branch',
                ):
                    step_metrics[name] = sparse_sums[name] / id_trigger_sum
                step_metrics['id_loss_triggered'] = id_trigger_sum
                step_metrics['id_loss_attempt_count'] = id_attempt_sum
                step_metrics['id_loss_skip_count'] = id_skip_sum
            if hair_trigger_sum > 0.0:
                step_metrics['hair_decode_seconds'] = (
                    sparse_sums['hair_decode_seconds'] / hair_trigger_sum
                )
                step_metrics['hair_loss_triggered'] = hair_trigger_sum
            if hair_valid_sum > 0.0:
                step_metrics['loss_hair'] = (
                    sparse_sums['loss_hair_sum'] / hair_valid_sum
                )
                step_metrics['loss_hair_dino'] = step_metrics['loss_hair']
                step_metrics['hair_cosine'] = (
                    sparse_sums['hair_cosine_sum'] / hair_valid_sum
                )
                step_metrics['hair_valid_count'] = hair_valid_sum
            if id_attempt_sum > 0.0:
                step_metrics['id_loss_skip_rate'] = id_skip_sum / id_attempt_sum
            if hair_attempt_sum > 0.0:
                step_metrics['hair_loss_attempt_count'] = hair_attempt_sum
                step_metrics['hair_loss_skip_count'] = hair_skip_sum
                step_metrics['hair_loss_skip_rate'] = hair_skip_sum / hair_attempt_sum
            for name in (
                'hair_schedule_sample_count',
                'hair_schedule_hit_count',
                'hair_skip_decode_freq_count',
                'hair_skip_tau_window_count',
                'hair_skip_area_count',
                'hair_skip_target_invalid_count',
                'hair_skip_parsing_failure_count',
            ):
                step_metrics[name] = sparse_sums[name]
            if hair_schedule_sum > 0.0:
                step_metrics['hair_schedule_hit_rate'] = (
                    hair_schedule_hit_sum / hair_schedule_sum
                )
                step_metrics['hair_skip_decode_freq_rate'] = (
                    sparse_sums['hair_skip_decode_freq_count']
                    / hair_schedule_sum
                )
                step_metrics['hair_skip_tau_window_rate'] = (
                    sparse_sums['hair_skip_tau_window_count']
                    / hair_schedule_sum
                )
            if hair_attempt_sum > 0.0:
                for reason in ('area', 'target_invalid', 'parsing_failure'):
                    step_metrics[f'hair_skip_{reason}_rate'] = (
                        sparse_sums[f'hair_skip_{reason}_count']
                        / hair_attempt_sum
                    )
            hair_metric_window.append({
                'attempts': hair_attempt_sum,
                'skips': hair_skip_sum,
                'valid': hair_valid_sum,
                'cosine_sum': sparse_sums['hair_cosine_sum'],
                'schedule_samples': hair_schedule_sum,
                'schedule_hits': hair_schedule_hit_sum,
                'decode_freq': sparse_sums['hair_skip_decode_freq_count'],
                'tau_window': sparse_sums['hair_skip_tau_window_count'],
                'area': sparse_sums['hair_skip_area_count'],
                'target_invalid': sparse_sums['hair_skip_target_invalid_count'],
                'parsing_failure': sparse_sums['hair_skip_parsing_failure_count'],
            })
            rolling_attempts = sum(row['attempts'] for row in hair_metric_window)
            rolling_valid = sum(row['valid'] for row in hair_metric_window)
            if rolling_attempts > 0.0:
                step_metrics['hair_skip_rate_rolling'] = (
                    sum(row['skips'] for row in hair_metric_window)
                    / rolling_attempts
                )
            if rolling_valid > 0.0:
                step_metrics['hair_cosine_rolling'] = (
                    sum(row['cosine_sum'] for row in hair_metric_window)
                    / rolling_valid
                )
            rolling_schedule = sum(
                row['schedule_samples'] for row in hair_metric_window
            )
            if rolling_schedule > 0.0:
                for reason in ('decode_freq', 'tau_window'):
                    step_metrics[f'hair_skip_{reason}_rate_rolling'] = (
                        sum(row[reason] for row in hair_metric_window)
                        / rolling_schedule
                    )
            if rolling_attempts > 0.0:
                for reason in ('area', 'target_invalid', 'parsing_failure'):
                    step_metrics[f'hair_skip_{reason}_rate_rolling'] = (
                        sum(row[reason] for row in hair_metric_window)
                        / rolling_attempts
                    )
            metric_totals.clear()
            metric_counts.clear()
            if differential_enabled and completed_run_steps % sampling_log_every == 0:
                sampling_stats_for_log = summarize_sampling_window(
                    sampling_distances,
                    sampling_relaxations,
                )
                sampling_distances.clear()
                sampling_relaxations.clear()
            if rank == 0 and (
                global_step % int(cfg['training']['log_every']) == 0
                or completed_run_steps <= 3
                or id_trigger_sum > 0.0
                or hair_trigger_sum > 0.0
                or hair_schedule_sum > 0.0
            ):
                log_dir.mkdir(parents=True, exist_ok=True)
                row = {
                    'step': global_step,
                    'run_step': completed_run_steps,
                    'lr': cfg['_runtime']['effective_lr'],
                    'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
                    'sample_ids': list(batch.get('sample_id', [])),
                    'cf_j_ids': list(batch.get('cf_j_id', [])),
                    'cf_k_ids': list(batch.get('cf_k_id', [])),
                }
                for name in (
                    'loss_total', 'loss_pair', 'loss_teach', 'loss_inv', 'loss_hinge',
                    'hinge_active_rate', 'face_diff_norm', 'diff_active_ratio',
                    'hinge_calibrating', 'hinge_g', 'head_pose_null_ratio',
                    'head_control_synthetic_ratio', 'garment_reference_tokens',
                    'hair_reference_tokens',
                    'tau_mean', 'z1_mean', 'controlnet_forward_count',
                    'transformer_forward_count', 'appearance_gate', 'garment_gate',
                    'hair_gate', 'head_pose_gate', 'loss_id_dir', 'loss_id_abs', 'sim_gap',
                    'loss_pair_unweighted', 'mse_cloth_safe', 'mse_hair', 'mse_face',
                    'mse_other', 'pair_weight_mean', 'pair_weight_max',
                    'id_loss_attempt_count', 'id_loss_skip_count', 'id_loss_skip_rate',
                    'id_loss_triggered', 'id_decode_seconds', 'id_decode_branch',
                    'loss_hair', 'loss_hair_dino', 'hair_cosine',
                    'hair_decode_seconds', 'hair_loss_triggered',
                    'hair_valid_count',
                    'hair_loss_attempt_count', 'hair_loss_skip_count',
                    'hair_loss_skip_rate', 'hair_skip_rate_rolling',
                    'hair_cosine_rolling',
                    'hair_schedule_sample_count', 'hair_schedule_hit_count',
                    'hair_schedule_hit_rate',
                    'hair_skip_decode_freq_count', 'hair_skip_tau_window_count',
                    'hair_skip_area_count', 'hair_skip_target_invalid_count',
                    'hair_skip_parsing_failure_count',
                    'hair_skip_decode_freq_rate', 'hair_skip_tau_window_rate',
                    'hair_skip_area_rate', 'hair_skip_target_invalid_rate',
                    'hair_skip_parsing_failure_rate',
                    'hair_skip_decode_freq_rate_rolling',
                    'hair_skip_tau_window_rate_rolling',
                    'hair_skip_area_rate_rolling',
                    'hair_skip_target_invalid_rate_rolling',
                    'hair_skip_parsing_failure_rate_rolling',
                ):
                    if name in step_metrics:
                        row[name] = step_metrics[name]
                if isinstance(core_model, DifferentialFlowModel) and core_model.hinge_g is not None:
                    row['hinge_g'] = float(core_model.hinge_g)
                if sampling_stats_for_log is not None:
                    sampling_row = {'step': global_step, 'run_step': completed_run_steps, **sampling_stats_for_log}
                    with (log_dir / 'sampling_d_jk.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(sampling_row, ensure_ascii=False) + '\n')
                    row.update({key: value for key, value in sampling_stats_for_log.items() if key != 'distances'})
                    sampling_stats_for_log = None
                with (log_dir / 'train.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(f'[rank0] {row}', flush=True)
            if not benchmark_done and launch_completed_steps >= int(cfg['training']['benchmark_steps']):
                elapsed = time.perf_counter() - benchmark_start
                imgs = int(cfg['_runtime']['global_batch']) * int(cfg['training']['benchmark_steps'])
                per_step = elapsed / int(cfg['training']['benchmark_steps'])
                bench = {
                    'steps': int(cfg['training']['benchmark_steps']),
                    'seconds': elapsed,
                    'seconds_per_optimizer_step': per_step,
                    'img_per_sec': imgs / elapsed,
                    'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
                    'estimated_continuation_hours': per_step * (target_step - start_step) / 3600,
                    'target_step': target_step,
                }
                if rank == 0:
                    log_dir.mkdir(parents=True, exist_ok=True)
                    (log_dir / 'benchmark.json').write_text(json.dumps(bench, indent=2), encoding='utf-8')
                    print(f'[rank0] benchmark={bench}', flush=True)
                benchmark_done = True
            checkpoint_every = int(cfg['training']['checkpoint_every'])
            if rank == 0 and global_step % checkpoint_every == 0:
                save_checkpoint(ckpt_dir / f'step-{global_step:06d}', core_model, optimizer, global_step, sampler, cfg)
            manual_review = review_status(cfg, output, completed_run_steps)
            if manual_review['due'] and not manual_review['approved']:
                if rank == 0:
                    review_checkpoint = ckpt_dir / f'step-{global_step:06d}'
                    if global_step % checkpoint_every != 0:
                        save_checkpoint(
                            review_checkpoint,
                            core_model,
                            optimizer,
                            global_step,
                            sampler,
                            cfg,
                        )
                    payload = {
                        'reason': 'manual_review_pending',
                        'checkpoint': str(review_checkpoint),
                        'warnings': [
                            'STOP-TRAINING: step-500 spatial-condition human review is pending'
                        ],
                        'manual_review': manual_review,
                    }
                    temporary = stop_marker.with_suffix('.tmp')
                    temporary.write_text(
                        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                        encoding='utf-8',
                    )
                    temporary.replace(stop_marker)
                    print(
                        f'[rank0] pausing for mandatory human review: {manual_review}',
                        flush=True,
                    )
            if sync_stop_requested(stop_marker, rank, device):
                stopped_by_watcher = True
                if rank == 0:
                    print(f'[rank0] watcher requested stop at step={global_step}: {stop_marker}', flush=True)
                break
            if global_step >= target_step:
                break
    if rank == 0:
        stop_reason = None
        if stopped_by_watcher and stop_marker.exists():
            try:
                stop_reason = json.loads(stop_marker.read_text(encoding='utf-8')).get('reason')
            except (OSError, json.JSONDecodeError):
                stop_reason = 'unknown'
        manual_pause = stop_reason == 'manual_review_pending'
        if cfg['training'].get('save_final', True) and not manual_pause:
            save_checkpoint(
                ckpt_dir / 'final',
                unwrap_model(ddp),
                optimizer,
                global_step,
                sampler,
                cfg,
            )
        (output / 'training_status.json').write_text(json.dumps({
            'status': (
                'paused_for_manual_review'
                if manual_pause
                else 'stopped_by_watcher' if stopped_by_watcher else 'complete'
            ),
            'step': global_step,
            'run_step': global_step - run_origin_step,
            'target_steps': target_step,
            'stop_marker': str(stop_marker) if stopped_by_watcher else None,
            'stop_reason': stop_reason,
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
