from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from train_paired import DirectedDifferentialFlowModel


PilotArm = Literal["C-H", "C-perm", "E-match"]


@dataclass(frozen=True)
class ArmStateKeys:
    j: str
    k: str


def cf_state_keys_for_arm(arm: PilotArm) -> ArmStateKeys:
    """Map a pilot arm to the CF endpoint states consumed by the model."""
    if arm == "C-H":
        return ArmStateKeys("target_latents", "target_latents")
    if arm == "C-perm":
        return ArmStateKeys("cf_k_state_latents", "cf_j_state_latents")
    if arm == "E-match":
        return ArmStateKeys("cf_j_state_latents", "cf_k_state_latents")
    raise ValueError(f"unknown pilot arm {arm!r}; expected C-H, C-perm, or E-match")


class StatePilotDirectedFlowModel(DirectedDifferentialFlowModel):
    """Directed A4 loss on branch-specific CF states for the M2H state pilot.

    The paired FM term is intentionally unchanged: it uses original paired H
    state and paired I_mid identity. Only the CF branch states vary by arm.
    """

    def __init__(self, *args: Any, pilot_arm: PilotArm, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pilot_arm = pilot_arm
        self.cf_state_keys = cf_state_keys_for_arm(pilot_arm)
        self.hinge_g = None
        self.diff_cfg["lambda_hinge"] = 0.0
        self.diff_cfg["hinge_g_resolved"] = None

    def load_differential_state(self, state: dict[str, Any]) -> None:
        if state.get("run_origin_step") is not None:
            self.run_origin_step = int(state["run_origin_step"])
        self.hinge_g = None
        self.diff_cfg["lambda_hinge"] = 0.0
        self.diff_cfg["hinge_g_resolved"] = None

    def differential_state_dict(self) -> dict[str, Any]:
        return {
            "hinge_g": None,
            "run_origin_step": self.run_origin_step,
            "state_pilot_arm": self.pilot_arm,
        }

    def _flow_from_state(
        self,
        batch: dict[str, torch.Tensor],
        z0: torch.Tensor,
        z1: torch.Tensor,
        tau: torch.Tensor,
        prompt: torch.Tensor,
        pooled: torch.Tensor,
        img_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_tau = (1.0 - tau.view(-1, 1, 1)) * z0 + tau.view(-1, 1, 1) * z1
        return {
            "z0": z0,
            "z1": z1,
            "z_tau": z_tau,
            "target_v": z1 - z0,
            "tau": tau,
            "prompt": prompt,
            "pooled": pooled,
            "img_ids": img_ids,
            "device": z0.device,
            "dtype": z0.dtype,
        }

    @staticmethod
    def _require_batch_keys(batch: dict[str, torch.Tensor], keys: tuple[str, ...]) -> None:
        missing = [key for key in keys if key not in batch]
        if missing:
            raise RuntimeError(f"state pilot batch is missing required keys {missing}")

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
        self._require_batch_keys(
            batch,
            (
                "target_latents",
                "cf_j_state_latents",
                "cf_k_state_latents",
                "pose_latents",
                "prompt_embeds",
                "pooled_prompt_embeds",
                "appearance",
                "cf_j_appearance",
                "cf_k_appearance",
                "pulid_id_embed",
                "cf_j_pulid_id_embed",
                "cf_k_pulid_id_embed",
                "cloth_safe_z",
                "body_bg_z",
                "face_z",
            ),
        )
        train_step = int(train_step or 0)
        paired_flow = self._prepare_flow(batch)
        dtype, device = paired_flow["dtype"], paired_flow["device"]
        garment = batch["garment"].to(device=device, dtype=dtype)
        head_pose = batch["head_pose"].to(device=device, dtype=dtype)

        paired_hair = self._hair_inputs(batch, device, dtype)
        paired_tokens = self._condition_tokens(
            paired_flow["prompt"],
            batch["appearance"].to(device=device, dtype=dtype),
            garment,
            head_pose,
            *paired_hair,
        )
        paired_cn = self._controlnet_forward(
            paired_flow["z_tau"],
            paired_flow["tau"],
            paired_flow["prompt"],
            paired_flow["pooled"],
            batch["pose_latents"],
            paired_flow["img_ids"],
        )
        pred_i = self._transformer_forward(
            paired_flow["z_tau"],
            paired_flow["tau"],
            paired_tokens,
            batch["pulid_id_embed"],
            paired_cn,
            pooled=paired_flow["pooled"],
            img_ids=paired_flow["img_ids"],
            garment_ref_latents=batch.get("garment_ref_latents"),
            hair_ref_latents=batch.get("hair_ref_latents"),
        )
        loss_pair, pair_metrics = self._paired_flow_loss(
            pred_i, paired_flow["target_v"], batch
        )

        zero = torch.zeros((), device=device, dtype=torch.float32)
        metrics = self._base_metrics(batch, paired_flow)
        tau_min = float(self.diff_cfg.get("tau_min", 0.2))
        tau_max = float(self.diff_cfg.get("tau_max", 0.8))
        diff_every = max(1, int(self.diff_cfg.get("diff_every", 1)))
        scheduled = train_step % diff_every == 0
        active = (
            (paired_flow["tau"].float() >= tau_min)
            & (paired_flow["tau"].float() <= tau_max)
        )
        if not scheduled:
            active = torch.zeros_like(active)

        loss_teach = zero
        loss_inv = zero
        face_diff_mean = zero
        extra_loss = zero
        extra_metrics = self._identity_metric_defaults(device)
        transformer_count = 1.0
        controlnet_count = 1.0
        if bool(active.any()):
            z0_j = batch[self.cf_state_keys.j].to(device=device, dtype=dtype).detach()
            z0_k = batch[self.cf_state_keys.k].to(device=device, dtype=dtype).detach()
            flow_j = self._flow_from_state(
                batch,
                z0_j,
                paired_flow["z1"],
                paired_flow["tau"],
                paired_flow["prompt"],
                paired_flow["pooled"],
                paired_flow["img_ids"],
            )
            flow_k = self._flow_from_state(
                batch,
                z0_k,
                paired_flow["z1"],
                paired_flow["tau"],
                paired_flow["prompt"],
                paired_flow["pooled"],
                paired_flow["img_ids"],
            )
            hair_j = self._hair_inputs(batch, device, dtype, prefix="cf_j_")
            hair_k = self._hair_inputs(batch, device, dtype, prefix="cf_k_")
            tokens_j = self._condition_tokens(
                paired_flow["prompt"],
                batch["cf_j_appearance"].to(device=device, dtype=dtype),
                garment,
                head_pose,
                *hair_j,
            )
            tokens_k = self._condition_tokens(
                paired_flow["prompt"],
                batch["cf_k_appearance"].to(device=device, dtype=dtype),
                garment,
                head_pose,
                *hair_k,
            )
            cn_j = self._controlnet_forward(
                flow_j["z_tau"],
                flow_j["tau"],
                flow_j["prompt"],
                flow_j["pooled"],
                batch["pose_latents"],
                flow_j["img_ids"],
            )
            cn_k = self._controlnet_forward(
                flow_k["z_tau"],
                flow_k["tau"],
                flow_k["prompt"],
                flow_k["pooled"],
                batch["pose_latents"],
                flow_k["img_ids"],
            )
            pred_j = self._transformer_forward(
                flow_j["z_tau"],
                flow_j["tau"],
                tokens_j,
                batch["cf_j_pulid_id_embed"],
                cn_j,
                pooled=flow_j["pooled"],
                img_ids=flow_j["img_ids"],
                garment_ref_latents=batch.get("garment_ref_latents"),
                hair_ref_latents=batch.get("cf_j_hair_ref_latents"),
            )
            pred_k = self._transformer_forward(
                flow_k["z_tau"],
                flow_k["tau"],
                tokens_k,
                batch["cf_k_pulid_id_embed"],
                cn_k,
                pooled=flow_k["pooled"],
                img_ids=flow_k["img_ids"],
                garment_ref_latents=batch.get("garment_ref_latents"),
                hair_ref_latents=batch.get("cf_k_hair_ref_latents"),
            )
            transformer_count = 3.0
            controlnet_count = 3.0
            tau_view = paired_flow["tau"].float().view(-1, 1, 1)
            z_hat_j = flow_j["z_tau"].float() - tau_view * pred_j.float()
            z_hat_k = flow_k["z_tau"].float() - tau_view * pred_k.float()
            teach_j = self._masked_l1_per_sample(
                z_hat_j - flow_j["z0"].float(), batch["cloth_safe_z"]
            )
            teach_k = self._masked_l1_per_sample(
                z_hat_k - flow_k["z0"].float(), batch["cloth_safe_z"]
            )
            teach_per = 0.5 * (teach_j + teach_k)
            inv_per = self._masked_l1_per_sample(
                z_hat_j - z_hat_k, batch["body_bg_z"]
            )
            face_per = self._masked_l1_per_sample(
                z_hat_j - z_hat_k, batch["face_z"]
            )
            loss_teach = self._active_mean(teach_per, active)
            loss_inv = self._active_mean(inv_per, active)
            face_diff_mean = self._active_mean(face_per, active)
            extra_loss, computed_extra_metrics = self._extra_differential_loss(
                batch,
                paired_flow,
                z_hat_j,
                z_hat_k,
                active,
                train_step,
                decode_trigger,
                identity_loss_accum_scale,
            )
            extra_metrics.update(computed_extra_metrics)

        lambda_teach = float(self.diff_cfg.get("lambda_teach", 0.5))
        lambda_inv = float(self.diff_cfg.get("lambda_inv", 0.2))
        total = loss_pair + lambda_teach * loss_teach + lambda_inv * loss_inv + extra_loss
        metrics.update(
            {
                "loss_total": total.detach(),
                "loss_pair": loss_pair.detach(),
                "loss_teach": loss_teach.detach(),
                "loss_inv": loss_inv.detach(),
                "loss_hinge": zero,
                "hinge_active_rate": zero,
                "face_diff_norm": face_diff_mean.detach(),
                "diff_active_ratio": active.float().mean().detach(),
                "hinge_calibrating": torch.ones((), device=device, dtype=torch.float32),
                "hinge_g": zero,
                "transformer_forward_count": torch.tensor(transformer_count, device=device),
                "controlnet_forward_count": torch.tensor(controlnet_count, device=device),
                "state_pilot_arm_index": torch.tensor(
                    {"C-H": 0.0, "C-perm": 1.0, "E-match": 2.0}[self.pilot_arm],
                    device=device,
                    dtype=torch.float32,
                ),
                "calibration_ratios": torch.empty((0,), device=device, dtype=torch.float32),
            }
        )
        metrics.update(pair_metrics)
        metrics.update(extra_metrics)
        metrics.update(
            self._decode_schedule_metrics(paired_flow, train_step, optimizer_boundary)
        )
        return total, metrics
