from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class FakeDirectedBase(nn.Module):
    def __init__(self, transformer, controlnet, adapter, pulid, vae, recognizer, detector, cfg):
        super().__init__()
        self.transformer = transformer
        self.controlnet = controlnet
        self.adapter = adapter
        self.pulid = pulid
        self.cfg = cfg
        self.diff_cfg = cfg["training"]["differential"]
        self.width = 16
        self.height = 16
        self.hinge_g = 1.0
        self.decode_cfg = self.diff_cfg["decode"]
        self.identity_cfg = self.diff_cfg["identity_loss"]

    @staticmethod
    def _masked_l1_per_sample(diff, mask):
        return (diff.abs() * mask.unsqueeze(-1)).mean(dim=(1, 2))

    @staticmethod
    def _active_mean(values, active):
        return (values * active.float()).sum() / active.float().sum().clamp_min(1.0)

    def _prepare_flow(self, batch):
        z0 = batch["target_latents"]
        z1 = batch["noise_override"]
        tau = batch["tau_override"]
        prompt = batch["prompt_embeds"]
        pooled = batch["pooled_prompt_embeds"]
        return {
            "z0": z0,
            "z1": z1,
            "z_tau": (1 - tau.view(-1, 1, 1)) * z0 + tau.view(-1, 1, 1) * z1,
            "target_v": z1 - z0,
            "tau": tau,
            "prompt": prompt,
            "pooled": pooled,
            "img_ids": torch.zeros(1),
            "device": z0.device,
            "dtype": z0.dtype,
        }

    def _hair_inputs(self, batch, device, dtype, prefix=""):
        return None, None, None

    def _condition_tokens(self, prompt, appearance, garment, head_pose, *args):
        return torch.cat([prompt, appearance[:, :1].reshape(appearance.shape[0], 1, 1).expand(-1, 1, prompt.shape[-1])], dim=1)

    def _controlnet_forward(self, z_tau, tau, prompt, pooled, pose_latents, img_ids):
        self.controlnet.calls.append(z_tau.detach().clone())
        return ([torch.zeros_like(z_tau)], [torch.zeros_like(z_tau)])

    def _transformer_forward(self, z_tau, tau, cond_tokens, pulid_embed, cn_samples, **kwargs):
        identity = pulid_embed.mean(dim=tuple(range(1, pulid_embed.ndim)))
        return self.transformer(z_tau, identity)

    def _paired_flow_loss(self, pred, target, batch):
        return (pred - target).square().mean(), {"loss_pair_unweighted": (pred - target).square().mean().detach()}

    def _base_metrics(self, batch, flow):
        return {"tau_mean": flow["tau"].mean().detach()}

    def _identity_metric_defaults(self, device):
        zero = torch.zeros((), device=device)
        return {"loss_id_dir": zero, "loss_id_abs": zero, "id_loss_attempt_count": zero, "id_loss_skip_count": zero}

    def _extra_differential_loss(self, batch, flow, z_hat_j, z_hat_k, active, train_step, decode_trigger, identity_loss_accum_scale):
        if not decode_trigger:
            return z_hat_j.sum() * 0.0, {}
        loss = (z_hat_j.mean() - batch["cf_j_train_embed"].mean()).square()
        return loss * identity_loss_accum_scale, {"loss_id_abs": loss.detach(), "id_loss_attempt_count": torch.ones(())}

    def _decode_schedule_metrics(self, flow, train_step, optimizer_boundary):
        return {}


class FakeControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []


class FakeAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, z_tau, identity):
        return z_tau * self.scale + identity.reshape(-1, 1, 1)


def install_fake_train_paired():
    fake = SimpleNamespace(DirectedDifferentialFlowModel=FakeDirectedBase)
    sys.modules["train_paired"] = fake


def config():
    return {
        "training": {
            "differential": {
                "lambda_teach": 0.5,
                "lambda_inv": 0.2,
                "lambda_hinge": 0.05,
                "tau_min": 0.2,
                "tau_max": 0.8,
                "diff_every": 1,
                "decode": {"enabled": True, "freq": 3},
                "identity_loss": {"enabled": True},
            }
        }
    }


def batch():
    return {
        "target_latents": torch.zeros(1, 2, 1),
        "cf_j_state_latents": torch.ones(1, 2, 1) * 10,
        "cf_k_state_latents": torch.ones(1, 2, 1) * 20,
        "noise_override": torch.ones(1, 2, 1) * 2,
        "tau_override": torch.tensor([0.5]),
        "pose_latents": torch.zeros(1, 2, 1),
        "prompt_embeds": torch.zeros(1, 1, 1),
        "pooled_prompt_embeds": torch.zeros(1, 1),
        "appearance": torch.ones(1, 1),
        "garment": torch.ones(1, 1),
        "head_pose": torch.ones(1, 1),
        "pulid_id_embed": torch.ones(1, 1, 1),
        "cf_j_pulid_id_embed": torch.ones(1, 1, 1) * 3,
        "cf_k_pulid_id_embed": torch.ones(1, 1, 1) * 4,
        "cf_j_appearance": torch.ones(1, 1) * 3,
        "cf_k_appearance": torch.ones(1, 1) * 4,
        "cf_j_train_embed": torch.ones(1, 1),
        "cf_k_train_embed": torch.ones(1, 1) * 2,
        "cloth_safe_z": torch.ones(1, 2),
        "body_bg_z": torch.ones(1, 2),
        "face_z": torch.ones(1, 2),
    }


def make_model(arm):
    install_fake_train_paired()
    module = load_module("state_pilot_model")
    return module.StatePilotDirectedFlowModel(
        FakeTransformer(),
        FakeControlNet(),
        FakeAdapter(),
        object(),
        object(),
        object(),
        object(),
        config(),
        pilot_arm=arm,
    )


def test_arm_state_mapping_and_multiset() -> None:
    install_fake_train_paired()
    module = load_module("state_pilot_model")
    assert module.cf_state_keys_for_arm("C-H").j == "target_latents"
    assert module.cf_state_keys_for_arm("E-match").j == "cf_j_state_latents"
    assert module.cf_state_keys_for_arm("C-perm").j == "cf_k_state_latents"
    assert sorted(module.cf_state_keys_for_arm("C-perm").__dict__.values()) == sorted(module.cf_state_keys_for_arm("E-match").__dict__.values())


def test_cf_controlnet_recomputed_for_each_branch_state() -> None:
    model = make_model("E-match")
    loss, metrics = model(batch(), train_step=0)
    assert float(metrics["controlnet_forward_count"]) == 3.0
    assert len(model.controlnet.calls) == 3
    assert torch.allclose(model.controlnet.calls[0], torch.ones(1, 2, 1))
    assert torch.allclose(model.controlnet.calls[1], torch.ones(1, 2, 1) * 6)
    assert torch.allclose(model.controlnet.calls[2], torch.ones(1, 2, 1) * 11)
    assert float(metrics["loss_hinge"]) == 0.0
    loss.backward()
    assert model.transformer.scale.grad is not None
    assert torch.isfinite(model.transformer.scale.grad)


def test_paired_fm_uses_original_h_not_teacher_state() -> None:
    base = batch()
    model = make_model("E-match")
    loss_a, metrics_a = model(base, train_step=0)
    changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in base.items()}
    changed["cf_j_state_latents"] = torch.ones(1, 2, 1) * 1000
    changed["cf_k_state_latents"] = torch.ones(1, 2, 1) * 2000
    _loss_b, metrics_b = model(changed, train_step=0)
    assert torch.allclose(metrics_a["loss_pair_unweighted"], metrics_b["loss_pair_unweighted"])
    assert float(loss_a) > 0.0


def test_teacher_endpoint_states_are_stop_gradient() -> None:
    base = batch()
    base["cf_j_state_latents"] = base["cf_j_state_latents"].clone().requires_grad_(True)
    base["cf_k_state_latents"] = base["cf_k_state_latents"].clone().requires_grad_(True)
    model = make_model("E-match")
    loss, _metrics = model(base, train_step=0, decode_trigger=True)
    loss.backward()
    assert base["cf_j_state_latents"].grad is None
    assert base["cf_k_state_latents"].grad is None
    assert model.transformer.scale.grad is not None


def test_deterministic_index_resume_no_sampler_drift() -> None:
    sys.modules.pop("train_paired", None)
    module = load_module("train_state_pilot")
    first = [
        module.deterministic_pool_index(step, micro, rank=2, world_size=3, pool_size=7)
        for step in range(20)
        for micro in range(6)
    ]
    resumed = [
        module.deterministic_pool_index(step, micro, rank=2, world_size=3, pool_size=7)
        for step in range(20)
        for micro in range(6)
    ]
    assert first == resumed
    assert first[20 * 6 - 1] == module.deterministic_pool_index(19, 5, 2, 3, 7)


def write_tiny_cache(tmp_path: Path, with_endpoints: bool = True) -> None:
    for name in ("M", "I", "H", "endpoints"):
        (tmp_path / name).mkdir()
    np.savez(
        tmp_path / "prompt.npz",
        prompt_embeds=np.zeros((1, 1), np.float32),
        pooled_prompt_embeds=np.zeros((1,), np.float32),
    )
    np.savez(
        tmp_path / "M/m.npz",
        pose_latents=np.zeros((2, 1), np.float32),
        garment_grid=np.zeros((1,), np.float32),
        head_pose=np.zeros((1,), np.float32),
    )
    for sid, value in (("m", 0.0), ("j", 1.0), ("k", 0.0)):
        embed = np.zeros((2,), np.float32)
        embed[int(value)] = 1.0
        np.savez(
            tmp_path / f"I/{sid}.npz",
            pulid_id_embed=np.ones((1, 1), np.float32),
            appearance=np.ones((1,), np.float32),
            train_embed=embed,
        )
    np.savez(
        tmp_path / "H/m.npz",
        target_latents=np.zeros((2, 1), np.float32),
        cloth_safe_z=np.ones((2,), np.float32),
        body_bg_z=np.ones((2,), np.float32),
        face_z=np.ones((2,), np.float32),
    )
    if with_endpoints:
        np.savez(
            tmp_path / "endpoints/m__j.npz",
            target_latents=np.ones((2, 1), np.float32),
        )
        np.savez(
            tmp_path / "endpoints/m__k.npz",
            target_latents=np.ones((2, 1), np.float32) * 2,
        )


def test_read_pool_accepts_actual_object_rows(tmp_path: Path) -> None:
    sys.modules.pop("train_paired", None)
    module = load_module("train_state_pilot")
    path = tmp_path / "pool.json"
    path.write_text(
        '{"schema":1,"rows":[{"mid":"m","jid":"j","kid":"k","source_group":"g"}]}',
        encoding="utf-8",
    )
    rows = module.read_pool(path)
    assert rows == [module.PilotRow("m", "j", "k", "g")]


def test_pilot_dataset_cache_contract(tmp_path: Path) -> None:
    sys.modules.pop("train_paired", None)
    module = load_module("train_state_pilot")
    module.LATENT_SHAPE = (2, 1)
    module.MASK_SHAPE = (2,)
    write_tiny_cache(tmp_path)
    ds = module.StatePilotDataset([module.PilotRow("m", "j", "k", "g")], tmp_path, "E-match")
    item = ds[0]
    assert item["sample_id"] == "m"
    assert torch.allclose(item["cf_j_state_latents"], torch.ones(2, 1))
    assert torch.allclose(item["cf_k_state_latents"], torch.ones(2, 1) * 2)


def test_c_h_does_not_require_teacher_endpoints(tmp_path: Path) -> None:
    sys.modules.pop("train_paired", None)
    module = load_module("train_state_pilot")
    module.LATENT_SHAPE = (2, 1)
    module.MASK_SHAPE = (2,)
    write_tiny_cache(tmp_path, with_endpoints=False)
    ds = module.StatePilotDataset([module.PilotRow("m", "j", "k", "g")], tmp_path, "C-H")
    item = ds[0]
    assert torch.allclose(item["cf_j_state_latents"], item["target_latents"])
    assert torch.allclose(item["cf_k_state_latents"], item["target_latents"])


def test_default_tau_is_original_zero_to_one_range() -> None:
    sys.modules.pop("train_paired", None)
    module = load_module("train_state_pilot")
    batch_in = {"target_latents": torch.zeros(128, 2, 1)}
    module.inject_deterministic_noise(batch_in, seed=123)
    assert float(batch_in["tau_override"].min()) >= 0.0
    assert float(batch_in["tau_override"].max()) <= 1.0
    assert float(batch_in["tau_override"].min()) < 0.2
    assert float(batch_in["tau_override"].max()) > 0.8
