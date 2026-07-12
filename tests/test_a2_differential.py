from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from build_region_masks_z import pool_token_mask
from dataset import IdentityBank
from train_paired import DifferentialFlowModel


class FakeAdapter(nn.Module):
    def forward(self, appearance, garment, head_pose):
        value = appearance[:, :1].reshape(appearance.shape[0], 1, 1)
        return value.expand(appearance.shape[0], 1, 4)

    def gate_values(self):
        return {
            'appearance_gate': 0.1,
            'garment_gate': 0.1,
            'head_pose_gate': 0.1,
        }


class FakeControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        hidden = kwargs['hidden_states']
        zeros = torch.zeros_like(hidden)
        return SimpleNamespace(
            controlnet_block_samples=[zeros],
            controlnet_single_block_samples=[zeros],
        )


class FakePuLID:
    ID_KEY = '_m2h_pulid_id_embedding'
    WEIGHT_KEY = '_m2h_pulid_id_weight'

    def __init__(self):
        self.context = None
        self.weight = 1.0

    def set_context(self, embedding, weight):
        self.context = embedding
        self.weight = float(weight)

    def context_kwargs(self):
        return {self.ID_KEY: self.context, self.WEIGHT_KEY: self.weight}

    def clear_context(self):
        self.context = None


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.context_means: list[float] = []
        self.appearance_means: list[float] = []

    def forward(self, hidden_states, encoder_hidden_states, joint_attention_kwargs, **kwargs):
        identity = joint_attention_kwargs[FakePuLID.ID_KEY]
        identity_mean = identity.mean(dim=(1, 2))
        appearance_mean = encoder_hidden_states[:, -1].mean(dim=-1)
        self.context_means.append(float(identity_mean.detach().cpu()[0]))
        self.appearance_means.append(float(appearance_mean.detach().cpu()[0]))
        value = (identity_mean + appearance_mean).view(-1, 1, 1)
        sample = torch.ones_like(hidden_states) * value * self.scale
        return SimpleNamespace(sample=sample)


def config() -> dict:
    return {
        'data': {'resolution': {'width': 16, 'height': 16}},
        'model': {
            'control_mode': 4,
            'controlnet_scale': 0.75,
            'pulid': {'id_weight': 1.0},
        },
        'training': {
            'differential': {
                'enabled': True,
                'lambda_teach': 0.5,
                'lambda_inv': 0.2,
                'lambda_hinge': 0.05,
                'tau_min': 0.2,
                'tau_max': 0.8,
                'diff_every': 1,
                'hinge_g_resolved': 10.0,
            }
        },
        'eval': {'guidance_scale': 3.5},
    }


def batch(tau: float = 0.5) -> dict[str, torch.Tensor]:
    return {
        'target_latents': torch.zeros(1, 1, 4),
        'pose_latents': torch.zeros(1, 1, 4),
        'prompt_embeds': torch.zeros(1, 2, 4),
        'pooled_prompt_embeds': torch.zeros(1, 4),
        'pulid_id_embed': torch.ones(1, 1, 1),
        'appearance': torch.ones(1, 2),
        'garment': torch.zeros(1, 1, 2),
        'head_pose': torch.zeros(1, 2),
        'head_pose_is_null': torch.zeros(1),
        'cf_j_pulid_id_embed': torch.full((1, 1, 1), 2.0),
        'cf_k_pulid_id_embed': torch.full((1, 1, 1), 4.0),
        'cf_j_appearance': torch.full((1, 2), 3.0),
        'cf_k_appearance': torch.full((1, 2), 5.0),
        'delta_arc_jk': torch.full((1,), 0.5),
        'cloth_safe_z': torch.ones(1, 1),
        'body_bg_z': torch.ones(1, 1),
        'face_z': torch.ones(1, 1),
        'tau_override': torch.full((1,), tau),
    }


def test_differential_step_reuses_controlnet_and_swaps_both_identity_routes() -> None:
    transformer = FakeTransformer()
    controlnet = FakeControlNet()
    model = DifferentialFlowModel(transformer, controlnet, FakeAdapter(), FakePuLID(), config())
    loss, metrics = model(batch(), train_step=0)
    assert controlnet.calls == 1
    assert transformer.context_means == [1.0, 2.0, 4.0]
    assert transformer.appearance_means == [1.0, 3.0, 5.0]
    assert float(metrics['transformer_forward_count']) == 3.0
    assert float(metrics['loss_teach']) > 0.0
    assert float(metrics['loss_inv']) > 0.0
    assert float(metrics['loss_hinge']) > 0.0
    loss.backward()
    assert transformer.scale.grad is not None
    assert torch.isfinite(transformer.scale.grad)


def test_tau_outside_window_is_paired_only() -> None:
    transformer = FakeTransformer()
    controlnet = FakeControlNet()
    model = DifferentialFlowModel(transformer, controlnet, FakeAdapter(), FakePuLID(), config())
    _loss, metrics = model(batch(tau=0.9), train_step=0)
    assert controlnet.calls == 1
    assert len(transformer.context_means) == 1
    assert float(metrics['transformer_forward_count']) == 1.0
    assert float(metrics['loss_teach']) == 0.0
    assert float(metrics['loss_inv']) == 0.0
    assert float(metrics['loss_hinge']) == 0.0


def test_region_mask_pooling_matches_packed_token_grid() -> None:
    mask = np.zeros((32, 16), dtype=np.uint8)
    mask[:16] = 255
    pooled, resized = pool_token_mask(mask, width=16, height=32)
    assert not resized
    np.testing.assert_array_equal(pooled, np.asarray([1.0, 0.0], dtype=np.float16))


def test_identity_bank_sampling_is_compatible_and_deterministic(tmp_path) -> None:
    ids = np.asarray(['a', 'b', 'c', 'd', 'e'])
    embeds = np.zeros((5, 512), dtype=np.float32)
    embeds[np.arange(5), np.arange(5)] = 1.0
    path = tmp_path / 'identity_bank.npz'
    np.savez(
        path,
        ids=ids,
        embeds=embeds,
        gender=np.asarray(['male', 'male', 'male', 'female', 'male']),
        age=np.asarray([30, 35, 45, 30, 50], dtype=np.float32),
        age_group=np.asarray(['adult'] * 5),
        skin_cluster=np.asarray([2, 3, 1, 2, 2], dtype=np.int16),
    )
    bank = IdentityBank(path)
    assert bank.validate_sources(['a']) == {
        'source_count': 1,
        'minimum_compatible_identities': 2,
    }
    first = bank.sample_pair('a', seed=123)
    second = bank.sample_pair('a', seed=123)
    assert first == second
    selected = {str(bank.ids[first[0]]), str(bank.ids[first[1]])}
    assert selected == {'b', 'c'}
    assert first[2] == 1.0
