from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from train_paired import DirectedDifferentialFlowModel
from train_recognizer import FaceGeometry


class FakeAdapter(nn.Module):
    def forward(self, appearance, garment, head_pose):
        return appearance[:, :1].reshape(appearance.shape[0], 1, 1).expand(-1, 1, 4)

    def gate_values(self):
        return {'appearance_gate': 0.1, 'garment_gate': 0.1, 'head_pose_gate': 0.1}


class FakeControlNet(nn.Module):
    def forward(self, **kwargs):
        zeros = torch.zeros_like(kwargs['hidden_states'])
        return SimpleNamespace(controlnet_block_samples=[zeros], controlnet_single_block_samples=[zeros])


class FakePuLID:
    def set_context(self, embedding, weight):
        self.embedding = embedding

    def context_kwargs(self):
        return {'identity': self.embedding}

    def clear_context(self):
        self.embedding = None


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden_states, encoder_hidden_states, joint_attention_kwargs, **kwargs):
        identity = joint_attention_kwargs['identity'].mean(dim=(1, 2))
        appearance = encoder_hidden_states[:, -1].mean(dim=1)
        value = (identity + appearance).reshape(-1, 1, 1)
        return SimpleNamespace(sample=torch.ones_like(hidden_states) * value * self.scale)


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter('dtype_anchor', nn.Parameter(torch.tensor(1.0), requires_grad=False))
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def decode(self, latents, return_dict=False):
        image = F.interpolate(latents[:, :1], size=(112, 112), mode='bilinear', align_corners=False)
        image = torch.cat([image, -image, image * 0.5], dim=1).tanh()
        return (image,)


class FakeRecognizer(nn.Module):
    input_size = 112

    def forward(self, images):
        means = images.mean(dim=(2, 3))
        features = torch.cat([means, torch.zeros(images.shape[0], 509, device=images.device)], dim=1)
        return F.normalize(features, dim=1)


class FakeDetector:
    TEMPLATE = np.asarray([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    def detect_tensor_batch(self, images):
        return [
            FaceGeometry(
                bbox=np.asarray([20.0, 20.0, 92.0, 105.0], dtype=np.float32),
                landmarks=self.TEMPLATE.copy(),
                confidence=0.99,
            )
            for _ in range(images.shape[0])
        ]


def config():
    return {
        'data': {'resolution': {'width': 16, 'height': 16}},
        'model': {'control_mode': 4, 'controlnet_scale': 0.75, 'pulid': {'id_weight': 1.0}},
        'training': {
            'differential': {
                'enabled': True,
                'lambda_teach': 0.5,
                'lambda_inv': 0.2,
                'lambda_hinge': 0.05,
                'tau_min': 0.2,
                'tau_max': 0.8,
                'diff_every': 1,
                'hinge_g_resolved': 1.0,
                'decode': {
                    'enabled': True,
                    'freq': 3,
                    'both': False,
                    'tau_min': 0.35,
                    'tau_max': 0.7,
                    'latent_scale': 1.0,
                    'gradient_checkpointing': True,
                },
                'identity_loss': {
                    'enabled': True,
                    'margin': 0.1,
                    'lambda_dir': 0.1,
                    'lambda_abs': 0.05,
                    'min_detection_confidence': 0.5,
                },
            },
        },
        'eval': {'guidance_scale': 3.5},
    }


def batch():
    ref_j = torch.zeros(1, 512)
    ref_k = torch.zeros(1, 512)
    ref_j[:, 0] = 1.0
    ref_k[:, 1] = 1.0
    ones = torch.ones(1, 1)
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
        'cf_j_train_embed': ref_j,
        'cf_k_train_embed': ref_k,
        'delta_arc_jk': torch.ones(1),
        'cloth_safe_z': ones,
        'body_bg_z': ones,
        'face_z': ones,
        'tau_override': torch.full((1,), 0.5),
    }


def test_directed_step_decodes_one_branch_and_backpropagates() -> None:
    transformer = FakeTransformer()
    model = DirectedDifferentialFlowModel(
        transformer,
        FakeControlNet(),
        FakeAdapter(),
        FakePuLID(),
        FakeVAE(),
        FakeRecognizer(),
        FakeDetector(),
        config(),
    )
    loss, metrics = model(batch(), train_step=0, decode_trigger=True, identity_loss_accum_scale=1.0)
    assert float(metrics['id_loss_triggered']) == 1.0
    assert float(metrics['id_loss_attempt_count']) == 1.0
    assert float(metrics['id_loss_skip_count']) == 0.0
    assert float(metrics['loss_id_dir']) > 0.0
    assert float(metrics['loss_id_abs']) > 0.0
    loss.backward()
    assert transformer.scale.grad is not None
    assert torch.isfinite(transformer.scale.grad)


def test_directed_step_alternates_to_k_branch() -> None:
    model = DirectedDifferentialFlowModel(
        FakeTransformer(),
        FakeControlNet(),
        FakeAdapter(),
        FakePuLID(),
        FakeVAE(),
        FakeRecognizer(),
        FakeDetector(),
        config(),
    )
    _loss, metrics = model(batch(), train_step=3, decode_trigger=True, identity_loss_accum_scale=1.0)
    assert float(metrics['id_loss_triggered']) == 1.0
    assert float(metrics['id_decode_branch']) == 1.0
