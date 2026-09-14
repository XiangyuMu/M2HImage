from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

from conditions import FluxConditionAdapter, make_image_ids, save_yaml
from eval_watcher import approve_manual_review
from dataset import sample_cache_keys
from manual_review_gate import review_status, write_approval
from spatial_conditions import (
    dino_hair_semantic_tokens,
    dino_hair_dense_tokens,
    downsample_reference_tokens,
    make_reference_image_ids,
    pad_control_samples_for_reference,
)
from synth_head_keypoints import synth_head_keypoints
from train_paired import DirectedDifferentialFlowModel, WarmupFlowModel


def test_reference_stride_ids_and_control_padding_stay_aligned() -> None:
    tokens = torch.arange(16, dtype=torch.float32).view(4, 4)
    sampled = downsample_reference_tokens(tokens, width=32, height=32, stride=2)
    assert sampled.shape == (1, 4)
    torch.testing.assert_close(sampled[0], tokens[0])

    ids = make_reference_image_ids(
        32, 32, x_offset=64, device=torch.device('cpu'), dtype=torch.float32, stride=2
    )
    assert ids.shape == (1, 3)
    torch.testing.assert_close(ids[0], torch.tensor([0.0, 0.0, 64.0]))

    residual = [torch.ones(1, 4, 8)]
    padded = pad_control_samples_for_reference(residual, reference_token_count=1)
    assert padded is not None
    assert padded[0].shape == (1, 5, 8)
    assert torch.all(padded[0][:, :4] == 1)
    assert torch.all(padded[0][:, 4:] == 0)


def test_hair_adapter_disables_legacy_garment_and_masks_padding() -> None:
    adapter = FluxConditionAdapter({
        'appearance_dim': 8,
        'garment_grid_dim': 8,
        'hair_ref_dim': 6,
        'hair_ref_max_tokens': 4,
        'token_dim': 16,
        'appearance_tokens': 2,
        'pose_tokens': 1,
        'gate_init': 0.1,
        'use_legacy_garment_tokens': False,
        'use_hair_tokens': True,
    })
    assert not adapter.garment_gate.requires_grad
    assert adapter.hair_gate.requires_grad
    output = adapter(
        torch.randn(1, 8),
        torch.randn(1, 4, 8),
        torch.randn(1, 7),
        hair_ref_tokens=torch.randn(1, 4, 6),
        hair_ref_positions=torch.randn(1, 4, 2),
        hair_ref_mask=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
    )
    assert output.shape == (1, 7, 16)
    assert torch.count_nonzero(output[:, 4:6]) == 0
    assert set(adapter.gate_values()) == {'appearance_gate', 'hair_gate', 'head_pose_gate'}


def test_dead_hair_route_keeps_resume_slots_but_appends_no_tokens() -> None:
    adapter = FluxConditionAdapter({
        'appearance_dim': 8,
        'garment_grid_dim': 8,
        'hair_ref_dim': 6,
        'hair_ref_max_tokens': 4,
        'token_dim': 16,
        'appearance_tokens': 2,
        'pose_tokens': 1,
        'gate_init': 0.1,
        'use_legacy_garment_tokens': False,
        'use_hair_tokens': False,
        'retain_legacy_hair_parameters_for_resume': True,
    })
    assert adapter.hair_gate.requires_grad
    assert adapter.hair_proj.weight.requires_grad
    assert adapter.token_count == 3
    output = adapter(
        torch.randn(1, 8),
        torch.randn(1, 4, 8),
        torch.randn(1, 7),
    )
    assert output.shape == (1, 3, 16)
    output.sum().backward()
    # Retained slots stay in the DDP graph but have no effect on model output.
    assert adapter.hair_gate.grad is not None
    assert adapter.hair_proj.weight.grad is not None
    assert adapter.hair_proj.bias.grad is not None
    assert adapter.hair_pos_proj.weight.grad is not None
    assert set(adapter.gate_values()) == {
        'appearance_gate', 'hair_gate', 'head_pose_gate'
    }


class FakeAdapter(nn.Module):
    use_hair_tokens = False

    def forward(self, appearance, garment, head_pose):
        return torch.zeros(appearance.shape[0], 1, 4)

    def gate_values(self):
        return {'appearance_gate': 0.1, 'head_pose_gate': 0.1}


class FakeControlNet(nn.Module):
    def forward(self, **kwargs):
        value = torch.ones(
            kwargs['hidden_states'].shape[0],
            kwargs['hidden_states'].shape[1],
            8,
            device=kwargs['hidden_states'].device,
            dtype=kwargs['hidden_states'].dtype,
        )
        return SimpleNamespace(
            controlnet_block_samples=[value],
            controlnet_single_block_samples=[value.clone()],
        )


class FakePuLID:
    def set_context(self, embedding, weight):
        self.embedding = embedding

    def context_kwargs(self):
        return {'identity': self.embedding}

    def clear_context(self):
        self.embedding = None


class RecordingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.last = None

    def forward(self, **kwargs):
        self.last = kwargs
        return SimpleNamespace(sample=kwargs['hidden_states'] * self.weight)


def test_model_concatenates_reference_but_returns_only_image_tokens() -> None:
    cfg = {
        'data': {'resolution': {'width': 32, 'height': 32}},
        'model': {
            'control_mode': 4,
            'controlnet_scale': 0.75,
            'identity_adapter': {'use_hair_tokens': True},
            'pulid': {'id_weight': 1.0},
            'spatial_conditions': {
                'garment_reference': {'enabled': True, 'stride': 2, 'x_offset': 64},
                'hair': {
                    'enabled': True,
                    'reference': {'enabled': True, 'stride': 2, 'y_offset': 64},
                },
            },
        },
        'eval': {'guidance_scale': 3.5},
    }
    transformer = RecordingTransformer()
    model = WarmupFlowModel(transformer, FakeControlNet(), FakeAdapter(), FakePuLID(), cfg)
    image_tokens = torch.randn(1, 4, 64)
    reference = torch.randn(1, 4, 64)
    prompt = torch.zeros(1, 2, 4)
    pooled = torch.zeros(1, 4)
    img_ids = make_image_ids(32, 32, torch.device('cpu'), torch.float32)
    cn = model._controlnet_forward(
        image_tokens, torch.tensor([0.5]), prompt, pooled, image_tokens, img_ids
    )
    output = model._transformer_forward(
        image_tokens,
        torch.tensor([0.5]),
        prompt,
        torch.zeros(1, 1, 1),
        cn,
        pooled=pooled,
        img_ids=img_ids,
        garment_ref_latents=reference,
        hair_ref_latents=reference,
    )
    assert output.shape == image_tokens.shape
    assert transformer.last['hidden_states'].shape == (1, 6, 64)
    assert transformer.last['img_ids'].shape == (6, 3)
    block = transformer.last['controlnet_block_samples'][0]
    assert block.shape == (1, 6, 8)
    assert torch.all(block[:, :4] == 1)
    assert torch.all(block[:, 4:] == 0)


def test_spatial_cache_keys_are_config_gated() -> None:
    old = {'model': {}}
    assert 'garment_ref_latents' not in sample_cache_keys(old)
    spatial = {
        'model': {
            'identity_adapter': {
                'use_hair_tokens': False,
                'retain_legacy_hair_parameters_for_resume': True,
            },
            'spatial_conditions': {
                'garment_reference': {'enabled': True},
                'head_control': {'enabled': True},
                'hair': {
                    'enabled': True,
                    'reference': {'enabled': True},
                },
            },
        },
    }
    keys = sample_cache_keys(spatial)
    assert {
        'garment_ref_latents', 'pose_synth_latents', 'hair_ref_latents',
        'hair_ref_empty', *('hair_ref_tokens', 'hair_ref_positions', 'hair_ref_mask')
    } <= set(keys)


class NoFaceDetector:
    def detect_tensor_batch(self, decoded):
        return [None] * decoded.shape[0]


class FakeHairSupervisor(nn.Module):
    def forward(self, decoded, target_tokens, target_valid, projected_hair_mask=None):
        del target_tokens, target_valid, projected_hair_mask
        loss = decoded.square().mean()
        count = torch.tensor(float(decoded.shape[0]), device=decoded.device)
        return loss, {
            'loss_hair_dino': loss.detach(),
            'hair_loss_attempt_count': count,
            'hair_loss_skip_count': torch.zeros_like(count),
        }


class DirectedDecodeHarness(DirectedDifferentialFlowModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.identity_cfg = {'min_detection_confidence': 0.5}
        self.hair_loss_cfg = {'lambda': 0.25}
        self.hair_supervisor = FakeHairSupervisor()
        self.face_detector = NoFaceDetector()

    def _decode_tokens(self, tokens):
        return tokens


def test_hair_loss_survives_face_detection_failure() -> None:
    model = DirectedDecodeHarness()
    decoded = torch.randn(2, 3, 8, 8, requires_grad=True)
    loss, metrics = model._decode_identity_branch(
        decoded,
        torch.randn(2, 4),
        torch.randn(2, 4),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        torch.randn(2, 4, 6),
        torch.ones(2, 4),
        torch.ones(2, 3072),
    )
    torch.testing.assert_close(loss, decoded.square().mean() * 0.25)
    assert metrics['id_loss_skip_count'].item() == 2.0
    assert metrics['hair_loss_attempt_count'].item() == 2.0
    assert metrics['hair_loss_skip_count'].item() == 0.0
    loss.backward()
    assert decoded.grad is not None
    assert torch.count_nonzero(decoded.grad) > 0


def test_manual_review_gate_requires_matching_explicit_approval(tmp_path) -> None:
    cfg = {
        'experiment': {'id': 'spatial-test'},
        'eval': {
            'manual_review_step': 500,
            'watcher_manual_checklist': [
                'garment_same_item',
                'hair_matches_reference',
                'head_follows_mannequin',
            ],
        },
    }
    early = review_status(cfg, tmp_path, 499)
    assert early['enabled']
    assert not early['due']
    pending = review_status(cfg, tmp_path, 500)
    assert pending['due']
    assert not pending['approved']
    assert pending['reason'] == 'approval_missing'

    approval = write_approval(cfg, tmp_path, reviewer='unit-test')
    assert approval.exists()
    approved = review_status(cfg, tmp_path, 500)
    assert approved['approved']
    assert approved['reason'] == 'approved'

    wrong_cfg = {
        **cfg,
        'experiment': {'id': 'different-run'},
    }
    mismatch = review_status(wrong_cfg, tmp_path, 500)
    assert not mismatch['approved']
    assert mismatch['reason'] == 'experiment_id_mismatch'


class FakeDenseDino(nn.Module):
    def forward(self, value):
        return torch.ones(value.shape[0], 4, 6, device=value.device, dtype=value.dtype)


def test_no_hair_parsing_produces_explicit_null_condition() -> None:
    image = Image.new("RGB", (8, 4), (80, 90, 100))
    tokens, positions, valid = dino_hair_dense_tokens(
        FakeDenseDino(),
        image,
        np.zeros((4, 8), dtype=np.uint8),
        device=torch.device("cpu"),
        dtype=torch.float32,
        image_size=8,
        max_tokens=4,
    )
    assert tokens.shape == (4, 6)
    assert positions.shape == (4, 2)
    assert valid.shape == (4,)
    assert tokens.dtype == np.float16
    assert np.count_nonzero(tokens) == 0
    assert np.count_nonzero(positions) == 0
    assert np.count_nonzero(valid) == 0


def test_synthesized_head_is_anchored_at_face_height_above_neck() -> None:
    neck = np.asarray([100.0, 200.0], dtype=np.float32)
    scale = 50.0
    points = synth_head_keypoints(
        {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        neck,
        scale,
    )
    expected_nose_y = neck[1] - 1.62 * scale + 0.02 * scale
    assert np.isclose(points[0, 1], expected_nose_y)
    assert float(points[:, 1].max()) < neck[1] - scale


def test_manual_approval_cannot_bypass_pause_or_automatic_failure(tmp_path) -> None:
    output_root = tmp_path / "runs"
    cfg = {
        "experiment": {
            "id": "spatial-test",
            "output_root": str(output_root),
        },
        "eval": {
            "manual_review_step": 500,
            "watcher_manual_checklist": [
                "garment_same_item",
                "hair_matches_reference",
                "head_follows_mannequin",
            ],
        },
    }
    config_path = tmp_path / "config.yaml"
    save_yaml(config_path, cfg)
    experiment_dir = output_root / "spatial-test"
    approval_path = experiment_dir / "HUMAN_REVIEW_APPROVED.json"

    with pytest.raises(RuntimeError, match="before training reaches"):
        approve_manual_review(str(config_path), "reviewer")
    assert not approval_path.exists()

    experiment_dir.mkdir(parents=True)
    stop_marker = experiment_dir / "STOP_TRAINING"
    stop_marker.write_text(
        json.dumps({"reason": "watcher_checks_failed"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="automatic failures"):
        approve_manual_review(str(config_path), "reviewer")
    assert not approval_path.exists()
    assert stop_marker.exists()

    stop_marker.write_text(
        json.dumps({
            "reason": "manual_review_pending",
            "checkpoint": "step-000500",
            "manual_review": {
                "due": True,
                "completed_run_steps": 500,
            },
        }),
        encoding="utf-8",
    )
    result = approve_manual_review(str(config_path), "reviewer")
    assert result["cleared_manual_stop"]
    assert result["reviewed_checkpoint"] == "step-000500"
    assert approval_path.exists()
    assert not stop_marker.exists()
