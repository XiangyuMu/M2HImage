from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn

from conditions import FluxConditionAdapter
from dataset import sample_cache_keys
from spatial_conditions import dino_hair_semantic_tokens


class TinyDino(nn.Module):
    def forward(self, value):
        batch = value.shape[0]
        return torch.arange(
            batch * 16 * 6, device=value.device, dtype=value.dtype
        ).view(batch, 16, 6)


def test_semantic_hair_tokens_crop_pad_and_cap_count() -> None:
    image = np.full((20, 40, 3), 127, dtype=np.uint8)
    image[2:18, 15:25] = np.asarray([20, 80, 160], dtype=np.uint8)
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[2:18, 15:25] = 1
    tokens, valid = dino_hair_semantic_tokens(
        TinyDino(),
        Image.fromarray(image, mode='RGB'),
        mask,
        device=torch.device('cpu'),
        dtype=torch.float32,
        image_size=56,
        max_tokens=4,
    )
    assert tokens.shape == (4, 6)
    assert valid.shape == (4,)
    assert 0 < int(valid.sum()) <= 4


def test_semantic_hair_adapter_uses_no_position_ids_but_keeps_ddp_slot() -> None:
    adapter = FluxConditionAdapter({
        'appearance_dim': 8,
        'garment_grid_dim': 8,
        'hair_ref_dim': 6,
        'hair_semantic_max_tokens': 4,
        'hair_token_mode': 'semantic_dense',
        'token_dim': 16,
        'appearance_tokens': 2,
        'pose_tokens': 1,
        'gate_init': 0.1,
        'use_legacy_garment_tokens': False,
        'use_hair_tokens': True,
        'retain_legacy_hair_parameters_for_resume': True,
    })
    output = adapter(
        torch.randn(1, 8),
        torch.randn(1, 4, 8),
        torch.randn(1, 7),
        hair_ref_tokens=torch.randn(1, 4, 6),
        hair_ref_positions=None,
        hair_ref_mask=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
    )
    assert output.shape == (1, 7, 16)
    assert torch.count_nonzero(output[:, 4:6]) == 0
    output.sum().backward()
    assert adapter.hair_pos_proj.weight.grad is not None
    assert torch.count_nonzero(adapter.hair_pos_proj.weight.grad) == 0


def test_semantic_hair_cache_keys_keep_loss_targets_separate() -> None:
    cfg = {
        'model': {
            'identity_adapter': {
                'use_hair_tokens': True,
                'hair_token_mode': 'semantic_dense',
            },
            'spatial_conditions': {'hair': {'reference': {'enabled': False}}},
        },
        'training': {'hair_loss': {'enabled': True}},
    }
    keys = set(sample_cache_keys(cfg))
    assert {'hair_semantic_tokens', 'hair_semantic_mask'} <= keys
    assert {'hair_ref_tokens', 'hair_ref_positions', 'hair_ref_mask'} <= keys
    assert 'hair_ref_latents' not in keys
