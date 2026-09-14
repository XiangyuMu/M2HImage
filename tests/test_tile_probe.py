from __future__ import annotations

import numpy as np
import torch

from spatial_conditions import add_control_samples, image_mask_to_token_weights, mask_control_samples
from tile_multicontrol_probe import scale_tag, select_probe_subset


def test_image_mask_to_token_weights_matches_flux_grid() -> None:
    mask = np.zeros((1024, 768), dtype=np.float32)
    mask[:16, :16] = 1.0
    weights = image_mask_to_token_weights(mask, 768, 1024)
    assert weights.shape == (1, 3072, 1)
    assert float(weights[0, 0, 0]) == 1.0
    assert float(weights.sum()) == 1.0


def test_control_samples_are_masked_before_addition() -> None:
    pose = [torch.ones(1, 4, 2)]
    tile = [torch.full((1, 4, 2), 2.0)]
    mask = torch.tensor([[[1.0], [0.5], [0.0], [0.0]]])
    merged = add_control_samples(pose, mask_control_samples(tile, mask))
    assert merged is not None
    expected = torch.tensor([[[3.0, 3.0], [2.0, 2.0], [1.0, 1.0], [1.0, 1.0]]])
    torch.testing.assert_close(merged[0], expected)


def test_probe_subset_is_deterministic_and_seed_zero() -> None:
    subset = {
        "seed": 7,
        "mannequins": ["m1", "m2", "m3"],
        "garment_types": {"m1": "top", "m2": "dress", "m3": "pants"},
        "pairs": [
            {"mannequin_id": mid, "identity_id": f"j{index}", "seeds": [0, 1]}
            for mid in ("m1", "m2", "m3") for index in range(3)
        ],
    }
    result = select_probe_subset(subset, mid_count=2, identities_per_mid=2, seeds=[0])
    assert result["mannequins"] == ["m1", "m2"]
    assert [(row["mannequin_id"], row["identity_id"]) for row in result["pairs"]] == [
        ("m1", "j0"), ("m1", "j1"), ("m2", "j0"), ("m2", "j1")
    ]
    assert all(row["seeds"] == [0] for row in result["pairs"])
    assert scale_tag(0.6) == "tile_0p6"
