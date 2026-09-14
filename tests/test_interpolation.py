from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from interp_common import (
    alpha_label,
    assert_trainable_compatible,
    farthest_pair,
    identity_image_name,
    interpolate_identity_batch,
    interpolate_trainable_state,
    slerp_tokens,
    stratified_mids,
)
from metrics.identity_interp_metrics import normalized_garment_drift


def fake_state(offset: float) -> dict[str, dict[str, torch.Tensor]]:
    return {
        'transformer_lora': {
            'layer.lora_A.weight': torch.tensor([[1.0, 2.0]]) + offset,
            'layer.lora_B.weight': torch.tensor([[3.0], [4.0]]) + offset,
        },
        'adapter': {
            'appearance_gate': torch.tensor(0.1 + offset),
            'projection.weight': torch.tensor([[5.0, 6.0]]) + offset,
        },
    }


def test_trainable_state_interpolation_and_schema_failure() -> None:
    start, end = fake_state(0.0), fake_state(2.0)
    summary = assert_trainable_compatible(start, end)
    assert summary['transformer_lora']['keys'] == 2
    midpoint = interpolate_trainable_state(start, end, 0.25)
    assert torch.allclose(midpoint['adapter']['projection.weight'], torch.tensor([[5.5, 6.5]]))
    broken = fake_state(2.0)
    broken['adapter'].pop('appearance_gate')
    with pytest.raises(RuntimeError, match='only_b2'):
        assert_trainable_compatible(start, broken)


def test_slerp_endpoints_norm_and_batch_condition_invariance() -> None:
    start = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    end = torch.tensor([[0.0, 4.0], [-5.0, 0.0]])
    mid, diagnostics = slerp_tokens(start, end, 0.5)
    assert diagnostics['token_count'] == 2
    assert torch.allclose(torch.linalg.vector_norm(mid, dim=-1), torch.tensor([3.0, 4.0]), atol=1e-5)
    assert torch.equal(slerp_tokens(start, end, 0.0)[0], start)
    assert torch.equal(slerp_tokens(start, end, 1.0)[0], end)

    invariant = torch.tensor([7.0])
    batch_j = {'pulid_id_embed': start, 'appearance': torch.tensor([1.0, 2.0]), 'garment': invariant}
    batch_k = {'pulid_id_embed': end, 'appearance': torch.tensor([3.0, 4.0]), 'garment': torch.tensor([9.0])}
    batch, _ = interpolate_identity_batch(batch_j, batch_k, 0.5)
    assert batch['garment'] is invariant
    assert torch.allclose(batch['appearance'], torch.tensor([2.0, 3.0]))


def test_labels_and_farthest_pair_are_deterministic() -> None:
    assert alpha_label(0.25) == 'alpha025'
    assert identity_image_name('m', 'j', 'k', 0.75, 0) == 'm__jj__kk__t075__seed0.png'
    embeddings = {
        'a': np.asarray([1.0, 0.0], dtype=np.float32),
        'b': np.asarray([0.0, 1.0], dtype=np.float32),
        'c': np.asarray([-1.0, 0.0], dtype=np.float32),
    }
    assert farthest_pair(['c', 'a', 'b'], embeddings) == ('a', 'c', 2.0)


def test_stratified_mid_selection() -> None:
    names = ('dress', 'pants', 'skirt', 'top')
    mids = [f'{index:02d}' for index in range(20)]
    subset = {
        'mannequins': mids,
        'garment_types': {mid: names[index % 4] for index, mid in enumerate(mids)},
        'pairs': [
            {'mannequin_id': mid, 'identity_id': f'i{identity}', 'garment_type': names[index % 4]}
            for index, mid in enumerate(mids)
            for identity in range(4)
        ],
    }
    selected = stratified_mids(subset, 12)
    counts = {name: sum(subset['garment_types'][mid] == name for mid in selected) for name in names}
    assert counts == {name: 3 for name in names}


def test_generation_modules_do_not_import_heldout_recognizer() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ('interp_common.py', 'interp_weights.py', 'interp_identity.py'):
        source = (root / name).read_text(encoding='utf-8').lower()
        assert 'metrics.heldout_id' not in source
        assert 'unifaceadafacerecognizer' not in source


def test_normalized_garment_drift_uses_only_sim_to_j_endpoint_swing() -> None:
    max_drift = 0.06
    sim_j_swing = 0.20
    sim_k_swing = 0.80
    result = normalized_garment_drift(max_drift, sim_j_swing)
    assert result == pytest.approx(0.30)
    assert result != pytest.approx(max_drift / ((sim_j_swing + sim_k_swing) / 2.0))

