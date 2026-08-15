from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from analyze_id_velocity import find_tau_bin, low_rank_analysis, select_stratified_pairs
from conditions import load_yaml
from sampling_protected import (
    build_protect_mask,
    mode_settings,
    normalized_region_partition,
    protected_velocity,
    subset_pairs_for_ablation,
    tau_is_protected,
    weak_settings,
)


def test_soft_region_partition_sums_to_one() -> None:
    masks = {
        'face_z': np.asarray([0.7, 0.0, 0.2], dtype=np.float32),
        'cloth_safe_z': np.asarray([0.5, 0.8, 0.0], dtype=np.float32),
        'body_bg_z': np.asarray([0.0, 0.5, 0.2], dtype=np.float32),
    }
    partition = normalized_region_partition(masks)
    assert np.allclose(sum(partition.values()), 1.0)
    assert all(np.all(value >= 0.0) for value in partition.values())


def test_token_dilation_and_protected_formula() -> None:
    values = np.zeros(16, dtype=np.float32)
    values[5] = 0.75
    masks = {'cloth_safe_z': values, 'body_bg_z': np.zeros_like(values)}
    mask = build_protect_mask(
        masks,
        'cloth_safe',
        1,
        64,
        64,
        device=torch.device('cpu'),
        dtype=torch.float32,
    )
    assert mask.shape == (1, 16, 1)
    assert int((mask > 0).sum()) == 9
    v_on = torch.full((1, 16, 2), 4.0)
    v_off = torch.full((1, 16, 2), 2.0)
    output = protected_velocity(v_on, v_off, mask)
    assert torch.allclose(output[:, 5], torch.full((1, 2), 2.5))
    assert torch.allclose(output[:, 15], torch.full((1, 2), 4.0))


def test_tau_window_and_weak_modes() -> None:
    assert tau_is_protected(0.5, [0.2, 0.8])
    assert not tau_is_protected(0.9, [0.2, 0.8])
    assert weak_settings('off') == (0.0, 0.0)
    assert weak_settings('scale03') == (0.3, 0.3)
    bins = [[0.8, 1.000001], [0.5, 0.8], [0.2, 0.5], [0.0, 0.2]]
    assert find_tau_bin(0.8, bins) == 0
    assert find_tau_bin(0.5, bins) == 1
    assert find_tau_bin(0.2, bins) == 2


def test_protection_yaml_preserves_off_as_a_string() -> None:
    cfg = load_yaml('configs/a4_protected.yaml')
    assert cfg['inference_protection']['weak_mode'] == 'off'
    assert cfg['inference_protection']['ablations']['tau_window']['co_primary'] is True


def test_tau_window_is_a_full_subset_co_primary(tmp_path) -> None:
    cfg = load_yaml('configs/a4_protected.yaml')
    analysis_dir = tmp_path / cfg['inference_protection']['analysis']['output_dir']
    analysis_dir.mkdir(parents=True)
    (analysis_dir / 'analysis_summary.json').write_text(
        '{"recommended_protect_tau_range": [0.0, 0.2]}\n', encoding='utf-8'
    )
    settings = mode_settings(cfg, 'tau_window', tmp_path)
    assert settings['protect_tau_range'] == [0.0, 0.2]
    assert settings['is_ablation'] is False


def make_subset(mid_count: int = 50) -> dict:
    garment_names = ('dress', 'pants', 'skirt', 'top')
    mids = [f'{index:05d}' for index in range(mid_count)]
    pairs = []
    garment_types = {}
    for mid_index, mid in enumerate(mids):
        garment_type = garment_names[mid_index % len(garment_names)]
        garment_types[mid] = garment_type
        for identity_index in range(4):
            pairs.append({
                'mannequin_id': mid,
                'identity_id': f'{10000 + mid_index * 4 + identity_index:05d}',
                'garment_type': garment_type,
                'seeds': [0, 1],
            })
    return {
        'mannequins': mids,
        'pairs': pairs,
        'garment_types': garment_types,
        'identity_pool': [],
    }


def test_ablation_subset_is_exactly_100_images() -> None:
    subset = subset_pairs_for_ablation(make_subset(), mid_count=25, identities_per_mid=2)
    assert len(subset['pairs']) == 50
    assert subset['protected_ablation_protocol']['images'] == 100
    assert sum(len(row['seeds']) for row in subset['pairs']) == 100


def test_analysis_selection_is_stratified_and_deterministic() -> None:
    subset = make_subset()
    first = select_stratified_pairs(subset, 20)
    second = select_stratified_pairs(subset, 20)
    assert first == second
    counts = {name: 0 for name in ('dress', 'pants', 'skirt', 'top')}
    for row in first:
        counts[row['garment_type']] += 1
        assert row['jid'] != row['kid']
    assert counts == {'dress': 5, 'pants': 5, 'skirt': 5, 'top': 5}


def test_analysis_and_sampling_do_not_import_heldout_recognizer() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ('analyze_id_velocity.py', 'sampling_protected.py'):
        source = (root / name).read_text(encoding='utf-8').lower()
        assert 'metrics.heldout_id' not in source
        assert 'adaface' not in source


def test_low_rank_analysis_writes_spectra(tmp_path) -> None:
    generator = torch.Generator().manual_seed(7)
    payloads = []
    for sample_index in range(3):
        v_j = torch.randn(4, 3, 2, generator=generator)
        payloads.append({
            'metadata': {'mid': str(sample_index)},
            'tau': torch.tensor([0.9, 0.6, 0.3, 0.1]),
            'v_j': v_j,
            'v_on': v_j,
            'v_k': torch.randn(4, 3, 2, generator=generator),
            'v_off': torch.randn(4, 3, 2, generator=generator),
        })
    bins = [[0.8, 1.000001], [0.5, 0.8], [0.2, 0.5], [0.0, 0.2]]
    rows = low_rank_analysis(payloads, bins, 3, 1, torch.device('cpu'), tmp_path)
    assert rows
    assert (tmp_path / 'delta_v_jk_singular_spectrum.png').exists()
    assert (tmp_path / 'delta_v_io_singular_spectrum.png').exists()
