from __future__ import annotations

import numpy as np

from dataset import IdentityBank


def write_bank(path, ids, embeds, gender, age, skin):
    np.savez(
        path,
        ids=np.asarray(ids),
        embeds=np.asarray(embeds, dtype=np.float32),
        gender=np.asarray(gender),
        age=np.asarray(age, dtype=np.float32),
        age_group=np.asarray(['adult'] * len(ids)),
        skin_cluster=np.asarray(skin, dtype=np.int16),
    )


def unit_vectors(count: int) -> np.ndarray:
    values = np.zeros((count, 512), dtype=np.float32)
    values[np.arange(count), np.arange(count)] = 1.0
    return values


def test_semihard_is_deterministic_and_selects_maximum_distance(tmp_path) -> None:
    ids = ['anchor', 'a', 'b', 'c', 'd']
    embeds = np.zeros((len(ids), 512), dtype=np.float32)
    embeds[0, 2] = 1.0
    embeds[1, :2] = (1.0, 0.0)
    embeds[2, :2] = (-1.0, 0.0)
    embeds[3, :2] = (0.0, 1.0)
    embeds[4, :2] = (0.0, -1.0)
    path = tmp_path / 'bank.npz'
    write_bank(path, ids, embeds, ['m'] * 5, [30] * 5, [2] * 5)
    bank = IdentityBank(path)
    first = bank.sample_pair_details('anchor', seed=7, sampling='semihard', semihard_pool=4)
    second = bank.sample_pair_details('anchor', seed=7, sampling='semihard', semihard_pool=4)
    assert first == second
    assert first['relaxation'] == 0
    assert first['delta_arc'] == 2.0
    pair = {str(bank.ids[int(first['j_index'])]), str(bank.ids[int(first['k_index'])])}
    assert pair in ({'a', 'b'}, {'c', 'd'})


def test_semihard_relaxes_age_before_skin(tmp_path) -> None:
    ids = ['anchor', 'a', 'b', 'c']
    path = tmp_path / 'bank.npz'
    write_bank(path, ids, unit_vectors(4), ['m'] * 4, [20, 50, 55, 60], [2] * 4)
    bank = IdentityBank(path)
    row = bank.sample_pair_details('anchor', seed=3, sampling='semihard', semihard_pool=3)
    assert row['relaxation'] == 1
    assert row['relaxation_name'] == 'relax_age'
    assert row['candidate_count'] == 3


def test_semihard_never_relaxes_gender(tmp_path) -> None:
    ids = ['anchor', 'strict_one', 'far_skin_a', 'far_skin_b', 'female_a', 'female_b']
    path = tmp_path / 'bank.npz'
    write_bank(
        path,
        ids,
        unit_vectors(len(ids)),
        ['m', 'm', 'm', 'm', 'f', 'f'],
        [30, 31, 32, 33, 30, 30],
        [2, 2, 8, 9, 2, 2],
    )
    bank = IdentityBank(path)
    row = bank.sample_pair_details('anchor', seed=9, sampling='semihard', semihard_pool=3)
    assert row['relaxation'] == 2
    selected = {str(bank.ids[int(row['j_index'])]), str(bank.ids[int(row['k_index'])])}
    assert not selected.intersection({'female_a', 'female_b'})
