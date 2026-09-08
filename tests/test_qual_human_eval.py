from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from human_eval_build import assign_questions, build_normal_questions, spread_questions
from human_eval_score import (
    attention_qc,
    krippendorff_alpha_ordinal,
    noninferiority_test,
)
from qual_eval_common import load_config, select_panel_mids


def synthetic_selection_inputs():
    garment_names = ('dress', 'pants', 'skirt', 'top')
    mids = [f'{index:02d}' for index in range(20)]
    garment_types = {
        mid: garment_names[index // 5]
        for index, mid in enumerate(mids)
    }
    subset = {
        'mannequins': mids,
        'garment_types': garment_types,
        'pairs': [
            {'mannequin_id': mid, 'identity_id': 'identity'}
            for mid in mids
        ],
    }
    b2_garment = {mid: 0.90 for mid in mids}
    a4_garment = {mid: 0.90 - index / 1000 for index, mid in enumerate(mids)}
    b2_delta = {(mid, 'identity', 0): {'delta_id': 0.0} for mid in mids}
    a4_delta = {
        (mid, 'identity', 0): {'delta_id': index / 1000}
        for index, mid in enumerate(mids)
    }
    runs = {
        'b2cont': {'garment': b2_garment, 'delta': b2_delta},
        'a4': {'garment': a4_garment, 'delta': a4_delta},
    }
    return subset, runs


def test_config_extends_without_training_imports() -> None:
    cfg = load_config('configs/qualitative.yaml')
    assert cfg['qualitative_eval']['human_eval']['rater_count'] == 8
    assert cfg['data']['resolution'] == {'width': 768, 'height': 1024}


def test_panel_selection_is_unique_and_retains_global_extremes() -> None:
    subset, runs = synthetic_selection_inputs()
    selected, tags, diagnostics = select_panel_mids(subset, runs, 16)
    assert len(selected) == len(set(selected)) == 16
    worst = {row['mid'] for row in diagnostics['global_worst_garment']}
    best = {row['mid'] for row in diagnostics['global_best_identity']}
    assert worst <= set(selected)
    assert best <= set(selected)
    assert all('worst-garment' in tags[mid] for mid in worst)
    assert all('best-identity' in tags[mid] for mid in best)


def fake_question_context():
    rows = []
    sources = {}
    for index in range(16):
        mid = f'm{index:02d}'
        jids = [f'{mid}_j{jid}' for jid in range(4)]
        rows.append({'mid': mid, 'identity_ids': jids})
        sources[mid] = {
            'mannequin': Path(f'/source/{mid}.png'),
            'faces': {jid: Path(f'/face/{jid}.png') for jid in jids},
            'generated': {
                (run, jid): Path(f'/gen/{run}/{mid}_{jid}.png')
                for run in ('b2cont', 'alpha075', 'a4')
                for jid in jids
            },
        }
    context = {
        'qcfg': {'seed_for_panels': 0, 'human_identity_positions': [0, 2]},
        'sources': sources,
    }
    return context, {'rows': rows}


def test_question_counts_assignment_and_no_adjacent_mid() -> None:
    context, manifest = fake_question_context()
    questions = build_normal_questions(context, manifest)
    assert len(questions) == 240
    assignments = assign_questions(questions, rater_count=8, ratings_per_question=3, seed=123)
    counts = {qid: 0 for qid in [row['question_id'] for row in questions]}
    mapping = {row['question_id']: row for row in questions}
    for rater_index, qids in enumerate(assignments.values()):
        assert len(qids) == 90
        ordered = spread_questions(qids, mapping, seed=1000 + rater_index)
        assert all(
            mapping[left]['mid'] != mapping[right]['mid']
            for left, right in zip(ordered, ordered[1:])
        )
        for qid in qids:
            counts[qid] += 1
    assert set(counts.values()) == {3}


def test_ordinal_alpha_and_noninferiority() -> None:
    perfect = (iter(values) for values in ([1, 1, 1], [3, 3, 3], [5, 5, 5]))
    assert krippendorff_alpha_ordinal(perfect) == pytest.approx(1.0)
    disagreement = krippendorff_alpha_ordinal([[1, 5, 1], [5, 1, 5], [2, 4, 3]])
    assert disagreement is not None and disagreement < 0.5
    reference = np.full(16, 4.0)
    assert noninferiority_test(np.full(16, 3.9), reference, 0.3, 0.05)['pass']
    assert not noninferiority_test(np.full(16, 3.4), reference, 0.3, 0.05)['pass']


def test_attention_failure_excludes_rater() -> None:
    key = {
        'protocol': {'max_attention_failures': 0},
        'assignments': {'rater_01': ['q1', 'q2']},
        'questions': {
            'q1': {'attention': True, 'expected': {'field': 'identity_match', 'operator': '>=', 'threshold': 4}},
            'q2': {'attention': True, 'expected': {'field': 'identity_match', 'operator': '<=', 'threshold': 2}},
        },
    }
    responses = [
        {'rater_id': 'rater_01', 'question_id': 'q1', 'scores': {'identity_match': 5}},
        {'rater_id': 'rater_01', 'question_id': 'q2', 'scores': {'identity_match': 4}},
    ]
    qc, included = attention_qc(responses, key)
    assert not included
    assert qc[0]['returned'] is True
    assert qc[0]['attention_failed'] == 1
