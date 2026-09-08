from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import wilcoxon

from qual_eval_common import load_config, read_json, resolve_path, write_csv, write_json


RUNS = ('b2cont', 'alpha075', 'a4')
FIELDS = ('garment_consistency', 'identity_match', 'garment_fidelity', 'realism')
FIELD_QUESTION = {
    'garment_consistency': 'q1',
    'identity_match': 'q2',
    'garment_fidelity': 'q3',
    'realism': 'q3',
}
FIELD_LABELS = {
    'garment_consistency': 'Q1 garment consistency',
    'identity_match': 'Q2 identity match',
    'garment_fidelity': 'Q3 garment fidelity',
    'realism': 'Q3 realism',
}


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 'N/A'
    return 'N/A' if not math.isfinite(number) else f'{number:.{digits}f}'


def read_response_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def score_value(row: dict[str, str], field: str, scale_min: int, scale_max: int) -> int:
    raw = row.get(field, '')
    if raw is None or str(raw).strip() == '':
        raise ValueError(f'missing score field {field}')
    value = int(float(raw))
    if not scale_min <= value <= scale_max:
        raise ValueError(f'{field}={value} outside [{scale_min},{scale_max}]')
    return value


def load_responses(
    paths: list[Path],
    key: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    questions = key['questions']
    assignments = {rater: set(values) for rater, values in key['assignments'].items()}
    scale_min, scale_max = [int(value) for value in key['protocol']['scale']]
    rows: list[dict[str, Any]] = []
    errors: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for line_number, row in enumerate(read_response_csv(path), start=2):
            rater = str(row.get('rater_id', '')).strip()
            qid = str(row.get('question_id', '')).strip()
            location = f'{path}:{line_number}'
            if rater not in assignments:
                errors[location].append(f'unknown rater_id={rater!r}')
                continue
            if qid not in questions:
                errors[location].append(f'unknown question_id={qid!r}')
                continue
            if qid not in assignments[rater]:
                errors[location].append(f'question {qid} was not assigned to {rater}')
                continue
            if (rater, qid) in seen:
                errors[location].append(f'duplicate response for {rater}/{qid}')
                continue
            seen.add((rater, qid))
            question = questions[qid]
            if row.get('question_type') != question['question_type']:
                errors[location].append(
                    f"question_type={row.get('question_type')!r}, expected {question['question_type']!r}"
                )
                continue
            scores = {}
            try:
                for field in question['score_fields']:
                    scores[field] = score_value(row, field, scale_min, scale_max)
            except (TypeError, ValueError) as exc:
                errors[location].append(str(exc))
                continue
            rows.append({
                'rater_id': rater,
                'question_id': qid,
                'question_type': question['question_type'],
                'scores': scores,
                'source_csv': str(path),
            })
    return rows, errors


def attention_qc(
    responses: list[dict[str, Any]],
    key: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    by_rater_qid = {(row['rater_id'], row['question_id']): row for row in responses}
    max_failures = int(key['protocol']['max_attention_failures'])
    qc_rows = []
    included = set()
    for rater, assigned in sorted(key['assignments'].items()):
        returned = any((rater, qid) in by_rater_qid for qid in assigned)
        checks = [qid for qid in assigned if key['questions'][qid]['attention']]
        failed = []
        missing = []
        for qid in checks:
            response = by_rater_qid.get((rater, qid))
            if response is None:
                missing.append(qid)
                failed.append(qid)
                continue
            expected = key['questions'][qid]['expected']
            value = response['scores'][expected['field']]
            threshold = int(expected['threshold'])
            passed = value >= threshold if expected['operator'] == '>=' else value <= threshold
            if not passed:
                failed.append(qid)
        keep = returned and len(failed) <= max_failures
        if keep:
            included.add(rater)
        qc_rows.append({
            'rater_id': rater,
            'attention_total': len(checks),
            'attention_failed': len(failed),
            'missing_checks': len(missing),
            'returned': returned,
            'included': keep,
            'failed_question_ids': ';'.join(failed),
        })
    return qc_rows, included


def response_coverage(
    responses: list[dict[str, Any]],
    key: dict[str, Any],
    included_raters: set[str],
) -> list[dict[str, Any]]:
    counts = Counter(
        row['question_id']
        for row in responses
        if row['rater_id'] in included_raters and not key['questions'][row['question_id']]['attention']
    )
    minimum = int(key['protocol']['min_retained_ratings'])
    missing = []
    for qid, question in key['questions'].items():
        if question['attention']:
            continue
        actual = int(counts.get(qid, 0))
        if actual < minimum:
            missing.append({
                'question_id': qid,
                'question_type': question['question_type'],
                'run': question['run'],
                'mid': question['mid'],
                'jid': question.get('jid'),
                'retained_ratings': actual,
                'required_ratings': minimum,
                'needed': minimum - actual,
            })
    return missing


def task_medians(
    responses: list[dict[str, Any]],
    key: dict[str, Any],
    included_raters: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in responses:
        if row['rater_id'] not in included_raters:
            continue
        if key['questions'][row['question_id']]['attention']:
            continue
        grouped[row['question_id']].append(row)
    output = []
    for qid, values in sorted(grouped.items()):
        question = key['questions'][qid]
        row: dict[str, Any] = {
            'question_id': qid,
            'question_type': question['question_type'],
            'run': question['run'],
            'mid': question['mid'],
            'jid': question.get('jid'),
            'rating_count': len(values),
        }
        for field in question['score_fields']:
            row[field] = float(np.median([value['scores'][field] for value in values]))
        output.append(row)
    return output


def bootstrap_mean_ci(values: Iterable[float], samples: int, seed: int) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {'count': 0, 'mean': None, 'ci95_low': None, 'ci95_high': None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(int(samples), len(array)))
    means = np.mean(array[indices], axis=1)
    return {
        'count': int(len(array)),
        'mean': float(np.mean(array)),
        'ci95_low': float(np.quantile(means, 0.025)),
        'ci95_high': float(np.quantile(means, 0.975)),
    }


def krippendorff_alpha_ordinal(units: Iterable[Iterable[int]], categories: tuple[int, ...] = (1, 2, 3, 4, 5)) -> float | None:
    rows = []
    for values in units:
        row = list(values)
        if len(row) >= 2:
            rows.append(row)
    if not rows:
        return None
    category_index = {value: index for index, value in enumerate(categories)}
    coincidence = np.zeros((len(categories), len(categories)), dtype=np.float64)
    for values in rows:
        counts = Counter(values)
        count = len(values)
        for left in categories:
            for right in categories:
                li, ri = category_index[left], category_index[right]
                if left == right:
                    coincidence[li, ri] += counts[left] * (counts[left] - 1) / (count - 1)
                else:
                    coincidence[li, ri] += counts[left] * counts[right] / (count - 1)
    marginals = coincidence.sum(axis=0)
    total = float(marginals.sum())
    if total <= 1.0:
        return None
    distances = np.zeros_like(coincidence)
    for left in range(len(categories)):
        for right in range(len(categories)):
            lo, hi = sorted((left, right))
            distance = marginals[lo:hi + 1].sum() - (marginals[lo] + marginals[hi]) / 2.0
            distances[left, right] = distance * distance
    observed = float(np.sum(coincidence * distances) / total)
    expected = np.outer(marginals, marginals) / (total - 1.0)
    np.fill_diagonal(expected, marginals * (marginals - 1.0) / (total - 1.0))
    expected_disagreement = float(np.sum(expected * distances) / total)
    if expected_disagreement <= 1e-12:
        return 1.0 if observed <= 1e-12 else None
    return float(1.0 - observed / expected_disagreement)


def reliability(
    responses: list[dict[str, Any]],
    key: dict[str, Any],
    included_raters: set[str],
) -> dict[str, dict[str, float | None]]:
    output = {}
    for field in FIELDS:
        question_type = FIELD_QUESTION[field]
        by_run: dict[str, dict[str, list[int]]] = {
            run: defaultdict(list) for run in RUNS
        }
        overall: dict[str, list[int]] = defaultdict(list)
        for row in responses:
            if row['rater_id'] not in included_raters or row['question_type'] != question_type:
                continue
            question = key['questions'][row['question_id']]
            if question['attention'] or field not in row['scores']:
                continue
            value = int(row['scores'][field])
            by_run[question['run']][row['question_id']].append(value)
            overall[row['question_id']].append(value)
        output[field] = {
            **{
                run: krippendorff_alpha_ordinal(values.values())
                for run, values in by_run.items()
            },
            'overall': krippendorff_alpha_ordinal(overall.values()),
        }
    return output


def per_mid_values(task_rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in task_rows:
        if field not in row:
            continue
        grouped[(str(row['run']), str(row['mid']))].append(float(row[field]))
    output: dict[str, dict[str, float]] = {run: {} for run in RUNS}
    for (run, mid), values in grouped.items():
        output[run][mid] = float(np.mean(values))
    return output


def paired_arrays(values: dict[str, dict[str, float]], left: str, right: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keys = sorted(set(values[left]) & set(values[right]))
    return (
        np.asarray([values[left][key] for key in keys], dtype=np.float64),
        np.asarray([values[right][key] for key in keys], dtype=np.float64),
        keys,
    )


def paired_test(left: np.ndarray, right: np.ndarray, alternative: str) -> dict[str, Any]:
    if not len(left) or len(left) != len(right):
        return {'count': 0, 'mean_delta': None, 'median_delta': None, 'p_value': None}
    difference = left - right
    if np.allclose(difference, 0.0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(left, right, alternative=alternative, zero_method='wilcox').pvalue)
    return {
        'count': int(len(left)),
        'mean_delta': float(np.mean(difference)),
        'median_delta': float(np.median(difference)),
        'p_value': p_value,
    }


def noninferiority_test(left: np.ndarray, right: np.ndarray, margin: float, alpha: float) -> dict[str, Any]:
    if not len(left) or len(left) != len(right):
        return {'count': 0, 'mean_delta': None, 'p_value': None, 'pass': False}
    difference = left - right
    shifted = difference + float(margin)
    p_value = 1.0 if np.allclose(shifted, 0.0) else float(
        wilcoxon(shifted, alternative='greater', zero_method='wilcox').pvalue
    )
    return {
        'count': int(len(left)),
        'mean_delta': float(np.mean(difference)),
        'margin': float(margin),
        'p_value': p_value,
        'pass': bool(float(np.mean(difference)) >= -float(margin) and p_value < float(alpha)),
    }


def summarize(
    task_rows: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    key: dict[str, Any],
    included_raters: set[str],
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    bootstrap_samples = int(eval_cfg['bootstrap_samples'])
    seed = int(key['protocol']['seed'])
    summaries = {}
    for field_index, field in enumerate(FIELDS):
        summaries[field] = {}
        for run_index, run in enumerate(RUNS):
            values = [float(row[field]) for row in task_rows if row['run'] == run and field in row]
            summaries[field][run] = bootstrap_mean_ci(
                values,
                bootstrap_samples,
                seed + field_index * 100 + run_index,
            )
    reliabilities = reliability(responses, key, included_raters)
    paired = {}
    per_mid = {}
    for field in FIELDS:
        values = per_mid_values(task_rows, field)
        per_mid[field] = values
        paired[field] = {}
        for left, right in (('a4', 'b2cont'), ('alpha075', 'b2cont'), ('alpha075', 'a4')):
            left_values, right_values, mids = paired_arrays(values, left, right)
            paired[field][f'{left}_vs_{right}'] = {
                **paired_test(left_values, right_values, 'two-sided'),
                'paired_mids': mids,
            }

    alpha = float(eval_cfg['significance_alpha'])
    visible_threshold = float(eval_cfg['visible_mean_threshold'])
    margin = float(eval_cfg['noninferiority_margin'])
    q1_a4, q1_b2, _ = paired_arrays(per_mid['garment_consistency'], 'a4', 'b2cont')
    garment_test = paired_test(q1_a4, q1_b2, 'two-sided')
    garment_invisible = bool(
        garment_test['p_value'] is not None
        and garment_test['p_value'] >= alpha
        and abs(float(garment_test['mean_delta'])) < visible_threshold
    )
    q2_a4, q2_b2, _ = paired_arrays(per_mid['identity_match'], 'a4', 'b2cont')
    identity_test = paired_test(q2_a4, q2_b2, 'greater')
    identity_visible = bool(
        identity_test['p_value'] is not None
        and identity_test['p_value'] < alpha
        and float(identity_test['mean_delta']) > 0.0
    )
    q1_alpha, q1_a4_ref, _ = paired_arrays(per_mid['garment_consistency'], 'alpha075', 'a4')
    q2_alpha, q2_a4_ref, _ = paired_arrays(per_mid['identity_match'], 'alpha075', 'a4')
    garment_noninferior = noninferiority_test(q1_alpha, q1_a4_ref, margin, alpha)
    identity_noninferior = noninferiority_test(q2_alpha, q2_a4_ref, margin, alpha)
    operating_point = bool(garment_noninferior['pass'] and identity_noninferior['pass'])
    verdicts = {
        'garment': (
            'GarmentSim -0.017 的回退在人眼下不可辨'
            if garment_invisible
            else 'GarmentSim -0.017 的回退在人眼下可辨，如实报告'
        ),
        'identity': (
            '身份增益肉眼可见'
            if identity_visible
            else '身份增益未达到肉眼显著门槛'
        ),
        'alpha075': (
            'alpha=0.75 身份与服装均不劣，支持无损运营点'
            if operating_point
            else 'alpha=0.75 未同时满足身份与服装不劣'
        ),
    }
    return {
        'summaries': summaries,
        'krippendorff_alpha_ordinal': reliabilities,
        'paired_two_sided': paired,
        'preregistered_tests': {
            'garment_a4_vs_b2cont': garment_test,
            'identity_a4_greater_b2cont': identity_test,
            'alpha075_garment_noninferiority_vs_a4': garment_noninferior,
            'alpha075_identity_noninferiority_vs_a4': identity_noninferior,
        },
        'verdicts': verdicts,
    }


def write_report(
    path: Path,
    result: dict[str, Any],
    qc_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> None:
    summaries = result['summaries']
    reliability_values = result['krippendorff_alpha_ordinal']
    tests = result['preregistered_tests']
    lines = [
        '# Blind Human Evaluation Report',
        '',
        f"**{result['verdicts']['garment']}**",
        '',
        f"**{result['verdicts']['identity']}**",
        '',
        f"**{result['verdicts']['alpha075']}**",
        '',
        'Scores are question-level rater medians. Means and deterministic bootstrap 95% CIs are computed over question medians.',
        '',
        '## Rater quality control',
        '',
        f"- Returned raters: `{sum(bool(row['returned']) for row in qc_rows)}`",
        f"- Included raters: `{sum(bool(row['included']) for row in qc_rows)}`",
        f"- Excluded raters: `{sum(not bool(row['included']) for row in qc_rows)}`",
        f"- Scored normal questions: `{len(task_rows)}`",
        '',
        '| rater | returned | checks | failed | missing | included |',
        '|---|---|---:|---:|---:|---|',
    ]
    for row in qc_rows:
        lines.append(
            f"| {row['rater_id']} | {row['returned']} | {row['attention_total']} | {row['attention_failed']} | "
            f"{row['missing_checks']} | {row['included']} |"
        )
    lines.extend([
        '',
        '## Question summaries',
        '',
        '| question / metric | B2-cont mean [95% CI] | alpha=0.75 mean [95% CI] | A4 mean [95% CI] | ordinal alpha (overall) |',
        '|---|---:|---:|---:|---:|',
    ])
    for field in FIELDS:
        cells = []
        for run in RUNS:
            value = summaries[field][run]
            cells.append(
                f"{fmt(value['mean'])} [{fmt(value['ci95_low'])}, {fmt(value['ci95_high'])}]"
            )
        lines.append(
            f"| {FIELD_LABELS[field]} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{fmt(reliability_values[field]['overall'])} |"
        )
    lines.extend([
        '',
        '## Ordinal Krippendorff alpha',
        '',
        '| question / metric | B2-cont | alpha=0.75 | A4 | overall |',
        '|---|---:|---:|---:|---:|',
    ])
    for field in FIELDS:
        values = reliability_values[field]
        lines.append(
            f"| {FIELD_LABELS[field]} | {fmt(values['b2cont'])} | {fmt(values['alpha075'])} | "
            f"{fmt(values['a4'])} | {fmt(values['overall'])} |"
        )
    lines.extend([
        '',
        '## Q1/Q2 paired Wilcoxon tests',
        '',
        '| metric | comparison | paired mids | mean delta | two-sided p |',
        '|---|---|---:|---:|---:|',
    ])
    comparisons = (
        ('a4_vs_b2cont', 'A4 - B2-cont'),
        ('alpha075_vs_b2cont', 'alpha=.75 - B2-cont'),
        ('alpha075_vs_a4', 'alpha=.75 - A4'),
    )
    for field in ('garment_consistency', 'identity_match'):
        for key, label in comparisons:
            value = result['paired_two_sided'][field][key]
            lines.append(
                f"| {FIELD_LABELS[field]} | {label} | {value['count']} | "
                f"{fmt(value['mean_delta'])} | {fmt(value['p_value'], 6)} |"
            )
    lines.extend([
        '',
        '## Preregistered tests',
        '',
        '| test | paired mids | mean delta | p | pass/readout |',
        '|---|---:|---:|---:|---|',
    ])
    test_rows = [
        ('Q1 A4 vs B2-cont, two-sided', tests['garment_a4_vs_b2cont'], result['verdicts']['garment']),
        ('Q2 A4 > B2-cont', tests['identity_a4_greater_b2cont'], result['verdicts']['identity']),
        ('Q1 alpha=.75 noninferior to A4', tests['alpha075_garment_noninferiority_vs_a4'], str(tests['alpha075_garment_noninferiority_vs_a4']['pass'])),
        ('Q2 alpha=.75 noninferior to A4', tests['alpha075_identity_noninferiority_vs_a4'], str(tests['alpha075_identity_noninferiority_vs_a4']['pass'])),
    ]
    for name, value, readout in test_rows:
        lines.append(
            f"| {name} | {value['count']} | {fmt(value['mean_delta'])} | "
            f"{fmt(value['p_value'], 6)} | {readout} |"
        )
    lines.extend([
        '',
        'The alpha=.75 noninferiority margin is 0.3 points on the five-point scale and is tested on per-mid paired differences. Q2 first averages the two registered identities within each mid.',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Score returned blind human-evaluation CSV files.')
    parser.add_argument('--config', default='configs/qualitative.yaml')
    parser.add_argument('--responses', nargs='*', default=None)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg['data']['root'])
    eval_cfg = cfg['qualitative_eval']['human_eval']
    key_path = resolve_path(root, eval_cfg['key'])
    key = read_json(key_path)
    if args.responses:
        paths = [Path(value) for value in args.responses]
    else:
        pattern = str(resolve_path(root, eval_cfg['response_dir']) / '*.csv')
        paths = [Path(value) for value in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(
            f'no returned response CSVs found under {resolve_path(root, eval_cfg["response_dir"])}'
        )
    responses, errors = load_responses(paths, key)
    if errors:
        raise RuntimeError('invalid response rows:\n' + json.dumps(errors, indent=2, ensure_ascii=False))
    qc_rows, included = attention_qc(responses, key)
    qc_path = resolve_path(root, eval_cfg['rater_qc'])
    write_csv(
        qc_path,
        qc_rows,
        ['rater_id', 'returned', 'attention_total', 'attention_failed', 'missing_checks', 'included', 'failed_question_ids'],
    )
    missing = response_coverage(responses, key, included)
    reassignment_path = resolve_path(root, eval_cfg['reassignment_needed'])
    write_csv(
        reassignment_path,
        missing,
        ['question_id', 'question_type', 'run', 'mid', 'jid', 'retained_ratings', 'required_ratings', 'needed'],
    )
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f'{len(missing)} normal questions have fewer than {key["protocol"]["min_retained_ratings"]} '
            f'retained ratings; see {reassignment_path} and collect replacements before final scoring'
        )
    medians = task_medians(responses, key, included)
    task_path = resolve_path(root, eval_cfg['task_medians'])
    write_csv(
        task_path,
        medians,
        [
            'question_id', 'question_type', 'run', 'mid', 'jid', 'rating_count',
            'garment_consistency', 'identity_match', 'garment_fidelity', 'realism',
        ],
    )
    result = summarize(medians, responses, key, included, eval_cfg)
    result['inputs'] = {
        'key': str(key_path),
        'response_csvs': [str(path) for path in paths],
        'included_raters': sorted(included),
        'excluded_raters': sorted(set(key['assignments']) - included),
        'coverage_shortfalls': len(missing),
    }
    report_path = resolve_path(root, eval_cfg['report'])
    write_report(report_path, result, qc_rows, medians)
    write_json(resolve_path(root, eval_cfg['report_json']), result)
    print(json.dumps({'verdicts': result['verdicts'], 'report': str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
