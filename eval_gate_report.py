from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, wilcoxon

from conditions import load_yaml


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'required gate-report input missing: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f'required gate-report CSV missing: {path}')
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return 'N/A'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return 'N/A' if not math.isfinite(number) else f'{number:.{digits}f}'


def garment_per_mid(metrics_dir: Path) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(metrics_dir / 'garment_pairwise_dino.csv'):
        try:
            grouped[row['mid']].append(float(row['dino_cosine']))
        except (KeyError, TypeError, ValueError):
            continue
    return {mid: float(np.mean(values)) for mid, values in grouped.items() if values}


def pose_variance_per_mid(metrics_dir: Path) -> dict[str, dict[str, float]]:
    by_mid_jid: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {'yaw': [], 'pitch': [], 'roll': []}
    )
    for row in read_csv(metrics_dir / 'headpose_per_image.csv'):
        if row.get('status') != 'ok':
            continue
        try:
            key = (row['mid'], row['jid'])
            by_mid_jid[key]['yaw'].append(float(row['pred_yaw']))
            by_mid_jid[key]['pitch'].append(float(row['pred_pitch']))
            by_mid_jid[key]['roll'].append(float(row['pred_roll']))
        except (KeyError, TypeError, ValueError):
            continue
    per_mid: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {'yaw': [], 'pitch': [], 'roll': []}
    )
    for (mid, _jid), axes in by_mid_jid.items():
        for axis in ('yaw', 'pitch', 'roll'):
            if axes[axis]:
                per_mid[mid][axis].append(float(np.mean(axes[axis])))
    result: dict[str, dict[str, float]] = {}
    for mid, axes in per_mid.items():
        if min(len(axes[axis]) for axis in ('yaw', 'pitch', 'roll')) < 2:
            continue
        variances = {axis: float(np.var(axes[axis], ddof=0)) for axis in ('yaw', 'pitch', 'roll')}
        variances['mean_axis_variance'] = float(np.mean(list(variances.values())))
        result[mid] = variances
    return result


def deltaid_per_image(metrics_dir: Path) -> dict[tuple[str, str, str], float]:
    result = {}
    for row in read_csv(metrics_dir / 'deltaid_per_image.csv'):
        if row.get('status') != 'ok':
            continue
        try:
            result[(row['mid'], row['jid'], row['seed'])] = float(row['delta_id'])
        except (KeyError, TypeError, ValueError):
            continue
    return result


def paired_values(
    left: dict[str, float] | dict[tuple[str, str, str], float],
    right: dict[str, float] | dict[tuple[str, str, str], float],
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    keys = sorted(set(left) & set(right))
    return (
        np.asarray([left[key] for key in keys], dtype=np.float64),
        np.asarray([right[key] for key in keys], dtype=np.float64),
        keys,
    )


def paired_test(a: np.ndarray, b: np.ndarray, alternative: str, higher_is_better: bool) -> dict[str, Any]:
    if len(a) == 0 or len(a) != len(b):
        return {'count': 0, 'p_value': None, 'rank_biserial': None, 'mean_delta': None}
    raw_diff = a - b
    improvement = raw_diff if higher_is_better else -raw_diff
    nonzero = improvement != 0
    if not bool(nonzero.any()):
        p_value = 1.0
        effect = 0.0
    else:
        p_value = float(wilcoxon(a, b, alternative=alternative, zero_method='wilcox').pvalue)
        ranks = rankdata(np.abs(improvement[nonzero]))
        positive = float(ranks[improvement[nonzero] > 0].sum())
        negative = float(ranks[improvement[nonzero] < 0].sum())
        effect = (positive - negative) / max(positive + negative, 1e-12)
    return {
        'count': int(len(a)),
        'p_value': p_value,
        'rank_biserial': effect,
        'mean_delta': float(np.mean(raw_diff)),
    }


def distribution(values: dict[str, float]) -> dict[str, float]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    if len(array) == 0:
        return {}
    count = max(1, int(math.ceil(len(array) * 0.25)))
    ordered = np.sort(array)
    return {
        'mean': float(np.mean(array)),
        'p10': float(np.quantile(array, 0.10)),
        'p25': float(np.quantile(array, 0.25)),
        'p50': float(np.quantile(array, 0.50)),
        'p75': float(np.quantile(array, 0.75)),
        'bottom_quartile_mean': float(np.mean(ordered[:count])),
    }


def plot_garment_hist(path: Path, runs: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    all_values = [value for rows in runs.values() for value in rows.values()]
    bins = np.linspace(min(all_values), max(all_values), 18) if all_values else 18
    for name, rows in runs.items():
        plt.hist(list(rows.values()), bins=bins, alpha=0.45, label=name, density=False)
    plt.xlabel('Per-mannequin cross-identity DINO mean')
    plt.ylabel('Mannequin count')
    plt.title('A2 vs equal-step B2-cont GarmentSim distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def fairness_check(root: Path, a2_cfg: dict, b2_cfg: dict) -> tuple[bool, dict[str, Any]]:
    a2_dir = Path(a2_cfg['experiment']['output_root']) / a2_cfg['experiment']['id']
    b2_dir = Path(b2_cfg['experiment']['output_root']) / b2_cfg['experiment']['id']
    a2_launch = load_json(a2_dir / 'launch.json')
    b2_launch = load_json(b2_dir / 'launch.json')
    a2_resolved = load_yaml(a2_dir / 'resolved_config.yaml')
    b2_resolved = load_yaml(b2_dir / 'resolved_config.yaml')
    fields = {
        'resume_trainable_hash': (a2_launch.get('resume_trainable_hash'), b2_launch.get('resume_trainable_hash')),
        'resume_sampler_state': (a2_launch.get('resume_sampler_state'), b2_launch.get('resume_sampler_state')),
        'seed': (a2_resolved['experiment']['seed'], b2_resolved['experiment']['seed']),
        'global_batch': (a2_resolved['_runtime']['global_batch'], b2_resolved['_runtime']['global_batch']),
        'effective_lr': (a2_resolved['_runtime']['effective_lr'], b2_resolved['_runtime']['effective_lr']),
        'continuation_steps': (a2_resolved['_runtime']['continuation_steps'], b2_resolved['_runtime']['continuation_steps']),
        'lora_rank': (a2_resolved['model']['lora_rank'], b2_resolved['model']['lora_rank']),
        'lr_schedule': (
            a2_resolved['training'].get('lr_schedule', 'constant'),
            b2_resolved['training'].get('lr_schedule', 'constant'),
        ),
    }
    mismatches = {name: values for name, values in fields.items() if values[0] != values[1]}
    return not mismatches, {
        'a2_run': str(a2_dir),
        'b2cont_run': str(b2_dir),
        'fields': fields,
        'mismatches': mismatches,
        'comparison': 'A2 vs B2-cont is the decision comparison; B2-prime is reference only.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Produce the A2 vs equal-step B2-cont decision report.')
    parser.add_argument('--a2-config', default='configs/a2_diff.yaml')
    parser.add_argument('--b2cont-config', default='configs/b2_cont.yaml')
    parser.add_argument('--a2-metrics', default='eval/a2_metrics')
    parser.add_argument('--b2cont-metrics', default='eval/b2cont_metrics')
    parser.add_argument('--b2p-metrics', default='eval/b2p_gatefix_metrics')
    parser.add_argument('--report', default='eval/gate_report.md')
    args = parser.parse_args()

    a2_cfg = load_yaml(args.a2_config)
    b2_cfg = load_yaml(args.b2cont_config)
    root = Path(a2_cfg['data']['root'])

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    metric_dirs = {
        'A2': resolve(args.a2_metrics),
        'B2-cont': resolve(args.b2cont_metrics),
        "B2' ref": resolve(args.b2p_metrics),
    }
    summaries = {
        name: {
            'delta': load_json(path / 'deltaid_summary.json'),
            'garment': load_json(path / 'garment_summary.json'),
            'headpose': load_json(path / 'headpose_summary.json'),
        }
        for name, path in metric_dirs.items()
    }
    garment = {name: garment_per_mid(path) for name, path in metric_dirs.items()}
    garment_dist = {name: distribution(values) for name, values in garment.items()}
    pose = {name: pose_variance_per_mid(path) for name, path in metric_dirs.items()}
    delta = {name: deltaid_per_image(path) for name, path in metric_dirs.items()}

    a2_g, b2_g, garment_keys = paired_values(garment['A2'], garment['B2-cont'])
    garment_test = paired_test(a2_g, b2_g, alternative='greater', higher_is_better=True)
    a2_pose_map = {mid: row['mean_axis_variance'] for mid, row in pose['A2'].items()}
    b2_pose_map = {mid: row['mean_axis_variance'] for mid, row in pose['B2-cont'].items()}
    a2_p, b2_p, pose_keys = paired_values(a2_pose_map, b2_pose_map)
    pose_test = paired_test(a2_p, b2_p, alternative='less', higher_is_better=False)
    a2_d, b2_d, delta_keys = paired_values(delta['A2'], delta['B2-cont'])
    delta_regression_test = paired_test(a2_d, b2_d, alternative='less', higher_is_better=True)

    hist_path = resolve('eval/gate_garment_per_mid_hist.png')
    plot_garment_hist(hist_path, garment)
    fair, fairness = fairness_check(root, a2_cfg, b2_cfg)
    mean_gain = garment_test['mean_delta']
    tail_gain = (
        garment_dist['A2']['bottom_quartile_mean']
        - garment_dist['B2-cont']['bottom_quartile_mean']
    )
    significant_mean_gain = (
        mean_gain is not None
        and mean_gain > 0.0
        and garment_test['p_value'] is not None
        and garment_test['p_value'] < 0.05
    )
    tail_win = tail_gain >= 0.02
    garment_win = significant_mean_gain or tail_win
    identity_regression = (
        delta_regression_test['mean_delta'] is not None
        and delta_regression_test['mean_delta'] < 0.0
        and delta_regression_test['p_value'] is not None
        and delta_regression_test['p_value'] < 0.05
    )
    if not fair:
        verdict = 'BLOCKED: A2/B2-cont fairness fields differ; no mechanism verdict is valid.'
    elif garment_win and identity_regression:
        verdict = 'MIXED: GarmentSim improved but identity regressed; inspect lambda balance and hinge calibration, then rerun once.'
    elif garment_win:
        verdict = 'PASS: 差分机制有效，进入 A4（定向对比身份损失）'
    else:
        verdict = (
            'FAIL: 差分无增量——停止堆 loss，回到方案层重估差分的价值锚点'
            '（姿势方差/弱身份区/更难身份对），先出诊断再决定 A3/A4 是否继续'
        )

    report_path = resolve(args.report)
    top = [f'# A2 vs B2-cont Gate Report', '', f'**{verdict}**']
    if identity_regression:
        top.extend(['', '**⚠ identity regression**'])
    lines = top + [
        '',
        'The decision comparison is A2 vs B2-cont with equal continuation steps. B2-prime is reference only.',
        '',
        '## Fairness',
        '',
        f"fairness status: {'PASS' if fair else 'BLOCKED'}",
        f"details: {fairness}",
        '',
        '## Three-metric Comparison',
        '',
        '| metric | A2 | B2-cont | B2-prime reference |',
        '|---|---:|---:|---:|',
    ]
    for label, getter in (
        ('DeltaID mean', lambda name: summaries[name]['delta'].get('mean')),
        ('held-out sim_target mean', lambda name: summaries[name]['delta'].get('sim_target_mean')),
        ('GarmentSim per-mid mean', lambda name: garment_dist[name].get('mean')),
        ('GarmentSim bottom-quartile mean', lambda name: garment_dist[name].get('bottom_quartile_mean')),
        ('head-pose yaw MAE', lambda name: summaries[name]['headpose'].get('yaw_mae_mean')),
        ('head-pose pitch MAE', lambda name: summaries[name]['headpose'].get('pitch_mae_mean')),
        ('head-pose roll MAE', lambda name: summaries[name]['headpose'].get('roll_mae_mean')),
        ('pose cross-ID mean-axis variance', lambda name: float(np.mean([row['mean_axis_variance'] for row in pose[name].values()]))),
    ):
        lines.append(
            f"| {label} | {fmt(getter('A2'))} | {fmt(getter('B2-cont'))} | {fmt(getter("B2' ref"))} |"
        )

    lines.extend([
        '',
        '## GarmentSim Tail Analysis',
        '',
        f'histogram: {hist_path}',
        '',
        '| run | P10 | P25 | P50 | P75 | bottom-quartile mean |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for name in ('A2', 'B2-cont', "B2' ref"):
        row = garment_dist[name]
        lines.append(
            f"| {name} | {fmt(row.get('p10'))} | {fmt(row.get('p25'))} | "
            f"{fmt(row.get('p50'))} | {fmt(row.get('p75'))} | {fmt(row.get('bottom_quartile_mean'))} |"
        )
    lines.extend([
        '',
        f'paired mids: {len(garment_keys)}',
        f"A2-B2-cont mean gain: {fmt(mean_gain)}",
        f'A2-B2-cont bottom-quartile gain: {fmt(tail_gain)}',
        f"Wilcoxon one-sided (A2 > B2-cont): p={fmt(garment_test['p_value'], 6)}, "
        f"rank-biserial effect={fmt(garment_test['rank_biserial'])}",
        '',
        '## Pose Cross-identity Variance',
        '',
        f'paired mids: {len(pose_keys)}',
        f"A2-B2-cont mean variance delta: {fmt(pose_test['mean_delta'])} (negative is better)",
        f"Wilcoxon one-sided (A2 < B2-cont): p={fmt(pose_test['p_value'], 6)}, "
        f"rank-biserial improvement effect={fmt(pose_test['rank_biserial'])}",
        '',
        '## Identity Preservation',
        '',
        f'paired images: {len(delta_keys)}',
        f"A2-B2-cont DeltaID mean delta: {fmt(delta_regression_test['mean_delta'])}",
        f"Wilcoxon regression test (A2 < B2-cont): p={fmt(delta_regression_test['p_value'], 6)}",
        f"identity regression: {'YES' if identity_regression else 'NO'}",
        '',
        '## Automatic Rule Inputs',
        '',
        f'significant mean GarmentSim gain: {significant_mean_gain}',
        f'bottom-quartile gain >= 0.02: {tail_win}',
        f'DeltaID significant regression: {identity_regression}',
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    bundle = {
        'verdict': verdict,
        'fairness': fairness,
        'garment_distribution': garment_dist,
        'garment_test': garment_test,
        'garment_tail_gain': tail_gain,
        'pose_test': pose_test,
        'deltaid_regression_test': delta_regression_test,
        'identity_regression': identity_regression,
        'report': str(report_path),
        'histogram': str(hist_path),
    }
    report_path.with_suffix('.json').write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    print(report_path)
    print(verdict)


if __name__ == '__main__':
    main()
