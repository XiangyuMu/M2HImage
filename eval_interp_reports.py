from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from conditions import load_yaml
from eval_a4_gate_report import detection_summary
from eval_gate_report import paired_test, paired_values, pose_variance_per_mid
from interp_common import alpha_label, read_json, resolve_root_path, write_json


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 'N/A'
    return 'N/A' if not math.isfinite(number) else f'{number:.{digits}f}'


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f'required interpolation CSV missing: {path}')
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def weight_metrics(metrics_dir: Path) -> dict[str, Any]:
    delta = read_json(metrics_dir / 'deltaid_summary.json')
    garment = read_json(metrics_dir / 'garment_summary.json')
    headpose = read_json(metrics_dir / 'headpose_summary.json')
    pose_rows = pose_variance_per_mid(metrics_dir)
    pose = [row['mean_axis_variance'] for row in pose_rows.values()]
    detection = detection_summary(metrics_dir)
    return {
        'sim_target': float(delta['sim_target_mean']),
        'delta_id': float(delta['mean']),
        'garment_sim': float(garment['cross_identity_group_mean_mean']),
        'pose_variance': float(np.mean(pose)),
        'detection_rate': float(detection['detection_rate']),
        'det_conf': float(detection['det_conf_mean']),
        'yaw_mae': float(headpose['yaw_mae_mean']),
        'pitch_mae': float(headpose['pitch_mae_mean']),
        'roll_mae': float(headpose['roll_mae_mean']),
        'metrics_dir': str(metrics_dir),
    }


def plot_pareto(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = [row['sim_target'] for row in rows]
    y = [row['garment_sim'] for row in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker='o', color='#2563eb', label='observed interpolation')
    plt.plot([x[0], x[-1]], [y[0], y[-1]], linestyle='--', color='#6b7280', label='endpoint metric line')
    for row in rows:
        plt.annotate(f"a={row['alpha']:.2f}", (row['sim_target'], row['garment_sim']), xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Held-out sim_target (higher is better)')
    plt.ylabel('GarmentSim (higher is better)')
    plt.title('B2-cont to A4 trainable-weight interpolation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def write_weight_report(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg['data']['root'])
    icfg = cfg['inference_interpolation']
    wcfg = icfg['weights']
    refs = icfg['endpoints']
    metrics_root = resolve_root_path(root, wcfg['metrics_root'])
    points = [0.0] + [float(value) for value in wcfg['alphas']] + [1.0]
    metric_dirs = {
        0.0: resolve_root_path(root, refs['b2cont_metrics_dir']),
        1.0: resolve_root_path(root, refs['a4_metrics_dir']),
        **{alpha: metrics_root / alpha_label(alpha) for alpha in points[1:-1]},
    }
    rows = [{'alpha': alpha, **weight_metrics(metric_dirs[alpha])} for alpha in points]
    x0, x1 = rows[0]['sim_target'], rows[-1]['sim_target']
    y0, y1 = rows[0]['garment_sim'], rows[-1]['garment_sim']
    for row in rows:
        metric_fraction = (row['sim_target'] - x0) / (x1 - x0) if abs(x1 - x0) > 1e-12 else row['alpha']
        expected = y0 + metric_fraction * (y1 - y0)
        deviation = row['garment_sim'] - expected
        row['endpoint_line_garment'] = expected
        row['frontier_vertical_delta'] = deviation
        row['frontier_shape'] = 'endpoint' if row['alpha'] in (0.0, 1.0) else ('above/favorable' if deviation > 0 else 'below/unfavorable')

    sim_values = np.asarray([row['sim_target'] for row in rows])
    garment_values = np.asarray([row['garment_sim'] for row in rows])
    sim_segments = np.diff(sim_values)
    garment_segments = np.diff(garment_values)
    sim_spearman = spearmanr(points, sim_values)
    garment_spearman = spearmanr(points, garment_values)
    sim_monotonic = bool(np.all(sim_segments >= 0.0))
    garment_monotonic = bool(np.all(garment_segments <= 0.0))

    endpoint_detection_min = min(rows[0]['detection_rate'], rows[-1]['detection_rate'])
    endpoint_conf_min = min(rows[0]['det_conf'], rows[-1]['det_conf'])
    realism_checks = []
    for row in rows[1:-1]:
        ok = (
            row['detection_rate'] >= float(wcfg['realism_detection_floor'])
            and row['detection_rate'] >= endpoint_detection_min - float(wcfg['realism_detection_drop_tolerance'])
            and row['det_conf'] >= endpoint_conf_min - float(wcfg['realism_confidence_drop_tolerance'])
        )
        realism_checks.append({'alpha': row['alpha'], 'pass': ok})
    realism_pass = all(row['pass'] for row in realism_checks)
    smooth = sim_monotonic and garment_monotonic and realism_pass
    verdict = (
        'SMOOTH-DIAL: 权重驻留证实 + 得到身份-服装可调拨盘'
        if smooth
        else 'NON-LINEAR: trade-off 在权重空间不可线性分离——loss 盆地几何结论，写入机制章'
    )

    favorable = []
    identity_total = x1 - x0
    garment_total = y0 - y1
    for row in rows[1:-1]:
        identity_fraction = (row['sim_target'] - x0) / identity_total if identity_total > 0 else float('-inf')
        garment_cost_fraction = (y0 - row['garment_sim']) / garment_total if garment_total > 0 else float('inf')
        row['identity_gain_fraction'] = identity_fraction
        row['garment_cost_fraction'] = garment_cost_fraction
        if (
            identity_fraction >= float(wcfg['favorable_identity_fraction'])
            and garment_cost_fraction < float(wcfg['favorable_garment_cost_fraction'])
        ):
            favorable.append(row['alpha'])
    if favorable:
        verdict += f"; favorable operating point alpha={','.join(f'{value:.2f}' for value in favorable)}"

    plot_path = resolve_root_path(root, wcfg['pareto_plot'])
    plot_pareto(plot_path, rows)
    result = {
        'verdict': verdict,
        'smooth_dial': smooth,
        'points': rows,
        'monotonicity': {
            'sim_target': {
                'pass': sim_monotonic,
                'spearman_rho': float(sim_spearman.statistic),
                'spearman_p': float(sim_spearman.pvalue),
                'segment_deltas': sim_segments.tolist(),
            },
            'garment_sim': {
                'pass': garment_monotonic,
                'spearman_rho': float(garment_spearman.statistic),
                'spearman_p': float(garment_spearman.pvalue),
                'segment_deltas': garment_segments.tolist(),
            },
        },
        'realism_checks': realism_checks,
        'favorable_operating_points': favorable,
        'pareto_plot': str(plot_path),
    }
    report_path = resolve_root_path(root, wcfg['report'])
    lines = [
        '# LoRA/Adapter Weight Interpolation Pareto Report',
        '',
        f'**{verdict}**',
        '',
        'This is a pure-inference diagnostic. Only aligned trainable tensors (LoRA A/B, condition projections, and gates) are interpolated; endpoint checkpoint files and frozen modules are unchanged.',
        '',
        '## Five-point curve',
        '',
        '| alpha | sim_target | DeltaID | GarmentSim | pose variance | detect rate | det conf | yaw/pitch/roll MAE | line delta | shape |',
        '|---:|---:|---:|---:|---:|---:|---:|---|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['alpha']:.2f} | {fmt(row['sim_target'])} | {fmt(row['delta_id'])} | "
            f"{fmt(row['garment_sim'])} | {fmt(row['pose_variance'])} | {fmt(row['detection_rate'])} | "
            f"{fmt(row['det_conf'])} | {fmt(row['yaw_mae'])}/{fmt(row['pitch_mae'])}/{fmt(row['roll_mae'])} | "
            f"{fmt(row['frontier_vertical_delta'], 5)} | {row['frontier_shape']} |"
        )
    lines.extend([
        '',
        f'Pareto plot: `{plot_path}`',
        '',
        '## Shape and monotonicity',
        '',
        f"sim_target monotonic increase: **{sim_monotonic}**; Spearman rho={fmt(sim_spearman.statistic)}, p={fmt(sim_spearman.pvalue, 6)}; segments={sim_segments.tolist()}",
        f"GarmentSim monotonic decrease: **{garment_monotonic}**; Spearman rho={fmt(garment_spearman.statistic)}, p={fmt(garment_spearman.pvalue, 6)}; segments={garment_segments.tolist()}",
        f'Realism/no-collapse check: **{realism_pass}**; details={realism_checks}',
        '',
        'Positive `line delta` means the point lies above the endpoint metric-space line: it retains more garment similarity than linear endpoint interpolation predicts at the observed identity score.',
        '',
        '## Preregistered readout',
        '',
        verdict,
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(resolve_root_path(root, wcfg['report_json']), result)
    return result


def _path_map(rows: list[dict[str, str]], checkpoint: str, metric: str) -> dict[str, float]:
    output = {}
    for row in rows:
        if row.get('checkpoint') != checkpoint or row.get('status') not in {'ok', 'partial'}:
            continue
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            output[row['mid']] = value
    return output


def _paired_metric(rows: list[dict[str, str]], metric: str, alternative: str, higher_is_better: bool) -> dict[str, Any]:
    a4, b2, keys = paired_values(_path_map(rows, 'a4', metric), _path_map(rows, 'b2cont', metric))
    test = paired_test(a4, b2, alternative=alternative, higher_is_better=higher_is_better)
    return {
        'metric': metric,
        'a4_mean': float(np.mean(a4)) if len(a4) else None,
        'b2cont_mean': float(np.mean(b2)) if len(b2) else None,
        'mean_delta': float(np.mean(a4 - b2)) if len(a4) else None,
        'paired_paths': len(keys),
        'test': test,
    }


def plot_identity_path_summary(path: Path, results: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ['path_efficiency', 'monotonic_violation_rate', 'garment_pairwise_mean', 'garment_normalized_drift']
    x = np.arange(len(names))
    a4 = [results[name]['a4_mean'] for name in names]
    b2 = [results[name]['b2cont_mean'] for name in names]
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x[:2] - width / 2, a4[:2], width, label='A4')
    axes[0].bar(x[:2] + width / 2, b2[:2], width, label='B2-cont')
    axes[0].set_xticks(x[:2], ['efficiency', 'monotonic violation'])
    axes[0].legend()
    axes[1].bar(x[2:] - width / 2, a4[2:], width, label='A4')
    axes[1].bar(x[2:] + width / 2, b2[2:], width, label='B2-cont')
    axes[1].set_xticks(x[2:], ['garment similarity', 'normalized drift'])
    axes[1].legend()
    fig.suptitle('Identity interpolation path metrics')
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_identity_report(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg['data']['root'])
    identity_cfg = cfg['inference_interpolation']['identity']
    metrics_dir = resolve_root_path(root, identity_cfg['metrics_dir'])
    rows = read_csv(metrics_dir / 'identity_interp_per_path.csv')
    results = {
        'path_efficiency': _paired_metric(rows, 'path_efficiency', 'greater', True),
        'monotonic_violation_rate': _paired_metric(rows, 'monotonic_violation_rate', 'less', False),
        'garment_pairwise_mean': _paired_metric(rows, 'garment_pairwise_mean', 'less', True),
        'garment_normalized_drift': _paired_metric(rows, 'garment_normalized_drift', 'greater', False),
    }
    alpha = float(identity_cfg['significance_alpha'])
    efficiency = results['path_efficiency']
    monotonicity = results['monotonic_violation_rate']
    normalized = results['garment_normalized_drift']
    efficiency_better = (
        efficiency['mean_delta'] is not None
        and efficiency['mean_delta'] > 0.0
        and efficiency['test']['p_value'] is not None
        and efficiency['test']['p_value'] < alpha
    )
    monotonicity_better = (
        monotonicity['mean_delta'] is not None
        and monotonicity['mean_delta'] < 0.0
        and monotonicity['test']['p_value'] is not None
        and monotonicity['test']['p_value'] < alpha
    )
    normalized_noninferior = (
        normalized['a4_mean'] is not None
        and normalized['b2cont_mean'] is not None
        and normalized['a4_mean'] <= normalized['b2cont_mean'] + float(identity_cfg['normalized_drift_tolerance'])
        and normalized['test']['p_value'] is not None
        and normalized['test']['p_value'] >= alpha
    )
    decoupled = efficiency_better and monotonicity_better and normalized_noninferior
    verdict = (
        'DECOUPLED: 差分训练整理了身份流形——核心机制第三路独立证据'
        if decoupled
        else 'EQUAL: 插值行为继承自 PuLID 预训练、非反事实训练塑造——如实写入'
    )
    summary = read_json(metrics_dir / 'identity_interp_summary.json')
    normalized_medians = {
        checkpoint: summary['branches'][checkpoint]['garment_normalized_drift']['median']
        for checkpoint in ('a4', 'b2cont')
    }
    plot_path = metrics_dir / 'identity_interp_metric_summary.png'
    plot_identity_path_summary(plot_path, results)
    result = {
        'verdict': verdict,
        'decoupled': decoupled,
        'criteria': {
            'efficiency_significantly_better': efficiency_better,
            'monotonicity_significantly_better': monotonicity_better,
            'normalized_garment_noninferior': normalized_noninferior,
        },
        'paired_metrics': results,
        'summary': summary,
        'plot': str(plot_path),
    }
    report_path = resolve_root_path(root, identity_cfg['report'])
    lines = [
        '# Identity-condition Interpolation Report',
        '',
        f'**{verdict}**',
        '',
        'Each A4/B2-cont path uses the same mannequin conditions, j/k endpoints, t grid, and initial noise. PuLID tokens use tokenwise slerp with linearly interpolated norm; appearance uses lerp.',
        '',
        '## Paired path tests',
        '',
        '| metric | A4 mean | B2-cont mean | A4-B2 | alternative | p | rank-biserial | criterion |',
        '|---|---:|---:|---:|---|---:|---:|---|',
    ]
    criterion_names = {
        'path_efficiency': str(efficiency_better),
        'monotonic_violation_rate': str(monotonicity_better),
        'garment_pairwise_mean': 'diagnostic only',
        'garment_normalized_drift': str(normalized_noninferior),
    }
    alternatives = {
        'path_efficiency': 'A4 greater',
        'monotonic_violation_rate': 'A4 less',
        'garment_pairwise_mean': 'A4 less (expected trade-off)',
        'garment_normalized_drift': 'A4 greater (regression)',
    }
    for name, result_row in results.items():
        lines.append(
            f"| {name} | {fmt(result_row['a4_mean'])} | {fmt(result_row['b2cont_mean'])} | "
            f"{fmt(result_row['mean_delta'])} | {alternatives[name]} | "
            f"{fmt(result_row['test']['p_value'], 6)} | {fmt(result_row['test']['rank_biserial'])} | "
            f"{criterion_names[name]} |"
        )
    lines.extend([
        '',
        f"Absolute garment stability difference is diagnostic and expected to reflect the known A4 trade-off; it does not enter the verdict. A4-B2={fmt(results['garment_pairwise_mean']['mean_delta'])}.",
        f"Normalized garment non-regression uses drift tolerance {float(identity_cfg['normalized_drift_tolerance']):.4f} and a one-sided regression test.",
        f"Normalized garment drift medians: A4={fmt(normalized_medians['a4'])}, B2-cont={fmt(normalized_medians['b2cont'])}.",
        'The registered denominator is abs(sim_to_j(t=1)-sim_to_j(t=0)). B2-cont paths with near-zero '
        'endpoint change create a heavy upper tail in the mean; values are retained without clipping or '
        'substituting the sim-to-k swing.',
        f'Plot: `{plot_path}`',
        '',
        '## Visual strips',
        '',
    ])
    for strip in summary.get('strips', []):
        lines.append(f'- `{strip}`')
    lines.extend([
        '',
        '## Preregistered readout',
        '',
        verdict,
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(resolve_root_path(root, identity_cfg['report_json']), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Write weight/identity interpolation reports.')
    parser.add_argument('--config', default='configs/interpolation.yaml')
    parser.add_argument('--part', choices=['weights', 'identity', 'all'], default='all')
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    output = {}
    if args.part in {'weights', 'all'}:
        output['weights'] = write_weight_report(cfg)
    if args.part in {'identity', 'all'}:
        output['identity'] = write_identity_report(cfg)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

