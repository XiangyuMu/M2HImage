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


LOSS_KEYS = ('loss_pair', 'loss_teach', 'loss_inv', 'loss_hinge')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f'A2 training log missing: {path}')
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'invalid JSONL at {path}:{line_no}: {exc}') from exc
            if all(key in row for key in LOSS_KEYS):
                rows.append(row)
    if not rows:
        raise RuntimeError(f'A2 training log contains no differential metric rows: {path}')
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f'diagnostic input missing: {path}')
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def rolling_mean(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < 2:
        return values.copy()
    window = min(window, len(values))
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window - 1, 0), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def paired_test(left: np.ndarray, right: np.ndarray, alternative: str) -> dict[str, Any]:
    if len(left) == 0 or len(left) != len(right):
        raise RuntimeError(f'invalid paired arrays: left={len(left)}, right={len(right)}')
    diff = left - right
    nonzero = np.abs(diff) > 1e-12
    if not bool(nonzero.any()):
        p_value = 1.0
        effect = 0.0
    else:
        p_value = float(wilcoxon(left, right, alternative=alternative, zero_method='wilcox').pvalue)
        ranks = rankdata(np.abs(diff[nonzero]))
        positive = float(ranks[diff[nonzero] > 0].sum())
        negative = float(ranks[diff[nonzero] < 0].sum())
        effect = (positive - negative) / max(positive + negative, 1e-12)
    return {
        'count': int(len(left)),
        'mean_left': float(np.mean(left)),
        'mean_right': float(np.mean(right)),
        'mean_gain': float(np.mean(diff)),
        'median_gain': float(np.median(diff)),
        'p_value': p_value,
        'rank_biserial': effect,
        'gain_p10': float(np.quantile(diff, 0.10)),
        'gain_p25': float(np.quantile(diff, 0.25)),
        'gain_p75': float(np.quantile(diff, 0.75)),
        'gain_p90': float(np.quantile(diff, 0.90)),
    }


def paired_deltaid(a2_path: Path, control_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    def load(path: Path) -> dict[tuple[str, str, str], float]:
        result = {}
        for row in read_csv(path):
            if row.get('status') != 'ok':
                continue
            try:
                result[(row['mid'], row['jid'], row['seed'])] = float(row['delta_id'])
            except (KeyError, TypeError, ValueError):
                continue
        return result

    a2 = load(a2_path)
    control = load(control_path)
    keys = sorted(set(a2) & set(control))
    if not keys:
        raise RuntimeError('A2/B2-cont DeltaID CSVs have no paired successful rows')
    left = np.asarray([a2[key] for key in keys], dtype=np.float64)
    right = np.asarray([control[key] for key in keys], dtype=np.float64)
    return paired_test(left, right, alternative='greater'), left - right


def garment_per_mid(path: Path) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(path):
        try:
            grouped[row['mid']].append(float(row['dino_cosine']))
        except (KeyError, TypeError, ValueError):
            continue
    return {mid: float(np.mean(values)) for mid, values in grouped.items() if values}


def garment_type_by_mid(deltaid_path: Path) -> dict[str, str]:
    result = {}
    for row in read_csv(deltaid_path):
        mid = row.get('mid')
        garment_type = row.get('garment_type')
        if mid and garment_type:
            result[mid] = garment_type
    return result


def garment_diagnostic(
    a2_pairwise: Path,
    control_pairwise: Path,
    deltaid_path: Path,
) -> dict[str, Any]:
    a2 = garment_per_mid(a2_pairwise)
    control = garment_per_mid(control_pairwise)
    keys = sorted(set(a2) & set(control))
    if not keys:
        raise RuntimeError('A2/B2-cont garment CSVs have no paired mannequin IDs')
    count = max(1, int(math.ceil(len(keys) * 0.25)))
    a2_tail = set(sorted(keys, key=lambda mid: a2[mid])[:count])
    control_tail = set(sorted(keys, key=lambda mid: control[mid])[:count])
    union = a2_tail | control_tail
    types = garment_type_by_mid(deltaid_path)
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for mid in keys:
        grouped[types.get(mid, 'unknown')].append((a2[mid], control[mid]))
    by_type = {}
    for garment_type, values in sorted(grouped.items()):
        left = np.asarray([value[0] for value in values], dtype=np.float64)
        right = np.asarray([value[1] for value in values], dtype=np.float64)
        stats = paired_test(left, right, alternative='greater')
        by_type[garment_type] = stats
    return {
        'paired_mids': len(keys),
        'tail_count': count,
        'a2_tail': sorted(a2_tail),
        'control_tail': sorted(control_tail),
        'tail_intersection': sorted(a2_tail & control_tail),
        'tail_jaccard': len(a2_tail & control_tail) / max(1, len(union)),
        'per_type': by_type,
    }


def plot_training(rows: list[dict[str, Any]], path: Path) -> None:
    steps = np.asarray([float(row['run_step']) for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    for key in LOSS_KEYS:
        values = np.asarray([float(row[key]) for row in rows])
        ax.plot(steps, rolling_mean(values), label=key)
    ax.set_title('A2 losses (rolling mean)')
    ax.set_xlabel('Continuation step')
    ax.set_ylabel('Loss')
    ax.legend()

    hinge = np.asarray([float(row.get('hinge_active_rate', 0.0)) for row in rows])
    axes[0, 1].plot(steps, rolling_mean(hinge), color='tab:red')
    axes[0, 1].set_title('Hinge activation rate')
    axes[0, 1].set_xlabel('Continuation step')
    axes[0, 1].set_ylim(0.0, 1.0)

    face = np.asarray([float(row.get('face_diff_norm', 0.0)) for row in rows])
    axes[1, 0].plot(steps, rolling_mean(face), color='tab:green')
    axes[1, 0].set_title('Face-region differential norm')
    axes[1, 0].set_xlabel('Continuation step')

    active = np.asarray([float(row.get('diff_active_ratio', 0.0)) for row in rows])
    axes[1, 1].plot(steps, rolling_mean(active), color='tab:purple')
    axes[1, 1].set_title('Differential branch active ratio')
    axes[1, 1].set_xlabel('Continuation step')
    axes[1, 1].set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_deltaid_gain(gains: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(gains, bins=32, alpha=0.85, color='tab:blue')
    plt.axvline(0.0, color='black', linestyle='--', linewidth=1)
    plt.axvline(float(np.mean(gains)), color='tab:red', linewidth=1.5, label=f'mean={np.mean(gains):.4f}')
    plt.xlabel('DeltaID(A2) - DeltaID(B2-cont)')
    plt.ylabel('Image count')
    plt.title('Paired held-out DeltaID gain')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return 'N/A'
    return f'{float(value):.{digits}f}'


def main() -> None:
    parser = argparse.ArgumentParser(description='Diagnose the completed A2 differential gate run.')
    parser.add_argument('--config', default='configs/a2_diff.yaml')
    parser.add_argument('--output', default='docs/results/a2_gate/diagnosis.md')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    run_dir = Path(cfg['experiment']['output_root']) / cfg['experiment']['id']
    resolved_path = run_dir / 'resolved_config.yaml'
    resolved = load_yaml(resolved_path)
    rows = read_jsonl(run_dir / 'logs/train.jsonl')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    training_plot = output.with_name('diagnosis_training_curves.png')
    delta_plot = output.with_name('diagnosis_deltaid_gain.png')
    plot_training(rows, training_plot)

    diff_cfg = resolved['training']['differential']
    hinge_g = diff_cfg.get('hinge_g_resolved')
    calibration_steps = int(diff_cfg.get('calibration_steps', 200))
    post = [row for row in rows if int(row['run_step']) > calibration_steps]
    if not post:
        raise RuntimeError('A2 log has no rows after hinge calibration')
    means = {key: float(np.mean([float(row[key]) for row in post])) for key in LOSS_KEYS}
    hinge_active_mean = float(np.mean([float(row.get('hinge_active_rate', 0.0)) for row in post]))
    hinge_active_max = float(np.max([float(row.get('hinge_active_rate', 0.0)) for row in post]))
    face_start = float(np.mean([float(row.get('face_diff_norm', 0.0)) for row in post[:50]]))
    face_end = float(np.mean([float(row.get('face_diff_norm', 0.0)) for row in post[-50:]]))
    weighted_diff = (
        float(diff_cfg.get('lambda_teach', 0.5)) * means['loss_teach']
        + float(diff_cfg.get('lambda_inv', 0.2)) * means['loss_inv']
        + float(diff_cfg.get('lambda_hinge', 0.05)) * means['loss_hinge']
    )
    weighted_ratio = weighted_diff / max(means['loss_pair'], 1e-12)
    if hinge_g is None or not np.isfinite(float(hinge_g)) or float(hinge_g) <= 0 or hinge_active_max <= 0:
        binding = 'DEAD'
    elif weighted_ratio < 0.05:
        binding = 'WEAK'
    else:
        binding = 'BOUND'

    delta_stats, gains = paired_deltaid(
        root / 'eval/a2_metrics/deltaid_per_image.csv',
        root / 'eval/b2cont_metrics/deltaid_per_image.csv',
    )
    plot_deltaid_gain(gains, delta_plot)
    delta_status = (
        'SIGNIFICANT'
        if delta_stats['mean_gain'] > 0 and delta_stats['p_value'] < 0.05
        else 'NOT-SIGNIFICANT'
    )
    garment = garment_diagnostic(
        root / 'eval/a2_metrics/garment_pairwise_dino.csv',
        root / 'eval/b2cont_metrics/garment_pairwise_dino.csv',
        root / 'eval/a2_metrics/deltaid_per_image.csv',
    )

    if delta_status == 'NOT-SIGNIFICANT':
        decision = 'STOP'
        decision_text = '转锚地基不成立；中止 A4，等待人工决策。'
    elif binding == 'DEAD':
        decision = 'PROCEED_ATTRIBUTION_UNCERTAIN'
        decision_text = (
            'A4 按计划继续，但 A2 身份增益不能归因于 hinge；'
            '可能来自 teach/inv 的间接效应，报告必须注明归因不确定。'
        )
    else:
        decision = 'PROCEED'
        decision_text = '转锚地基成立，按计划进入 A4。'

    lines = [
        '# A2 Diagnostic Report',
        '',
        f'- A2 run: `{run_dir}`',
        f'- resolved config: `{resolved_path}`',
        f'- logged metric rows: {len(rows)}',
        '',
        '## Differential Loss Binding',
        '',
        f'- calibrated hinge_g: **{fmt(hinge_g)}**',
        f'- post-calibration loss_pair mean: {fmt(means["loss_pair"])}',
        f'- post-calibration loss_teach mean: {fmt(means["loss_teach"])}',
        f'- post-calibration loss_inv mean: {fmt(means["loss_inv"])}',
        f'- post-calibration loss_hinge mean: {fmt(means["loss_hinge"])}',
        f'- weighted differential / loss_pair: {weighted_ratio:.2%}',
        f'- hinge activation mean/max: {hinge_active_mean:.2%} / {hinge_active_max:.2%}',
        f'- face differential norm start/end: {fmt(face_start)} / {fmt(face_end)}',
        f'- curves: [diagnosis_training_curves.png]({training_plot.name})',
        '',
        f'**Binding conclusion: {binding}**',
        '',
        '## Paired Held-out DeltaID Gain',
        '',
        f'- paired images: {delta_stats["count"]}',
        f'- A2 mean: {fmt(delta_stats["mean_left"])}',
        f'- B2-cont mean: {fmt(delta_stats["mean_right"])}',
        f'- mean/median gain: {fmt(delta_stats["mean_gain"])} / {fmt(delta_stats["median_gain"])}',
        f'- Wilcoxon greater p: {fmt(delta_stats["p_value"], 9)}',
        f'- rank-biserial effect: {fmt(delta_stats["rank_biserial"])}',
        f'- gain P10/P25/P75/P90: {fmt(delta_stats["gain_p10"])} / {fmt(delta_stats["gain_p25"])} / '
        f'{fmt(delta_stats["gain_p75"])} / {fmt(delta_stats["gain_p90"])}',
        f'- histogram: [diagnosis_deltaid_gain.png]({delta_plot.name})',
        '',
        f'**DeltaID conclusion: {delta_status}**',
        '',
        '## Residual Garment Instability',
        '',
        f'- paired mannequin IDs: {garment["paired_mids"]}',
        f'- bottom-quartile size: {garment["tail_count"]}',
        f'- low-tail overlap: {len(garment["tail_intersection"])}',
        f'- low-tail Jaccard: {garment["tail_jaccard"]:.4f}',
        '',
        '| garment type | count | A2 mean | B2-cont mean | gain | greater p |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for garment_type, stats in garment['per_type'].items():
        lines.append(
            f'| {garment_type} | {stats["count"]} | {stats["mean_left"]:.4f} | '
            f'{stats["mean_right"]:.4f} | {stats["mean_gain"]:+.4f} | {stats["p_value"]:.6f} |'
        )
    lines.extend([
        '',
        f'A2 low-tail mids: `{garment["a2_tail"]}`',
        '',
        f'B2-cont low-tail mids: `{garment["control_tail"]}`',
        '',
        '## Preregistered Diagnostic Decision',
        '',
        f'**{decision}: {decision_text}**',
        '',
    ])
    output.write_text('\n'.join(lines), encoding='utf-8')
    payload = {
        'binding': {
            'status': binding,
            'hinge_g': hinge_g,
            'loss_means': means,
            'weighted_differential_ratio': weighted_ratio,
            'hinge_activation_mean': hinge_active_mean,
            'hinge_activation_max': hinge_active_max,
            'face_diff_start': face_start,
            'face_diff_end': face_end,
        },
        'deltaid': {**delta_stats, 'status': delta_status},
        'garment': garment,
        'decision': decision,
        'decision_text': decision_text,
    }
    output.with_suffix('.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(output)
    print(f'BINDING={binding}')
    print(f'DELTAID={delta_status}')
    print(f'A4_DECISION={decision}')
    if decision == 'STOP':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
