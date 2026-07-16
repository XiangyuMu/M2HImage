from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from conditions import load_yaml
from dataset import IdentityBank
from eval_gate_report import (
    fmt,
    garment_per_mid,
    load_json,
    paired_test,
    paired_values,
    pose_variance_per_mid,
)


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f'required A4 gate CSV missing: {path}')
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def identity_rows(metrics_dir: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    result = {}
    for row in read_csv(metrics_dir / 'deltaid_per_image.csv'):
        if row.get('status') != 'ok':
            continue
        try:
            result[(row['mid'], row['jid'], row['seed'])] = {
                'sim_target': float(row['sim_target']),
                'delta_id': float(row['delta_id']),
                'det_conf': float(row['det_conf']),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return result


def metric_map(rows: dict[Any, dict[str, float]], key: str) -> dict[Any, float]:
    return {index: values[key] for index, values in rows.items()}


def detection_summary(metrics_dir: Path) -> dict[str, float]:
    rows = read_csv(metrics_dir / 'deltaid_per_image.csv')
    valid = []
    for row in rows:
        if row.get('status') != 'ok':
            continue
        try:
            valid.append(float(row['det_conf']))
        except (KeyError, TypeError, ValueError):
            continue
    return {
        'total': len(rows),
        'detected': len(valid),
        'detection_rate': len(valid) / max(1, len(rows)),
        'det_conf_mean': float(np.mean(valid)) if valid else float('nan'),
        'det_conf_median': float(np.median(valid)) if valid else float('nan'),
    }


def fairness_check(a4_cfg: dict, b2_cfg: dict) -> tuple[bool, dict[str, Any]]:
    a4_dir = Path(a4_cfg['experiment']['output_root']) / a4_cfg['experiment']['id']
    b2_dir = Path(b2_cfg['experiment']['output_root']) / b2_cfg['experiment']['id']
    a4_launch = load_json(a4_dir / 'launch.json')
    b2_launch = load_json(b2_dir / 'launch.json')
    a4_resolved = load_yaml(a4_dir / 'resolved_config.yaml')
    b2_resolved = load_yaml(b2_dir / 'resolved_config.yaml')
    fields = {
        'resume_trainable_hash': (a4_launch.get('resume_trainable_hash'), b2_launch.get('resume_trainable_hash')),
        'resume_sampler_state': (a4_launch.get('resume_sampler_state'), b2_launch.get('resume_sampler_state')),
        'train_ids_hash': (a4_launch.get('train_ids_hash'), b2_launch.get('train_ids_hash')),
        'train_sample_count': (a4_launch.get('train_sample_count'), b2_launch.get('train_sample_count')),
        'excluded_train_ids': (a4_launch.get('excluded_train_ids'), b2_launch.get('excluded_train_ids')),
        'seed': (a4_resolved['experiment']['seed'], b2_resolved['experiment']['seed']),
        'global_batch': (a4_resolved['_runtime']['global_batch'], b2_resolved['_runtime']['global_batch']),
        'effective_lr': (a4_resolved['_runtime']['effective_lr'], b2_resolved['_runtime']['effective_lr']),
        'continuation_steps': (a4_resolved['_runtime']['continuation_steps'], b2_resolved['_runtime']['continuation_steps']),
        'lora_rank': (a4_resolved['model']['lora_rank'], b2_resolved['model']['lora_rank']),
        'lr_schedule': (
            a4_resolved['training'].get('lr_schedule', 'constant'),
            b2_resolved['training'].get('lr_schedule', 'constant'),
        ),
    }
    mismatches = {name: values for name, values in fields.items() if values[0] != values[1]}
    return not mismatches, {'fields': fields, 'mismatches': mismatches}


def load_sampling_log(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    distances: list[float] = []
    relax = {'strict': 0, 'relax_age': 0, 'relax_skin': 0}
    if not path.exists():
        raise FileNotFoundError(f'A4 treatment-strength log missing: {path}')
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            distances.extend(float(value) for value in row.get('distances', []))
            relax['strict'] += int(row.get('relax_strict', 0))
            relax['relax_age'] += int(row.get('relax_age', 0))
            relax['relax_skin'] += int(row.get('relax_skin', 0))
    if not distances:
        raise RuntimeError(f'A4 treatment-strength log contains no distances: {path}')
    return np.asarray(distances, dtype=np.float64), relax


def replay_random_bank_v2(cfg: dict, count: int) -> np.ndarray:
    root = Path(cfg['data']['root'])
    bank = IdentityBank(root / cfg['data']['identity_bank_v2'])
    excluded = set(str(value) for value in cfg['data'].get('exclude_train_ids', []))
    ids = [sample_id for sample_id in bank.ids.tolist() if sample_id not in excluded]
    base_seed = int(cfg['experiment']['seed'])
    values = []
    cursor = 0
    while len(values) < count:
        index = cursor % len(ids)
        epoch = cursor // len(ids)
        sample_seed = (base_seed * 1_000_003 + epoch * 97_409 + index * 65_537) & ((1 << 63) - 1)
        row = bank.sample_pair_details(ids[index], sample_seed, sampling='random')
        values.append(float(row['delta_arc']))
        cursor += 1
    return np.asarray(values, dtype=np.float64)


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        'count': int(len(values)),
        'mean': float(np.mean(values)),
        'p25': float(np.quantile(values, 0.25)),
        'p50': float(np.quantile(values, 0.50)),
        'p75': float(np.quantile(values, 0.75)),
    }


def plot_sampling(path: Path, random_values: np.ndarray, semihard_values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    low = min(float(random_values.min()), float(semihard_values.min()))
    high = max(float(random_values.max()), float(semihard_values.max()))
    bins = np.linspace(low, high, 40)
    plt.figure(figsize=(8, 5))
    plt.hist(random_values, bins=bins, alpha=0.5, density=True, label='A2 random policy replay (bank v2)')
    plt.hist(semihard_values, bins=bins, alpha=0.5, density=True, label='A4 semi-hard actual')
    plt.xlabel('d(j,k) = 1 - cosine in F_train space')
    plt.ylabel('density')
    plt.title('Differential treatment strength')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def training_identity_summary(path: Path) -> dict[str, Any]:
    rows = []
    if path.exists():
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                if float(row.get('id_loss_triggered', 0.0)) <= 0.0:
                    continue
                rows.append(row)
    attempts = sum(float(row.get('id_loss_attempt_count', 0.0)) for row in rows)
    skips = sum(float(row.get('id_loss_skip_count', 0.0)) for row in rows)
    gaps = [float(row['sim_gap']) for row in rows if row.get('sim_gap') is not None]
    return {
        'triggered_log_rows': len(rows),
        'attempts': attempts,
        'skips': skips,
        'skip_rate': skips / attempts if attempts > 0.0 else None,
        'sim_gap_first_quartile_mean': float(np.mean(gaps[:max(1, len(gaps) // 4)])) if gaps else None,
        'sim_gap_last_quartile_mean': float(np.mean(gaps[-max(1, len(gaps) // 4):])) if gaps else None,
        'sim_gap_latest': gaps[-1] if gaps else None,
    }


def decide_verdict(
    fair: bool,
    treatment_strengthened: bool,
    identity_pass: bool,
    garment_pass: bool,
    pose_pass: bool,
    face_pass: bool,
) -> str:
    if not fair:
        return 'BLOCKED: A4/B2-cont fairness fields differ; the preregistered verdict is invalid.'
    if not treatment_strengthened:
        return 'BLOCKED: semi-hard d(j,k) distribution was not stronger than the random-policy replay; treatment validity failed.'
    if not identity_pass:
        return 'FAIL: 机制在两条轴上均无足量增量——收敛为 adapter 系统 + 头朝向可控 + 负结果分析，不开第三轮'
    if garment_pass and pose_pass and face_pass:
        return 'PASS: 核心贡献重写为 identity-directed counterfactual training；A2 服装轴结果作为诚实分析章节'
    return 'MIXED: 身份-服装 trade-off 成立，作为发现如实报告，训练侧不再重跑，λ 权衡留给论文讨论'


def main() -> None:
    parser = argparse.ArgumentParser(description='Produce the one-shot preregistered A4 identity-axis gate report.')
    parser.add_argument('--a4-config', default='configs/a4_directed.yaml')
    parser.add_argument('--b2cont-config', default='configs/b2_cont.yaml')
    parser.add_argument('--a4-metrics', default='eval/a4_metrics')
    parser.add_argument('--b2cont-metrics', default='eval/b2cont_metrics')
    parser.add_argument('--a2-metrics', default='eval/a2_metrics')
    parser.add_argument('--report', default='eval/a4_gate_report.md')
    args = parser.parse_args()

    a4_cfg = load_yaml(args.a4_config)
    b2_cfg = load_yaml(args.b2cont_config)
    root = Path(a4_cfg['data']['root'])
    metric_dirs = {
        'A4': resolve(root, args.a4_metrics),
        'B2-cont': resolve(root, args.b2cont_metrics),
        'A2 ref': resolve(root, args.a2_metrics),
    }
    summaries = {
        name: {
            'delta': load_json(path / 'deltaid_summary.json'),
            'garment': load_json(path / 'garment_summary.json'),
            'headpose': load_json(path / 'headpose_summary.json'),
            'detection': detection_summary(path),
        }
        for name, path in metric_dirs.items()
    }
    identities = {name: identity_rows(path) for name, path in metric_dirs.items()}
    garments = {name: garment_per_mid(path) for name, path in metric_dirs.items()}
    poses = {name: pose_variance_per_mid(path) for name, path in metric_dirs.items()}

    a4_sim, b2_sim, sim_keys = paired_values(
        metric_map(identities['A4'], 'sim_target'), metric_map(identities['B2-cont'], 'sim_target')
    )
    sim_test = paired_test(a4_sim, b2_sim, alternative='greater', higher_is_better=True)
    a4_delta, b2_delta, delta_keys = paired_values(
        metric_map(identities['A4'], 'delta_id'), metric_map(identities['B2-cont'], 'delta_id')
    )
    delta_test = paired_test(a4_delta, b2_delta, alternative='greater', higher_is_better=True)
    a4_garment, b2_garment, garment_keys = paired_values(garments['A4'], garments['B2-cont'])
    garment_regression_test = paired_test(a4_garment, b2_garment, alternative='less', higher_is_better=True)
    a4_pose_map = {mid: row['mean_axis_variance'] for mid, row in poses['A4'].items()}
    b2_pose_map = {mid: row['mean_axis_variance'] for mid, row in poses['B2-cont'].items()}
    a4_pose, b2_pose, pose_keys = paired_values(a4_pose_map, b2_pose_map)
    pose_regression_test = paired_test(a4_pose, b2_pose, alternative='greater', higher_is_better=False)
    a4_conf, b2_conf, conf_keys = paired_values(
        metric_map(identities['A4'], 'det_conf'), metric_map(identities['B2-cont'], 'det_conf')
    )
    face_regression_test = paired_test(a4_conf, b2_conf, alternative='less', higher_is_better=True)

    fair, fairness = fairness_check(a4_cfg, b2_cfg)
    sim_gain = float(np.mean(a4_sim - b2_sim))
    identity_pass = sim_gain >= 0.03 and sim_test['p_value'] is not None and sim_test['p_value'] < 0.05
    garment_mean_a4 = float(np.mean(a4_garment))
    garment_mean_b2 = float(np.mean(b2_garment))
    garment_pass = (
        garment_mean_a4 >= garment_mean_b2 - 0.005
        and garment_regression_test['p_value'] is not None
        and garment_regression_test['p_value'] >= 0.05
    )
    pose_mean_a4 = float(np.mean(a4_pose))
    pose_mean_b2 = float(np.mean(b2_pose))
    pose_pass = (
        pose_mean_a4 <= pose_mean_b2 + 0.005
        and pose_regression_test['p_value'] is not None
        and pose_regression_test['p_value'] >= 0.05
    )
    detection_pass = summaries['A4']['detection']['detection_rate'] >= 0.95
    face_quality_pass = (
        summaries['A4']['detection']['det_conf_mean'] >= summaries['B2-cont']['detection']['det_conf_mean'] - 0.01
        and face_regression_test['p_value'] is not None
        and face_regression_test['p_value'] >= 0.05
    )
    face_pass = detection_pass and face_quality_pass

    a4_run = Path(a4_cfg['experiment']['output_root']) / a4_cfg['experiment']['id']
    semihard_values, relaxation_counts = load_sampling_log(a4_run / 'logs' / 'sampling_d_jk.jsonl')
    random_values = replay_random_bank_v2(a4_cfg, len(semihard_values))
    random_dist = distribution(random_values)
    semihard_dist = distribution(semihard_values)
    treatment_strengthened = semihard_dist['mean'] > random_dist['mean']
    verdict = decide_verdict(
        fair,
        treatment_strengthened,
        identity_pass,
        garment_pass,
        pose_pass,
        face_pass,
    )
    sampling_plot = resolve(root, 'eval/a4_treatment_strength.png')
    plot_sampling(sampling_plot, random_values, semihard_values)
    training_summary = training_identity_summary(a4_run / 'logs' / 'train.jsonl')

    report_path = resolve(root, args.report)
    lines = [
        '# A4 One-shot Identity-axis Gate Report',
        '',
        f'**{verdict}**',
        '',
        'This is the preregistered final mechanism run. The decision comparison is A4 vs the existing equal-step B2-cont; A2 is reference only. No third rescue run is allowed.',
        '',
        '## Fairness',
        '',
        f"fairness: {'PASS' if fair else 'BLOCKED'}",
        f'details: {fairness}',
        '',
        '## Three-run Comparison',
        '',
        '| metric | A4 | B2-cont | A2 reference |',
        '|---|---:|---:|---:|',
    ]
    for label, getter in (
        ('held-out sim_target mean', lambda name: summaries[name]['delta'].get('sim_target_mean')),
        ('DeltaID mean', lambda name: summaries[name]['delta'].get('mean')),
        ('GarmentSim per-mid mean', lambda name: float(np.mean(list(garments[name].values())))),
        ('pose cross-ID mean-axis variance', lambda name: float(np.mean([row['mean_axis_variance'] for row in poses[name].values()]))),
        ('face detection rate', lambda name: summaries[name]['detection']['detection_rate']),
        ('face detector confidence mean', lambda name: summaries[name]['detection']['det_conf_mean']),
        ('head-pose yaw MAE', lambda name: summaries[name]['headpose'].get('yaw_mae_mean')),
        ('head-pose pitch MAE', lambda name: summaries[name]['headpose'].get('pitch_mae_mean')),
        ('head-pose roll MAE', lambda name: summaries[name]['headpose'].get('roll_mae_mean')),
    ):
        lines.append(f"| {label} | {fmt(getter('A4'))} | {fmt(getter('B2-cont'))} | {fmt(getter('A2 ref'))} |")

    lines.extend([
        '',
        '## Identity Primary Gate',
        '',
        f'- paired images: {len(sim_keys)}',
        f'- sim_target mean gain A4-B2-cont: {sim_gain:.6f} (required >=0.03)',
        f"- Wilcoxon greater p={fmt(sim_test['p_value'], 8)}, rank-biserial={fmt(sim_test['rank_biserial'])}",
        f"- DeltaID mean gain: {fmt(delta_test['mean_delta'], 6)}, greater p={fmt(delta_test['p_value'], 8)}",
        f"- identity condition: {'PASS' if identity_pass else 'FAIL'}",
        '',
        '## Treatment Strength',
        '',
        f'- histogram: `{sampling_plot}`',
        f'- semi-hard relaxation counts: {relaxation_counts}',
        f"- strengthened in F_train space: {'YES' if treatment_strengthened else 'NO'}",
        '- The random baseline replays the exact A2 random pairing policy but measures both policies in bank-v2 F_train space; old-bank and bank-v2 cosine distances are not mixed.',
        '',
        '| policy | count | mean | P25 | P50 | P75 |',
        '|---|---:|---:|---:|---:|---:|',
        f"| A2 random policy replay, bank v2 | {random_dist['count']} | {random_dist['mean']:.4f} | {random_dist['p25']:.4f} | {random_dist['p50']:.4f} | {random_dist['p75']:.4f} |",
        f"| A4 semi-hard actual | {semihard_dist['count']} | {semihard_dist['mean']:.4f} | {semihard_dist['p25']:.4f} | {semihard_dist['p50']:.4f} | {semihard_dist['p75']:.4f} |",
        '',
        f'training sim_gap/skip summary: {training_summary}',
        '',
        '## Non-regression Gates',
        '',
        '| constraint | A4 | B2-cont | one-sided deterioration p | result |',
        '|---|---:|---:|---:|---|',
        f"| GarmentSim per-mid mean | {garment_mean_a4:.4f} | {garment_mean_b2:.4f} | {fmt(garment_regression_test['p_value'], 8)} | {'PASS' if garment_pass else 'REGRESSION'} |",
        f"| pose cross-ID variance | {pose_mean_a4:.4f} | {pose_mean_b2:.4f} | {fmt(pose_regression_test['p_value'], 8)} | {'PASS' if pose_pass else 'REGRESSION'} |",
        f"| face detector confidence | {summaries['A4']['detection']['det_conf_mean']:.4f} | {summaries['B2-cont']['detection']['det_conf_mean']:.4f} | {fmt(face_regression_test['p_value'], 8)} | {'PASS' if face_quality_pass else 'REGRESSION'} |",
        f"| face detection rate | {summaries['A4']['detection']['detection_rate']:.2%} | {summaries['B2-cont']['detection']['detection_rate']:.2%} | N/A | {'PASS' if detection_pass else 'REGRESSION'} |",
        '',
        'Face detector confidence is the automatic face-realism proxy available in the frozen shared runner; recognition remains held-out AdaFace IR-101.',
        '',
        '## Automatic Rule Inputs',
        '',
        f'- identity primary condition: {identity_pass}',
        f'- garment non-regression: {garment_pass}',
        f'- pose non-regression: {pose_pass}',
        f'- face detection/realism non-regression: {face_pass}',
        f'- treatment strength increased: {treatment_strengthened}',
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    payload = {
        'verdict': verdict,
        'fairness': fairness,
        'identity': {'pass': identity_pass, 'sim_gain': sim_gain, 'sim_test': sim_test, 'delta_test': delta_test},
        'garment': {'pass': garment_pass, 'a4_mean': garment_mean_a4, 'b2_mean': garment_mean_b2, 'test': garment_regression_test},
        'pose': {'pass': pose_pass, 'a4_mean': pose_mean_a4, 'b2_mean': pose_mean_b2, 'test': pose_regression_test},
        'face': {'pass': face_pass, 'detection_pass': detection_pass, 'quality_pass': face_quality_pass, 'test': face_regression_test},
        'treatment': {'strengthened': treatment_strengthened, 'random': random_dist, 'semihard': semihard_dist, 'relaxation': relaxation_counts},
        'training_identity': training_summary,
    }
    report_path.with_suffix('.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(report_path)
    print(verdict)


if __name__ == '__main__':
    main()
