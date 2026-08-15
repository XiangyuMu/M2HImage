from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from conditions import load_yaml
from eval_a4_gate_report import detection_summary, identity_rows, metric_map
from eval_gate_report import fmt, garment_per_mid, load_json, paired_test, paired_values, pose_variance_per_mid
from sampling_protected import resolve_root_path


PASS_TEXT = (
    'PASS: 推理时服装保护消除了 trade-off 的主要部分；论文 MIXED 讨论升级为 '
    '"trade-off 可被无训练干预中和"'
)
PARTIAL_TEXT = 'PARTIAL: 作为缓解手段与剩余 trade-off 一并报告'
NULL_TEXT = (
    'NULL: 回退主要烙在训练权重而非推理条件注入——本身是机制定位结论，写入分析章；'
    '（future note：B2-cont/A4 的 LoRA 权重插值是另一条推理侧路径，本任务不做）'
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f'required protected gate CSV missing: {path}')
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def headpose_error_map(metrics_dir: Path) -> dict[tuple[str, str, str], float]:
    result = {}
    for row in read_csv(metrics_dir / 'headpose_per_image.csv'):
        if row.get('status') != 'ok':
            continue
        try:
            result[(row['mid'], row['jid'], row['seed'])] = float(np.mean([
                float(row['yaw_abs_err']),
                float(row['pitch_abs_err']),
                float(row['roll_abs_err']),
            ]))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def decide_verdict(
    *,
    all_pass: bool,
    identity_pass: bool,
    garment_mean: float,
    a4_garment_mean: float,
    partial_threshold: float,
    cloth_energy_below_threshold: bool,
) -> str:
    if all_pass and not cloth_energy_below_threshold:
        return PASS_TEXT
    recovered = garment_mean > a4_garment_mean
    if not cloth_energy_below_threshold and recovered and garment_mean >= partial_threshold and identity_pass:
        return PARTIAL_TEXT
    return NULL_TEXT


def summarize_run(metrics_dir: Path) -> dict[str, Any]:
    identities = identity_rows(metrics_dir)
    garments = garment_per_mid(metrics_dir)
    poses = pose_variance_per_mid(metrics_dir)
    return {
        'delta': load_json(metrics_dir / 'deltaid_summary.json'),
        'garment_summary': load_json(metrics_dir / 'garment_summary.json'),
        'headpose': load_json(metrics_dir / 'headpose_summary.json'),
        'detection': detection_summary(metrics_dir),
        'identities': identities,
        'garments': garments,
        'poses': poses,
        'headpose_errors': headpose_error_map(metrics_dir),
    }


def metric_value(run: dict[str, Any], name: str) -> float:
    if name == 'sim_target':
        return float(run['delta']['sim_target_mean'])
    if name == 'delta_id':
        return float(run['delta']['mean'])
    if name == 'garment':
        return float(np.mean(list(run['garments'].values())))
    if name == 'pose':
        return float(np.mean([row['mean_axis_variance'] for row in run['poses'].values()]))
    if name == 'detection_rate':
        return float(run['detection']['detection_rate'])
    if name == 'det_conf':
        return float(run['detection']['det_conf_mean'])
    if name in {'yaw', 'pitch', 'roll'}:
        return float(run['headpose'][f'{name}_mae_mean'])
    raise KeyError(name)


def optional_ablation(metrics_dir: Path) -> dict[str, Any] | None:
    required = [
        metrics_dir / 'deltaid_summary.json',
        metrics_dir / 'garment_summary.json',
        metrics_dir / 'headpose_summary.json',
        metrics_dir / 'deltaid_per_image.csv',
        metrics_dir / 'garment_pairwise_dino.csv',
        metrics_dir / 'headpose_per_image.csv',
    ]
    return summarize_run(metrics_dir) if all(path.exists() for path in required) else None


def garment_gap_recovery_percent(candidate: float, a4: float, b2cont: float) -> float:
    gap = float(b2cont) - float(a4)
    if abs(gap) < 1e-12:
        return float('nan')
    return 100.0 * (float(candidate) - float(a4)) / gap


def evaluate_candidate(
    candidate: dict[str, Any],
    a4: dict[str, Any],
    b2cont: dict[str, Any],
    gate: dict[str, Any],
    *,
    cloth_energy_below_threshold: bool,
) -> dict[str, Any]:
    alpha = float(gate['significance_alpha'])

    candidate_g, b2_g, garment_keys = paired_values(candidate['garments'], b2cont['garments'])
    garment_test = paired_test(candidate_g, b2_g, alternative='less', higher_is_better=True)
    garment_mean = float(np.mean(candidate_g))
    b2_garment_mean = float(np.mean(b2_g))
    a4_garment_mean = metric_value(a4, 'garment')
    garment_pass = (
        garment_mean >= b2_garment_mean - float(gate['garment_tolerance'])
        and garment_test['p_value'] is not None
        and garment_test['p_value'] >= alpha
    )

    candidate_sim, b2_sim, sim_b2_keys = paired_values(
        metric_map(candidate['identities'], 'sim_target'),
        metric_map(b2cont['identities'], 'sim_target'),
    )
    sim_b2_test = paired_test(candidate_sim, b2_sim, alternative='greater', higher_is_better=True)
    candidate_a4_sim, a4_sim, sim_a4_keys = paired_values(
        metric_map(candidate['identities'], 'sim_target'),
        metric_map(a4['identities'], 'sim_target'),
    )
    sim_a4_test = paired_test(candidate_a4_sim, a4_sim, alternative='less', higher_is_better=True)
    candidate_sim_mean = float(np.mean(candidate_sim))
    a4_sim_mean = float(np.mean(a4_sim))
    identity_pass = (
        candidate_sim_mean >= a4_sim_mean - float(gate['identity_a4_tolerance'])
        and float(np.mean(candidate_sim - b2_sim)) > 0.0
        and sim_b2_test['p_value'] is not None
        and sim_b2_test['p_value'] < alpha
    )

    candidate_pose_map = {mid: row['mean_axis_variance'] for mid, row in candidate['poses'].items()}
    b2_pose_map = {mid: row['mean_axis_variance'] for mid, row in b2cont['poses'].items()}
    candidate_pose, b2_pose, pose_keys = paired_values(candidate_pose_map, b2_pose_map)
    pose_test = paired_test(candidate_pose, b2_pose, alternative='greater', higher_is_better=False)
    pose_pass = (
        float(np.mean(candidate_pose))
        <= float(np.mean(b2_pose)) + float(gate['pose_variance_tolerance'])
        and pose_test['p_value'] is not None
        and pose_test['p_value'] >= alpha
    )

    candidate_head, b2_head, head_keys = paired_values(
        candidate['headpose_errors'], b2cont['headpose_errors']
    )
    head_test = paired_test(candidate_head, b2_head, alternative='greater', higher_is_better=False)
    axis_values = {
        axis: {
            'candidate': metric_value(candidate, axis),
            'b2cont': metric_value(b2cont, axis),
        }
        for axis in ('yaw', 'pitch', 'roll')
    }
    axis_pass = all(
        values['candidate'] <= values['b2cont'] + float(gate['headpose_mae_tolerance_deg'])
        for values in axis_values.values()
    )
    headpose_pass = axis_pass and head_test['p_value'] is not None and head_test['p_value'] >= alpha

    candidate_conf, b2_conf, conf_keys = paired_values(
        metric_map(candidate['identities'], 'det_conf'),
        metric_map(b2cont['identities'], 'det_conf'),
    )
    face_test = paired_test(candidate_conf, b2_conf, alternative='less', higher_is_better=True)
    candidate_detection_rate = metric_value(candidate, 'detection_rate')
    detection_pass = candidate_detection_rate >= float(gate['face_detection_min'])
    face_quality_pass = (
        float(np.mean(candidate_conf))
        >= float(np.mean(b2_conf)) - float(gate['face_confidence_tolerance'])
        and face_test['p_value'] is not None
        and face_test['p_value'] >= alpha
    )
    face_pass = detection_pass and face_quality_pass

    all_pass = garment_pass and identity_pass and pose_pass and headpose_pass and face_pass
    verdict = decide_verdict(
        all_pass=all_pass,
        identity_pass=identity_pass,
        garment_mean=garment_mean,
        a4_garment_mean=a4_garment_mean,
        partial_threshold=float(gate['garment_partial_threshold']),
        cloth_energy_below_threshold=cloth_energy_below_threshold,
    )
    return {
        'verdict': verdict,
        'all_pass': all_pass,
        'garment': {
            'pass': garment_pass,
            'candidate': garment_mean,
            'protected': garment_mean,
            'b2cont': b2_garment_mean,
            'a4': a4_garment_mean,
            'gap_recovery_percent': garment_gap_recovery_percent(
                garment_mean, a4_garment_mean, b2_garment_mean
            ),
            'test': garment_test,
            'paired_mids': len(garment_keys),
        },
        'identity': {
            'pass': identity_pass,
            'candidate': candidate_sim_mean,
            'protected': candidate_sim_mean,
            'a4': a4_sim_mean,
            'b2cont': float(np.mean(b2_sim)),
            'drop_vs_a4': float(np.mean(a4_sim - candidate_a4_sim)),
            'gain_vs_b2cont': float(np.mean(candidate_sim - b2_sim)),
            'vs_b2cont': sim_b2_test,
            'vs_a4': sim_a4_test,
            'paired_b2cont_images': len(sim_b2_keys),
            'paired_a4_images': len(sim_a4_keys),
        },
        'pose': {
            'pass': pose_pass,
            'candidate': float(np.mean(candidate_pose)),
            'b2cont': float(np.mean(b2_pose)),
            'test': pose_test,
            'paired_mids': len(pose_keys),
        },
        'headpose': {
            'pass': headpose_pass,
            'axis_pass': axis_pass,
            'candidate': float(np.mean(candidate_head)),
            'b2cont': float(np.mean(b2_head)),
            'axes': axis_values,
            'test': head_test,
            'paired_images': len(head_keys),
        },
        'face': {
            'pass': face_pass,
            'detection_pass': detection_pass,
            'quality_pass': face_quality_pass,
            'candidate_detection_rate': candidate_detection_rate,
            'b2cont_detection_rate': metric_value(b2cont, 'detection_rate'),
            'candidate_confidence': float(np.mean(candidate_conf)),
            'b2cont_confidence': float(np.mean(b2_conf)),
            'test': face_test,
            'paired_images': len(conf_keys),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Preregistered A4 protected-sampling inference gate.')
    parser.add_argument('--config', default='configs/a4_protected.yaml')
    parser.add_argument('--protected-metrics', default=None)
    parser.add_argument('--tau-metrics', default=None)
    parser.add_argument('--a4-metrics', default=None)
    parser.add_argument('--b2cont-metrics', default=None)
    parser.add_argument('--b2p-metrics', default=None)
    parser.add_argument('--analysis-summary', default=None)
    parser.add_argument('--report', default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    pcfg = cfg['inference_protection']
    refs = pcfg['references']
    tau_cfg = pcfg['ablations']['tau_window']
    if not bool(tau_cfg.get('co_primary', False)):
        raise RuntimeError('tau_window must be marked co_primary before running the dual-primary gate')
    metric_dirs = {
        'All-step protected': resolve_root_path(root, args.protected_metrics or pcfg['metrics_dir']),
        'Tau-window protected': resolve_root_path(
            root, args.tau_metrics or tau_cfg['metrics_dir']
        ),
        'A4': resolve_root_path(root, args.a4_metrics or refs['a4_metrics_dir']),
        'B2-cont': resolve_root_path(root, args.b2cont_metrics or refs['b2cont_metrics_dir']),
        "B2' ref": resolve_root_path(root, args.b2p_metrics or refs['b2p_metrics_dir']),
    }
    runs = {name: summarize_run(path) for name, path in metric_dirs.items()}
    a4, b2 = runs['A4'], runs['B2-cont']
    gate = pcfg['gate']

    analysis_path = resolve_root_path(
        root,
        args.analysis_summary or Path(pcfg['analysis']['output_dir']) / 'analysis_summary.json',
    )
    analysis = load_json(analysis_path)
    energy_null = bool(analysis['cloth_energy_below_threshold'])
    evaluations = {
        name: evaluate_candidate(
            runs[name],
            a4,
            b2,
            gate,
            cloth_energy_below_threshold=energy_null,
        )
        for name in ('All-step protected', 'Tau-window protected')
    }

    ablation_dirs = {
        'scale03 100': resolve_root_path(root, pcfg['ablations']['scale03']['metrics_dir']),
    }
    ablations = {name: optional_ablation(path) for name, path in ablation_dirs.items()}
    report_path = resolve_root_path(root, args.report or gate['report'])
    columns = ['All-step protected', 'Tau-window protected', 'A4', 'B2-cont', "B2' ref"]
    lines = [
        '# A4 Inference-time Garment Protection Gate',
        '',
        '## Two Numbers First',
        '',
        '| co-primary configuration | GarmentSim | A4-to-B2-cont gap recovered | sim_target | '
        'sim_target drop vs A4 | verdict |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for name, result in evaluations.items():
        lines.append(
            f"| {name} | {result['garment']['candidate']:.6f} | "
            f"{result['garment']['gap_recovery_percent']:+.2f}% | "
            f"{result['identity']['candidate']:.6f} | {result['identity']['drop_vs_a4']:+.6f} | "
            f"{result['verdict'].split(':', 1)[0]} |"
        )
    lines.extend([
        '',
        'This is a pure-inference post-hoc analysis. No checkpoint was trained or modified, and the frozen A4 '
        'preregistered MIXED verdict remains unchanged. The all-step and analysis-selected tau-window protocols are '
        'co-primary and are each judged independently by the unchanged fixed rule.',
        '',
        '## Five-run Comparison',
        '',
        '| metric | all-step protected | tau-window protected | A4 | B2-cont | B2-prime reference |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    metric_rows = [
        ('held-out sim_target mean', 'sim_target'),
        ('DeltaID mean', 'delta_id'),
        ('GarmentSim per-mid mean', 'garment'),
        ('pose cross-ID mean-axis variance', 'pose'),
        ('face detection rate', 'detection_rate'),
        ('face detector confidence mean', 'det_conf'),
        ('head-pose yaw MAE', 'yaw'),
        ('head-pose pitch MAE', 'pitch'),
        ('head-pose roll MAE', 'roll'),
    ]
    for label, key in metric_rows:
        values = ' | '.join(fmt(metric_value(runs[name], key)) for name in columns)
        lines.append(f'| {label} | {values} |')
    lines.extend([
        '',
        '## Shared Fixed Inputs',
        '',
        f"- Part A Delta v_io cloth_safe energy share: "
        f"{float(analysis['delta_v_io_cloth_energy_share']):.2%}; below 10%: {energy_null}",
        f"- full-gap reference: A4={metric_value(a4, 'garment'):.6f}, "
        f"B2-cont={metric_value(b2, 'garment'):.6f}",
        f"- PASS garment tolerance: B2-cont - {float(gate['garment_tolerance']):.3f}; "
        f"PARTIAL threshold: {float(gate['garment_partial_threshold']):.4f}",
        f"- maximum allowed sim_target drop from A4: {float(gate['identity_a4_tolerance']):.3f}",
    ])
    for name, result in evaluations.items():
        garment = result['garment']
        identity = result['identity']
        lines.extend([
            '',
            f'## {name} Fixed Gate',
            '',
            f"**{result['verdict']}**",
            '',
            f"- GarmentSim={garment['candidate']:.6f}; recovered "
            f"**{garment['gap_recovery_percent']:+.2f}%** of the A4-to-B2-cont gap",
            f"- GarmentSim deterioration p={fmt(garment['test']['p_value'], 8)} over "
            f"{garment['paired_mids']} paired mids; garment condition: "
            f"{'PASS' if garment['pass'] else 'FAIL'}",
            f"- sim_target={identity['candidate']:.6f}; drop from A4="
            f"**{identity['drop_vs_a4']:+.6f}**; allowed drop={float(gate['identity_a4_tolerance']):.3f}",
            f"- sim_target gain vs B2-cont={identity['gain_vs_b2cont']:+.6f}, greater p="
            f"{fmt(identity['vs_b2cont']['p_value'], 10)} over "
            f"{identity['paired_b2cont_images']} images",
            f"- sim_target loss-side p vs A4={fmt(identity['vs_a4']['p_value'], 8)}; "
            f"identity condition: {'PASS' if identity['pass'] else 'FAIL'}",
            '',
            '| non-regression constraint | candidate | B2-cont | deterioration p | result |',
            '|---|---:|---:|---:|---|',
            f"| pose cross-ID variance | {result['pose']['candidate']:.4f} | "
            f"{result['pose']['b2cont']:.4f} | {fmt(result['pose']['test']['p_value'], 8)} | "
            f"{'PASS' if result['pose']['pass'] else 'FAIL'} |",
            f"| mean per-image head-pose error | {result['headpose']['candidate']:.4f} | "
            f"{result['headpose']['b2cont']:.4f} | {fmt(result['headpose']['test']['p_value'], 8)} | "
            f"{'PASS' if result['headpose']['pass'] else 'FAIL'} |",
            f"| face detector confidence | {result['face']['candidate_confidence']:.4f} | "
            f"{result['face']['b2cont_confidence']:.4f} | {fmt(result['face']['test']['p_value'], 8)} | "
            f"{'PASS' if result['face']['quality_pass'] else 'FAIL'} |",
            f"| face detection rate | {result['face']['candidate_detection_rate']:.2%} | "
            f"{result['face']['b2cont_detection_rate']:.2%} | N/A | "
            f"{'PASS' if result['face']['detection_pass'] else 'FAIL'} |",
        ])
    lines.extend([
        '',
        'Head-pose non-regression additionally requires each yaw/pitch/roll MAE to remain within the fixed 0.5 degree '
        'tolerance. Face confidence is the frozen shared face-realism proxy; held-out AdaFace remains confined to the '
        'metric runner.',
        '',
        '## Ablation (Not Used for Either Co-primary Verdict)',
        '',
        '| ablation | images | sim_target | DeltaID | GarmentSim | yaw/pitch/roll MAE |',
        '|---|---:|---:|---:|---:|---|',
    ])
    for name, run in ablations.items():
        if run is None:
            lines.append(f'| {name} | not run | N/A | N/A | N/A | N/A |')
            continue
        count = int(run['delta'].get('count', 0))
        pose_text = '/'.join(fmt(metric_value(run, axis)) for axis in ('yaw', 'pitch', 'roll'))
        lines.append(
            f"| {name} | {count} | {fmt(metric_value(run, 'sim_target'))} | "
            f"{fmt(metric_value(run, 'delta_id'))} | {fmt(metric_value(run, 'garment'))} | {pose_text} |"
        )
    lines.extend([
        '',
        '## Fixed Rule Inputs',
        '',
        f"- all-step all PASS constraints: {evaluations['All-step protected']['all_pass']}",
        f"- tau-window all PASS constraints: {evaluations['Tau-window protected']['all_pass']}",
        f'- Part A null signal: {energy_null}',
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    payload = {
        # Keep the original all-step fields for consumers of the first report schema.
        'verdict': evaluations['All-step protected']['verdict'],
        'verdicts': {name: result['verdict'] for name, result in evaluations.items()},
        'analysis': analysis,
        'metrics_dirs': {name: str(path) for name, path in metric_dirs.items()},
        'co_primary': evaluations,
        'garment': evaluations['All-step protected']['garment'],
        'identity': evaluations['All-step protected']['identity'],
        'pose': evaluations['All-step protected']['pose'],
        'headpose': evaluations['All-step protected']['headpose'],
        'face': evaluations['All-step protected']['face'],
        'ablations': {
            name: str(ablation_dirs[name]) if run is not None else None
            for name, run in ablations.items()
        },
    }
    report_path.with_suffix('.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    print(report_path)
    for name, result in evaluations.items():
        print(f"{name}: {result['verdict']}")


if __name__ == '__main__':
    main()
