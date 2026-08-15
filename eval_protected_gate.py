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


def main() -> None:
    parser = argparse.ArgumentParser(description='Preregistered A4 protected-sampling inference gate.')
    parser.add_argument('--config', default='configs/a4_protected.yaml')
    parser.add_argument('--protected-metrics', default=None)
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
    metric_dirs = {
        'A4-protected': resolve_root_path(root, args.protected_metrics or pcfg['metrics_dir']),
        'A4': resolve_root_path(root, args.a4_metrics or refs['a4_metrics_dir']),
        'B2-cont': resolve_root_path(root, args.b2cont_metrics or refs['b2cont_metrics_dir']),
        "B2' ref": resolve_root_path(root, args.b2p_metrics or refs['b2p_metrics_dir']),
    }
    runs = {name: summarize_run(path) for name, path in metric_dirs.items()}
    protected, a4, b2 = runs['A4-protected'], runs['A4'], runs['B2-cont']
    gate = pcfg['gate']
    alpha = float(gate['significance_alpha'])

    prot_g, b2_g, garment_keys = paired_values(protected['garments'], b2['garments'])
    garment_test = paired_test(prot_g, b2_g, alternative='less', higher_is_better=True)
    garment_mean = float(np.mean(prot_g))
    b2_garment_mean = float(np.mean(b2_g))
    a4_garment_mean = metric_value(a4, 'garment')
    garment_pass = (
        garment_mean >= b2_garment_mean - float(gate['garment_tolerance'])
        and garment_test['p_value'] is not None
        and garment_test['p_value'] >= alpha
    )

    prot_sim, b2_sim, sim_b2_keys = paired_values(
        metric_map(protected['identities'], 'sim_target'),
        metric_map(b2['identities'], 'sim_target'),
    )
    sim_b2_test = paired_test(prot_sim, b2_sim, alternative='greater', higher_is_better=True)
    prot_a4_sim, a4_sim, sim_a4_keys = paired_values(
        metric_map(protected['identities'], 'sim_target'),
        metric_map(a4['identities'], 'sim_target'),
    )
    sim_a4_test = paired_test(prot_a4_sim, a4_sim, alternative='less', higher_is_better=True)
    protected_sim_mean = float(np.mean(prot_sim))
    identity_pass = (
        protected_sim_mean >= float(np.mean(a4_sim)) - float(gate['identity_a4_tolerance'])
        and float(np.mean(prot_sim - b2_sim)) > 0.0
        and sim_b2_test['p_value'] is not None
        and sim_b2_test['p_value'] < alpha
    )

    protected_pose_map = {mid: row['mean_axis_variance'] for mid, row in protected['poses'].items()}
    b2_pose_map = {mid: row['mean_axis_variance'] for mid, row in b2['poses'].items()}
    prot_pose, b2_pose, pose_keys = paired_values(protected_pose_map, b2_pose_map)
    pose_test = paired_test(prot_pose, b2_pose, alternative='greater', higher_is_better=False)
    pose_pass = (
        float(np.mean(prot_pose)) <= float(np.mean(b2_pose)) + float(gate['pose_variance_tolerance'])
        and pose_test['p_value'] is not None
        and pose_test['p_value'] >= alpha
    )

    prot_head, b2_head, head_keys = paired_values(protected['headpose_errors'], b2['headpose_errors'])
    head_test = paired_test(prot_head, b2_head, alternative='greater', higher_is_better=False)
    axis_pass = all(
        metric_value(protected, axis)
        <= metric_value(b2, axis) + float(gate['headpose_mae_tolerance_deg'])
        for axis in ('yaw', 'pitch', 'roll')
    )
    headpose_pass = axis_pass and head_test['p_value'] is not None and head_test['p_value'] >= alpha

    prot_conf, b2_conf, conf_keys = paired_values(
        metric_map(protected['identities'], 'det_conf'),
        metric_map(b2['identities'], 'det_conf'),
    )
    face_test = paired_test(prot_conf, b2_conf, alternative='less', higher_is_better=True)
    detection_pass = metric_value(protected, 'detection_rate') >= float(gate['face_detection_min'])
    face_quality_pass = (
        float(np.mean(prot_conf)) >= float(np.mean(b2_conf)) - float(gate['face_confidence_tolerance'])
        and face_test['p_value'] is not None
        and face_test['p_value'] >= alpha
    )
    face_pass = detection_pass and face_quality_pass

    analysis_path = resolve_root_path(
        root,
        args.analysis_summary or Path(pcfg['analysis']['output_dir']) / 'analysis_summary.json',
    )
    analysis = load_json(analysis_path)
    energy_null = bool(analysis['cloth_energy_below_threshold'])
    all_pass = garment_pass and identity_pass and pose_pass and headpose_pass and face_pass
    verdict = decide_verdict(
        all_pass=all_pass,
        identity_pass=identity_pass,
        garment_mean=garment_mean,
        a4_garment_mean=a4_garment_mean,
        partial_threshold=float(gate['garment_partial_threshold']),
        cloth_energy_below_threshold=energy_null,
    )

    ablation_dirs = {
        'tau-window 100': resolve_root_path(root, pcfg['ablations']['tau_window']['metrics_dir']),
        'scale03 100': resolve_root_path(root, pcfg['ablations']['scale03']['metrics_dir']),
    }
    ablations = {name: optional_ablation(path) for name, path in ablation_dirs.items()}
    report_path = resolve_root_path(root, args.report or gate['report'])
    columns = ['A4-protected', 'A4', 'B2-cont', "B2' ref"]
    lines = [
        '# A4 Inference-time Garment Protection Gate',
        '',
        f'**{verdict}**',
        '',
        'This is a pure-inference post-hoc analysis. No checkpoint was trained or modified, and the frozen A4 '
        'preregistered MIXED verdict remains unchanged.',
        '',
        '## Four-run Comparison',
        '',
        '| metric | A4-protected | A4 | B2-cont | B2-prime reference |',
        '|---|---:|---:|---:|---:|',
    ]
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
        '## Preregistered Main Gate',
        '',
        f"- Part A Delta v_io cloth_safe energy share: "
        f"{float(analysis['delta_v_io_cloth_energy_share']):.2%}; below 10%: {energy_null}",
        f"- protected GarmentSim={garment_mean:.6f}, B2-cont={b2_garment_mean:.6f}, "
        f"tolerance={float(gate['garment_tolerance']):.3f}",
        f"- GarmentSim deterioration Wilcoxon p={fmt(garment_test['p_value'], 8)} over "
        f"{len(garment_keys)} paired mids; garment condition: {'PASS' if garment_pass else 'FAIL'}",
        f"- protected sim_target={protected_sim_mean:.6f}, A4={float(np.mean(a4_sim)):.6f}, "
        f"allowed loss={float(gate['identity_a4_tolerance']):.3f}",
        f"- sim_target vs B2-cont gain={float(np.mean(prot_sim - b2_sim)):+.6f}, "
        f"greater p={fmt(sim_b2_test['p_value'], 10)} over {len(sim_b2_keys)} images",
        f"- sim_target vs A4 delta={float(np.mean(prot_a4_sim - a4_sim)):+.6f}, "
        f"loss-side p={fmt(sim_a4_test['p_value'], 8)} over {len(sim_a4_keys)} images; "
        f"identity condition: {'PASS' if identity_pass else 'FAIL'}",
        '',
        '| non-regression constraint | protected | B2-cont | deterioration p | result |',
        '|---|---:|---:|---:|---|',
        f"| pose cross-ID variance | {float(np.mean(prot_pose)):.4f} | {float(np.mean(b2_pose)):.4f} | "
        f"{fmt(pose_test['p_value'], 8)} | {'PASS' if pose_pass else 'FAIL'} |",
        f"| mean per-image head-pose error | {float(np.mean(prot_head)):.4f} | {float(np.mean(b2_head)):.4f} | "
        f"{fmt(head_test['p_value'], 8)} | {'PASS' if headpose_pass else 'FAIL'} |",
        f"| face detector confidence | {float(np.mean(prot_conf)):.4f} | {float(np.mean(b2_conf)):.4f} | "
        f"{fmt(face_test['p_value'], 8)} | {'PASS' if face_quality_pass else 'FAIL'} |",
        f"| face detection rate | {metric_value(protected, 'detection_rate'):.2%} | "
        f"{metric_value(b2, 'detection_rate'):.2%} | N/A | {'PASS' if detection_pass else 'FAIL'} |",
        '',
        'Head-pose non-regression additionally requires each yaw/pitch/roll MAE to remain within the fixed 0.5 degree '
        'tolerance. Face confidence is the frozen shared face-realism proxy; held-out AdaFace remains confined to the '
        'metric runner.',
        '',
        '## Ablations (Not Used for Main Verdict)',
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
        f'- all PASS constraints: {all_pass}',
        f"- partial garment threshold: {float(gate['garment_partial_threshold']):.4f}",
        f'- recovered any garment gap over A4: {garment_mean > a4_garment_mean}',
        f'- identity condition: {identity_pass}',
        f'- Part A null signal: {energy_null}',
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    payload = {
        'verdict': verdict,
        'analysis': analysis,
        'metrics_dirs': {name: str(path) for name, path in metric_dirs.items()},
        'garment': {
            'pass': garment_pass,
            'protected': garment_mean,
            'b2cont': b2_garment_mean,
            'a4': a4_garment_mean,
            'test': garment_test,
        },
        'identity': {
            'pass': identity_pass,
            'protected': protected_sim_mean,
            'a4': float(np.mean(a4_sim)),
            'vs_b2cont': sim_b2_test,
            'vs_a4': sim_a4_test,
        },
        'pose': {'pass': pose_pass, 'test': pose_test, 'paired_mids': len(pose_keys)},
        'headpose': {'pass': headpose_pass, 'test': head_test, 'paired_images': len(head_keys)},
        'face': {
            'pass': face_pass,
            'detection_pass': detection_pass,
            'quality_pass': face_quality_pass,
            'test': face_test,
            'paired_images': len(conf_keys),
        },
        'ablations': {
            name: str(ablation_dirs[name]) if run is not None else None
            for name, run in ablations.items()
        },
    }
    report_path.with_suffix('.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    print(report_path)
    print(verdict)


if __name__ == '__main__':
    main()
