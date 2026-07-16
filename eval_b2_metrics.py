from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from conditions import load_yaml
from metrics.common import MetricUnavailable, expected_rows, fmt, load_json, safe_mean, safe_median, write_json
from metrics.garment_sim import run_garment_sim
from metrics.headpose_mae import run_headpose_mae
from metrics.heldout_id import run_deltaid


def resolve_root_path(root: Path, path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return root / path


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'status': 'not_run', 'reason': f'missing {path}'}
    return load_json(path)


def deltaid_conclusion(summary: dict[str, Any]) -> str:
    if summary.get('status') != 'ok':
        return '⚠ BLOCKED: held-out DeltaID is unavailable; install/configure AdaFace IR-101 or CurricularFace before using B2 as the A2 identity gate.'
    mean_val = summary.get('mean')
    median_val = summary.get('median')
    if mean_val is None:
        return '⚠ BLOCKED: DeltaID produced no valid samples.'
    if abs(float(mean_val)) < 0.05:
        return '⚠ BLOCKER: ΔID ≈ 0, identity injection is likely ineffective; fix adapter before A2.'
    if float(mean_val) > 0:
        return 'ΔID 明显 > 0: adapter 起效，B2 成立，以下数字为 A2 对照基线.'
    return f'⚠ DeltaID mean is negative ({fmt(mean_val)}; median {fmt(median_val)}), generated faces are closer to source identity than target identity.'


def garment_warning(summary: dict[str, Any], threshold: float) -> str:
    if summary.get('status') != 'ok':
        return '⚠ GarmentSim unavailable.'
    value = summary.get('cross_identity_group_mean_mean')
    if value is None:
        return '⚠ GarmentSim produced no valid DINO pairs.'
    if float(value) > float(threshold):
        return '⚠ B2 服装稳定性已高，A2 差分改善空间有限，生死线可能难拉开.'
    return 'GarmentSim leaves measurable room for A2 comparison.'


def b2p_identity_acceptance(summary: dict[str, Any], threshold: float) -> str:
    if summary.get('status') != 'ok':
        return "⚠ B2' identity acceptance unavailable."
    sim_target = summary.get('sim_target_mean')
    delta_mean = summary.get('mean')
    if sim_target is None or delta_mean is None:
        return "⚠ B2' identity acceptance produced no valid summary."
    if float(sim_target) >= float(threshold) and float(delta_mean) > 0:
        return (
            f"B2' identity acceptance PASS: sim_target={fmt(sim_target)} >= {threshold:.3f} "
            f"and DeltaID mean={fmt(delta_mean)} > 0."
        )
    return (
        f"⚠ B2' identity acceptance FAIL: sim_target={fmt(sim_target)} "
        f"(required >= {threshold:.3f}), DeltaID mean={fmt(delta_mean)}."
    )


def garment_baseline_comparison(summary: dict[str, Any], previous: float) -> str:
    if summary.get('status') != 'ok' or summary.get('cross_identity_group_mean_mean') is None:
        return '⚠ GarmentSim comparison to previous B2 is unavailable.'
    current = float(summary['cross_identity_group_mean_mean'])
    delta = current - float(previous)
    interpretation = (
        'The expected drop exposes real room for A2 differential improvement.'
        if delta < 0
        else 'No drop was observed; A2 differentiation space may remain narrow.'
    )
    return (
        f'GarmentSim vs previous dead-adapter B2: current={current:.4f}, '
        f'previous={float(previous):.4f}, delta={delta:+.4f}. {interpretation}'
    )


def mae_line(summary: dict[str, Any]) -> str:
    if summary.get('status') != 'ok':
        return '⚠ Head pose MAE unavailable.'
    return (
        f"Head pose MAE: yaw={fmt(summary.get('yaw_mae_mean'))}, "
        f"pitch={fmt(summary.get('pitch_mae_mean'))}, roll={fmt(summary.get('roll_mae_mean'))}"
    )


def write_report(
    cfg: dict[str, Any],
    subset: dict[str, Any],
    gen_dir: Path,
    out_dir: Path,
    report_path: Path,
) -> None:
    differential = cfg.get('training', {}).get('differential', {})
    is_a4 = bool(
        differential.get('enabled', False)
        and differential.get('identity_loss', {}).get('enabled', False)
    )
    report_title = (
        '# A4 Directed Counterfactual Offline Metric Report'
        if is_a4
        else "# B2' PuLID Adapter-only Baseline Report"
    )
    rows = expected_rows(cfg, subset, gen_dir)
    actual = sum(1 for row in rows if row['path'].exists())
    delta = load_summary(out_dir / 'deltaid_summary.json')
    headpose = load_summary(out_dir / 'headpose_summary.json')
    garment = load_summary(out_dir / 'garment_summary.json')
    threshold = float(cfg.get('metrics', {}).get('garment', {}).get('high_similarity_warning_threshold', 0.92))
    previous_garment = float(cfg.get('metrics', {}).get('garment', {}).get('previous_dead_adapter_baseline', 0.924))
    sim_target_threshold = float(cfg.get('metrics', {}).get('heldout_id', {}).get('sim_target_acceptance', 0.3))

    lines: list[str] = [
        report_title,
        '',
        f"subset: {len(subset.get('mannequins', []))} mannequins / {len(subset.get('identity_pool', []))} identity pool / {len(subset.get('pairs', []))} pairs",
        f"garment type counts: {subset.get('garment_type_counts', {})}",
        f"expected generated images: {len(rows)}",
        f"actual generated images: {actual}",
        f"generation status: {'complete' if actual == len(rows) else 'incomplete'}",
        '',
        '## Metric Provenance',
        '',
        f"Held-out identity recognizer: {delta.get('recognizer', 'AdaFace IR-101')} | status={delta.get('status')} | hash={delta.get('checkpoint_hash', 'n/a')}",
        'Face detection/alignment for DeltaID uses InsightFace RetinaFace only; recognition uses the held-out recognizer when configured.',
        f"DINO garment encoder: {garment.get('dino', 'GSVTON vendored DINOv2 ViT-B/14')} | status={garment.get('status')} | hash={garment.get('dino_checkpoint_hash', 'n/a')}",
        f"Head pose runner: {headpose.get('runner', 'not configured')} | status={headpose.get('status')} | hash={headpose.get('checkpoint_hash', 'n/a')}",
        f"Mask projection: {garment.get('mask_projection', 'source mid cloth_safe mask resized to generated frame')}",
        '',
        '## Held-out DeltaID',
        '',
    ]
    if delta.get('status') == 'ok':
        lines.extend([
            f"valid images: {delta.get('count')} / {len(rows)}; failed: {delta.get('failed')}",
            f"DeltaID mean={fmt(delta.get('mean'))}, median={fmt(delta.get('median'))}",
            f"sim_target mean={fmt(delta.get('sim_target_mean'))}; sim_source mean={fmt(delta.get('sim_source_mean'))}",
            '',
            '| face_size bucket | count | mean | median |',
            '|---|---:|---:|---:|',
        ])
        for name, payload in delta.get('face_size_buckets', {}).items():
            lines.append(f"| {name} | {payload.get('count', 0)} | {fmt(payload.get('mean'))} | {fmt(payload.get('median'))} |")
        lines.extend(['', f"CSV: `{delta.get('csv')}`", f"Histogram: `{delta.get('histogram')}`"])
    else:
        lines.extend([
            f"Status: BLOCKED. {delta.get('reason', 'held-out recognizer unavailable')}",
            f"Manual setup: {delta.get('manual_download', '')}",
        ])

    lines.extend([
        '',
        '## Official GarmentSim',
        '',
    ])
    if garment.get('status') == 'ok':
        lines.extend([
            'Feature: DINOv2 patch tokens pooled only over `cloth_safe` mask patches; mask outside set to gray.',
            f"cross-identity per-mid DINO mean={fmt(garment.get('cross_identity_group_mean_mean'))}, median={fmt(garment.get('cross_identity_group_mean_median'))}",
            f"pairwise DINO mean={fmt(garment.get('pairwise_mean'))}, median={fmt(garment.get('pairwise_median'))}, pairs={garment.get('pair_count')}",
            f"generated-vs-source DINO mean={fmt(garment.get('source_similarity_mean'))}, median={fmt(garment.get('source_similarity_median'))}",
            f"secondary cloth LPIPS mean={fmt(garment.get('lpips_mean'))}, median={fmt(garment.get('lpips_median'))}",
            f"secondary cloth SSIM mean={fmt(garment.get('ssim_mean'))}, median={fmt(garment.get('ssim_median'))}",
            f"CSV pairwise: `{garment.get('csv_pairwise')}`",
            f"CSV source: `{garment.get('csv_source')}`",
        ])
    else:
        lines.append(f"Status: BLOCKED. {garment.get('reason', 'DINO runner unavailable')}")

    lines.extend([
        '',
        '## Head Pose MAE',
        '',
    ])
    if headpose.get('status') == 'ok':
        lines.extend([
            f"valid images: {headpose.get('count')} / {len(rows)}; failed: {headpose.get('failed')}",
            f"yaw MAE mean={fmt(headpose.get('yaw_mae_mean'))}, median={fmt(headpose.get('yaw_mae_median'))}",
            f"pitch MAE mean={fmt(headpose.get('pitch_mae_mean'))}, median={fmt(headpose.get('pitch_mae_median'))}",
            f"roll MAE mean={fmt(headpose.get('roll_mae_mean'))}, median={fmt(headpose.get('roll_mae_median'))}",
            '',
            '| target yaw bucket | count | yaw MAE mean | yaw MAE median |',
            '|---|---:|---:|---:|',
        ])
        for name, payload in headpose.get('yaw_bucket_yaw_mae', {}).items():
            lines.append(f"| {name} | {payload.get('count', 0)} | {fmt(payload.get('mean'))} | {fmt(payload.get('median'))} |")
        lines.extend(['', f"CSV: `{headpose.get('csv')}`", f"Histogram: `{headpose.get('histogram')}`"])
    else:
        lines.extend([
            f"Status: BLOCKED. {headpose.get('reason', '6DRepNet runner unavailable')}",
            f"Manual setup: {headpose.get('manual_setup', '')}",
        ])

    if is_a4:
        lines.extend([
            '',
            '## A4 Readout',
            '',
            'These frozen metrics are inputs to `eval_a4_gate_report.py`; A4 has no standalone verdict against B2-prime.',
            deltaid_conclusion(delta),
            garment_warning(garment, threshold),
            mae_line(headpose),
        ])
    else:
        lines.extend([
            '',
            '## B2 Readout',
            '',
            deltaid_conclusion(delta),
            b2p_identity_acceptance(delta, sim_target_threshold),
            garment_warning(garment, threshold),
            garment_baseline_comparison(garment, previous_garment),
            mae_line(headpose),
        ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(out_dir / 'b2_metric_bundle.json', {
        'deltaid': delta,
        'garment': garment,
        'headpose': headpose,
        'report': str(report_path),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description='Offline official B2 metric runners; never regenerates images.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--subset', required=True)
    parser.add_argument('--gen-dir', required=True)
    parser.add_argument('--metrics', choices=['deltaid', 'headpose', 'garment', 'all'], default='all')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--report', default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    subset_path = resolve_root_path(root, args.subset)
    gen_dir = resolve_root_path(root, args.gen_dir)
    subset = load_json(subset_path)
    out_dir = resolve_root_path(root, args.out_dir or cfg.get('metrics', {}).get('output_dir', 'eval/metrics'))
    report_path = resolve_root_path(root, args.report or cfg.get('eval', {}).get('report_path', 'eval/b2_report.md'))
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = ['deltaid', 'headpose', 'garment'] if args.metrics == 'all' else [args.metrics]
    errors: list[str] = []
    for name in selected:
        try:
            if name == 'deltaid':
                run_deltaid(cfg, subset, gen_dir, out_dir, device=args.device, fail_on_unavailable=args.metrics != 'all')
            elif name == 'headpose':
                run_headpose_mae(cfg, subset, gen_dir, out_dir, device=args.device, fail_on_unavailable=args.metrics != 'all')
            elif name == 'garment':
                run_garment_sim(cfg, subset, gen_dir, out_dir, device=args.device)
        except MetricUnavailable as exc:
            errors.append(f'{name}: {exc}')
        except Exception as exc:  # noqa: BLE001
            if args.metrics != 'all':
                raise
            errors.append(f'{name}: {exc}')
            write_json(out_dir / f'{name}_summary.json', {'status': 'blocked', 'reason': str(exc)})

    write_report(cfg, subset, gen_dir, out_dir, report_path)
    if errors:
        print('metric issues:', file=sys.stderr)
        for err in errors:
            print(f'- {err}', file=sys.stderr)
        if args.metrics != 'all':
            raise SystemExit(2)
    print(f'wrote report={report_path}')


if __name__ == '__main__':
    main()
