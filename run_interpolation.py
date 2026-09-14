from __future__ import annotations

import argparse
import sys
from pathlib import Path

from conditions import load_yaml
from interp_common import alpha_label, resolve_root_path, run_command, run_official_metrics


def distributed_command(script: str, config: str, nproc: int, extra: list[str]) -> list[str]:
    return [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--standalone',
        '--nproc-per-node',
        str(nproc),
        script,
        '--config',
        config,
        *extra,
    ]


def generate_weights(args: argparse.Namespace) -> None:
    extra = ['--overwrite'] if args.overwrite else []
    if args.smoke:
        run_command([sys.executable, 'interp_weights.py', '--config', args.config, '--device', args.device, '--smoke', *extra])
    else:
        run_command(distributed_command('interp_weights.py', args.config, args.nproc, extra))


def generate_identity(args: argparse.Namespace) -> None:
    prep = [
        sys.executable, 'interp_identity.py', '--config', args.config,
        '--device', args.device, '--prepare-selection',
    ]
    if args.overwrite:
        prep.append('--overwrite')
    run_command(prep)
    extra = ['--overwrite'] if args.overwrite else []
    if args.smoke:
        run_command([sys.executable, 'interp_identity.py', '--config', args.config, '--device', args.device, '--smoke', *extra])
    else:
        run_command(distributed_command('interp_identity.py', args.config, args.nproc, extra))


def metrics_weights(args: argparse.Namespace, cfg: dict) -> None:
    root = Path(cfg['data']['root'])
    icfg = cfg['inference_interpolation']
    wcfg = icfg['weights']
    for alpha in [float(value) for value in wcfg['alphas']]:
        label = alpha_label(alpha)
        run_official_metrics(
            args.config,
            icfg['subset'],
            Path(wcfg['output_root']) / label,
            Path(wcfg['metrics_root']) / label,
            Path(wcfg['metrics_root']) / label / 'report.md',
            args.metrics_device,
        )
        expected = resolve_root_path(root, Path(wcfg['metrics_root']) / label / 'deltaid_summary.json')
        if not expected.exists():
            raise RuntimeError(f'official metrics did not produce {expected}')


def metrics_identity(args: argparse.Namespace) -> None:
    run_command([
        sys.executable,
        '-m',
        'metrics.identity_interp_metrics',
        '--config', args.config,
        '--device', args.metrics_device,
    ])


def report(args: argparse.Namespace, part: str) -> None:
    run_command([sys.executable, 'eval_interp_reports.py', '--config', args.config, '--part', part])


def main() -> None:
    parser = argparse.ArgumentParser(description='Orchestrate both pure-inference interpolation experiments.')
    parser.add_argument('--config', default='configs/interpolation.yaml')
    parser.add_argument('--part', choices=['weights', 'identity', 'all'], default='all')
    parser.add_argument('--stage', choices=['generate', 'metrics', 'report', 'all'], default='all')
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--metrics-device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    parts = ('weights', 'identity') if args.part == 'all' else (args.part,)
    if args.stage in {'generate', 'all'}:
        for part in parts:
            generate_weights(args) if part == 'weights' else generate_identity(args)
    if args.smoke:
        return
    if args.stage in {'metrics', 'all'}:
        for part in parts:
            metrics_weights(args, cfg) if part == 'weights' else metrics_identity(args)
    if args.stage in {'report', 'all'}:
        report(args, args.part)


if __name__ == '__main__':
    main()

