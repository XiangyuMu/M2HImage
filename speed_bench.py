from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 1 benchmark helper. Runs train_paired.py for the configured 20-step benchmark.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--nproc', type=int, default=3)
    parser.add_argument('--all-gpus-train', action='store_true')
    args = parser.parse_args()
    from conditions import load_yaml
    cfg = load_yaml(args.config)
    cmd = ['torchrun', f'--nproc_per_node={args.nproc}', 'train_paired.py', '--config', args.config, '--override-total-steps', str(int(cfg['training']['benchmark_steps']))]
    if args.all_gpus_train:
        cmd.append('--all-gpus-train')
    print(' '.join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
