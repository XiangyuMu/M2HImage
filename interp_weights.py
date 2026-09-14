from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from conditions import load_yaml
from eval_b2 import make_cf_batch
from eval_watcher import decode_tokens, generate
from interp_common import (
    alpha_label,
    apply_trainable_state,
    assert_model_schema,
    assigned_mids,
    interpolate_trainable_state,
    load_endpoint_states,
    load_inference_model,
    read_json,
    resolve_root_path,
    runtime_shard,
    subset_for_mids,
    write_csv,
    write_json,
)


STATUS_FIELDS = (
    'alpha', 'mid', 'jid', 'seed', 'path', 'status', 'error', 'seconds', 'rank', 'world_size'
)


def generation_rows(subset: dict[str, Any], default_seeds: list[int]) -> list[dict[str, Any]]:
    return [
        {
            'mid': str(pair['mannequin_id']),
            'jid': str(pair['identity_id']),
            'seed': int(seed),
        }
        for pair in subset['pairs']
        for seed in pair.get('seeds', default_seeds)
    ]


def run(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    icfg = cfg['inference_interpolation']
    wcfg = icfg['weights']
    subset_path = resolve_root_path(root, args.subset or icfg['subset'])
    subset = read_json(subset_path)
    rank, world_size, device = runtime_shard(args)

    if args.smoke:
        mids = list(subset['mannequins'])[: int(wcfg['smoke_mid_count'])]
        subset = subset_for_mids(subset, mids)
        alphas = [float(args.alpha if args.alpha is not None else wcfg['smoke_alpha'])]
        output_root = resolve_root_path(root, args.output_root or wcfg['smoke_output_root'])
        rank, world_size = 0, 1
    else:
        alphas = [float(args.alpha)] if args.alpha is not None else [float(value) for value in wcfg['alphas']]
        output_root = resolve_root_path(root, args.output_root or wcfg['output_root'])

    ordered_mids = [str(mid) for mid in subset['mannequins']]
    keep = assigned_mids(ordered_mids, rank, world_size)
    rows = [
        row for row in generation_rows(subset, list(cfg['eval'].get('b2_seeds', [0, 1])))
        if row['mid'] in keep
    ]
    cache = root / cfg['data']['cache_dir'] / 'samples'
    text_cache = root / cfg['data']['cache_dir'] / 'text' / 'prompt.npz'
    status_root = resolve_root_path(root, wcfg['status_root'])
    output_root.mkdir(parents=True, exist_ok=True)

    b2_state, a4_state, endpoint_metadata = load_endpoint_states(cfg)
    model, vae, dtype = load_inference_model(cfg, device)
    assert_model_schema(model, b2_state)
    if rank == 0:
        manifest = {
            'protocol': 'theta(alpha)=(1-alpha)*theta_B2cont+alpha*theta_A4',
            'interpolated_groups': ['transformer_lora', 'adapter'],
            'frozen_groups': ['base_transformer', 'controlnet', 'pulid', 'vae'],
            'alphas': alphas,
            'subset': str(subset_path),
            'steps': int(icfg['steps']),
            'endpoint_validation': endpoint_metadata,
        }
        write_json(output_root / 'interpolation_manifest.json', manifest)
        print(json.dumps(manifest, indent=2), flush=True)

    batch_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    all_status: list[dict[str, Any]] = []
    for alpha in alphas:
        label = alpha_label(alpha)
        state = interpolate_trainable_state(b2_state, a4_state, alpha)
        apply_trainable_state(model, state)
        del state
        out_dir = output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        alpha_status: list[dict[str, Any]] = []
        for row in rows:
            mid, jid, seed = row['mid'], row['jid'], row['seed']
            output = out_dir / f'{mid}__id{jid}__seed{seed}.png'
            base = {
                'alpha': alpha,
                'mid': mid,
                'jid': jid,
                'seed': seed,
                'path': str(output),
                'error': '',
                'rank': rank,
                'world_size': world_size,
            }
            if output.exists() and not args.overwrite:
                alpha_status.append({**base, 'status': 'existing', 'seconds': 0.0})
                continue
            started = time.perf_counter()
            try:
                key = (mid, jid)
                if key not in batch_cache:
                    batch_cache[key] = make_cf_batch(cache, text_cache, mid, jid)
                tokens = generate(
                    model,
                    batch_cache[key],
                    int(icfg['steps']),
                    seed=seed,
                    device=device,
                    dtype=dtype,
                )
                decode_tokens(vae, tokens, cfg['data']['resolution']).save(output)
                alpha_status.append({**base, 'status': 'ok', 'seconds': time.perf_counter() - started})
            except Exception as exc:  # noqa: BLE001
                alpha_status.append({
                    **base,
                    'status': 'failed',
                    'error': f'{type(exc).__name__}: {exc}',
                    'seconds': time.perf_counter() - started,
                })
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        status_path = status_root / f'{label}.shard{rank}.csv'
        if args.smoke:
            status_path = status_root / f'{label}.smoke.csv'
        write_csv(status_path, alpha_status, list(STATUS_FIELDS))
        all_status.extend(alpha_status)
        print(json.dumps({
            'alpha': alpha,
            'rank': rank,
            'world_size': world_size,
            'items': len(alpha_status),
            'failed': sum(row['status'] == 'failed' for row in alpha_status),
            'output_dir': str(out_dir),
            'status_csv': str(status_path),
        }, indent=2), flush=True)

    failed = sum(row['status'] == 'failed' for row in all_status)
    if failed:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description='Pure-inference B2-cont/A4 trainable-weight interpolation.')
    parser.add_argument('--config', default='configs/interpolation.yaml')
    parser.add_argument('--subset', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--output-root', default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()

