"""One frozen checkpoint/protocol arm, fresh output directory, exact pair slice."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from revalidation_common import (SOURCE_KEYS, IDENTITY_KEYS, ARMS, sha256, write_json,
                                load_pairs, install_guard, bait_test, batch_from_arrays)


def main():
    p = argparse.ArgumentParser()
    for n in ('repo', 'root', 'cache', 'pairs', 'config', 'checkpoint', 'out'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--arm', choices=ARMS + ('b2_legacyM_oldI', 'a4_legacyM_oldI'), required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-hours', type=float, default=2)
    a = p.parse_args()
    pairs = load_pairs(a.pairs)[a.start:a.start + a.count]
    if a.start < 0 or len(pairs) != a.count or not pairs or a.steps != 20:
        raise ValueError('invalid fixed pair slice or solver')
    if not (a.cache / 'READY').is_file() or not (a.checkpoint / 'READY').is_file():
        raise RuntimeError('cache/checkpoint not ready')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'outputs').mkdir()
    sys.path.insert(0, str(a.repo))
    import torch
    from diffusers import AutoencoderKL
    from conditions import load_yaml, seed_everything
    from train_paired import load_components, WarmupFlowModel, load_checkpoint
    from eval_watcher import generate, decode_tokens
    cfg = load_yaml(a.config)
    # This is a no-optimizer evaluation. Only VAE loading differs from training.
    cfg['model']['load_vae_in_train'] = True
    device = torch.device(a.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(20260907)
    dtype = torch.bfloat16
    start = time.monotonic()
    input_only = 'inputOnly' in a.arm
    fresh_i = 'freshI' in a.arm
    old = a.root / cfg['data']['cache_dir'] / 'samples'
    audit_cache = json.loads((a.cache / 'audit.json').read_text())
    mh = {r['mid']: r['cache_sha256'] for r in audit_cache['M_roles']}
    ih = {r['jid']: r['cache_sha256'] for r in audit_cache['I_roles']}
    manifest = {'arm': a.arm, 'pairs': pairs, 'config': cfg, 'checkpoint': str(a.checkpoint),
                'checkpoint_sha256': sha256(a.checkpoint / 'trainable.pt'),
                'cache_audit_sha256': sha256(a.cache / 'audit.json'),
                'script_sha256': sha256(__file__), 'pair_manifest_sha256': sha256(a.pairs),
                'training_steps': 0, 'sampler': 'Euler tau1->0 steps20',
                'I_policy': 'shared fresh full-I features' if fresh_i else 'old-I bridge only',
                'condition_policy': 'M/I only' if input_only else 'historical M conditions include H-derived head/garment; diagnostic only',
                'max_hours': a.max_hours, 'status': 'initializing', 'row_input_hashes': []}
    write_json(a.out / 'manifest.json', manifest)
    try:
        transformer, controlnet, vae, adapter, pulid, note = load_components(cfg, device, dtype)
        model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg)
        step = load_checkpoint(a.checkpoint, model)
        if step != 8400:
            raise RuntimeError('checkpoint step changed: ' + str(step))
        model.eval().requires_grad_(False)
        if vae is None:
            vae = AutoencoderKL.from_pretrained(cfg['model']['base'], subfolder='vae',
                    torch_dtype=dtype, local_files_only=True).to(device).eval()
        allowed = []
        if not input_only:
            allowed += [old / f'{r["mid"]}.npz' for r in pairs]
        if not fresh_i:
            allowed += [old / f'{r["jid"]}.npz' for r in pairs]
        events = install_guard(a.root, allowed)
        manifest['read_guard'] = events
        manifest['target_denial_test_passed'] = bait_test(a.root, pairs[0]['mid'])
        manifest['model_note'] = note
        data = {}
        for r in pairs:
            mid, jid = r['mid'], r['jid']
            mp = a.cache / 'M' / f'{mid}.npz' if input_only else old / f'{mid}.npz'
            ip = a.cache / 'I' / f'{jid}.npz' if fresh_i else old / f'{jid}.npz'
            for path, keys, expected_hash in [(mp, SOURCE_KEYS, mh[mid] if input_only else None),
                                              (ip, IDENTITY_KEYS, ih[jid] if fresh_i else None)]:
                if str(path) not in data:
                    digest = sha256(path)
                    if expected_hash and digest != expected_hash:
                        raise ValueError('cache changed: ' + str(path))
                    with np.load(path, allow_pickle=False) as z:
                        data[str(path)] = {k: np.array(z[k]) for k in keys}
                    manifest['row_input_hashes'].append({'path': str(path), 'sha256': digest, 'selected_keys': keys})
            with np.load(a.cache / 'prompt.npz', allow_pickle=False) as z:
                text = {k: np.array(z[k]) for k in ('prompt_embeds', 'pooled_prompt_embeds')}
            batch_from_arrays(data[str(mp)], data[str(ip)], text)
        manifest['initialization_seconds'] = time.monotonic() - start
        manifest['status'] = 'generating'
        write_json(a.out / 'manifest.json', manifest)
        durations = []
        with (a.out / 'rows.jsonl').open('x') as handle, torch.inference_mode():
            for idx, r in enumerate(pairs):
                if time.monotonic() - start > a.max_hours * 3600:
                    raise TimeoutError('generation wall budget exceeded')
                mid, jid = r['mid'], r['jid']
                mp = a.cache / 'M' / f'{mid}.npz' if input_only else old / f'{mid}.npz'
                ip = a.cache / 'I' / f'{jid}.npz' if fresh_i else old / f'{jid}.npz'
                batch = batch_from_arrays(data[str(mp)], data[str(ip)], text)
                tick = time.monotonic()
                tokens = generate(model, batch, a.steps, int(r['seed']), device, dtype)
                if not torch.isfinite(tokens).all():
                    raise ValueError('nonfinite generated latent')
                path = a.out / 'outputs' / f'{mid}__id{jid}__seed{r["seed"]}.png'
                decode_tokens(vae, tokens, cfg['data']['resolution']).save(path)
                torch.cuda.synchronize(device)
                durations.append(time.monotonic() - tick)
                row = {**r, 'branch': a.arm, 'tau': 0, 'k': 1,
                       'path': str(path.resolve()), 'sha256': sha256(path),
                       'clean_carryin_path': str(path.resolve()),
                       'carryin_adapter_note': 'self path only to reuse metric runner; ALL carryin fields discarded',
                       'seconds': durations[-1], 'metric_status': 'pending'}
                handle.write(json.dumps(row, allow_nan=False) + '\n')
                handle.flush()
                progress = {'status': 'generating', 'completed': idx + 1, 'expected': len(pairs),
                            'elapsed_seconds': time.monotonic() - start,
                            'median_image_seconds': float(np.median(durations)),
                            'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3}
                write_json(a.out / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        if len(events['denied_reads']) != 1 or events['denied_subprocesses']:
            raise RuntimeError('unexpected guard event')
        manifest['status'] = 'complete'
        manifest['resources'] = {'seconds': time.monotonic() - start,
                                 'gpu_hours': (time.monotonic() - start) / 3600,
                                 'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
                                 'image_seconds': durations}
        write_json(a.out / 'manifest.json', manifest)
        write_json(a.out / 'summary.json', manifest['resources'])
        (a.out / 'READY').write_text('generation complete; metrics pending\n')
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = repr(exc)
        manifest['seconds'] = time.monotonic() - start
        write_json(a.out / 'manifest.json', manifest)
        write_json(a.out / 'FAILED.json', {'error': repr(exc), 'seconds': time.monotonic() - start})
        raise


if __name__ == '__main__':
    main()
