"""Exploratory M-only condition builder with role-separated I cache extraction.

Not a clean final benchmark: old-val inputs and historical checkpoint are
allowed only for plumbing tests. No old M garment/head features are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

IDENTITY_KEYS = ('pulid_id_embed', 'appearance', 'hair_ref_latents', 'hair_ref_empty',
                 'hair_ref_tokens', 'hair_ref_positions', 'hair_ref_mask',
                 'hair_semantic_tokens', 'hair_semantic_mask')
SOURCE_KEYS = ('pose_latents', 'garment_grid', 'garment_ref_latents', 'head_pose')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_ids(ids, valid):
    if len(ids) != len(set(ids)) or len(ids) < 2:
        raise ValueError('need at least two distinct IDs')
    if any(not isinstance(x, str) or not x.isdigit() for x in ids):
        raise ValueError('IDs must be numeric strings, not paths')
    if not set(ids).issubset(valid):
        raise ValueError('probe inputs must be old-val only')


def identity_payload(cache):
    for key in ('pulid_id_embed', 'appearance'):
        if key not in cache:
            raise ValueError('missing identity key: ' + key)
    return {key: np.array(cache[key], copy=True) for key in IDENTITY_KEYS if key in cache}


def source_path(root, role, sid):
    if role not in ('images/mannequin', 'dwpose/with_head/mannequin') or not sid.isdigit():
        raise ValueError('forbidden source role/path')
    base = Path(root).resolve()
    for suffix in ('.png', '.jpg', '.jpeg'):
        p = base / role / (sid + suffix)
        if p.is_file():
            if not p.resolve().is_relative_to((base / role).resolve()):
                raise ValueError('source symlink escapes approved role')
            return p
    raise FileNotFoundError(f'{role}/{sid}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--ids-json', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=32)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--generate', action='store_true')
    p.add_argument('--steps', type=int, default=20)
    args = p.parse_args()
    if args.limit < 1:
        p.error('--limit must be positive')
    ids = json.loads(args.ids_json.read_text())
    validate_ids(ids, set((args.root / 'splits/val.txt').read_text().split()))
    if len(ids) < args.limit * 2:
        p.error('need disjoint M/I pools: at least 2*limit IDs')
    mids, jids = ids[:args.limit], ids[-args.limit:]
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ('M', 'I', 'masks', 'outputs'):
        (args.out / name).mkdir()
    sys.path.insert(0, str(args.repo))
    import torch
    from PIL import Image
    import cv2
    from diffusers import AutoencoderKL
    from conditions import load_yaml, load_clip_vision, clip_patch_grid_feature
    from build_cache import encode_image_to_packed_latents
    from metrics_v2.parsing import FashnParser
    cfg = load_yaml(args.repo / 'configs/spatial_hair_ab_space.yaml')
    old = args.root / cfg['data']['cache_dir']
    device = torch.device(args.device)
    dtype = torch.bfloat16
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    vae = AutoencoderKL.from_pretrained(cfg['model']['base'], subfolder='vae',
                                       torch_dtype=dtype, local_files_only=True).to(device).eval()
    vae.requires_grad_(False)
    clip = load_clip_vision(cfg['cache']['clip_vision_model'], device, dtype)
    parser = FashnParser(args.repo / 'models/hf/fashn-ai/fashn-human-parser', args.device)
    audit = {'formal_result': False, 'status': 'building', 'roles': [],
             'script_sha256': sha256(__file__), 'ids_sha256': sha256(args.ids_json),
             'head_policy': 'null 7-vector; synthetic H-derived head control disabled',
             'limitations': ['Old I-only feature lineage is code-traced, not rebuilt here.',
                            'Python path checks are not an OS/subprocess sandbox.',
                            'Historical initialization is not clean final training.'],
             'pairs': [{'mid': m, 'jid': j, 'seed': 0} for m, j in zip(mids, jids)],
             'config': cfg}
    textpath = old / 'text/prompt.npz'
    with np.load(textpath, allow_pickle=False) as payload:
        np.savez(args.out / 'prompt.npz', **{k: payload[k] for k in ('prompt_embeds', 'pooled_prompt_embeds')})
    audit['text_source'] = {'path': str(textpath), 'sha256': sha256(textpath), 'role': 'fixed text'}
    for mid, jid in zip(mids, jids):
        image_path = source_path(args.root, 'images/mannequin', mid)
        pose_path = source_path(args.root, 'dwpose/with_head/mannequin', mid)
        image = Image.open(image_path).convert('RGB')
        if image.size != (768, 1024):
            raise ValueError('unexpected native image resolution')
        arr = np.asarray(image)
        labels = parser.predict([arr])[0]
        mask = np.isin(labels, (3, 4, 5, 6, 7, 10)).astype('uint8')
        if mask.mean() < .005:
            raise ValueError('M parser failed for ' + mid)
        Image.fromarray(mask * 255).save(args.out / 'masks' / (mid + '.png'))
        white = np.where(mask[:, :, None] > 0, arr, 255).astype('uint8')
        yy, xx = np.nonzero(mask)
        crop = Image.fromarray(white).crop((int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1))
        eroded = cv2.erode(mask, np.ones((5, 5), np.uint8))
        if not eroded.any():
            raise ValueError('eroded M mask empty for ' + mid)
        canvas = Image.fromarray(np.where(eroded[:, :, None] > 0, arr, 227).astype('uint8'))
        source = {
            'pose_latents': encode_image_to_packed_latents(vae, Image.open(pose_path).convert('RGB'),
                (768, 1024), device, dtype),
            'garment_grid': clip_patch_grid_feature(clip, crop, device, dtype, max_tokens=64),
            'garment_ref_latents': encode_image_to_packed_latents(vae, canvas, (768, 1024), device, dtype),
            'head_pose': np.zeros(7, dtype='float32'),
        }
        np.savez(args.out / 'M' / (mid + '.npz'), **source)
        identity_path = old / 'samples' / (jid + '.npz')
        with np.load(identity_path, allow_pickle=False) as loaded:
            identity = identity_payload(loaded)
        np.savez(args.out / 'I' / (jid + '.npz'), **identity)
        audit['roles'].append({'mid': mid, 'jid': jid,
            'M_inputs': {str(f): sha256(f) for f in (image_path, pose_path)},
            'I_cache': {'path': str(identity_path), 'sha256': sha256(identity_path), 'selected_keys': list(identity)},
            'M_keys': list(source), 'mask_area_fraction': float(mask.mean())})
        print(json.dumps({'phase': 'source_cache', 'mid': mid, 'completed': len(audit['roles'])}), flush=True)
        (args.out / 'audit.json').write_text(json.dumps(audit, indent=2))
    del clip, parser, vae
    torch.cuda.empty_cache()
    audit['status'] = 'source_cache_complete_not_full_isolation_verified'
    (args.out / 'CACHE_READY').write_text('source-only M cache; I lineage traced; formal gate remains open\n')
    if args.generate:
        from train_paired import load_components, WarmupFlowModel, load_checkpoint
        from eval_watcher import generate, decode_tokens
        transformer, controlnet, vae, adapter, pulid, _ = load_components(cfg, device, dtype)
        if vae is None:
            vae = AutoencoderKL.from_pretrained(cfg['model']['base'], subfolder='vae',
                torch_dtype=dtype, local_files_only=True).to(device)
        model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg)
        ckpt = args.root / 'phase1/phase1_spatial_hair_ab_space_r16_8400_768x1024/checkpoints/final'
        load_checkpoint(ckpt, model)
        model.eval()
        audit['checkpoint'] = str(ckpt)
        # Only the new M/I/text payloads enter model generation; no dataset loader.
        for mid, jid in zip(mids, jids):
            with np.load(args.out / 'M' / (mid + '.npz')) as m, np.load(args.out / 'I' / (jid + '.npz')) as j, np.load(args.out / 'prompt.npz') as text:
                batch = {k: torch.from_numpy(np.array(m[k])).float() for k in SOURCE_KEYS}
                batch['garment'] = batch.pop('garment_grid')
                batch.update({k: torch.from_numpy(np.array(j[k])).float() for k in j.files})
                batch.update({k: torch.from_numpy(np.array(text[k])).float() for k in text.files})
            with torch.inference_mode():
                tokens = generate(model, batch, args.steps, seed=0, device=device, dtype=dtype)
                decode_tokens(vae, tokens, cfg['data']['resolution']).save(args.out / 'outputs' / f'{mid}__id{jid}__seed0.png')
            print(json.dumps({'phase': 'generated', 'mid': mid, 'jid': jid}), flush=True)
        audit['status'] = 'inference_complete_not_full_isolation_verified'
    torch.cuda.synchronize(device)
    audit['resources'] = {'seconds': time.monotonic() - start,
                          'gpu_hours': (time.monotonic() - start) / 3600,
                          'peak_gib': torch.cuda.max_memory_allocated(device) / 1024 ** 3}
    (args.out / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({'status': audit['status'], 'resources': audit['resources']}), flush=True)


if __name__ == '__main__':
    main()
