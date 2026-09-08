"""Rebuild only M and I conditions required by the historical A4/B2 models.

I is regenerated from the supplied full reference image, with no derived crops
or old feature files. Both protocol arms use this same I. An old-I bridge is
evaluated separately. No H_mid is read, even during cache building.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from revalidation_common import (sha256, write_json, load_pairs, image_path,
                                install_guard, bait_test, validate_payload,
                                SOURCE_KEYS, IDENTITY_KEYS)


def main():
    p = argparse.ArgumentParser()
    for name in ('repo', 'root', 'pairs', 'config', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    pairs = load_pairs(a.pairs)
    mids = sorted({r['mid'] for r in pairs})
    jids = sorted({r['jid'] for r in pairs})
    a.out.mkdir(parents=True, exist_ok=False)
    for n in ('M', 'I', 'masks', 'appearance_crops'):
        (a.out / n).mkdir()
    sys.path.insert(0, str(a.repo))
    import torch
    import cv2
    from PIL import Image
    from diffusers import AutoencoderKL
    from conditions import (load_yaml, load_clip_vision, clip_patch_grid_feature,
                            pooled_clip_feature, expanded_head_crop, mask_bbox)
    from build_cache import encode_image_to_packed_latents
    from pulid_flux import PuLIDIdentityEmbedder
    from metrics_v2.parsing import FashnParser
    cfg = load_yaml(a.config)
    if cfg['model'].get('spatial_conditions', {}).get('enabled', False):
        raise ValueError('this cache targets the historical non-spatial model')
    device = torch.device(a.device)
    dtype = torch.bfloat16
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    audit = {'status': 'initializing', 'training_steps': 0, 'formal_result': False,
             'config': cfg, 'pairs': pairs, 'M_roles': [], 'I_roles': [],
             'script_sha256': sha256(__file__), 'pairs_sha256': sha256(a.pairs),
             'I_policy': 'fresh full I RGB -> PuLID; fresh largest-face bbox -> expanded head CLIP; no old crops/features',
             'M_policy': 'M-only mask/garment; original without-head M pose; zero7 head token',
             'limitation': 'Python dataset/subprocess guard; not universal native syscall proof'}
    write_json(a.out / 'audit.json', audit)
    vae = AutoencoderKL.from_pretrained(cfg['model']['base'], subfolder='vae',
            torch_dtype=dtype, local_files_only=True).to(device).eval().requires_grad_(False)
    clip = load_clip_vision(cfg['cache']['clip_vision_model'], device, dtype)
    parser = FashnParser(a.repo / 'models/hf/fashn-ai/fashn-human-parser', a.device)
    embedder = PuLIDIdentityEmbedder(cfg['model']['pulid'], device, dtype)
    # Initialize all libraries before the read guard; only declared input data
    # and the fixed historical text cache are allowed after initialization.
    textpath = a.root / cfg['data']['cache_dir'] / 'text/prompt.npz'
    allowed = [textpath]
    allowed += [image_path(a.root, role, m) for m in mids for role in
                ('images/mannequin', 'dwpose/without_head/mannequin')]
    allowed += [image_path(a.root, 'images/human', j) for j in jids]
    audit['extractor_hashes'] = {}
    for path in [Path(cfg['model']['pulid']['weight_path']),
                 Path(cfg['model']['pulid']['antelopev2_dir']) / 'glintr100.onnx',
                 Path(cfg['model']['pulid']['antelopev2_dir']) / 'scrfd_10g_bnkps.onnx']:
        audit['extractor_hashes'][str(path)] = sha256(path)
    for base in [Path(cfg['cache']['clip_vision_model']),
                 a.repo / 'models/hf/fashn-ai/fashn-human-parser']:
        for f in sorted(base.rglob('*')):
            if f.is_file() and f.suffix in ('.safetensors', '.bin', '.json'):
                audit['extractor_hashes'][str(f)] = sha256(f)
    for base in [Path(cfg['model']['pulid']['repo']) / 'facexlib/weights',
                 Path(cfg['model']['pulid']['repo']) / 'models']:
        if base.exists():
            for f in sorted(base.glob('*.pth')):
                audit['extractor_hashes'][str(f)] = sha256(f)
    events = install_guard(a.root, allowed)
    audit['read_guard'] = events
    audit['allowed_dataset_paths'] = [str(x.resolve()) for x in allowed]
    audit['target_denial_test_passed'] = bait_test(a.root, mids[0])
    try:
        with np.load(textpath, allow_pickle=False) as z:
            text = {k: np.array(z[k]) for k in ('prompt_embeds', 'pooled_prompt_embeds')}
        np.savez(a.out / 'prompt.npz', **text)
        audit['text_source'] = {'path': str(textpath), 'sha256': sha256(textpath)}
        with torch.inference_mode():
            for mid in mids:
                mp = image_path(a.root, 'images/mannequin', mid)
                pp = image_path(a.root, 'dwpose/without_head/mannequin', mid)
                im = Image.open(mp).convert('RGB')
                if im.size != (768, 1024):
                    raise ValueError('unexpected source size')
                arr = np.asarray(im)
                mask = np.isin(parser.predict([arr])[0], (3, 4, 5, 6, 7, 10)).astype(np.uint8)
                if mask.mean() < .005:
                    raise ValueError('empty M mask: ' + mid)
                maskim = Image.fromarray(mask * 255)
                maskim.save(a.out / 'masks' / f'{mid}.png')
                crop = Image.fromarray(np.where(mask[:, :, None], arr, 255).astype(np.uint8)).crop(mask_bbox(maskim))
                payload = {
                    'pose_latents': encode_image_to_packed_latents(vae, Image.open(pp), cfg['data']['resolution'], device, dtype, output_dtype='float16'),
                    'garment_grid': clip_patch_grid_feature(clip, crop, device, dtype, max_tokens=64),
                    'head_pose': np.zeros(7, np.float32)}
                validate_payload(payload, SOURCE_KEYS)
                out = a.out / 'M' / f'{mid}.npz'
                np.savez(out, **payload)
                audit['M_roles'].append({'mid': mid, 'source_sha256': {str(x): sha256(x) for x in (mp, pp)},
                                        'keys': list(payload), 'cache_sha256': sha256(out),
                                        'mask_sha256': sha256(a.out / 'masks' / f'{mid}.png')})
                audit['status'] = 'building_M'
                write_json(a.out / 'audit.json', audit)
                print(json.dumps({'M': len(audit['M_roles']), 'expected': len(mids)}), flush=True)
            for jid in jids:
                ip = image_path(a.root, 'images/human', jid)
                im = Image.open(ip).convert('RGB')
                faces = embedder.pipeline.app.get(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
                if not faces:
                    raise ValueError('reference face missing: ' + jid)
                face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
                bbox = tuple(float(x) for x in face.bbox)
                crop = expanded_head_crop(im, bbox)
                crop.save(a.out / 'appearance_crops' / f'{jid}.png')
                payload = {'pulid_id_embed': embedder.embed_image(im).astype('float32'),
                           'appearance': pooled_clip_feature(clip, crop, device, dtype).astype('float16')}
                validate_payload(payload, IDENTITY_KEYS)
                out = a.out / 'I' / f'{jid}.npz'
                np.savez(out, **payload)
                audit['I_roles'].append({'jid': jid, 'input': str(ip), 'input_sha256': sha256(ip),
                                        'keys': list(payload), 'cache_sha256': sha256(out),
                                        'bbox': bbox, 'det_score': float(face.det_score),
                                        'pulid_input': 'full I RGB, no fallback',
                                        'appearance_crop_sha256': sha256(a.out / 'appearance_crops' / f'{jid}.png')})
                audit['status'] = 'building_I'
                write_json(a.out / 'audit.json', audit)
                print(json.dumps({'I': len(audit['I_roles']), 'expected': len(jids)}), flush=True)
        if len(events['denied_reads']) != 1 or events['denied_subprocesses']:
            raise RuntimeError('unexpected guard event')
        audit['status'] = 'fresh_MI_cache_complete'
        torch.cuda.synchronize(device)
        audit['resources'] = {'seconds': time.monotonic() - start,
                              'gpu_hours': (time.monotonic() - start) / 3600,
                              'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3}
        write_json(a.out / 'audit.json', audit)
        (a.out / 'READY').write_text('fresh M/I cache ready; not clean historical training\n')
    except Exception as exc:
        audit['status'] = 'failed'
        audit['error'] = repr(exc)
        write_json(a.out / 'audit.json', audit)
        raise


if __name__ == '__main__':
    main()
