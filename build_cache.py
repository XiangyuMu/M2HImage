from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from conditions import (
    arcface_embedding_from_path, assert_real_controlnet, choose_dtype, find_one, garment_crop, load_clip_vision,
    load_head_pose_token, load_text_embeddings, load_yaml, pack_latents, pil_to_tensor, pooled_clip_feature,
    read_ids, seed_everything, short_hash_path,
)
from dataset import write_cache_manifest


def encode_image_to_packed_latents(vae, image: Image.Image, resolution: int, device, dtype) -> np.ndarray:
    tensor = pil_to_tensor(image, resolution).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        posterior = vae.encode(tensor).latent_dist
        latents = posterior.mean
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        packed = pack_latents(latents)
    return packed[0].float().cpu().numpy().astype('float32')


def build_one(root: Path, sample_id: str, vae, clip_model, resolution: int, device, dtype, cfg: dict) -> dict[str, np.ndarray]:
    human = Image.open(find_one(root / 'images/human', sample_id)).convert('RGB')
    pose = Image.open(find_one(root / 'dwpose/without_head/mannequin', sample_id)).convert('RGB')
    face_path = find_one(root / 'derived/face_crops/human', sample_id)
    face = Image.open(face_path).convert('RGB')
    target_latents = encode_image_to_packed_latents(vae, human, resolution, device, dtype)
    pose_latents = encode_image_to_packed_latents(vae, pose, resolution, device, dtype)
    device_id = device.index if getattr(device, 'type', None) == 'cuda' and device.index is not None else 0
    identity = arcface_embedding_from_path(
        face_path,
        helper_python=cfg['cache'].get('arcface_helper_python'),
        helper_script=cfg['cache'].get('arcface_helper_script'),
        model_root=cfg['cache'].get('arcface_model_root', '/data/muxiangyu/modelLibrary/insightface'),
        device_id=device_id,
    )
    appearance = pooled_clip_feature(clip_model, face, device, dtype)
    garment = pooled_clip_feature(clip_model, garment_crop(root, sample_id), device, dtype)
    head_pose = load_head_pose_token(root, sample_id)
    return {
        'target_latents': target_latents,
        'pose_latents': pose_latents,
        'identity': identity.astype('float32'),
        'appearance': appearance.astype('float32'),
        'garment': garment.astype('float32'),
        'head_pose': head_pose.astype('float32'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Build Phase 1 offline cache: VAE latents, text embeddings, garment/id/head tokens.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--split', default='train,val,test')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1 and args.num_shards == 1:
        args.num_shards = int(os.environ['WORLD_SIZE'])
        args.shard_index = int(os.environ['RANK'])
        args.device = f"cuda:{int(os.environ.get('LOCAL_RANK', args.shard_index))}"
    cfg = load_yaml(args.config)
    seed_everything(int(cfg['experiment']['seed']))
    root = Path(cfg['data']['root'])
    cache_dir = root / cfg['data']['cache_dir']
    sample_dir = cache_dir / 'samples'
    text_dir = cache_dir / 'text'
    sample_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    dtype = choose_dtype(cfg['model']['precision'])
    device = torch.device(args.device)

    assert_real_controlnet(cfg['model']['controlnet'])
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(cfg['model']['base'], subfolder='vae', torch_dtype=dtype, local_files_only=True)
    vae.eval().requires_grad_(False).to(device)
    clip_model = load_clip_vision(cfg['cache']['clip_vision_model'], device, dtype)

    prompt_cache = text_dir / 'prompt.npz'
    if args.overwrite or not prompt_cache.exists():
        prompt_embeds, pooled, text_ids = load_text_embeddings(cfg['model']['base'], cfg['data']['prompt'], device, dtype)
        tmp_prompt = prompt_cache.with_name(f'{prompt_cache.stem}.rank{args.shard_index}.npz.tmp')
        with tmp_prompt.open('wb') as handle:
            np.savez_compressed(handle, prompt_embeds=prompt_embeds.float().numpy(), pooled_prompt_embeds=pooled.float().numpy(), text_ids=text_ids.float().numpy())
        tmp_prompt.replace(prompt_cache)

    ids: list[str] = []
    for split in args.split.split(','):
        split = split.strip()
        if not split:
            continue
        ids.extend(read_ids(root / cfg['data'][f'{split}_split']))
    ids = sorted(set(ids))
    if args.limit is not None:
        ids = ids[: args.limit]
    ids = [sid for i, sid in enumerate(ids) if i % args.num_shards == args.shard_index]
    failures = []
    for sid in tqdm(ids, desc=f'cache shard {args.shard_index}/{args.num_shards}'):
        out = sample_dir / f'{sid}.npz'
        if out.exists() and not args.overwrite:
            continue
        try:
            payload = build_one(root, sid, vae, clip_model, int(cfg['data']['resolution']), device, dtype, cfg)
            tmp = out.with_suffix('.npz.tmp')
            with tmp.open('wb') as handle:
                np.savez_compressed(handle, **payload)
            tmp.replace(out)
        except Exception as exc:  # noqa: BLE001
            failures.append({'id': sid, 'error': str(exc)})
            with (cache_dir / 'cache_errors.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(failures[-1], ensure_ascii=False) + '\n')
    manifest = {
        'config': args.config,
        'root': str(root),
        'resolution': cfg['data']['resolution'],
        'base_hash': short_hash_path(cfg['model']['base']),
        'controlnet': assert_real_controlnet(cfg['model']['controlnet']),
        'clip_vision_model': cfg['cache']['clip_vision_model'],
        'shard_index': args.shard_index,
        'num_shards': args.num_shards,
        'failures': len(failures),
    }
    write_cache_manifest(cache_dir, manifest)
    if failures:
        raise RuntimeError(f'cache build completed with {len(failures)} failures; see {cache_dir / "cache_errors.jsonl"}')


if __name__ == '__main__':
    main()
