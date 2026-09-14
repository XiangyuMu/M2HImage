from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from conditions import (
    assert_real_controlnet, choose_dtype, clip_patch_grid_feature, find_one, garment_crop,
    get_resolution, head_crop_from_original, load_clip_vision, load_head_pose_token, load_text_embeddings, load_yaml, pack_latents,
    pil_to_tensor, pooled_clip_feature,
    read_ids, resolution_tag, seed_everything, short_hash_path,
)
from dataset import write_cache_manifest
from pulid_flux import PuLIDIdentityEmbedder
from spatial_conditions import (
    dino_hair_dense_tokens,
    dino_hair_semantic_tokens,
    face_hair_appearance_crop,
    load_fashn_labels,
    masked_garment_image,
    masked_hair_image,
)
from synth_head_keypoints import synth_head_control_image


BASE_CACHE_KEYS = (
    'target_latents',
    'pose_latents',
    'pulid_id_embed',
    'appearance',
    'garment_grid',
    'head_pose',
)
SPATIAL_CACHE_KEYS = (
    'garment_ref_latents',
    'hair_ref_latents',
    'hair_ref_empty',
    'pose_latents',
    'pose_synth_latents',
    'appearance',
    'hair_ref_tokens',
    'hair_ref_positions',
    'hair_ref_mask',
    'hair_semantic_tokens',
    'hair_semantic_mask',
)
ALL_CACHE_KEYS = tuple(dict.fromkeys(BASE_CACHE_KEYS + SPATIAL_CACHE_KEYS))


def requested_cache_keys(value: str) -> set[str]:
    names = {item.strip() for item in value.split(',') if item.strip()}
    if names == {'all'}:
        return set(ALL_CACHE_KEYS)
    if names == {'spatial'}:
        return set(SPATIAL_CACHE_KEYS)
    unknown = sorted(names - set(ALL_CACHE_KEYS))
    if unknown:
        raise ValueError(f'unknown cache keys: {unknown}; valid={ALL_CACHE_KEYS} plus all/spatial')
    return names


def encode_image_to_packed_latents(
    vae,
    image: Image.Image,
    resolution,
    device,
    dtype,
    output_dtype: str = 'float32',
) -> np.ndarray:
    tensor = pil_to_tensor(image, resolution).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        posterior = vae.encode(tensor).latent_dist
        latents = posterior.mean
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        packed = pack_latents(latents)
    return packed[0].float().cpu().numpy().astype(output_dtype)


def _load_hair_dino(cfg: dict[str, Any], device: torch.device):
    from metrics.garment_sim import load_dino_model

    hair_cfg = cfg['cache']['hair_dino']
    proxy = {
        'metrics': {
            'garment': {
                'dino_repo_root': hair_cfg['repo_root'],
                'dino_checkpoint': hair_cfg['checkpoint'],
            },
        },
    }
    return load_dino_model(proxy, device)


def build_one(
    root: Path,
    sample_id: str,
    *,
    requested: set[str],
    vae,
    clip_model,
    pulid_embedder: PuLIDIdentityEmbedder | None,
    hair_dino,
    resolution,
    device,
    dtype,
    cfg: dict[str, Any],
    debug: bool = False,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    human_path = find_one(root / 'images/human', sample_id)
    image_keys = {
        'target_latents', 'appearance', 'hair_ref_tokens', 'hair_ref_positions',
        'hair_ref_mask', 'hair_ref_latents', 'hair_ref_empty',
        'hair_semantic_tokens', 'hair_semantic_mask',
    }
    human = Image.open(human_path).convert('RGB') if requested & image_keys else None
    if 'target_latents' in requested:
        assert vae is not None and human is not None
        payload['target_latents'] = encode_image_to_packed_latents(
            vae, human, resolution, device, dtype, output_dtype='float32'
        )
    if 'pose_latents' in requested:
        assert vae is not None
        head_cfg = cfg.get('model', {}).get('spatial_conditions', {}).get('head_control', {})
        folder = 'dwpose/with_head/mannequin' if head_cfg.get('enabled', False) else 'dwpose/without_head/mannequin'
        pose = Image.open(find_one(root / folder, sample_id)).convert('RGB')
        payload['pose_latents'] = encode_image_to_packed_latents(
            vae, pose, resolution, device, dtype, output_dtype='float16'
        )
    if 'pose_synth_latents' in requested:
        assert vae is not None
        head_cfg = cfg.get('model', {}).get('spatial_conditions', {}).get('head_control', {})
        synth_pose = synth_head_control_image(
            root,
            sample_id,
            resolution,
            nose_offset_scale=float(head_cfg.get('nose_offset_scale', 1.62)),
            shoulder_scale=float(head_cfg.get('shoulder_scale', 0.48)),
        )
        payload['pose_synth_latents'] = encode_image_to_packed_latents(
            vae, synth_pose, resolution, device, dtype, output_dtype='float16'
        )
        if debug:
            debug_dir = root / cfg['data']['cache_dir'] / 'debug_synth_head_controls'
            debug_dir.mkdir(parents=True, exist_ok=True)
            synth_pose.save(debug_dir / f'{sample_id}.png')
    if 'pulid_id_embed' in requested:
        if pulid_embedder is None:
            raise RuntimeError('pulid_id_embed requested but PuLID embedder was not loaded')
        face_path = find_one(root / 'derived/face_crops/human', sample_id)
        try:
            value = pulid_embedder.embed_image(face_path)
        except RuntimeError:
            value = pulid_embedder.embed_image(human_path)
        payload['pulid_id_embed'] = value.astype('float32')
    if 'appearance' in requested:
        if clip_model is None or human is None:
            raise RuntimeError('appearance requested but CLIP/human image was not loaded')
        hair_cfg = cfg.get('model', {}).get('spatial_conditions', {}).get('hair', {})
        if hair_cfg.get('clean_appearance', False):
            crop, _ = face_hair_appearance_crop(
                root,
                sample_id,
                neutral_gray=int(cfg['cache'].get('neutral_gray', 127)),
                bbox_pad_fraction=float(cfg['cache'].get('appearance_bbox_pad_fraction', 0.08)),
            )
            debug_name = 'debug_clean_appearance'
        else:
            device_id = device.index if getattr(device, 'type', None) == 'cuda' and device.index is not None else 0
            crop = head_crop_from_original(
                human,
                model_root=cfg['cache'].get('arcface_model_root', '/data/muxiangyu/modelLibrary/insightface'),
                device_id=device_id,
                image_path=human_path,
                helper_python=cfg['cache'].get('arcface_helper_python'),
                helper_script=cfg['cache'].get('arcface_helper_script'),
            )
            debug_name = 'debug_head_crops'
        payload['appearance'] = pooled_clip_feature(clip_model, crop, device, dtype).astype('float16')
        if debug:
            debug_dir = root / cfg['data']['cache_dir'] / debug_name
            debug_dir.mkdir(parents=True, exist_ok=True)
            crop.save(debug_dir / f'{sample_id}.png')
    if 'garment_grid' in requested:
        if clip_model is None:
            raise RuntimeError('garment_grid requested but CLIP was not loaded')
        payload['garment_grid'] = clip_patch_grid_feature(
            clip_model,
            garment_crop(root, sample_id),
            device,
            dtype,
            max_tokens=int(cfg['model']['identity_adapter'].get('garment_grid_max_tokens', 64)),
        ).astype('float32')
    if 'garment_ref_latents' in requested:
        if vae is None:
            raise RuntimeError('garment_ref_latents requested but VAE was not loaded')
        garment_image, _ = masked_garment_image(
            root,
            sample_id,
            resolution,
            int(
                cfg['cache'].get(
                    'reference_neutral_gray',
                    cfg['cache'].get('neutral_gray', 127),
                )
            ),
            erosion_px=int(cfg['cache'].get('reference_mask_erosion_px', 0)),
        )
        payload['garment_ref_latents'] = encode_image_to_packed_latents(
            vae, garment_image, resolution, device, dtype, output_dtype='float16'
        )
    hair_latent_keys = {'hair_ref_latents', 'hair_ref_empty'}
    if requested & hair_latent_keys:
        if not hair_latent_keys.issubset(requested):
            raise RuntimeError(
                f'hair latent cache keys must be rebuilt together: {sorted(hair_latent_keys)}'
            )
        if vae is None:
            raise RuntimeError('hair_ref_latents requested but VAE was not loaded')
        hair_image, _, hair_empty = masked_hair_image(
            root,
            sample_id,
            resolution,
            neutral_gray=int(
                cfg['cache'].get(
                    'reference_neutral_gray',
                    cfg['cache'].get('neutral_gray', 127),
                )
            ),
            hair_label=int(cfg['cache'].get('hair_label', 2)),
            min_area_fraction=float(
                cfg['cache'].get('hair_ref_min_area_fraction', 0.01)
            ),
            erosion_px=int(cfg['cache'].get('reference_mask_erosion_px', 0)),
        )
        payload['hair_ref_latents'] = encode_image_to_packed_latents(
            vae, hair_image, resolution, device, dtype, output_dtype='float16'
        )
        payload['hair_ref_empty'] = np.asarray(int(hair_empty), dtype=np.uint8)
        if debug:
            debug_dir = root / cfg['data']['cache_dir'] / 'debug_hair_reference'
            debug_dir.mkdir(parents=True, exist_ok=True)
            hair_image.save(debug_dir / f'{sample_id}.png')
    hair_keys = {'hair_ref_tokens', 'hair_ref_positions', 'hair_ref_mask'}
    if requested & hair_keys:
        if not hair_keys.issubset(requested):
            raise RuntimeError(f'hair cache keys must be rebuilt together: {sorted(hair_keys)}')
        if hair_dino is None or human is None:
            raise RuntimeError('hair reference tokens requested but DINO/human image was not loaded')
        labels = load_fashn_labels(root, sample_id)
        hair_mask = (labels == int(cfg['cache'].get('hair_label', 2))).astype(np.uint8)
        hair_dino_cfg = cfg['cache']['hair_dino']
        tokens, positions, valid = dino_hair_dense_tokens(
            hair_dino,
            human,
            hair_mask,
            device=device,
            dtype=next(hair_dino.parameters()).dtype,
            image_size=int(hair_dino_cfg.get('image_size', 518)),
            max_tokens=int(cfg['model']['identity_adapter'].get('hair_ref_max_tokens', 64)),
            neutral_gray=int(cfg['cache'].get('neutral_gray', 127)),
        )
        payload['hair_ref_tokens'] = tokens
        payload['hair_ref_positions'] = positions
        payload['hair_ref_mask'] = valid
    semantic_hair_keys = {'hair_semantic_tokens', 'hair_semantic_mask'}
    if requested & semantic_hair_keys:
        if not semantic_hair_keys.issubset(requested):
            raise RuntimeError(
                f'semantic hair cache keys must be rebuilt together: {sorted(semantic_hair_keys)}'
            )
        if hair_dino is None or human is None:
            raise RuntimeError('semantic hair tokens requested but DINO/human image was not loaded')
        labels = load_fashn_labels(root, sample_id)
        hair_mask = (labels == int(cfg['cache'].get('hair_label', 2))).astype(np.uint8)
        hair_dino_cfg = cfg['cache']['hair_dino']
        tokens, valid = dino_hair_semantic_tokens(
            hair_dino,
            human,
            hair_mask,
            device=device,
            dtype=next(hair_dino.parameters()).dtype,
            image_size=int(hair_dino_cfg.get('image_size', 518)),
            max_tokens=int(cfg['model']['identity_adapter'].get('hair_semantic_max_tokens', 32)),
            neutral_gray=int(cfg['cache'].get('neutral_gray', 127)),
        )
        payload['hair_semantic_tokens'] = tokens
        payload['hair_semantic_mask'] = valid
    if 'head_pose' in requested:
        payload['head_pose'] = load_head_pose_token(root, sample_id, dropout_p=0.0).astype('float32')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description='Build or incrementally upgrade the Phase 1 offline cache.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--split', default='train,val,test')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--overwrite-prompt', action='store_true')
    parser.add_argument('--keys', default='all', help='all, spatial, or a comma-separated cache-key list')
    args = parser.parse_args()
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1 and args.num_shards == 1:
        args.num_shards = int(os.environ['WORLD_SIZE'])
        args.shard_index = int(os.environ['RANK'])
        args.device = f"cuda:{int(os.environ.get('LOCAL_RANK', args.shard_index))}"
    cfg = load_yaml(args.config)
    seed_everything(int(cfg['experiment']['seed']))
    root = Path(cfg['data']['root'])
    requested = requested_cache_keys(args.keys)
    if not requested:
        raise RuntimeError('no cache keys requested')
    cache_dir = root / cfg['data']['cache_dir']
    sample_dir = cache_dir / 'samples'
    text_dir = cache_dir / 'text'
    sample_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    error_path = cache_dir / f'cache_errors.rank{args.shard_index:02d}.jsonl'
    if error_path.exists():
        error_path.unlink()
    dtype = choose_dtype(cfg['model']['precision'])
    device = torch.device(args.device)

    assert_real_controlnet(cfg['model']['controlnet'])
    latent_keys = {
        'target_latents', 'pose_latents', 'pose_synth_latents',
        'garment_ref_latents', 'hair_ref_latents',
    }
    vae = None
    if requested & latent_keys:
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(
            cfg['model']['base'], subfolder='vae', torch_dtype=dtype, local_files_only=True
        )
        vae.eval().requires_grad_(False).to(device)
    clip_model = (
        load_clip_vision(cfg['cache']['clip_vision_model'], device, dtype)
        if requested & {'appearance', 'garment_grid'}
        else None
    )
    pulid_embedder = (
        PuLIDIdentityEmbedder(cfg['model']['pulid'], device, dtype)
        if 'pulid_id_embed' in requested
        else None
    )
    hair_keys = {'hair_ref_tokens', 'hair_ref_positions', 'hair_ref_mask'}
    semantic_hair_keys = {'hair_semantic_tokens', 'hair_semantic_mask'}
    hair_dino = _load_hair_dino(cfg, device) if requested & (hair_keys | semantic_hair_keys) else None
    hair_latent_keys = {'hair_ref_latents', 'hair_ref_empty'}

    base_cache_value = cfg['cache'].get('base_cache_dir')
    base_cache_dir = root / base_cache_value if base_cache_value else None
    if base_cache_dir is not None and base_cache_dir.resolve() == cache_dir.resolve():
        raise RuntimeError('cache.base_cache_dir must differ from data.cache_dir')
    base_sample_dir = base_cache_dir / 'samples' if base_cache_dir is not None else None

    prompt_cache = text_dir / 'prompt.npz'
    if args.overwrite_prompt or not prompt_cache.exists():
        tmp_prompt = prompt_cache.with_name(f'{prompt_cache.stem}.rank{args.shard_index}.npz.tmp')
        base_prompt = base_cache_dir / 'text' / 'prompt.npz' if base_cache_dir is not None else None
        if base_prompt is not None and base_prompt.exists() and not args.overwrite_prompt:
            shutil.copy2(base_prompt, tmp_prompt)
        else:
            prompt_embeds, pooled, text_ids = load_text_embeddings(
                cfg['model']['base'], cfg['data']['prompt'], device, dtype
            )
            with tmp_prompt.open('wb') as handle:
                np.savez_compressed(
                    handle,
                    prompt_embeds=prompt_embeds.float().numpy(),
                    pooled_prompt_embeds=pooled.float().numpy(),
                    text_ids=text_ids.float().numpy(),
                )
        tmp_prompt.replace(prompt_cache)

    ids: list[str] = []
    for split in args.split.split(','):
        split = split.strip()
        if not split:
            continue
        ids.extend(read_ids(root / cfg['data'][f'{split}_split']))
    ids = sorted(set(ids))
    debug_ids = set(ids[: int(cfg['cache'].get('spatial_debug_count', 20))])
    if args.limit is not None:
        ids = ids[: args.limit]
    ids = [sid for i, sid in enumerate(ids) if i % args.num_shards == args.shard_index]
    failures = []
    for sid in tqdm(ids, desc=f'cache shard {args.shard_index}/{args.num_shards}'):
        out = sample_dir / f'{sid}.npz'
        try:
            source = out if out.exists() else (
                base_sample_dir / f'{sid}.npz' if base_sample_dir is not None else None
            )
            payload: dict[str, np.ndarray] = {}
            source_keys: set[str] = set()
            needed = set(requested)
            if source is not None and source.exists():
                with np.load(source, allow_pickle=False) as existing:
                    source_keys = set(existing.files)
                    needed = (
                        set(requested)
                        if args.overwrite or not out.exists()
                        else requested - source_keys
                    )
                    if not needed:
                        continue
                    payload = {key: np.asarray(existing[key]) for key in existing.files}
            if needed & hair_keys:
                needed.update(hair_keys)
            if needed & semantic_hair_keys:
                needed.update(semantic_hair_keys)
            if needed & hair_latent_keys:
                needed.update(hair_latent_keys)
            payload.update(build_one(
                root,
                sid,
                requested=needed,
                vae=vae,
                clip_model=clip_model,
                pulid_embedder=pulid_embedder,
                hair_dino=hair_dino,
                resolution=cfg['data']['resolution'],
                device=device,
                dtype=dtype,
                cfg=cfg,
                debug=sid in debug_ids,
            ))
            tmp = out.with_suffix('.npz.tmp')
            with tmp.open('wb') as handle:
                np.savez_compressed(handle, **payload)
            tmp.replace(out)
        except Exception as exc:  # noqa: BLE001
            failures.append({'id': sid, 'error': str(exc)})
            with error_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(failures[-1], ensure_ascii=False) + '\n')
    manifest = {
        'config': args.config,
        'root': str(root),
        'cache_dir': str(cache_dir),
        'resolution': cfg['data']['resolution'],
        'resolution_tag': resolution_tag(cfg['data']['resolution']),
        'pulid': {'repo': cfg['model']['pulid']['repo'], 'weight_path': cfg['model']['pulid']['weight_path']},
        'base_hash': short_hash_path(cfg['model']['base']),
        'controlnet': assert_real_controlnet(cfg['model']['controlnet']),
        'clip_vision_model': cfg['cache']['clip_vision_model'],
        'garment_grid_max_tokens': cfg['model']['identity_adapter'].get('garment_grid_max_tokens', 64),
        'head_crop_debug_dir': str(cache_dir / 'debug_head_crops'),
        'clean_appearance_debug_dir': str(cache_dir / 'debug_clean_appearance'),
        'synthetic_head_debug_dir': str(cache_dir / 'debug_synth_head_controls'),
        'hair_reference_debug_dir': str(cache_dir / 'debug_hair_reference'),
        'requested_keys': sorted(requested),
        'base_cache_dir': str(base_cache_dir) if base_cache_dir is not None else None,
        'spatial_conditions': cfg.get('model', {}).get('spatial_conditions', {}),
        'hair_dino': cfg['cache'].get('hair_dino'),
        'reference_preprocessing': {
            'neutral_gray': int(
                cfg['cache'].get(
                    'reference_neutral_gray',
                    cfg['cache'].get('neutral_gray', 127),
                )
            ),
            'mask_erosion_px': int(
                cfg['cache'].get('reference_mask_erosion_px', 0)
            ),
            'hair_ref_min_area_fraction': float(
                cfg['cache'].get('hair_ref_min_area_fraction', 0.01)
            ),
        },
        'shard_index': args.shard_index,
        'num_shards': args.num_shards,
        'failures': len(failures),
    }
    shard_manifest = cache_dir / f'manifest.rank{args.shard_index:02d}.json'
    shard_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.shard_index == 0:
        write_cache_manifest(cache_dir, manifest)
    if failures:
        raise RuntimeError(
            f'cache build completed with {len(failures)} failures; see {error_path}'
        )


if __name__ == '__main__':
    main()
