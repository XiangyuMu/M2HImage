from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from build_region_masks_z import SOURCE_KEYS, pool_token_mask
from conditions import choose_dtype, get_resolution, load_yaml, make_image_ids, make_text_ids, seed_everything


ALLOWED_PROTECT_MASKS = {'cloth_safe', 'cloth_safe+body_bg'}
ALLOWED_WEAK_MODES = {'off', 'scale03'}


def resolve_root_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def ensure_batch(tensor: torch.Tensor, ndim_without_batch: int) -> torch.Tensor:
    return tensor.unsqueeze(0) if tensor.ndim == ndim_without_batch else tensor


def load_token_masks(
    root: Path,
    sample_id: str,
    width: int,
    height: int,
    region_masks_z_dir: str | Path = 'derived/region_masks_z',
) -> tuple[dict[str, np.ndarray], str]:
    """Load packed masks, using an exact in-memory projection for uncovered eval IDs."""
    expected = (height // 16) * (width // 16)
    packed_path = resolve_root_path(root, region_masks_z_dir) / f'{sample_id}.npz'
    if packed_path.exists():
        data = np.load(packed_path)
        masks = {key: np.asarray(data[key], dtype=np.float32).reshape(-1) for key in SOURCE_KEYS}
        source = 'derived/region_masks_z'
    else:
        source_path = root / 'derived/region_masks' / f'{sample_id}.npz'
        if not source_path.exists():
            raise FileNotFoundError(f'no region mask for id={sample_id}: {packed_path} or {source_path}')
        data = np.load(source_path)
        masks = {
            output_key: pool_token_mask(np.asarray(data[source_key]), width, height)[0].astype(np.float32)
            for output_key, source_key in SOURCE_KEYS.items()
        }
        source = 'in-memory 16x16 projection from derived/region_masks (no asset written)'
    for key, value in masks.items():
        if value.shape != (expected,):
            raise RuntimeError(f'{sample_id} {key} shape={value.shape}; expected {(expected,)}')
        if not np.isfinite(value).all() or value.min(initial=0.0) < 0.0 or value.max(initial=1.0) > 1.0:
            raise RuntimeError(f'{sample_id} {key} contains invalid values')
        masks[key] = np.clip(value, 0.0, 1.0)
    return masks, source


def normalized_region_partition(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return soft face/cloth/body/rest weights that sum to one per token."""
    stacked = np.stack([
        masks['face_z'].astype(np.float32),
        masks['cloth_safe_z'].astype(np.float32),
        masks['body_bg_z'].astype(np.float32),
    ], axis=0)
    residual = np.clip(1.0 - stacked.sum(axis=0), 0.0, 1.0)[None, :]
    weights = np.concatenate([stacked, residual], axis=0)
    weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-8)
    return {
        'face': weights[0],
        'cloth_safe': weights[1],
        'body_bg': weights[2],
        'other': weights[3],
    }


def build_protect_mask(
    masks: dict[str, np.ndarray],
    protect_mask: str,
    dilation: int,
    width: int,
    height: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if protect_mask not in ALLOWED_PROTECT_MASKS:
        raise ValueError(f'protect_mask must be one of {sorted(ALLOWED_PROTECT_MASKS)}, got {protect_mask}')
    selected = masks['cloth_safe_z'].astype(np.float32).copy()
    if protect_mask == 'cloth_safe+body_bg':
        selected = np.maximum(selected, masks['body_bg_z'].astype(np.float32))
    grid_h, grid_w = height // 16, width // 16
    tensor = torch.from_numpy(selected).to(device=device, dtype=torch.float32).view(1, 1, grid_h, grid_w)
    if int(dilation) > 0:
        kernel = 2 * int(dilation) + 1
        tensor = F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=int(dilation))
    return tensor.reshape(1, grid_h * grid_w, 1).clamp_(0.0, 1.0).to(dtype=dtype)


def prepare_conditions(
    model,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    *,
    appearance_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt = ensure_batch(batch['prompt_embeds'].to(device=device, dtype=dtype), 2)
    pooled = ensure_batch(batch['pooled_prompt_embeds'].to(device=device, dtype=dtype), 1)
    appearance = ensure_batch(batch['appearance'].to(device=device, dtype=dtype), 1)
    garment = ensure_batch(batch['garment'].to(device=device, dtype=dtype), 2)
    head_pose = ensure_batch(batch['head_pose'].to(device=device, dtype=dtype), 1)
    adapter_tokens = model.adapter(appearance, garment, head_pose)
    if float(appearance_scale) != 1.0:
        adapter_tokens = adapter_tokens.clone()
        count = int(model.adapter.appearance_tokens)
        adapter_tokens[:, :count] *= float(appearance_scale)
    return prompt, pooled, torch.cat([prompt, adapter_tokens], dim=1)


def transformer_velocity(
    model,
    z: torch.Tensor,
    tau: torch.Tensor,
    cond_tokens: torch.Tensor,
    pulid_embed: torch.Tensor,
    cn_samples,
    *,
    pooled: torch.Tensor,
    img_ids: torch.Tensor,
    id_weight: float,
) -> torch.Tensor:
    """Inference-only FLUX forward with an explicit PuLID weight."""
    device, dtype = z.device, z.dtype
    embed = ensure_batch(pulid_embed.to(device=device, dtype=dtype), 2)
    model.pulid.set_context(embed, float(id_weight))
    try:
        return model.transformer(
            hidden_states=z,
            encoder_hidden_states=cond_tokens,
            pooled_projections=pooled,
            timestep=tau,
            img_ids=img_ids,
            txt_ids=make_text_ids(cond_tokens.shape[1], device, dtype),
            guidance=torch.full((z.shape[0],), model.guidance_scale, device=device, dtype=dtype),
            joint_attention_kwargs=model.pulid.context_kwargs(),
            controlnet_block_samples=cn_samples[0],
            controlnet_single_block_samples=cn_samples[1],
            return_dict=True,
        ).sample
    finally:
        model.pulid.clear_context()


def weak_settings(mode: str, weak_scale: float = 0.3) -> tuple[float, float]:
    if mode not in ALLOWED_WEAK_MODES:
        raise ValueError(f'weak_mode must be one of {sorted(ALLOWED_WEAK_MODES)}, got {mode}')
    scale = 0.0 if mode == 'off' else float(weak_scale)
    return scale, scale


def tau_is_protected(tau: float, tau_range: tuple[float, float] | list[float]) -> bool:
    low, high = float(tau_range[0]), float(tau_range[1])
    if low > high:
        raise ValueError(f'invalid protect_tau_range={tau_range}')
    return low <= float(tau) <= high


def protected_velocity(v_on: torch.Tensor, v_weak: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return v_on - mask * (v_on - v_weak)


@torch.no_grad()
def generate_protected(
    model,
    batch: dict[str, torch.Tensor],
    masks: dict[str, np.ndarray],
    *,
    steps: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    protect_mask: str = 'cloth_safe',
    protect_dilation: int = 1,
    protect_tau_range: tuple[float, float] | list[float] = (0.0, 1.0),
    weak_mode: str = 'off',
    weak_scale: float = 0.3,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if steps <= 0:
        raise ValueError('steps must be positive')
    generator = torch.Generator(device=device).manual_seed(int(seed))
    z = torch.randn(
        1,
        (model.height // 16) * (model.width // 16),
        64,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    prompt, pooled, cond_on = prepare_conditions(model, batch, device, dtype, appearance_scale=1.0)
    appearance_scale, id_weight = weak_settings(weak_mode, weak_scale)
    _, _, cond_weak = prepare_conditions(model, batch, device, dtype, appearance_scale=appearance_scale)
    img_ids = make_image_ids(model.width, model.height, device, dtype)
    mask = build_protect_mask(
        masks,
        protect_mask,
        protect_dilation,
        model.width,
        model.height,
        device=device,
        dtype=dtype,
    )
    pose = ensure_batch(batch['pose_latents'].to(device=device, dtype=dtype), 2)
    pulid_embed = ensure_batch(batch['pulid_id_embed'].to(device=device, dtype=dtype), 2)
    protected_steps = 0
    weak_forwards = 0
    delta_norms: list[float] = []
    for index in range(steps):
        tau_value = 1.0 - index / steps
        tau = torch.full((1,), tau_value, device=device, dtype=dtype)
        cn_samples = model._controlnet_forward(z, tau, prompt, pooled, pose, img_ids)
        # The on branch uses the unchanged A4 forward path, preserving seed/path parity.
        v_on = model._transformer_forward(
            z,
            tau,
            cond_on,
            pulid_embed,
            cn_samples,
            pooled=pooled,
            img_ids=img_ids,
        )
        if tau_is_protected(tau_value, protect_tau_range):
            v_weak = transformer_velocity(
                model,
                z,
                tau,
                cond_weak,
                pulid_embed,
                cn_samples,
                pooled=pooled,
                img_ids=img_ids,
                id_weight=id_weight,
            )
            delta_norms.append(float((v_on - v_weak).float().square().mean().sqrt().cpu()))
            velocity = protected_velocity(v_on, v_weak, mask)
            protected_steps += 1
            weak_forwards += 1
        else:
            velocity = v_on
        z = z - (1.0 / steps) * velocity
    return z, {
        'seed': int(seed),
        'steps': int(steps),
        'protected_steps': protected_steps,
        'on_forwards': int(steps),
        'weak_forwards': weak_forwards,
        'protect_mask': protect_mask,
        'protect_dilation': int(protect_dilation),
        'protect_tau_range': [float(protect_tau_range[0]), float(protect_tau_range[1])],
        'weak_mode': weak_mode,
        'weak_scale': float(id_weight),
        'mask_mean': float(mask.float().mean().cpu()),
        'identity_delta_rms_mean': float(np.mean(delta_norms)) if delta_norms else None,
    }


def subset_pairs_for_ablation(subset: dict[str, Any], mid_count: int, identities_per_mid: int) -> dict[str, Any]:
    selected_mids = list(subset['mannequins'])[: int(mid_count)]
    counts = {mid: 0 for mid in selected_mids}
    pairs = []
    for row in subset['pairs']:
        mid = row['mannequin_id']
        if mid not in counts or counts[mid] >= int(identities_per_mid):
            continue
        pairs.append(dict(row))
        counts[mid] += 1
    expected = len(selected_mids) * int(identities_per_mid)
    if len(pairs) != expected:
        raise RuntimeError(f'ablation subset expected {expected} pairs, got {len(pairs)}')
    payload = dict(subset)
    payload['mannequins'] = selected_mids
    payload['pairs'] = pairs
    payload['identity_pool'] = sorted({row['identity_id'] for row in pairs})
    payload['garment_types'] = {mid: subset.get('garment_types', {}).get(mid, 'unknown') for mid in selected_mids}
    payload['garment_type_counts'] = {
        name: sum(1 for value in payload['garment_types'].values() if value == name)
        for name in sorted(set(payload['garment_types'].values()))
    }
    payload['protected_ablation_protocol'] = {
        'first_mids': int(mid_count),
        'identities_per_mid': int(identities_per_mid),
        'images': sum(len(row.get('seeds', [0, 1])) for row in pairs),
    }
    return payload


def expected_generation_items(subset: dict[str, Any], default_seeds: list[int]) -> list[dict[str, Any]]:
    return [
        {
            'mid': row['mannequin_id'],
            'jid': row['identity_id'],
            'seed': int(seed),
            'garment_type': row.get('garment_type', 'unknown'),
        }
        for row in subset['pairs']
        for seed in row.get('seeds', default_seeds)
    ]


def read_recommended_tau_range(root: Path, output_dir: str | Path) -> list[float]:
    summary_path = resolve_root_path(root, output_dir) / 'analysis_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f'Part A summary required for tau-window ablation: {summary_path}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    value = summary.get('recommended_protect_tau_range')
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError(f'{summary_path} has no valid recommended_protect_tau_range')
    return [float(value[0]), float(value[1])]


def mode_settings(cfg: dict[str, Any], mode: str, root: Path) -> dict[str, Any]:
    pcfg = cfg['inference_protection']
    settings = {
        'output_dir': pcfg['output_dir'],
        'metrics_dir': pcfg['metrics_dir'],
        'protect_mask': pcfg['protect_mask'],
        'protect_dilation': int(pcfg['protect_dilation']),
        'protect_tau_range': list(pcfg['protect_tau_range']),
        'weak_mode': pcfg['weak_mode'],
        'weak_scale': float(pcfg.get('weak_scale', 0.3)),
        'is_ablation': False,
    }
    if mode == 'tau_window':
        settings['output_dir'] = pcfg['ablations']['tau_window']['output_dir']
        settings['metrics_dir'] = pcfg['ablations']['tau_window']['metrics_dir']
        settings['protect_tau_range'] = read_recommended_tau_range(root, pcfg['analysis']['output_dir'])
        settings['weak_mode'] = 'off'
        settings['is_ablation'] = True
    elif mode == 'scale03':
        settings['output_dir'] = pcfg['ablations']['scale03']['output_dir']
        settings['metrics_dir'] = pcfg['ablations']['scale03']['metrics_dir']
        settings['weak_mode'] = 'scale03'
        settings['is_ablation'] = True
    elif mode != 'main':
        raise ValueError(f'unknown mode={mode}')
    return settings


def write_status(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'mid', 'jid', 'seed', 'path', 'status', 'error', 'seconds', 'mask_source',
        'protect_mask', 'protect_dilation', 'protect_tau_range', 'weak_mode',
        'protected_steps', 'identity_delta_rms_mean',
    ]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})


def run_generation(args: argparse.Namespace) -> None:
    from diffusers import AutoencoderKL

    from eval_b2 import make_cf_batch
    from eval_watcher import decode_tokens
    from train_paired import WarmupFlowModel, load_checkpoint, load_components

    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    pcfg = cfg['inference_protection']
    settings = mode_settings(cfg, args.mode, root)
    subset_path = resolve_root_path(root, args.subset or pcfg['subset'])
    subset = json.loads(subset_path.read_text(encoding='utf-8'))
    if settings['is_ablation']:
        acfg = pcfg['ablations']
        subset = subset_pairs_for_ablation(subset, acfg['mid_count'], acfg['identities_per_mid'])
        ablation_subset_path = resolve_root_path(root, acfg[args.mode]['subset_path'])
        if int(args.shard_index) == 0:
            ablation_subset_path.parent.mkdir(parents=True, exist_ok=True)
            ablation_subset_path.write_text(
                json.dumps(subset, indent=2, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

    items = expected_generation_items(subset, list(cfg['eval'].get('b2_seeds', [0, 1])))
    if args.smoke is not None:
        items = items[: int(args.smoke)]
        output_dir = resolve_root_path(root, args.output_dir or pcfg['smoke_output_dir'])
        num_shards, shard_index = 1, 0
    else:
        output_dir = resolve_root_path(root, args.output_dir or settings['output_dir'])
        num_shards, shard_index = int(args.num_shards), int(args.shard_index)
    mids = list(dict.fromkeys(item['mid'] for item in items))
    assigned = {mid for index, mid in enumerate(mids) if index % num_shards == shard_index}
    items = [item for item in items if item['mid'] in assigned]
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(cfg['experiment']['seed']))
    device = torch.device(args.device)
    dtype = choose_dtype(cfg['model']['precision'])
    transformer, controlnet, vae, adapter, pulid, _ = load_components(cfg, device, dtype)
    if vae is None:
        vae = AutoencoderKL.from_pretrained(
            cfg['model']['base'], subfolder='vae', torch_dtype=dtype, local_files_only=True
        ).to(device).eval()
    model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg)
    checkpoint = Path(args.ckpt or pcfg['checkpoint'])
    load_checkpoint(checkpoint, model)
    model.eval().requires_grad_(False)
    width, height = get_resolution(cfg['data']['resolution'])
    cache = root / cfg['data']['cache_dir'] / 'samples'
    text_cache = root / cfg['data']['cache_dir'] / 'text' / 'prompt.npz'
    mask_cache: dict[str, tuple[dict[str, np.ndarray], str]] = {}
    batch_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    status_rows: list[dict[str, Any]] = []

    for item in items:
        mid, jid, seed = item['mid'], item['jid'], int(item['seed'])
        path = output_dir / f'{mid}__id{jid}__seed{seed}.png'
        base = {'mid': mid, 'jid': jid, 'seed': seed, 'path': str(path), 'error': ''}
        if path.exists() and not args.overwrite:
            status_rows.append({**base, 'status': 'existing'})
            continue
        started = time.perf_counter()
        try:
            if mid not in mask_cache:
                mask_cache[mid] = load_token_masks(
                    root,
                    mid,
                    width,
                    height,
                    cfg['data'].get('region_masks_z_dir', 'derived/region_masks_z'),
                )
            key = (mid, jid)
            if key not in batch_cache:
                batch_cache[key] = make_cf_batch(cache, text_cache, mid, jid)
            tokens, diagnostics = generate_protected(
                model,
                batch_cache[key],
                mask_cache[mid][0],
                steps=int(pcfg.get('steps', cfg['eval']['generate_steps'])),
                seed=seed,
                device=device,
                dtype=dtype,
                protect_mask=settings['protect_mask'],
                protect_dilation=settings['protect_dilation'],
                protect_tau_range=settings['protect_tau_range'],
                weak_mode=settings['weak_mode'],
                weak_scale=settings['weak_scale'],
            )
            decode_tokens(vae, tokens, cfg['data']['resolution']).save(path)
            status_rows.append({
                **base,
                **diagnostics,
                'status': 'ok',
                'seconds': time.perf_counter() - started,
                'mask_source': mask_cache[mid][1],
                'protect_tau_range': json.dumps(settings['protect_tau_range']),
            })
        except Exception as exc:  # noqa: BLE001
            status_rows.append({
                **base,
                'status': 'failed',
                'error': f'{type(exc).__name__}: {exc}',
                'seconds': time.perf_counter() - started,
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    status_base = resolve_root_path(root, pcfg['generation_status'])
    suffix = f'.{args.mode}.smoke{args.smoke}' if args.smoke is not None else f'.{args.mode}.shard{shard_index}'
    status_path = status_base.parent / f'{status_base.name}{suffix}.csv'
    write_status(status_path, status_rows)
    failed = sum(row.get('status') == 'failed' for row in status_rows)
    print(json.dumps({
        'mode': args.mode,
        'shard': [shard_index, num_shards],
        'items': len(items),
        'failed': failed,
        'output_dir': str(output_dir),
        'status_csv': str(status_path),
        'settings': settings,
    }, indent=2))
    if failed:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description='Inference-only identity velocity removal in protected garment regions.')
    parser.add_argument('--config', default='configs/a4_protected.yaml')
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--subset', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode', choices=['main', 'tau_window', 'scale03'], default='main')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--smoke', type=int, default=None, metavar='N')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1 and args.num_shards == 1:
        args.num_shards = int(os.environ['WORLD_SIZE'])
        args.shard_index = int(os.environ['RANK'])
        args.device = f"cuda:{int(os.environ.get('LOCAL_RANK', args.shard_index))}"
    run_generation(args)


if __name__ == '__main__':
    main()
