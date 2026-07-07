from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from conditions import (
    arcface_embedding_from_path, choose_dtype, find_one, load_yaml, seed_everything, unpack_latents,
)
from dataset import PairedWarmupDataset
from train_paired import WarmupFlowModel, load_checkpoint, load_components


def decode_tokens(vae, tokens: torch.Tensor, resolution: int) -> Image.Image:
    latents = unpack_latents(tokens, resolution)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0][0]
    arr = ((image.float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype('uint8')
    return Image.fromarray(arr)


def generate(model: WarmupFlowModel, batch: dict, steps: int, seed: int, device, dtype) -> torch.Tensor:
    z = torch.randn(1, (model.resolution // 16) ** 2, 64, device=device, dtype=dtype, generator=torch.Generator(device=device).manual_seed(seed))
    # Deterministic Euler solver for the trained flow field: integrate from tau=1 to tau=0.
    for i in range(steps):
        tau = torch.full((1,), 1.0 - i / steps, device=device, dtype=dtype)
        model_timestep = tau
        local = batch
        prompt = local['prompt_embeds'].to(device=device, dtype=dtype).unsqueeze(0) if local['prompt_embeds'].ndim == 2 else local['prompt_embeds'].to(device=device, dtype=dtype)
        pooled = local['pooled_prompt_embeds'].to(device=device, dtype=dtype).unsqueeze(0) if local['pooled_prompt_embeds'].ndim == 1 else local['pooled_prompt_embeds'].to(device=device, dtype=dtype)
        adapter_tokens = model.adapter(local['identity'].to(device=device, dtype=dtype).unsqueeze(0), local['appearance'].to(device=device, dtype=dtype).unsqueeze(0), local['garment'].to(device=device, dtype=dtype).unsqueeze(0), local['head_pose'].to(device=device, dtype=dtype).unsqueeze(0))
        from conditions import make_image_ids, make_text_ids
        img_ids = make_image_ids(model.resolution, device, dtype)
        guidance_scale = float(model.cfg.get('eval', {}).get('guidance_scale', 3.5))
        with torch.no_grad():
            cn = model.controlnet(hidden_states=z, controlnet_cond=local['pose_latents'].to(device=device, dtype=dtype).unsqueeze(0), controlnet_mode=torch.full((1, 1), model.control_mode, device=device, dtype=torch.long), conditioning_scale=model.controlnet_scale, encoder_hidden_states=prompt, pooled_projections=pooled, timestep=model_timestep, img_ids=img_ids, txt_ids=make_text_ids(prompt.shape[1], device, dtype), guidance=torch.full((1,), guidance_scale, device=device, dtype=dtype), return_dict=True)
            v = model.transformer(hidden_states=z, encoder_hidden_states=torch.cat([prompt, adapter_tokens], dim=1), pooled_projections=pooled, timestep=model_timestep, img_ids=img_ids, txt_ids=make_text_ids(prompt.shape[1] + adapter_tokens.shape[1], device, dtype), guidance=torch.full((1,), guidance_scale, device=device, dtype=dtype), controlnet_block_samples=cn.controlnet_block_samples, controlnet_single_block_samples=cn.controlnet_single_block_samples, return_dict=True).sample
        z = z - (1.0 / steps) * v
    return z


def make_panel(root: Path, sample_id: str, generated: Image.Image, resolution: int, swap: Image.Image | None = None, swap_id: str | None = None) -> Image.Image:
    parts = [
        Image.open(find_one(root / 'images/mannequin', sample_id)).convert('RGB').resize((resolution, resolution)),
        Image.open(find_one(root / 'dwpose/without_head/mannequin', sample_id)).convert('RGB').resize((resolution, resolution)),
        generated.resize((resolution, resolution)),
    ]
    labels = ['m_i', 'pose', 'generated c_i']
    if swap is not None:
        parts.append(swap.resize((resolution, resolution)))
        labels.append(f'swap c_{swap_id}')
    parts.append(Image.open(find_one(root / 'images/human', sample_id)).convert('RGB').resize((resolution, resolution)))
    labels.append('h_i')
    canvas = Image.new('RGB', (resolution * len(parts), resolution + 24), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, image in enumerate(parts):
        canvas.paste(image, (i * resolution, 24))
        draw.text((i * resolution + 8, 6), labels[i], fill=(0, 0, 0))
    return canvas


def swap_identity(batch: dict, donor: dict) -> dict:
    out = dict(batch)
    out['identity'] = donor['identity']
    out['appearance'] = donor['appearance']
    return out


def embedding_for_image(path: Path, cfg: dict, device_index: int) -> np.ndarray | None:
    try:
        return arcface_embedding_from_path(
            path,
            helper_python=cfg['cache'].get('arcface_helper_python'),
            helper_script=cfg['cache'].get('arcface_helper_script'),
            model_root=cfg['cache'].get('arcface_model_root', '/data/muxiangyu/modelLibrary/insightface'),
            device_id=device_index,
        )
    except Exception:
        return None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_once(config_path: str, ckpt: Path, device: str) -> None:
    cfg = load_yaml(config_path)
    seed_everything(int(cfg['experiment']['seed']))
    torch_device = torch.device(device)
    device_index = torch_device.index if torch_device.type == 'cuda' and torch_device.index is not None else 0
    dtype = choose_dtype(cfg['model']['precision'])
    transformer, controlnet, vae, adapter, _ = load_components(cfg, torch_device, dtype)
    if vae is None:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(cfg['model']['base'], subfolder='vae', torch_dtype=dtype, local_files_only=True).to(torch_device)
    model = WarmupFlowModel(transformer, controlnet, adapter, cfg)
    load_checkpoint(ckpt, model)
    model.eval()
    ds = PairedWarmupDataset(cfg, 'val', require_coverage=True)
    ids = ds.ids[: int(cfg['eval']['fixed_val_count'])]
    swap_count = min(int(cfg['eval'].get('identity_swap_count', 0)), len(ids))
    out = Path(cfg['data']['root']) / 'phase1' / cfg['experiment']['id'] / 'warmup_vis' / ckpt.name
    out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg['data']['root'])
    generated_paths: list[Path] = []
    swap_rows = []
    for i, sid in enumerate(ids):
        batch = ds[ds.ids.index(sid)]
        tokens = generate(model, batch, int(cfg['eval']['generate_steps']), seed=1000 + i, device=torch_device, dtype=dtype)
        image = decode_tokens(vae, tokens, int(cfg['data']['resolution']))
        gen_path = out / f'{sid}_generated.png'
        image.save(gen_path)
        generated_paths.append(gen_path)
        swap_image = None
        swap_id = None
        swap_path = None
        if i < swap_count:
            swap_id = ids[(i + 1) % len(ids)]
            donor = ds[ds.ids.index(swap_id)]
            swap_batch = swap_identity(batch, donor)
            swap_tokens = generate(model, swap_batch, int(cfg['eval']['generate_steps']), seed=2000 + i, device=torch_device, dtype=dtype)
            swap_image = decode_tokens(vae, swap_tokens, int(cfg['data']['resolution']))
            swap_path = out / f'{sid}_swap_{swap_id}.png'
            swap_image.save(swap_path)
            generated_paths.append(swap_path)
        make_panel(root, sid, image, int(cfg['data']['resolution']), swap=swap_image, swap_id=swap_id).save(out / f'{sid}.png')
        if swap_path is not None:
            swap_rows.append({'sample_id': sid, 'swap_id': swap_id, 'paired_path': gen_path, 'swap_path': swap_path})

    embeddings: dict[Path, np.ndarray | None] = {path: embedding_for_image(path, cfg, device_index) for path in generated_paths}
    detected = sum(emb is not None for emb in embeddings.values())
    face_rate = detected / max(1, len(generated_paths))
    swap_results = []
    adapter_not_responding = False
    for row in swap_rows:
        a = embeddings.get(row['paired_path'])
        b = embeddings.get(row['swap_path'])
        cos = cosine(a, b) if a is not None and b is not None else None
        if cos is not None and cos > 0.9:
            adapter_not_responding = True
        swap_results.append({**row, 'cosine': cos})

    warnings = []
    if face_rate < 0.90:
        warnings.append('⚠ RED: face detection collapsed')
    if adapter_not_responding:
        warnings.append('⚠ adapter not responding')
    report = ['# Warmup Watcher Report', '']
    report.extend(warnings or ['status: automatic checks passed thresholds'])
    report.extend(['', f'checkpoint: `{ckpt}`', '', '## Automatic Checks', ''])
    report.append(f'face detection rate: {detected}/{len(generated_paths)} = {face_rate:.2%}')
    report.append('')
    report.append('| sample | swap_id | ArcFace cos(generated c_i, swap c_j) | status |')
    report.append('|---|---|---:|---|')
    for row in swap_results:
        cos = row['cosine']
        status = 'N/A detection failed' if cos is None else ('adapter not responding' if cos > 0.9 else 'responding')
        report.append(f"| {row['sample_id']} | {row['swap_id']} | {'N/A' if cos is None else f'{cos:.4f}'} | {status} |")
    report.extend(['', '## Manual Checklist', '', 'Inspect plastic feel, garment fidelity, pose following, and identity swap response in the five-column panels.'])
    (out / 'watcher_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='GPU3 checkpoint watcher for Phase 1 warmup.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--ckpt-dir', required=False)
    parser.add_argument('--ckpt', default=None, help='Run one specific checkpoint directory and exit.')
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.ckpt:
        run_once(args.config, Path(args.ckpt), args.device)
        return
    if not args.ckpt_dir:
        raise SystemExit('--ckpt-dir is required unless --ckpt is provided')
    seen = set()
    while True:
        ready = sorted(Path(args.ckpt_dir).glob('*/READY'))
        for marker in ready:
            ckpt = marker.parent
            if str(ckpt) in seen:
                continue
            run_once(args.config, ckpt, args.device)
            seen.add(str(ckpt))
        if args.once:
            break
        time.sleep(load_yaml(args.config)['eval']['watcher_poll_seconds'])


if __name__ == '__main__':
    main()
