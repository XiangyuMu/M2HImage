from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from conditions import choose_dtype, find_one, load_yaml, pil_to_tensor, seed_everything, unpack_latents
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
    # Simple deterministic Euler solver for the trained flow field: integrate from tau=1 to tau=0.
    for i in range(steps):
        tau = torch.full((1,), 1.0 - i / steps, device=device, dtype=dtype)
        model_timestep = tau * 1000.0
        local = batch
        prompt = local['prompt_embeds'].to(device=device, dtype=dtype).unsqueeze(0) if local['prompt_embeds'].ndim == 2 else local['prompt_embeds'].to(device=device, dtype=dtype)
        pooled = local['pooled_prompt_embeds'].to(device=device, dtype=dtype).unsqueeze(0) if local['pooled_prompt_embeds'].ndim == 1 else local['pooled_prompt_embeds'].to(device=device, dtype=dtype)
        adapter_tokens = model.adapter(local['identity'].to(device=device, dtype=dtype).unsqueeze(0), local['appearance'].to(device=device, dtype=dtype).unsqueeze(0), local['garment'].to(device=device, dtype=dtype).unsqueeze(0), local['head_pose'].to(device=device, dtype=dtype).unsqueeze(0))
        from conditions import make_image_ids, make_text_ids
        img_ids = make_image_ids(model.resolution, device, dtype)
        with torch.no_grad():
            cn = model.controlnet(hidden_states=z, controlnet_cond=local['pose_latents'].to(device=device, dtype=dtype).unsqueeze(0), controlnet_mode=torch.full((1, 1), model.control_mode, device=device, dtype=torch.long), conditioning_scale=model.controlnet_scale, encoder_hidden_states=prompt, pooled_projections=pooled, timestep=model_timestep, img_ids=img_ids, txt_ids=make_text_ids(prompt.shape[1], device, dtype), guidance=torch.full((1,), 3.5, device=device, dtype=dtype), return_dict=True)
            v = model.transformer(hidden_states=z, encoder_hidden_states=torch.cat([prompt, adapter_tokens], dim=1), pooled_projections=pooled, timestep=model_timestep, img_ids=img_ids, txt_ids=make_text_ids(prompt.shape[1] + adapter_tokens.shape[1], device, dtype), guidance=torch.full((1,), 3.5, device=device, dtype=dtype), controlnet_block_samples=cn.controlnet_block_samples, controlnet_single_block_samples=cn.controlnet_single_block_samples, return_dict=True).sample
        z = z - (1.0 / steps) * v
    return z


def make_panel(root: Path, sample_id: str, generated: Image.Image, resolution: int) -> Image.Image:
    parts = [
        Image.open(find_one(root / 'images/mannequin', sample_id)).convert('RGB').resize((resolution, resolution)),
        Image.open(find_one(root / 'dwpose/without_head/mannequin', sample_id)).convert('RGB').resize((resolution, resolution)),
        generated.resize((resolution, resolution)),
        Image.open(find_one(root / 'images/human', sample_id)).convert('RGB').resize((resolution, resolution)),
    ]
    canvas = Image.new('RGB', (resolution * 4, resolution + 24), (255, 255, 255))
    labels = ['m_i', 'pose', 'generated', 'h_i']
    draw = ImageDraw.Draw(canvas)
    for i, image in enumerate(parts):
        canvas.paste(image, (i * resolution, 24))
        draw.text((i * resolution + 8, 6), labels[i], fill=(0, 0, 0))
    return canvas


def run_once(config_path: str, ckpt: Path, device: str) -> None:
    cfg = load_yaml(config_path)
    seed_everything(int(cfg['experiment']['seed']))
    torch_device = torch.device(device)
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
    out = Path(cfg['data']['root']) / 'phase1' / cfg['experiment']['id'] / 'warmup_vis' / ckpt.name
    out.mkdir(parents=True, exist_ok=True)
    report = ['# Warmup Watcher Report', '', f'checkpoint: `{ckpt}`', '', 'Checklist: plastic feel, garment fidelity, pose following, identity swap response.']
    for i, sid in enumerate(ids):
        batch = ds[ds.ids.index(sid)]
        tokens = generate(model, batch, int(cfg['eval']['generate_steps']), seed=1000 + i, device=torch_device, dtype=dtype)
        image = decode_tokens(vae, tokens, int(cfg['data']['resolution']))
        make_panel(Path(cfg['data']['root']), sid, image, int(cfg['data']['resolution'])).save(out / f'{sid}.png')
    report.append('')
    report.append('Identity spot-check: inspect generated swap panels; if face does not change, mark adapter BLOCKER before B2.')
    (out / 'watcher_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='GPU3 checkpoint watcher for Phase 1 warmup.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--ckpt-dir', required=True)
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
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
