from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps


def custom_prompt_only(pipe: FluxPipeline, prompt: str, steps: int, seed: int, height: int, width: int, guidance_scale: float, device: torch.device):
    dtype = pipe.transformer.dtype
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(1, num_channels_latents, height, width, dtype, device, generator)
    sigmas = None if getattr(pipe.scheduler.config, 'use_flow_sigmas', False) else np.linspace(1.0, 1 / steps, steps)
    mu = calculate_shift(
        latents.shape[1],
        pipe.scheduler.config.get('base_image_seq_len', 256),
        pipe.scheduler.config.get('max_image_seq_len', 4096),
        pipe.scheduler.config.get('base_shift', 0.5),
        pipe.scheduler.config.get('max_shift', 1.15),
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=mu)
    guidance = torch.full([latents.shape[0]], guidance_scale, device=device, dtype=torch.float32) if pipe.transformer.config.guidance_embeds else None
    pipe.scheduler.set_begin_index(0)
    for t in timesteps:
        timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000.0
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type='pil')[0]


def main() -> None:
    parser = argparse.ArgumentParser(description='Sanity-check FLUX timestep scaling in the custom prompt-only denoise path.')
    parser.add_argument('--base', default='/data/muxiangyu/pythonPrograms/M2HImage/models/hf/black-forest-labs/FLUX.1-dev')
    parser.add_argument('--prompt', default='a photorealistic human wearing a denim jacket, clean studio lighting')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--out-dir', default='/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/timestep_sanity')
    args = parser.parse_args()

    device = torch.device(args.device)
    pipe = FluxPipeline.from_pretrained(args.base, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        official = pipe(
            prompt=args.prompt,
            prompt_2=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=3.5,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        ).images[0]
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    with torch.inference_mode():
        custom = custom_prompt_only(pipe, args.prompt, args.steps, args.seed, args.height, args.width, 3.5, device)
    official.save(out_dir / 'official_flux_pipeline.png')
    custom.save(out_dir / 'custom_prompt_only_tau_0_1.png')
    arr = np.asarray(custom.convert('RGB'), dtype=np.float32)
    print({'custom_mean': float(arr.mean()), 'custom_std': float(arr.std()), 'out_dir': str(out_dir)})


if __name__ == '__main__':
    main()
