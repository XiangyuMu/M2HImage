from __future__ import annotations

from pathlib import Path
from typing import Any


class MCICSDXLModel:
    """SDXL-inpainting wrapper with LoRA and gated condition token injection."""

    def __new__(cls, config: dict[str, Any], torch_dtype=None):
        import torch
        from diffusers import AutoPipelineForInpainting, DDPMScheduler
        from diffusers.models.attention_processor import AttnProcessor2_0
        from peft import LoraConfig
        from torch import nn
        from transformers import CLIPVisionModel

        from mcic.models.conditioning import IdentityProjector, MannequinProjector
        from mcic.models.gated_attention import GatedConditionAttnProcessor

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                dtype = torch_dtype or torch.float32
                pipeline = AutoPipelineForInpainting.from_pretrained(
                    config["model"]["base_model"], torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
                )
                self.vae = pipeline.vae
                self.unet = pipeline.unet
                self.text_encoder = pipeline.text_encoder
                self.text_encoder_2 = pipeline.text_encoder_2
                self.tokenizer = pipeline.tokenizer
                self.tokenizer_2 = pipeline.tokenizer_2
                self.noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
                self.inference_scheduler_config = pipeline.scheduler.config
                self.vision_encoder = CLIPVisionModel.from_pretrained(config["model"]["mannequin_encoder"])
                for module in (self.vae, self.text_encoder, self.text_encoder_2, self.vision_encoder, self.unet):
                    module.requires_grad_(False)
                rank = int(config["model"].get("lora_rank", 16))
                self.unet.add_adapter(
                    LoraConfig(
                        r=rank,
                        lora_alpha=int(config["model"].get("lora_alpha", rank)),
                        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                        init_lora_weights="gaussian",
                    )
                )
                cross_dim = int(self.unet.config.cross_attention_dim)
                vision_dim = int(self.vision_encoder.config.hidden_size)
                self.mannequin_projector = MannequinProjector(
                    vision_dim, cross_dim, int(config["model"]["mannequin_tokens"])
                )
                self.identity_projector = IdentityProjector(
                    int(config["model"]["identity_embedding_dim"]),
                    cross_dim,
                    int(config["model"]["identity_tokens"]),
                )
                processors = {}
                block_channels = list(self.unet.config.block_out_channels)
                for name in self.unet.attn_processors.keys():
                    if name.endswith("attn1.processor"):
                        processors[name] = AttnProcessor2_0()
                        continue
                    if name.startswith("mid_block"):
                        hidden_size = block_channels[-1]
                    elif name.startswith("up_blocks"):
                        hidden_size = list(reversed(block_channels))[int(name.split(".")[1])]
                    else:
                        hidden_size = block_channels[int(name.split(".")[1])]
                    processors[name] = GatedConditionAttnProcessor(
                        hidden_size,
                        cross_dim,
                        int(config["model"]["mannequin_tokens"]),
                        int(config["model"]["identity_tokens"]),
                    )
                self.unet.set_attn_processor(processors)
                self.prompt = config["model"]["prompt"]
                self.height = config["image"]["height"]
                self.width = config["image"]["width"]
                self.vae_scale = self.vae.config.scaling_factor
                del pipeline

            def trainable_parameters(self):
                return [parameter for parameter in self.parameters() if parameter.requires_grad]

            def encode_latents(self, images):
                return self.vae.encode(images).latent_dist.sample() * self.vae_scale

            def decode_latents(self, latents):
                return self.vae.decode(latents / self.vae_scale).sample.clamp(-1, 1)

            def encode_prompt(self, batch_size: int, device, dtype):
                import torch

                prompt = [self.prompt] * batch_size
                prompt_1 = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                prompt_2 = self.tokenizer_2(prompt, padding="max_length", max_length=self.tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                output_1 = self.text_encoder(prompt_1, output_hidden_states=True)
                output_2 = self.text_encoder_2(prompt_2, output_hidden_states=True)
                prompt_embeds = torch.cat([output_1.hidden_states[-2], output_2.hidden_states[-2]], dim=-1).to(dtype)
                pooled = output_2[0].to(dtype)
                return prompt_embeds, pooled

            def mannequin_tokens(self, images):
                import torch.nn.functional as functional

                pixels = functional.interpolate((images + 1) / 2, size=(224, 224), mode="bicubic", align_corners=False)
                mean = pixels.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
                std = pixels.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
                vision = self.vision_encoder((pixels - mean) / std).last_hidden_state[:, 1:]
                return self.mannequin_projector(vision)

            def conditioning(self, mannequin, identity_embedding):
                import torch

                text, pooled = self.encode_prompt(mannequin.shape[0], mannequin.device, mannequin.dtype)
                man_tokens = self.mannequin_tokens(mannequin)
                id_tokens = self.identity_projector(identity_embedding)
                return torch.cat([text, man_tokens, id_tokens], dim=1), pooled

            def forward_unet(self, noisy_latents, masks, masked_latents, timesteps, mannequin, identity):
                import torch
                import torch.nn.functional as functional

                masks = functional.interpolate(masks, size=noisy_latents.shape[-2:], mode="nearest")
                sample = torch.cat([noisy_latents, masks, masked_latents], dim=1)
                hidden, pooled = self.conditioning(mannequin, identity)
                time_ids = noisy_latents.new_tensor(
                    [self.height, self.width, 0, 0, self.height, self.width]
                ).unsqueeze(0).repeat(noisy_latents.shape[0], 1)
                return self.unet(
                    sample,
                    timesteps,
                    encoder_hidden_states=hidden,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
                ).sample

            def forward(self, noisy_latents, masks, masked_latents, timesteps, mannequin, identity):
                return self.forward_unet(
                    noisy_latents, masks, masked_latents, timesteps, mannequin, identity
                )

            def save_trainable(self, folder: str | Path) -> None:
                import torch

                path = Path(folder)
                path.mkdir(parents=True, exist_ok=True)
                trainable_unet = {
                    name: value.detach().cpu()
                    for name, value in self.unet.state_dict().items()
                    if "lora" in name or ".processor." in name
                }
                torch.save(trainable_unet, path / "unet_trainable.pt")
                torch.save(
                    {
                        "mannequin_projector": self.mannequin_projector.state_dict(),
                        "identity_projector": self.identity_projector.state_dict(),
                    },
                    path / "conditioners.pt",
                )

            def load_trainable(self, folder: str | Path) -> None:
                import torch

                payload = torch.load(Path(folder) / "conditioners.pt", map_location="cpu", weights_only=True)
                self.mannequin_projector.load_state_dict(payload["mannequin_projector"])
                self.identity_projector.load_state_dict(payload["identity_projector"])
                unet_payload = torch.load(Path(folder) / "unet_trainable.pt", map_location="cpu", weights_only=True)
                self.unet.load_state_dict(unet_payload, strict=False)

        return Module()
