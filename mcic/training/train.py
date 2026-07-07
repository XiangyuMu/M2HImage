from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcic.utils.config import add_config_args, load_config, write_json, write_yaml


def train(config: dict) -> None:
    import torch
    import torch.nn.functional as functional
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    from mcic.data.dataset import MCICDataset
    from mcic.losses.core import (
        cosine_identity_loss,
        garment_l1_loss,
        identity_triplet_loss,
        weighted_diffusion_loss,
    )
    from mcic.models.identity import build_differentiable_identity_encoder, reference_face_box
    from mcic.models.sdxl_mcic import MCICSDXLModel
    from mcic.training.branches import crop_face_boxes, crop_identity_region, masked_source, predict_x0

    training = config["training"]
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training["gradient_accumulation"]),
        mixed_precision=training.get("mixed_precision", "bf16"),
    )
    torch.manual_seed(int(training.get("seed", 42)))
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }.get(training.get("mixed_precision"), torch.float32)
    dataset = MCICDataset(config, "train")
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size_per_gpu"]),
        shuffle=True,
        num_workers=int(training.get("num_workers", 2)),
        drop_last=True,
    )
    model = MCICSDXLModel(config, dtype)
    model.to(dtype=dtype)
    init_checkpoint = config.get("experiment", {}).get("init_checkpoint")
    if init_checkpoint:
        model.load_trainable(init_checkpoint)
    identity_encoder = build_differentiable_identity_encoder(config)
    identity_encoder.requires_grad_(False)
    if training.get("gradient_checkpointing", True):
        model.unet.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=float(training["learning_rate"]))
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    identity_encoder.to(accelerator.device)
    output = Path(config["experiment"].get("output_root", "outputs")) / config["experiment"]["id"]
    run_metadata = {
        "resolution": [config["image"]["height"], config["image"]["width"]],
        "batch_size_per_gpu": training["batch_size_per_gpu"],
        "gradient_accumulation": training["gradient_accumulation"],
        "mixed_precision": training.get("mixed_precision"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        write_yaml(output / "resolved_config.yaml", config)
        write_json(output / "run_metadata.json", run_metadata)
    global_step = 0
    log_path = output / "logs" / "train.jsonl"
    while global_step < int(training["max_steps"]):
        for batch in loader:
            with accelerator.accumulate(model):
                core_model = accelerator.unwrap_model(model)
                human = batch["human"].to(dtype)
                mannequin = batch["mannequin"].to(dtype)
                paired_mask = batch["paired_mask"].to(dtype)
                identity = batch["identity_embedding"].to(dtype)
                with torch.no_grad():
                    target_latents = core_model.encode_latents(human)
                    masked_latents = core_model.encode_latents(masked_source(mannequin, paired_mask))
                noise = torch.randn_like(target_latents)
                timestep = torch.randint(
                    0, core_model.noise_scheduler.config.num_train_timesteps, (human.shape[0],), device=human.device
                ).long()
                noisy = core_model.noise_scheduler.add_noise(target_latents, noise, timestep)
                prediction = model(noisy, paired_mask, masked_latents, timestep, mannequin, identity)
                latent_mask = functional.interpolate(paired_mask, target_latents.shape[-2:], mode="nearest")
                pair_loss = weighted_diffusion_loss(
                    prediction,
                    noise,
                    latent_mask,
                    float(training.get("editable_loss_weight", 1.0)),
                    float(training.get("context_loss_weight", 0.1)),
                )
                total = float(config["loss"]["lambda_pair"]) * pair_loss
                metrics = {"loss_pair": pair_loss.detach().float().item()}
                if config.get("cf", {}).get("enabled", False):
                    negative_indices = dataset.sample_negative_indices(batch["index"])
                    negative_identity = dataset.identity_embeddings(negative_indices).to(human.device, dtype=dtype)
                    cf_mask = batch["cf_mask"].to(dtype)
                    cloth_mask = batch["cloth_safe_mask"].to(dtype)
                    with torch.no_grad():
                        cf_source = core_model.encode_latents(masked_source(mannequin, cf_mask))
                    cf_timestep = torch.randint(
                        int(config["cf"].get("timestep_min", 20)),
                        int(config["cf"].get("timestep_max", 500)) + 1,
                        (human.shape[0],),
                        device=human.device,
                    ).long()
                    cf_noisy = core_model.noise_scheduler.add_noise(target_latents, noise, cf_timestep)
                    cf_prediction = model(
                        cf_noisy, cf_mask, cf_source, cf_timestep, mannequin, negative_identity
                    )
                    cf_image = core_model.decode_latents(
                        predict_x0(cf_noisy, cf_prediction, cf_timestep, core_model.noise_scheduler)
                    )
                    cloth_loss = garment_l1_loss(cf_image, human, cloth_mask)
                    cf_loss = float(config["loss"]["lambda_cloth"]) * cloth_loss
                    metrics["loss_cloth"] = cloth_loss.detach().float().item()
                    if config["cf"].get("identity_loss_enabled", False):
                        valid_indices = list(range(cf_image.shape[0]))
                        boxes = []
                        use_filter = (
                            config["cf"].get("face_quality_filter", True)
                            and config["identity"].get("backend") == "facenet"
                        )
                        if use_filter:
                            from PIL import Image

                            valid_indices = []
                            for image_index, image in enumerate(cf_image.detach().float().cpu()):
                                array = ((image.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
                                try:
                                    boxes.append(reference_face_box(Image.fromarray(array), config))
                                    valid_indices.append(image_index)
                                except RuntimeError:
                                    continue
                        metrics["identity_face_pass_rate"] = len(valid_indices) / cf_image.shape[0]
                        if valid_indices:
                            chosen = cf_image[valid_indices]
                            generated_faces = (
                                crop_face_boxes(chosen, boxes)
                                if use_filter
                                else crop_identity_region(chosen, cf_mask[valid_indices])
                            )
                            generated_embedding = identity_encoder(generated_faces.float())
                            id_loss = cosine_identity_loss(generated_embedding, negative_identity[valid_indices].float())
                            tri_loss = identity_triplet_loss(
                                generated_embedding,
                                negative_identity[valid_indices].float(),
                                identity[valid_indices].float(),
                                float(config["loss"]["triplet_margin"]),
                            )
                            cf_loss = cf_loss + float(config["loss"]["lambda_id"]) * id_loss
                            cf_loss = cf_loss + float(config["loss"]["lambda_tri"]) * tri_loss
                            metrics.update(loss_identity=id_loss.detach().float().item(), loss_triplet=tri_loss.detach().float().item())
                    total = total + float(config["loss"]["lambda_cf_max"]) * cf_loss
                accelerator.backward(total)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(core_model.trainable_parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                metrics.update(step=global_step, loss_total=total.detach().float().item())
                if accelerator.is_main_process and global_step % int(training.get("log_every", 10)) == 0:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metrics) + "\n")
                    accelerator.print(metrics)
                if accelerator.is_main_process and global_step % int(training.get("checkpoint_every", 1000)) == 0:
                    accelerator.unwrap_model(model).save_trainable(output / "checkpoints" / f"step-{global_step}")
                if global_step >= int(training["max_steps"]):
                    break
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_trainable(output / "checkpoints" / "final")
        run_metadata["peak_gpu_memory_bytes"] = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        )
        run_metadata["steps_completed"] = global_step
        write_json(output / "run_metadata.json", run_metadata)
        summary = (
            f"# {config['experiment']['id']} Run Summary\n\n"
            f"- Resolution: `{config['image']['height']}x{config['image']['width']}`\n"
            f"- Steps completed: `{global_step}`\n"
            f"- Batch size per GPU: `{training['batch_size_per_gpu']}`\n"
            f"- Gradient accumulation: `{training['gradient_accumulation']}`\n"
            f"- Final checkpoint: `checkpoints/final`\n"
            f"- Training log: `logs/train.jsonl`\n\n"
            "Add metric results and visual observations using `docs/EXPERIMENT_LOG_TEMPLATE.md`.\n"
        )
        (output / "run_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MCIC with SDXL inpainting and LoRA.")
    add_config_args(parser)
    args = parser.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
