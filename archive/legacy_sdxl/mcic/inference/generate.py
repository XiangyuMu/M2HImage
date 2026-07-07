from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from mcic.models.identity import offline_identity_embedding, reference_face_box
from mcic.preprocess.parsing import compose_training_masks, heuristic_masks
from mcic.utils.config import add_config_args, load_config, write_json, write_yaml
from mcic.utils.image import compute_canvas_transform, fit_to_canvas, tensor_from_image, tensor_from_mask


def generate_image(config: dict, checkpoint: str, mannequin_path: str, reference_path: str, mask_path: str | None = None):
    import numpy as np
    import torch
    from diffusers import EulerDiscreteScheduler

    from mcic.models.sdxl_mcic import MCICSDXLModel
    from mcic.training.branches import masked_source

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    height, width = config["image"]["height"], config["image"]["width"]
    mannequin_raw = Image.open(mannequin_path).convert("RGB")
    reference_raw = Image.open(reference_path).convert("RGB")
    transform = compute_canvas_transform(mannequin_raw, height, width)
    mannequin = fit_to_canvas(mannequin_raw, transform, pad_value=config["image"].get("pad_value", 255))
    if mask_path:
        mask = fit_to_canvas(Image.open(mask_path), transform, is_mask=True)
    else:
        mask = compose_training_masks(heuristic_masks(mannequin), {
            "preprocess": {"boundary_dilate_radius": 7, "cloth_erode_radius": 3}
        })["paired_mask"]
    face_box = reference_face_box(reference_raw, config)
    embedding = offline_identity_embedding(reference_raw, face_box, config)
    source = tensor_from_image(mannequin).unsqueeze(0).to(device, dtype)
    mask_tensor = tensor_from_mask(mask).unsqueeze(0).to(device, dtype)
    identity = torch.from_numpy(np.asarray(embedding)).unsqueeze(0).to(device, dtype)
    model = MCICSDXLModel(config, dtype).to(device, dtype).eval()
    model.load_trainable(checkpoint)
    scheduler = EulerDiscreteScheduler.from_config(model.inference_scheduler_config)
    steps = int(config.get("inference", {}).get("num_inference_steps", 30))
    scheduler.set_timesteps(steps, device=device)
    generator = torch.Generator(device=device).manual_seed(int(config.get("inference", {}).get("seed", 42)))
    latents = torch.randn((1, 4, height // 8, width // 8), generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma
    with torch.no_grad():
        masked_latents = model.encode_latents(masked_source(source, mask_tensor))
        for timestep in scheduler.timesteps:
            batch_timestep = timestep.expand(1)
            scaled_latents = scheduler.scale_model_input(latents, timestep)
            prediction = model(scaled_latents, mask_tensor, masked_latents, batch_timestep, source, identity)
            latents = scheduler.step(prediction, timestep, latents).prev_sample
        result = model.decode_latents(latents)
        result = result * mask_tensor + source * (1 - mask_tensor)
    array = ((result[0].float().cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
    return Image.fromarray(array), mannequin, mask


def save_panel(output: Path, mannequin: Image.Image, reference: Image.Image, mask: Image.Image, generated: Image.Image) -> None:
    size = generated.size
    reference = reference.convert("RGB").resize(size, Image.Resampling.BICUBIC)
    mask_rgb = Image.merge("RGB", (mask, mask, mask)).resize(size)
    canvas = Image.new("RGB", (size[0] * 4, size[1] + 30), "white")
    for index, (label, image) in enumerate(
        [("mannequin", mannequin), ("identity", reference), ("mask", mask_rgb), ("output", generated)]
    ):
        canvas.paste(image.resize(size), (size[0] * index, 30))
        ImageDraw.Draw(canvas).text((size[0] * index + 5, 8), label, fill="black")
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a humanized mannequin image with MCIC.")
    add_config_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mannequin", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--output", default="outputs/inference/generated.png")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    generated, mannequin, mask = generate_image(
        config, args.checkpoint, args.mannequin, args.reference, args.mask
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generated.save(output)
    save_panel(output.with_name(f"{output.stem}_panel.jpg"), mannequin, Image.open(args.reference), mask, generated)
    write_yaml(output.with_name("resolved_config.yaml"), config)
    write_json(
        output.with_name(f"{output.stem}_metadata.json"),
        {
            "mannequin": args.mannequin,
            "reference": args.reference,
            "mask": args.mask,
            "checkpoint": args.checkpoint,
            "output": str(output),
            "resolution": [config["image"]["height"], config["image"]["width"]],
        },
    )
    print(f"Generated image written to {output}")


if __name__ == "__main__":
    main()
