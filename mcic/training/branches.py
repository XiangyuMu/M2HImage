from __future__ import annotations


def masked_source(source, mask):
    return source * (1 - mask)


def predict_x0(noisy_latents, noise_prediction, timesteps, scheduler):
    import torch

    alpha = scheduler.alphas_cumprod.to(noisy_latents.device)[timesteps].view(-1, 1, 1, 1)
    return (noisy_latents - (1 - alpha).sqrt() * noise_prediction) / alpha.sqrt()


def crop_identity_region(images, masks, output_size: int = 160):
    import torch.nn.functional as functional

    crops = []
    for image, mask in zip(images, masks):
        positions = (mask[0] > 0.5).nonzero(as_tuple=False)
        if positions.numel() == 0:
            crop = image.unsqueeze(0)
        else:
            top, left = positions.min(dim=0).values
            bottom, right = positions.max(dim=0).values + 1
            crop = image[:, top:bottom, left:right].unsqueeze(0)
        crops.append(functional.interpolate(crop, (output_size, output_size), mode="bilinear", align_corners=False))
    return __import__("torch").cat(crops, dim=0)


def crop_face_boxes(images, boxes, output_size: int = 160):
    import torch.nn.functional as functional

    crops = []
    for image, box in zip(images, boxes):
        left, top, right, bottom = [int(value) for value in box]
        crop = image[:, top:bottom, left:right].unsqueeze(0)
        crops.append(functional.interpolate(crop, (output_size, output_size), mode="bilinear", align_corners=False))
    return __import__("torch").cat(crops, dim=0)
