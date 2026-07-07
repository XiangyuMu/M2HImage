from __future__ import annotations


def weighted_diffusion_loss(prediction, target, editable_mask, editable_weight: float, context_weight: float):
    import torch.nn.functional as functional

    error = functional.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=1, keepdim=True)
    weights = editable_mask * editable_weight + (1 - editable_mask) * context_weight
    return (error * weights).mean()


def garment_l1_loss(prediction, target, cloth_mask):
    denominator = cloth_mask.sum().clamp_min(1.0) * prediction.shape[1]
    return ((prediction.float() - target.float()).abs() * cloth_mask).sum() / denominator


def cosine_identity_loss(prediction_embedding, target_embedding):
    import torch.nn.functional as functional

    return (1 - functional.cosine_similarity(prediction_embedding, target_embedding, dim=-1)).mean()


def identity_triplet_loss(prediction_embedding, target_embedding, source_embedding, margin: float):
    import torch
    import torch.nn.functional as functional

    positive = 1 - functional.cosine_similarity(prediction_embedding, target_embedding, dim=-1)
    negative = 1 - functional.cosine_similarity(prediction_embedding, source_embedding, dim=-1)
    return torch.relu(positive - negative + margin).mean()
