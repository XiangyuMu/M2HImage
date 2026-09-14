from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class DifferentiableHairDinoLoss(nn.Module):
    """Differentiable frozen-DINO hair loss with fixed or parsed mask geometry.

    This training-side implementation intentionally does not import the held-out
    metrics_v2 feature runner. It may use the same DINO checkpoint, which is
    disclosed as a limitation and is cross-checked by human review and LAB color.
    """

    def __init__(self, cfg: dict[str, Any], device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        dino_repo = Path(cfg['dino_repo_root'])
        dino_checkpoint = Path(cfg['dino_checkpoint'])
        self.mask_source = str(cfg.get('mask_source', 'generated_parser'))
        required = [
            (dino_repo, 'DINOv2 repo'),
            (dino_checkpoint, 'DINOv2 checkpoint'),
        ]
        parser_dir = Path(cfg.get('parser_model_dir', '.'))
        if self.mask_source == 'generated_parser':
            required.append((parser_dir, 'FASHN parser'))
        elif self.mask_source != 'target_projection':
            raise ValueError(f'unsupported hair loss mask_source={self.mask_source}')
        for path, role in required:
            if not path.exists():
                raise FileNotFoundError(f'hair loss {role} is missing: {path}')
        from metrics.garment_sim import load_dino_model

        proxy = {
            'metrics': {
                'garment': {
                    'dino_repo_root': str(dino_repo),
                    'dino_checkpoint': str(dino_checkpoint),
                },
            },
        }
        precision = str(cfg.get('dino_precision', 'float32')).lower()
        dino_dtypes = {
            'float32': torch.float32,
            'fp32': torch.float32,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
        }
        if precision not in dino_dtypes:
            raise ValueError(
                f'unsupported training DINO precision={precision!r}; expected fp32 or bf16'
            )
        self.dino_dtype = dino_dtypes[precision]
        self.dino = load_dino_model(proxy, device)
        self.dino.to(device=device, dtype=self.dino_dtype).eval().requires_grad_(False)
        self.parser = None
        if self.mask_source == 'generated_parser':
            from metrics_v2.parsing import FashnParser

            self.parser = FashnParser(
                parser_dir,
                device=str(device),
                input_size=(
                    int(cfg.get('parser_input_width', 384)),
                    int(cfg.get('parser_input_height', 576)),
                ),
            )
        self.image_size = int(cfg.get('dino_image_size', 518))
        self.hair_label = int(cfg.get('hair_label', 2))
        self.min_area = float(cfg.get('min_hair_area_fraction', 0.005))
        self.use_checkpoint = bool(cfg.get('gradient_checkpointing', True))
        self.register_buffer(
            'mean',
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'std',
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        # The DINO loader moves only its own model. Move this wrapper as well so
        # normalization buffers follow decoded tensors during differentiable loss.
        self.to(device)

    @torch.no_grad()
    def _parse(self, decoded: torch.Tensor) -> torch.Tensor:
        if self.parser is None:
            raise RuntimeError('generated parsing requested but parser is disabled')
        arrays = (
            ((decoded.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5)
            .permute(0, 2, 3, 1)
            .byte()
            .cpu()
            .numpy()
        )
        labels = self.parser.predict([np.asarray(image) for image in arrays])
        masks = np.stack([(label == self.hair_label).astype(np.float32) for label in labels])
        return torch.from_numpy(masks).unsqueeze(1).to(device=decoded.device)

    @staticmethod
    def _projected_mask(decoded: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        token_mask = token_mask.to(device=decoded.device, dtype=torch.float32)
        if token_mask.ndim == 3 and token_mask.shape[1] == 1:
            token_mask = token_mask[:, 0]
        if token_mask.ndim != 2:
            raise RuntimeError(
                f'projected hair mask must be [B,N], got {tuple(token_mask.shape)}'
            )
        token_count = int(token_mask.shape[1])
        aspect = float(decoded.shape[-2]) / float(decoded.shape[-1])
        grid_h = int(round((token_count * aspect) ** 0.5))
        grid_w = token_count // max(1, grid_h)
        if grid_h * grid_w != token_count:
            raise RuntimeError(
                f'cannot infer hair token grid for N={token_count}, decoded={tuple(decoded.shape[-2:])}'
            )
        mask = token_mask.reshape(token_mask.shape[0], 1, grid_h, grid_w)
        return F.interpolate(
            mask,
            size=decoded.shape[-2:],
            mode='bilinear',
            align_corners=False,
        ).clamp(0.0, 1.0)

    @staticmethod
    def _square_pad(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = image.shape[-2:]
        side = max(height, width)
        left = (side - width) // 2
        right = side - width - left
        top = (side - height) // 2
        bottom = side - height - top
        return (
            F.pad(image, (left, right, top, bottom), value=0.5),
            F.pad(mask, (left, right, top, bottom), value=0.0),
        )

    def _dino_forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.dino(value)
        return output[0] if isinstance(output, (tuple, list)) else output

    def forward(
        self,
        decoded: torch.Tensor,
        target_tokens: torch.Tensor,
        target_valid: torch.Tensor,
        projected_hair_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attempt = torch.tensor(float(decoded.shape[0]), device=decoded.device)
        if self.mask_source == 'target_projection':
            if projected_hair_mask is None:
                raise RuntimeError(
                    'target_projection hair loss requires cached projected_hair_mask'
                )
            masks = self._projected_mask(decoded, projected_hair_mask)
        else:
            try:
                masks = self._parse(decoded)
            except Exception:  # noqa: BLE001
                # Runtime parsing is auxiliary geometry. A parser failure is
                # observable and skippable; missing projected-mask wiring is not.
                zero = decoded.sum() * 0.0
                return zero, {
                    'loss_hair_dino': zero.detach().float(),
                    'hair_cosine': zero.detach().float(),
                    'loss_hair_sum': zero.detach().float(),
                    'hair_cosine_sum': zero.detach().float(),
                    'hair_valid_count': zero.detach().float(),
                    'hair_loss_attempt_count': attempt,
                    'hair_loss_skip_count': attempt,
                    'hair_skip_area_count': zero.detach().float(),
                    'hair_skip_target_invalid_count': zero.detach().float(),
                    'hair_skip_parsing_failure_count': attempt,
                }
        area = masks.float().mean(dim=(1, 2, 3))
        target_valid = target_valid.to(device=decoded.device, dtype=torch.float32)
        area_valid = area >= self.min_area
        target_is_valid = target_valid.sum(dim=1) >= 1.0
        valid = area_valid & target_is_valid
        skip = (~valid).float().sum()
        area_skip = (~area_valid).float().sum()
        target_skip = (~target_is_valid).float().sum()
        if not bool(valid.any()):
            zero = decoded.sum() * 0.0
            return zero, {
                'loss_hair_dino': zero.detach().float(),
                'hair_cosine': zero.detach().float(),
                'loss_hair_sum': zero.detach().float(),
                'hair_cosine_sum': zero.detach().float(),
                'hair_valid_count': zero.detach().float(),
                'hair_loss_attempt_count': attempt,
                'hair_loss_skip_count': skip,
                'hair_skip_area_count': area_skip,
                'hair_skip_target_invalid_count': target_skip,
                'hair_skip_parsing_failure_count': zero.detach().float(),
            }

        images = (decoded.index_select(0, torch.nonzero(valid, as_tuple=False).flatten()).float() + 1.0) * 0.5
        selected_masks = masks.index_select(0, torch.nonzero(valid, as_tuple=False).flatten()).float()
        images = images * selected_masks + 0.5 * (1.0 - selected_masks)
        images, selected_masks = self._square_pad(images, selected_masks)
        images = F.interpolate(images, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        images = ((images - self.mean) / self.std).to(dtype=self.dino_dtype)
        patches = checkpoint(self._dino_forward, images, use_reentrant=False) if self.use_checkpoint else self._dino_forward(images)
        count = patches.shape[1]
        side = int(round(count ** 0.5))
        if side * side != count:
            raise RuntimeError(f'hair-loss DINO output is not a square grid: {tuple(patches.shape)}')
        weights = F.interpolate(selected_masks, size=(side, side), mode='area').flatten(2).transpose(1, 2)
        generated = (patches.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
        chosen = torch.nonzero(valid, as_tuple=False).flatten()
        target = target_tokens.index_select(0, chosen).to(device=decoded.device, dtype=torch.float32)
        target_weights = target_valid.index_select(0, chosen).unsqueeze(-1)
        target = (target * target_weights).sum(dim=1) / target_weights.sum(dim=1).clamp_min(1e-6)
        cosine = F.cosine_similarity(generated, target, dim=1)
        loss = (1.0 - cosine).mean()
        return loss, {
            'loss_hair_dino': loss.detach(),
            'hair_cosine': cosine.detach().mean(),
            'loss_hair_sum': (1.0 - cosine.detach()).sum(),
            'hair_cosine_sum': cosine.detach().sum(),
            'hair_valid_count': torch.tensor(
                float(cosine.numel()), device=decoded.device
            ),
            'hair_loss_attempt_count': attempt,
            'hair_loss_skip_count': skip,
            'hair_skip_area_count': area_skip,
            'hair_skip_target_invalid_count': target_skip,
            'hair_skip_parsing_failure_count': torch.zeros(
                (), device=decoded.device, dtype=torch.float32
            ),
        }
