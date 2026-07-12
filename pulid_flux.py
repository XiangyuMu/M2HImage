
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch import nn

from conditions import sha256_file


PULID_DOWNLOAD_GUIDE = """PuLID-FLUX assets are required and no fallback is allowed.
Download/prepare:
  repo: https://github.com/ToTheBeginning/PuLID -> /data/muxiangyu/modelLibrary/PuLID
  weight: https://huggingface.co/guozinan/PuLID/blob/main/pulid_flux_v0.9.1.safetensors
  place: /data/muxiangyu/modelLibrary/PuLID/models/pulid_flux_v0.9.1.safetensors
  antelopev2: /data/muxiangyu/modelLibrary/PuLID/models/antelopev2
  EVA cache: HF_HOME=/data/muxiangyu/modelLibrary, file QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt
  dependencies: facexlib, insightface, onnxruntime, timm, safetensors
"""


@contextmanager
def _pulid_import_context(repo: str | Path):
    repo = Path(repo)
    if not repo.exists():
        raise FileNotFoundError(f'PuLID repo missing: {repo}. {PULID_DOWNLOAD_GUIDE}')
    old_cwd = Path.cwd()
    inserted = False
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
        inserted = True
    try:
        os.chdir(repo)
        yield repo
    finally:
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(str(repo))
            except ValueError:
                pass


def _require_file(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f'{label} missing: {path}. {PULID_DOWNLOAD_GUIDE}')
    return path


def _require_dir(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f'{label} missing: {path}. {PULID_DOWNLOAD_GUIDE}')
    return path


class PuLIDFluxAdapter(nn.Module):
    """Frozen PuLID-FLUX cross-attention side path for diffusers FLUX transformer blocks."""

    def __init__(self, cfg: dict[str, Any], device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        self.repo = _require_dir(cfg.get('repo', ''), 'PuLID repo')
        self.weight_path = _require_file(cfg.get('weight_path', ''), 'PuLID-FLUX weight')
        self.double_interval = int(cfg.get('double_interval', 2))
        self.single_interval = int(cfg.get('single_interval', 4))
        self.id_weight = float(cfg.get('id_weight', 1.0))
        with _pulid_import_context(self.repo):
            from pulid.encoders_transformer import PerceiverAttentionCA

            num_ca = 19 // self.double_interval + 38 // self.single_interval
            if 19 % self.double_interval != 0:
                num_ca += 1
            if 38 % self.single_interval != 0:
                num_ca += 1
            self.pulid_ca = nn.ModuleList([PerceiverAttentionCA() for _ in range(num_ca)])
        state = load_file(str(self.weight_path), device='cpu')
        ca_state = {}
        for key, value in state.items():
            if key.startswith('pulid_ca.'):
                ca_state[key[len('pulid_ca.'):]] = value
        missing, unexpected = self.pulid_ca.load_state_dict(ca_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f'PuLID CA state mismatch: missing={missing}, unexpected={unexpected}')
        self.pulid_ca.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self._handles = []
        self._original_forwards = []
        self._debug_hook_calls = 0
        self._debug_delta_norm = 0.0
        self._context_id: torch.Tensor | None = None
        self._context_weight: float = self.id_weight
        self.weight_hash = sha256_file(self.weight_path)[:16]

    def launch_note(self) -> dict[str, Any]:
        return {
            'type': 'pulid_flux_v0.9.1',
            'repo': str(self.repo),
            'weight_path': str(self.weight_path),
            'weight_hash': self.weight_hash,
            'antelopev2_dir': str(self.cfg.get('antelopev2_dir', self.repo / 'models' / 'antelopev2')),
            'hf_home': str(self.cfg.get('hf_home', '/data/muxiangyu/modelLibrary')),
            'double_interval': self.double_interval,
            'single_interval': self.single_interval,
            'id_weight': self.id_weight,
            'trainable': 0,
        }

    def attach_to_transformer(self, transformer: nn.Module) -> None:
        if self._original_forwards:
            return
        ca_idx = 0
        for block_idx, block in enumerate(getattr(transformer, 'transformer_blocks')):
            if block_idx % self.double_interval == 0:
                self._wrap_block_forward(block, ca_idx)
                ca_idx += 1
        for block_idx, block in enumerate(getattr(transformer, 'single_transformer_blocks')):
            if block_idx % self.single_interval == 0:
                self._wrap_block_forward(block, ca_idx)
                ca_idx += 1
        if ca_idx != len(self.pulid_ca):
            raise RuntimeError(f'PuLID hook count mismatch: hooks={ca_idx}, ca_modules={len(self.pulid_ca)}')

    def _wrap_block_forward(self, block: nn.Module, ca_idx: int) -> None:
        original_forward = block.forward

        def forward_with_pulid(*args, **kwargs):
            return self._apply_pulid_residual(ca_idx, original_forward(*args, **kwargs))

        block.forward = forward_with_pulid
        self._original_forwards.append((block, original_forward))

    def set_context(self, id_embedding: torch.Tensor, id_weight: float | None = None) -> None:
        if id_embedding.ndim == 2:
            id_embedding = id_embedding.unsqueeze(0)
        if id_embedding.ndim != 3:
            raise RuntimeError(f'pulid_id_embed must have shape (B,N,D) or (N,D), got {tuple(id_embedding.shape)}')
        self._context_id = id_embedding
        self._context_weight = self.id_weight if id_weight is None else float(id_weight)

    def clear_context(self) -> None:
        self._context_id = None

    def _apply_pulid_residual(self, ca_idx: int, output):
        if self._context_id is None:
            return output
        if not isinstance(output, tuple) or len(output) != 2:
            return output
        encoder_hidden_states, hidden_states = output
        id_embedding = self._context_id.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if id_embedding.shape[0] == 1 and hidden_states.shape[0] != 1:
            id_embedding = id_embedding.expand(hidden_states.shape[0], -1, -1)
        delta = self.pulid_ca[ca_idx](id_embedding, hidden_states)
        self._debug_hook_calls += 1
        self._debug_delta_norm += float(torch.norm(delta.detach().float()).item())
        hidden_states = hidden_states + float(self._context_weight) * delta
        return encoder_hidden_states, hidden_states

    def _make_hook(self, ca_idx: int):
        def hook(_module, _args, output):
            return self._apply_pulid_residual(ca_idx, output)
        return hook

    @torch.no_grad()
    def self_check(self, hidden_size: int = 3072, tokens: int = 32, id_tokens: int = 32, device: torch.device | None = None, dtype: torch.dtype | None = None) -> float:
        ca = self.pulid_ca[0]
        device = device or next(ca.parameters()).device
        dtype = dtype or next(ca.parameters()).dtype
        generator = torch.Generator(device=device).manual_seed(1234)
        hidden = torch.randn(1, tokens, hidden_size, device=device, dtype=dtype, generator=generator)
        ids = torch.randn(1, id_tokens, 2048, device=device, dtype=dtype, generator=generator)
        out = hidden + self.id_weight * ca(ids, hidden)
        return float(torch.norm((out - hidden).float()).item())


    @torch.no_grad()
    def transformer_self_check(
        self,
        transformer: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        token_count: int = 256,
        text_tokens: int = 16,
    ) -> float:
        generator = torch.Generator(device=device).manual_seed(4321)
        hidden = torch.randn(1, token_count, 64, device=device, dtype=dtype, generator=generator)
        encoder = torch.randn(1, text_tokens, 4096, device=device, dtype=dtype, generator=generator)
        pooled = torch.randn(1, 768, device=device, dtype=dtype, generator=generator)
        ids = torch.zeros(text_tokens, 3, device=device, dtype=dtype)
        img_ids = torch.zeros(token_count, 3, device=device, dtype=dtype)
        img_ids[:, 1] = torch.arange(token_count, device=device, dtype=dtype) // 4
        img_ids[:, 2] = torch.arange(token_count, device=device, dtype=dtype) % 4
        timestep = torch.full((1,), 0.5, device=device, dtype=dtype)
        guidance = torch.full((1,), 3.5, device=device, dtype=dtype)
        id_embedding = torch.randn(1, 32, 2048, device=device, dtype=dtype, generator=generator)
        self._debug_hook_calls = 0
        self._debug_delta_norm = 0.0
        self.set_context(id_embedding, id_weight=0.0)
        out0 = transformer(
            hidden_states=hidden.clone(),
            encoder_hidden_states=encoder.clone(),
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=ids,
            guidance=guidance,
            return_dict=True,
        ).sample.detach().clone()
        calls_after_zero = self._debug_hook_calls
        delta_after_zero = self._debug_delta_norm
        self._debug_hook_calls = 0
        self._debug_delta_norm = 0.0
        self.set_context(id_embedding, id_weight=self.id_weight)
        out1 = transformer(
            hidden_states=hidden.clone(),
            encoder_hidden_states=encoder.clone(),
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=ids,
            guidance=guidance,
            return_dict=True,
        ).sample.detach().clone()
        self._last_transformer_out0_norm = float(torch.norm(out0.float()).item())
        self._last_transformer_out1_norm = float(torch.norm(out1.float()).item())
        self._last_transformer_hook_calls = self._debug_hook_calls
        self._last_transformer_delta_norm = self._debug_delta_norm
        self._last_transformer_zero_hook_calls = calls_after_zero
        self._last_transformer_zero_delta_norm = delta_after_zero
        return float(torch.norm((out1 - out0).float()).item())



class PuLIDIdentityEmbedder:
    """Offline PuLID ID embedding extractor using the official PuLID-FLUX pipeline."""

    def __init__(self, cfg: dict[str, Any], device: torch.device, dtype: torch.dtype) -> None:
        self.cfg = dict(cfg)
        self.repo = _require_dir(cfg.get('repo', ''), 'PuLID repo')
        self.weight_path = _require_file(cfg.get('weight_path', ''), 'PuLID-FLUX weight')
        self.antelopev2_dir = _require_dir(cfg.get('antelopev2_dir', self.repo / 'models' / 'antelopev2'), 'PuLID antelopev2 dir')
        self.hf_home = Path(cfg.get('hf_home', '/data/muxiangyu/modelLibrary'))
        os.environ.setdefault('HF_HOME', str(self.hf_home))
        os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
        self.device = device
        self.dtype = dtype
        with _pulid_import_context(self.repo):
            import pulid.pipeline_flux as pipeline_flux

            class _DummyDiT:
                pass

            original_snapshot_download = pipeline_flux.snapshot_download

            def local_snapshot_download(repo_id, local_dir=None, *args, **kwargs):
                if repo_id == 'DIAMONIK7777/antelopev2':
                    target = Path(local_dir or 'models/antelopev2')
                    if not target.is_absolute():
                        target = self.repo / target
                    if target.resolve() != self.antelopev2_dir.resolve():
                        raise RuntimeError(
                            f'PuLID antelopev2 path mismatch: requested {target}, configured {self.antelopev2_dir}'
                        )
                    _require_dir(target, 'PuLID antelopev2 dir')
                    return str(target)
                return original_snapshot_download(repo_id, local_dir=local_dir, *args, **kwargs)

            pipeline_flux.snapshot_download = local_snapshot_download
            try:
                provider = 'cpu' if str(device) == 'cpu' else str(cfg.get('onnx_provider', 'gpu'))
                self.pipeline = pipeline_flux.PuLIDPipeline(_DummyDiT(), device=device, weight_dtype=dtype, onnx_provider=provider)
            finally:
                pipeline_flux.snapshot_download = original_snapshot_download
            self._load_local_pretrain()
            self.pipeline.eval().requires_grad_(False)

    def _load_local_pretrain(self) -> None:
        state = load_file(str(self.weight_path), device='cpu')
        grouped: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in state.items():
            module, _, subkey = key.partition('.')
            if not subkey:
                continue
            grouped.setdefault(module, {})[subkey] = value
        required = {'pulid_encoder', 'pulid_ca'}
        missing_modules = sorted(required - set(grouped))
        if missing_modules:
            raise RuntimeError(f'PuLID-FLUX local weight missing modules {missing_modules}: {self.weight_path}')
        for module, module_state in grouped.items():
            target = getattr(self.pipeline, module, None)
            if target is None:
                continue
            missing, unexpected = target.load_state_dict(module_state, strict=True)
            if missing or unexpected:
                raise RuntimeError(f'PuLID {module} state mismatch: missing={missing}, unexpected={unexpected}')

    @torch.no_grad()
    def embed_image(self, image: Image.Image | str | Path) -> np.ndarray:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        arr = np.asarray(image.convert('RGB'), dtype=np.uint8)
        with _pulid_import_context(self.repo):
            emb, _ = self.pipeline.get_id_embedding(arr, cal_uncond=False)
        return emb[0].detach().float().cpu().numpy().astype('float32')
