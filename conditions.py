from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
PROMPT = 'a photorealistic human wearing the same garment, same body pose, natural skin and hair'


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml
    with Path(path).open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    import yaml
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def read_ids(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def find_one(folder: str | Path, sample_id: str) -> Path:
    folder = Path(folder)
    for ext in IMAGE_EXTS:
        path = folder / f'{sample_id}{ext}'
        if path.exists():
            return path
    matches = sorted(folder.glob(f'{sample_id}.*'))
    if matches:
        return matches[0]
    raise FileNotFoundError(f'missing file for id={sample_id} under {folder}')


def sha256_file(path: str | Path, block_mb: int = 16) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            chunk = handle.read(block_mb * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def short_hash_path(path: str | Path) -> str:
    path = Path(path)
    if path.is_file():
        return sha256_file(path)[:16]
    if path.is_dir():
        pieces = []
        for rel in ('config.json', 'model_index.json', 'diffusion_pytorch_model.safetensors'):
            p = path / rel
            if p.exists():
                pieces.append(f'{rel}:{sha256_file(p)[:16]}')
        return ','.join(pieces) if pieces else 'dir-no-known-files'
    return 'missing'


def assert_real_controlnet(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = path / 'config.json'
    weight = path / 'diffusion_pytorch_model.safetensors'
    if not cfg.exists() or not weight.exists():
        raise FileNotFoundError(
            f'真实 ControlNet 权重缺失: expected {cfg} and {weight}; Phase 1 禁止占位回退。'
        )
    payload = json.loads(cfg.read_text(encoding='utf-8'))
    if payload.get('_class_name') != 'FluxControlNetModel':
        raise RuntimeError(f'ControlNet config class mismatch: {payload.get("_class_name")}')
    return {'path': str(path), 'config_hash': sha256_file(cfg)[:16], 'weight_hash': sha256_file(weight)[:16], 'config': payload}


def pil_to_tensor(image: Image.Image, resolution: int) -> torch.Tensor:
    image = image.convert('RGB').resize((resolution, resolution), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def pil_to_clip_tensor(image: Image.Image, size: int = 224) -> torch.Tensor:
    image = image.convert('RGB').resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def mask_bbox(mask: Image.Image, pad: int = 12) -> tuple[int, int, int, int]:
    arr = np.asarray(mask.convert('L')) > 127
    ys, xs = np.where(arr)
    if len(xs) == 0:
        return 0, 0, mask.width, mask.height
    x0, x1 = max(0, xs.min() - pad), min(mask.width, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(mask.height, ys.max() + pad + 1)
    return int(x0), int(y0), int(x1), int(y1)


def garment_crop(root: Path, sample_id: str) -> Image.Image:
    mannequin = Image.open(find_one(root / 'images/mannequin', sample_id)).convert('RGB')
    mask = Image.open(find_one(root / 'clothes_bySAM/masks/human', sample_id)).convert('L')
    if mask.size != mannequin.size:
        mask = mask.resize(mannequin.size, Image.Resampling.NEAREST)
    garment = Image.new('RGB', mannequin.size, (255, 255, 255))
    garment.paste(mannequin, (0, 0), mask.point(lambda v: 255 if v > 127 else 0))
    return garment.crop(mask_bbox(mask))


def load_head_pose_token(root: Path, sample_id: str, dropout_p: float = 0.0, rng: random.Random | None = None) -> np.ndarray:
    path = root / 'derived/head_pose_6drepnet/human' / f'{sample_id}.json'
    if not path.exists():
        return np.zeros(7, dtype=np.float32)
    pose = json.loads(path.read_text(encoding='utf-8'))
    status = str(pose.get('status', 'unknown'))
    conf = float(pose.get('mean_confidence', pose.get('det_confidence', 0.0)) or 0.0)
    drop = rng.random() < dropout_p if rng is not None and dropout_p > 0 else False
    if status != 'ok' or conf < 0.1 or drop:
        return np.zeros(7, dtype=np.float32)
    vals = []
    for key in ('yaw', 'pitch', 'roll'):
        rad = math.radians(float(pose.get(key, 0.0)))
        vals.extend([math.sin(rad), math.cos(rad)])
    vals.append(1.0)
    return np.asarray(vals, dtype=np.float32)


def make_image_ids(resolution: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    h = resolution // 16
    w = resolution // 16
    rows = torch.arange(h, device=device, dtype=dtype)
    cols = torch.arange(w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(rows, cols, indexing='ij')
    return torch.stack((torch.zeros_like(gy), gy, gx), dim=-1).reshape(-1, 3)


def make_text_ids(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(length, 3, device=device, dtype=dtype)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    return latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_latents(tokens: torch.Tensor, resolution: int) -> torch.Tensor:
    b, n, c = tokens.shape
    h = resolution // 8
    w = resolution // 8
    return tokens.view(b, h // 2, w // 2, c // 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c // 4, h, w)


class FluxIdentityAdapter(nn.Module):
    """PuLID/InstantID-style local dual-path adapter for Phase 1 B2.

    It projects cached identity embedding, 1.8x face appearance token, garment token and head-pose token
    into additional FLUX text-stream tokens. This is intentionally adapter-only and paired-flow only;
    there is no counterfactual branch or CF loss in Phase 1 warmup.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        token_dim = int(cfg.get('token_dim', 4096))
        self.id_tokens = int(cfg.get('id_tokens', 4))
        self.appearance_tokens = int(cfg.get('appearance_tokens', 4))
        self.garment_tokens = int(cfg.get('garment_tokens', 8))
        self.pose_tokens = int(cfg.get('pose_tokens', 1))
        self.id_proj = nn.Sequential(nn.Linear(int(cfg.get('id_dim', 512)), token_dim * self.id_tokens), nn.SiLU())
        self.app_proj = nn.Sequential(nn.Linear(int(cfg.get('appearance_dim', 1024)), token_dim * self.appearance_tokens), nn.SiLU())
        self.garment_proj = nn.Sequential(nn.Linear(int(cfg.get('garment_dim', 1024)), token_dim * self.garment_tokens), nn.SiLU())
        self.pose_proj = nn.Sequential(nn.Linear(7, token_dim * self.pose_tokens), nn.SiLU())
        self.norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(float(cfg.get('dropout', 0.0)))
        self.token_dim = token_dim

    @property
    def token_count(self) -> int:
        return self.id_tokens + self.appearance_tokens + self.garment_tokens + self.pose_tokens

    def forward(
        self,
        identity: torch.Tensor,
        appearance: torch.Tensor,
        garment: torch.Tensor,
        head_pose: torch.Tensor,
    ) -> torch.Tensor:
        b = identity.shape[0]
        pieces = [
            self.id_proj(identity).view(b, self.id_tokens, self.token_dim),
            self.app_proj(appearance).view(b, self.appearance_tokens, self.token_dim),
            self.garment_proj(garment).view(b, self.garment_tokens, self.token_dim),
            self.pose_proj(head_pose).view(b, self.pose_tokens, self.token_dim),
        ]
        return self.dropout(self.norm(torch.cat(pieces, dim=1)))


def load_text_embeddings(base: str | Path, prompt: str, device: torch.device, dtype: torch.dtype, max_length: int = 512):
    base = Path(base)
    try:
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(str(base), torch_dtype=dtype, local_files_only=base.exists())
        pipe.to(device)
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_length,
            )
        del pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return prompt_embeds[0].detach().cpu(), pooled_prompt_embeds[0].detach().cpu(), text_ids.detach().cpu()
    except Exception:
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        clip_tokenizer = CLIPTokenizer.from_pretrained(str(base / 'tokenizer'), local_files_only=True)
        clip_encoder = CLIPTextModel.from_pretrained(str(base / 'text_encoder'), torch_dtype=dtype, local_files_only=True).to(device).eval()
        t5_tokenizer = T5TokenizerFast.from_pretrained(str(base / 'tokenizer_2'), local_files_only=True)
        t5_encoder = T5EncoderModel.from_pretrained(str(base / 'text_encoder_2'), torch_dtype=dtype, local_files_only=True).to(device).eval()
        with torch.no_grad():
            clip_inputs = clip_tokenizer(prompt, padding='max_length', max_length=77, truncation=True, return_tensors='pt').to(device)
            clip_out = clip_encoder(**clip_inputs)
            pooled = clip_out.pooler_output
            t5_inputs = t5_tokenizer(prompt, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt').to(device)
            prompt_embeds = t5_encoder(**t5_inputs).last_hidden_state
        text_ids = torch.zeros(max_length, 3, dtype=dtype)
        del clip_encoder, t5_encoder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return prompt_embeds[0].detach().cpu(), pooled[0].detach().cpu(), text_ids.detach().cpu()


def load_clip_vision(model_path: str | Path, device: torch.device, dtype: torch.dtype):
    from transformers import CLIPVisionModel
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f'CLIP/DINO garment encoder not found: {path}')
    model = CLIPVisionModel.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)
    model.eval().requires_grad_(False).to(device)
    return model


def pooled_clip_feature(model, image: Image.Image, device: torch.device, dtype: torch.dtype) -> np.ndarray:
    tensor = pil_to_clip_tensor(image).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        out = model(pixel_values=tensor)
        pooled = out.pooler_output if getattr(out, 'pooler_output', None) is not None else out.last_hidden_state[:, 0]
    return pooled[0].float().cpu().numpy().astype('float32')


def arcface_embedding(face: Image.Image, model_root: str | Path = '/data/muxiangyu/modelLibrary/insightface') -> np.ndarray:
    try:
        import cv2
        import insightface
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError('insightface/opencv are required for real identity cache; no mock fallback allowed') from exc
    app = getattr(arcface_embedding, '_app', None)
    if app is None:
        app = insightface.app.FaceAnalysis(name='antelopev2', root=str(model_root), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(224, 224))
        setattr(arcface_embedding, '_app', app)
    arr = np.asarray(face.convert('RGB'))[:, :, ::-1]
    faces = app.get(arr)
    if not faces:
        raise RuntimeError('ArcFace detector found no face')
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = np.asarray(best.normed_embedding, dtype='float32')
    return emb


_ARC_HELPERS: dict[tuple[str, str, str, int], subprocess.Popen] = {}


def _close_arcface_helpers() -> None:
    for proc in list(_ARC_HELPERS.values()):
        if proc.poll() is None:
            proc.terminate()
    _ARC_HELPERS.clear()


atexit.register(_close_arcface_helpers)


def _arcface_embedding_via_server(
    face_path: str | Path,
    helper_python: str,
    helper_script: str | Path,
    model_root: str | Path,
    device_id: int,
) -> np.ndarray:
    key = (str(helper_python), str(helper_script), str(model_root), int(device_id))
    proc = _ARC_HELPERS.get(key)
    if proc is None or proc.poll() is not None:
        cmd = [
            str(helper_python),
            str(helper_script),
            '--model-root',
            str(model_root),
            '--device-id',
            str(device_id),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        _ARC_HELPERS[key] = proc
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps({'image': str(face_path)}, ensure_ascii=False) + '\n')
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f'ArcFace helper server exited with code {proc.poll()}')
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        break
    if not payload.get('ok'):
        raise RuntimeError(f'ArcFace helper server failed for {face_path}: {payload.get("error")}')
    emb = np.asarray(payload['embedding'], dtype='float32')
    if emb.shape != (512,):
        raise RuntimeError(f'ArcFace helper returned invalid shape {emb.shape}, expected (512,)')
    return emb


def arcface_embedding_from_path(
    face_path: str | Path,
    helper_python: str | None = None,
    helper_script: str | Path | None = None,
    model_root: str | Path = '/data/muxiangyu/modelLibrary/insightface',
    device_id: int = 0,
) -> np.ndarray:
    try:
        return arcface_embedding(Image.open(face_path).convert('RGB'), model_root=model_root)
    except RuntimeError as local_exc:
        if not helper_python:
            raise
        script = Path(helper_script or 'tools/arcface_embed.py')
        if 'server' in script.name:
            return _arcface_embedding_via_server(face_path, helper_python, script, model_root, device_id)
        cmd = [
            str(helper_python),
            str(script),
            '--image',
            str(face_path),
            '--model-root',
            str(model_root),
            '--device-id',
            str(device_id),
        ]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as helper_exc:
            raise RuntimeError(
                f'ArcFace helper failed after local import failed ({local_exc}); stderr={helper_exc.stderr.strip()}'
            ) from helper_exc
        json_line = next((line for line in reversed(proc.stdout.splitlines()) if line.strip().startswith('[')), '')
        if not json_line:
            raise RuntimeError(f'ArcFace helper produced no JSON embedding; stdout={proc.stdout[-1000:]}')
        emb = np.asarray(json.loads(json_line), dtype='float32')
        if emb.shape != (512,):
            raise RuntimeError(f'ArcFace helper returned invalid shape {emb.shape}, expected (512,)')
        return emb


def choose_dtype(name: str) -> torch.dtype:
    return {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}.get(name, torch.bfloat16)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)
