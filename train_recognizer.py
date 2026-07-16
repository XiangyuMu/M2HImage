from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from conditions import sha256_file


# Evaluation-only AdaFace is deliberately not imported here. Training references,
# semi-hard distances, and A4 losses must stay in this Glint360K ArcFace space.


@dataclass(frozen=True)
class FaceGeometry:
    bbox: np.ndarray
    landmarks: np.ndarray
    confidence: float


class TrainArcFaceRecognizer(nn.Module):
    def __init__(self, cfg: dict[str, Any], device: torch.device):
        super().__init__()
        checkpoint = Path(cfg.get('checkpoint', ''))
        if not checkpoint.exists():
            raise FileNotFoundError(
                f'F_train ArcFace checkpoint missing: {checkpoint}. '
                'Expected InsightFace Glint360K glintr100.onnx or a configured equivalent.'
            )
        if 'adaface' in str(checkpoint).lower():
            raise RuntimeError('held-out AdaFace checkpoint is forbidden in training code')
        backend = str(cfg.get('backend', 'onnx2torch_glintr100'))
        architecture = str(cfg.get('architecture', 'iresnet100'))
        if backend != 'onnx2torch_glintr100' or architecture != 'iresnet100':
            raise RuntimeError(
                f'unsupported F_train configuration backend={backend}, architecture={architecture}; '
                'no fallback is allowed'
            )
        import onnx
        from onnx2torch import convert

        self.cfg = cfg
        self.checkpoint = checkpoint
        self.checkpoint_hash = sha256_file(checkpoint)[:16]
        self.model = convert(onnx.load(str(checkpoint), load_external_data=True))
        self.model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
        self.device = device
        self.embedding_dim = int(cfg.get('embedding_dim', 512))
        self.input_size = int(cfg.get('input_size', 112))
        self.eval()

    def train(self, mode: bool = True) -> 'TrainArcFaceRecognizer':
        super().train(False)
        self.model.eval()
        return self

    def forward(self, aligned_rgb: torch.Tensor) -> torch.Tensor:
        if aligned_rgb.ndim != 4 or tuple(aligned_rgb.shape[-2:]) != (self.input_size, self.input_size):
            raise RuntimeError(
                f'F_train expects (B,3,{self.input_size},{self.input_size}), got {tuple(aligned_rgb.shape)}'
            )
        output = self.model(aligned_rgb.to(device=self.device, dtype=torch.float32))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim != 2 or output.shape[1] != self.embedding_dim:
            raise RuntimeError(f'F_train output shape={tuple(output.shape)}, expected (B,{self.embedding_dim})')
        return F.normalize(output.float(), dim=1)

    @torch.no_grad()
    def embed_aligned_rgb_arrays(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        array = np.stack(images).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
        return self(tensor).cpu().numpy().astype(np.float32)

    def launch_note(self) -> dict[str, Any]:
        return {
            'type': 'PyTorch-native ArcFace via onnx2torch',
            'architecture': 'InsightFace arcface_torch-family iresnet100',
            'training_data': 'Glint360K',
            'checkpoint': str(self.checkpoint),
            'checkpoint_hash': self.checkpoint_hash,
            'backend': 'onnx2torch_glintr100',
            'heldout_adaface_used': False,
            'trainable': 0,
        }


class RetinaFaceGeometryDetector:
    def __init__(self, cfg: dict[str, Any], device_id: int = -1):
        import onnxruntime as ort
        from insightface.model_zoo import model_zoo

        self.cfg = dict(cfg)
        model_root = Path(cfg.get('detector_model_root', '/data/muxiangyu/modelLibrary/insightface'))
        if not model_root.exists():
            raise FileNotFoundError(f'RetinaFace model root missing: {model_root}')
        detector_checkpoint = Path(cfg.get(
            'detector_checkpoint',
            model_root / 'models' / str(cfg.get('detector_name', 'antelopev2')) / 'scrfd_10g_bnkps.onnx',
        ))
        if not detector_checkpoint.exists():
            raise FileNotFoundError(f'RetinaFace detector checkpoint missing: {detector_checkpoint}')
        providers = ['CPUExecutionProvider']
        if int(device_id) >= 0:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = int(cfg.get('detector_intra_op_threads', 8))
        session_options.inter_op_num_threads = int(cfg.get('detector_inter_op_threads', 1))
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.det_model = model_zoo.get_model(
            str(detector_checkpoint),
            providers=providers,
            sess_options=session_options,
        )
        det_size = int(cfg.get('det_size', 640))
        self.min_face_crop_px = int(cfg.get('detector_min_face_crop_px', 256))
        self.det_model.prepare(
            ctx_id=int(device_id),
            input_size=(det_size, det_size),
            det_thresh=float(cfg.get('detector_threshold', 0.5)),
        )

    def detect_bgr(self, bgr: np.ndarray) -> FaceGeometry:
        bboxes, keypoints = self.det_model.detect(bgr, max_num=0, metric='default')
        if bboxes.shape[0] == 0 or keypoints is None:
            raise RuntimeError('RetinaFace found no face')
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        index = int(np.argmax(areas))
        landmarks = np.asarray(keypoints[index], dtype=np.float32)
        if landmarks.shape != (5, 2):
            raise RuntimeError(f'RetinaFace landmarks shape={landmarks.shape}, expected (5,2)')
        return FaceGeometry(
            bbox=np.asarray(bboxes[index, :4], dtype=np.float32),
            landmarks=landmarks,
            confidence=float(bboxes[index, 4]),
        )

    def align_path(self, path: str | Path, image_size: int = 112) -> tuple[np.ndarray, FaceGeometry]:
        from insightface.utils.face_align import norm_crop

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'failed to read face crop: {path}')
        pad_fraction = float(self.cfg.get('detector_face_crop_pad', 0.35))
        if pad_fraction > 0.0:
            pad_y = int(round(bgr.shape[0] * pad_fraction))
            pad_x = int(round(bgr.shape[1] * pad_fraction))
            bgr = cv2.copyMakeBorder(
                bgr,
                pad_y,
                pad_y,
                pad_x,
                pad_x,
                borderType=cv2.BORDER_CONSTANT,
                value=(127, 127, 127),
            )
        short_side = min(bgr.shape[:2])
        if short_side < self.min_face_crop_px:
            scale = self.min_face_crop_px / max(short_side, 1)
            bgr = cv2.resize(
                bgr,
                (int(round(bgr.shape[1] * scale)), int(round(bgr.shape[0] * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        geometry = self.detect_bgr(bgr)
        aligned = norm_crop(bgr, landmark=geometry.landmarks, image_size=int(image_size))
        return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB), geometry

    @torch.no_grad()
    def detect_tensor_batch(self, images_rgb_minus1_1: torch.Tensor) -> list[FaceGeometry | None]:
        arrays = (
            ((images_rgb_minus1_1.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
            .permute(0, 2, 3, 1)
            .numpy()
            .round()
            .astype(np.uint8)
        )
        result: list[FaceGeometry | None] = []
        for rgb in arrays:
            try:
                result.append(self.detect_bgr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
            except RuntimeError:
                result.append(None)
        return result


def similarity_matrices(geometries: list[FaceGeometry], image_size: int = 112) -> np.ndarray:
    from insightface.utils.face_align import estimate_norm

    matrices = []
    for geometry in geometries:
        matrix = estimate_norm(geometry.landmarks, image_size=int(image_size), mode='arcface')
        matrices.append(np.asarray(matrix, dtype=np.float32))
    return np.stack(matrices)


def differentiable_face_align(
    images: torch.Tensor,
    matrices_source_to_aligned: np.ndarray | torch.Tensor,
    image_size: int = 112,
) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise RuntimeError(f'face alignment expects BCHW RGB tensor, got {tuple(images.shape)}')
    batch, _, height, width = images.shape
    matrices = torch.as_tensor(
        matrices_source_to_aligned,
        device=images.device,
        dtype=torch.float32,
    )
    if matrices.shape != (batch, 2, 3):
        raise RuntimeError(f'alignment matrices shape={tuple(matrices.shape)}, expected {(batch, 2, 3)}')
    homogeneous = torch.eye(3, device=images.device, dtype=torch.float32).unsqueeze(0).repeat(batch, 1, 1)
    homogeneous[:, :2] = matrices
    inverse = torch.linalg.inv(homogeneous)
    ys, xs = torch.meshgrid(
        torch.arange(image_size, device=images.device, dtype=torch.float32),
        torch.arange(image_size, device=images.device, dtype=torch.float32),
        indexing='ij',
    )
    target = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).reshape(1, -1, 3)
    source = torch.bmm(target.expand(batch, -1, -1), inverse.transpose(1, 2))
    x = source[..., 0].reshape(batch, image_size, image_size)
    y = source[..., 1].reshape(batch, image_size, image_size)
    grid = torch.stack([
        2.0 * x / max(width - 1, 1) - 1.0,
        2.0 * y / max(height - 1, 1) - 1.0,
    ], dim=-1)
    return F.grid_sample(
        images.float(),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
