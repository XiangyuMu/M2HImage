from __future__ import annotations

from typing import Any

import numpy as np

_FACENET_MODEL = None
_MTCNN_MODEL = None

def crop_face_array(image, face_box: tuple[int, int, int, int]) -> np.ndarray:
    return np.asarray(image.crop(face_box).resize((160, 160)), dtype=np.float32) / 255.0


def mock_embedding(image, face_box: tuple[int, int, int, int], embedding_dim: int = 512) -> np.ndarray:
    crop = crop_face_array(image, face_box)
    pooled = crop.reshape(10, 16, 10, 16, 3).mean(axis=(1, 3)).reshape(-1)
    rng = np.random.default_rng(1234)
    projection = rng.standard_normal((pooled.shape[0], embedding_dim), dtype=np.float32)
    embedding = pooled @ projection
    return embedding / max(np.linalg.norm(embedding), 1e-8)


def reference_face_box(image, config: dict[str, Any]) -> tuple[int, int, int, int]:
    backend = config["identity"].get("backend", "mock")
    if backend == "facenet":
        from facenet_pytorch import MTCNN

        global _MTCNN_MODEL
        if _MTCNN_MODEL is None:
            _MTCNN_MODEL = MTCNN(keep_all=False)
        boxes, _ = _MTCNN_MODEL.detect(image)
        if boxes is None or len(boxes) == 0:
            raise RuntimeError("No face detected in the reference identity image.")
        left, top, right, bottom = boxes[0]
        left, top = max(0, round(left)), max(0, round(top))
        right, bottom = min(image.width, round(right)), min(image.height, round(bottom))
        if right <= left or bottom <= top:
            raise RuntimeError("Detected face bounding box is outside the image.")
        return (left, top, right, bottom)
    width, height = image.size
    return (round(width * 0.38), round(height * 0.08), round(width * 0.62), round(height * 0.28))


def offline_identity_embedding(image, face_box: tuple[int, int, int, int], config: dict[str, Any]) -> np.ndarray:
    backend = config["identity"].get("backend", "mock")
    if backend == "mock":
        return mock_embedding(image, face_box, int(config["identity"].get("embedding_dim", 512)))
    if backend == "facenet":
        import torch
        from facenet_pytorch import InceptionResnetV1

        global _FACENET_MODEL
        crop = crop_face_array(image, face_box)
        tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        if _FACENET_MODEL is None:
            _FACENET_MODEL = InceptionResnetV1(pretrained="vggface2").eval()
        with torch.no_grad():
            return _FACENET_MODEL(tensor).squeeze(0).cpu().numpy()
    raise ValueError(f"Unsupported identity backend: {backend}")


def build_differentiable_identity_encoder(config: dict[str, Any]):
    import torch
    from torch import nn

    backend = config["identity"].get("backend", "mock")
    embedding_dim = int(config["identity"].get("embedding_dim", 512))
    if backend == "facenet":
        from facenet_pytorch import InceptionResnetV1

        model = InceptionResnetV1(pretrained="vggface2").eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    class MockIdentityEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            generator = torch.Generator().manual_seed(1234)
            matrix = torch.randn(300, embedding_dim, generator=generator)
            self.register_buffer("projection", matrix)

        def forward(self, images):
            pooled = nn.functional.adaptive_avg_pool2d((images + 1) / 2, (10, 10)).flatten(1)
            embedding = pooled @ self.projection
            return nn.functional.normalize(embedding, dim=-1)

    return MockIdentityEncoder()
