from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from train_recognizer import TrainArcFaceRecognizer, differentiable_face_align


CHECKPOINT = Path('/data/muxiangyu/modelLibrary/insightface/models/antelopev2/glintr100.onnx')


def test_differentiable_face_alignment_propagates_gradient() -> None:
    image = torch.randn(1, 3, 112, 112, requires_grad=True)
    matrix = np.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32)
    aligned = differentiable_face_align(image, matrix, image_size=112)
    assert aligned.shape == image.shape
    aligned.square().mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert float(image.grad.norm()) > 0.0


@pytest.mark.skipif(not CHECKPOINT.exists(), reason='local Glint360K ArcFace checkpoint unavailable')
def test_onnx2torch_arcface_matches_onnxruntime() -> None:
    import onnxruntime as ort

    cfg = {
        'checkpoint': str(CHECKPOINT),
        'backend': 'onnx2torch_glintr100',
        'architecture': 'iresnet100',
        'embedding_dim': 512,
        'input_size': 112,
    }
    recognizer = TrainArcFaceRecognizer(cfg, torch.device('cpu'))
    rng = np.random.default_rng(19)
    values = rng.uniform(-1.0, 1.0, size=(1, 3, 112, 112)).astype(np.float32)
    session = ort.InferenceSession(str(CHECKPOINT), providers=['CPUExecutionProvider'])
    expected = session.run(None, {session.get_inputs()[0].name: values})[0]
    expected = F.normalize(torch.from_numpy(expected).float(), dim=1).numpy()
    actual = recognizer(torch.from_numpy(values)).detach().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-5)

