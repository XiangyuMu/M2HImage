from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import cv2
import insightface
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description='Persistent ArcFace bank server: detection + recognition only.')
    parser.add_argument('--model-root', default='/data/muxiangyu/modelLibrary/insightface')
    parser.add_argument('--device-id', type=int, default=0)
    args = parser.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        app = insightface.app.FaceAnalysis(
            name='antelopev2',
            root=str(args.model_root),
            allowed_modules=['detection', 'recognition'],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        app.prepare(ctx_id=args.device_id, det_size=(224, 224))

    for line in sys.stdin:
        try:
            request = json.loads(line)
            image_path = Path(request['image'])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'failed to read image: {image_path}')
            with contextlib.redirect_stdout(sys.stderr):
                faces = app.get(image)
            if not faces:
                raise RuntimeError(f'ArcFace detector found no face: {image_path}')
            best = max(
                faces,
                key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
            )
            embedding = np.asarray(best.normed_embedding, dtype=np.float32)
            if embedding.shape != (512,):
                raise RuntimeError(f'invalid embedding shape: {embedding.shape}')
            print(json.dumps({
                'ok': True,
                'embedding': embedding.tolist(),
                'bbox': [float(value) for value in best.bbox],
                'det_score': float(getattr(best, 'det_score', 0.0)),
            }), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({'ok': False, 'error': str(exc)}), flush=True)


if __name__ == '__main__':
    main()
