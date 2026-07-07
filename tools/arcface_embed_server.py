from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import cv2
import insightface
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description='Persistent ArcFace JSONL embedding server.')
    parser.add_argument('--model-root', default='/data/muxiangyu/modelLibrary/insightface')
    parser.add_argument('--device-id', type=int, default=0)
    args = parser.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        app = insightface.app.FaceAnalysis(
            name='antelopev2',
            root=str(args.model_root),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        app.prepare(ctx_id=args.device_id, det_size=(640, 640))

    for line in sys.stdin:
        try:
            req = json.loads(line)
            image_path = Path(req['image'])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'failed to read image: {image_path}')
            with contextlib.redirect_stdout(sys.stderr):
                faces = app.get(image)
            if not faces:
                raise RuntimeError(f'ArcFace detector found no face: {image_path}')
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = np.asarray(best.normed_embedding, dtype=np.float32)
            if emb.shape != (512,):
                raise RuntimeError(f'invalid embedding shape: {emb.shape}')
            print(json.dumps({'ok': True, 'embedding': emb.tolist(), 'bbox': [float(v) for v in best.bbox], 'det_score': float(getattr(best, 'det_score', 0.0))}), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({'ok': False, 'error': str(exc)}), flush=True)


if __name__ == '__main__':
    main()
