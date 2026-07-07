from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import cv2
import insightface
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description='Emit a real ArcFace normed embedding for one face crop as JSON.')
    parser.add_argument('--image', required=True)
    parser.add_argument('--model-root', default='/data/muxiangyu/modelLibrary/insightface')
    parser.add_argument('--device-id', type=int, default=0)
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f'failed to read image: {image_path}')

    app = insightface.app.FaceAnalysis(
        name='antelopev2',
        root=str(args.model_root),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    app.prepare(ctx_id=args.device_id, det_size=(224, 224))
    faces = app.get(image)
    if not faces:
        raise SystemExit(f'ArcFace detector found no face: {image_path}')
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = np.asarray(best.normed_embedding, dtype=np.float32)
    if emb.shape != (512,):
        raise SystemExit(f'invalid embedding shape: {emb.shape}')
    print(json.dumps(emb.tolist()))


if __name__ == '__main__':
    main()
