"""Automatic all-split identity-group audit features, never a final evaluator.

Uses the same existing ArcFace model for every split, only for dataset
grouping. Does not make train/dev/final assignments or identity truth claims.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model-root', type=Path, default=Path('/data/muxiangyu/modelLibrary/insightface'))
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--max-hours', type=float, default=8)
    args = p.parse_args()
    model_dir = args.model_root / 'models/antelopev2'
    for name in ('glintr100.onnx', 'scrfd_10g_bnkps.onnx'):
        if not (model_dir / name).is_file():
            raise FileNotFoundError('Local model required; will not download: ' + name)
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    import cv2
    import onnxruntime as ort
    import insightface
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('CUDA ONNX runtime unavailable: do not silently run entire corpus on CPU')
    app = insightface.app.FaceAnalysis(name='antelopev2', root=str(args.model_root),
        allowed_modules=['detection', 'recognition'],
        providers=[('CUDAExecutionProvider', {'device_id': args.device}), 'CPUExecutionProvider'])
    app.prepare(ctx_id=args.device, det_size=(640, 640))
    effective = {key: model.session.get_providers() for key, model in app.models.items()}
    if any('CUDAExecutionProvider' not in providers for providers in effective.values()):
        raise RuntimeError('Model session silently fell back from CUDA: ' + str(effective))
    with (args.root / 'metadata/manifest.csv').open(newline='') as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: r['id'])
    if args.limit:
        rows = rows[:args.limit]
    manifest = {'purpose': 'all-split dataset grouping only, NOT model evaluation',
        'expected': len(rows), 'device': args.device, 'max_hours': args.max_hours,
        'det_size': [640, 640], 'min_detection_confidence': .3,
        'effective_providers': effective,
        'script_sha256': sha256(__file__),
        'manifest_sha256': sha256(args.root / 'metadata/manifest.csv'),
        'models': {name: sha256(model_dir / name) for name in ('glintr100.onnx', 'scrfd_10g_bnkps.onnx')}}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    start = time.monotonic()
    all_ids, embeddings, chunk_ids = [], [], []
    counts = {'ok': 0, 'failed': 0}
    part = 0
    completed = 0
    timed_out = False

    def flush_chunk():
        nonlocal part
        if chunk_ids:
            np.savez(args.out / f'embeddings_{part:04d}.npz',
                     ids=np.asarray(chunk_ids), embeds=np.stack(embeddings))
            embeddings.clear()
            chunk_ids.clear()
            part += 1

    with (args.out / 'per_image.jsonl').open('x') as output:
        for row in rows:
            if time.monotonic() - start > args.max_hours * 3600:
                timed_out = True
                break
            sid = row['id']
            path = Path(row['human_path'])
            entry = {'id': sid, 'split': row['split'], 'path': str(path)}
            try:
                bgr = cv2.imread(str(path))
                if bgr is None:
                    raise ValueError('unreadable source image')
                faces = app.get(bgr)
                if not faces:
                    raise ValueError('face not detected')
                face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
                emb = np.asarray(face.normed_embedding, dtype=np.float32)
                if float(face.det_score) < .3 or emb.shape != (512,) or not np.isfinite(emb).all():
                    raise ValueError('face/embedding fails fixed validity threshold')
                chunk_ids.append(sid)
                embeddings.append(emb)
                all_ids.append(sid)
                entry.update(status='ok', det_score=float(face.det_score), input_sha256=sha256(path))
                counts['ok'] += 1
            except Exception as exc:
                entry.update(status='failed', error=str(exc))
                counts['failed'] += 1
            output.write(json.dumps(entry, allow_nan=False) + '\n')
            output.flush()
            completed += 1
            if len(chunk_ids) >= 256:
                flush_chunk()
            if completed % 100 == 0 or completed == len(rows):
                progress = {'completed': completed, 'expected': len(rows), **counts,
                            'seconds': time.monotonic() - start}
                (args.out / 'progress.json').write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)
        flush_chunk()
    summary = {'expected': len(rows), 'completed': completed, **counts,
               'timed_out': timed_out, 'seconds': time.monotonic() - start,
               'gpu_hours_reserved_estimate': (time.monotonic() - start) / 3600,
               'status': 'features_complete' if completed == len(rows) else 'partial',
               'formal_split_ready': False,
               'pending': 'cluster with calibrated thresholds and source/duplicate union; unresolved faces quarantined'}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    if completed == len(rows):
        (args.out / 'FEATURES_READY').write_text('coverage audit complete; identity clustering not yet done\n')
    print(json.dumps(summary), flush=True)
    if timed_out:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
