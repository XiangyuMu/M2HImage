from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from conditions import find_one
from metrics.common import (
    MetricUnavailable, cosine, expected_rows, face_size_bucket, group_summary, plot_histogram, safe_mean, safe_median,
    sha256_short, write_csv, write_json,
)


MANUAL_ADAFACE = (
    'Install the official AdaFace repo and IR-101 checkpoint, then set metrics.heldout_id.repo and '
    'metrics.heldout_id.checkpoint in configs/warmup.yaml. Expected model: AdaFace IR-101, e.g. '
    'adaface_ir101_webface12m.ckpt. Do not use InsightFace antelopev2/glintr100 for this metric.'
)


class RetinaFaceAligner:
    def __init__(self, model_root: str | Path, device_id: int = 0, det_size: int = 640):
        import insightface

        self.app = insightface.app.FaceAnalysis(
            name='antelopev2',
            root=str(model_root),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.app.prepare(ctx_id=int(device_id), det_size=(int(det_size), int(det_size)))

    def align(self, path: str | Path, expand: float = 1.3, min_crop: int = 256) -> tuple[np.ndarray, float, float]:
        from insightface.utils.face_align import norm_crop

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'failed to read image: {path}')
        faces = self.app.get(bgr)
        if not faces:
            raise RuntimeError('RetinaFace found no face')
        face = max(faces, key=lambda f: (float(f.bbox[2]) - float(f.bbox[0])) * (float(f.bbox[3]) - float(f.bbox[1])))
        x0, y0, x1, y1 = [float(v) for v in face.bbox]
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        face_size = max(w, h)
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        side = max(w, h) * float(expand)
        ix0 = max(0, int(round(cx - side * 0.5)))
        ix1 = min(bgr.shape[1], int(round(cx + side * 0.5)))
        iy0 = max(0, int(round(cy - side * 0.5)))
        iy1 = min(bgr.shape[0], int(round(cy + side * 0.5)))
        if ix1 <= ix0 or iy1 <= iy0:
            raise RuntimeError('expanded face crop is empty')
        crop = bgr[iy0:iy1, ix0:ix1]
        kps = np.asarray(face.kps, dtype=np.float32) - np.asarray([ix0, iy0], dtype=np.float32)
        scale = 1.0
        crop_side = max(crop.shape[:2])
        if crop_side < int(min_crop):
            scale = float(min_crop) / float(crop_side)
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            kps = kps * scale
        aligned_bgr = norm_crop(crop, landmark=kps, image_size=112)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        det_conf = float(getattr(face, 'det_score', 0.0))
        return aligned_rgb, face_size, det_conf


def resize_face_crop_rgb(path: str | Path, image_size: int = 112) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f'failed to read image: {path}')
    resized = cv2.resize(bgr, (int(image_size), int(image_size)), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


class UnifaceAdaFaceRecognizer:
    def __init__(self, cache_dir: str | Path, checkpoint: str | Path, device: torch.device):
        from uniface import set_cache_dir
        from uniface.constants import AdaFaceWeights
        from uniface.recognition import AdaFace

        cache_dir = Path(cache_dir)
        checkpoint = Path(checkpoint)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['UNIFACE_CACHE_DIR'] = str(cache_dir)
        set_cache_dir(str(cache_dir))
        if not checkpoint.exists():
            raise MetricUnavailable(f'UniFace AdaFace IR101 checkpoint not found: {checkpoint}. {MANUAL_ADAFACE}')
        providers = ['CPUExecutionProvider']
        if str(device).startswith('cuda'):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.model = AdaFace(model_name=AdaFaceWeights.IR_101, providers=providers)
        self.device = device

    def embed_aligned_rgb(self, aligned_rgb: np.ndarray) -> np.ndarray:
        aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
        emb = self.model.get_embedding(aligned_bgr, landmarks=None).reshape(-1).astype('float32')
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb


class AdaFaceRecognizer:
    def __init__(self, repo: str | Path, checkpoint: str | Path, device: torch.device, architecture: str = 'ir_101'):
        repo = Path(repo)
        checkpoint = Path(checkpoint)
        if not repo.exists():
            raise MetricUnavailable(f'AdaFace repo not found: {repo}. {MANUAL_ADAFACE}')
        if not checkpoint.exists():
            raise MetricUnavailable(f'AdaFace checkpoint not found: {checkpoint}. {MANUAL_ADAFACE}')
        net_py = repo / 'net.py'
        if not net_py.exists():
            raise MetricUnavailable(f'AdaFace repo is missing net.py: {repo}. {MANUAL_ADAFACE}')
        spec = importlib.util.spec_from_file_location('adaface_net_external', net_py)
        if spec is None or spec.loader is None:
            raise MetricUnavailable(f'failed to import AdaFace net.py from {net_py}')
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(repo))
        try:
            spec.loader.exec_module(module)
        finally:
            try:
                sys.path.remove(str(repo))
            except ValueError:
                pass
        if not hasattr(module, 'build_model'):
            raise MetricUnavailable(f'AdaFace net.py has no build_model(): {net_py}')
        self.model = module.build_model(architecture)
        payload = torch.load(str(checkpoint), map_location='cpu')
        state = payload.get('state_dict', payload)
        cleaned = {}
        for key, value in state.items():
            new_key = key
            for prefix in ('model.', 'module.', 'backbone.'):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned[new_key] = value
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if len(missing) > 100:
            raise MetricUnavailable(f'AdaFace checkpoint load looks incompatible: missing={len(missing)}, unexpected={len(unexpected)}')
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def embed_aligned_rgb(self, aligned_rgb: np.ndarray) -> np.ndarray:
        arr = aligned_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.to(self.device)
        out = self.model(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = torch.nn.functional.normalize(out.float(), dim=1)
        return out[0].detach().cpu().numpy().astype('float32')


def _metric_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get('metrics', {}).get('heldout_id', {})


def _blocked_summary(out_dir: Path, reason: str, cfg: dict[str, Any]) -> dict[str, Any]:
    mcfg = _metric_cfg(cfg)
    summary = {
        'status': 'blocked',
        'reason': reason,
        'recognizer': 'AdaFace IR-101',
        'checkpoint': str(mcfg.get('checkpoint', '')),
        'checkpoint_hash': sha256_short(mcfg.get('checkpoint')),
        'manual_download': MANUAL_ADAFACE,
        'csv': str(out_dir / 'deltaid_per_image.csv'),
    }
    write_csv(out_dir / 'deltaid_per_image.csv', [], [
        'mid', 'jid', 'seed', 'path', 'status', 'error', 'sim_target', 'sim_source', 'delta_id', 'face_size_px',
        'face_size_bucket', 'det_conf', 'garment_type',
    ])
    write_json(out_dir / 'deltaid_summary.json', summary)
    return summary


def run_deltaid(
    cfg: dict[str, Any],
    subset: dict[str, Any],
    gen_dir: str | Path,
    out_dir: str | Path,
    device: str = 'cuda:0',
    fail_on_unavailable: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mcfg = _metric_cfg(cfg)
    repo = Path(mcfg.get('repo', ''))
    checkpoint = Path(mcfg.get('checkpoint', ''))
    try:
        if not str(checkpoint):
            raise MetricUnavailable(f'AdaFace checkpoint is not configured. {MANUAL_ADAFACE}')
        torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith('cuda') else 'cpu')
        recognizer_name = str(mcfg.get('recognizer', 'adaface_ir101')).lower()
        if recognizer_name in {'uniface_adaface_ir101', 'uniface_adaface_ir_101'}:
            recognizer = UnifaceAdaFaceRecognizer(
                cache_dir=mcfg.get('cache_dir', checkpoint.parent),
                checkpoint=checkpoint,
                device=torch_device,
            )
        else:
            if not str(repo):
                raise MetricUnavailable(f'AdaFace repo is not configured. {MANUAL_ADAFACE}')
            recognizer = AdaFaceRecognizer(repo, checkpoint, torch_device, architecture=mcfg.get('architecture', 'ir_101'))
        detector = RetinaFaceAligner(
            model_root=mcfg.get('detector_model_root', cfg.get('cache', {}).get('arcface_model_root', '/data/muxiangyu/modelLibrary/insightface')),
            device_id=int(mcfg.get('detector_device_id', 0 if str(torch_device).startswith('cuda') else -1)),
            det_size=int(mcfg.get('det_size', 640)),
        )
    except MetricUnavailable as exc:
        summary = _blocked_summary(out_dir, str(exc), cfg)
        if fail_on_unavailable:
            raise
        return summary

    root = Path(cfg['data']['root'])
    rows = expected_rows(cfg, subset, gen_dir)
    refs: dict[str, np.ndarray] = {}
    ref_errors: dict[str, str] = {}

    def ref_embedding(sample_id: str) -> np.ndarray:
        if sample_id in refs:
            return refs[sample_id]
        if sample_id in ref_errors:
            raise RuntimeError(ref_errors[sample_id])
        try:
            face_path = find_one(root / 'derived/face_crops/human', sample_id)
            try:
                aligned, _, _ = detector.align(
                    face_path,
                    expand=float(mcfg.get('ref_expand', 1.1)),
                    min_crop=int(mcfg.get('min_crop_px', 256)),
                )
            except RuntimeError as exc:
                if 'found no face' not in str(exc):
                    raise
                aligned = resize_face_crop_rgb(face_path)
            refs[sample_id] = recognizer.embed_aligned_rgb(aligned)
            return refs[sample_id]
        except Exception as exc:  # noqa: BLE001
            ref_errors[sample_id] = str(exc)
            raise

    csv_rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for row in tqdm(rows, desc='heldout deltaID'):
        out = {
            'mid': row['mid'],
            'jid': row['jid'],
            'seed': row['seed'],
            'path': str(row['path']),
            'garment_type': row.get('garment_type', 'unknown'),
            'status': 'ok',
            'error': '',
            'sim_target': '',
            'sim_source': '',
            'delta_id': '',
            'face_size_px': '',
            'face_size_bucket': '',
            'det_conf': '',
        }
        try:
            if not row['path'].exists():
                raise RuntimeError('generated image missing')
            aligned, face_size, det_conf = detector.align(
                row['path'],
                expand=float(mcfg.get('gen_expand', 1.3)),
                min_crop=int(mcfg.get('min_crop_px', 256)),
            )
            gen_emb = recognizer.embed_aligned_rgb(aligned)
            target = ref_embedding(row['jid'])
            source = ref_embedding(row['mid'])
            sim_target = cosine(gen_emb, target)
            sim_source = cosine(gen_emb, source)
            delta = sim_target - sim_source
            out.update({
                'sim_target': sim_target,
                'sim_source': sim_source,
                'delta_id': delta,
                'face_size_px': face_size,
                'face_size_bucket': face_size_bucket(face_size),
                'det_conf': det_conf,
            })
            deltas.append(delta)
        except Exception as exc:  # noqa: BLE001
            out['status'] = 'failed'
            out['error'] = str(exc)
        csv_rows.append(out)

    write_csv(out_dir / 'deltaid_per_image.csv', csv_rows, [
        'mid', 'jid', 'seed', 'path', 'status', 'error', 'sim_target', 'sim_source', 'delta_id', 'face_size_px',
        'face_size_bucket', 'det_conf', 'garment_type',
    ])
    plot_histogram(out_dir / 'deltaid_hist.png', deltas, 'Held-out DeltaID', 'sim(target identity) - sim(source identity)')
    ok_rows = [row for row in csv_rows if row['status'] == 'ok']
    summary = {
        'status': 'ok',
        'recognizer': 'UniFace AdaFace IR-101 ONNX' if str(mcfg.get('recognizer', '')).lower().startswith('uniface') else 'AdaFace IR-101',
        'repo': str(repo),
        'checkpoint': str(checkpoint),
        'checkpoint_hash': sha256_short(checkpoint),
        'detector': 'InsightFace antelopev2 RetinaFace detection/alignment only; recognition is held-out AdaFace',
        'count': len(ok_rows),
        'failed': len(csv_rows) - len(ok_rows),
        'mean': safe_mean([row.get('delta_id') for row in ok_rows]),
        'median': safe_median([row.get('delta_id') for row in ok_rows]),
        'sim_target_mean': safe_mean([row.get('sim_target') for row in ok_rows]),
        'sim_source_mean': safe_mean([row.get('sim_source') for row in ok_rows]),
        'face_size_buckets': group_summary(ok_rows, 'face_size_bucket', 'delta_id'),
        'garment_type': group_summary(ok_rows, 'garment_type', 'delta_id'),
        'csv': str(out_dir / 'deltaid_per_image.csv'),
        'histogram': str(out_dir / 'deltaid_hist.png'),
    }
    write_json(out_dir / 'deltaid_summary.json', summary)
    return summary

