from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from metrics.common import (
    MetricUnavailable, angle_diff_deg, expected_rows, group_summary, load_json, plot_histogram, safe_mean,
    safe_median, sha256_short, write_csv, write_json, yaw_bucket,
)


MANUAL_6DREPNET = (
    'Configure metrics.headpose.command to the same 6DRepNet runner used for dataset extraction. The command must '
    'print JSON containing yaw, pitch, roll, status, det_confidence, and face_size_px for one image. Template variables: '
    '{image}, {device}, {checkpoint}. Example: python path/to/run_6drepnet_one.py --image {image} --device {device}.'
)


class UnifaceHeadPoseRunner:
    def __init__(self, cache_dir: str | Path, checkpoint: str | Path, providers: list[str] | None = None):
        from uniface import set_cache_dir
        from uniface.constants import HeadPoseWeights
        from uniface.headpose import HeadPose

        cache_dir = Path(cache_dir)
        checkpoint = Path(checkpoint)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['UNIFACE_CACHE_DIR'] = str(cache_dir)
        set_cache_dir(str(cache_dir))
        if not checkpoint.exists():
            raise MetricUnavailable(f'UniFace headpose checkpoint not found: {checkpoint}. {MANUAL_6DREPNET}')
        self.model = HeadPose(model_name=HeadPoseWeights.RESNET18, providers=providers or ['CPUExecutionProvider'])
        self.det_conf_key = 'det_confidence'

    @staticmethod
    def _crop_largest_face(path: str | Path) -> tuple[np.ndarray, float, float]:
        import insightface

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'failed to read image: {path}')
        app = getattr(UnifaceHeadPoseRunner, '_detector', None)
        if app is None:
            app = insightface.app.FaceAnalysis(
                name='antelopev2',
                root='/data/muxiangyu/modelLibrary/insightface',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            UnifaceHeadPoseRunner._detector = app
        faces = app.get(bgr)
        if not faces:
            raise RuntimeError('RetinaFace found no face')
        face = max(faces, key=lambda f: (float(f.bbox[2]) - float(f.bbox[0])) * (float(f.bbox[3]) - float(f.bbox[1])))
        x0, y0, x1, y1 = [float(v) for v in face.bbox]
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        face_size = max(w, h)
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        side = max(w, h) * 1.3
        ix0 = max(0, int(round(cx - side * 0.5)))
        ix1 = min(bgr.shape[1], int(round(cx + side * 0.5)))
        iy0 = max(0, int(round(cy - side * 0.5)))
        iy1 = min(bgr.shape[0], int(round(cy + side * 0.5)))
        crop = bgr[iy0:iy1, ix0:ix1]
        if crop.size == 0:
            raise RuntimeError('expanded face crop is empty')
        return crop, face_size, float(getattr(face, 'det_score', 0.0))

    def predict(self, image: str | Path) -> dict[str, Any]:
        crop, face_size, det_conf = self._crop_largest_face(image)
        result = self.model.estimate(crop)
        return {
            'status': 'ok',
            'yaw': float(result.yaw),
            'pitch': float(result.pitch),
            'roll': float(result.roll),
            'det_confidence': det_conf,
            'face_size_px': face_size,
        }


class External6DRepNetRunner:
    def __init__(self, command: str, checkpoint: str | None, device: str):
        if not command:
            raise MetricUnavailable(f'6DRepNet command is not configured. {MANUAL_6DREPNET}')
        self.command = command
        self.checkpoint = checkpoint or ''
        self.device = device

    def predict(self, image: str | Path) -> dict[str, Any]:
        cmd = self.command.format(image=str(image), device=self.device, checkpoint=self.checkpoint)
        proc = subprocess.run(shlex.split(cmd), check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f'6DRepNet command failed rc={proc.returncode}: {proc.stderr.strip()[:500]}')
        text = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ''
        if not text:
            raise RuntimeError('6DRepNet command produced no JSON stdout')
        payload = json.loads(text)
        for key in ('yaw', 'pitch', 'roll'):
            if key not in payload:
                raise RuntimeError(f'6DRepNet output missing {key}')
        return payload


def _metric_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get('metrics', {}).get('headpose', {})


def _blocked_summary(out_dir: Path, reason: str, cfg: dict[str, Any]) -> dict[str, Any]:
    mcfg = _metric_cfg(cfg)
    summary = {
        'status': 'blocked',
        'reason': reason,
        'runner': mcfg.get('command', ''),
        'checkpoint': str(mcfg.get('checkpoint', '')),
        'checkpoint_hash': sha256_short(mcfg.get('checkpoint')),
        'manual_setup': MANUAL_6DREPNET,
        'csv': str(out_dir / 'headpose_per_image.csv'),
    }
    write_csv(out_dir / 'headpose_per_image.csv', [], [
        'mid', 'jid', 'seed', 'path', 'status', 'error', 'target_yaw', 'target_pitch', 'target_roll', 'pred_yaw',
        'pred_pitch', 'pred_roll', 'yaw_abs_err', 'pitch_abs_err', 'roll_abs_err', 'target_yaw_bucket',
        'det_conf', 'face_size_px',
    ])
    write_json(out_dir / 'headpose_summary.json', summary)
    return summary


def target_pose(root: Path, theta_source: str) -> dict[str, Any]:
    path = root / 'derived/head_pose_6drepnet/human' / f'{theta_source}.json'
    payload = load_json(path)
    if payload.get('status', 'ok') != 'ok':
        raise RuntimeError(f'target head pose status is not ok: {path}')
    return payload


def run_headpose_mae(
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
    try:
        runner_name = str(mcfg.get('runner') or '').lower()
        if runner_name == 'uniface_headpose':
            providers = ['CPUExecutionProvider']
            if str(device).startswith('cuda'):
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            runner = UnifaceHeadPoseRunner(
                cache_dir=mcfg.get('cache_dir', Path(str(mcfg.get('checkpoint') or '')).parent),
                checkpoint=str(mcfg.get('checkpoint') or ''),
                providers=providers,
            )
        else:
            runner = External6DRepNetRunner(
                command=str(mcfg.get('command') or ''),
                checkpoint=str(mcfg.get('checkpoint') or ''),
                device=device,
            )
    except MetricUnavailable as exc:
        summary = _blocked_summary(out_dir, str(exc), cfg)
        if fail_on_unavailable:
            raise
        return summary

    root = Path(cfg['data']['root'])
    rows = expected_rows(cfg, subset, gen_dir)
    targets = {}
    csv_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc='headpose MAE'):
        out = {
            'mid': row['mid'],
            'jid': row['jid'],
            'seed': row['seed'],
            'path': str(row['path']),
            'status': 'ok',
            'error': '',
            'target_yaw': '',
            'target_pitch': '',
            'target_roll': '',
            'pred_yaw': '',
            'pred_pitch': '',
            'pred_roll': '',
            'yaw_abs_err': '',
            'pitch_abs_err': '',
            'roll_abs_err': '',
            'target_yaw_bucket': '',
            'det_conf': '',
            'face_size_px': '',
        }
        try:
            if not row['path'].exists():
                raise RuntimeError('generated image missing')
            theta_source = row.get('theta_source', row['mid'])
            if theta_source not in targets:
                targets[theta_source] = target_pose(root, theta_source)
            tgt = targets[theta_source]
            pred = runner.predict(row['path'])
            if pred.get('status', 'ok') != 'ok':
                raise RuntimeError(f'generated headpose status={pred.get("status")}')
            out.update({
                'target_yaw': float(tgt['yaw']),
                'target_pitch': float(tgt['pitch']),
                'target_roll': float(tgt['roll']),
                'pred_yaw': float(pred['yaw']),
                'pred_pitch': float(pred['pitch']),
                'pred_roll': float(pred['roll']),
                'yaw_abs_err': angle_diff_deg(pred['yaw'], tgt['yaw']),
                'pitch_abs_err': angle_diff_deg(pred['pitch'], tgt['pitch']),
                'roll_abs_err': angle_diff_deg(pred['roll'], tgt['roll']),
                'target_yaw_bucket': yaw_bucket(float(tgt['yaw'])),
                'det_conf': pred.get('det_confidence', pred.get('det_conf', '')),
                'face_size_px': pred.get('face_size_px', ''),
            })
        except Exception as exc:  # noqa: BLE001
            out['status'] = 'failed'
            out['error'] = str(exc)
        csv_rows.append(out)

    fieldnames = [
        'mid', 'jid', 'seed', 'path', 'status', 'error', 'target_yaw', 'target_pitch', 'target_roll', 'pred_yaw',
        'pred_pitch', 'pred_roll', 'yaw_abs_err', 'pitch_abs_err', 'roll_abs_err', 'target_yaw_bucket',
        'det_conf', 'face_size_px',
    ]
    write_csv(out_dir / 'headpose_per_image.csv', csv_rows, fieldnames)
    ok_rows = [row for row in csv_rows if row['status'] == 'ok']
    plot_histogram(out_dir / 'headpose_yaw_abs_err_hist.png', [row.get('yaw_abs_err') for row in ok_rows], 'Head Pose Yaw MAE', '|generated yaw - target yaw|')
    summary = {
        'status': 'ok',
        'runner': mcfg.get('runner') or mcfg.get('command'),
        'checkpoint': str(mcfg.get('checkpoint', '')),
        'checkpoint_hash': sha256_short(mcfg.get('checkpoint')),
        'angle_convention': 'generated head pose from configured runner; target yaw/pitch/roll from derived/head_pose_6drepnet; errors folded into (-90, 90]',
        'count': len(ok_rows),
        'failed': len(csv_rows) - len(ok_rows),
        'yaw_mae_mean': safe_mean([row.get('yaw_abs_err') for row in ok_rows]),
        'yaw_mae_median': safe_median([row.get('yaw_abs_err') for row in ok_rows]),
        'pitch_mae_mean': safe_mean([row.get('pitch_abs_err') for row in ok_rows]),
        'pitch_mae_median': safe_median([row.get('pitch_abs_err') for row in ok_rows]),
        'roll_mae_mean': safe_mean([row.get('roll_abs_err') for row in ok_rows]),
        'roll_mae_median': safe_median([row.get('roll_abs_err') for row in ok_rows]),
        'yaw_bucket_yaw_mae': group_summary(ok_rows, 'target_yaw_bucket', 'yaw_abs_err'),
        'csv': str(out_dir / 'headpose_per_image.csv'),
        'histogram': str(out_dir / 'headpose_yaw_abs_err_hist.png'),
    }
    write_json(out_dir / 'headpose_summary.json', summary)
    return summary

