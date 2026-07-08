from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from PIL import Image

from conditions import find_one


B2_NAME_RE = re.compile(r'^(?P<mid>.+)__id(?P<jid>.+)__seed(?P<seed>\d+)\.png$')


class MetricUnavailable(RuntimeError):
    """Raised when an official metric cannot run because required weights/code are absent."""


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def sha256_short(path: str | Path | None, length: int = 12) -> str:
    if path is None:
        return 'none'
    path = Path(path)
    if not path.exists() or not path.is_file():
        return 'missing'
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()[:length]


def safe_values(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def safe_mean(values: list[Any]) -> float | None:
    vals = safe_values(values)
    return mean(vals) if vals else None


def safe_median(values: list[Any]) -> float | None:
    vals = safe_values(values)
    return median(vals) if vals else None


def percentile(values: list[Any], q: float) -> float | None:
    vals = safe_values(values)
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=np.float32), q))


def fmt(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 'n/a'
    if not math.isfinite(v):
        return 'n/a'
    return f'{v:.{digits}f}'


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def parse_b2_name(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    match = B2_NAME_RE.match(path.name)
    if not match:
        raise ValueError(f'not a B2 generated filename: {path.name}')
    row = match.groupdict()
    row['seed'] = int(row['seed'])
    return row


def expected_rows(cfg: dict[str, Any], subset: dict[str, Any], gen_dir: str | Path) -> list[dict[str, Any]]:
    gen_dir = Path(gen_dir)
    default_seeds = cfg.get('eval', {}).get('b2_seeds', [0, 1])
    rows = []
    for pair in subset['pairs']:
        mid = pair['mannequin_id']
        jid = pair['identity_id']
        for seed in pair.get('seeds', default_seeds):
            rows.append({
                'mid': mid,
                'jid': jid,
                'seed': int(seed),
                'garment_type': pair.get('garment_type', subset.get('garment_types', {}).get(mid, 'unknown')),
                'theta_source': pair.get('theta_source', mid),
                'path': gen_dir / f'{mid}__id{jid}__seed{int(seed)}.png',
            })
    return rows


def read_rgb(path: str | Path, size: tuple[int, int] | int | None = None) -> np.ndarray:
    image = Image.open(path).convert('RGB')
    if isinstance(size, int):
        size = (size, size)
    if size is not None and image.size != tuple(size):
        image = image.resize(tuple(size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def image_path(root: Path, folder: str, sample_id: str) -> Path:
    return find_one(root / folder, sample_id)


def face_size_bucket(face_size_px: float | None) -> str:
    if face_size_px is None or not math.isfinite(float(face_size_px)):
        return 'n/a'
    value = float(face_size_px)
    if value < 80:
        return '<80'
    if value <= 120:
        return '80-120'
    return '>120'


def yaw_bucket(yaw: float | None) -> str:
    if yaw is None or not math.isfinite(float(yaw)):
        return 'n/a'
    value = abs(float(yaw))
    if value < 15:
        return 'front'
    if value < 45:
        return 'three-quarter'
    return 'side'


def angle_diff_deg(pred: float, target: float) -> float:
    diff = (float(pred) - float(target) + 180.0) % 360.0 - 180.0
    if diff <= -90.0:
        diff += 180.0
    if diff > 90.0:
        diff -= 180.0
    return abs(diff)


def plot_histogram(path: str | Path, values: list[float], title: str, xlabel: str) -> None:
    vals = safe_values(values)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not vals:
        return
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=30, color='#3b82f6', edgecolor='white')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def group_summary(rows: list[dict[str, Any]], key: str, value_key: str) -> dict[str, Any]:
    groups: dict[str, list[Any]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, 'unknown')), []).append(row.get(value_key))
    return {
        name: {
            'count': len(safe_values(values)),
            'mean': safe_mean(values),
            'median': safe_median(values),
        }
        for name, values in sorted(groups.items())
    }

