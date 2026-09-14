from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from conditions import read_ids
from eval_b2 import garment_type
from spatial_conditions import load_fashn_labels


WATCHER_PROTOCOL_VERSION = "m2h-watcher-fixed-v1"
GARMENT_TYPES = ("top", "dress", "pants", "skirt")


def _canonical_payload(payload: dict[str, Any]) -> bytes:
    content = {key: value for key, value in payload.items() if key != "protocol_hash"}
    return json.dumps(
        content,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def protocol_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(payload)).hexdigest()


def build_watcher_eval_set(
    root: str | Path,
    split: str | Path,
    *,
    per_type: int = 4,
    min_hair_fraction: float = 0.01,
) -> dict[str, Any]:
    root = Path(root)
    split_path = Path(split)
    if not split_path.is_absolute():
        split_path = root / split_path
    candidates: dict[str, list[dict[str, Any]]] = {
        kind: [] for kind in GARMENT_TYPES
    }
    for sample_id in sorted(read_ids(split_path)):
        kind = garment_type(root, sample_id)
        if kind not in candidates:
            continue
        labels = load_fashn_labels(root, sample_id)
        hair_fraction = float((labels == 2).mean())
        if hair_fraction < min_hair_fraction:
            continue
        candidates[kind].append(
            {
                "id": sample_id,
                "garment_type": kind,
                "hair_fraction": hair_fraction,
            }
        )
    missing = {
        kind: len(rows)
        for kind, rows in candidates.items()
        if len(rows) < per_type
    }
    if missing:
        raise RuntimeError(
            f"watcher fixed set needs {per_type} visible-hair samples per type; "
            f"insufficient candidates={missing}"
        )
    selected_by_type = {
        kind: candidates[kind][:per_type] for kind in GARMENT_TYPES
    }
    # Interleave strata so any explicitly configured prefix remains balanced.
    samples = [
        selected_by_type[kind][rank]
        for rank in range(per_type)
        for kind in GARMENT_TYPES
    ]
    payload: dict[str, Any] = {
        "protocol_version": WATCHER_PROTOCOL_VERSION,
        "selection_rule": (
            "val split; ids lexicographically sorted; first four samples per "
            "top/dress/pants/skirt with FASHN human hair label-2 area >=1%; "
            "stored order is round-robin across garment types"
        ),
        "split": str(Path(split)),
        "hair_label": 2,
        "min_hair_fraction": float(min_hair_fraction),
        "per_garment_type": int(per_type),
        "garment_types": list(GARMENT_TYPES),
        "sample_ids": [row["id"] for row in samples],
        "samples": samples,
        "counts_by_garment_type": dict(
            Counter(row["garment_type"] for row in samples)
        ),
    }
    payload["protocol_hash"] = protocol_hash(payload)
    return payload


def write_watcher_eval_set(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_watcher_eval_set(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"frozen watcher evaluation set is missing: {path}; run "
            "python tools/build_watcher_eval_set.py"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != WATCHER_PROTOCOL_VERSION:
        raise RuntimeError(
            f"unsupported watcher protocol version in {path}: "
            f"{payload.get('protocol_version')!r}"
        )
    expected_hash = protocol_hash(payload)
    if payload.get("protocol_hash") != expected_hash:
        raise RuntimeError(
            f"watcher protocol hash mismatch in {path}: "
            f"stored={payload.get('protocol_hash')} computed={expected_hash}"
        )
    sample_ids = [str(value) for value in payload.get("sample_ids", [])]
    samples = payload.get("samples", [])
    if len(sample_ids) != 16 or len(samples) != 16:
        raise RuntimeError(
            f"watcher protocol must contain exactly 16 samples, got "
            f"ids={len(sample_ids)}, rows={len(samples)}"
        )
    if sample_ids != [str(row.get("id")) for row in samples]:
        raise RuntimeError(f"watcher protocol sample_ids and samples disagree: {path}")
    counts = Counter(str(row.get("garment_type")) for row in samples)
    if any(counts.get(kind, 0) != 4 for kind in GARMENT_TYPES):
        raise RuntimeError(
            f"watcher protocol is not 4x4 garment-stratified: {dict(counts)}"
        )
    minimum = float(payload.get("min_hair_fraction", 0.01))
    below = [
        str(row.get("id"))
        for row in samples
        if float(row.get("hair_fraction", 0.0)) < minimum
    ]
    if below:
        raise RuntimeError(f"watcher protocol contains no-hair samples: {below}")
    payload["path"] = str(path)
    return payload


def watcher_eval_set_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    path = cfg.get("eval", {}).get("watcher_eval_set")
    if not path:
        raise RuntimeError(
            "eval.watcher_eval_set is required; dynamic watcher sampling is forbidden"
        )
    return load_watcher_eval_set(path)

