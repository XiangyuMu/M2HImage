#!/usr/bin/env python3
"""Verify safetensors files tensor-by-tensor and emit a reproducibility report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--require-same-keys", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> tuple[dict[str, object], tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    empty_tensors: list[str] = []
    nonfinite_tensors: list[str] = []
    dtypes: dict[str, int] = {}
    elements = 0
    size_bytes = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = tuple(handle.keys())
        if not keys:
            raise RuntimeError(f"No tensors in {path}")
        for key in keys:
            tensor = handle.get_tensor(key)
            if tensor.numel() == 0:
                empty_tensors.append(key)
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
                torch.isfinite(tensor).all().item()
            ):
                nonfinite_tensors.append(key)
            dtype = str(tensor.dtype)
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
            elements += tensor.numel()
            size_bytes += tensor.numel() * tensor.element_size()

    if empty_tensors:
        raise RuntimeError(
            f"{path} contains {len(empty_tensors)} empty tensors; first={empty_tensors[0]}"
        )
    if nonfinite_tensors:
        raise RuntimeError(
            f"{path} contains {len(nonfinite_tensors)} non-finite tensors; "
            f"first={nonfinite_tensors[0]}"
        )

    return (
        {
            "bytes": size_bytes,
            "dtypes": dict(sorted(dtypes.items())),
            "elements": elements,
            "empty_tensors": 0,
            "file_bytes": path.stat().st_size,
            "nonfinite_tensors": 0,
            "path": str(path),
            "sha256": sha256_file(path),
            "tensors": len(keys),
        },
        keys,
    )


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    args = parse_args()
    resolved = [path.resolve() for path in args.paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate input paths")

    files: dict[str, object] = {}
    key_sets: list[tuple[str, ...]] = []
    for path in resolved:
        stats, keys = verify(path)
        files[str(path)] = stats
        key_sets.append(keys)
        print(
            f"Verified {path.name}: {stats['tensors']} tensors, "
            f"sha256={stats['sha256']}",
            flush=True,
        )

    if args.require_same_keys and any(keys != key_sets[0] for keys in key_sets[1:]):
        raise RuntimeError("Safetensors files do not contain identical key sets")

    report: dict[str, object] = {
        "files": files,
        "require_same_keys": args.require_same_keys,
        "status": "pass",
    }
    if args.output is not None:
        write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
