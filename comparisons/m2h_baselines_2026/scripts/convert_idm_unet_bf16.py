#!/usr/bin/env python3
"""Convert IDM-VTON's monolithic FP32 UNet checkpoint to sharded BF16 safetensors."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-unet", type=Path, required=True)
    parser.add_argument("--output-unet", type=Path, required=True)
    parser.add_argument("--max-shard-size-gb", type=float, default=3.0)
    return parser.parse_args()


def target_nbytes(tensor: torch.Tensor) -> int:
    element_size = 2 if tensor.is_floating_point() else tensor.element_size()
    return tensor.numel() * element_size


def group_keys(
    state_dict: dict[str, torch.Tensor], max_shard_bytes: int
) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for key, tensor in state_dict.items():
        tensor_bytes = target_nbytes(tensor)
        if current and current_bytes + tensor_bytes > max_shard_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(key)
        current_bytes += tensor_bytes
    if current:
        groups.append(current)
    return groups


def main() -> None:
    args = parse_args()
    source_unet = args.source_unet.resolve()
    output_unet = args.output_unet.resolve()
    source_weights = source_unet / "diffusion_pytorch_model.bin"
    source_config = source_unet / "config.json"
    single_marker = output_unet / "diffusion_pytorch_model.safetensors"
    index_marker = output_unet / "diffusion_pytorch_model.safetensors.index.json"

    if (
        (single_marker.is_file() or index_marker.is_file())
        and (output_unet / "config.json").is_file()
    ):
        print(f"BF16 UNet already ready: {output_unet}")
        return
    if output_unet.exists():
        raise FileExistsError(
            f"Refusing to replace incomplete output directory: {output_unet}"
        )
    if not source_weights.is_file() or not source_config.is_file():
        raise FileNotFoundError(f"Incomplete source UNet: {source_unet}")
    if args.max_shard_size_gb <= 0:
        raise ValueError("--max-shard-size-gb must be positive")

    print(f"Memory-mapping source checkpoint: {source_weights}", flush=True)
    loaded = torch.load(
        str(source_weights),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        loaded = loaded["state_dict"]
    state_dict = {key: value for key, value in loaded.items() if torch.is_tensor(value)}
    if not state_dict or len(state_dict) != len(loaded):
        raise TypeError("Expected a tensor-only PyTorch state dict")

    max_shard_bytes = int(args.max_shard_size_gb * 1024**3)
    groups = group_keys(state_dict, max_shard_bytes)
    total_size = sum(target_nbytes(tensor) for tensor in state_dict.values())
    output_unet.parent.mkdir(parents=True, exist_ok=True)
    temp_unet = Path(
        tempfile.mkdtemp(prefix=".idm-unet-bf16-", dir=output_unet.parent)
    )
    weight_map: dict[str, str] = {}

    for index, keys in enumerate(groups, start=1):
        if len(groups) == 1:
            filename = "diffusion_pytorch_model.safetensors"
        else:
            filename = (
                f"diffusion_pytorch_model-{index:05d}-of-{len(groups):05d}.safetensors"
            )
        shard: dict[str, torch.Tensor] = {}
        for key in keys:
            tensor = state_dict[key]
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=torch.bfloat16)
            shard[key] = tensor.contiguous()
            weight_map[key] = filename
        shard_bytes = sum(tensor.numel() * tensor.element_size() for tensor in shard.values())
        print(
            f"Writing shard {index}/{len(groups)} ({shard_bytes / 1024**3:.2f} GiB)",
            flush=True,
        )
        save_file(shard, str(temp_unet / filename), metadata={"format": "pt"})
        del shard
        gc.collect()

    shutil.copy2(source_config, temp_unet / "config.json")
    if len(groups) > 1:
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with (temp_unet / index_marker.name).open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
            handle.write("\n")
    os.replace(temp_unet, output_unet)
    print(
        f"Created {len(groups)} BF16 shards ({total_size / 1024**3:.2f} GiB): {output_unet}",
        flush=True,
    )


if __name__ == "__main__":
    main()
