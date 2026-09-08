#!/usr/bin/env python3
"""Extract and verify MCLD inference components from a full FSDP model checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch


COMPONENTS = (
    "reference_unet",
    "denoising_unet",
    "pose_guider",
    "image_proj_model",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--expected-ip-adapter-tensors", type=int, default=32)
    return parser.parse_args()


def load_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    loaded = torch.load(
        str(path),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if "state_dict" in loaded and isinstance(loaded["state_dict"], Mapping):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, Mapping) or not loaded:
        raise TypeError(f"Expected a non-empty state dict: {path}")
    if any(not isinstance(key, str) or not torch.is_tensor(value) for key, value in loaded.items()):
        raise TypeError(f"Expected a tensor-only state dict: {path}")
    return loaded


def validate_component(
    component: str,
    state_dict: Mapping[str, torch.Tensor],
    expected_ip_adapter_tensors: int,
) -> dict[str, object]:
    if not state_dict:
        raise RuntimeError(f"Empty MCLD component: {component}")

    empty_tensors: list[str] = []
    nonfinite_tensors: list[str] = []
    dtypes: dict[str, int] = {}
    elements = 0
    size_bytes = 0
    for key, tensor in state_dict.items():
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
            f"{component} contains {len(empty_tensors)} empty tensors; first={empty_tensors[0]}"
        )
    if nonfinite_tensors:
        raise RuntimeError(
            f"{component} contains {len(nonfinite_tensors)} non-finite tensors; "
            f"first={nonfinite_tensors[0]}"
        )

    ip_adapter_tensors = sum(
        ".processor.to_k_ip." in key or ".processor.to_v_ip." in key
        for key in state_dict
    )
    if component == "denoising_unet" and ip_adapter_tensors != expected_ip_adapter_tensors:
        raise RuntimeError(
            "MCLD denoising UNet IP-Adapter tensor count mismatch: "
            f"{ip_adapter_tensors} != {expected_ip_adapter_tensors}"
        )

    return {
        "bytes": size_bytes,
        "dtypes": dict(sorted(dtypes.items())),
        "elements": elements,
        "empty_tensors": 0,
        "ip_adapter_tensors": ip_adapter_tensors,
        "nonfinite_tensors": 0,
        "tensors": len(state_dict),
    }


def main() -> None:
    args = parse_args()
    if args.step < 1:
        raise ValueError("--step must be positive")
    if args.expected_ip_adapter_tensors < 1:
        raise ValueError("--expected-ip-adapter-tensors must be positive")

    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    destinations = {
        component: output_dir / f"{component}-{args.step}.pth"
        for component in COMPONENTS
    }
    report_path = output_dir / f"mcld-components-{args.step}.validation.json"
    existing = [path for path in (*destinations.values(), report_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing output: {existing[0]}")

    print(f"Memory-mapping full MCLD checkpoint: {source}", flush=True)
    full_state_dict = load_state_dict(source)
    prefixes = {key.partition(".")[0] for key in full_state_dict}
    if prefixes != set(COMPONENTS):
        raise RuntimeError(
            f"Unexpected MCLD component prefixes: {sorted(prefixes)}"
        )

    temp_dir = Path(tempfile.mkdtemp(prefix=".mcld-components-", dir=output_dir))
    report: dict[str, object] = {
        "components": {},
        "expected_ip_adapter_tensors": args.expected_ip_adapter_tensors,
        "source": str(source),
        "status": "pass",
        "step": args.step,
    }
    try:
        for component in COMPONENTS:
            prefix = f"{component}."
            component_state = {
                key[len(prefix) :]: value
                for key, value in full_state_dict.items()
                if key.startswith(prefix)
            }
            temporary_path = temp_dir / destinations[component].name
            print(
                f"Writing {component} ({len(component_state)} tensors)",
                flush=True,
            )
            torch.save(component_state, temporary_path)
            del component_state

            reloaded = load_state_dict(temporary_path)
            stats = validate_component(
                component,
                reloaded,
                args.expected_ip_adapter_tensors,
            )
            stats["path"] = str(destinations[component])
            report["components"][component] = stats
            print(
                f"Verified {component}: {stats['tensors']} tensors, "
                f"{stats['bytes'] / 1024**3:.2f} GiB",
                flush=True,
            )
            del reloaded

        temporary_report = temp_dir / report_path.name
        with temporary_report.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")

        for component in COMPONENTS:
            os.replace(temp_dir / destinations[component].name, destinations[component])
        os.replace(temporary_report, report_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
