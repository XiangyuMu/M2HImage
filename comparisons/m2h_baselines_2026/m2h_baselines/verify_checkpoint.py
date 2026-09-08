from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = {"refton", "ita_mdt", "idm_vton", "mcld", "ominicontrol"}


def _safe_files(checkpoint: Path, names: object, label: str) -> list[str]:
    if not isinstance(names, list) or not names:
        raise ValueError(f"checkpoint metadata has no {label}")
    verified: list[str] = []
    for value in names:
        name = str(value)
        if Path(name).name != name:
            raise ValueError(f"unsafe checkpoint filename in {label}: {name}")
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty checkpoint file: {path}")
        verified.append(name)
    return verified


def verify_checkpoint(
    checkpoint: Path,
    *,
    method: str,
    protocol_sha256: str,
) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"unsupported checkpoint method: {method}")
    checkpoint_root = checkpoint if checkpoint.is_dir() else checkpoint.parent
    metadata_path = checkpoint_root / "m2h_checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("task") != "mannequin_to_human":
        raise ValueError("checkpoint was not recorded as mannequin_to_human")
    if metadata.get("protocol_sha256") != protocol_sha256:
        raise ValueError("checkpoint protocol SHA256 does not match the frozen study")
    if int(metadata.get("global_step", 0)) < 1:
        raise ValueError("checkpoint has no completed optimizer step")
    contract = metadata.get("conditioning_contract")
    if not isinstance(contract, dict) or not contract:
        raise ValueError("checkpoint has no conditioning contract")

    if method == "refton":
        files = _safe_files(checkpoint_root, metadata.get("adapter_files"), "adapter_files")
        if files != ["pytorch_lora_weights.safetensors"]:
            raise ValueError(f"unexpected RefTon adapters: {files}")
    elif method == "ita_mdt":
        files = _safe_files(checkpoint_root, metadata.get("checkpoint_files"), "checkpoint_files")
        if checkpoint.name not in files or not checkpoint.name.startswith("ema_0.9999_"):
            raise ValueError(f"ITA checkpoint is not the recorded EMA artifact: {checkpoint}")
    elif method == "idm_vton":
        model = checkpoint_root / "unet" / "diffusion_pytorch_model.safetensors"
        if not model.is_file() or model.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty IDM UNet: {model}")
        files = [str(model.relative_to(checkpoint_root))]
    elif method == "mcld":
        files = _safe_files(checkpoint_root, metadata.get("component_files"), "component_files")
        required = {"reference_unet", "denoising_unet", "pose_guider", "image_proj_model"}
        prefixes = {name.rsplit("-", 1)[0] for name in files}
        if prefixes != required:
            raise ValueError(f"unexpected MCLD components: {sorted(prefixes)}")
    else:
        files = _safe_files(checkpoint_root, metadata.get("adapter_files"), "adapter_files")
        if set(files) != {
            "mannequin.safetensors",
            "identity.safetensors",
            "pose.safetensors",
        }:
            raise ValueError(f"unexpected OminiControl adapters: {sorted(files)}")

    return {
        "status": "pass",
        "method": method,
        "checkpoint": str(checkpoint.resolve()),
        "global_step": int(metadata["global_step"]),
        "protocol_sha256": protocol_sha256,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a task-adapted M2H checkpoint.")
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    report = verify_checkpoint(
        args.checkpoint,
        method=args.method,
        protocol_sha256=args.protocol_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
