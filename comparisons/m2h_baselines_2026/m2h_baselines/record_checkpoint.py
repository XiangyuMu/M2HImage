from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CONTRACTS = {
    "refton": {
        "source": "mannequin(mid)",
        "identity": "face(jid)+identity_card(jid)",
        "target": "human(mid)",
    },
    "ita_mdt": {
        "source": "agnostic(mid)+pose_dense(mannequin(mid))",
        "identity": "identity_person(jid)+face(jid)",
        "target": "human(mid)",
    },
}


def _step_from_name(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.pt$)", path.name)
    if match is None:
        raise ValueError(f"cannot parse checkpoint step from {path}")
    return int(match.group(1))


def record_checkpoint(
    *,
    method: str,
    run_dir: Path,
    protocol_sha256: str,
    base_model: str,
    data_dir: str,
) -> dict[str, object]:
    run_dir = run_dir.expanduser().resolve()
    if method == "refton":
        weight = run_dir / "pytorch_lora_weights.safetensors"
        losses = run_dir / "all_losses.json"
        if not weight.is_file() or weight.stat().st_size == 0:
            raise FileNotFoundError(weight)
        values = json.loads(losses.read_text(encoding="utf-8"))
        if not isinstance(values, list) or not values:
            raise ValueError(f"invalid RefTon loss history: {losses}")
        global_step = len(values)
        artifact_key = "adapter_files"
        artifacts = [weight.name]
    elif method == "ita_mdt":
        ema_files = sorted(run_dir.glob("ema_0.9999_*.pt"), key=_step_from_name)
        if not ema_files:
            raise FileNotFoundError(f"no ITA-MDT EMA checkpoint under {run_dir}")
        ema = ema_files[-1]
        global_step = _step_from_name(ema)
        model = run_dir / f"model{global_step:06d}.pt"
        if not model.is_file() or model.stat().st_size == 0:
            raise FileNotFoundError(model)
        artifact_key = "checkpoint_files"
        artifacts = [model.name, ema.name]
    else:
        raise ValueError(f"unsupported method: {method}")

    payload: dict[str, object] = {
        "format_version": 1,
        "method": method,
        "task": "mannequin_to_human",
        "dataset_type": "m2h",
        "global_step": global_step,
        "protocol_sha256": protocol_sha256,
        "base_model": str(Path(base_model).expanduser().resolve()),
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "resolution": [512, 512],
        "seed": 42,
        "conditioning_contract": CONTRACTS[method],
        artifact_key: artifacts,
    }
    destination = run_dir / "m2h_checkpoint.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Record RefTon/ITA-MDT M2H checkpoint provenance.")
    parser.add_argument("--method", choices=sorted(CONTRACTS), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    payload = record_checkpoint(
        method=args.method,
        run_dir=args.run_dir,
        protocol_sha256=args.protocol_sha256,
        base_model=args.base_model,
        data_dir=args.data_dir,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
