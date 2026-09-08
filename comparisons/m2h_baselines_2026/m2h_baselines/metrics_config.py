from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .collect import protocol_from_records
from .validate import load_manifest


def write_metrics_config(
    *,
    base_config: Path,
    manifest: Path,
    gen_dir: Path,
    data_root: Path,
    output_config: Path,
    output_root: Path,
    run_name: str,
    label: str,
) -> tuple[Path, Path]:
    if not base_config.is_file():
        raise FileNotFoundError(base_config)
    records = load_manifest(manifest)
    if not records:
        raise ValueError(f"empty manifest: {manifest}")
    resolution = str(records[0]["resolution"])
    sizes = {"low": (512, 512), "native": (768, 1024)}
    width, height = sizes[resolution]
    protocol_path = output_config.with_suffix(".protocol.json")
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(
        json.dumps(protocol_from_records(records), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    payload = {
        "extends": str(base_config.resolve()),
        "data": {
            "root": str(data_root.resolve()),
            "cf_subset": str(protocol_path.resolve()),
            "resolution": {"width": width, "height": height},
        },
        "metrics_v2": {
            "output_root": str(output_root.resolve()),
            "default_runs": [run_name],
            "runs": {
                run_name: {
                    "label": label,
                    "gen_dir": str(gen_dir.resolve()),
                    "legacy_metrics_dir": str((output_root / "legacy_unavailable").resolve()),
                }
            },
        },
    }
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return output_config, protocol_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Metrics-v2 config for one baseline run.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gen-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    config, protocol = write_metrics_config(
        base_config=Path(args.base_config),
        manifest=Path(args.manifest),
        gen_dir=Path(args.gen_dir),
        data_root=Path(args.data_root),
        output_config=Path(args.output_config),
        output_root=Path(args.output_root),
        run_name=args.run_name,
        label=args.label,
    )
    print(json.dumps({"config": str(config), "protocol": str(protocol)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
