from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import configured_path, load_config
from .materialize import MATERIALIZERS, materialize
from .prepare import prepare_split
from .protocol import prepare_counterfactual
from .validate import validate_manifest


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "study.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="m2h-baselines")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="build shared M2H conditions")
    prepare.add_argument("--split", choices=("train", "val", "test"), required=True)
    prepare.add_argument("--resolution", choices=("low", "native"), default="low")
    prepare.add_argument("--data-root")
    prepare.add_argument("--output-root")
    prepare.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    prepare.add_argument("--limit", type=int)
    prepare_cf = sub.add_parser("prepare-cf", help="build the frozen mid/jid counterfactual manifest")
    prepare_cf.add_argument("--resolution", choices=("low", "native"), default="low")
    prepare_cf.add_argument("--data-root")
    prepare_cf.add_argument("--output-root")
    prepare_cf.add_argument("--protocol")
    prepare_cf.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    prepare_cf.add_argument("--limit-pairs", type=int)


    validate = sub.add_parser("validate", help="validate a prepared JSONL manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--prepared-root", required=True)

    layout = sub.add_parser("materialize", help="build method-specific symlink layouts")
    layout.add_argument("--method", required=True, choices=sorted(MATERIALIZERS))
    layout.add_argument("--manifest", required=True)
    layout.add_argument("--prepared-root", required=True)
    layout.add_argument("--output-root", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        data_root = configured_path(config, "data_root", args.data_root)
        project_root = configured_path(config, "project_root")
        output_root = Path(args.output_root).expanduser().resolve() if args.output_root else project_root / "prepared"
        excluded = config["study"].get("excluded_train_ids", []) if args.split == "train" else []
        manifest = prepare_split(
            data_root=data_root,
            output_root=output_root,
            split=args.split,
            resolution=args.resolution,
            prep_config=config["preprocess"],
            workers=args.workers,
            limit=args.limit,
            excluded_ids=excluded,
        )
        print(json.dumps({"status": "ok", "manifest": str(manifest)}, ensure_ascii=False))
    elif args.command == "prepare-cf":
        data_root = configured_path(config, "data_root", args.data_root)
        project_root = configured_path(config, "project_root")
        output_root = Path(args.output_root).expanduser().resolve() if args.output_root else project_root / "prepared"
        protocol = (
            Path(args.protocol).expanduser().resolve()
            if args.protocol
            else data_root / "eval" / "cf_subset.json"
        )
        manifest = prepare_counterfactual(
            data_root=data_root,
            output_root=output_root,
            protocol_path=protocol,
            resolution=args.resolution,
            prep_config=config["preprocess"],
            workers=args.workers,
            limit_pairs=args.limit_pairs,
        )
        print(json.dumps({"status": "ok", "manifest": str(manifest)}, ensure_ascii=False))
    elif args.command == "validate":
        report = validate_manifest(Path(args.manifest), Path(args.prepared_root))
        print(json.dumps(report, indent=2, ensure_ascii=False))

    elif args.command == "materialize":
        root = materialize(
            args.method,
            Path(args.manifest).expanduser().resolve(),
            Path(args.prepared_root).expanduser().resolve(),
            Path(args.output_root).expanduser().resolve(),
            config.get("preprocess", {}),
        )
        print(json.dumps({"status": "ok", "layout_root": str(root)}, ensure_ascii=False))

if __name__ == "__main__":
    main()

