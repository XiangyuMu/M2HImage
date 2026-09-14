from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conditions import load_yaml
from watcher_protocol import build_watcher_eval_set, write_watcher_eval_set


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the immutable 4x4 visible-hair watcher evaluation set."
    )
    parser.add_argument(
        "--config",
        default="configs/spatial_warmup_resume_hair_incontext.yaml",
    )
    parser.add_argument("--output", default="configs/watcher_eval_set.json")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    payload = build_watcher_eval_set(
        cfg["data"]["root"],
        cfg["data"]["val_split"],
        per_type=4,
        min_hair_fraction=0.01,
    )
    output = write_watcher_eval_set(Path(args.output), payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "protocol_hash": payload["protocol_hash"],
                "sample_ids": payload["sample_ids"],
                "counts_by_garment_type": payload["counts_by_garment_type"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
