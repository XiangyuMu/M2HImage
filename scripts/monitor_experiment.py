#!/usr/bin/env python3
"""Emit a compact, append-only health snapshot for a training experiment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _latest(path: Path, pattern: str) -> str | None:
    items = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime)
    return str(items[-1]) if items else None


def snapshot(root: Path) -> dict[str, object]:
    logs = root / "logs"
    checkpoints = root / "checkpoints"
    status: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid_alive": False,
        "latest_checkpoint": _latest(checkpoints, "step-*"),
        "latest_log": _latest(logs, "*.jsonl"),
    }
    pid_file = root / "train.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            status["pid"] = pid
            status["pid_alive"] = True
        except (OSError, ValueError):
            status["pid"] = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        status["gpu"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        status["gpu"] = []
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--interval-seconds", type=int, default=7200)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = Path(args.experiment_root)
    root.mkdir(parents=True, exist_ok=True)
    output = root / "monitoring" / "health.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        row = snapshot(root)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if args.once:
            return
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    main()
