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
            proc = Path(f"/proc/{pid}")
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            status["pid"] = pid
            status["process_command"] = cmdline
            status["process_state"] = stat[0]
            status["start_ticks"] = stat[19]
            status["pid_alive"] = stat[0] != "Z" and "train_paired.py" in cmdline and root.name in cmdline
            status["cpu_seconds"] = (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK")
            status["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
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
    latest = status.get("latest_log")
    if latest:
        log = Path(str(latest))
        status["log_age_seconds"] = round(time.time() - log.stat().st_mtime, 1)
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 8192))
            status["log_tail"] = handle.read().decode(errors="replace").splitlines()[-3:]
    status["health"] = "running" if status["pid_alive"] else "not_running"
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
