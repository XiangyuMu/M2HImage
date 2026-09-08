from __future__ import annotations

"""Build a durable, machine-readable summary for the five full M2H baselines.

The full experiment queue is deliberately tolerant of an OOM in one method.
Consequently, the final report cannot assume that every method produced images
or metrics.  This module inventories phase status, checkpoints, canonical
outputs, and Metrics-v2 artifacts without mutating any experiment output.
"""

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


METHODS = ("refton", "ita_mdt", "idm_vton", "mcld", "ominicontrol")
PHASES = ("train", "inference", "nativeize", "verify", "metrics")
TERMINAL_STATES = {"completed", "oom", "paused", "failed", "failed_validation"}
_OOM_RE = re.compile(
    r"out[\s_-]*of[\s_-]*memory|cublas_status_alloc_failed|"
    r"cuda error.*memory|memory allocation failed|cannot allocate memory",
    re.IGNORECASE,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(payload: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _step_from_path(path: Path) -> int:
    matches = re.findall(r"(?:checkpoint[-_]?|ema_0\.9999_)(\d+)", str(path))
    return max((int(value) for value in matches), default=-1)


def _status_rows(status_tsv: Path) -> list[dict[str, str]]:
    if not status_tsv.is_file():
        return []
    with status_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"method", "phase", "status", "rc", "log"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            return []
        return [dict(row) for row in reader]


def _checkpoint_inventory(
    method: str, method_run: Path, canonical_summary: dict[str, Any] | None
) -> dict[str, Any]:
    canonical_checkpoint = (
        str(canonical_summary.get("checkpoint"))
        if canonical_summary and canonical_summary.get("checkpoint")
        else None
    )
    candidates: list[tuple[int, float, Path, str, Path | None]] = []

    for metadata in method_run.rglob("m2h_checkpoint.json") if method_run.is_dir() else ():
        payload = _read_json(metadata) or {}
        try:
            step = int(payload.get("global_step", _step_from_path(metadata.parent)))
        except (TypeError, ValueError):
            step = _step_from_path(metadata.parent)
        candidates.append((step, metadata.stat().st_mtime, metadata.parent, "metadata", metadata))

    for directory in method_run.rglob("checkpoint-*") if method_run.is_dir() else ():
        if directory.is_dir():
            candidates.append(
                (_step_from_path(directory), directory.stat().st_mtime, directory, "checkpoint_dir", None)
            )

    if method == "ita_mdt" and method_run.is_dir():
        for checkpoint in method_run.rglob("ema_0.9999_*.pt"):
            candidates.append(
                (_step_from_path(checkpoint), checkpoint.stat().st_mtime, checkpoint, "ema", None)
            )

    if canonical_checkpoint:
        checkpoint_path = Path(canonical_checkpoint)
        metadata_path = (
            checkpoint_path / "m2h_checkpoint.json"
            if checkpoint_path.is_dir()
            else checkpoint_path.parent / "m2h_checkpoint.json"
        )
        metadata = _read_json(metadata_path)
        step_value = _nested(metadata, "global_step")
        try:
            global_step = int(step_value) if step_value is not None else _step_from_path(checkpoint_path)
        except (TypeError, ValueError):
            global_step = _step_from_path(checkpoint_path)
        return {
            "path": canonical_checkpoint,
            "exists": checkpoint_path.exists(),
            "source": "canonical_summary",
            "global_step": None if global_step < 0 else global_step,
            "metadata": str(metadata_path) if metadata_path.is_file() else None,
        }

    if not candidates:
        # RefTon writes the final LoRA in the run root rather than a nested
        # checkpoint directory.  It is useful to expose even after an unusual
        # exit that happened after the final save but before provenance write.
        refton_weight = method_run / "pytorch_lora_weights.safetensors"
        if refton_weight.is_file():
            return {
                "path": str(method_run),
                "exists": True,
                "source": "refton_weight",
                "global_step": None,
                "metadata": None,
            }
        return {
            "path": None,
            "exists": False,
            "source": None,
            "global_step": None,
            "metadata": None,
        }

    step, _, path, source, metadata = max(candidates, key=lambda item: (item[0], item[1]))
    return {
        "path": str(path),
        "exists": path.exists(),
        "source": source,
        "global_step": None if step < 0 else step,
        "metadata": str(metadata) if metadata else None,
    }


def _artifact_inventory(
    *,
    method: str,
    output_root: Path,
    report_root: Path,
    expected_records: int,
    native_manifest_sha256: str | None,
) -> tuple[dict[str, Any], list[str]]:
    canonical_dir = output_root / method / "full" / "canonical"
    canonical_summary_path = canonical_dir / "summary.json"
    canonical_summary = _read_json(canonical_summary_path)
    png_count = sum(1 for path in canonical_dir.glob("*.png") if path.is_file())
    inference_manifest = canonical_dir / "inference_manifest.jsonl"

    run_dir = report_root / "generated" / method / "full" / f"{method}_full"
    verification_path = run_dir / "metrics_verification.json"
    verification = _read_json(verification_path)
    garment = _read_json(run_dir / "garment_summary.json")
    hair = _read_json(run_dir / "hair_summary.json")
    pose = _read_json(run_dir / "pose_summary.json")
    identity = _read_json(run_dir / "identity_summary.json")
    distribution = _read_json(run_dir / "distribution_summary.json")
    report_path = run_dir / "report.md"

    errors: list[str] = []
    if canonical_summary is not None:
        if canonical_summary.get("status") != "pass":
            errors.append("canonical summary status is not pass")
        if int(canonical_summary.get("records", -1)) != expected_records:
            errors.append("canonical summary record count mismatch")
        if canonical_summary.get("resolution") != "native":
            errors.append("canonical summary resolution is not native")
        if list(canonical_summary.get("size", [])) != [768, 1024]:
            errors.append("canonical summary size is not 768x1024")
        if native_manifest_sha256 and canonical_summary.get("manifest_sha256") != native_manifest_sha256:
            errors.append("canonical manifest hash mismatch")
        if png_count != expected_records:
            errors.append(f"canonical PNG count is {png_count}, expected {expected_records}")
        if not inference_manifest.is_file():
            errors.append("canonical inference manifest is missing")

    if verification is not None:
        if verification.get("status") != "ok":
            errors.append("Metrics-v2 verification status is not ok")
        if int(verification.get("expected_count", -1)) != expected_records:
            errors.append("Metrics-v2 verified count mismatch")
        if distribution is None or distribution.get("status") != "ok":
            errors.append("FID/KID distribution summary is missing or not ok")
        if not report_path.is_file() or report_path.stat().st_size == 0:
            errors.append("Metrics-v2 report is missing or empty")

    artifacts = {
        "canonical_dir": str(canonical_dir),
        "canonical_summary": str(canonical_summary_path) if canonical_summary is not None else None,
        "canonical_status": canonical_summary.get("status") if canonical_summary else None,
        "canonical_records": canonical_summary.get("records") if canonical_summary else None,
        "png_count": png_count,
        "checkpoint": canonical_summary.get("checkpoint") if canonical_summary else None,
        "metrics_dir": str(run_dir),
        "metrics_verification": str(verification_path) if verification is not None else None,
        "metrics_status": verification.get("status") if verification else None,
        "quality_status": verification.get("quality_status") if verification else None,
        "report": str(report_path) if report_path.is_file() else None,
        "metrics": {
            "garment_dino_mean": _number(_nested(garment, "garment_dino", "mean")),
            "garment_lpips_mean": _number(_nested(garment, "garment_lpips", "mean")),
            "garment_hf_lpips_mean": _number(_nested(garment, "garment_hf_lpips", "mean")),
            "hair_dino_mean": _number(_nested(hair, "hair_dino", "mean")),
            "hair_lab_distance_mean": _number(_nested(hair, "hair_lab_distance", "mean")),
            "body_distance_mean": _number(_nested(pose, "body_distance", "mean")),
            "head5_distance_mean": _number(_nested(pose, "head5_distance", "mean")),
            "identity_delta_mean": _number(_nested(identity, "mean")),
            "identity_target_mean": _number(_nested(identity, "sim_target_mean")),
            "face_detection_rate": _number(_nested(identity, "face_detection_rate")),
            "fid": _number(_nested(distribution, "fid")),
            "kid_mean": _number(_nested(distribution, "kid_mean")),
            "kid_std": _number(_nested(distribution, "kid_std")),
        },
    }
    return artifacts, errors


def _latest_phase_logs(orchestrator_logs: Path, method: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for phase in PHASES:
        matches = sorted(orchestrator_logs.glob(f"{method}_{phase}_*.log"))
        if matches:
            result[phase] = str(matches[-1])
    return result


def _log_has_oom(path: Path) -> bool:
    """Check a bounded phase-log tail for an OOM signature."""

    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 4 * 1024 * 1024))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return _OOM_RE.search(text) is not None


def _method_state(
    phase_status: dict[str, str],
    phase_logs: dict[str, str],
    artifact_errors: list[str],
    *,
    queue_finished: bool = False,
) -> str:
    statuses = set(phase_status.values())
    if "paused" in statuses:
        return "paused"
    if "oom" in statuses:
        return "oom"
    if "failed" in statuses:
        return "failed"
    if all(phase_status.get(phase) == "ok" for phase in PHASES):
        return "failed_validation" if artifact_errors else "completed"
    if queue_finished:
        # A system-level OOM or abrupt worker kill can terminate the queue
        # before run_phase gets a chance to append its status row. Once the
        # queue session is known to be gone, unresolved methods must not remain
        # misleadingly marked as running/pending in the strict final report.
        if any(_log_has_oom(Path(path)) for path in phase_logs.values()):
            return "oom"
        return "failed"
    if phase_status:
        return "in_progress"
    if phase_logs:
        return "running"
    return "pending"


def _fmt(value: Any, digits: int = 4) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Five-method M2H full comparison",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        f"Overall status: **{payload['status']}**. Expected canonical outputs per successful method: "
        f"**{payload['expected_records']}** at **768×1024**.",
        "",
        "| Method | State | Outputs | Metrics | Garment DINO ↑ | Garment HF-LPIPS ↓ | Hair DINO ↑ | Head-5 dist. ↓ | DeltaID ↑ | FID ↓ | KID ↓ |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = payload["methods"][method]
        artifacts = item["artifacts"]
        metrics = artifacts["metrics"]
        lines.append(
            "| {method} | {state} | {outputs}/{expected} | {metrics_status} | {gd} | {ghf} | "
            "{hair} | {head} | {identity} | {fid} | {kid} |".format(
                method=method,
                state=item["state"],
                outputs=artifacts["png_count"],
                expected=payload["expected_records"],
                metrics_status=artifacts["metrics_status"] or "—",
                gd=_fmt(metrics["garment_dino_mean"]),
                ghf=_fmt(metrics["garment_hf_lpips_mean"]),
                hair=_fmt(metrics["hair_dino_mean"]),
                head=_fmt(metrics["head5_distance_mean"]),
                identity=_fmt(metrics["identity_delta_mean"]),
                fid=_fmt(metrics["fid"]),
                kid=_fmt(metrics["kid_mean"], 6),
            )
        )

    lines.extend(["", "## Artifact and failure inventory", ""])
    for method in METHODS:
        item = payload["methods"][method]
        checkpoint = item["checkpoint"]
        phases = ", ".join(
            f"{phase}={item['phase_status'].get(phase, 'not_recorded')}" for phase in PHASES
        )
        lines.extend(
            [
                f"### {method}",
                "",
                f"- State: `{item['state']}`; phases: {phases}",
                f"- Checkpoint: `{checkpoint['path'] or 'none'}`",
                f"- Canonical outputs: `{item['artifacts']['canonical_dir']}`",
                f"- Metrics report: `{item['artifacts']['report'] or 'none'}`",
            ]
        )
        if item["artifact_errors"]:
            lines.append("- Artifact errors: " + "; ".join(item["artifact_errors"]))
        if item["terminal_log"]:
            lines.append(f"- Terminal/current log: `{item['terminal_log']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_tsv(path: Path, payload: dict[str, Any]) -> None:
    fields = [
        "method",
        "state",
        "checkpoint",
        "global_step",
        "png_count",
        "metrics_status",
        "quality_status",
        "garment_dino_mean",
        "garment_hf_lpips_mean",
        "hair_dino_mean",
        "head5_distance_mean",
        "identity_delta_mean",
        "fid",
        "kid_mean",
        "terminal_log",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for method in METHODS:
            item = payload["methods"][method]
            artifacts = item["artifacts"]
            metrics = artifacts["metrics"]
            writer.writerow(
                {
                    "method": method,
                    "state": item["state"],
                    "checkpoint": item["checkpoint"]["path"],
                    "global_step": item["checkpoint"]["global_step"],
                    "png_count": artifacts["png_count"],
                    "metrics_status": artifacts["metrics_status"],
                    "quality_status": artifacts["quality_status"],
                    "garment_dino_mean": metrics["garment_dino_mean"],
                    "garment_hf_lpips_mean": metrics["garment_hf_lpips_mean"],
                    "hair_dino_mean": metrics["hair_dino_mean"],
                    "head5_distance_mean": metrics["head5_distance_mean"],
                    "identity_delta_mean": metrics["identity_delta_mean"],
                    "fid": metrics["fid"],
                    "kid_mean": metrics["kid_mean"],
                    "terminal_log": item["terminal_log"],
                }
            )
    temporary.replace(path)


def summarize_full_native(
    *,
    run_root: Path,
    output_root: Path,
    report_root: Path,
    status_tsv: Path,
    expected_records: int = 400,
    native_manifest: Path | None = None,
    protocol_sha256: str | None = None,
    destination: Path | None = None,
    queue_finished: bool = False,
) -> dict[str, Any]:
    if expected_records < 1:
        raise ValueError("expected_records must be positive")
    rows = _status_rows(status_tsv)
    manifest_sha256 = _sha256(native_manifest) if native_manifest and native_manifest.is_file() else None
    orchestrator_logs = status_tsv.parent / "logs"
    methods: dict[str, Any] = {}

    for method in METHODS:
        method_rows = [row for row in rows if row.get("method") == method]
        phase_status = {
            row["phase"]: row["status"] for row in method_rows if row.get("phase") in PHASES
        }
        phase_logs = _latest_phase_logs(orchestrator_logs, method)
        artifacts, artifact_errors = _artifact_inventory(
            method=method,
            output_root=output_root,
            report_root=report_root,
            expected_records=expected_records,
            native_manifest_sha256=manifest_sha256,
        )
        checkpoint = _checkpoint_inventory(
            method, run_root / method / "full", _read_json(Path(artifacts["canonical_summary"]))
            if artifacts["canonical_summary"]
            else None
        )
        state = _method_state(
            phase_status,
            phase_logs,
            artifact_errors,
            queue_finished=queue_finished,
        )
        if queue_finished and state in {"failed", "oom"}:
            explicit_terminal = bool(
                set(phase_status.values()) & {"oom", "failed"}
            )
            if not explicit_terminal:
                artifact_errors.append(
                    "queue ended before all phase status records were written"
                )
        terminal_log = None
        if method_rows:
            terminal_log = method_rows[-1].get("log") or None
        elif phase_logs:
            terminal_log = phase_logs.get("train") or list(phase_logs.values())[-1]
        methods[method] = {
            "state": state,
            "phase_status": phase_status,
            "phase_logs": phase_logs,
            "terminal_log": terminal_log,
            "checkpoint": checkpoint,
            "artifacts": artifacts,
            "artifact_errors": artifact_errors,
        }

    states = [methods[method]["state"] for method in METHODS]
    if all(state == "completed" for state in states):
        overall = "complete"
    elif all(state in TERMINAL_STATES for state in states):
        overall = (
            "finished_with_oom"
            if set(states).issubset({"completed", "oom"})
            else "finished_with_failures"
        )
    else:
        overall = "in_progress"

    destination = destination or report_root / "full_native_summary"
    payload: dict[str, Any] = {
        "status": overall,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "methods_order": list(METHODS),
        "expected_records": expected_records,
        "native_size": [768, 1024],
        "native_manifest": str(native_manifest) if native_manifest else None,
        "native_manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "status_tsv": str(status_tsv),
        "terminal_count": sum(state in TERMINAL_STATES for state in states),
        "completed_count": states.count("completed"),
        "oom_count": states.count("oom"),
        "paused_count": states.count("paused"),
        "methods": methods,
    }
    _write_json(destination / "summary.json", payload)
    (destination / "report.md").write_text(_markdown(payload), encoding="utf-8")
    _write_tsv(destination / "comparison.tsv", payload)
    return payload


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize the five-method full native M2H queue.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--status-tsv", required=True, type=Path)
    parser.add_argument("--expected-records", type=int, default=400)
    parser.add_argument("--native-manifest", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--require-terminal",
        action="store_true",
        help="Exit nonzero unless all five methods have a terminal result.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = summarize_full_native(
        run_root=args.run_root,
        output_root=args.output_root,
        report_root=args.report_root,
        status_tsv=args.status_tsv,
        expected_records=args.expected_records,
        native_manifest=args.native_manifest,
        protocol_sha256=args.protocol_sha256,
        destination=args.destination,
        queue_finished=args.require_terminal,
    )
    print(json.dumps(payload, ensure_ascii=False))
    if args.require_terminal and payload["terminal_count"] != len(METHODS):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
