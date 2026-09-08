from __future__ import annotations

import json
from pathlib import Path

from m2h_baselines.summarize_full_native import summarize_full_native


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_summary_combines_completed_oom_and_pending_methods(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    output_root = tmp_path / "outputs"
    report_root = tmp_path / "reports"
    status_tsv = run_root / "orchestrator" / "status.tsv"
    status_tsv.parent.mkdir(parents=True)
    status_tsv.write_text(
        "method\tphase\tstatus\trc\tlog\n"
        "refton\ttrain\tok\t0\t/train.log\n"
        "refton\tinference\tok\t0\t/infer.log\n"
        "refton\tnativeize\tok\t0\t/native.log\n"
        "refton\tverify\tok\t0\t/verify.log\n"
        "refton\tmetrics\tok\t0\t/metrics.log\n"
        "ita_mdt\ttrain\toom\t125\t/ita.log\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text("{}\n{}\n", encoding="utf-8")

    checkpoint = run_root / "refton" / "full"
    checkpoint.mkdir(parents=True)
    (checkpoint / "pytorch_lora_weights.safetensors").write_bytes(b"weights")
    _json(checkpoint / "m2h_checkpoint.json", {"global_step": 64})

    canonical = output_root / "refton" / "full" / "canonical"
    canonical.mkdir(parents=True)
    (canonical / "a.png").write_bytes(b"png")
    (canonical / "b.png").write_bytes(b"png")
    (canonical / "inference_manifest.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    import hashlib

    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    _json(
        canonical / "summary.json",
        {
            "status": "pass",
            "records": 2,
            "resolution": "native",
            "size": [768, 1024],
            "manifest_sha256": manifest_hash,
            "checkpoint": str(checkpoint),
        },
    )

    metrics = report_root / "generated" / "refton" / "full" / "refton_full"
    _json(metrics / "metrics_verification.json", {"status": "ok", "quality_status": "warning", "expected_count": 2})
    _json(metrics / "garment_summary.json", {"garment_dino": {"mean": 0.8}, "garment_hf_lpips": {"mean": 0.2}})
    _json(metrics / "hair_summary.json", {"hair_dino": {"mean": 0.7}})
    _json(metrics / "pose_summary.json", {"head5_distance": {"mean": 0.03}})
    _json(metrics / "identity_summary.json", {"mean": 0.4, "face_detection_rate": 1.0})
    _json(metrics / "distribution_summary.json", {"status": "ok", "fid": 12.5, "kid_mean": 0.01})
    (metrics / "report.md").write_text("report\n", encoding="utf-8")

    result = summarize_full_native(
        run_root=run_root,
        output_root=output_root,
        report_root=report_root,
        status_tsv=status_tsv,
        expected_records=2,
        native_manifest=manifest,
        protocol_sha256="frozen",
    )

    assert result["status"] == "in_progress"
    assert result["methods"]["refton"]["state"] == "completed"
    assert result["methods"]["refton"]["checkpoint"]["global_step"] == 64
    assert result["methods"]["refton"]["artifacts"]["metrics"]["fid"] == 12.5
    assert result["methods"]["ita_mdt"]["state"] == "oom"
    assert result["methods"]["idm_vton"]["state"] == "pending"
    assert (report_root / "full_native_summary" / "summary.json").is_file()
    assert "refton | completed | 2/2" in (report_root / "full_native_summary" / "report.md").read_text(encoding="utf-8")
    assert (report_root / "full_native_summary" / "comparison.tsv").is_file()


def test_running_state_comes_from_unfinished_phase_log(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    output_root = tmp_path / "outputs"
    report_root = tmp_path / "reports"
    status_tsv = run_root / "orchestrator" / "status.tsv"
    logs = status_tsv.parent / "logs"
    logs.mkdir(parents=True)
    status_tsv.write_text("method\tphase\tstatus\trc\tlog\n", encoding="utf-8")
    (logs / "refton_train_20260824-110750.log").write_text("training\n", encoding="utf-8")

    result = summarize_full_native(
        run_root=run_root,
        output_root=output_root,
        report_root=report_root,
        status_tsv=status_tsv,
        expected_records=2,
    )

    assert result["methods"]["refton"]["state"] == "running"
    assert result["methods"]["ita_mdt"]["state"] == "pending"

def test_paused_method_is_a_terminal_user_decision(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    output_root = tmp_path / "outputs"
    report_root = tmp_path / "reports"
    status_tsv = run_root / "orchestrator" / "status.tsv"
    status_tsv.parent.mkdir(parents=True)
    status_tsv.write_text(
        "method\tphase\tstatus\trc\tlog\n"
        "refton\ttrain\tpaused\t130\t/refton.log\n",
        encoding="utf-8",
    )

    result = summarize_full_native(
        run_root=run_root,
        output_root=output_root,
        report_root=report_root,
        status_tsv=status_tsv,
        expected_records=2,
    )

    assert result["status"] == "in_progress"
    assert result["terminal_count"] == 1
    assert result["paused_count"] == 1
    assert result["methods"]["refton"]["state"] == "paused"



def test_finished_queue_turns_unrecorded_phases_into_terminal_results(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    output_root = tmp_path / "outputs"
    report_root = tmp_path / "reports"
    status_tsv = run_root / "orchestrator" / "status.tsv"
    logs = status_tsv.parent / "logs"
    logs.mkdir(parents=True)
    status_tsv.write_text(
        "method\tphase\tstatus\trc\tlog\n", encoding="utf-8"
    )
    (logs / "refton_train_20260824-110750.log").write_text(
        "RuntimeError: CUDA out of memory\n", encoding="utf-8"
    )

    result = summarize_full_native(
        run_root=run_root,
        output_root=output_root,
        report_root=report_root,
        status_tsv=status_tsv,
        expected_records=2,
        queue_finished=True,
    )

    assert result["status"] == "finished_with_failures"
    assert result["terminal_count"] == 5
    assert result["methods"]["refton"]["state"] == "oom"
    assert result["methods"]["ita_mdt"]["state"] == "failed"
    assert (
        "queue ended before all phase status records were written"
        in result["methods"]["refton"]["artifact_errors"]
    )
