from __future__ import annotations

from pathlib import Path
from typing import Any

from metrics.common import safe_mean, sha256_short
from metrics.heldout_id import run_deltaid
from metrics_v2.common import read_csv, write_json


# Evaluation-only boundary: held-out AdaFace must never be imported by training code.
def run_identity_metrics(
    cfg: dict[str, Any], subset: dict[str, Any], gen_dir: str | Path, out_dir: str | Path, device: str
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    identity_dir = out_dir / "identity"
    checkpoint = Path(cfg["metrics"]["heldout_id"]["checkpoint"])
    actual_hash = sha256_short(checkpoint)
    expected_hash = str(cfg["metrics_v2"]["identity"]["expected_adaface_hash"])
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"held-out AdaFace hash mismatch: expected {expected_hash}, got {actual_hash} for {checkpoint}"
        )
    summary = run_deltaid(cfg, subset, gen_dir, identity_dir, device=device, fail_on_unavailable=True)
    rows = read_csv(identity_dir / "deltaid_per_image.csv")
    valid = [row for row in rows if row.get("status") == "ok"]
    summary = dict(summary)
    summary.update(
        {
            "measures_against": "reference person jid and original source person mid using held-out AdaFace IR-101",
            "face_detection_rate": len(valid) / len(rows) if rows else None,
            "det_conf_mean": safe_mean([row.get("det_conf") for row in valid]),
            "csv": str(identity_dir / "deltaid_per_image.csv"),
        }
    )
    write_json(out_dir / "identity_summary.json", summary)
    return summary
