from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tools.evaluate_role_flow import (
    METRIC_NAMES,
    _normalise_row,
    calibrate_tar,
    load_eval_rows,
    sha256_path,
)


def test_calibrate_tar_freezes_high_quantile_and_computes_tar() -> None:
    result = calibrate_tar([0.8, 0.9, 0.95], [0.1, 0.2, 0.3, 0.4], far=1e-3)
    assert result["threshold"] == pytest.approx(0.4)
    assert result["empirical_far"] == pytest.approx(0.25)
    assert result["tar"] == pytest.approx(1.0)


def test_calibrate_tar_rejects_empty_protocol() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        calibrate_tar([], [0.1])


def test_manifest_rows_are_sorted_and_duplicate_keys_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [
                    {"mid": "m2", "jid": "j2", "seed": 1, "split": "test", "generated_path": "/g/2.png", "mannequin_path": "/m/2.png", "human_path": "/h/2.png"},
                    {"mid": "m1", "jid": "j1", "seed": 0, "split": "test", "generated_path": "/g/1.png", "mannequin_path": "/m/1.png", "human_path": "/h/1.png"},
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = load_eval_rows(manifest, tmp_path / "generated", tmp_path / "dataset", "test")
    assert [(row["mid"], row["jid"]) for row in rows] == [("m1", "j1"), ("m2", "j2")]

    manifest.write_text(
        json.dumps({"rows": [{"mid": "m1", "jid": "j1", "seed": 0, "split": "test", "generated_path": "/g/1.png", "mannequin_path": "/m/1.png", "human_path": "/h/1.png"}] * 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_rows(manifest, tmp_path / "generated", tmp_path / "dataset", "test")


def test_normalise_row_preserves_explicit_paths(tmp_path: Path) -> None:
    row = _normalise_row(
        {
            "mid": "m",
            "jid": "j",
            "seed": "2",
            "split": "val",
            "generated_path": "/generated/out.png",
            "mannequin_path": "/data/m.png",
            "human_path": "/data/h.png",
        },
        tmp_path / "generated",
        tmp_path / "dataset",
    )
    assert row["generated_path"] == "/generated/out.png"
    assert row["seed"] == 2
    assert row["split"] == "val"


def test_sha256_path_directory_is_order_independent(tmp_path: Path) -> None:
    root = tmp_path / "d"
    (root / "b").mkdir(parents=True)
    (root / "a").write_bytes(b"a")
    (root / "b" / "c").write_bytes(b"c")
    first = sha256_path(root)
    (root / "b" / "c").touch()
    assert sha256_path(root) == first


def test_metric_contract_has_exact_eight_metrics() -> None:
    assert METRIC_NAMES == (
        "id_cosine",
        "tar_at_1e-3",
        "garment_dino",
        "garment_iou",
        "pose_pck",
        "bg_ssim",
        "bg_lpips",
        "fid",
    )
