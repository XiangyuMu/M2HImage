from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dataset import ASYNC_FLOW_MASK_KEYS, PairedWarmupDataset
from tools.evaluate_role_flow import (
    METRIC_NAMES,
    _cfg_with_data_root,
    _dwpose_target_path,
    _face_crop_path,
    _normalise_row,
    _source_garment_mask,
    calibrate_tar,
    load_eval_rows,
    resolve_asset_root,
    sha256_path,
)




def test_asset_root_defaults_to_manifest_read_only_source_root(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"provenance": {"read_only_source_root": "/source/assets"}, "rows": []}),
        encoding="utf-8",
    )

    assert resolve_asset_root(manifest, "/clean/root") == Path("/source/assets")
    assert resolve_asset_root(manifest, "/clean/root", "/override/assets") == Path("/override/assets")

    manifest.write_text(json.dumps({"provenance": {}, "rows": []}), encoding="utf-8")
    assert resolve_asset_root(manifest, "/clean/root") == Path("/clean/root")


def test_source_asset_helpers_use_asset_root_without_changing_dataset_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clean = tmp_path / "clean"
    asset = tmp_path / "asset"
    face_dir = asset / "derived/face_crops/human"
    face_dir.mkdir(parents=True)
    (face_dir / "id1.png").write_bytes(b"face")
    cfg = {"data": {"root": str(clean), "asset_root": str(asset)}}

    assert _face_crop_path(cfg, "id1") == face_dir / "id1.png"
    assert _dwpose_target_path(cfg, "m1") == asset / "dwpose/keypoints/mannequin/m1.npz"
    assert _cfg_with_data_root(cfg, asset)["data"]["root"] == str(asset)
    assert cfg["data"]["root"] == str(clean)

    calls = []

    def fake_source_garment_mask(inner_cfg, mid, size):
        calls.append((inner_cfg, mid, size))
        return np.ones((2, 2), dtype=np.uint8)

    monkeypatch.setattr("metrics_v2.parsing.source_garment_mask", fake_source_garment_mask)
    mask = _source_garment_mask(cfg, "m1", (2, 2))

    assert mask.shape == (2, 2)
    assert calls[0][0]["data"]["root"] == str(asset)
    assert calls[0][1:] == ("m1", (2, 2))

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


def test_async_flow_dataset_requests_all_required_region_masks(tmp_path: Path) -> None:
    (tmp_path / "train.txt").write_text("00001\n", encoding="utf-8")
    cfg = {
        "experiment": {"seed": 0},
        "experiment_method": {"name": "C"},
        "data": {
            "root": str(tmp_path),
            "train_split": "train.txt",
            "cache_dir": "cache",
            "resolution": {"width": 768, "height": 1024},
        },
        "model": {},
        "training": {},
    }

    dataset = PairedWarmupDataset(cfg, "train", require_coverage=False)

    assert dataset.async_flow_enabled
    assert dataset.region_mask_keys == set(ASYNC_FLOW_MASK_KEYS)


def test_c_async_flow_is_cumulative_with_b_region_weighting() -> None:
    config_root = Path(__file__).resolve().parents[1] / "configs" / "role_selective"
    with (config_root / "B_garment_weighted.yaml").open("r", encoding="utf-8") as handle:
        config_b = yaml.safe_load(handle)
    with (config_root / "C_async_flow.yaml").open("r", encoding="utf-8") as handle:
        config_c = yaml.safe_load(handle)

    assert config_c["experiment_method"]["name"] == "C"
    assert "B plus" in config_c["experiment_method"]["description"]
    assert config_c["experiment_method"]["region_schedule"] == {
        "background": -0.5,
        "garment": -0.5,
        "boundary": 0.0,
        "identity": 0.3,
    }
    assert config_c["training"]["paired_region_weighting"]["enabled"] is True
    assert (
        config_c["training"]["paired_region_weighting"]
        == config_b["training"]["paired_region_weighting"]
    )
    assert config_b["training"]["resume"] is None
    assert config_c["training"]["resume"] is None
