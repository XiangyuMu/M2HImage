from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools import evaluate_role_flow_v2 as v2


def _sha(path: Path) -> str:
    return v2.sha256_file(path)


def _write_rgb(path: Path, color: tuple[int, int, int] = (1, 2, 3), size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _write_label(path: Path, label: int = 3, size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0]), label, dtype=np.uint8), mode="L").save(path)


@pytest.fixture()
def tiny_protocol(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    counts = {"val": 3, "test": 5}
    monkeypatch.setattr(v2, "EXPECTED_EVAL_COUNT", sum(counts.values()))
    monkeypatch.setattr(v2, "EXPECTED_SPLIT_COUNTS", counts)
    monkeypatch.setattr(v2, "EXPECTED_FID_REFERENCE_COUNT", 9)
    return {"split_counts": counts, "eval_count": sum(counts.values()), "fid_count": 9, "size": (32, 32)}


def _fixture(tmp_path: Path) -> dict[str, Path]:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"ckpt")
    config.write_text(
        "data:\n  root: /unused\n  resolution:\n    width: 32\n    height: 32\n"
        "metrics:\n  heldout_id:\n    checkpoint: /unused/adaface.ckpt\n",
        encoding="utf-8",
    )
    asset = tmp_path / "asset"
    generated = tmp_path / "generated"
    rows = []
    val_count = int(v2.EXPECTED_SPLIT_COUNTS["val"])
    for index in range(v2.EXPECTED_EVAL_COUNT):
        split = "val" if index < val_count else "test"
        mid = f"m{index:04d}"
        jid = f"j{index:04d}"
        seed = index % 7
        human = asset / "human" / f"{jid}.png"
        mannequin = asset / "mannequin" / f"{mid}.png"
        generated_path = generated / f"{mid}__id{jid}__seed{seed}.png"
        _write_rgb(human, color=(index % 255, 2, 3))
        _write_rgb(mannequin, color=(1, index % 255, 3))
        _write_rgb(generated_path, color=(1, 2, index % 255))
        _write_label(asset / "human_parsing" / "fashn" / "masks" / "mannequin" / f"{mid}.png", label=3)
        rows.append(
            {
                "split": split,
                "mid": mid,
                "jid": jid,
                "seed": seed,
                "human_path": str(human),
                "human_sha256": _sha(human),
                "mannequin_path": str(mannequin),
                "mannequin_sha256": _sha(mannequin),
            }
        )
    manifest = tmp_path / "eval_manifest.json"
    manifest_payload = {
        "schema_version": "m2h-role-flow-eval-v1",
        "content_sha256": "manifest-content",
        "rows": rows,
    }
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    expected = [v2.expected_filename(row) for row in rows]
    images = [{"filename": name, "sha256": _sha(generated / name), "readable": True, "width": 32, "height": 32, "mode": "RGB"} for name in expected]
    generation_provenance = generated / "generation_provenance.json"
    generation_provenance.write_text(
        json.dumps(
            {
                "schema_version": "m2h-role-flow-generation-v2",
                "status": "complete",
                "inputs": {
                    "config": {"path": str(config), "sha256": _sha(config)},
                    "checkpoint": {"path": str(checkpoint), "type": "directory", "sha256": v2.sha256_path(checkpoint)},
                    "manifest": {"path": str(manifest), "sha256": _sha(manifest), "content_sha256": "manifest-content"},
                },
                "selection": {
                    "split": "all",
                    "selected_row_count": v2.EXPECTED_EVAL_COUNT,
                    "row_keys": [{"split": r["split"], "mid": r["mid"], "jid": r["jid"], "seed": r["seed"]} for r in rows],
                    "expected_filenames": expected,
                },
                "images": images,
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    threshold = tmp_path / "tar_calibration.json"
    threshold.write_text(
        json.dumps(
            {
                "schema_version": "m2h-role-flow-metric-protocol-v1",
                "artifact": "tar_calibration",
                "far_target": 1e-3,
                "threshold": 0.42,
                "content_sha256": "tar-content",
                "model": {"recognizer": "heldout-fake"},
            }
        ),
        encoding="utf-8",
    )
    fid_rows = []
    for index in range(v2.EXPECTED_FID_REFERENCE_COUNT):
        path = asset / "fid_human" / f"t{index:04d}.png"
        _write_rgb(path, color=(index % 255, 8, 9))
        fid_rows.append({"id": f"t{index:04d}", "split": "test", "human_path": str(path), "human_sha256": _sha(path)})
    fid = tmp_path / "fid_reference_manifest.json"
    fid.write_text(json.dumps({"artifact": "fid_reference_manifest", "count": v2.EXPECTED_FID_REFERENCE_COUNT, "content_sha256": "fid-content", "rows": fid_rows}), encoding="utf-8")
    return {
        "config": config,
        "checkpoint": checkpoint,
        "asset": asset,
        "generated": generated,
        "manifest": manifest,
        "generation_provenance": generation_provenance,
        "threshold": threshold,
        "fid": fid,
        "output": tmp_path / "metrics",
    }


def test_preflight_accepts_exact_frozen_protocol(tmp_path: Path, tiny_protocol: dict[str, object]) -> None:
    paths = _fixture(tmp_path)
    protocol = v2.preflight_protocol(
        config_path=paths["config"],
        manifest_path=paths["manifest"],
        generated_dir=paths["generated"],
        generation_provenance_path=paths["generation_provenance"],
        checkpoint_path=paths["checkpoint"],
        asset_root=paths["asset"],
        identity_threshold_path=paths["threshold"],
        fid_reference_manifest_path=paths["fid"],
        output_dir=paths["output"],
    )

    assert protocol.split_counts == tiny_protocol["split_counts"]
    assert len(protocol.rows) == tiny_protocol["eval_count"]
    assert protocol.identity_threshold == pytest.approx(0.42)
    assert protocol.fid_reference_count == tiny_protocol["fid_count"]


@pytest.mark.parametrize("drop_count,match", [(1, "expected exactly"), (-1, "expected exactly")])
def test_preflight_rejects_wrong_row_count(tmp_path: Path, tiny_protocol: dict[str, object], drop_count: int, match: str) -> None:
    paths = _fixture(tmp_path)
    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if drop_count > 0:
        payload["rows"] = payload["rows"][:-drop_count]
    else:
        payload["rows"].append(dict(payload["rows"][-1], mid="extra", jid="extra", seed=0))
    paths["manifest"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        v2.preflight_protocol(
            config_path=paths["config"],
            manifest_path=paths["manifest"],
            generated_dir=paths["generated"],
            generation_provenance_path=paths["generation_provenance"],
            checkpoint_path=paths["checkpoint"],
            asset_root=paths["asset"],
            identity_threshold_path=paths["threshold"],
            fid_reference_manifest_path=paths["fid"],
            output_dir=paths["output"],
        )


def test_preflight_rejects_corrupt_or_wrong_size_generated_image(tmp_path: Path, tiny_protocol: dict[str, object]) -> None:
    paths = _fixture(tmp_path)
    bad = next(paths["generated"].glob("*.png"))
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="unreadable generated image"):
        v2.preflight_protocol(
            config_path=paths["config"],
            manifest_path=paths["manifest"],
            generated_dir=paths["generated"],
            generation_provenance_path=paths["generation_provenance"],
            checkpoint_path=paths["checkpoint"],
            asset_root=paths["asset"],
            identity_threshold_path=paths["threshold"],
            fid_reference_manifest_path=paths["fid"],
            output_dir=paths["output"],
        )

    _write_rgb(bad, size=(16, 16))
    with pytest.raises(ValueError, match="expected RGB"):
        v2.preflight_protocol(
            config_path=paths["config"],
            manifest_path=paths["manifest"],
            generated_dir=paths["generated"],
            generation_provenance_path=paths["generation_provenance"],
            checkpoint_path=paths["checkpoint"],
            asset_root=paths["asset"],
            identity_threshold_path=paths["threshold"],
            fid_reference_manifest_path=paths["fid"],
            output_dir=paths["output"],
        )


def test_output_dir_must_be_empty_and_protocol_hashes_must_match(tmp_path: Path, tiny_protocol: dict[str, object]) -> None:
    paths = _fixture(tmp_path)
    paths["output"].mkdir()
    (paths["output"] / "old.csv").write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output directory must be empty"):
        v2.preflight_protocol(
            config_path=paths["config"],
            manifest_path=paths["manifest"],
            generated_dir=paths["generated"],
            generation_provenance_path=paths["generation_provenance"],
            checkpoint_path=paths["checkpoint"],
            asset_root=paths["asset"],
            identity_threshold_path=paths["threshold"],
            fid_reference_manifest_path=paths["fid"],
            output_dir=paths["output"],
        )

    paths["output"].unlink() if paths["output"].is_file() else None
    for item in paths["output"].iterdir():
        item.unlink()
    provenance = json.loads(paths["generation_provenance"].read_text(encoding="utf-8"))
    provenance["inputs"]["config"]["sha256"] = "bad"
    paths["generation_provenance"].write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="config sha256 mismatch"):
        v2.preflight_protocol(
            config_path=paths["config"],
            manifest_path=paths["manifest"],
            generated_dir=paths["generated"],
            generation_provenance_path=paths["generation_provenance"],
            checkpoint_path=paths["checkpoint"],
            asset_root=paths["asset"],
            identity_threshold_path=paths["threshold"],
            fid_reference_manifest_path=paths["fid"],
            output_dir=paths["output"],
        )


def test_source_garment_mask_uses_mannequin_fashn_labels_without_fallback(tmp_path: Path) -> None:
    asset = tmp_path / "asset"
    label_path = asset / "human_parsing" / "fashn" / "masks" / "mannequin" / "m1.png"
    labels = np.array([[0, 3, 4], [5, 6, 10], [7, 2, 1]], dtype=np.uint8)
    label_path.parent.mkdir(parents=True)
    Image.fromarray(labels, mode="L").save(label_path)
    mask, record = v2.load_source_garment_mask(asset, "m1", size=None)

    assert mask.astype(int).tolist() == [[0, 1, 1], [1, 1, 1], [1, 0, 0]]
    assert record["source_role"] == "mannequin"
    assert record["source_path"] == str(label_path)
    assert record["source_sha256"] == _sha(label_path)


def test_generated_low_garment_area_yields_null_metrics_without_fallback() -> None:
    source = np.ones((10, 10), dtype=bool)
    generated = np.zeros((10, 10), dtype=np.uint8)

    result = v2.garment_iou_from_masks(generated, source, min_area_fraction=0.005)

    assert result["garment_iou"] is None
    assert result["status"] == "generated_garment_invalid"


def test_common_background_uses_full_foreground_and_11x11_erosion() -> None:
    source_labels = np.zeros((32, 32), dtype=np.uint8)
    generated_labels = np.zeros((32, 32), dtype=np.uint8)
    source_labels[8:16, 8:16] = 3
    generated_labels[10:18, 10:18] = 1

    common = v2.common_background_mask(source_labels, generated_labels)

    assert not common[12, 12]
    assert common[0, 0]
    assert not common[7, 7]


def test_masked_ssim_averages_the_ssim_map_over_background(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_map = np.zeros((4, 4, 3), dtype=np.float32)
    expected_map[..., 0] = 0.2
    expected_map[..., 1] = 0.6
    expected_map[..., 2] = 1.0

    def fake_ssim(*_args, **kwargs):
        assert kwargs["full"] is True
        return 0.0, expected_map

    skimage = types.ModuleType("skimage")
    metrics = types.ModuleType("skimage.metrics")
    metrics.structural_similarity = fake_ssim
    skimage.metrics = metrics
    monkeypatch.setitem(sys.modules, "skimage", skimage)
    monkeypatch.setitem(sys.modules, "skimage.metrics", metrics)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    result = v2._masked_ssim(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.uint8),
        mask,
    )

    assert result == pytest.approx(0.6)


def test_fixed_identity_threshold_is_loaded_without_recalibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    threshold = tmp_path / "threshold.json"
    threshold.write_text(json.dumps({"artifact": "tar_calibration", "threshold": 0.5, "far_target": 1e-3}), encoding="utf-8")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("calibration must not run in evaluator v2")

    monkeypatch.setattr(v2, "calibrate_tar", fail_if_called, raising=False)

    payload = v2.load_identity_threshold(threshold)
    assert payload["threshold"] == pytest.approx(0.5)


def test_fid_is_summary_only_not_per_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_protocol: dict[str, object]) -> None:
    paths = _fixture(tmp_path)

    def metric_rows(fields: tuple[str, ...]) -> dict[tuple[str, str, int], dict[str, object]]:
        rows = {}
        for row in json.loads(paths["manifest"].read_text(encoding="utf-8"))["rows"]:
            key = (row["mid"], row["jid"], int(row["seed"]))
            value = {"generated_path": str(paths["generated"] / v2.expected_filename(row)), "status": "ok", "error": ""}
            value.update({field: 0.5 for field in fields})
            rows[key] = value
        return rows

    monkeypatch.setattr(v2, "_run_identity_metrics", lambda *a, **k: metric_rows(("id_cosine", "tar_at_1e-3")))
    monkeypatch.setattr(v2, "_run_garment_metrics", lambda *a, **k: metric_rows(("garment_dino", "garment_iou")))
    monkeypatch.setattr(v2, "_run_background_metrics", lambda *a, **k: metric_rows(("bg_ssim", "bg_lpips")))
    monkeypatch.setattr(v2, "_run_pose_metrics", lambda *a, **k: metric_rows(("pose_pck",)))
    monkeypatch.setattr(v2, "_run_fid", lambda *a, **k: 12.5)

    summary = v2.evaluate_v2(
        config_path=paths["config"],
        manifest_path=paths["manifest"],
        generated_dir=paths["generated"],
        generation_provenance_path=paths["generation_provenance"],
        checkpoint_path=paths["checkpoint"],
        asset_root=paths["asset"],
        identity_threshold_path=paths["threshold"],
        fid_reference_manifest_path=paths["fid"],
        output_dir=paths["output"],
        device="cpu",
        strict=True,
    )

    per_pair = (paths["output"] / "per_pair_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    set_metrics = json.loads((paths["output"] / "set_metrics.json").read_text(encoding="utf-8"))
    assert "fid" not in per_pair
    assert set_metrics["fid"]["value"] == pytest.approx(12.5)
    assert summary["status"] == "complete"
    assert (paths["output"] / "READY").is_file()


def test_strict_rejects_missing_metric_rows_and_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_protocol: dict[str, object],
) -> None:
    paths = _fixture(tmp_path)

    def complete_rows(fields: tuple[str, ...]) -> dict[tuple[str, str, int], dict[str, object]]:
        result = {}
        for row in json.loads(paths["manifest"].read_text(encoding="utf-8"))["rows"]:
            key = (row["mid"], row["jid"], int(row["seed"]))
            record = {"generated_path": str(paths["generated"] / v2.expected_filename(row)), "status": "ok", "error": ""}
            record.update({field: 0.5 for field in fields})
            result[key] = record
        return result

    identity = complete_rows(("id_cosine", "tar_at_1e-3"))
    identity.pop(next(iter(identity)))
    garment = complete_rows(("garment_dino",))
    monkeypatch.setattr(v2, "_run_identity_metrics", lambda *a, **k: identity)
    monkeypatch.setattr(v2, "_run_garment_metrics", lambda *a, **k: garment)
    monkeypatch.setattr(v2, "_run_background_metrics", lambda *a, **k: complete_rows(("bg_ssim", "bg_lpips")))
    monkeypatch.setattr(v2, "_run_pose_metrics", lambda *a, **k: complete_rows(("pose_pck",)))
    monkeypatch.setattr(v2, "_run_fid", lambda *a, **k: 1.0)

    with pytest.raises(RuntimeError, match="metric failures"):
        v2.evaluate_v2(
            config_path=paths["config"],
            manifest_path=paths["manifest"],
            generated_dir=paths["generated"],
            generation_provenance_path=paths["generation_provenance"],
            checkpoint_path=paths["checkpoint"],
            asset_root=paths["asset"],
            identity_threshold_path=paths["threshold"],
            fid_reference_manifest_path=paths["fid"],
            output_dir=paths["output"],
            device="cpu",
            strict=True,
        )

    failures = (paths["output"] / "failures.csv").read_text(encoding="utf-8")
    assert "missing metric row" in failures
    assert "missing fields garment_iou" in failures
