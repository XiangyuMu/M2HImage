from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = Path(__file__).with_name("b_d0_probe.py")
SPEC = importlib.util.spec_from_file_location("b_d0_probe", SCRIPT)
b_d0_probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = b_d0_probe
SPEC.loader.exec_module(b_d0_probe)


def test_select_probe_ids_is_deterministic_and_unique() -> None:
    ids = ["a", "b", "b", "c", "d", "e", ""]
    first = b_d0_probe.select_probe_ids(ids, limit=3, seed=7)
    second = b_d0_probe.select_probe_ids(ids, limit=3, seed=7)
    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3
    assert all(item in {"a", "b", "c", "d", "e"} for item in first)


def test_select_probe_ids_can_preserve_explicit_order() -> None:
    ids = ["42143", "17500", "19461", "01708", "00430"]
    assert b_d0_probe.select_probe_ids(ids, limit=4, seed=999, preserve_order=True) == ids[:4]


def test_validate_sample_id_blocks_path_traversal() -> None:
    assert b_d0_probe.validate_sample_id("01708") == "01708"
    for bad in ("../01708", "a/b", "..", ".", "01708.png"):
        try:
            b_d0_probe.validate_sample_id(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe id was accepted: {bad}")


def test_assert_subset_of_old_val() -> None:
    b_d0_probe.assert_subset_of_old_val(["42143", "17500"], ["17500", "42143", "99999"])
    try:
        b_d0_probe.assert_subset_of_old_val(["42143", "not_val"], ["42143"])
    except ValueError as exc:
        assert "outside old-val" in str(exc)
    else:
        raise AssertionError("non-old-val explicit id was accepted")


def test_binary_mask_from_labels() -> None:
    labels = np.array([[0, 3, 4], [5, 9, 10]], dtype=np.uint8)
    mask = b_d0_probe.binary_mask_from_labels(labels, [3, 5, 10])
    np.testing.assert_array_equal(mask, np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8))


def test_masked_mse_and_psnr() -> None:
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    b = np.zeros((2, 2, 3), dtype=np.uint8)
    b[0, 0, :] = 255
    mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    mse = b_d0_probe.masked_mse(a, b, mask)
    assert mse == 1.0
    assert b_d0_probe.psnr_from_mse(mse) == 0.0
    assert math.isinf(b_d0_probe.psnr_from_mse(0.0))
    assert math.isnan(b_d0_probe.masked_mse(a, b, np.zeros((2, 2), dtype=np.uint8)))


def test_to_json_safe_converts_non_finite_numbers_to_none() -> None:
    payload = b_d0_probe.to_json_safe({"nan": float("nan"), "inf": float("inf"), "ok": np.float32(1.5)})
    assert payload == {"nan": None, "inf": None, "ok": 1.5}


def test_high_frequency_mse_identical_is_zero() -> None:
    image = np.full((8, 8, 3), 127, dtype=np.uint8)
    assert b_d0_probe.high_frequency_mse(image, image) == 0.0


def test_flux_latent_scaling_roundtrip() -> None:
    latents = np.array([-1.0, 0.0, 0.5, 2.0], dtype=np.float32)
    scaled = b_d0_probe.flux_scale_latents(latents, scaling_factor=0.3611, shift_factor=0.1159)
    restored = b_d0_probe.flux_unscale_latents(scaled, scaling_factor=0.3611, shift_factor=0.1159)
    np.testing.assert_allclose(restored, latents, rtol=1e-6, atol=1e-6)


def test_mask_bbox_with_padding_and_empty() -> None:
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[3:5, 4:7] = 1
    assert b_d0_probe.mask_bbox(mask, pad=2) == (2, 1, 9, 7)
    assert b_d0_probe.mask_bbox(np.zeros((3, 3), dtype=np.uint8)) is None


def test_prepare_probe_paths_never_reuses_existing_run(tmp_path: Path) -> None:
    first = b_d0_probe.prepare_probe_paths(tmp_path, run_name="fixed")
    second = b_d0_probe.prepare_probe_paths(tmp_path, run_name="fixed")
    assert first.run_dir.name == "fixed"
    assert second.run_dir.name == "fixed_001"
    assert first.metrics_jsonl != second.metrics_jsonl
    assert first.masks_dir.exists()
    assert second.recon_dir.exists()


def test_build_summary_marks_diagnostic_not_formal(tmp_path: Path) -> None:
    paths = b_d0_probe.prepare_probe_paths(tmp_path, run_name="summary")
    args = SimpleNamespace(
        ids_json=None,
        repo_root=Path("/repo"),
        data_root=Path("/data"),
        vae_dir=Path("/repo/models/hf/black-forest-labs/FLUX.1-dev/vae"),
        fashn_model_dir=Path("/repo/models/hf/fashn-ai/fashn-human-parser"),
        device="cuda:3",
        dtype="float16",
        limit=4,
        seed=1,
        parser_batch_size=1,
        parser_input_width=384,
        parser_input_height=576,
        save_recon_limit=2,
        lpips_mode="auto",
    )
    rows = [{"status": "ok", "mse_full": 0.25, "psnr_full": 6.0}, {"status": "failed"}]
    summary = b_d0_probe.build_summary(
        args,
        rows,
        ["00006", "00019"],
        paths,
        "0.1.4",
        True,
        {"script": {"sha256": "abc"}},
        {"enabled": False},
        started_at=0.0,
    )
    assert summary["formal_result"] is False
    assert summary["source_boundary"]["source"] == "old-val"
    assert "old-test" in summary["source_boundary"]["forbidden_sources"]
    assert summary["metrics"]["count_ok"] == 1
    assert summary["metrics"]["count_failed"] == 1
    assert summary["lpips"]["version"] == "0.1.4"
    assert summary["lpips"]["enabled"] is True
    assert summary["hashes"]["script"]["sha256"] == "abc"


def test_read_explicit_ids_accepts_fixed_json_list(tmp_path: Path) -> None:
    ids_path = tmp_path / "old_val_probe_ids.json"
    ids_path.write_text('["42143", "17500", "19461", "01708"]', encoding="utf-8")
    assert b_d0_probe.read_explicit_ids(ids_path) == ["42143", "17500", "19461", "01708"]


def test_parse_args_no_lpips_alias() -> None:
    args = b_d0_probe.parse_args(["--no-lpips", "--limit", "4"])
    assert args.lpips_mode == "off"
    assert args.dtype == "float32"
