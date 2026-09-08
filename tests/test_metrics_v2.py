from __future__ import annotations

import numpy as np

from metrics_v2.common import deterministic_panel_cases, paired_test, write_csv, write_json
from metrics_v2.features import (
    detect_print_region,
    gaussian_high_pass,
    masked_gradient_cosine,
)
from metrics_v2.parsing import labels_mask
from metrics_v2.pose import normalized_keypoint_distance
from metrics_v2.report import write_compare_report


def test_labels_mask_unions_requested_classes() -> None:
    labels = np.asarray([[0, 2, 3], [4, 7, 12]], dtype=np.uint8)
    assert labels_mask(labels, [3, 4, 7]).tolist() == [[0, 0, 1], [1, 1, 0]]


def test_normalized_keypoint_distance_uses_diagonal() -> None:
    generated = np.zeros((18, 2), dtype=np.float32)
    target = np.zeros((18, 2), dtype=np.float32)
    target[0] = [1.0, 1.0]
    scores = np.ones(18, dtype=np.float32)
    distance, count = normalized_keypoint_distance(generated, scores, target, scores, [0], 3, 4, 0.3)
    assert count == 1
    assert np.isclose(distance, 1.0)


def test_paired_wilcoxon_aligns_on_triplet_key() -> None:
    a = [
        {"mid": "a", "jid": "x", "seed": 0, "value": 2.0},
        {"mid": "b", "jid": "y", "seed": 0, "value": 4.0},
    ]
    b = [
        {"mid": "b", "jid": "y", "seed": 0, "value": 2.0},
        {"mid": "a", "jid": "x", "seed": 0, "value": 1.0},
    ]
    result = paired_test(a, b, "value")
    assert result["count"] == 2
    assert np.isclose(result["mean_diff"], 1.5)


def test_panel_selection_is_deterministic() -> None:
    subset = {
        "pairs": [
            {"mannequin_id": f"{mid:02d}", "identity_id": f"{jid:02d}"}
            for mid in range(10)
            for jid in range(4)
        ]
    }
    first = deterministic_panel_cases(subset, 7, mid_count=3, identities_per_mid=2)
    second = deterministic_panel_cases(subset, 7, mid_count=3, identities_per_mid=2)
    assert first == second
    assert len(first) == 6


def test_high_pass_removes_constant_component() -> None:
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    assert np.max(np.abs(gaussian_high_pass(image, sigma=2.0))) < 1e-4


def test_gradient_similarity_is_one_for_identical_texture() -> None:
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    image[:, 20:28] = 255
    mask = np.ones((48, 48), dtype=np.uint8)
    assert np.isclose(masked_gradient_cosine(image, mask, image, mask), 1.0)


def test_print_region_finds_dense_internal_texture() -> None:
    image = np.full((128, 96, 3), 128, dtype=np.uint8)
    mask = np.zeros((128, 96), dtype=np.uint8)
    mask[12:116, 8:88] = 1
    for x in range(34, 63, 4):
        image[42:78, x : x + 2] = 240
    bbox, score = detect_print_region(
        image,
        mask,
        gradient_percentile=75,
        density_threshold=0.05,
        min_component_fraction=0.0001,
    )
    x0, y0, x1, y1 = bbox
    assert x0 < 48 < x1
    assert y0 < 60 < y1
    assert score > 0


def test_compare_report_accepts_configured_run_names(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for directory, offset in ((left, 0.1), (right, 0.0)):
        key = {"mid": "m", "jid": "j", "seed": 0}
        write_csv(
            directory / "garment_per_image.csv",
            [{
                **key,
                "garment_dino_to_mannequin": 0.8 + offset,
                "garment_lpips_to_mannequin": 0.3 - offset,
                "garment_hf_lpips_to_mannequin": 0.2 - offset,
                "garment_gradient_sim_to_mannequin": 0.6 + offset,
                "print_region_hf_lpips": 0.25 - offset,
            }],
            [
                "mid", "jid", "seed", "garment_dino_to_mannequin",
                "garment_lpips_to_mannequin", "garment_hf_lpips_to_mannequin",
                "garment_gradient_sim_to_mannequin", "print_region_hf_lpips",
            ],
        )
        write_csv(
            directory / "pose_per_image.csv",
            [{
                **key,
                "body_distance_to_mannequin": 0.02,
                "head5_distance_to_mannequin": 0.03,
            }],
            ["mid", "jid", "seed", "body_distance_to_mannequin", "head5_distance_to_mannequin"],
        )
        write_csv(
            directory / "hair_per_image.csv",
            [{
                **key,
                "hair_dino_to_reference": 0.7 + offset,
                "hair_lab_distance": 5.0 - offset,
            }],
            ["mid", "jid", "seed", "hair_dino_to_reference", "hair_lab_distance"],
        )
        write_csv(
            directory / "identity/deltaid_per_image.csv",
            [{
                **key,
                "status": "ok",
                "sim_target": 0.4 + offset,
                "sim_source": 0.1,
                "delta_id": 0.3 + offset,
                "det_conf": 0.99,
            }],
            [
                "mid", "jid", "seed", "status", "sim_target", "sim_source",
                "delta_id", "det_conf",
            ],
        )
        write_json(directory / "distribution_summary.json", {"fid": 10.0, "kid_mean": 0.01})
        write_json(directory / "garment_summary.json", {
            "garment_dino": {"mean": 0.8 + offset},
            "garment_hf_lpips": {"mean": 0.2 - offset},
        })
        write_json(directory / "hair_summary.json", {
            "hair_dino": {"mean": 0.7 + offset},
        })
        write_json(directory / "pose_summary.json", {
            "head5_distance": {"mean": 0.03},
        })

    left_legacy = tmp_path / "legacy_left"
    right_legacy = tmp_path / "legacy_right"
    write_json(left_legacy / "garment_summary.json", {"cross_identity_group_mean_mean": 0.9})
    write_json(right_legacy / "garment_summary.json", {"cross_identity_group_mean_mean": 0.8})
    cfg = {
        "data": {"root": str(tmp_path)},
        "metrics_v2": {
            "runs": {
                "spatial": {
                    "label": "Spatial repair",
                    "legacy_metrics_dir": str(left_legacy),
                },
                "a4": {
                    "label": "A4",
                    "legacy_metrics_dir": str(right_legacy),
                },
            },
        },
    }
    report = write_compare_report(
        cfg,
        left,
        right,
        tmp_path / "compare.md",
        write_baseline=False,
        left_name="spatial",
        right_name="a4",
    )
    text = report.read_text(encoding="utf-8")
    assert "Spatial repair vs A4" in text
    assert "Spatial repair - A4" in text
    assert "four quality gates" in text
    assert "Garment-HF-LPIPS" in text
