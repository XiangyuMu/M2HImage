"""Pure array tests; unittest only, no models, data files, SSH, or downloads.

python -B -m unittest discover -s <scripts> -p test_b_alignment_boundary.py -v
"""
from dataclasses import replace
import unittest

import cv2
import numpy as np

from b_alignment_boundary import (
    Config, METHODS, aggregate, alignment_decision, calibration_cases,
    correspondence_support, crossing_statistics, estimate_alignment, feather_alpha,
    fixed_regions, measure, method_output, multiband_blend, reference_context,
    sample_translation, summarize_calibration, synthetic_perturbation,
)


def fixture(height=128, width=160):
    rng = np.random.default_rng(1701)
    image = cv2.GaussianBlur(rng.uniform(25, 230, (height, width, 3)).astype(np.float32), (0, 0), 1.2)
    image = np.rint(image).astype(np.uint8)
    # Distinct low-frequency structures prevent periodic translation ambiguity.
    cv2.circle(image, (width // 3, height // 2), 12, (210, 55, 120), -1)
    cv2.rectangle(image, (width // 2, height // 3), (width // 2 + 20, height // 3 + 12), (40, 200, 75), -1)
    mask = np.zeros((height, width), bool)
    mask[16:-16, 16:-16] = True
    return image, mask


def translated_forward(image, dx, dy):
    """Independent OpenCV forward warp defines expected visual motion."""
    return cv2.warpAffine(image, np.array([[1, 0, dx], [0, 1, dy]], np.float32),
                          (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


class WarpTests(unittest.TestCase):
    def test_sample_identity_exact_including_borders(self):
        source, _ = fixture()
        output, valid = sample_translation(source, (0, 0))
        np.testing.assert_array_equal(output, source)
        self.assertTrue(valid.all())

    def test_sampling_direction_known_integer_translation(self):
        source, _ = fixture()
        expected = translated_forward(source, 3, -2)
        output, valid = sample_translation(source, (-3, 2))
        np.testing.assert_array_equal(output[valid], expected[valid])
        self.assertFalse(valid[:, :3].any())
        self.assertFalse(valid[-2:, :].any())
        self.assertTrue(valid[:-2, 3:].all())
        wrong, _ = sample_translation(source, (3, -2))
        self.assertGreater(float(np.mean(np.abs(output[valid].astype(float) - wrong[valid]))), 2)

    def test_ecc_identity_and_all_methods_identity(self):
        source, mask = fixture()
        aligned = estimate_alignment(source, source, mask)
        self.assertTrue(aligned["accepted"], aligned)
        np.testing.assert_allclose(aligned["applied_sample_dxdy"], [0, 0], atol=.05)
        for method in METHODS:
            output, _, _ = method_output(method, source, source, mask, aligned)
            np.testing.assert_array_equal(output, source)

    def test_ecc_forward_backward_known_translations_and_resize_units(self):
        source, mask = fixture(192, 240)
        config = replace(Config(), ecc_max_side=160)
        for dx, dy in ((3, -2), (-2, 3), (1.5, -.75)):
            with self.subTest(dx=dx, dy=dy):
                base = translated_forward(source, dx, dy)
                result = estimate_alignment(source, base, mask, config)
                self.assertTrue(result["accepted"], result)
                np.testing.assert_allclose(result["applied_sample_dxdy"], [-dx, -dy], atol=.4)
                np.testing.assert_allclose(result["reverse_sample_dxdy"], [dx, dy], atol=.4)
                self.assertLess(result["fb_error_px"], config.fb_limit_px)
                corrected, _ = sample_translation(source, result["applied_sample_dxdy"])
                interior = fixed_regions(mask)["interior"]
                before = float(np.mean(np.abs(source[interior].astype(float) - base[interior])))
                after = float(np.mean(np.abs(corrected[interior].astype(float) - base[interior])))
                self.assertLess(after, before * .3)

    def test_alignment_policy_bound_fb_correlation_and_nonfinite(self):
        _, mask = fixture()
        cases = [([7, 0], [-7, 0], [.95, .95], "displacement_cap"),
                 ([4, 0], [-1, 0], [.95, .95], "forward_backward_inconsistent"),
                 ([1, 0], [-1, 0], [.1, .95], "low_ecc_correlation"),
                 ([np.nan, 0], [0, 0], [.95, .95], "nonfinite_ecc")]
        for forward, backward, corr, reason in cases:
            result = alignment_decision(forward, backward, corr, mask)
            self.assertFalse(result["accepted"])
            self.assertIn(reason, result["reasons"])
            self.assertEqual(result["applied_sample_dxdy"], [0, 0])

    def test_displacement_cap_is_euclidean_and_checks_reverse(self):
        _, mask = fixture()
        for fwd, rev in (([5, 5], [-5, -5]), ([5.5, 0], [-6.1, 0])):
            result = alignment_decision(fwd, rev, [.95, .95], mask)
            self.assertIn("displacement_cap", result["reasons"])

    def test_low_texture_fallback_explicit_and_feather_equals_inplace(self):
        _, mask = fixture()
        source = np.full((*mask.shape, 3), 100, np.uint8)
        base = np.full_like(source, 120)
        result = estimate_alignment(source, base, mask)
        self.assertFalse(result["accepted"])
        self.assertIn("insufficient_gray_texture", result["reasons"])
        a, _, _ = method_output("inplace_feather", source, base, mask, result)
        b, _, _ = method_output("warp_feather", source, base, mask, result)
        np.testing.assert_array_equal(a, b)

    def test_fractional_source_support_excludes_mask_and_frame_extrapolation(self):
        _, mask = fixture()
        support = correspondence_support(mask, (.5, 0))
        self.assertTrue(np.all(~support | mask))
        self.assertFalse(support[:, -17].any())
        full = np.ones((12, 14), bool)
        support = correspondence_support(full, (-.5, 0))
        self.assertFalse(support[:, 0].any())
        self.assertTrue(support[:, 1:].all())


class BlendMetricTests(unittest.TestCase):
    def test_fixed_partition_and_zero_gate_do_not_shrink_metric_roi(self):
        source, mask = fixture()
        base = np.clip(source.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        regions = fixed_regions(mask)
        np.testing.assert_array_equal(regions["interior"].astype(int) + regions["boundary"] + regions["exterior"], np.ones(mask.shape))
        output = multiband_blend(source, base, np.zeros(mask.shape, np.float32))
        np.testing.assert_array_equal(output, base)
        result = measure(source, base, output, mask)
        self.assertEqual(result["garment_area"], int(mask.sum()))
        self.assertGreater(result["garment_mse_to_source"], 0)
        self.assertEqual(result["outside_M_edit_fraction"], 0)

    def test_all_methods_strict_no_spill_on_odd_dimensions(self):
        source, mask = fixture(129, 161)
        base = np.flip(source, axis=1).copy()
        aligned = {"applied_sample_dxdy": [3, -2]}
        for method in METHODS:
            output, _, coverage = method_output(method, source, base, mask, aligned)
            np.testing.assert_array_equal(output[~mask], base[~mask])
            result = measure(source, base, output, mask)
            self.assertEqual(result["outside_M_changed_pixels"], 0)
            self.assertEqual(result["outside_allowed_changed_pixels"], 0)
            self.assertEqual(result["garment_area"], int(mask.sum()))
            self.assertGreaterEqual(coverage["alpha_positive_fraction_of_M"], 0)
            self.assertLessEqual(coverage["alpha_positive_fraction_of_M"], 1)

    def test_multiband_zero_one_and_identity(self):
        source, _ = fixture(65, 81)
        base = np.full_like(source, 60)
        for alpha, expected in ((np.zeros(source.shape[:2], np.float32), base),
                                (np.ones(source.shape[:2], np.float32), source)):
            np.testing.assert_array_equal(multiband_blend(source, base, alpha), expected)
        a = np.full(source.shape[:2], .35, np.float32)
        np.testing.assert_array_equal(multiband_blend(source, source, a), source)

    def test_feather_native_width_and_empty_full_mask_defined(self):
        mask = np.zeros((80, 80), bool)
        mask[10:70, 10:70] = True
        alpha = feather_alpha(mask)
        self.assertEqual(alpha[10, 40], 0)
        self.assertEqual(alpha[13, 40], 0)
        self.assertAlmostEqual(float(alpha[14, 40]), 1 / 8)
        self.assertEqual(alpha[30, 40], 1)
        self.assertEqual(float(feather_alpha(np.zeros_like(mask)).max()), 0)
        self.assertEqual(feather_alpha(np.ones_like(mask))[0, 0], 0)

    def test_identity_fidelity_zero_and_empty_regions_null(self):
        source, mask = fixture()
        result = measure(source, source, source, mask)
        for key, value in result.items():
            if key.endswith("_to_source") and value is not None:
                self.assertEqual(value, 0, key)
        result = measure(source, source, source, np.zeros_like(mask))
        self.assertEqual(result["garment_area"], 0)
        self.assertIsNone(result["garment_mse_to_source"])
        self.assertIsNone(result["boundary_cross_rgb_mean"])

    def test_spill_fraction_fixed_denominator_and_strict_threshold(self):
        source, mask = fixture()
        base = np.zeros_like(source)
        output = base.copy()
        output[0, 0] = 3
        output[0, 1] = 2
        result = measure(source, base, output, mask)
        self.assertEqual(result["outside_allowed_changed_pixels"], 1)
        self.assertEqual(result["outside_allowed_edit_fraction"], 1 / result["outside_allowed_area"])

    def test_texture_eligibility_depends_only_on_source(self):
        source, mask = fixture()
        context = reference_context(source, mask)
        clean = measure(source, source, source, mask, context=context)
        blurred = cv2.GaussianBlur(source, (0, 0), 3)
        result = measure(source, source, blurred, mask, context=context)
        self.assertGreater(result["strong_texture_area"], 0)
        self.assertEqual(result["strong_texture_area"], clean["strong_texture_area"])
        self.assertGreater(result["strong_texture_gradient_l1_to_source"], 0)
        self.assertGreater(result["strong_texture_hf_l1_to_source"], 0)


class CalibrationTests(unittest.TestCase):
    def test_splice_crossing_score_increases_on_known_flat_fixture(self):
        _, mask = fixture()
        clean = np.full((*mask.shape, 3), 100, np.uint8)
        scores = []
        for severity in (0, 8, 24):
            out = synthetic_perturbation(clean, mask, "splice_offset", severity)
            np.testing.assert_array_equal(out[~mask], clean[~mask])
            scores.append(crossing_statistics(out, mask)["boundary_cross_excess_mean"])
        self.assertEqual(scores[0], 0)
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[2], scores[1])

    def test_misalignment_score_sensitive_on_nonperiodic_texture(self):
        source, mask = fixture()
        perturbed = synthetic_perturbation(source, mask, "misalignment", 3)
        np.testing.assert_array_equal(perturbed[~mask], source[~mask])
        expected = translated_forward(source, 3, 0)
        np.testing.assert_array_equal(perturbed[mask], expected[mask])
        before = measure(source, source, source, mask)
        after = measure(source, source, perturbed, mask)
        self.assertGreater(after["boundary_gradient_l1_to_source"], before["boundary_gradient_l1_to_source"])
        self.assertGreater(after["boundary_cross_excess_mean"], before["boundary_cross_excess_mean"])

    def test_calibration_reports_insensitivity_instead_of_claiming_success(self):
        rows = []
        for kind, severity in calibration_cases():
            rows.append({"mid": "one", "kind": kind, "severity": severity,
                         "boundary_cross_rgb_mean": 0.0,
                         "synthetic_boundary_rgb_mae_to_known_clean": float(severity) / 255})
        summary = summarize_calibration(rows)
        metric = summary["cases"]["misalignment_3"]["metrics"]["boundary_cross_rgb_mean"]
        self.assertEqual(metric["fraction_increased_vs_clean"], 0)
        self.assertEqual(metric["n"], 1)

    def test_failure_count_not_removed_from_aggregation_denominator(self):
        rows = [{"name": "a", "method": "base", "status": "ok", "metrics": {"score": 1.0}, "coverage": {}},
                {"name": "b", "method": "base", "status": "failed"},
                {"name": "a", "method": "warp_feather", "status": "ok", "alignment_status": "identity_fallback",
                 "metrics": {"score": .5}, "coverage": {}},
                {"name": "b", "method": "warp_feather", "status": "failed"}]
        summary = aggregate(rows)
        self.assertEqual(summary["base"]["expected"], 2)
        self.assertEqual(summary["base"]["failed"], 1)
        self.assertEqual(summary["base"]["metrics"]["score"]["missing_including_failures"], 1)
        self.assertEqual(summary["warp_feather"]["identity_fallback"], 1)
        self.assertEqual(summary["warp_feather"]["paired_delta_vs_base"]["score"]["mean"], -.5)


if __name__ == "__main__":
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    unittest.main()
