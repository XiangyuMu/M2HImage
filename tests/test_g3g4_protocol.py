from __future__ import annotations

import unittest

from complete_2x2_blocks import BlockProtocolError, deterministic_replay, intention_to_test_output_status, validate_complete_blocks
from directional_metrics import MetricProtocolError, dose_spearman, identity_direction, interaction_coherence, mannequin_direction
from injected_metric_validation import balanced_accuracy, cohens_kappa, false_positive_rate, reproduce_a4_cloth_leakage, second_parser_direction_preserved


def rows():
    out = []
    for i in (0, 1):
        for j in (0, 1):
            out.append({
                "block_id": "b01", "cell_id": f"I{i}M{j}", "identity_index": i, "mannequin_index": j,
                "identity_id": f"id{i}", "mannequin_id": f"m{j}", "noise_seed": 7, "noise_hash": "a" * 64,
                "split": "dev", "source_dataset": "synthetic_a", "garment_sku": "sku0", "context_id": "ctx0",
                "intervention_type": "garment",
                "input_identity_path": f"/id{i}.jpg", "input_mannequin_path": f"/m{j}.png",
            })
    for row in out:
        row["garment_sku"] = f"sku{row['mannequin_index']}"
    return out


class BlockTests(unittest.TestCase):
    def test_complete_same_noise_block_and_replay(self):
        result = validate_complete_blocks(rows(), expected_split="dev")
        self.assertTrue(result["complete"])
        self.assertTrue(deterministic_replay(rows(), rows()))

    def test_rejects_missing_cell_or_noise_mismatch(self):
        bad = rows()[:-1]
        with self.assertRaises(BlockProtocolError):
            validate_complete_blocks(bad)

    def test_missing_outputs_remain_in_intention_to_test(self):
        manifest = rows()
        status = intention_to_test_output_status(manifest, [{"block_id": "b01", "cell_id": "I0M0", "valid_output": True}])
        self.assertEqual(status["intention_to_test_cells"], 4)
        self.assertEqual(status["missing_cells"], 3)
        self.assertEqual(status["worst_case_cells"], 3)
        bad = rows(); bad[1]["noise_hash"] = "b" * 64
        with self.assertRaises(BlockProtocolError):
            validate_complete_blocks(bad)


class MetricTests(unittest.TestCase):
    def test_directional_owner_response_and_floor(self):
        table = []
        for row in rows():
            i, j = row["identity_index"], row["mannequin_index"]
            table.append(dict(row, q_i1=0.9 if i == 1 else 0.2, q_i0=0.9 if i == 0 else 0.2, q_m=0.8 + 0.01 * j))
        result = identity_direction(table, owner_floor=0.5)
        self.assertTrue(result.owner_floor_pass)
        self.assertGreater(result.direction, 0.5)

    def test_mannequin_stratum_is_directional(self):
        table = []
        for row in rows():
            i, j = row["identity_index"], row["mannequin_index"]
            table.append(dict(row, q_m1=0.9 if j == 1 else 0.2, q_m0=0.9 if j == 0 else 0.2, q_i=0.8 + 0.01 * i))
        result = mannequin_direction(table, owner_floor=0.5)
        self.assertTrue(result.owner_floor_pass)
        self.assertGreater(result.direction, 0.5)

    def test_interaction_zero_denominator_is_undefined(self):
        result = interaction_coherence([True], [0.0])
        self.assertFalse(result["defined"])
        self.assertIsNone(result["interaction_coherence"])
        with self.assertRaises(MetricProtocolError):
            interaction_coherence([True], [-1.0])

    def test_injected_dose_is_monotonic(self):
        doses = [0.0, 0.10, 0.25, 0.50, 1.0]
        responses = [0.01, 0.12, 0.28, 0.53, 0.98]
        self.assertGreaterEqual(dose_spearman(doses, responses), 0.90)

    def test_classification_and_secondary_parser_checks(self):
        expected = [False, False, True, True]
        predicted = [False, False, True, True]
        self.assertAlmostEqual(balanced_accuracy(predicted, expected), 1.0)
        self.assertAlmostEqual(cohens_kappa(predicted, expected), 1.0)
        self.assertEqual(false_positive_rate([False, False, True, True], [False, False, True, True]), 0.0)
        self.assertTrue(second_parser_direction_preserved(0.4, 0.2))
        self.assertFalse(second_parser_direction_preserved(0.4, -0.2))
        self.assertTrue(reproduce_a4_cloth_leakage(0.15)["pass"])


if __name__ == "__main__":
    unittest.main()
