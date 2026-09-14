from __future__ import annotations

import unittest

import numpy as np

from source_ownership_ontology import (
    ProtocolError,
    evaluate_hair_gate,
    exclusive_leakage_units,
    fleiss_kappa,
    mask_iou,
    nominal_krippendorff_alpha,
    validate_ontology_rows,
    validate_support_separation,
)


def ontology_rows(policy: str = "main_hair") -> list[dict[str, str]]:
    hair_attribute = "reliable_local_hair" if policy == "main_hair" else "unreliable_local_hair"
    hair_support = "S_I" if policy == "main_hair" else "S_U"
    specs = (
        ("face", "identity_morphology", "face_morphology", "S_I"),
        ("hair", "hair_appearance_geometry", hair_attribute, hair_support),
        ("head", "global_geometry_pose", "global_head_pose", "S_M"),
        ("body", "global_geometry_pose", "body_morphology_pose", "S_M"),
        ("garment", "garment_appearance", "garment_design", "S_M"),
        ("drape", "garment_drape", "garment_gross_drape", "S_X"),
        ("scene", "scene", "background_scene", "S_M"),
        ("contact", "occlusion", "hair_garment_contact", "S_X"),
        ("unknown", "occlusion", "unresolved", "S_U"),
    )
    return [
        {
            "unit_id": unit,
            "feature_family": family,
            "attribute": attribute,
            "support_label": support,
            "primary_policy": policy,
        }
        for unit, family, attribute, support in specs
    ]


class OntologyValidationTests(unittest.TestCase):
    def test_accepts_disjoint_exhaustive_main_hair_ontology(self) -> None:
        validate_ontology_rows(ontology_rows(), "main_hair")

    def test_accepts_face_only_fallback_without_identity_owned_hair(self) -> None:
        rows = ontology_rows("face_only")
        validate_ontology_rows(rows, "face_only")
        hair = next(row for row in rows if row["feature_family"] == "hair_appearance_geometry")
        self.assertEqual(hair["support_label"], "S_U")

    def test_rejects_wrong_owner_mapping(self) -> None:
        rows = ontology_rows()
        rows[0]["support_label"] = "S_M"
        with self.assertRaises(ProtocolError):
            validate_ontology_rows(rows, "main_hair")

    def test_rejects_duplicate_unit_and_mixed_policy(self) -> None:
        rows = ontology_rows()
        rows[1]["unit_id"] = rows[0]["unit_id"]
        with self.assertRaises(ProtocolError):
            validate_ontology_rows(rows, "main_hair")
        rows = ontology_rows()
        rows[1]["primary_policy"] = "face_only"
        with self.assertRaises(ProtocolError):
            validate_ontology_rows(rows, "main_hair")

    def test_excludes_interaction_and_uncertain_from_exclusive_leakage(self) -> None:
        units = exclusive_leakage_units(ontology_rows())
        self.assertEqual(set(units), {"face", "hair", "head", "body", "garment", "scene"})

    def test_support_pipeline_separation_requires_independent_frozen_parser(self) -> None:
        metadata = {
            "conditioning_supports": {
                "parser_name": "sam_input_v1",
                "frozen": True,
                "artifact_sha256": "a" * 64,
                "allowed_inputs": ["input_m", "input_i", "frozen_preprocessor"],
            },
            "evaluation_supports": {
                "parser_name": "fashn_segformer_b4",
                "frozen": True,
                "artifact_sha256": "b" * 64,
            },
        }
        validate_support_separation(metadata)
        metadata["conditioning_supports"]["allowed_inputs"].append("output_mask")
        with self.assertRaises(ProtocolError):
            validate_support_separation(metadata)


class AgreementTests(unittest.TestCase):
    def test_agreement_helpers(self) -> None:
        perfect = [["S_I"] * 3, ["S_M"] * 3, ["S_X"] * 3]
        self.assertAlmostEqual(nominal_krippendorff_alpha(perfect) or 0.0, 1.0)
        self.assertAlmostEqual(fleiss_kappa(perfect) or 0.0, 1.0)
        mask = np.ones((6, 6), dtype=bool)
        self.assertAlmostEqual(mask_iou(mask, mask), 1.0)

    def test_main_hair_gate_passes_all_frozen_thresholds(self) -> None:
        ratings = {
            "u1": ["S_I", "S_I", "S_I"],
            "u2": ["S_M", "S_M", "S_M"],
            "u3": ["S_X", "S_X", "S_X"],
        }
        mask = np.ones((8, 8), dtype=bool)
        boundaries = {unit: [mask.copy(), mask.copy(), mask.copy()] for unit in ratings}
        occlusion = {
            "o1": ["hair_over_face"] * 3,
            "o2": ["hair_over_garment"] * 3,
            "o3": ["none"] * 3,
        }
        result = evaluate_hair_gate(ratings, boundaries, occlusion, {"a": 0.05, "b": 0.10}, 0)
        self.assertTrue(result["gate_pass"])
        self.assertEqual(result["decision"], "main_hair")

    def test_failed_first_pass_requires_exactly_one_revision(self) -> None:
        ratings = {
            "u1": ["S_I", "S_M", "S_U"],
            "u2": ["S_M", "S_X", "S_U"],
        }
        mask = np.ones((4, 4), dtype=bool)
        empty = np.zeros((4, 4), dtype=bool)
        boundaries = {unit: [mask, empty, mask] for unit in ratings}
        occlusion = {"o1": ["hair_over_face", "none", "ambiguous"]}
        first = evaluate_hair_gate(ratings, boundaries, occlusion, {"a": 0.25}, 0)
        self.assertFalse(first["gate_pass"])
        self.assertEqual(first["decision"], "revise_once")
        revised = evaluate_hair_gate(ratings, boundaries, occlusion, {"a": 0.25}, 1)
        self.assertFalse(revised["gate_pass"])
        self.assertEqual(revised["decision"], "face_only")
        frozen = evaluate_hair_gate(ratings, boundaries, occlusion, {"a": 0.25}, 1, fallback_policy_frozen=True)
        self.assertTrue(frozen["gate_pass"])
        self.assertEqual(frozen["decision"], "face_only")


if __name__ == "__main__":
    unittest.main()
