from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SUPPORT_LABELS = ("S_I", "S_M", "S_X", "S_U")
FEATURE_FAMILIES = (
    "identity_morphology",
    "hair_appearance_geometry",
    "global_geometry_pose",
    "garment_appearance",
    "garment_drape",
    "scene",
    "occlusion",
)
PRIMARY_POLICIES = ("main_hair", "face_only")

ATTRIBUTE_SUPPORTS = {
    "face_morphology": frozenset({"S_I"}),
    "reliable_local_hair": frozenset({"S_I"}),
    "unreliable_local_hair": frozenset({"S_U"}),
    "global_head_pose": frozenset({"S_M"}),
    "body_morphology_pose": frozenset({"S_M"}),
    "garment_design": frozenset({"S_M"}),
    "garment_gross_drape": frozenset({"S_M", "S_X"}),
    "background_scene": frozenset({"S_M"}),
    "hair_face_contact": frozenset({"S_X"}),
    "hair_garment_contact": frozenset({"S_X"}),
    "unresolved": frozenset({"S_U"}),
}

FORBIDDEN_CONDITIONING_TOKENS = (
    "output_mask",
    "generated_mask",
    "photographic_target",
    "ground_truth",
    "manual_eval",
    "evaluator_annotation",
    "evaluation_support",
)


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class HairGateThresholds:
    krippendorff_alpha: float = 0.80
    median_boundary_iou: float = 0.75
    occlusion_kappa: float = 0.70
    max_su_fraction: float = 0.10
    min_images_within_su_limit: float = 0.90


def _require_text(row: Mapping[str, Any], field: str) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise ProtocolError(f"missing required field {field!r}")
    return value


def validate_support_separation(metadata: Mapping[str, Any]) -> None:
    conditioning = metadata.get("conditioning_supports")
    evaluation = metadata.get("evaluation_supports")
    if not isinstance(conditioning, Mapping) or not isinstance(evaluation, Mapping):
        raise ProtocolError("conditioning_supports and evaluation_supports are required")

    conditioning_parser = _require_text(conditioning, "parser_name")
    evaluation_parser = _require_text(evaluation, "parser_name")
    if conditioning_parser == evaluation_parser:
        raise ProtocolError("evaluation supports must use an independent parser")
    if not bool(conditioning.get("frozen")) or not bool(evaluation.get("frozen")):
        raise ProtocolError("both support pipelines must be frozen")
    if not _require_text(conditioning, "artifact_sha256") or not _require_text(evaluation, "artifact_sha256"):
        raise ProtocolError("both support artifacts require SHA-256 provenance")

    allowed_inputs = conditioning.get("allowed_inputs")
    if not isinstance(allowed_inputs, Sequence) or isinstance(allowed_inputs, (str, bytes)):
        raise ProtocolError("conditioning allowed_inputs must be a sequence")
    normalized = {str(value).strip() for value in allowed_inputs}
    permitted = {"input_m", "input_i", "frozen_preprocessor"}
    if not normalized or not normalized <= permitted:
        raise ProtocolError(f"conditioning supports may only use {sorted(permitted)}")

    serialized = " ".join(str(value).lower() for value in conditioning.values())
    forbidden = [token for token in FORBIDDEN_CONDITIONING_TOKENS if token in serialized]
    if forbidden:
        raise ProtocolError(f"conditioning supports contain forbidden inputs: {forbidden}")


def validate_ontology_rows(rows: Iterable[Mapping[str, Any]], primary_policy: str) -> None:
    if primary_policy not in PRIMARY_POLICIES:
        raise ProtocolError(f"unknown primary policy {primary_policy!r}")
    seen: set[str] = set()
    observed_policies: set[str] = set()
    count = 0
    for row in rows:
        count += 1
        unit_id = _require_text(row, "unit_id")
        if unit_id in seen:
            raise ProtocolError(f"duplicate unit_id {unit_id!r}")
        seen.add(unit_id)

        family = _require_text(row, "feature_family")
        support = _require_text(row, "support_label")
        attribute = _require_text(row, "attribute")
        row_policy = _require_text(row, "primary_policy")
        observed_policies.add(row_policy)
        if family not in FEATURE_FAMILIES:
            raise ProtocolError(f"unknown feature_family {family!r}")
        if support not in SUPPORT_LABELS:
            raise ProtocolError(f"unit {unit_id!r} has invalid support {support!r}")
        if attribute not in ATTRIBUTE_SUPPORTS:
            raise ProtocolError(f"unit {unit_id!r} has unknown attribute {attribute!r}")
        if support not in ATTRIBUTE_SUPPORTS[attribute]:
            raise ProtocolError(f"unit {unit_id!r}: {attribute!r} cannot use {support!r}")
        if row_policy != primary_policy:
            raise ProtocolError("primary policies cannot be mixed")
        if primary_policy == "face_only" and family == "hair_appearance_geometry" and support == "S_I":
            raise ProtocolError("face-only policy cannot claim hair as identity-owned")
    if count == 0:
        raise ProtocolError("ontology must contain at least one unit")
    if observed_policies != {primary_policy}:
        raise ProtocolError("primary policy is not frozen consistently")


def exclusive_leakage_units(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        _require_text(row, "unit_id")
        for row in rows
        if _require_text(row, "support_label") in {"S_I", "S_M"}
    ]


def nominal_krippendorff_alpha(units: Iterable[Sequence[str]]) -> float | None:
    usable = [list(values) for values in units if len(values) >= 2]
    if not usable:
        return None
    pair_total = 0
    disagreements = 0
    category_counts: dict[str, int] = {}
    total_ratings = 0
    for values in usable:
        for value in values:
            category_counts[value] = category_counts.get(value, 0) + 1
            total_ratings += 1
        for left, right in combinations(values, 2):
            pair_total += 1
            disagreements += int(left != right)
    if pair_total == 0 or total_ratings < 2:
        return None
    observed = disagreements / pair_total
    same_probability = sum(count * (count - 1) for count in category_counts.values())
    same_probability /= total_ratings * (total_ratings - 1)
    expected = 1.0 - same_probability
    if expected <= 1e-12:
        return 1.0 if observed <= 1e-12 else None
    return float(1.0 - observed / expected)


def fleiss_kappa(units: Iterable[Sequence[str]]) -> float | None:
    usable = [list(values) for values in units if len(values) >= 2]
    if not usable:
        return None
    rater_counts = {len(values) for values in usable}
    if len(rater_counts) != 1:
        raise ProtocolError("Fleiss kappa requires the same rater count for every unit")
    n_raters = next(iter(rater_counts))
    categories = sorted({value for values in usable for value in values})
    category_totals = {category: 0 for category in categories}
    agreements = []
    for values in usable:
        counts = {category: values.count(category) for category in categories}
        agreements.append(sum(count * (count - 1) for count in counts.values()) / (n_raters * (n_raters - 1)))
        for category, count in counts.items():
            category_totals[category] += count
    observed = float(np.mean(agreements))
    total = len(usable) * n_raters
    expected = sum((count / total) ** 2 for count in category_totals.values())
    if 1.0 - expected <= 1e-12:
        return 1.0 if observed >= 1.0 - 1e-12 else None
    return float((observed - expected) / (1.0 - expected))


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    if a.shape != b.shape:
        raise ProtocolError(f"boundary mask shape mismatch: {a.shape} != {b.shape}")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def median_pairwise_boundary_iou(boundary_masks_by_unit: Mapping[str, Sequence[np.ndarray]]) -> float | None:
    values: list[float] = []
    for masks in boundary_masks_by_unit.values():
        if len(masks) != 3:
            raise ProtocolError("each boundary unit must have exactly three annotator masks")
        values.extend(mask_iou(left, right) for left, right in combinations(masks, 2))
    return None if not values else float(np.median(values))


def evaluate_hair_gate(
    support_ratings_by_unit: Mapping[str, Sequence[str]],
    boundary_masks_by_unit: Mapping[str, Sequence[np.ndarray]],
    occlusion_ratings_by_unit: Mapping[str, Sequence[str]],
    su_fraction_by_image: Mapping[str, float],
    guideline_revision: int,
    fallback_policy_frozen: bool = False,
    thresholds: HairGateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or HairGateThresholds()
    if guideline_revision < 0:
        raise ProtocolError("guideline_revision must be non-negative")
    if not support_ratings_by_unit or not boundary_masks_by_unit or not occlusion_ratings_by_unit:
        raise ProtocolError("support, boundary, and occlusion annotations are all required")
    if not su_fraction_by_image:
        raise ProtocolError("S_U coverage is required")
    for name, groups in (
        ("support", support_ratings_by_unit),
        ("occlusion", occlusion_ratings_by_unit),
    ):
        invalid = [unit for unit, values in groups.items() if len(values) != 3]
        if invalid:
            raise ProtocolError(f"{name} units require exactly three annotators: {invalid[:5]}")

    alpha = nominal_krippendorff_alpha(support_ratings_by_unit.values())
    boundary_iou = median_pairwise_boundary_iou(boundary_masks_by_unit)
    kappa = fleiss_kappa(occlusion_ratings_by_unit.values())
    fractions = [float(value) for value in su_fraction_by_image.values()]
    if any(value < 0.0 or value > 1.0 for value in fractions):
        raise ProtocolError("S_U fractions must lie in [0, 1]")
    within_limit = float(np.mean([value <= thresholds.max_su_fraction for value in fractions]))

    checks = {
        "krippendorff_alpha": alpha is not None and alpha >= thresholds.krippendorff_alpha,
        "median_boundary_iou": boundary_iou is not None and boundary_iou >= thresholds.median_boundary_iou,
        "occlusion_kappa": kappa is not None and kappa >= thresholds.occlusion_kappa,
        "su_coverage": within_limit >= thresholds.min_images_within_su_limit,
    }
    main_hair_pass = all(checks.values())
    if main_hair_pass:
        decision = "main_hair"
        gate_pass = True
    elif guideline_revision >= 1:
        decision = "face_only"
        gate_pass = bool(fallback_policy_frozen)
    else:
        decision = "revise_once"
        gate_pass = False

    return {
        "gate_pass": gate_pass,
        "main_hair_threshold_pass": main_hair_pass,
        "fallback_policy_frozen": bool(fallback_policy_frozen),
        "decision": decision,
        "guideline_revision": guideline_revision,
        "annotator_count": 3,
        "metrics": {
            "krippendorff_alpha": alpha,
            "median_boundary_iou": boundary_iou,
            "occlusion_kappa": kappa,
            "images_with_su_at_most_limit": within_limit,
            "image_count": len(fractions),
        },
        "thresholds": asdict(thresholds),
        "checks": checks,
    }
