from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from directional_metrics import MetricProtocolError, dose_spearman


INJECTION_DOSES = (0.0, 0.10, 0.25, 0.50, 1.00)


def inject_wrong_source_response(owner_score: float, wrong_source_score: float, dose: float) -> float:
    if not 0.0 <= dose <= 1.0:
        raise MetricProtocolError("injection dose must lie in [0, 1]")
    return float((1.0 - dose) * owner_score + dose * wrong_source_score)


def validate_dose_response(doses: Sequence[float], responses: Sequence[float], minimum_rho: float = 0.90) -> dict[str, Any]:
    rho = dose_spearman(doses, responses)
    return {"spearman_rho": rho, "minimum_rho": minimum_rho, "pass": bool(rho >= minimum_rho)}


def false_positive_rate(predicted_leakage: Sequence[bool], legal_interaction: Sequence[bool]) -> float:
    if len(predicted_leakage) != len(legal_interaction) or not legal_interaction:
        raise MetricProtocolError("false-positive inputs are empty or misaligned")
    negatives = [not bool(value) for value in legal_interaction]
    if not any(negatives):
        return 0.0
    false_positives = sum(bool(pred) and neg for pred, neg in zip(predicted_leakage, negatives))
    return float(false_positives / sum(negatives))


def balanced_accuracy(predicted: Sequence[bool], expected: Sequence[bool]) -> float:
    if len(predicted) != len(expected) or not expected:
        raise MetricProtocolError("classification inputs are empty or misaligned")
    predicted = [bool(value) for value in predicted]
    expected = [bool(value) for value in expected]
    positives = [index for index, value in enumerate(expected) if value]
    negatives = [index for index, value in enumerate(expected) if not value]
    if not positives or not negatives:
        raise MetricProtocolError("balanced accuracy requires both classes")
    tpr = sum(predicted[index] for index in positives) / len(positives)
    tnr = sum(not predicted[index] for index in negatives) / len(negatives)
    return float((tpr + tnr) / 2.0)


def cohens_kappa(predicted: Sequence[bool], expected: Sequence[bool]) -> float:
    if len(predicted) != len(expected) or not expected:
        raise MetricProtocolError("classification inputs are empty or misaligned")
    p_o = np.mean([bool(a) == bool(b) for a, b in zip(predicted, expected)])
    p_pred = np.mean([bool(value) for value in predicted])
    p_exp = np.mean([bool(value) for value in expected])
    p_e = p_pred * p_exp + (1.0 - p_pred) * (1.0 - p_exp)
    if 1.0 - p_e <= 1e-12:
        return 1.0 if p_o >= 1.0 - 1e-12 else 0.0
    return float((p_o - p_e) / (1.0 - p_e))


def second_parser_direction_preserved(primary_direction: float, secondary_direction: float) -> bool:
    if not np.isfinite(primary_direction) or not np.isfinite(secondary_direction):
        raise MetricProtocolError("parser directions must be finite")
    return bool(np.sign(primary_direction) == np.sign(secondary_direction) and np.sign(primary_direction) != 0)


def reproduce_a4_cloth_leakage(measured_fraction: float, target_fraction: float = 0.14, tolerance: float = 0.02) -> dict[str, Any]:
    if not 0.0 <= measured_fraction <= 1.0:
        raise MetricProtocolError("cloth leakage fraction must lie in [0, 1]")
    error = abs(float(measured_fraction) - target_fraction)
    return {"measured_fraction": float(measured_fraction), "target_fraction": target_fraction, "tolerance": tolerance, "absolute_error": error, "pass": bool(error <= tolerance)}
