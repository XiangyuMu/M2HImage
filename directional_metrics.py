from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class MetricProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class DirectionalResult:
    direction: float
    unchanged_owner_delta: float
    owner_floor: float
    owner_floor_pass: bool
    relative_leakage: float | None
    denominator: float | None
    count: int


def _q(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise MetricProtocolError(f"non-finite metric {key}")
    return value


def _rows_by_cell(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[int, int], Mapping[str, Any]]:
    result: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["identity_index"]), int(row["mannequin_index"]))
        if key in result:
            raise MetricProtocolError(f"duplicate cell {key}")
        result[key] = row
    if set(result) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise MetricProtocolError("directional metrics require a complete 2x2 block")
    return result


def identity_direction(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric_for_new_identity: str = "q_i1",
    metric_for_old_identity: str = "q_i0",
    unchanged_mannequin_metric: str = "q_m",
    owner_floor: float = 0.0,
    eta: float = 1e-8,
) -> DirectionalResult:
    cells = _rows_by_cell(rows)
    direction = np.mean([
        (_q(cells[(1, j)], metric_for_new_identity) - _q(cells[(1, j)], metric_for_old_identity))
        - (_q(cells[(0, j)], metric_for_new_identity) - _q(cells[(0, j)], metric_for_old_identity))
        for j in (0, 1)
    ])
    unchanged = np.mean([_q(cells[(1, j)], unchanged_mannequin_metric) - _q(cells[(0, j)], unchanged_mannequin_metric) for j in (0, 1)])
    return _result(direction, unchanged, owner_floor, eta)


def mannequin_direction(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric_for_new_mannequin: str = "q_m1",
    metric_for_old_mannequin: str = "q_m0",
    unchanged_identity_metric: str = "q_i",
    owner_floor: float = 0.0,
    eta: float = 1e-8,
) -> DirectionalResult:
    cells = _rows_by_cell(rows)
    direction = np.mean([
        (_q(cells[(i, 1)], metric_for_new_mannequin) - _q(cells[(i, 1)], metric_for_old_mannequin))
        - (_q(cells[(i, 0)], metric_for_new_mannequin) - _q(cells[(i, 0)], metric_for_old_mannequin))
        for i in (0, 1)
    ])
    unchanged = np.mean([_q(cells[(i, 1)], unchanged_identity_metric) - _q(cells[(i, 0)], unchanged_identity_metric) for i in (0, 1)])
    return _result(direction, unchanged, owner_floor, eta)


def _result(direction: float, unchanged: float, floor: float, eta: float) -> DirectionalResult:
    if eta <= 0:
        raise MetricProtocolError("eta must be positive")
    passed = bool(direction > floor)
    relative = None if not passed else abs(unchanged) / (direction + eta)
    return DirectionalResult(float(direction), float(unchanged), float(floor), passed, relative, None if not passed else float(direction + eta), 4)


def interaction_coherence(correct: Sequence[int | bool], weights: Sequence[float], expected_defined: Sequence[bool] | None = None) -> dict[str, Any]:
    if len(correct) != len(weights):
        raise MetricProtocolError("correct and weights must have equal length")
    if expected_defined is None:
        expected_defined = [True] * len(correct)
    if len(expected_defined) != len(correct):
        raise MetricProtocolError("expected_defined length mismatch")
    eligible = [(bool(value), float(weight)) for value, weight, defined in zip(correct, weights, expected_defined) if defined]
    if any(weight < 0 for _, weight in eligible):
        raise MetricProtocolError("interaction weights must be non-negative")
    denominator = sum(weight for _, weight in eligible)
    if denominator <= 0:
        return {"defined": False, "interaction_coherence": None, "eligible_units": 0, "excluded_undefined_direction": sum(not flag for flag in expected_defined)}
    value = sum(weight for ok, weight in eligible if ok) / denominator
    return {"defined": True, "interaction_coherence": float(value), "eligible_units": len(eligible), "excluded_undefined_direction": sum(not flag for flag in expected_defined)}


def dose_spearman(doses: Sequence[float], responses: Sequence[float]) -> float:
    if len(doses) != len(responses) or len(doses) < 2:
        raise MetricProtocolError("dose and response lengths are invalid")
    x = np.asarray(doses, dtype=float)
    y = np.asarray(responses, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise MetricProtocolError("dose/response contains non-finite values")
    return _pearson(_rank(x), _rank(y))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        indices = np.flatnonzero(values == value)
        ranks[indices] = np.mean(ranks[indices])
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denominator <= 1e-12:
        raise MetricProtocolError("Spearman correlation undefined for a constant vector")
    return float(np.sum(x * y) / denominator)
