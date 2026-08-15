import pytest

from eval_protected_gate import (
    NULL_TEXT,
    PARTIAL_TEXT,
    PASS_TEXT,
    decide_verdict,
    garment_gap_recovery_percent,
)


def test_protected_gate_preregistered_outcomes() -> None:
    assert decide_verdict(
        all_pass=True,
        identity_pass=True,
        garment_mean=0.892,
        a4_garment_mean=0.878,
        partial_threshold=0.8865,
        cloth_energy_below_threshold=False,
    ) == PASS_TEXT
    assert decide_verdict(
        all_pass=False,
        identity_pass=True,
        garment_mean=0.889,
        a4_garment_mean=0.878,
        partial_threshold=0.8865,
        cloth_energy_below_threshold=False,
    ) == PARTIAL_TEXT
    assert decide_verdict(
        all_pass=True,
        identity_pass=True,
        garment_mean=0.892,
        a4_garment_mean=0.878,
        partial_threshold=0.8865,
        cloth_energy_below_threshold=True,
    ) == NULL_TEXT
    assert decide_verdict(
        all_pass=False,
        identity_pass=False,
        garment_mean=0.900,
        a4_garment_mean=0.878,
        partial_threshold=0.8865,
        cloth_energy_below_threshold=False,
    ) == NULL_TEXT


def test_garment_gap_recovery_percent() -> None:
    a4 = 0.877802
    b2cont = 0.895092
    assert garment_gap_recovery_percent((a4 + b2cont) / 2.0, a4, b2cont) == pytest.approx(50.0)
    assert garment_gap_recovery_percent(0.866307, a4, b2cont) == pytest.approx(-66.48, abs=0.02)
