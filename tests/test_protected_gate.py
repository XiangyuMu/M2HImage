from eval_protected_gate import NULL_TEXT, PARTIAL_TEXT, PASS_TEXT, decide_verdict


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
