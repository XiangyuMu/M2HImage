from eval_a4_gate_report import decide_verdict


def test_a4_gate_verdicts_are_preregistered() -> None:
    assert decide_verdict(True, True, True, True, True, True).startswith('PASS:')
    assert decide_verdict(True, True, False, True, True, True).startswith('FAIL:')
    assert decide_verdict(True, True, True, False, True, True).startswith('MIXED:')
    assert decide_verdict(False, True, True, True, True, True).startswith('BLOCKED:')
    assert decide_verdict(True, False, True, True, True, True).startswith('BLOCKED:')
