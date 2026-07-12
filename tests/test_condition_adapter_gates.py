from __future__ import annotations

import torch

from conditions import FluxConditionAdapter


def make_adapter() -> FluxConditionAdapter:
    return FluxConditionAdapter({
        'appearance_dim': 8,
        'garment_grid_dim': 8,
        'garment_grid_max_tokens': 4,
        'token_dim': 16,
        'appearance_tokens': 2,
        'pose_tokens': 1,
        'gate_init': 0.1,
    })


def test_condition_gates_are_post_norm_and_trainable() -> None:
    torch.manual_seed(7)
    adapter = make_adapter().to_compute(device=torch.device('cpu'), dtype=torch.bfloat16)
    assert adapter.appearance_gate.dtype == torch.float32
    assert adapter.garment_gate.dtype == torch.float32
    assert adapter.pose_gate.dtype == torch.float32

    appearance = torch.randn(2, 8, dtype=torch.bfloat16)
    garment = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    head_pose = torch.randn(2, 7, dtype=torch.bfloat16)

    out_01 = adapter(appearance, garment, head_pose)
    with torch.no_grad():
        adapter.appearance_gate.fill_(0.2)
        adapter.garment_gate.fill_(0.2)
        adapter.pose_gate.fill_(0.2)
    out_02 = adapter(appearance, garment, head_pose)
    torch.testing.assert_close(out_02.float(), out_01.float() * 2.0, rtol=0.03, atol=0.01)

    adapter.zero_grad(set_to_none=True)
    weights = torch.randn_like(out_02.float())
    (out_02.float() * weights).sum().backward()
    for name in ('appearance_gate', 'garment_gate', 'pose_gate'):
        grad = getattr(adapter, name).grad
        assert grad is not None
        assert torch.isfinite(grad)
        assert grad.abs().item() > 1e-6
