from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from pulid_flux import PuLIDFluxAdapter


class FakeCA(nn.Module):
    def forward(self, identity, hidden):
        return identity.mean() * hidden


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        return encoder_hidden_states, hidden_states * self.weight


def make_adapter() -> PuLIDFluxAdapter:
    adapter = PuLIDFluxAdapter.__new__(PuLIDFluxAdapter)
    nn.Module.__init__(adapter)
    adapter.pulid_ca = nn.ModuleList([FakeCA()])
    adapter.id_weight = 1.0
    adapter._context_id = None
    adapter._context_weight = 1.0
    adapter._original_forwards = []
    adapter._debug_hook_calls = 0
    adapter._debug_delta_norm = 0.0
    return adapter


def test_explicit_context_survives_checkpoint_recompute() -> None:
    adapter = make_adapter()
    block = FakeBlock()
    adapter._wrap_block_forward(block, 0)
    hidden = torch.tensor([[[2.0]]], requires_grad=True)
    encoder = torch.zeros_like(hidden)
    temb = torch.zeros(1, 1)
    id_a = torch.ones(1, 1, 1)
    id_b = torch.full((1, 1, 1), 3.0)
    adapter.set_context(id_a, 1.0)
    joint = adapter.context_kwargs()

    def run(h, e, t, context):
        return block(h, e, t, None, context)[1]

    output = checkpoint(run, hidden, encoder, temb, joint, use_reentrant=False)
    adapter.set_context(id_b, 1.0)
    output.sum().backward()
    torch.testing.assert_close(block.weight.grad, torch.tensor(4.0))
