from __future__ import annotations


class GatedConditionAttnProcessor:
    """Adds zero-initialized mannequin and identity cross-attention residuals."""

    def __new__(
        cls,
        hidden_size: int,
        cross_attention_dim: int,
        mannequin_tokens: int,
        identity_tokens: int,
    ):
        import torch
        from torch import nn

        class Processor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mannequin_tokens = mannequin_tokens
                self.identity_tokens = identity_tokens
                self.to_k_mannequin = nn.Linear(cross_attention_dim, hidden_size, bias=False)
                self.to_v_mannequin = nn.Linear(cross_attention_dim, hidden_size, bias=False)
                self.to_k_identity = nn.Linear(cross_attention_dim, hidden_size, bias=False)
                self.to_v_identity = nn.Linear(cross_attention_dim, hidden_size, bias=False)
                self.mannequin_gate = nn.Parameter(torch.zeros(()))
                self.identity_gate = nn.Parameter(torch.zeros(()))

            @staticmethod
            def _attend(attn, query, key, value, attention_mask=None):
                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)
                probabilities = attn.get_attention_scores(query, key, attention_mask)
                hidden_states = torch.bmm(probabilities, value)
                return attn.batch_to_head_dim(hidden_states)

            def forward(
                self,
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
                *args,
                **kwargs,
            ):
                residual = hidden_states
                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)
                input_ndim = hidden_states.ndim
                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
                text_end = encoder_hidden_states.shape[1] - self.mannequin_tokens - self.identity_tokens
                text = encoder_hidden_states[:, :text_end]
                mannequin = encoder_hidden_states[:, text_end : text_end + self.mannequin_tokens]
                identity = encoder_hidden_states[:, text_end + self.mannequin_tokens :]
                if attn.norm_cross:
                    text = attn.norm_encoder_hidden_states(text)
                    mannequin = attn.norm_encoder_hidden_states(mannequin)
                    identity = attn.norm_encoder_hidden_states(identity)
                batch_size = hidden_states.shape[0]
                attention_mask = attn.prepare_attention_mask(attention_mask, text.shape[1], batch_size)
                query = attn.to_q(hidden_states)
                base = self._attend(attn, query, attn.to_k(text), attn.to_v(text), attention_mask)
                man = self._attend(
                    attn,
                    query,
                    self.to_k_mannequin(mannequin),
                    self.to_v_mannequin(mannequin),
                )
                identity_out = self._attend(
                    attn,
                    query,
                    self.to_k_identity(identity),
                    self.to_v_identity(identity),
                )
                hidden_states = base + self.mannequin_gate * man + self.identity_gate * identity_out
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
                if attn.residual_connection:
                    hidden_states = hidden_states + residual
                return hidden_states / attn.rescale_output_factor

        return Processor()
