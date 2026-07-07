from __future__ import annotations


def _linear_tokens(input_dim: int, output_dim: int, token_count: int):
    from torch import nn

    return nn.Sequential(
        nn.Linear(input_dim, output_dim * token_count),
        nn.LayerNorm(output_dim * token_count),
    )


class IdentityProjector:
    def __new__(cls, input_dim: int, output_dim: int, token_count: int):
        from torch import nn

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.project = _linear_tokens(input_dim, output_dim, token_count)
                self.token_count = token_count
                self.output_dim = output_dim

            def forward(self, embeddings):
                return self.project(embeddings).view(-1, self.token_count, self.output_dim)

        return Module()


class MannequinProjector:
    def __new__(cls, input_dim: int, output_dim: int, token_count: int):
        from torch import nn

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = nn.AdaptiveAvgPool1d(token_count)
                self.project = nn.Linear(input_dim, output_dim)

            def forward(self, patch_tokens):
                tokens = self.pool(patch_tokens.transpose(1, 2)).transpose(1, 2)
                return self.project(tokens)

        return Module()
