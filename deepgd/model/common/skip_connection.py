import torch
from torch import nn


class SkipConnection(nn.Module):

    def __init__(self, *,
                 in_dim: int,
                 out_dim: int):
        super().__init__()

        self.same_dim: bool = in_dim == out_dim
        self.proj: nn.Module = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, *,
                block_input: torch.FloatTensor,
                block_output: torch.FloatTensor) -> torch.FloatTensor:
        if self.same_dim:
            return block_input + block_output
        else:
            return self.proj(block_input) + block_output
