from .module_factory import ModuleFactory
from .skip_connection import SkipConnection

from dataclasses import dataclass
from typing import Optional
from attrs import define, frozen

import torch
from torch import nn, jit


@define(kw_only=True, eq=False, repr=False, slots=False)
class LinearLayer(nn.Module):

    @dataclass(kw_only=True)
    class Params:
        in_dim: int
        out_dim: int

    @dataclass(kw_only=True)
    class Config:
        bn: Optional[str] = "batch_norm"
        act: Optional[str] = "leaky_relu"
        dp: float = 0.0
        residual: bool = False

    params: Params
    config: Config = Config()

    def __attrs_post_init__(self):
        super().__init__()

        # Define flags
        self.with_bn: bool = self.config.bn is not None
        self.with_act: bool = self.config.act is not None
        self.with_dp: bool = self.config.dp > 0.0
        self.residual: bool = self.config.residual

        # Define nn modules
        self.dense: nn.Module = nn.Linear(self.params.in_dim, self.params.out_dim)
        self.bn: nn.Module = ModuleFactory(self.config.bn)(self.params.out_dim)
        self.act: nn.Module = ModuleFactory(self.config.act)()
        self.dp: nn.Module = nn.Dropout(self.config.dp)
        self.skip: SkipConnection = SkipConnection(in_dim=self.params.in_dim, out_dim=self.params.out_dim)

    def forward(self, feat: torch.FloatTensor) -> torch.FloatTensor:
        inputs = outputs = feat
        outputs = self.dense(outputs)
        if self.with_bn:
            outputs = self.bn(outputs)
        if self.with_act:
            outputs = self.act(outputs)
        if self.with_dp:
            outputs = self.dp(outputs)
        if self.residual:
            outputs = self.skip(block_input=inputs, block_output=outputs)
        return outputs


LinearLayer.__annotations__.clear()
