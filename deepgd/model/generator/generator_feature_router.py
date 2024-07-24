from deepgd.constants import EPS
from ..common import EdgeFeatureExpansion

from attrs import define, frozen
from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn, jit


@define(kw_only=True, eq=False, repr=False, slots=False)
class GeneratorFeatureRouter(nn.Module):

    @dataclass(kw_only=True)
    class Config:
        block_input_dim: int
        raw_input_dim: int
        edge_attr_dim: int
        input_source: Optional[str]

    config: Config
    edge_feat_expansion: EdgeFeatureExpansion.Expansions = EdgeFeatureExpansion.Expansions()
    eps: float = EPS

    def __attrs_post_init__(self):
        super().__init__()

        self.input_source: Optional[str] = self.config.input_source
        self.edge_feature_provider: EdgeFeatureExpansion = EdgeFeatureExpansion(
            config=EdgeFeatureExpansion.Config(
                node_feat_dim=dict(
                    block=self.config.block_input_dim,
                    raw=self.config.raw_input_dim,
                    null=0
                ).get(self.config.input_source, None),
                edge_attr_dim=self.config.edge_attr_dim
            ),
            expansions=dict(
                block=self.edge_feat_expansion,
                raw=self.edge_feat_expansion,
                null=EdgeFeatureExpansion.Expansions()
            ).get(self.config.input_source, EdgeFeatureExpansion.Expansions()),
            eps=self.eps
        )

    def get_output_channels(self):
        return self.edge_feature_provider.get_feature_channels()

    def forward(self, *,
                block_input: torch.FloatTensor,
                raw_input: torch.FloatTensor,
                edge_attr: torch.FloatTensor,
                edge_index: torch.LongTensor):
        if self.input_source is None:
            return edge_attr
        null_input = torch.zeros(len(block_input), 0).to(block_input)
        return self.edge_feature_provider(
            node_feat=(
                block_input if self.input_source == "block" else
                raw_input if self.input_source == "raw" else
                null_input if self.input_source == "null" else
                null_input
            ),
            edge_attr=edge_attr,
            edge_index=edge_index
        )


GeneratorFeatureRouter.__annotations__.clear()
