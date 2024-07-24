from deepgd.constants import EPS
from ..common import NNConvLayer, NNConvBasicLayer, EdgeFeatureExpansion

from typing import Optional
from attrs import define, frozen
from dataclasses import dataclass

import torch
from torch import nn, jit, FloatTensor, LongTensor


@define(kw_only=True, eq=False, repr=False, slots=False)
class GeneratorLayer(nn.Module):

    @dataclass(kw_only=True)
    class Config:
        in_dim: int
        out_dim: int
        node_feat_dim: int
        edge_feat_dim: int

    @dataclass(kw_only=True)
    class EdgeNetConfig:
        width: int = 0
        depth: int = 0
        hidden_act: str = "leaky_relu"
        out_act: Optional[str] = "tanh"
        bn: Optional[str] = "batch_norm"
        dp: float = 0.0
        residual: bool = False

    @dataclass(kw_only=True)
    class GNNConfig:
        aggr: str = "mean"
        root_weight: bool = True
        dense: bool = False
        bn: Optional[str] = "batch_norm"
        act: Optional[str] = "leaky_relu"
        dp: float = 0.0

    layer_index: int
    config: Config
    edge_net_config: EdgeNetConfig = EdgeNetConfig()
    gnn_config: GNNConfig = GNNConfig()
    edge_feat_expansion: EdgeFeatureExpansion.Expansions = EdgeFeatureExpansion.Expansions(),
    eps: float = EPS

    def __attrs_post_init__(self):
        super().__init__()

        self.edge_feat_provider: EdgeFeatureExpansion = EdgeFeatureExpansion(
            config=EdgeFeatureExpansion.Config(
                node_feat_dim=self.config.node_feat_dim,
                edge_attr_dim=self.config.edge_feat_dim
            ),
            expansions=self.edge_feat_expansion,
            eps=self.eps
        )

        self.gnn_layer: NNConvLayer = NNConvLayer(
            layer_index=self.layer_index,
            params=NNConvBasicLayer.Params(
                in_dim=self.config.in_dim,
                out_dim=self.config.out_dim,
                edge_feat_dim=self.edge_feat_provider.get_feature_channels()
            ),
            nnconv_config=NNConvLayer.NNConvConfig(
                dense=self.gnn_config.dense,
                bn=self.gnn_config.bn,
                act=self.gnn_config.act,
                dp=self.gnn_config.dp,
                residual=False,
                aggr=self.gnn_config.aggr,
                root_weight=self.gnn_config.root_weight
            ),
            edge_net_config=NNConvLayer.EdgeNetConfig(
                hidden_dims=[self.edge_net_config.width] * self.edge_net_config.depth,
                hidden_act=self.edge_net_config.hidden_act,
                out_act=self.edge_net_config.out_act,
                bn=self.edge_net_config.bn,
                dp=self.edge_net_config.dp,
                residual=self.edge_net_config.residual
            )
        )

    def forward(self, *,
                node_feat: FloatTensor,
                edge_feat: FloatTensor,
                edge_index: LongTensor,
                batch_index: LongTensor,
                num_sampled_nodes_per_hop: list[int],
                num_sampled_edges_per_hop: list[int]) -> tuple[FloatTensor, LongTensor, FloatTensor]:
        return self.gnn_layer(
            node_feat=node_feat,
            edge_feat=self.edge_feat_provider(
                node_feat=node_feat,
                edge_index=edge_index,
                edge_attr=edge_feat
            ),
            edge_index=edge_index,
            batch_index=batch_index,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop
        )


GeneratorLayer.__annotations__.clear()
