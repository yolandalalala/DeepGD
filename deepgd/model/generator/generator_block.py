from deepgd.constants import EPS
from ..common import SkipConnection, EdgeFeatureExpansion
from .generator_layer import GeneratorLayer
from .generator_feature_router import GeneratorFeatureRouter

from attrs import define, frozen
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from functools import cached_property

import torch
from torch import nn, jit, FloatTensor, LongTensor

import torch_geometric as pyg


@define(kw_only=True, eq=False, repr=False, slots=False)
class GeneratorBlock(nn.Module):

    # TODO: separate into Params and Config
    @dataclass(kw_only=True)
    class Config:
        in_dim: int
        out_dim: int
        edge_attr_dim: int
        node_attr_dim: int
        residual: bool
        hidden_dims: list[int]  # TODO: use width and depth instead
        dynamic_edge_feat_mode: Optional[str]
        """dynamic_edge_feat_mode: {
            "simple": block input to first layer, 
            "jump": block input to each layer,
            "chain": every layer output to next layer,
            "raw_simple": raw attribute to first layer,
            "raw_jump": raw attribute to each layer,
            None: no relative edge features
        }
        """

    start_layer_index: int
    config: Config
    edge_net_config: GeneratorLayer.EdgeNetConfig
    gnn_config: GeneratorLayer.GNNConfig
    edge_feat_expansion: EdgeFeatureExpansion.Expansions = EdgeFeatureExpansion.Expansions()
    eps: float = EPS

    def __attrs_post_init__(self):
        super().__init__()

        self.residual = self.config.residual

        self.skip: SkipConnection = SkipConnection(
            in_dim=self.config.in_dim,
            out_dim=self.config.out_dim
        )

        self.first_layer_feature_router: GeneratorFeatureRouter = self._make_feature_router(
            input_source=self.first_layer_input_source
        )

        self.rest_layers_feature_router: GeneratorFeatureRouter = self._make_feature_router(
            input_source=self.rest_layers_input_source
        )

        dims = [self.config.in_dim] + self.config.hidden_dims + [self.config.out_dim]

        self.first_layer: GeneratorLayer = self._make_layer(
            index=self.start_layer_index,
            in_dim=dims[0],
            out_dim=dims[1],
            router=self.first_layer_feature_router
        )

        self.rest_layers: nn.ModuleList[GeneratorLayer] = nn.ModuleList()
        for layer_index, (in_dim, out_dim) in enumerate(zip(dims[1:-1], dims[2:]), start=self.start_layer_index + 1):
            self.rest_layers.append(self._make_layer(
                index= layer_index,
                in_dim=in_dim,
                out_dim=out_dim,
                router=self.rest_layers_feature_router
            ))

    @cached_property
    def first_layer_input_source(self) -> Optional[str]:
        return defaultdict(
            lambda: "null",
            simple="block",
            jump="block",
            chain="block",
            raw_simple="raw",
            raw_jump="raw",
        )[self.config.dynamic_edge_feat_mode]

    @cached_property
    def rest_layers_input_source(self) -> Optional[str]:
        return defaultdict(
            lambda: "null",
            simple="null",
            jump="block",
            chain=None,
            raw_simple="null",
            raw_jump="raw",
        )[self.config.dynamic_edge_feat_mode]

    def _make_feature_router(self, *, input_source: Optional[str]) -> GeneratorFeatureRouter:
        return GeneratorFeatureRouter(
            config=GeneratorFeatureRouter.Config(
                input_source=input_source,
                block_input_dim=self.config.in_dim,
                raw_input_dim=self.config.node_attr_dim,
                edge_attr_dim=self.config.edge_attr_dim
            ),
            edge_feat_expansion=self.edge_feat_expansion,
            eps=self.eps
        )

    def _make_layer(self, *,
                    index: int,
                    in_dim: int,
                    out_dim: int,
                    router: GeneratorFeatureRouter) -> GeneratorLayer:
        return GeneratorLayer(
            layer_index=index,
            config=GeneratorLayer.Config(
                in_dim=in_dim,
                out_dim=out_dim,
                node_feat_dim=in_dim,
                edge_feat_dim=(router.get_output_channels()
                               if router.input_source else self.config.edge_attr_dim)
            ),
            edge_net_config=self.edge_net_config,
            gnn_config=self.gnn_config,
            edge_feat_expansion=(EdgeFeatureExpansion.Expansions() if router.input_source
                                 else self.edge_feat_expansion),
            eps=self.eps
        )

    def forward(self, *,
                node_feat: FloatTensor,
                node_attr: FloatTensor,
                edge_index: LongTensor,
                edge_attr: FloatTensor,
                batch_index: LongTensor,
                num_sampled_nodes_per_hop: list[int],
                num_sampled_edges_per_hop: list[int]) -> tuple[FloatTensor, FloatTensor, LongTensor, FloatTensor]:

        init_edge_index = edge_index

        first_layer_edge_feat = self.first_layer_feature_router(
            block_input=node_feat,
            raw_input=node_attr,
            edge_attr=edge_attr,
            edge_index=init_edge_index
        )

        # _, _, rest_layers_edge_feat = pyg.utils.trim_to_layer(
        #     layer=self.start_layer_index,
        #     x=node_feat,
        #     edge_index=init_edge_index,
        #     edge_attr=self.rest_layers_feature_router(
        #         block_input=node_feat,
        #         raw_input=node_attr,
        #         edge_attr=edge_attr,
        #         edge_index=init_edge_index
        #     ),
        #     num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
        #     num_sampled_edges_per_hop=num_sampled_edges_per_hop
        # )

        rest_layers_edge_feat = self.rest_layers_feature_router(
            block_input=node_feat,
            raw_input=node_attr,
            edge_attr=edge_attr,
            edge_index=init_edge_index
        )

        outputs, edge_index, _ = self.first_layer(
            node_feat=node_feat,
            edge_feat=first_layer_edge_feat,
            edge_index=init_edge_index,
            batch_index=batch_index,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop
        )

        for layer in self.rest_layers:
            outputs, edge_index, rest_layers_edge_feat = layer(
                node_feat=outputs,
                edge_feat=rest_layers_edge_feat,
                edge_index=edge_index,
                batch_index=batch_index,
                num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
                num_sampled_edges_per_hop=num_sampled_edges_per_hop
            )

        edge_index = init_edge_index
        # for layer in range(self.start_layer_index, self.next_layer_index):
            # node_feat, _, _ = pyg.utils.trim_to_layer(
            #     layer=layer,
            #     x=node_feat,
            #     edge_index=edge_index,
            #     num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            #     num_sampled_edges_per_hop=num_sampled_edges_per_hop
            # )
            # node_attr, edge_index, edge_attr = pyg.utils.trim_to_layer(
            #     layer=layer,
            #     x=node_attr,
            #     edge_index=edge_index,
            #     edge_attr=edge_attr,
            #     num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            #     num_sampled_edges_per_hop=num_sampled_edges_per_hop
            # )

        if self.residual:
            outputs = self.skip(
                block_input=node_feat,
                block_output=outputs
            )
        return outputs, node_attr, edge_index, edge_attr

    def __len__(self):
        return 1 + len(self.rest_layers)

    @property
    def next_layer_index(self):
        return self.start_layer_index + len(self)


GeneratorBlock.__annotations__.clear()
