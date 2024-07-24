from .transforms import (
    NormalizeGraph,
    AddAdjacencyInfo,
    ComputeShortestPath,
    Delaunay,
    GenerateEdgePairs,
    GeneratePermutationEdges,
    GenerateRandomLayout,
    PopulateGraphAttrs,
    SampleAggregationEdges,
    GabrielGraph,
    RandomNeighborhoodGraph,
)
from .base_data import BaseData

from typing import Any, Optional, Union, Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.typing import OptTensor
import torch_geometric.transforms as T
from typing_extensions import Self
import networkx as nx


class BaseGraphDrawingData(BaseData):

    @dataclass
    class Field:
        stage: str
        transform: T.BaseTransform
        optional: bool = False

    # Initialization  # TODO: move to BaseData
    G:                         nx.Graph = Field(stage="init", transform=NormalizeGraph())
    perm_index:                Tensor = Field(stage="init",
                                              transform=GeneratePermutationEdges(attr_name="perm_index"))

    # pre_transform (Storage Footprint -- Saved to disk)
    edge_metaindex:            Tensor = Field(stage="pre_transform",
                                              transform=AddAdjacencyInfo(attr_name="edge_metaindex"))
    apsp_attr:                 Tensor = Field(stage="pre_transform",
                                              transform=(compute_shortest_path := ComputeShortestPath(
                                                  cutoff=None,
                                                  attr_name="apsp_attr",
                                                  weight_name="perm_weight"
                                              )))
    perm_weight:               Tensor = Field(stage="pre_transform",
                                              transform=compute_shortest_path)

    # static_transform (Memory Footprint -- Generated everytime when dataset_name being loaded to memory)
    name:                      str = Field(stage="static_transform",
                                           transform=(populate_graph_attrs := PopulateGraphAttrs()))
    dataset:                   str = Field(stage="static_transform", transform=populate_graph_attrs)
    n:                         Tensor = Field(stage="static_transform", transform=populate_graph_attrs)
    m:                         Tensor = Field(stage="static_transform", transform=populate_graph_attrs)
    laplacian_eigenvector_pe:  OptTensor = Field(stage="static_transform",
                                                 transform=T.AddLaplacianEigenvectorPE(
                                                     k=3,
                                                     is_undirected=True,
                                                     attr_name="laplacian_eigenvector_pe"
                                                 ),
                                                 optional=True)
    edge_pair_metaindex:       OptTensor = Field(stage="static_transform",
                                                 transform=GenerateEdgePairs(
                                                     attr_name="edge_pair_metaindex"
                                                 ),
                                                 optional=True)

    # dynamic_transform (Memory/CPU Footprint -- Generated everytime when batch is sampled from the dataset_name)
    aggr_metaindex:            Tensor = Field(stage="dynamic_transform",
                                              transform=SampleAggregationEdges(attr_name="aggr_metaindex"))
    pos:                       OptTensor = Field(stage="dynamic_transform", transform=GenerateRandomLayout())

    # post_transform (Memory/CPU Footprint -- Generated everytime when needed)
    # TODO: raise exception if field is None, then catch the exception and perform corresponding transform
    face:                      OptTensor = Field(stage="post_transform",
                                                 transform=Delaunay(),
                                                 optional=True)
    gabriel_index:             OptTensor = Field(stage="post_transform",
                                                 transform=GabrielGraph(attr_name="gabriel_index"),
                                                 optional=True)
    rng_index:                 OptTensor = Field(stage="post_transform",
                                                 transform=RandomNeighborhoodGraph(attr_name="rng_index"),
                                                 optional=True)

    # Dynamic (CPU/GPU Footprint -- Generated on the fly when being accessed)
    # ------------------------------------ pre_transform
    x:                         OptTensor
    perm_attr:                 OptTensor
    edge_index:                Tensor
    edge_attr:                 Tensor
    edge_weight:               Tensor
    edge_pair_index:           Tensor
    # -------------------------------- dynamic_transform
    aggr_index:                Tensor
    aggr_attr:                 Tensor
    aggr_weight:               Tensor

    @property
    def x(self) -> OptTensor:
        try:
            assert self.laplacian_eigenvector_pe is not None
        except AssertionError:
            return None
        return torch.cat([
            self.laplacian_eigenvector_pe,
        ], dim=1).float()

    @property
    def perm_attr(self) -> OptTensor:
        try:
            assert self.apsp_attr is not None
        except AssertionError:
            return None
        return torch.cat([self.apsp_attr.unsqueeze(1)], dim=1).float()

    @property
    def edge_index(self) -> Tensor:
        return NotImplemented  # override as NotImplemented to force dynamic generation

    @property
    def edge_attr(self) -> Tensor:
        return NotImplemented

    @property
    def edge_weight(self) -> Tensor:
        return NotImplemented

    @property
    def device(self) -> torch.device:
        return self.perm_index.device

    @classmethod
    def field_annotations(cls) -> dict[str, type]:
        return BaseGraphDrawingData.__annotations__

    @classmethod
    def fields(cls, stage: Optional[str] = None) -> dict[str, Field]:
        def generate_fields():
            for field_name in cls.field_annotations():
                if not hasattr(cls, field_name):
                    continue
                field_info = getattr(cls, field_name)
                if not isinstance(field_info, cls.Field):
                    continue
                if field_info.optional and field_name not in cls.get_optional_fields():
                    continue
                if stage in [None, field_info.stage]:
                    yield field_name, field_info
        return dict(generate_fields())

    # noinspection PyPep8Naming
    def __init__(self, G: Optional[nx.Graph] = None):
        super().__init__(G=G)
        if G is not None:  # Allow empty GraphDrawingData when constructing Batch
            self.num_nodes = self.G.number_of_nodes()
            for transform in self._transforms(stage="init"):
                transform(self)

    def pre_filter(self) -> bool:
        assert not isinstance(self, Batch)
        return nx.is_connected(self.G.to_undirected())

    def _transforms(self, stage: str):
        transforms = []
        for field_info in self.fields(stage=stage).values():
            if field_info.transform not in transforms:
                transforms.append(field_info.transform)
        return transforms

    def pre_transform(self) -> Self:
        assert not isinstance(self, Batch)
        return T.Compose(self._transforms(stage="pre_transform"))(self)

    def static_transform(self) -> Self:
        assert not isinstance(self, Batch)
        return T.Compose(self._transforms(stage="static_transform"))(self)

    def dynamic_transform(self) -> Self:
        assert not isinstance(self, Batch)
        return T.Compose(self._transforms(stage="dynamic_transform"))(self)

    def post_transform(self) -> Self:
        transforms = T.Compose(self._transforms(stage="post_transform"))
        if isinstance(self, Batch):
            return Batch.from_data_list(list(map(transforms, self.to_data_list())))
        return transforms(self)

    def load_pos_dict(self, pos_dict: dict[str, np.ndarray]) -> Self:
        names = self.name
        if isinstance(names, str):
            names = [names]
        pos = np.concatenate(list(map(pos_dict.__getitem__, names)), axis=0)
        self.pos = torch.tensor(pos).to(self.device).float()
        return self

    def pos_dict(self) -> dict[str, np.ndarray]:
        if isinstance(self, Batch):
            return {data.name: data.pos.detach().cpu().numpy() for data in self.to_data_list()}
        return {self.name: self.pos.detach().cpu().numpy()}
