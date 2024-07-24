from ..graph_struct import GraphStruct
from ..base_graph_drawing_data import BaseGraphDrawingData

from typing import Any, Optional, Mapping, Union, Iterable, Callable
from functools import singledispatchmethod
import numpy as np
import torch
from torch_geometric.data import Data, Batch


class StructMixin(BaseGraphDrawingData):

    def transform_struct(self, transform: Callable, struct: GraphStruct):
        return self.make_struct(transform(struct))

    def apply_transform(self, transform: Callable):
        struct = self.make_struct()
        self.make_struct(transform(struct))
        return self

    def make_struct(self, value: Any = None) -> GraphStruct:
        return self._struct(value)

    # TODO: logic overlap with make_struct
    def sync_pos(self, value: Any = None) -> None:
        self._struct(value)
        return self

    @singledispatchmethod
    def _struct(self, value: Any) -> GraphStruct:
        raise NotImplementedError(f"Unknown type {type(value)}!")

    @_struct.register
    def _(self, pos: None) -> GraphStruct:
        assert self.pos is not None
        data = self.post_transform()
        batch_index = data.batch if isinstance(data, Batch) else torch.zeros(data.num_nodes).long().to(data.device)
        num_graphs = data.num_graphs if isinstance(data, Batch) else 1
        return GraphStruct(
            pos=data.pos,
            n=data.n,
            m=data.m,
            x=data.x,
            batch=batch_index,
            num_nodes=data.num_nodes,
            num_edges=data.num_edges,
            num_graphs=num_graphs,
            perm_index=data.perm_index,
            perm_attr=data.perm_attr,
            perm_weight=data.perm_weight,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            edge_weight=data.edge_weight,
            aggr_index=data.aggr_index,
            aggr_attr=data.aggr_attr,
            aggr_weight=data.aggr_weight,
            apsp_attr=data.apsp_attr,
            gabriel_index=data.gabriel_index,
            rng_index=data.rng_index,
            edge_pair_index=data.edge_pair_index
        )

    @_struct.register
    def _(self, pos: np.ndarray) -> GraphStruct:
        return self._struct(torch.tensor(pos))

    @_struct.register
    def _(self, pos: torch.Tensor) -> GraphStruct:
        self.pos = pos.to(self.device).float()
        return self._struct(None)

    @_struct.register
    def _(self, store: dict) -> GraphStruct:
        self.load_pos_dict(store)
        return self._struct(None)

    @_struct.register
    def _(self, struct: GraphStruct) -> GraphStruct:
        return self._struct(struct.pos)
