import re
from dataclasses import dataclass, asdict, fields
from types import SimpleNamespace as namespace
from typing import Mapping, Optional, Union, Iterable
from typing_extensions import Self

import torch
from torch import jit
from torch import FloatTensor, LongTensor, Tensor
import torch_geometric as pyg
from torch_geometric.typing import OptTensor


@dataclass(eq=False, repr=False)
class GraphStruct:

    pos:                FloatTensor
    n:                  LongTensor
    m:                  LongTensor
    x:                  FloatTensor
    batch:              LongTensor
    num_nodes:          int
    num_edges:          int
    num_graphs:         int
    perm_index:         LongTensor
    perm_attr:          FloatTensor
    perm_weight:        FloatTensor
    edge_index:         LongTensor
    edge_attr:          FloatTensor
    edge_weight:        FloatTensor
    aggr_index:         LongTensor
    aggr_attr:          FloatTensor
    aggr_weight:        FloatTensor
    apsp_attr:          LongTensor
    gabriel_index:      LongTensor
    rng_index:          LongTensor
    edge_pair_index:    LongTensor

    @property
    def perm_src_index(self) -> Tensor:
        return self.perm_index[0]

    @property
    def perm_dst_index(self) -> Tensor:
        return self.perm_index[1]

    @property
    def perm_src_pos(self) -> torch.FloatTensor:
        return self.pos[self.perm_src_index]

    @property
    def perm_dst_pos(self) -> torch.FloatTensor:
        return self.pos[self.perm_dst_index]

    @property
    def perm_batch_index(self) -> torch.LongTensor:
        return self.batch[self.perm_src_index]

    @property
    def edge_src_index(self) -> Tensor:
        return self.edge_index[0]

    @property
    def edge_dst_index(self) -> Tensor:
        return self.edge_index[1]

    @property
    def edge_src_pos(self) -> torch.FloatTensor:
        return self.pos[self.edge_src_index]

    @property
    def edge_dst_pos(self) -> torch.FloatTensor:
        return self.pos[self.edge_dst_index]

    @property
    def edge_batch_index(self) -> torch.LongTensor:
        return self.batch[self.edge_src_index]

    def __call__(self, pos: torch.FloatTensor):  # TODO: figure out how to use `-> Self`
        # TODO: use __annotations__
        return GraphStruct(
            pos=pos,
            n=self.n,
            m=self.m,
            x=self.x,
            batch=self.batch,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            num_graphs=self.num_graphs,
            perm_index=self.perm_index,
            perm_attr=self.perm_attr,
            perm_weight=self.perm_weight,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            edge_weight=self.edge_weight,
            aggr_index=self.aggr_index,
            aggr_attr=self.aggr_attr,
            aggr_weight=self.aggr_weight,
            apsp_attr=self.apsp_attr,
            gabriel_index=self.gabriel_index,
            rng_index=self.rng_index,
            edge_pair_index=self.edge_pair_index
        )

    @jit.unused
    def __repr__(self) -> str:
        shape_dict = {
            k.name: list(val.shape) if isinstance(val := getattr(self, k.name), Tensor) else val
            for k in fields(self)
            if getattr(self, k.name) is not None
        }
        return type(self).__name__ + re.sub("namespace", "", repr(namespace(**shape_dict)))

