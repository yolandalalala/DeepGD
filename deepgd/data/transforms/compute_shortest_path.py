from typing import Optional
from itertools import permutations

import networkx as nx
import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class ComputeShortestPath(BaseTransform):

    def __init__(self,
                 cutoff: Optional[int] = None,
                 attr_name: Optional[str] = "apsp_attr",
                 weight_name: Optional[str] = None):
        super().__init__()
        self.cutoff = cutoff
        self.attr_name = attr_name
        self.weight_name = weight_name

    def __call__(self, data: Data) -> Data:
        apsp = dict(nx.all_pairs_shortest_path_length(data.G, cutoff=self.cutoff))
        perm = list(permutations(range(data.num_nodes), 2))
        attr = torch.tensor(np.array([apsp[u][v] for u, v in perm])).to(data.device)
        if self.attr_name is not None:
            data[self.attr_name] = attr
        elif "edge_attr" in data:
            data.edge_attr = torch.cat([data.edge_attr, attr.unsqueeze(1)], dim=1)
        else:
            data.edge_attr = attr.unsqueeze(1)
        if self.weight_name:
            data[self.weight_name] = 1 / attr.square()
        return data
