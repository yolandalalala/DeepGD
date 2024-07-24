import networkx as nx
import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class AddAdjacencyInfo(BaseTransform):

    def __init__(self,
                 attr_name: str = "edge_metaindex"):
        super().__init__()
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        perm_u, perm_v = data.perm_index
        perm_unique_index = (perm_u * data.num_nodes + perm_v) * (2 * (perm_u < perm_v).long() - 1)
        edge_unique_index = torch.tensor([
            (u * data.num_nodes + v) * (2 * (u < v) - 1) for u, v in data.G.edges
        ]).to(perm_unique_index)
        pui_sorted, sorted_idx = perm_unique_index.sort()
        eui_sorted = edge_unique_index.sort().values
        search_idx = torch.searchsorted(pui_sorted, eui_sorted)
        data[self.attr_name] = sorted_idx[search_idx]
        return data
