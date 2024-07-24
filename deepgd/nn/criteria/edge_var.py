from ...functions import *

import torch
from torch import nn
from torch_geometric.utils import scatter


class EdgeVar(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce

    def forward(self, node_pos, batch):
        edge_idx = batch.raw_edge_index.T
        start, end = get_raw_edges(node_pos, batch)
        eu = end.sub(start).norm(dim=1)
        edge_var = eu.sub(1).square()
        index = batch.batch[batch.raw_edge_index[0]]
        graph_var = scatter(edge_var, index, reduce="mean")
        return graph_var if self.reduce is None else self.reduce(graph_var)