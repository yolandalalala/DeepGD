from ...functions import *

import torch
from torch import nn
from torch_geometric.utils import scatter


class Stress(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce
        
    def forward(self, node_pos, batch):
        start, end = get_full_edges(node_pos, batch)
        eu = (start - end).norm(dim=1)
        d = batch.full_edge_attr[:, 0]
        edge_stress = eu.sub(d).abs().div(d).square()
        index = batch.batch[batch.full_edge_index[0]]
        graph_stress = scatter(edge_stress, index)
        return graph_stress if self.reduce is None else self.reduce(graph_stress)
    