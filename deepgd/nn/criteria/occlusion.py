from ...functions import *

import torch
from torch import nn
from torch_geometric.utils import scatter


class Occlusion(nn.Module):
    def __init__(self, gamma=1, reduce=torch.mean):
        super().__init__()
        self.gamma = gamma
        self.reduce = reduce
        
    def forward(self, node_pos, batch):
        start, end = get_full_edges(node_pos, batch)
        eu = end.sub(start).norm(dim=1)
        edge_occusion = eu.mul(-self.gamma).exp()
        index = batch.batch[batch.full_edge_index[0]]
        graph_occusion = scatter(edge_occusion, index)
        return graph_occusion if self.reduce is None else self.reduce(graph_occusion)