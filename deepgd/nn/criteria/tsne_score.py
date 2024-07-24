from ...functions import *

import torch
from torch import nn
from torch_geometric.utils import scatter


class TSNEScore(nn.Module):
    def __init__(self, sigma=1, reduce=torch.mean):
        super().__init__()
        self.sigma = sigma
        self.reduce = reduce
        
    def forward(self, node_pos, batch):
        p = batch.full_edge_attr[:, 0].div(-2 * self.sigma**2).exp()
        sum_src = scatter(p, batch.full_edge_index[0])[batch.full_edge_index[0]]
        sum_dst = scatter(p, batch.full_edge_index[1])[batch.full_edge_index[1]]
        p = (p / sum_src + p / sum_dst) / (2 * batch.n[batch.batch[batch.edge_index[0]]])
        start, end = get_full_edges(node_pos, batch)
        eu = end.sub(start).norm(dim=1)
        index = batch.batch[batch.full_edge_index[0]]
        q = 1 / (1 + eu.square())
        q /= scatter(q, index)[index]
        edge_kl = (p.log() - q.log()).mul(p)
        graph_kl = scatter(edge_kl, index)
        return graph_kl if self.reduce is None else self.reduce(graph_kl)