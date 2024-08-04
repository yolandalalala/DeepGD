import torch
from torch import nn
from torch_geometric.utils import scatter


class Stress(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, node_pos, edge_index, apsp, batch_index) -> torch.Tensor:
        start, end = node_pos[edge_index[0]], node_pos[edge_index[1]]
        dist = torch.norm(end - start, 2, 1)
        edge_stress = dist.sub(apsp).abs().div(apsp).square()
        return scatter(edge_stress, batch_index, reduce="sum").mean()
