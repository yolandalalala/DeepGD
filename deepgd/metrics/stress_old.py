import torch
from torch import nn


class Stress(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, node_pos, edge_index, apsp):
        start, end = node_pos[edge_index[0]], node_pos[edge_index[1]]
        eu = (start - end).norm(dim=1)
        edge_stress = eu.sub(apsp).abs().div(apsp).square().sum()
        return edge_stress
    