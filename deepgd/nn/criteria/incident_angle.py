from ...functions import *

import torch
from torch import nn
import numpy as np
from torch_geometric.utils import scatter


class IncidentAngle(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce
    
    def forward(self, node_pos, batch):
        theta, degrees, indices = get_radians(node_pos, batch, 
                                              return_node_degrees=True, 
                                              return_node_indices=True)
        phi = degrees.float().pow(-1).mul(2*np.pi)
        angle_l1 = phi.sub(theta).abs()
        index = batch.batch[indices]
        graph_l1 = scatter(angle_l1, index)
        return graph_l1 if self.reduce is None else self.reduce(graph_l1)
    