import networkx as nx
import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class GenerateRandomLayout(BaseTransform):

    def __call__(self, data: Data) -> Data:
        layout = nx.drawing.random_layout(data.G)
        data.pos = torch.tensor(np.array([layout[i] for i in range(len(layout))])).to(data.device)
        return data
