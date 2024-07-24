import networkx as nx
import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class PopulateGraphAttrs(BaseTransform):

    def __call__(self, data: Data) -> Data:
        data.name = data.G.graph.get("name", None)
        data.dataset = data.G.graph.get("dataset_name", None)
        data.n = torch.tensor(data.G.number_of_nodes()).to(data.device)
        assert data.G.number_of_edges() % 2 == 0
        data.m = torch.tensor(data.G.number_of_edges()).to(data.device) // 2  # Get undirected count
        return data
