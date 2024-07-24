import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class GenerateEdgePairs(BaseTransform):

    def __init__(self,
                 attr_name: str = "edge_pair_metaindex"):
        super().__init__()
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        unique_edge_metaindex = data.edge_metaindex[data.edge_index[0] < data.edge_index[1]]
        data[self.attr_name] = torch.combinations(unique_edge_metaindex).T
        return data
