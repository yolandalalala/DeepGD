import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class SampleAggregationEdges(BaseTransform):

    def __init__(self,
                 attr_name: str = "aggr_metaindex"):
        super().__init__()
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        data[self.attr_name] = torch.arange(data.num_nodes * (data.num_nodes - 1)).to(data.device)
        return data
