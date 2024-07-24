from itertools import permutations

import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class GeneratePermutationEdges(BaseTransform):

    def __init__(self,
                 attr_name: str = "perm_index"):
        super().__init__()
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        data[self.attr_name] = torch.tensor(np.array(list(permutations(range(data.num_nodes), 2)))).T
        return data
