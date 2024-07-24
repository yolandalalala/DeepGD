from deepgd.constants import EPS

from itertools import permutations

from scipy import spatial
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class GabrielGraph(BaseTransform):

    def __init__(self,
                 attr_name: str = "gabriel_index",
                 eps: float = EPS):
        super().__init__()
        self.attr_name = attr_name
        self.eps = eps

    def __call__(self, data: Data) -> Data:
        delaunay_edges = data.face[list(permutations(range(3), 2)), :].transpose(1, 2).flatten(end_dim=1).unique(dim=0)
        tree = spatial.KDTree(data.pos.detach().cpu().numpy())
        c = data.pos[delaunay_edges]
        m = c.mean(dim=1)
        d = (c[:, 0, :] - c[:, 1, :]).norm(dim=1)
        dm = torch.tensor(tree.query(x=m.detach().cpu().numpy(), k=1)[0]).to(m)
        data[self.attr_name] = delaunay_edges[dm >= d / 2 * (1 - self.eps)].T
        return data
