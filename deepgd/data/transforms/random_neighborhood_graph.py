from deepgd.constants import EPS

from itertools import permutations

import numpy as np
from scipy import spatial, sparse
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class RandomNeighborhoodGraph(BaseTransform):

    def __init__(self,
                 attr_name: str = "rng_index",
                 eps: float = EPS):
        super().__init__()
        self.attr_name = attr_name
        self.eps = eps

    def __call__(self, data: Data) -> Data:
        delaunay_edges = data.face[list(permutations(range(3), 2)), :].transpose(1, 2).flatten(end_dim=1).unique(dim=0)
        tree = spatial.KDTree(data.pos.detach().cpu().numpy())
        c = data.pos[delaunay_edges]
        src, dst = c[:, 0, :], c[:, 1, :]
        d = (src - dst).norm(dim=1)
        r = (d * (1 - self.eps)).detach().cpu().numpy()

        p0 = tree.query_ball_point(x=src.detach().cpu().numpy(), r=r)
        p0m = sparse.lil_matrix((len(delaunay_edges), data.num_nodes))
        p0m.rows, p0m.data = p0, list(map(np.ones_like, p0))
        p0idx = torch.tensor(p0m.toarray(), device=src.device, dtype=torch.bool)

        p1 = tree.query_ball_point(x=dst.detach().cpu().numpy(), r=r)
        p1m = sparse.lil_matrix((len(delaunay_edges), data.num_nodes))
        p1m.rows, p1m.data = p1, list(map(np.ones_like, p1))
        p1idx = torch.tensor(p1m.toarray(), device=dst.device, dtype=torch.bool)

        data[self.attr_name] = delaunay_edges[~(p0idx & p1idx).any(dim=1)].T
        return data
