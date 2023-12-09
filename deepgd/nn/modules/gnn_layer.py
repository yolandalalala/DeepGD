import torch
from torch import nn
import torch_geometric as pyg

from .weighted_nnconv import WeightedNNConv

class GNNLayer(nn.Module):
    def __init__(self,
                 nfeat_dims,
                 efeat_dim,
                 aggr,
                 edge_net=None, 
                 dense=False,
                 bn=True, 
                 act=True, 
                 dp=None,
                 root_weight=True,
                 skip=True):
        super().__init__()
        try:
            in_dim = nfeat_dims[0]
            out_dim = nfeat_dims[1]
        except:
            in_dim = nfeat_dims
            out_dim = nfeat_dims
        self.enet = nn.Linear(efeat_dim, in_dim * out_dim) if edge_net is None and efeat_dim > 0 else edge_net
        self.conv = WeightedNNConv(in_dim, out_dim, nn=self.enet, aggr=aggr, root_weight=root_weight)
        self.dense = nn.Linear(out_dim, out_dim) if dense else nn.Identity()
        self.bn = pyg.nn.BatchNorm(out_dim) if bn else nn.Identity()
        self.act = nn.LeakyReLU() if act else nn.Identity()
        self.dp = dp and nn.Dropout(dp) or nn.Identity()
        self.skip = skip
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        
    def forward(self, v, e, data):
        v_ = v
        v = self.conv(v, data.edge_index, e, data.edge_weight)
        v = self.dense(v)
        v = self.bn(v)
        v = self.act(v)
        v = self.dp(v)
        return v + self.proj(v_) if self.skip else v
    
    