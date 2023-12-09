from .gnn_layer import *
from ...functions import *

from itertools import chain

import torch
from torch import nn
import torch_geometric as pyg


class GNNBlock(nn.Module):
    def __init__(self, 
                 feat_dims, 
                 efeat_hid_dims=[], 
                 efeat_hid_act=nn.LeakyReLU,
                 efeat_out_act=nn.Tanh,
                 bn=False,
                 act=True,
                 dp=None,
                 aggr='mean',
                 root_weight=True,
                 static_efeats=2,
                 dynamic_efeats='skip',
                 euclidian=False,
                 direction=False,
                 n_weights=0,
                 residual=False):
        '''
        dynamic_efeats: {
            skip: block input to each layer, 
            first: block input to first layer, 
            prev: previous layer output to next layer, 
            orig: original node feature to each layer
        }
        '''
        super().__init__()
        self.static_efeats = static_efeats
        self.dynamic_efeats = dynamic_efeats
        self.euclidian = euclidian
        self.direction = direction
        self.n_weights = n_weights
        self.residual = residual
        self.gnn = nn.ModuleList()
        self.n_layers = len(feat_dims) - 1

        for idx, (in_feat, out_feat) in enumerate(zip(feat_dims[:-1], feat_dims[1:])):
            direction_dim = (feat_dims[idx] if self.dynamic_efeats == 'prev'
                             else 2 if self.dynamic_efeats == 'orig'
                             else feat_dims[0])
            in_efeat_dim = self.static_efeats
            if self.dynamic_efeats != 'first': 
                in_efeat_dim += self.euclidian + self.direction * direction_dim + self.n_weights
            edge_net = nn.Sequential(*chain.from_iterable(
                [nn.Linear(idim, odim),
                 nn.BatchNorm1d(odim),
                 act()]
                for idim, odim, act in zip([in_efeat_dim] + efeat_hid_dims,
                                           efeat_hid_dims + [in_feat * out_feat],
                                           [efeat_hid_act] * len(efeat_hid_dims) + [efeat_out_act])
            ))
            self.gnn.append(GNNLayer(nfeat_dims=(in_feat, out_feat), 
                                     efeat_dim=in_efeat_dim, 
                                     edge_net=edge_net,
                                     bn=bn, 
                                     act=act, 
                                     dp=dp,
                                     aggr=aggr,
                                     root_weight=root_weight,
                                     skip=False))
        
    def _get_edge_feat(self, pos, data, euclidian=False, direction=False):
        e = data.edge_attr[:, :self.static_efeats]
        if euclidian or direction:
            start_pos, end_pos = get_edges(pos, data)
            v, u = l2_normalize(end_pos - start_pos, return_norm=True)
            if euclidian:
                e = torch.cat([e, u], dim=1)
            if direction:
                e = torch.cat([e, v], dim=1)
        return e
        
    def forward(self, v, data):
        vres = v
        for layer in range(self.n_layers):
            vsrc = (v if self.dynamic_efeats == 'prev' 
                    else data.pos if self.dynamic_efeats == 'orig' 
                    else vres)
            get_extra = not (self.dynamic_efeats == 'first' and layer != 0)
            e = self._get_edge_feat(vsrc, data,
                                    euclidian=self.euclidian and get_extra, 
                                    direction=self.direction and get_extra)
            v = self.gnn[layer](v, e, data)
        return v + vres if self.residual else v
