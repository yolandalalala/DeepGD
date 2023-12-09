from .gnn_block import *

from itertools import chain

import torch
from torch import nn
import torch_geometric as pyg

class DeepGD(nn.Module):
    def __init__(self, 
                 num_blocks=9, 
                 num_layers=3,
                 num_enet_layers=2,
                 layer_dims=None,
                 n_weights=0, 
                 dynamic_efeats='skip',
                 euclidian=True,
                 direction=True,
                 residual=True,
                 normalize=None):
        super().__init__()

        self.in_blocks = nn.ModuleList([
            GNNBlock(feat_dims=[2, 8, 8 if layer_dims is None else layer_dims[0]], bn=True, dp=0.2, static_efeats=2)
        ])
        self.hid_blocks = nn.ModuleList([
            GNNBlock(feat_dims=layer_dims or ([8] + [8] * num_layers), 
                     efeat_hid_dims=[16] * (num_enet_layers - 1),
                     bn=True, 
                     act=True,
                     dp=0.2, 
                     static_efeats=2,
                     dynamic_efeats=dynamic_efeats,
                     euclidian=euclidian,
                     direction=direction,
                     n_weights=n_weights,
                     residual=residual)
            for _ in range(num_blocks)
        ])
        self.out_blocks = nn.ModuleList([
            GNNBlock(feat_dims=[8 if layer_dims is None else layer_dims[-1], 8], bn=True, static_efeats=2),
            GNNBlock(feat_dims=[8, 2], act=False, static_efeats=2)
        ])
        self.normalize = normalize

    def forward(self, data, output_hidden=False, numpy=False):
        v = data.init_pos if data.init_pos is not None else generate_rand_pos(len(data.x)).to(data.x.device)
        if self.normalize is not None:
            v = self.normalize(v, data)
        
        hidden = []
        for block in chain(self.in_blocks, 
                           self.hid_blocks, 
                           self.out_blocks):
            v = block(v, data)
            if output_hidden:
                hidden.append(v.detach().cpu().numpy() if numpy else v)
        if not output_hidden:
            vout = v.detach().cpu().numpy() if numpy else v
            if self.normalize is not None:
                vout = self.normalize(vout, data)
        
        return hidden if output_hidden else vout
    
