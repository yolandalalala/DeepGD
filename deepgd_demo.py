from itertools import chain

import torch
from torch import nn
import torch_geometric as pyg
import torch_scatter

import os.path
import re
import random

from tqdm.autonotebook import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch_geometric as pyg

DATA_ROOT = 'deepgd_data'


class RomeDataset(pyg.data.InMemoryDataset):
    def __init__(self, *,
                 url='http://www.graphdrawing.org/download/rome-graphml.tgz',
                 root=f'{DATA_ROOT}/Rome',
                 layout_initializer=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.url = url
        self.initializer = layout_initializer or nx.drawing.random_layout
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        metafile = "rome/Graph.log"
        if os.path.exists(metadata_path := f'{self.raw_dir}/{metafile}'):
            return list(map(lambda f: f'rome/{f}.graphml',
                            self.get_graph_names(metadata_path)))
        else:
            return [metafile]

    @property
    def processed_file_names(self):
        return ['data.pt']

    @classmethod
    def get_graph_names(cls, logfile):
        with open(logfile) as fin:
            for line in fin.readlines():
                if match := re.search(r'name: (grafo\d+\.\d+)', line):
                    yield f'{match.group(1)}'

    def process_raw(self):
        graphmls = sorted(self.raw_paths,
                          key=lambda x: int(re.search(r'grafo(\d+)', x).group(1)))
        for file in tqdm(graphmls, desc=f"Loading graphs"):
            G = nx.read_graphml(file)
            if nx.is_connected(G):
                yield nx.convert_node_labels_to_integers(G)

    def convert(self, G):
        apsp = dict(nx.all_pairs_shortest_path_length(G))
        init_pos = torch.tensor(np.array(list(self.initializer(G).values())))
        full_edges, attr_d = zip(*[((u, v), d) for u in apsp for v, d in apsp[u].items()])
        raw_edge_index = pyg.utils.to_undirected(torch.tensor(list(G.edges)).T)
        full_edge_index, d = pyg.utils.remove_self_loops(*pyg.utils.to_undirected(
            torch.tensor(full_edges).T, torch.tensor(attr_d)
        ))
        k = 1 / d ** 2
        full_edge_attr = torch.stack([d, k], dim=-1)
        return pyg.data.Data(
            G=G,
            x=init_pos,
            init_pos=init_pos,
            edge_index=full_edge_index,
            edge_attr=full_edge_attr,
            raw_edge_index=raw_edge_index,
            full_edge_index=full_edge_index,
            full_edge_attr=full_edge_attr,
            d=d,
        )

    def download(self):
        pyg.data.download_url(self.url, self.raw_dir)
        pyg.data.extract_tar(f'{self.raw_dir}/rome-graphml.tgz', self.raw_dir)

    def process(self):
        data_list = map(self.convert, self.process_raw())

        if self.pre_filter is not None:
            data_list = filter(self.pre_filter, data_list)

        if self.pre_transform is not None:
            data_list = map(self.pre_transform, data_list)

        data, slices = self.collate(list(data_list))
        torch.save((data, slices), self.processed_paths[0])
        
        
def l2_normalize(x, return_norm=False, eps=1e-5):
    if type(x) is torch.Tensor:
        norm = x.norm(dim=1).unsqueeze(dim=1) 
    else:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
    unit_vec = x / (norm + eps)
    if return_norm:
        return unit_vec, norm
    else:
        return unit_vec
    
    
def get_edges(node_pos, batch):
    edges = node_pos[batch.edge_index.T]
    return edges[:, 0, :], edges[:, 1, :]


def get_full_edges(node_pos, batch):
    edges = node_pos[batch.full_edge_index.T]
    return edges[:, 0, :], edges[:, 1, :]


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
        self.conv = pyg.nn.NNConv(in_dim, out_dim, nn=self.enet, aggr=aggr, root_weight=root_weight)
        self.dense = nn.Linear(out_dim, out_dim) if dense else nn.Identity()
        self.bn = pyg.nn.BatchNorm(out_dim) if bn else nn.Identity()
        self.act = nn.LeakyReLU() if act else nn.Identity()
        self.dp = dp and nn.Dropout(dp) or nn.Identity()
        self.skip = skip
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        
    def forward(self, v, e, data):
        v_ = v
        v = self.conv(v, data.edge_index, e)
        v = self.dense(v)
        v = self.bn(v)
        v = self.act(v)
        v = self.dp(v)
        return v + self.proj(v_) if self.skip else v
    
    
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
        
    def _get_edge_feat(self, pos, data, euclidian=False, direction=False, weights=None):
        e = data.edge_attr[:, :self.static_efeats]
        if euclidian or direction:
            start_pos, end_pos = get_edges(pos, data)
            v, u = l2_normalize(end_pos - start_pos, return_norm=True)
            if euclidian:
                e = torch.cat([e, u], dim=1)
            if direction:
                e = torch.cat([e, v], dim=1)
        if weights is not None:
            w = weights.repeat(len(e), 1)
            e = torch.cat([e, w], dim=1)
        return e
        
    def forward(self, v, data, weights=None):
        vres = v
        for layer in range(self.n_layers):
            vsrc = (v if self.dynamic_efeats == 'prev' 
                    else data.pos if self.dynamic_efeats == 'orig' 
                    else vres)
            get_extra = not (self.dynamic_efeats == 'first' and layer != 0)
            e = self._get_edge_feat(vsrc, data,
                                    euclidian=self.euclidian and get_extra, 
                                    direction=self.direction and get_extra,
                                    weights=weights if get_extra and self.n_weights > 0 else None)
            v = self.gnn[layer](v, e, data)
        return v + vres if self.residual else v


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

    def forward(self, data, weights=None, output_hidden=False, numpy=False):
        v = data.init_pos if data.init_pos is not None else generate_rand_pos(len(data.x)).to(data.x.device)
        if self.normalize is not None:
            v = self.normalize(v, data)
        
        hidden = []
        for block in chain(self.in_blocks, 
                           self.hid_blocks, 
                           self.out_blocks):
            v = block(v, data, weights)
            if output_hidden:
                hidden.append(v.detach().cpu().numpy() if numpy else v)
        if not output_hidden:
            vout = v.detach().cpu().numpy() if numpy else v
            if self.normalize is not None:
                vout = self.normalize(vout, data)
        
        return hidden if output_hidden else vout
    
    
class Stress(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce
        
    def forward(self, node_pos, batch):
        start, end = get_full_edges(node_pos, batch)
        eu = (start - end).norm(dim=1)
        d = batch.full_edge_attr[:, 0]
        edge_stress = eu.sub(d).abs().div(d).square()
        index = batch.batch[batch.edge_index[0]]
        graph_stress = torch_scatter.scatter(edge_stress, index)
        return graph_stress if self.reduce is None else self.reduce(graph_stress)