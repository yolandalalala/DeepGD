from . import *

import os
import re

import numpy as np
import torch
import torch_geometric as pyg
import networkx as nx
from tqdm.auto import tqdm


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
            n=G.number_of_nodes(),
            m=G.number_of_edges(),
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