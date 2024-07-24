from deepgd.constants import DATASET_ROOT
from deepgd.data import BaseData, GraphDrawingData

import os
from typing import Callable, Optional, TypeVar, Iterator

import ssgetpy
import scipy
import torch
import torch_geometric as pyg
from torch_geometric.data import Dataset
import networkx as nx
from tqdm.auto import tqdm


DATATYPE = TypeVar("DATATYPE", bound=BaseData)


class SuiteSparseDataset(Dataset):

    DEFAULT_NAME = "SuiteSparse"

    def __init__(self, *,
                 root: str = DATASET_ROOT,
                 name: str = DEFAULT_NAME,
                 min_nodes=300,
                 max_nodes=3000,
                 limit=10000,
                 datatype: type[DATATYPE] = GraphDrawingData):
        self.dataset_name: str = name
        self.datatype: type[DATATYPE] = datatype
        self.graph_list = ssgetpy.search(colbounds=(min_nodes, max_nodes), limit=limit)
        super().__init__(
            root=os.path.join(root, name),
            transform=self.datatype.dynamic_transform,
            pre_transform=self.datatype.pre_transform,
            pre_filter=self.datatype.pre_filter
        )
        with open(self.processed_paths[0], "r") as index_file:
            self.index = index_file.read().strip().split("\n")

    @property
    def raw_file_names(self):
        return list(map(lambda graph: f"{graph.name}.mtx", self.graph_list))

    @property
    def processed_file_names(self):
        # TODO: put params in index file name
        file_names = ["index.txt"]
        index_path = os.path.join(self.processed_dir, file_names[0])
        if os.path.exists(index_path):
            with open(index_path, "r") as index_file:
                name_list = index_file.read().strip().split("\n")
            file_names.extend(list(map(lambda name: os.path.join("data", f"{name}.pt"), name_list)))
        return file_names

    def generate(self) -> Iterator[nx.Graph]:
        for graph, raw_path in zip(tqdm(self.graph_list, desc=f"Preprocess graphs"), self.raw_paths):
            mat = scipy.io.mmread(raw_path)
            if mat.shape[0] != mat.shape[1]:
                continue
            G = nx.from_scipy_sparse_matrix(mat)
            G.graph.update(dict(
                name=graph.name,
                dataset=self.dataset_name
            ))
            yield G

    def process(self):
        def get_path(data):
            return os.path.join(self.processed_dir, "data", f"{data.G.graph['name']}.pt")

        def filter_cached_and_save_index(data_list):
            name_list = []
            for data in data_list:
                name_list.append(data.G.graph['name'])
                if not os.path.exists(get_path(data)):
                    yield data
            with open(self.processed_paths[0], "w") as index_file:
                index_file.write("\n".join(name_list))

        data_list = map(self.datatype, self.generate())
        data_list = filter(self.pre_filter, data_list)
        data_list = map(self.pre_transform, filter_cached_and_save_index(data_list))
        for idx, data in enumerate(data_list):
            torch.save(data, get_path(data))

    def download(self):
        for graph in self.graph_list:
            raw_path = os.path.join(self.raw_dir, f"{graph.name}.mtx")
            if os.path.exists(raw_path):
                continue
            graph.download(destpath=self.raw_dir, extract=True)
            os.rename(os.path.join(self.raw_dir, graph.name, f"{graph.name}.mtx"), raw_path)
            os.rmdir(os.path.join(self.raw_dir, graph.name))

    def len(self):
        return len(self.index)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, "data", f"{self.index[idx]}.pt"))
        return data.static_transform()
