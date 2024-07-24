from deepgd.constants import DATASET_ROOT
from deepgd.data import BaseData, GraphDrawingData

import os
import re
import hashlib
from typing import Callable, Optional, TypeVar, Iterator

from tqdm.auto import tqdm
import numpy as np
import torch
import torch_geometric as pyg
import networkx as nx


DATATYPE = TypeVar("DATATYPE", bound=BaseData)


class RomeDataset(pyg.data.InMemoryDataset):

    DEFAULT_NAME = "Rome"
    DEFAULT_URL = "https://www.graphdrawing.org/download/rome-graphml.tgz"
    GRAPH_NAME_REGEX = re.compile(r"grafo(\d+)\.(\d+)")

    def __init__(self, *,
                 url: str = DEFAULT_URL,
                 root: str = DATASET_ROOT,
                 name: str = DEFAULT_NAME,
                 index: Optional[list[str]] = None,
                 datatype: type[DATATYPE] = GraphDrawingData):
        self.url: str = url
        self.dataset_name: str = name
        self.index: Optional[list[str]] = index
        self.datatype: type[DATATYPE] = datatype
        super().__init__(
            root=os.path.join(root, name),
            transform=self.datatype.dynamic_transform,
            pre_transform=self.datatype.pre_transform,
            pre_filter=self.datatype.pre_filter
        )
        self.data, self.slices = torch.load(self.data_path)
        with open(self.index_path, "r") as index_file:
            self.index = index_file.read().strip().split("\n")
        data_list = map(datatype.static_transform, tqdm(self, desc=f"Transform graphs"))
        data_dict = {data.G.graph["name"]: data for data in data_list}
        data_list = [data_dict[name] for name in self.index]
        self.data, self.slices = self.collate(list(data_list))

    def _parse_metadata(self, logfile: str) -> Iterator[str]:
        with open(logfile) as fin:
            for line in fin.readlines():
                if match := self.GRAPH_NAME_REGEX.search(line):
                    yield match.group(0)

    @property
    def raw_file_names(self) -> list[str]:
        metadata_file = "rome/Graph.log"
        if os.path.exists(metadata_path := os.path.join(self.raw_dir, metadata_file)):
            return list(map(lambda f: f"rome/{f}.graphml", self._parse_metadata(metadata_path)))
        return [metadata_file]

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt", "index.txt"]

    @property
    def data_path(self) -> str:
        return self.processed_paths[0]

    @property
    def index_path(self) -> str:
        return self.processed_paths[1]

    def generate(self) -> Iterator[nx.Graph]:
        def key(path):
            match = self.GRAPH_NAME_REGEX.search(path)
            return int(match.group(1)), int(match.group(2))
        for file in tqdm(sorted(self.raw_paths, key=key), desc=f"Loading graphs"):
            G = nx.read_graphml(file)
            G.graph.update(dict(
                name=self.GRAPH_NAME_REGEX.search(file).group(0),
                dataset=self.dataset_name
            ))
            yield G

    def download(self) -> None:
        pyg.data.download_url(self.url, self.raw_dir)
        pyg.data.extract_tar(f'{self.raw_dir}/rome-graphml.tgz', self.raw_dir)

    def process(self) -> None:
        def filter_and_save_index(data_list):
            name_list = []
            for data in data_list:
                if self.pre_filter(data):
                    name_list.append(data.G.graph["name"])
                    yield data
            if self.index is None:
                self.index = name_list
            else:
                self.index = [name for name in self.index if name in name_list]
            with open(self.index_path, "w") as index_file:
                index_file.write("\n".join(self.index))

        data_list = map(self.datatype, self.generate())
        data_list = filter_and_save_index(data_list)
        data_list = map(self.pre_transform, data_list)
        data, slices = self.collate(list(data_list))
        torch.save((data, slices), self.processed_paths[0])
