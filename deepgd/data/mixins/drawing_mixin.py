from ..base_data import BaseData

from typing import Optional
from abc import ABC

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex


class DrawingMixin(BaseData, ABC):

    def _get_default_cmap(self):
        base1 = plt.get_cmap("coolwarm")
        base2 = LinearSegmentedColormap.from_list("CMAP", [to_hex(base1(i)) for i in [
            0.3, 0.7
        ]])
        cmap = LinearSegmentedColormap.from_list("CMAP", [to_hex(base1(i)) for i in [
            0.0, 0.03, 0.10, 0.21
        ]] + [to_hex(base2(0.5))] + [to_hex(base1(i)) for i in [
            0.79, 0.90, 0.97, 1.0
        ]])
        return cmap

    def draw(self, attr: Optional[dict] = None):
        attr = attr or {}
        pos = {i: self.pos[i].tolist() for i in range(len(self.pos))}
        nx.draw_networkx(
            G=self.G.to_undirected(),
            pos=pos,
            **dict(
                with_labels=False,
                node_size=60,
            ) | attr
        )
        plt.axis("equal")
        plt.axis("off")

    def draw_paper(self, cmap: Optional[LinearSegmentedColormap] = None, attr: Optional[dict] = None):
        cmap = cmap or self._get_default_cmap()
        attr = attr or {}
        pos = {i: self.pos[i].tolist() for i in range(len(self.pos))}
        length = np.sqrt(np.square(
            self.pos[self.edge_index[0]].numpy()
            - self.pos[self.edge_index[1]].numpy()
        ).sum(axis=1))
        min_len = length.min()
        max_len = length.max()
        ratio = (length - min_len) / (max_len - min_len + 1e-4)
        nx.draw_networkx(
            G=self.G.to_undirected(),
            pos=pos,
            **dict(node_size=0,
                   with_labels=False,
                   # labels=dict(zip(list(G.nodes), map(lambda n: n if type(n) is int else n[1:], list(G.nodes)))),
                   font_color="white",
                   font_weight="bold",
                   edge_color=list(map(lambda r: to_hex(cmap(1-r)), ratio)),
                   font_size=12,
                   width=1) | attr
        )
        plt.axis("equal")
        plt.axis("off")
