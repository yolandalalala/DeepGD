import networkx as nx
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class NormalizeGraph(BaseTransform):

    def __call__(self, data: Data) -> Data:
        data.G = nx.convert_node_labels_to_integers(data.G).to_directed()
        data.G.remove_edges_from(nx.selfloop_edges(data.G))
        nx.set_edge_attributes(data.G, None, "weight")
        return data
