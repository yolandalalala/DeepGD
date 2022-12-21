import torch
import torch_scatter
import numpy as np

        
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


def get_raw_edges(node_pos, batch):
    edges = node_pos[batch.raw_edge_index.T]
    return edges[:, 0, :], edges[:, 1, :]


def get_per_graph_property(batch, property_getter):
    return torch.tensor(list(map(property_getter, batch.to_data_list())), 
                        device=batch.x.device)


def map_node_indices_to_graph_property(batch, node_index, property_getter):
    return get_per_graph_property(batch, property_getter)[batch.batch][node_index]


def map_node_indices_to_node_degrees(real_edges, node_indices):
    node, degrees = np.unique(real_edges[:, 0].detach().cpu().numpy(), return_counts=True)
    return torch.tensor(degrees[node_indices], device=real_edges.device)


def get_counter_clockwise_sorted_angle_vertices(edges, pos):
    if type(pos) is torch.Tensor:
        edges = edges.cpu().detach().numpy()
        pos = pos.cpu().detach().numpy()
    u, v = edges[:, 0], edges[:, 1]
    diff = pos[v] - pos[u]
    diff_normalized = l2_normalize(diff)
    # get cosine angle between uv and y-axis
    cos = diff_normalized @ np.array([[1],[0]])
    # get radian between uv and y-axis
    radian = np.arccos(cos) * np.expand_dims(np.sign(diff[:, 1]), axis=1)
    # for each u, sort edges based on the position of v
    sorted_idx = sorted(np.arange(len(edges)), key=lambda e: (u[e], radian[e]))
    sorted_v = v[sorted_idx]
    # get start index for each u
    idx = np.unique(u, return_index=True)[1]
    roll_idx = np.arange(1, len(u) + 1)
    roll_idx[np.roll(idx - 1, -1)] = idx
    rolled_v = sorted_v[roll_idx]
    return np.stack([u, sorted_v, rolled_v]).T[sorted_v != rolled_v]


def get_radians(pos, batch, 
                return_node_degrees=False, 
                return_node_indices=False, 
                return_num_nodes=False, 
                return_num_real_edges=False):
    real_edges = batch.raw_edge_index.T
    angles = get_counter_clockwise_sorted_angle_vertices(real_edges, pos)
    u, v1, v2 = angles[:, 0], angles[:, 1], angles[:, 2]
    e1 = l2_normalize(pos[v1] - pos[u])
    e2 = l2_normalize(pos[v2] - pos[u])
    radians = (e1 * e2).sum(dim=1).acos()
    result = (radians,)
    if return_node_degrees:
        degrees = map_node_indices_to_node_degrees(real_edges, u)
        result += (degrees,)
    if return_node_indices:
        result += (u,)
    if return_num_nodes:
        node_counts = map_node_indices_to_graph_property(batch, angles[:,0], lambda g: g.num_nodes)
        result += (node_counts,)
    if return_num_real_edges:
        edge_counts = map_node_indices_to_graph_property(batch, angles[:,0], lambda g: len(g.raw_edge_index.T))
        result += (edge_counts,)
    return result[0] if len(result) == 1 else result
