import torch
import torch, numpy as np, scipy.sparse as sp
import torch_sparse
from torch_scatter import scatter, scatter_add
from torch_geometric.utils import dropout_adj



def drop_incidence(hyperedge_index, p):
    if p == 0.0:
        return hyperedge_index

    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p

    row, col = row[mask], col[mask]
    hyperedge_index = torch.stack([row, col], dim=0)

    return hyperedge_index


def drop_feature(x, p):
    if p == 0.0:
        return x

    mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p # <p_f的位置为True，反之为False
    x = x.clone()
    x[:, mask] = 0  # mask为True的位置上的节点特征置为0

    return x



# membership aug
def valid_node_edge_mask(hyperedge_index, num_nodes, num_edges):
    '''
    This function calculates the effective node mask and edge mask
    :param hyperedge_index:
    :param hyperedge_weigth:
    :return:
    '''
    ones = hyperedge_index.new_ones(hyperedge_index.shape[1])

    Dn = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)  # 每个节点连接的超边数量。
    De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_edges)  # 每条边连接的节点数量。

    node_mask = Dn != 0 #创建了一个节点掩码 (node_mask)，其中 True 表示对应节点连接有超边。
    edge_mask = De != 0 #创建了一个边掩码 (edge_mask)，其中 True 表示对应边连接有节点。

    return node_mask, edge_mask



# 基于给定的节点掩码 (node_mask) 和边掩码 (edge_mask) 来过滤超图的边索引 (hyperedge_index)。
def hyperedge_index_masking(hyperedge_index, node_mask, edge_mask, num_nodes, num_edges):
    if node_mask is None and edge_mask is None:
        return hyperedge_index

    # 创建一个稀疏张量 H，其中每个元素表示对应的超边索引是否存在。然后，通过 to_dense 方法得到一个矩阵，其中行表示节点，列表示超边，元素为 1 表示对应的节点与超边相连。
    H = torch.sparse_coo_tensor(hyperedge_index, hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()

    # 根据提供的节点掩码和边掩码，进行不同的过滤操作

    if node_mask is not None and edge_mask is not None:
        masked_hyperedge_index = H[node_mask][:, edge_mask].to_sparse().indices()

    elif node_mask is None and edge_mask is not None:
        masked_hyperedge_index = H[:, edge_mask].to_sparse().indices()

    elif node_mask is not None and edge_mask is None:
        masked_hyperedge_index = H[node_mask].to_sparse().indices()

    return masked_hyperedge_index



def remove_hyperedge_by_weight(hyperedge_index, num_nodes, num_edges, p):
    # 根据超边重要性(权重) 删除超边， p为超边移除的概率，

    # 将Hyperedge_index 转为稀疏邻接矩阵
    H = torch.sparse_coo_tensor(hyperedge_index, hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges)).to_dense()

    # 计算节点的度中心性
    degrees = torch.sum(H, dim=1)
    degree_centrality = degrees / torch.sum(degrees)

    # 计算边的权重
    edge_weights = torch.sum(H * degree_centrality.unsqueeze(1), dim=0)

    # 权重归一化[0, 1]
    edge_weights_norm = edge_weights / edge_weights.sum()
    # edge_weights_norm = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())

    # 超边被移除的概率： 权重越大，被移除的概率越小
    edge_remove_prob = 1 - edge_weights_norm

    retain_indices = torch.multinomial(input=edge_weights_norm, num_samples=int(H.shape[1] * (1-p)))

    H = H[:, retain_indices]

    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index



def remove_hyperedge_by_weight1(hyperedge_index, num_nodes, num_edges, p):
    # 根据超边重要性(权重) 删除超边， p为超边移除的概率，

    # 将Hyperedge_index 转为稀疏邻接矩阵
    H = torch.sparse_coo_tensor(hyperedge_index, hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges)).to_dense()

    # 计算节点的度中心性
    degrees = torch.sum(H, dim=1)
    degree_centrality = degrees / torch.sum(degrees)

    # 计算边的权重
    edge_weights = torch.sum(H * degree_centrality.unsqueeze(1), dim=0)

    # 权重归一化[0, 1]
    # edge_weights_norm = edge_weights / edge_weights.sum()
    edge_weights_norm = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())

    # 超边被移除的概率： 权重越大，被移除的概率越小
    edge_weights = edge_weights_norm / edge_weights_norm.mean() * p

    sel_mask = torch.bernoulli(1 - edge_weights).to(torch.bool)

    H = H[:, sel_mask]

    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index

