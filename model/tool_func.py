import numpy as np
import torch


def preprocess_features(feature):
    """
    Row-normalize feature matrix and convert to tuple representation
    :param feature:
    :return:
    """
    col_var = np.var(feature, axis=1, keepdims=True)
    col_mean = np.mean(feature, axis=1, keepdims=True)
    c_inv = np.power(col_var, -0.5)
    c_inv[np.isinf(c_inv)] = 0.
    feature = np.multiply((feature - col_mean), c_inv)
    return feature


def k_order_adj(adj, neigh_order=1):
    '''compute k-order adjacent matrix'''

    diffused_adj = np.eye(adj.shape[1])
    for _ in range(neigh_order): # 一阶邻接矩阵
        diffused_adj = np.matmul(diffused_adj, (adj + np.eye(adj.shape[1])))
    diffused_adj = np.where(diffused_adj > 0, 1.0, diffused_adj)

    return diffused_adj



def construct_G_from_H(H):

    G_n = torch.mm(H, H.t())
    G_e = torch.mm(H.t(), H)

    G_n1 = torch.where(G_n != 0, 1, 0)
    G_e1 = torch.where(G_e != 0, 1, 0)

    G_n2 = G_n1 - torch.eye(G_n1.size(0))
    G_e2 = G_e1 - torch.eye(G_e1.size(0))

    node_adj = G_n2
    edge_adj = G_e2

    return node_adj, edge_adj


def sigmoid(z):
    return 1/(1 + np.exp(-z))



def cal_degree_of_each_pair(hyperedge_index, num_nodes, num_edges):
    '''
    :param H: incidence matrix
    :return:
    '''
    H = torch.sparse_coo_tensor(hyperedge_index, hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges)).to_dense()


    degree_mat = torch.zeros(num_nodes, num_nodes)

    for hyperedge_idx in range(num_edges):
        hyperedge = H[:, hyperedge_idx]  # 获取当前超边的连接情况
        connected_nodes = torch.nonzero(hyperedge).squeeze()  # 获取连接的节点索引

        for node_i in connected_nodes:
            for node_j in connected_nodes:
                degree_mat[node_i,node_j] = degree_mat[node_i,node_j] + 1 #对角线是节点度，其余是节点对的度

    return degree_mat




def cal_homogeneity_hyperedge(hyperedge_index, num_nodes, num_edges, degree_mat):
    '''
    :param H: incidence matrix
    :param degree_mat: degree of each pair
    :return:
    '''
    H = torch.sparse_coo_tensor(hyperedge_index, hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges)).to_dense()
    homogeneity = []
    for hyperedge_idx in range(num_edges):
        hyperedge = H[:, hyperedge_idx]  # 获取当前超边的连接情况
        connected_nodes = torch.nonzero(hyperedge).squeeze()  # 获取连接的节点索引

        if len(connected_nodes) > 1:
            homo = 0
            for node_i in connected_nodes:
                for node_j in connected_nodes:
                    if node_i != node_j:
                        homo = homo + degree_mat[node_i, node_j].item()

            homo /= (len(connected_nodes) * (len(connected_nodes) - 1))
            homogeneity.append(sigmoid(homo))
            # homogeneity.append(homo)

        else:
            homogeneity.append(1)

    homogeneity = torch.tensor(homogeneity)

    return homogeneity



def hyperedge_similarity(hyperedge_index, num_nodes, num_edges, manner=''):
    '''
    :param H:
    :return: hyperedge similarity matrix
    '''
    H = torch.sparse_coo_tensor(hyperedge_index, hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges)).to_dense()
    H = H.float()
    if manner == 'jaccard':
        # 计算超边之间的 Jaccard 相似性
        intersection = torch.mm(H.t(), H)  # 计算超边之间的交集
        union = torch.sum(H, dim=0, keepdim=True).t() + torch.sum(H, dim=0, keepdim=True) - intersection  # 计算超边之间的并集
        jaccard_sim = intersection / union
        return jaccard_sim

    if manner == 'cosine':
        dot_product = torch.mm(H.t(), H)  # 计算超边之间的点积
        norm_A = torch.sqrt(torch.sum(H.t() * H.t(), dim=1, keepdim=True))  # 计算每个超边的范数
        norm_B = torch.sqrt(torch.sum(H * H, dim=0, keepdim=True))  # 计算每个超边的范数
        cosine_sim = dot_product / (norm_A * norm_B)
        return cosine_sim



def homo_distance_matrix(homogeneity):
    # 超边数量
    num_hyperedges = len(homogeneity)
    # 初始化距离矩阵
    distance_mat = torch.zeros(num_hyperedges, num_hyperedges)

    # 计算距离矩阵
    for i in range(num_hyperedges):
        for j in range(num_hyperedges):
            # 计算两个同质性的差的绝对值作为距离
            distance = abs(homogeneity[i] - homogeneity[j])
            distance_mat[i][j] = distance

    return distance_mat



def edge_neg_mask(homo_delta):
    prob = homo_delta /homo_delta.max()

    neg_mask = torch.rand_like(homo_delta) < prob

    return neg_mask




if __name__ == '__main__':
    from torch_geometric.utils import degree, to_undirected
    import torch
    import torch.nn.functional as F

    # convert incidence matrix to adj
    H = torch.tensor([[1, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 1, 0],
         [0, 1, 1, 0],
         [0, 1, 0, 1],
         [1, 1, 0, 1]])

    G_n = torch.mm(H, H.T)
    G_e = torch.mm(H.T, H)
    # print(G_n)
    # print(G_e)

    G_n1 = torch.where(G_n != 0, 1, 0)
    G_e1 = torch.where(G_e != 0, 1, 0)
    # print(G_n1)
    # print(G_e1)

    G_n2 = G_n1 - torch.eye(G_n1.size(0))
    G_e2 = G_e1 - torch.eye(G_e1.size(0))
    # print(G_n2)
    # print(G_e2)

    adj = torch.tensor([[0, 1, 0, 0, 1, 0],
                        [1, 0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 1, 1, 0, 1, 0],
                        [1, 0, 0, 1, 0, 1],
                        [0, 0, 0 ,0, 1, 0]])
    adj_invert = 1 - adj - torch.eye(adj.size(0))
    # print(adj_invert)
    G_n2_invert = 1 - G_n2 - torch.eye(G_n2.size(0))
    G_e2_invert = 1 - G_e2 - torch.eye(G_e2.size(0))
    # print(G_e2_invert)


    G_or_adj = torch.logical_or(adj_invert, G_n2_invert)
    G_xor_adj = torch.logical_xor(adj_invert, G_n2)
    # print(G_or_adj)
    # print(G_xor_adj)

    degree_mat = cal_degree_of_each_pair(H)
    homo = cal_homogeneity_hyperedge(H, degree_mat)
    print(homo)

    # print(hyperedge_similarity(H, manner='jaccard'))
    # print(hyperedge_similarity(H, manner='cosine'))


    # 构建距离矩阵
    homo_delta = homo_distance_matrix(homo) # 同质性差值
    print(homo_delta)





























