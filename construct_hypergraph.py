import numpy as np

def construct_hypergraph_with_threshold(mat, t):
    '''
    threshold-based hypergraph construction
    :return: 二值关联矩阵
    '''
    simi_mtx = np.copy(mat)

    H = np.where(simi_mtx >= t, 1, 0)

    valid_col = np.where(np.sum(H, axis=0) > 1)[0]

    H = H[:, valid_col]

    return H


def construct_hypergraph_with_knn(mat, k_neig, is_probH=False):
    '''
    construct hypregraph incidence matrix from similarity matrix (src, dst, poi)
    :param mat: node similarity matrix
    :param k: K nearest neighbor
    :param is_probH: probability Vertex-Edge matrix or binary matrix
    :return: N_object X N_hyperedge
    '''

    adj_sp = np.copy(mat)

    # 为每个 region 截取前 k 个correlation 最强的邻居
    for i in range(adj_sp.shape[0]):
        adj_sp[np.argsort(adj_sp[:, i])[:-k_neig], i] = 0
        adj_sp[i, np.argsort(adj_sp[i, :])[:-k_neig]] = 0

    H = np.where(adj_sp != 0, 1, 0)

    # H = H - np.eye(H.shape[0])

    valid_col = np.where(np.sum(H, axis=0) > 1)[0]

    H_prob = adj_sp[:, valid_col]
    H_bin = H[:, valid_col]

    if is_probH:
        H = H_prob
    else:
        H = H_bin

    return H





def construct_graph_with_knn(mat, k):
    '''
    基于knn构建graph，
    传入的mat是mobility flow（双向，非对称矩阵）
    :return: 图邻接矩阵
    '''
    # mtx = mat / np.mean(mat, axis=(0, 1))
    mtx = mat

    # 为每个 region 截取前 k 个 最强的邻居
    for i in range(mtx.shape[0]):
        mtx[np.argsort(mtx[:, i])[:-k], i] = 0
        mtx[i, np.argsort(mtx[i, :])[:-k]] = 0

    # 转换为一阶/（多阶）邻接矩阵，矩阵扩散
    neigh_order = 1
    diffused_adj = np.eye(mtx.shape[1])
    for _ in range(neigh_order): # 一阶邻接矩阵
        diffused_adj = np.matmul(diffused_adj, (mtx + np.eye(mtx.shape[1])))
    diffused_adj = np.where(diffused_adj > 0, 1.0, diffused_adj)
    # diffused_adj = -1e9 * (1.0 - diffused_adj)  #扩散邻接矩阵中小于等于0的元素置为一个极小的负数（这里使用了-1e9）。这样可以将邻接矩阵中的零元素置为一个较小的负数，以避免在神经网络中对这些元素进行操作。

    # np.save('./data/new_test_data/mob_adj_25.npy', diffused_adj)
    return diffused_adj


def construct_graph_with_adj(mat):
    '''
    构建邻接graph
    :return: 图结构邻接矩阵
    '''
    # 添加自环

    adj_sp = mat
    neigh_order = 1
    diffused_adj = np.eye(adj_sp.shape[1])
    for _ in range(neigh_order):  # 一阶邻接矩阵
        diffused_adj = np.matmul(diffused_adj, (adj_sp + np.eye(adj_sp.shape[1])))
    diffused_adj = np.where(diffused_adj > 0, 1.0, diffused_adj)

    return diffused_adj

def merge_graph(G_list):
    '''
    合并多个图
    :return: 图邻接矩阵
    '''
    merge_G = np.sum(G_list, axis=0)
    G = np.where(merge_G > 0, 1.0, merge_G)
    # remove self-loop
    G = G - np.diag(np.diag(G))

    return G

def merge_hypergraph(H_list):
    '''
    合并多个超图
    :return:
    '''
    H = np.concatenate(H_list, axis=1)
    return H






if __name__ == '__main__':
    from load_basic_data import load_basic_data

    data = load_basic_data()

    adj = data['adj']
    mob_flow = data['mob_flow']
    src = data['src_simi']
    dst = data['dst_simi']
    poi = data['poi_simi']


    t1 = {'t_src': 0.94,
         't_dst': 0.95,
         't_poi': 0.94,
         't_chk': 0.80
         }

    t2 = {'t_src': 0.96,
          't_dst': 0.96,
          't_poi': 0.94,
          't_chk': 0.85,
          't_bld': 0.989,
          't_img': 0.28
          }

    '''
    # graph
    k1 = 10
    k2 = 10
    G_mob1 = construct_graph_with_knn(mob_flow, k1)
    G_mob2 = construct_graph_with_knn(mob_flow.T, k1)
    G_poi = construct_graph_with_knn(poi, k2)

    G = np.logical_or(G_poi, np.logical_or(adj, np.logical_or(G_mob1, G_mob2)))
    G = G - np.eye(G.shape[0])
    np.save('./data/hypergraph/G_mob12_10_poi_10_adj.npy', G)

    '''



    k1 = 5
    k2 = 5
    H_mob1 = construct_graph_with_knn(mob_flow, k1 + 1)
    H_mob1 = construct_hypergraph_with_threshold(H_mob1, 1)
    H_mob2 = construct_graph_with_knn(mob_flow.T, k1 + 1)
    H_mob2 = construct_hypergraph_with_threshold(H_mob2, 1)

    # H_poi = construct_hypergraph_with_threshold(poi, 0.85)
    H_poi = construct_hypergraph_with_knn(poi, k2 + 1)

    adj = adj + np.eye(adj.shape[0])  # 添加自环

    H_list = [H_mob1, H_mob2, H_poi]  # (180, 531)
    H = merge_hypergraph(H_list)
    print(H.shape)

    H_unique, indices = np.unique(H, axis=1, return_index=True)  # (180, 483)
    print(H_unique.shape)

    np.save('./data/hypergraph/H_mob12_5_poi_5.npy', H)
    np.save('./data/hypergraph/H_mob12_5_poi_5_u.npy', H_unique)






    '''
    # hypergraph
    k1 = 15
    k2 = 10

    H_mob1 = construct_graph_with_knn(mob_flow, k1 + 1)
    H_mob1 = construct_hypergraph_with_threshold(H_mob1, 1)
    H_mob2 = construct_graph_with_knn(mob_flow.T, k1 + 1)
    H_mob2 = construct_hypergraph_with_threshold(H_mob2, 1)

    H_src = construct_hypergraph_with_knn(src, k1 + 1)
    H_dst = construct_hypergraph_with_knn(dst, k1 + 1)
    H_poi = construct_hypergraph_with_knn(poi, k2 + 1)

    H_list = [H_mob1, H_mob2, H_poi, adj]  #(180, 531)
    H = merge_hypergraph(H_list)
    # print(H.shape)

    H_unique, indices = np.unique(H, axis=1, return_index=True)  # (180, 483)
    # print(H_unique.shape)

    # np.save('./data/hypergraph/H_mob12_15_poi_10_adj_u.npy', H_unique)
    '''