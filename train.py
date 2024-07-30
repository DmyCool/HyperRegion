import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import os.path
import matplotlib.pyplot as plt
import torch
import yaml
import numpy as np
from data_loader import DataLoader
from model import HyperRegionCL
from aug import *
from do_tasks import do_tasks
from sklearn.model_selection import GridSearchCV
from tool_func import *
from  sklearn.model_selection import ParameterGrid

import random



def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(node_neg_mask, hedge_neg_mask):
    features, hyperedge_index, edge_index, num_nodes, num_edges = \
        data.features, data.hyperedge_index, data.edge_index, data.num_nodes, data.num_edges

    model.train()
    optimizer.zero_grad()

    # feature aug(graph, hypergraph)
    n_x1 = drop_feature(features, params['drop_feature_rate'])
    n_x2 = drop_feature(features, params['drop_feature_rate'])

    hn_x1 = drop_feature(features, params['drop_feature_rate'])
    hn_x2 = drop_feature(features, params['drop_feature_rate'])


    # graph edge aug
    edge_index1 = dropout_adj(edge_index, p=params['drop_adj_rate'])[0]
    edge_index2 = dropout_adj(edge_index, p=params['drop_adj_rate'])[0]

    # hypergraph edge aug
    hyperedge_index1 = drop_incidence(hyperedge_index, params['drop_incidence_rate'])
    hyperedge_index2 = drop_incidence(hyperedge_index, params['drop_incidence_rate'])
    # hyperedge_index1 = remove_hyperedge_by_weight1(hyperedge_index, num_nodes, num_edges, 0.2)
    # hyperedge_index2 = remove_hyperedge_by_weight1(hyperedge_index, num_nodes, num_edges, 0.2)


    ### check valid node and edge
    hg_node_mask1, hg_edge_mask1 = valid_node_edge_mask(hyperedge_index1, num_nodes, num_edges)
    hg_node_mask2, hg_edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)
    node_mask = hg_node_mask1 & hg_node_mask2
    edge_mask = hg_edge_mask1 & hg_edge_mask2

    # graph and hypergraph encoder
    n1, hn1, he1 = model(n_x1, hn_x1, edge_index1, hyperedge_index1, num_nodes, num_edges)
    n2, hn2, he2 = model(n_x2, hn_x2, edge_index2, hyperedge_index2, num_nodes, num_edges)

    # projection
    n1, n2 = model.n_projection(n1), model.n_projection(n2)
    hn1, hn2 = model.hn_projection(hn1), model.hn_projection(hn2)
    he1, he2 = model.he_projection(he1), model.he_projection(he2)


    # graph/hypergraph loss
    loss_n = model.g_node_loss(n1, n2, params['tau_n'])

    loss_hn = model.hg_node_loss(hn1, hn2, params['tau_hn'], node_neg_mask)

    if hedge_neg_mask is None:
        loss_he = model.hg_hyperedge_loss(he1[edge_mask], he2[edge_mask], params['tau_he'], hedge_neg_mask)
    else:
        loss_he = model.hg_hyperedge_loss(he1[edge_mask], he2[edge_mask], params['tau_he'], hedge_neg_mask[edge_mask][:, edge_mask])

    # membership
    masked_index1 = hyperedge_index_masking(hyperedge_index, None, hg_edge_mask1, num_nodes, num_edges)
    masked_index2 = hyperedge_index_masking(hyperedge_index, None, hg_edge_mask2, num_nodes, num_edges)
    loss_hm1 = model.hg_membership_loss(hn1, he2[hg_edge_mask2], masked_index2, params['tau_hm'])
    loss_hm2 = model.hg_membership_loss(hn2, he1[hg_edge_mask1], masked_index1, params['tau_hm'])
    loss_hm = (loss_hm1 + loss_hm2) * 0.5

    loss_hg = params['w_hn'] * loss_hn + params['w_he'] * loss_he + params['w_hm'] * loss_hm

    '''不同融合策略 fusion strategies (fs) '''
    # fs1: 图与超图直接对比
    # loss_cross = (model.cross_loss(n1, hn1, 0.5, None) + model.cross_loss(n2, hn2, 0.5, None))*0.5  # bad

    # fs2: 图与图求和，超图与超图求和 #68.9831， 0.746
    # loss_cross = model.cross_loss((n1 + n2), (hn1 + hn2), 0.5, None)

    # fs3: 图与图拼接，超图与超图拼接 # 69.1701， 0.73
    # loss_cross = model.cross_loss(torch.cat((n1, n2)), torch.cat((hn1, hn2)), 0.5, None)


    # fs4: 任意组合增强对比 crime=68.502
    # loss_cross = (model.cross_loss(n1, hn1, 0.5, None) + model.cross_loss(n2, hn2, 0.5, None) +
    #               model.cross_loss(n1, hn2, 0.5, None) + model.cross_loss(n2, hn1, 0.5, None)) * 0.25

    # fs5: 任意拼接对比 crime=68.95
    # (n1||hn1)<-->(n2||hn2) + (n1||hn2)<-->(n2||hn1)
    # ca1, ca2 = model.fusion_forward(n1, hn1), model.fusion_forward(n2, hn2)
    # ca3, ca4 = model.fusion_forward(n1, hn2), model.fusion_forward(n2, hn1)
    # fs5-1:
    # loss_cross = model.cross_loss(ca1, ca2, 0.5, None)
    # fs5-2:
    # loss_cross = (model.cross_loss(ca1, ca2, 0.5, None) + model.cross_loss(ca3, ca4, 0.5, None)) * 0.5

    # fs6: 任意加权对比
    # (a * n1 + (1-a) * hn1)<-->(a * n2 + (1-a) * hn2)
    alpha = 0.2
    f1 = alpha * n1 + (1 - alpha) * hn1
    f2 = alpha * n2 + (1 - alpha) * hn2
    # f3 = 0.2 * n1 + 0.8 * hn2
    # f4 = 0.2 * n2 + 0.8 * hn1
    # fs6-1: (0.2, 0.8)
    loss_cross = model.cross_loss(f1, f2, params['tau_f'], None)  # [crime=66.61, 0.5] [65.087, crime=0.2]
    # fs6-2: (0.2, 0.8)
    # loss_cross = (model.cross_loss(f1, f2, 0.5, None) + model.cross_loss(f3, f4, 0.5, None)) * 0.5


    # fs7: 注意力融合: Attention或者加权融合可以用来对比，但是最终的下游任务一定要是cat(n,hn)，其他的效果都不好,
    # fa1 = model.fusion_forward(n1, hn1)
    # fa2 = model.fusion_forward(n2, hn2)
    # fa3 = model.fusion_forward(n1, hn2)
    # fa4 = model.fusion_forward(n2, hn1)
    # fs7-1:
    # loss_cross = model.cross_loss(fa1, fa2, 0.5, None)
    # fs7_2:
    # loss_cross = (model.cross_loss(fa1, fa2, 0.5, None) + model.cross_loss(fa3, fa4, 0.5, None)) * 0.5  # 65.68-cat

    # print(loss_n, loss_hg, loss_cross)

    # all loss
    loss = loss_n + params['w_hg'] * loss_hg + params['w_cross'] * loss_cross



    loss.backward()
    optimizer.step()

    return loss.item()



@torch.no_grad()
def evaluation(flag_save=False):
    model.eval()
    n, hn, he = model(data.features, data.features, data.edge_index, data.hyperedge_index, data.num_nodes, data.num_edges)

    if flag_save:
        np.save('./data/embeddings/n.npy', n.cpu().detach().numpy())
        np.save('./data/embeddings/hn.npy', hn.cpu().detach().numpy())


    # np.save('./data/embeddings/he.npy', he.cpu().detach().numpy())

    # emb = n + hn  # sum -----> perform average
    # emb = n - hn  # difference
    # emb = n * hn  # multiply
    # emb = (n + hn) * 0.5  # mean ---> perform bad
    # emb = torch.min(n, hn)  # min
    # emb = torch.max(n, hn)  # max
    # emb = 0.4 * n + 0.6 * hn  # weighted

    emb = torch.cat((n, hn), dim=1)  # cat  ----> perform better

    emb = emb.cpu().detach().numpy()

    crime_mae, crime_rmse, crime_r2, check_mae, check_rmse, check_r2, nmi, ars = do_tasks(emb)


    return crime_mae, crime_rmse, crime_r2, check_mae, check_rmse, check_r2, nmi, ars, emb, he



def negative_sel(data, manner=''):
    adj = data.G
    H = data.H

    if manner == 'k_order_neg':
        # k阶邻接矩阵以内节点作为负样本
        adj_k = k_order_adj(adj, neigh_order=6)
        adj_k = torch.as_tensor(adj_k).to(args.device)
        neg_mask = adj_k
        return neg_mask, None

    if manner == 'one_order_exclude':
        # 一阶邻接矩阵以外的节点作为负样本
        adj = torch.as_tensor(adj).to(args.device)
        neg_mask = 1 - adj - torch.eye(adj.size(0)).to(args.device)
        return neg_mask, None

    if manner == 'mix_adj_H':
        # 节点邻接矩阵与超图节点邻接矩阵进行or/xor后作为超节点负样本，超边邻接矩阵作为超边负样本
        node_adj, hedge_adj = construct_G_from_H(torch.tensor(H))
        node_adj, hedge_adj = node_adj.to(args.device), hedge_adj.to(args.device)

        adj = torch.as_tensor(adj).to(args.device)
        adj_invert = 1 - adj - torch.eye(adj.size(0)).to(args.device)  # 一阶邻接矩阵以外的节点作为负样本

        node_adj_invert = 1 - node_adj - torch.eye(node_adj.size(0)).to(args.device)  # 一阶邻接矩阵以外的节点作为负样本
        hedge_adj_invert = 1 - hedge_adj - torch.eye(hedge_adj.size(0)).to(args.device)  # 一阶邻接矩阵以外的超边作为负样本

        node_neg_mask = torch.logical_or(adj_invert, node_adj_invert)
        hedge_neg_mask = hedge_adj_invert

        # node_neg_mask1 = torch.logical_not((node_adj * adj) + torch.eye(adj.size(0)).to(args.device))

        return node_neg_mask, hedge_neg_mask










if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hypergraph region representation learning')
    parser.add_argument('--data_dir', type=str, default='./data/hypergraph/', help='data path')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--seeds', type=int, default=2, help='seed for randomness')
    parser.add_argument('--model_dir', type=str, default='./data/checkpoints/', help='Path for saving the trained model')
    parser.add_argument('--emb_dir', type=str, default='./data/embeddings/', help='Path for saving the embeddings')
    parser.add_argument('--optimizer', type=str, default='AdamW')
    args = parser.parse_args()

    params = yaml.safe_load(open('config.yaml'))

    data = DataLoader(args.data_dir, args.device)
    data = data.to(args.device)

    # node feature ablation
    # data.features = torch.randn_like(data.features).to(args.device)  # 随机化节点特征


    # mask
    ## homo_mask
    # degree_mat = cal_degree_of_each_pair(data.hyperedge_index, data.num_nodes, data.num_edges)
    # homo = cal_homogeneity_hyperedge(data.hyperedge_index, data.num_nodes, data.num_edges, degree_mat)
    # homo_delta = homo_distance_matrix(homo)  # 同质性差越大，越容易成为负样本
    # hedge_neg_mask = edge_neg_mask(homo_delta)
    # hedge_neg_mask = hedge_neg_mask.to(args.device)
    # node_neg_mask, _ = negative_sel(data, manner='one_order_exclude')

    ## edge_simi mask
    # hedge_simi = hyperedge_similarity(data.hyperedge_index, data.num_nodes, data.num_edges, manner='jaccard')
    # hedge_neg_mask = torch.where(hedge_simi <= 0.70, 1, 0)
    # hedge_neg_mask = hedge_neg_mask.to(args.device)
    # node_neg_mask, _ = negative_sel(data, manner='one_order_exclude')

    # node_neg_mask = None
    node_neg_mask, _ = negative_sel(data, manner='mix_adj_H') #(one_order_exclude, mix_adj_H)
    hedge_neg_mask = None


    # node_neg_mask, hedge_neg_mask = negative_sel(data, manner='mix_adj_H')


    fix_seed(args.seeds)

    model_metrics_list = []
    best_epoch_list = []

    for run in range(1):
        model = HyperRegionCL(data.features.shape[1], params['he_dim'], params['hn_dim'], params['n_dim'], params['proj_dim'], n_layers=2).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        # optimizer
        # if args.optimizer == 'AdamW':
        #     optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        # elif args.optimizer == 'Adam':
        #     optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        # elif args.optimizer == 'SGD':
        #     optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'], momentum=0.9)



        ## train
        cnt_wait = 0
        best_loss = 1e9
        best_epoch_emb = 0
        best_epoch_model = 0
        patience = 40

        best_rmse = 10000
        best_mae = 10000
        best_r2 = 0
        best_emb = 0



        for epoch in range(params['epochs']):
            # training
            loss = train(node_neg_mask, hedge_neg_mask)

            ## Part 4: evaluation

            print("Epoch {}, Loss {}".format(epoch, loss))
            crime_mae, crime_rmse, crime_r2, check_mae, check_rmse, check_r2, nmi, ars, emb, he = evaluation()
            print('crime:', crime_mae, crime_rmse, crime_r2)
            print('check:', check_mae, check_rmse, check_r2)
            print('land:', nmi, ars)

            # check min, ref HREP
            if crime_mae < best_mae and crime_rmse < best_rmse and best_r2 < crime_r2:
                best_mae = crime_mae
                best_rmse = crime_rmse
                best_r2 = crime_r2
                best_epoch_emb = epoch
                np.save(f'./data/embeddings/emb_{run}_{best_epoch_emb}.npy', emb)
                np.save(f'./data/embeddings/he_{run}_{best_epoch_emb}.npy', he.cpu().detach().numpy())

            # if best_r2 < crime_r2:
            #     best_r2 = crime_r2
            #     best_epoch_emb = epoch
            #     np.save(f'./data/embeddings/emb_{best_epoch_emb}.npy', emb)




            # best epoch and early stopping
            if loss < best_loss:
                best_loss = loss
                best_epoch_model = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), args.model_dir + 'model.pkl')


            else:
                cnt_wait += 1

            if cnt_wait == patience:
                print('=========Early Stopping')
                break

            # print('')
            # print(f'=====best_loss: {best_loss}, best_epoch:{best_epoch}=====')

        print('best_rmse:', best_rmse)
        print('best_mae:', best_mae)
        print('best_r2:', best_r2)
        print('best_epoch_emb:', best_epoch_emb)

        best_epoch_list.append(best_epoch_emb)



        print('========================evaluation==============')
        model.load_state_dict(torch.load(args.model_dir + 'model.pkl'))
        crime_mae, crime_rmse, crime_r2, check_mae, check_rmse, check_r2, nmi, ars, _, _ = evaluation()
        # print('crime:', crime_mae, crime_rmse, crime_r2)
        # print('check:', check_mae, check_rmse, check_r2)
        # print('land:', nmi, ars)
        all_metrics = [crime_mae, crime_rmse, crime_r2, check_mae, check_rmse, check_r2, nmi, ars]
        print('all metrics', crime_mae, crime_rmse, crime_r2, check_mae, check_rmse, check_r2, nmi, ars)
        model_metrics_list.append(all_metrics)

        # crime_mae, crime_rmse, crime_r2, check_mae, check_rmse, check_r2, nmi, ars = evaluation()


    # print(best_epoch_list)
    # for i in model_metrics_list:
    #     print(i)

