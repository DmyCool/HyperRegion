import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from layers import *
import numpy as np


class CatFusion(nn.Module):
    def __init__(self):
        super(CatFusion, self).__init__()

    def forward(self, g_embed, h_embed):
        fusion_embed = torch.cat((g_embed, h_embed), dim=-1)
        return fusion_embed


class WeightedSumFusion(nn.Module):
    def __init__(self, in_channels):
        super(WeightedSumFusion, self).__init__()
        self.weight_g = nn.Parameter(torch.Tensor([0.2]))
        self.weight_h = nn.Parameter(torch.Tensor([0.8]))

    def forward(self, g_embed, h_embed):
        fusion_embed = self.weight_g * g_embed + self.weight_h * h_embed
        return fusion_embed


class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.fc = nn.Linear(in_channels * 2, in_channels)
        self.attention_weights = nn.Parameter(torch.Tensor(in_channels * 2, 1))
        nn.init.xavier_uniform_(self.attention_weights.data, gain=1.414)

    def forward(self, g_embed, h_embed):
        concat_features = torch.cat([g_embed, h_embed], dim=-1)
        attention_scores = torch.sigmoid(torch.matmul(concat_features, self.attention_weights))
        fusion_embed = attention_scores * g_embed + (1 - attention_scores) * h_embed
        return fusion_embed


class MLPFusion(nn.Module):
    def __init__(self, in_channels):
        super(MLPFusion, self).__init__()
        self.fc1 = nn.Linear(in_channels * 2, in_channels)
        self.fc2 = nn.Linear(in_channels, in_channels)

    def forward(self, g_embed, h_embed):
        concat_features = torch.cat([g_embed, h_embed], dim=-1)
        hidden = F.relu(self.fc1(concat_features))
        fusion_embed = self.fc2(hidden)
        return fusion_embed


class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super(GatedFusion, self).__init__()
        self.gate = nn.Linear(in_channels * 2, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g_embed, h_embed):
        concat_features = torch.cat([g_embed, h_embed], dim=-1)
        gate = self.sigmoid(self.gate(concat_features))
        fusion_embed = gate * g_embed + (1 - gate) * h_embed
        return fusion_embed


class FusionLayer(nn.Module):
    def __init__(self, in_channels, fusion_type):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if fusion_type == 'cat':
            self.fusion = CatFusion()
        elif fusion_type == 'weighted':
            self.fusion = WeightedSumFusion(in_channels)
        elif fusion_type == 'attention':
            self.fusion = AttentionFusion(in_channels)
        elif fusion_type == 'mlp':
            self.fusion = MLPFusion(in_channels)
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(in_channels)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, g_embed, h_embed):
        return self.fusion(g_embed, h_embed)






###################################################################################

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GCNConv(self.in_channels, 2 * self.out_channels)
        self.conv2 = GCNConv(2 * self.out_channels, self.out_channels)
        # self.conv3 = GCNConv(self.out_channels, self.out_channels)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)

        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv3(x, edge_index)


        return x




class HypergraphEncoder(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_layers=2):
        '''
        :param in_channels: input_feature
        :param hid_channels: edge_dim
        :param out_channels: node_dim
        :param num_layers:
        '''
        super(HypergraphEncoder, self).__init__()

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.act = nn.PReLU()

        # hconv layer
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(Hypergraph_conv(self.in_channels, self.hid_channels, self.out_channels))
        else:
            # first layer
            self.convs.append(Hypergraph_conv(self.in_channels, self.hid_channels, self.out_channels))
            # middle layer
            for _ in range(num_layers - 2):
                self.convs.append(Hypergraph_conv(self.out_channels, self.hid_channels, self.out_channels))
            # last layer
            self.convs.append(Hypergraph_conv(self.out_channels, self.hid_channels, self.out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, x, hyperedge_index, num_nodes, num_edges):
        for i in range(self.num_layers - 1):
            x, e = self.convs[i](x, hyperedge_index, num_nodes, num_edges)
            x = self.act(x)
            x = F.dropout(x, p=0.2, training=self.training)

        x, e = self.convs[-1](x, hyperedge_index, num_nodes, num_edges)
        x = self.act(x)

        return x, e




class HyperRegionCL(nn.Module):
    def __init__(self, in_dim, hedge_dim, hnode_dim, node_dim, proj_dim, n_layers):
        super(HyperRegionCL, self).__init__()

        self.in_dim = in_dim
        self.hedge_dim = hedge_dim
        self.hnode_dim = hnode_dim
        self.node_dim = node_dim
        self.proj_dim = proj_dim
        self.n_layers = n_layers

        self.graph_encoder = GraphEncoder(self.in_dim, self.node_dim)
        self.hypergraph_encoder = HypergraphEncoder(self.in_dim, self.hedge_dim, self.hnode_dim, self.n_layers)

        self.fc1_n = nn.Linear(self.node_dim, self.proj_dim)
        self.fc2_n = nn.Linear(self.proj_dim, self.node_dim)

        self.fc1_hn = nn.Linear(self.hnode_dim, self.proj_dim)
        self.fc2_hn = nn.Linear(self.proj_dim, self.hnode_dim)

        self.fc1_he = nn.Linear(self.hedge_dim, self.proj_dim)
        self.fc2_he = nn.Linear(self.proj_dim, self.hedge_dim)

        self.disc = nn.Bilinear(self.hnode_dim, self.hedge_dim, 1)


        # self.fusion = FusionGate(self.node_dim)
        self.fusion = FusionLayer(self.node_dim, fusion_type='weighted')


    def forward(self, x_n, x_hn, edge_index, hyperedge_index, num_nodes, num_edges):

        n = self.graph_encoder(x_n, edge_index)

        node_idx = torch.arange(0, num_nodes, device=x_hn.device)
        edge_idx = torch.arange(num_edges, num_edges + num_nodes, device=x_hn.device)
        self_loop = torch.stack([node_idx, edge_idx])
        self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
        hn, he = self.hypergraph_encoder(x_hn, self_loop_hyperedge_index, num_nodes, num_edges + num_nodes)

        return n, hn, he[:num_edges]



    def fusion_forward(self, n, hn):
        f = self.fusion(n, hn)
        return f







    def n_projection(self, z):
        out = self.fc2_n(F.elu(self.fc1_n(z)))
        return out

    def hn_projection(self, z):
        out = self.fc2_hn(F.elu(self.fc1_hn(z)))
        return out

    def he_projection(self, z):
        return self.fc2_he(F.elu(self.fc1_he(z)))

    def f(self, x, tau):
        return torch.exp(x / tau)

    def cosine_similarity(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def disc_similarity(self, z1, z2):
        return torch.sigmoid(self.disc(z1, z2)).squeeze()

    # #####################################################################
    # graph-loss
    def __semi_loss_g(self, h1, h2, tau):
        # between_sim = self.f(self.cosine_similarity(h1, h2), tau)
        # return -torch.log(between_sim.diag() / between_sim.sum(1))
        refl_simi = self.f(self.cosine_similarity(h1, h1), tau)
        between_simi = self.f(self.cosine_similarity(h1, h2), tau)
        return -torch.log(between_simi.diag() / (refl_simi.sum(1) + between_simi.sum(1) - refl_simi.diag()))

    def __loss_g(self, z1, z2, tau):
        l1 = self.__semi_loss_g(z1, z2, tau)
        l2 = self.__semi_loss_g(z2, z1, tau)
        loss = (l1 + l2) * 0.5
        loss = loss.mean()
        return loss

    def g_node_loss(self, z1, z2, tau):

        loss = self.__loss_g(z1, z2, tau)
        return loss

##################################################################################################################

    # hypergraph-loss
    def __semi_loss(self, h1, h2, tau, neg_mask):
        if neg_mask is None:
            between_sim = self.f(self.cosine_similarity(h1, h2), tau)
            return -torch.log(between_sim.diag() / between_sim.sum(1))

        else:
            ## refl_sim = self.f(self.cosine_similarity(h1, h1), tau)
            between_sim = self.f(self.cosine_similarity(h1, h2), tau)
            masked_between_sim = between_sim * neg_mask
            return -torch.log(between_sim.diag() / (masked_between_sim.sum(1) - masked_between_sim.diag() + between_sim.diag()))

            # between_sim = self.f(self.cosine_similarity(h1, h2), tau)
            # pos_sum = (between_sim * torch.logical_not(neg_mask)).sum(1)
            # neg_sum = (between_sim * neg_mask).sum(1)
            # return -torch.log(pos_sum / (pos_sum + neg_sum))

    def __loss(self, z1, z2, tau, neg_mask):
        l1 = self.__semi_loss(z1, z2, tau, neg_mask)
        l2 = self.__semi_loss(z2, z1, tau, neg_mask)
        loss = (l1 + l2) * 0.5
        loss = loss.mean()
        return loss


    def hg_node_loss(self, n1, n2, tau, neg_mask):
        loss = self.__loss(n1, n2, tau, neg_mask)
        return loss

    def hg_hyperedge_loss(self, e1, e2, tau, neg_mask):
        loss = self.__loss(e1, e2, tau, neg_mask)
        return loss

    def hg_membership_loss(self, n, e, hyperedge_index, tau):
        e_perm = e[torch.randperm(e.size(0))]
        n_perm = n[torch.randperm(n.size(0))]
        pos = self.f(self.disc_similarity(n[hyperedge_index[0]], e[hyperedge_index[1]]), tau)
        neg_n = self.f(self.disc_similarity(n[hyperedge_index[0]], e_perm[hyperedge_index[1]]), tau)
        neg_e = self.f(self.disc_similarity(n_perm[hyperedge_index[0]], e[hyperedge_index[1]]), tau)
        loss_n = -torch.log(pos / (pos + neg_n))
        loss_e = -torch.log(pos / (pos + neg_e))
        loss_n = loss_n[~torch.isnan(loss_n)]
        loss_e = loss_e[~torch.isnan(loss_e)]
        loss = loss_n + loss_e
        loss = loss.mean()

        return loss

#######################################################################################################
    def cross_loss(self, n, hn, tau, neg_mask):
        n_hn_loss1 = self.__semi_loss_g(n, hn, tau)
        n_hn_loss2 = self.__semi_loss_g(hn, n, tau)
        loss = (n_hn_loss1 + n_hn_loss2) * 0.5
        loss = loss.mean()
        return loss































