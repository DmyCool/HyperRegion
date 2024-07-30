from torch.nn import Linear
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
import torch.nn.functional as F


# def glorot(tensor):
#     if tensor is not None:
#         stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
#         tensor.data.uniform_(-stdv, stdv)
#
# def zeros(tensor):
#     if tensor is not None:
#         tensor.data.fill_(0)


def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.lin = Linear(in_ft, out_ft, bias=bias)
#         self.weight = Parameter(torch.Tensor(in_ft, out_ft))
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_ft))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):
        x = self.lin(x)
#         x = x.matmul(self.weight)
#         if self.bias is not None:
#             x = x + self.bias
        x = torch.matmul(edge_index, x)
        return x


class Hypergraph_conv(MessagePassing):
    def __init__(self, in_dim, hid_dim, out_dim, dropout= 0.0, bias=True, row_norm=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.row_norm = row_norm
        self.act = nn.PReLU()

        self.lin_n2e = Linear(in_dim, hid_dim, bias=False, weight_initializer='glorot')
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False, weight_initializer='glorot')


        if bias:
            self.bias_n2e = Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias_n2e', None)
            self.register_parameter('bias_e2n', None)

        self.reset_parameters()


    def reset_parameters(self):
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()
        zeros(self.bias_n2e)
        zeros(self.bias_e2n)

    def forward(self, x, hyperedge_index, num_nodes=None, num_edges=None):
        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        hyperedge_weight = x.new_ones(num_edges)

        node_idx, edge_idx = hyperedge_index



        Dn = scatter_add(hyperedge_weight[hyperedge_index[1]],
                         hyperedge_index[0], dim=0, dim_size=num_nodes)
        De = scatter_add(x.new_ones(hyperedge_index.shape[1]),
                         hyperedge_index[1], dim=0, dim_size=num_edges)

        if self.row_norm:
            Dn = 1.0 / Dn
            Dn[Dn == float('inf')] = 0

            De = 1.0 / De
            De[De == float('inf')] = 0

            norm_n2e = De[edge_idx]
            norm_e2n = Dn[node_idx]


        else:
            Dn = Dn.pow(-0.5)
            Dn[Dn == float('inf')] = 0

            De = De.pow(-0.5)
            De[De == float('inf')] = 0

            norm = De[edge_idx] * Dn[node_idx]
            norm_n2e = norm
            norm_e2n = norm

        x = self.lin_n2e(x)
        e = self.propagate(hyperedge_index, x=x, norm=norm_n2e,
                           size=(num_nodes, num_edges))  # Node to edge

        if self.bias_n2e is not None:
            e = e + self.bias_n2e
        e = self.act(e)
        e = F.dropout(e, p=self.dropout, training=self.training)

        x = self.lin_e2n(e)  # remove ?
        n = self.propagate(hyperedge_index.flip([0]), x=x, norm=norm_e2n,
                           size=(num_edges, num_nodes))  # Edge to node

        if self.bias_e2n is not None:
            n = n + self.bias_e2n

        return n, e  # No act, act

    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j



# v1: X -> XW -> AXW -> norm
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=2, dropout=0., negative_slope=0.2, use_norm=True):
        super().__init__()
        # TODO: bias?
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.use_norm = use_norm

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, x, hyperedge_index):
        vertex = hyperedge_index[0]
        edges = hyperedge_index[1]

        N = x.shape[0]

        # X0 = X # NOTE: reserved for skip connection

        X = self.W(x)

        Xve = X[vertex]  # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce='mean')  # [E, C]

        Xev = Xe[edges]  # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)  # [N, C]
        X = X + Xv

        if self.use_norm:
            X = normalize_l2(X)
            Xe = normalize_l2(Xe)

        # NOTE: concat heads or mean heads?
        # NOTE: normalize here?
        # NOTE: skip concat here?

        return X, Xe




class GATConv(nn.Module):

    def __init__(self, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2, skip_sum=False, use_norm=True):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_v = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_drop = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.skip_sum = skip_sum
        self.use_norm = use_norm
        self.reset_parameters()

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def reset_parameters(self):
        glorot(self.att_v)
        glorot(self.att_e)

    def forward(self, x, hyperedge_index):
        X = x
        vertex = hyperedge_index[0]
        edges = hyperedge_index[1]

        H, C, N = self.heads, self.out_channels, X.shape[0]

        # X0 = X # NOTE: reserved for skip connection

        X0 = self.W(X)
        X = X0.view(N, H, C)

        Xve = X[vertex]  # [nnz, H, C]
        Xe = scatter(Xve, edges, dim=0, reduce='mean')  # [E, H, C]

        alpha_e = (Xe * self.att_e).sum(-1)  # [E, H, 1]
        a_ev = alpha_e[edges]
        alpha = a_ev  # Recommed to use this
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, vertex, num_nodes=N)
        alpha = self.attn_drop(alpha)
        alpha = alpha.unsqueeze(-1)

        Xev = Xe[edges]  # [nnz, H, C]
        Xev = Xev * alpha
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)  # [N, H, C]
        X = Xv
        X = X.view(N, H * C)

        if self.use_norm:
            X = normalize_l2(X)
            Xe = normalize_l2(Xe)

        if self.skip_sum:
            X = X + X0

            # NOTE: concat heads or mean heads?
        # NOTE: skip concat here?

        return X, Xe.squeeze(1)


