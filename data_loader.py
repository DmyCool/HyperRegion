import numpy as np
from construct_hypergraph import *
import os.path as osp
import torch
from torch_scatter import scatter_add
from tool_func import preprocess_features


class DataLoader(object):
    def __init__(self, data_dir, device=None):
        self.data_dir = data_dir
        self.device = device

        self.load_data()
        self.preprocess_data()

    def load_data(self):

        self.features = np.load(osp.join(self.data_dir, 'img_feat.npy'))
        # self.features = preprocess_features(self.features)

        self.G = np.load(osp.join(self.data_dir, 'G_mob12_10_adj.npy'))

        self.H = np.load(osp.join(self.data_dir, 'H_mob12_10_poi_10.npy'))


    def preprocess_data(self):
        self.features = torch.as_tensor(self.features, dtype=torch.float32)

        self.hyperedge_index = torch.tensor(self.H).to_sparse().indices()
        self.edge_index = torch.tensor(self.G).to_sparse().indices()

        self.num_nodes = int(self.hyperedge_index[0].max()) + 1
        self.num_edges = int(self.hyperedge_index[1].max()) + 1


        weight = torch.ones(self.num_edges)
        Dn = scatter_add(weight[self.hyperedge_index[1]], self.hyperedge_index[0], dim=0, dim_size=self.num_nodes)
        De = scatter_add(torch.ones(self.hyperedge_index.shape[1]), self.hyperedge_index[1], dim=0, dim_size=self.num_edges)

        # print('=============== Dataset Stats ===============')
        # print(f'features size: [{self.features.shape[0]}, {self.features.shape[1]}]')
        # print(f'num nodes: {self.num_nodes}')
        # print(f'num edges: {self.num_edges}')
        # print(f'num connections: {self.hyperedge_index.shape[1]}')
        # print(f'avg hyperedge size: {torch.mean(De).item():.2f}+-{torch.std(De).item():.2f}')
        # print(f'avg hypernode degree: {torch.mean(Dn).item():.2f}+-{torch.std(Dn).item():.2f}')
        # print(f'max node size: {Dn.max().item()}')
        # print(f'max edge size: {De.max().item()}')
        # print('=============================================')

        self.to(self.device)

    def to(self, device: str):
        self.features = self.features.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        self.edge_index = self.edge_index.to(device)
        self.device = device
        return self





if __name__ == '__main__':
    data = DataLoader('./data/new_test_data1/')
    print(data.hyperedge_index)

