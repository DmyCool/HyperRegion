import numpy as np
import torch.nn as nn
import torch
from time import time
import torch.optim as optim
from torch.utils.data import DataLoader

from config import PeMS08_config, STGCN_config, DEVICE, NYC_config
from dataset import NYCDataset
from utils import calculate_adj_matrix, calculate_cheb_polynomial
from utils import compute_val_loss

from model import STGCN


if __name__ == '__main__':
    emb = torch.tensor(np.load('./data/emb.npy'), dtype=torch.float32).to(DEVICE)
    emb_dim = emb.size(1)


    # load configuration info
    ncy_config = NYC_config
    train_config = STGCN_config

    # load graph data
    data_name = ncy_config['dataset_name']
    data_path = ncy_config['processed_signals'] + 'dataset_1_predstep_4.npz'
    train_set = NYCDataset(data_name, data_path, 'train')
    val_set = NYCDataset(data_name, data_path, 'val')

    train_loader, val_loader = DataLoader(train_set, train_config['batch_size'], shuffle=True), \
                               DataLoader(val_set, train_config['batch_size'], shuffle=False)

    # load graph-related matrices
    adj = torch.tensor(np.load('./data/adj_neighbor.npy'), dtype=torch.float32)

    # adj = calculate_adj_matrix(dist_filename=ncy_config['distance_filename'],
    #                            num_of_nodes=ncy_config['num_of_vertices'],
    #                            adj_type=ncy_config['adj_type'],
    #                            sensor_id_file=ncy_config['id_filename'])


    cheb_polynomials = calculate_cheb_polynomial(adj, train_config['Ks'], DEVICE)
    # create model
    stgcn = STGCN(emb_dim=emb_dim, in_dim=ncy_config['num_of_features'], temporal_dim=train_config['temporal_dim'], Kt=train_config['Kt'],
                  Ks=train_config['Ks'], cheb_polynomials=cheb_polynomials, spatial_dim=train_config['spatial_dim'],
                  seq_len=12, pred_step=ncy_config['num_for_predict']).to(DEVICE)

    # parameter init
    for param in stgcn.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.uniform_(param)

    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(stgcn.parameters(), lr=train_config['learning_rate'])

    # training
    best_epoch = 0
    best_val_loss = np.inf

    print("Start train on " + str(DEVICE))

    start_time = time()

    for epoch in range(train_config['epochs']):
        stgcn.train()
        epoch_loss = []
        for batch_idx, batch_data in enumerate(train_loader):
            x, y = batch_data
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_pred = stgcn(x, emb)
            loss = criterion(y_pred, y)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss = np.mean(epoch_loss)
        print('EPOCH: %s, TRAINING LOSS: %.2f, time: %.2fs' % (epoch + 1, epoch_loss, time() - start_time))

        val_loss = compute_val_loss(stgcn, val_loader, criterion, emb, DEVICE)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stgcn.save(stop_epoch=epoch, lr=train_config['learning_rate'], data_name=ncy_config['dataset_name'])
        print()

    print('best epoch:', best_epoch)
