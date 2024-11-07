import torch
import numpy as np
from torch.utils.data import DataLoader

from config import NYC_config, STGCN_config, DEVICE
from dataset import NYCDataset

from utils import calculate_adj_matrix, calculate_cheb_polynomial
from utils import metric, step_metric

from model import STGCN

emb = torch.tensor(np.load('./data/emb.npy'), dtype=torch.float32).to(DEVICE)
emb_dim = emb.size(1)

# load configuration info
nyc_config = NYC_config
train_config = STGCN_config

# load graph data
data_name = nyc_config['dataset_name']
data_path = nyc_config['processed_signals'] + 'dataset_1_predstep_4.npz'
test_set = NYCDataset(data_name, data_path, 'test')
test_loader = DataLoader(test_set, train_config['batch_size'], shuffle=False)

# load graph-related matrices
adj = torch.tensor(np.load('./data/adj_neighbor.npy'), dtype=torch.float32)
# adj = calculate_adj_matrix(dist_filename=nyc_config['distance_filename'],
#                            num_of_nodes=nyc_config['num_of_vertices'],
#                            adj_type='connectivity', sensor_id_file=nyc_config['id_filename'])

cheb_polynomials = calculate_cheb_polynomial(adj, train_config['Ks'], DEVICE)

# create model
stgcn = STGCN(emb_dim=emb_dim, in_dim=nyc_config['num_of_features'], temporal_dim=train_config['temporal_dim'], Kt=train_config['Kt'],
              Ks=train_config['Ks'], cheb_polynomials=cheb_polynomials, spatial_dim=train_config['spatial_dim'],
              seq_len=12, pred_step=nyc_config['num_for_predict']).to(DEVICE)

stgcn.load('checkpoints/STGCN_1104_10_25_epoch_470_lr_0.001_nyc_pickup.pth')
stgcn.eval()

all_true = []
all_pred = []
with torch.no_grad():
    for batch_idx, batch_data in enumerate(test_loader):
        x, y = batch_data
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = stgcn(x, emb)
        all_true.append(y)
        all_pred.append(y_pred)
trues = torch.cat(all_true, dim=0)
preds = torch.cat(all_pred, dim=0)
# print(trues.shape)

mae, mape, rmse = metric(trues, preds)
print('MAE:', mae, '\t MAPE(%):', mape, '\t RMSE:', rmse)

steps_mae, steps_mape, steps_rmse = step_metric(trues, preds)
print('step_MAE:', steps_mae, '\nstep_MAPE(%):', steps_mape, '\nstep_RMSE:', steps_rmse)
