import torch
import numpy as np
import pandas as pd


def calculate_adj_matrix(dist_filename, num_of_nodes, adj_type, sensor_id_file=None):
    adj_mat = torch.zeros((num_of_nodes, num_of_nodes), dtype=torch.float)
    dist_df = pd.read_csv(dist_filename)
    if sensor_id_file:
        with open(sensor_id_file, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}

        if adj_type == 'connectivity':
            for _, row in dist_df.iterrows():
                i, j, dist = int(row['from']), int(row['to']), row['cost']
                adj_mat[id_dict[i], id_dict[j]] = 1
                adj_mat[id_dict[j], id_dict[i]] = 1
        elif adj_type == 'distance':
            for _, row in dist_df.iterrows():
                i, j, dist = int(row['from']), int(row['to']), row['cost']
                adj_mat[id_dict[i], id_dict[j]] = dist
                adj_mat[id_dict[j], id_dict[i]] = dist
        else:
            raise ValueError("type_ error, must be "
                             "connectivity or distance!")
    else:
        if adj_type == 'connectivity':
            for _, row in dist_df.iterrows():
                i, j, dist = int(row['from']), int(row['to']), row['cost']
                adj_mat[i, j] = 1
                adj_mat[j, i] = 1
            adj_mat += torch.eye(num_of_nodes)
        elif adj_type == 'distance':
            for _, row in dist_df.iterrows():
                i, j, dist = int(row['from']), int(row['to']), row['cost']
                adj_mat[i, j] = 1. / dist
                adj_mat[j, i] = 1. / dist
        else:
            raise ValueError("type_ error, must be "
                             "connectivity or distance!")
    return adj_mat


def calculate_self_loop_symmetric_adj(adj):
    """
    :param adj: (N, N)
    :return: D^{-0.5}(adj + I)D^{-0.5}
    """
    adj = adj + torch.eye(adj.size(0))  # add self-loop
    degree_mat = torch.diag(adj.sum(dim=0))
    degree_inv_sqrt = degree_mat ** (-0.5)
    degree_inv_sqrt.masked_fill_(degree_inv_sqrt == float('inf'), 0)
    return degree_inv_sqrt.mm(adj).mm(degree_inv_sqrt)


def calculate_symmetric_normalized_laplacian(adj):
    degree_mat = torch.diag(adj.sum(dim=0))
    degree_inv_sqrt = degree_mat ** (-0.5)
    degree_inv_sqrt.masked_fill_(degree_inv_sqrt == float('inf'), 0)
    return torch.eye(adj.size(0)) - degree_inv_sqrt.mm(adj).mm(degree_inv_sqrt)


def calculate_scaled_laplacian(adj, lambda_max=2):
    assert adj.shape[0] == adj.shape[1]
    degree_mat = torch.diag(adj.sum(dim=0))
    laplacian = degree_mat - adj
    # laplacian = calculate_normalized_laplacian(adj)
    if lambda_max is None:
        evals = torch.linalg.eigvals(laplacian).real
        lambda_max = torch.max(evals)
    scaled_laplacian = (2 * laplacian) / lambda_max - torch.eye(adj.size(0))
    return scaled_laplacian


def calculate_cheb_polynomial(adj, K, DEVICE):
    scaled_laplacian = calculate_scaled_laplacian(adj, None).to(DEVICE)
    num_nodes = scaled_laplacian.shape[0]
    cheb_polynomials = [torch.eye(num_nodes, device=DEVICE)]
    if K == 1:
        return cheb_polynomials
    else:
        cheb_polynomials.append(scaled_laplacian)
        if K == 2:
            return cheb_polynomials
        else:
            for i in range(2, K):
                cheb_polynomials.append(2 * torch.mm(scaled_laplacian,
                                                     cheb_polynomials[i - 1]) - cheb_polynomials[i - 2])
    return cheb_polynomials


def compute_val_loss(model, val_loader, criterion, x_emb, device):
    model.eval()
    with torch.no_grad():
        loss_of_batches = []
        for batch_idx, batch_data in enumerate(val_loader):
            x, y = batch_data
            x, y = x.to(device), y.to(device)
            y_pred = model(x, x_emb)
            loss = criterion(y_pred, y)
            loss_of_batches.append(loss.item())
            # if batch_idx % 20 == 0:
            #     print('validation batch %s / %s, loss: %.2f' % (batch_idx + 1, num_val_batch, loss.item()))
        val_loss = np.mean(loss_of_batches)
        print('validation set loss: ', val_loss)
    return val_loss


def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    ape = torch.abs((y_true - y_pred) / y_true)
    ape = torch.where(torch.isinf(ape), torch.full_like(ape, 0.), ape)
    return torch.mean(ape)


def root_mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2) ** 0.5


def metric(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred).item()
    mape = mean_absolute_percentage_error(y_true, y_pred).item() * 100
    rmse = root_mean_squared_error(y_true, y_pred).item()
    return mae, mape, rmse


def step_metric(y_true, y_pred):
    """
    :param y_true: (num_samples, pred_steps, num_nodes, num_features)
    :param y_pred: (num_samples, pred_steps, num_nodes, num_features)
    :return:
    """
    pred_steps = y_true.shape[1]
    steps_mae = []
    steps_mape = []
    steps_rmse = []
    for step in range(pred_steps):
        step_y_true = y_true[:, step, :, :]
        step_y_pred = y_pred[:, step, :, :]
        step_mae, step_mape, step_rmse = metric(step_y_true, step_y_pred)
        steps_mae.append(step_mae)
        steps_mape.append(step_mape)
        steps_rmse.append(step_rmse)
    return steps_mae, steps_mape, steps_rmse
