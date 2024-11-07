import torch

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

NYC_config = {
    'dataset_name': 'nyc_pickup',
    'adj_type': 'connectivity',  # or 'distance'
    'id_filename': None,

    'distance_filename': 'data/nyc_dis.csv',
    'graph_signal_matrix_filename': 'data/nyc_pickup.npz',
    'processed_signals': 'data/',

    'normalize': 'z-score',
    'num_of_vertices': 180,
    'points_per_hour': 2,
    'num_for_predict': 4,
    'num_of_features': 1,
    'time_span': ['2016-01-01 00:00:00', '2016-06-30 00:00:00']
}















PeMS08_config = {
    'dataset_name': 'PeMS08',
    'adj_type': 'connectivity',  # or 'distance'
    'id_filename': None,

    'distance_filename': 'data/PEMS08.csv',
    'graph_signal_matrix_filename': 'data/PEMS08.npz',
    'processed_signals': 'data/dataset.npz',

    'normalize': 'z-score',
    'num_of_vertices': 170,
    'points_per_hour': 12,
    'num_for_predict': 1, #预测长度？
    'num_of_features': 1,
    'time_span': ['2016-07-01 00:00:00', '2016-08-31 23:55:00']
}

STGCN_config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    "epochs": 500,
    'temporal_dim': 64,
    'Kt': 3,
    'spatial_dim': 16,
    'Ks': 3,
}
