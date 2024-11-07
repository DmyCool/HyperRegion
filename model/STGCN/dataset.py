from torch.utils import data
import torch
import numpy as np


class NYCDataset(data.Dataset):
    def __init__(self, data_name, data_path, mode):
        pems = np.load(data_path)
        if mode == 'train':
            self.X, self.Y = pems['train_x'], pems['train_y']
        elif mode == 'val':
            self.X, self.Y = pems['val_x'], pems['val_y']
        elif mode == 'test':
            self.X, self.Y = pems['test_x'], pems['test_y']
        else:
            raise ValueError("dataset mode error, must be "
                             "train, val or test!")
        assert self.X.shape[0] == self.Y.shape[0], 'Data Error!'

        self.X, self.Y = torch.from_numpy(self.X).float(), torch.from_numpy(self.Y).float()
        print(data_name + ' ' + mode + ' dataset created !', 'X:', self.X.shape, 'Y:', self.Y.shape)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return len(self.Y)
