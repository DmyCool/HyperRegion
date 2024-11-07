import torch.nn as nn
import torch
import time


class TemporalGatedConv(nn.Module):
    def __init__(self, in_dim, out_dim, Kt):
        super(TemporalGatedConv, self).__init__()
        self.conv = nn.Conv2d(in_dim, 2 * out_dim, (1, Kt))  # Kt is the kernel size for temporal dim

    def forward(self, x):
        """
        shorten the seq_len by (Kt-1), and transform the feature dim from C to C'
        :param x: in shape of (B, T, N, C)
        :return: in shape of (B, T', N, C')
        """
        # to fit the input format of nn.Conv2d : (batch_size, channels, height, width)
        x = x.permute(0, 3, 2, 1)  # (B, C, N, T)

        x = self.conv(x)  # (B, 2C', N, T')
        x = x.permute(0, 2, 3, 1)  # (B, N, T', 2C')

        # Gated Linear Units
        lhs, rhs = torch.chunk(x, 2, dim=-1)  # (B, N, T', C') & (B, N, T', C')
        out = lhs * torch.sigmoid(rhs)  # (B, N, T', C')

        return out.permute(0, 2, 1, 3)  # (B, T', N, C')


class SpatialGraphConv(nn.Module):
    def __init__(self, K, cheb_polynomials, in_dim, out_dim):
        super(SpatialGraphConv, self).__init__()
        # for chebconv
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        # for feature transform
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_dim, out_dim)) for _ in range(K)])

    def forward(self, x):
        """
        :param x: (B, T, N, C)
        :return: (B, T, N, C')
        """
        batch_size, seq_len, num_of_nodes, c_in = x.shape
        outputs = []
        for step in range(seq_len):
            step_x = x[:, step, :, :]  # (B, N, C)
            step_output = torch.zeros(batch_size, num_of_nodes, self.out_dim, device=self.Theta[0].device)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]
                step_x = torch.matmul(T_k, step_x)  # (N, N) * (B, N, C) --> (B, N, C)
                theta_k = self.Theta[k]  # (C, C')
                step_output += step_x.matmul(theta_k)  # (B, N, C')
            outputs.append(step_output.unsqueeze(1))  # add (B, 1, N, C') for each step of seq
        output = torch.relu(torch.cat(outputs, dim=1))  # (B, T, N, C')
        return output


class STConvBlock(nn.Module):
    def __init__(self, in_dim, temporal_dim, Kt, K, cheb_polynomials, spatial_dim):
        super(STConvBlock, self).__init__()
        self.temporal_gated_conv1 = TemporalGatedConv(in_dim, temporal_dim, Kt)
        self.spatial_graph_conv = SpatialGraphConv(K, cheb_polynomials, temporal_dim, spatial_dim)
        self.temporal_gated_conv2 = TemporalGatedConv(spatial_dim, temporal_dim, Kt)

    def forward(self, x):
        """
        :param x: (B, T, N, C)
        :return:
        """
        x = self.temporal_gated_conv1(x)
        x = self.spatial_graph_conv(x)
        x = self.temporal_gated_conv2(x)
        return x


class STGCN(nn.Module):
    def __init__(self,emb_dim, in_dim, temporal_dim, Kt, Ks, cheb_polynomials, spatial_dim, seq_len, pred_step):
        super(STGCN, self).__init__()
        self.seq_len = seq_len
        self.pred_step = pred_step

        self.st_conv_block1 = STConvBlock(in_dim, temporal_dim, Kt, Ks, cheb_polynomials, spatial_dim)
        self.st_conv_block2 = STConvBlock(temporal_dim, temporal_dim, Kt, Ks, cheb_polynomials, spatial_dim)
        self.temporal_gated_conv = TemporalGatedConv(temporal_dim, temporal_dim, Kt)
        self.fc = nn.Linear((seq_len - (Kt - 1) * 5) * temporal_dim + emb_dim, pred_step)

        self.model_name = str(type(self).__name__)

    def forward(self, x, x_emb):
        """
        :param x: (B, T, N, C)
        :return: (B, pred_step, N, C)
        """
        # x_emb(N, F)----(B, N, F)
        x_emb = x_emb.unsqueeze(0)
        x_emb = x_emb.repeat(x.shape[0], 1, 1)

        x = self.st_conv_block1(x)
        x = self.st_conv_block2(x)
        x = self.temporal_gated_conv(x)
        x = x.permute(0, 2, 1, 3)  # (B, N, T', C')
        x = x.reshape((x.shape[0], x.shape[1], -1))  # (B, N, T'*C)
        x = torch.cat((x, x_emb), dim=-1) # (B, N, F+(T'*C))
        x = self.fc(x)  # (B, N, pred_step)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, pred_step, N, C)
        return x

    def save(self, stop_epoch, lr, data_name, file_path=None):
        prefix = 'checkpoints/'
        if file_path is None:
            prefix = prefix + self.model_name + "_"
            file_path = time.strftime(prefix + '%m%d_%H_%M_epoch_' + str(stop_epoch)
                                      + '_lr_' + str(lr) + '_' + str(data_name) + '.pth')
        else:
            file_path = prefix + file_path + '.pth'
        torch.save(self.state_dict(), file_path)
        print('save parameters to file: %s' % file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print('load parameters from file: %s' % file_path)
