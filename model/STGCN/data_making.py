from config import PeMS08_config, NYC_config
import numpy as np

# 滑动窗口，得到训练样本
def get_samples(split_data, pred_step, hour_for_train, points_per_hour):
    samples = []
    for idx in range(split_data.shape[0]):
        if hour_for_train * points_per_hour <= idx and idx + pred_step <= split_data.shape[0]:
            sample_x = split_data[idx - hour_for_train * points_per_hour: idx, :, :1]
            sample_y = split_data[idx: idx + pred_step, :, :1]
            samples.append((sample_x, sample_y))
    samples = [np.stack(i, axis=0) for i in zip(*samples)]
    return samples

def graph_signals_process(file, pred_step, hour_for_train, points_per_hour, normalize, save_path):
    data = np.load(file)['data']  #一年数据：(17568, 180) or # (366*48, 180)
    # 假设截取前3个月数据测试, [31, 91, 182]
    month = 1
    data = data[:31*48, :]
    data = np.expand_dims(data, axis=2)

    print(data.shape)
    num_of_samples, num_of_nodes, num_of_features = data.shape
    assert (hour_for_train > 0), 'Invalid train hour!'

    split_line1 = int(num_of_samples * 0.6) #
    split_line2 = int(num_of_samples * 0.8)

    train_data = data[:split_line1] # (10713, 170, 3)
    val_data = data[split_line1:split_line2] # (3571, 170, 3)
    test_data = data[split_line2:] # (3572, 170, 3)

    train_samples = get_samples(train_data, pred_step, hour_for_train, points_per_hour)
    val_samples = get_samples(val_data, pred_step, hour_for_train, points_per_hour)
    test_samples = get_samples(test_data, pred_step, hour_for_train, points_per_hour)

    train_x, train_y = train_samples
    val_x, val_y = val_samples
    test_x, test_y = test_samples

    if normalize == 'z-score':
        mean = train_x.mean()
        std = train_x.std()

        def z_score(x):
            return (x - mean) / std

        train_x = z_score(train_x)
        val_x = z_score(val_x)
        test_x = z_score(test_x)

    print(train_x.shape, train_y.shape)
    print(val_x.shape, val_y.shape)
    print(test_x.shape, test_y.shape)
    save_path_name = save_path + f'dataset_{str(month)}_predstep_{str(pred_step)}.npz'
    np.savez_compressed(save_path_name,
                        train_x=train_x, train_y=train_y,
                        val_x=val_x, val_y=val_y,
                        test_x=test_x, test_y=test_y)

if __name__ == '__main__':

    config = NYC_config
    graph_signals_process(file=config['graph_signal_matrix_filename'], pred_step=config['num_for_predict'],
                          hour_for_train=6, points_per_hour=config['points_per_hour'], normalize=config['normalize'],
                          save_path=config['processed_signals'])