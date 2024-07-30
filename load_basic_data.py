import os.path
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from scipy.sparse import coo_matrix
from tool_func import preprocess_features


def load_basic_data():
    """
    load data all basic data
    :return: data dictionary
    """
    data_raw_path = "./data/raw_data/"

    # load region ID
    region_id = np.load(data_raw_path + "region_id.npy")

    adj = np.load(data_raw_path + "adj_neighbor.npy")

    mob_flow = np.load(data_raw_path + "mobility_adj.npy").squeeze()
    src_simi = np.load(data_raw_path + "source_adj.npy")
    dst_simi = np.load(data_raw_path + "destination_adj.npy")
    poi_simi = np.load(data_raw_path + "poi_similarity.npy")

    # node features
    img_feat = np.load(data_raw_path + "img_feat.npy").squeeze(1)  #(180, 128)

    # feature normalization
    # img_feat = preprocess_features(img_feat)


    # feature normalization: feature Standardization or [feature min-max normalization]
    # scaler = StandardScaler()
    # feature = scaler.fit_transform(feature)


    # save as a dict
    out = {
        "region_id": region_id,
        "adj": adj,
        "mob_flow": mob_flow,
        "src_simi": src_simi,
        "dst_simi": dst_simi,
        "poi_simi": poi_simi,
        "img_feat": img_feat
    }

    return out


if __name__ == '__main__':
    data = load_basic_data()
    print(data['mob_flow'].shape)






