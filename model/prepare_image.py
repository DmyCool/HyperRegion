import os
import random
import pandas as pd
import geopandas as gpd
import numpy as np
import shutil
import shapely
from shapely.geometry import Point
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import time



# Find the image ID contained in each region, dict={region_id: img_list}
def region_contain_img_id():
    # image point file
    img_points_df = pd.read_csv('./data/img/GSV_point.txt', delimiter=',')

    # region file
    region_gdf = gpd.read_file('./data/raw_data/ma_shp_180/mh-180.shp')


    # convert point to GeoDataFrame
    geometry_points = [Point(xy) for xy in zip(img_points_df['long'], img_points_df['lat'])]
    img_points_gdf = gpd.GeoDataFrame(img_points_df, geometry=geometry_points)

    result_dict = {}

    for index, region in region_gdf.iterrows():
        region_id = region['region_id']
        points_in_region = img_points_gdf[img_points_gdf.within(region['geometry'])]
        point_ids_in_region = points_in_region['GSVID'].tolist()
        result_dict[region_id] = point_ids_in_region

    # print(result_dict)
    return result_dict


# Using pretrained models (which pretrained on cityscape dataset) to extract features
def feature_extract(imgpath):
    im = Image.open(imgpath)

    model_extractor = models.inception_v3(pretrained=True)

    model_extractor.fc = nn.Linear(2048, 128)
    model_extractor.eval()

    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = transform(im)
    img.unsqueeze_(dim=0)
    feat = model_extractor(img)
    feat = feat.detach().numpy()
    return feat



def get_region_img_feat(region_img_dict, img_path_folder):
    # define a dict to save the final img feat of each region
    region_img_feat_dict = {}

    # loop each 'key:value'
    for key, values in region_img_dict.items():

        key_features = []
        print('processing current region {}:'.format(key))

        # loop each images in current key(region)   --->sample some 50 images in each region, if num(img)<50, select all.
        k = 50 # The number of pictures to be sampled (50, 75, 117, 200)
        if len(values) < k:
            sub_values = values
        else:
            sub_values = random.sample(values, k)

        for filename in sub_values:
            image_path = os.path.join(img_path_folder, f'{filename}.jpg')

            # extract the image feature
            print('Extract current image feature with GSV_id={}'.format(str(filename)))
            feature = feature_extract(image_path)

            key_features.append(feature)

        # aggregated all image features in current key
        aggregated_feat = np.mean(key_features, axis=0)

        region_img_feat_dict[key] = aggregated_feat

        '''
        分段：每30个区域保存一次
        '''
        # count = int(key) + 1
        # if count % 30 == 0:
        #     print()
        #     feat_tmp = {key: value for key, value in region_img_feat_dict.items() if (key >= count - 30 and key < count)}
        #     feat_array = np.array(list(feat_tmp.values()))
        #     np.save('data/img_feat_seg/img_feat_{}.npy'.format(count), feat_array)

    feat_array = np.array(list(region_img_feat_dict.values()))


    if len(feat_array.shape) == 3:
        feat_array = feat_array.squeeze()

    print(feat_array.shape)

    # save region-img feats (180, 1, 128)
    if not os.path.exists('./data/img/img_feat.npy'):
        np.save('./data/img/img_feat.npy', feat_array)

    return feat_array







if __name__ == '__main__':
    img_path_folder = '../data/Panorama/Panorama/'
    region_img_dict = region_contain_img_id()

    get_region_img_feat(region_img_dict, img_path_folder)

