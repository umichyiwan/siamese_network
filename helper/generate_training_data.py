from generate_random_image_pair import generate_random_image_pair
from generate_rois_and_labels import generate_rois_and_labels
from siamese.utils import get_dataset_and_name
import numpy as np
import os
import json
import time


t1 = time.clock()
data_dir = '/home/yiwan/Desktop/siamese_network/data/'
year = '2DMOT2015'
rois_path = data_dir + 'rois/'
rois_siamese_path = data_dir + 'rois_siamese/'
labels_path = data_dir + 'labels/'
total_pair_list = []
train_data_size = 40000
if not os.path.exists(rois_path):
    os.mkdir(rois_path)
if not os.path.exists(rois_siamese_path):
    os.mkdir(rois_siamese_path)
if not os.path.exists(labels_path):
    os.mkdir(labels_path)
for dataset_name in os.listdir(data_dir):
    if dataset_name[:len(year)] != year:
        continue
    # generate image pairs
    dataset_path = data_dir + dataset_name
    image_dir = dataset_path + '/img/'

    print dataset_path
    image_pair_list = generate_random_image_pair(image_dir)
    # generate rois and labels, bad image pairs will be deleted in this stage
    with open(dataset_path + '/groundtruth.json') as f:
        groundtruth = json.load(f)
    rois, rois_siamese, labels = generate_rois_and_labels(groundtruth, image_pair_list)
    total_pair_list += image_pair_list
    for index in range(len(labels)):
        _, image1_name = get_dataset_and_name(image_pair_list[index][0])
        _, image2_name = get_dataset_and_name(image_pair_list[index][1])
        np.save(os.path.join(rois_path, dataset_name + '_' + image1_name + '_' + dataset_name + '_' + image2_name), rois[index])
        np.save(os.path.join(rois_siamese_path, dataset_name + '_' + image1_name + '_' + dataset_name + '_' + image2_name), rois_siamese[index])
        np.save(os.path.join(labels_path, dataset_name + '_' + image1_name + '_' + dataset_name + '_' + image2_name), labels[index])
np.save(data_dir + 'train_list', total_pair_list[:train_data_size])
np.save(data_dir + 'val_list', total_pair_list[train_data_size:])
t2 = time.clock()
print t2 - t1

