from random import shuffle
import os
import numpy as np


# def generate_random_num_pair(num_images, max_pair_dist, num_pairs):
#     pair_list = []
#     for i in range(1, num_images + 1):
#         for j in range(1, max_pair_dist + 1):
#             if i + j <= num_images:
#                 pair_list.append([i, i + j])
#     shuffle(pair_list)
#     pair_list = pair_list[:num_pairs]
#     return pair_list

def generate_random_num_pair(num_images, dist, num_pairs):
    pair_list = []
    for i in range(1, num_images + 1):
        for j in dist:
            if i + j <= num_images:
                pair_list.append([i, i + j])
    shuffle(pair_list)
    pair_list = pair_list[:num_pairs]
    return pair_list


def generate_random_image_pair(image_dir):
    num_images = len(os.listdir(image_dir))
    max_pair_dist = 10
    num_pairs = num_images * 10
    # pair_list = generate_random_num_pair(num_images, max_pair_dist, num_pairs)
    pair_list = generate_random_num_pair(num_images, [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 36, 42, 50], num_pairs)
    image_pair_list = []
    for pair in pair_list:
        image1_name = '%0*d' % (6, pair[0]) + '.jpg'
        image1_path = image_dir + image1_name
        image2_name = '%0*d' % (6, pair[1]) + '.jpg'
        image2_path = image_dir + image2_name
        image_pair_list.append([image1_path, image2_path])
    return image_pair_list


if __name__ == '__main__':
    image_dir = '/home/yiwan/Desktop/vbase-trajectory/data/2DMOT2015/train/ADL-Rundle-6/img1'
    image_pair_list_path = '/home/yiwan/Desktop/siamese_network/image_pair_list'
    image_pair_list = generate_random_image_pair(image_dir)
    np.save(image_pair_list_path, image_pair_list)




