import os
import json
import cv2
import numpy as np


def resize_image_with_zero_padding(image, image_size):
    zeros = np.zeros(image_size)
    if 1. * image.shape[0] / image.shape[1] > 1. * zeros.shape[0] / zeros.shape[1]:
        rescale = 1. * zeros.shape[0] / image.shape[0]
        image = cv2.resize(image, (int(image.shape[1] * rescale), zeros.shape[0]))
    else:
        rescale = 1. * zeros.shape[1] / image.shape[1]
        image = cv2.resize(image, (zeros.shape[1], int(image.shape[0] * rescale)))
    zeros[:image.shape[0], :image.shape[1], :] = image
    return zeros, rescale


def txt2json(data_dir, year, output_basedir, image_size):
    for dataset in os.listdir(data_dir + year + '/train/'):
        print dataset
        dataset_path = data_dir + year + '/train/' + dataset
        input_path = dataset_path + '/gt/gt.txt'
        output_dir = output_basedir + year + '_' + dataset + '/'
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        output_img_dir = output_basedir + year + '_' + dataset + '/img/'
        if not os.path.isdir(output_img_dir):
            os.makedirs(output_img_dir)
        output_path = output_dir + 'groundtruth.json'
        output_json = []

        image_dir = dataset_path + '/img1/'
        total_frame_num = len(os.listdir(image_dir))
        for i in range(total_frame_num):
            frame_info = {'frame': i + 1, 'feature': {'objs': []}}
            output_json.append(frame_info)
            image = cv2.imread(image_dir + '%0*d' % (6, i + 1) + '.jpg')
            image, rescale = resize_image_with_zero_padding(image, image_size)
            cv2.imwrite(output_img_dir + '%0*d' % (6, i + 1) + '.jpg', image)

        with open(input_path) as f:
            for line in f:
                line = line.strip()
                line = line.split(',')
                frame = int(line[0])
                output_json[frame - 1]['feature']['objs'].append({'left': float(line[2]) * rescale,
                                                                   'top': float(line[3]) * rescale,
                                                                   'right': (float(line[2]) + float(line[4])) * rescale,
                                                                   'bottom': (float(line[3]) + float(line[5])) * rescale,
                                                                   'id': int(line[1]),
                                                                   'confidence': float(line[6]),
                                                                   'type': 'PEDESTRIAN'})
        with open(output_path, 'w') as f:
            json.dump(output_json, f, encoding='utf-8', indent=2)


if __name__ == '__main__':
    data_dir = '/home/yiwan/Desktop/siamese_network/raw_data/'
    year = '2DMOT2015'
    output_basedir = '/home/yiwan/Desktop/siamese_network/data/'
    txt2json(data_dir, year, output_basedir, (360, 640, 3))
