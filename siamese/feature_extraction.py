import ConfigParser
import sys
import numpy as np
import os
import mxnet as mx
import logging
from symbol import get_single_branch_test
from symbol import get_siamese_test
from utils import load_params
from utils import get_dataset_and_name
from data_loader import SingleDataLoader
import cv2
import time


class FeatureExtractor:
    def __init__(self):
        try:
            config_path = '/home/yiwan/Desktop/siamese_network/siamese/test.cfg'
            config = ConfigParser.RawConfigParser()
            config.read(config_path)
            self.config = config
            # multi thread loading data or not
            self.multi_thread = config.getboolean('model', 'multi_thread')

            # Fine-tune options
            self.finetune = config.getboolean('model', 'finetune')
            self.finetune_model = config.get('model', 'finetune_model')
            self.model_epoch = int(config.get('model', 'model_epoch'))
            self.make_new_dir = config.getboolean('model', 'make_new_dir')

            # Data directory
            self.test_list_path = config.get('model', 'test_list_path')
            self.rois_dir = config.get('model', 'rois_dir')

            # Data info
            channel = 3
            img_width = int(config.get('model', 'img_width'))
            img_height = int(config.get('model', 'img_height'))
            self.batch_size = int(config.get('model', 'batch_size'))
            self.image_size = (img_height, img_width, channel)

            # load model
            self.use_gpu_idx = config.get('model', 'use_gpu_idx')
            # symbol = get_siamese_test()
            symbol = get_single_branch_test()
            args, auxs = load_params(os.path.join(self.finetune_model, 'test_v0'), self.model_epoch)
            ctx = mx.cpu() if self.use_gpu_idx == 'None' else [
                mx.gpu(int(i)) for i in self.use_gpu_idx.split(',')]
            self.model = mx.model.FeedForward(symbol=symbol, arg_params=args, aux_params=auxs, ctx=ctx)

        except ValueError:
            logging.error('Config parameter error')

    def extract(self, image, rois):
        # data_iter = SingleDataLoader(
        #     image=image,
        #     detections=detections)
        data_iter = SingleDataLoader(
            image=image,
            rois=rois,
        )
        pred = self.model.predict(data_iter)
        return pred


def read_list(data_list_path):
    """Get image list
    Args:
        data_list_path (str): full path of train_lst or val_list
    Returns:
        list: a list of image paths
    """
    image_pair_list = np.load(data_list_path)
    image_pair_list = image_pair_list.tolist()
    return image_pair_list


if __name__ == '__main__':
    config_path = sys.argv[1]
    config = ConfigParser.RawConfigParser()
    config.read(config_path)
    extracter = FeatureExtractor()
    data_list_path = config.get('model', 'test_list_path')
    image_pairs_list = read_list(config.get('model', 'test_list_path'))
    label_dir = config.get('model', 'label_dir')
    rois_dir = config.get('model', 'rois_dir')
    rois_siamese_dir = config.get('model', 'rois_siamese_dir')
    mean = np.array([103.939, 116.779, 123.68])
    for image_pair_path in image_pairs_list:
        print image_pair_path
        image = cv2.imread(image_pair_path[0])
        dataset_name, image1_name = get_dataset_and_name(image_pair_path[0])
        dataset_name, image2_name = get_dataset_and_name(image_pair_path[1])
        rois = np.load(
            os.path.join(rois_dir, dataset_name + '_' + image1_name + '_' + dataset_name + '_' + image2_name) + '.npy')
        a = extracter.extract(image, rois)
        # b = extracter.extract(image_siamese, rois_siamese)
        # print np.sum(np.square(b - a), axis=1)
        # print label
        pass