import ConfigParser
import sys
import numpy as np
import os
import mxnet as mx
import logging
from data_loader import SiameseDataLoader
from symbol import get_siamese_test
from utils import load_params
import time
import cv2
from utils import get_dataset_and_name


class SiameseTester:
    def __init__(self, config):
        try:
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
            self.label_dir = config.get('model', 'label_dir')
            self.rois_dir = config.get('model', 'rois_dir')
            self.rois_siamese_dir = config.get('model', 'rois_siamese_dir')

            # Data info
            channel = 3
            img_width = int(config.get('model', 'img_width'))
            img_height = int(config.get('model', 'img_height'))
            self.batch_size = int(config.get('model', 'batch_size'))
            self.image_size = (img_height, img_width, channel)

            # load model
            self.use_gpu_idx = config.get('model', 'use_gpu_idx')
            symbol = get_siamese_test()
            args, auxs = load_params(os.path.join(self.finetune_model, 'test_v0'), self.model_epoch)
            ctx = mx.cpu() if self.use_gpu_idx == 'None' else [
                mx.gpu(int(i)) for i in self.use_gpu_idx.split(',')]
            self.model = mx.model.FeedForward(symbol=symbol, arg_params=args, aux_params=auxs, ctx=ctx)

        except ValueError:
            logging.error('Config parameter error')


    @staticmethod
    def get_data_iter(image_size, label_dir, rois_dir, rois_siamese_dir, test_list_path, batch_size,
                      multi_thread, mode):

        test_data_iter = SiameseDataLoader(
            image_size=image_size,
            data_list_path=test_list_path,
            label_dir=label_dir,
            rois_dir=rois_dir,
            rois_siamese_dir=rois_siamese_dir,
            batch_size=batch_size,
            multi_thread=multi_thread,
            mode=mode)

        return test_data_iter

    def test_image_pairs(self):
        t1 = time.clock()
        test = self.get_data_iter(self.image_size, self.label_dir, self.rois_dir, self.rois_siamese_dir,
                                        self.test_list_path, self.batch_size, self.multi_thread, 'train')
        pred = self.model.predict(X=test)

        pred1 = pred[0].squeeze()
        pred2 = pred[1].squeeze()
        label = pred[2].squeeze()
        result = []
        for i in predict:
            if i < 0.2:
                result.append(1)
            else:
                result.append(0)
        result = np.array(result)
        label = np.array(label)
        t2 = time.clock()
        print t2 - t1
        print predict[np.where(result != label)]
        print label[np.where(result != label)]
        print 1 - sum(abs(result - label)) / len(result)
        pass

    def extract(self, image, rois, image_siamese, rois_siamese):

        batch_size = 1
        y = np.zeros([1, 1, 1, 1, 1, 1, 0, 0])
        mean = np.array([103.939, 116.779, 123.68])
        image = image - mean
        image = np.transpose(image, (2, 0, 1))
        image_siamese = image_siamese - mean
        image_siamese = np.transpose(image_siamese, (2, 0, 1))
        data = [[image], [rois], [image_siamese], [rois_siamese]]
        iter = CustomNDArrayIter(data, y, batch_size, shuffle=False)
        pred = self.model.predict(iter)
        return pred

if __name__ == '__main__':
    config_path = sys.argv[1]
    config = ConfigParser.RawConfigParser()
    config.read(config_path)
    tester = SiameseTester(config)
    tester.test_image_pairs()


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


# if __name__ == '__main__':
#     config_path = sys.argv[1]
#     config = ConfigParser.RawConfigParser()
#     config.read(config_path)
#     tester = SiameseTester(config)
#     image_pairs_list = read_list(config.get('model', 'test_list_path'))
#     label_dir = config.get('model', 'label_dir')
#     rois_dir = config.get('model', 'rois_dir')
#     rois_siamese_dir = config.get('model', 'rois_siamese_dir')
#     mean = np.array([103.939, 116.779, 123.68])
#     for image_pair_path in image_pairs_list:
#         print image_pair_path
#         image = cv2.imread(image_pair_path[0])
#         image_siamese = cv2.imread(image_pair_path[1])
#         dataset_name, image1_name = get_dataset_and_name(image_pair_path[0])
#         dataset_name, image2_name = get_dataset_and_name(image_pair_path[1])
#         rois = np.load(
#             os.path.join(rois_dir, dataset_name + '_' + image1_name + '_' + dataset_name + '_' + image2_name) + '.npy')
#         rois_siamese = np.load(os.path.join(rois_siamese_dir,
#                                             dataset_name + '_' + image1_name + '_' + dataset_name + '_' + image2_name) + '.npy')
#         label = np.load(
#             os.path.join(label_dir, dataset_name + '_' + image1_name + '_' + dataset_name + '_' + image2_name + '.npy'))
#         rois = np.hstack((np.ones((8, 1)), rois))
#         rois_siamese = np.hstack((np.ones((8, 1)), rois_siamese))
#         a = tester.extract(image, rois, image_siamese, rois_siamese)
#         # b = extracter.extract(image_siamese, rois_siamese)
#         # print np.sum(np.square(b - a), axis=1)
#         # print label
#         pass