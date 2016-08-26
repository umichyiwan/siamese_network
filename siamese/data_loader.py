import mxnet as mx
import logging
import multiprocessing as mp
import numpy as np
import cv2
import os
import random
import Queue
import atexit
from utils import get_dataset_and_name
from utils import read_list


class SiameseDataLoader(mx.io.DataIter):
    """Data loader for siamese network."""
    def __init__(self, **kwargs):
        super(SiameseDataLoader, self).__init__()
        self.input_args = kwargs
        self.image_size = kwargs.get('image_size')
        self.image_pairs_list = read_list(kwargs.get('data_list_path'))
        self.label_dir = kwargs.get('label_dir')
        self.rois_dir = kwargs.get('rois_dir')
        self.rois_siamese_dir = kwargs.get('rois_siamese_dir')
        self.multi_thread = kwargs.get('multi_thread', True)
        self.n_thread = kwargs.get('n_thread', 7)
        self.stop_word = kwargs.get('stop_word', '==STOP--')
        self.batch_size = kwargs.pop('batch_size', 10)
        self.mode = kwargs.pop('mode', 'train')
        self.data_num = len(self.image_pairs_list)
        self.current = 0
        self.worker_proc = None
        self._get_next(True)

        if self.multi_thread:
            self.stop_flag = mp.Value('b', False)
            self.result_queue = mp.Queue(maxsize=self.batch_size*30)
            self.data_queue = mp.Queue()

    def _insert_queue(self):
        for item in self.image_pairs_list:
            self.data_queue.put(item)
        for i in range(self.n_thread):
            self.data_queue.put(self.stop_word)

    def _thread_start(self):
        self.stop_flag = False
        self.worker_proc = [mp.Process(target=SiameseDataLoader._worker,
                                       args=[self.data_queue,
                                             self.result_queue,
                                             self.stop_word,
                                             self.stop_flag,
                                             self.input_args
                                             ]) for i in range(self.n_thread)]
        [item.start() for item in self.worker_proc]

        def cleanup():
            self.shutdown()
        atexit.register(cleanup)

    @staticmethod
    def _worker(data_queue, result_queue, stop_word, stop_flag, input_args):
        count = 0
        while True:
            item = data_queue.get()
            if item == stop_word or stop_flag == 1:
                break
            img1, img2, rois, rois_siamese, label = SiameseDataLoader.get_pair_data(item, input_args)
            result_queue.put((img1, img2, rois, rois_siamese, label))
            count += 1

    @property
    def provide_data(self):
        if self.mode == 'train':
            p_data = [('data', self.data[0].shape),
                      ('data_siamese', self.data[1].shape),
                      ('rois', self.data[2].shape),
                      ('rois_siamese', self.data[3].shape)]
        else:
            p_data = [('data', self.data[0].shape),
                      ('data_siamese', self.data[1].shape),
                      ('rois', self.data[2].shape),
                      ('rois_siamese', self.data[3].shape),
                      ('label', self.label[0].shape)]
        return p_data

    @property
    def provide_label(self):
        if self.mode == 'train':
            p_label = [('label', (8, 1))]
        else:
            p_label = [('label', self.label[0].shape)]
        return p_label

    def reset(self):
        self.data_num = len(self.image_pairs_list)
        self.current = 0
        # self.shuffle()
        if self.multi_thread:
            self.shutdown()            # Shutdown data-reading threads
            self._insert_queue()       # Initialize data_queue
            self._thread_start()       # Start multi thread

    def shutdown(self):
        if self.multi_thread:
            # clean queue
            while True:
                try:
                    self.result_queue.get(timeout=1)
                except Queue.Empty:
                    break
            while True:
                try:
                    self.data_queue.get(timeout=1)
                except Queue.Empty:
                    break
            # stop worker
            self.stop_flag = True
            if self.worker_proc:
                for i, worker in enumerate(self.worker_proc):
                    worker.join(timeout=1)
                    if worker.is_alive():
                        logging.error('worker {} fails to join'.format(i))
                        worker.terminate()

    def shuffle(self):
        random.shuffle(self.image_pairs_list)
        pass

    def next(self):
        if self._get_next(False):
            self.current += self.batch_size
            if self.mode == 'train':
                return mx.io.DataBatch(data=self.data, label=self.label, pad=0, index=None)
            else:
                return mx.io.DataBatch(data=self.data + self.label, label=self.label, pad=0, index=None)
        else:
            raise StopIteration

    def _get_next(self, pre_fetch):
        batch_size = self.batch_size
        if self.current + batch_size > self.data_num:
            return False
        data = []
        data_siamese = []
        rois = []
        rois_siamese = []
        label = []
        cnt = 0
        for i in range(self.current, self.current + batch_size):
            if self.multi_thread and not pre_fetch:
                img1, img2, det1, det2, pair_label = self.result_queue.get()
            else:
                img1, img2, det1, det2, pair_label = SiameseDataLoader.get_pair_data(self.image_pairs_list[i], self.input_args)
            batch_num_padding1 = np.ones((det1.shape[0], 1)) * cnt
            batch_num_padding2 = np.ones((det2.shape[0], 1)) * cnt
            det1 = np.hstack((batch_num_padding1, det1))
            det2 = np.hstack((batch_num_padding2, det2))
            data.append(img1)
            data_siamese.append(img2)
            rois.append(det1)
            rois_siamese.append(det2)
            label.append(pair_label)
            cnt += 1
        data = mx.ndarray.array(data)
        data_siamese = mx.ndarray.array(data_siamese)
        rois = mx.ndarray.array(rois)
        rois_siamese = mx.ndarray.array(rois_siamese)
        label = mx.ndarray.array(label)
        self.data = [data, data_siamese, rois, rois_siamese]
        self.label = [label]

        return True

    @staticmethod
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

    @staticmethod
    def get_pair_data(image_pair_path, input_args):
        rois_dir = input_args.get('rois_dir')
        rois_siamese_dir = input_args.get('rois_siamese_dir')
        label_dir = input_args.get('label_dir')
        # image_size = input_args.get('image_size')

        mean = np.array([103.939, 116.779, 123.68])
        image = cv2.imread(image_pair_path[0])
        # size_list = [(420, 720, 3), (560, 560, 3), (400, 600, 3)]
        # random.shuffle(size_list)
        # image, rescale = DataLoader.resize_image_with_zero_padding(image, size_list[0])

        image_siamese = cv2.imread(image_pair_path[1])
        # image_siamese, rescale_siamese = DataLoader.resize_image_with_zero_padding(image_siamese, size_list[1])
        print image_pair_path
        dataset_name, image1_name = get_dataset_and_name(image_pair_path[0])
        dataset_name, image2_name = get_dataset_and_name(image_pair_path[1])

        rois = np.load(os.path.join(rois_dir, dataset_name + '_' + image1_name + '_' + dataset_name + '_' + image2_name) + '.npy')
        # rois = rois * rescale

        rois_siamese = np.load(os.path.join(rois_siamese_dir, dataset_name + '_' + image1_name + '_' + dataset_name + '_' + image2_name) + '.npy')
        # rois_siamese = rois_siamese * rescale_siamese

        label = np.load(os.path.join(label_dir, dataset_name + '_' + image1_name + '_' + dataset_name + '_' + image2_name + '.npy'))

        # image1 = image.astype(np.uint8)
        # image2 = image_siamese.astype(np.uint8)
        # cv2.imshow('img1', image1)
        # cv2.imshow('img2', image2)
        # for i in range(8):
        #     bbox1 = image1[rois[i][1]:rois[i][3], rois[i][0]:rois[i][2], :]
        #     bbox2 = image2[rois_siamese[i][1]:rois_siamese[i][3], rois_siamese[i][0]:rois_siamese[i][2], :]
        #     cv2.imshow('bb1', bbox1)
        #     cv2.imshow('bb2', bbox2)
        #     print label[i], bbox1.shape, bbox2.shape
        #     cv2.waitKey(0)

        image = image - mean
        image_siamese = image_siamese - mean
        image = np.transpose(image, (2, 0, 1))
        image_siamese = np.transpose(image_siamese, (2, 0, 1))

        return image, image_siamese, rois, rois_siamese, label


class SingleDataLoader(mx.io.DataIter):
    """Data loader for siamese network."""

    def __init__(self, **kwargs):
        super(SingleDataLoader, self).__init__()
        self.input_args = kwargs
        self.image = kwargs.get('image')
        self.rois = kwargs.get('rois')
        self.data_num = 1
        self.current = 0
        self.batch_size = 1
        self._get_next()

    @property
    def provide_data(self):
        p_data = [('data', self.data[0].shape),
                  ('rois', self.data[1].shape)]
        return p_data

    @property
    def provide_label(self):
        p_label = [('label', self.label[0].shape)]
        return p_label

    def next(self):
        if self._get_next():
            self.current += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label, pad=0, index=None)
        else:
            raise StopIteration

    def _get_next(self):
        if self.current + self.batch_size > self.data_num:
            return False
        data = []
        rois = []
        image = SingleDataLoader.preprocess(self.image)
        batch_num_padding = np.zeros((self.rois.shape[0], 1))
        detection = np.hstack((batch_num_padding, self.rois))
        data.append(image)
        rois.append(detection)
        data = mx.ndarray.array(data)
        rois = mx.ndarray.array(rois)
        self.data = [data,  rois]
        self.label = [np.zeros((1,4))]

        return True

    @staticmethod
    def preprocess(image):
        mean = np.array([103.939, 116.779, 123.68])
        image = image - mean
        image = np.transpose(image, (2, 0, 1))
        return image