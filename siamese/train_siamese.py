"""
Train the siamese network model
"""
import mxnet as mx
import sys
import os
import numpy as np
import logging
from eval_metric import Loss, CompositeEvalMetric
from data_loader import DataLoader
from symbol import get_siamese_train
import utils
import ConfigParser
from datetime import datetime


class Fitter:
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

            # Save model options
            self.model_dir = config.get('model', 'model_dir')
            self.model_prefix = config.get('model', 'model_prefix')

            # Data directory
            self.train_list_path = config.get('model', 'train_list_path')
            self.val_list_path = config.get('model', 'val_list_path')
            self.label_dir = config.get('model', 'label_dir')
            self.rois_dir = config.get('model', 'rois_dir')
            self.rois_siamese_dir = config.get('model', 'rois_siamese_dir')

            # Data info
            channel = 3
            img_width = int(config.get('model', 'img_width'))
            img_height = int(config.get('model', 'img_height'))
            self.image_size = (img_height, img_width, channel)

            # SGD options
            self.lr = float(config.get('model', 'learning_rate'))
            self.momentum = float(config.get('model', 'momentum'))
            self.num_epoch = int(config.get('model', 'num_epoch'))
            self.weight_decay = float(config.get('model', 'weight_decay'))
            self.lr_factor = float(config.get('model', 'lr_factor'))
            self.lr_factor_epoch = int(config.get('model', 'lr_factor_epoch'))

            # Training options
            self.use_gpu_idx = config.get('model', 'use_gpu_idx')
            self.ctx = mx.cpu() if self.use_gpu_idx == 'None' else [
                mx.gpu(int(i)) for i in self.use_gpu_idx.split(',')]

            # Compute epoch size, input_shape, output_shape
            self.train_size = len(np.load(self.train_list_path))
            self.batch_size = int(config.get('model', 'batch_size'))
            self.epoch_size = self.train_size / self.batch_size

        except ValueError:
            logging.error('Config parameter error')

    def build_model(self):
        net = get_siamese_train(self.batch_size)
        # dot = mx.viz.plot_network(symbol=net,
        #                           shape={"data": (2, 3, 224, 224),
        #                                  "data_siamese": (2, 3, 224, 224),
        #                                  "rois": (2, 128, 5),
        #                                  "rois_siamese": (2, 128, 5),
        #                                  "label": (2, 128)},
        #                           node_attrs={"shape": 'rect', "fixedsize": 'false'})
        # dot.render('output.gv', view=True)
        model_args = {}
        if self.finetune:
            arg_params, aux_params = utils.load_params(os.path.join(self.finetune_model, 'test_v0'), self.model_epoch)
            model_args = {'arg_params': arg_params,
                          'aux_params': aux_params}
            begin_epoch = self.model_epoch + 1
        else:
            begin_epoch = 0

        # Learning rate scheduler
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step=max(1, int(self.epoch_size * self.lr_factor_epoch)),
            factor=self.lr_factor)

        model = mx.model.FeedForward(
            ctx=self.ctx,
            symbol=net,
            num_epoch=self.num_epoch,
            learning_rate=self.lr,
            momentum=self.momentum,
            wd=self.weight_decay,
            initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
            begin_epoch=begin_epoch,
            **model_args
        )
        return model

    @staticmethod
    def get_data_iter(train_list_path, val_list_path, rois_dir, rois_siamese_dir, label_dir, image_size, batch_size,
                      multi_thread, mode):

        train_data_iter = DataLoader(
            data_list_path=train_list_path,
            rois_dir=rois_dir,
            rois_siamese_dir=rois_siamese_dir,
            label_dir=label_dir,
            image_size=image_size,
            batch_size=batch_size,
            multi_thread=multi_thread,
            mode=mode)

        val_data_iter = DataLoader(
            data_list_path=val_list_path,
            rois_dir=rois_dir,
            rois_siamese_dir=rois_siamese_dir,
            label_dir=label_dir,
            image_size=image_size,
            batch_size=batch_size,
            multi_thread=multi_thread,
            mode=mode)

        return train_data_iter, val_data_iter

    def fit(self):
        if self.make_new_dir:
            # Make a folder to save model
            model_path = os.path.join(self.model_dir, self.model_prefix)
            if not os.path.isdir(model_path):
                os.mkdir(model_path)

            model_full_path = os.path.join(model_path, datetime.now().strftime('%Y_%m_%d_%H:%M:%S'))
            if not os.path.isdir(model_full_path):
                os.mkdir(model_full_path)
        else:
            model_full_path = self.finetune_model

        # Save config in model folder
        with open(os.path.join(model_full_path, 'train_' + datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '.cfg'), 'w') as f:
            self.config.write(f)
        utils.save_log(model_full_path)     # Save event log

        # Build mxnet model and train
        checkpoint = mx.callback.do_checkpoint(os.path.join(model_full_path, 'test_v0'))
        model = self.build_model()

        train, val = self.get_data_iter(self.train_list_path, self.val_list_path, self.rois_dir, self.rois_siamese_dir,
                                        self.label_dir, self.image_size, self.batch_size, self.multi_thread, 'mode')

        eval_metric = CompositeEvalMetric(metrics=[Loss()])
        call_back = utils.get_callback(3)

        model.fit(
            X=train,
            eval_data=val,
            eval_metric=eval_metric,
            epoch_end_callback=checkpoint,
            batch_end_callback=call_back
        )


if __name__ == '__main__':
    config_path = sys.argv[1]
    config = ConfigParser.RawConfigParser()
    config.read(config_path)
    fitter = Fitter(config)
    fitter.fit()