import mxnet as mx
import logging
import numpy as np
from mxnet.metric import EvalMetric


class Evalmetric(EvalMetric):
    def __init__(self, name):
        super(Evalmetric, self).__init__(name)
        self.precision = 0.0
        self.precision_inst = 0
        self.recall = 0.0
        self.recall_inst = 0

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        for i in range(len(labels)):
            pred_label = preds[i].asnumpy()
            pred_label = np.int32(pred_label > 0.5)
            label = labels[i].asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)

            self.sum_metric += np.sum(np.int32(pred_label.flat == label.flat))
            self.precision += np.sum(label[np.where(pred_label == 1)])
            self.precision_inst += np.sum(pred_label)
            self.recall += np.sum(pred_label[np.where(label == 1)])
            self.recall_inst += np.sum(label)
            self.num_inst += len(label.flat)

    def reset(self):
        super(Evalmetric, self).reset()
        self.precision = 0.0
        self.precision_inst = 0
        self.recall = 0.0
        self.recall_inst = 0

    def print_log(self):
        logging.info('Accuracy = %6f,  Precision = %4f,  Recall = %4f', self.sum_metric*1.0/max(self.num_inst, 1),
                     self.precision/max(self.precision_inst, 1), self.recall/max(self.recall_inst, 1))


class CompositeEvalMetric(EvalMetric):
    """Manage multiple evaluation metrics."""

    def __init__(self, **kwargs):
        super(CompositeEvalMetric, self).__init__('composite')
        try:
            self.metrics = kwargs['metrics']
        except KeyError:
            self.metrics = []

    def add(self, metric):
        """
        Add a child metric
        :param metric: Evaluation metric
        """
        self.metrics.append(metric)

    def get_metric(self, index):
        """
        Get a child metric
        :param index: Dst metric index
        :return:
        """
        try:
            return self.metrics[index]
        except IndexError:
            return ValueError("Metric index {} is out of range 0 and {}".format(
                index, len(self.metrics)))

    def update(self, labels, preds):
        for metric in self.metrics:
            metric.update(labels, preds)

    def reset(self):
        try:
            for metric in self.metrics:
                metric.reset()
        except AttributeError:
            pass

    def get(self):
        names = []
        results = []
        for metric in self.metrics:
            result = metric.get()
            names.append(result[0])
            results.append(result[1])
        return names, results

    def print_log(self):
        names, results = self.get()
        logging.info('; '.join(['{}: {}'.format(name, val) for name, val in zip(names, results)]))


class Loss(EvalMetric):
    """Calculate accuracy"""

    def __init__(self):
        super(Loss, self).__init__('loss')

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        pred_label = preds[0].asnumpy()
        label = labels[0].asnumpy().astype('int32')

        mx.metric.check_label_shapes(label, pred_label)
        print pred_label

        self.sum_metric += np.sum(pred_label)
        self.num_inst += len(label.flat)