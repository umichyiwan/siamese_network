import logging
from datetime import datetime
import mxnet as mx
import os
import sys
import numpy as np


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


def load_params(prefix, epoch):
    """Load pre-trained models
    Args:
        prefix (str): model prefix
        epoch (int): epoch
    Returns:
        arg_params (dict): weight and bias
        aux_params (dict): other auxiliary parameters
    """
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def get_callback(period):
    """Callback function in training
    Args:
        period (int): Every certain period, call evaluation function
    Returns:
        func: callback function
    """
    def _callback(param):
        if param.nbatch % period == 0:
            logging.info('Iter[%d] Batch[%d]', param.epoch, param.nbatch)
            param.eval_metric.print_log()
    return _callback


def save_log(output_dir):
    """Save log in training process
    Args:
        output_dir (str): folder to save log
    """
    fmt = '%(asctime)s %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt=date_fmt,
                        filename=os.path.join(output_dir,
                                              'event_' + datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def analyze_log(log_file):
    f = open(log_file, 'r')
    val_precision = []
    val_recall = []

    while True:
        line = f.readline()
        if 'Validation-precision' in line:
            precision = float(line.strip().split('=')[1])
            val_precision.append(precision)

        if 'Validation-recall' in line:
            validation = float(line.strip().split('=')[1])
            val_recall.append(validation)

        if not line:
            break

    print 'Precision: '
    print val_precision
    print 'Recall: '
    print val_recall


def get_dataset_and_name(image_path):
    image_path = image_path.strip()
    image_path = image_path.split('/')
    dataset = image_path[-3]
    name = image_path[-1].split('.')[0]
    return dataset, name


def precision(pred, label):
    pred_label = np.int32(pred > 0.5)        # binary
    label = label.astype('int32')     # binary
    sum_metric = np.sum(label[np.where(pred_label == 1)])
    num_inst = np.sum(pred_label)
    if num_inst == 0:
        return 1
    return float(sum_metric) / float(num_inst)


def recall(pred, label):
    pred_label = np.int32(pred > 0.5)        # binary
    label = label.astype('int32')     # binary
    sum_metric = np.sum(pred_label[np.where(label == 1)])
    num_inst = np.sum(label)
    if num_inst == 0:
        return 1
    return float(sum_metric) / float(num_inst)


def preprocess_img(img):
    """Preprocess image, resize the image to fixed size
    Args:
        img (numpy.ndarray): image, shape is (w, h, 3)
    Returns:
        mx.ndarray: pre processed image
    """
    img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1)) - 128
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = mx.ndarray.array(img)
    return img


def angle_mapping(ang):
    """Non-linear mapping of angle, enforce large gap
    in 0 and 180, y = (x - 90)^3 / 8100 + 90
    Args:
        ang (np.ndarray): angle, [0, 180)
    Returns:
        int: angle after mapping
    """
    return (ang - 90.0)**3 / 8100.0 + 90.0


def angle_visualization(cls_mask, reg_mask):
    """
        Using the same colormap for orientation as SIFT flow
        http://people.csail.mit.edu/celiu/SIFTflow/
    Args:
        cls_mask (numpy.ndarray): [0, 1] masks of lane probability
        reg_mask (numpy.ndarray): [0, 180] mask of lane orientation, e.g.
                                  /
                    theta = 135  /
                  -------------------
                               /
    Returns:
        numpy.ndarray: bgr image of visualization
    """
    hsv = np.zeros(cls_mask.shape + (3,), dtype=np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 0] = reg_mask
    hsv[:, :, 2] = cv.normalize(cls_mask, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def label2mask(label, cls_width):
    """Convert network predicted label to flat mask
    Args:
        label: e.g. 64*7*7
        cls_width: e.g. 8
    Returns:
        np.ndarray: flatten mask
    """
    h = label.shape[1] * cls_width
    w = label.shape[2] * cls_width
    mask = np.zeros((h, w))
    for i in range(cls_width):
        for j in range(cls_width):
            mask[i:h:cls_width, j:w:cls_width] = label[i*cls_width + j, :, :]
    return mask


def postprocess(output, threshold, cls_width):
    pred_prob = output
    pred_label = np.where(pred_prob > threshold, pred_prob, 0)
    mask = label2mask(pred_label, cls_width)
    return mask

if __name__ == '__main__':
    analyze_log(sys.argv[1])
