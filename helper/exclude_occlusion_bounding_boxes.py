import json
import numpy as np
import os
import cv2


def exclude_occlusion_bounding_boxes(input_json_path, image_size):
    with open(input_json_path) as f:
        result = json.load(f)
    for i in range(len(result)):
        result[i]['objs'] = result[i]['feature']['objs']
        del result[i]['feature']

    for frame in result:
        bboxes = frame['objs']
        for bbox in bboxes:
            other_bboxes = [bb for bb in bboxes if bb is not bbox]
            zeros = np.zeros(image_size)
            for other_bbox in other_bboxes:
                top = max(0, int(other_bbox['top']))
                left = max(0, int(other_bbox['left']))
                bottom = min(image_size[0] - 1, int(other_bbox['bottom']))
                right = min(image_size[1] - 1, int(other_bbox['right']))
                zeros[top:bottom, left:right] = 1
            top = max(0, int(bbox['top']))
            left = max(0, int(bbox['left']))
            bottom = min(image_size[0] - 1, int(bbox['bottom']))
            right = min(image_size[1] - 1, int(bbox['right']))
            area = (right - left) * (bottom - top)
            covered_ratio = np.sum(zeros[top:bottom, left:right]) / area
            bbox['covered_ratio'] = covered_ratio

    for i in range(len(result)):
        result[i]['feature'] = {'objs': result[i]['objs']}
        del result[i]['objs']
    return result


if __name__ == '__main__':
    data_dir = '/home/yiwan/Desktop/siamese_network/raw_data/'
    year = '2DMOT2015'
    output_dir = '/home/yiwan/Desktop/siamese_network/data/'
    for dataset_name in os.listdir(data_dir + year + '/train/'):
        dataset_path = output_dir + '2DMOT2015_' + dataset_name
        groundtruth_json_path = dataset_path + '/groundtruth.json'
        output_path = groundtruth_json_path
        image_size = (cv2.imread(dataset_path + '/img/000001.jpg', 0)).shape
        result = exclude_occlusion_bounding_boxes(groundtruth_json_path, image_size)
        with open(output_path, 'w') as f:
            json.dump(result, f, encoding='utf-8', indent=2)