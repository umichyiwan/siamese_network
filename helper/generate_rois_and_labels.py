from random import shuffle
from siamese.utils import get_dataset_and_name
import cv2


def render(bboxes, output):
    for i in range(len(bboxes)):
        id = bboxes[i].get('id', '')
        color = (255, 0, 0)
        left = max(0, int(bboxes[i]['left']))
        top = min(output.shape[1] - 1, int(bboxes[i]['top']))
        right = max(0, int(bboxes[i]['right']))
        bottom = min(output.shape[1] - 1, int(bboxes[i]['bottom']))
        if id != '':
            color = ((255 / 10 * id) % 256, (255 / 5 * id) % 256, (255 / 20 * id) % 256)
            text = str(bboxes[i]['id']) + " " + str(round(bboxes[i]['covered_ratio'], 2))
            position = (left / 2 + right / 2, top / 2 + bottom / 2)
            cv2.putText(output, text, position, cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
        # draw bounding box
        cv2.rectangle(output, (left, top), (right, bottom), color, 4)


def generate_rois_and_labels(groundtruth, image_pair_list):
    index = 0
    rois1 = []
    rois2 = []
    labels = []
    while index < len(image_pair_list):
        image1_path = image_pair_list[index][0]
        image2_path = image_pair_list[index][1]
        image1 = cv2.imread(image1_path, 0)
        image1_size = image1.shape
        image2 = cv2.imread(image2_path, 0)
        image2_size = image2.shape
        _, image1_name = get_dataset_and_name(image1_path)
        _, image2_name = get_dataset_and_name(image2_path)
        image1_info = groundtruth[int(image1_name) - 1]
        image2_info = groundtruth[int(image2_name) - 1]
        # im = cv2.imread(image1_path)
        # output = im.copy()
        # render([bbox for bbox in image1_info['feature']['objs'] if bbox['covered_ratio'] < 1], output)
        # cv2.imshow('img', output)
        # cv2.waitKey(0)
        assert int(image1_name) == int(image1_info['frame'])
        image1_bboxes = image1_info['feature']['objs']
        image2_bboxes = image2_info['feature']['objs']
        image1_bboxes = [bbox for bbox in image1_bboxes if bbox['covered_ratio'] < 0.7 and (min(image1_size[1], bbox['right']) - max(0, bbox['left'])) > 16 and (min(image1_size[0], bbox['bottom']) - max(0, bbox['top'])) > 32]
        image2_bboxes = [bbox for bbox in image2_bboxes if bbox['covered_ratio'] < 0.7 and (min(image2_size[1], bbox['right']) - max(0, bbox['left'])) > 16 and (min(image2_size[0], bbox['bottom']) - max(0, bbox['top'])) > 32]
        # output = im.copy()
        # render(image1_bboxes, output)
        # cv2.imshow('img', output)
        # cv2.waitKey(0)
        if len(image1_bboxes) * len(image2_bboxes) < 8:
            del image_pair_list[index]
        else:
            rois1.append([])
            rois2.append([])
            labels.append([])
            negative_pair_list = []
            for bbox1 in image1_bboxes:
                for bbox2 in image2_bboxes:
                    if bbox1['id'] == bbox2['id']:
                        rois1[-1].append([max(0, int(bbox1['left'])),
                                      max(0, int(bbox1['top'])),
                                      min(int(bbox1['right']), image1_size[1] - 1),
                                      min(int(bbox1['bottom']), image1_size[0] - 1)])
                        rois2[-1].append([max(0, int(bbox2['left'])),
                                      max(0, int(bbox2['top'])),
                                      min(int(bbox2['right']), image2_size[1] - 1),
                                      min(int(bbox2['bottom']), image2_size[0] - 1)])
                        labels[-1].append(1)
                    else:
                        negative_pair_list.append([[max(0, int(bbox1['left'])),
                                                    max(0, int(bbox1['top'])),
                                                    min(int(bbox1['right']), image1_size[1] - 1),
                                                    min(int(bbox1['bottom']), image1_size[0] - 1)],
                                                   [max(0, int(bbox2['left'])),
                                                    max(0, int(bbox2['top'])),
                                                    min(int(bbox2['right']), image2_size[1] - 1),
                                                    min(int(bbox2['bottom']), image2_size[0] - 1)]])
                    if len(labels[-1]) == 8:
                        break
                if len(labels[-1]) == 8:
                    break
            if len(labels[-1]) != 8:
                shuffle(negative_pair_list)
                negative_pair_list = negative_pair_list[:8 - len(labels[-1])]
                for negative_pair in negative_pair_list:
                    rois1[-1].append(negative_pair[0])
                    rois2[-1].append(negative_pair[1])
                    labels[-1].append(0)
            index += 1
    return rois1, rois2, labels



