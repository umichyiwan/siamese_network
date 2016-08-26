import cv2
import json

frame_size = (640, 360)


def render(bboxes, output, writer):
    for i in range(len(bboxes)):
        id = bboxes[i].get('id', '')
        color = (255, 0, 0)
        left = max(0, int(bboxes[i]['left']))
        top = min(output.shape[1] - 1, int(bboxes[i]['top']))
        right = max(0, int(bboxes[i]['right']))
        bottom = min(output.shape[1] - 1, int(bboxes[i]['bottom']))
        if id != '':
            color = ((255 / 10 * id) % 256, (255 / 5 * id) % 256, (255 / 20 * id) % 256)
            text = str(round(bboxes[i]['covered_ratio'], 2))
            position = (left / 2 + right / 2, top / 2 + bottom / 2)
            cv2.putText(output, text, position, cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
        # draw bounding box
        left = int(bboxes[i]['left'])
        top = int(bboxes[i]['top'])
        right = int(bboxes[i]['right'])
        bottom = int(bboxes[i]['bottom'])
        cv2.rectangle(output, (left, top), (right, bottom), color, 4)
    # write video
    writer.write(output)


def visualize(image_dir, visualization_type, input_json_path, output_dir_path):
    with open(input_json_path) as f:
        result = json.load(f)
    if visualization_type == 'detection':
        # load detection output
        detection_video_path = output_dir_path + 'detection.avi'
        writer = cv2.VideoWriter(detection_video_path, cv2.cv.CV_FOURCC(*'XVID'), 15, frame_size)
    elif visualization_type == 'tracking':
        # load tracking output
        tracking_video_path = output_dir_path + 'tracking.avi'
        writer = cv2.VideoWriter(tracking_video_path, cv2.cv.CV_FOURCC(*'XVID'), 15, frame_size)
    elif visualization_type == 'postprocessing':
        # load tracking output
        postprocessing_video_path = output_dir_path + 'postprocessing.avi'
        writer = cv2.VideoWriter(postprocessing_video_path, cv2.cv.CV_FOURCC(*'XVID'), 15, frame_size)
    elif visualization_type == 'groundtruth':
        # load tracking output
        groundtruth_video_path = output_dir_path + 'groundtruth.avi'
        writer = cv2.VideoWriter(groundtruth_video_path, cv2.cv.CV_FOURCC(*'XVID'), 15, frame_size)
    else:
        print "no such visualization type"
        return

    for i in range(len(result)):
        result[i]['objs'] = result[i]['feature']['objs']
        del result[i]['feature']

    frame_count = 0
    frame_num = 1

    while True:
        print frame_num
        image_name = '%0*d' % (6, frame_num) + '.jpg'
        image_path = image_dir + image_name
        image = cv2.imread(image_path)
        if image is None:
            break
        if frame_count == len(result):
            break
        if result[frame_count]['frame'] != frame_num:
            bboxes = []
        else:
            bboxes = result[frame_count]['objs']
            frame_count += 1
        output = image.copy()
        # render(bboxes, output, writer)
        writer.write(output)
        # cv2.imshow('frame', output)
        # cv2.waitKey(15)
        frame_num += 1

if __name__ == '__main__':
    image_dir = '/home/yiwan/Desktop/siamese_network/data/2DMOT2015_ADL-Rundle-6/img/'
    visualization_type = 'groundtruth'
    input_json_path = '/home/yiwan/Desktop/siamese_network/data/2DMOT2015_ADL-Rundle-6/groundtruth.json'
    output_dir_path = '/home/yiwan/Desktop/siamese_network/data/2DMOT2015_ADL-Rundle-6/'
    visualize(image_dir, visualization_type, input_json_path, output_dir_path)