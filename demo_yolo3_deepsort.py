import os
import cv2
import time
import argparse
import numpy as np
from distutils.util import strtobool

from YOLOv3 import YOLOv3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

from read_json import Read_Json

import matplotlib.pyplot as plt
from math import *

# ==========================================================
# ↓↓↓↓↓↓↓↓↓↓↓↓↓Params↓↓↓↓↓↓↓↓↓↓↓↓↓
# ==========================================================
# detector part
SHOW_PERSON_DETECTOR_RESULT_ONLY = False

USE_OPENPOSE = False
OUTPUT_OPENPOSE_KEYPOINTS_AND_ID_NPY = False

SHOW_YOLOv3_FLAGS = False
ANALYSE_AND_UPDATE_YOLOv3_BBOX = True
OUTPUT_YOLOv3_BBOX = False

# reid part
Target_People_Num = 2
USE_IOU = True

# others
START_FRAME = 4
SAVE_VIDEO = False
PRINT_TIME_SPEND = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, default="./test_videos/2019_2person_chinese_cut.mp4")
    parser.add_argument("--yolo_cfg", type=str, default="./YOLOv3/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights", type=str, default="./YOLOv3/yolov3.weights")
    parser.add_argument("--yolo_names", type=str, default="./YOLOv3/cfg/coco.names")
    # 按 置信度<conf_thresh 删掉错误的目标检测结果
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    # 按 交并比>nms_thresh 删掉重复的目标检测结果
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--deepsort_checkpoint", type=str, default="./deep_sort/deep/checkpoint/ckpt.t7")
    if USE_OPENPOSE:
        parser.add_argument("--max_dist", type=float, default=100000000000000000)
    else:
        parser.add_argument("--max_dist", type=float, default=0.3)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default='./demos/demo_for_' +
                                                         parser.parse_args().VIDEO_PATH.split('/')[
                                                             len(parser.parse_args().VIDEO_PATH.split('/')) - 1].split(
                                                             '.')[0] + '.avi')
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args()


# ==========================================================
# ↑↑↑↑↑↑↑↑↑↑↑↑↑Params↑↑↑↑↑↑↑↑↑↑↑↑↑
# ==========================================================

# Todo : tracker.py -> USE_IOU -> 增加镜头切换检测，防止第一步按位置匹配出现错误
# Done : 在tracks的bbox面积远小于图片1/3时抑制YOLOv3输出结果中面积占图片1/3以上的bbox产生，否则会导致使用YOLOv3进行目标检测后目标匹配出现问题

class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.yolo3 = YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names, is_xywh=True,
                            conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, use_cuda=use_cuda)
        self.deepsort = DeepSort(args.deepsort_checkpoint, Target_People_Num, USE_IOU, args.max_dist, use_cuda)
        self.class_names = self.yolo3.class_names

        self.read_json = Read_Json()

        self.current_time = 0
        self.start_frame = 0

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path and SAVE_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def time_cut_output(self, output_message, output=True):
        if output:
            print(output_message, '---->', int((time.time() - self.current_time) * 1000), 'ms')
        self.current_time = time.time()

    def get_ids_after_remove_big_bboxes(self, image_shape, bboxes, max_bbox_area_scale_to_im,
                                        min_bbox_area_scale_to_im):
        bbox_ids = []
        im_h, im_w, _ = image_shape

        total_im_area = im_w * im_h
        average_bbox_area = 0
        bbox_areas = []

        if bboxes is not None:
            for bbox in bboxes:
                current_bbox_area = bbox[2] * bbox[3]
                average_bbox_area += current_bbox_area
                bbox_areas.append(current_bbox_area)

            average_bbox_area /= len(bboxes)

            if average_bbox_area / total_im_area < max_bbox_area_scale_to_im:
                for i in range(len(bboxes)):
                    if bboxes[i][2] > 0.8 * im_w:
                        continue
                    if min_bbox_area_scale_to_im < bbox_areas[i] / total_im_area < max_bbox_area_scale_to_im:
                        bbox_ids.append(i)

        return bbox_ids

    def get_ids_after_remove_small_bboxes(self, bboxes, bbox_ids_in, min_bbox_area_scale_to_max_bbox):
        bbox_ids = []

        max_bbox_area = 0
        bbox_areas = []

        if bboxes is not None and len(bbox_ids_in) > 0:
            for idx in bbox_ids_in:
                current_bbox_area = bboxes[idx][2] * bboxes[idx][3]
                if current_bbox_area > max_bbox_area:
                    max_bbox_area = current_bbox_area
                bbox_areas.append(current_bbox_area)

            for i in range(len(bbox_ids_in)):
                if bbox_areas[i] / max_bbox_area > min_bbox_area_scale_to_max_bbox:
                    bbox_ids.append(bbox_ids_in[i])

        return bbox_ids

    def detect(self):
        # bbox_pre = []
        # bbox_pre.append([[565., 283., 691., 562.], [655., 230., 848., 567.]])
        # bbox_pre.append([[566., 280., 693., 561.], [667., 231., 845., 566.]])
        # bbox_pre.append([[565., 276., 684., 565.], [660., 240., 834., 559.]])

        x = [0]
        # [Target_People_Num, 8, [data....]]
        means = []
        # [Target_People_Num, 8, 8, [data....]]
        covariances = []
        for i in range(Target_People_Num):
            means.append([])
            covariances.append([])
            for j in range(8):
                means[i].append([0])
                covariances[i].append([])
                for k in range(8):
                    covariances[i][j].append([0])

        total_start = time.time()
        current_frame = 0
        while self.vdo.grab():
            self.time_cut_output('\n', False)
            current_frame += 1
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            self.time_cut_output('vdo.retrieve', PRINT_TIME_SPEND)
            if current_frame - 1 < self.start_frame:
                continue
            # im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            im = ori_im
            temp_bbox_xcycwh, temp_cls_conf, temp_cls_ids = self.yolo3(im, OUTPUT_YOLOv3_BBOX)
            self.time_cut_output('yolo3', PRINT_TIME_SPEND)

            if ANALYSE_AND_UPDATE_YOLOv3_BBOX:
                small_bbox_ids = self.get_ids_after_remove_big_bboxes(im.shape, temp_bbox_xcycwh, 0.33, 0.01)

                used_bbox_ids = self.get_ids_after_remove_small_bboxes(temp_bbox_xcycwh, small_bbox_ids, 0.5)

                bbox_xcycwh, cls_conf, cls_ids = None, None, None

                if len(used_bbox_ids) > 0:
                    bbox_xcycwh = [temp_bbox_xcycwh[i] for i in used_bbox_ids]
                    cls_conf = [temp_cls_conf[i] for i in used_bbox_ids]
                    cls_ids = [temp_cls_ids[i] for i in used_bbox_ids]

                    bbox_xcycwh = np.array(bbox_xcycwh)
                    cls_conf = np.array(cls_conf)
                    cls_ids = np.array(cls_ids)

                self.time_cut_output('remove_big_bbox', PRINT_TIME_SPEND)
            else:
                bbox_xcycwh, cls_conf, cls_ids = temp_bbox_xcycwh, temp_cls_conf, temp_cls_ids

            if bbox_xcycwh is not None:
                if len(bbox_xcycwh) == 0:
                    continue

                # select class person
                if not SHOW_PERSON_DETECTOR_RESULT_ONLY:
                    mask = cls_ids == 0

                    # set_first_frame_num = 4
                    # if OUTPUT_YOLOv3_BBOX:
                    #     set_first_frame_num = 0
                    #
                    # if current_frame < set_first_frame_num:
                    #     bbox_xcycwh = []
                    #     cls_conf = []
                    #     temp_bbox = bbox_pre[current_frame - 1]
                    #     for bbox in temp_bbox:
                    #         bbox_xcycwh.append(
                    #             [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0], bbox[3] - bbox[1]])
                    #         cls_conf.append(1)
                    #     bbox_xcycwh = np.array(bbox_xcycwh)
                    #     cls_conf = np.array(cls_conf)
                    # else:
                    bbox_xcycwh = bbox_xcycwh[mask]
                    cls_conf = cls_conf[mask]

                if not OUTPUT_YOLOv3_BBOX:
                    bbox_xcycwh[:, 3] *= 1.2
                self.time_cut_output('update_bbox', PRINT_TIME_SPEND)

                outputs = self.deepsort.update(bbox_xcycwh, bbox_xcycwh, cls_conf, cls_ids, im,
                                               SHOW_PERSON_DETECTOR_RESULT_ONLY)
                self.time_cut_output('deepsort.update', PRINT_TIME_SPEND)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    confidences = outputs[:, 4]
                    if len(outputs[0]) == 5:
                        confidences = None
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, confidences, (0, 0), SHOW_YOLOv3_FLAGS)
                    self.time_cut_output('draw_bboxes', PRINT_TIME_SPEND)

                if OUTPUT_YOLOv3_BBOX:
                    if not os.path.exists(os.getcwd() + '/bbox/'):
                        os.makedirs(os.getcwd() + '/bbox/')
                    with open('./bbox/' + args.VIDEO_PATH.split('/')[len(args.VIDEO_PATH.split('/')) - 1].split('.')[
                        0] + '_' + str(current_frame - 1) + '.txt', 'w') as f:
                        for i in range(len(bbox_xcycwh)):
                            x, y, w, h = bbox_xcycwh[i]
                            x1 = x - w / 2
                            x2 = x + w / 2
                            y1 = y - h / 2
                            y2 = y + h / 2
                            f.write(
                                str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(cls_conf[i]) + '\n')
                    self.time_cut_output('OUTPUT_YOLOv3_BBOX', PRINT_TIME_SPEND)

            # x.append(current_frame - 1)
            #
            # tracks_num = len(self.deepsort.tracker.tracks)
            #
            # for i in range(tracks_num):
            #     for j in range(8):
            #         means[i][j].append(self.deepsort.tracker.tracks[i].mean[j])
            #         for k in range(8):
            #             covariances[i][j][k].append(self.deepsort.tracker.tracks[i].covariance[j][k])
            # for i in range(tracks_num, Target_People_Num):
            #     for j in range(8):
            #         means[i][j].append(means[i][j][len(means[i][j]) - 1])
            #         for k in range(8):
            #             covariances[i][j][k].append(covariances[i][j][k][len(covariances[i][j][k]) - 1])
            #
            # for i in range(Target_People_Num):
            #     plt.ion()
            #     fig, axes = plt.subplots(8, 9)
            #     for j in range(8):
            #         axes[j, 0].plot(x, means[i][j], '-r')
            #         for k in range(8):
            #             axes[j, k + 1].plot(x, covariances[i][j][k], '-r')
            #     plt.draw()
            #     time.sleep(10)

            end = time.time()
            if end != start and PRINT_TIME_SPEND:
                print('\rtime: {}ms, fps: {}'.format(int((end - start) * 1000), int(1 / (end - start))), end='')

            if self.args.display:
                cv2.putText(ori_im, 'Frame:' + str(current_frame - 1),
                            (int(0.01 * ori_im.shape[1]), int(0.99 * ori_im.shape[0])), cv2.FONT_HERSHEY_PLAIN, 2,
                            [0, 0, 255], 2)
                cv2.putText(ori_im, 'FPS:' + str(int(1 / (end - start))),
                            (int(0.88 * ori_im.shape[1]), int(0.99 * ori_im.shape[0])), cv2.FONT_HERSHEY_PLAIN, 2,
                            [0, 0, 255], 2)
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if SAVE_VIDEO:
                if self.args.save_path:
                    if not os.path.exists(os.getcwd() + '/demos/'):
                        os.makedirs(os.getcwd() + '/demos/')
                    self.output.write(ori_im)

        total_end = time.time()
        print("average_time: {} ms, average_fps: {}".format(int(((total_end - total_start) / current_frame) * 1000),
                                                            float(
                                                                int((current_frame / (
                                                                            total_end - total_start)) * 100)) / 100))

    def openpose_detect(self):
        total_start = time.time()
        current_frame = 0
        video_name = args.VIDEO_PATH.split('/')[len(args.VIDEO_PATH.split('/')) - 1].split('.')[0]
        json_path = os.getcwd() + '/json/' + video_name
        if not os.path.exists(json_path + '/'):
            os.makedirs(json_path + '/')

        frame_idx = 0

        if OUTPUT_OPENPOSE_KEYPOINTS_AND_ID_NPY:
            total_data = []
            total_match = []

            ff = open('./test.txt', 'r')

            current_match = [0, 0, 0]
            for line in ff.readlines():
                if 'Frame' in line:
                    current_match[0] = int(line.split(':')[1])
                elif 'match' in line:
                    current_match[1] = int(line.split(':')[1].split(',')[0])
                    current_match[2] = int(line.split(':')[1].split(',')[1])
                    total_match.append(current_match.copy())

            ff.close()

            current_match_idx = 0

            while self.vdo.grab():
                len_zero = 12 - len(str(frame_idx))
                frame_str = ''
                for i in range(len_zero):
                    frame_str += '0'
                frame_str += str(frame_idx)
                json_name = json_path + '/' + video_name + '_' + frame_str + '_keypoints.json'

                pose_keypoints_2d = self.read_json.load_json(json_name)

                current_people_match = []

                while total_match[current_match_idx][0] == frame_idx:
                    for i in range(25):
                        pose_keypoints_2d[total_match[current_match_idx][1]][i][3] = total_match[current_match_idx][2]
                    current_people_match.append(pose_keypoints_2d[total_match[current_match_idx][1]].copy())

                    current_match_idx += 1

                current_people_match = np.array(current_people_match)

                total_data.append(current_people_match.copy())

                frame_idx += 1

            total_data = np.array(total_data)

            np.save('./test.npy', total_data)
        else:
            while self.vdo.grab():
                len_zero = 12 - len(str(frame_idx))
                frame_str = ''
                for i in range(len_zero):
                    frame_str += '0'
                frame_str += str(frame_idx)
                json_name = json_path + '/' + video_name + '_' + frame_str + '_keypoints.json'

                self.read_json.load_json(json_name)

                person_bbox_2d, bbox_feature_2d, bbox_2d = self.read_json.get_bbox_xywh(frame_idx)

                person_bbox_2d = np.array(person_bbox_2d)
                bbox_feature_2d = np.array(bbox_feature_2d)
                bbox_2d = np.array(bbox_2d)

                frame_idx += 1
                current_frame += 1
                start = time.time()
                _, ori_im = self.vdo.retrieve()
                # im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
                im = ori_im

                if person_bbox_2d is not None:
                    bbox_xcycwh = []
                    bbox_feature = []
                    cls_conf = []
                    # max_area_1 = 0
                    # max_area_2 = 0
                    # max_idx_1 = 0
                    # max_idx_2 = 0
                    # current_idx = 0
                    # for person_bbox in person_bbox_2d:
                    #     if person_bbox[4] > args.conf_thresh and person_bbox[2] < 0.2 * im.shape[1] and person_bbox[3] < 0.6 * im.shape[0]:
                    #         current_area = person_bbox[2]*person_bbox[3]/pow(((person_bbox[0] - im.shape[1]/2)*(person_bbox[0] - im.shape[1]/2) + (person_bbox[1] - im.shape[0]/2)*(person_bbox[1] - im.shape[0]/2)), 0.2)
                    #         if current_area > max_area_1:
                    #             max_area_1 = current_area
                    #             max_idx_1 = current_idx
                    #         elif current_area > max_area_2:
                    #             max_area_2 = current_area
                    #             max_idx_2 = current_idx
                    #     current_idx += 1
                    for i in range(len(person_bbox_2d)):
                        person_bbox = person_bbox_2d[i]
                        if person_bbox[4] > args.conf_thresh:
                            bbox_xcycwh.append(person_bbox[:4])
                            cls_conf.append(person_bbox[4])
                            if i < len(bbox_feature_2d):
                                bbox_feature.append(bbox_feature_2d[i][:4])
                            else:
                                bbox_feature.append([0., 0., 0., 0.])

                    bbox_xcycwh = np.array(bbox_xcycwh)
                    bbox_feature = np.array(bbox_feature)
                    cls_conf = np.array(cls_conf)

                    if len(bbox_xcycwh) > 0:
                        bbox_xcycwh[:, 3] *= 1.2
                        bbox_feature[:, 3] *= 1.2

                        outputs = self.deepsort.update(bbox_xcycwh, bbox_xcycwh, cls_conf, im,
                                                       SHOW_PERSON_DETECTOR_RESULT_ONLY)
                        # outputs = self.deepsort.openpose_update(bbox_xcycwh, bbox_feature, cls_conf, im, bbox_2d, SHOW_YOLOv3_RESULT)
                        if len(outputs) > 0:
                            bbox_xyxy = outputs[:, :4]
                            identities = outputs[:, -1]
                            ori_im = draw_bboxes(ori_im, bbox_xyxy, identities)

                end = time.time()
                if end != start:
                    print('\rtime: {}ms, fps: {}'.format(int((end - start) * 1000), int(1 / (end - start))), end='')

                if self.args.display:
                    cv2.imshow("test", ori_im)
                    cv2.waitKey(1)

                if SAVE_VIDEO:
                    if self.args.save_path:
                        self.output.write(ori_im)

            total_end = time.time()
            print(
                "\naverage_time: {} ms, average_fps: {}".format(int(((total_end - total_start) / current_frame) * 1000),
                                                                float(
                                                                    int((current_frame / (
                                                                                total_end - total_start)) * 100)) / 100))


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.start_frame = START_FRAME
        if USE_OPENPOSE:
            det.openpose_detect()
        else:
            det.detect()
