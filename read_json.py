import os
import time
import json

class Read_Json:
    def __init__(self):
        self.bbox_pre = []
        self.bbox_pre.append([[565, 283, 691, 562], [655, 230, 848, 567]])
        self.bbox_pre.append([[566, 280, 693, 561], [667, 231, 845, 566]])
        self.bbox_pre.append([[565, 276, 684, 565], [660, 240, 834, 559]])

        self.pose_keypoints_2d = []

    def load_json(self, json_path):
        while not os.path.exists(json_path):
            time.sleep(0.01)

        # {0,  "Nose"},{1,  "Neck"},{2,  "RShoulder"},{3,  "RElbow"},{4,  "RWrist"},{5,  "LShoulder"},{6,  "LElbow"}
        # {7,  "LWrist"},{8,  "MidHip"},{9,  "RHip"},{10, "RKnee"},{11, "RAnkle"},{12, "LHip"},{13, "LKnee"}
        # {14, "LAnkle"},{15, "REye"},{16, "LEye"},{17, "REar"},{18, "LEar"},{19, "LBigToe"},{20, "LSmallToe"}
        # {21, "LHeel"},{22, "RBigToe"},{23, "RSmallToe"},{24, "RHeel"},{25, "Background"}
        self.pose_keypoints_2d = []
        stem = [5, 2, 9, 12]
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
            for person_dict in json_dict['people']:
                temp_keypoints_2d = []
                for i in range(int(len(person_dict['pose_keypoints_2d']) / 3)):
                    temp_keypoints_2d.append(
                        [person_dict['pose_keypoints_2d'][3 * i], person_dict['pose_keypoints_2d'][3 * i + 1],
                         person_dict['pose_keypoints_2d'][3 * i + 2], -1])

                self.pose_keypoints_2d.append(temp_keypoints_2d)

        return self.pose_keypoints_2d

    def get_bbox_xywh(self, frame_idx):
        person_bbox_2d = []
        bbox_2d = []
        bbox_feature_2d = []

        for keypoints_set in self.pose_keypoints_2d:
            xy_min_idx = 0
            while keypoints_set[xy_min_idx][2] == 0 and xy_min_idx < len(keypoints_set):
                xy_min_idx += 1
            x_min = keypoints_set[xy_min_idx][0]
            y_min = keypoints_set[xy_min_idx][1]
            x_max = x_min
            y_max = y_min
            conf = 0
            conf_num = 0
            zero_num = 1
            for keypoint in keypoints_set:
                if int(keypoint[0]) + int(keypoint[1]) > 0:
                    if keypoint[0] < x_min:
                        x_min = keypoint[0]
                    if keypoint[0] > x_max:
                        x_max = keypoint[0]
                    if keypoint[1] < y_min:
                        y_min = keypoint[1]
                    if keypoint[1] > y_max:
                        y_max = keypoint[1]
                    conf_num += 1
                    conf += keypoint[2]
                else:
                    zero_num *= 0.95
            # if x_max == x_min:
            #     x_max += 1
            # if y_max == y_min:
            #     y_max += 1
            # person_bbox_2d.append([x_min, y_min, x_max, y_max, conf / conf_num])
            ls = keypoints_set[5]
            rs = keypoints_set[2]
            rh = keypoints_set[9]
            lh = keypoints_set[12]
            xmin_feature = xmax_feature = 0
            ymin_feature = ymax_feature = 0
            for j in [5, 2, 9, 12, 13, 14, 19, 20, 21, 22, 23, 24]:
                if int(keypoints_set[j][0]) + int(keypoints_set[j][1]) > 0:
                    xmin_feature = min(xmin_feature, keypoints_set[j][0])
                    xmax_feature = max(xmax_feature, keypoints_set[j][0])
                    ymin_feature = min(ymin_feature, keypoints_set[j][1])
                    ymax_feature = max(ymax_feature, keypoints_set[j][1])

            if ls[2] * rs[2] * rh[2] * lh[2] > 0:
                bbox_2d.append([ls, rs, rh, lh])
            if conf_num > 0:
                person_bbox_2d.append([(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min,
                                       zero_num * conf / conf_num])
                if (xmax_feature - xmin_feature) * (ymax_feature - ymin_feature) == 0:
                    xmax_feature += 1
                    ymax_feature += 1
                bbox_feature_2d.append(
                    [(xmin_feature + xmax_feature) / 2, (ymin_feature + ymax_feature) / 2, xmax_feature - xmin_feature,
                     ymax_feature - ymin_feature])
        if frame_idx < 3:
            person_bbox_2d = []
            temp_bbox = self.bbox_pre[frame_idx]
            for bbox in temp_bbox:
                person_bbox_2d.append(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0], bbox[3] - bbox[1], 1.])
            bbox_feature_2d = person_bbox_2d.copy()

        return person_bbox_2d, bbox_feature_2d, bbox_2d