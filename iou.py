import numpy as np
from math import sqrt

class IOU:
    def __init__(self):
        self.iou_pairs = []
        self.close_rect_pairs = []

        self.detections_matches_by_dist2 = []
        self.detections_match_dists_by_dist2 = []

        self.detections_matches_by_dist = []
        self.detections_match_dists_by_dist = []

        self.detections_matches_by_descriptor = []
        self.detections_match_dists_by_descriptor = []

    def get_iou(self, bbox_1, bbox_2):
        if bbox_1[0] > bbox_2[0] + bbox_2[2]:
            return 0
        if bbox_1[1] > bbox_2[1] + bbox_2[3]:
            return 0
        if bbox_1[0] + bbox_1[2] < bbox_2[0]:
            return 0
        if bbox_1[1] + bbox_1[3] < bbox_2[1]:
            return 0

        colInt = min(bbox_1[0] + bbox_1[2], bbox_2[0] + bbox_2[2]) - max(bbox_1[0], bbox_2[0])
        rowInt = min(bbox_1[1] + bbox_1[3], bbox_2[1] + bbox_2[3]) - max(bbox_1[1], bbox_2[1])

        intersection = colInt * rowInt

        areaA = bbox_1[2] * bbox_1[3]
        areaB = bbox_2[2] * bbox_2[3]

        return intersection / (areaA + areaB - intersection)

    def get_iou_pairs(self, bboxs):
        self.iou_pairs = []

        for i in range(len(bboxs)):
            self.iou_pairs.append([])
            for j in range(len(bboxs)):
                self.iou_pairs[i].append(0)

            self.iou_pairs[i][i] = 1

        for i in range(len(bboxs)):
            for j in range(i):
                self.iou_pairs[i][j] = self.get_iou(bboxs[i], bboxs[j])
                self.iou_pairs[j][i] = self.iou_pairs[i][j]

        return self.iou_pairs

    def get_close_bbox_id_pairs(self):
        self.close_rect_pairs = []

        for i in range(len(self.iou_pairs)):
            for j in range(i):
                if self.iou_pairs[i][j] > 0:
                    self.close_rect_pairs.append([j, i])

        return self.close_rect_pairs

    def get_bbox_dist2(self, bbox_1, bbox_2):
        rect_dist = 0

        rect_dist += (bbox_1[0] - bbox_2[0]) * (bbox_1[0] - bbox_2[0])
        rect_dist += (bbox_1[1] - bbox_2[1]) * (bbox_1[1] - bbox_2[1])
        rect_dist += (bbox_1[0] + bbox_1[2] - bbox_2[0] - bbox_2[2]) * (
                    bbox_1[0] + bbox_1[2] - bbox_2[0] - bbox_2[2])
        rect_dist += (bbox_1[1] + bbox_1[3] - bbox_2[1] - bbox_2[3]) * (
                    bbox_1[1] + bbox_1[3] - bbox_2[1] - bbox_2[3])

        return rect_dist

    def get_bbox_dist(self, bbox_1, bbox_2):
        return sqrt(self.get_bbox_dist2(bbox_1, bbox_2))

    def get_bbox_match_by_dist2(self, detection_bboxs, track_bboxs):
        self.detections_matches_by_dist2 = []
        self.detections_match_dists_by_dist2 = []

        for i in range(len(detection_bboxs)):
            self.detections_matches_by_dist2.append(-1)
            self.detections_match_dists_by_dist2.append(-1)

        for i in range(len(detection_bboxs)):
            for j in range(len(track_bboxs)):
                current_dist2 = self.get_bbox_dist2(detection_bboxs[i], track_bboxs[j])

                if self.detections_match_dists_by_dist2[i] == -1 or current_dist2 < self.detections_match_dists_by_dist2[i]:
                    self.detections_match_dists_by_dist2[i] = current_dist2
                    self.detections_matches_by_dist2[i] = j

        return self.detections_matches_by_dist2

    def get_bbox_match_by_dist(self, detection_bboxs, track_bboxs):
        self.detections_matches_by_dist = []
        self.detections_match_dists_by_dist = []

        for i in range(len(detection_bboxs)):
            self.detections_matches_by_dist.append(-1)
            self.detections_match_dists_by_dist.append(-1)

        for i in range(len(detection_bboxs)):
            for j in range(len(track_bboxs)):
                current_dist = self.get_bbox_dist(detection_bboxs[i], track_bboxs[j])

                if self.detections_match_dists_by_dist[i] == -1 or current_dist < self.detections_match_dists_by_dist[i]:
                    self.detections_match_dists_by_dist[i] = current_dist
                    self.detections_matches_by_dist[i] = j

        return self.detections_matches_by_dist

    def get_descriptor_dist(self, descriptor_1, descriptor_2):
        descriptor_1 = np.asarray(descriptor_1)
        descriptor_2 = np.asarray(descriptor_2)

        xy = np.dot(descriptor_1, descriptor_2.T)
        xx = np.dot(descriptor_1, descriptor_1.T)
        yy = np.dot(descriptor_2, descriptor_2.T)
        norm = sqrt(xx * yy) + 1e-6

        return 1 - xy / norm

    def get_bbox_match_by_descriptor(self, detection_descriptors, track_descriptors):
        self.detections_matches_by_descriptor = []
        self.detections_match_dists_by_descriptor = []

        for i in range(len(detection_descriptors)):
            self.detections_matches_by_descriptor.append(-1)
            self.detections_match_dists_by_descriptor.append(-1)

        for i in range(len(detection_descriptors)):
            for j in range(len(track_descriptors)):
                current_dist = self.get_descriptor_dist(detection_descriptors[i], track_descriptors[j])

                if self.detections_match_dists_by_descriptor[i] == -1 or current_dist < self.detections_match_dists_by_descriptor[i]:
                    self.detections_match_dists_by_descriptor[i] = current_dist
                    self.detections_matches_by_descriptor[i] = j

        return self.detections_matches_by_descriptor