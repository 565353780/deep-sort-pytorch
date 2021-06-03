# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track

from iou import IOU

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, Target_People_Num=-1,USE_IOU=False, max_iou_distance=0.7, max_age=600, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 0

        self.Target_People_Num = Target_People_Num
        self.USE_IOU = USE_IOU
        self.IOU = IOU()

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = [], [], []

        if self.USE_IOU and len(detections) < len(self.tracks) and 0 < self.Target_People_Num <= len(self.tracks):
            print('\nActivate dist+descriptor+iou match...')

            detections_matches = [-1 for detection_ in detections]

            detection_bboxs = [detection_.tlwh for detection_ in detections]
            track_bboxs = [track_.to_tlwh() for track_ in self.tracks]

            detection_descriptors = [detection_.feature for detection_ in detections]
            track_descriptors = [track_.features[0] for track_ in self.tracks]

            self.IOU.get_iou_pairs(track_bboxs)
            self.IOU.get_close_bbox_id_pairs()
            self.IOU.get_bbox_match_by_dist2(detection_bboxs, track_bboxs)
            self.IOU.get_bbox_match_by_descriptor(detection_descriptors, track_descriptors)

            for ii in range(len(detections)):
                print("------pair : (", ii, ",", self.IOU.detections_matches_by_dist2[ii], "&&", self.IOU.detections_matches_by_descriptor[ii], ")")
                print("------bbox dist : ", self.IOU.detections_match_dists_by_dist2[ii])
                print("------descroptor dist : ", self.IOU.detections_match_dists_by_descriptor[ii])
                for jj in range(len(self.tracks)):
                    if self.IOU.detections_matches_by_dist2[ii] > -1 and self.IOU.detections_matches_by_dist2[ii] != jj and self.IOU.iou_pairs[self.IOU.detections_matches_by_dist2[ii]][jj] > 0:
                        print("------iou for track : (", self.IOU.detections_matches_by_dist2[ii], ",", jj, ") : ", self.IOU.iou_pairs[self.IOU.detections_matches_by_dist2[ii]][jj])
                    if self.IOU.detections_matches_by_descriptor[ii] > -1 and self.IOU.detections_matches_by_descriptor[ii] != jj and self.IOU.iou_pairs[self.IOU.detections_matches_by_descriptor[ii]][jj] > 0:
                        print("------iou for track : (", self.IOU.detections_matches_by_descriptor[ii], ",", jj, ") : ", self.IOU.iou_pairs[self.IOU.detections_matches_by_descriptor[ii]][jj])

                max_current_iou = 0

                for jj in range(len(self.tracks)):
                    if jj != self.IOU.detections_matches_by_dist2[ii] and self.IOU.iou_pairs[jj][self.IOU.detections_matches_by_dist2[ii]] > max_current_iou:
                        max_current_iou = self.IOU.iou_pairs[jj][self.IOU.detections_matches_by_dist2[ii]]

                if self.IOU.detections_matches_by_dist2[ii] != -1 and self.IOU.detections_matches_by_dist2[ii] == self.IOU.detections_matches_by_descriptor[ii]:
                    detections_matches[ii] = self.IOU.detections_matches_by_dist2[ii]
                elif self.IOU.detections_matches_by_dist2[ii] != -1 and max_current_iou < 0.8 and self.IOU.detections_match_dists_by_descriptor[ii] > 0.2:
                    detections_matches[ii] = self.IOU.detections_matches_by_dist2[ii]

                if self.IOU.detections_matches_by_descriptor[ii] != -1 and self.IOU.detections_matches_by_dist2[ii] != self.IOU.detections_matches_by_descriptor[ii] and self.IOU.detections_match_dists_by_descriptor[ii] < 0.2:
                    if detections_matches[ii] != -1:
                        print("reset current match : (", ii, " , ", self.IOU.detections_matches_by_dist2[ii], ") to (", ii, " , ", self.IOU.detections_matches_by_descriptor[ii] << ")")
                        detections_matches[ii] = self.IOU.detections_matches_by_descriptor[ii]

                if detections_matches[ii] == -1:
                    print("!!!!!!!!!!!!!!!!!!!!!!!detection match failed : ", ii)
                elif max_current_iou > 0.8:
                    self.tracks[detections_matches[ii]].update_descriptor = False

            for ii in range(len(detections)):
                matches.append((detections_matches[ii], ii))

            for ii in range(len(self.tracks)):
                if self.tracks[ii].track_id not in detections_matches:
                    unmatched_tracks.append(self.tracks[ii].track_id)

        else:
            matches, unmatched_tracks, unmatched_detections = \
                self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        if len(self.tracks) < self.Target_People_Num or self.Target_People_Num == -1:
            for track_idx in unmatched_tracks:
                self.tracks[track_idx].mark_missed()
            for detection_idx in unmatched_detections:
                self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            # track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
