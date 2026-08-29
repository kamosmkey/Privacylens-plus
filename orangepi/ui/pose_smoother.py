"""Lightweight temporal smoothing for pose detections."""

from __future__ import annotations

import math
import time

import numpy as np


class _PoseTrack:
    def __init__(self, detection, timestamp):
        box, score, keypoints = detection
        self.box = np.asarray(box, dtype=np.float32).copy()
        self.score = float(score)
        self.keypoints = np.asarray(keypoints, dtype=np.float32).copy()
        self.raw_xy = self.keypoints[:, :2].copy()
        self.velocity = np.zeros_like(self.raw_xy)
        self.timestamp = timestamp


class PoseSmoother:
    """One Euro filtering with nearest-person association between inferences."""

    def __init__(self, min_cutoff=1.2, beta=0.035, derivative_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.tracks = []

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _match_cost(box, previous_box):
        box = np.asarray(box, dtype=np.float32)
        center = (box[:2] + box[2:]) * 0.5
        previous_center = (previous_box[:2] + previous_box[2:]) * 0.5
        scale = max(
            1.0,
            float(np.linalg.norm(previous_box[2:] - previous_box[:2])),
        )
        return float(np.linalg.norm(center - previous_center) / scale)

    def update(self, detections, timestamp=None):
        timestamp = time.perf_counter() if timestamp is None else float(timestamp)
        detections = list(detections)
        unmatched_tracks = set(range(len(self.tracks)))
        updated_tracks = []

        for detection in detections:
            best_index = None
            best_cost = float("inf")
            for index in unmatched_tracks:
                cost = self._match_cost(detection[0], self.tracks[index].box)
                if cost < best_cost:
                    best_cost = cost
                    best_index = index

            # A center displacement larger than one previous box diagonal is
            # treated as a new person rather than forcing a bad association.
            if best_index is None or best_cost > 1.0:
                track = _PoseTrack(detection, timestamp)
            else:
                unmatched_tracks.remove(best_index)
                track = self.tracks[best_index]
                self._filter_track(track, detection, timestamp)
            updated_tracks.append(track)

        self.tracks = updated_tracks
        return [
            (track.box.copy(), track.score, track.keypoints.copy())
            for track in self.tracks
        ]

    def _filter_track(self, track, detection, timestamp):
        box, score, keypoints = detection
        keypoints = np.asarray(keypoints, dtype=np.float32)
        raw_xy = keypoints[:, :2]
        dt = min(0.25, max(1e-3, timestamp - track.timestamp))

        raw_velocity = (raw_xy - track.raw_xy) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        track.velocity += derivative_alpha * (raw_velocity - track.velocity)

        speed = np.linalg.norm(track.velocity, axis=1)
        cutoff = self.min_cutoff + self.beta * speed
        alpha = np.asarray(
            [self._alpha(value, dt) for value in cutoff], dtype=np.float32
        )[:, None]
        track.keypoints[:, :2] += alpha * (raw_xy - track.keypoints[:, :2])
        track.keypoints[:, 2] = keypoints[:, 2]

        # A modest box EMA keeps person association stable without affecting
        # the skeleton's responsiveness.
        track.box += 0.5 * (np.asarray(box, dtype=np.float32) - track.box)
        track.score = float(score)
        track.raw_xy = raw_xy.copy()
        track.timestamp = timestamp
