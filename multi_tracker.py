"""Multi-object tracking: assigns a stable ID to each detected object across
frames, smooths its position with its own Kalman filter, computes its speed,
and samples its dominant color.

Association is nearest-centroid matching (greedy, by distance) - simple and
fast, sufficient for a handful of objects moving in one direction on a
conveyor. Not a full Hungarian/IoU tracker.
"""

import time
from collections import deque

import cv2
import numpy as np

import config
from kalman import build_kalman_filter, predict_and_correct, reset as kalman_reset

COLOR_NAMES = {
    "red": ((0, 100, 80), (10, 255, 255)),
    "orange": ((11, 100, 80), (25, 255, 255)),
    "yellow": ((26, 100, 80), (35, 255, 255)),
    "green": ((36, 60, 60), (85, 255, 255)),
    "blue": ((86, 60, 60), (130, 255, 255)),
    "purple": ((131, 60, 60), (155, 255, 255)),
    "red2": ((156, 100, 80), (180, 255, 255)),  # red wraps around hue 0/180
}

MAX_MATCH_DISTANCE_PX = 150  # don't associate a detection to a track further than this
MAX_MISSED_FRAMES = 15       # drop a track after this many frames with no matching detection


def white_balance(frame):
    """Gray-world white balance: scales each channel so their means match,
    removing an ambient color cast (e.g. cyan-tinted industrial lighting)
    that would otherwise bias hue-based classification - a yellow object
    under cyan light reads with a green-shifted hue without this."""
    b, g, r = cv2.split(frame.astype(np.float32))
    b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
    gray_mean = (b_mean + g_mean + r_mean) / 3.0

    # guard against a near-black frame (mean ~0) causing a huge/unstable gain
    if min(b_mean, g_mean, r_mean) < 1.0:
        return frame

    b *= gray_mean / b_mean
    g *= gray_mean / g_mean
    r *= gray_mean / r_mean
    return cv2.merge([b, g, r]).clip(0, 255).astype(np.uint8)


def classify_color(frame, bbox):
    """Sample the color inside the bbox and classify it into a name.

    Uses the center ~50% of the box (edges are the likeliest place to catch
    background/conveyor pixels bleeding in) and a per-pixel HSV median
    (robust to specular highlights and shadow edges, unlike a BGR mean)."""
    x, y, w, h = bbox
    inset_x, inset_y = w // 4, h // 4
    cx0, cy0 = x + inset_x, y + inset_y
    cx1, cy1 = x + w - inset_x, y + h - inset_y
    roi = frame[cy0:cy1, cx0:cx1]
    if roi.size == 0:
        roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return "unknown", (128, 128, 128)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue, sat, val = (int(v) for v in np.median(hsv_roi, axis=0))

    median_bgr_uint8 = np.uint8([[[hue, sat, val]]])
    median_bgr = cv2.cvtColor(median_bgr_uint8, cv2.COLOR_HSV2BGR)[0][0]
    color_bgr = tuple(int(c) for c in median_bgr)

    # Low-saturation pixels are grayscale/metallic regardless of measured
    # hue - ambient light tint alone can otherwise push them into a color
    # bucket. Require a real amount of saturation before trusting hue.
    if sat < 60 or val < 40:
        name = "gray/black" if val < 100 else "white/gray"
        return name, color_bgr

    for name, (lower, upper) in COLOR_NAMES.items():
        if lower[0] <= hue <= upper[0]:
            return name.rstrip("2"), color_bgr

    return "unknown", color_bgr


class TrackedObject:
    def __init__(self, track_id, bbox, color_name, color_bgr, timestamp):
        self.id = track_id
        self.bbox = bbox
        self.color_name = color_name
        self.color_bgr = color_bgr
        self.kf = build_kalman_filter()

        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        kalman_reset(self.kf, cx, cy)
        self.position = (cx, cy)

        self.prev_position = None
        self.prev_timestamp = None
        self.last_seen_timestamp = timestamp

        self.speed_history = deque(maxlen=config.SPEED_SMOOTHING_WINDOW)
        self.trail = deque(maxlen=config.TRAIL_LENGTH)
        self.trail.append(self.position)
        self.missed_frames = 0

    @property
    def speed_cm_s(self):
        if not self.speed_history:
            return 0.0
        return sum(self.speed_history) / len(self.speed_history)

    def update(self, bbox, color_name, color_bgr, pixels_per_cm, timestamp):
        self.bbox = bbox
        self.color_name = color_name
        self.color_bgr = color_bgr
        self.missed_frames = 0
        self.last_seen_timestamp = timestamp

        x, y, w, h = bbox
        raw_cx, raw_cy = x + w // 2, y + h // 2
        cx, cy = predict_and_correct(self.kf, raw_cx, raw_cy)
        cx, cy = int(cx), int(cy)
        self.position = (cx, cy)
        self.trail.append(self.position)

        if self.prev_position is not None:
            dt = timestamp - self.prev_timestamp
            if dt > 0:
                dist_px = float(np.hypot(cx - self.prev_position[0], cy - self.prev_position[1]))
                dist_cm = dist_px / pixels_per_cm
                raw_speed = dist_cm / dt
                if raw_speed <= config.MAX_SPEED_CM_S:
                    self.speed_history.append(raw_speed)

        self.prev_position = self.position
        self.prev_timestamp = timestamp

    def mark_missed(self):
        self.missed_frames += 1


class MultiObjectTracker:
    def __init__(self, pixels_per_cm):
        self.pixels_per_cm = pixels_per_cm
        self.tracks = {}
        self._next_id = 1

    def update(self, frame, bboxes):
        timestamp = time.time()
        balanced_frame = white_balance(frame)
        detections = []
        for bbox in bboxes:
            color_name, color_bgr = classify_color(balanced_frame, bbox)
            x, y, w, h = bbox
            cx, cy = x + w // 2, y + h // 2
            detections.append((bbox, color_name, color_bgr, (cx, cy)))

        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        # Greedy nearest-centroid matching: repeatedly pick the closest
        # (track, detection) pair under the distance threshold.
        pairs = []
        for track_id in unmatched_tracks:
            tx, ty = self.tracks[track_id].position
            for det_idx in unmatched_detections:
                _, _, _, (dx, dy) = detections[det_idx]
                dist = np.hypot(tx - dx, ty - dy)
                if dist <= MAX_MATCH_DISTANCE_PX:
                    pairs.append((dist, track_id, det_idx))
        pairs.sort(key=lambda p: p[0])

        matched_tracks = set()
        matched_detections = set()
        for _dist, track_id, det_idx in pairs:
            if track_id in matched_tracks or det_idx in matched_detections:
                continue
            bbox, color_name, color_bgr, _ = detections[det_idx]
            self.tracks[track_id].update(bbox, color_name, color_bgr, self.pixels_per_cm, timestamp)
            matched_tracks.add(track_id)
            matched_detections.add(det_idx)

        for track_id in unmatched_tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id].mark_missed()

        for det_idx in range(len(detections)):
            if det_idx not in matched_detections:
                bbox, color_name, color_bgr, _ = detections[det_idx]
                track_id = self._next_id
                self._next_id += 1
                self.tracks[track_id] = TrackedObject(track_id, bbox, color_name, color_bgr, timestamp)

        stale_ids = [tid for tid, t in self.tracks.items() if t.missed_frames > MAX_MISSED_FRAMES]
        for tid in stale_ids:
            del self.tracks[tid]

        return list(self.tracks.values())
