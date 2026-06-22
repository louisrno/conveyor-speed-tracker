"""Object detectors that return a bounding box for the moving object."""

import cv2
import numpy as np

import config


class MOG2Tracker:
    """Background subtraction based detector. Good for any object as long as
    the background stays mostly static."""

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY,
            varThreshold=config.MOG2_VAR_THRESHOLD,
            detectShadows=True,
        )
        self.kernel = np.ones((5, 5), np.uint8)

    def detect(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = self._largest_valid_contour(contours)
        return bbox, fg_mask

    @staticmethod
    def _largest_valid_contour(contours):
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < config.MIN_CONTOUR_AREA:
            return None
        return cv2.boundingRect(largest)


class HSVTracker:
    """Color mask based detector. More stable than MOG2 if the object has a
    distinct, consistent color and the background does not share it."""

    def __init__(self, lower=config.HSV_LOWER, upper=config.HSV_UPPER):
        self.lower = np.array(lower, dtype=np.uint8)
        self.upper = np.array(upper, dtype=np.uint8)
        self.kernel = np.ones((5, 5), np.uint8)

    def set_range(self, lower, upper):
        self.lower = np.array(lower, dtype=np.uint8)
        self.upper = np.array(upper, dtype=np.uint8)

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = MOG2Tracker._largest_valid_contour(contours)
        return bbox, mask
