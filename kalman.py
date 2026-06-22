"""2D constant-velocity Kalman filter to smooth the centroid position before
computing speed. Reduces jitter from contour-detection noise."""

import cv2
import numpy as np

import config


def build_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)  # state: [x, y, vx, vy], measurement: [x, y]

    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float32
    )
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * config.KALMAN_PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * config.KALMAN_MEASUREMENT_NOISE
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    return kf


def reset(kf, x, y):
    kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)


def predict_and_correct(kf, x, y):
    kf.predict()
    corrected = kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))
    return float(corrected[0, 0]), float(corrected[1, 0])
