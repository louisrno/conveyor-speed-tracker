"""Tunable constants for the conveyor speed tracker."""

# Calibration: pixels per centimeter. Use the calibration mode (--calibrate)
# to set this interactively instead of guessing.
PIXELS_PER_CM = 20.0

# Detection
MIN_CONTOUR_AREA = 800
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 16

# HSV color mask defaults (orange-ish target). Override with sliders in
# --mode hsv --tune.
HSV_LOWER = (5, 120, 120)
HSV_UPPER = (25, 255, 255)

# Speed pipeline
SPEED_SMOOTHING_WINDOW = 8
MAX_SPEED_CM_S = 500.0

# Kalman filter process/measurement noise. Lower process noise = smoother but
# laggier position; lower measurement noise = trusts raw detection more.
KALMAN_PROCESS_NOISE = 1e-2
KALMAN_MEASUREMENT_NOISE = 1e-1

# Trajectory trail length (number of past centroids drawn)
TRAIL_LENGTH = 64
