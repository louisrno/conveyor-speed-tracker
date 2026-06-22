# Conveyor Speed Tracker

Webcam prototype for measuring the speed of an object moving across a conveyor belt, using OpenCV. Built as a desk-test stand-in before connecting an industrial camera.

## Pipeline

1. Detect the moving object (background subtraction with MOG2, or HSV color mask).
2. Extract the bounding box and centroid.
3. Smooth the centroid with a 2D constant-velocity Kalman filter.
4. Compute euclidean distance between consecutive positions and divide by elapsed time.
5. Convert pixels to centimeters using a calibration constant, apply a moving average, display on screen.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```

Must run from a real Terminal/iTerm window so macOS can prompt for camera access on first launch.

### Options

```bash
python3 main.py --mode hsv          # color-mask detector instead of MOG2
python3 main.py --calibrate         # interactive pixels-per-cm calibration (click 2 points of known distance)
python3 main.py --log speeds.csv    # log timestamp,speed_cm_s rows to CSV
python3 main.py --camera 1          # use a different camera index
```

Press `q` in the video window to quit.

## Tuning

All constants live in [config.py](config.py):

- `PIXELS_PER_CM` — set via `--calibrate`, or measure manually (known-width object in frame, pixels / real cm).
- `MOG2_VAR_THRESHOLD` — lower = more sensitive to motion, more false positives from lighting changes.
- `MIN_CONTOUR_AREA` — raise if small artifacts (hands, shadows) get tracked as the object.
- `HSV_LOWER` / `HSV_UPPER` — color range for `--mode hsv`, tune to your object's color.
- `SPEED_SMOOTHING_WINDOW` — number of samples averaged for the displayed speed.
- `KALMAN_PROCESS_NOISE` / `KALMAN_MEASUREMENT_NOISE` — lower process noise = smoother but laggier position.

## Project structure

```
main.py        # CLI entry point, capture loop, HUD, CSV logging, calibration mode
trackers.py     # MOG2Tracker and HSVTracker detectors
kalman.py       # Kalman filter setup/update helpers
config.py       # tunable constants
requirements.txt
```

## Moving to an industrial camera

- GigE/USB3 SDK camera (Basler, FLIR, Cognex...): swap `cv2.VideoCapture` in `main.py` for the SDK's frame grab, keep the rest of the pipeline unchanged.
- RTSP/GigE Vision standard stream: `cv2.VideoCapture("rtsp://...")` may work directly.
- Re-run `--calibrate` and re-tune detection thresholds for the new lens/resolution/lighting — nothing else in the pipeline needs to change.
