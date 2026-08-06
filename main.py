"""
Conveyor object speed tracker - webcam prototype.

Pipeline: detector (MOG2 or HSV) -> bounding box -> centroid -> Kalman
smoothing -> euclidean distance over time -> px/cm calibration -> moving
average -> HUD + CSV log.

Usage:
    python main.py                     # MOG2 detector, default calibration
    python main.py --mode hsv          # color-mask detector
    python main.py --calibrate         # interactive pixels-per-cm calibration
    python main.py --log speeds.csv    # write timestamp,speed_cm_s rows
"""

import argparse
import csv
import time
from collections import deque

import cv2
import numpy as np

import config
from kalman import build_kalman_filter, predict_and_correct, reset as kalman_reset
from trackers import MOG2Tracker, HSVTracker


def run_calibration(cap):
    """Click two points of a known real-world distance to derive
    PIXELS_PER_CM. Returns the computed value, or None if cancelled."""

    points = []

    def on_click(event, x, y, flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    window = "Calibration - click 2 points of known distance, press q to cancel"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)

    while True:
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)

        for p in points:
            cv2.circle(frame, p, 5, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(frame, points[0], points[1], (0, 255, 0), 2)

        cv2.putText(frame, f"Points: {len(points)}/2", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(window)
            return None
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            # window was closed via its X button rather than 'q' - cv2.imshow
            # would otherwise silently recreate it on the next iteration
            return None
        if len(points) == 2:
            break

    cv2.destroyWindow(window)
    dist_px = float(np.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]))

    try:
        real_cm = float(input(f"Distance in pixels = {dist_px:.1f}. Enter the real distance in cm: "))
    except ValueError:
        print("Invalid input, calibration cancelled.")
        return None

    if real_cm <= 0:
        print("Distance must be positive, calibration cancelled.")
        return None

    pixels_per_cm = dist_px / real_cm
    print(f"PIXELS_PER_CM = {pixels_per_cm:.3f} (update config.py to persist this value)")
    return pixels_per_cm


def main():
    parser = argparse.ArgumentParser(description="Conveyor object speed tracker")
    parser.add_argument("--mode", choices=["mog2", "hsv"], default="mog2",
                         help="Detection method (default: mog2)")
    parser.add_argument("--calibrate", action="store_true",
                         help="Run interactive pixels-per-cm calibration before tracking")
    parser.add_argument("--log", metavar="FILE.csv", default=None,
                         help="Write timestamp,speed_cm_s rows to this CSV file")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--gige-ip", metavar="IP", default=None,
                         help="Connect directly to a GigE Vision camera at this IP (same network segment as this Mac)")
    parser.add_argument("--stream-url", metavar="URL", default=None,
                         help="Read an MJPEG/RTSP stream URL instead of a local webcam, "
                              "e.g. http://<windows-ip>:5000/video from windows_server.py")
    parser.add_argument("--debug", action="store_true",
                         help="Print verbose GenTL/device discovery info (GigE Vision mode only)")
    parser.add_argument("--cti-path", metavar="PATH", default=None,
                         help="Force a specific GenTL .cti producer path instead of auto-discovery "
                              "(GigE Vision mode only), e.g. the vendor's GEV producer")
    parser.add_argument("--max-width", type=int, default=None,
                         help="Downscale (via sensor binning) to at most this width, useful on "
                              "bandwidth-limited links like a 100 Mbit USB-Ethernet dock (GigE mode only)")
    parser.add_argument("--target-fps", type=float, default=None,
                         help="Cap the camera's acquisition frame rate (GigE mode only)")
    parser.add_argument("--packet-size", type=int, default=None,
                         help="Force GevSCPSPacketSize in bytes (GigE mode only). Default 1400, "
                              "safely under a standard 1500 MTU. Lower it further (e.g. 1000) if "
                              "fetches still time out.")
    args = parser.parse_args()

    if args.gige_ip:
        from gige_capture import GigeCapture
        cap = GigeCapture(args.gige_ip, cti_path=args.cti_path, debug=args.debug,
                           max_width=args.max_width, target_fps=args.target_fps,
                           packet_size=args.packet_size)
    elif args.stream_url:
        cap = cv2.VideoCapture(args.stream_url)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open stream at {args.stream_url}. Check the Windows "
                "server is running and both machines are on the same network."
            )
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check camera permissions for Terminal/IDE "
                "in System Settings > Privacy & Security > Camera."
            )

    pixels_per_cm = config.PIXELS_PER_CM
    if args.calibrate:
        result = run_calibration(cap)
        if result is not None:
            pixels_per_cm = result

    tracker = MOG2Tracker() if args.mode == "mog2" else HSVTracker()
    kf = build_kalman_filter()
    kalman_initialized = False

    prev_position = None
    prev_timestamp = None
    speed_history = deque(maxlen=config.SPEED_SMOOTHING_WINDOW)
    trail = deque(maxlen=config.TRAIL_LENGTH)

    fps_timestamps = deque(maxlen=30)

    csv_file = open(args.log, "w", newline="") if args.log else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(["timestamp", "speed_cm_s"])

    print(f"Mode: {args.mode} | PIXELS_PER_CM: {pixels_per_cm:.3f} | press q to quit")

    consecutive_failures = 0
    max_consecutive_failures = 30

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"{max_consecutive_failures} consecutive frame reads failed, stopping.")
                    break
                continue
            consecutive_failures = 0

            frame = cv2.flip(frame, 1)
            now = time.time()
            fps_timestamps.append(now)

            bbox, mask = tracker.detect(frame)
            speed_cm_s = 0.0

            if bbox is not None:
                x, y, w, h = bbox
                raw_cx, raw_cy = x + w // 2, y + h // 2

                if not kalman_initialized:
                    kalman_reset(kf, raw_cx, raw_cy)
                    kalman_initialized = True

                cx, cy = predict_and_correct(kf, raw_cx, raw_cy)
                cx, cy = int(cx), int(cy)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                trail.append((cx, cy))

                if prev_position is not None:
                    dt = now - prev_timestamp
                    if dt > 0:
                        dist_px = float(np.hypot(cx - prev_position[0], cy - prev_position[1]))
                        dist_cm = dist_px / pixels_per_cm
                        raw_speed = dist_cm / dt

                        if raw_speed <= config.MAX_SPEED_CM_S:
                            speed_history.append(raw_speed)
                            if csv_writer:
                                csv_writer.writerow([f"{now:.3f}", f"{raw_speed:.2f}"])

                prev_position = (cx, cy)
                prev_timestamp = now
            else:
                prev_position = None
                prev_timestamp = None
                speed_history.clear()
                trail.clear()
                kalman_initialized = False

            if speed_history:
                speed_cm_s = sum(speed_history) / len(speed_history)

            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], (0, 200, 255), 2)

            fps = 0.0
            if len(fps_timestamps) >= 2:
                fps = (len(fps_timestamps) - 1) / (fps_timestamps[-1] - fps_timestamps[0])

            cv2.putText(frame, f"Speed: {speed_cm_s:.1f} cm/s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.0f}  mode: {args.mode}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Conveyor Speed Tracker", frame)
            cv2.imshow("Detection Mask", mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if (cv2.getWindowProperty("Conveyor Speed Tracker", cv2.WND_PROP_VISIBLE) < 1
                    or cv2.getWindowProperty("Detection Mask", cv2.WND_PROP_VISIBLE) < 1):
                break
    finally:
        if csv_file:
            csv_file.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
