"""
Run this on the Windows PC where the Cognex GigE Vision camera is connected.
Captures frames and serves them as an MJPEG stream over HTTP so the Mac can
read it with plain cv2.VideoCapture(url) - no GigE/GenTL libraries needed on
the Mac side.

Install (Windows, in a venv):
    pip install opencv-python flask

If the Cognex GigE Vision Configuration Tool installed a DirectShow filter
for the camera, it shows up as a normal capture device and --camera <index>
below will work directly with cv2.VideoCapture. If cv2.VideoCapture cannot
open it, use the camera vendor's own SDK/driver to grab frames instead and
feed them into `latest_frame` the same way (see NOTE in capture_loop).

Run:
    python windows_server.py --camera 0
    python windows_server.py --camera 0 --port 5000

Then on the Mac:
    python3 main.py --stream-url http://<windows-ip>:5000/video
"""

import argparse
import threading
import time

import cv2
from flask import Flask, Response

app = Flask(__name__)

latest_frame = None
frame_lock = threading.Lock()


def capture_loop(camera_index):
    global latest_frame

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. If this is the "
            "Cognex GigE camera and it does not appear as a DirectShow "
            "device, use the vendor SDK to grab frames and assign them to "
            "`latest_frame` here instead of cv2.VideoCapture."
        )

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        with frame_lock:
            latest_frame = frame


def mjpeg_generator():
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )


@app.route("/video")
def video():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return "<html><body><img src='/video'></body></html>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index on this Windows PC")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    thread = threading.Thread(target=capture_loop, args=(args.camera,), daemon=True)
    thread.start()

    print(f"Streaming on http://0.0.0.0:{args.port}/video")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
