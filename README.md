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

## Moving to an industrial camera (Cognex GigE Vision)

### Option A — camera and code on the same machine (e.g. everything on Windows)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install harvesters
python main.py --gige-ip 192.168.1.125 --calibrate
```

Requires a GenTL `.cti` producer on that machine (macOS: `brew install aravis`; Windows: install a GenTL-compliant SDK such as MatrixVision mvIMPACT Acquire, Pleora eBUS SDK, Cognex GigE Vision Configuration Tool, HuarayTech MV Viewer, or the camera vendor's driver). [gige_capture.py](gige_capture.py) auto-discovers it under `/opt/homebrew`, `/usr/local`, `C:/Program Files`, `C:/Program Files (x86)`, or the `GENICAM_GENTL64_PATH` env var.

```bash
# search for the .cti yourself if auto-discovery fails
find /opt/homebrew -name "*.cti"                     # macOS
```

```powershell
# Windows (PowerShell) - search and view result without scrolling a long list
Get-ChildItem -Path "C:\Program Files","C:\Program Files (x86)" -Recurse -Filter *.cti -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName | Out-File -FilePath "$env:USERPROFILE\Desktop\cti_path.txt"
notepad "$env:USERPROFILE\Desktop\cti_path.txt"

# check a specific vendor's install folder, e.g. HuarayTech MV Viewer or Cognex tools
Get-ChildItem "C:\Program Files\MVS","C:\Program Files\HuarayTech","C:\Program Files (x86)\MVS","C:\Program Files (x86)\HuarayTech" -ErrorAction SilentlyContinue
```

### Debugging a Cognex GigE Vision camera (example IP 192.168.125.20)

Run with `--debug` to print GenTL discovery details: producer path used, every device Harvesters can see on the network, the chosen device's pixel format/resolution, and the reason if connection or frame fetch fails.

```powershell
venv\Scripts\activate
pip install harvesters
python main.py --gige-ip 192.168.125.20 --debug --calibrate
```

Common causes if `--debug` shows `device_info_list is EMPTY`:
- Windows PC is not on the same subnet as the camera (camera at `192.168.125.20` needs the PC's NIC set to e.g. `192.168.125.10 / 255.255.255.0`).
- Windows Firewall blocks GigE Vision discovery (UDP). Temporarily disable or add an inbound rule for the Python interpreter.
- The Cognex GigE Vision Configuration Tool (or another app) already has the camera open — close it before running the script, GigE Vision cameras generally only allow one consumer at a time.
- No `.cti` producer installed at all — `find_cti_path` raises before reaching discovery; install one of the SDKs listed above.

If the device is found but `device_info is None` (IP not matched) or frames don't arrive, check the IP printed in the `[gige] found device: ...` line against `192.168.125.20` exactly — Harvesters matches on a substring of the device info string, so confirm it's actually present there.

### Option B — camera on Windows, code/display on Mac

Run a small MJPEG bridge on the Windows PC where the camera is plugged in, then point the Mac at it over the LAN.

Windows (camera side):
```powershell
pip install opencv-python flask
python windows_server.py --camera 0
```

Mac (display/processing side):
```bash
python3 main.py --stream-url http://<windows-ip>:5000/video
```

Use Option B if the camera cannot be reached directly from the Mac's network interface (different subnet, camera physically wired only to the Windows PC). See [windows_server.py](windows_server.py) for the bridge implementation, and adapt `capture_loop` there if the camera does not appear as a plain DirectShow device and needs the vendor SDK instead.

### General notes

- Re-run `--calibrate` and re-tune detection thresholds in [config.py](config.py) for the new lens/resolution/lighting — nothing else in the pipeline needs to change.
- Pixel format (`Mono8`, `BayerRG8`...) is set in the Cognex GigE Vision Configuration Tool; adjust the conversion in `GigeCapture.read()` in [gige_capture.py](gige_capture.py) if colors look wrong.
- Enable jumbo frames (MTU 9000) on the network interface for high-resolution GigE Vision streams to avoid dropped/corrupted frames.
