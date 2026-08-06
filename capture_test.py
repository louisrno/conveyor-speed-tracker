"""
Standalone script to grab and save one frame from the Cognex GigE Vision
camera. Copy this single file to the Windows PC and run it there.

Install (once):
    pip install harvesters opencv-python numpy

Run:
    python capture_test.py
"""

import cv2
import numpy as np
from harvesters.core import Harvester

CTI_PATH = r"C:\Program Files\HuarayTech\MV Viewer\Runtime\x64\MVProducerGEV.cti"
CAMERA_IP = "169.254.55.20"
OUTPUT_FILE = "frame.png"


def main():
    harvester = Harvester()
    harvester.add_file(CTI_PATH)
    harvester.update()

    print("Devices found:")
    for d in harvester.device_info_list:
        print(f"  {d}")

    if not harvester.device_info_list:
        print("No devices found at all.")
        return

    # device_info_list entries don't expose the IP directly (just model/serial/
    # vendor), so just grab the first (and presumably only) camera instead of
    # filtering by CAMERA_IP.
    device_info = harvester.device_info_list[0]
    acquirer = harvester.create(device_info)

    try:
        actual_ip = acquirer.remote_device.node_map.GevCurrentIPAddress.value
        print(f"Connected device IP: {actual_ip}")
    except Exception:
        pass

    try:
        node_map = acquirer.remote_device.node_map
        print(f"PixelFormat: {node_map.PixelFormat.value}")
        print(f"Width x Height: {node_map.Width.value} x {node_map.Height.value}")
    except Exception as exc:
        print(f"Could not read node map: {exc}")

    acquirer.start()
    print("Fetching one frame...")

    with acquirer.fetch(timeout=5) as buffer:
        component = buffer.payload.components[0]
        frame = component.data.reshape(component.height, component.width)

        pixel_format = component.data_format
        if "Mono" in pixel_format:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif "Bayer" in pixel_format:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        else:
            frame_bgr = frame

        cv2.imwrite(OUTPUT_FILE, frame_bgr)
        print(f"Saved {OUTPUT_FILE} ({frame_bgr.shape[1]}x{frame_bgr.shape[0]})")

    acquirer.stop()
    acquirer.destroy()
    harvester.reset()


if __name__ == "__main__":
    main()
