"""GigE Vision capture via Harvesters + Aravis GenTL producer.

Drop-in replacement for cv2.VideoCapture: exposes .read() -> (ok, frame_bgr)
and .release(), so main.py only needs to swap the capture object.
"""

import glob

import cv2
import numpy as np
from harvesters.core import Harvester


def find_cti_path():
    candidates = glob.glob("/opt/homebrew/**/*.cti", recursive=True)
    candidates += glob.glob("/usr/local/**/*.cti", recursive=True)
    if not candidates:
        raise RuntimeError(
            "No .cti GenTL producer found. Install Aravis: brew install aravis"
        )
    return candidates[0]


class GigeCapture:
    def __init__(self, ip_address, cti_path=None):
        self.harvester = Harvester()
        self.harvester.add_file(cti_path or find_cti_path())
        self.harvester.update()

        device_info = next(
            (d for d in self.harvester.device_info_list if ip_address in str(d)),
            None,
        )
        if device_info is None:
            available = [str(d) for d in self.harvester.device_info_list]
            raise RuntimeError(
                f"No GigE Vision device found at {ip_address}. "
                f"Devices seen: {available or 'none'}"
            )

        self.acquirer = self.harvester.create(device_info)
        self.acquirer.start()

    def read(self):
        try:
            with self.acquirer.fetch(timeout=2) as buffer:
                component = buffer.payload.components[0]
                frame = component.data.reshape(component.height, component.width)

                pixel_format = component.data_format
                if "Mono" in pixel_format:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif "Bayer" in pixel_format:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
                else:
                    frame_bgr = frame  # already BGR/RGB-like, adjust if colors look swapped

                return True, frame_bgr.copy()
        except Exception:
            return False, None

    def isOpened(self):
        return True

    def release(self):
        self.acquirer.stop()
        self.acquirer.destroy()
        self.harvester.reset()
