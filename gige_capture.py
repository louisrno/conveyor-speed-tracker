"""GigE Vision capture via Harvesters + a GenTL producer.

On macOS the producer comes from Aravis (brew install aravis). On Windows it
comes from whichever GenTL-compliant SDK is installed (e.g. MatrixVision
mvIMPACT Acquire, Pleora eBUS SDK, or the camera vendor's own driver).

Drop-in replacement for cv2.VideoCapture: exposes .read() -> (ok, frame_bgr)
and .release(), so main.py only needs to swap the capture object.
"""

import glob
import os

import cv2
import numpy as np
from harvesters.core import Harvester


def find_cti_path(debug=False):
    candidates = glob.glob("/opt/homebrew/**/*.cti", recursive=True)
    candidates += glob.glob("/usr/local/**/*.cti", recursive=True)
    candidates += glob.glob("C:/Program Files/**/*.cti", recursive=True)
    candidates += glob.glob("C:/Program Files (x86)/**/*.cti", recursive=True)

    env_path = os.environ.get("GENICAM_GENTL64_PATH") or os.environ.get("GENICAM_GENTL32_PATH")
    if env_path:
        for folder in env_path.split(os.pathsep):
            candidates += glob.glob(os.path.join(folder, "*.cti"))

    if debug:
        print(f"[gige] GENICAM_GENTL64_PATH={os.environ.get('GENICAM_GENTL64_PATH')}")
        print(f"[gige] GENICAM_GENTL32_PATH={os.environ.get('GENICAM_GENTL32_PATH')}")
        print(f"[gige] .cti candidates found: {candidates or 'none'}")

    if not candidates:
        raise RuntimeError(
            "No .cti GenTL producer found. macOS: brew install aravis. "
            "Windows: install a GenTL-compliant SDK (mvIMPACT Acquire, "
            "Pleora eBUS SDK, Cognex GigE Vision Configuration Tool, or the "
            "camera vendor's driver) and retry. Use --debug to print search paths."
        )
    if debug:
        print(f"[gige] using: {candidates[0]}")
    return candidates[0]


class GigeCapture:
    def __init__(self, ip_address, cti_path=None, debug=False):
        self.debug = debug

        self.harvester = Harvester()
        resolved_cti = cti_path or find_cti_path(debug=debug)
        self.harvester.add_file(resolved_cti)
        self.harvester.update()

        if debug:
            print(f"[gige] looking for device at IP {ip_address}")
            if not self.harvester.device_info_list:
                print("[gige] device_info_list is EMPTY - no GigE Vision device discovered at all. "
                      "Check: same subnet as camera, firewall allows UDP discovery, "
                      "camera not already opened by another app (e.g. Cognex Configuration Tool).")
            for d in self.harvester.device_info_list:
                print(f"[gige] found device: {d}")

        device_info = next(
            (d for d in self.harvester.device_info_list if ip_address in str(d)),
            None,
        )
        if device_info is None:
            available = [str(d) for d in self.harvester.device_info_list]
            raise RuntimeError(
                f"No GigE Vision device found at {ip_address}. "
                f"Devices seen: {available or 'none'}. Run with --debug for more detail, "
                "and make sure no other tool (e.g. the Cognex GigE Vision Configuration "
                "Tool itself) is holding the camera open."
            )

        self.acquirer = self.harvester.create(device_info)

        if debug:
            try:
                node_map = self.acquirer.remote_device.node_map
                print(f"[gige] PixelFormat: {node_map.PixelFormat.value}")
                print(f"[gige] Width x Height: {node_map.Width.value} x {node_map.Height.value}")
            except Exception as exc:
                print(f"[gige] could not read node map: {exc}")

        self.acquirer.start()
        if debug:
            print("[gige] acquisition started")

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
        except Exception as exc:
            if self.debug:
                print(f"[gige] fetch failed: {exc}")
            return False, None

    def isOpened(self):
        return True

    def release(self):
        self.acquirer.stop()
        self.acquirer.destroy()
        self.harvester.reset()
