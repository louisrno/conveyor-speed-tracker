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
    def __init__(self, ip_address, cti_path=None, debug=False, max_width=None,
                 target_fps=None, packet_size=None):
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

        if not self.harvester.device_info_list:
            raise RuntimeError(
                "No GigE Vision device found at all. Run with --debug for more detail, "
                "and make sure no other tool (e.g. the Cognex GigE Vision Configuration "
                "Tool or VisionPro) is holding the camera open."
            )

        # device_info entries don't expose the IP directly (just model/serial/
        # vendor), so if there's exactly one device just use it. Otherwise try
        # to match by opening each and reading its actual IP from the node map.
        if len(self.harvester.device_info_list) == 1:
            device_info = self.harvester.device_info_list[0]
        else:
            device_info = None
            for candidate in self.harvester.device_info_list:
                probe = self.harvester.create(candidate)
                try:
                    candidate_ip = probe.remote_device.node_map.GevCurrentIPAddress.value
                    if debug:
                        print(f"[gige] candidate {candidate} -> IP {candidate_ip}")
                    if str(candidate_ip) == ip_address:
                        device_info = candidate
                        probe.destroy()
                        break
                except Exception as exc:
                    if debug:
                        print(f"[gige] could not read IP for {candidate}: {exc}")
                probe.destroy()

            if device_info is None:
                available = [str(d) for d in self.harvester.device_info_list]
                raise RuntimeError(
                    f"No GigE Vision device matched IP {ip_address}. "
                    f"Devices seen: {available}. Run with --debug for more detail."
                )

        self.acquirer = self.harvester.create(device_info)

        node_map = self.acquirer.remote_device.node_map

        if debug:
            try:
                print(f"[gige] PixelFormat: {node_map.PixelFormat.value}")
                print(f"[gige] Width x Height: {node_map.Width.value} x {node_map.Height.value}")
            except Exception as exc:
                print(f"[gige] could not read node map: {exc}")

        # Bandwidth-limited links (e.g. a 100 Mbit USB-Ethernet dock) cannot
        # sustain a full-resolution continuous stream from a multi-megapixel
        # sensor - reduce resolution via binning and/or cap the frame rate.
        if max_width is not None:
            try:
                sensor_width = node_map.Width.value
                sensor_height = node_map.Height.value
                if sensor_width > max_width:
                    scale = max_width / sensor_width
                    new_width = int(sensor_width * scale) - (int(sensor_width * scale) % 4)
                    new_height = int(sensor_height * scale) - (int(sensor_height * scale) % 4)

                    # OffsetX/Y must be reset to 0 before shrinking Width/Height on
                    # most GenICam cameras, otherwise the new size can exceed the
                    # sensor bounds relative to the current offset.
                    try:
                        node_map.OffsetX.value = 0
                        node_map.OffsetY.value = 0
                    except Exception:
                        pass

                    node_map.Width.value = new_width
                    node_map.Height.value = new_height
                    if debug:
                        print(f"[gige] applied ROI crop, new size "
                              f"{node_map.Width.value}x{node_map.Height.value}")
            except Exception as exc:
                if debug:
                    print(f"[gige] could not apply ROI/Width-Height reduction: {exc}")

        # Classic GigE Vision failure mode: the camera's stream channel packet
        # size (GevSCPSPacketSize) defaults to something larger than the
        # host NIC's MTU (standard Ethernet = 1500). Every packet then gets
        # silently dropped and fetch() times out forever, regardless of
        # resolution. Force it down to a size that fits a standard MTU.
        try:
            current_packet_size = node_map.GevSCPSPacketSize.value
            if debug:
                print(f"[gige] current GevSCPSPacketSize: {current_packet_size}")
            # 1500 is the standard Ethernet MTU, but GevSCPSPacketSize counts
            # the full packet including IP/UDP/GVSP headers (~36-42 bytes) on
            # most cameras, so requesting exactly 1500 still overflows the MTU
            # and gets silently dropped by the NIC/driver. Leave real headroom
            # unless the user overrides it explicitly.
            safe_packet_size = packet_size if packet_size is not None else 1400
            if packet_size is not None or current_packet_size > safe_packet_size:
                node_map.GevSCPSPacketSize.value = safe_packet_size
                if debug:
                    print(f"[gige] set GevSCPSPacketSize to {safe_packet_size} "
                          f"(was {current_packet_size})")
        except Exception as exc:
            if debug:
                print(f"[gige] could not read/set GevSCPSPacketSize: {exc!r}")

        if target_fps is not None:
            try:
                node_map.AcquisitionFrameRateEnable.value = True
                node_map.AcquisitionFrameRate.value = target_fps
                if debug:
                    print(f"[gige] capped AcquisitionFrameRate to {target_fps}")
            except Exception as exc:
                if debug:
                    print(f"[gige] could not set frame rate: {exc}")

        self.acquirer.start()
        if debug:
            print("[gige] acquisition started")

    def read(self):
        try:
            with self.acquirer.fetch(timeout=5) as buffer:
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
                print(f"[gige] fetch failed: {type(exc).__name__}: {exc!r}")
            return False, None

    def isOpened(self):
        return True

    def release(self):
        self.acquirer.stop()
        self.acquirer.destroy()
        self.harvester.reset()
