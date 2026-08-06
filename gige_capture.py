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
            for node_name in ("TriggerMode", "TriggerSource", "TriggerSelector", "AcquisitionMode"):
                try:
                    value = getattr(node_map, node_name).value
                    print(f"[gige] {node_name}: {value}")
                except Exception as exc:
                    print(f"[gige] could not read {node_name}: {exc!r}")

        # If the camera is waiting for an external hardware/software trigger
        # (common in an automation cell wired to a PLC), free-run fetch()
        # calls will hang/timeout forever since no frame is ever produced on
        # its own. Force free-run acquisition so this script can pull frames
        # continuously on demand instead of waiting for an external trigger.
        try:
            if node_map.TriggerMode.value == "On":
                if debug:
                    print("[gige] TriggerMode was On (camera waiting for external trigger) "
                          "- forcing it Off for free-run capture")
                node_map.TriggerMode.value = "Off"
        except Exception as exc:
            if debug:
                print(f"[gige] could not read/set TriggerMode: {exc!r}")

        # AcquisitionMode=SingleFrame means the camera produces exactly one
        # frame per start() and then stops - every fetch() after the first
        # times out forever since nothing new is ever captured. Force
        # Continuous so it keeps producing frames for the whole session.
        try:
            if node_map.AcquisitionMode.value != "Continuous":
                if debug:
                    print(f"[gige] AcquisitionMode was {node_map.AcquisitionMode.value} "
                          "- forcing it to Continuous")
                node_map.AcquisitionMode.value = "Continuous"
        except Exception as exc:
            if debug:
                print(f"[gige] could not read/set AcquisitionMode: {exc!r}")

        # Width/Height/Offset persist on the camera across script runs (not
        # reset on start()), so always reset to the sensor's true maximum
        # first - otherwise a previous run's crop becomes the new "full
        # frame" for this run, regardless of whether --max-width is used now.
        try:
            node_map.OffsetX.value = 0
            node_map.OffsetY.value = 0
            node_map.Width.value = node_map.WidthMax.value
            node_map.Height.value = node_map.HeightMax.value
            if debug:
                print(f"[gige] reset to full sensor size "
                      f"{node_map.Width.value}x{node_map.Height.value}")
        except Exception as exc:
            if debug:
                print(f"[gige] could not reset to full sensor size: {exc!r}")

        # Bandwidth-limited links (e.g. a 100 Mbit USB-Ethernet dock) cannot
        # sustain a full-resolution continuous stream from a multi-megapixel
        # sensor. Use Decimation (pixel/line subsampling) rather than a
        # Width/Height ROI crop: decimation reduces resolution while keeping
        # the FULL field of view, whereas a Width/Height crop only keeps a
        # smaller window of the sensor (an actual zoom-in, not a downscale).
        if max_width is not None:
            try:
                sensor_width = node_map.Width.value
                if sensor_width > max_width:
                    factor = max(1, round(sensor_width / max_width))
                    applied = False
                    try:
                        node_map.DecimationHorizontal.value = factor
                        node_map.DecimationVertical.value = factor
                        applied = True
                    except Exception as exc:
                        if debug:
                            print(f"[gige] camera does not support Decimation: {exc!r}")

                    if applied and debug:
                        try:
                            offset_x = node_map.OffsetX.value
                            offset_y = node_map.OffsetY.value
                        except Exception:
                            offset_x = offset_y = "unknown"
                        print(f"[gige] applied {factor}x decimation (full FOV kept), new size "
                              f"{node_map.Width.value}x{node_map.Height.value}, "
                              f"offset ({offset_x}, {offset_y}), "
                              f"sensor max {node_map.WidthMax.value}x{node_map.HeightMax.value}")
                    elif not applied and debug:
                        print("[gige] leaving full resolution - decimation unsupported, "
                              "and a Width/Height crop would lose field of view instead of "
                              "just reducing bandwidth. Consider a Gigabit link instead.")
            except Exception as exc:
                if debug:
                    print(f"[gige] could not apply decimation: {exc!r}")

        # Classic GigE Vision failure mode: the camera's stream channel packet
        # size (GevSCPSPacketSize) defaults to something larger than the
        # host NIC's MTU (standard Ethernet = 1500). Every packet then gets
        # silently dropped and fetch() times out forever, regardless of
        # resolution. Force it down to a size that fits a standard MTU.
        try:
            current_packet_size = node_map.GevSCPSPacketSize.value
            if debug:
                print(f"[gige] current GevSCPSPacketSize: {current_packet_size}")
            # Only override if the caller explicitly asked for a specific size -
            # a mismatched MTU isn't necessarily the issue (e.g. jumbo frames
            # may already be enabled on the NIC), so don't fight the camera's
            # own default unless told to.
            if packet_size is not None:
                node_map.GevSCPSPacketSize.value = packet_size
                if debug:
                    print(f"[gige] set GevSCPSPacketSize to {packet_size} "
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
                    # Edge-aware demosaicing (_EA) gives noticeably more
                    # accurate colors than the plain bilinear variant, at a
                    # small CPU cost - worth it since color classification
                    # depends on it.
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR_EA)
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
