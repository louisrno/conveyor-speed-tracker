"""
Run this on the Windows PC connected to the Cognex camera. No install needed,
uses only the Python standard library.

Steps:
1. Prints this PC's network adapters and IPs.
2. Sends a GigE Vision discovery broadcast (UDP 3956) and lists every camera
   that responds, with its IP - this works even if the camera got a
   different/unexpected IP.

Run:
    python windows_diagnose.py
"""

import socket
import struct
import subprocess


def print_adapters():
    print("=== Network adapters (ipconfig) ===")
    result = subprocess.run(["ipconfig"], capture_output=True, text=True, shell=True)
    print(result.stdout)


def discover_gige_vision(timeout=3.0):
    print("=== GigE Vision discovery (UDP broadcast on port 3956) ===")

    # GVCP DISCOVERY_CMD packet: header + command
    # See GigE Vision spec: 0x42 flags, 0x0002 = DISCOVERY_CMD, req id, size 0
    packet = struct.pack(">BBHHH", 0x42, 0x11, 0x0002, 0x0000, 0x0001)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(timeout)
    sock.bind(("", 0))

    sock.sendto(packet, ("255.255.255.255", 3956))

    found = []
    try:
        while True:
            data, addr = sock.recvfrom(2048)
            found.append(addr[0])
            print(f"Camera responded from IP: {addr[0]} ({len(data)} bytes)")
    except socket.timeout:
        pass
    finally:
        sock.close()

    if not found:
        print("No GigE Vision device responded. Possible causes:")
        print("- Camera not powered / cable not seated")
        print("- Windows Firewall blocking UDP 3956 (try disabling temporarily)")
        print("- PC's NIC not on the same subnet as the camera")
        print("- Another application (Cognex Configuration Tool) already has an exclusive lock")
    else:
        print(f"\n{len(found)} device(s) found: {found}")


if __name__ == "__main__":
    print_adapters()
    discover_gige_vision()
