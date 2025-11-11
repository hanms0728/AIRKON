import carla
import time
import socket
import json
import signal
import sys
import numpy as np

FPS = 30
UDP_IP = "0.0.0.0"
UDP_PORT = 60200 

# --- CARLA init ---
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.find('vehicle.audi.a2')

vehicle_map = {}
z_offset = 0.2

def move_vehicle_to(vehicle, x, y, z,pitch ,yaw_deg):
    """Teleport vehicle to (x, y, z) with yaw/pitch (degrees)."""
    transform = carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(pitch=pitch, yaw=yaw_deg)
    )
    vehicle.set_transform(transform)

def move_or_spawn_vehicles(info_dict):
    """
    info_dict: { id: (x, y, z, pitch,yaw_deg), ... }
    - Destroy vehicles that disappeared
    - Move existing or spawn new ones
    """
    global vehicle_map
    current_ids = set(info_dict.keys())
    existing_ids = set(vehicle_map.keys())

    # Destroy vehicles no longer present
    for vid in existing_ids - current_ids:
        try:
            vehicle_map[vid].destroy()
        except:
            pass
        vehicle_map.pop(vid, None)

    # Move existing or spawn new
    for vid, (x, y, z,pitch ,yaw) in info_dict.items():
        if vid in vehicle_map:
            move_vehicle_to(vehicle_map[vid], x, y, z+z_offset, pitch ,yaw)
        else:
            transform = carla.Transform(
                carla.Location(x=x, y=y, z=z+z_offset),
                carla.Rotation(pitch=pitch, yaw=yaw)
            )
            vehicle = world.try_spawn_actor(vehicle_bp, transform)
            if vehicle:
                vehicle_map[vid] = vehicle
            else:
                # Spawn failed (collision, blocked, etc.)
                pass

def cleanup():
    """Destroy all spawned vehicles."""
    for v in list(vehicle_map.values()):
        try:
            v.destroy()
        except:
            pass
    vehicle_map.clear()

def main():
    # --- UDP socket (JSON receiver) ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Increase receive buffer to avoid drop under burst
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)  # non-blocking loop with timeout

    print(f"[UDP] Listening on {UDP_IP}:{UDP_PORT}")

    frame_idx = 0
    try:
        while True:
            try:
                # Use large size for UDP datagrams carrying JSON
                data, addr = sock.recvfrom(65507)
            except socket.timeout:
                continue

            # Parse JSON of the form:
            # {
            #   "type": "global_tracks",
            #   "timestamp": 1731292345.123,
            #   "items": [
            #     {"id": 12, "class": 0, "center": [cx, cy, cz], "yaw": 12.3, ...}
            #   ]
            # }
            try:
                msg = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                # Not JSON → ignore silently
                continue

            if msg.get("type") != "global_tracks":
                # Unknown packet type → ignore
                continue

            items = msg.get("items", [])
            vehicle_info = {}

            for it in items:
                try:
                    vid = int(it["id"])
                    center = it.get("center", [0.0, 0.0, 0.0])
                    cx = float(center[0])
                    cy = float(center[1])
                    cz = float(center[2])
                    pitch = float(it.get("pitch", 0.0))  # degree
                    yaw = float(it.get("yaw", 0.0))  # degree
                    # NOTE: If your BEV Y-axis is inverted against CARLA's Y,
                    # you may apply cy = -cy or yaw = -yaw here.
                    vehicle_info[vid] = (cx, cy, cz, pitch,yaw)
                except Exception:
                    # Skip malformed item
                    continue

            print(frame_idx, ":", vehicle_info)
            move_or_spawn_vehicles(vehicle_info)

            # If you want to throttle, uncomment:
            # time.sleep(1.0 / FPS)

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[STOP] KeyboardInterrupt")
    finally:
        try:
            sock.close()
        except:
            pass
        cleanup()

if __name__ == "__main__":
    main()

# https://drive.google.com/file/d/1DEJ_Eiz1gdWv6-t8sDTDfnGi7_5iHqI9/view?usp=sharing