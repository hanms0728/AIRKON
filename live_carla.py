import carla
import time
import socket
import json


COLOR_HEX_MAP = {
    "red": "#f52629",
    "pink": "#f53e96",
    "green": "#48ad0d",
    "white": "#f0f0f0",
    "yellow": "#ffdd00",
    "purple": "#781de7",
    "black": "#000000",
}

DEFAULT_COLOR_LABEL = "white"


def _hex_to_rgb_tuple(value):
    """Convert a #RRGGBB string into an (R, G, B) tuple."""
    if not value:
        return None
    hex_value = value.lstrip("#")
    if len(hex_value) != 6:
        return None
    try:
        r = int(hex_value[0:2], 16)
        g = int(hex_value[2:4], 16)
        b = int(hex_value[4:6], 16)
        return (r, g, b)
    except ValueError:
        return None


COLOR_RGB_MAP = {
    name: _hex_to_rgb_tuple(hex_value)
    for name, hex_value in COLOR_HEX_MAP.items()
}


def normalize_color_label(value):
    """Return a normalized color label if supported, otherwise None."""
    if value is None:
        return None
    color = str(value).strip().lower()
    return color if color in COLOR_RGB_MAP else None


def color_label_to_attr_value(color_label):
    rgb = COLOR_RGB_MAP.get(color_label)
    if not rgb:
        return None
    return f"{rgb[0]},{rgb[1]},{rgb[2]}"


def color_label_to_carla_color(color_label):
    rgb = COLOR_RGB_MAP.get(color_label)
    if not rgb:
        return None
    return carla.Color(*rgb)

FPS = 30
UDP_IP = "0.0.0.0"
UDP_PORT = 60200 

# --- CARLA init ---
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()
try:
    carla_map = world.get_map()
except RuntimeError as exc:
    print(f"[CARLA][WARN] Unable to fetch map ({exc}); ground snapping disabled.")
    carla_map = None


def get_vehicle_blueprint_with_color(blueprint_library):
    preferred_ids = [
        'vehicle.vehicle.coloredxycar'
    ]

    for vid in preferred_ids:
        try:
            bp = blueprint_library.find(vid)
        except RuntimeError:
            bp = None
        if bp and bp.has_attribute('color'):
            return bp

    for bp in blueprint_library.filter('vehicle.*'):
        if bp.has_attribute('color'):
            return bp

    for vid in preferred_ids:
        try:
            bp = blueprint_library.find(vid)
            if bp:
                return bp
        except RuntimeError:
            continue

    return None


vehicle_bp = get_vehicle_blueprint_with_color(bp_lib)
if vehicle_bp is None:
    raise RuntimeError("No vehicle blueprint available")

VEHICLE_BP_ID = vehicle_bp.id
VEHICLE_BP_SUPPORTS_COLOR = vehicle_bp.has_attribute('color')

print(f"[CARLA] Using blueprint: {VEHICLE_BP_ID} (color attribute: {VEHICLE_BP_SUPPORTS_COLOR})")

vehicle_map = {}
vehicle_colors = {}
z_offset = 0.2
DISABLE_COLLISIONS = True
AUTO_GROUND_HEIGHT = True
try:
    LANE_TYPE_ANY = carla.LaneType.Any
except AttributeError:
    LANE_TYPE_ANY = carla.LaneType.Driving


def build_vehicle_blueprint(color_label):
    bp = bp_lib.find(VEHICLE_BP_ID)
    if bp is None:
        return None
    attr_value = color_label_to_attr_value(color_label)
    if attr_value and bp.has_attribute('color'):
        bp.set_attribute('color', attr_value)
    return bp


def apply_vehicle_color(vehicle, color_label):
    carla_color = color_label_to_carla_color(color_label)
    if not carla_color:
        return False
    try:
        vehicle.set_color(carla_color)
        return True
    except Exception:
        return False

def disable_vehicle_collisions(vehicle):
    """Disable collision response for a vehicle actor if supported."""
    if not DISABLE_COLLISIONS or vehicle is None:
        return
    try:
        vehicle.set_collision_enabled(False)
        return
    except AttributeError:
        # Older CARLA versions do not expose set_collision_enabled
        pass
    except RuntimeError:
        # Vehicle may be pending spawn
        return
    try:
        vehicle.set_simulate_physics(False)
    except Exception:
        pass

def get_ground_z(x, y, fallback_z):
    """Query CARLA map for ground height at (x, y)."""
    if not AUTO_GROUND_HEIGHT or carla_map is None:
        return fallback_z
    try:
        location = carla.Location(x=x, y=y, z=fallback_z)
        waypoint = carla_map.get_waypoint(location, project_to_road=True, lane_type=LANE_TYPE_ANY)
        if waypoint:
            return waypoint.transform.location.z
    except RuntimeError:
        # Query failed (e.g., far from road), fall back to provided z
        pass
    return fallback_z

def move_vehicle_to(vehicle, x, y, z,pitch ,yaw_deg):
    """Teleport vehicle to (x, y, z) with yaw/pitch (degrees)."""
    transform = carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(pitch=pitch, yaw=yaw_deg)
    )
    vehicle.set_transform(transform)

def move_or_spawn_vehicles(info_dict):
    """
    info_dict: { id: (x, y, z, pitch, yaw_deg, color), ... }
    - Destroy vehicles that disappeared
    - Move existing or spawn new ones and update their colors
    """
    global vehicle_map, vehicle_colors
    current_ids = set(info_dict.keys())
    existing_ids = set(vehicle_map.keys())

    # Destroy vehicles no longer present
    for vid in existing_ids - current_ids:
        try:
            vehicle_map[vid].destroy()
        except:
            pass
        vehicle_map.pop(vid, None)
        vehicle_colors.pop(vid, None)

    # Move existing or spawn new
    for vid, (x, y, z, pitch, yaw, color_label) in info_dict.items():
        normalized_color = normalize_color_label(color_label)
        ground_z = get_ground_z(x, y, z)
        target_z = ground_z + z_offset
        if vid in vehicle_map:
            vehicle = vehicle_map[vid]
            disable_vehicle_collisions(vehicle)
            move_vehicle_to(vehicle, x, y, target_z, pitch, yaw)
            if normalized_color and normalized_color != vehicle_colors.get(vid):
                if apply_vehicle_color(vehicle, normalized_color):
                    vehicle_colors[vid] = normalized_color
        else:
            transform = carla.Transform(
                carla.Location(x=x, y=y, z=target_z),
                carla.Rotation(pitch=pitch, yaw=yaw)
            )
            spawn_color = normalized_color or DEFAULT_COLOR_LABEL
            bp = build_vehicle_blueprint(spawn_color)
            if bp is None:
                continue
            vehicle = world.try_spawn_actor(bp, transform)
            if vehicle:
                vehicle_map[vid] = vehicle
                disable_vehicle_collisions(vehicle)
                if spawn_color and apply_vehicle_color(vehicle, spawn_color):
                    vehicle_colors[vid] = spawn_color
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
    vehicle_colors.clear()

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
                    cz = float(3.5)
                    pitch = float(it.get("pitch", 0.0))  # degree
                    yaw = float(it.get("yaw", 0.0))  # degree
                    color = normalize_color_label(it.get("color"))  # normalized color label

                    # NOTE: If your BEV Y-axis is inverted against CARLA's Y,
                    # you may apply cy = -cy or yaw = -yaw here.
                    vehicle_info[vid] = (cx, cy, cz, pitch, yaw, color)
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
