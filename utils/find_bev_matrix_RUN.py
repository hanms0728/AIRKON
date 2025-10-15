import subprocess

CAMERA_SETUPS = [
    {"name": "cam1",  "pos": {"x": -46,  "y": -74, "z": 9}, "rot": {"pitch": -40, "yaw": 90,  "roll": 0}},
    {"name": "cam2",  "pos": {"x": -35,  "y": -36, "z": 9}, "rot": {"pitch": -45, "yaw": 197, "roll": 0}},
    {"name": "cam3",  "pos": {"x": -35,  "y": -36, "z": 9}, "rot": {"pitch": -45, "yaw": 163, "roll": 0}},
    {"name": "cam4",  "pos": {"x": -35,  "y": 0,   "z": 9}, "rot": {"pitch": -45, "yaw": 190, "roll": 0}},
    {"name": "cam5",  "pos": {"x": -35,  "y": 5,   "z": 9}, "rot": {"pitch": -45, "yaw": 135, "roll": 0}},
    {"name": "cam6",  "pos": {"x": -77,  "y": 7,   "z": 9}, "rot": {"pitch": -40, "yaw": 73,  "roll": 0}},
    {"name": "cam7",  "pos": {"x": -77,  "y": 7,   "z": 9}, "rot": {"pitch": -40, "yaw": 107, "roll": 0}},
    {"name": "cam8",  "pos": {"x": -122, "y": 19,  "z": 9}, "rot": {"pitch": -40, "yaw": 0,   "roll": 0}},
    {"name": "cam9",  "pos": {"x": -95,  "y": -20, "z": 9}, "rot": {"pitch": -40, "yaw": 150, "roll": 0}},
    {"name": "cam10", "pos": {"x": -121, "y": -15, "z": 9}, "rot": {"pitch": -45, "yaw": -17, "roll": 0}},
    {"name": "cam11", "pos": {"x": -113, "y": -63, "z": 9}, "rot": {"pitch": -40, "yaw": 40,  "roll": 0}},
    {"name": "cam12", "pos": {"x": -60,  "y": -76, "z": 9}, "rot": {"pitch": -40, "yaw": 120, "roll": 0}},
    {"name": "cam13", "pos": {"x": -77,  "y": 34,  "z": 9}, "rot": {"pitch": -45, "yaw": -73, "roll": 0}},
    {"name": "cam14", "pos": {"x": -68,  "y": 34,  "z": 9}, "rot": {"pitch": -45, "yaw": -30, "roll": 0}},
    {"name": "cam15", "pos": {"x": -120, "y": -40, "z": 9}, "rot": {"pitch": -40, "yaw": 30,  "roll": 0}},
    {"name": "cam16", "pos": {"x": -61,  "y": -15, "z": 9}, "rot": {"pitch": -45, "yaw": 0,   "roll": 0}},
]

for row in CAMERA_SETUPS:
    x, y, z = row["pos"]["x"], row["pos"]["y"], row["pos"]["z"]
    pitch, yaw, roll = row["rot"]["pitch"], row["rot"]["yaw"], row["rot"]["roll"]
    print(f"[{row['name']}] running find_bev_matrix.py ...")
    subprocess.run([
        "python", "find_bev_matrix.py",
        f"--carla_camera_position={x},{y},{z}",
        f"--carla_camera_rotation={pitch},{yaw},{roll}",
        f"--frame_id={row['name']}",
        f"--output_dir=/AIRKON/utils/find_bev_matrix/",
    ], check=True)

print("Done.")