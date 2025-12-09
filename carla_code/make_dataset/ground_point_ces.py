#python ground_point.py
#!/usr/bin/env python




import glob
import os
import sys
import time

import math
import random

import numpy as np

from scipy.spatial.transform import Rotation

BASE_PATH = "./output"

os.makedirs(BASE_PATH, exist_ok=True)

# CARLA 모듈 경로 추가q
try:
    egg = glob.glob(
        '../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major, 
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'
        )
    )[0]
    sys.path.append(egg)
except IndexError:
    pass

import carla
import pygame
from pygame.locals import K_ESCAPE, K_q
import threading
import queue
import cv2

VIEW_WIDTH   = 1536
VIEW_HEIGHT  = 864
VIEW_FOV     = 95
BB_COLOR     = (248, 64, 24)
CENTER_COLOR = (64, 248, 24)
FRONT_COLOR  = (24, 128, 248)
TARGET_FPS   = int(10.0)
FIXED_DT     = float(1.0 / float(TARGET_FPS))  # 30 FPS simulation
MAX_DISTANCE = 100.0
MIN_BOX_AREA = 300
DISPLAY_SCALE = 6
PITCH = -45

CLASS_ID_CAR = 0
CLASS_ID_TRAFFIC_CONE = 1

TRAFFIC_CONE_BLUEPRINTS = [
    'static.prop.trafficcone01',
    'static.prop.constructioncone'
]
TRAFFIC_CONE_TYPE_IDS = {bp.casefold() for bp in TRAFFIC_CONE_BLUEPRINTS}

# Maximum allowed pixels outside image boundary per side
MAX_OUTSIDE_PIXELS = 1500

USE_SYNC_WORLD = True
USE_TM_SYNC = True
TM_PORT = 8000
SPAWN_VEHICLES = True
NUM_VEHICLES = 4

INCLUDE_VEHICLE_KEYWORDS = [
    # 이 목록에 포함된 키워드가 차량 블루프린트 ID(또는 카테고리)에
    # 매칭될 때만 스폰합니다. (대소문자 구분 없음)
    #'colored_xycar'
    'TeslaM3', 'Audi'
]

FRAME_COUNTER = 0

# World-space region of interest (x, y)
REGION_MIN = (-65.0, -60.0)
REGION_MAX = (65.0,10.0)

CAMERA_SETUPS = [
    {
        "name": "cam1_-45_4", #cam2와 대응
        "transform": carla.Transform(
            carla.Location(x=10, y=7, z=10),
            carla.Rotation(pitch=-45, yaw=-135, roll=0)
        ),
    },
    {
        "name": "cam2_-45_4", #cam1과 대응
        "transform": carla.Transform(
            carla.Location(x=-12, y=-14, z=10),
            carla.Rotation(pitch=-45, yaw=25, roll=0)
        ),
    },
    # {
    #     "name": "cam3",  #cam7과 대응
    #     "transform": carla.Transform(
    #         carla.Location(x=-60, y=0, z=10),
    #         carla.Rotation(pitch=-35, yaw=-45, roll=0)
    #     ),
    # },
    # {   
    #     "name": "cam4", #cam12와 대응
    #     "transform": carla.Transform(
    #         carla.Location(x=-60, y=-57, z=10),
    #         carla.Rotation(pitch=-35, yaw=45, roll=0)
    #     ),
    # },
    # {
    #     "name": "cam5", #alone
    #     "transform": carla.Transform(
    #         carla.Location(x=0, y=-37, z=10),
    #         carla.Rotation(pitch=-40, yaw=90, roll=0)
    #     ),
    # },
    # {
    #     "name": "cam6", #cam13과 대응
    #     "transform": carla.Transform(
    #         carla.Location(x=24, y=-56, z=10),
    #         carla.Rotation(pitch=-35, yaw=45, roll=0)
    #     ),
    # },
    # {
    #     "name": "cam7", #cam3과 대응
    #     "transform": carla.Transform(
    #         carla.Location(x=60, y=0, z=10),
    #         carla.Rotation(pitch=-35, yaw=-135, roll=0)
    #     ),
    # },
    # {
    #     "name": "cam8", #cam10과 대응
    #     "transform": carla.Transform(
    #         carla.Location(x=30, y=-10, z=10),
    #         carla.Rotation(pitch=-35, yaw=125, roll=0)
    #     ),
    # },
    # {
    #     "name": "cam9", #cam11과 대응
    #     "transform": carla.Transform(
    #         carla.Location(x=30, y=2, z=10),
    #         carla.Rotation(pitch=-35, yaw=-55, roll=0)
    #     ),
    # },
    # {
    #     "name": "cam10", #cam8과 대응
    #     "transform": carla.Transform(
    #         carla.Location(x=-30, y=-10, z=10),
    #         carla.Rotation(pitch=-35, yaw=55, roll=0)
    #     ),
    # },
    # {
    #     "name": "cam11", #cam9과 대응
    #     "transform": carla.Transform(
    #         carla.Location(x=-30, y=2, z=10),
    #         carla.Rotation(pitch=-35, yaw=-125, roll=0)
    #     ),
    # },
    # {
    #     "name": "cam12", #cam12와 대응
    #     "transform": carla.Transform(
    #         carla.Location(x=60, y=-57, z=10),
    #         carla.Rotation(pitch=-35, yaw=135, roll=0)
    #     ),
    # },
    # {
    #     "name": "cam13", #cam6과 대응
    #     "transform": carla.Transform(
    #         carla.Location(x=-24, y=-56, z=10),
    #         carla.Rotation(pitch=-35, yaw=135, roll=0)
    #     ),
    # },
    # {
    #     "name": "cam14", #cam15와 대응
    #     "transform": carla.Transform(
    #         carla.Location(x=-32, y=-22, z=10),
    #         carla.Rotation(pitch=-35, yaw=-45, roll=0)
    #     ),
    # },
    # {
    #     "name": "cam15", #cam14 와 대응
    #     "transform": carla.Transform(
    #         carla.Location(x=32, y=-22, z=10),
    #         carla.Rotation(pitch=-35, yaw=-125, roll=0)
    #     ),
    # },
    # {
    #     "name": "cam16", 
    #     "transform": carla.Transform(
    #         carla.Location(x=63, y=-30, z=10),
    #         carla.Rotation(pitch=-45, yaw=180, roll=0)
    #     ),
    # },
]   

def compute_H_img_to_ground(
    img_resolution,        # (width, height)
    fov_x_deg,             # FOV in degrees
    camera_pos_world,      # (X, Y, Z) in Carla world
    camera_rot_world       # (Pitch, Yaw, Roll) in Carla world
):
    """
    이미지 좌표 → 지면 좌표 (Y=0 평면)로의 호모그래피 행렬을 계산합니다.
    BEV 변환이나 이미지 입출력은 포함하지 않습니다.
    """
    width, height = img_resolution
    fov_x_rad = np.deg2rad(fov_x_deg)

    # 1️⃣ 내부 파라미터 K
    fx = (width / 2.0) / np.tan(fov_x_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    # 2️⃣ Carla → OpenCV 좌표계 변환
    axes_carla_to_cv = np.array([
        [0.0,  1.0,  0.0],   # Carla Y → CV X
        [0.0,  0.0, -1.0],   # Carla Z → CV Y (아래 방향이 +)
        [1.0,  0.0,  0.0],   # Carla X → CV Z
    ], dtype=np.float64)

    cam_pos_carla = np.array([-camera_pos_world[1], camera_pos_world[0], camera_pos_world[2]], dtype=np.float64)
    cam_pos_cv = axes_carla_to_cv @ cam_pos_carla

    yaw_carla   = camera_rot_world[1] + 90.0
    pitch_carla = -camera_rot_world[0]
    roll_carla  = camera_rot_world[2]

    rotation_ue = Rotation.from_euler('ZYX', [yaw_carla, pitch_carla, roll_carla], degrees=True)

    R_cam_to_world_carla = rotation_ue.as_matrix()
    R_world_to_cam_carla = R_cam_to_world_carla.T
    R_world_to_cam = axes_carla_to_cv @ R_world_to_cam_carla @ axes_carla_to_cv.T
    t_world_to_cam = -R_world_to_cam @ cam_pos_cv

    # 3️⃣ 이미지-지면 호모그래피 계산
    r1 = R_world_to_cam[:, 0]  # X축
    r3 = R_world_to_cam[:, 2]  # Z축
    H_ground_to_img = K @ np.column_stack((r1, r3, t_world_to_cam))

    H_img_to_ground = np.linalg.inv(H_ground_to_img)
    return H_img_to_ground

class ClientSideBoundingBoxes:
    @staticmethod
    def get_bounding_boxes(vehicles, camera, snapshot, max_distance=50.0):
        boxes = []
        vehicles_in_box = []
        # 카메라 월드 위치
        cam_loc = camera.get_transform().location

        for v in vehicles:
            actor_snap = snapshot.find(v.id)
            if actor_snap is None:
                continue

            # 차량 월드 상 중심점 (vehicle_transform.location + bbox center offset)
            veh_tf = actor_snap.get_transform()
            distance = cam_loc.distance(veh_tf.location)

            # 50미터 이내의 차량만 처리
            if distance > max_distance:
                continue

            bb = ClientSideBoundingBoxes.get_bounding_box(v, camera, veh_tf, snapshot)
            # 앞쪽(depth>0)인 경우만 처리
            if np.all(bb[:, 2] > 0):
                # 스크린 공간에서 박스의 투영된 면적 계산
                u = bb[:, 0]
                v_ = bb[:, 1]
                min_u, max_u = u.min(), u.max()
                min_v, max_v = v_.min(), v_.max()
                # skip boxes fully outside image frame
                if max_u < 0 or min_u > VIEW_WIDTH or max_v < 0 or min_v > VIEW_HEIGHT:
                    continue
                # skip boxes that extend outside the frame by more than threshold pixels
                out_left   = max(0,    -min_u)
                out_right  = max(0,    max_u - VIEW_WIDTH)
                out_top    = max(0,    -min_v)
                out_bottom = max(0,    max_v - VIEW_HEIGHT)
                if max(out_left, out_right, out_top, out_bottom) > MAX_OUTSIDE_PIXELS:
                    continue
                area = (max_u - min_u) * (max_v - min_v)
                # 면적이 임계값 이상일 때만 추가
                if area >= MIN_BOX_AREA:
                    boxes.append(bb)
                    vehicles_in_box.append(v)

        return boxes, vehicles_in_box

    @staticmethod
    def draw_ground_overlays(display, ground_polygons, ground_centers, front_pairs=None):
        surf = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        surf.set_colorkey((0, 0, 0))
        for poly in ground_polygons:
            if len(poly) >= 3:
                pygame.draw.polygon(surf, BB_COLOR, poly, 2)
        for c in ground_centers:
            pygame.draw.circle(surf, CENTER_COLOR, c, 4)
        if front_pairs:
            for left_pt, right_pt in front_pairs:
                pygame.draw.circle(surf, FRONT_COLOR, left_pt, 4)
                pygame.draw.circle(surf, FRONT_COLOR, right_pt, 4)
                pygame.draw.line(surf, FRONT_COLOR, left_pt, right_pt, 2)
        display.blit(surf, (0, 0))

    @staticmethod
    def get_bounding_box(vehicle, camera, vehicle_transform, snapshot=None):
        # 1) Define 8 local corners of the bounding box in homogeneous coords
        ext = vehicle.bounding_box.extent
        cords = np.array([
            [ ext.x,  ext.y, -ext.z, 1],
            [-ext.x,  ext.y, -ext.z, 1],
            [-ext.x, -ext.y, -ext.z, 1],
            [ ext.x, -ext.y, -ext.z, 1],
            [ ext.x,  ext.y,  ext.z, 1],
            [-ext.x,  ext.y,  ext.z, 1],
            [-ext.x, -ext.y,  ext.z, 1],
            [ ext.x, -ext.y,  ext.z, 1],
        ]).T  # 4x8

        # 2) World-space transform of the box
        offset_tf = carla.Transform(vehicle.bounding_box.location, carla.Rotation())
        world_from_box = ClientSideBoundingBoxes._get_matrix(vehicle_transform) @ ClientSideBoundingBoxes._get_matrix(offset_tf)

        # 3) Compute projection matrix inline: P_full = K · C · (world→cam)
        K = camera.calibration  # 3×3
        # Use camera transform from the same snapshot/frame if available
        try:
            cam_snap = snapshot.find(camera.id)
            cam_tf = cam_snap.get_transform() if cam_snap is not None else camera.get_transform()
        except Exception:
            cam_tf = camera.get_transform()
        cam_from_world = np.linalg.inv(ClientSideBoundingBoxes._get_matrix(cam_tf))
        # CARLA→KITTI axis mapping (4×4)
        C4 = np.array([
            [0,  1,  0, 0],
            [0,  0, -1, 0],
            [1,  0,  0, 0],
            [0,  0,  0, 1],
        ], dtype=float)
        # Full projection: 3×4
        P4 = C4 @ cam_from_world      # 4×4
        P_full = K @ P4[0:3, :]       # 3×4

        # 4) Apply world transform then project
        pts_world = world_from_box @ cords            # 4x8
        pts_cam_img = P_full @ pts_world               # 3x8

        # 5) Normalize to pixel coordinates and depth
        u = pts_cam_img[0, :] / pts_cam_img[2, :]
        v = pts_cam_img[1, :] / pts_cam_img[2, :]
        depth = pts_cam_img[2, :]

        # 6) Return Nx3 array of [u, v, depth]
        return np.vstack([u, v, depth]).T  # 8x3

    @staticmethod
    def _get_matrix(transform):
        m = np.identity(4, dtype=float)
        loc, rot = transform.location, transform.rotation
        # translation
        m[0,3], m[1,3], m[2,3] = loc.x, loc.y, loc.z
        # rotation
        cy, sy = np.cos(np.radians(rot.yaw)),   np.sin(np.radians(rot.yaw))
        cr, sr = np.cos(np.radians(rot.roll)),  np.sin(np.radians(rot.roll))
        cp, sp = np.cos(np.radians(rot.pitch)), np.sin(np.radians(rot.pitch))
        m[0,0] =  cp*cy
        m[0,1] =  cy*sp*sr - sy*cr
        m[0,2] = -cy*sp*cr - sy*sr
        m[1,0] =  sy*cp
        m[1,1] =  sy*sp*sr + cy*cr
        m[1,2] = -sy*sp*cr + cy*sr
        m[2,0] =  sp
        m[2,1] = -cp*sr
        m[2,2] =  cp*cr
        return m


class CameraRig:
    def __init__(self, name, camera_actor, calibration, output_dirs):
        self.name = name
        self.camera = camera_actor
        self.K = calibration
        self.rgb_queue = queue.Queue()
        self.camera.listen(lambda image, q=self.rgb_queue: q.put(image))
        self.image_dir = output_dirs["images"]
        self.label_dir = output_dirs["labels"]
        self.overlay_dir = output_dirs["overlays"]
        self.calib_dir = output_dirs["calib"]
        self.display_position = (0, 0)

class FixedCameraClient:
    def __init__(self):
        if len(CAMERA_SETUPS) == 0:
            raise ValueError("CAMERA_SETUPS must contain at least one camera definition")
        if len(CAMERA_SETUPS) > 16:
            raise ValueError("CAMERA_SETUPS supports up to 16 cameras")

        pygame.init()
        self.clock = pygame.time.Clock()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self._orig_settings = self.world.get_settings()

        if USE_SYNC_WORLD:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = FIXED_DT
            self.world.apply_settings(settings)

        self.tm = None
        if USE_TM_SYNC:
            self.tm = self.client.get_trafficmanager(TM_PORT)
            self.tm.set_synchronous_mode(True)

        if USE_SYNC_WORLD:
            self.world.tick()
        else:
            try:
                self.world.wait_for_tick(1.0)
            except RuntimeError:
                pass

        self.blueprint_library = self.world.get_blueprint_library()

        self.spawned_vehicles = []
        if SPAWN_VEHICLES:
            self.spawned_vehicles = self._spawn_traffic(NUM_VEHICLES)

        self.camera_rigs = []
        for idx, cfg in enumerate(CAMERA_SETUPS):
            self.camera_rigs.append(self._spawn_camera_rig(cfg, idx))

        self._configure_display()

        self.vehicles = self.world.get_actors().filter('vehicle.*')
        self.annotated_actors = list(self.vehicles)

        self.save_queue = queue.Queue()
        threading.Thread(target=self._saving_worker, daemon=True).start()

        self.region_dir = os.path.join(BASE_PATH, "regions")
        os.makedirs(self.region_dir, exist_ok=True)

    def _spawn_traffic(self, num):
        spawn_points = list(self.world.get_map().get_spawn_points())
        random.shuffle(spawn_points)
        if not spawn_points:
            print("[Spawn] No spawn points available.")
            return []
        num = min(num, len(spawn_points))

        allow_set = {kw.casefold() for kw in INCLUDE_VEHICLE_KEYWORDS}
        vehicle_blueprints = []
        for bp in self.blueprint_library.filter('vehicle.*'):
            if bp.has_attribute('number_of_wheels'):
                try:
                    if int(bp.get_attribute('number_of_wheels').as_int()) != 4:
                        continue
                except Exception:
                    continue
            bp_id_lower = bp.id.casefold()

            # 화이트리스트: ID에 allow_set 키워드가 포함되거나,
            # category 속성이 allow_set에 포함될 때만 허용
            id_match = any(kw in bp_id_lower for kw in allow_set) if allow_set else False
            cat_match = False
            if bp.has_attribute('category'):
                try:
                    cat = bp.get_attribute('category').as_string().casefold()
                    cat_match = cat in allow_set
                except Exception:
                    pass

            if not (id_match or cat_match):
                continue

            # 특수 초소형 차량은 기본 제외 (원 코드 유지)
            if bp.id.endswith('isetta') or bp.id.endswith('microlino'):
                continue

            vehicle_blueprints.append(bp)

        if not vehicle_blueprints:
            print("[Spawn] No eligible vehicle blueprints found.")
            return []

        tm_port = self.tm.get_port() if self.tm else 0
        commands = []
        for idx in range(num):
            bp = random.choice(vehicle_blueprints)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            if bp.has_attribute('driver_id'):
                driver_id = random.choice(bp.get_attribute('driver_id').recommended_values)
                bp.set_attribute('driver_id', driver_id)
            bp.set_attribute('role_name', 'autopilot')
            transform = spawn_points[idx]
            commands.append(
                carla.command.SpawnActor(bp, transform).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)
                )
            )

        try:
            if USE_SYNC_WORLD:
                responses = self.client.apply_batch_sync(commands, True)
            else:
                responses = self.client.apply_batch(commands, True)
        except Exception as exc:
            print(f"[Spawn] Vehicle spawn batch failed: {exc}")
            return []

        spawned = []
        for response in responses:
            if response.error:
                print("[Spawn] Error:", response.error)
                continue
            try:
                actor = self.world.get_actor(response.actor_id)
                if actor is not None:
                    spawned.append(actor)
            except Exception:
                pass

        print(f"[Spawn] Spawned {len(spawned)}/{num} vehicles (TM port={tm_port}).")
        return spawned

    def _spawn_camera_rig(self, config, index):
        name = config.get("name", f"cam{index}")
        transform = config.get("transform")
        if not isinstance(transform, carla.Transform):
            raise ValueError(f"Camera '{name}' requires a valid carla.Transform")

        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        camera_actor = self.world.spawn_actor(camera_bp, transform)
        if USE_SYNC_WORLD:
            self.world.tick()
        else:
            try:
                self.world.wait_for_tick(1.0)
            except RuntimeError:
                pass

        calibration = self._build_calibration()
        camera_actor.calibration = calibration
        # Build per-camera output directories: BASE_PATH/<camera_name>/{images,labels,overlays,calib}
        cam_base = os.path.join(BASE_PATH, name)
        output_dirs = {
            "images": os.path.join(cam_base, "images"),
            "labels": os.path.join(cam_base, "labels"),
            "overlays": os.path.join(cam_base, "overlays"),
            "calib": os.path.join(cam_base, "calib"),
        }
        for path in output_dirs.values():
            os.makedirs(path, exist_ok=True)
        return CameraRig(name, camera_actor, calibration, output_dirs)

    def _configure_display(self):
        count = len(self.camera_rigs)
        cols = max(1, min(count, 4))
        rows = math.ceil(count / cols)
        self.scaled_width = max(1, VIEW_WIDTH // DISPLAY_SCALE)
        self.scaled_height = max(1, VIEW_HEIGHT // DISPLAY_SCALE)
        window_w = cols * self.scaled_width
        window_h = rows * self.scaled_height
        self.display = pygame.display.set_mode((window_w, window_h))
        pygame.display.set_caption("CARLA Fixed Camera Client")
        for idx, rig in enumerate(self.camera_rigs):
            col = idx % cols
            row = idx // cols
            rig.display_position = (col * self.scaled_width, row * self.scaled_height)

    @staticmethod
    def _build_calibration():
        hfov = np.deg2rad(VIEW_FOV)
        vfov = 2.0 * np.arctan(np.tan(hfov / 2.0) * (VIEW_HEIGHT / VIEW_WIDTH))
        fx = VIEW_WIDTH / (2.0 * np.tan(hfov / 2.0))
        fy = VIEW_HEIGHT / (2.0 * np.tan(vfov / 2.0))
        cx = VIEW_WIDTH / 2.0
        cy = VIEW_HEIGHT / 2.0
        return np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=float)

    def game_loop(self):
        global FRAME_COUNTER
        try:
            while True:
                if USE_SYNC_WORLD:
                    try:
                        self.world.tick()
                    except RuntimeError as exc:
                        print(f"Warning: world tick failed ({exc}); retrying...")
                        time.sleep(0.5)
                        continue
                    snapshot = self.world.get_snapshot()
                else:
                    try:
                        snapshot = self.world.wait_for_tick(1.0)
                    except RuntimeError as exc:
                        print(f"Warning: world tick failed ({exc}); retrying...")
                        time.sleep(0.5)
                        continue
                if snapshot is None:
                    print("Warning: world snapshot unavailable; retrying...")
                    continue
                actor_list = self.world.get_actors()
                self.vehicles = actor_list.filter('vehicle.*')
                self.annotated_actors = self._collect_annotation_actors(actor_list, self.vehicles)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN and event.key in (K_ESCAPE, K_q):
                        return

                self.display.fill((0, 0, 0))

                current_frame = FRAME_COUNTER
                region_entries = self._collect_region_entries(snapshot)

                for rig in self.camera_rigs:
                    image = self._retrieve_sensor_frame(rig.rgb_queue, snapshot.frame, f"RGB[{rig.name}]")
                    if image is None:
                        continue

                    array = np.frombuffer(image.raw_data, dtype=np.uint8)
                    array = array.reshape((image.height, image.width, 4))[:, :, :3]
                    frame = np.ascontiguousarray(array[:, :, ::-1])

                    bbs, filtered_actors = ClientSideBoundingBoxes.get_bounding_boxes(
                        self.annotated_actors, rig.camera, snapshot, MAX_DISTANCE
                    )

                    cam_snap = snapshot.find(rig.camera.id)
                    cam_tf = cam_snap.get_transform() if cam_snap is not None else rig.camera.get_transform()
                    cam_location = cam_tf.location
                    cam_rotation = cam_tf.rotation
                    homography_img_to_ground = compute_H_img_to_ground(
                        (VIEW_WIDTH, VIEW_HEIGHT),
                        VIEW_FOV,
                        (cam_location.x, cam_location.y, cam_location.z),
                        (cam_rotation.pitch, cam_rotation.yaw, cam_rotation.roll)
                    )
                    cam_from_world = np.linalg.inv(ClientSideBoundingBoxes._get_matrix(cam_tf))
                    C4 = np.array([
                        [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]
                    ], dtype=float)
                    camk_from_world = C4 @ cam_from_world

                    annotations = []
                    ground_polygons = []
                    ground_centers = []
                    front_markers = []

                    for bb, actor in zip(bbs, filtered_actors):
                        class_id = self._determine_class_id(actor)
                        if class_id is None:
                            continue
                        actor_snap = snapshot.find(actor.id)
                        if actor_snap is None:
                            continue
                        veh_tf = actor_snap.get_transform()
                        offset_tf = carla.Transform(actor.bounding_box.location, carla.Rotation())
                        world_from_box = ClientSideBoundingBoxes._get_matrix(veh_tf) @ ClientSideBoundingBoxes._get_matrix(offset_tf)
                        camk_from_box = camk_from_world @ world_from_box

                        ext = actor.bounding_box.extent
                        bottom_center_cam = camk_from_box @ np.array([0.0, 0.0, -ext.z, 1.0], dtype=float)
                        bottom_center_uv = self._project_cam_point(rig.K, bottom_center_cam)
                        if bottom_center_uv is None:
                            continue
                        if not self._is_point_inside_frame(bottom_center_uv):
                            continue

                        bottom_indices = [0, 1, 2, 3]
                        if np.any(bb[bottom_indices, 2] <= 0):
                            continue
                        ground_uv = [(float(bb[idx, 0]), float(bb[idx, 1])) for idx in bottom_indices]

                        front_left_uv, front_right_uv = self._select_front_points(bb, class_id)
                        if not (self._is_point_inside_frame(front_left_uv) or self._is_point_inside_frame(front_right_uv)):
                            continue

                        annotations.append(
                            f"{class_id} {bottom_center_uv[0]:.2f} {bottom_center_uv[1]:.2f} "
                            f"{front_left_uv[0]:.2f} {front_left_uv[1]:.2f} "
                            f"{front_right_uv[0]:.2f} {front_right_uv[1]:.2f}"
                        )
                        ground_polygons.append([(int(round(u)), int(round(v))) for (u, v) in ground_uv])
                        ground_centers.append((int(round(bottom_center_uv[0])), int(round(bottom_center_uv[1]))))
                        front_markers.append(
                            (
                                (int(round(front_left_uv[0])), int(round(front_left_uv[1]))),
                                (int(round(front_right_uv[0])), int(round(front_right_uv[1]))),
                            )
                        )

                    bbox_surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    ClientSideBoundingBoxes.draw_ground_overlays(bbox_surf, ground_polygons, ground_centers, front_markers)

                    scaled_surface = pygame.transform.smoothscale(
                        bbox_surf, (self.scaled_width, self.scaled_height)
                    )
                    self.display.blit(scaled_surface, rig.display_position)

                    frame_copy = frame.copy()
                    overlay_copy = bbox_surf.copy()

                    # 카메라별 pitch 값을 파일명 태그로 사용
                    pitch_tag_str = f"{cam_rotation.pitch:g}"

                    self.save_queue.put((self._save_frame, (rig, current_frame, frame_copy, pitch_tag_str)))
                    self.save_queue.put((self._save_overlay, (rig, current_frame, overlay_copy, pitch_tag_str)))
                    self.save_queue.put((self._save_annotations, (rig, current_frame, list(annotations), pitch_tag_str)))
                    self.save_queue.put((self._save_calibration, (rig, current_frame, homography_img_to_ground.copy(), pitch_tag_str)))

                self.save_queue.put((self._save_region_data, (self.region_dir, current_frame, list(region_entries))))

                pygame.display.flip()
                self.clock.tick(TARGET_FPS)

                FRAME_COUNTER += 1
                if FRAME_COUNTER == 500:
                    break

        finally:
            self.save_queue.join()
            self.save_queue.put(None)

            for rig in self.camera_rigs:
                try:
                    rig.camera.stop()
                except Exception:
                    pass
                try:
                    rig.camera.destroy()
                except Exception:
                    pass

            for actor in getattr(self, "spawned_vehicles", []):
                try:
                    actor.destroy()
                except Exception:
                    pass

            try:
                if self.tm and USE_TM_SYNC:
                    self.tm.set_synchronous_mode(False)
            except Exception:
                pass

            try:
                if USE_SYNC_WORLD:
                    self.world.apply_settings(self._orig_settings)
            except Exception:
                pass

            pygame.quit()

    @staticmethod
    def _retrieve_sensor_frame(sensor_queue, frame_number, label):
        try:
            while True:
                data = sensor_queue.get(timeout=1.0)
                if data.frame == frame_number:
                    return data
        except queue.Empty:
            print(f"Warning: No {label} for frame {frame_number}")
        return None

    @staticmethod
    def _project_cam_point(K, cam_point):
        x, y, z = cam_point[:3]
        if z <= 0:
            return None
        u = (K[0, 0] * (x / z)) + K[0, 2]
        v = (K[1, 1] * (y / z)) + K[1, 2]
        return (float(u), float(v))
    
    @staticmethod
    def _is_point_inside_frame(point_uv):
        u, v = point_uv
        return (0.0 <= u < VIEW_WIDTH) and (0.0 <= v < VIEW_HEIGHT)

    def _collect_annotation_actors(self, actor_list, vehicle_list):
        tracked = {}
        for actor in list(vehicle_list):
            tracked[actor.id] = actor
        for bp_filter in TRAFFIC_CONE_BLUEPRINTS:
            try:
                props = actor_list.filter(bp_filter)
            except RuntimeError:
                props = []
            for actor in props:
                tracked[actor.id] = actor
        return list(tracked.values())

    @staticmethod
    def _determine_class_id(actor):
        type_id = actor.type_id.casefold()
        if type_id.startswith('vehicle.'):
            return CLASS_ID_CAR
        if type_id in TRAFFIC_CONE_TYPE_IDS:
            return CLASS_ID_TRAFFIC_CONE
        return None

    def _select_front_points(self, bb, class_id):
        if class_id == CLASS_ID_TRAFFIC_CONE:
            bottom_indices = [0, 1, 2, 3]
            closest = sorted(bottom_indices, key=lambda idx: bb[idx, 2])[:2]
            closest.sort(key=lambda idx: bb[idx, 0])
            left_idx, right_idx = closest
        else:
            left_idx = 3
            right_idx = 0
        front_left_uv = (float(bb[left_idx, 0]), float(bb[left_idx, 1]))
        front_right_uv = (float(bb[right_idx, 0]), float(bb[right_idx, 1]))
        return front_left_uv, front_right_uv

    def _collect_region_entries(self, snapshot):
        min_x, min_y = REGION_MIN
        max_x, max_y = REGION_MAX
        entries = []
        for vehicle in self.vehicles:
            actor_snap = snapshot.find(vehicle.id)
            if actor_snap is None:
                continue
            vehicle_tf = actor_snap.get_transform()
            bb_center = vehicle_tf.transform(vehicle.bounding_box.location)
            if not (min_x <= bb_center.x <= max_x and min_y <= bb_center.y <= max_y):
                continue
            entries.append(self._format_region_entry(vehicle, vehicle_tf, bb_center))
        return entries

    @staticmethod
    def _format_region_entry(vehicle, vehicle_tf, bb_center):
        bbox = vehicle.bounding_box.extent
        height = 2.0 * bbox.z
        width = 2.0 * bbox.y
        length = 2.0 * bbox.x
        rotation = vehicle_tf.rotation
        return (
            f"{vehicle.id},{vehicle.type_id},"
            f"{bb_center.x:.2f},{bb_center.y:.2f},{bb_center.z:.2f},"
            f"{height:.2f},{width:.2f},{length:.2f},{rotation.yaw:.2f}"
        )

    def _saving_worker(self):
        while True:
            task = self.save_queue.get()
            if task is None:
                break
            func, args = task
            func(*args)
            self.save_queue.task_done()

    @staticmethod
    def _save_frame(rig, frame_id, frame, pitch_tag):
        image_filename = os.path.join(rig.image_dir, f"{rig.name}_frame_{frame_id:06d}_{pitch_tag}.jpg")
        cv2.imwrite(image_filename, frame[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    @staticmethod
    def _save_overlay(rig, frame_id, overlay_surf, pitch_tag):
        overlay_array = pygame.surfarray.array3d(overlay_surf).swapaxes(0, 1)
        overlay_path = os.path.join(rig.overlay_dir, f"{rig.name}_frame_{frame_id:06d}_{pitch_tag}.jpg")
        cv2.imwrite(overlay_path, overlay_array[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    @staticmethod
    def _save_annotations(rig, frame_id, annotations, pitch_tag):
        label_filename = os.path.join(rig.label_dir, f"{rig.name}_frame_{frame_id:06d}_{pitch_tag}.txt")
        with open(label_filename, 'w') as f:
            f.write("\n".join(annotations))

    @staticmethod
    def _save_calibration(rig, frame_id, homography, pitch_tag):
        calib_path = os.path.join(rig.calib_dir, f"{rig.name}_frame_{frame_id:06d}_{pitch_tag}.txt")
        matrix = np.asarray(homography, dtype=np.float64)
        np.savetxt(calib_path, matrix, fmt="%.8f", delimiter=" ", comments="")

    @staticmethod
    def _save_region_data(region_dir, frame_id, entries):
        region_path = os.path.join(region_dir, f"region_frame_{frame_id:06d}.txt")
        with open(region_path, 'w') as f:
            f.write("\n".join(entries))

def main():
    client = FixedCameraClient()
    client.game_loop()
    print("EXIT")

if __name__=='__main__':
    main()
