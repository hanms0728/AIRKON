#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA (0.9.x) — Multi-Camera Pixel→World LUT + Global Fused PLYs

What you get
- Per-camera capture (RGB/Depth/Semantic), 3D reconstruction (camera→world), ground filtering
- Per-camera outputs (rgb/semantic/depth heatmaps & npz) saved under out_root/cam_<name>/
- Global fused PLYs:
    1) global_fused_full.ply       : all fused ground points
    2) global_lane_only.ply        : semantic == lane_id (default 24)
    3) global_road_lane.ply        : semantic in {road_ids, lane_id} (default road_ids="1")

Notes
- World coord = CARLA/UE4 (left-handed): X forward, Y right, Z up.
- Depth is Euclidean distance to first hit. Reconstruction uses `forward` by default (stable in your tests).

Usage (example)
python multi_cam_pixel2world_and_fuse_inline.py \
  --host 127.0.0.1 --port 2000 \
  --width 1920 --height 1080 --fov 89 \
  --ray-model forward --floor-ids 1,2,10 \
  --lane-id 24 --road-ids 1 \
  --cloud-ground-only 1 --voxel 0.05 \
  --out-root ./multi_out
"""

import os, json, time, math, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import open3d as o3d
import carla
from contextlib import contextmanager

# ------------------------------------------------------------
# Inline camera list (can be overridden by --cam-config JSON)
# ------------------------------------------------------------
CAM_LIST_DEFAULT = [
    {"name":"cam1",  "x":10.0,  "y":7.0,   "z":10.0,  "pitch":-30.0, "yaw":-135.0, "roll":0.0, "fov":89.0},
    {"name":"cam2",  "x":-12.0, "y":-14.0, "z":10.0,  "pitch":-30.0, "yaw":25.0,   "roll":0.0, "fov":89.0},
    {"name":"cam3",  "x":-60.0, "y":0.0,   "z":10.0,  "pitch":-35.0, "yaw":-45.0,  "roll":0.0, "fov":89.0},
    {"name":"cam4",  "x":-60.0, "y":-57.0, "z":10.0,  "pitch":-35.0, "yaw":45.0,   "roll":0.0, "fov":89.0},
    {"name":"cam5",  "x":0.0,   "y":-37.0, "z":10.0,  "pitch":-40.0, "yaw":90.0,   "roll":0.0, "fov":89.0},
    {"name":"cam6",  "x":24.0,  "y":-56.0, "z":10.0,  "pitch":-35.0, "yaw":45.0,   "roll":0.0, "fov":89.0},
    {"name":"cam7",  "x":60.0,  "y":0.0,   "z":10.0,  "pitch":-35.0, "yaw":-135.0, "roll":0.0, "fov":89.0},
    {"name":"cam8",  "x":30.0,  "y":-10.0, "z":10.0,  "pitch":-35.0, "yaw":125.0,  "roll":0.0, "fov":89.0},
    {"name":"cam9",  "x":30.0,  "y":2.0,   "z":10.0,  "pitch":-35.0, "yaw":-55.0,  "roll":0.0, "fov":89.0},
    {"name":"cam10", "x":-30.0, "y":-10.0, "z":10.0,  "pitch":-35.0, "yaw":55.0,   "roll":0.0, "fov":89.0},
    {"name":"cam11", "x":-30.0, "y":2.0,   "z":10.0,  "pitch":-35.0, "yaw":-125.0, "roll":0.0, "fov":89.0},
    {"name":"cam12", "x":60.0,  "y":-57.0, "z":10.0,  "pitch":-35.0, "yaw":135.0,  "roll":0.0, "fov":89.0},
    {"name":"cam13", "x":-24.0, "y":-56.0, "z":10.0,  "pitch":-35.0, "yaw":135.0,  "roll":0.0, "fov":89.0},
    {"name":"cam14", "x":-32.0, "y":-22.0, "z":10.0,  "pitch":-35.0, "yaw":-45.0,  "roll":0.0, "fov":89.0},
    {"name":"cam15", "x":32.0,  "y":-22.0, "z":10.0,  "pitch":-35.0, "yaw":-125.0, "roll":0.0, "fov":89.0},
    {"name":"cam16", "x":65.0,  "y":-30.0, "z":10.0,  "pitch":-40.0, "yaw":180.0, "roll":0.0, "fov":89.0},
    {"name":"top", "x":0.0,  "y":-25.0, "z":100.0,  "pitch":-90.0, "yaw":0.0, "roll":90.0, "fov":89.0}
]

# ---------------- Utilities ----------------

def depth_raw_to_meters(image_bgra: np.ndarray) -> np.ndarray:
    B = image_bgra[:, :, 0].astype(np.uint32)
    G = image_bgra[:, :, 1].astype(np.uint32)
    R = image_bgra[:, :, 2].astype(np.uint32)
    denom = (256 ** 3 - 1)
    depth_norm = (R + (G << 8) + (B << 16)).astype(np.float64) / denom
    return (1000.0 * depth_norm).astype(np.float32)


def build_intrinsics(width: int, height: int, fov_deg: float):
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[f, 0, cx],[0, f, cy],[0, 0, 1]], dtype=np.float64)
    return K, f, cx, cy


def colorize_semantic(sem: np.ndarray) -> np.ndarray:
    H, W = sem.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[:, :, :] = (80, 80, 80)
    def paint(ids, color):
        mask = np.isin(sem, ids); out[mask] = color
    # 기본 CARLA 팔레트 (필요시 변경)
    paint([7], (50,150,50))      # Road
    paint([10], (255,255,0))     # LaneMarking
    paint([8], (244,35,232))     # Sidewalk
    paint([12], (70,70,70))      # Building
    paint([13], (190,153,153))   # Fence
    paint([14], (153,153,153))   # Pole
    paint([18], (0,0,142))       # Vehicle
    paint([19], (220,20,60))     # Pedestrian
    paint([5,6,9,11,15,16,17,20,21,22], (100,100,100))
    return out


def overlay_mask_on_rgb(rgb: np.ndarray, mask: np.ndarray, alpha=0.5, color=(0,255,0)) -> np.ndarray:
    ov = rgb.copy().astype(np.float32)
    m = (mask > 0)[..., None].astype(np.float32)
    color_arr = np.array(color, dtype=np.float32).reshape(1,1,3)
    ov = ov * (1 - alpha*m) + color_arr * (alpha*m)
    return np.clip(ov, 0, 255).astype(np.uint8)


def overlay_gray_on_rgb(rgb: np.ndarray, gray_u8: np.ndarray, alpha=0.5) -> np.ndarray:
    gray3 = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    return np.clip(rgb.astype(np.float32)*(1-alpha) + gray3.astype(np.float32)*alpha, 0, 255).astype(np.uint8)


def pick_sem_channel(arr_bgra: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'r':
        sem = arr_bgra[:, :, 2]
    elif mode == 'g':
        sem = arr_bgra[:, :, 1]
    elif mode == 'b':
        sem = arr_bgra[:, :, 0]
    else:
        cands = [arr_bgra[:,:,2], arr_bgra[:,:,1], arr_bgra[:,:,0]]
        uniq = [len(np.unique(c)) for c in cands]
        sem = cands[int(np.argmax(uniq))]
    return sem.astype(np.uint8)


@contextmanager
def synchronous_mode(world, fps=20):
    original = world.get_settings()
    settings = world.get_settings()
    try:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(fps)
        world.apply_settings(settings)
        yield
    finally:
        world.apply_settings(original)


# -------------- Per-camera capture & reconstruction --------------

def process_one_camera(world, bp_lib, args, cam_cfg, out_dir_cam: Path):
    out_dir_cam.mkdir(parents=True, exist_ok=True)
    W, H = args.width, args.height
    FOV = float(cam_cfg.get("fov", args.fov))

    depth_bp = bp_lib.find('sensor.camera.depth')
    rgb_bp   = bp_lib.find('sensor.camera.rgb')
    sem_bp   = bp_lib.find('sensor.camera.semantic_segmentation')
    for bp in (depth_bp, rgb_bp, sem_bp):
        bp.set_attribute('image_size_x', str(W))
        bp.set_attribute('image_size_y', str(H))
        bp.set_attribute('fov',          str(FOV))

    cam_tf = carla.Transform(
        carla.Location(cam_cfg["x"], cam_cfg["y"], cam_cfg["z"]),
        carla.Rotation(pitch=cam_cfg["pitch"], yaw=cam_cfg["yaw"], roll=cam_cfg.get("roll",0.0))
    )

    depth_buf = {"arr":None}; rgb_buf={"arr":None}; sem_buf={"arr":None}
    def on_depth(img: carla.Image):
        depth_buf["arr"] = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(H, W, 4)
    def on_rgb(img: carla.Image):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(H, W, 4)
        rgb_buf["arr"] = arr[:,:,:3][:,:,::-1]
    def on_sem(img: carla.Image):
        sem_buf["arr"] = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(H, W, 4)

    sensor_depth = sensor_rgb = sensor_sem = None
    K, f, cx, cy = build_intrinsics(W, H, FOV)

    try:
        with synchronous_mode(world, fps=args.fps):
            sensor_depth = world.spawn_actor(depth_bp, cam_tf)
            sensor_rgb   = world.spawn_actor(rgb_bp,   cam_tf)
            sensor_sem   = world.spawn_actor(sem_bp,   cam_tf)
            sensor_depth.listen(on_depth); sensor_rgb.listen(on_rgb); sensor_sem.listen(on_sem)

            for _ in range(4):
                world.tick()

            deadline = time.time() + 6.0
            while (depth_buf["arr"] is None or rgb_buf["arr"] is None or sem_buf["arr"] is None) and time.time() < deadline:
                world.tick()
            if depth_buf["arr"] is None or rgb_buf["arr"] is None or sem_buf["arr"] is None:
                raise RuntimeError("Timeout on sensors")

            # Save RGB/Semantic for reference
            Image.fromarray(rgb_buf["arr"]).save(out_dir_cam/"rgb_image.png")
            sem_id = pick_sem_channel(sem_buf["arr"], args.sem_channel)
            Image.fromarray(sem_id).save(out_dir_cam/"semantic_raw.png")
            Image.fromarray(colorize_semantic(sem_id)).save(out_dir_cam/"semantic_color.png")
            vals, counts = np.unique(sem_id, return_counts=True)
            order = np.argsort(counts)[::-1]
            with open(out_dir_cam/"semantic_histogram.txt","w") as hf:
                hf.write("# class_id,count\n")
                for vi,ci in zip(vals[order],counts[order]):
                    hf.write(f"{int(vi)},{int(ci)}\n")

            # Depth & vis
            depth_m = depth_raw_to_meters(depth_buf["arr"])  # HxW float32
            depth_gray = np.clip((depth_m - args.min_depth)/(args.max_depth-args.min_depth+1e-9), 0,1)
            depth_gray_u8 = (depth_gray*255).astype(np.uint8)
            Image.fromarray(depth_gray_u8).save(out_dir_cam/"depth_gray.png")

            # Reconstruct 3D in camera frame
            u = np.arange(W, dtype=np.float64); v = np.arange(H, dtype=np.float64)
            uu, vv = np.meshgrid(u, v)
            D = depth_m.astype(np.float64)
            if args.ray_model == "normalized":
                dir_x = np.ones_like(uu); dir_y = (uu - cx)/f; dir_z = -(vv - cy)/f
                norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                ux,uy,uz = dir_x/norm, dir_y/norm, dir_z/norm
                Xc,Yc,Zc = D*ux, D*uy, D*uz
            else:
                Xc = D; Yc = ((uu - cx)/f)*D; Zc = (-(vv - cy)/f)*D

            valid = (D > args.min_depth) & (D < args.max_depth) & np.isfinite(D)

            # Cam→World
            # sensor_depth 기준 transform 사용
            M_c2w = np.array(sensor_depth.get_transform().get_matrix(), dtype=np.float64)
            R = M_c2w[:3,:3]; t = M_c2w[:3,3]
            pts_c = np.stack([Xc,Yc,Zc], axis=0).reshape(3,-1)
            pts_w = (R @ pts_c) + t.reshape(3,1)
            Xw = pts_w[0].reshape(H,W).astype(np.float32)
            Yw = pts_w[1].reshape(H,W).astype(np.float32)
            Zw = pts_w[2].reshape(H,W).astype(np.float32)

            # Floor / ground masks
            floor_ids = set(int(x) for x in args.floor_ids.split(",")) if args.floor_ids else {7,10}
            sem_floor = np.isin(sem_id, list(floor_ids))
            Image.fromarray((sem_floor*255).astype(np.uint8)).save(out_dir_cam/"floor_mask.png")
            Image.fromarray(overlay_mask_on_rgb(rgb_buf["arr"], sem_floor.astype(np.uint8), 0.45, (0,255,0))).save(out_dir_cam/"floor_on_rgb.png")
            Image.fromarray(overlay_gray_on_rgb(rgb_buf["arr"], depth_gray_u8, 0.45)).save(out_dir_cam/"depth_on_rgb.png")

            ground_valid_mask = valid & sem_floor
            ground_z_med = None
            Zw_floor = Zw[ground_valid_mask]
            if Zw_floor.size > 50:
                ground_z_med = float(np.median(Zw_floor))
            if args.ground_z_tol > 0 and Zw_floor.size > 100:
                Z0 = float(np.median(Zw_floor)); tol = float(args.ground_z_tol)
                ground_valid_mask = ground_valid_mask & (np.abs(Zw - Z0) < tol)
                ground_z_med = Z0
                print(f"[{cam_cfg['name']}] ground Z median={Z0:.3f}, tol={tol:.3f}")

            # Save NPZ (for downstream use)
            np.savez_compressed(
                out_dir_cam/"pixel2world_lut.npz",
                X=Xw, Y=Yw, Z=Zw,
                valid_mask=valid.astype(np.uint8),
                floor_mask=sem_floor.astype(np.uint8),
                ground_valid_mask=ground_valid_mask.astype(np.uint8),
                K=K.astype(np.float32),
                cam_pose=np.array([cam_cfg["x"],cam_cfg["y"],cam_cfg["z"],
                                   cam_cfg["pitch"],cam_cfg["yaw"],cam_cfg.get("roll",0.0)],dtype=np.float32),
                width=np.int32(W), height=np.int32(H), fov=np.float32(FOV),
                ray_model=np.array(args.ray_model, dtype='U'),
                sem_channel=np.array(args.sem_channel, dtype='U'),
                floor_ids=np.array(sorted(list(floor_ids)), dtype=np.int32),
                M_c2w=M_c2w.astype(np.float64)
            )

            # -----------------------
            # Point cloud selection
            # -----------------------
            # 기본: ground_valid_mask 또는 valid를 사용해서 "full" 클라우드 구성
            mask_for_cloud = (ground_valid_mask if args.cloud_ground_only else valid)
            idx_full = np.where(mask_for_cloud)
            pts_full = np.stack([Xw[idx_full], Yw[idx_full], Zw[idx_full]], axis=1)
            rgb_full = rgb_buf["arr"][idx_full] / 255.0

            # lane-only (semantic == lane_id)
            lane_mask = (sem_id == args.lane_id) & valid
            idx_lane = np.where(lane_mask)
            pts_lane = np.stack([Xw[idx_lane], Yw[idx_lane], Zw[idx_lane]], axis=1) if idx_lane[0].size > 0 else np.zeros((0,3),dtype=np.float32)
            rgb_lane = (rgb_buf["arr"][idx_lane] / 255.0) if idx_lane[0].size > 0 else np.zeros((0,3),dtype=np.float32)

            # road+lane (semantic in {road_ids, lane_id})
            road_ids_set = set(int(x) for x in args.road_ids.split(",")) if args.road_ids else {1}
            road_lane_mask = np.isin(sem_id, list(road_ids_set | {args.lane_id})) & valid
            idx_rl = np.where(road_lane_mask)
            pts_rl = np.stack([Xw[idx_rl], Yw[idx_rl], Zw[idx_rl]], axis=1) if idx_rl[0].size > 0 else np.zeros((0,3),dtype=np.float32)
            rgb_rl = (rgb_buf["arr"][idx_rl] / 255.0) if idx_rl[0].size > 0 else np.zeros((0,3),dtype=np.float32)

            return {
                "name": cam_cfg["name"],
                "points_full": pts_full.astype(np.float64),
                "colors_full": rgb_full.astype(np.float32),
                "points_lane": pts_lane.astype(np.float64),
                "colors_lane": rgb_lane.astype(np.float32),
                "points_road_lane": pts_rl.astype(np.float64),
                "colors_road_lane": rgb_rl.astype(np.float32),
            }

    finally:
        for s in (sensor_depth, sensor_rgb, sensor_sem):
            if s is not None:
                s.stop(); s.destroy()


# -------------- Main --------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--fps", type=int, default=20)

    ap.add_argument("--cam-config", type=str, default="", help="(optional) JSON list of cameras")

    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--min-depth", type=float, default=0.05)
    ap.add_argument("--max-depth", type=float, default=200.0)
    ap.add_argument("--ray-model", choices=["normalized","forward"], default="forward")
    ap.add_argument("--sem-channel", choices=["auto","r","g","b"], default="auto")
    ap.add_argument("--floor-ids", type=str, default="7,10")
    ap.add_argument("--ground-z-tol", type=float, default=0.0)
    ap.add_argument("--sample-step", type=int, default=0)
    ap.add_argument("--out-root", type=str, default="./multi_out")
    ap.add_argument("--voxel", type=float, default=0.0, help=">0: voxel downsample fused points (m)")
    ap.add_argument("--cloud-ground-only", type=int, default=0, help="1=only ground for points, 0=all valid")

    # 추가: lane/road semantic id 설정
    ap.add_argument("--lane-id", type=int, default=24, help="semantic id for lane marking")
    ap.add_argument("--road-ids", type=str, default="1", help="comma-separated semantic ids for road surface")

    args = ap.parse_args()
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    # Camera list
    if args.cam_config and Path(args.cam_config).exists():
        with open(args.cam_config,"r") as f:
            cam_list = json.load(f)
        assert isinstance(cam_list, list) and len(cam_list) > 0
        print(f"[INFO] Using external cam-config: {args.cam_config} (n={len(cam_list)})")
    else:
        cam_list = CAM_LIST_DEFAULT
        print(f"[INFO] Using inline CAM_LIST_DEFAULT (n={len(cam_list)})")

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    per_cam = []
    for i, cam in enumerate(cam_list, 1):
        name = cam.get("name", f"cam_{i:02d}")
        out_dir_cam = out_root / f"cam_{name}"
        print(f"\n[+] Processing camera: {name}")
        rec = process_one_camera(world, bp_lib, args, cam, out_dir_cam)
        if rec is None or rec["points_full"].shape[0] == 0:
            print(f"[!] {name}: no points")
            continue
        per_cam.append(rec)

    if not per_cam:
        print("[!] No camera produced any points. Exit.")
        return

    # ---------- 1) Fused FULL ----------
    fused_full_pts = np.concatenate([c["points_full"] for c in per_cam if c["points_full"].shape[0] > 0], axis=0)
    fused_full_cols = np.concatenate([c["colors_full"] for c in per_cam if c["colors_full"].shape[0] > 0], axis=0)

    # ---------- 2) Fused LANE-only ----------
    fused_lane_pts = np.concatenate([c["points_lane"] for c in per_cam if c["points_lane"].shape[0] > 0], axis=0) \
                     if any(c["points_lane"].shape[0] > 0 for c in per_cam) else np.zeros((0,3),dtype=np.float64)
    fused_lane_cols = np.concatenate([c["colors_lane"] for c in per_cam if c["colors_lane"].shape[0] > 0], axis=0) \
                      if fused_lane_pts.shape[0] > 0 else np.zeros((0,3),dtype=np.float32)

    # ---------- 3) Fused ROAD+LANE ----------
    fused_rl_pts = np.concatenate([c["points_road_lane"] for c in per_cam if c["points_road_lane"].shape[0] > 0], axis=0) \
                   if any(c["points_road_lane"].shape[0] > 0 for c in per_cam) else np.zeros((0,3),dtype=np.float64)
    fused_rl_cols = np.concatenate([c["colors_road_lane"] for c in per_cam if c["colors_road_lane"].shape[0] > 0], axis=0) \
                    if fused_rl_pts.shape[0] > 0 else np.zeros((0,3),dtype=np.float32)

    # Optional voxel downsample (각 fused cloud별로)
    def voxel_downsample_if_needed(pts, cols, voxel_size):
        if voxel_size <= 0 or pts.shape[0] == 0:
            return pts, cols
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        if cols is not None and cols.shape[0] == pts.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
        pts_ds = np.asarray(pcd.points)
        cols_ds = np.asarray(pcd.colors) if np.asarray(pcd.colors).shape[0] == pts_ds.shape[0] else None
        if cols_ds is None:
            cols_ds = np.ones((pts_ds.shape[0],3),dtype=np.float32)*0.5
        return pts_ds, cols_ds

    if args.voxel and args.voxel > 0:
        fused_full_pts, fused_full_cols = voxel_downsample_if_needed(fused_full_pts, fused_full_cols, args.voxel)
        if fused_lane_pts.shape[0] > 0:
            fused_lane_pts, fused_lane_cols = voxel_downsample_if_needed(fused_lane_pts, fused_lane_cols, args.voxel)
        if fused_rl_pts.shape[0] > 0:
            fused_rl_pts, fused_rl_cols = voxel_downsample_if_needed(fused_rl_pts, fused_rl_cols, args.voxel)

    # ---- Save PLYs ----
    def save_ply(path, pts, cols):
        if pts.shape[0] == 0:
            print(f"[WARN] {path} : no points to save, skip.")
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        if cols is not None and cols.shape[0] == pts.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.ones((pts.shape[0],3),dtype=np.float64)*0.5)
        o3d.io.write_point_cloud(str(path), pcd, write_ascii=True)
        print(f"[OK] Saved PLY: {path} ({pts.shape[0]} pts)")

    # 1) full
    save_ply(out_root / "global_fused_full.ply", fused_full_pts, fused_full_cols)

    # 2) lane-only
    if fused_lane_pts.shape[0] > 0:
        save_ply(out_root / "global_lane_only.ply", fused_lane_pts, fused_lane_cols)
    else:
        print("[INFO] No lane points (semantic == lane-id). global_lane_only.ply not created.")

    # 3) road+lane
    if fused_rl_pts.shape[0] > 0:
        save_ply(out_root / "global_road_lane.ply", fused_rl_pts, fused_rl_cols)
    else:
        print("[INFO] No road+lane points (semantic in road_ids ∪ {lane-id}). global_road_lane.ply not created.")


if __name__ == "__main__":
    main()

"""
예시 실행:

python multi_cam_pixel2world_and_fuse_inline.py \
  --host 127.0.0.1 --port 2000 \
  --width 1920 --height 1080 --fov 89 \
  --ray-model forward \
  --floor-ids 1,2,10,24 \
  --lane-id 24 \
  --road-ids 1 \
  --cloud-ground-only 0 \
  --voxel 0 \
  --out-root ./multi_out_dense
"""