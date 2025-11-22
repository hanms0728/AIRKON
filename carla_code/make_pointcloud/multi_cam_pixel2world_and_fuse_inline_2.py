#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA (0.9.x) — Multi‑Camera Pixel→World LUT + Global Fused OBJ with Colored Camera Boundaries

What you get
- Per‑camera capture (RGB/Depth/Semantic), 3D reconstruction (camera→world), ground filtering
- Per‑camera outputs (rgb/semantic/depth heatmaps & npz) saved under out_root/cam_<name>/
- Single **OBJ** `global_fused_combo.obj` that contains:
  (A) All fused points as either
      • point primitives (lighter, but some viewers hide tiny points), or
      • tiny quads (faces) per point (heavier, but shows everywhere). See `--points-as-quads`.
  (B) Per‑camera **colored boundary strips** (convex hull on ground XY) as thin face ribbons.

Notes
- World coord = CARLA/UE4 (left‑handed): X forward, Y right, Z up.
- Depth is Euclidean distance to first hit. Reconstruction uses `forward` by default (stable in your tests).
- No external SciPy/Shapely required; convex hull via monotone chain.

Usage (example)
python multi_cam_fuse_with_colored_boundaries.py \
  --host 127.0.0.1 --port 2000 \
  --width 1920 --height 1080 --fov 89 \
  --ray-model forward --floor-ids 1,2,10,24 \
  --cloud-ground-only 1 --voxel 0.05 \
  --strip-width 0.08 \
  --points-as-quads 1 --point-quad-size 0.04 --max-point-quads 200000 \
  --out-root ./multi_out

If you want inline cams (default) or override with JSON: see --cam-config.
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
]

# Distinct boundary colors per camera (RGB 0..1)
CAM_PALETTE = [
    (0.90,0.10,0.10), (0.10,0.70,0.10), (0.10,0.35,0.95), (0.90,0.60,0.10),
    (0.65,0.25,0.90), (0.10,0.80,0.80), (0.95,0.35,0.35), (0.30,0.90,0.30),
    (0.35,0.60,0.95), (0.95,0.80,0.35), (0.80,0.30,0.95), (0.35,0.95,0.90),
    (0.95,0.35,0.65), (0.60,0.95,0.35), (0.35,0.35,0.35)
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


# -------------- Geometry helpers --------------

def convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    """Monotone chain convex hull. points_xy: (N,2). Returns hull points in CCW order (M,2)."""
    pts = np.unique(points_xy.astype(np.float64), axis=0)
    if pts.shape[0] <= 2:
        return pts
    pts = pts[np.lexsort((pts[:,1], pts[:,0]))]  # sort by x, then y
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull


def polyline_to_strip_quads(poly_xy: np.ndarray, width: float, closed: bool=True) -> np.ndarray:
    """Make thin quads (two triangles each) along a polyline on XY. Return (4*K, 3) vertices (Z=0).
    We'll set Z later by adding per-vertex z.
    """
    if poly_xy.shape[0] < 2:
        return np.zeros((0,3), dtype=np.float64)
    quads = []
    n = poly_xy.shape[0]
    idx_iter = range(n) if closed else range(n-1)
    half = width * 0.5
    for i in idx_iter:
        j = (i+1) % n
        p0 = poly_xy[i]; p1 = poly_xy[j]
        e = p1 - p0
        L = np.hypot(e[0], e[1])
        if L < 1e-6:  # skip degenerate
            continue
        nrm = np.array([-e[1]/L, e[0]/L])  # left normal (XY)
        # Offset left/right
        v0 = p0 + nrm*half; v1 = p0 - nrm*half
        v2 = p1 - nrm*half; v3 = p1 + nrm*half
        quads.append([v0, v1, v2, v3])
    if not quads:
        return np.zeros((0,3), dtype=np.float64)
    Q = np.array(quads, dtype=np.float64).reshape(-1,2)  # (4*K,2)
    Q3 = np.concatenate([Q, np.zeros((Q.shape[0],1), dtype=np.float64)], axis=1)  # add Z=0
    return Q3  # (4*K,3)


def build_point_quads_xy(points_xyz: np.ndarray, size: float) -> np.ndarray:
    """Each point → tiny square (XY plane), centered at point, preserving Z. Return (4*N,3)."""
    if points_xyz.shape[0] == 0:
        return np.zeros((0,3), dtype=np.float64)
    half = float(size) * 0.5
    base = np.array([[-half, -half, 0.0],
                     [ half, -half, 0.0],
                     [ half,  half, 0.0],
                     [-half,  half, 0.0]], dtype=np.float64)
    N = points_xyz.shape[0]
    quads = np.repeat(points_xyz.astype(np.float64), 4, axis=0) + np.tile(base, (N,1))
    return quads


# -------------- Simple OBJ writer (with vertex colors) --------------
class ObjWriter:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, 'w', encoding='utf-8')
        self.v_count = 0
        self._write_header()
    def _write_header(self):
        self.f.write("# global_fused_combo.obj (points + colored boundary strips)\n")
    def add_vertices_with_color(self, verts: np.ndarray, colors01: np.ndarray) -> int:
        assert verts.shape[0] == colors01.shape[0]
        start = self.v_count + 1
        for i in range(verts.shape[0]):
            x,y,z = verts[i]
            r,g,b = colors01[i]
            self.f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
        self.v_count += verts.shape[0]
        return start
    def add_point_primitives(self, start_idx: int, count: int):
        # OBJ 'p' uses vertex indices (1-based). Many viewers ignore 'p'; kept for completeness.
        idxs = [str(i) for i in range(start_idx, start_idx+count)]
        self.f.write("p " + " ".join(idxs) + "\n")
    def add_faces_quads_as_tris(self, i0, i1, i2, i3):
        self.f.write(f"f {i0} {i1} {i2}\n")
        self.f.write(f"f {i0} {i2} {i3}\n")
    def close(self):
        self.f.close()


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

            for _ in range(4): world.tick()

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
                for vi,ci in zip(vals[order],counts[order]): hf.write(f"{int(vi)},{int(ci)}\n")

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

            # Collect point cloud (choose mask)
            mask_for_cloud = (ground_valid_mask if args.cloud_ground_only else valid)
            idx = np.where(mask_for_cloud)
            pts = np.stack([Xw[idx], Yw[idx], Zw[idx]], axis=1)
            rgb = rgb_buf["arr"][idx] / 255.0

            # For boundary hull: use ground XY points (downsample for speed)
            gx, gy = Xw[ground_valid_mask], Yw[ground_valid_mask]
            if gx.size > 0:
                gxy = np.stack([gx, gy], axis=1)
                if gxy.shape[0] > 200000:
                    sel = np.random.choice(gxy.shape[0], 200000, replace=False)
                    gxy = gxy[sel]
                hull_xy = convex_hull_xy(gxy)
            else:
                hull_xy = np.zeros((0,2), dtype=np.float64)

            if ground_z_med is None:
                # Fallback: camera base height projected: use cam z minus some pitch dependent offset
                ground_z_med = float(cam_cfg["z"]) - 10.0  # heuristic fallback

            return {
                "name": cam_cfg["name"],
                "points": pts.astype(np.float64),
                "colors": rgb.astype(np.float32),
                "hull_xy": hull_xy.astype(np.float64),
                "hull_z": float(ground_z_med)
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
    ap.add_argument("--cloud-ground-only", type=int, default=1, help="1=only ground for points, 0=all valid")

    # Visual fusion settings
    ap.add_argument("--strip-width", type=float, default=0.05, help="boundary strip width (m)")
    ap.add_argument("--points-as-quads", type=int, default=0, help="1: output fused points as tiny quads (faces)")
    ap.add_argument("--point-quad-size", type=float, default=0.04, help="quad edge length (m) for per-point quads")
    ap.add_argument("--max-point-quads", type=int, default=250000, help="cap #quads for file size control")

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
        if rec is None or rec["points"].shape[0] == 0:
            print(f"[!] {name}: no points")
            continue
        # Optional per-cam voxel downsample BEFORE fusion (speed/memory)
        if args.voxel and args.voxel > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(rec["points"].astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(rec["colors"].astype(np.float64))
            pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel))
            rec["points"] = np.asarray(pcd.points)
            rec["colors"] = np.asarray(pcd.colors)
        per_cam.append(rec)

    if not per_cam:
        print("[!] No camera produced any points. Exit.")
        return

    # Build OBJ: points (as quads or points) + boundary strips (colored per cam)
    obj_path = out_root / "global_fused_combo.obj"
    ow = ObjWriter(str(obj_path))

    # 1) Fused points
    fused_pts = np.concatenate([c["points"] for c in per_cam], axis=0)
    fused_cols = np.concatenate([c["colors"] for c in per_cam], axis=0)
    if fused_pts.shape[0] > 0:
        if args.points_as_quads:
            N = fused_pts.shape[0]
            keep = min(args.max_point_quads, N)
            sel = np.arange(N) if N <= keep else np.random.choice(N, keep, replace=False)
            pts_sel = fused_pts[sel].astype(np.float64)
            cols_sel = fused_cols[sel].astype(np.float32)
            quads_xyz = build_point_quads_xy(pts_sel, size=args.point_quad_size)   # (4*K,3)
            quads_col = np.repeat(cols_sel, 4, axis=0)
            v0 = ow.add_vertices_with_color(quads_xyz, quads_col)
            K = quads_xyz.shape[0] // 4
            for q in range(K):
                i0 = v0 + q*4 + 0
                i1 = v0 + q*4 + 1
                i2 = v0 + q*4 + 2
                i3 = v0 + q*4 + 3
                ow.add_faces_quads_as_tris(i0, i1, i2, i3)
        else:
            v0 = ow.add_vertices_with_color(fused_pts.astype(np.float64), fused_cols.astype(np.float32))
            ow.add_point_primitives(v0, fused_pts.shape[0])

    # 2) Colored boundary strips per cam
    for idx, rec in enumerate(per_cam):
        hull_xy = rec["hull_xy"]
        if hull_xy.shape[0] < 2:
            continue
        color = CAM_PALETTE[idx % len(CAM_PALETTE)]
        # Build thin strip in XY
        strip_xyz = polyline_to_strip_quads(hull_xy, width=float(args.strip_width), closed=True)  # (4*K,3) with Z=0
        if strip_xyz.shape[0] == 0:
            continue
        # Place at ground_z (+ small epsilon to float above points)
        strip_xyz[:,2] = rec["hull_z"] + 0.02
        cols = np.tile(np.array(color, dtype=np.float32).reshape(1,3), (strip_xyz.shape[0],1))
        v1 = ow.add_vertices_with_color(strip_xyz, cols)
        K = strip_xyz.shape[0] // 4
        for q in range(K):
            i0 = v1 + q*4 + 0
            i1 = v1 + q*4 + 1
            i2 = v1 + q*4 + 2
            i3 = v1 + q*4 + 3
            ow.add_faces_quads_as_tris(i0, i1, i2, i3)

    ow.close()
    print(f"\n[OK] Saved single-file OBJ: {obj_path}")


if __name__ == "__main__":
    main()

"""
예시 실행:

python multi_cam_pixel2world_and_fuse_inline.py \
--host 127.0.0.1 --port 2000 \
  --width 1920 --height 1080 --fov 89 \
  --ray-model forward \
  --floor-ids 1,2,10,24 \
  --cloud-ground-only 0 \
  --voxel 0 \
  --points-as-quads 1 \
  --point-quad-size 0.02 \
  --max-point-quads 2000000 \
  --strip-width 0.06 \
  --out-root ./multi_out_dense
"""