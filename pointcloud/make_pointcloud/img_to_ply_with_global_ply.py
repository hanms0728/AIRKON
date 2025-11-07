#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-view visible-point extraction & colorization.

기능:
- 전역 PLY를 입력받아,
- 지정된 카메라 위치/자세/FOV 기반으로
- 해당 시야 내에서 실제로 보이는 점만 추출(Z-buffer)
- RGB 이미지 색상을 투영해 색칠한 후
- 그 가시 포인트클라우드만 별도 PLY로 저장.

즉, CARLA 카메라가 보는 실제 시야의 PLY를 재현.
- 위치 좌표계 선택( --pos-space carla|cv ) 및 Y부호 플립 옵션( --flip-y ) 지원
- 디버그: CARLA→CV 좌표 변환 결과를 콘솔에 출력.
"""

import os, math, argparse
import numpy as np
import cv2
import open3d as o3d
from typing import Optional, Tuple
import time
import random
import threading
from collections import deque

# --- Helper: integer disk offsets for splatting ---
def _disk_offsets(radius: int):
    """
    Return list of (dx, dy) integer offsets inside a filled disk of given radius.
    radius=1 → only center pixel (no expansion). radius=2 → ~5px, radius=3 → ~13px, etc.
    """
    r = int(radius)
    if r <= 1:
        return [(0, 0)]
    off = []
    r2 = r * r
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx*dx + dy*dy <= r2:
                off.append((dx, dy))
    return off

# Deterministic RNG for stable sampling (prevents frame-to-frame flicker)
_RNG = np.random.default_rng(42)
# ---------------- main ----------------
class RenderWorker:
    """
    Background renderer: projects subsampled points and builds BEV preview
    whenever the latest pose/FOV changes. The main UI thread only handles
    key inputs and displays the most recent frame from this worker.
    """
    def __init__(self, pts_live, cols_live, img_bgr, z_near, z_far, splat, initial_pose, initial_fov, bev_size=(1600, 780), pad=15):
        self.pts_live = pts_live
        self.cols_live = cols_live
        self.img_bgr = img_bgr
        self.z_near = float(z_near)
        self.z_far = float(z_far)
        self.splat = int(splat)
        self.bev_w, self.bev_h = bev_size
        self.pad = int(pad)

        # shared state
        self._lock = threading.Lock()
        self._pose = list(initial_pose)  # [x,y,z,yaw,pitch,roll]
        self._fov = float(initial_fov)
        self._dirty = True
        self._stop = False

        # double buffer for latest frame
        self._frame = None
        self._hud_text = ""
        self._render_every_n = 1      # render every request
        self._skip_bev_every = 1      # compute BEV every frame for snappier updates
        self._counter = 0

        self._frame_seq = 0
        self._last_bev = None
        # Precompute a stable subset for BEV global drawing to avoid flicker
        N = len(self.pts_live)
        self._idx_bev_global = None
        try:
            target = 200000  # lighter BEV draw for faster refresh
            if N > target:
                self._idx_bev_global = _RNG.choice(N, target, replace=False)
        except Exception:
            self._idx_bev_global = None

        # Overlay blending parameters
        self.cam_alpha = 0.70
        self.global_gain = 0.60

        self._thread = threading.Thread(target=self._run, daemon=True)
    def adjust_cam_alpha(self, delta):
        with self._lock:
            self.cam_alpha = float(np.clip(self.cam_alpha + delta, 0.0, 1.0))
            self._dirty = True

    def adjust_global_gain(self, delta):
        with self._lock:
            # allow up to 1.5 to brighten global layer if desired
            self.global_gain = float(np.clip(self.global_gain + delta, 0.0, 1.5))
            self._dirty = True

    def start(self):
        self._thread.start()

    def stop(self):
        with self._lock:
            self._stop = True
        self._thread.join(timeout=1.0)

    def update(self, x, y, z, yaw, pitch, roll, fov, hud_text=""):
        """Update latest target pose/FOV from UI thread."""
        with self._lock:
            self._pose = [float(x), float(y), float(z), float(yaw), float(pitch), float(roll)]
            self._fov = float(fov)
            self._hud_text = hud_text
            self._dirty = True

    def get_frame(self):
        """Return the most recent composed frame (np.uint8 HxWx3) and frame sequence number, or (None, seq)."""
        with self._lock:
            if self._frame is None:
                return None, self._frame_seq
            return self._frame.copy(), self._frame_seq

    def _run(self):
        # local references (avoid attribute lookups in loop)
        pts = self.pts_live
        cols = self.cols_live
        img = self.img_bgr
        H, W = img.shape[:2]

        while True:
            with self._lock:
                if self._stop:
                    break
                if not self._dirty:
                    # sleep a bit to avoid busy spin
                    need_sleep = True
                else:
                    x, y, z, yaw, pitch, roll = self._pose
                    fov = self._fov
                    hud_text = self._hud_text
                    self._dirty = False
                    need_sleep = False

            if need_sleep:
                time.sleep(0.01)
                continue

            try:
                # compute projection for current pose
                K_live = K_from_fov(W, H, fov)
                Rt_live, R_wc_live, _ = pose_to_extrinsic(x, y, z, yaw, pitch, roll)
                pts_vis_live, cols_vis_live = project_visible_points(
                    pts, img, K_live, Rt_live, self.z_near, self.z_far, self.splat
                )

                self._counter += 1
                # compute BEV not every frame to lighten load
                if (self._counter % self._skip_bev_every) == 0:
                    bev = make_bev_preview(pts, cols, pts_vis_live, cols_vis_live,
                                           bev_w=self.bev_w, bev_h=self.bev_h, pad=self.pad,
                                           idx_global_subset=self._idx_bev_global,
                                           cam_alpha=self.cam_alpha, global_gain=self.global_gain)
                    self._last_bev = bev.copy()
                else:
                    # reuse previous BEV to avoid flicker/blank frames
                    bev = self._last_bev.copy() if self._last_bev is not None else np.zeros((self.bev_h, self.bev_w, 3), np.uint8)

                # overlay HUD text at bottom of BEV
                if hud_text:
                    cv2.putText(bev, hud_text, (20, self.bev_h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # store frame
                with self._lock:
                    self._frame = bev
                    self._frame_seq += 1

            except Exception as e:
                # Don't crash the thread; write a small error frame
                err = np.zeros((200, 800, 3), np.uint8)
                cv2.putText(err, f"[RenderWorker] {type(e).__name__}: {e}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                with self._lock:
                    self._frame = err
                time.sleep(0.05)

def timestamp_str():
    return time.strftime("%Y%m%d_%H%M%S")

def format_pose(x,y,z,yaw,pitch,roll,fov):
    return f"pos=({x:.2f},{y:.2f},{z:.2f})  rot(ypr)=({yaw:.2f},{pitch:.2f},{roll:.2f})  fov={fov:.2f}°"

def subsample_points(pts: np.ndarray, cols: np.ndarray, limit: int):
    if limit is None or limit <= 0 or len(pts) <= limit:
        return pts, cols
    idx = _RNG.choice(len(pts), limit, replace=False)
    return pts[idx], cols[idx]

def make_bev_preview(pts_cv: np.ndarray, cols_cv: np.ndarray,
                     pts_vis_cv: np.ndarray, cols_vis: np.ndarray,
                     bev_w: int = 1600, bev_h: int = 780, pad: int = 15,
                     idx_global_subset: Optional[np.ndarray] = None,
                     cam_alpha: float = 0.70, global_gain: float = 0.60) -> np.ndarray:
    """
    Build a BEV (CARLA X-Y) preview image:
    - global PLY footprint (dim, using its colors)
    - visible footprint (vivid)
    Input pts are in CV/world; we convert back to CARLA to use (X,Y).
    """
    if pts_cv.size == 0:
        return np.zeros((bev_h, bev_w, 3), np.uint8)

    T_CV_TO_CARLA = np.linalg.inv(_T_CARLA_WORLD_TO_CV_WORLD)
    pts_carla = (pts_cv @ T_CV_TO_CARLA.T)
    xy_global = pts_carla[:, [0, 1]].astype(np.float64)

    # bounds/scale
    gmin = xy_global.min(axis=0)
    gmax = xy_global.max(axis=0)
    grng = np.maximum(gmax - gmin, 1e-6)

    # Fit-to-rectangle scaling (independent X/Y ranges), then center the content.
    scale_x = (bev_w - 2 * pad) / float(grng[0])
    scale_y = (bev_h - 2 * pad) / float(grng[1])
    scale = min(scale_x, scale_y)

    # Centering offsets so the map is not stuck at the top-left
    draw_w = grng[0] * scale
    draw_h = grng[1] * scale
    left = (bev_w - draw_w) * 0.5
    top  = (bev_h - draw_h) * 0.5

    def xy_to_px(xy):
        # X grows right, Y grows up in CARLA → invert Y into image coords
        px = left + (xy[:, 0] - gmin[0]) * scale
        py = top  + (gmax[1] - xy[:, 1]) * scale
        return np.stack([px, py], axis=1).astype(np.int32)

    # Two layers for clear composition
    global_layer = np.zeros((bev_h, bev_w, 3), np.uint8)
    cam_layer    = np.zeros((bev_h, bev_w, 3), np.uint8)

    # draw global dimmed (use stable subset if provided)
    if idx_global_subset is not None:
        xy_g = xy_global[idx_global_subset]
        cols_g = cols_cv[idx_global_subset]
    else:
        xy_g = xy_global
        cols_g = cols_cv

    px_g = xy_to_px(xy_g)
    mask_g = (px_g[:, 0] >= 0) & (px_g[:, 0] < bev_w) & (px_g[:, 1] >= 0) & (px_g[:, 1] < bev_h)
    px_g = px_g[mask_g]
    cols_g = np.clip(cols_g[mask_g] * global_gain, 0.0, 1.0)  # apply brightness gain on global
    for (px, py), c in zip(px_g, cols_g):
        color = (np.clip(c[::-1], 0.0, 1.0) * 255).astype(np.uint8)  # RGB->BGR
        cv2.circle(global_layer, (int(px), int(py)), 1, tuple(int(x) for x in color), -1)

    # draw visible vivid, thicker
    if pts_vis_cv is not None and len(pts_vis_cv) > 0:
        pts_vis_carla = (pts_vis_cv @ T_CV_TO_CARLA.T)
        xy_v = pts_vis_carla[:, [0, 1]].astype(np.float64)
        px_v = xy_to_px(xy_v)
        mask_v = (px_v[:, 0] >= 0) & (px_v[:, 0] < bev_w) & (px_v[:, 1] >= 0) & (px_v[:, 1] < bev_h)
        px_v = px_v[mask_v]
        cols_v = cols_vis[mask_v] if len(cols_vis) == len(pts_vis_cv) else np.ones((len(px_v), 3))
        for (px, py), c in zip(px_v, np.clip(cols_v, 0.0, 1.0)):
            color = (c[::-1] * 255).astype(np.uint8)
            cv2.circle(cam_layer, (int(px), int(py)), 2, tuple(int(x) for x in color), -1)

    # compose with alpha
    bev = cv2.addWeighted(global_layer, 1.0, cam_layer, float(np.clip(cam_alpha, 0.0, 1.0)), 0.0)
    # frame & legend
    cv2.rectangle(bev, (0, 0), (bev_w - 1, bev_h - 1), (90, 90, 90), 1)
    cv2.putText(bev, "Global PLY (CARLA X-Y) x gain", (24 + 22, 24 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(bev, (24, 24), (24 + 14, 24 + 14), (120, 120, 120), -1)
    cv2.putText(bev, "Camera-visible (CARLA X-Y) alpha blend", (24 + 22, 46 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(bev, (24, 46), (24 + 14, 46 + 14), (0, 255, 255), -1)
    return bev
# ---------------- Frames & Utilities ----------------
# CARLA→CV mapping used in your H computation (vector transform)
_AXES_CARLA_TO_CV = np.array([
    [0.0,  1.0,  0.0],   # Carla Y → CV X
    [0.0,  0.0, -1.0],   # Carla Z → CV Y (down is +)
    [1.0,  0.0,  0.0],   # Carla X → CV Z
], dtype=np.float64)

# The additional XY-swap-with-sign matrix used before applying _AXES_CARLA_TO_CV
_M_CARLA_SWAP = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
], dtype=np.float64)

# Combined CARLA-world point → CV-world point transform that matches your H code
_T_CARLA_WORLD_TO_CV_WORLD = _AXES_CARLA_TO_CV @ _M_CARLA_SWAP  # results in [X, -Z, -Y]


def carla_world_points_to_cv(points: np.ndarray) -> np.ndarray:
    """Map Nx3 CARLA world points (X,Y,Z in CARLA) into the CV/world frame
    consistent with compute_H_img_to_ground().
    """
    if points is None or len(points) == 0:
        return points
    pts = np.ascontiguousarray(points, dtype=np.float64)
    return (pts @ _T_CARLA_WORLD_TO_CV_WORLD.T)


def parse_vec3_csv(text: str) -> Tuple[float,float,float]:
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 3:
        raise ValueError("Expected 3 comma-separated values, got: %s" % text)
    return vals[0], vals[1], vals[2]


def print_bounds(tag: str, pts: np.ndarray) -> None:
    if pts is None or len(pts) == 0:
        print(f"[DEBUG] {tag}: empty")
        return
    gmin = np.min(pts, axis=0)
    gmax = np.max(pts, axis=0)
    center = (gmin + gmax) * 0.5
    extent = (gmax - gmin)
    print(f"[DEBUG] {tag} bounds:")
    print(f"        min: {gmin}")
    print(f"        max: {gmax}")
    print(f"        cen: {center}")
    print(f"        ext: {extent}")


# ---------------- Rotation / Pose ----------------
def rot_from_yaw_pitch_roll(yaw_deg, pitch_deg, roll_deg):
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    cz, sz = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)

    Rz = np.array([[cz, -sz, 0.0],
                   [sz,  cz, 0.0],
                   [0.0, 0.0, 1.0]])
    Ry = np.array([[ cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cr, -sr],
                   [0.0,  sr,  cr]])
    return Rz @ Ry @ Rx  # CARLA/UE convention


def pose_to_extrinsic(x, y, z, yaw, pitch, roll):
    """
    Build world->camera extrinsic (Rt) using the SAME CARLA→OpenCV mapping
    as compute_H_img_to_ground():
      axes_carla_to_cv = [[0,1,0],[0,0,-1],[1,0,0]]
      cam_pos_carla = [-Y, X, Z]
      cam_pos_cv = axes * cam_pos_carla
      yaw_carla = yaw + 90
      pitch_carla = -pitch
      roll_carla = roll
      R_cam_to_world_carla = Rz(yaw_carla) @ Ry(pitch_carla) @ Rx(roll_carla)
      R_world_to_cam_carla = R_cam_to_world_carla.T
      R_world_to_cam = axes * R_world_to_cam_carla * axes.T
      t = - R_world_to_cam * cam_pos_cv
    Note: We now also optionally transform the GLOBAL PLY into the same CV/world frame so axes & pose align.
    """
    axes_carla_to_cv = np.array([
        [0.0,  1.0,  0.0],   # Carla Y → CV X
        [0.0,  0.0, -1.0],   # Carla Z → CV Y (down is +)
        [1.0,  0.0,  0.0],   # Carla X → CV Z
    ], dtype=np.float64)

    # position: CARLA (x,y,z) -> cam_pos_carla = [-y, x, z] -> cam_pos_cv
    cam_pos_carla = np.array([-y, x, z], dtype=np.float64)
    cam_pos_cv = axes_carla_to_cv @ cam_pos_carla.reshape(3, 1)

    # orientation: match H computation
    yaw_carla   = yaw + 90.0
    pitch_carla = -pitch
    roll_carla  = roll

    R_cam_to_world_carla = rot_from_yaw_pitch_roll(yaw_carla, pitch_carla, roll_carla)
    R_world_to_cam_carla = R_cam_to_world_carla.T
    R_world_to_cam = axes_carla_to_cv @ R_world_to_cam_carla @ axes_carla_to_cv.T

    t = -R_world_to_cam @ cam_pos_cv
    Rt = np.hstack([R_world_to_cam, t])
    return Rt, R_world_to_cam, cam_pos_cv



def K_from_fov(W, H, fov_x_deg):
    fx = (W / 2.0) / math.tan(math.radians(fov_x_deg * 0.5))
    fy = fx
    cx, cy = W / 2.0, H / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)

def K_from_fov_hv(W, H, fov_x_deg, fov_y_deg):
    fx = (W / 2.0) / math.tan(math.radians(fov_x_deg * 0.5))
    fy = (H / 2.0) / math.tan(math.radians(fov_y_deg * 0.5))
    cx, cy = W / 2.0, H / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


def _axes_pointcloud(origin_xyz, R_axes, scale=5.0, n_pts=100, include_z=True, color_strength=1.0):
    """
    Build a point-sampled XYZ axes (as a PointCloud) for visualization.
    - origin_xyz: (3,) world position where axes start
    - R_axes: 3x3 rotation whose columns are axis directions in WORLD frame
    - scale: length in meters for each axis
    - n_pts: number of samples along each axis
    - include_z: if False, only X/Y axes are drawn
    - color_strength: 0..1 multiplier for colors (use <1 for dimmer axes)
    Returns: (pts[N,3], cols[N,3]) in world frame
    """
    origin_xyz = np.asarray(origin_xyz, dtype=np.float64).reshape(3)
    R_axes = np.asarray(R_axes, dtype=np.float64).reshape(3, 3)

    # Unit axis directions in world coordinates
    x_dir = R_axes[:, 0]
    y_dir = R_axes[:, 1]
    z_dir = R_axes[:, 2]

    ts = np.linspace(0.0, 1.0, n_pts, dtype=np.float64)
    pts_list = []
    cols_list = []

    # X axis (red)
    for t in ts:
        pts_list.append(origin_xyz + x_dir * (t * scale))
        cols_list.append(np.array([1.0, 0.0, 0.0]) * color_strength)

    # Y axis (green)
    for t in ts:
        pts_list.append(origin_xyz + y_dir * (t * scale))
        cols_list.append(np.array([0.0, 1.0, 0.0]) * color_strength)

    if include_z:
        # Z axis (blue)
        for t in ts:
            pts_list.append(origin_xyz + z_dir * (t * scale))
            cols_list.append(np.array([0.0, 0.0, 1.0]) * color_strength)

    # Add labeled direction markers (at tip + 0.5*scale)
    tip_offsets = [x_dir, y_dir]
    labels_colors = [np.array([1,0.3,0.3]), np.array([0.3,1,0.3])]
    if include_z:
        tip_offsets.append(z_dir)
        labels_colors.append(np.array([0.3,0.3,1]))
    for d, c in zip(tip_offsets, labels_colors):
        pts_list.append(origin_xyz + d * (scale * 1.05))
        cols_list.append(c)

    pts = np.vstack(pts_list) if pts_list else np.zeros((0, 3), dtype=np.float64)
    cols = np.vstack(cols_list) if pts_list else np.zeros((0, 3), dtype=np.float64)
    return pts, cols


def save_axes_ply(out_path, scene_pts, cam_pos_world, R_world_to_cam, include_global=True, include_camera=True, xy_only=False, base_ply_path=None, base_pts: Optional[np.ndarray]=None, base_cols: Optional[np.ndarray]=None):
    """
    Create a single PLY that shows:
      - Global axes at world origin (dim)
      - Camera-local axes at the camera position (vivid)
    The axes are point-sampled (so they display everywhere).
    - xy_only=True draws only X/Y axes (no Z).
    Axis lengths are chosen from scene bounds (10% of max extent, clamped).
    Optionally merges axes with an existing global PLY to visualize axes in context.
    """
    # Scene-based scale
    if scene_pts is not None and len(scene_pts) > 0:
        gmin = np.min(scene_pts, axis=0)
        gmax = np.max(scene_pts, axis=0)
        extent = np.max(gmax - gmin)
        scale = float(np.clip(0.10 * extent, 1.0, 20.0))  # 10% of scene, between [1m, 20m]
    else:
        scale = 5.0

    # Determine whether to include Z
    include_z = not xy_only

    all_pts = []
    all_cols = []

    if include_global:
        I = np.eye(3, dtype=np.float64)
        pts_g, cols_g = _axes_pointcloud(origin_xyz=np.zeros(3), R_axes=I, scale=scale, n_pts=120,
                                         include_z=include_z, color_strength=0.5)  # dim
        all_pts.append(pts_g)
        all_cols.append(cols_g)

    if include_camera:
        # Camera axes in WORLD frame are columns of R_cam_to_world = R_world_to_cam.T
        R_cam_to_world = R_world_to_cam.T
        pts_c, cols_c = _axes_pointcloud(origin_xyz=np.asarray(cam_pos_world, dtype=np.float64),
                                         R_axes=R_cam_to_world, scale=scale, n_pts=120,
                                         include_z=include_z, color_strength=1.0)  # vivid
        all_pts.append(pts_c)
        all_cols.append(cols_c)

    if len(all_pts) == 0:
        # nothing to save
        return False

    pts = np.vstack(all_pts)
    cols = np.vstack(all_cols)

    # Optionally merge with base global PLY (prefer in-memory arrays if provided)
    if base_pts is not None and len(base_pts) > 0:
        try:
            pts = np.vstack([base_pts, pts])
            if base_cols is not None and len(base_cols) == len(base_pts):
                cols = np.vstack([base_cols, cols])
            else:
                cols = np.vstack([np.ones_like(base_pts)*0.3, cols])
        except Exception as e:
            print(f"[WARN] Could not merge in-memory base PLY: {e}")
    elif base_ply_path and os.path.exists(base_ply_path):
        try:
            pcd_base = o3d.io.read_point_cloud(base_ply_path)
            pts_base = np.asarray(pcd_base.points)
            cols_base = np.asarray(pcd_base.colors)
            if pts_base.size > 0:
                pts = np.vstack([pts_base, pts])
                if cols_base.size == pts_base.size:
                    cols = np.vstack([cols_base, cols])
                else:
                    cols = np.vstack([np.ones_like(pts_base)*0.3, cols])
        except Exception as e:
            print(f"[WARN] Could not merge base PLY ({base_ply_path}): {e}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
    return True


# ---------------- Main colorization ----------------
def project_visible_points(pts_world, img_bgr, K, Rt, z_near=0.05, z_far=200.0, splat=1):
    H, W = img_bgr.shape[:2]
    R = Rt[:, :3]
    t = Rt[:, 3:4]
    Xc = (R @ pts_world.T) + t
    Z = Xc[2, :]
    valid = (Z > z_near) & (Z < z_far)
    if not np.any(valid):
        return np.empty((0,3)), np.empty((0,3))
    Xc = Xc[:, valid]
    if Xc.shape[1] == 0:
        return np.empty((0,3)), np.empty((0,3))

    uv_h = K @ Xc
    w = uv_h[2, :]
    finite = np.isfinite(uv_h).all(axis=0) & np.isfinite(w)
    if not np.any(finite):
        return np.empty((0,3)), np.empty((0,3))
    uv_h = uv_h[:, finite]
    Xc = Xc[:, finite]
    w = uv_h[2, :]

    u = (uv_h[0, :] / w).astype(np.int32)
    v = (uv_h[1, :] / w).astype(np.int32)

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[inside], v[inside]
    Z, idx = Z[valid][finite][inside], np.nonzero(valid)[0][finite][inside]

    depth = np.full((H, W), np.inf)
    pidx = np.full((H, W), -1, np.int32)

    if int(splat) <= 1:
        for uu, vv, zz, wi in zip(u, v, Z, idx):
            if zz < depth[vv, uu]:
                depth[vv, uu] = zz
                pidx[vv, uu] = wi
    else:
        offsets = _disk_offsets(int(splat))
        for uu, vv, zz, wi in zip(u, v, Z, idx):
            for dx, dy in offsets:
                xx = uu + dx
                yy = vv + dy
                if 0 <= xx < W and 0 <= yy < H:
                    if zz < depth[yy, xx]:
                        depth[yy, xx] = zz
                        pidx[yy, xx] = wi

    hit_mask = pidx >= 0
    ys, xs = np.nonzero(hit_mask)
    hit_indices = pidx[ys, xs]
    colors = img_bgr[ys, xs][:, ::-1] / 255.0  # BGR→RGB
    pts_visible = pts_world[hit_indices]

    # 평균화 (같은 점에 여러 픽셀 대응 시)
    uniq_idx, inv, counts = np.unique(hit_indices, return_inverse=True, return_counts=True)
    color_accum = np.zeros((len(uniq_idx), 3))
    np.add.at(color_accum, inv, colors)
    color_accum /= counts[:, None]

    pts_final = pts_world[uniq_idx]
    cols_final = color_accum
    return pts_final, cols_final


# ---- Added: Per-pixel XYZ map/z-buffer helper ----
def project_visible_points_with_maps(pts_world, img_bgr, K, Rt, z_near=0.05, z_far=200.0, splat=1):
    """
    Same as project_visible_points, but also returns dense per-pixel maps (and honors `splat` as a disk radius for pixel fill):
      - Xmap, Ymap, Zmap in WORLD frame of pts_world (same frame as pts_world)
      - hit_mask (H,W) where True means a 3D point was visible at that pixel
    Non-hit pixels in the maps are set to NaN.
    """
    H, W = img_bgr.shape[:2]
    R = Rt[:, :3]
    t = Rt[:, 3:4]
    Xc = (R @ pts_world.T) + t
    Z = Xc[2, :]
    valid = (Z > z_near) & (Z < z_far)
    if not np.any(valid):
        return np.empty((0,3)), np.empty((0,3)), np.full((H, W), np.nan, dtype=np.float32), np.full((H, W), np.nan, dtype=np.float32), np.full((H, W), np.nan, dtype=np.float32), np.zeros((H, W), dtype=bool)
    Xc = Xc[:, valid]
    if Xc.shape[1] == 0:
        return np.empty((0,3)), np.empty((0,3)), np.full((H, W), np.nan, dtype=np.float32), np.full((H, W), np.nan, dtype=np.float32), np.full((H, W), np.nan, dtype=np.float32), np.zeros((H, W), dtype=bool)

    uv_h = K @ Xc
    w = uv_h[2, :]
    finite = np.isfinite(uv_h).all(axis=0) & np.isfinite(w)
    if not np.any(finite):
        return np.empty((0,3)), np.empty((0,3)), np.full((H, W), np.nan, dtype=np.float32), np.full((H, W), np.nan, dtype=np.float32), np.full((H, W), np.nan, dtype=np.float32), np.zeros((H, W), dtype=bool)
    uv_h = uv_h[:, finite]
    Xc = Xc[:, finite]
    w = uv_h[2, :]

    u = (uv_h[0, :] / w).astype(np.int32)
    v = (uv_h[1, :] / w).astype(np.int32)

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[inside], v[inside]
    Z, idx = Z[valid][finite][inside], np.nonzero(valid)[0][finite][inside]

    depth = np.full((H, W), np.inf)
    pidx = np.full((H, W), -1, np.int32)

    # z-buffer resolve with optional splat (disk) fill
    if int(splat) <= 1:
        for uu, vv, zz, wi in zip(u, v, Z, idx):
            if zz < depth[vv, uu]:
                depth[vv, uu] = zz
                pidx[vv, uu] = wi
    else:
        offsets = _disk_offsets(int(splat))
        for uu, vv, zz, wi in zip(u, v, Z, idx):
            for dx, dy in offsets:
                xx = uu + dx
                yy = vv + dy
                if 0 <= xx < W and 0 <= yy < H:
                    if zz < depth[yy, xx]:
                        depth[yy, xx] = zz
                        pidx[yy, xx] = wi

    hit_mask = pidx >= 0
    ys, xs = np.nonzero(hit_mask)
    if ys.size == 0:
        return np.empty((0,3)), np.empty((0,3)), np.full((H, W), np.nan, dtype=np.float32), np.full((H, W), np.nan, dtype=np.float32), np.full((H, W), np.nan, dtype=np.float32), np.zeros((H, W), dtype=bool)

    hit_indices = pidx[ys, xs]
    colors = img_bgr[ys, xs][:, ::-1] / 255.0  # BGR→RGB

    # Average colors if multiple pixels hit the same 3D point
    uniq_idx, inv, counts = np.unique(hit_indices, return_inverse=True, return_counts=True)
    color_accum = np.zeros((len(uniq_idx), 3), dtype=np.float64)
    np.add.at(color_accum, inv, colors)
    color_accum /= counts[:, None]
    pts_final = pts_world[uniq_idx]
    cols_final = color_accum.astype(np.float64)

    # Build dense XYZ maps (WORLD frame same as pts_world)
    Xmap = np.full((H, W), np.nan, dtype=np.float32)
    Ymap = np.full((H, W), np.nan, dtype=np.float32)
    Zmap = np.full((H, W), np.nan, dtype=np.float32)
    # For pixels, use their selected winning point coords
    Xmap[ys, xs] = pts_world[hit_indices, 0].astype(np.float32)
    Ymap[ys, xs] = pts_world[hit_indices, 1].astype(np.float32)
    Zmap[ys, xs] = pts_world[hit_indices, 2].astype(np.float32)

    return pts_final, cols_final, Xmap, Ymap, Zmap, hit_mask


# ---------------- I/O ----------------
def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if cols is None or len(cols) != len(pts):
        cols = np.ones((len(pts), 3), dtype=np.float64)  # white fallback
    return pts, cols


def save_ply(pts, cols, out):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(out, pcd, write_ascii=True)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser("Extract visible region of global PLY via image projection.")
    ap.add_argument("--ply", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pos", type=str, required=True, help="x,y,z (m)")
    ap.add_argument("--rot", type=str, required=True, help="yaw,pitch,roll (deg)")
    ap.add_argument("--fov", type=float, required=True)
    ap.add_argument("--fov-y", type=float, default=None, help="(optional) vertical field of view in degrees; if set, use both HFOV/VFOV")
    ap.add_argument("--z-near", type=float, default=0.05)
    ap.add_argument("--z-far", type=float, default=300.0)
    ap.add_argument("--splat", type=int, default=3)
    ap.add_argument("--strict-carla", action="store_true",
                    help="Interpret --pos/--rot strictly as CARLA world pose (default). Matches compute_H_img_to_ground mapping.")
    ap.add_argument("--debug-pos", action="store_true",
                    help="CARLA→CV 좌표 변환 및 사용된 Rt 요약을 출력" )
    ap.add_argument("--axes-out", type=str, default=None,
                    help="If set, save a PLY that visualizes XY(Z) axes: global origin (dim) + camera axes at camera position (vivid).")
    ap.add_argument("--xy-only", action="store_true",
                    help="When dumping axes, draw only X/Y axes (no Z).")
    ap.add_argument("--ply-frame", choices=["carla","cv"], default="carla",
                    help="What frame the input PLY is in. If 'carla', we convert points to CV/world to match H & pose mapping.")
    ap.add_argument("--origin", type=str, default=None,
                    help="Optional origin offset to subtract from points (ox,oy,oz) AFTER --ply-frame transform. Helps align your expected (0,0,0).")
    ap.add_argument("--dump-axes-merged", type=str, default=None,
                    help="If set, also save a merged PLY of (global scene + axes) to inspect origins/axes in one file.")
    ap.add_argument("--save-frame", choices=["cv","carla"], default="carla",
                    help="Coordinate frame of the output visible PLY. 'carla' converts back to CARLA world axes; 'cv' keeps the compute frame.")
    ap.add_argument("--interactive", action="store_true",
                    help="Enable live tuning: adjust pose/FOV with keys and preview coverage in real time.")
    ap.add_argument("--step-pos", type=float, default=0.20,
                    help="Position step (meters) for interactive mode. Default: 0.20")
    ap.add_argument("--step-ang", type=float, default=1.0,
                    help="Angle step (degrees) for interactive mode. Default: 1.0")
    ap.add_argument("--subsample", type=int, default=1000000,
                    help="Max points to use for live preview (random subsample, lower = smoother UI). Use all points when saving with Enter.")
    ap.add_argument("--live-save-dir", type=str, default=None,
                    help="If set, pressing Enter saves PLY/coverage PNG/JSON here with a timestamped filename.")
    ap.add_argument("--quit-key", type=str, default="",
                    help="Optional extra quit key (besides ESC). Leave empty to disable.")
    ap.add_argument("--debug-keys", action="store_true",
                    help="Print raw key codes received from the OpenCV window (for troubleshooting).")
    ap.add_argument("--force-focus", action="store_true",
                    help="Try to keep the OpenCV window on top (macOS focus quirk).")
    ap.add_argument("--cam-alpha", type=float, default=0.70,
                    help="Alpha of camera-visible overlay when composing BEV preview (0..1, default 0.70)")
    ap.add_argument("--global-gain", type=float, default=0.60,
                    help="Brightness gain for global PLY layer in BEV preview (e.g., 0.3~1.2, default 0.60)")
    args = ap.parse_args()
    origin_cv = None

    pts, cols = load_ply(args.ply)
    print_bounds("PLY(raw)", pts)

    # Frame conversion
    if args.ply_frame == "carla":
        pts = carla_world_points_to_cv(pts)
        print("[DEBUG] Converted PLY points CARLA→CV to match pose/H frame.")
        print_bounds("PLY(cv)", pts)

    # Optional origin shift
    if args.origin:
        ox, oy, oz = parse_vec3_csv(args.origin)
        origin = np.array([ox, oy, oz], dtype=np.float64)
        pts = pts - origin
        origin_cv = origin.copy()
        print(f"[DEBUG] Applied origin shift: -{origin}")
        print_bounds("PLY(after origin)", pts)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    H, W = img.shape[:2]

    if args.fov_y is not None:
        K = K_from_fov_hv(W, H, args.fov, args.fov_y)
    else:
        K = K_from_fov(W, H, args.fov)
    x,y,z = [float(v) for v in args.pos.split(",")]
    yaw,pitch,roll = [float(v) for v in args.rot.split(",")]

    Rt, R_wc, cam_pos_cv = pose_to_extrinsic(x,y,z,yaw,pitch,roll)

    # ---------------- Interactive live-tuning mode ----------------
    if args.interactive:
        # Use higher number of visible points for BEV preview, but subsample for rendering efficiency
        pts_live, cols_live = subsample_points(pts, cols, min(args.subsample * 2, len(pts)))

        step_p = float(args.step_pos)
        step_a = float(args.step_ang)
        fov = float(args.fov)

        # Prepare the OpenCV window and bring to front if requested
        win_name = "colorize/live"
        try:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        except Exception:
            cv2.namedWindow(win_name)
        # Set window to 1600x900 (16:9)
        cv2.resizeWindow(win_name, 1600, 900)
        cv2.moveWindow(win_name, 60, 60)
        if args.force_focus:
            try:
                cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass

        # --- start background renderer ---
        worker = RenderWorker(
            pts_live=pts_live,
            cols_live=cols_live,
            img_bgr=img,
            z_near=args.z_near,
            z_far=args.z_far,
            splat=args.splat,
            initial_pose=(x, y, z, yaw, pitch, roll),
            initial_fov=fov,
            bev_size=(1600, 780),
            pad=15
        )
        worker.start()
        # Set blending parameters from CLI
        worker.cam_alpha = float(np.clip(args.cam_alpha, 0.0, 1.0))
        worker.global_gain = float(np.clip(args.global_gain, 0.0, 1.5))

        try:
            show_bev = True
            last_frame = None
            last_update = 0.0

            while True:
                # Compose HUD text
                hud_text = "Move A/D X-+ W/S Y+- Q/E Z-+ | Rot J/L Yaw I/K Pitch U/O Roll | FOV Z/X | Step [/] | α -/= | Gain ,/. | Quit ESC"
                hud_text2 = format_pose(x, y, z, yaw, pitch, roll, fov) + f"  alpha={worker.cam_alpha:.2f}  gain={worker.global_gain:.2f}"

                # Push latest pose every loop for immediate responsiveness
                worker.update(x, y, z, yaw, pitch, roll, fov, hud_text2)

                # Get latest rendered frame (may be from previous pose; okay for UI)
                frame, seq = worker.get_frame()
                # Keep the last composed canvas to avoid flicker when no new frame
                if 'last_seq' not in locals():
                    last_seq = -1
                if 'last_canvas' not in locals():
                    last_canvas = np.zeros((900, 1600, 3), np.uint8)
                    cv2.putText(last_canvas, "[Initializing renderer...]", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

                if frame is not None and seq != last_seq:
                    canvas = last_canvas.copy()
                    canvas[0:780, 0:1600] = frame
                    last_canvas = canvas
                    last_seq = seq
                else:
                    canvas = last_canvas

                # HUD area at bottom 120px
                hud = np.zeros((120, 1600, 3), np.uint8)
                cv2.putText(hud, hud_text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(hud, hud_text2, (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
                canvas[780:900, 0:1600] = hud

                cv2.imshow(win_name, canvas)
                key_raw = cv2.waitKeyEx(1)
                key = key_raw & 0xFF if key_raw != -1 else -1
                if args.debug_keys and key_raw != -1:
                    print(f"[KEY] raw={key_raw} masked={key} chr={chr(key) if 32 <= key <= 126 else '?'}")

                # Quit conditions (ESC or custom)
                try:
                    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                        cv2.destroyWindow(win_name)
                        return
                except Exception:
                    pass
                # Quit: ESC only (and optional custom key); 'q' is now used for z-position
                if key == 27 or (args.quit_key and key == ord(args.quit_key)):
                    cv2.destroyWindow(win_name)
                    return

                # Position controls
                if key == ord('a'): x -= step_p
                if key == ord('d'): x += step_p
                if key == ord('w'): y += step_p
                if key == ord('s'): y -= step_p
                if key == ord('q'): z -= step_p  # Now z-position down, not quit
                if key == ord('e'): z += step_p

                # Rotation controls
                if key == ord('j'): yaw  -= step_a
                if key == ord('l'): yaw  += step_a
                if key == ord('i'): pitch += step_a
                if key == ord('k'): pitch -= step_a
                if key == ord('u'): roll += step_a
                if key == ord('o'): roll -= step_a

                # FOV & steps
                if key == ord('z'): fov = max(1.0, fov - 0.5)
                if key == ord('x'): fov = min(179.0, fov + 0.5)
                if key == ord('['):
                    step_p = max(0.01, step_p * 0.5); step_a = max(0.1, step_a * 0.5)
                if key == ord(']'):
                    step_p = min(5.0, step_p * 2.0); step_a = min(10.0, step_a * 2.0)

                # Overlay tuning
                if key == ord('-'):  # camera overlay alpha down
                    worker.adjust_cam_alpha(-0.05)
                if key == ord('=') or key == ord('+'):  # camera overlay alpha up
                    worker.adjust_cam_alpha(+0.05)
                if key == ord(','):  # global layer gain down
                    worker.adjust_global_gain(-0.05)
                if key == ord('.'):  # global layer gain up
                    worker.adjust_global_gain(+0.05)

                # Force immediate update after any key-adjusted change
                if key != -1:
                    worker.update(x, y, z, yaw, pitch, roll, fov, hud_text2)

                # Toggle BEV draw request (handled inside worker through hud text; here we simply force refresh)
                if key == ord('v'):
                    # Simply mark dirty so worker recomputes immediately
                    worker.update(x, y, z, yaw, pitch, roll, fov, hud_text2)

                # Save current precise result (full-resolution projection, not subsampled)
                if key == 13:  # Enter
                    from pathlib import Path
                    # 1) Decide save directory
                    save_dir = args.live_save_dir if args.live_save_dir else os.path.dirname(args.out)
                    os.makedirs(save_dir, exist_ok=True)

                    # 2) Build base stem from --out (e.g., ./out_visible/visible_cam9.ply → visible_cam9)
                    base_stem = Path(args.out).stem if args.out else "visible"

                    # 3) Compose filename that embeds current pose & FOV
                    #    Example: visible_cam9_x0.00_y0.00_z10.00_yaw55.00_pit-35.00_rol0.00_f88.80.ply
                    fname_stem = (
                        f"{base_stem}"
                        f"_x{float(x):.2f}_y{float(y):.2f}_z{float(z):.2f}"
                        f"_yaw{float(yaw):.2f}_pit{float(pitch):.2f}_rol{float(roll):.2f}"
                        f"_f{float(fov):.2f}"
                    )
                    ply_path = os.path.join(save_dir, f"{fname_stem}.ply")
                    png_path = os.path.join(save_dir, f"{fname_stem}_coverage.png")

                    # 4) Full-resolution visible points with the current intrinsics (respect fov_y if provided)
                    Rt_full, R_wc_full, _ = pose_to_extrinsic(x, y, z, yaw, pitch, roll)
                    if args.fov_y is not None:
                        K_full = K_from_fov_hv(W, H, fov, args.fov_y)
                    else:
                        K_full = K_from_fov(W, H, fov)
                    pts_vis_full, cols_vis_full = project_visible_points(
                        pts, img, K_full, Rt_full, args.z_near, args.z_far, args.splat
                    )

                    # 5) Frame transform selection for saving
                    T_CV_TO_CARLA = np.linalg.inv(_T_CARLA_WORLD_TO_CV_WORLD)
                    pts_out = pts_vis_full
                    if args.save_frame == "carla":
                        pts_out = (pts_out @ T_CV_TO_CARLA.T)

                    # 6) NaN/Inf guard and save PLY
                    if pts_out is None or len(pts_out) == 0:
                        print("[WARN] No points to save (pts_out empty). Skipping PLY save.")
                    else:
                        finite_mask = np.isfinite(pts_out).all(axis=1)
                        pts_out = pts_out[finite_mask]
                        if cols_vis_full is not None and len(cols_vis_full) == len(finite_mask):
                            cols_vis_full = cols_vis_full[finite_mask]
                        elif cols_vis_full is not None and len(cols_vis_full) != len(pts_out):
                            # fallback if lengths mismatch
                            cols_vis_full = np.ones((len(pts_out), 3), dtype=np.float64)
                        save_ply(pts_out, cols_vis_full, ply_path)

                    # 7) Save coverage PNG with current overlay settings
                    bev_img = make_bev_preview(
                        pts, cols, pts_vis_full, cols_vis_full,
                        bev_w=1600, bev_h=780, pad=15,
                        cam_alpha=worker.cam_alpha, global_gain=worker.global_gain
                    )
                    cv2.imwrite(png_path, bev_img)

                    print(f"[SAVE] PLY: {ply_path}")
                    print(f"[SAVE] PNG: {png_path}")
                    # 8) Also save per-pixel Pixel→World LUT NPZ (same schema as CARLA script)
                    # --- For NPZ: always save CARLA-frame world coordinates (to match multi-cam script) ---
                    pts_vis_map, cols_vis_map, Xmap_cv, Ymap_cv, Zmap_cv, hit_mask = project_visible_points_with_maps(
                        pts, img, K_full, Rt_full, args.z_near, args.z_far, args.splat
                    )
                    Xmap_out = Xmap_cv.copy()
                    Ymap_out = Ymap_cv.copy()
                    Zmap_out = Zmap_cv.copy()
                    T_CV_TO_CARLA = np.linalg.inv(_T_CARLA_WORLD_TO_CV_WORLD)

                    valid_pix = hit_mask
                    if np.any(valid_pix):
                        flat_cv = np.stack([Xmap_cv[valid_pix], Ymap_cv[valid_pix], Zmap_cv[valid_pix]], axis=1).astype(np.float64)
                        flat_car = (flat_cv @ T_CV_TO_CARLA.T)
                        Xmap_out[valid_pix] = flat_car[:, 0].astype(np.float32)
                        Ymap_out[valid_pix] = flat_car[:, 1].astype(np.float32)
                        Zmap_out[valid_pix] = flat_car[:, 2].astype(np.float32)

                    # Fill non-hit pixels with 0.0 to avoid NaNs (multi-cam NPZs are dense float arrays)
                    inv_pix = ~valid_pix
                    if np.any(inv_pix):
                        Xmap_out[inv_pix] = 0.0
                        Ymap_out[inv_pix] = 0.0
                        Zmap_out[inv_pix] = 0.0

                    # Build auxiliary fields (match multi-cam: valid_mask, floor_mask, ground_valid_mask, floor_ids)
                    valid_mask = hit_mask.astype(np.uint8)
                    floor_mask = np.zeros_like(valid_mask, dtype=np.uint8)          # no semantics here; zeros like multi-cam when IDs absent
                    ground_valid_mask = valid_mask.copy()
                    floor_ids_arr = np.array([7, 10], dtype=np.int32)               # match multi-cam default

                    # Intrinsics / pose (match multi-cam: cam_pose ordering [x,y,z,pitch,yaw,roll])
                    K_save = K_full.astype(np.float32)
                    cam_pose = np.array([x, y, z, pitch, yaw, roll], dtype=np.float32)

                    # Build CARLA-frame camera-to-world 4x4 (matches CARLA Transform matrix)
                    R_carla = rot_from_yaw_pitch_roll(yaw, pitch, roll)  # CARLA/UE convention
                    M_c2w_carla = np.eye(4, dtype=np.float64)
                    M_c2w_carla[:3, :3] = R_carla
                    M_c2w_carla[:3, 3]  = np.array([x, y, z], dtype=np.float64)

                    npz_path = os.path.join(save_dir, f"{fname_stem}_pixel2world_lut.npz")
                    np.savez_compressed(
                        npz_path,
                        X=Xmap_out.astype(np.float32),
                        Y=Ymap_out.astype(np.float32),
                        Z=Zmap_out.astype(np.float32),
                        valid_mask=valid_mask.astype(np.uint8),
                        floor_mask=floor_mask.astype(np.uint8),
                        ground_valid_mask=ground_valid_mask.astype(np.uint8),
                        K=K_full.astype(np.float32),
                        cam_pose=cam_pose.astype(np.float32),                            # [x,y,z,pitch,yaw,roll]
                        width=np.int32(W), height=np.int32(H), fov=np.float32(fov),
                        ray_model=np.array('forward', dtype='U'),                        # fixed to match multi-cam default
                        sem_channel=np.array('auto', dtype='U'),                         # fixed to match multi-cam default
                        floor_ids=floor_ids_arr.astype(np.int32),
                        M_c2w=M_c2w_carla.astype(np.float64)
                    )
                    print(f"[SAVE] NPZ: {npz_path}")
        finally:
            worker.stop()

    if args.debug_pos:
        print(f"[DEBUG] CARLA --pos: ({x:.3f}, {y:.3f}, {z:.3f}), --rot(y,p,r): ({yaw:.3f}, {pitch:.3f}, {roll:.3f})")
        print(f"[DEBUG] Derived cam_pos_cv: [{cam_pos_cv[0,0]:.3f}, {cam_pos_cv[1,0]:.3f}, {cam_pos_cv[2,0]:.3f}]")
        print("[DEBUG] Mapping: cam_pos_carla = [-Y, X, Z], cam_pos_cv = axes_carla_to_cv @ cam_pos_carla")
        print("[DEBUG] axes_carla_to_cv = [[0,1,0],[0,0,-1],[1,0,0]]; yaw' = yaw+90, pitch' = -pitch, roll' = roll")
        print("[DEBUG] Rt (world→cam):\n", Rt)

    # Camera position in WORLD frame (CARLA world): given directly by args.pos
    cam_pos_world = np.array([x, y, z], dtype=np.float64)

    # Optionally dump an axes PLY that shows global axes and camera-local axes
    if args.axes_out is not None and len(args.axes_out) > 0:
        ok_axes = save_axes_ply(out_path=args.axes_out,
                                scene_pts=pts,
                                cam_pos_world=np.array([x, y, z], dtype=np.float64),
                                R_world_to_cam=R_wc,
                                include_global=True,
                                include_camera=True,
                                xy_only=args.xy_only,
                                base_ply_path=None,
                                base_pts=pts,
                                base_cols=cols)
        if ok_axes:
            print(f"[OK] Saved axes visualization PLY: {args.axes_out}")
        else:
            print("[WARN] Axes PLY was not created (no content).")

    if args.dump_axes_merged is not None and len(args.dump_axes_merged) > 0:
        # produce an explicit merged axes+scene PLY for inspection
        ok_axes2 = save_axes_ply(out_path=args.dump_axes_merged,
                                 scene_pts=pts,
                                 cam_pos_world=np.array([x, y, z], dtype=np.float64),
                                 R_world_to_cam=R_wc,
                                 include_global=True,
                                 include_camera=True,
                                 xy_only=args.xy_only,
                                 base_ply_path=None,
                                 base_pts=pts,
                                 base_cols=cols)
        if ok_axes2:
            print(f"[OK] Saved merged (scene+axes) PLY: {args.dump_axes_merged}")

    print(f"[INFO] Global points: {len(pts)}")
    pts_vis, cols_vis = project_visible_points(pts, img, K, Rt, args.z_near, args.z_far, args.splat)
    print(f"[INFO] Visible points: {len(pts_vis)}")
    if len(pts_vis) > 0:
        min_xyz = pts_vis.min(axis=0)
        max_xyz = pts_vis.max(axis=0)
        center = pts_vis.mean(axis=0)
        bbox_info = {
            "camera_position": [x, y, z],
            "camera_rotation": [yaw, pitch, roll],
            "bbox_min": min_xyz.tolist(),
            "bbox_max": max_xyz.tolist(),
            "center": center.tolist(),
            "visible_point_count": int(len(pts_vis))
        }
        bbox_json = os.path.splitext(args.out)[0] + "_coverage.json"
        import json
        with open(bbox_json, "w") as f:
            json.dump(bbox_info, f, indent=2)
        print(f"[OK] Saved coverage metadata: {bbox_json}")

        # Also save a 2D BEV projection PNG that shows:
        # - the GLOBAL PLY footprint (dim gray)
        # - the camera-covered (visible) footprint (bright color) overlaid
        # Coordinate: top-down (CARLA X right, Y up) with Y inverted into image coordinates for visualization
        bev = make_bev_preview(pts, cols, pts_vis, cols_vis, bev_w=1200, bev_h=1200, pad=20,
                               cam_alpha=args.cam_alpha, global_gain=args.global_gain)
        bev_path = os.path.splitext(args.out)[0] + "_coverage.png"
        cv2.imwrite(bev_path, bev)
        print(f"[OK] Saved BEV coverage preview: {bev_path}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Prepare output frame transform and points
    pts_out = pts_vis  # placeholder, then override per frame selection
    T_CV_TO_CARLA = np.linalg.inv(_T_CARLA_WORLD_TO_CV_WORLD)
    if args.save_frame == "carla":
        pts_tmp = pts_vis + origin_cv if origin_cv is not None else pts_vis
        pts_out = (pts_tmp @ T_CV_TO_CARLA.T)
    else:
        pts_out = pts_vis if origin_cv is None else (pts_vis + origin_cv)
    print(f"[DEBUG] Saving visible PLY in '{args.save_frame}' frame.")
    print_bounds("VISIBLE(save bounds)", pts_out)
    save_ply(pts_out, cols_vis, args.out)
    print(f"[OK] Saved visible PLY: {args.out}")


if __name__ == "__main__":
    main()

"""
python img_to_ply_with_global_ply.py \
  --ply ./global_fused.ply \
  --image ./multi_out_1/cam_cam9/semantic_color.png \
  --out ./out_visible/visible_cam9.ply \
  --pos 30.0,2.0,10.0 \
  --rot=-55.0,-35.0,0.0 \
  --fov 89 \
  --interactive 
"""