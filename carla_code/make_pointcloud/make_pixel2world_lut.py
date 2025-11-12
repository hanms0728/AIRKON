#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA 0.9.16 — Pixel→World LUT generator (UE4 coord, floor filter, heatmaps & contour maps)

Outputs in --out-dir:
  - pixel2world_lut.npz        : per-pixel world coords (X,Y,Z), valid_mask, floor_mask, ground_valid_mask, intrinsics/extrinsics
  - pixel2world_sample.csv     : (optional via --sample-step N)
  - rgb_image.png              : RGB snapshot
  - semantic_raw.png           : raw class-id (single channel vis)
  - semantic_color.png         : colorized semantic (simple palette)
  - semantic_histogram.txt     : histogram of semantic IDs in the frame
  - depth_gray.png             : depth (m) → [0..255] grayscale (clipped to --min-depth..--max-depth)
  - floor_mask.png             : binary mask of floor-like classes (white=floor)
  - floor_on_rgb.png           : floor mask overlay on RGB
  - depth_on_rgb.png           : depth grayscale overlay on RGB
  - X_map.png / Y_map.png / Z_map.png : world 좌표 히트맵(자동 대비 스트레치, 바닥 위주)
  - X_map_legend.txt / Y_map_legend.txt / Z_map_legend.txt : 히트맵 범위/통계
  - X_contours.png / Y_contours.png / Z_contours.png : 가독성 높은 등고선 지도(옵션, 라벨 값 출력)

Notes:
- UE4/CARLA camera frame: X=forward, Y=right, Z=up (left-handed).
- Depth sensor encodes Euclidean distance D (meters) from camera center to first hit surface.
- 3D reconstruction:
    normalized : P_cam = D * normalize([1, (u-cx)/f, -(v-cy)/f])  (정확)
    forward    : X=D, Y=((u-cx)/f)*D, Z=(-(v-cy)/f)*D             (근사)
- NumPy 2.0 safe: strings saved as np.array(..., dtype='U').
"""

import argparse
import math
import os
import time
import numpy as np
from PIL import Image
import carla
from contextlib import contextmanager

# ----- Matplotlib (등고선/라벨) -----
import matplotlib
matplotlib.use("Agg")  # GUI 없는 서버에서도 저장 가능
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe


# ---------------- Utilities ----------------

def depth_raw_to_meters(image_bgra: np.ndarray) -> np.ndarray:
    """CARLA depth BGRA (uint8) -> meters (float32, 0..1000)."""
    B = image_bgra[:, :, 0].astype(np.uint32)
    G = image_bgra[:, :, 1].astype(np.uint32)
    R = image_bgra[:, :, 2].astype(np.uint32)
    denom = (256 ** 3 - 1)
    depth_norm = (R + (G << 8) + (B << 16)).astype(np.float64) / denom
    return (1000.0 * depth_norm).astype(np.float32)


def build_intrinsics(width: int, height: int, fov_deg: float):
    """Return K, f(px), cx, cy for horizontal FOV."""
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)
    return K, f, cx, cy


def colorize_semantic(sem: np.ndarray) -> np.ndarray:
    """아주 간단한 팔레트(필요시 확장)."""
    H, W = sem.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[:, :, :] = (80, 80, 80)  # default gray

    def paint(ids, color):
        mask = np.isin(sem, ids)
        out[mask] = color

    paint([7], (50, 150, 50))      # Road
    paint([10], (255, 255, 0))     # LaneMarking
    paint([8], (244, 35, 232))     # Sidewalk
    paint([12], (70, 70, 70))      # Building
    paint([13], (190, 153, 153))   # Fence
    paint([14], (153, 153, 153))   # Pole
    paint([18], (0, 0, 142))       # Vehicle
    paint([19], (220, 20, 60))     # Pedestrian
    paint([5,6,9,11,15,16,17,20,21,22], (100,100,100))  # Others
    return out


def overlay_mask_on_rgb(rgb: np.ndarray, mask: np.ndarray, alpha=0.5, color=(0,255,0)) -> np.ndarray:
    """Overlay binary mask on RGB image."""
    ov = rgb.copy().astype(np.float32)
    m = (mask > 0)[..., None].astype(np.float32)
    color_arr = np.array(color, dtype=np.float32).reshape(1,1,3)
    ov = ov * (1 - alpha*m) + color_arr * (alpha*m)
    return np.clip(ov, 0, 255).astype(np.uint8)


def overlay_gray_on_rgb(rgb: np.ndarray, gray_u8: np.ndarray, alpha=0.5) -> np.ndarray:
    """Overlay grayscale (uint8) onto RGB."""
    gray3 = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    rgbf = rgb.astype(np.float32)
    grayf = gray3.astype(np.float32)
    out = rgbf*(1-alpha) + grayf*alpha
    return np.clip(out, 0, 255).astype(np.uint8)


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


def pick_sem_channel(arr_bgra: np.ndarray, mode: str) -> np.ndarray:
    """
    Returns HxW uint8 of class ids.
    mode: 'auto'|'r'|'g'|'b'
    많은 CARLA 빌드에서 클래스 ID는 R채널(arr[:,:,2])에 있음.
    """
    if mode == 'r':
        sem = arr_bgra[:, :, 2]
    elif mode == 'g':
        sem = arr_bgra[:, :, 1]
    elif mode == 'b':
        sem = arr_bgra[:, :, 0]
    else:
        candidates = [arr_bgra[:, :, 2], arr_bgra[:, :, 1], arr_bgra[:, :, 0]]
        uniq = [len(np.unique(c)) for c in candidates]
        sem = candidates[int(np.argmax(uniq))]
    return sem.astype(np.uint8)


# --------- Heatmap utils (auto contrast + percentile clip + custom colormap) ---------

def _linear_colormap01(x01: np.ndarray) -> np.ndarray:
    """
    [0,1] → RGB(0..255). Blue→Cyan→Yellow→Red piecewise 컬러맵.
    """
    x = np.clip(x01, 0.0, 1.0)
    r = np.zeros_like(x)
    g = np.zeros_like(x)
    b = np.zeros_like(x)
    # 0.00~0.33: Blue(0,0,1) → Cyan(0,1,1)
    m = (x <= 1/3)
    t = np.zeros_like(x); t[m] = x[m] * 3.0
    r[m] = 0.0
    g[m] = t[m]
    b[m] = 1.0
    # 0.33~0.66: Cyan(0,1,1) → Yellow(1,1,0)
    m = (x > 1/3) & (x <= 2/3)
    t[m] = (x[m] - 1/3) * 3.0
    r[m] = t[m]
    g[m] = 1.0
    b[m] = 1.0 - t[m]
    # 0.66~1.00: Yellow(1,1,0) → Red(1,0,0)
    m = (x > 2/3)
    t[m] = (x[m] - 2/3) * 3.0
    r[m] = 1.0
    g[m] = 1.0 - t[m]
    b[m] = 0.0
    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def save_value_heatmap_png(
    out_path_png: str,
    values: np.ndarray,          # HxW float
    valid_mask: np.ndarray,      # HxW bool/uint8
    use_ground_only: bool = True,
    ground_mask: np.ndarray | None = None,
    clip_pct: float = 1.0,       # e.g., 1.0 → [1,99] percentile
    legend_txt_path: str | None = None,
    title: str = ""
):
    """
    값 분포의 (clip_pct, 100-clip_pct) 퍼센타일로 자동 스트레치해서 컬러 히트맵 저장.
    - use_ground_only=True → ground_mask ∧ valid_mask 픽셀만으로 범위 추정(추천)
    - 유효 픽셀 부족하면 전체 valid로 대체
    """
    H, W = values.shape
    vm = (valid_mask.astype(bool))
    if use_ground_only and (ground_mask is not None):
        vm = vm & ground_mask.astype(bool)

    vals = values[vm]
    vals = vals[np.isfinite(vals)]
    if vals.size < 50:
        vm = (valid_mask.astype(bool))
        vals = values[vm]
        vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        img = np.full((H, W, 3), 128, np.uint8)
        Image.fromarray(img).save(out_path_png)
        if legend_txt_path:
            with open(legend_txt_path, "w") as f:
                f.write(f"{title}\nNO VALID PIXELS\n")
        return

    lo = np.percentile(vals, clip_pct)
    hi = np.percentile(vals, 100.0 - clip_pct)
    if not np.isfinite(lo): lo = np.nanmin(vals)
    if not np.isfinite(hi): hi = np.nanmax(vals)
    if (hi - lo) < 1e-6:
        med = float(np.median(vals))
        lo, hi = med - 0.05, med + 0.05  # ±5cm

    x01 = (values - lo) / max(hi - lo, 1e-9)
    x01[~vm] = np.nan

    rgb = _linear_colormap01(np.nan_to_num(x01, nan=0.0))
    invalid = ~vm
    if invalid.any():
        rgb[invalid] = np.array([30, 30, 30], dtype=np.uint8)

    Image.fromarray(rgb).save(out_path_png)

    if legend_txt_path:
        with open(legend_txt_path, "w") as f:
            f.write(f"{title}\n")
            f.write(f"clip_pct={clip_pct:.2f}  range=[{lo:.6f}, {hi:.6f}]\n")
            f.write(f"valid_pixels={int(vm.sum())}/{H*W}\n")


# -------- 등고선 시각화(다크 배경 + 라벨 외곽선) --------

def save_contour_map_png(
    out_path_png: str,
    values: np.ndarray,             # HxW float
    valid_mask: np.ndarray,         # HxW bool/uint8
    ground_mask: np.ndarray | None, # 선택
    vmin: float | None = None,
    vmax: float | None = None,
    step: float = 1.0,
    draw_labels: bool = True,
    bg: str = "dark",               # "dark" | "light"
    overlay_heatmap: bool = False,  # 등고선 아래 히트맵
    clip_pct: float = 1.0,
    title: str = "",
):
    H, W = values.shape
    vm = valid_mask.astype(bool)
    if ground_mask is not None:
        vm = vm & ground_mask.astype(bool)

    vals = values[vm]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        bg_color = (0.12,0.12,0.12) if bg=="dark" else (1,1,1)
        fig, ax = plt.subplots(figsize=(W/200, H/200), dpi=200)
        ax.set_facecolor(bg_color); ax.axis("off")
        fig.savefig(out_path_png, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return

    # vmin/vmax 결정
    if vmin is None or vmax is None:
        vmin_auto = np.percentile(vals, clip_pct)
        vmax_auto = np.percentile(vals, 100.0 - clip_pct)
        if (vmax_auto - vmin_auto) < 1e-6:
            med = float(np.median(vals))
            vmin_auto, vmax_auto = med - 0.05, med + 0.05
        if vmin is None: vmin = vmin_auto
        if vmax is None: vmax = vmax_auto
    if vmin > vmax:
        vmin, vmax = vmax, vmin

    eps = 1e-9
    n_levels = max(2, int(np.floor((vmax - vmin) / max(step, eps))) + 1)
    levels = np.linspace(vmin, vmin + (n_levels-1)*step, n_levels)

    arr = np.full_like(values, np.nan, dtype=float)
    arr[vm] = values[vm]

    fig_w, fig_h, dpi = W/200, H/200, 200
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    bg_color = (0.12,0.12,0.12) if bg=="dark" else (1,1,1)
    ax.set_facecolor(bg_color)
    ax.set_aspect("equal")
    ax.axis("off")

    if overlay_heatmap:
        hm = ax.imshow(arr, interpolation="nearest", vmin=vmin, vmax=vmax, cmap="plasma", alpha=0.85)
        hm.set_zorder(0)

    line_color = "w" if bg=="dark" else "k"
    cs = ax.contour(arr, levels=levels, colors=line_color, linewidths=1.2, antialiased=True)
    cs.set_zorder(1)

    if draw_labels:
        fmt = mticker.FormatStrFormatter("%.2f")
        cl = ax.clabel(cs, inline=True, fmt=fmt, fontsize=8, inline_spacing=3)
        text_color = "black" if bg=="light" else "white"
        outline = "white" if text_color=="black" else "black"
        for txt in cl:
            txt.set_color(text_color)
            txt.set_path_effects([
                pe.Stroke(linewidth=2.2, foreground=outline, alpha=0.95),
                pe.Normal()
            ])

    fig.savefig(out_path_png, bbox_inches="tight", pad_inches=0, facecolor=bg_color)
    plt.close(fig)


# ---------------- Main generator ----------------

def generate_pixel2world_lut(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    W, H, FOV = args.width, args.height, args.fov
    os.makedirs(args.out_dir, exist_ok=True)

    # Blueprints
    depth_bp = bp_lib.find('sensor.camera.depth')
    rgb_bp   = bp_lib.find('sensor.camera.rgb')
    sem_bp   = bp_lib.find('sensor.camera.semantic_segmentation')

    for bp in (depth_bp, rgb_bp, sem_bp):
        bp.set_attribute('image_size_x', str(W))
        bp.set_attribute('image_size_y', str(H))
        bp.set_attribute('fov',          str(FOV))

    # Camera pose (world frame)
    cam_tf = carla.Transform(
        carla.Location(args.cam_x, args.cam_y, args.cam_z),
        carla.Rotation(pitch=args.pitch, yaw=args.yaw, roll=args.roll)
    )

    # Buffers
    depth_buf = {"arr": None}
    rgb_buf   = {"arr": None}
    sem_buf   = {"arr": None}

    def on_depth(img: carla.Image):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(H, W, 4)
        depth_buf["arr"] = arr

    def on_rgb(img: carla.Image):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(H, W, 4)
        rgb = arr[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        rgb_buf["arr"] = rgb

    def on_sem(img: carla.Image):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(H, W, 4)
        sem_buf["arr"] = arr

    sensor_depth = sensor_rgb = sensor_sem = None

    # Intrinsics
    K, f, cx, cy = build_intrinsics(W, H, FOV)

    try:
        with synchronous_mode(world, fps=args.fps):
            sensor_depth = world.spawn_actor(depth_bp, cam_tf)
            sensor_rgb   = world.spawn_actor(rgb_bp,   cam_tf)
            sensor_sem   = world.spawn_actor(sem_bp,   cam_tf)

            sensor_depth.listen(on_depth)
            sensor_rgb.listen(on_rgb)
            sensor_sem.listen(on_sem)

            # Warm-up ticks
            for _ in range(4):
                world.tick()

            # Wait frames
            deadline = time.time() + 6.0
            while (
                depth_buf["arr"] is None or
                rgb_buf["arr"]   is None or
                sem_buf["arr"]   is None
            ) and time.time() < deadline:
                world.tick()

            if depth_buf["arr"] is None or rgb_buf["arr"] is None or sem_buf["arr"] is None:
                raise RuntimeError("Timeout: did not receive depth/RGB/semantic frames.")

            # Save RGB snapshot
            rgb_path = os.path.join(args.out_dir, "rgb_image.png")
            Image.fromarray(rgb_buf["arr"]).save(rgb_path)
            print(f"[OK] Saved RGB: {rgb_path}")

            # Semantic class ids
            sem_raw_bgra = sem_buf["arr"]
            sem_id = pick_sem_channel(sem_raw_bgra, args.sem_channel)  # HxW uint8
            Image.fromarray(sem_id).save(os.path.join(args.out_dir, "semantic_raw.png"))
            sem_color = colorize_semantic(sem_id)
            Image.fromarray(sem_color).save(os.path.join(args.out_dir, "semantic_color.png"))
            print("[OK] Saved semantic visualizations")

            # Semantic histogram
            vals, counts = np.unique(sem_id, return_counts=True)
            order = np.argsort(counts)[::-1]
            hist_path = os.path.join(args.out_dir, "semantic_histogram.txt")
            with open(hist_path, "w") as hf:
                hf.write("# class_id,count\n")
                for vi, ci in zip(vals[order], counts[order]):
                    hf.write(f"{int(vi)},{int(ci)}\n")
            print(f"[OK] Saved semantic histogram: {hist_path}")

            # Depth (meters) and grayscale vis
            depth_m = depth_raw_to_meters(depth_buf["arr"])  # HxW float32
            depth_gray = np.clip((depth_m - args.min_depth)/(args.max_depth-args.min_depth+1e-9), 0, 1)
            depth_gray_u8 = (depth_gray*255).astype(np.uint8)
            Image.fromarray(depth_gray_u8).save(os.path.join(args.out_dir, "depth_gray.png"))
            print("[OK] Saved depth grayscale")

            # ---------------- 3D reconstruction ----------------
            u = np.arange(W, dtype=np.float64)
            v = np.arange(H, dtype=np.float64)
            uu, vv = np.meshgrid(u, v)
            D = depth_m.astype(np.float64)

            if args.ray_model == "normalized":
                dir_x = np.ones_like(uu, dtype=np.float64)      # X forward
                dir_y = (uu - cx) / f                           # Y right
                dir_z = -(vv - cy) / f                          # Z up
                norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                ux, uy, uz = dir_x/norm, dir_y/norm, dir_z/norm
                Xc, Yc, Zc = D*ux, D*uy, D*uz
            else:  # "forward" (approx)
                Xc = D
                Yc = ((uu - cx) / f) * D
                Zc = (-(vv - cy) / f) * D

            # Valid by depth range
            valid = (D > args.min_depth) & (D < args.max_depth) & np.isfinite(D)

            # Camera→World
            M_c2w = np.array(sensor_depth.get_transform().get_matrix(), dtype=np.float64)  # 4x4
            R = M_c2w[:3, :3]
            t = M_c2w[:3, 3]
            pts_c = np.stack([Xc, Yc, Zc], axis=0).reshape(3, -1)
            pts_w = (R @ pts_c) + t.reshape(3, 1)
            Xw = pts_w[0, :].reshape(H, W).astype(np.float32)
            Yw = pts_w[1, :].reshape(H, W).astype(np.float32)
            Zw = pts_w[2, :].reshape(H, W).astype(np.float32)

            # ---------------- Floor-like mask (Road/Lane/... user-configurable) ----------------
            floor_ids = set(int(x) for x in args.floor_ids.split(",")) if args.floor_ids else {7, 10}
            floor_mask = np.isin(sem_id, list(floor_ids))
            Image.fromarray((floor_mask*255).astype(np.uint8)).save(os.path.join(args.out_dir, "floor_mask.png"))

            # Overlays
            rgb = rgb_buf["arr"]
            floor_on_rgb = overlay_mask_on_rgb(rgb, floor_mask.astype(np.uint8), alpha=0.45, color=(0,255,0))
            Image.fromarray(floor_on_rgb).save(os.path.join(args.out_dir, "floor_on_rgb.png"))

            depth_on_rgb = overlay_gray_on_rgb(rgb, depth_gray_u8, alpha=0.45)
            Image.fromarray(depth_on_rgb).save(os.path.join(args.out_dir, "depth_on_rgb.png"))

            # ---------------- Ground (optional Z heuristic) ----------------
            ground_valid_mask = valid & floor_mask
            if args.ground_z_tol > 0:
                Zw_floor = Zw[ground_valid_mask]
                if Zw_floor.size > 100:
                    Z0 = float(np.median(Zw_floor))
                    tol = float(args.ground_z_tol)
                    ground_valid_mask = ground_valid_mask & (np.abs(Zw - Z0) < tol)
                    print(f"[INFO] Ground Z median={Z0:.3f} m, tol={tol:.3f} m (filtered)")

            # ---------------- Save NPZ LUT (NumPy 2.0-safe strings) ----------------
            ground_valid_mask_u8 = ground_valid_mask.astype(np.uint8)
            npz_path = os.path.join(args.out_dir, "pixel2world_lut.npz")
            np.savez_compressed(
                npz_path,
                X=Xw, Y=Yw, Z=Zw,
                valid_mask=valid.astype(np.uint8),
                floor_mask=floor_mask.astype(np.uint8),
                ground_valid_mask=ground_valid_mask_u8,
                K=K.astype(np.float32),
                cam_pose=np.array([args.cam_x, args.cam_y, args.cam_z,
                                   args.pitch, args.yaw, args.roll], dtype=np.float32),
                width=np.int32(W), height=np.int32(H), fov=np.float32(FOV),
                ray_model=np.array(args.ray_model, dtype='U'),
                sem_channel=np.array(args.sem_channel, dtype='U'),
                floor_ids=np.array(sorted(list(floor_ids)), dtype=np.int32)
            )
            print(f"[OK] Saved LUT: {npz_path}")

            # ---------------- Heatmaps (auto contrast) ----------------
            save_value_heatmap_png(
                out_path_png=os.path.join(args.out_dir, "X_map.png"),
                values=Xw, valid_mask=valid,
                use_ground_only=True, ground_mask=ground_valid_mask_u8,
                clip_pct=1.0, legend_txt_path=os.path.join(args.out_dir, "X_map_legend.txt"),
                title="X (world forward) [m]"
            )
            save_value_heatmap_png(
                out_path_png=os.path.join(args.out_dir, "Y_map.png"),
                values=Yw, valid_mask=valid,
                use_ground_only=True, ground_mask=ground_valid_mask_u8,
                clip_pct=1.0, legend_txt_path=os.path.join(args.out_dir, "Y_map_legend.txt"),
                title="Y (world right) [m]"
            )
            save_value_heatmap_png(
                out_path_png=os.path.join(args.out_dir, "Z_map.png"),
                values=Zw, valid_mask=valid,
                use_ground_only=True, ground_mask=ground_valid_mask_u8,
                clip_pct=1.0, legend_txt_path=os.path.join(args.out_dir, "Z_map_legend.txt"),
                title="Z (world up) [m]"
            )
            print("[OK] Saved X/Y/Z heatmaps with auto contrast")

            # -------- 등고선 저장 (축별 간격 별도 설정) --------
            if args.contours:
                def _parse_range(s):
                    if s is None: return (None, None)
                    a,b = s.split(",")
                    return float(a), float(b)

                xvmin, xvmax = _parse_range(args.x_range)
                yvmin, yvmax = _parse_range(args.y_range)
                zvmin, zvmax = _parse_range(args.z_range)

                step_x = args.contour_step_x if args.contour_step_x is not None else args.contour_step
                step_y = args.contour_step_y if args.contour_step_y is not None else args.contour_step
                step_z = args.contour_step_z if args.contour_step_z is not None else args.contour_step

                save_contour_map_png(
                    out_path_png=os.path.join(args.out_dir, "X_contours.png"),
                    values=Xw, valid_mask=valid,
                    ground_mask=ground_valid_mask_u8,
                    vmin=xvmin, vmax=xvmax,
                    step=step_x,
                    draw_labels=bool(args.label_contours),
                    bg=args.contour_bg,
                    overlay_heatmap=bool(args.contour_overlay_heatmap),
                    clip_pct=1.0,
                    title="X (world forward) [m]"
                )
                save_contour_map_png(
                    out_path_png=os.path.join(args.out_dir, "Y_contours.png"),
                    values=Yw, valid_mask=valid,
                    ground_mask=ground_valid_mask_u8,
                    vmin=yvmin, vmax=yvmax,
                    step=step_y,
                    draw_labels=bool(args.label_contours),
                    bg=args.contour_bg,
                    overlay_heatmap=bool(args.contour_overlay_heatmap),
                    clip_pct=1.0,
                    title="Y (world right) [m]"
                )
                save_contour_map_png(
                    out_path_png=os.path.join(args.out_dir, "Z_contours.png"),
                    values=Zw, valid_mask=valid,
                    ground_mask=ground_valid_mask_u8,
                    vmin=zvmin, vmax=zvmax,
                    step=step_z,
                    draw_labels=bool(args.label_contours),
                    bg=args.contour_bg,
                    overlay_heatmap=bool(args.contour_overlay_heatmap),
                    clip_pct=1.0,
                    title="Z (world up) [m]"
                )
                print("[OK] Saved X/Y/Z contour maps")

            # ---------------- Optional sparse CSV sample ----------------
            if args.sample_step > 0:
                csv_path = os.path.join(args.out_dir, "pixel2world_sample.csv")
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("u,v,X,Y,Z,valid,floor,ground\n")
                    step = args.sample_step
                    for vv_i in range(0, H, step):
                        for uu_i in range(0, W, step):
                            f.write(f"{uu_i},{vv_i},"
                                    f"{Xw[vv_i,uu_i]:.6f},{Yw[vv_i,uu_i]:.6f},{Zw[vv_i,uu_i]:.6f},"
                                    f"{int(valid[vv_i,uu_i])},{int(floor_mask[vv_i,uu_i])},{int(ground_valid_mask_u8[vv_i,uu_i])}\n")
                print(f"[OK] Saved CSV sample: {csv_path}")

    finally:
        for s in (sensor_depth, sensor_rgb, sensor_sem):
            if s is not None:
                s.stop()
                s.destroy()


def main():
    p = argparse.ArgumentParser(
        description="Generate Pixel→World LUT (CARLA 0.9.16, UE4 coord) with floor filter & rich visualizations."
    )
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fov", type=float, default=90.0, help="Horizontal FOV (deg)")

    # Pose
    p.add_argument("--cam-x", type=float, default=10.0)
    p.add_argument("--cam-y", type=float, default=0.0)
    p.add_argument("--cam-z", type=float, default=8.0)
    p.add_argument("--pitch", type=float, default=-30.0)
    p.add_argument("--yaw",   type=float, default=0.0)
    p.add_argument("--roll",  type=float, default=0.0)

    # Depth & reconstruction
    p.add_argument("--min-depth", type=float, default=0.05)
    p.add_argument("--max-depth", type=float, default=200.0)
    p.add_argument("--ray-model", choices=["normalized","forward"], default="normalized",
                   help="'normalized' = D * normalize([1,dx,dy]) (accurate), 'forward' = X=D approx.")
    p.add_argument("--fps", type=int, default=20)

    # Semantic options
    p.add_argument("--sem-channel", choices=["auto","r","g","b"], default="auto",
                   help="Which channel contains class ids in raw semantic BGRA.")
    p.add_argument("--floor-ids", type=str, default="7,10",
                   help="Comma-separated class IDs considered as floor-like (e.g., '7,10,8,21,6').")

    # Ground Z heuristic
    p.add_argument("--ground-z-tol", type=float, default=0.0,
                   help="If >0, keep floor pixels with |Z - median(Z_floor)| < tol (meters).")

    # IO
    p.add_argument("--out-dir", type=str, default="./")
    p.add_argument("--sample-step", type=int, default=0,
                   help="If >0, dump sparse CSV sampling every N pixels.")

    # Contour options
    p.add_argument("--contours", type=int, default=0, help="1이면 X/Y/Z 등고선 png 저장")
    p.add_argument("--contour-step", type=float, default=1.0, help="기본 등고선 간격 (미지정 축에 공통 적용)")
    p.add_argument("--contour-step-x", type=float, default=None, help="X 전용 등고선 간격")
    p.add_argument("--contour-step-y", type=float, default=None, help="Y 전용 등고선 간격")
    p.add_argument("--contour-step-z", type=float, default=None, help="Z 전용 등고선 간격")
    p.add_argument("--label-contours", type=int, default=1, help="등고선 라벨 표시(값 숫자)")
    p.add_argument("--contour-bg", choices=["dark","light"], default="dark", help="등고선 배경")
    p.add_argument("--contour-overlay-heatmap", type=int, default=0, help="등고선 아래 히트맵 함께 표시")

    # 값 범위를 강제로 지정(절대좌표 확인용)
    p.add_argument("--x-range", type=str, default=None, help="예: '-70,70'")
    p.add_argument("--y-range", type=str, default=None, help="예: '-70,20'")
    p.add_argument("--z-range", type=str, default=None, help="예: '-1,5'")

    args = p.parse_args()
    generate_pixel2world_lut(args)


if __name__ == "__main__":
    main()

"""
python make_pixel2world_lut.py \
  --width 1920 --height 1080 --fov 89 \
  --cam-x 30 --cam-y 2 --cam-z 10 \
  --pitch -35 --yaw -55 --roll 0.0 \
  --out-dir ./lut_out --sample-step 40 \
  --floor-ids 1,2,10,24 \
  --ray-model forward \
  --contours 1 --label-contours 1 --contour-bg dark --contour-overlay-heatmap 1 \
  --contour-step 2.0 \
  --contour-step-z 0.05 \
  --x-range='-70,70' --y-range='-70,20' --z-range='-1,5'
"""