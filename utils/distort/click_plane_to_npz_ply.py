#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Planar Image->World Projection (Homography-based), NPZ/PLY exporter

- Load one image
- Click N>=4 image points; after each click, type the corresponding world-plane (x y)
- Compute H (RANSAC / LMEDS / LSQ)
- Use H to map *every image pixel* to world plane (X,Y, Z=z_plane)
- Save:
  1) NPZ LUT with the same schema as img_to_ply_with_global_ply.py
  2) Reconstructed colored PLY from image pixels (optionally downsampled by --stride)

Notes:
- This script *does not* need camera intrinsics/extrinsics; it's purely point-to-point planar mapping.
- We keep npz keys/structure identical for downstream compatibility.
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path

# ----------------------------
# UI/help overlays
# ----------------------------
HELP = [
    "Controls:",
    " Left click : add image point then type world-plane (x y)",
    " u : undo last pair   r : reset all",
    " h : compute H (RANSAC)   m : compute H (LMEDS)   l : compute H (LSQ)",
    " s : save NPZ/PLY using current H",
    " q/ESC : quit",
]

def draw_help(img):
    y = 22
    for t in HELP:
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        y += 22

def draw_points(img, pts):
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (int(x), int(y)), 6, (0, 255, 255), -1)
        cv2.putText(img, f"{i+1}", (int(x)+8, int(y)-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, f"{i+1}", (int(x)+8, int(y)-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

def prompt_world_xy():
    while True:
        try:
            s = input("  → 실제 평면 좌표 입력 (x y): ").strip()
            x_str, y_str = s.split()
            return float(x_str), float(y_str)
        except Exception:
            print("  ⚠️ 형식: x y (예: 0 5)")

# ----------------------------
# H estimation & RMSE
# ----------------------------
def reproj_rmse(H, src_xy, dst_xy):
    """RMSE in dst-plane units (same unit as input world coords)."""
    if H is None or len(src_xy) == 0:
        return None
    src_h = np.hstack([src_xy, np.ones((src_xy.shape[0],1), dtype=np.float32)])  # Nx3
    proj_h = (H @ src_h.T).T
    w = proj_h[:, 2:3]
    proj = proj_h[:, :2] / np.clip(w, 1e-12, None)
    err = proj - dst_xy
    rmse = np.sqrt((err**2).sum(axis=1)).mean()
    return rmse

def compute_H(src_points, dst_points, method="RANSAC"):
    if len(src_points) < 4:
        print("[Warn] 최소 4쌍이 필요합니다.")
        return None, None
    src = np.asarray(src_points, dtype=np.float32)
    dst = np.asarray(dst_points, dtype=np.float32)

    method = method.upper()
    if method == "RANSAC":
        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC,
                                     ransacReprojThreshold=3.0,  # NOTE: unit = dst unit!
                                     maxIters=2000, confidence=0.999)
    elif method == "LMEDS":
        H, mask = cv2.findHomography(src, dst, method=cv2.LMEDS)
    else:
        H, mask = cv2.findHomography(src, dst, method=0)

    if H is None:
        print("[Error] Homography 추정 실패. 점 분포/대응을 확인하세요.")
        return None, None

    inliers = int(mask.sum()) if mask is not None else len(src_points)
    rmse_all = reproj_rmse(H, src, dst)
    print(f"\n[✅ {method}] H =\n{H}")
    print(f"[Info] 대응쌍: {len(src_points)}  인라이어: {inliers}")
    print(f"[RMSE] 전체: {rmse_all:.6f}")

    if mask is not None:
        in_mask = mask.ravel().astype(bool)
        rmse_in = reproj_rmse(H, src[in_mask], dst[in_mask])
        rmse_out = reproj_rmse(H, src[~in_mask], dst[~in_mask]) if (~in_mask).any() else 0.0
        print(f"[RMSE] 인라이어: {rmse_in:.6f} / 아웃라이어: {rmse_out:.6f} (아웃라이어 {int((~in_mask).sum())}개)")
    print()
    return H, mask

# ----------------------------
# NPZ & PLY writers
# ----------------------------
def save_npz_lut(out_npz_path, Xmap, Ymap, Zmap,
                 valid_mask, width, height,
                 z_plane,
                 # schema compatibility fields:
                 K=None, cam_pose=None, fov=0.0,
                 ray_model="forward", sem_channel="auto",
                 floor_ids=(7,10), M_c2w=None):
    """
    Save NPZ with *the same* schema as img_to_ply_with_global_ply.py
    """
    if K is None:
        # Keep a reasonable identity-like K; not used downstream if X/Y/Z maps are consumed directly.
        cx, cy = width/2.0, height/2.0
        K = np.array([[1.0, 0.0, cx],
                      [0.0, 1.0, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
    if cam_pose is None:
        # [x,y,z,pitch,yaw,roll] - unknown here; keep zeros
        cam_pose = np.zeros((6,), dtype=np.float32)
    if M_c2w is None:
        M_c2w = np.eye(4, dtype=np.float64)

    floor_mask = np.zeros_like(valid_mask, dtype=np.uint8)
    ground_valid_mask = valid_mask.astype(np.uint8)
    floor_ids_arr = np.array(floor_ids, dtype=np.int32)

    np.savez_compressed(
        out_npz_path,
        X=Xmap.astype(np.float32),
        Y=Ymap.astype(np.float32),
        Z=Zmap.astype(np.float32),
        valid_mask=valid_mask.astype(np.uint8),
        floor_mask=floor_mask.astype(np.uint8),
        ground_valid_mask=ground_valid_mask.astype(np.uint8),
        K=K.astype(np.float32),
        cam_pose=cam_pose.astype(np.float32),
        width=np.int32(width),
        height=np.int32(height),
        fov=np.float32(fov),
        ray_model=np.array(ray_model, dtype='U'),
        sem_channel=np.array(sem_channel, dtype='U'),
        floor_ids=floor_ids_arr.astype(np.int32),
        M_c2w=M_c2w.astype(np.float64)
    )
    print(f"[SAVE] NPZ: {out_npz_path}")

def write_ascii_ply(out_path, pts_xyz, cols_rgb01):
    """
    Minimal ASCII PLY writer (no external deps).
    pts_xyz: (N,3) float64/float32
    cols_rgb01: (N,3) float in [0,1]
    """
    N = int(pts_xyz.shape[0])
    cols_u8 = np.clip(cols_rgb01 * 255.0, 0, 255).astype(np.uint8)

    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]) + "\n"

    with open(out_path, "w") as f:
        f.write(header)
        for (x,y,z), (r,g,b) in zip(pts_xyz, cols_u8):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    print(f"[SAVE] PLY: {out_path}")

# ----------------------------
# Full image→world warping (per-pixel)
# ----------------------------
def build_maps_via_H(H, W, Ht, z_plane=0.0):
    """
    For every pixel (u,v), compute world-plane (X,Y,Z=z_plane) by H.
    Returns Xmap, Ymap, Zmap, valid_mask
    """
    # Build a full grid of pixel coords (u,v)
    us = np.arange(W, dtype=np.float32)
    vs = np.arange(Ht, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)        # shape (Ht, W)

    ones = np.ones_like(uu, dtype=np.float32)
    img_h = np.stack([uu, vv, ones], axis=-1)     # (Ht, W, 3)

    # Apply H
    Htens = H.astype(np.float64)
    proj = img_h @ Htens.T                          # (Ht, W, 3)
    w = proj[..., 2:3]
    # valid if w != 0 and finite
    valid = np.isfinite(proj).all(axis=-1) & np.isfinite(w[...,0]) & (np.abs(w[...,0]) > 1e-12)
    proj_xy = proj[..., :2] / np.clip(w, 1e-12, None)

    Xmap = np.full((Ht, W), np.float32(np.nan), dtype=np.float32)
    Ymap = np.full((Ht, W), np.float32(np.nan), dtype=np.float32)
    Zmap = np.full((Ht, W), np.float32(np.nan), dtype=np.float32)

    Xmap[valid] = proj_xy[..., 0][valid].astype(np.float32)
    Ymap[valid] = proj_xy[..., 1][valid].astype(np.float32)
    Zmap[valid] = np.float32(z_plane)

    valid_mask = valid.astype(np.uint8)
    return Xmap, Ymap, Zmap, valid_mask

def reconstruct_ply_from_image(img_bgr, Xmap, Ymap, Zmap, valid_mask, stride=1):
    """
    Create dense colored point cloud from per-pixel XY(Z) maps.
    stride>1 will downsample.
    Returns pts(N,3), cols(N,3 in 0..1, RGB)
    """
    Ht, W = valid_mask.shape
    ys, xs = np.nonzero(valid_mask)
    if stride > 1:
        mask_stride = ((ys % stride) == 0) & ((xs % stride) == 0)
        ys = ys[mask_stride]
        xs = xs[mask_stride]

    if ys.size == 0:
        return np.zeros((0,3), dtype=np.float64), np.zeros((0,3), dtype=np.float64)

    X = Xmap[ys, xs].astype(np.float64)
    Y = Ymap[ys, xs].astype(np.float64)
    Z = Zmap[ys, xs].astype(np.float64)
    pts = np.stack([X, Y, Z], axis=1)

    # BGR->RGB in [0,1]
    cols = img_bgr[ys, xs][:, ::-1] / 255.0
    return pts, cols

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser("Click-to-Plane Homography → NPZ/PLY exporter (no global PLY needed)")
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--z-plane", type=float, default=0.0, help="World plane Z (default: 0.0)")
    ap.add_argument("--method", choices=["RANSAC","LMEDS","LSQ"], default="RANSAC", help="Homography estimation method key")
    ap.add_argument("--stride", type=int, default=1, help="PLY downsample stride (>=1)")
    ap.add_argument("--prefix", type=str, default=None, help="Output file prefix (default: image stem)")
    args = ap.parse_args()

    img_path = args.image
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)
    Ht, W = img.shape[:2]

    prefix = args.prefix if args.prefix else Path(img_path).stem
    out_npz = os.path.join(out_dir, f"{prefix}_pixel2world_lut.npz")
    out_ply = os.path.join(out_dir, f"{prefix}_reconstructed.ply")

    # Interactive point picking
    window = "Click -> input world (x y) | h: RANSAC, m: LMEDS, l: LSQ, s: save, u/r undo/reset, q/ESC quit"
    src_points = []
    dst_points = []
    Hmat = None

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 800, 500)
    cv2.setWindowProperty(window, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    def on_mouse(event, x, y, flags, param):
        nonlocal src_points, dst_points
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"\n[{len(src_points)+1}] 이미지 좌표: ({x}, {y})")
            wx, wy = prompt_world_xy()
            src_points.append([float(x), float(y)])
            dst_points.append([wx, wy])
            print(f"  ✅ 등록: 이미지({x},{y}) → 세계({wx},{wy})")
    # cv2.resizeWindow(window, 500, 300)
    cv2.setMouseCallback(window, on_mouse)

    print("\n🖱️ 점을 4개 이상 클릭하세요. (각 클릭 후 터미널에 해당 점의 실제 평면 좌표 (x y) 입력)")
    print("   robust 추정을 원하면 6~12점 정도가 안정적입니다.\n")

    while True:
        vis = img.copy()
        draw_help(vis)
        draw_points(vis, src_points)
        cv2.imshow(window, vis)

        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')):  # ESC/q
            break
        elif k == ord('u'):
            if src_points:
                src_points.pop()
                dst_points.pop()
                print("↩️  마지막 한 쌍 제거")
        elif k == ord('r'):
            src_points.clear()
            dst_points.clear()
            Hmat = None
            print("🧹 초기화 완료")
        elif k == ord('h'):
            Hmat, _ = compute_H(src_points, dst_points, method="RANSAC")
        elif k == ord('m'):
            Hmat, _ = compute_H(src_points, dst_points, method="LMEDS")
        elif k == ord('l'):
            Hmat, _ = compute_H(src_points, dst_points, method="LSQ")
        elif k == ord('s'):
            if Hmat is None:
                print("⚠️ 먼저 H를 계산하세요 (h/m/l).")
                continue

            # 1) Per-pixel XY(Z) maps via H
            Xmap, Ymap, Zmap, valid_mask = build_maps_via_H(Hmat, W, Ht, z_plane=args.z_plane)

            # Fill NaNs for non-valid with zeros to match multi-cam dense array style
            inv = (valid_mask == 0)
            if np.any(inv):
                Xmap[inv] = 0.0
                Ymap[inv] = 0.0
                Zmap[inv] = 0.0

            # 2) Save NPZ with the SAME schema keys as existing pipeline
            #    K/cam_pose/M_c2w are placeholders for schema compatibility.
            cx, cy = W/2.0, Ht/2.0
            K = np.array([[1.0, 0.0, cx],
                          [0.0, 1.0, cy],
                          [0.0, 0.0, 1.0]], dtype=np.float32)
            cam_pose = np.zeros((6,), dtype=np.float32)  # [x,y,z,pitch,yaw,roll] not used here
            M_c2w = np.eye(4, dtype=np.float64)
            save_npz_lut(out_npz, Xmap, Ymap, Zmap,
                         valid_mask, W, Ht,
                         args.z_plane,
                         K=K, cam_pose=cam_pose, fov=0.0,
                         ray_model="forward", sem_channel="auto",
                         floor_ids=(7,10), M_c2w=M_c2w)

            # 3) Reconstruct dense PLY from image pixels (optionally strided)
            pts, cols = reconstruct_ply_from_image(img, Xmap, Ymap, Zmap, valid_mask, stride=max(1, int(args.stride)))
            write_ascii_ply(out_ply, pts, cols)

            # 4) Save a tiny metadata JSON (optional)
            meta = {
                "image": os.path.abspath(img_path),
                "H": Hmat.tolist(),
                "width": int(W),
                "height": int(Ht),
                "z_plane": float(args.z_plane),
                "stride": int(args.stride),
                "pairs": {
                    "image_xy": np.asarray(src_points, dtype=float).tolist(),
                    "world_xy": np.asarray(dst_points, dtype=float).tolist()
                }
            }
            with open(os.path.join(out_dir, f"{prefix}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[SAVE] META: {os.path.join(out_dir, f'{prefix}_meta.json')}")

            print("\n[OK] 저장 완료. 계속 편집하려면 더 찍고(h/m/l) 다시 's' 누르세요.\n")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()