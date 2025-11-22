#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import open3d as o3d
import argparse


# ----------------------
# PLY LOAD (XY + COLOR)
# ----------------------
def load_ply_xy_with_color(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    xy = pts[:, :2]
    return pts, xy, cols


# ----------------------
# DRAW XY CANVAS (비율 유지)
# ----------------------
def draw_ply_xy_colored(xy, cols, base=1300):
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)

    x_range = xmax - xmin
    y_range = ymax - ymin

    ratio = x_range / y_range

    W = base
    H = int(W / ratio)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    X = xy[:, 0]
    Y = xy[:, 1]

    u = ((X - xmin) / x_range) * (W - 1)
    v = ((ymax - Y) / y_range) * (H - 1)

    u = u.astype(np.int32)
    v = v.astype(np.int32)

    cols255 = (cols * 255).astype(np.uint8)

    for i in range(len(u)):
        if 0 <= u[i] < W and 0 <= v[i] < H:
            canvas[v[i], u[i]] = cols255[i]

    return canvas, xmin, xmax, ymin, ymax, W, H


# ----------------------
# APPLY COLOR TO PLY
# ----------------------
def apply_color_to_ply(pts, bev, xy, xmin, xmax, ymin, ymax, 
                       canvas_W, canvas_H, ox, oy, scale, rot_deg, bev_raw, orig_cols):
    """
    Color PLY points using EXACT SAME coordinate logic as GUI overlay.
    """

    # 1) Convert XY → canvas pixel (same as draw_ply_xy_colored)
    X = xy[:, 0]
    Y = xy[:, 1]

    u = ((X - xmin) / (xmax - xmin)) * (canvas_W - 1)
    v = ((ymax - Y) / (ymax - ymin)) * (canvas_H - 1)

    u = u.astype(np.int32)
    v = v.astype(np.int32)

    # 2) Build transformed BEV exactly as GUI
    bev_transformed = transform_bev(bev_raw, scale, rot_deg)
    bh, bw = bev_transformed.shape[:2]

    # GUI overlay placement
    x0 = ox + canvas_W // 2
    y0 = oy + canvas_H // 2
    x1 = x0 + bw
    y1 = y0 + bh

    cols = []

    for idx, (px, py) in enumerate(zip(u, v)):
        # If the point falls outside BEV region → fill with solid background color
        if not (x0 <= px < x1 and y0 <= py < y1):
            # Fill uncovered region with solid background color
            bg = np.array([0x04, 0x04, 0x07], dtype=np.float32)  # BGR in 0–255
            cols.append(bg)
            continue

        # Map canvas pixel → BEV pixel
        bx = px - x0
        by = py - y0

        # Use exact BEV pixel color (preserve thin line fidelity)
        col = bev_transformed[by, bx].astype(np.float32)
        cols.append(col)

    cols = np.array(cols)[:, ::-1] / 255.0
    return cols


# ----------------------
# TRANSFORM BEV
# ----------------------
def transform_bev(bev_raw, scale, rot_deg):
    bev = cv2.resize(bev_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    M = cv2.getRotationMatrix2D((bev.shape[1] // 2, bev.shape[0] // 2), rot_deg, 1.0)
    bev = cv2.warpAffine(bev, M, (bev.shape[1], bev.shape[0]))
    return bev


# ----------------------
# MAIN PROGRAM
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--bev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flip", type=str, default="none",
                    help="Flip BEV image: none | h | v | hv")
    args = ap.parse_args()

    pts, xy_raw, cols_raw = load_ply_xy_with_color(args.ply)
    bev_raw = cv2.imread(args.bev)
    # --- Strong multi-stage dilation to keep lines visible under scaling ---
    # Stage 1: 9x9 kernel (base thickness)
    kernel1 = np.ones((9, 9), np.uint8)
    bev_raw = cv2.dilate(bev_raw, kernel1, iterations=1)

    # Stage 2: 5x5 kernel (smooth growth)
    kernel2 = np.ones((5, 5), np.uint8)
    bev_raw = cv2.dilate(bev_raw, kernel2, iterations=1)

    # Stage 3: 3x3 kernel (fine reinforcement)
    kernel3 = np.ones((3, 3), np.uint8)
    bev_raw = cv2.dilate(bev_raw, kernel3, iterations=1)

    # --- Optional BEV flipping ---
    if args.flip.lower() == "h":
        bev_raw = cv2.flip(bev_raw, 1)
    elif args.flip.lower() == "v":
        bev_raw = cv2.flip(bev_raw, 0)
    elif args.flip.lower() in ("hv", "vh"):
        bev_raw = cv2.flip(bev_raw, -1)

    base_canvas, xmin_base, xmax_base, ymin_base, ymax_base, W, H = draw_ply_xy_colored(
        xy_raw, cols_raw, base=1300
    )

    # 초기 값
    scale = 1.0
    ox = 0
    oy = 0
    rot = 0.0
    alpha = 0.4

    # ----------------------
    # Create Window + Trackbars
    # ----------------------
    cv2.namedWindow("ALIGN", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ALIGN", W, H)

    cv2.createTrackbar("Scale x10000", "ALIGN", int(scale * 10000), 50000, lambda x: None)
    cv2.createTrackbar("Offset X",   "ALIGN", ox + 2000, 4000, lambda x: None)
    cv2.createTrackbar("Offset Y",   "ALIGN", oy + 2000, 4000, lambda x: None)
    cv2.createTrackbar("Rotate x10", "ALIGN", int(rot * 10) + 1800, 3600, lambda x: None)
    cv2.createTrackbar("Alpha x100", "ALIGN", int(alpha * 100), 100, lambda x: None)

    print("\n=========== KEY CONTROL ===========")
    print("W / A / S / D : 이동(Offset 미세 조절)")
    print("- / =         : 스케일 ±0.01")
    print("Q / E         : 회전 ±0.1°")
    print("Z / X         : 투명도 ±0.01")
    print("S             : 색칠된 PLY 저장")
    print("ESC           : 종료")
    print("===================================\n")

    while True:

        # ---- 슬라이더 변화 감지 ----
        new_scale = cv2.getTrackbarPos("Scale x10000", "ALIGN") / 10000.0
        new_ox    = cv2.getTrackbarPos("Offset X", "ALIGN") - 2000
        new_oy    = cv2.getTrackbarPos("Offset Y", "ALIGN") - 2000
        new_rot   = (cv2.getTrackbarPos("Rotate x10", "ALIGN") - 1800) / 10.0
        new_alpha = cv2.getTrackbarPos("Alpha x100", "ALIGN") / 100.0

        slider_changed = False
        if abs(new_scale - scale) > 1e-6:
            scale = new_scale
            slider_changed = True
        if new_ox != ox:
            ox = new_ox
            slider_changed = True
        if new_oy != oy:
            oy = new_oy
            slider_changed = True
        if abs(new_rot - rot) > 1e-6:
            rot = new_rot
            slider_changed = True
        if abs(new_alpha - alpha) > 1e-6:
            alpha = new_alpha
            slider_changed = True

        # ----------------------
        # 변환된 BEV 만들기
        # ----------------------
        bev = transform_bev(bev_raw, scale, rot)

        overlay = base_canvas.copy()

        # BEV 위치 계산
        bh, bw = bev.shape[:2]
        x0 = ox + W // 2
        y0 = oy + H // 2
        x1 = x0 + bw
        y1 = y0 + bh

        # ROI 영역 계산
        if x1 > 0 and y1 > 0 and x0 < W and y0 < H:
            xs0 = max(0, x0)
            ys0 = max(0, y0)
            xs1 = min(W, x1)
            ys1 = min(H, y1)

            bx0 = xs0 - x0
            by0 = ys0 - y0
            bx1 = bx0 + (xs1 - xs0)
            by1 = by0 + (ys1 - ys0)

            roi_ov  = overlay[ys0:ys1, xs0:xs1]
            roi_bev = bev[by0:by1, bx0:bx1]

            merged = cv2.addWeighted(roi_ov, 1-alpha, roi_bev, alpha, 0)
            overlay[ys0:ys1, xs0:xs1] = merged

        # ---- Display float values on screen ----
        debug_text = f"Scale: {scale:.4f}   Offset: ({ox}, {oy})   Rot: {rot:.2f}°   Alpha: {alpha:.2f}"
        cv2.putText(
            overlay, debug_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imshow("ALIGN", overlay)

        # ----------------------
        # 키 입력
        # ----------------------
        key = cv2.waitKey(10)

        if key == 27:  # ESC
            break

        changed = False

        # 이동 (WASD)
        if key == ord('a'): ox -= 2; changed = True  # move left
        if key == ord('d'): ox += 2; changed = True  # move right
        if key == ord('w'): oy -= 2; changed = True  # move up
        if key == ord('s'): oy += 2; changed = True  # move down

        # scale (now 'k' and 'l' keys)
        if key == ord('k'): scale -= 0.01; changed = True
        if key == ord('l'): scale += 0.01; changed = True

        # rotate
        if key == ord('q'): rot -= 0.1; changed = True
        if key == ord('e'): rot += 0.1; changed = True

        # alpha
        if key == ord('z'): alpha = max(0.0, alpha - 0.01); changed = True
        if key == ord('x'): alpha = min(1.0, alpha + 0.01); changed = True

        # Clamp values after keyboard-based adjustments
        scale = max(0.01, min(scale, 5.0))
        rot = max(-180.0, min(rot, 180.0))
        alpha = max(0.0, min(alpha, 1.0))

        # ----------------------
        # 키로 바뀌었거나 슬라이더로 바뀌었으면 → 슬라이더 업데이트
        # ----------------------
        if changed or slider_changed:
            scale_pos = int(scale * 10000)
            scale_pos = max(0, min(scale_pos, 50000))

            ox_pos = ox + 2000
            ox_pos = max(0, min(ox_pos, 4000))

            oy_pos = oy + 2000
            oy_pos = max(0, min(oy_pos, 4000))

            rot_pos = int(rot * 10) + 1800
            rot_pos = max(0, min(rot_pos, 3600))

            alpha_pos = int(alpha * 100)
            alpha_pos = max(0, min(alpha_pos, 100))

            cv2.setTrackbarPos("Scale x10000", "ALIGN", scale_pos)
            cv2.setTrackbarPos("Offset X",   "ALIGN", ox_pos)
            cv2.setTrackbarPos("Offset Y",   "ALIGN", oy_pos)
            cv2.setTrackbarPos("Rotate x10", "ALIGN", rot_pos)
            cv2.setTrackbarPos("Alpha x100", "ALIGN", alpha_pos)

        # ----------------------
        # 저장
        # ----------------------
        if key == ord('s'):
            print("[INFO] Saving PLY...")

            transformed_bev = transform_bev(bev_raw, scale, rot)

            # Use original XY (no offset in world units) and original colors as fallback
            cols_new = apply_color_to_ply(
                pts,
                bev_raw,
                xy_raw,
                xmin_base,
                xmax_base,
                ymin_base,
                ymax_base,
                W, H,
                ox, oy,
                scale, rot,
                bev_raw,
                cols_raw
            )

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols_new)
            o3d.io.write_point_cloud(args.out, pcd, write_ascii=True)

            print(f"[OK] Saved → {args.out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()