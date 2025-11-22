#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLY 색 편집 도구 (XY 탑뷰 기반)
- 전체 포인트를 탑뷰 이미지로 투영해 사진처럼 확인 후 색칠
- 브러시 / 지우개 / 스포이드 + Undo + 저장
"""

import os
import argparse
from collections import deque
from typing import Tuple

import numpy as np
import cv2
import open3d as o3d

BACKGROUND_BGR = (18, 18, 18)
UNDO_HISTORY = 80


def load_ply(path: str) -> Tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64)
    if cols is None or len(cols) != len(pts):
        cols = np.ones((len(pts), 3), dtype=np.float64) * 0.8
    return pts, cols


def save_ply(pts: np.ndarray, cols: np.ndarray, out_path: str):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0).astype(np.float64))
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
    print(f"[OK] Saved: {out_path} (N={len(pts)})")


def compute_view_xy_to_img(xy: np.ndarray, W: int, H: int, pad: int = 20):
    """
    xy: (N,2) [X,Y], X→오른쪽 +, Y→위쪽 + 라고 가정.
    이미지에서는 Y가 아래로 증가하므로, Y를 뒤집어서 렌더링.
    """
    gmin = xy.min(axis=0)
    gmax = xy.max(axis=0)
    grng = np.maximum(gmax - gmin, 1e-6)

    scale_x = (W - 2 * pad) / grng[0]
    scale_y = (H - 2 * pad) / grng[1]
    scale = min(scale_x, scale_y)

    draw_w = grng[0] * scale
    draw_h = grng[1] * scale
    left = (W - draw_w) * 0.5
    top = (H - draw_h) * 0.5

    def xy_to_px(xy_):
        px = left + (xy_[:, 0] - gmin[0]) * scale
        py = top + (gmax[1] - xy_[:, 1]) * scale
        return np.stack([px, py], axis=1).astype(np.int32)

    def px_to_xy(pix_):
        x = (pix_[:, 0] - left) / scale + gmin[0]
        y = gmax[1] - (pix_[:, 1] - top) / scale
        return np.stack([x, y], axis=1)

    return xy_to_px, px_to_xy


def build_topdown_image(
    xy: np.ndarray,
    cols: np.ndarray,
    W: int,
    H: int,
    pad: int = 30,
    spread_ksize: int = 7,
):
    xy_to_px, _ = compute_view_xy_to_img(xy, W, H, pad=pad)
    px_all = xy_to_px(xy).astype(np.int32)
    px = px_all[:, 0]
    py = px_all[:, 1]

    in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px_valid = px[in_bounds]
    py_valid = py[in_bounds]
    color_sum = np.zeros((H, W, 3), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    if px_valid.size:
        cols_valid = cols[in_bounds].astype(np.float32)
        np.add.at(color_sum[:, :, 0], (py_valid, px_valid), cols_valid[:, 0])
        np.add.at(color_sum[:, :, 1], (py_valid, px_valid), cols_valid[:, 1])
        np.add.at(color_sum[:, :, 2], (py_valid, px_valid), cols_valid[:, 2])
        np.add.at(count, (py_valid, px_valid), 1.0)

    count_exp = count[..., None]
    color_avg = np.zeros_like(color_sum)
    np.divide(
        color_sum,
        np.maximum(count_exp, 1e-6),
        out=color_avg,
        where=count_exp > 0,
    )
    mask = count > 0

    if spread_ksize > 1:
        k = max(1, int(spread_ksize))
        if k % 2 == 0:
            k += 1
        color_avg = cv2.GaussianBlur(color_avg, (k, k), 0)
        mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
        mask = mask_blur > 1e-3

    color_img = (np.clip(color_avg[:, :, ::-1], 0.0, 1.0) * 255).astype(np.uint8)
    mask_img = mask.astype(np.uint8) * 255
    return color_img, mask_img, px_all


def main():
    ap = argparse.ArgumentParser("PLY XY painter (top-down image based)")
    ap.add_argument("--ply", required=True, help="입력 PLY 경로")
    ap.add_argument("--out", default="", help="출력 PLY (미지정 시 *_paint.ply)")
    ap.add_argument("--win-w", type=int, default=1200, help="탑뷰 이미지 가로 크기")
    ap.add_argument("--win-h", type=int, default=1200, help="탑뷰 이미지 세로 크기")
    ap.add_argument("--pad", type=int, default=30, help="탑뷰 바운더리 여백 (픽셀)")
    ap.add_argument(
        "--spread-ksize",
        type=int,
        default=7,
        help="포인트를 퍼뜨릴 커널 크기(홀수). 1이면 퍼뜨리지 않음.",
    )
    args = ap.parse_args()

    src_path = args.ply
    out_path = args.out if args.out else os.path.splitext(src_path)[0] + "_paint.ply"

    pts, cols = load_ply(src_path)
    if pts.size == 0:
        print("[ERR] 빈 포인트클라우드입니다.")
        return

    xy = pts[:, [0, 1]].astype(np.float64)

    color_img, mask_img, px_all = build_topdown_image(
        xy,
        cols,
        args.win_w,
        args.win_h,
        pad=args.pad,
        spread_ksize=args.spread_ksize,
    )

    win = "PLY Painter"
    controls_win = "Controls"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.win_w, args.win_h)
    cv2.namedWindow(controls_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(controls_win, 360, 150)

    undo_stack = deque(maxlen=UNDO_HISTORY)
    mode = "brush"  # 'brush', 'erase', 'picker'
    brush_radius = 8
    paint_color = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # RGB
    canvas = None

    cv2.createTrackbar("R", controls_win, int(paint_color[0] * 255), 255, lambda v: None)
    cv2.createTrackbar("G", controls_win, int(paint_color[1] * 255), 255, lambda v: None)
    cv2.createTrackbar("B", controls_win, int(paint_color[2] * 255), 255, lambda v: None)
    cv2.createTrackbar("Brush", controls_win, brush_radius, 150, lambda v: None)

    help1 = "LMB drag: paint/erase | E:Erase  B:Brush  C:Picker"
    help2 = "S:save  Z:undo  ESC:quit"
    info3 = f"INPUT: {os.path.basename(src_path)} -> OUTPUT: {os.path.basename(out_path)}"

    def redraw():
        nonlocal canvas
        canvas = color_img.copy()
        blank = (mask_img == 0)
        if np.any(blank):
            canvas[blank] = BACKGROUND_BGR

        if mode == "erase":
            mode_txt = f"MODE: BRUSH ERASE (r={brush_radius})"
        elif mode == "brush":
            mode_txt = f"MODE: BRUSH PAINT (r={brush_radius})"
        else:
            mode_txt = "MODE: COLOR PICKER"

        paint_preview = (np.clip(paint_color[::-1], 0, 1) * 255).astype(np.uint8)

        cv2.putText(canvas, help1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, help2, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, info3, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, mode_txt, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (120, 220, 255), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (20, 130), (100, 170), paint_preview.tolist(), -1)
        cv2.rectangle(canvas, (20, 130), (100, 170), (255, 255, 255), 1)
        cv2.putText(canvas, "Brush Color", (110, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)

        cv2.imshow(win, canvas)

    def record_patch(x0, y0, x1, y1):
        undo_stack.append({
            "rect": (x0, y0, x1, y1),
            "color": color_img[y0:y1, x0:x1].copy(),
            "mask": mask_img[y0:y1, x0:x1].copy(),
        })

    def apply_brush(x, y):
        nonlocal brush_radius
        if mode not in ("brush", "erase"):
            return
        x = int(x)
        y = int(y)
        x0 = max(0, x - brush_radius)
        x1 = min(args.win_w, x + brush_radius + 1)
        y0 = max(0, y - brush_radius)
        y1 = min(args.win_h, y + brush_radius + 1)
        if x0 >= x1 or y0 >= y1:
            return
        record_patch(x0, y0, x1, y1)

        if mode == "brush":
            brush_bgr = tuple(int(np.clip(c, 0, 255)) for c in (paint_color[::-1] * 255))
            cv2.circle(color_img, (x, y), brush_radius, brush_bgr, -1, lineType=cv2.LINE_AA)
            cv2.circle(mask_img, (x, y), brush_radius, 255, -1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(color_img, (x, y), brush_radius, BACKGROUND_BGR, -1, lineType=cv2.LINE_AA)
            cv2.circle(mask_img, (x, y), brush_radius, 0, -1, lineType=cv2.LINE_AA)
        redraw()

    def pick_color_at(x, y):
        nonlocal paint_color
        x = int(np.clip(x, 0, args.win_w - 1))
        y = int(np.clip(y, 0, args.win_h - 1))
        if mask_img[y, x] == 0:
            print("[PICKER] 포인트가 없는 영역입니다.")
            return
        bgr = color_img[y, x].astype(np.float32)
        paint_color = (bgr[::-1] / 255.0)
        r = int(np.clip(paint_color[0], 0, 1) * 255)
        g = int(np.clip(paint_color[1], 0, 1) * 255)
        b = int(np.clip(paint_color[2], 0, 1) * 255)
        cv2.setTrackbarPos("R", controls_win, r)
        cv2.setTrackbarPos("G", controls_win, g)
        cv2.setTrackbarPos("B", controls_win, b)
        redraw()
        print(f"[PICKER] picked BGR={bgr} at ({x}, {y})")

    def undo_last():
        if not undo_stack:
            print("[UNDO] nothing to undo.")
            return
        change = undo_stack.pop()
        x0, y0, x1, y1 = change["rect"]
        color_img[y0:y1, x0:x1] = change["color"]
        mask_img[y0:y1, x0:x1] = change["mask"]
        redraw()
        print("[UNDO] reverted last patch.")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == "picker":
                pick_color_at(x, y)
            else:
                apply_brush(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if mode in ("brush", "erase"):
                apply_brush(x, y)

    cv2.setMouseCallback(win, on_mouse)

    redraw()
    print("[INFO] Controls 창에서 R,G,B,Brush 조절 가능.")
    print("[INFO] E(erase), B(brush), C(picker), S(save), Z(undo), ESC(quit)")

    while True:
        r = cv2.getTrackbarPos("R", controls_win)
        g = cv2.getTrackbarPos("G", controls_win)
        b = cv2.getTrackbarPos("B", controls_win)
        br = cv2.getTrackbarPos("Brush", controls_win)
        br = max(1, br)
        paint_color = np.array([r, g, b], dtype=np.float64) / 255.0
        brush_radius = br

        key = cv2.waitKeyEx(20)
        if key == 27:
            break
        elif key in (ord('e'), ord('E')):
            mode = "erase"
            print("[MODE] Brush -> ERASE")
            redraw()
        elif key in (ord('b'), ord('B')):
            mode = "brush"
            print("[MODE] Brush -> PAINT")
            redraw()
        elif key in (ord('c'), ord('C')):
            mode = "picker"
            print("[MODE] COLOR PICKER")
            redraw()
        elif key in (ord('z'), ord('Z')):
            undo_last()
        elif key in (ord('s'), ord('S')):
            width = args.win_w
            height = args.win_h
            px = px_all[:, 0]
            py = px_all[:, 1]
            valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            px_clip = np.clip(px, 0, width - 1)
            py_clip = np.clip(py, 0, height - 1)
            mask_values = mask_img[py_clip, px_clip] > 0
            keep_idx = np.where(valid & mask_values)[0]
            outside_idx = np.where(~valid)[0]

            outputs_pts = []
            outputs_cols = []
            if keep_idx.size:
                sampled = (color_img[py_clip[keep_idx], px_clip[keep_idx]][:, ::-1].astype(np.float32) / 255.0)
                outputs_pts.append(pts[keep_idx])
                outputs_cols.append(sampled)
            if outside_idx.size:
                outputs_pts.append(pts[outside_idx])
                outputs_cols.append(cols[outside_idx])

            if not outputs_pts:
                print("[WARN] 저장할 포인트가 없습니다.")
                continue

            pts_save = np.vstack(outputs_pts)
            cols_save = np.vstack(outputs_cols)
            save_ply(pts_save, cols_save, out_path)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
