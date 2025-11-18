#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLY를 XY 평면(탑뷰) 기준 '자유형(폴리곤)'으로 크롭하는 도구
- 마우스로 다각형 꼭짓점을 순서대로 클릭 → Enter(또는 Return)로 확정 → S로 저장
- I: 선택 반전(폴리곤 바깥만 남기기), R: 폴리곤 리셋, ESC: 종료
- Z는 변경하지 않음(그대로 유지). 색상도 유지.
"""

import os, sys, argparse
import numpy as np
import cv2
import open3d as o3d

# ------------------------ 유틸 ------------------------

def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64)
    if cols is None or len(cols) != len(pts):
        cols = np.ones((len(pts), 3), dtype=np.float64) * 0.8  # 회색 기본
    return pts, cols

def save_ply(pts, cols, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0).astype(np.float64))
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
    print(f"[OK] Saved: {out_path}  (N={len(pts)})")

def compute_view_xy_to_img(xy, W, H, pad=20):
    gmin = xy.min(axis=0)
    gmax = xy.max(axis=0)
    grng = np.maximum(gmax - gmin, 1e-6)

    scale_x = (W - 2*pad) / grng[0]
    scale_y = (H - 2*pad) / grng[1]
    scale = min(scale_x, scale_y)

    draw_w = grng[0] * scale
    draw_h = grng[1] * scale
    left = (W - draw_w) * 0.5
    top  = (H - draw_h) * 0.5

    def xy_to_px(xy_):
        # X 오른쪽 +, Y 위쪽 + → 이미지에선 Y를 아래로 늘려야 하므로 invert
        px = left + (xy_[:, 0] - gmin[0]) * scale
        py = top  + (gmax[1] - xy_[:, 1]) * scale
        return np.stack([px, py], axis=1).astype(np.int32)

    def px_to_xy(pix_):
        # 역변환 (정확한 역산, 다각형을 XY로 옮길 때 사용)
        x = (pix_[:, 0] - left) / scale + gmin[0]
        y = gmax[1] - (pix_[:, 1] - top) / scale
        return np.stack([x, y], axis=1)

    return xy_to_px, px_to_xy

def point_in_poly_batch(xy_pts, poly_xy):
    """
    다각형 포함검사 (레이 캐스팅). xy_pts:(N,2), poly_xy:(M,2)
    반환: inside(bool N,)
    """
    N = xy_pts.shape[0]
    inside = np.zeros(N, dtype=bool)
    x = xy_pts[:, 0]
    y = xy_pts[:, 1]
    px = poly_xy[:, 0]
    py = poly_xy[:, 1]
    M = poly_xy.shape[0]

    # 벡터화 레이 캐스팅
    for i in range(M):
        j = (i + 1) % M
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        # 에지(y 범위)와의 교차여부
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= intersect
    return inside

# ------------------------ 인터랙티브 도구 ------------------------

def main():
    ap = argparse.ArgumentParser("XY 자유형(폴리곤) 크롭")
    ap.add_argument("--ply", required=True, help="입력 PLY")
    ap.add_argument("--out", default="", help="출력 PLY (미지정 시 *_crop.ply)")
    ap.add_argument("--win-w", type=int, default=1400)
    ap.add_argument("--win-h", type=int, default=1000)
    args = ap.parse_args()

    src_path = args.ply
    out_path = args.out if args.out else os.path.splitext(src_path)[0] + "_crop.ply"

    pts, cols = load_ply(src_path)
    if pts.shape[0] == 0:
        print("[ERR] 빈 포인트클라우드입니다.")
        return

    xy = pts[:, [0, 1]].astype(np.float64)
    xy_to_px, px_to_xy = compute_view_xy_to_img(xy, args.win_w, args.win_h, pad=30)

    # 베이스 캔버스 준비
    base = np.zeros((args.win_h, args.win_w, 3), np.uint8)
    base[:, :] = (18, 18, 18)

    # 샘플링(표시 속도)
    N = xy.shape[0]
    show_target = 400000
    idx_show = np.arange(N) if N <= show_target else np.random.choice(N, show_target, replace=False)
    px_show = xy_to_px(xy[idx_show])
    # 실제 포인트 색상 반영 (RGB→BGR 변환 후 0~255 범위)
    cols_show = (np.clip(cols[idx_show][:, ::-1], 0, 1) * 255).astype(np.uint8)
    for (px, py), c in zip(px_show, cols_show):
        if 0 <= px < args.win_w and 0 <= py < args.win_h:
            base[py, px] = c

    canvas = base.copy()
    poly_pix = []   # 픽셀좌표로 폴리곤 저장
    closed = False
    invert = False  # I로 반전 여부

    help1 = "Mouse: add points | Enter: close | I: invert | R: reset | S: save | ESC: quit"
    help2 = f"Input: {os.path.basename(src_path)}  ->  Output: {os.path.basename(out_path)}"
    cv2.putText(canvas, help1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, help2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    win = "XY Crop (polygon)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.win_w, args.win_h)

    def on_mouse(event, x, y, flags, param):
        nonlocal canvas, poly_pix, closed
        if event == cv2.EVENT_LBUTTONDOWN and not closed:
            poly_pix.append((x, y))
        redraw()

    def redraw():
        nonlocal canvas, poly_pix, closed
        canvas = base.copy()
        # 점/선 그리기
        if len(poly_pix) > 0:
            for i, (px, py) in enumerate(poly_pix):
                cv2.circle(canvas, (px, py), 4, (0, 255, 255), -1, cv2.LINE_AA)
                if i > 0:
                    cv2.line(canvas, poly_pix[i-1], poly_pix[i], (0, 255, 255), 2, cv2.LINE_AA)
        if closed and len(poly_pix) >= 3:
            cv2.polylines(canvas, [np.array(poly_pix, np.int32)], True, (0, 200, 0), 2, cv2.LINE_AA)
        # 도움말
        cv2.putText(canvas, help1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, help2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        mode_txt = f"MODE: {'Keep INSIDE polygon' if not invert else 'Keep OUTSIDE polygon'}"
        cv2.putText(canvas, mode_txt, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)

    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKeyEx(20)
        if key == 27:  # ESC
            break
        elif key == 13 or key == 10:  # Enter
            if len(poly_pix) >= 3:
                closed = True
                redraw()
        elif key in (ord('r'), ord('R')):
            poly_pix = []
            closed = False
            redraw()
        elif key in (ord('i'), ord('I')):
            invert = not invert
            redraw()
        elif key in (ord('s'), ord('S')):
            if not closed or len(poly_pix) < 3:
                print("[INFO] 폴리곤을 먼저 Enter로 확정하세요.")
                continue
            # 픽셀 폴리곤 → XY 폴리곤
            poly_xy = px_to_xy(np.array(poly_pix, dtype=np.float64))
            # 포함검사
            inside = point_in_poly_batch(xy, poly_xy)
            mask = (~inside) if invert else inside
            kept = np.where(mask)[0]
            if kept.size == 0:
                print("[WARN] 선택된 점이 없습니다. 저장을 취소합니다.")
                continue
            save_ply(pts[kept], cols[kept], out_path)
            # 저장 후 미리 반전해서 한번 더 선택하고 싶다면 계속 진행 가능
            print("[HINT] 계속 다른 폴리곤으로도 크롭할 수 있습니다. (R로 리셋)")
        # 계속 갱신
        # (마우스 이벤트로 redraw 수행하므로 여기선 생략 가능)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()