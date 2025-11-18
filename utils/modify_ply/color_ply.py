#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLY 색 편집 도구 (XY 탑뷰 기준)
- 브러시 지우개 / 브러시 색칠 / 스포이드

조작키:
  마우스:
    - Erase/Brush 모드(E/B): 왼쪽 클릭/드래그로 편집
    - Color Picker 모드(C): 점이 있는 곳을 클릭 → 그 점 색을 브러시 색으로 설정

  키보드:
    - E : Erase 브러시 모드
    - B : Brush 페인트 모드
    - C : Color Picker (스포이드) 모드
    - S : 현재 결과 저장 (alive+색 수정 포함)
    - Z : 직전 편집(브러시/지우기) 되돌리기
    - ESC : 종료

Controls 창:
    - R, G, B : 브러시 색상 (0~255)
    - Brush   : 브러시 반경 (픽셀)
"""

import os
import argparse
from collections import deque
import numpy as np
import cv2
import open3d as o3d

# ------------------------ 유틸 ------------------------

def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64)
    if cols is None or len(cols) != len(pts):
        cols = np.ones((len(pts), 3), dtype=np.float64) * 0.8  # 기본 회색
    return pts, cols

def save_ply(pts, cols, out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0).astype(np.float64))
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
    print(f"[OK] Saved: {out_path}  (N={len(pts)})")

def compute_view_xy_to_img(xy, W, H, pad=20):
    """
    xy: (N,2) [X,Y], X→오른쪽 +, Y→위쪽 + 라고 가정.
    이미지에서는 Y가 아래로 증가하므로, Y를 뒤집어서 렌더링.
    """
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
        px = left + (xy_[:, 0] - gmin[0]) * scale
        # Y 축 뒤집어서 그림 (위쪽 + → 이미지 아래쪽 +)
        py = top  + (gmax[1] - xy_[:, 1]) * scale
        return np.stack([px, py], axis=1).astype(np.int32)

    def px_to_xy(pix_):
        x = (pix_[:, 0] - left) / scale + gmin[0]
        y = gmax[1] - (pix_[:, 1] - top) / scale
        return np.stack([x, y], axis=1)

    return xy_to_px, px_to_xy

# ------------------------ 메인 ------------------------

def main():
    ap = argparse.ArgumentParser("PLY XY painter (brush + erase + picker)")
    ap.add_argument("--ply", required=True, help="입력 PLY 경로")
    ap.add_argument("--out", default="", help="출력 PLY (미지정 시 *_paint.ply)")
    ap.add_argument("--win-w", type=int, default=1000)
    ap.add_argument("--win-h", type=int, default=1000)
    args = ap.parse_args()

    src_path = args.ply
    if args.out:
        out_path = args.out
    else:
        out_path = os.path.splitext(src_path)[0] + "_paint.ply"

    pts, cols = load_ply(src_path)
    if pts.shape[0] == 0:
        print("[ERR] 빈 포인트클라우드입니다.")
        return

    N = pts.shape[0]
    xy = pts[:, [0, 1]].astype(np.float64)

    # 전체 인덱스와 시각화용 서브셋 인덱스
    all_idx = np.arange(N)
    show_target = 100000
    if N > show_target:
        vis_idx = np.random.choice(all_idx, show_target, replace=False)
    else:
        vis_idx = all_idx

    xy_to_px, _ = compute_view_xy_to_img(xy, args.win_w, args.win_h, pad=30)
    px_all = xy_to_px(xy)  # 모든 점의 화면 좌표 (N,2), int32

    alive = np.ones(N, dtype=bool)  # 삭제 여부
    undo_stack = deque(maxlen=50)   # 최근 50회 편집까지 되돌리기

    # 상태 변수들
    base = None      # 배경 (점 렌더링)
    canvas = None    # 배경 + UI 오버레이
    mode = "brush"   # 'brush', 'erase', 'picker'
    brush_radius = 6 # 브러시 반경 (픽셀)

    # 브러시 색 (0~1)
    paint_color = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # 빨강 기본

    win = "PLY Painter"
    controls_win = "Controls"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.win_w, args.win_h)

    cv2.namedWindow(controls_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(controls_win, 350, 130)

    # Trackbar: R,G,B, Brush
    cv2.createTrackbar("R", controls_win, int(paint_color[0] * 255), 255, lambda v: None)
    cv2.createTrackbar("G", controls_win, int(paint_color[1] * 255), 255, lambda v: None)
    cv2.createTrackbar("B", controls_win, int(paint_color[2] * 255), 255, lambda v: None)
    cv2.createTrackbar("Brush", controls_win, brush_radius, 100, lambda v: None)

    help1 = "LMB drag: edit | E:erase  B:brush  C:picker"
    help2 = "S:save  Z:undo  ESC:quit"
    info3 = f"INPUT: {os.path.basename(src_path)}  ->  OUTPUT: {os.path.basename(out_path)}"

    def redraw_base():
        nonlocal base
        base = np.zeros((args.win_h, args.win_w, 3), np.uint8)
        base[:, :] = (18, 18, 18)

        # vis_idx 중 살아있는 점만 표시
        idx = vis_idx[alive[vis_idx]]
        if idx.size == 0:
            return

        pts_px = px_all[idx]
        cols_vis = (np.clip(cols[idx][:, ::-1], 0, 1) * 255).astype(np.uint8)

        h, w = base.shape[:2]
        px = pts_px[:, 0]
        py = pts_px[:, 1]

        valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if not np.any(valid):
            return

        px_valid = px[valid]
        py_valid = py[valid]
        cols_valid = cols_vis[valid]

        base[py_valid, px_valid] = cols_valid

    def redraw():
        nonlocal canvas
        canvas = base.copy()

        # 도움말 텍스트
        cv2.putText(canvas, help1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, help2, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, info3, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 180, 180), 1, cv2.LINE_AA)

        # 모드 표시
        if mode == "erase":
            mode_txt = f"MODE: BRUSH ERASE (r={brush_radius})"
        elif mode == "brush":
            mode_txt = f"MODE: BRUSH PAINT (r={brush_radius})"
        else:
            mode_txt = "MODE: COLOR PICKER"

        cv2.putText(canvas, mode_txt, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (120, 220, 255), 2, cv2.LINE_AA)

        # 현재 브러시 색 미리보기 박스
        color_box = (np.clip(paint_color[::-1], 0, 1) * 255).astype(np.uint8).tolist()  # BGR
        cv2.rectangle(canvas, (20, 130), (90, 160), color_box, -1)
        cv2.rectangle(canvas, (20, 130), (90, 160), (255, 255, 255), 1)
        cv2.putText(canvas, "Brush Color", (100, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)

        cv2.imshow(win, canvas)

    def undo_last():
        """최근 브러시/지우기 동작을 되돌린다."""
        nonlocal alive, cols
        if not undo_stack:
            print("[UNDO] nothing to undo.")
            return

        change = undo_stack.pop()
        idx = change["idx"]

        if "prev_alive" in change:
            alive[idx] = change["prev_alive"]
        if "prev_cols" in change:
            cols[idx] = change["prev_cols"]

        redraw_base()
        redraw()
        print(f"[UNDO] reverted {len(idx)} points.")

    def apply_brush(x, y):
        nonlocal alive, cols, base
        # vis_idx 서브셋에 대해서만 거리 계산 (화면에 보이는 점들만 편집)
        dx = px_all[vis_idx, 0].astype(np.float64) - float(x)
        dy = px_all[vis_idx, 1].astype(np.float64) - float(y)
        dist2 = dx*dx + dy*dy
        hit_local = dist2 <= (brush_radius * brush_radius)
        hit_local &= alive[vis_idx]

        if not np.any(hit_local):
            return

        hit_idx = vis_idx[hit_local]
        change = {"idx": hit_idx}

        if mode == "erase":
            # 이미 삭제된 점은 기록/처리하지 않음
            changed_mask = alive[hit_idx]
            if not np.any(changed_mask):
                return
            changed_idx = hit_idx[changed_mask]
            change["idx"] = changed_idx
            change["prev_alive"] = alive[changed_idx].copy()
            alive[changed_idx] = False
        elif mode == "brush":
            # 현재 색과 다른 점만 기록/변경
            need_change = np.any(np.abs(cols[hit_idx] - paint_color) > 1e-6, axis=1)
            if not np.any(need_change):
                return
            changed_idx = hit_idx[need_change]
            change["idx"] = changed_idx
            change["prev_cols"] = cols[changed_idx].copy()
            cols[changed_idx] = paint_color
        else:
            return

        undo_stack.append(change)

        redraw_base()
        redraw()

    def pick_color_at(x, y):
        nonlocal paint_color
        # 클릭 지점 근처에서 vis_idx 중 가장 가까운 점의 색을 가져옴
        dx = px_all[vis_idx, 0].astype(np.float64) - float(x)
        dy = px_all[vis_idx, 1].astype(np.float64) - float(y)
        dist2 = dx*dx + dy*dy
        # 이미 지워진 점은 무시
        alive_vis = alive[vis_idx]
        dist2[~alive_vis] = np.inf

        idx_local = np.argmin(dist2)
        if not np.isfinite(dist2[idx_local]):
            print("[PICKER] no point near click.")
            return
        # 너무 멀면 무시 (예: 10픽셀 이상)
        if dist2[idx_local] > (10.0 * 10.0):
            print("[PICKER] click too far from any point.")
            return

        idx_global = vis_idx[idx_local]
        paint_color = cols[idx_global].copy()
        print(f"[PICKER] picked color from point #{idx_global}: {paint_color}")

        # Trackbar도 동기화
        r = int(np.clip(paint_color[0], 0, 1) * 255)
        g = int(np.clip(paint_color[1], 0, 1) * 255)
        b = int(np.clip(paint_color[2], 0, 1) * 255)
        cv2.setTrackbarPos("R", controls_win, r)
        cv2.setTrackbarPos("G", controls_win, g)
        cv2.setTrackbarPos("B", controls_win, b)

        redraw()

    def on_mouse(event, x, y, flags, param):
        nonlocal mode
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == "picker":
                pick_color_at(x, y)
                # 스포이드 후 바로 브러시로 돌아가고 싶으면 아래 주석 해제:
                # mode = "brush"
                return
            else:
                apply_brush(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if mode in ("erase", "brush"):
                apply_brush(x, y)

    cv2.setMouseCallback(win, on_mouse)

    # 초기 렌더
    redraw_base()
    redraw()

    print("[INFO] Controls window에서 R,G,B,Brush 조절 가능.")
    print("[INFO] E(erase), B(brush), C(picker), S(save), Z(undo), ESC(quit)")

    while True:
        # Trackbar 값 읽어서 paint_color / brush_radius 업데이트
        r = cv2.getTrackbarPos("R", controls_win)
        g = cv2.getTrackbarPos("G", controls_win)
        b = cv2.getTrackbarPos("B", controls_win)
        br = cv2.getTrackbarPos("Brush", controls_win)
        br = max(1, br)

        paint_color[:] = np.array([r, g, b], dtype=np.float64) / 255.0
        brush_radius = br

        key = cv2.waitKeyEx(20)
        if key == 27:  # ESC
            break
        elif key in (ord('e'), ord('E')):
            mode = "erase"
            print("[MODE] brush erase")
            redraw()
        elif key in (ord('b'), ord('B')):
            mode = "brush"
            print("[MODE] brush paint")
            redraw()
        elif key in (ord('c'), ord('C')):
            mode = "picker"
            print("[MODE] color picker (click on point)")
            redraw()
        elif key in (ord('z'), ord('Z')):
            undo_last()
        elif key in (ord('s'), ord('S')):
            # 저장 시점: vis_idx의 편집 결과를 전체 포인트로 전파
            kept_vis = vis_idx[alive[vis_idx]]
            if kept_vis.size == 0:
                print("[WARN] no alive points in visible subset. cancel save.")
                continue

            # vis_idx 기준 포인트클라우드 생성 (KD-Tree용)
            pc_vis = o3d.geometry.PointCloud()
            pc_vis.points = o3d.utility.Vector3dVector(pts[vis_idx].astype(np.float64))
            kdtree = o3d.geometry.KDTreeFlann(pc_vis)

            new_cols = np.zeros_like(cols)
            final_alive = np.ones(N, dtype=bool)

            # 전체 포인트에 대해, 가장 가까운 vis_idx의 색/삭제 상태를 복사
            for i in range(N):
                # 가까운 vis_idx 찾기
                k, idx_knn, _ = kdtree.search_knn_vector_3d(pts[i].astype(np.float64), 1)
                if k == 0:
                    # 이론상 거의 없겠지만, 매칭 실패 시 원본 유지
                    new_cols[i] = cols[i]
                    final_alive[i] = alive[i]
                    continue

                j_vis_local = idx_knn[0]
                j_global = vis_idx[j_vis_local]

                # 삭제 상태 전파
                if not alive[j_global]:
                    final_alive[i] = False
                else:
                    final_alive[i] = True

                # 색 전파 (편집된 vis_idx 색 사용)
                new_cols[i] = cols[j_global]

            kept_idx = np.where(final_alive)[0]
            if kept_idx.size == 0:
                print("[WARN] no alive points after propagation. cancel save.")
                continue

            save_ply(pts[kept_idx], new_cols[kept_idx], out_path)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
