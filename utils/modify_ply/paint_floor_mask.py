#!/usr/bin/env python3
import numpy as np
import cv2
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser("floor_mask painter (image + lut.npz)")
    ap.add_argument("--lut", required=True)
    ap.add_argument("--image", required=True)   # 언디스토트된 jpg
    ap.add_argument("--out", default="")        # 미지정 시 *_floor.npz
    ap.add_argument("--brush", type=int, default=10)
    args = ap.parse_args()

    lut_path = Path(args.lut)
    out_path = Path(args.out) if args.out else lut_path.with_name(lut_path.stem + "_floor.npz")

    data = dict(np.load(str(lut_path), allow_pickle=False))
    H = int(data.get("height"))
    W = int(data.get("width"))
    floor = data.get("floor_mask", None)
    if floor is None:
        # 없으면 0으로 초기화하거나, ground_valid_mask로부터 시작
        gv = data.get("ground_valid_mask", None)
        if gv is not None:
            floor = (gv.astype(bool)).astype(np.uint8)
            print("[INFO] init floor_mask from ground_valid_mask")
        else:
            floor = np.zeros((H, W), np.uint8)
            print("[INFO] init floor_mask as zeros")
    else:
        floor = floor.astype(np.uint8)
    if floor.shape != (H, W):
        raise SystemExit(f"floor_mask shape mismatch: {floor.shape} vs ({H},{W})")

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"failed to read image {args.image}")
    if img.shape[:2] != (H, W):
        raise SystemExit(f"image shape {img.shape[:2]} != LUT ({H},{W})")

    brush_radius = args.brush
    mode = "brush"  # or "erase"
    drawing = False

    def make_vis():
        base = img.copy()
        overlay = img.copy()
        # floor=1인 부분에 색 입히기 (예: 시안)
        overlay[floor > 0] = (overlay[floor > 0] * 0.3 + np.array([255, 255, 0]) * 0.7).astype(np.uint8)
        alpha = 0.5
        vis = cv2.addWeighted(overlay, alpha, base, 1-alpha, 0)
        mode_txt = f"MODE: {mode}  |  brush={brush_radius}  |  S:save  B:brush  E:erase  +/-:brush size"
        cv2.putText(vis, mode_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
        return vis

    vis = make_vis()
    win = "floor_mask painter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W//2, H//2)

    def apply_brush(x, y):
        nonlocal floor
        # (x,y)는 이미지 좌표, floor는 (H,W)
        yy, xx = np.ogrid[:H, :W]
        dist2 = (xx - x)**2 + (yy - y)**2
        hit = dist2 <= brush_radius*brush_radius
        if mode == "brush":
            floor[hit] = 1
        elif mode == "erase":
            floor[hit] = 0

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, vis
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            apply_brush(x, y)
            vis = make_vis()
            cv2.imshow(win, vis)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            apply_brush(x, y)
            vis = make_vis()
            cv2.imshow(win, vis)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            vis = make_vis()
            cv2.imshow(win, vis)

    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, vis)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('b'), ord('B')):
            mode = "brush"
            vis = make_vis()
            cv2.imshow(win, vis)
        elif key in (ord('e'), ord('E')):
            mode = "erase"
            vis = make_vis()
            cv2.imshow(win, vis)
        elif key in (ord('+'), ord('=')):
            brush_radius = min(200, brush_radius + 2)
            vis = make_vis()
            cv2.imshow(win, vis)
        elif key in (ord('-'), ord('_')):
            brush_radius = max(1, brush_radius - 2)
            vis = make_vis()
            cv2.imshow(win, vis)
        elif key in (ord('s'), ord('S')):
            # floor_mask만 업데이트해서 새 npz로 저장
            data["floor_mask"] = floor.astype(np.uint8)
            np.savez_compressed(str(out_path), **data)
            print(f"[OK] saved: {out_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()