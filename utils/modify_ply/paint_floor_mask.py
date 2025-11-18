#!/usr/bin/env python3
import numpy as np
import cv2
import argparse
from pathlib import Path

def fill_small_holes(X, Y, Z, valid_mask, floor_mask, max_iter=2):
    """
    Fill small holes in LUT (X,Y,Z,valid_mask) using local neighbors, but only
    inside floor_mask==1.
    - X,Y,Z: (H,W) float32
    - valid_mask: (H,W) uint8/bool (ground_valid_mask)
    - floor_mask: (H,W) uint8/bool (user-painted floor)
    max_iter: number of refinement iterations (1~2 recommended)
    """
    X = np.asarray(X).copy()
    Y = np.asarray(Y).copy()
    Z = np.asarray(Z).copy()
    valid = np.asarray(valid_mask).astype(bool).copy()
    floor = np.asarray(floor_mask).astype(bool)

    H, W = X.shape
    if Y.shape != (H, W) or Z.shape != (H, W) or valid.shape != (H, W):
        raise ValueError("fill_small_holes: shape mismatch")

    for it in range(max_iter):
        holes = floor & (~valid)
        if not np.any(holes):
            break

        # Accumulate neighbor sums & counts
        sumX = np.zeros_like(X, dtype=np.float64)
        sumY = np.zeros_like(Y, dtype=np.float64)
        sumZ = np.zeros_like(Z, dtype=np.float64)
        cnt  = np.zeros_like(X, dtype=np.int32)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                # source (neighbor) region
                ys_src = slice(max(0, -dy), H - max(0, dy))
                xs_src = slice(max(0, -dx), W - max(0, dx))
                # destination (hole) region
                ys_dst = slice(max(0, dy), H - max(0, -dy))
                xs_dst = slice(max(0, dx), W - max(0, -dx))

                neigh_valid = valid[ys_src, xs_src]
                # positions where current pixel is hole and neighbor is valid
                mask = holes[ys_dst, xs_dst] & neigh_valid
                if not np.any(mask):
                    continue

                sumX_region = sumX[ys_dst, xs_dst]
                sumY_region = sumY[ys_dst, xs_dst]
                sumZ_region = sumZ[ys_dst, xs_dst]
                cnt_region  = cnt[ys_dst, xs_dst]

                sumX_region[mask] += X[ys_src, xs_src][mask]
                sumY_region[mask] += Y[ys_src, xs_src][mask]
                sumZ_region[mask] += Z[ys_src, xs_src][mask]
                cnt_region[mask]  += 1

        fill_mask = holes & (cnt > 0)
        if not np.any(fill_mask):
            break

        X[fill_mask] = (sumX[fill_mask] / cnt[fill_mask]).astype(X.dtype)
        Y[fill_mask] = (sumY[fill_mask] / cnt[fill_mask]).astype(Y.dtype)
        Z[fill_mask] = (sumZ[fill_mask] / cnt[fill_mask]).astype(Z.dtype)
        valid[fill_mask] = True

        filled_count = int(np.sum(fill_mask))
        print(f"[INFO] fill_small_holes iter {it+1}: filled {filled_count} pixels")

    return X, Y, Z, valid.astype(valid_mask.dtype)

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
            # floor_mask 업데이트
            data["floor_mask"] = floor.astype(np.uint8)

            # 선택적으로 LUT의 작은 구멍을 채우기 (X/Y/Z + ground_valid_mask가 있을 때만)
            Xmap = data.get("X", None)
            Ymap = data.get("Y", None)
            Zmap = data.get("Z", None)
            gv   = data.get("ground_valid_mask", None)

            if Xmap is not None and Ymap is not None and Zmap is not None and gv is not None:
                try:
                    X_f, Y_f, Z_f, gv_f = fill_small_holes(Xmap, Ymap, Zmap, gv, floor, max_iter=2)
                    newly_filled = int(np.sum((gv_f.astype(bool)) & (~np.asarray(gv).astype(bool))))
                    data["X"] = X_f
                    data["Y"] = Y_f
                    data["Z"] = Z_f
                    data["ground_valid_mask"] = gv_f
                    print(f"[INFO] hole-fill applied on LUT (newly filled={newly_filled} pixels)")
                except Exception as e:
                    print(f"[WARN] hole-fill skipped due to error: {e}")
            else:
                print("[INFO] X/Y/Z/ground_valid_mask not found in LUT; skip hole-fill")

            # floor_mask와 (필요시 보정된 LUT)를 새 npz로 저장
            np.savez_compressed(str(out_path), **data)
            print(f"[OK] saved: {out_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()