#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LUT npz + 이미지로 floor_mask 시각적으로 확인하는 뷰어
- 원본 이미지 / 마스크 오버레이 / 마스크만 보기 모드 지원
- ESC: 종료
- 1: 원본만
- 2: floor_mask 오버레이
- 3: floor_mask만 (그레이)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser("floor_mask viewer (image + lut.npz)")
    ap.add_argument("--lut", required=True, help="LUT npz 경로")
    ap.add_argument("--image", required=True, help="LUT 기준 언디스토트된 이미지 (jpg/png)")
    ap.add_argument("--mask-key", default="floor_mask",
                    help="시각화할 마스크 키 (기본: floor_mask, 필요시 ground_valid_mask 등)")
    args = ap.parse_args()

    lut_path = Path(args.lut)
    if not lut_path.is_file():
        raise SystemExit(f"[ERR] LUT not found: {lut_path}")

    data = dict(np.load(str(lut_path), allow_pickle=False))

    H = int(data.get("height"))
    W = int(data.get("width"))
    if H <= 0 or W <= 0:
        raise SystemExit(f"[ERR] invalid (height,width) in LUT: ({H},{W})")

    # ---- 마스크 로드 ----
    mask = data.get(args.mask_key, None)
    if mask is None:
        raise SystemExit(f"[ERR] '{args.mask_key}' not found in LUT: keys = {list(data.keys())}")
    mask = mask.astype(np.uint8)
    if mask.shape != (H, W):
        raise SystemExit(f"[ERR] mask shape {mask.shape} != ({H},{W})")

    print(f"[INFO] LUT: {lut_path.name}")
    print(f"       size      : {W} x {H}")
    print(f"       mask key  : {args.mask_key}")
    print(f"       mask sum  : {int(mask.sum())} (non-zero count={int((mask>0).sum())})")

    # ---- 이미지 로드 ----
    img_path = Path(args.image)
    if not img_path.is_file():
        raise SystemExit(f"[ERR] image not found: {img_path}")

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"[ERR] failed to read image: {img_path}")
    if img.shape[:2] != (H, W):
        raise SystemExit(
            f"[ERR] image shape {img.shape[:2]} != LUT size ({H},{W}). "
            "언디스토트/리사이즈가 LUT 만들 때와 같은지 확인 필요."
        )

    # ---- 뷰 모드 3가지 ----
    MODE_ORIG = 1
    MODE_OVERLAY = 2
    MODE_MASK_ONLY = 3
    mode = MODE_OVERLAY

    def make_overlay():
        """원본 + 마스크 오버레이."""
        base = img.copy()
        overlay = img.copy()

        # 마스크 True인 곳에 색 입히기 (노란색 계열)
        color = np.array([0, 255, 255], dtype=np.uint8)  # BGR
        overlay[mask > 0] = (overlay[mask > 0] * 0.3 + color * 0.7).astype(np.uint8)

        alpha = 0.55
        vis = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

        txt = f"MODE: overlay('{args.mask_key}') | 1:orig  2:overlay  3:mask only  ESC:quit"
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return vis

    def make_mask_only():
        """마스크만 흑백/컬러로 보기."""
        # 흑백 이미지로 만들기
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        vis = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        txt = f"MODE: mask only('{args.mask_key}') | 1:orig  2:overlay  3:mask only  ESC:quit"
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return vis

    def make_orig():
        vis = img.copy()
        txt = f"MODE: original | 1:orig  2:overlay('{args.mask_key}')  3:mask only  ESC:quit"
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return vis

    def render():
        if mode == MODE_ORIG:
            return make_orig()
        elif mode == MODE_OVERLAY:
            return make_overlay()
        elif mode == MODE_MASK_ONLY:
            return make_mask_only()
        else:
            return make_overlay()

    win = "floor_mask viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # 적당히 축소해서 보기 (너무 크면)
    scale = 0.7 if max(W, H) > 1200 else 1.0
    cv2.resizeWindow(win, int(W * scale), int(H * scale))

    vis = render()
    cv2.imshow(win, vis)

    print("[INFO] 키 조작:")
    print("  1: 원본만 보기")
    print("  2: 마스크 오버레이 보기")
    print("  3: 마스크만 보기")
    print("  ESC: 종료")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('1'):
            mode = MODE_ORIG
            vis = render()
            cv2.imshow(win, vis)
        elif key == ord('2'):
            mode = MODE_OVERLAY
            vis = render()
            cv2.imshow(win, vis)
        elif key == ord('3'):
            mode = MODE_MASK_ONLY
            vis = render()
            cv2.imshow(win, vis)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()