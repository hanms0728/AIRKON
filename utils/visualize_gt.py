#!/usr/bin/env python3
# visualize_gt.py
# - 데이터셋 GT sanity-check: 이미지에 GT 삼각형 + 평행사변형 덧그리기
# - train.py와 동일 스케일(리사이즈)로 확인

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def parallelogram_from_triangle(p0, p1, p2):
    # 논문 정의: p3 = 2*p0 - p1, p4 = 2*p0 - p2
    p3 = 2 * p0 - p1
    p4 = 2 * p0 - p2
    # 폴리곤 그릴 때 보기좋게 [p1, p2, p3, p4] 순서로
    return np.stack([p1, p2, p3, p4], axis=0).astype(np.int32)

def draw_triangle(img, tri, color=(0, 255, 255), thickness=2):
    # tri: (3,2) float
    tri_i = tri.astype(np.int32)
    cv2.line(img, tuple(tri_i[0]), tuple(tri_i[1]), color, thickness)
    cv2.line(img, tuple(tri_i[1]), tuple(tri_i[2]), color, thickness)
    cv2.line(img, tuple(tri_i[2]), tuple(tri_i[0]), color, thickness)
    # p0에 표시(빨강 점)
    cv2.circle(img, tuple(tri_i[0]), 4, (0, 0, 255), -1)

def draw_parallelogram(img, poly4, color=(0, 255, 0), thickness=2):
    cv2.polylines(img, [poly4], isClosed=True, color=color, thickness=thickness)

def parse_label_line(line):
    # 포맷: "0 p0x p0y p1x p1y p2x p2y"
    parts = line.strip().split()
    if len(parts) != 7:
        return None
    _, p0x, p0y, p1x, p1y, p2x, p2y = parts
    p0 = np.array([float(p0x), float(p0y)], dtype=np.float32)
    p1 = np.array([float(p1x), float(p1y)], dtype=np.float32)
    p2 = np.array([float(p2x), float(p2y)], dtype=np.float32)
    return p0, p1, p2

def main():
    ap = argparse.ArgumentParser(description="Visualize GT triangles & parallelograms")
    ap.add_argument("--root", type=str, required=True,
                    help="데이터 루트 (images/ 와 labels/ 포함)")
    ap.add_argument("--out", type=str, default="./viz_gt",
                    help="시각화 이미지 저장 폴더")
    ap.add_argument("--img-size", type=str, default="1080,1920",
                    help="리사이즈 크기 H,W (train.py와 동일하게)")
    ap.add_argument("--limit", type=int, default=None,
                    help="최대 시각화 개수 (None이면 전체)")
    ap.add_argument("--draw-aabb", action="store_true", help="AABB(바운딩박스)도 표시")
    args = ap.parse_args()

    Ht, Wt = [int(x) for x in args.img_size.split(",")]
    img_dir = os.path.join(args.root, "images")
    lab_dir = os.path.join(args.root, "labels")

    os.makedirs(args.out, exist_ok=True)

    img_names = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if args.limit is not None:
        img_names = img_names[:args.limit]

    print(f"[GT-Viz] total images = {len(img_names)}  | size = (H={Ht}, W={Wt})")
    for name in tqdm(img_names):
        img_path = os.path.join(img_dir, name)
        lab_path = os.path.join(lab_dir, os.path.splitext(name)[0] + ".txt")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f" ! skip (failed to read): {img_path}")
            continue

        H0, W0 = img_bgr.shape[:2]
        sX, sY = Wt / W0, Ht / H0

        # 학습과 동일: 단순 리사이즈
        img_rs = cv2.resize(img_bgr, (Wt, Ht), interpolation=cv2.INTER_LINEAR)

        if not os.path.exists(lab_path):
            # 라벨 없으면 그냥 리사이즈 저장
            cv2.imwrite(os.path.join(args.out, name), img_rs)
            continue

        with open(lab_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        for ln in lines:
            parsed = parse_label_line(ln)
            if parsed is None:
                continue
            p0, p1, p2 = parsed
            # 리사이즈 스케일 적용
            p0_s = np.array([p0[0] * sX, p0[1] * sY], dtype=np.float32)
            p1_s = np.array([p1[0] * sX, p1[1] * sY], dtype=np.float32)
            p2_s = np.array([p2[0] * sX, p2[1] * sY], dtype=np.float32)

            tri = np.stack([p0_s, p1_s, p2_s], axis=0)  # (3,2)
            poly4 = parallelogram_from_triangle(p0_s, p1_s, p2_s)    # (4,2) int

            # 그리기
            draw_triangle(img_rs, tri, color=(0, 255, 255), thickness=2)  # 노랑
            draw_parallelogram(img_rs, poly4, color=(0, 255, 0), thickness=2)  # 초록

            if args.draw_aabb:
                x0, y0 = poly4[:,0].min(), poly4[:,1].min()
                x1, y1 = poly4[:,0].max(), poly4[:,1].max()
                cv2.rectangle(img_rs, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)  # 파랑

        cv2.imwrite(os.path.join(args.out, name), img_rs)

    print(f"[GT-Viz] done. saved to: {args.out}")

if __name__ == "__main__":
    main()