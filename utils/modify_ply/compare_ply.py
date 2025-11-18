#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_ply.py — PLY 포인트클라우드 비교 스크립트
용도:
  두 개의 포인트클라우드(src, tgt)를 비교하여
  RMSE / Chamfer Distance 계산 및 시각화
"""

import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def chamfer_distance(a: np.ndarray, b: np.ndarray):
    """Compute symmetric Chamfer Distance between two point clouds."""
    tree_a, tree_b = cKDTree(a), cKDTree(b)
    dist_a, _ = tree_a.query(b)
    dist_b, _ = tree_b.query(a)
    return float((dist_a.mean() + dist_b.mean()) / 2.0)

def main():
    parser = argparse.ArgumentParser(description="Compare two PLY point clouds")
    parser.add_argument("--src", required=True, help="Source PLY (비교 대상, 예: img_lifted_cloud_stride1.ply)")
    parser.add_argument("--tgt", required=True, help="Target PLY (정답 ground truth)")
    parser.add_argument("--voxel", type=float, default=0.05, help="Downsample voxel size (m)")
    parser.add_argument("--threshold", type=float, default=0.2, help="ICP correspondence threshold (m)")
    args = parser.parse_args()

    print(f"[INFO] Loading point clouds...")
    src = o3d.io.read_point_cloud(args.src)
    tgt = o3d.io.read_point_cloud(args.tgt)

    print(f"[INFO] Downsampling with voxel={args.voxel:.3f}m ...")
    src_ds = src.voxel_down_sample(args.voxel)
    tgt_ds = tgt.voxel_down_sample(args.voxel)

    print(f"[INFO] Aligning centers for initial guess...")
    src_ds.translate(-src_ds.get_center())
    tgt_ds.translate(-tgt_ds.get_center())

    print(f"[INFO] Running ICP registration...")
    reg = o3d.pipelines.registration.registration_icp(
        src_ds, tgt_ds, args.threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    print(f"\n[RESULT]")
    print(f"  - Inlier RMSE     : {reg.inlier_rmse:.4f} m")
    print(f"  - Fitness (inlier ratio): {reg.fitness:.3f}")
    print(f"  - Transformation:\n{reg.transformation}")

    print(f"[INFO] Computing Chamfer Distance...")
    src_pts = np.asarray(src_ds.points)
    tgt_pts = np.asarray(tgt_ds.points)
    cd = chamfer_distance(src_pts, tgt_pts)
    print(f"  - Chamfer Distance : {cd:.4f} m")

    # Apply transformation to visualize aligned result
    src_ds.paint_uniform_color([1, 0, 0])  # red: result
    tgt_ds.paint_uniform_color([0, 1, 0])  # green: ground truth
    src_ds.transform(reg.transformation)

    print(f"[INFO] Visualization window opened.")
    o3d.visualization.draw_geometries(
        [src_ds, tgt_ds],
        window_name="PLY Comparison (Red=Result, Green=Ground Truth)",
        width=1600, height=900, point_show_normal=False
    )

    print(f"[DONE] Comparison complete.")

if __name__ == "__main__":
    main()


"""
python compare_ply.py \
  --src ./out_visible/visible_20251104_002444.ply \
  --tgt ./cloud_rgb_9.ply \
  --voxel 0.05 \
  --threshold 0.2
"""