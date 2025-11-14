#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize a pixel2world_lut.npz valid mask on top of a global PLY.

- Loads the global PLY as the base scene
- Loads LUT (npz) and extracts X/Y/Z arrays
- Applies a chosen mask (ground_valid_mask, valid_mask, or a custom key)
- Builds a point cloud of only the valid samples and overlays it in Open3D

Usage example:
    python pointcloud/make_pointcloud/view_lut_mask_on_ply.py \
        --lut outputs/cam_1_pixel2world_lut.npz \
        --ply pointcloud/global_fused_small.ply \
        --mask-key ground_valid_mask \
        --max-mask-points 200000 \
        --mask-color 0 1 0
"""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path


def pick_mask(lut, key: str = None):
    if key:
        if key not in lut:
            raise KeyError(f"Mask key '{key}' not found in LUT.")
        return lut[key].astype(bool), key
    for cand in ("ground_valid_mask", "valid_mask", "floor_mask"):
        if cand in lut:
            return lut[cand].astype(bool), cand
    X = np.asarray(lut["X"])
    Y = np.asarray(lut["Y"])
    return (np.isfinite(X) & np.isfinite(Y)), "finite(X,Y)"


def build_mask_cloud(X, Y, Z, mask, color, max_points):
    coords = np.stack([X[mask], Y[mask], Z[mask]], axis=1).astype(np.float64)
    if coords.size == 0:
        return None
    if max_points and len(coords) > max_points:
        idx = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[idx]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(coords)
    rgb = np.array(color, dtype=np.float64).reshape(1, 3)
    cols = np.repeat(rgb, len(coords), axis=0)
    cloud.colors = o3d.utility.Vector3dVector(cols)
    return cloud


def parse_args():
    ap = argparse.ArgumentParser("Visualize LUT valid mask points on a PLY")
    ap.add_argument("--lut", required=True, help="pixel2world_lut.npz file")
    ap.add_argument("--ply", required=True, help="Global PLY (world frame matches LUT)")
    ap.add_argument("--mask-key", help="Override mask key (default: ground_valid_mask→valid_mask→floor_mask)")
    ap.add_argument("--max-mask-points", type=int, default=250000,
                    help="Subsample mask points to this number for visualization (0 = no limit)")
    ap.add_argument("--mask-color", type=float, nargs=3, default=(0.0, 1.0, 0.0),
                    help="RGB color (0-1) for mask points (default: green)")
    ap.add_argument("--mask-size", type=float, default=0.02,
                    help="Point size for mask cloud (kept via material shader)")
    ap.add_argument("--global-color", type=float, nargs=3, default=(0.5, 0.5, 0.5),
                    help="Uniform RGB color for the base PLY (default dim gray)")
    ap.add_argument("--global-voxel", type=float, default=0.0,
                    help="Voxel size (meters) to downsample the base PLY for faster rendering")
    ap.add_argument("--show-invalid", action="store_true",
                    help="Also draw invalid LUT points (in red) to inspect mask boundaries")
    ap.add_argument("--max-invalid-points", type=int, default=100000,
                    help="Subsample limit for invalid points (0 = no limit)")
    return ap.parse_args()


def main():
    args = parse_args()
    ply_path = Path(args.ply)
    lut_path = Path(args.lut)
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)
    if not lut_path.is_file():
        raise FileNotFoundError(lut_path)

    print(f"[LOAD] PLY: {ply_path}")
    base_cloud = o3d.io.read_point_cloud(str(ply_path))
    if base_cloud.is_empty():
        raise RuntimeError("Base PLY contains no points.")
    if args.global_voxel and args.global_voxel > 0.0:
        base_cloud = base_cloud.voxel_down_sample(args.global_voxel)
    base_cloud.paint_uniform_color(np.clip(args.global_color, 0.0, 1.0))

    lut = np.load(str(lut_path), allow_pickle=False)
    if not {"X", "Y", "Z"}.issubset(lut.keys()):
        raise KeyError("LUT must contain X, Y, Z arrays.")
    mask, used_key = pick_mask(lut, args.mask_key)
    print(f"[INFO] Using mask '{used_key}' with {mask.sum()} valid pixels.")
    X = np.asarray(lut["X"], dtype=np.float64)
    Y = np.asarray(lut["Y"], dtype=np.float64)
    Z = np.asarray(lut["Z"], dtype=np.float64)
    if X.shape != mask.shape:
        raise ValueError("Mask shape does not match X/Y/Z.")

    mask_cloud = build_mask_cloud(X, Y, Z, mask, args.mask_color, args.max_mask_points)
    if mask_cloud is None:
        raise RuntimeError("No valid mask points to visualize.")

    geometries = [base_cloud, mask_cloud]
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = float(args.mask_size)

    gui = o3d.visualization.gui.Application.instance
    try_gui = hasattr(gui, "initialize")
    use_legacy = (not try_gui) or (not hasattr(gui, "renderer"))

    if use_legacy:
        print("[INFO] Falling back to legacy Visualizer (GUI renderer unavailable).")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="LUT mask on PLY", width=1280, height=720)
        vis.add_geometry(base_cloud)
        vis.add_geometry(mask_cloud)
        invalid_cloud = None
        if args.show_invalid:
            inv = (~mask) & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
            if inv.any():
                invalid_cloud = build_mask_cloud(
                    X,
                    Y,
                    Z,
                    inv,
                    color=(1.0, 0.0, 0.0),
                    max_points=args.max_invalid_points)
                if invalid_cloud is not None:
                    vis.add_geometry(invalid_cloud)
        vis.run()
        vis.destroy_window()
        return

    gui.initialize()
    win = gui.create_window("LUT mask on PLY", 1280, 720)
    scene = o3d.visualization.gui.SceneWidget()
    scene.scene = o3d.visualization.rendering.Open3DScene(gui.renderer)
    scene.scene.add_geometry("base", base_cloud, o3d.visualization.rendering.MaterialRecord())
    scene.scene.add_geometry("mask", mask_cloud, material)

    if args.show_invalid:
        inv = (~mask) & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
        if inv.any():
            invalid_cloud = build_mask_cloud(
                X,
                Y,
                Z,
                inv,
                color=(1.0, 0.0, 0.0),
                max_points=args.max_invalid_points)
            if invalid_cloud is not None:
                material_inv = o3d.visualization.rendering.MaterialRecord()
                material_inv.shader = "defaultUnlit"
                material_inv.point_size = float(max(args.mask_size * 0.8, 1.0))
                scene.scene.add_geometry("invalid", invalid_cloud, material_inv)
                geometries.append(invalid_cloud)

    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(np.vstack([np.asarray(g.points) for g in geometries])))
    scene.setup_camera(60.0, bbox, bbox.get_center())
    scene.scene.show_axes(True)
    win.add_child(scene)
    gui.run()


if __name__ == "__main__":
    main()
