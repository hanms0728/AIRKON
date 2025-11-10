#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge multiple ASCII PLY point clouds (x y z [r g b]) into a single PLY.

- 입력 PLY:
  * ASCII PLY만 지원 (binary는 미지원)
  * vertex만 있는 단순 포맷 가정
  * 컬러는 없거나(r,g,b 없음) 또는 uchar 0..255 혹은 float 0..1 모두 허용

- 옵션:
  --voxel s       : 격자 다운샘플(미터 단위). s>0이면 동일 격자 칸 포인트들을 평균 병합
  --dedup-eps e   : 완전중복/초근접 포인트 제거 (거리 e 이하를 동일 포인트로 간주, voxel 후에 적용 권장)

출력:
  - ASCII PLY (x y z r g b). 컬러가 없는 입력은 흰색(255,255,255)로 출력
"""

import os
import sys
import glob
import math
import argparse
import numpy as np

# ---------------------------
# PLY I/O (ASCII 전용)
# ---------------------------

def read_ascii_ply(path):
    """
    Return: (pts Nx3 float64, cols Nx3 float64 in 0..1 or None)
    Accepts vertex properties with flexible order. Requires x,y,z. Optional r,g,b or red,green,blue.
    """
    with open(path, "r") as f:
        header = []
        line = f.readline().strip()
        if line != "ply":
            raise ValueError(f"[{path}] not a PLY file")
        header.append(line)

        format_line = f.readline().strip()
        header.append(format_line)
        if not format_line.startswith("format ascii"):
            raise ValueError(f"[{path}] only ASCII PLY supported (got: {format_line})")

        # Parse header
        n_vertices = None
        prop_names = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"[{path}] unexpected EOF while reading header")
            s = line.strip()
            header.append(s)
            if s.startswith("element vertex"):
                n_vertices = int(s.split()[-1])
            elif s.startswith("property"):
                # e.g., "property float x" or "property uchar red"
                toks = s.split()
                name = toks[-1]
                prop_names.append(name)
            elif s == "end_header":
                break

        if n_vertices is None:
            raise ValueError(f"[{path}] no 'element vertex' in header")

        # Read body
        data = []
        for _ in range(n_vertices):
            row = f.readline()
            if not row:
                break
            data.append(row.strip().split())

    if len(data) != n_vertices:
        raise ValueError(f"[{path}] vertex count mismatch (expect {n_vertices}, got {len(data)})")

    data = np.array(data, dtype=np.float64)  # 숫자만 있다고 가정
    # Locate columns
    def idx(name_candidates):
        for n in name_candidates:
            if n in prop_names:
                return prop_names.index(n)
        return -1

    ix = idx(["x"])
    iy = idx(["y"])
    iz = idx(["z"])
    if min(ix,iy,iz) < 0:
        raise ValueError(f"[{path}] x/y/z properties not found; got {prop_names}")

    # color can be red/green/blue or r/g/b
    ir = idx(["red","r"])
    ig = idx(["green","g"])
    ib = idx(["blue","b"])

    xyz = data[:, [ix,iy,iz]].astype(np.float64)

    cols = None
    if ir >= 0 and ig >= 0 and ib >= 0:
        cols_raw = data[:, [ir,ig,ib]].astype(np.float64)
        # Heuristic: if any value >1.0 we assume 0..255, else 0..1
        if (cols_raw > 1.0).any():
            cols = np.clip(cols_raw / 255.0, 0.0, 1.0)
        else:
            cols = np.clip(cols_raw, 0.0, 1.0)

    return xyz, cols

def write_ascii_ply(path, pts_xyz, cols_rgb01=None):
    """
    Write ASCII PLY with x y z r g b (uchar 0..255).
    If cols_rgb01 is None, use white.
    """
    N = int(pts_xyz.shape[0])
    if cols_rgb01 is None:
        cols_u8 = np.full((N,3), 255, dtype=np.uint8)
    else:
        cols_u8 = np.clip(cols_rgb01 * 255.0, 0, 255).astype(np.uint8)

    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]) + "\n"

    with open(path, "w") as f:
        f.write(header)
        for (x,y,z), (r,g,b) in zip(pts_xyz, cols_u8):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

# ---------------------------
# Merging & Post-process
# ---------------------------

def voxel_downsample(xyz, rgb01, voxel):
    """
    Grid-based downsampling:
      - key = floor(coord/voxel)
      - aggregate: average xyz & colors per cell
    Returns: (xyz_ds, rgb_ds)
    """
    if voxel <= 0:
        return xyz, rgb01

    keys = np.floor(xyz / voxel).astype(np.int64)
    keys = np.ascontiguousarray(keys)
    # Hashing keys
    # Use structured array as dict key
    key_view = keys.view([('kx', np.int64), ('ky', np.int64), ('kz', np.int64)]).reshape(-1)

    # group by key
    unique_keys, inv = np.unique(key_view, return_inverse=True)
    cnt = np.bincount(inv)

    xyz_sum = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
    np.add.at(xyz_sum, inv, xyz)

    if rgb01 is None:
        rgb_sum = None
        rgb_out = None
    else:
        rgb_sum = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
        np.add.at(rgb_sum, inv, rgb01)

    xyz_out = xyz_sum / cnt[:, None]
    if rgb_sum is not None:
        rgb_out = rgb_sum / cnt[:, None]

    return xyz_out, rgb_out

def dedup_by_eps(xyz, rgb01, eps):
    """
    Remove near-duplicate points within distance <= eps.
    Simple grid binning on eps, then keep first (or mean).
    Here we keep the first occurrence per cell to be fast.
    """
    if eps <= 0:
        return xyz, rgb01

    keys = np.floor(xyz / eps).astype(np.int64)
    keys = np.ascontiguousarray(keys)
    key_view = keys.view([('kx', np.int64), ('ky', np.int64), ('kz', np.int64)]).reshape(-1)

    # first occurrence indices per cell
    _, idx = np.unique(key_view, return_index=True)
    idx.sort()
    if rgb01 is None:
        return xyz[idx], None
    else:
        return xyz[idx], rgb01[idx]

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser("Merge multiple ASCII PLYs (x y z [r g b])")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Input PLY paths (globs allowed). Example: ./out/*.ply ./cam2/*.ply")
    ap.add_argument("--out", required=True, help="Output merged PLY path")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="Voxel size (meters) for grid downsampling (default: 0=off)")
    ap.add_argument("--dedup-eps", type=float, default=0.0,
                    help="Duplicate-merge distance eps (meters). Applied after voxel (default: 0=off)")
    args = ap.parse_args()

    # Expand globs / files / directories
    paths = []
    unmatched = []
    for pat in args.inputs:
        # If it's a directory, take all .ply inside
        if os.path.isdir(pat):
            found = glob.glob(os.path.join(pat, "*.ply"))
            if not found:
                unmatched.append(pat + " (empty dir or no .ply)")
            else:
                paths.extend(found)
            continue

        # Glob patterns
        g = glob.glob(pat)
        if g:
            paths.extend(g)
            continue

        # Exact file path
        if os.path.isfile(pat):
            paths.append(pat)
        else:
            unmatched.append(pat)

    # De-duplicate & sort
    paths = sorted(set(paths))

    # Diagnostics
    if unmatched:
        print("[WARN] The following inputs matched nothing:", file=sys.stderr)
        for u in unmatched:
            print(f"  - {u}", file=sys.stderr)

    if not paths:
        print("No input PLYs found. (All patterns unmatched)", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Matched input files:")
    for p in paths:
        print(f"  - {p}")

    all_xyz = []
    all_rgb = []
    total_pts = 0

    print("[INFO] Reading PLYs...")
    for p in paths:
        xyz, col = read_ascii_ply(p)
        total_pts += xyz.shape[0]
        all_xyz.append(xyz)
        all_rgb.append(col)
        print(f"  - {os.path.basename(p)} : {xyz.shape[0]:,} pts "
              f"{'(color)' if col is not None else '(no color)'}")

    # Concatenate
    xyz = np.vstack(all_xyz)
    if any(c is None for c in all_rgb):
        # If any file has no color, fill white for those; else concatenate
        cols_list = []
        for c, x in zip(all_rgb, all_xyz):
            if c is None:
                cols_list.append(np.ones((x.shape[0],3), dtype=np.float64))
            else:
                cols_list.append(c)
        rgb = np.vstack(cols_list)
    else:
        rgb = np.vstack(all_rgb)

    print(f"[INFO] Total before merge: {total_pts:,}  → concatenated: {xyz.shape[0]:,}")

    # Optional voxel downsample
    if args.voxel > 0:
        xyz, rgb = voxel_downsample(xyz, rgb, args.voxel)
        print(f"[INFO] After voxel({args.voxel} m): {xyz.shape[0]:,}")

    # Optional dedup by eps
    if args.dedup_eps > 0:
        xyz, rgb = dedup_by_eps(xyz, rgb, args.dedup_eps)
        print(f"[INFO] After dedup-eps({args.dedup_eps} m): {xyz.shape[0]:,}")

    # Write
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_ascii_ply(args.out, xyz, rgb)
    print("\n[DONE]")
    print(f"  Inputs : {len(paths)} files")
    print(f"  Output : {args.out}")
    print(f"  Points : {xyz.shape[0]:,}")

if __name__ == "__main__":
    main()