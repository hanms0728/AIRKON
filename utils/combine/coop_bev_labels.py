'''글로벌좌표에 bev labels에 있는 거 동시 투영.   
같은 시점의 14개의 캠에서의 사진 총 100번 반복  
한번 14사진 넣으면 출력 2장 (bev labels만 합친거, region의 정답값도 함께시각화된버전)  
캠별로 색 다르게, region은 빨간색.  
bev맵크기는 -120,-80 a  ~  -30,40 d   

class, 중심 x좌표, 중심 y좌표, 길이, 너비,차량의 방향(단위: degree)'''
import argparse
import glob
import math
import os
import re
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def parse_pred_file(path: str) -> List[dict]:
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) == 6:
                _, cx, cy, L, W, yaw_deg = toks
            elif len(toks) == 5:
                cx, cy, L, W, yaw_deg = toks
            else:
                raise ValueError(f"Unexpected format in {path}: '{line}'")
            labels.append({
                "type": "obb",
                "cx": float(cx),
                "cy": float(cy),
                "l":  float(L),
                "w":  float(W),
                "yaw_deg": float(yaw_deg),
            })
    return labels


def obb_to_quad(cx: float, cy: float, L: float, W: float, yaw_rad: float) -> np.ndarray:
    dx, dy = L / 2.0, W / 2.0  
    corners = np.array([
        [ dx,  dy],
        [ dx, -dy],
        [-dx, -dy],
        [-dx,  dy],
    ])  
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    R = np.array([[c, -s],
                  [s,  c]])
    return corners @ R.T + np.array([cx, cy])


def distinct_colors(n: int):
    cmap = plt.colormaps.get_cmap("tab20")
    return [cmap(i) for i in range(n)]


def make_axes(xlim: Tuple[float, float], ylim: Tuple[float, float], title: str):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    return fig, ax


def draw_quads(ax, quads: List[np.ndarray], edgecolor, label=None, lw=2.0, alpha=0.95):
    for q in quads:
        poly = np.vstack([q, q[0]])
        ax.plot(poly[:, 0], poly[:, 1], color=edgecolor, lw=lw, alpha=alpha)
    if label is not None:
        ax.plot([], [], color=edgecolor, lw=lw, label=label)

def parse_gt_file(path: str) -> List[dict]:
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                print(f"[warn] Skipping malformed line in {path}: {line}")
                continue
            _, cls_name, x, y, z, _, w, l, yaw_deg = parts[:9]
            labels.append({
                "type": "obb",
                "cx": float(x),
                "cy": float(y),
                "l":  float(l),
                "w":  float(w),
                "yaw_deg": float(yaw_deg),
            })
    return labels

# ---------- 메인 ----------
def main():
    p = argparse.ArgumentParser(description="Merge multi-cam BEV labels of the SAME frame into one global plot.")
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--frame", required=True)
    p.add_argument("--gt_dir", default=None)
    p.add_argument("--xlim", nargs=2, type=float, default=[-120.0, -30.0])
    p.add_argument("--ylim", nargs=2, type=float, default=[-80.0, 40.0])
    p.add_argument("--out_prefix", default="global_bev")
    p.add_argument("--bg_path", default=None)
    p.add_argument("--bg_alpha", type=float, default=0.6)
    args = p.parse_args()

    frame_key = args.frame 

    pattern = os.path.join(args.pred_dir, f"cam*_frame_{frame_key}.txt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No prediction files found: {pattern}")

    cam_name_pat = re.compile(r"(cam[^/\\]+)_frame_")
    bev_by_cam: Dict[str, List[dict]] = {}
    for path in paths:
        m = cam_name_pat.search(os.path.basename(path))
        cam = m.group(1) if m else os.path.basename(path)
        bev_by_cam[cam] = parse_pred_file(path)

    cams = sorted(bev_by_cam.keys())
    colors = distinct_colors(len(cams))

    fig1, ax1 = make_axes(tuple(args.xlim), tuple(args.ylim),
                          title=f"Pred-only | frame {frame_key}")
    if args.bg_path is not None and os.path.exists(args.bg_path):
        try:
            bg_img = plt.imread(args.bg_path)
            y_min, y_max = min(args.ylim[0], args.ylim[1]), max(args.ylim[0], args.ylim[1])
            extent = (args.xlim[0], args.xlim[1], y_min, y_max)
            ax1.imshow(bg_img, extent=extent, origin='lower', alpha=args.bg_alpha, zorder=0)
        except Exception:
            pass
    for i, cam in enumerate(cams):
        lab = bev_by_cam[cam]
        quads = []
        for d in lab:
            q = obb_to_quad(d["cx"], d["cy"], d["l"], d["w"],
                            math.radians(d["yaw_deg"]))
            quads.append(q)
        draw_quads(ax1, quads, edgecolor=colors[i], label=cam)
    ax1.legend(loc="upper right", fontsize=9, ncol=2)
    pred_path = f"{args.out_prefix}_{frame_key}_pred.png"
    fig1.tight_layout()
    fig1.savefig(pred_path, dpi=200)
    plt.close(fig1)

    gt_path = None
    gt_labels = []
    if args.gt_dir:
        cand = os.path.join(args.gt_dir, f"region_frame_{frame_key}.txt")
        if os.path.exists(cand):
            gt_path = cand
            gt_labels = parse_gt_file(cand)
        else:
            print(f"[warn] GT file not found: {cand}")

    fig2, ax2 = make_axes(tuple(args.xlim), tuple(args.ylim),
                          title=f"GT(red) + Pred | frame {frame_key}")

    if args.bg_path is not None and os.path.exists(args.bg_path):
        try:
            bg_img = plt.imread(args.bg_path)
            y_min, y_max = min(args.ylim[0], args.ylim[1]), max(args.ylim[0], args.ylim[1])
            extent = (args.xlim[0], args.xlim[1], y_min, y_max)
            ax2.imshow(bg_img, extent=extent, origin='lower', alpha=args.bg_alpha, zorder=0)
        except Exception:
            pass
    if gt_labels:
        gt_quads = []
        for d in gt_labels:
            q = obb_to_quad(d["cx"], d["cy"], d["l"], d["w"],
                            math.radians(d["yaw_deg"]))
            gt_quads.append(q)
        draw_quads(ax2, gt_quads, edgecolor="red", label="GT", lw=2.5, alpha=0.9)

    for i, cam in enumerate(cams):
        lab = bev_by_cam[cam]
        quads = []
        for d in lab:
            q = obb_to_quad(d["cx"], d["cy"], d["l"], d["w"],
                            math.radians(d["yaw_deg"]))
            quads.append(q)
        draw_quads(ax2, quads, edgecolor=colors[i], label=cam)

    ax2.legend(loc="upper right", fontsize=9, ncol=2)
    gt_pred_path = f"{args.out_prefix}_{frame_key}_gt_pred.png"
    fig2.tight_layout()
    fig2.savefig(gt_pred_path, dpi=200)
    plt.close(fig2)

    print("Saved:", pred_path, gt_pred_path)
    if gt_path:
        print("GT used:", gt_path)


if __name__ == "__main__":
    main()
