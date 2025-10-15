import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
'''
기능: /merge_dist_wbf 의 예측 결과와 /val_dataset45/regions 의 GT를 불러와서 글로벌 좌표계에 맞게 시각화
출력: /draw_global_dist_wbf/merged_pred_xxxxxx.png(예측만), merged_gt_pred_xxxxxx.png(GT+예측)
'''
def parse_pred_file(path):
    """예측 라벨: class cx cy l w yaw_deg"""
    boxes = []
    with open(path, "r") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) != 6:
                continue
            _, cx, cy, l, w, yaw = vals
            boxes.append({
                "cx": float(cx),
                "cy": float(cy),
                "l": float(l),
                "w": float(w),
                "yaw": float(yaw)
            })
    return boxes

def parse_gt_file(path):
    """region GT 파일: CSV 포맷"""
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            _, cls_name, x, y, z, _, w, l, yaw_deg = parts[:9]
            boxes.append({
                "cx": float(x),
                "cy": float(y),
                "l": float(l),
                "w": float(w),
                "yaw": float(yaw_deg)
            })
    return boxes

def obb_to_quad(cx, cy, l, w, yaw_deg):
    yaw = math.radians(yaw_deg)
    dx, dy = l / 2, w / 2
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return corners @ R.T + np.array([cx, cy])

def draw_boxes(ax, boxes, color, label=None):
    for b in boxes:
        q = obb_to_quad(b["cx"], b["cy"], b["l"], b["w"], b["yaw"])
        q = np.vstack([q, q[0]])  # 닫힌 형태
        ax.plot(q[:, 0], q[:, 1], color=color, linewidth=1.5, alpha=0.9)
    if label:
        ax.plot([], [], color=color, label=label)


def visualize(pred_dir, gt_dir, out_dir,
              xlim=(-120, -30), ylim=(40, -80),
              bg_path: str = None,
              bg_alpha: float = 0.6,
              bg_pixels: tuple = (690, 920),
              dpi: int = 100):
    os.makedirs(out_dir, exist_ok=True)
    pred_files = sorted([f for f in os.listdir(pred_dir)
                         if f.startswith("merged_frame_") and f.endswith(".txt")])

    for fname in tqdm(pred_files, desc="Visualizing frames"):
        frame_key = fname.replace("merged_frame_", "").replace(".txt", "")
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, f"region_frame_{frame_key}.txt")

        pred_boxes = parse_pred_file(pred_path)
        gt_boxes = parse_gt_file(gt_path) if os.path.exists(gt_path) else []

        # ----- (1) 예측만 -----
        fig_w = bg_pixels[0] / float(dpi)
        fig_h = bg_pixels[1] / float(dpi)
        fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax1.set_aspect("equal", "box")
        ax1.set_xlim(*xlim)
        ax1.set_ylim(*ylim)
        ax1.grid(True, ls="--", alpha=0.4)
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_title(f"Pred Only | {frame_key}")

        # 배경이미지 
        if bg_path is not None and os.path.exists(bg_path):
            try:
                bg_img = plt.imread(bg_path)
                y_min, y_max = min(ylim[0], ylim[1]), max(ylim[0], ylim[1])
                extent = (xlim[0], xlim[1], y_min, y_max)
                ax1.imshow(bg_img, extent=extent, origin='lower', alpha=bg_alpha, zorder=0)
            except Exception:
                pass

        draw_boxes(ax1, pred_boxes, color="blue", label="Pred")
        ax1.legend(loc="upper right", fontsize=9)
        fig1.tight_layout()
        fig1.savefig(os.path.join(out_dir, f"merged_pred_{frame_key}.png"), dpi=200)
        plt.close(fig1)

        # ----- (2) GT + 예측 -----
        fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax2.set_aspect("equal", "box")
        ax2.set_xlim(*xlim)
        ax2.set_ylim(*ylim)
        ax2.grid(True, ls="--", alpha=0.4)
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_title(f"GT (red) + Pred (blue) | {frame_key}")

        if bg_path is not None and os.path.exists(bg_path):
            try:
                bg_img = plt.imread(bg_path)
                y_min, y_max = min(ylim[0], ylim[1]), max(ylim[0], ylim[1])
                extent = (xlim[0], xlim[1], y_min, y_max)
                ax2.imshow(bg_img, extent=extent, origin='lower', alpha=bg_alpha, zorder=0)
            except Exception:
                pass

        if gt_boxes:
            draw_boxes(ax2, gt_boxes, color="red", label="GT")
        draw_boxes(ax2, pred_boxes, color="blue", label="Pred")

        ax2.legend(loc="upper right", fontsize=9)
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, f"merged_gt_pred_{frame_key}.png"), dpi=200)
        plt.close(fig2)


if __name__ == "__main__":
    pred_dir = "/merge_dist_wbf"
    gt_dir = "/val_dataset45/regions"
    out_dir = "/draw_global_dist_wbf"
    bg_path = "/칼라맵.png"

    visualize(pred_dir, gt_dir, out_dir, bg_path=bg_path)
