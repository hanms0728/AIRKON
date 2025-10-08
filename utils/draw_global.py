'''
merge_multi_cam_wbf 에서 생성된 텍스트 파일을 시각화 
'''
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 파서
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


# 파서 정답라벨
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


# OBB → 꼭짓점
def obb_to_quad(cx, cy, l, w, yaw_deg):
    yaw = math.radians(yaw_deg)
    dx, dy = l / 2, w / 2
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return corners @ R.T + np.array([cx, cy])


# 시각화 
def draw_boxes(ax, boxes, color, label=None):
    for b in boxes:
        q = obb_to_quad(b["cx"], b["cy"], b["l"], b["w"], b["yaw"])
        q = np.vstack([q, q[0]])  
        ax.plot(q[:, 0], q[:, 1], color=color, linewidth=1.5, alpha=0.9)
    if label:
        ax.plot([], [], color=color, label=label)


def visualize(pred_dir, gt_dir, out_dir,
              xlim=(-120, -30), ylim=(-80, 40)):
    os.makedirs(out_dir, exist_ok=True)
    pred_files = sorted([f for f in os.listdir(pred_dir)
                         if f.startswith("merged_frame_") and f.endswith(".txt")])

    for fname in tqdm(pred_files, desc="Visualizing frames"):
        frame_key = fname.replace("merged_frame_", "").replace(".txt", "")
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, f"region_frame_{frame_key}.txt")

        pred_boxes = parse_pred_file(pred_path)
        gt_boxes = parse_gt_file(gt_path) if os.path.exists(gt_path) else []

        # 사진1 - 예측만 
        fig1, ax1 = plt.subplots(figsize=(7, 7))
        ax1.set_aspect("equal", "box")
        ax1.set_xlim(*xlim)
        ax1.set_ylim(*ylim)
        ax1.grid(True, ls="--", alpha=0.4)
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_title(f"Pred Only | {frame_key}")

        draw_boxes(ax1, pred_boxes, color="blue", label="Pred")
        ax1.legend(loc="upper right", fontsize=9)
        fig1.tight_layout()
        fig1.savefig(os.path.join(out_dir, f"merged_pred_{frame_key}.png"), dpi=200)
        plt.close(fig1)

        # 사진2 - GT + 예측
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        ax2.set_aspect("equal", "box")
        ax2.set_xlim(*xlim)
        ax2.set_ylim(*ylim)
        ax2.grid(True, ls="--", alpha=0.4)
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_title(f"GT (red) + Pred (blue) | {frame_key}")

        if gt_boxes:
            draw_boxes(ax2, gt_boxes, color="red", label="GT")
        draw_boxes(ax2, pred_boxes, color="blue", label="Pred")

        ax2.legend(loc="upper right", fontsize=9)
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, f"merged_gt_pred_{frame_key}.png"), dpi=200)
        plt.close(fig2)


if __name__ == "__main__":
    pred_dir = "/merge_cam_wbf_dist"
    gt_dir = "/val_dataset45/regions"
    out_dir = "/draw_global_dist"

    visualize(pred_dir, gt_dir, out_dir)
