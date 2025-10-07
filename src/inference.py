# infer.py
# 학습된 yolo11_2p5d *.pth 로드 → 디코딩 → 시각화/라벨 저장
# (+선택: GT와 함께 시각화 & IoU 표기, +선택: 2D 성능지표 계산)

import os
import cv2
import math
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from ultralytics import YOLO
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.lines import Line2D
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MplPolygon = None
    Line2D = None
    _MATPLOTLIB_AVAILABLE = False

# 학습 코드와 동일한 유틸 (반드시 train과 동일 파일이어야 함)
from geometry_utils import parallelogram_from_triangle
from evaluation_utils import (
    decode_predictions,
    evaluate_single_image,
    compute_detection_metrics,
    orientation_error_deg,
)

# ---------------------------
# Model (train.py와 동일)
# ---------------------------
class TriHead(nn.Module):
    def __init__(self, in_ch, nc, prior_p=0.20):
        super().__init__()
        self.reg = nn.Conv2d(in_ch, 6, kernel_size=1, stride=1, padding=0)
        self.obj = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0)
        self.cls = nn.Conv2d(in_ch, nc, kernel_size=1, stride=1, padding=0)

        prior_b = math.log(prior_p/(1-prior_p))
        nn.init.constant_(self.obj.bias, prior_b)
        nn.init.constant_(self.cls.bias, prior_b)
        nn.init.zeros_(self.reg.weight)
        nn.init.zeros_(self.reg.bias)

    def forward(self, x):
        return self.reg(x), self.obj(x), self.cls(x)

class YOLO11_2p5D(nn.Module):
    def __init__(self, yolo11_path='yolo11m.pt', num_classes=1, img_size=(864,1536)):
        super().__init__()
        base = YOLO(yolo11_path).model
        self.backbone_neck = base.model[:-1]
        self.save_idx = base.save
        detect = base.model[-1]
        self.strides = detect.stride
        self.f_indices = detect.f

        # in_channels 추출
        self.backbone_neck.eval()
        with torch.no_grad():
            dummy = torch.zeros(1,3,img_size[0],img_size[1])
            feats_memory = []
            x = dummy
            for m in self.backbone_neck:
                if m.f != -1:
                    x = feats_memory[m.f] if isinstance(m.f, int) else [x if j== -1 else feats_memory[j] for j in m.f]
                x = m(x)
                feats_memory.append(x if m.i in self.save_idx else None)
            feat_list = [feats_memory[i] for i in self.f_indices]
            in_chs = [f.shape[1] for f in feat_list]
        self.backbone_neck.train()

        self.heads = nn.ModuleList([TriHead(c, num_classes) for c in in_chs])
        self.num_classes = num_classes

    def forward(self, x):
        feats_memory = []
        for m in self.backbone_neck:
            if m.f != -1:
                x = feats_memory[m.f] if isinstance(m.f, int) else [x if j== -1 else feats_memory[j] for j in m.f]
            x = m(x)
            feats_memory.append(x if m.i in self.save_idx else None)
        feat_list = [feats_memory[i] for i in self.f_indices]
        outs = [head(f) for head, f in zip(self.heads, feat_list)]
        return outs  # list of (reg,obj,cls)

# ---------------------------
# GT 로더 (원본 해상도 기준 라벨 가정: "cls p0x p0y p1x p1y p2x p2y")
# ---------------------------
def load_gt_triangles(label_path: str) -> np.ndarray:
    if not os.path.isfile(label_path):
        return np.zeros((0,3,2), dtype=np.float32)
    tris = []
    with open(label_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 7:
                continue
            _, p0x, p0y, p1x, p1y, p2x, p2y = p
            tris.append([[float(p0x), float(p0y)],
                         [float(p1x), float(p1y)],
                         [float(p2x), float(p2y)]])
    if len(tris) == 0:
        return np.zeros((0,3,2), dtype=np.float32)
    return np.asarray(tris, dtype=np.float32)

# ---------------------------
# 기하 유틸: 폴리곤 IoU (우선 정확: convex-intersection, 실패 시 AABB로 폴백)
# ---------------------------
def poly_from_tri(tri: np.ndarray) -> np.ndarray:
    """tri(3,2) → 평행사변형(4,2) float32"""
    p0, p1, p2 = tri[0], tri[1], tri[2]
    return parallelogram_from_triangle(p0, p1, p2).astype(np.float32)

def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:,0], poly[:,1]
    return float(abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1))) * 0.5)

def iou_polygon(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """OpenCV intersectConvexConvex 사용, 실패 시 AABB IoU."""
    try:
        pa = poly_a.astype(np.float32)
        pb = poly_b.astype(np.float32)
        inter_area, _ = cv2.intersectConvexConvex(pa, pb)
        if inter_area <= 0:
            return 0.0
        ua = polygon_area(pa)
        ub = polygon_area(pb)
        union = ua + ub - inter_area
        return float(inter_area / max(union, 1e-9))
    except Exception:
        xa1, ya1 = poly_a[:,0].min(), poly_a[:,1].min()
        xa2, ya2 = poly_a[:,0].max(), poly_a[:,1].max()
        xb1, yb1 = poly_b[:,0].min(), poly_b[:,1].min()
        xb2, yb2 = poly_b[:,0].max(), poly_b[:,1].max()
        inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
        inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
        inter = inter_w * inter_h
        ua = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        ub = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union = ua + ub - inter
        return float(inter / max(union, 1e-9))

# ---------------------------
# Inference 시각화
# ---------------------------
def draw_pred_only(image_bgr, dets, save_path_img, save_path_txt, W, H, W0, H0):
    """
    image_bgr: (H, W)로 리사이즈된 BGR 이미지 (시각화용)
    dets: [{"tri":(3,2), "score":float}, ...]  # (H, W) 좌표계
    save_path_img: 시각화 이미지 저장 경로 (리사이즈 좌표계로 그림)
    save_path_txt: 라벨 저장 경로 (==> 원본 해상도 좌표계로 저장)
    W,H:   모델 입력(리사이즈) 가로/세로
    W0,H0: 원본 해상도 가로/세로
    """
    os.makedirs(os.path.dirname(save_path_img), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_txt), exist_ok=True)

    # 시각화는 기존처럼 (리사이즈 좌표계)로 그림
    img = image_bgr.copy()

    # 스케일 팩터: (리사이즈 → 원본)
    sx, sy = float(W0) / float(W), float(H0) / float(H)

    lines = []
    tri_orig_list: List[np.ndarray] = []
    for d in dets:
        tri = np.asarray(d["tri"], dtype=np.float32)  # (3,2) in (H,W)
        score = float(d["score"])

        # 시각화(리사이즈 기준)
        p0, p1, p2 = tri[0], tri[1], tri[2]
        poly4 = parallelogram_from_triangle(p0, p1, p2).astype(np.int32)
        cv2.polylines(img, [poly4], isClosed=True, color=(0,255,0), thickness=2)
        cx, cy = int(p0[0]), int(p0[1])
        cv2.putText(img, f"{score:.2f}", (cx, max(0, cy-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # === 라벨(.txt)은 원본 해상도 좌표계로 저장 ===
        tri_orig = tri.copy()
        tri_orig[:, 0] *= sx
        tri_orig[:, 1] *= sy
        tri_orig_list.append(tri_orig.copy())
        p0o, p1o, p2o = tri_orig[0], tri_orig[1], tri_orig[2]
        lines.append(
            f"0 {p0o[0]:.2f} {p0o[1]:.2f} {p1o[0]:.2f} {p1o[1]:.2f} {p2o[0]:.2f} {p2o[1]:.2f} {score:.4f}"
        )

    # 저장
    cv2.imwrite(save_path_img, img)
    with open(save_path_txt, "w") as f:
        f.write("\n".join(lines))

    return tri_orig_list

def draw_pred_with_gt(image_bgr_resized, dets, gt_tris_resized, save_path_img_mix, iou_thr=0.5):
    """
    - dets: [{"tri":(3,2), "score":float}, ...]  (리사이즈 좌표계)
    - gt_tris_resized: (Ng,3,2)  (리사이즈 좌표계로 스케일된 GT)
    결과: pred(빨강), GT(초록), 점수/IoU 텍스트 색상 구분
    """
    os.makedirs(os.path.dirname(save_path_img_mix), exist_ok=True)
    img = image_bgr_resized.copy()

    # 1) GT 박스 & 점 (초록)
    for g in gt_tris_resized:
        poly_g = poly_from_tri(g).astype(np.int32)
        cv2.polylines(img, [poly_g], True, (0,255,0), 2)   # green polygon
        for k in range(3):
            x = int(round(float(g[k,0])))
            y = int(round(float(g[k,1])))
            cv2.circle(img, (x, y), 3, (0,255,0), -1)      # green dots

    # 2) Pred 박스 & 점수 (빨강 박스, 점수는 초록 글씨)
    for d in dets:
        tri = np.asarray(d["tri"], dtype=np.float32)
        score = float(d["score"])
        poly4 = poly_from_tri(tri).astype(np.int32)
        cv2.polylines(img, [poly4], True, (0,0,255), 2)   # red polygon
        p0 = tri[0].astype(int)
        cv2.putText(img, f"{score:.2f}",
                    (int(p0[0]), max(0, int(p0[1])-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,0), 1, cv2.LINE_AA)             # green text

    # 3) IoU 매칭 결과 (IoU는 빨간색 글씨)
    if gt_tris_resized.shape[0] > 0 and len(dets) > 0:
        det_idx_sorted = np.argsort([-float(d["score"]) for d in dets])
        matched = np.zeros((gt_tris_resized.shape[0],), dtype=bool)
        for di in det_idx_sorted:
            d = dets[di]
            tri_d = np.asarray(d["tri"], dtype=np.float32)
            poly_d = poly_from_tri(tri_d)

            best_j, best_iou = -1, 0.0
            for j, gtri in enumerate(gt_tris_resized):
                if matched[j]:
                    continue
                poly_g = poly_from_tri(gtri)
                iou = iou_polygon(poly_d, poly_g)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_j >= 0:
                matched[best_j] = True
                p0 = tri_d[0].astype(int)
                cv2.putText(img, f"IoU {best_iou:.2f}",
                            (int(p0[0]), int(p0[1]) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,0,255), 1, cv2.LINE_AA)     # red text

    cv2.imwrite(save_path_img_mix, img)


def normalize_angle_deg(angle: float) -> float:
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.copy()
    flat = pts.reshape(-1, 2)
    ones = np.ones((flat.shape[0], 1), dtype=np.float64)
    homog = np.concatenate([flat, ones], axis=1)
    proj = homog @ H.T
    denom = proj[:, 2:3]
    result = np.full_like(proj[:, :2], np.nan, dtype=np.float64)
    valid = np.abs(denom[:, 0]) > 1e-9
    if np.any(valid):
        result[valid] = proj[valid, :2] / denom[valid]
    return result.reshape(pts.shape)


def load_homography(calib_dir: str, image_name: str, cache: dict, invert: bool = False) -> Optional[np.ndarray]:
    base = os.path.splitext(os.path.basename(image_name))[0]
    if base in cache:
        return cache[base]

    search_order = [base + ext for ext in (".txt", ".npy", ".csv")]
    H = None
    for candidate in search_order:
        c_path = Path(calib_dir) / candidate
        if c_path.is_file():
            H = _read_h_matrix(c_path)
            break

    if H is None:
        matches = sorted(Path(calib_dir).glob(base + ".*"))
        for c_path in matches:
            H = _read_h_matrix(c_path)
            if H is not None:
                break

    if H is not None and invert:
        try:
            H = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H = None

    cache[base] = H
    return H


def _read_h_matrix(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix.lower() == ".npy":
            data = np.load(path)
        else:
            data = np.loadtxt(path)
    except Exception:
        return None

    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 9:
        arr = arr.reshape(3, 3)
    if arr.shape != (3, 3):
        return None
    return arr


def compute_bev_properties(poly4: np.ndarray) -> Optional[Tuple[Tuple[float, float], float, float, float]]:
    poly = np.asarray(poly4, dtype=np.float64)
    if poly.shape != (4, 2) or not np.all(np.isfinite(poly)):
        return None

    edges = [np.linalg.norm(poly[(i + 1) % 4] - poly[i]) for i in range(2)]
    if edges[0] < 1e-6 or edges[1] < 1e-6:
        return None
    if edges[0] >= edges[1]:
        length = edges[0]
        width = edges[1]
        vec = poly[1] - poly[0]
    else:
        length = edges[1]
        width = edges[0]
        vec = poly[2] - poly[1]
    if np.linalg.norm(vec) < 1e-6:
        vec = poly[1] - poly[0]
    yaw = math.degrees(math.atan2(vec[1], vec[0]))
    yaw = normalize_angle_deg(yaw)
    center = poly.mean(axis=0)
    return (float(center[0]), float(center[1])), float(length), float(width), float(yaw)


def write_bev_labels(save_path: str, bev_dets: List[dict]):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lines = []
    for det in bev_dets:
        cx, cy = det["center"]
        length = det["length"]
        width = det["width"]
        yaw = det["yaw"]
        lines.append(f"0 {cx:.4f} {cy:.4f} {length:.4f} {width:.4f} {yaw:.2f}")
    with open(save_path, "w") as f:
        f.write("\n".join(lines))


def _prepare_bev_canvas(polygons: List[np.ndarray], padding: float = 1.0, target: float = 800.0):
    if not polygons:
        return None
    pts = np.concatenate(polygons, axis=0)
    if pts.size == 0:
        return None
    min_x = float(np.nanmin(pts[:, 0]) - padding)
    max_x = float(np.nanmax(pts[:, 0]) + padding)
    min_y = float(np.nanmin(pts[:, 1]) - padding)
    max_y = float(np.nanmax(pts[:, 1]) + padding)
    range_x = max(max_x - min_x, 1e-3)
    range_y = max(max_y - min_y, 1e-3)
    scale = target / max(range_x, range_y)
    width = int(max(range_x * scale, 300))
    height = int(max(range_y * scale, 300))
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "scale": scale,
        "width": width,
        "height": height,
    }


def _to_canvas(points: np.ndarray, params: dict) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    xs = (pts[:, 0] - params["min_x"]) * params["scale"]
    ys = (params["max_y"] - pts[:, 1]) * params["scale"]
    return np.stack([xs, ys], axis=1).astype(np.int32)


def draw_bev_visualization(
    preds_bev: List[dict],
    gt_tris_bev: Optional[np.ndarray],
    save_path_img: str,
    title: str,
):
    pred_polys = [det["poly"] for det in preds_bev]
    gt_polys = [poly_from_tri(tri) for tri in gt_tris_bev] if gt_tris_bev is not None else []
    polygons = pred_polys + gt_polys

    params = _prepare_bev_canvas(polygons)
    os.makedirs(os.path.dirname(save_path_img), exist_ok=True)

    if params is None:
        canvas = np.full((360, 360, 3), 240, dtype=np.uint8)
        cv2.putText(canvas, "No BEV data", (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(save_path_img, canvas)
        return

    if _MATPLOTLIB_AVAILABLE:
        # Matplotlib 렌더링 (plt 스타일)
        try:
            fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=220)
            ax.set_facecolor("#f7f7f7")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title, fontsize=11, pad=10)
            ax.set_xlabel("X (m)", fontsize=10)
            ax.set_ylabel("Y (m)", fontsize=10)
            ax.set_xlim(params["min_x"], params["max_x"])
            ax.set_ylim(params["max_y"], params["min_y"])

            if params["min_x"] <= 0.0 <= params["max_x"]:
                ax.axvline(0.0, color="#999999", linewidth=0.9, linestyle="--", alpha=0.8)
            if params["min_y"] <= 0.0 <= params["max_y"]:
                ax.axhline(0.0, color="#999999", linewidth=0.9, linestyle="--", alpha=0.8)

            legend_handles = []

            if gt_polys:
                for poly in gt_polys:
                    if not np.all(np.isfinite(poly)):
                        continue
                    patch = MplPolygon(poly, closed=True, fill=False, edgecolor="#27ae60", linewidth=1.8)
                    ax.add_patch(patch)
                    ax.scatter(poly[:, 0], poly[:, 1], s=8, color="#27ae60", alpha=0.9)
                legend_handles.append(Line2D([0], [0], color="#27ae60", lw=2, label="GT"))

            if preds_bev:
                dy = max(params["max_y"] - params["min_y"], 1e-3)
                text_offset = 0.015 * dy
                for det in preds_bev:
                    poly = det["poly"]
                    if not np.all(np.isfinite(poly)):
                        continue
                    patch = MplPolygon(poly, closed=True, fill=False, edgecolor="#e74c3c", linewidth=1.8)
                    ax.add_patch(patch)
                    center = det["center"]
                    ax.scatter(center[0], center[1], s=20, color="#e74c3c", alpha=0.9)
                    label = f"{det['score']:.2f} / {det['yaw']:.1f}°"
                    ax.text(center[0], center[1] + text_offset, label,
                            fontsize=7.5, color="#e74c3c", ha="center", va="bottom",
                            bbox=dict(facecolor="#ffffff", alpha=0.6, edgecolor="none", pad=1.5))
                legend_handles.append(Line2D([0], [0], color="#e74c3c", lw=2, label="Pred"))

            if legend_handles:
                ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.75, fontsize=8)

            fig.tight_layout(pad=0.6)
            fig.savefig(save_path_img, dpi=220)
            plt.close(fig)
            return
        except Exception:
            plt.close("all")

    # Fallback: OpenCV 기반 간단 시각화
    canvas = np.full((params["height"], params["width"], 3), 255, dtype=np.uint8)
    axis_color = (120, 120, 120)
    axis_thickness = 1
    if params["min_x"] <= 0.0 <= params["max_x"]:
        axis_x = np.array([[0.0, params["min_y"]], [0.0, params["max_y"]]], dtype=np.float64)
        axis_x_px = _to_canvas(axis_x, params)
        cv2.line(canvas, tuple(axis_x_px[0]), tuple(axis_x_px[1]), axis_color, axis_thickness, cv2.LINE_AA)
    if params["min_y"] <= 0.0 <= params["max_y"]:
        axis_y = np.array([[params["min_x"], 0.0], [params["max_x"], 0.0]], dtype=np.float64)
        axis_y_px = _to_canvas(axis_y, params)
        cv2.line(canvas, tuple(axis_y_px[0]), tuple(axis_y_px[1]), axis_color, axis_thickness, cv2.LINE_AA)

    for poly in gt_polys:
        if not np.all(np.isfinite(poly)):
            continue
        poly_px = _to_canvas(poly, params)
        cv2.polylines(canvas, [poly_px], True, (0, 200, 0), 2)
        center_px = _to_canvas(poly.mean(axis=0, keepdims=True), params)[0]
        cv2.circle(canvas, tuple(center_px), 4, (0, 150, 0), -1)

    for det in preds_bev:
        poly = det["poly"]
        if not np.all(np.isfinite(poly)):
            continue
        poly_px = _to_canvas(poly, params)
        cv2.polylines(canvas, [poly_px], True, (0, 0, 255), 2)
        center_px = _to_canvas(np.asarray(det["center"]).reshape(1, 2), params)[0]
        cv2.circle(canvas, tuple(center_px), 4, (0, 0, 255), -1)
        label = f"{det['score']:.2f} / {det['yaw']:.1f}°"
        text_pos = (center_px[0] + 4, max(center_px[1] - 8, 10))
        cv2.putText(canvas, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
    coord_text = f"x:[{params['min_x']:.2f},{params['max_x']:.2f}]  y:[{params['min_y']:.2f},{params['max_y']:.2f}]"
    cv2.putText(canvas, coord_text, (10, params["height"] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    cv2.imwrite(save_path_img, canvas)


def evaluate_single_image_bev(preds_bev: List[dict], gt_tris_bev: np.ndarray, iou_thr=0.5):
    gt_arr = np.asarray(gt_tris_bev)
    num_gt = gt_arr.shape[0] if gt_arr.ndim == 3 else 0
    preds_sorted = sorted(preds_bev, key=lambda d: d["score"], reverse=True)

    if num_gt == 0:
        records = [(det["score"], 0, 0.0, None) for det in preds_sorted]
        return records, 0

    gt_polys = [poly_from_tri(tri) for tri in gt_arr]
    matched = np.zeros((num_gt,), dtype=bool)
    records = []

    for det in preds_sorted:
        poly_d = det["poly"]
        if not np.all(np.isfinite(poly_d)):
            records.append((det["score"], 0, 0.0, None))
            continue
        best_iou, best_idx = 0.0, -1
        for idx, poly_g in enumerate(gt_polys):
            if matched[idx] or not np.all(np.isfinite(poly_g)):
                continue
            iou = iou_polygon(poly_d, poly_g)
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_idx >= 0 and best_iou >= iou_thr:
            matched[best_idx] = True
            orient_err = orientation_error_deg(det["tri"], gt_arr[best_idx])
            records.append((det["score"], 1, best_iou, orient_err))
        else:
            records.append((det["score"], 0, best_iou, None))

    return records, int(matched.sum())

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser("YOLO11 2.5D Inference (+GT overlay & IoU + 2D metrics)")
    ap.add_argument("--input-dir", type=str, required=True, help="이미지 폴더")
    ap.add_argument("--output-dir", type=str, default="./inference_results", help="결과 저장 루트")
    ap.add_argument("--weights", type=str, required=True, help="학습된 *.pth (state_dict)")
    ap.add_argument("--base-model", type=str, default="yolo11m.pt", help="학습시 사용한 YOLO11 가중치")
    ap.add_argument("--img-size", type=str, default="864,1536", help="HxW (학습과 동일)")
    ap.add_argument("--score-mode", type=str, default="obj*cls", choices=["obj","cls","obj*cls"])
    ap.add_argument("--conf", type=float, default=0.80)
    ap.add_argument("--nms-iou", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--clip-cells", type=float, default=None, help="디코딩 시 tanh clip 반경(셀). None이면 미적용")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--exts", type=str, default=".jpg,.jpeg,.png")

    # (선택) GT와 함께 그리기 + 평가
    ap.add_argument("--gt-label-dir", type=str, default=None, help="GT 라벨 폴더(원본 해상도 기준)")
    ap.add_argument("--eval-iou-thr", type=float, default=0.5, help="평가 IoU 임계값 (2D)")
    ap.add_argument("--labels-are-original-size", action="store_true", default=True, 
                    help="GT 좌표가 원본 이미지 기준이면 켜세요(평가/오버레이 시 모델 입력 크기로 스케일)")
    ap.add_argument("--calib-dir", type=str, default=None, help="이미지→지면 투영 H 행렬(.txt/.npy) 폴더")
    ap.add_argument("--invert-calib", action="store_true", help="H 행렬을 역행렬로 사용(img2ground ↔ ground2img)")
    ap.add_argument("--bev-scale", type=float, default=1.0, help="BEV 좌표/길이에 곱할 스케일 (예: meter/pixel)")

    args = ap.parse_args()

    H, W = map(int, args.img_size.split(","))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # 모델 로드 (학습과 동일 구조)
    model = YOLO11_2p5D(yolo11_path=args.base_model, num_classes=1, img_size=(H, W)).to(device)
    if isinstance(model.strides, torch.Tensor):
        model.strides = [float(s.item()) for s in model.strides]
    else:
        model.strides = [float(s) for s in model.strides]

    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # I/O 준비
    out_img_dir = os.path.join(args.output_dir, "images")
    out_lab_dir = os.path.join(args.output_dir, "labels")
    out_mix_dir = os.path.join(args.output_dir, "images_with_gt")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lab_dir, exist_ok=True)
    os.makedirs(out_mix_dir, exist_ok=True)

    use_bev = args.calib_dir is not None
    if use_bev and not os.path.isdir(args.calib_dir):
        print(f"[Warn] calib dir {args.calib_dir} 미존재 → BEV 출력 비활성화")
        use_bev = False

    if use_bev:
        out_bev_img_dir = os.path.join(args.output_dir, "bev_images")
        out_bev_lab_dir = os.path.join(args.output_dir, "bev_labels")
        out_bev_mix_dir = os.path.join(args.output_dir, "bev_images_with_gt")
        os.makedirs(out_bev_img_dir, exist_ok=True)
        os.makedirs(out_bev_lab_dir, exist_ok=True)
        os.makedirs(out_bev_mix_dir, exist_ok=True)
    else:
        out_bev_img_dir = out_bev_lab_dir = out_bev_mix_dir = None

    homography_cache = {} if use_bev else {}
    missing_h_names = set()

    # 이미지 목록
    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    names = [f for f in os.listdir(args.input_dir) if f.lower().endswith(exts)]
    names.sort()

    # 2D 평가 여부
    do_eval_2d = args.gt_label_dir is not None and os.path.isdir(args.gt_label_dir)

    # 평가 누적 버퍼
    metric_records = []
    total_gt = 0
    metric_records_bev = []
    total_gt_bev = 0

    print(
        f"[Infer] device={device}, strides={model.strides}, imgs={len(names)}, "
        f"eval2D={do_eval_2d}, evalBEV={use_bev}"
    )

    with torch.inference_mode():
        # autocast: GPU면 켠다
        use_amp = device.type == "cuda"
        try:
            amp_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp)
        except Exception:
            from torch.cuda.amp import autocast as _autocast
            amp_ctx = _autocast(enabled=use_amp)

        for name in tqdm(names, desc="[Infer]"):
            path = os.path.join(args.input_dir, name)
            img_bgr0 = cv2.imread(path)
            if img_bgr0 is None:
                continue

            # 전처리 (resize → RGB → tensor)
            H0, W0 = img_bgr0.shape[:2]
            img_bgr = cv2.resize(img_bgr0, (W, H), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb.transpose(2,0,1)).float().div_(255.0).unsqueeze(0).to(device, non_blocking=True)

            scale_resize_x = W / float(W0)
            scale_resize_y = H / float(H0)
            scale_to_orig_x = float(W0) / float(W)
            scale_to_orig_y = float(H0) / float(H)

            with amp_ctx:
                outs = model(img_t)

            # 디코딩
            dets = decode_predictions(
                outs,
                model.strides,
                clip_cells=args.clip_cells,
                conf_th=args.conf,
                nms_iou=args.nms_iou,
                topk=args.topk,
                score_mode=args.score_mode,
                use_gpu_nms=(device.type=="cuda")
            )[0]  # batch size=1

            # 저장: pred-only (리사이즈 좌표계 + 원본 좌표 반환)
            save_img = os.path.join(out_img_dir, name)
            save_txt = os.path.join(out_lab_dir, os.path.splitext(name)[0] + ".txt")
            pred_tris_orig = draw_pred_only(img_bgr, dets, save_img, save_txt, W, H, W0, H0)

            # GT 준비 (평가/GT 오버레이용 리사이즈 좌표 & BEV용 원본 좌표)
            gt_for_eval = np.zeros((0, 3, 2), dtype=np.float32)
            gt_tri_orig_for_bev = np.zeros((0, 3, 2), dtype=np.float32)

            if do_eval_2d:
                lab_path = os.path.join(args.gt_label_dir, os.path.splitext(name)[0] + ".txt")
                gt_tri_raw = load_gt_triangles(lab_path)
                if gt_tri_raw.shape[0] > 0:
                    if args.labels_are_original_size:
                        gt_tri_orig_for_bev = gt_tri_raw.astype(np.float32)
                        gt_for_eval = gt_tri_raw.copy()
                        gt_for_eval[:, :, 0] *= scale_resize_x
                        gt_for_eval[:, :, 1] *= scale_resize_y
                    else:
                        gt_for_eval = gt_tri_raw.astype(np.float32)
                        gt_tri_orig_for_bev = gt_tri_raw.copy()
                        gt_tri_orig_for_bev[:, :, 0] *= scale_to_orig_x
                        gt_tri_orig_for_bev[:, :, 1] *= scale_to_orig_y

                save_img_mix = os.path.join(out_mix_dir, name)
                draw_pred_with_gt(img_bgr, dets, gt_for_eval, save_img_mix, iou_thr=args.eval_iou_thr)

                if gt_for_eval.shape[0] > 0:
                    records, _ = evaluate_single_image(dets, gt_for_eval, iou_thr=args.eval_iou_thr)
                    metric_records.extend(records)
                    total_gt += gt_for_eval.shape[0]

            # BEV 변환/저장/평가
            if use_bev:
                H_img2ground = load_homography(args.calib_dir, name, homography_cache, invert=args.invert_calib)
                if H_img2ground is None:
                    missing_h_names.add(os.path.splitext(name)[0])
                else:
                    bev_dets = []
                    if pred_tris_orig:
                        pred_stack_orig = np.asarray(pred_tris_orig, dtype=np.float64)
                        pred_tris_bev = apply_homography(pred_stack_orig, H_img2ground)
                        pred_tris_bev = pred_tris_bev * float(args.bev_scale)
                        for det, tri_bev in zip(dets, pred_tris_bev):
                            if not np.all(np.isfinite(tri_bev)):
                                continue
                            poly_bev = poly_from_tri(tri_bev)
                            props = compute_bev_properties(poly_bev)
                            if props is None:
                                continue
                            center, length, width, yaw = props
                            bev_dets.append({
                                "score": float(det["score"]),
                                "tri": tri_bev,
                                "poly": poly_bev,
                                "center": center,
                                "length": length,
                                "width": width,
                                "yaw": yaw,
                            })

                    if gt_tri_orig_for_bev.size > 0:
                        gt_tris_bev = apply_homography(gt_tri_orig_for_bev.astype(np.float64), H_img2ground)
                        gt_tris_bev = gt_tris_bev * float(args.bev_scale)
                    else:
                        gt_tris_bev = np.zeros((0, 3, 2), dtype=np.float64)

                    bev_img_path = os.path.join(out_bev_img_dir, name)
                    draw_bev_visualization(bev_dets, None, bev_img_path, f"{name} | Pred BEV")

                    bev_mix_path = os.path.join(out_bev_mix_dir, name)
                    draw_bev_visualization(bev_dets, gt_tris_bev, bev_mix_path, f"{name} | Pred & GT BEV")

                    bev_label_path = os.path.join(out_bev_lab_dir, os.path.splitext(name)[0] + ".txt")
                    write_bev_labels(bev_label_path, bev_dets)

                    if gt_tris_bev.size > 0:
                        records_bev, _ = evaluate_single_image_bev(bev_dets, gt_tris_bev, iou_thr=args.eval_iou_thr)
                        metric_records_bev.extend(records_bev)
                        total_gt_bev += gt_tris_bev.shape[0]

    # (선택) 2D 결과 통계 출력
    if do_eval_2d:
        metrics = compute_detection_metrics(metric_records, total_gt)
        print("== 2D Eval (dataset-wide) ==")
        print("Precision:  {:.4f}".format(metrics["precision"]))
        print("Recall:     {:.4f}".format(metrics["recall"]))
        print("mAP@50:     {:.4f}".format(metrics["map50"]))
        print("mAOE(deg):  {:.2f}".format(metrics["mAOE_deg"]))

    if use_bev:
        metrics_bev = compute_detection_metrics(metric_records_bev, total_gt_bev)
        if total_gt_bev > 0 or metric_records_bev:
            print("== BEV Eval (dataset-wide) ==")
            print("Precision:  {:.4f}".format(metrics_bev["precision"]))
            print("Recall:     {:.4f}".format(metrics_bev["recall"]))
            print("APbev@50:   {:.4f}".format(metrics_bev["map50"]))
            print("mAOE_bev:   {:.2f}".format(metrics_bev["mAOE_deg"]))
        else:
            print("[Info] BEV 평가는 GT 또는 유효한 H 행렬이 부족해 계산하지 않았습니다.")

    if use_bev and missing_h_names:
        samples = sorted(missing_h_names)
        preview = ", ".join(samples[:5])
        more = "" if len(samples) <= 5 else " ..."
        print(f"[Warn] H 행렬 누락 이미지 {len(samples)}개 (예: {preview}{more})")

    print("Done.")

if __name__ == "__main__":
    main()
