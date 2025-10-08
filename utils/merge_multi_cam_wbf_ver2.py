import os
import math
import glob
import numpy as np
from typing import List, Dict, Tuple

# ================== 1) CARLA -> BEV 평면 좌표 변환 ==================
def carla_to_bev_xy(x_carla: float, y_carla: float, z_carla: float = 0.0) -> Tuple[float, float]:
    """
    find_bev_matrix.py와 동일한 축 변환을 따라
    CARLA (X,Y,Z) -> CV 좌표 (x_cv, z_cv) = 우리가 쓰는 BEV 평면 (X,Y)
    """
    cam_pos_carla = np.array([-y_carla, x_carla, z_carla], dtype=np.float64)
    axes_carla_to_cv = np.array([[0.0, 1.0,  0.0],
                                 [0.0, 0.0, -1.0],
                                 [1.0, 0.0,  0.0]], dtype=np.float64)
    cam_pos_cv = axes_carla_to_cv @ cam_pos_carla   # (x_cv, y_cv, z_cv)
    x_bev = float(cam_pos_cv[0])
    y_bev = float(cam_pos_cv[2])  # 주의: z_cv를 BEV의 Y로 사용
    return x_bev, y_bev

def build_cam_ground_from_simple(camera_setups: List[dict]) -> Dict[str, Tuple[float, float]]:
    """
    CAMERA_SETUPS (순수 dict: {'name', 'pos': {'x','y','z'}, 'rot': {...}}) 로부터
    {'cam1': (x_bev, y_bev), ...} 생성
    """
    cam_ground = {}
    for item in camera_setups:
        name = item["name"]
        p = item["pos"]
        x, y, z = float(p["x"]), float(p["y"]), float(p["z"])
        cam_ground[name] = carla_to_bev_xy(x, y, z)
    return cam_ground

# ================== 2) I/O: cam별 라벨 + 출처(cam 이름) 보존 ==================
def load_cam_labels(pred_dir: str, frame_key: str) -> List[Tuple[str, np.ndarray]]:
    """
    cam*_frame_{frame_key}.txt -> [(cam_name, arr(N,5)), ...]
    각 행: [cx, cy, L, W, yaw_deg]
    """
    paths = sorted(glob.glob(os.path.join(pred_dir, f"cam*_frame_{frame_key}.txt")))
    result = []
    for path in paths:
        cam_name = os.path.basename(path).split("_frame_")[0]
        rows = []
        with open(path, "r") as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) != 6:
                    continue
                _, cx, cy, L, W, yaw = vals
                rows.append([float(cx), float(cy), float(L), float(W), float(yaw)])
        if rows:
            result.append((cam_name, np.array(rows, dtype=float)))
    return result

# ================== 3) 회전 박스 유틸 ==================
def obb_to_corners(cx, cy, L, W, yaw_deg):
    yaw = math.radians(yaw_deg)
    dx, dy = L/2.0, W/2.0
    corners = np.array([[ dx,  dy],
                        [ dx, -dy],
                        [-dx, -dy],
                        [-dx,  dy]], dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s],[s,  c]], dtype=float)
    return corners @ R.T + np.array([cx, cy], dtype=float)

def _poly_area(poly: np.ndarray) -> float:
    x, y = poly[:,0], poly[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _clip_polygon(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """
    Sutherland–Hodgman polygon clipping (convex clipper).
    항상 (N,2) np.ndarray 를 반환. 교집합 없으면 shape (0,2) 빈 배열 반환.
    """
    def inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0

    def inter(p1, p2, a, b):
        x1,y1 = p1; x2,y2 = p2; x3,y3 = a; x4,y4 = b
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-9:
            return p2  # 평행/거의 평행: 근사치
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
        return np.array([px, py], dtype=float)

    # 시작은 ndarray 보장
    output = np.asarray(subject, dtype=float)
    for i in range(len(clipper)):
        input_list = np.asarray(output, dtype=float)
        if input_list.size == 0:
            return np.empty((0, 2), dtype=float)

        A = np.asarray(clipper[i], dtype=float)
        B = np.asarray(clipper[(i+1) % len(clipper)], dtype=float)

        new_output = []
        S = input_list[-1]
        for E in input_list:
            if inside(E, A, B):
                if not inside(S, A, B):
                    new_output.append(inter(S, E, A, B))
                new_output.append(E)
            elif inside(S, A, B):
                new_output.append(inter(S, E, A, B))
            S = E

        output = np.asarray(new_output, dtype=float)
        if output.size == 0:
            return np.empty((0, 2), dtype=float)

    return output  # ndarray 보장

def oriented_iou(box1, box2) -> float:
    p1 = obb_to_corners(*box1)
    p2 = obb_to_corners(*box2)
    inter_poly = _clip_polygon(p1, p2)
    if inter_poly.size == 0:
        inter_poly = _clip_polygon(p2, p1)
        if inter_poly.size == 0:
            return 0.0
    inter = _poly_area(inter_poly)
    a1 = _poly_area(p1)
    a2 = _poly_area(p2)
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0

# ================== 4) 클러스터 평균 (거리 가중치 지원) ==================
def merge_cluster(boxes: List[np.ndarray], weights=None):
    if len(boxes) == 1:
        return boxes[0]
    boxes = np.array(boxes, dtype=float)
    if weights is None:
        weights = np.ones(len(boxes), dtype=float)
    w = np.array(weights, dtype=float)
    w /= (w.sum() + 1e-12)
    mean = np.average(boxes, axis=0, weights=w)
    # yaw는 원형평균
    yaw_s = np.average(np.sin(np.deg2rad(boxes[:,4])), weights=w)
    yaw_c = np.average(np.cos(np.deg2rad(boxes[:,4])), weights=w)
    mean[4] = np.rad2deg(np.arctan2(yaw_s, yaw_c))
    return mean

# ================== 5) 거리 가중치 WBF ==================
def weighted_box_fusion_distance(
    cam_arrays: List[Tuple[str, np.ndarray]],
    cam_ground_xy: Dict[str, Tuple[float,float]],
    iou_thr: float = 0.15,
    dist_k: float = 0.6,
    yaw_thr_deg: float = 25.0,
    dist_eps: float = 1.0,          # 너무 큰 weight 방지 ε
    weight_pow: float = 1.0         # 1/dist^p 의 p (1~2 권장)
) -> np.ndarray:
    """
    - cam_arrays: [(cam_name, arr(N,5)), ...]
    - cam_ground_xy: {'cam1': (x_bev, y_bev), ...}
    - 가중치: w_i = 1 / (dist(cam_i, box_i)^p + dist_eps)
    - 연결: Oriented IoU >= iou_thr  OR  (거리/각도 게이트)
    """
    # 1) 모든 박스와 출처 cam 나란히 모으기
    boxes, cams = [], []
    for cam, arr in cam_arrays:
        if arr.size == 0:
            continue
        for row in arr:
            boxes.append(row)
            cams.append(cam)
    if not boxes:
        return np.zeros((0,5), dtype=float)
    boxes = np.array(boxes, dtype=float)
    N = len(boxes)

    # 2) 연결 조건
    def linkable(b1, b2):
        if oriented_iou(b1, b2) >= iou_thr:
            return True
        x1,y1,L1,W1,yaw1 = b1
        x2,y2,L2,W2,yaw2 = b2
        d = math.hypot(x1-x2, y1-y2)
        diag = 0.5*(math.hypot(L1,W1) + math.hypot(L2,W2))
        dyaw = abs(((yaw1 - yaw2 + 180) % 360) - 180)
        return (d <= dist_k*diag) and (dyaw <= yaw_thr_deg)

    used = np.zeros(N, dtype=bool)
    merged_boxes = []

    for i in range(N):
        if used[i]:
            continue
        cluster_idx = [i]
        used[i] = True
        # 연결된 것들 계속 편입 (BFS 느낌)
        changed = True
        while changed:
            changed = False
            for j in range(N):
                if used[j]:
                    continue
                if any(linkable(boxes[j], boxes[k]) for k in cluster_idx):
                    cluster_idx.append(j)
                    used[j] = True
                    changed = True

        # 3) 클러스터 가중치 (카메라-박스 거리 기반)
        cluster_boxes = [boxes[k] for k in cluster_idx]
        weights = []
        for k in cluster_idx:
            cam = cams[k]
            cx, cy = boxes[k][:2]
            cam_xy = cam_ground_xy.get(cam, (0.0, 0.0))
            d = math.hypot(cx - cam_xy[0], cy - cam_xy[1])  # m
            w = 1.0 / ((d ** weight_pow) + dist_eps)        # 가까울수록 큼
            weights.append(w)

        merged = merge_cluster(cluster_boxes, weights=weights)
        merged_boxes.append(merged)

    return np.array(merged_boxes, dtype=float)

# ================== 6) 포함 박스 억제 ==================
def is_fully_inside(small, big, margin_frac=0.05):
    cx_b, cy_b, L_b, W_b, yaw_b = big
    corners_s = obb_to_corners(*small)
    yaw = math.radians(yaw_b)
    c, s = math.cos(-yaw), math.sin(-yaw)
    R_inv = np.array([[c, -s],[s,  c]])
    local = (corners_s - np.array([cx_b, cy_b])) @ R_inv.T
    mL = margin_frac * L_b
    mW = margin_frac * W_b
    cond_x = np.all(np.abs(local[:,0]) <= (L_b/2.0 - mL))
    cond_y = np.all(np.abs(local[:,1]) <= (W_b/2.0 - mW))
    return bool(cond_x and cond_y)

def suppress_nested_boxes(boxes: np.ndarray,
                          area_ratio_thr: float = 0.7,
                          margin_frac: float = 0.06,
                          min_area: float = 0.5) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    areas = boxes[:,2] * boxes[:,3]
    keep = np.ones(len(boxes), dtype=bool)
    order = np.argsort(-areas)  # 큰 것부터
    for ii, i in enumerate(order):
        if not keep[i]:
            continue
        if areas[i] < min_area:
            keep[i] = False
            continue
        big = boxes[i]
        for j in order[ii+1:]:
            if not keep[j]:
                continue
            small = boxes[j]
            if areas[j]/areas[i] <= area_ratio_thr and is_fully_inside(small, big, margin_frac):
                keep[j] = False
    return boxes[keep]

# ================== 7) 메인 (사용 예) ==================
def merge_frame_with_distance_weight(
    pred_dir: str,
    frame_key: str,
    out_dir: str,
    camera_setups: List[dict]
):
    os.makedirs(out_dir, exist_ok=True)

    # (a) 카메라 지면 좌표 계산 (순수 dict 포맷)
    cam_ground_xy = build_cam_ground_from_simple(camera_setups)  # {'cam1': (x,y), ...}

    # (b) 프레임별 cam 라벨 로드 (cam 이름 보존)
    cam_arrays = load_cam_labels(pred_dir, frame_key)            # [(cam, arr), ...]
    if not cam_arrays:
        print(f"[warn] no inputs for frame {frame_key}")
        return None

    # (c) 거리 가중치 WBF
    merged = weighted_box_fusion_distance(
        cam_arrays,
        cam_ground_xy,
        iou_thr=0.1,      # O-IoU 기준 (0.1~0.2 사이 튜닝)
        dist_k=0.6,        # 거리 게이트 강도
        yaw_thr_deg=25.0,  # 각도 게이트
        dist_eps=1.0,      # 1/(d^p + eps) 안정화
        weight_pow=1.0     # p=1 (p=1.5~2로 올리면 가까운 카메라 더 우대)
    )

    # (d) 포함 억제 후처리
    merged = suppress_nested_boxes(merged, area_ratio_thr=0.7, margin_frac=0.06, min_area=0.5)

    # (e) 저장
    out_path = os.path.join(out_dir, f"merged_frame_{frame_key}.txt")
    with open(out_path, "w") as f:
        for b in merged:
            cx, cy, L, W, yaw = b
            f.write(f"0 {cx:.4f} {cy:.4f} {L:.4f} {W:.4f} {yaw:.2f}\n")
    print(f"✅ saved: {out_path} ({len(merged)} objects)")
    return out_path

# ================== 카메라 세팅 (순수 dict) ==================
CAMERA_SETUPS = [
    {"name": "cam1",  "pos": {"x": -46,  "y": -74, "z": 9}, "rot": {"pitch": -45, "yaw":  90,  "roll": 0}},
    {"name": "cam2",  "pos": {"x": -35,  "y": -36, "z": 9}, "rot": {"pitch": -45, "yaw": 197,  "roll": 0}},
    {"name": "cam3",  "pos": {"x": -35,  "y": -36, "z": 9}, "rot": {"pitch": -45, "yaw": 163,  "roll": 0}},
    {"name": "cam4",  "pos": {"x": -35,  "y":   5, "z": 9}, "rot": {"pitch": -45, "yaw": 197,  "roll": 0}},
    {"name": "cam5",  "pos": {"x": -35,  "y":   5, "z": 9}, "rot": {"pitch": -45, "yaw": 135,  "roll": 0}},
    {"name": "cam6",  "pos": {"x": -77,  "y":   7, "z": 9}, "rot": {"pitch": -45, "yaw":  73,  "roll": 0}},
    {"name": "cam7",  "pos": {"x": -77,  "y":   7, "z": 9}, "rot": {"pitch": -45, "yaw": 107,  "roll": 0}},
    {"name": "cam8",  "pos": {"x": -121, "y":  19, "z": 9}, "rot": {"pitch": -45, "yaw":   0,  "roll": 0}},
    {"name": "cam9",  "pos": {"x": -121, "y": -15, "z": 9}, "rot": {"pitch": -45, "yaw":  17,  "roll": 0}},
    {"name": "cam10", "pos": {"x": -121, "y": -15, "z": 9}, "rot": {"pitch": -45, "yaw": -17,  "roll": 0}},
    {"name": "cam11", "pos": {"x": -107, "y": -57, "z": 9}, "rot": {"pitch": -45, "yaw":  40,  "roll": 0}},
    {"name": "cam12", "pos": {"x": -75,  "y": -74, "z": 9}, "rot": {"pitch": -45, "yaw":  90,  "roll": 0}},
    {"name": "cam13", "pos": {"x": -77,  "y":  34, "z": 9}, "rot": {"pitch": -45, "yaw": -73,  "roll": 0}},
    {"name": "cam14", "pos": {"x": -77,  "y":  34, "z": 9}, "rot": {"pitch": -45, "yaw": -107, "roll": 0}},
]

if __name__ == "__main__":
    pred_dir = "/inference_dataset/bev_labels"
    out_dir  = "/merge_cam_wbf_dist"
    os.makedirs(out_dir, exist_ok=True)

    for i in range(100):
        frame_key = f"{i:06d}_-45"
        merge_frame_with_distance_weight(pred_dir, frame_key, out_dir, CAMERA_SETUPS)
