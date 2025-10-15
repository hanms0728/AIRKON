import os
import math
import glob
import numpy as np
from typing import List, Dict, Tuple
'''
기능: 여러 카메라에서 예측한 BEV 라벨들을 거리 가중치 기반으로 병합
규칙:
- AABB 기준으로 부분 포함 제거
- AABB IoU 기준으로 클러스터링
- 클러스터별 대표 라벨 선택:
    - 같은 카메라만 있으면 간단 평균(yaw는 원형평균)
    - 여러 카메라가 섞였으면:
        기본: 카메라-박스 거리가 가장 가까운 라벨 채택  
        단, 클러스터 내 박스 중심들이 같다면(center_eps), 
            yaw-중앙성(카메라->객체 bearing과 박스 yaw의 차이가 최소)으로 선택
입력: /inference_dataset/bev_labels/cam*_frame_000000_-45.txt
출력: /merge_dist_wbf/merged_frame_000000_-45.txt
'''
CAMERA_SETUPS = [
    {"name": "cam1",  "pos": {"x": -46,  "y": -74, "z": 9}, "rot": {"pitch": -40, "yaw": 90,  "roll": 0}},
    {"name": "cam2",  "pos": {"x": -35,  "y": -36, "z": 9}, "rot": {"pitch": -45, "yaw": 197, "roll": 0}},
    {"name": "cam3",  "pos": {"x": -35,  "y": -36, "z": 9}, "rot": {"pitch": -45, "yaw": 163, "roll": 0}},
    {"name": "cam4",  "pos": {"x": -35,  "y": 0,   "z": 9}, "rot": {"pitch": -45, "yaw": 190, "roll": 0}},
    {"name": "cam5",  "pos": {"x": -35,  "y": 5,   "z": 9}, "rot": {"pitch": -45, "yaw": 135, "roll": 0}},
    {"name": "cam6",  "pos": {"x": -77,  "y": 7,   "z": 9}, "rot": {"pitch": -40, "yaw": 73,  "roll": 0}},
    {"name": "cam7",  "pos": {"x": -77,  "y": 7,   "z": 9}, "rot": {"pitch": -40, "yaw": 107, "roll": 0}},
    {"name": "cam8",  "pos": {"x": -122, "y": 19,  "z": 9}, "rot": {"pitch": -40, "yaw": 0,   "roll": 0}},
    {"name": "cam9",  "pos": {"x": -95,  "y": -20, "z": 9}, "rot": {"pitch": -40, "yaw": 150, "roll": 0}},
    {"name": "cam10", "pos": {"x": -121, "y": -15, "z": 9}, "rot": {"pitch": -45, "yaw": -17, "roll": 0}},
    {"name": "cam11", "pos": {"x": -113, "y": -63, "z": 9}, "rot": {"pitch": -40, "yaw": 40,  "roll": 0}},
    {"name": "cam12", "pos": {"x": -60,  "y": -76, "z": 9}, "rot": {"pitch": -40, "yaw": 120, "roll": 0}},
    {"name": "cam13", "pos": {"x": -77,  "y": 34,  "z": 9}, "rot": {"pitch": -45, "yaw": -73, "roll": 0}},
    {"name": "cam14", "pos": {"x": -68,  "y": 34,  "z": 9}, "rot": {"pitch": -45, "yaw": -30, "roll": 0}},
    {"name": "cam15", "pos": {"x": -120, "y": -40, "z": 9}, "rot": {"pitch": -40, "yaw": 30,  "roll": 0}},
    {"name": "cam16", "pos": {"x": -61,  "y": -15, "z": 9}, "rot": {"pitch": -45, "yaw": 0,   "roll": 0}},
]

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

def aabb_iou_axis_aligned(b1, b2) -> float:
    x1,y1,L1,W1,_ = b1
    x2,y2,L2,W2,_ = b2
    A = [x1 - L1/2, y1 - W1/2, x1 + L1/2, y1 + W1/2]
    B = [x2 - L2/2, y2 - W2/2, x2 + L2/2, y2 + W2/2]
    ix1, iy1 = max(A[0], B[0]), max(A[1], B[1])
    ix2, iy2 = min(A[2], B[2]), min(A[3], B[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (L1*W1) + (L2*W2) - inter
    return inter/union if union > 1e-12 else 0.0

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

def oriented_iou(box1, box2) -> float: # box: [cx, cy, L, W, yaw_deg]
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
    return float(inter / union) if union > 0 else 0.0 # 1 ~ 0

def partial_inclusion_suppression_aabb(
    boxes: np.ndarray,
    incl_frac_thr: float = 0.55
) -> np.ndarray:
    """
    AABB(축정렬) 기준으로 부분 포함 제거.
    두 박스의 교차 면적 / 작은 박스 면적 >= incl_frac_thr 면 작은 박스를 제거.
    회전(yaw)은 무시.
    """
    if boxes.size == 0:
        return boxes
    N = len(boxes)
    areas = boxes[:,2] * boxes[:,3]
    keep = np.ones(N, dtype=bool)

    order = np.argsort(-areas)  # 큰 박스부터 검사

    for ii, i in enumerate(order):
        if not keep[i]:
            continue
        xi, yi, Li, Wi, _ = boxes[i]
        for j in order[ii+1:]:
            if not keep[j]:
                continue
            xj, yj, Lj, Wj, _ = boxes[j]

            # AABB 영역 정의
            Ai = [xi - Li/2, yi - Wi/2, xi + Li/2, yi + Wi/2]
            Aj = [xj - Lj/2, yj - Wj/2, xj + Lj/2, yj + Wj/2]

            ix1, iy1 = max(Ai[0], Aj[0]), max(Ai[1], Aj[1])
            ix2, iy2 = min(Ai[2], Aj[2]), min(Ai[3], Aj[3])
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih

            if inter <= 0:
                continue

            small_area = min(Li*Wi, Lj*Wj)
            if inter / small_area >= incl_frac_thr:
                # 작은 박스 제거
                if Li*Wi >= Lj*Wj:
                    keep[j] = False
                else:
                    keep[i] = False
    return boxes[keep]

def cluster_by_aabb_iou(boxes: np.ndarray, iou_cluster_thr: float = 0.15) -> List[List[int]]:
    """
    AABB IoU(각도 무시) >= iou_cluster_thr 이면 같은 클러스터
    """
    if boxes.size == 0:
        return []
    N = len(boxes)
    used = np.zeros(N, dtype=bool)
    clusters: List[List[int]] = []
    for i in range(N):
        if used[i]:
            continue
        q = [i]
        used[i] = True
        cluster = [i]
        while q:
            k = q.pop()
            for j in range(N):
                if used[j]:
                    continue
                if aabb_iou_axis_aligned(boxes[k], boxes[j]) >= iou_cluster_thr:
                    used[j] = True
                    q.append(j)
                    cluster.append(j)
        clusters.append(cluster)
    return clusters

def _wrap_angle_deg(a):
    return ((a + 180.0) % 360.0) - 180.0

def _bearing_deg(from_xy, to_xy):
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    return math.degrees(math.atan2(dy, dx))

def pick_representative_for_cluster(
    boxes: np.ndarray,
    cams: List[str],
    idxs: List[int],
    cam_ground_xy: Dict[str, Tuple[float,float]],
    center_eps: float = 0.5,     # "위치 동일" 판단 허용오차 [m]
    use_yaw_centrality: bool = True
) -> np.ndarray:
    """
    규칙:
    - 서로 다른 카메라에서 온 라벨이 섞여 있으면:
        기본: 카메라-박스 거리가 가장 가까운 라벨 채택
        단, 클러스터 내 박스 중심들이 거의 같다면(center_eps),
            yaw-중앙성(카메라->객체 bearing과 박스 yaw의 차이가 최소)으로 선택 <- 수정 필요 
    - 같은 카메라만 있으면: 간단 평균( yaw는 원형평균 )
    """
    # 클러스터 구성
    cluster_boxes = [boxes[i] for i in idxs]
    cluster_cams  = [cams[i]  for i in idxs]

    unique_cams = set(cluster_cams)

    # (a) 같은 카메라만
    if len(unique_cams) == 1:
        arr = np.array(cluster_boxes, dtype=float)
        w  = np.ones(len(arr), dtype=float)
        mean = np.average(arr, axis=0, weights=w)
        # yaw 원형평균
        ang = np.deg2rad(arr[:,4])
        s = np.average(np.sin(ang), weights=w)
        c = np.average(np.cos(ang), weights=w)
        mean[4] = math.degrees(math.atan2(s, c))
        return mean

    # (b) 여러 카메라가 섞인 경우
    # 1) "센터 거의 동일?" 검사
    centers = np.array([[b[0], b[1]] for b in cluster_boxes], dtype=float)
    c_ref = centers.mean(axis=0)
    close_all = np.all(np.hypot(centers[:,0]-c_ref[0], centers[:,1]-c_ref[1]) <= center_eps)

    # 2) 기본: 카메라-박스 거리 최솟값
    def cam_dist_for(idx_local):
        b = cluster_boxes[idx_local]; cam_name = cluster_cams[idx_local]
        cx, cy = b[0], b[1]
        cam_xy = cam_ground_xy.get(cam_name, (0.0, 0.0))
        return math.hypot(cx - cam_xy[0], cy - cam_xy[1])

    if not close_all or not use_yaw_centrality:
        k = int(np.argmin([cam_dist_for(t) for t in range(len(cluster_boxes))]))
        return cluster_boxes[k]

    # 3) 센터 동일 → yaw-중앙성으로 선택
    # 카메라→객체 bearing 과 박스 yaw의 차이가 작은 라벨을 선택
    def centrality_cost(idx_local):
        b = cluster_boxes[idx_local]; cam_name = cluster_cams[idx_local]
        cam_xy = cam_ground_xy.get(cam_name, (0.0, 0.0))
        bearing = _bearing_deg(cam_xy, (b[0], b[1]))  # 카메라 → 객체 방향
        # yaw와의 최소 차이 (정/역방향 동일시)
        dyaw1 = abs(_wrap_angle_deg(bearing - b[4]))
        dyaw2 = abs(_wrap_angle_deg(bearing - (b[4] + 180.0)))
        return min(dyaw1, dyaw2)

    k = int(np.argmin([centrality_cost(t) for t in range(len(cluster_boxes))]))
    return cluster_boxes[k]

def merge_frame_with_distance_weight(
    pred_dir: str,
    frame_key: str,
    out_dir: str,
    camera_setups: List[dict],
    incl_frac_thr: float = 0.55,     # 포함 제거 임계
    iou_cluster_thr: float = 0.15,   # AABB IoU 클러스터 임계
    center_eps: float = 0.5          # "위치 동일" 판단 오차
):
    os.makedirs(out_dir, exist_ok=True)

    # a) 카메라 좌표 dict: {'cam1': (x,y), ...}
    cam_ground_xy = {item["name"]: (float(item["pos"]["x"]), float(item["pos"]["y"])) for item in camera_setups}

    # b) 라벨 로드: [(cam, arr(N,5)), ...]
    cam_arrays = load_cam_labels(pred_dir, frame_key)
    if not cam_arrays:
        print(f"[warn] no inputs for frame {frame_key}")
        return None

    # c) 한 배열로 합치고, 출처 cam 이름 나란히 보관
    boxes, cams = [], []
    for cam, arr in cam_arrays:
        for row in arr:
            boxes.append(row)
            cams.append(cam)
    boxes = np.array(boxes, dtype=float)
    if boxes.size == 0:
        out_path = os.path.join(out_dir, f"merged_frame_{frame_key}.txt")
        open(out_path, "w").close()
        print(f"✅ saved: {out_path} (0 objects)")
        return out_path

    # 1) 부분 포함 제거
    boxes = partial_inclusion_suppression_aabb(boxes, incl_frac_thr=incl_frac_thr)

    keep_set = {tuple(b) for b in boxes}
    new_boxes, new_cams = [], []
    for b, c in zip(np.array([row for cam, arr in cam_arrays for row in arr], dtype=float), cams):
        if tuple(b) in keep_set:
            new_boxes.append(b)
            new_cams.append(c)
    boxes = np.array(new_boxes, dtype=float)
    cams  = list(new_cams)

    if boxes.size == 0:
        out_path = os.path.join(out_dir, f"merged_frame_{frame_key}.txt")
        open(out_path, "w").close()
        print(f"✅ saved: {out_path} (0 objects)")
        return out_path

    # 2) AABB IoU 기반 클러스터링
    clusters = cluster_by_aabb_iou(boxes, iou_cluster_thr=iou_cluster_thr)
    
    # 3) 클러스터별 대표 라벨 선택
    merged_list = []
    for idxs in clusters:
        rep = pick_representative_for_cluster(
            boxes, cams, idxs, cam_ground_xy,
            center_eps=center_eps, use_yaw_centrality=True
        )
        merged_list.append(rep)

    merged = np.array(merged_list, dtype=float)

    # 저장
    out_path = os.path.join(out_dir, f"merged_frame_{frame_key}.txt")
    with open(out_path, "w") as f:
        for cx, cy, L, W, yaw in merged:
            f.write(f"0 {cx:.4f} {cy:.4f} {L:.4f} {W:.4f} {yaw:.2f}\n")
    print(f"✅ saved: {out_path} ({len(merged)} objects)")
    return out_path


if __name__ == "__main__":
    pred_dir = "/inference_dataset/bev_labels"
    out_dir  = "/merge_dist_wbf"
    os.makedirs(out_dir, exist_ok=True)

    for i in range(100):
        frame_key = f"{i:06d}_-45"
        merge_frame_with_distance_weight(pred_dir, frame_key, out_dir, CAMERA_SETUPS) # , incl_frac_thr= 0.1, iou_cluster_thr=0.1