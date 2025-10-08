'''
동일 프레임에 대해 여러 카메라에서 예측된 BEV 박스들을 WBF 방식으로 병합하여
텍스트 파일 생성
'''
import os
import math
import glob
import numpy as np
from typing import List, Dict

def load_cam_labels(pred_dir: str, frame_key: str) -> Dict[str, np.ndarray]:
    '''
    cam별로 저장된 BEV 라벨 불러오기
    '''
    # 예: /path/preds/cam0_frame_000013_-45.txt 같은 파일들을 모두 찾음
    paths = sorted(glob.glob(os.path.join(pred_dir, f"cam*_frame_{frame_key}.txt")))
    bev_by_cam = {}
    for path in paths:
        cam_name = os.path.basename(path).split("_frame_")[0] 
        data = [] 
        with open(path, "r") as f: 
            for line in f: 
                vals = line.strip().split() 
                if len(vals) != 6:
                    continue 
                _, cx, cy, l, w, yaw = vals 
                data.append([float(cx), float(cy), float(l), float(w), float(yaw)])
        if len(data) > 0: 
            bev_by_cam[cam_name] = np.array(data) 
    return bev_by_cam

def obb_iou(box1, box2):
    """단순 IoU"""
    x1, y1, l1, w1, _ = box1 # box1: 중심(cx, cy)과 크기(L, W), 각도(yaw)
    x2, y2, l2, w2, _ = box2
    # 축 정렬 박스 변환
    boxA = [x1 - l1 / 2, y1 - w1 / 2, x1 + l1 / 2, y1 + w1 / 2]
    boxB = [x2 - l2 / 2, y2 - w2 / 2, x2 + l2 / 2, y2 + w2 / 2]
    # 교집합 영역 계산
    ix1, iy1 = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    ix2, iy2 = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih # 교집합 면적
    union = (l1 * w1) + (l2 * w2) - inter # 합집합 면적
    return inter / union if union > 0 else 0 # IoU 반환 (0~1)

def obb_to_corners(cx, cy, L, W, yaw_deg):
    # OBB(중심, 길이L, 폭W, 각도yaw_deg(도)) -> 꼭짓점 4개 좌표 (시각화, 포함판정용 근데 시각화 지금 ㅌ)
    yaw = math.radians(yaw_deg) # 디그리->라디안 
    dx, dy = L/2.0, W/2.0 
    corners = np.array([[ dx,  dy],
                        [ dx, -dy],
                        [-dx, -dy],
                        [-dx,  dy]])  
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s],[s, c]])
    return corners @ R.T + np.array([cx, cy])

def merge_cluster(boxes: List[np.ndarray], weights=None):
    """클러스터(겹치는 박스 그룹) 내 박스들 가중 평균으로 하나의 대표 박스 계산"""
    if len(boxes) == 1: # 박스가 1개면 그대로 반환
        return boxes[0]
    boxes = np.array(boxes) # shape: (K,5) = [cx,cy,L,W,yaw_deg]
    if weights is None:
        weights = np.ones(len(boxes))  # 기본 가중치 = 1
    weights = np.array(weights)
    weights /= weights.sum() # 가중치 정규화(합=1)
    # cx, cy, L, W는 가중 평균
    mean_box = np.average(boxes, axis=0, weights=weights)
    # yaw는 각도 평균이 따로 있음 
    yaw_sin = np.average(np.sin(np.deg2rad(boxes[:, 4])), weights=weights)
    yaw_cos = np.average(np.cos(np.deg2rad(boxes[:, 4])), weights=weights)
    mean_box[4] = np.rad2deg(np.arctan2(yaw_sin, yaw_cos)) # 아크탄젠트 써서 이제 디그리로 변환 
    return mean_box


# =============== WBF core ===============
def weighted_box_fusion(bev_by_cam: Dict[str, np.ndarray], iou_thr=0.3) -> np.ndarray:
    all_boxes = [] # 모든 캠의 박스를 하나의 리스트로 모음
    for cam, arr in bev_by_cam.items():
        all_boxes.extend(arr) # arr shape: (Ni,5)
    if len(all_boxes) == 0:
        return np.zeros((0,5)) # 박스가 전혀 없으면 빈 배열 반환
    all_boxes = np.array(all_boxes) # shape: (N,5)

    used = np.zeros(len(all_boxes), dtype=bool) # 이미 클러스터에 넣은 박스 표시
    merged_boxes = [] # 결과 대표 박스들

    for i, box in enumerate(all_boxes): # 박스 i를 기준으로
        if used[i]: # 이미 사용했으면 스킵
            continue
        cluster = [box] # 새 클러스터 시작
        used[i] = True
        for j in range(i+1, len(all_boxes)): # 나머지 박스들과 IoU 비교
            if used[j]:
                continue
            if obb_iou(box, all_boxes[j]) > iou_thr: # IoU 높으면 같은 그룹
                cluster.append(all_boxes[j])
                used[j] = True
        merged = merge_cluster(cluster)
        merged_boxes.append(merged)

    return np.array(merged_boxes)  # shape: (M,5)

def is_fully_inside(small, big, margin_frac=0.05):
    """
    포함시 트루 - 덜 포함되어도 트루하도록 개선 필요? <-아직안했음
    """
    cx_b, cy_b, L_b, W_b, yaw_b = big # 큰 박스의 파라미터
    corners_s = obb_to_corners(*small)  # 작은 박스의 꼭짓점 4개
    yaw = math.radians(yaw_b)
    c, s = math.cos(-yaw), math.sin(-yaw)  # inverse 회전 
    R_inv = np.array([[c, -s],[s, c]])
    local = (corners_s - np.array([cx_b, cy_b])) @ R_inv.T # 큰 거 기준 로컬 좌표로 변환
    # 마진값처리 
    mL = margin_frac * L_b
    mW = margin_frac * W_b
    cond_x = np.all(np.abs(local[:,0]) <= (L_b/2.0 - mL)) # x축= 방향으로 모두 내부?
    cond_y = np.all(np.abs(local[:,1]) <= (W_b/2.0 - mW)) # y축 방향으로 모두 내부?
    return bool(cond_x and cond_y)           # 두 축 모두 내부면 완전 포함 True

def suppress_nested_boxes(boxes: np.ndarray,
                          area_ratio_thr: float = 0.55,
                          margin_frac: float = 0.06,
                          min_area: float = 0.0) -> np.ndarray:
    """
    포함 박스 제거 
    """
    if boxes.size == 0: # 빈 입력이면 그대로 반환
        return boxes

    areas = boxes[:,2] * boxes[:,3] # 각 박스 면적(=L*W)
    keep_mask = np.ones(len(boxes), dtype=bool) # True=유지, False=제거

    # 면적 내림차순(큰 박스부터 검사 -> 작은 것 제거)
    order = np.argsort(-areas) # 내림차순 인덱스

    for ii, i in enumerate(order): # i: 큰 박스 인덱스
        if not keep_mask[i]:  # 이미 제거된 박스면 스킵
            continue
        if areas[i] < min_area: # 면적이 너무 작으면 자체 제거...는 돌긴 도는데 아니 잘 돔 
            keep_mask[i] = False
            continue
        big = boxes[i] # 기준이 되는 큰 박스
        for j in order[ii+1:]: # 그보다 작은 박스들만 검사
            if not keep_mask[j]:
                continue
            small = boxes[j]
            # 충분히 작고 & 완전 포함이면 작은박스 제거
            if areas[j] / areas[i] <= area_ratio_thr and is_fully_inside(small, big, margin_frac=margin_frac):
                keep_mask[j] = False

    return boxes[keep_mask] # 유지인 박스만 반환

# =============== Main ===============
def merge_frame(pred_dir: str, frame_key: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True) 
    bev_by_cam = load_cam_labels(pred_dir, frame_key) # 같은 프레임에 대한 카메라별 라벨 로드
    merged = weighted_box_fusion(bev_by_cam, iou_thr=0.1) # IoU 기반 클러스터링 + 가중 평균으로 병합

    # 중첩 작은 박스 억제
    merged = suppress_nested_boxes(
        merged,
        area_ratio_thr=0.7,  # 작다 판단 임계(필요시 0.5~0.7 튜닝)
        margin_frac=0.06,     # 포함 판정 여유(5~8% 튜닝)
        min_area=0.5          # 극소 박스 제거 싫으
    )

    # 결과 저장 
    out_path = os.path.join(out_dir, f"merged_frame_{frame_key}.txt") 
    with open(out_path, "w") as f:
        for b in merged:
            cx, cy, L, W, yaw = b
            f.write(f"0 {cx:.4f} {cy:.4f} {L:.4f} {W:.4f} {yaw:.2f}\n")
    print(f"✅ saved: {out_path} ({len(merged)} objects)")
    return out_path

if __name__ == "__main__":
    pred_dir = "/inference_dataset/bev_labels"
    out_dir = "/merge_cam_wbf"
    # 0 ~ 99 프레임 반복 처리 (키 형식: "000000_-45" ~ "000099_-45")
    for i in range(100):
        frame_key = f"{i:06d}_-45"
        merge_frame(pred_dir, frame_key, out_dir)
