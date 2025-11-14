import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import glob
import os
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

COLOR_LABELS = ("red", "blue", "green", "white", "yellow", "purple")
VALID_COLORS = {color: color for color in COLOR_LABELS}


def normalize_color_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    color = str(value).strip().lower()
    return VALID_COLORS.get(color)

def wrap_deg(angle):
    """[-180, 180)로 정규화"""
    a = (angle + 180.0) % 360.0
    if a < 0:
        a += 360.0
    return a - 180.0

def nearest_equivalent_deg(meas, ref, period=360.0):
    """
    ref에 '가장 가까운' 동치각으로 meas를 변환.
    period=360이면 일반 각도, 180이면 앞/뒤 대칭 모델 대비.
    """
    d = meas - ref
    d = (d + period/2) % period - period/2
    return ref + d

def circ_mean_deg(angles_deg, weights=None, period=360.0):
    """대표각 계산 시 atan2(sum w sin, sum w cos). 주기 180이면 2θ 처리 권장."""
    ang = np.deg2rad(angles_deg)
    if period == 180.0:
        ang = 2.0 * ang  # 180° 주기면 2θ 공간에서 평균
    if weights is None:
        weights = np.ones_like(ang)
    s = np.sum(weights * np.sin(ang))
    c = np.sum(weights * np.cos(ang))
    mean = math.degrees(math.atan2(s, c))
    if period == 180.0:
        mean *= 0.5
    return wrap_deg(mean)


def carla_to_aabb(detection):
    # detection: [class, x_c, y_c, l, w, yaw_deg]
    x_c, y_c, l, w, yaw_deg = detection[1:6]
    yaw = math.radians(yaw_deg)
    
    # 1. OBB의 4개 코너 계산
    dx, dy = l / 2.0, w / 2.0
    corners = np.array([[ dx,  dy], [ dx, -dy], [-dx, -dy], [-dx,  dy]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s,  c]])
    rotated_corners = corners @ R.T + np.array([x_c, y_c])
    
    # 2. AABB (Axis-Aligned BBOX) 계산
    x_min = np.min(rotated_corners[:, 0])
    x_max = np.max(rotated_corners[:, 0])
    y_min = np.min(rotated_corners[:, 1])
    y_max = np.max(rotated_corners[:, 1])

    aabb_width = x_max - x_min
    aabb_height = y_max - y_min
    
    # AABB: [x_min, y_min, width, height]
    return np.array([x_min, y_min, aabb_width, aabb_height])

def iou_bbox(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2]) * max(0.0, boxA[3])
    areaB = max(0.0, boxB[2]) * max(0.0, boxB[3])

    denom = areaA + areaB - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


# BBOX 목록과 Track 목록 간의 IOU 비용 행렬 계산
def iou_batch(detections_carla, tracks):
    cost_matrix = np.zeros((len(detections_carla), len(tracks)), dtype=np.float32)
    for i, det_carla in enumerate(detections_carla):
        det_aabb = carla_to_aabb(det_carla)  # [x_min, y_min, w, h]

        for j, track in enumerate(tracks):
            # KF 예측 중심
            pred_xc, pred_yc = track.kf_pos.x[:2].flatten()
            # 마지막 길이/너비/각도 사용해서 임시 OBB 구성
            temp_obb = np.array([0, pred_xc, pred_yc, track.car_length, track.car_width, track.car_yaw])
            pred_aabb = carla_to_aabb(temp_obb)

            cost_matrix[i, j] = 1.0 - iou_bbox(det_aabb, pred_aabb)
    return cost_matrix

class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    DELETED = 4

class Track:
    track_id_counter = 0 # 클래스 변수: 트랙 ID 부여
    def __init__(self, bbox_init, confirm_hits=3, color: Optional[str] = None): # bbox_init은 [class, x_c, y_c, l, w, yaw_deg] 형식
        self.id = Track.track_id_counter
        Track.track_id_counter += 1
        
        # OBB 정보 초기 저장
        self.car_length = bbox_init[3]
        self.car_width = bbox_init[4]
        self.car_yaw = bbox_init[5] # Yaw (deg)

        # 1. 중심 위치 KF (4D: x_c, y_c, vx, vy)
        self.kf_pos = KalmanFilter(dim_x=4, dim_z=2)
        self.kf_pos.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf_pos.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf_pos.x[:2] = bbox_init[1:3].reshape((2, 1))
        
        # 💡 파라미터 조정
        self.kf_pos.P *= 1000.
        self.kf_pos.Q *= 0.1
        self.kf_pos.R *= 10.
        
        # 2. 크기 및 각도 KF (2D: value, d_value/dt)
        # 상태: [값, 변화율] / 측정: [값]

        # 2-1. Yaw KF (yaw, dyaw/dt)
        self.kf_yaw = self._init_2d_kf(initial_value=self.car_yaw, Q_scale=0.01, R_scale=1.0)
        
        # 2-2. Length KF (l, dl/dt)
        self.kf_length = self._init_2d_kf(initial_value=self.car_length, Q_scale=0.001, R_scale=1.0)
        
        # 2-3. Width KF (w, dw/dt)
        self.kf_width = self._init_2d_kf(initial_value=self.car_width, Q_scale=0.001, R_scale=1.0)

        # 시간 및 상태 관리 변수
        self.time_since_update = 0
        self.hits = 1
        self.age = 1
        self.state = TrackState.TENTATIVE
        self.history = []
        self.confirm_hits = confirm_hits
        self.color_counts: Counter = Counter()
        self.current_color: Optional[str] = None
        self.total_color_votes = 0
        self._update_color(color)

    def _init_2d_kf(self, initial_value, Q_scale=0.1, R_scale=1.0):
        """2D 칼만 필터 초기화 유틸리티 (값, 변화율)"""
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1, 1], [0, 1]]) # 상태 전이: x_k+1 = x_k + v_k * dt, v_k+1 = v_k
        kf.H = np.array([[1, 0]])         # 측정 행렬: z_k = x_k
        kf.x[0] = initial_value           # 초기 값
        kf.P *= 10.                       # 초기 불확실성
        kf.Q *= Q_scale                   # 프로세스 잡음 (느린 변화 가정)
        kf.R *= R_scale                   # 측정 잡음 (측정의 신뢰도)
        return kf

    def predict(self):
        self.kf_pos.predict()
        self.kf_yaw.predict()
        self.kf_length.predict()
        self.kf_width.predict()

        self.car_yaw = self.kf_yaw.x[0, 0]
        self.car_length = self.kf_length.x[0, 0]
        self.car_width = self.kf_width.x[0, 0]

        # 각도 상태 정규화 ~ 안하면 돈다 
        self.car_yaw = wrap_deg(self.car_yaw)
        self.kf_yaw.x[0, 0] = self.car_yaw

        self.age += 1
        if self.state != TrackState.DELETED:
            self.time_since_update += 1
        
    def update(self, detections_carla, detection_colors: Optional[List[Optional[str]]] = None):
        """
        ???�레?�의 ?��? 결과�?받아???�랙???�데?�트?�고 결과�?반환?�니??
        detections_carla: [class, x_center, y_center, length, width, angle] 배열
        detection_colors: 각 detection에 대해 normalized color string 또는 None
        """
        self.last_matches = []
        # 1. 기존 ?�랙 ?�측
        for track in self.tracks:
            track.predict()

        # ?�재 ?�성 ?�랙 (DELETED ?�외)
        active_tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        # 2. IOU 기반 매칭 비용 계산 �?Hungarian ?�고리즘 ?�용
        matched_indices = []
        unmatched_detections = list(range(len(detections_carla)))
        unmatched_tracks = list(range(len(active_tracks)))
        det_colors = self._prepare_detection_colors(detection_colors, len(detections_carla))

        if len(detections_carla) > 0 and len(active_tracks) > 0:
            # IOU 비용 ?�렬 (1 - IOU)
            cost_matrix = iou_batch(detections_carla, active_tracks)
            if det_colors:
                for i, det_color in enumerate(det_colors):
                    for j, track in enumerate(active_tracks):
                        cost_matrix[i, j] += self._color_cost(det_color, track.get_color())

            # ?�렬???�과 ?�에 ?�???�형 ?�당 (최소 비용 매칭)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 매칭 결과 처리
            for r, c in zip(row_ind, col_ind):
                # IOU ?�계�??�인
                if 1.0 - cost_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c)) # (detection_index, track_index)
                    # 매칭???�덱??목록?�서 ?�거
                    if r in unmatched_detections:
                        unmatched_detections.remove(r)
                    if c in unmatched_tracks:
                        unmatched_tracks.remove(c)

        # 3. 매칭???�랙 ?�데?�트
        for det_idx, track_idx in matched_indices:
            track = active_tracks[track_idx]
            color = det_colors[det_idx] if det_colors else None
            track.update(detections_carla[det_idx], color=color)
            track.history.append(track.get_state().copy())
            self.last_matches.append((track.id, det_idx))

        # 4. 매칭?��? ?��? ?�랙 ?�태 변�?
        for track_idx in unmatched_tracks:
            track = active_tracks[track_idx]
            if track.state == TrackState.CONFIRMED:
                # Confirmed ?�태 -> Lost
                track.state = TrackState.LOST
            elif track.state == TrackState.LOST:
                # Lost ?�태?�서 max_age�??�으�???��
                if track.time_since_update > self.max_age:
                    track.state = TrackState.DELETED
            elif track.state == TrackState.TENTATIVE:
                # Tentative ?�태 -> 바로 ??��
                track.state = TrackState.DELETED

        # 5. 매칭?��? ?��? ?��? 결과�??�로???�랙 ?�성
        for det_idx in unmatched_detections:
            color = det_colors[det_idx] if det_colors else None
            new_track = Track(detections_carla[det_idx], confirm_hits=self.min_hits, color=color)
            self.tracks.append(new_track)

        # 6. ??��???�랙 ?�리 �?결과 반환
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        output_results = []
        for track in self.tracks:
            if track.state == TrackState.CONFIRMED or track.state == TrackState.LOST:
                state = track.get_state()
                # [track_id, class, x_c, y_c, l, w, angle] ?�식?�로 출력
                output_results.append(np.array([track.id, *state]))

        return np.array(output_results) if output_results else np.array([])

    def _prepare_detection_colors(self, detection_colors, count: int) -> List[Optional[str]]:
        if not detection_colors:
            return [None] * count
        colors: List[Optional[str]] = []
        for idx in range(count):
            val = detection_colors[idx] if idx < len(detection_colors) else None
            colors.append(normalize_color_label(val))
        return colors

    def _color_cost(self, detection_color: Optional[str], track_color: Optional[str]) -> float:
        if not detection_color or not track_color:
            return 0.0
        if detection_color == track_color:
            return 0.0
        return self.color_penalty

    def get_latest_matches(self) -> List[Tuple[int, int]]:
        return list(self.last_matches)

    def get_track_attributes(self) -> Dict[int, dict]:
        attrs: Dict[int, dict] = {}
        for track in self.tracks:
            if track.state in (TrackState.CONFIRMED, TrackState.LOST):
                attrs[track.id] = {
                    "color": track.get_color(),
                    "color_confidence": track.get_color_confidence(),
                }
        return attrs

def load_detections_from_file(filepath):
    try:
        if os.path.getsize(filepath) == 0:
            return np.array([])
        
        data = pd.read_csv(
            filepath, 
            header=None, 
            sep='[,\s]+', 
            engine='python',
            dtype=float
        ).values
        
        if data.shape[1] != 6:
            raise ValueError(f"파일 {filepath}의 열 개수가 예상 (6개)와 다릅니다: {data.shape[1]}")
        return data
        
    except Exception as e:
        print(f"파일 로드 오류: {filepath}. 오류: {e}")
        return np.array([])

def main_tracking():
    input_folder = "/merge_dist_wbf_drop"
    
    # 500개 프레임 파일 목록을 순서대로 정렬
    file_pattern = os.path.join(input_folder, "merged_frame_*.txt")
    frame_files = sorted(glob.glob(file_pattern))
    
    if not frame_files:
        print(f"오류: '{input_folder}' 폴더에서 파일을 찾을 수 없습니다. 경로와 파일명을 확인해주세요.")
        return

    # SORT 트래커 초기화 (파라미터는 필요에 따라 튜닝 가능)
    tracker = SortTracker(max_age=10, min_hits=3, iou_threshold=0.15)

    # 모든 프레임의 최종 추적 결과를 저장할 리스트
    all_tracking_results = []

    print(f"총 {len(frame_files)}개의 프레임 파일 로드됨. 추적 시작...")
    
    for frame_idx, filepath in enumerate(frame_files):
        # 1. 탐지 결과 로드
        detections = load_detections_from_file(filepath)
        
        # 2. 트래커 업데이트
        tracked_objects = tracker.update(detections)
        
        # 3. 추적 결과 저장 (프레임 인덱스, 트랙 ID, 나머지 정보)
        # tracked_objects 형식: [track_id, class, x_c, y_c, l, w, angle]
        if len(tracked_objects) > 0:
            # 프레임 인덱스(0부터 시작) 추가
            frame_id_column = np.full((tracked_objects.shape[0], 1), frame_idx)
            # [frame_id, track_id, class, x_c, y_c, l, w, angle]
            frame_results = np.hstack((frame_id_column, tracked_objects))
            all_tracking_results.append(frame_results)

        # 진행 상황 출력 (선택 사항)
        if (frame_idx + 1) % 50 == 0 or frame_idx == len(frame_files) - 1:
            print(f"--- 프레임 {frame_idx + 1} / {len(frame_files)} 처리 완료. 현재 활성 트랙 수: {len(tracker.tracks)}")

    # 최종 결과 통합
    if all_tracking_results:
        try:
            # 리스트에 있는 모든 NumPy 배열을 세로로 합칩니다.
            final_results = np.vstack(all_tracking_results)
            
            # 최종 결과 저장
            print("\n✅ 추적 완료. 최종 결과를 'tracking_output.txt'에 저장합니다.")
            
            header = "frame_id, track_id, class, x_center, y_center, length, width, angle"
            np.savetxt(
                "tracking_output.txt", 
                final_results, 
                fmt=['%d', '%d', '%d', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f'], 
                delimiter=',', 
                header=header, 
                comments=''
            )
        except Exception as e:
            print(f"\n⚠️ 최종 결과 통합/저장 중 오류 발생: {e}")
    else:
        print("\n⚠️ 추적된 객체가 없습니다. (모든 프레임에서 Confirmed/Lost 상태의 트랙이 없었음)")

# main_tracking() 
