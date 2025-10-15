import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import glob
import os
import math
'''
기능: SORT 알고리즘을 사용하여 /merge_dist_wbf 의 예측 결과에 트랙 ID를 부여
출력: tracking_output.txt (frame_id, track_id, class, x_center, y_center, length, width, angle)
'''
# BBOX 회전 무시하고 2D AABB로 변환(칼만 필터용)
def carla_to_aabb(detection):
    x_c, y_c, l, w, yaw_deg = detection[1:6]
    yaw = math.radians(yaw_deg)
    
    dx, dy = l / 2.0, w / 2.0
    corners = np.array([[ dx,  dy], [ dx, -dy], [-dx, -dy], [-dx,  dy]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s,  c]])
    rotated_corners = corners @ R.T + np.array([x_c, y_c])
    
    x_min = np.min(rotated_corners[:, 0])
    x_max = np.max(rotated_corners[:, 0])
    y_min = np.min(rotated_corners[:, 1])
    y_max = np.max(rotated_corners[:, 1])

    aabb_width = x_max - x_min
    aabb_height = y_max - y_min
    
    return np.array([x_min, y_min, aabb_width, aabb_height])

# BBOX IOU 계산(2D AABB 기준, 회전 무시)
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
        det_aabb = carla_to_aabb(det_carla) 

        for j, track in enumerate(tracks):
            pred_xc, pred_yc = track.kf.x[:2].flatten()
            
            temp_obb = np.array([0, pred_xc, pred_yc, track.car_length, track.car_width, track.car_yaw])
            pred_aabb = carla_to_aabb(temp_obb) # 예측 obb -> aabb

            cost_matrix[i, j] = 1.0 - iou_bbox(det_aabb, pred_aabb)
    return cost_matrix

class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    DELETED = 4

class Track:
    track_id_counter = 0 
    def __init__(self, bbox_init, confirm_hits=3): 
        self.id = Track.track_id_counter
        Track.track_id_counter += 1

        self.kf = KalmanFilter(dim_x=4, dim_z=2) 
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        self.kf.x[:2] = bbox_init[1:3].reshape((2, 1))
        
        self.kf.P *= 1000. 
        self.kf.Q *= 0.1   
        self.kf.R *= 10.   
        
        self.car_length = bbox_init[3]
        self.car_width = bbox_init[4]
        self.car_yaw = bbox_init[5] 

        self.time_since_update = 0
        self.hits = 1
        self.age = 1
        self.state = TrackState.TENTATIVE
        self.history = []
        self.confirm_hits = confirm_hits  

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.state != TrackState.DELETED:
            self.time_since_update += 1
        
    def update(self, bbox_carla):
        z = bbox_carla[1:3].reshape((2,1))
        self.kf.update(z)

        self.car_length = bbox_carla[3]
        self.car_width  = bbox_carla[4]
        self.car_yaw    = bbox_carla[5]

        self.time_since_update = 0
        self.hits += 1

        if self.state in (TrackState.TENTATIVE, TrackState.LOST):
            if self.hits >= self.confirm_hits:
                self.state = TrackState.CONFIRMED

    def get_state(self):
        x_c, y_c = self.kf.x[:2].flatten()
        
        length = self.car_length
        width = self.car_width
        yaw = self.car_yaw 
        return np.array([0, x_c, y_c, length, width, yaw])

class SortTracker:
    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3):
        self.tracks = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, detections_carla):
        for track in self.tracks:
            track.predict()

        active_tracks = [t for t in self.tracks if t.state != TrackState.DELETED]
        
        matched_indices = []
        unmatched_detections = list(range(len(detections_carla)))
        unmatched_tracks = list(range(len(active_tracks)))
        
        if len(detections_carla) > 0 and len(active_tracks) > 0:
            cost_matrix = iou_batch(detections_carla, active_tracks)
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if 1.0 - cost_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c))
                    if r in unmatched_detections:
                        unmatched_detections.remove(r)
                    if c in unmatched_tracks:
                        unmatched_tracks.remove(c)
        
        for det_idx, track_idx in matched_indices:
            track = active_tracks[track_idx]
            track.update(detections_carla[det_idx])
            track.history.append(track.get_state().copy())

        for track_idx in unmatched_tracks:
            track = active_tracks[track_idx]

            if track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST
            elif track.state == TrackState.LOST:
                if track.time_since_update > self.max_age:
                    track.state = TrackState.DELETED
            elif track.state == TrackState.TENTATIVE:
                track.state = TrackState.DELETED
        
        for det_idx in unmatched_detections:
            new_track = Track(detections_carla[det_idx], confirm_hits=self.min_hits)
            self.tracks.append(new_track)
        
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        output_results = []
        for track in self.tracks:
            if track.state == TrackState.CONFIRMED or track.state == TrackState.LOST:
                state = track.get_state()
                output_results.append(np.array([track.id, *state]))
        
        return np.array(output_results) if output_results else np.array([])


def load_detections_from_file(filepath):
    try:
        if os.path.getsize(filepath) == 0:
            return np.array([])
        
        data = pd.read_csv(filepath, header=None, sep='[,\s]+', engine='python', dtype=float).values
        
        if data.shape[1] != 6:
            raise ValueError(f"파일 {filepath}의 열 개수가 예상 (6개)와 다릅니다: {data.shape[1]}")

        return data
        
    except Exception as e:
        print(f"파일 로드 오류: {filepath}. 오류: {e}")
        return np.array([])

def main_tracking():
    input_folder = "/merge_dist_wbf"
    
    file_pattern = os.path.join(input_folder, "merged_frame_*.txt")
    frame_files = sorted(glob.glob(file_pattern))
    
    if not frame_files:
        print(f"오류: '{input_folder}' 폴더에서 파일을 찾을 수 없습니다. 경로와 파일명을 확인해주세요.")
        return

    tracker = SortTracker(max_age=10, min_hits=1, iou_threshold=0.15)

    all_tracking_results = []

    print(f"총 {len(frame_files)}개의 프레임 파일 로드됨. 추적 시작...")
    
    for frame_idx, filepath in enumerate(frame_files):
        detections = load_detections_from_file(filepath)
        tracked_objects = tracker.update(detections)
        
        if len(tracked_objects) > 0:
            frame_id_column = np.full((tracked_objects.shape[0], 1), frame_idx)
            frame_results = np.hstack((frame_id_column, tracked_objects))
            all_tracking_results.append(frame_results)

        if (frame_idx + 1) % 50 == 0 or frame_idx == len(frame_files) - 1:
            print(f"--- 프레임 {frame_idx + 1} / {len(frame_files)} 처리 완료. 현재 활성 트랙 수: {len(tracker.tracks)}")

    if all_tracking_results:
        try:
            final_results = np.vstack(all_tracking_results)
            
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

main_tracking()