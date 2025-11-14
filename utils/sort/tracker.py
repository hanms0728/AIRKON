import glob
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

COLOR_LABELS = ("red", "pink", "green", "white", "yellow", "purple")
VALID_COLORS = {color: color for color in COLOR_LABELS}


def normalize_color_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    color = str(value).strip().lower()
    return VALID_COLORS.get(color)


def wrap_deg(angle: float) -> float:
    """Normalize angle into [-180, 180)."""
    a = (angle + 180.0) % 360.0
    if a < 0:
        a += 360.0
    return a - 180.0


def nearest_equivalent_deg(meas: float, ref: float, period: float = 360.0) -> float:
    """
    Convert measurement into the equivalent angle closest to the reference.
    period=360 for general angle, 180 for fore/aft symmetric models.
    """
    d = meas - ref
    d = (d + period / 2.0) % period - period / 2.0
    return ref + d


def carla_to_aabb(detection: np.ndarray) -> np.ndarray:
    # detection: [class, x_c, y_c, l, w, yaw_deg]
    x_c, y_c, l, w, yaw_deg = detection[1:6]
    yaw = math.radians(yaw_deg)

    dx, dy = l / 2.0, w / 2.0
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rotated_corners = corners @ R.T + np.array([x_c, y_c])

    x_min = np.min(rotated_corners[:, 0])
    x_max = np.max(rotated_corners[:, 0])
    y_min = np.min(rotated_corners[:, 1])
    y_max = np.max(rotated_corners[:, 1])

    aabb_width = x_max - x_min
    aabb_height = y_max - y_min

    return np.array([x_min, y_min, aabb_width, aabb_height])


def iou_bbox(boxA: np.ndarray, boxB: np.ndarray) -> float:
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
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def iou_batch(detections_carla: np.ndarray, tracks: List["Track"]) -> np.ndarray:
    cost_matrix = np.zeros((len(detections_carla), len(tracks)), dtype=np.float32)
    for i, det_carla in enumerate(detections_carla):
        det_aabb = carla_to_aabb(det_carla)
        for j, track in enumerate(tracks):
            pred_xc, pred_yc = track.kf_pos.x[:2].flatten()
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
    track_id_counter = 0

    def __init__(self, bbox_init: np.ndarray, confirm_hits: int = 3, color: Optional[str] = None):
        # bbox_init: [class, x_c, y_c, l, w, yaw_deg]
        self.id = Track.track_id_counter
        Track.track_id_counter += 1

        self.cls = bbox_init[0]
        self.car_length = bbox_init[3]
        self.car_width = bbox_init[4]
        self.car_yaw = bbox_init[5]

        self.kf_pos = KalmanFilter(dim_x=4, dim_z=2)
        self.kf_pos.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf_pos.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf_pos.x[:2] = bbox_init[1:3].reshape((2, 1))
        self.kf_pos.P *= 1000.0
        self.kf_pos.Q *= 0.1
        self.kf_pos.R *= 10.0

        self.kf_yaw = self._init_2d_kf(initial_value=self.car_yaw, Q_scale=0.01, R_scale=1.0)
        self.kf_length = self._init_2d_kf(initial_value=self.car_length, Q_scale=0.001, R_scale=1.0)
        self.kf_width = self._init_2d_kf(initial_value=self.car_width, Q_scale=0.001, R_scale=1.0)

        self.time_since_update = 0
        self.hits = 1
        self.age = 1
        self.state = TrackState.TENTATIVE
        self.history: List[np.ndarray] = []
        self.confirm_hits = confirm_hits

        self.color_counts: Counter = Counter()
        self.current_color: Optional[str] = None
        self.total_color_votes = 0
        self._update_color(color)

    def _init_2d_kf(self, initial_value: float, Q_scale: float = 0.1, R_scale: float = 1.0) -> KalmanFilter:
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1, 1], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.x[0] = initial_value
        kf.P *= 10.0
        kf.Q *= Q_scale
        kf.R *= R_scale
        return kf

    def predict(self) -> None:
        self.kf_pos.predict()
        self.kf_yaw.predict()
        self.kf_length.predict()
        self.kf_width.predict()

        self.car_yaw = wrap_deg(self.kf_yaw.x[0, 0])
        self.kf_yaw.x[0, 0] = self.car_yaw
        self.car_length = max(0.0, self.kf_length.x[0, 0])
        self.car_width = max(0.0, self.kf_width.x[0, 0])

        self.age += 1
        if self.state != TrackState.DELETED:
            self.time_since_update += 1

    def update(self, bbox: np.ndarray, color: Optional[str] = None) -> None:
        measurement = np.asarray(bbox, dtype=float)
        self.cls = measurement[0]

        self.kf_pos.update(measurement[1:3].reshape((2, 1)))

        yaw_meas = nearest_equivalent_deg(measurement[5], self.kf_yaw.x[0, 0])
        self.kf_yaw.update(np.array([[yaw_meas]]))
        self.car_yaw = wrap_deg(self.kf_yaw.x[0, 0])
        self.kf_yaw.x[0, 0] = self.car_yaw

        self.kf_length.update(np.array([[measurement[3]]]))
        self.kf_width.update(np.array([[measurement[4]]]))
        self.car_length = max(0.0, self.kf_length.x[0, 0])
        self.car_width = max(0.0, self.kf_width.x[0, 0])

        self.time_since_update = 0
        self.hits += 1
        if self.state == TrackState.TENTATIVE and self.hits >= self.confirm_hits:
            self.state = TrackState.CONFIRMED
        elif self.state in (TrackState.CONFIRMED, TrackState.LOST):
            self.state = TrackState.CONFIRMED

        self._update_color(color)

    def _update_color(self, color: Optional[str]) -> None:
        normalized = normalize_color_label(color)
        if not normalized:
            return
        self.color_counts[normalized] += 1
        self.total_color_votes += 1
        self.current_color = self.color_counts.most_common(1)[0][0]

    def get_color(self) -> Optional[str]:
        return self.current_color

    def get_color_confidence(self) -> float:
        if not self.current_color or self.total_color_votes == 0:
            return 0.0
        return self.color_counts[self.current_color] / float(self.total_color_votes)

    def get_state(self) -> np.ndarray:
        return np.array([
            self.cls,
            self.kf_pos.x[0, 0],
            self.kf_pos.x[1, 0],
            self.car_length,
            self.car_width,
            self.car_yaw,
        ], dtype=float)


class SortTracker:
    def __init__(self, max_age: int = 3, min_hits: int = 3, iou_threshold: float = 0.3, color_penalty: float = 0.3):
        self.tracks: List[Track] = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.color_penalty = color_penalty
        self.last_matches: List[Tuple[int, int]] = []

    def update(
        self,
        detections_carla: np.ndarray,
        detection_colors: Optional[List[Optional[str]]] = None,
    ) -> np.ndarray:
        """
        Update tracker with detections.
        detections_carla: Nx6 array [class, x_center, y_center, length, width, angle]
        detection_colors: optional list aligned with detections (None entries allowed)
        """
        if detections_carla is None:
            detections_carla = np.zeros((0, 6), dtype=float)
        detections_carla = np.asarray(detections_carla, dtype=float)
        self.last_matches = []

        for track in self.tracks:
            track.predict()

        active_tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        matched_indices: List[Tuple[int, int]] = []
        unmatched_detections = list(range(len(detections_carla)))
        unmatched_tracks = list(range(len(active_tracks)))
        det_colors = self._prepare_detection_colors(detection_colors, len(detections_carla))

        if len(detections_carla) > 0 and len(active_tracks) > 0:
            cost_matrix = iou_batch(detections_carla, active_tracks)
            for i, det_color in enumerate(det_colors):
                if not det_color:
                    continue
                for j, track in enumerate(active_tracks):
                    cost_matrix[i, j] += self._color_cost(det_color, track.get_color())

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
            color = det_colors[det_idx] if det_colors else None
            track.update(detections_carla[det_idx], color=color)
            track.history.append(track.get_state().copy())
            self.last_matches.append((track.id, det_idx))

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
            color = det_colors[det_idx] if det_colors else None
            new_track = Track(detections_carla[det_idx], confirm_hits=self.min_hits, color=color)
            self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        output_results = []
        for track in self.tracks:
            if track.state in (TrackState.CONFIRMED, TrackState.LOST):
                state = track.get_state()
                output_results.append(np.array([track.id, *state], dtype=float))

        return np.array(output_results) if output_results else np.array([])

    def _prepare_detection_colors(
        self,
        detection_colors: Optional[List[Optional[str]]],
        count: int,
    ) -> List[Optional[str]]:
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


def load_detections_from_file(filepath: str) -> np.ndarray:
    try:
        if os.path.getsize(filepath) == 0:
            return np.array([])

        data = pd.read_csv(
            filepath,
            header=None,
            sep=r'[,\s]+',
            engine='python',
            dtype=float,
        ).values

        if data.shape[1] != 6:
            raise ValueError(f"파일 {filepath}의 열 개수가 예상 (6개)와 다릅니다: {data.shape[1]}")
        return data

    except Exception as e:
        print(f"파일 로드 오류: {filepath}. 오류: {e}")
        return np.array([])


def main_tracking():
    input_folder = "/merge_dist_wbf_drop"
    file_pattern = os.path.join(input_folder, "merged_frame_*.txt")
    frame_files = sorted(glob.glob(file_pattern))

    if not frame_files:
        print(f"오류: '{input_folder}' 폴더에서 파일을 찾을 수 없습니다. 경로와 파일명을 확인해주세요.")
        return

    tracker = SortTracker(max_age=10, min_hits=3, iou_threshold=0.15)
    all_tracking_results = []

    print(f"총 {len(frame_files)}개의 프레임 파일 로드됨. 추적 시작...")

    for frame_idx, filepath in enumerate(frame_files):
        detections = load_detections_from_file(filepath)
        tracked_objects = tracker.update(detections, None)

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
                comments='',
            )
        except Exception as e:
            print(f"\n⚠️ 최종 결과 통합/저장 중 오류 발생: {e}")
    else:
        print("\n⚠️ 추적된 객체가 없습니다. (모든 프레임에서 Confirmed/Lost 상태의 트랙이 없었음)")


# main_tracking()
