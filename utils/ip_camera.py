#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import cv2
import threading
import time
from collections import deque
import ffmpeg
import numpy as np
'''
python utils/ip_camera.py --capture-mode sequence --sequence-fps 1 --sequence-dir data

python utils/ip_camera.py --capture-mode snapshot --snapshot-dir real_image
'''

def parse_keys(value):
    """
    Accepts strings like "enter", "space", "s", "a,b" or digit codes and
    returns a list of OpenCV key codes.
    """
    if value is None:
        return []

    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    codes = []
    for part in parts:
        lower = part.lower()
        if lower in ("enter", "return"):
            codes.extend([10, 13])
            continue
        if lower == "space":
            codes.append(32)
            continue
        if lower.isdigit():
            codes.append(int(lower))
            continue
        codes.append(ord(part[0]))
    # Preserve order while removing duplicates
    seen = set()
    unique_codes = []
    for c in codes:
        if c in seen:
            continue
        seen.add(c)
        unique_codes.append(c)
    return unique_codes


class IPCameraStreamer:
    def __init__(
        self,
        capture_mode="snapshot",
        snapshot_dir="cam_28",
        snapshot_keys=None,
        sequence_dir="sequence_capture",
        sequence_fps=5.0,
        start_key=ord("s"),
        end_key=ord("e"),
    ):
        # 6개 카메라 구성
        self.camera_configs = [
            {'ip': '192.168.0.30', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
             'camera_id': 0, 'transport': 'tcp', 'width': 1536, 'height': 864},
            # {'ip': '192.168.0.25', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 1, 'transport': 'tcp', 'width': 1536, 'height': 864},
            #  {'ip': '192.168.0.36', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 2, 'transport': 'tcp', 'width': 1536, 'height': 864},
            # {'ip': '192.168.0.27', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 3, 'transport': 'tcp', 'width': 1536, 'height': 864},
            #  {'ip': '192.168.0.51', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 4, 'transport': 'tcp', 'width': 1536, 'height': 864},
            #  {'ip': '192.168.0.26', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 5, 'transport': 'tcp', 'width': 1536, 'height': 864},
            #  {'ip': '192.168.0.28', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 6, 'transport': 'tcp', 'width': 1536, 'height': 864},
            #  {'ip': '192.168.0.50', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 7, 'transport': 'tcp', 'width': 1536, 'height': 864},
            #  {'ip': '192.168.0.32', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 8, 'transport': 'tcp', 'width': 1536, 'height': 864},
            # {'ip': '192.168.0.21', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 9, 'transport': 'tcp', 'width': 1536, 'height': 864},
            # {'ip': '192.168.0.31', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 10, 'transport': 'tcp', 'width': 1536, 'height': 864},
            # {'ip': '192.168.0.35', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 11, 'transport': 'tcp', 'width': 1536, 'height': 864},
            #  {'ip': '192.168.0.29', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 12, 'transport': 'tcp', 'width': 1536, 'height': 864},
            # {'ip': '192.168.0.37', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 13, 'transport': 'tcp', 'width': 1536, 'height': 864},
             
        ]

        # 프레임 저장 버퍼(최신 1장)
        self.latest = {cfg['camera_id']: deque(maxlen=1) for cfg in self.camera_configs}

        self.captures = {}        # FFmpeg 프로세스 저장
        self.running = True
        self.connected = {cfg['camera_id']: False for cfg in self.camera_configs}
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # 모드/키 설정
        self.capture_mode = capture_mode
        self.snapshot_keys = set(snapshot_keys or (10, 13))
        self.sequence_dir = sequence_dir
        os.makedirs(self.sequence_dir, exist_ok=True)
        self.sequence_fps = max(float(sequence_fps), 0.1)
        self.sequence_interval = 1.0 / self.sequence_fps
        self.start_key = start_key
        self.end_key = end_key

        # 구간 저장 상태
        self.sequence_active = False
        # self.sequence_session_dir = None
        self.sequence_count = 0
        self.sequence_last_saved = 0.0

        # 카메라 스레드 시작
        self.threads = []
        for cfg in self.camera_configs:
            t = threading.Thread(target=self.camera_thread, args=(cfg,))
            t.daemon = True
            t.start()
            self.threads.append(t)

        # 시각화 스레드
        self.vis_thread = threading.Thread(target=self.visualizer)
        self.vis_thread.daemon = True
        self.vis_thread.start()

        print("[INFO] IPCameraStreamer initialized (6 cameras).")

    # --------------------------------------------------------------------
    # RTSP URL 생성
    # --------------------------------------------------------------------
    def create_urls(self, cfg):
        return [
            f"rtsp://{cfg['username']}:{cfg['password']}@{cfg['ip']}:{cfg['port']}/stream1",
            f"rtsp://{cfg['username']}:{cfg['password']}@{cfg['ip']}:{cfg['port']}/stream2",
        ]

    # --------------------------------------------------------------------
    # FFmpeg 프로세스 생성
    # --------------------------------------------------------------------
    def spawn_ffmpeg(self, url, width, height, transport):
        try:
            p = (
                ffmpeg
                .input(url,
                       rtsp_transport=transport,
                       fflags='nobuffer',
                       flags='low_delay',
                       probesize='16k',
                       analyzeduration='0')
                .filter('scale', width, height)
                .output('pipe:', format='rawvideo', pix_fmt='bgr24', vsync='passthrough')
                .global_args('-loglevel', 'error', '-nostats')
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            return p
        except Exception as e:
            print("[FFmpeg Error]", e)
            return None

    # --------------------------------------------------------------------
    # 카메라 스트리밍 스레드
    # --------------------------------------------------------------------
    def camera_thread(self, cfg):
        cam_id = cfg['camera_id']
        width, height = cfg['width'], cfg['height']
        bytes_per_frame = width * height * 3

        urls = self.create_urls(cfg)
        process = None

        print(f"[INFO] Trying camera {cam_id} connection...")

        # 1) 각 URL(stream1/stream2) 시도 + 실패하면 동일 URL 2번 추가 재시도
        for url in urls:
            for attempt in range(1, 4):  # 총 3번 시도
                print(f"[INFO] Camera {cam_id} → Try {attempt}/3 : {url}")
                process = self.spawn_ffmpeg(url, width, height, cfg['transport'])

                if not process:
                    continue

                # 첫 프레임 테스트
                try:
                    test = process.stdout.read(bytes_per_frame)
                    if test and len(test) == bytes_per_frame:
                        print(f"[OK] Camera {cam_id} connected: {url}")
                        self.connected[cam_id] = True
                        break
                except:
                    pass

                # 실패 → 프로세스 종료하고 다음 시도
                try:
                    process.stdout.close()
                    process.stderr.close()
                    process.wait(timeout=1)
                except:
                    pass
                process = None

            if self.connected[cam_id]:
                break

        if not self.connected[cam_id]:
            print(f"[FAIL] Camera {cam_id} failed to connect after retries.")
            return

        self.captures[cam_id] = process

        # 계속 프레임 읽기
        while self.running:
            try:
                in_bytes = process.stdout.read(bytes_per_frame)
                if not in_bytes:
                    print(f"[WARN] Camera {cam_id} lost connection.")
                    self.connected[cam_id] = False
                    break

                frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))
                d = self.latest[cam_id]
                d.clear()
                d.append(frame)

            except Exception as e:
                print(f"[ERROR] Camera {cam_id} streaming error:", e)
                break

        print(f"[INFO] Camera {cam_id} thread finished.")

    # --------------------------------------------------------------------
    # 실시간 시각화 스레드
    # --------------------------------------------------------------------
    def visualizer(self):
        while self.running:
            for cfg in self.camera_configs:
                cam_id = cfg['camera_id']
                if self.latest[cam_id]:
                    frame = self.latest[cam_id][-1]
                    small = cv2.resize(frame, (0, 0), fx=1, fy=1)
                    cv2.imshow(f"Camera {cam_id}", small)

            key = cv2.waitKey(1) & 0xFF
            if self.capture_mode == "snapshot":
                if key in self.snapshot_keys:
                    self.save_snapshot()
            elif self.capture_mode == "sequence":
                if key == self.start_key:
                    self.start_sequence_capture()
                if key == self.end_key:
                    self.stop_sequence_capture()

            if self.capture_mode == "sequence" and self.sequence_active:
                self.save_sequence_frame()
            time.sleep(0.01)


    # --------------------------------------------------------------------
    # Enter 키로 스냅샷 저장
    # --------------------------------------------------------------------
    def save_snapshot(self):
        timestamp = time.strftime("%H%M%S")
        # target_dir = os.path.join(self.snapshot_dir, timestamp)
        os.makedirs(self.snapshot_dir, exist_ok=True)

        saved = 0
        for cam_id, frames in self.latest.items():
            if not frames:
                continue
            frame = frames[-1].copy()
            filename = os.path.join(self.snapshot_dir, f"{cam_id}_{self.snapshot_dir}_{timestamp}.jpg")
            try:
                cv2.imwrite(filename, frame)
                saved += 1
            except Exception as e:
                print(f"[ERROR] Failed to save Enter snapshot for camera {cam_id}:", e)

        if saved:
            print(f"[SAVE] Enter snapshot saved ({saved} cameras) → {self.snapshot_dir}")
        else:
            print("[WARN] Enter pressed but no frames were available to save.")

    # --------------------------------------------------------------------
    # Start/End 키 사이 구간 동안 fps에 맞춰 저장
    # --------------------------------------------------------------------
    def start_sequence_capture(self):
        if self.sequence_active:
            return
        session_ts = time.strftime("%Y%m%d_%H%M%S")
        # self.sequence_session_dir = os.path.join(self.sequence_dir, session_ts)
        # os.makedirs(self.sequence_session_dir, exist_ok=True)
        self.sequence_count = 0
        self.sequence_last_saved = 0.0
        self.sequence_active = True
        print(f"[SEQ] Start capture → {self.sequence_dir} @ {self.sequence_fps} fps")

    def stop_sequence_capture(self):
        if not self.sequence_active:
            return
        self.sequence_active = False
        print(f"[SEQ] Stop capture (saved {self.sequence_count} frames per camera).")

    def save_sequence_frame(self):
        if not self.sequence_active:
            return
        now = time.time()
        if now - self.sequence_last_saved < self.sequence_interval:
            return
        self.sequence_last_saved = now

        timestamp = time.strftime("%H%M%S", time.localtime(now))
        saved = 0
        for cam_id, frames in self.latest.items():
            if not frames:
                continue
            frame = frames[-1].copy()
            filename = os.path.join(
                self.sequence_dir,
                f"cam{cam_id}_{timestamp}_{self.sequence_count:06d}.jpg"
            )
            try:
                cv2.imwrite(filename, frame)
                saved += 1
            except Exception as e:
                print(f"[ERROR] Failed to save sequence frame for camera {cam_id}:", e)

        if saved:
            self.sequence_count += 1

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=1)
        self.vis_thread.join(timeout=1)
        cv2.destroyAllWindows()
        print("[INFO] IPCameraStreamer stopped.")


# ----------------------------
# 실행
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IP Camera streamer with snapshot/sequence capture options.")
    parser.add_argument("--capture-mode", choices=["snapshot", "sequence"], default="snapshot",
                        help="snapshot: Enter 키로 1장 저장 / sequence: start~end 사이 fps에 맞춰 저장")
    parser.add_argument("--snapshot-dir", default="cam_28", help="스냅샷 저장 폴더")
    parser.add_argument("--snapshot-key", default="enter",
                        help="스냅샷을 트리거할 키. 콤마로 여러 개 가능 (예: enter,space)")
    parser.add_argument("--sequence-dir", default="sequence_capture", help="구간 저장용 기본 폴더")
    parser.add_argument("--sequence-fps", type=float, default=5.0, help="구간 저장 fps")
    parser.add_argument("--start-key", default="s", help="구간 저장 시작 키")
    parser.add_argument("--end-key", default="e", help="구간 저장 종료 키")
    args = parser.parse_args()

    snapshot_keys = parse_keys(args.snapshot_key)
    start_key = parse_keys(args.start_key)
    end_key = parse_keys(args.end_key)

    streamer = IPCameraStreamer(
        capture_mode=args.capture_mode,
        snapshot_dir=args.snapshot_dir,
        snapshot_keys=snapshot_keys or (10, 13),
        sequence_dir=args.sequence_dir,
        sequence_fps=args.sequence_fps,
        start_key=start_key[0] if start_key else ord("s"),
        end_key=end_key[0] if end_key else ord("e"),
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        streamer.stop()
