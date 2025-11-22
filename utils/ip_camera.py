#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import threading
import time
from collections import deque
import ffmpeg
import numpy as np


class IPCameraStreamer:
    def __init__(self):
        # 6개 카메라 구성
        self.camera_configs = [
            # {'ip': '192.168.0.51', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 1, 'transport': 'tcp', 'width': 1536, 'height': 864},

            # {'ip': '192.168.0.35', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 2, 'transport': 'tcp', 'width': 1536, 'height': 864},

            # {'ip': '192.168.0.25', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 3, 'transport': 'tcp', 'width': 1536, 'height': 864},

            # {'ip': '192.168.0.30', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 4, 'transport': 'tcp', 'width': 1536, 'height': 864},

            # {'ip': '192.168.0.50', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 5, 'transport': 'tcp', 'width': 1536, 'height': 864},

            # {'ip': '192.168.0.21', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 6, 'transport': 'tcp', 'width': 1536, 'height': 864},
            # {'ip': '192.168.0.36', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 1, 'transport': 'tcp', 'width': 1536, 'height': 864},

            # {'ip': '192.168.0.32', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 2, 'transport': 'tcp', 'width': 1536, 'height': 864},

            # {'ip': '192.168.0.31', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 3, 'transport': 'tcp', 'width': 1536, 'height': 864},

            # {'ip': '192.168.0.29', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 4, 'transport': 'tcp', 'width': 1536, 'height': 864},

            # {'ip': '192.168.0.37', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 5, 'transport': 'tcp', 'width': 1536, 'height': 864},

            # {'ip': '192.168.0.27', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 6, 'transport': 'tcp', 'width': 1536, 'height': 864},


            # {'ip': '192.168.0.26', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
            #  'camera_id': 7, 'transport': 'tcp', 'width': 1536, 'height': 864},

            {'ip': '192.168.0.28', 'port': 554, 'username': 'admin', 'password': 'zjsxmfhf',
             'camera_id': 7, 'transport': 'tcp', 'width': 1536, 'height': 864},



        ]

        # 프레임 저장 버퍼(최신 1장)
        self.latest = {cfg['camera_id']: deque(maxlen=1) for cfg in self.camera_configs}
        # 최초 프레임 저장 여부
        self.saved_first_frame = {cfg['camera_id']: False for cfg in self.camera_configs}

        self.captures = {}        # FFmpeg 프로세스 저장
        self.running = True
        self.connected = {cfg['camera_id']: False for cfg in self.camera_configs}
        self.save_dir = "first_frames"
        os.makedirs(self.save_dir, exist_ok=True)
        self.snapshot_dir = "cam_26"
        os.makedirs(self.snapshot_dir, exist_ok=True)

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

                # 각 카메라 첫 프레임만 저장
                if not self.saved_first_frame[cam_id]:
                    filename = os.path.join(self.save_dir, f"cam_{cam_id}_{cfg['ip'].split('.')[-1]}.jpg")
                    try:
                        cv2.imwrite(filename, frame)
                        print(f"[SAVE] Camera {cam_id} first frame saved: {filename}")
                        self.saved_first_frame[cam_id] = True
                    except Exception as e:
                        print(f"[ERROR] Failed to save first frame for camera {cam_id}:", e)

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
            if key in (10, 13):
                self.save_snapshot()
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
            filename = os.path.join(self.snapshot_dir, f"{self.snapshot_dir}_{timestamp}.jpg")
            try:
                cv2.imwrite(filename, frame)
                saved += 1
            except Exception as e:
                print(f"[ERROR] Failed to save Enter snapshot for camera {cam_id}:", e)

        if saved:
            print(f"[SAVE] Enter snapshot saved ({saved} cameras) → {self.snapshot_dir}")
        else:
            print("[WARN] Enter pressed but no frames were available to save.")

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
    streamer = IPCameraStreamer()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        streamer.stop()
