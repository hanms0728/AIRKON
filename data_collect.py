#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ffmpeg
import numpy as np
import cv2
import os
import time
import threading

# ============================================================
# 공통 설정
# ============================================================
WIDTH = 1536        # 전체 카메라 공통 해상도
HEIGHT = 864
NUM_FRAMES = 300    # 한 카메라당 저장할 프레임 수
TRANSPORT = "tcp"   # RTSP transport


# ------------------------------------------------------------
# 저장 디렉토리 내부에서 다음 파일 인덱스 찾기 (덮어쓰기 방지)
# ------------------------------------------------------------
def get_next_index(save_dir):
    """저장 디렉토리 내에서 다음 저장 번호 반환"""

    os.makedirs(save_dir, exist_ok=True)

    existing = [
        f for f in os.listdir(save_dir)
        if f.endswith(".jpg") and f[:4].isdigit()
    ]

    if not existing:
        return 1

    nums = [int(f[:4]) for f in existing]
    return max(nums) + 1


# ============================================================
# 카메라 스트림 캡처 함수
# ============================================================
def capture_rtsp_frames(cam):

    cam_name = cam["cam_name"]
    ip       = cam["ip"]
    port     = cam["port"]
    username = cam["username"]
    password = cam["password"]
    save_dir = os.path.abspath(os.path.expanduser(cam["save_dir"]))

    bytes_per_frame = WIDTH * HEIGHT * 3

    # 연결 후보 URL — stream1 → stream2
    urls = [
        f"rtsp://{username}:{password}@{ip}:{port}/stream1",
        f"rtsp://{username}:{password}@{ip}:{port}/stream2"
    ]

    print(f"\n[INFO] ====== {cam_name} connecting... ======")

    process = None

    # --------------------------------------------------------
    # URL별 3회씩 총 6회 연결 시도
    # --------------------------------------------------------
    for url in urls:
        for attempt in range(1, 4):
            print(f"[INFO] {cam_name} Try {attempt}/3 : {url}")

            try:
                process = (
                    ffmpeg
                    .input(url,
                           rtsp_transport=TRANSPORT,
                           fflags='nobuffer',
                           flags='low_delay',
                           probesize='16k',
                           analyzeduration='0')
                    .filter('scale', WIDTH, HEIGHT)
                    .output(
                        'pipe:',
                        format='rawvideo',
                        pix_fmt='bgr24',
                        vsync='passthrough'
                    )
                    .global_args('-loglevel', 'error', '-nostats')
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )
            except:
                process = None
                continue

            # 첫 프레임 테스트
            try:
                test_bytes = process.stdout.read(bytes_per_frame)
                if test_bytes and len(test_bytes) == bytes_per_frame:
                    print(f"[OK] {cam_name} Connected: {url}")
                    break
            except:
                pass

            # 실패하면 프로세스 종료
            try:
                process.stdout.close()
                process.stderr.close()
                process.wait(timeout=1)
            except:
                pass

            process = None

        if process:
            break

    if not process:
        print(f"[FAIL] {cam_name} could NOT connect after retries.")
        return

    # --------------------------------------------------------
    # 데이터 저장 시작
    # --------------------------------------------------------
    print(f"[INFO] {cam_name} Streaming started.")

    next_idx = get_next_index(save_dir)
    frame_count = 0
    start_time = time.time()

    try:
        while frame_count < NUM_FRAMES:
            in_bytes = process.stdout.read(bytes_per_frame)
            if not in_bytes:
                print(f"[WARN] {cam_name} stream ended")
                break

            frame = np.frombuffer(in_bytes, np.uint8).reshape((HEIGHT, WIDTH, 3))

            filename = os.path.join(save_dir, f"{next_idx:04d}.jpg")
            cv2.imwrite(filename, frame)

            frame_count += 1
            next_idx += 1

            if frame_count % 20 == 0:
                elapsed = time.time() - start_time
                print(f"[INFO] {cam_name} {frame_count}/{NUM_FRAMES} saved ({elapsed:.1f}s)")

    except KeyboardInterrupt:
        print(f"[INFO] {cam_name} interrupted by user")

    finally:
        try:
            process.stdout.close()
            process.stderr.close()
            process.wait(timeout=1)
        except:
            pass

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"[INFO] {cam_name} DONE → {frame_count} frames ({elapsed:.2f}s, {fps:.2f}fps)")


# ============================================================
# 6개 카메라 실행
# ============================================================
def run_multi_cameras():

    cams = [
        {"cam_name": "cam1", "ip": "192.168.0.32", "port": 554, "username": "admin", "password": "zjsxmfhf", "save_dir": "dataset/cam1"},
        #{"cam_name": "cam2", "ip": "192.168.0.21", "port": 554, "username": "admin", "password": "zjsxmfhf", "save_dir": "dataset/cam2"},
        {"cam_name": "cam3", "ip": "192.168.0.29", "port": 554, "username": "admin", "password": "zjsxmfhf", "save_dir": "dataset/cam3"},
        {"cam_name": "cam4", "ip": "192.168.0.27", "port": 554, "username": "admin", "password": "zjsxmfhf", "save_dir": "dataset/cam4"},
        {"cam_name": "cam5", "ip": "192.168.0.36", "port": 554, "username": "admin", "password": "zjsxmfhf", "save_dir": "dataset/cam5"},
        {"cam_name": "cam6", "ip": "192.168.0.37", "port": 554, "username": "admin", "password": "zjsxmfhf", "save_dir": "dataset/cam6"}
    ]

    threads = []

    for cam in cams:
        t = threading.Thread(target=capture_rtsp_frames, kwargs={"cam": cam}, daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("\n[INFO] ===== ALL CAMERAS DONE =====")


# ============================================================
if __name__ == "__main__":
    run_multi_cameras()