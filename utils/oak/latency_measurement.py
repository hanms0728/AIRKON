#!/usr/bin/env python3
import depthai as dai
import cv2
import numpy as np
import time

pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setFps(60)

# RAW stream enable
xout = pipeline.createXLinkOut()
xout.setStreamName("raw")
cam.raw.link(xout.input)

with dai.Device(pipeline) as device:
    print("USB MODE:", device.getUsbSpeed())

    q = device.getOutputQueue("raw", maxSize=1, blocking=False)
    diffs = []

    while True:
        frame = q.tryGet()
        if not frame:
            continue

        # ---- 정확한 latency 측정 (host 타임 기준) ----
        latency = (time.time() - frame.getTimestamp().total_seconds()) * 1000
        diffs.append(latency)

        print(f"Latency: {latency:.2f} ms | Avg: {np.mean(diffs):.2f} ms | Std: {np.std(diffs):.2f}")

        # ---- RAW10 unpack ----
        raw = frame.getData()
        w, h = frame.getWidth(), frame.getHeight()

        arr = np.frombuffer(raw, dtype=np.uint8).copy()
        row_size = (w // 4) * 5
        arr = arr[:row_size * h]        # stride 제거
        arr = arr.reshape((h, row_size))

        img10 = np.zeros((h, w), dtype=np.uint16)

        for r in range(h):
            row = arr[r].reshape(-1, 5)
            unpack = np.zeros((row.shape[0] * 4,), dtype=np.uint16)

            unpack[0::4] = (row[:,0] << 2) | (row[:,4] & 0b00000011)
            unpack[1::4] = (row[:,1] << 2) | ((row[:,4] & 0b00001100) >> 2)
            unpack[2::4] = (row[:,2] << 2) | ((row[:,4] & 0b00110000) >> 4)
            unpack[3::4] = (row[:,3] << 2) | ((row[:,4] & 0b11000000) >> 6)

            img10[r] = unpack[:w]

        img16 = (img10 * 64).astype(np.uint16)

        # ---- W Sensor 기준 색상 변환 (IMX577: BGGR → BGR) ----
        rgb = cv2.cvtColor(img16, cv2.COLOR_BayerBG2BGR)

        cv2.imshow("RAW → RGB", rgb)

        if cv2.waitKey(1) == ord('q'):
            break