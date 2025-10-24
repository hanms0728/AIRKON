#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import math
import queue
import signal
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import deque
from collections import defaultdict

from src.inference_lstm_onnx import (
    ONNXTemporalRunner,
    decode_predictions,          # evaluation_utils에서 import됨
    tiny_filter_on_dets,        # geometry_utils 경유
    apply_homography,
    poly_from_tri,
    draw_pred_only,
    draw_bev_visualization,
    write_bev_labels,
)
import ffmpeg
from collections import deque
import threading, sys

# --- 기존 SimpleRTSPStreamer 대신 사용 ---
import cv2, time, threading
import numpy as np
from collections import deque

# ==== 추가: 전송 유틸 ====
import threading, queue, time, shutil, requests, os
from pathlib import Path

import socket, json

# 테스트용 더미데이터 받아서 ㄱㄱ
import glob, cv2, time, threading
from collections import deque
from pathlib import Path

class ImageFolderStreamer:
    """
    dataset/cam1/*.jpg, dataset/cam2/*.jpg 같은 폴더를 1fps 등으로 재생.
    - camera_cfgs 입력 대신 cam_dirs: {camera_id: folder_path}
    - latest 프레임 1장만 유지 (get_latest와 인터페이스 동일)
    """
    def __init__(self, cam_dirs: dict, fps: float = 1.0, loop: bool = False):
        self.cam_dirs = cam_dirs              # {1: "dataset/cam1", 2:"dataset/cam2"}
        self.fps = max(0.01, float(fps))
        self.dt = 1.0 / self.fps
        self.loop = loop
        self.latest = {cid: deque(maxlen=1) for cid in self.cam_dirs.keys()}
        self._threads = []
        self._running = False

        # 각 카메라 파일 리스트 정렬
        self._filelists = {}
        for cid, d in self.cam_dirs.items():
            pats = []
            for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
                pats.extend(glob.glob(str(Path(d) / ext)))
            files = sorted(pats)
            if not files:
                print(f"[FolderStreamer] cam{cid} 이미지가 없습니다: {d}")
            self._filelists[cid] = files

    def start(self):
        if self._running: return
        self._running = True
        for cid in self.cam_dirs.keys():
            th = threading.Thread(target=self._reader, args=(cid,), daemon=True)
            th.start()
            self._threads.append(th)
        print("[FolderStreamer] started")

    def stop(self):
        self._running = False
        for th in self._threads:
            th.join(timeout=1.0)
        self._threads.clear()
        print("[FolderStreamer] stopped")

    def _reader(self, cid: int):
        files = self._filelists.get(cid, [])
        if not files:
            print(f"[FolderStreamer] cam{cid} 파일 없음; 종료")
            return
        idx = 0
        last = 0.0
        while self._running:
            now = time.time()
            if now - last < self.dt:
                time.sleep(0.005)
                continue
            last = now

            fp = files[idx]
            img = cv2.imread(fp)
            if img is None:
                print(f"[FolderStreamer] cam{cid} 읽기 실패: {fp}")
            else:
                dq = self.latest[cid]
                dq.clear(); dq.append(img)

            idx += 1
            if idx >= len(files):
                if self.loop:
                    idx = 0
                else:
                    print(f"[FolderStreamer] cam{cid} 재생 완료")
                    break
        print(f"[FolderStreamer] cam{cid} thread exit")

    def get_latest(self, cam_id: int):
        dq = self.latest.get(cam_id)
        return dq[-1] if dq and len(dq) else None


class UDPSender:
    def __init__(self, host: str, port: int, fmt: str = "json", max_bytes: int = 65000):
        """
        host: 수신 서버 IP (로컬 테스트면 127.0.0.1)
        port: 수신 UDP 포트
        fmt : 'json' | 'text'  (보낼 포맷)
        max_bytes: UDP 패킷 최대 크기(분할 전송용 한계)
        """
        self.addr = (host, int(port))
        self.fmt = fmt
        self.max_bytes = max_bytes
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self):
        try: self.sock.close()
        except: pass

    def _pack_json(self, cam_id: int, ts: float, fname: str, bev_dets):
        """
        bev_dets 항목 예시(필요 최소 필드만): 
        [{"center":[x,y],"length":L,"width":W,"yaw":deg,"score":s}, ...]
        """
        msg = {
            "type": "bev_labels",
            "camera_id": cam_id,
            "timestamp": ts,
            "filename": fname,
            "items": [
                {
                    "center": [float(d["center"][0]), float(d["center"][1])],
                    "length": float(d["length"]),
                    "width": float(d["width"]),
                    "yaw": float(d["yaw"]),
                    "score": float(d["score"]),
                } for d in bev_dets
            ],
        }
        return json.dumps(msg, ensure_ascii=False).encode("utf-8")

    def _pack_text(self, cam_id: int, ts: float, fname: str, bev_txt_path: str):
        """
        기존 bev_labels/*.txt 파일 내용 라인 그대로 보내기 (라벨 파서가 이미 있다면 이게 편함)
        첫 줄에 메타를 붙이고, 그 다음 줄부터 라벨 라인을 붙여 보냄.
        """
        lines = []
        try:
            with open(bev_txt_path, "r", encoding="utf-8") as f:
                lines = [ln.rstrip("\n") for ln in f]
        except Exception:
            pass
        header = f"# cam_id={cam_id} ts={ts} file={Path(bev_txt_path).name}"
        payload = "\n".join([header] + lines)
        return payload.encode("utf-8")

    def send(self, cam_id: int, ts: float, bev_txt_path: str, bev_dets=None):
        """
        fmt='json'이면 파싱된 bev_dets를 JSON으로,
        fmt='text'면 bev_labels 텍스트 파일 내용을 그대로 보냄.
        큰 페이로드는 조각내서 순차 전송(간단한 분할 헤더 포함).
        """
        fname = Path(bev_txt_path).name
        if self.fmt == "json":
            payload = self._pack_json(cam_id, ts, fname, bev_dets or [])
        else:
            payload = self._pack_text(cam_id, ts, fname, bev_txt_path)

        # 필요 시 분할 전송(간단한 프레이밍)
        if len(payload) <= self.max_bytes:
            self.sock.sendto(payload, self.addr)
            return

        chunk_id = os.urandom(4).hex()
        total = (len(payload) + self.max_bytes - 1) // self.max_bytes
        for idx in range(total):
            part = payload[idx*self.max_bytes:(idx+1)*self.max_bytes]
            # 1줄짜리 헤더: chunk_id,total,idx\n + part
            prefix = f"CHUNK {chunk_id} {total} {idx}\n".encode("utf-8")
            self.sock.sendto(prefix + part, self.addr)


class OpenCVRTSPStreamer: # 오픈cvRTSPStreamer:
    """
    OpenCV VideoCapture 기반 RTSP 스트리머.
    - 카메라별 paths를 순차 시도
    - 최신 프레임 1장만 유지
    - ffmpeg 설치/파이프 이슈 우회
    """
    def __init__(self, camera_cfgs, transport="tcp"):
        self.cfgs = camera_cfgs
        self.latest = {c["camera_id"]: deque(maxlen=1) for c in self.cfgs}
        self._threads = []
        self._caps = {}
        self._running = False
        self.transport = transport  # "tcp" / "udp" (OpenCV에서는 URL에 영향 적음)

    def _urls(self, cfg):
        user, pw = cfg.get("username",""), cfg.get("password","")
        auth = f"{user}:{pw}@" if (user or pw) else ""
        ip, port = cfg["ip"], cfg.get("port", 554)
        # 카메라별 후보 경로
        paths = cfg.get("paths", ["/stream1","/stream2"])
        return [f"rtsp://{auth}{ip}:{port}{p}" for p in paths]

    def _open_cap(self, url):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        # 지연 줄이기 위한 힌트(드라이버/빌드에 따라 무시될 수 있음)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
        return cap

    def _reader(self, cfg):
        cam_id = cfg["camera_id"]
        urls = self._urls(cfg)

        while self._running:
            cap = None
            for i, url in enumerate(urls):
                print(f"[Cam{cam_id}] Try(OpenCV): {url}")
                cap = self._open_cap(url)
                # isOpened가 True여도 첫 read가 실패하는 경우가 있어 바로 테스트
                ok, frame = cap.read()
                if ok and frame is not None:
                    self._caps[cam_id] = cap
                    self.latest[cam_id].clear(); self.latest[cam_id].append(frame)
                    h, w = frame.shape[:2]
                    print(f"[Cam{cam_id}] ✅ connected (OpenCV) {w}x{h} @ {url}")
                    break
                else:
                    print(f"[Cam{cam_id}] ❌ failed: {url}")
                    cap.release()
                    cap = None
                    time.sleep(0.2)

            if cap is None:
                print(f"[Cam{cam_id}] all URLs failed. retry soon...")
                time.sleep(1.0)
                continue

            # streaming loop
            last_t, n = time.time(), 0
            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print(f"[Cam{cam_id}] stream ended. reconnect...")
                    break
                dq = self.latest[cam_id]
                dq.clear(); dq.append(frame)
                n += 1
                now = time.time()
                if now - last_t >= 5.0:
                    fps = n / (now - last_t)
                    print(f"[Cam{cam_id}] fps≈{fps:.2f}")
                    last_t, n = now, 0

            try:
                cap.release()
            except Exception:
                pass
            self._caps.pop(cam_id, None)
            time.sleep(0.3)

        print(f"[Cam{cam_id}] reader exit")

    def start(self):
        if self._running: return
        self._running = True
        for cfg in self.cfgs:
            th = threading.Thread(target=self._reader, args=(cfg,), daemon=True)
            th.start()
            self._threads.append(th)
        print("[Streamer(OpenCV)] started")

    def stop(self):
        self._running = False
        for th in self._threads:
            th.join(timeout=1.0)
        for cap in list(self._caps.values()):
            try: cap.release()
            except Exception: pass
        self._threads.clear(); self._caps.clear()
        print("[Streamer(OpenCV)] stopped")

    def get_latest(self, cam_id):
        dq = self.latest.get(cam_id)
        return dq[-1] if dq and len(dq) else None

# ===================== H 로더 (카메라 ID 기준 파일) =====================
def load_H_for_cam(calib_dir: str, cam_id: int) -> Optional[np.ndarray]:
    """
    calib_dir 안에서 cam{cam_id}.txt / .npy / .csv 등 찾기
    """
    base_candidates = [f"cam{cam_id}", f"{cam_id}"]
    for stem in base_candidates:
        for ext in (".txt",".npy",".csv"):
            p = Path(calib_dir) / (stem + ext)
            if p.is_file():
                try:
                    if p.suffix == ".npy":
                        H = np.load(p)
                    else:
                        H = np.loadtxt(p)
                    H = np.asarray(H, dtype=np.float64)
                    if H.size == 9:
                        H = H.reshape(3,3)
                    if H.shape == (3,3):
                        return H
                except Exception:
                    pass
    return None

# ===================== 실시간 추론 파이프라인 =====================
class EdgeInfer:
    def __init__(
        self,
        weights_path: str,
        output_root: str,
        img_size: Tuple[int,int] = (864, 1536),
        strides: Tuple[float,...] = (8,16,32),
        conf: float = 0.80,
        nms_iou: float = 0.20,
        topk: int = 50,
        score_mode: str = "obj*cls",
        bev_scale: float = 1.0,
        calib_dir: Optional[str] = None,
        use_cuda: bool = True,
        # sender=None,
        udp_sender=None,
    ):
        # self.sender = sender
        self.H_infer, self.W_infer = img_size
        self.strides = list(strides)
        self.conf = conf
        self.nms_iou = nms_iou
        self.topk = topk
        self.score_mode = score_mode
        self.bev_scale = bev_scale
        self.calib_dir = calib_dir
        os.makedirs(output_root, exist_ok=True)
        self.out_root = output_root
        self.udp_sender = udp_sender

        providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        # 카메라별 LSTM state 유지 → runner를 카메라별로
        self.runners: Dict[int, ONNXTemporalRunner] = {}
        self.H_cache: Dict[int, Optional[np.ndarray]] = {}

        # 카메라별 출력 폴더
        self.out_dirs: Dict[int, Dict[str,str]] = {}

        self.weights_path = weights_path
        print("[EdgeInfer] ready.")

    def _ensure_runner(self, cam_id: int):
        if cam_id in self.runners:
            return
        runner = ONNXTemporalRunner(
            self.weights_path,
            providers=("CUDAExecutionProvider","CPUExecutionProvider"),
            state_stride_hint=32,
            default_hidden_ch=256,
        )
        self.runners[cam_id] = runner

        # 폴더 준비
        cam_root = os.path.join(self.out_root, f"cam{cam_id}")
        dirs = {
            "img":  os.path.join(cam_root, "images"),
            "lab":  os.path.join(cam_root, "labels"),
            "bev_img": os.path.join(cam_root, "bev_images"),
            "bev_lab": os.path.join(cam_root, "bev_labels"),
        }
        for d in dirs.values(): os.makedirs(d, exist_ok=True)
        self.out_dirs[cam_id] = dirs

        # H 로딩
        if self.calib_dir:
            self.H_cache[cam_id] = load_H_for_cam(self.calib_dir, cam_id)
            if self.H_cache[cam_id] is None:
                print(f"[EdgeInfer] WARN: cam{cam_id} H not found in {self.calib_dir}")
        else:
            self.H_cache[cam_id] = None

    def process_frame(self, cam_id: int, frame_bgr: np.ndarray, stamp: float):
        self._ensure_runner(cam_id)
        runner = self.runners[cam_id]
        dirs = self.out_dirs[cam_id]
        H_img, W_img = frame_bgr.shape[:2]

        # resize + normalize
        img_bgr = cv2.resize(frame_bgr, (self.W_infer, self.H_infer), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = img_rgb.transpose(2,0,1).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)

        # 추론
        outs = runner.forward(x)
        dets = decode_predictions(
            outs, self.strides,
            clip_cells=None,
            conf_th=self.conf, nms_iou=self.nms_iou, topk=self.topk,
            score_mode=self.score_mode, use_gpu_nms=True
        )[0]
        dets = tiny_filter_on_dets(dets, min_area=20.0, min_edge=3.0)

        # 저장 이름 (타임스탬프 기반)
        name = f"{int(stamp*1000):013d}.jpg"
        save_img = os.path.join(dirs["img"], name)
        save_txt = os.path.join(dirs["lab"], Path(name).with_suffix(".txt").name)

        # 시각화 + 2D 라벨(원본 이미지 크기 기준 좌표)
        img, _= draw_pred_only(
            image_bgr=img_bgr, dets=dets,
            save_path_img=save_img, save_path_txt=save_txt,
            W=self.W_infer, H=self.H_infer, W0=W_img, H0=H_img
        )

        # ===== BEV =====
        H_img2ground = self.H_cache.get(cam_id)
        if H_img2ground is not None:
            # draw_pred_only가 원본 좌표로 변환한 삼각형을 반환하니,
            # 원본 해상도 기준으로 다시 읽어오자
            # (draw_pred_only 안에서 반환 리스트가 tri_orig_list 형태였음)
            # → 여기선 파일을 다시 읽지 않고, 동일 로직을 한 번 더 수행
            tri_orig_list: List[np.ndarray] = []
            sx, sy = float(W_img)/float(self.W_infer), float(H_img)/float(self.H_infer)
            for d in dets:
                tri = np.asarray(d["tri"], dtype=np.float32).copy()
                tri[:, 0] *= sx; tri[:, 1] *= sy
                tri_orig_list.append(tri)

            bev_dets = []
            if tri_orig_list:
                pred_stack_orig = np.asarray(tri_orig_list, dtype=np.float64)
                tris_bev = apply_homography(pred_stack_orig, H_img2ground) * float(self.bev_scale)
                for d, tri_bev in zip(dets, tris_bev):
                    if not np.all(np.isfinite(tri_bev)): continue
                    poly_bev = poly_from_tri(tri_bev)
                    # 길이/너비/각도
                    p0, p1, p2 = tri_bev
                    front_center = (p1 + p2) / 2.0
                    yaw = math.degrees(math.atan2((front_center - p0)[1], (front_center - p0)[0]))
                    yaw = (yaw + 180) % 360 - 180
                    # 평행사변형 4점에서 길이/너비 근사
                    edges = [np.linalg.norm(poly_bev[(i+1)%4]-poly_bev[i]) for i in range(4)]
                    length, width = float(max(edges)), float(min(edges))
                    center = poly_bev.mean(axis=0)

                    bev_dets.append({
                        "score": float(d["score"]),
                        "tri": tri_bev,
                        "poly": poly_bev,
                        "center": (float(center[0]), float(center[1])),
                        "length": length,
                        "width": width,
                        "yaw": float(yaw),
                        "front_edge": (p1, p2),
                    })

            # 저장
            bev_img = os.path.join(dirs["bev_img"], name)
            draw_bev_visualization(bev_dets, None, bev_img, f"cam{cam_id} | Pred BEV")
            bev_lab = os.path.join(dirs["bev_lab"], Path(name).with_suffix(".txt").name)
            write_bev_labels(bev_lab, bev_dets)

            # if hasattr(self, "sender") and self.sender:
            #     self.sender.enqueue(bev_lab, cam_id, stamp)
            if self.udp_sender is not None:
                try:
                    self.udp_sender.send(cam_id, stamp, bev_lab, bev_dets)
                except Exception as e:
                    print(f"[UDP] send error: {e}")
            
        return img

# ===================== 메인 (스트리머와 결합) =====================
def main():
    import argparse
    ap = argparse.ArgumentParser("Realtime Edge Infer")
    ap.add_argument("--weights", required=True, type=str)
    ap.add_argument("--output-dir", default="./inference_results_realtime", type=str)
    ap.add_argument("--img-size", default="864,1536", type=str)
    ap.add_argument("--strides", default="8,16,32", type=str)
    ap.add_argument("--conf", default=0.30, type=float)
    ap.add_argument("--nms-iou", default=0.20, type=float)
    ap.add_argument("--topk", default=50, type=int)
    ap.add_argument("--score-mode", default="obj*cls", choices=["obj","cls","obj*cls"])
    ap.add_argument("--bev-scale", default=1.0, type=float)
    ap.add_argument("--calib-dir", default="./calib", type=str)
    ap.add_argument("--every-n", default=2, type=int, help="프레임 N개마다 1회 추론")
    ap.add_argument("--transport", default="tcp", choices=["tcp","udp"])
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--udp-enable", action="store_true",
                    help="BEV 라벨을 UDP로 실시간 전송")
    ap.add_argument("--udp-host", default="127.0.0.1",
                    help="수신 서버 IP (동일 PC 테스트면 127.0.0.1)")
    ap.add_argument("--udp-port", type=int, default=50050,
                    help="수신 UDP 포트 (예: 50050)")
    ap.add_argument("--udp-format", choices=["json","text"], default="json",
                    help="UDP 페이로드 포맷 (json: 구조화, text: bev_labels 그대로)")
    ap.add_argument("--dummy-cam-dirs",
                default="",
                help="더미데이터 테스트: 예) cam1=dataset/cam1,cam2=dataset/cam2")
    ap.add_argument("--dummy-fps", type=float, default=1.0,
                    help="폴더 재생 FPS (기본 1)")
    ap.add_argument("--dummy-loop", action="store_true",
                    help="폴더 끝나면 반복 재생")
    args = ap.parse_args()

    H, W = map(int, args.img_size.split(","))
    strides = tuple(float(s) for s in args.strides.split(","))

    # UDP 전송기 생성
    udp_sender = None
    if args.udp_enable:
        udp_sender = UDPSender(args.udp_host, args.udp_port, fmt=args.udp_format)

    # 카메라 설정(예시: IP/계정/경로 후보) 근데 이거 직접 찾으셔야돼 이게 맞나??
    camera_cfgs = [
        {"camera_id": 1, "ip": "192.168.0.3", "port": 554, "username": "admin", "password": "zjsxmfhf"},
        {"camera_id": 2, "ip": "192.168.0.4", "port": 554, "username": "admin", "password": "zjsxmfhf"},
    ]

    streamer = None
    if args.dummy_cam_dirs:
        cam_dirs = {}
        for tok in args.dummy_cam_dirs.split(","):
            tok = tok.strip()
            if not tok: continue
            name, path = tok.split("=")
            name = name.strip().lower()
            cid = 1 if name.endswith("1") else (2 if name.endswith("2") else int(name.replace("cam","")))
            cam_dirs[cid] = path.strip()
        streamer = ImageFolderStreamer(cam_dirs=cam_dirs, fps=args.dummy_fps, loop=args.dummy_loop)
    else:
        streamer = OpenCVRTSPStreamer(camera_cfgs, transport=args.transport)

    streamer.start()
    if args.dummy_cam_dirs:
        active_cam_ids = sorted(cam_dirs.keys())  
    else:
        active_cam_ids = [cfg["camera_id"] for cfg in camera_cfgs]
    counters = defaultdict(int)

    infer = EdgeInfer(
        weights_path=args.weights,
        output_root=args.output_dir,
        img_size=(H, W),
        strides=strides,
        conf=args.conf, nms_iou=args.nms_iou, topk=args.topk,
        score_mode=args.score_mode, bev_scale=args.bev_scale,
        calib_dir=args.calib_dir, use_cuda=True,
        udp_sender=udp_sender,  
    )

    # GUI 여부
    def gui_available() -> bool:
        try:
            cv2.namedWindow("TEST"); cv2.destroyWindow("TEST")
            print("GUI 사용 가능")
            return True
        except Exception:
            print("GUI 사용 불가")
            return False
    USE_GUI = (not args.no_gui) and gui_available()

    print("[Main] running. Press Ctrl+C to stop.")
    running = True

    def _sigint(_1,_2):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _sigint)

    try:
        while running:
            now = time.time()
            for cam_id in active_cam_ids:

                frame = streamer.get_latest(cam_id)
                if frame is None:
                    # print(f"[Main] cam{cam_id} 프레임 없ㄷ다")
                    continue

                # if USE_GUI:
                #     window_name = f"cam{cam_id}"
                #     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 창 크기 조절 허용
                #     cv2.resizeWindow(window_name, 640, 360)          # 원하는 크기 (가로 x 세로)
                #     cv2.imshow(window_name, frame)
                #     print("뿌림")
                #     # cv2.imshow(f"cam{cam_id}", frame)
                #     if (cam_id == camera_cfgs[0]["camera_id"]) and (cv2.waitKey(1) & 0xFF) == 27:
                #         running = False; break

                counters[cam_id] += 1
                if counters[cam_id] % max(1, args.every_n) != 0:
                    continue

                # 추론
                img = infer.process_frame(cam_id, frame, now)

                # if USE_GUI and img:
                #     window_name = f"cam{cam_id}"
                #     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 창 크기 조절 허용
                #     cv2.resizeWindow(window_name, 640, 360)          # 원하는 크기 (가로 x 세로)
                #     cv2.imshow(window_name, img)
                #     # cv2.imshow(f"cam{cam_id}", img)
                #     if (cam_id == camera_cfgs[0]["camera_id"]) and (cv2.waitKey(1) & 0xFF) == 27:
                #         running = False; break
                if USE_GUI and img is not None:
                    window_name = f"cam{cam_id}"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 1600, 900)
                    cv2.imshow(window_name, img)
                    if (cam_id == camera_cfgs[0]["camera_id"]) and (cv2.waitKey(1) & 0xFF) == 27:
                        running = False
                        break

                


            time.sleep(0.002)
            if args.dummy_cam_dirs:
                no_frames = all(streamer.get_latest(cid) is None for cid in active_cam_ids)
                if no_frames:
                    running = False
    finally:
        streamer.stop()
        if udp_sender: udp_sender.close()
        if USE_GUI:
            try: cv2.destroyAllWindows()
            except Exception: pass
        print("[Main] stopped.")

if __name__ == "__main__":
    main()
