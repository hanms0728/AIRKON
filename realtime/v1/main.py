from realtime_edge import IPCameraStreamerUltraLL as Streamer
from batch_infer import BatchedTemporalRunner
from src.inference_lstm_onnx import (
    decode_predictions,       
    tiny_filter_on_dets,       
    apply_homography,
    poly_from_tri,
    compute_bev_properties
)
from pathlib import Path
import argparse, signal, time, math, threading
import onnxruntime as ort
import numpy as np
import cv2
import queue
from typing import Optional, Dict
import socket, json
import os

# 잘돌아야 정상
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

class InferWorker(threading.Thread):
    """
    - Streamer에서 최신 프레임을 모아 배치 추론
    - UDP 전송(옵션)
    - GUI 큐에 (cam_id, resized_bgr, dets) 전달
    """
    def __init__(self, *, streamer, cam_ids, img_hw, strides, onnx_path,
                 score_mode="obj*cls", conf=0.3, nms_iou=0.2, topk=50,
                 calib_dir=None, bev_scale=1.0, providers=None,
                 gui_queue=None, udp_sender=None, backend="onnxruntime",
                 trt_options=None):
        super().__init__(daemon=True)
        self.streamer = streamer
        self.cam_ids = list(cam_ids)
        self.H, self.W = img_hw
        self.strides = list(map(float, strides))
        self.score_mode = score_mode
        self.conf = float(conf); self.nms_iou = float(nms_iou)
        self.topk = int(topk); self.bev_scale = float(bev_scale)
        self.gui_queue = gui_queue
        self.udp_sender = udp_sender
        self.stop_evt = threading.Event()
        
        self.H_cache: Dict[int, Optional[np.ndarray]] = {}

        backend = (backend or "onnxruntime").lower()
        trt_options = dict(trt_options or {})
        if backend == "tensorrt":
            from batch_infer_trt import TensorRTBatchedTemporalRunner
            self.runner = TensorRTBatchedTemporalRunner(
                onnx_path=onnx_path,
                cam_ids=self.cam_ids,
                img_size=(self.H, self.W),
                temporal="lstm",
                state_stride_hint=32,
                default_hidden_ch=256,
                **trt_options,
            )
        else:
            if providers is None:
                providers = ["CUDAExecutionProvider","CPUExecutionProvider"] \
                    if (ort.get_device().upper() == "GPU") else ["CPUExecutionProvider"]
            self.runner = BatchedTemporalRunner(
                onnx_path=onnx_path,
                cam_ids=self.cam_ids,
                img_size=(self.H, self.W),
                temporal="lstm", 
                providers=providers,
                state_stride_hint=32,
                default_hidden_ch=256,
            )
        # 호모그래피 캐시(없으면 None)
        self.H_cache = {cid: (load_H_for_cam(calib_dir, cid) if calib_dir else None)
                        for cid in self.cam_ids}
        for cid in self.cam_ids:
            if self.H_cache[cid] is None:
                    print(f"[EdgeInfer] WARN: cam{cid} H not found in {calib_dir}") 

    def _preprocess(self, frame_bgr):
        """BGR → (H,W) 리사이즈 → RGB CHW float32[0,1] + 리사이즈 BGR(시각화용)"""
        if frame_bgr.shape[:2] != (self.H, self.W):
            bgr = cv2.resize(frame_bgr, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        else:
            bgr = frame_bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2,0,1).astype(np.float32) / 255.0
        return chw, bgr

    def _decode(self, outs):
        dets = decode_predictions(
            outs, self.strides,
            clip_cells=None,
            conf_th=self.conf, nms_iou=self.nms_iou,
            topk=self.topk, score_mode=self.score_mode,
            use_gpu_nms=True
        )[0]
        return tiny_filter_on_dets(dets, min_area=20.0, min_edge=3.0)

    def _make_bev(self, dets, H_img2ground):
        if H_img2ground is None or not dets:
            print("호모그래피나 dets가 없음")
            return []
        tris = np.asarray([d["tri"] for d in dets], dtype=np.float64)
        tris_bev = apply_homography(tris, H_img2ground) * self.bev_scale
        bev = []
        for d, tri_bev in zip(dets, tris_bev):
            if not np.all(np.isfinite(tri_bev)): continue
            poly_bev = poly_from_tri(tri_bev)
            props = compute_bev_properties(tri_bev)
            if props is None: continue
            center, length, width, yaw, front_edge = props
            bev.append({
                "score": float(det["score"]),
                "tri": tri_bev,
                "poly": poly_bev,
                "center": center,
                "length": length,
                "width": width,
                "yaw": yaw,
                "front_edge": front_edge,
            })
            
            '''p0, p1, p2 = tri_bev
            front_center = (p1 + p2) / 2.0
            yaw = math.degrees(math.atan2((front_center - p0)[1], (front_center - p0)[0]))
            yaw = (yaw + 180) % 360 - 180
            edges = [np.linalg.norm(poly_bev[(i+1)%4]-poly_bev[i]) for i in range(4)]
            length, width = float(max(edges)), float(min(edges))
            center = poly_bev.mean(axis=0)
            bev.append({
                "score": float(d["score"]),
                "tri": tri_bev,
                "poly": poly_bev,
                "center": (float(center[0]), float(center[1])),
                "length": length, "width": width, "yaw": float(yaw),
                "front_edge": (p1, p2),
            })'''
        return bev

    def run(self):
        wrk = StageTimer(name="WORKER", print_every=5.0)

        while not self.stop_evt.is_set():
            # 1) 최신 프레임 수집
            with wrk.span("grab"):
                ready = True
                imgs_chw = {}
                bgr_for_gui = {}
                for cid in self.cam_ids:
                    fr= self.streamer.get_latest(cid)
                    # ts_capture = time.time() # frame에서 얻어온 시각
                    if fr is None:
                        ready = False
                        break
                    with wrk.span("preproc"):
                        fr, ts_capture = fr
                        chw, bgr = self._preprocess(fr)
                    imgs_chw[cid] = chw
                    bgr_for_gui[cid] = bgr

            if not ready:
                time.sleep(0.002)
                wrk.bump()
                continue

            # 2) 배치 주입
            with wrk.span("enqueue"):
                for cid, chw in imgs_chw.items():
                    self.runner.enqueue_frame(cid, chw)

            # 3) 배치 실행
            with wrk.span("infer"):
                per_cam_outs = self.runner.run_if_ready()
            if per_cam_outs is None:
                time.sleep(0.001)
                wrk.bump()
                continue

            ts = time.time()

            # 4) 디코드/BEV/UDP/GUI 큐
            for cid, outs in per_cam_outs.items():
                with wrk.span("decode"):
                    dets = self._decode(outs)

                with wrk.span("bev"):
                    bev = self._make_bev(dets, self.H_cache.get(cid))

                if self.udp_sender is not None:
                    with wrk.span("udp"):
                        try:
                            self.udp_sender.send(cam_id=cid, ts=ts, bev_dets=bev)
                            print(bev)
                        except Exception as e:
                            print(f"[UDP] send error cam{cid}: {e}")

                # GUI 큐로 전달(여기서 큐 대기/드롭이 발생할 수 있음)
                with wrk.span("gui.put"):
                    try:
                        self.gui_queue.put_nowait((cid, bgr_for_gui[cid], dets, ts_capture))
                    except queue.Full:
                        try:
                            _ = self.gui_queue.get_nowait()
                            self.gui_queue.put_nowait((cid, bgr_for_gui[cid], dets))
                        except Exception:
                            pass

            wrk.bump()

    # def run(self):
        # while not self.stop_evt.is_set():
        #     # 최신 프레임 수집
        #     ready = True
        #     imgs_chw = {}
        #     bgr_for_gui = {}
        #     for cid in self.cam_ids:
        #         fr = self.streamer.get_latest(cid)
        #         if fr is None:
        #             ready = False
        #             break
        #         chw, bgr = self._preprocess(fr)
        #         imgs_chw[cid] = chw
        #         bgr_for_gui[cid] = bgr

        #     if not ready:
        #         time.sleep(0.002)
        #         continue

        #     # 배치 싹 실행
        #     for cid, chw in imgs_chw.items():
        #         self.runner.enqueue_frame(cid, chw)
        #     per_cam_outs = self.runner.run_if_ready()
        #     if per_cam_outs is None:
        #         # 프레임 사이클이 덜 찼다면 잠깐 쉼
        #         time.sleep(0.001)
        #         continue

        #     # 전처리한거 udp로 넘기기 + GUI 큐에넣기
        #     ts = time.time()
        #     for cid, outs in per_cam_outs.items():
        #         dets = self._decode(outs)
        #         # UDP
        #         if self.udp_sender is not None:
        #             bev = self._make_bev(dets, self.H_cache.get(cid))
        #             try:
        #                 self.udp_sender.send(cam_id=cid, timestamp=ts, bev=bev)
        #             except Exception as e:
        #                 print(f"[UDP] send error cam{cid}: {e}")
        #         # GUI 큐 (스레드-세이프)
        #         try:
        #             self.gui_queue.put_nowait((cid, bgr_for_gui[cid], dets))
        #         except queue.Full:
        #             # 최신만 보여주면 되면, 오래된 항목 하나 버리고 재시도(드롭-올더스트)
        #             try:
        #                 _ = self.gui_queue.get_nowait()
        #                 self.gui_queue.put_nowait((cid, bgr_for_gui[cid], dets))
        #             except Exception:
        #                 pass

    def stop(self):
        self.stop_evt.set()

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

    def _pack_json(self, cam_id: int, ts: float, bev_dets):
        """
        bev_dets 항목 예시(필요 최소 필드만): 
        [{"center":[x,y],"length":L,"width":W,"yaw":deg,"score":s}, ...]
        """
        msg = {
            "type": "bev_labels",
            "camera_id": cam_id,
            "timestamp": ts,
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

    def _pack_text(self, cam_id: int, ts: float, fname: str, bev_txt_path: str): # 안쓸거임
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

    def send(self, cam_id: int, ts: float, bev_dets=None):
        """
        fmt='json'이면 파싱된 bev_dets를 JSON으로,
        fmt='text'면 bev_labels 텍스트 파일 내용을 그대로 보냄.
        큰 페이로드는 조각내서 순차 전송(간단한 분할 헤더 포함).
        """
        if self.fmt == "json":
            payload = self._pack_json(cam_id, ts, bev_dets or [])
        # else:
            # payload = self._pack_text(cam_id, ts, fname, bev_txt_path)

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

class FPSTicker:
    """
    GUI 루프의 FPS를 일정하게 유지하고, 'q' 입력으로 종료할 수 있게 도와주는 유틸.
    """
    def __init__(self, target_fps=30, print_every=5.0, name="GUI"):
        self.target_fps = max(1, int(target_fps))
        self.spf = 1.0 / self.target_fps  # seconds per frame
        self.next_deadline = time.time()
        self.running = True
        
        # FPS 측정용
        self._print_every = float(print_every)
        self._name = str(name)
        self._cnt = 0
        self._t0 = time.time()
        
    def set_fps(self, target_fps: int):
        self.target_fps = max(1, int(target_fps))
        self.spf = 1.0 / self.target_fps

    def tick(self):
        """한 프레임마다 호출 — FPS 안정화 + 'q' 입력 감지"""
        if not self.running:
            return False

        now = time.time()

        # FPS 유지 (너무 빠를 때 살짝 sleep)
        if now < self.next_deadline:
            time.sleep(self.next_deadline - now)
        self.next_deadline = time.time() + self.spf

        # GUI 이벤트 처리 + 종료 감지
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[FPSTicker] 'q' pressed → stopping.")
            self.running = False
            return False
        
        # 출력 fps 로그용,, 주석 가능
        self._cnt += 1
        if self._print_every > 0:
            t = time.time()
            if (t - self._t0) >= self._print_every:
                fps_out = self._cnt / (t - self._t0)
                print(f"[FPSTicker-{self._name}] render fps ≈ {fps_out:.2f} (target={self.target_fps})")
                self._cnt = 0
                self._t0 = t

        return True

import time
from contextlib import contextmanager

class StageTimer:
    """스테이지별 누적 시간/횟수 집계 + 주기적 출력"""
    def __init__(self, name="prof", print_every=5.0):
        self.name = name
        self.print_every = float(print_every)
        self.t0 = time.perf_counter()
        self.last_print = self.t0
        self.sum = {}   # stage -> seconds accumulated
        self.cnt = {}   # stage -> count
        self.tmp = {}   # stage -> start_time

    @contextmanager
    def span(self, stage: str):
        t = time.perf_counter()
        self.tmp[stage] = t
        try:
            yield
        finally:
            dt = time.perf_counter() - t
            self.sum[stage] = self.sum.get(stage, 0.0) + dt
            self.cnt[stage] = self.cnt.get(stage, 0) + 1

    def bump(self):
        """주기적으로 평균 ms 출력"""
        now = time.perf_counter()
        if (now - self.last_print) >= self.print_every:
            parts = []
            for k in sorted(self.sum.keys()):
                s = self.sum[k]
                n = max(1, self.cnt.get(k, 1))
                parts.append(f"{k}={1000.0*s/n:.2f}ms")
                # 리셋(롤링)
                self.sum[k] = 0.0
                self.cnt[k] = 0
            if parts:
                print(f"[{self.name}] " + " | ".join(parts))
            self.last_print = now

def main():
    ap = argparse.ArgumentParser("Realtime Edge Infer")
    ap.add_argument("--weights", required=True, type=str)
    ap.add_argument("--backend", default="onnxruntime",
                    choices=["onnxruntime","tensorrt"],
                    help="Select inference backend.")
    ap.add_argument("--trt-engine", type=str,
                    help="Existing TensorRT engine path (optional).")
    ap.add_argument("--trt-save-engine", type=str,
                    help="Where to save a newly built TensorRT engine.")
    ap.add_argument("--trt-workspace-mb", type=int, default=2048)
    ap.add_argument("--trt-fp16", action="store_true")
    ap.add_argument("--trt-max-batch", type=int, default=None)
    ap.add_argument("--img-size", default="864,1536", type=str)
    ap.add_argument("--strides", default="8,16,32", type=str)
    ap.add_argument("--conf", default=0.30, type=float)
    ap.add_argument("--nms-iou", default=0.20, type=float)
    ap.add_argument("--topk", default=50, type=int)
    ap.add_argument("--score-mode", default="obj*cls", choices=["obj","cls","obj*cls"])
    ap.add_argument("--bev-scale", default=1.0, type=float)
    ap.add_argument("--calib-dir", default="./calib", type=str)
    ap.add_argument("--transport", default="tcp", choices=["tcp","udp"])
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--udp-enable", action="store_true")
    ap.add_argument("--udp-host", default="127.0.0.1")
    ap.add_argument("--udp-port", type=int, default=50050)
    ap.add_argument("--udp-format", choices=["json","text"], default="json")
    ap.add_argument("--visual-size", default="216,384", type=str)
    ap.add_argument("--target-fps", default=30, type=int)
    args = ap.parse_args()
    backend = args.backend.lower()
    
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
    
    H, W = map(int, args.img_size.split(","))
    strides = tuple(float(s) for s in args.strides.split(","))
    vis_H, vis_W = map(int, args.visual_size.split(","))
    Target_fps = args.target_fps
    
    # 카메라에서 프레임 받아오기 
    camera_configs = [
        {
            'ip': '192.168.0.3',
            'port': 554,
            'username': 'admin',
            'password': 'zjsxmfhf',
            'camera_id': 1,
            'width': W,
            'height': H,
            'force_tcp': (args.transport == "tcp"),   # UDP 우선(저지연)
        },
        {
            'ip': '192.168.0.2',
            'port': 554,
            'username': 'admin',
            'password': 'zjsxmfhf',
            'camera_id': 2,
            'width': W,
            'height': H,
            'force_tcp': (args.transport == "tcp"),    # 이 카메라만 TCP 강제(손실/초록깨짐 방지)
        },
    ]
    
    cam_ids  = sorted([cfg["camera_id"] for cfg in camera_configs])
    # print(cam_ids)
    shutdown_evt = threading.Event()
    
    streamer = Streamer(
        camera_configs,
        show_windows=False,           # 미리보기는 본 루프에서 처리하거나 꺼둠
        target_fps=Target_fps,
        snapshot_dir=None,
        snapshot_interval_sec=None,
        catchup_seconds=0.5,
        overlay_ts=False,
        laytency_check=False
    )
    
    weights_for_runner: Optional[str] = args.weights
    trt_engine_path = args.trt_engine
    if backend == "tensorrt" and not trt_engine_path:
        w_lower = args.weights.lower()
        if w_lower.endswith(".engine") and os.path.isfile(args.weights):
            trt_engine_path = args.weights
            weights_for_runner = None
    if backend != "tensorrt" and weights_for_runner is None:
        raise ValueError("ONNX weights are required for the onnxruntime backend.")

    providers = None
    if backend == "onnxruntime":
        providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
        if args.no_cuda or ort.get_device().upper() != "GPU":
            providers = ["CPUExecutionProvider"]

    trt_options = {
        "engine_path": trt_engine_path,
        "save_engine_path": args.trt_save_engine,
        "workspace_mb": args.trt_workspace_mb,
        "fp16": args.trt_fp16,
        "max_batch_size": args.trt_max_batch,
    }
    
    
    udp_sender = None
    if args.udp_enable:
        udp_sender = UDPSender(args.udp_host, args.udp_port, fmt=args.udp_format)

    # 시작
    streamer.start()
    # -------- GUI 큐 & 워커 --------
    gui_queue = queue.Queue(maxsize=128)
    worker = InferWorker(
        streamer=streamer, cam_ids=cam_ids, img_hw=(H, W), strides=strides,
        onnx_path=weights_for_runner, score_mode=args.score_mode, conf=args.conf, nms_iou=args.nms_iou, topk=args.topk,
        calib_dir=args.calib_dir, bev_scale=args.bev_scale, providers=providers,
        gui_queue=gui_queue, udp_sender=udp_sender, backend=backend, trt_options=trt_options 
    )
    worker.start()
    # -------- 메인 스레드: 오직 GUI --------
    # 창 준비
    if USE_GUI:
        for cid in cam_ids:
            cv2.namedWindow(f"cam{cid}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"cam{cid}", vis_W, vis_H)

    running = True
    def _sigint(sig, frm):
        # ESC나 Ctrl+C 시 모두 같은 경로로 종료
        try:
            ticker.running = False
        except Exception:
            pass
        shutdown_evt.set() 
        # 추가로 창 이벤트를 깨워서 waitKey가 즉시 빠지도록
        try:
            if USE_GUI:
                cv2.waitKey(1)
        except Exception:
            pass

    signal.signal(signal.SIGINT, _sigint)
    ticker = FPSTicker(target_fps=Target_fps)
    gui_timer = StageTimer(name="GUI", print_every=5.0)

    # try:
    #     while ticker.running:
    #         # GUI 큐에서 결과 꺼내 그리기
    #         try:
    #             cid, bgr, dets = gui_queue.get(timeout=0.005)
    #         except queue.Empty:
    #             if USE_GUI: cv2.waitKey(1)
    #             continue

    #         if not USE_GUI:
    #             continue

    #         vis = bgr.copy()
    #         for d in dets:
    #             tri = np.asarray(d["tri"], dtype=np.int32)
    #             poly4 = poly_from_tri(tri).astype(np.int32)
    #             cv2.polylines(vis, [poly4], isClosed=True, color=(0,255,0), thickness=2)
                
    #             s = f"{d.get('score', 0.0):.2f}" 
    #             p0 = (int(tri[0,0]), int(tri[0,1]))
    #             cv2.putText(vis, s, p0, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            
    #         cv2.imshow(f"cam{cid}", vis)
    #         ticker.tick()
    try:
        while ticker.running:
            with gui_timer.span("gui.frame"):
                # GUI 큐에서 결과 꺼내 그리기
                with gui_timer.span("gui.get"):
                    try:
                        cid, bgr, dets, ts_capture = gui_queue.get(timeout=0.005) 
                    except ValueError:
                        cid, bgr, dets = gui_queue.get(timeout=0.005) 
                    except queue.Empty:
                        if USE_GUI: cv2.waitKey(1)
                        if not ticker.tick():
                            break
                        # bump는 루프마다 가볍게 호출해도 부담 거의 없음
                        gui_timer.bump()
                        continue

                if not USE_GUI:
                    if not ticker.tick():
                        break
                    gui_timer.bump()
                    continue

                with gui_timer.span("gui.draw"):
                    vis = bgr.copy()
                    for d in dets:
                        tri = np.asarray(d["tri"], dtype=np.int32)
                        poly4 = poly_from_tri(tri).astype(np.int32)
                        cv2.polylines(vis, [poly4], isClosed=True, color=(0,255,0), thickness=2)
                        s = f"{d.get('score', 0.0):.2f}"
                        p0 = (int(tri[0,0]), int(tri[0,1]))
                        cv2.putText(vis, s, p0, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

                with gui_timer.span("gui.show"):
                    cv2.imshow(f"cam{cid}", vis)
                    e2e_ms = (time.time() - ts_capture) * 1000.0
                    print(f"++++++++++++++캡처시점-렌더시점: {e2e_ms}")
                    if not ticker.tick():
                        break

            gui_timer.bump()
    finally:
        worker.stop()
        worker.join(timeout=1)
        streamer.stop()
        if udp_sender: udp_sender.close()
        if USE_GUI:
            try: cv2.destroyAllWindows()
            except: pass
        print("[Main] stopped.")


if __name__ == "__main__":
    main()
