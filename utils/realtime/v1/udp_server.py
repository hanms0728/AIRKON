#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import queue
import socket
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from utils.combine.coop_bev_labels import obb_to_quad, distinct_colors  
from utils.merge.merge_dist_wbf import (  
    CAMERA_SETUPS, cluster_by_aabb_iou, fuse_cluster_weighted
) 
from utils.sort.draw import color_for_track, obb_to_quad as obb_to_quad_draw  
from utils.sort.tracker import SortTracker  

# ---일단 저장용---
import os, gzip, json, csv, hashlib, threading, queue, time
from datetime import datetime, timezone

def _md5sum(path, bufsize=1<<20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b: break
            h.update(b)
    return h.hexdigest()

class RollingJsonlSaver:
    """
    kind별(예: 'fused', 'tracks')로 jsonl.gz 파일을 일정 간격/행수로 롤링하여 저장.
    - 파일 닫을 때 md5 계산해서 manifest.csv에 기록
    - meta.json 1회 생성
    - 비동기 저장(메인 루프 비차단)
    """
    def __init__(self, root_dir: str, roll_secs: int = 60, roll_max_rows: int = 1000):
        self.root = root_dir
        self.roll_secs = int(roll_secs)
        self.roll_max_rows = int(roll_max_rows)
        os.makedirs(self.root, exist_ok=True)

        # 하위 디렉토리
        self.dir_tracks = os.path.join(self.root, "tracks")
        os.makedirs(self.dir_tracks, exist_ok=True)

        self.manifest_path = os.path.join(self.root, "manifest.csv")
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["filename","kind","t_start","t_end","rows","md5"])

        # 작업 큐/스레드
        self.q = queue.Queue(maxsize=10000)
        self._running = False
        self._th = None

        # 현재 오픈 파일 상태(kind별 별도 유지)
        self._state = {
            "fused":  {"fp": None, "path": None, "rows": 0, "t_start": None, "last_ts": None},
            "tracks": {"fp": None, "path": None, "rows": 0, "t_start": None, "last_ts": None},
        }
        self._lock = threading.Lock()

        # meta.json은 외부에서 set_meta로 세팅
        self._meta_written = False
        self._meta_obj = None

    def set_meta(self, obj: dict):
        """서버 시작 시 한 번 호출해 meta.json 작성"""
        self._meta_obj = obj
        meta_path = os.path.join(self.root, "meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            self._meta_written = True
        except Exception as e:
            print(f"[Saver] meta write error: {e}")

    def start(self):
        if self._running: return
        self._running = True
        self._th = threading.Thread(target=self._worker, daemon=True)
        self._th.start()

    def stop(self, timeout=1.0):
        self._running = False
        if self._th:
            self._th.join(timeout=timeout)
        # 열려있는 파일들 마감
        with self._lock:
            for kind in ["fused","tracks"]:
                self._close_and_manifest(kind)

    def enqueue(self, kind: str, row: dict):
        """row: JSON 직렬화 가능한 dict (프레임 단위 레코드)"""
        try:
            self.q.put_nowait((kind, row))
        except queue.Full:
            # 가장 오래된 것 drop
            try:
                self.q.get_nowait()
                self.q.put_nowait((kind, row))
            except Exception:
                pass

    # ---------- 내부 ----------
    def _worker(self):
        while self._running or not self.q.empty():
            try:
                kind, row = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._write_row(kind, row)
            except Exception as e:
                print(f"[Saver] write error: {e}")

    def _ensure_open(self, kind: str, ts: float):
        st = self._state[kind]
        dir_ = self.dir_fused if kind == "fused" else self.dir_tracks
        now = datetime.fromtimestamp(ts, tz=timezone.utc)
        # 새 파일 이름: kind_YYYY-mm-ddTHH-MM.jsonl.gz (UTC, 분 단위)
        fname = f"{kind}_{now.strftime('%Y-%m-%dT%H-%M')}.jsonl.gz"
        fpath = os.path.join(dir_, fname)

        if st["fp"] is None:
            fp = gzip.open(fpath, "at", encoding="utf-8")
            st.update({"fp": fp, "path": fpath, "rows": 0, "t_start": ts, "last_ts": ts})
            return

        # 롤링 조건: 시간/행수
        hit_time = (ts - (st["t_start"] or ts)) >= self.roll_secs
        hit_rows = st["rows"] >= self.roll_max_rows
        # 또한 “분”이 바뀌었는데 같은 파일명으로 쓰고 있으면 롤링
        cur_min_name = os.path.basename(fpath)
        opened_min_name = os.path.basename(st["path"]) if st["path"] else ""
        hit_minute_changed = (cur_min_name != opened_min_name)

        if hit_time or hit_rows or hit_minute_changed:
            self._close_and_manifest(kind)
            fp = gzip.open(fpath, "at", encoding="utf-8")
            st.update({"fp": fp, "path": fpath, "rows": 0, "t_start": ts, "last_ts": ts})

    def _close_and_manifest(self, kind: str):
        st = self._state[kind]
        fp = st["fp"]
        if fp is None:
            return
        try:
            fp.close()
        except Exception:
            pass
        # manifest 기록
        try:
            md5 = _md5sum(st["path"])
            with open(self.manifest_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                t0 = st["t_start"] or 0.0
                t1 = st["last_ts"] or t0
                w.writerow([os.path.relpath(st["path"], self.root), kind, f"{t0:.6f}", f"{t1:.6f}", st["rows"], md5])
        except Exception as e:
            print(f"[Saver] manifest write error: {e}")
        # 상태 초기화
        st.update({"fp": None, "path": None, "rows": 0, "t_start": None, "last_ts": None})

    def _write_row(self, kind: str, row: dict):
        ts = float(row.get("ts", time.time()))
        with self._lock:
            self._ensure_open(kind, ts)
            st = self._state[kind]
            # 한 줄 JSON
            st["fp"].write(json.dumps(row, ensure_ascii=False) + "\n")
            st["rows"] += 1
            st["last_ts"] = ts
# --- 끝 저장용 ---

# -----------------------------
# 수신부: 카메라별 UDP 포트를 바인딩해서 메시지를 받아 파싱
# -----------------------------
class UDPReceiver:
    """
    - cam_port_map: {"cam1":50050, "cam2":50051, ...}
    - 포맷1(JSON): {"type":"bev_labels","camera_id":1,"timestamp":..., "items":[{"center":[x,y],"length":L,"width":W,"yaw":deg}, ...]}
    - 포맷2(TEXT): 첫줄 메타, 이후 각 줄 "cls cx cy L W yaw"
    """
    def __init__(self, cam_port_map: Dict[str, int], host: str = "0.0.0.0", max_bytes: int = 65507):
        self.cam_port_map = cam_port_map
        self.host = host
        self.max_bytes = max_bytes
        self.socks: Dict[str, socket.socket] = {}
        self.ths: List[threading.Thread] = []
        self.running = False

        # 출력 큐: 각 카메라별 최신 프레임 버킷을 큐에 넣음
        self.q = queue.Queue(maxsize=2048)

    def start(self):
        if self.running:
            return
        self.running = True
        for cam, port in self.cam_port_map.items():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((self.host, int(port)))
            self.socks[cam] = s
            th = threading.Thread(target=self._rx_loop, args=(cam, s), daemon=True)
            th.start()
            self.ths.append(th)
        print(f"[UDPReceiver] listening on: " + ", ".join([f"{c}:{p}" for c, p in self.cam_port_map.items()]))

    def stop(self):
        self.running = False
        for s in self.socks.values():
            try: s.close()
            except: pass
        for th in self.ths:
            th.join(timeout=0.5)
        self.socks.clear()
        self.ths.clear()

    def _rx_loop(self, cam: str, sock: socket.socket):
        while self.running:
            try:
                data, addr = sock.recvfrom(self.max_bytes)
                ts = time.time()
                dets = self._parse_payload(data)
                if dets is None:
                    continue
                # dets: List[[cls,cx,cy,L,W,yaw_deg]]
                self.q.put_nowait({"cam": cam, "ts": ts, "dets": dets})
            except Exception as e:
                # 소켓 close 중이거나 과도한 부하 시 무시
                continue

    def _parse_payload(self, data: bytes):
        # JSON 먼저 시도
        try:
            msg = json.loads(data.decode("utf-8"))
            if isinstance(msg, dict) and msg.get("type") == "bev_labels":
                dets = []
                for it in msg.get("items", []):
                    cx, cy = it["center"]
                    L, W, yaw = it["length"], it["width"], it["yaw"]
                    dets.append([0, float(cx), float(cy), float(L), float(W), float(yaw)])
                return dets
        except Exception:
            pass

        # TEXT 라인 파싱 (header 무시)
        try:
            text = data.decode("utf-8", errors="ignore").strip()
            lines = [ln for ln in text.splitlines() if ln and not ln.lstrip().startswith("#")]
            dets = []
            for ln in lines:
                toks = ln.split()
                if len(toks) == 6:
                    c, cx, cy, L, W, yaw = toks
                elif len(toks) == 5:
                    # class 생략 포맷은 0으로 간주
                    cx, cy, L, W, yaw = toks
                    c = 0
                else:
                    continue
                dets.append([int(c), float(cx), float(cy), float(L), float(W), float(yaw)])
            return dets if dets else None
        except Exception:
            return None
# ㄴ대신 단일포트로 N개 수신
class UDPReceiverSingle:
    """
    하나의 UDP 포트에서 모든 카메라 패킷을 수신.
    payload(JSON)에 포함된 camera_id를 사용해 cam 이름을 f"cam{camera_id}"로 태깅.
    """
    def __init__(self, port: int, host: str = "0.0.0.0", max_bytes: int = 65507):
        self.host = host
        self.port = int(port)
        self.max_bytes = max_bytes
        self.sock = None
        self.th = None
        self.running = False
        self.q = queue.Queue(maxsize=4096)

    def start(self):
        if self.running: return
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        self.th = threading.Thread(target=self._rx_loop, daemon=True)
        self.th.start()
        print(f"[UDPReceiverSingle] listening on {self.host}:{self.port}")

    def stop(self):
        self.running = False
        if self.sock:
            try: self.sock.close()
            except: pass
        if self.th:
            self.th.join(timeout=0.5)

    def _rx_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(self.max_bytes)
                ts = time.time()
                cam, dets = self._parse_payload(data)
                if dets is None: 
                    continue
                self.q.put_nowait({"cam": cam, "ts": ts, "dets": dets})
            except Exception:
                continue

    def _parse_payload(self, data: bytes):
        # JSON 우선
        try:
            msg = json.loads(data.decode("utf-8"))
            if isinstance(msg, dict) and msg.get("type") == "bev_labels":
                cam_id = int(msg.get("camera_id", 0) or 0)
                cam = f"cam{cam_id}" if cam_id else "cam?"
                dets = []
                for it in msg.get("items", []):
                    cx, cy = it["center"]
                    L, W, yaw = it["length"], it["width"], it["yaw"]
                    dets.append([0, float(cx), float(cy), float(L), float(W), float(yaw)])
                return cam, dets if dets else []
        except Exception:
            pass

        # TEXT 형식 (첫 줄 header 무시)
        try:
            text = data.decode("utf-8", errors="ignore").strip()
            lines = [ln for ln in text.splitlines() if ln and not ln.lstrip().startswith("#")]
            dets = []
            for ln in lines:
                toks = ln.split()
                if len(toks) == 6:
                    c, cx, cy, L, W, yaw = toks
                elif len(toks) == 5:
                    cx, cy, L, W, yaw = toks; c = 0
                else:
                    continue
                dets.append([int(c), float(cx), float(cy), float(L), float(W), float(yaw)])
            # 텍스트만으로는 cam 판별 불가 → cam=?, 필요하면 헤더에 cam 넣어 보내도록 확장
            return "cam?", dets if dets else []
        except Exception:
            return "cam?", None
# ---근데 실시간에 쓸수잇을진 몰겟다 

# -----------------------------
# 파이프라인: 수집 → 통합/융합 → 추적 → 시각화
# -----------------------------
class RealtimeFusionServer:
    def __init__(
        self,
        cam_ports: Dict[str, int],
        xlim: Tuple[float, float] = (-120.0, -30.0),
        ylim: Tuple[float, float] = (-80.0, 40.0),
        fps: float = 10.0,
        iou_cluster_thr: float = 0.25,
        bg_path: str = None,
        show_windows: bool = True,
        single_port: int = 50050,
        save_dir: str = None, roll_secs: int = 60, roll_max_rows: int = 1000 # 저장용
    ):
        self.receiver = UDPReceiver(cam_ports)
        self.fps = fps
        self.dt = 1.0 / max(1e-3, fps)
        self.xlim = xlim
        self.ylim = ylim
        self.iou_thr = iou_cluster_thr
        self.bg_path = bg_path
        self.show = show_windows
        self.save_dir = save_dir # 저장용
        self.saver = None

        # 단일 소켓 리시버
        self.receiver = UDPReceiverSingle(single_port)

        # 카메라 위치(가중치/거리 계산에 사용)
        self.cam_xy = {item["name"]: (float(item["pos"]["x"]), float(item["pos"]["y"])) for item in CAMERA_SETUPS}  # :contentReference[oaicite:10]{index=10}
         # ✅ 카메라 동적 관리: buffer / 순서 / 색상
        self.buffer = {}                        # cam_name -> deque(maxlen=1)
        self.cam_order = []                     # 보이는 순서 유지용
        self._palette = distinct_colors(128)    # 충분한 색
        self.color_map = {}    

        if self.save_dir:
            self.saver = RollingJsonlSaver(self.save_dir, roll_secs=roll_secs, roll_max_rows=roll_max_rows)
            # meta.json 작성
            meta = {
                "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "xlim": self.xlim, "ylim": self.ylim, "fps": self.fps, "iou_thr": self.iou_thr,
                # "cam_ports": self.receiver.cam_port_map,
                "udp_port": single_port,    
                "cam_positions": self.cam_xy,   # CAMERA_SETUPS에서 읽은 글로벌 위치
                "version": "realtime_bev_udp_server.v1"
            }
            self.saver.set_meta(meta)
            self.saver.start() # 여기까지 저장용

        
        # 프레임 버퍼(최근 T초 동안 카메라별 최신)
        self.buffer: Dict[str, deque] = {cam: deque(maxlen=1) for cam in cam_ports.keys()}

        # 추적기
        self.tracker = SortTracker(max_age=10, min_hits=3, iou_threshold=0.15)  # :contentReference[oaicite:11]{index=11}

        # matplotlib 창 준비
        if self.show:
            plt.ion()
            self.fig_raw, self.ax_raw = self._make_axes("① 통합(카메라별)")
            self.fig_fused, self.ax_fused = self._make_axes("② 융합(중복 제거)")
            self.fig_track, self.ax_track = self._make_axes("③ 추적 결과(ID)")
            self.colors = distinct_colors(len(cam_ports))  # :contentReference[oaicite:12]{index=12}
            self.cam_names = list(cam_ports.keys())
    
    def _register_cam_if_needed(self, cam_name: str): # 수신 시 신규카메라 등록 
        if cam_name not in self.buffer:
            self.buffer[cam_name] = deque(maxlen=1)
            self.cam_order.append(cam_name)
            # 색 지정
            self.color_map[cam_name] = self._palette[(len(self.cam_order)-1) % len(self._palette)]

    # axes 생성 (draw_global 스타일 축/그리드) :contentReference[oaicite:13]{index=13}
    def _make_axes(self, title: str):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        # 배경 이미지(선택)
        if self.bg_path and Path(self.bg_path).exists():
            try:
                bg_img = plt.imread(self.bg_path)
                y_min, y_max = min(self.ylim[0], self.ylim[1]), max(self.ylim[0], self.ylim[1])
                extent = (self.xlim[0], self.xlim[1], y_min, y_max)
                ax.imshow(bg_img, extent=extent, origin="lower", alpha=0.6, zorder=0)
            except Exception:
                pass
        return fig, ax

    def start(self):
        self.receiver.start()
        self._main_loop()

    def _main_loop(self):
        last = time.time()
        while True:
            # 수신 큐에서 가능한 만큼 비움 → 최신으로 buffer 업데이터
            try:
                while True:
                    item = self.receiver.q.get_nowait()
                    cam = item["cam"]; ts = item["ts"]; dets = item["dets"]
                    # ✅ 신규 카메라면 등록
                    self._register_cam_if_needed(cam)
                    self.buffer[cam].clear()
                    self.buffer[cam].append({"ts": ts, "dets": dets})
            except queue.Empty:
                pass

            now = time.time()
            if now - last < self.dt:
                time.sleep(0.005)
                continue
            last = now

            # ---- ① 통합(그냥 카메라별 결과를 한 축에 올림) ----
            raw_boxes, raw_cams = self._gather_current()
            if self.show:
                self._draw_raw(raw_boxes, raw_cams)

            # ---- ② 융합(중복 제거/객체 병합) ----
            fused = self._fuse_boxes(raw_boxes, raw_cams)
            if self.show:
                self._draw_fused(fused)

            # ---- ③ 추적(SORT) ----
            # tracker는 [class, x_c, y_c, l, w, angle] Nx6 입력을 받게 맞춤 :contentReference[oaicite:14]{index=14}
            dets_for_tracker = np.array([[0, *b] for b in fused], dtype=float) if len(fused) else np.zeros((0,6), dtype=float)
            tracks = self.tracker.update(dets_for_tracker)  # shape: [N, 8] = [track_id, class, x, y, l, w, yaw]
            if self.show:
                self._draw_tracks(tracks)

            # ==== 저장(롤링) ====
            if self.saver:
                # ③ tracks 저장
                tracks_list = []
                if tracks is not None and len(tracks):
                    for row in tracks:
                        tid = int(row[0]); cls = int(row[1])
                        cx, cy, L, W, yaw = map(float, row[2:7])
                        tracks_list.append({"id": tid, "cls": cls, "cx": cx, "cy": cy, "L": L, "W": W, "yaw": yaw})
                tracks_record = {"ts": now, "tracks": tracks_list}
                self.saver.enqueue("tracks", tracks_record)

            # GUI 갱신
            if self.show:
                plt.pause(0.001)

    def _gather_current(self):
        raw_boxes = []
        raw_cams = []
        for cam in self.buffer.keys():
            if not self.buffer[cam]:
                continue
            dets = self.buffer[cam][-1]["dets"]
            for c, cx, cy, L, W, yaw in dets:
                raw_boxes.append([float(cx), float(cy), float(L), float(W), float(yaw)])
                raw_cams.append(cam)
        return raw_boxes, raw_cams


    # merge_dist_wbf 의 클러스터링/가중통합 로직을 그대로 사용 :contentReference[oaicite:15]{index=15}
    def _fuse_boxes(self, raw_boxes: List[List[float]], raw_cams: List[str]) -> List[List[float]]:
        if not raw_boxes:
            return []
        boxes = np.array(raw_boxes, dtype=float)
        cams  = list(raw_cams)
        clusters = cluster_by_aabb_iou(boxes, iou_cluster_thr=self.iou_thr)
        fused_list = []
        for idxs in clusters:
            rep = fuse_cluster_weighted(
                boxes, cams, idxs, self.cam_xy,
                d0=5.0, p=2.0
            )
            fused_list.append(rep.tolist())  # [cx,cy,L,W,yaw]
        return fused_list

    # -------- draw helpers --------
    def _draw_raw(self, raw_boxes: List[List[float]], raw_cams: List[str]):
        ax = self.ax_raw
        ax.cla()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*self.xlim); ax.set_ylim(*self.ylim)
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title("① 통합(카메라별)")

        # 배경
        if self.bg_path and Path(self.bg_path).exists():
            try:
                bg_img = plt.imread(self.bg_path)
                y_min, y_max = min(self.ylim[0], self.ylim[1]), max(self.ylim[0], self.ylim[1])
                extent = (self.xlim[0], self.xlim[1], y_min, y_max)
                ax.imshow(bg_img, extent=extent, origin="lower", alpha=0.6, zorder=0)
            except Exception:
                pass

        # # 카메라별 색상
        for b, cam in zip(raw_boxes, raw_cams):
            cx, cy, L, W, yaw = b
            q = obb_to_quad(cx, cy, L, W, math.radians(yaw))
            poly = np.vstack([q, q[0]])
            color = self.color_map.get(cam, (0,0,1,1))
            ax.plot(poly[:, 0], poly[:, 1], color=color, lw=1.8, alpha=0.95)

        # 범례 와근데 이거 4개밖에 안뜨던데ㅋㅎ
        for cam in self.cam_order:
            ax.plot([], [], color=self.color_map.get(cam, (0,0,1,1)), lw=2, label=cam)
        ax.legend(loc="upper right", fontsize=9, ncol=2)

    def _draw_fused(self, fused_boxes: List[List[float]]):
        ax = self.ax_fused
        ax.cla()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*self.xlim); ax.set_ylim(*self.ylim)
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title("② 융합(중복 제거)")

        if self.bg_path and Path(self.bg_path).exists():
            try:
                bg_img = plt.imread(self.bg_path)
                y_min, y_max = min(self.ylim[0], self.ylim[1]), max(self.ylim[0], self.ylim[1])
                extent = (self.xlim[0], self.xlim[1], y_min, y_max)
                ax.imshow(bg_img, extent=extent, origin="lower", alpha=0.6, zorder=0)
            except Exception:
                pass

        for b in fused_boxes:
            cx, cy, L, W, yaw = b
            q = obb_to_quad(cx, cy, L, W, math.radians(yaw))  # :contentReference[oaicite:17]{index=17}
            poly = np.vstack([q, q[0]])
            ax.plot(poly[:, 0], poly[:, 1], color="blue", lw=2.0, alpha=0.95)

    def _draw_tracks(self, tracks: np.ndarray):
        ax = self.ax_track
        ax.cla()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*self.xlim); ax.set_ylim(*self.ylim)
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title("③ 추적 결과(ID)")

        if self.bg_path and Path(self.bg_path).exists():
            try:
                bg_img = plt.imread(self.bg_path)
                y_min, y_max = min(self.ylim[0], self.ylim[1]), max(self.ylim[0], self.ylim[1])
                extent = (self.xlim[0], self.xlim[1], y_min, y_max)
                ax.imshow(bg_img, extent=extent, origin="lower", alpha=0.6, zorder=0)
            except Exception:
                pass

        # tracks: [track_id, class, x, y, l, w, yaw]
        if tracks is None or not len(tracks):
            return
        for row in tracks:
            tid = int(row[1]) 
            # 여기선 update() 반환이 [track_id, class, x, y, l, w, angle]
            tid = int(row[0]); cls = int(row[1])
            cx, cy, L, W, yaw = map(float, row[2:7])

            # draw.py와 동일 스타일 색상/표시 사용 
            color = color_for_track(tid)
            q = obb_to_quad_draw(cx, cy, L, W, yaw)  # yaw는 deg, draw.py의 obb_to_quad는 deg 인풋
            poly = np.vstack([q, q[0]])
            ax.plot(poly[:, 0], poly[:, 1], color=color, lw=2.0, alpha=0.95, zorder=3)
            ax.text(cx, cy, str(tid),
                    fontsize=9, ha="center", va="center",
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
                    zorder=4)

# -----------------------------
# 실행부
# -----------------------------
def parse_cam_ports(text: str) -> Dict[str, int]:
    """
    예: "cam1:50050,cam2:50051"
    """
    out = {}
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, port = tok.split(":")
        out[name.strip()] = int(port)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam-ports", default="cam1:50050,cam2:50051",
                    help="카메라명:UDP포트 목록 (쉼표로 구분)")
    ap.add_argument("--xlim", default="0,20", help="글로벌 X축 표시 범위 (예: -120,-30)")
    ap.add_argument("--ylim", default="0,20", help="글로벌 Y축 표시 범위 (예: -80,40)")
    ap.add_argument("--fps", type=float, default=10.0, help="갱신 FPS")
    ap.add_argument("--iou-thr", type=float, default=0.25, help="AABB IoU 군집 임계값(융합 단계)")
    ap.add_argument("--bg", default=None, help="배경 이미지 경로(선택)")
    ap.add_argument("--no-gui", action="store_true", help="시각화 창 없이 서버만 구동")
    ap.add_argument("--save-dir", default=None, help="롤링 저장 루트 디렉토리 (예: ./logs)")
    ap.add_argument("--roll-secs", type=int, default=60, help="롤링 간격(초) 기본 60s")
    ap.add_argument("--roll-max-rows", type=int, default=1000, help="파일당 최대 행수 기본 1000")
    ap.add_argument("--udp-port", type=int, default=50050, help="단일 UDP 수신 포트")

    args = ap.parse_args()

    x0, x1 = [float(v) for v in args.xlim.split(",")]
    y0, y1 = [float(v) for v in args.ylim.split(",")]
    cam_ports = parse_cam_ports(args.cam_ports)

    server = RealtimeFusionServer(
        cam_ports=cam_ports,
        xlim=(x0, x1),
        ylim=(y0, y1),
        fps=args.fps,
        iou_cluster_thr=args.iou_thr,
        bg_path=args.bg,
        show_windows=(not args.no_gui),
        save_dir=args.save_dir,
        roll_secs=args.roll_secs,
        roll_max_rows=args.roll_max_rows,
        single_port=args.udp_port,
    )
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.receiver.stop()
        if server.saver:
            server.saver.stop()
        if not args.no_gui:
            try:
                plt.ioff()
                plt.close("all")
            except Exception:
                pass

if __name__ == "__main__":
    main()