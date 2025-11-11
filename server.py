import argparse
import json
import queue
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from utils.merge.merge_dist_wbf import (
    CAMERA_SETUPS, cluster_by_aabb_iou, fuse_cluster_weighted
)
from utils.sort.tracker import SortTracker
from realtime_show_result.viz_utils import VizSizeConfig, prepare_visual_item

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

class TrackBroadcaster:
    """
    UDP/TCP 브로드캐스터.
    track 패킷 포맷:
    {
        "type": "global_tracks",
        "timestamp": unix_ts,
        "items": [
            {"id": tid, "class": cls, "center": [cx, cy, cz], "length": L,
             "width": W, "yaw": yaw_deg, "pitch": pitch, "roll": roll,
             "score": score, "sources": ["cam1", ...]}
        ]
    }
    """
    def __init__(self, host: str, port: int, protocol: str = "udp"):
        self.host = host
        self.port = int(port)
        self.protocol = protocol.lower()
        if self.protocol not in ("udp", "tcp"):
            raise ValueError("protocol must be 'udp' or 'tcp'")
        if self.protocol == "udp":
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.addr = (self.host, self.port)
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.addr = None

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def send(self, tracks: np.ndarray, extras: Dict[int, dict], ts: float):
        payload = {
            "type": "global_tracks",
            "timestamp": ts,
            "items": []
        }
        if tracks is not None and len(tracks):
            for row in tracks:
                tid = int(row[0])
                cls = int(row[1])
                cx, cy, L, W, yaw = map(float, row[2:7])
                extra = extras.get(tid, {})
                payload["items"].append({
                    "id": tid,
                    "class": cls,
                    "center": [cx, cy, float(extra.get("cz", 0.0))],
                    "length": L,
                    "width": W,
                    "yaw": yaw,
                    "pitch": float(extra.get("pitch", 0.0)),
                    "roll": float(extra.get("roll", 0.0)),
                    "score": float(extra.get("score", 0.0)),
                    "sources": list(extra.get("source_cams", [])),
                })
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            if self.protocol == "udp":
                self.sock.sendto(data, self.addr)
            else:
                self.sock.sendall(data + b"\n")
        except Exception as e:
            print(f"[TrackBroadcaster] send error: {e}")

class GlobalWebServer:
    """
    FastAPI 기반 웹 서버.
    - /assets/global.ply, /assets/vehicle.glb 제공
    - /api/raw, /api/fused, /api/tracks 로 최신 결과 반환
    - /fusion/raw, /fusion/fused, /fusion/tracks 페이지에서 공용 뷰어(main.js) 사용
    """
    def __init__(
        self,
        *,
        global_ply: str,
        vehicle_glb: str,
        host: str = "0.0.0.0",
        port: int = 18090,
        static_root: Optional[Path] = None,
        viz_config: VizSizeConfig = VizSizeConfig(),
        client_config: Optional[dict] = None,
        tracker_fixed_length: Optional[float] = None,
        tracker_fixed_width: Optional[float] = None,
    ):
        self.global_ply = str(Path(global_ply).resolve())
        self.vehicle_glb = str(Path(vehicle_glb).resolve())
        self.host = host
        self.port = int(port)
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._state = {
            "raw": [],
            "fused": [],
            "tracks": [],
            "timestamp": None,
            "cameras": [],
        }
        if static_root is None:
            static_root = Path(__file__).resolve().parent / "realtime_show_result" / "static"
        self.static_root = static_root
        self.page_map = {
            "raw": self.static_root / "fusion_raw.html",
            "fused": self.static_root / "fusion_fused.html",
            "tracks": self.static_root / "fusion_tracks.html",
        }
        self.viz_cfg = viz_config
        self.client_config = dict(client_config or {})
        self.client_config.setdefault("normalizeVehicle", True)
        self.client_config.setdefault("vehicleYAxisUp", True)
        self.client_config.setdefault("flipPlyY", False)
        self.client_config.setdefault("showSceneAxes", False)
        self.client_config.setdefault("showDebugMarker", False)
        self.client_config.setdefault("mode", "fusion")
        self.client_config["vizConfig"] = self.viz_cfg.as_client_dict()

        self.app = FastAPI(title="AIRKON Fusion Server", version="0.1.0")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        if self.static_root.exists():
            self.app.mount("/static", StaticFiles(directory=str(self.static_root), html=True), name="static")
        self._register_routes()

        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info", access_log=False)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        print(f"[Web] serving http://{self.host}:{self.port}")

    # ----------------- FastAPI Routes -----------------
    def _register_routes(self):
        @self.app.get("/")
        def _root():
            page = self.page_map.get("raw")
            if page and page.exists():
                return FileResponse(str(page))
            return {"status": "ok", "message": "fusion_raw.html missing"}

        @self.app.get("/fusion/raw")
        def _view_raw():
            page = self.page_map.get("raw")
            if page and page.exists():
                return FileResponse(str(page))
            raise HTTPException(status_code=404, detail="fusion_raw.html missing")

        @self.app.get("/fusion/fused")
        def _view_fused():
            page = self.page_map.get("fused")
            if page and page.exists():
                return FileResponse(str(page))
            raise HTTPException(status_code=404, detail="fusion_fused.html missing")

        @self.app.get("/fusion/tracks")
        def _view_tracks():
            page = self.page_map.get("tracks")
            if page and page.exists():
                return FileResponse(str(page))
            raise HTTPException(status_code=404, detail="fusion_tracks.html missing")

        @self.app.get("/healthz")
        def _health():
            return {"ok": True, "since": self.started_at}

        @self.app.get("/api/site")
        def _site():
            with self._lock:
                return {
                    "started_at": self.started_at,
                    "timestamp": self._state["timestamp"],
                    "cameras": self._state["cameras"],
                    "config": self.client_config,
                }

        @self.app.get("/api/config")
        def _config():
            return self.client_config.copy()

        @self.app.get("/api/raw")
        def _raw():
            with self._lock:
                return {"timestamp": self._state["timestamp"], "items": self._state["raw"]}

        @self.app.get("/api/fused")
        def _fused():
            with self._lock:
                return {"timestamp": self._state["timestamp"], "items": self._state["fused"]}

        @self.app.get("/api/tracks")
        def _tracks():
            with self._lock:
                return {"timestamp": self._state["timestamp"], "items": self._state["tracks"]}

        @self.app.get("/assets/global.ply")
        def _global_ply():
            if not os.path.exists(self.global_ply):
                raise HTTPException(status_code=404, detail="global ply missing")
            return FileResponse(self.global_ply, filename="global.ply")

        @self.app.get("/assets/vehicle.glb")
        def _vehicle_glb():
            if not os.path.exists(self.vehicle_glb):
                raise HTTPException(status_code=404, detail="vehicle glb missing")
            return FileResponse(self.vehicle_glb, filename="vehicle.glb")

    # ----------------- Update Interface -----------------
    def update(self, *, raw, fused, tracks, timestamp: float, cameras: List[str]):
        with self._lock:
            self._state["raw"] = raw
            self._state["fused"] = fused
            self._state["tracks"] = tracks
            self._state["timestamp"] = timestamp
            self._state["cameras"] = sorted(set(cameras))

    def stop(self):
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=2.0)
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
                    dets.append({
                        "cls": int(it.get("class", 0)),
                        "cx": float(cx),
                        "cy": float(cy),
                        "length": float(it.get("length", 0.0)),
                        "width": float(it.get("width", 0.0)),
                        "yaw": float(it.get("yaw", 0.0)),
                        "score": float(it.get("score", 0.0)),
                        "cz": float(it.get("cz", 0.0)),
                        "pitch": float(it.get("pitch", 0.0)),
                        "roll": float(it.get("roll", 0.0)),
                    })
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
                if len(toks) >= 6:
                    c, cx, cy, L, W, yaw = toks[:6]
                    rest = toks[6:]
                else:
                    continue
                cz = float(rest[0]) if len(rest) >= 1 else 0.0
                pitch = float(rest[1]) if len(rest) >= 2 else 0.0
                roll = float(rest[2]) if len(rest) >= 3 else 0.0
                score = float(rest[3]) if len(rest) >= 4 else 0.0
                dets.append({
                    "cls": int(c),
                    "cx": float(cx),
                    "cy": float(cy),
                    "length": float(L),
                    "width": float(W),
                    "yaw": float(yaw),
                    "cz": cz,
                    "pitch": pitch,
                    "roll": roll,
                    "score": score,
                })
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

    def _log_packet(self, cam: str, dets: List[dict], meta: Optional[dict] = None):
        """
        수신 데이터 확인용 로그. meta는 JSON payload 전체(dict)일 수도 있고 None일 수도 있다.
        """
        try:
            cnt = len(dets) if dets else 0
            ts = meta.get("timestamp") if isinstance(meta, dict) else None
            capture_ts = meta.get("capture_ts") if isinstance(meta, dict) else None
            camera_id = meta.get("camera_id") if isinstance(meta, dict) else None
            print(
                "[UDPReceiverSingle] recv",
                f"cam={cam}",
                f"camera_id={camera_id}",
                f"timestamp={ts}",
                f"capture_ts={capture_ts}",
                f"detections={cnt}",
            )
            if dets:
                # 한 건만 찍어도 값 흐름을 확인할 수 있으므로 첫 항목만 노출
                sample = json.dumps(dets[0], ensure_ascii=False)
                print(f"  sample_det={sample}")
        except Exception as exc:
            print(f"[UDPReceiverSingle] log error: {exc}")

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
                    dets.append({
                        "cls": int(it.get("class", 0)),
                        "cx": float(cx),
                        "cy": float(cy),
                        "length": float(it.get("length", 0.0)),
                        "width": float(it.get("width", 0.0)),
                        "yaw": float(it.get("yaw", 0.0)),
                        "score": float(it.get("score", 0.0)),
                        "cz": float(it.get("cz", 0.0)),
                        "pitch": float(it.get("pitch", 0.0)),
                        "roll": float(it.get("roll", 0.0)),
                    })
                self._log_packet(cam, dets if dets else [], meta=msg)
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
                if len(toks) >= 6:
                    c, cx, cy, L, W, yaw = toks[:6]
                    rest = toks[6:]
                else:
                    continue
                cz = float(rest[0]) if len(rest) >= 1 else 0.0
                pitch = float(rest[1]) if len(rest) >= 2 else 0.0
                roll = float(rest[2]) if len(rest) >= 3 else 0.0
                score = float(rest[3]) if len(rest) >= 4 else 0.0
                dets.append({
                    "cls": int(c),
                    "cx": float(cx),
                    "cy": float(cy),
                    "length": float(L),
                    "width": float(W),
                    "yaw": float(yaw),
                    "cz": cz,
                    "pitch": pitch,
                    "roll": roll,
                    "score": score,
                })
            # 텍스트만으로는 cam 판별 불가 → cam=?, 필요하면 헤더에 cam 넣어 보내도록 확장
            self._log_packet("cam?", dets if dets else [])
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
        fps: float = 10.0,
        iou_cluster_thr: float = 0.25,
        single_port: int = 50050,
        save_dir: str = None, roll_secs: int = 60, roll_max_rows: int = 1000, # 저장용
        tx_host: Optional[str] = None, tx_port: int = 60050, tx_protocol: str = "udp",
        carla_host: Optional[str] = None, carla_port: int = 61000,
        global_ply: str = "pointcloud/global_fused_small.ply",
        vehicle_glb: str = "pointcloud/car.glb",
        web_host: str = "0.0.0.0",
        web_port: int = 18090,
        enable_web: bool = True,
        viz_config: VizSizeConfig = VizSizeConfig(),
        client_config: Optional[dict] = None,
        tracker_fixed_length: Optional[float] = None,
        tracker_fixed_width: Optional[float] = None,
    ):
        self.fps = fps
        self.dt = 1.0 / max(1e-3, fps)
        self.iou_thr = iou_cluster_thr
        self.save_dir = save_dir # 저장용
        self.saver = None
        self.track_meta: Dict[int, dict] = {}
        self.active_cams = set()

        # 단일 소켓 리시버 (엣지→서버 UDP)
        self.receiver = UDPReceiverSingle(single_port)

        # 카메라 위치(가중치/거리 계산에 사용)
        self.cam_xy = {item["name"]: (float(item["pos"]["x"]), float(item["pos"]["y"])) for item in CAMERA_SETUPS}  # :contentReference[oaicite:10]{index=10}
         # ✅ 카메라 동적 관리: buffer / 순서 / 색상
        self.buffer = {}                        # cam_name -> deque(maxlen=1)

        self.track_tx = TrackBroadcaster(tx_host, tx_port, tx_protocol) if tx_host else None
        self.carla_tx = TrackBroadcaster(carla_host, carla_port) if carla_host else None
        self.viz_cfg = viz_config
        self.client_config = client_config or {}
        self.tracker_fixed_length = float(tracker_fixed_length) if tracker_fixed_length is not None else None
        self.tracker_fixed_width = float(tracker_fixed_width) if tracker_fixed_width is not None else None
        self.web = GlobalWebServer(
            global_ply=global_ply,
            vehicle_glb=vehicle_glb,
            host=web_host,
            port=web_port,
            viz_config=self.viz_cfg,
            client_config=self.client_config,
        ) if enable_web else None

        if self.save_dir:
            self.saver = RollingJsonlSaver(self.save_dir, roll_secs=roll_secs, roll_max_rows=roll_max_rows)
            # meta.json 작성
            meta = {
                "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "fps": self.fps,
                "iou_thr": self.iou_thr,
                "udp_port": single_port,
                "cam_positions": self.cam_xy,
                "version": "realtime_bev_udp_server.v2"
            }
            self.saver.set_meta(meta)
            self.saver.start() # 여기까지 저장용

        
        # 프레임 버퍼(최근 T초 동안 카메라별 최신)
        self.buffer: Dict[str, deque] = {cam: deque(maxlen=1) for cam in cam_ports.keys()}

        # 추적기
        self.tracker = SortTracker(max_age=10, min_hits=3, iou_threshold=0.15)  # :contentReference[oaicite:11]{index=11}
        self._log_interval = 1.0
        self._next_log_ts = 0.0

    def _register_cam_if_needed(self, cam_name: str): # 수신 시 신규카메라 등록 
        if cam_name not in self.buffer:
            self.buffer[cam_name] = deque(maxlen=1)
        self.active_cams.add(cam_name)

    def _should_log(self) -> bool:
        now = time.time()
        if now >= self._next_log_ts:
            self._next_log_ts = now + self._log_interval
            return True
        return False

    def _log_pipeline(self, raw_stats: Dict[str, int], fused: List[dict], tracks: List[dict], timestamp: float,
                      timings: Optional[Dict[str, float]] = None):
        total_raw = sum(raw_stats.values())
        fused_count = len(fused)
        track_count = len(tracks)
        cams_str = ", ".join([f"{cam}:{cnt}" for cam, cnt in sorted(raw_stats.items())]) or "-"
        print(
            f"[Fusion] ts={timestamp:.3f} total_raw={total_raw} cams=({cams_str}) "
            f"clusters={fused_count} tracks={track_count}"
        )
        if fused_count:
            sample_keys = ["cx","cy","length","width","yaw","score","source_cams"]
            sample_fused = {k: fused[0][k] for k in sample_keys if k in fused[0]}
            print(f"  fused_sample={json.dumps(sample_fused, ensure_ascii=False)}")
        if track_count:
            sample_track = json.dumps(tracks[0], ensure_ascii=False)
            print(f"  track_sample={sample_track}")
        if timings:
            timing_str = " ".join(f"{k}={v:.2f}ms" for k, v in timings.items())
            print(f"  timings {timing_str}")

    def start(self):
        self.receiver.start()
        self._main_loop()

    def _main_loop(self):
        last = time.time()
        while True:
            timings: Dict[str, float] = {}
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

            # ---- ① 카메라별 최신 → ② 융합 ----
            t0 = time.perf_counter()
            raw_dets = self._gather_current()
            timings["gather"] = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            fused = self._fuse_boxes(raw_dets)
            timings["fuse"] = (time.perf_counter() - t1) * 1000.0

            # ---- ③ 추적(SORT) ----
            # tracker는 [class, x_c, y_c, l, w, angle] Nx6 입력을 받게 맞춤 :contentReference[oaicite:14]{index=14}
            dets_for_tracker = np.array(
                [[
                    0,
                    det["cx"],
                    det["cy"],
                    (self.tracker_fixed_length if self.tracker_fixed_length is not None else det["length"]),
                    (self.tracker_fixed_width if self.tracker_fixed_width is not None else det["width"]),
                    det["yaw"],
                ] for det in fused],
                dtype=float
            ) if fused else np.zeros((0,6), dtype=float)
            t2 = time.perf_counter()
            tracks = self.tracker.update(dets_for_tracker)  # shape: [N, 8] = [track_id, class, x, y, l, w, yaw]
            timings["track"] = (time.perf_counter() - t2) * 1000.0
            self._update_track_metadata(tracks, fused)
            self._broadcast_tracks(tracks, now)
            tracks_for_output = self._tracks_to_dicts(tracks)
            raw_payload = self._serialize_raw(raw_dets)
            fused_payload = self._serialize_fused(fused)
            if self.web:
                self.web.update(
                    raw=raw_payload,
                    fused=fused_payload,
                    tracks=tracks_for_output,
                    timestamp=now,
                    cameras=list(self.active_cams),
                )

            if self._should_log():
                stats = {}
                for det in raw_dets:
                    cam = det.get("cam", "?")
                    stats[cam] = stats.get(cam, 0) + 1
                self._log_pipeline(stats, fused_payload, tracks_for_output, now, timings)

            # ==== 저장(롤링) ====
            if self.saver:
                saver_rows = [
                    {"id": t["id"], "cls": t["class"], "cx": t["cx"], "cy": t["cy"], "L": t["length"], "W": t["width"], "yaw": t["yaw"]}
                    for t in tracks_for_output
                ]
                tracks_record = {"ts": now, "tracks": saver_rows}
                self.saver.enqueue("tracks", tracks_record)

    def _gather_current(self):
        detections = []
        for cam, dq in self.buffer.items():
            if not dq:
                continue
            entry = dq[-1]
            ts = entry.get("ts", time.time())
            for det in entry["dets"]:
                det_copy = det.copy()
                det_copy["cam"] = cam
                det_copy["ts"] = ts
                detections.append(det_copy)
        return detections


    # merge_dist_wbf 의 클러스터링/가중통합 로직을 그대로 사용 :contentReference[oaicite:15]{index=15}
    def _fuse_boxes(self, raw_detections: List[dict]) -> List[dict]:
        if not raw_detections:
            return []
        boxes = np.array([[d["cx"], d["cy"], d["length"], d["width"], d["yaw"]] for d in raw_detections], dtype=float)
        cams  = [d.get("cam", "?") for d in raw_detections]
        clusters = cluster_by_aabb_iou(boxes, iou_cluster_thr=self.iou_thr)
        fused_list = []
        for idxs in clusters:
            rep = fuse_cluster_weighted(
                boxes, cams, idxs, self.cam_xy,
                d0=5.0, p=2.0
            )
            extras = self._aggregate_cluster(raw_detections, idxs)
            fused_list.append({
                "cx": float(rep[0]),
                "cy": float(rep[1]),
                "length": float(rep[2]),
                "width": float(rep[3]),
                "yaw": float(rep[4]),
                **extras,
            })
        return fused_list

    def _aggregate_cluster(self, detections: List[dict], idxs: List[int]) -> dict:
        subset = [detections[i] for i in idxs]
        if not subset:
            return {"cz": 0.0, "pitch": 0.0, "roll": 0.0, "score": 0.0, "source_cams": []}
        score = np.mean([float(d.get("score", 0.0)) for d in subset])
        cz = np.mean([float(d.get("cz", 0.0)) for d in subset])
        pitch = np.mean([float(d.get("pitch", 0.0)) for d in subset])
        roll = np.mean([float(d.get("roll", 0.0)) for d in subset])
        cams = [d.get("cam", "?") for d in subset]
        return {
            "cz": float(cz),
            "pitch": float(pitch),
            "roll": float(roll),
            "score": float(score),
            "source_cams": cams,
        }


    def _update_track_metadata(self, tracks: np.ndarray, fused_boxes: List[dict]):
        active_ids = set()
        centers = None
        if fused_boxes:
            centers = np.array([[det["cx"], det["cy"]] for det in fused_boxes], dtype=float)
        track_rows = tracks if (tracks is not None and len(tracks)) else []
        for row in track_rows:
            tid = int(row[0])
            active_ids.add(tid)
            if centers is None or not len(centers):
                continue
            cx, cy = float(row[2]), float(row[3])
            idx = int(np.argmin(np.linalg.norm(centers - np.array([cx, cy]), axis=1)))
            det = fused_boxes[idx]
            self.track_meta[tid] = {
                "cz": float(det.get("cz", 0.0)),
                "pitch": float(det.get("pitch", 0.0)),
                "roll": float(det.get("roll", 0.0)),
                "score": float(det.get("score", 0.0)),
                "source_cams": det.get("source_cams", []),
            }
        # remove stale
        stale = [tid for tid in self.track_meta.keys() if tid not in active_ids]
        for tid in stale:
            self.track_meta.pop(tid, None)

    def _broadcast_tracks(self, tracks: np.ndarray, ts: float):
        if self.track_tx:
            self.track_tx.send(tracks, self.track_meta, ts)
        if self.carla_tx:
            self.carla_tx.send(tracks, self.track_meta, ts)

    def _serialize_raw(self, raw_detections: List[dict]) -> List[dict]:
        payload = []
        for det in raw_detections:
            vis = prepare_visual_item(
                class_id=int(det.get("cls", 0)),
                cx=float(det.get("cx", 0.0)),
                cy=float(det.get("cy", 0.0)),
                cz=float(det.get("cz", 0.0)),
                length=float(det.get("length", 0.0)),
                width=float(det.get("width", 0.0)),
                yaw_deg=float(det.get("yaw", 0.0)),
                pitch_deg=float(det.get("pitch", 0.0)),
                roll_deg=float(det.get("roll", 0.0)),
                cfg=self.viz_cfg,
                score=det.get("score", 0.0),
            )
            vis["cam"] = det.get("cam")
            vis["timestamp"] = float(det.get("ts", 0.0))
            vis["cx"] = float(det.get("cx", 0.0))
            vis["cy"] = float(det.get("cy", 0.0))
            vis["cz"] = float(det.get("cz", 0.0))
            payload.append(vis)
        return payload

    def _serialize_fused(self, fused_list: List[dict]) -> List[dict]:
        payload = []
        for det in fused_list:
            vis = prepare_visual_item(
                class_id=int(det.get("cls", 0)),
                cx=float(det.get("cx", 0.0)),
                cy=float(det.get("cy", 0.0)),
                cz=float(det.get("cz", 0.0)),
                length=float(det.get("length", 0.0)),
                width=float(det.get("width", 0.0)),
                yaw_deg=float(det.get("yaw", 0.0)),
                pitch_deg=float(det.get("pitch", 0.0)),
                roll_deg=float(det.get("roll", 0.0)),
                cfg=self.viz_cfg,
                score=det.get("score", 0.0),
            )
            vis["sources"] = list(det.get("source_cams", []))
            vis["source_cams"] = list(det.get("source_cams", []))
            vis["cx"] = float(det.get("cx", 0.0))
            vis["cy"] = float(det.get("cy", 0.0))
            vis["cz"] = float(det.get("cz", 0.0))
            payload.append(vis)
        return payload

    def _tracks_to_dicts(self, tracks: np.ndarray) -> List[dict]:
        rows = tracks if (tracks is not None and len(tracks)) else []
        payload = []
        for row in rows:
            tid = int(row[0]); cls = int(row[1])
            cx, cy, L, W, yaw = map(float, row[2:7])
            extra = self.track_meta.get(tid, {})
            vis = prepare_visual_item(
                class_id=cls,
                cx=cx,
                cy=cy,
                cz=float(extra.get("cz", 0.0)),
                length=L,
                width=W,
                yaw_deg=yaw,
                pitch_deg=float(extra.get("pitch", 0.0)),
                roll_deg=float(extra.get("roll", 0.0)),
                cfg=self.viz_cfg,
                score=extra.get("score", 0.0),
            )
            vis["id"] = tid
            vis["track_id"] = tid
            vis["class"] = cls
            vis["sources"] = list(extra.get("source_cams", []))
            vis["cx"] = cx
            vis["cy"] = cy
            payload.append(vis)
        return payload

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
    ap.add_argument("--fps", type=float, default=30.0, help="갱신 FPS")
    ap.add_argument("--iou-thr", type=float, default=0.25, help="AABB IoU 군집 임계값(융합 단계)")
    ap.add_argument("--save-dir", default=None, help="롤링 저장 루트 디렉토리 (예: ./logs)")
    ap.add_argument("--roll-secs", type=int, default=60, help="롤링 간격(초) 기본 60s")
    ap.add_argument("--roll-max-rows", type=int, default=1000, help="파일당 최대 행수 기본 1000")
    ap.add_argument("--udp-port", type=int, default=50050, help="단일 UDP 수신 포트")
    ap.add_argument("--tx-host", default=None, help="추적 결과를 전송할 상대 호스트(IP)")
    ap.add_argument("--tx-port", type=int, default=60050, help="추적 결과 전송 포트")
    ap.add_argument("--tx-protocol", choices=["udp","tcp"], default="udp", help="추적 결과 전송 프로토콜")
    ap.add_argument("--carla-host", default=None, help="CARLA 나 다른 시뮬레이터로 보낼 호스트 (선택)")
    ap.add_argument("--carla-port", type=int, default=61000, help="CARLA/시각화 전용 포트")
    ap.add_argument("--global-ply", default="pointcloud/merged.ply", help="웹뷰에서 사용할 글로벌 PLY")
    ap.add_argument("--vehicle-glb", default="pointcloud/car.glb", help="웹뷰 차량 GLB 경로")
    ap.add_argument("--web-host", default="0.0.0.0", help="웹 서버 호스트")
    ap.add_argument("--web-port", type=int, default=18090, help="웹 서버 포트")
    ap.add_argument("--no-web", action="store_true", help="FastAPI 웹 뷰어 비활성화")
    ap.add_argument("--tracker-fixed-length", type=float, default=None,
                    help="SORT Tracker에 입력할 길이를 고정 (미지정 시 detect된 길이 사용)")
    ap.add_argument("--tracker-fixed-width", type=float, default=None,
                    help="SORT Tracker에 입력할 너비를 고정 (미지정 시 detect된 너비 사용)")
    ap.add_argument("--size-mode", choices=["bbox","fixed","mesh"], default="mesh",
                    help="웹 뷰에서 사용할 차량 스케일링 모드")
    ap.add_argument("--fixed-length", type=float, default=4.5,
                    help="size-mode=fixed 일 때 차량 길이")
    ap.add_argument("--fixed-width", type=float, default=1.8,
                    help="size-mode=fixed 일 때 차량 너비")
    ap.add_argument("--height-scale", type=float, default=0.5,
                    help="bbox/fixed 모드일 때 차량 높이 = width * height_scale")
    ap.add_argument("--mesh-scale", type=float, default=1.0,
                    help="size-mode=mesh 일 때 GLB에 곱할 유니폼 스케일")
    ap.add_argument("--mesh-height", type=float, default=0.0,
                    help="size-mode=mesh 일 때 지면 높이 계산용 높이(0이면 mesh-scale 사용)")
    ap.add_argument("--z-offset", type=float, default=0.0,
                    help="모든 박스에 추가할 z 오프셋")
    ap.add_argument("--invert-bev-y", dest="invert_bev_y", action="store_true",
                    help="Y축 반전 적용")
    ap.add_argument("--no-invert-bev-y", dest="invert_bev_y", action="store_false",
                    help="Y축 반전 해제")
    ap.set_defaults(invert_bev_y=True)
    ap.add_argument("--normalize-vehicle", dest="normalize_vehicle", action="store_true",
                    help="GLB를 최대 변 1.0으로 정규화")
    ap.add_argument("--no-normalize-vehicle", dest="normalize_vehicle", action="store_false",
                    help="GLB 정규화 끄기")
    ap.set_defaults(normalize_vehicle=True)
    ap.add_argument("--vehicle-y-up", dest="vehicle_y_up", action="store_true",
                    help="GLB가 Y-up이면 +90° 회전 적용(기본)")
    ap.add_argument("--vehicle-z-up", dest="vehicle_y_up", action="store_false",
                    help="GLB가 이미 Z-up이면 회전 생략")
    ap.set_defaults(vehicle_y_up=True)
    ap.add_argument("--flip-ply-y", dest="flip_ply_y", action="store_true",
                    help="global ply의 Y축을 반전하여 로드")
    ap.add_argument("--no-flip-ply-y", dest="flip_ply_y", action="store_false",
                    help="global ply Y축 반전하지 않음")
    ap.set_defaults(flip_ply_y=False)

    args = ap.parse_args()

    cam_ports = parse_cam_ports(args.cam_ports)
    viz_cfg = VizSizeConfig(
        size_mode=args.size_mode,
        fixed_length=args.fixed_length,
        fixed_width=args.fixed_width,
        height_scale=args.height_scale,
        mesh_scale=args.mesh_scale,
        mesh_height=args.mesh_height,
        z_offset=args.z_offset,
        invert_bev_y=args.invert_bev_y,
    )
    client_config = {
        "flipPlyY": bool(args.flip_ply_y),
        "normalizeVehicle": bool(args.normalize_vehicle),
        "vehicleYAxisUp": bool(args.vehicle_y_up),
    }

    server = RealtimeFusionServer(
        cam_ports=cam_ports,
        fps=args.fps,
        iou_cluster_thr=args.iou_thr,
        save_dir=args.save_dir,
        roll_secs=args.roll_secs,
        roll_max_rows=args.roll_max_rows,
        single_port=args.udp_port,
        tx_host=args.tx_host,
        tx_port=args.tx_port,
        tx_protocol=args.tx_protocol,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        global_ply=args.global_ply,
        vehicle_glb=args.vehicle_glb,
        web_host=args.web_host,
        web_port=args.web_port,
        enable_web=(not args.no_web),
        viz_config=viz_cfg,
        client_config=client_config,
        tracker_fixed_length=args.tracker_fixed_length,
        tracker_fixed_width=args.tracker_fixed_width,
    )
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.receiver.stop()
        if server.saver:
            server.saver.stop()
        if server.track_tx:
            server.track_tx.close()
        if server.carla_tx:
            server.carla_tx.close()
        if server.web:
            server.web.stop()

if __name__ == "__main__":
    main()

'''
python server.py \
  --udp-port 50050 \
  --tx-host 192.168.0.200 --tx-port 60100 \
  --carla-host 127.0.0.1 --carla-port 60200 <- 여기로 칼라고ㅗ고하시면됨
  '''
