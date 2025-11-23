import argparse
import json
import queue
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np
import open3d as o3d
try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

from utils.merge.merge_dist_wbf import (
    cluster_by_aabb_iou, fuse_cluster_weighted
)
from utils.sort.tracker import SortTracker
from realtime_show_result.viz_utils import VizSizeConfig, prepare_visual_item

COLOR_LABELS = ("red", "pink", "green", "white", "yellow", "purple")
VALID_COLORS = {color: color for color in COLOR_LABELS}
COLOR_HEX_MAP = {
    "red": "#f52629",
    "pink": "#f53e96",
    "green": "#48ad0d",
    "white": "#f0f0f0",
    "yellow": "#ffdd00",
    "purple": "#781de7",
}

def normalize_color_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    color = str(value).strip().lower()
    if not color or color == "none":
        return None
    return VALID_COLORS.get(color)


def color_label_to_hex(color: Optional[str]) -> Optional[str]:
    if not color:
        return None
    return COLOR_HEX_MAP.get(color)


def _normalize_base_url(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    base = str(value).strip()
    if not base:
        return None
    return base.rstrip("/")

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

def _safe_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def _slugify_label(text: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", str(text)).strip("-").lower()
    return slug or "cam"

def _dedup_slug(base: str, used: set) -> str:
    slug = base
    idx = 2
    while slug in used:
        slug = f"{base}-{idx}"
        idx += 1
    used.add(slug)
    return slug

def _filter_candidates_by_cam_id(candidates, cam_id: Optional[int]):
    if cam_id is None:
        return candidates
    pattern = re.compile(rf"^cam_?{cam_id}(?!\d)", re.IGNORECASE)
    filtered = [c for c in candidates if pattern.search(c.name)]
    return filtered or candidates

def _guess_local_ply(root: Path, cam_id: Optional[int], name: Optional[str]) -> Optional[Path]:
    if cam_id is None:
        return None
    candidates = []
    patterns = [
        f"cam_{cam_id}_*.ply",
        f"cam{cam_id}_*.ply",
    ]
    if name:
        name_slug = _slugify_label(name).replace("-", "_")
        patterns.append(f"{name_slug}*.ply")
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    if not candidates:
        for pattern in patterns:
            candidates.extend(root.rglob(pattern))
    if not candidates:
        return None
    candidates = _filter_candidates_by_cam_id(candidates, cam_id)
    candidates.sort()
    return candidates[0].resolve()

def _guess_local_lut(root: Path, cam_id: Optional[int], name: Optional[str]) -> Optional[Path]:
    if cam_id is None:
        return None
    candidates = []
    patterns = [
        f"cam_{cam_id}_*.npz",
        f"cam{cam_id}_*.npz",
    ]
    if name:
        name_slug = _slugify_label(name).replace("-", "_")
        patterns.append(f"{name_slug}*.npz")
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    if not candidates:
        for pattern in patterns:
            candidates.extend(root.rglob(pattern))
    if not candidates:
        return None
    candidates = _filter_candidates_by_cam_id(candidates, cam_id)
    candidates.sort()
    return candidates[0].resolve()

def _lut_mask_from_obj(obj: dict) -> Optional[np.ndarray]:
    for key in ("ground_valid_mask", "valid_mask", "floor_mask"):
        if key in obj:
            mask = np.asarray(obj[key]).astype(bool)
            if "X" in obj and mask.shape == np.asarray(obj["X"]).shape:
                return mask
    if all(k in obj for k in ("X", "Y", "Z")):
        X = np.asarray(obj["X"])
        Y = np.asarray(obj["Y"])
        Z = np.asarray(obj["Z"])
        return np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    return None

def _load_visible_xyz_from_lut(path: Path) -> Optional[Tuple[np.ndarray, bool]]:
    try:
        with np.load(str(path)) as data:
            if not {"X", "Y", "Z"}.issubset(set(data.files)):
                print(f"[Fusion] LUT missing XYZ -> {path}")
                return None
            mask = _lut_mask_from_obj(data)
            if mask is None:
                print(f"[Fusion] LUT mask missing -> {path}")
                return None
            X = np.asarray(data["X"], dtype=np.float32)
            Y = np.asarray(data["Y"], dtype=np.float32)
            Z = np.asarray(data["Z"], dtype=np.float32)
            xyz = np.stack([X[mask], Y[mask], Z[mask]], axis=1).astype(np.float32)
            colors = None
            has_rgb = False
            if {"R", "G", "B"}.issubset(set(data.files)):
                R = np.asarray(data["R"], dtype=np.float32)
                G = np.asarray(data["G"], dtype=np.float32)
                B = np.asarray(data["B"], dtype=np.float32)
                try:
                    colors = np.stack([R[mask], G[mask], B[mask]], axis=1).astype(np.float32)
                    # Normalize 0-255 to 0-1 if needed
                    max_val = float(np.nanmax(colors)) if colors.size else 0.0
                    if max_val > 1.01:
                        colors /= 255.0
                    has_rgb = True
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[Fusion] LUT RGB load failed for {path}: {exc}")
                    colors = None
                    has_rgb = False
    except Exception as exc:
        print(f"[Fusion] failed to load LUT {path}: {exc}")
        return None
    if xyz.size == 0:
        return None
    if colors is not None and colors.shape[0] == xyz.shape[0]:
        pts = np.concatenate([xyz, colors], axis=1)
        return pts, True
    return xyz, False

def load_camera_markers(path: Optional[str], local_ply_root: Optional[str] = None,
                        local_lut_root: Optional[str] = None) -> List[dict]:
    markers: List[dict] = []
    if not path:
        return markers
    json_path = Path(path)
    if not json_path.exists():
        print(f"[Fusion] camera position file not found: {json_path}")
        return markers
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Fusion] failed to read camera position file {json_path}: {exc}")
        return markers

    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        cams = raw.get("cameras")
        if isinstance(cams, list):
            entries = cams
        else:
            entries = [raw]
    else:
        print(f"[Fusion] camera position file {json_path} has unexpected format")
        return markers

    base_dir = json_path.parent
    ply_root = Path(local_ply_root).resolve() if local_ply_root else None
    lut_root = Path(local_lut_root).resolve() if local_lut_root else None
    used_slugs = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("label")
        camera_id = entry.get("camera_id", entry.get("id"))
        if camera_id is not None:
            try:
                camera_id = int(camera_id)
            except (TypeError, ValueError):
                camera_id = None
        if camera_id is None and name:
            nums = re.findall(r"\d+", str(name))
            if nums:
                try:
                    camera_id = int(nums[-1])
                except ValueError:
                    camera_id = None

        pos_src = entry.get("pos") if isinstance(entry.get("pos"), dict) else {}
        x = _safe_float(pos_src.get("x", entry.get("x")))
        y = _safe_float(pos_src.get("y", entry.get("y")))
        if x is None or y is None:
            continue
        z = _safe_float(pos_src.get("z", entry.get("z")), 0.0)

        rot_src = entry.get("rot") if isinstance(entry.get("rot"), dict) else {}
        rotation = {
            "pitch": _safe_float(rot_src.get("pitch", entry.get("pitch")), 0.0),
            "yaw": _safe_float(rot_src.get("yaw", entry.get("yaw")), 0.0),
            "roll": _safe_float(rot_src.get("roll", entry.get("roll")), 0.0),
        }

        local_ref = entry.get("local_ply") or entry.get("visible_ply")
        ply_path = None
        if isinstance(local_ref, str) and local_ref:
            candidate = Path(local_ref)
            ply_path = candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
        if (ply_path is None or not ply_path.exists()) and ply_root and ply_root.exists():
            guessed = _guess_local_ply(ply_root, camera_id, name)
            if guessed and guessed.exists():
                ply_path = guessed
        if ply_path and not ply_path.exists():
            print(f"[Fusion] WARN: local ply for {name or camera_id} missing: {ply_path}")
            ply_path = None

        lut_ref = entry.get("local_lut") or entry.get("lut") or entry.get("lut_npz")
        lut_path = None
        if isinstance(lut_ref, str) and lut_ref:
            candidate = Path(lut_ref)
            lut_path = candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
        if (lut_path is None or not lut_path.exists()) and lut_root and lut_root.exists():
            guessed_lut = _guess_local_lut(lut_root, camera_id, name)
            if guessed_lut and guessed_lut.exists():
                lut_path = guessed_lut
        if lut_path and not lut_path.exists():
            print(f"[Fusion] WARN: local LUT for {name or camera_id} missing: {lut_path}")
            lut_path = None

        display_name = str(name or (f"cam{camera_id}" if camera_id is not None else f"marker{idx+1}"))
        slug = _dedup_slug(_slugify_label(display_name), used_slugs)
        overlay_base_url = entry.get("overlay_base_url") or entry.get("overlay_url") or entry.get("overlay_host")
        if overlay_base_url is not None:
            overlay_base_url = str(overlay_base_url).strip()
            if overlay_base_url:
                overlay_base_url = overlay_base_url.rstrip("/")
            else:
                overlay_base_url = None
        markers.append({
            "key": slug,
            "name": display_name,
            "camera_id": camera_id,
            "position": {"x": float(x), "y": float(y), "z": float(z)},
            "rotation": rotation,
            "local_ply": str(ply_path) if ply_path else None,
            "local_lut": str(lut_path) if lut_path else None,
            "overlay_base_url": overlay_base_url,
        })

    if not markers:
        print(f"[Fusion] camera position file {json_path} contained no usable entries")
    return markers

class GroundHeightLookup:
    """
    간단한 지면 높이 질의용: global ply의 XY→Z 근접 값을 가져온다.
    """
    def __init__(self, ply_path: Optional[str], *, flip_y: bool = False, max_points: int = 500_000):
        self.enabled = False
        self.xy: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None
        self.tree = None

        if not ply_path:
            return
        path = Path(ply_path)
        if not path.exists():
            print(f"[GroundZ] ply not found: {path}")
            return
        try:
            cloud = o3d.io.read_point_cloud(str(path))
            pts = np.asarray(cloud.points, dtype=np.float32)
        except Exception as exc:
            print(f"[GroundZ] failed to load ply {path}: {exc}")
            return
        if pts.size == 0:
            print(f"[GroundZ] empty ply: {path}")
            return
        if flip_y:
            pts[:, 1] *= -1.0
        if max_points and len(pts) > max_points:
            stride = max(1, int(len(pts) / max_points))
            pts = pts[::stride]
            print(f"[GroundZ] subsampled ply to {len(pts)} points (stride={stride})")
        self.xy = pts[:, :2].astype(np.float32, copy=False)
        self.z = pts[:, 2].astype(np.float32, copy=False)
        if cKDTree is not None:
            try:
                self.tree = cKDTree(self.xy)
            except Exception as exc:
                print(f"[GroundZ] cKDTree build failed: {exc}")
                self.tree = None
        self.enabled = True
        print(f"[GroundZ] loaded {len(self.z)} points from {path.name} (flip_y={flip_y}, kdtree={'yes' if self.tree else 'no'})")

    def query(self, x: float, y: float, *, default: float = 0.0, k: int = 5) -> float:
        if not self.enabled or self.xy is None or self.z is None:
            return float(default)
        try:
            if self.tree is not None:
                k_use = min(max(1, int(k)), len(self.z))
                dists, idxs = self.tree.query([float(x), float(y)], k=k_use)
                idx_arr = np.atleast_1d(idxs)
                z_vals = self.z[idx_arr]
                if k_use > 1:
                    dist_arr = np.atleast_1d(dists)
                    weights = 1.0 / np.maximum(dist_arr, 1e-3)
                    return float(np.average(z_vals, weights=weights))
                return float(z_vals[0])
            # fallback: 최근접 점 하나
            diff = self.xy - np.array([float(x), float(y)], dtype=np.float32)
            d2 = np.sum(diff * diff, axis=1)
            idx = int(np.argmin(d2))
            return float(self.z[idx])
        except Exception:
            return float(default)

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
                    "color": extra.get("color"),
                    "color_hex": extra.get("color_hex"),
                    "color_confidence": float(extra.get("color_confidence", 0.0)),
                })
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            if self.protocol == "udp":
                self.sock.sendto(data, self.addr)
            else:
                self.sock.sendall(data + b"\n")
        except Exception as e:
            print(f"[TrackBroadcaster] send error: {e}")


class CommandServer:
    """
    간단한 TCP 명령 서버. 각 연결은 JSON 한 줄을 보내고 응답을 받는다.
    """
    def __init__(self, host: str, port: int, command_queue: "queue.Queue[dict]", response_timeout: float = 2.0):
        self.host = host
        self.port = int(port)
        self.q = command_queue
        self.response_timeout = response_timeout
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._serve_forever, daemon=True)
        self._thread.start()
        print(f"[CommandServer] listening on {self.host}:{self.port}")

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    def _serve_forever(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((self.host, self.port))
                srv.listen()
                self._sock = srv
                while self._running:
                    try:
                        conn, addr = srv.accept()
                    except OSError:
                        break
                    threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
        except Exception as exc:
            print(f"[CommandServer] server error: {exc}")

    def _handle_client(self, conn: socket.socket, addr):
        with conn:
            try:
                data = self._recv_all(conn)
                if not data:
                    return
                response = self._process_payload(data)
            except Exception as exc:
                response = {"status": "error", "message": str(exc)}
            try:
                payload = (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
                conn.sendall(payload)
            except Exception:
                pass

    def _recv_all(self, conn: socket.socket) -> str:
        buf = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"\n" in chunk:
                break
        return buf.decode("utf-8").strip()

    def _process_payload(self, data: str) -> dict:
        if not data:
            return {"status": "error", "message": "empty payload"}
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return {"status": "error", "message": "invalid json"}
        if not isinstance(payload, dict):
            return {"status": "error", "message": "payload must be object"}
        cmd = payload.get("cmd")
        if not cmd:
            return {"status": "error", "message": "cmd required"}
        response_q: queue.Queue = queue.Queue(maxsize=1)
        item = {"cmd": cmd, "payload": payload, "response": response_q}
        try:
            self.q.put_nowait(item)
        except queue.Full:
            return {"status": "error", "message": "server busy"}
        try:
            return response_q.get(timeout=self.response_timeout)
        except queue.Empty:
            return {"status": "error", "message": "command timeout"}

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
        camera_markers: Optional[List[dict]] = None,
        command_handler: Optional[Callable[[dict, float], dict]] = None,
        overlay_sources: Optional[Dict[int, str]] = None,
        overlay_proxy_prefix: str = "/proxy",
        overlay_proxy_timeout: float = 3.0,
        overlay_default_base: Optional[str] = None,
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
        self.command_handler = command_handler
        prefix = str(overlay_proxy_prefix or "/proxy").strip()
        if not prefix.startswith("/"):
            prefix = "/" + prefix
        self.overlay_proxy_prefix = prefix.rstrip("/") or "/proxy"
        self.overlay_proxy_timeout = max(0.5, float(overlay_proxy_timeout))
        self.overlay_sources: Dict[int, str] = {}
        for key, url in (overlay_sources or {}).items():
            norm = _normalize_base_url(url)
            if norm is None:
                continue
            try:
                cam_id = int(key)
            except (TypeError, ValueError):
                continue
            self.overlay_sources[cam_id] = norm
        self.overlay_default_base = _normalize_base_url(overlay_default_base)
        self.overlay_proxy_enabled = bool(self.overlay_sources)
        if static_root is None:
            static_root = Path(__file__).resolve().parent / "realtime_show_result" / "static"
        self.static_root = static_root
        self.page_map = {
            "raw": self.static_root / "fusion_raw.html",
            "fused": self.static_root / "fusion_fused.html",
            "tracks": self.static_root / "fusion_tracks.html",
            "admin": self.static_root / "fusion_admin.html",
        }
        self.viz_cfg = viz_config
        self.client_config = dict(client_config or {})
        self.client_config.setdefault("normalizeVehicle", True)
        self.client_config.setdefault("vehicleYAxisUp", True)
        self.client_config.setdefault("flipPlyY", False)
        self.client_config.setdefault("showSceneAxes", False)
        self.client_config.setdefault("showDebugMarker", False)
        self.client_config.setdefault("mode", "fusion")
        self.client_config.setdefault("flipMarkerX", False)
        self.client_config.setdefault("flipMarkerY", False)
        self.client_config["vizConfig"] = self.viz_cfg.as_client_dict()
        self.client_config["initialViewTarget"] = [0,-50,50]
        self.client_config["initialViewOffset"] = [0, -50, 60]

        self.camera_marker_payload: List[dict] = []
        self.marker_local_map: Dict[str, str] = {}
        self.marker_visible_arrays: Dict[str, np.ndarray] = {}
        self.marker_visible_meta: Dict[str, dict] = {}
        for entry in camera_markers or []:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("key") or entry.get("name") or entry.get("camera_id") or len(self.camera_marker_payload))
            pos = entry.get("position")
            if not isinstance(pos, dict):
                pos = {}
            rot = entry.get("rotation")
            if not isinstance(rot, dict):
                rot = {}
            marker_payload = {
                "key": key,
                "name": str(entry.get("name") or key),
                "camera_id": entry.get("camera_id"),
                "position": {
                    "x": _safe_float(pos.get("x"), 0.0) or 0.0,
                    "y": _safe_float(pos.get("y"), 0.0) or 0.0,
                    "z": _safe_float(pos.get("z"), 0.0) or 0.0,
                },
                "rotation": {
                    "pitch": _safe_float(rot.get("pitch"), 0.0) or 0.0,
                    "yaw": _safe_float(rot.get("yaw"), 0.0) or 0.0,
                    "roll": _safe_float(rot.get("roll"), 0.0) or 0.0,
                },
                "overlay_base_url": entry.get("overlay_base_url"),
            }
            local_ply = entry.get("local_ply")
            if local_ply and os.path.exists(local_ply):
                marker_payload["local_ply_url"] = f"/assets/cameras/{key}/local.ply"
                marker_payload["has_local_ply"] = True
                self.marker_local_map[key] = str(Path(local_ply).resolve())
            else:
                marker_payload["local_ply_url"] = None
                marker_payload["has_local_ply"] = False

            visible_arr = None
            visible_has_rgb = False
            local_lut = entry.get("local_lut")
            if local_lut and os.path.exists(local_lut):
                loaded = _load_visible_xyz_from_lut(Path(local_lut))
                if loaded is not None:
                    visible_arr, visible_has_rgb = loaded
            if visible_arr is not None:
                stride = int(visible_arr.shape[1])
                self.marker_visible_arrays[key] = visible_arr
                meta = {
                    "count": int(visible_arr.shape[0]),
                    "stride": stride,
                    "source": str(local_lut),
                    "has_rgb": bool(visible_has_rgb),
                }
                self.marker_visible_meta[key] = meta
                marker_payload["has_local_visible"] = True
                marker_payload["local_visible_url"] = f"/assets/cameras/{key}/visible.bin"
                marker_payload["local_visible_stride"] = stride
                marker_payload["local_visible_count"] = meta["count"]
                marker_payload["local_visible_has_rgb"] = bool(visible_has_rgb)
            else:
                marker_payload["has_local_visible"] = False
                marker_payload["local_visible_url"] = None
                marker_payload["local_visible_stride"] = None
                marker_payload["local_visible_count"] = 0
                marker_payload["local_visible_has_rgb"] = False
            self.camera_marker_payload.append(marker_payload)

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
            page = self.page_map.get("tracks")
            if page and page.exists():
                return FileResponse(str(page))
            return {"status": "ok", "message": "fusion_tracks.html missing"}

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

        @self.app.get("/fusion/admin")
        def _view_admin():
            page = self.page_map.get("admin")
            if page and page.exists():
                return FileResponse(str(page))
            raise HTTPException(status_code=404, detail="fusion_admin.html missing")

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
                    "camera_positions": self.camera_marker_payload,
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

        def _run_admin_command(payload: dict) -> dict:
            if not self.command_handler:
                raise HTTPException(status_code=503, detail="admin command unavailable")
            try:
                resp = self.command_handler(payload, 2.0)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))
            if not isinstance(resp, dict):
                raise HTTPException(status_code=502, detail="invalid admin response")
            return resp

        @self.app.get("/api/admin/tracks")
        def _admin_tracks():
            resp = _run_admin_command({"cmd": "list_tracks"})
            if resp.get("status") != "ok":
                raise HTTPException(status_code=400, detail=resp.get("message", "command failed"))
            return resp

        @self.app.post("/api/admin/tracks/{track_id}/color")
        def _admin_set_color(track_id: int, body: Optional[dict] = None):
            payload = {"cmd": "set_color", "track_id": track_id, "color": None}
            if isinstance(body, dict) and "color" in body:
                payload["color"] = body.get("color")
            resp = _run_admin_command(payload)
            if resp.get("status") != "ok":
                raise HTTPException(status_code=400, detail=resp.get("message", "command failed"))
            return resp

        @self.app.post("/api/admin/tracks/{track_id}/yaw")
        def _admin_set_yaw(track_id: int, body: Optional[dict] = None):
            if not isinstance(body, dict) or "yaw" not in body:
                raise HTTPException(status_code=400, detail="yaw required")
            payload = {"cmd": "set_yaw", "track_id": track_id, "yaw": body.get("yaw")}
            resp = _run_admin_command(payload)
            if resp.get("status") != "ok":
                raise HTTPException(status_code=400, detail=resp.get("message", "command failed"))
            return resp

        @self.app.post("/api/admin/tracks/{track_id}/flip")
        def _admin_flip(track_id: int, body: Optional[dict] = None):
            delta = body.get("delta", 180.0) if isinstance(body, dict) else 180.0
            payload = {"cmd": "flip_yaw", "track_id": track_id, "delta": delta}
            resp = _run_admin_command(payload)
            if resp.get("status") != "ok":
                raise HTTPException(status_code=400, detail=resp.get("message", "command failed"))
            return resp

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

        @self.app.get("/assets/cameras/{marker_key}/local.ply")
        def _marker_local(marker_key: str):
            path = self.marker_local_map.get(str(marker_key))
            if not path or not os.path.exists(path):
                raise HTTPException(status_code=404, detail="local ply missing")
            return FileResponse(path, filename=f"{marker_key}.ply")

        @self.app.get("/assets/cameras/{marker_key}/visible.bin")
        def _marker_visible(marker_key: str):
            arr = self.marker_visible_arrays.get(str(marker_key))
            if arr is None:
                raise HTTPException(status_code=404, detail="visible cloud missing")
            meta = self.marker_visible_meta.get(str(marker_key)) or {}
            headers = {
                "X-Point-Count": str(meta.get("count", arr.shape[0])),
                "X-Stride": str(meta.get("stride", arr.shape[1] if arr.ndim == 2 else 3)),
                "X-Has-RGB": "1" if meta.get("has_rgb") else "0",
            }
            return Response(content=arr.tobytes(), media_type="application/octet-stream", headers=headers)

        @self.app.get(f"{self.overlay_proxy_prefix}/api/cameras/{{camera_id}}/overlay.jpg")
        def _proxy_overlay(camera_id: int):
            if not self.overlay_proxy_enabled:
                raise HTTPException(status_code=404, detail="overlay proxy disabled")
            return self._proxy_overlay_request(int(camera_id))

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

    def _proxy_overlay_request(self, camera_id: int):
        base = self.overlay_sources.get(int(camera_id))
        if not base and self.overlay_default_base:
            base = self.overlay_default_base
        if not base:
            raise HTTPException(status_code=404, detail="overlay source not configured")
        target_url = f"{base}/api/cameras/{camera_id}/overlay.jpg"
        try:
            with urllib.request.urlopen(target_url, timeout=self.overlay_proxy_timeout) as resp:
                data = resp.read()
                content_type = resp.headers.get("Content-Type") or "image/jpeg"
        except urllib.error.HTTPError as exc:
            raise HTTPException(status_code=exc.code, detail=f"overlay upstream error: {exc.reason}")
        except urllib.error.URLError as exc:
            raise HTTPException(status_code=502, detail=f"overlay upstream unreachable: {exc.reason}")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"overlay proxy failed: {exc}")
        return Response(content=data, media_type=content_type)


class UDPReceiverSingle:
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
                    # self.q.clear()
                    continue
                self.q.put_nowait({"cam": cam, "ts": ts, "dets": dets})
            except Exception:
                continue

    def _parse_payload(self, data: bytes):
        try:
            msg = json.loads(data.decode("utf-8"))
            if isinstance(msg, dict) and msg.get("type") == "bev_labels":
                cam_id = int(msg.get("camera_id", 0) or 0)
                cam = f"cam{cam_id}" if cam_id else "cam?"
                dets = []
                for it in msg.get("items", []):
                    cx, cy = it["center"]
                    color = normalize_color_label(it.get("color"))
                    # color_hex = color_label_to_hex(color)
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
                        "color": color,
                        # "color_hex": color_hex,
                    })
                self._log_packet(cam, dets if dets else [], meta=msg)
                return cam, dets if dets else []
        except Exception:
            pass

# -----------------------------
# 파이프라인: 수집 → 통합/융합 → 추적 → 시각화
# -----------------------------
class RealtimeFusionServer:
    def __init__(
        self,
        cam_ports: Dict[str, int],
        cam_positions_path: Optional[str] = None,
        local_ply_dir: Optional[str] = None,
        local_lut_dir: Optional[str] = None,
        fps: float = 10.0,
        iou_cluster_thr: float = 0.25,
        single_port: int = 50050,
        tx_host: Optional[str] = None, tx_port: int = 60050, tx_protocol: str = "udp",
        carla_host: Optional[str] = None, carla_port: int = 61000,
        global_ply: str = "real_global_ply.ply",
        vehicle_glb: str = "pointcloud/car.glb",
        web_host: str = "0.0.0.0",
        web_port: int = 18090,
        enable_web: bool = True,
        viz_config: VizSizeConfig = VizSizeConfig(),
        client_config: Optional[dict] = None,
        tracker_fixed_length: Optional[float] = None,
        tracker_fixed_width: Optional[float] = None,
        command_host: Optional[str] = None,
        command_port: Optional[int] = None,
    ):
        self.fps = fps
        self.dt = 1.0 / max(1e-3, fps)
        # Drop cached detections once they become too old so we don't replay stale frames forever
        self.buffer_ttl = max(self.dt * 2.0, 1.0)
        self.iou_thr = iou_cluster_thr
        self.track_meta: Dict[int, dict] = {}
        self.active_cams = set()
        self.color_bias_strength = 0.3
        self.color_bias_min_votes = 2
        delta = min(max(self.color_bias_strength * 0.25, 0.0), 0.00)
        # delta = 1
        self.color_cluster_bonus = delta
        self.color_cluster_penalty = delta

        # 단일 소켓 리시버 (엣지→서버 UDP)
        self.receiver = UDPReceiverSingle(single_port)

        # 카메라 위치(가중치/거리 계산 및 UI용)
        self.camera_markers = load_camera_markers(cam_positions_path, local_ply_dir, local_lut_dir)
        self.cam_xy: Dict[str, Tuple[float, float]] = {}
        self.overlay_sources: Dict[int, str] = {}
        for marker in self.camera_markers:
            pos = marker.get("position") or {}
            x = _safe_float(pos.get("x"))
            y = _safe_float(pos.get("y"))
            if x is None or y is None:
                continue
            name = marker.get("name") or marker.get("key")
            if not name:
                continue
            self.cam_xy[str(name)] = (float(x), float(y))
            try:
                cam_id_int = int(marker.get("camera_id"))
            except (TypeError, ValueError):
                cam_id_int = None
            overlay_base = _normalize_base_url(marker.get("overlay_base_url"))
            if overlay_base and cam_id_int is not None:
                self.overlay_sources[cam_id_int] = overlay_base
        if not self.cam_xy:
            print("[Fusion] WARN: no camera positions loaded; distance weighting falls back to origin.")
        self.buffer = {}

        flip_ply_y = bool((client_config or {}).get("flipPlyY", False))
        self.ground_height = GroundHeightLookup(global_ply, flip_y=flip_ply_y)

        self.track_tx = TrackBroadcaster(tx_host, tx_port, tx_protocol) if tx_host else None
        self.carla_tx = TrackBroadcaster(carla_host, carla_port) if carla_host else None
        self.viz_cfg = viz_config
        self.client_config = client_config or {}
        self.overlay_default_base = _normalize_base_url(self.client_config.get("overlayBaseUrl"))
        self.overlay_proxy_prefix = "/proxy"
        if self.overlay_sources:
            self.client_config["overlayBaseUrl"] = self.overlay_proxy_prefix
        self.tracker_fixed_length = float(tracker_fixed_length) if tracker_fixed_length is not None else None
        self.tracker_fixed_width = float(tracker_fixed_width) if tracker_fixed_width is not None else None
        self.web = GlobalWebServer(
            global_ply=global_ply,
            vehicle_glb=vehicle_glb,
            host=web_host,
            port=web_port,
            viz_config=self.viz_cfg,
            client_config=self.client_config,
            camera_markers=self.camera_markers,
            command_handler=(self._send_command if command_host and command_port else None),
            overlay_sources=self.overlay_sources,
            overlay_proxy_prefix=self.overlay_proxy_prefix,
            overlay_default_base=self.overlay_default_base,
        ) if enable_web else None

        # 프레임 버퍼(최근 T초 동안 카메라별 최신)
        self.buffer: Dict[str, deque] = {cam: deque(maxlen=1) for cam in cam_ports.keys()}

        # 추적기
        self.tracker = SortTracker(max_age=20, min_hits=10, iou_threshold=0.15) 
        self._log_interval = 1.0
        self._next_log_ts = 0.0
        self.command_queue: Optional[queue.Queue] = None
        self.command_server: Optional[CommandServer] = None
        if command_host and command_port:
            self.command_queue = queue.Queue()
            self.command_server = CommandServer(command_host, command_port, self.command_queue)

    def _register_cam_if_needed(self, cam_name: str): # 수신 시 신규카메라 등록 
        if cam_name not in self.buffer:
            self.buffer[cam_name] = deque(maxlen=1)
        self.active_cams.add(cam_name)

    def _ground_height_at(self, x: float, y: float, default: float = 0.0) -> float:
        if hasattr(self, "ground_height") and self.ground_height and self.ground_height.enabled:
            return self.ground_height.query(x, y, default=default)
        return float(default)

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
            sample_keys = ["cx","cy","length","width","yaw","score","source_cams","color"]
            sample_fused = {k: fused[0][k] for k in sample_keys if k in fused[0]}
            print(f"  fused_sample={json.dumps(sample_fused, ensure_ascii=False)}")
        if track_count:
            sample_track = json.dumps(tracks[0], ensure_ascii=False)
            print(f"  track_sample={sample_track}")
        if timings:
            timing_str = " ".join(f"{k}={v:.2f}ms" for k, v in timings.items())
            print(f"  timings {timing_str}")

    def _send_command(self, payload: dict, timeout: float = 2.0) -> dict:
        if not self.command_queue:
            raise RuntimeError("command queue not initialized")
        cmd = payload.get("cmd")
        if not cmd:
            raise RuntimeError("cmd required")
        resp_q: queue.Queue = queue.Queue(maxsize=1)
        item = {"cmd": cmd, "payload": payload, "response": resp_q}
        try:
            self.command_queue.put_nowait(item)
        except queue.Full:
            raise RuntimeError("command queue busy")
        try:
            return resp_q.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError("command timeout") from exc

    def start(self):
        self.receiver.start()
        if self.command_server:
            self.command_server.start()
        self._main_loop()

    def _main_loop(self):
        last = time.time()
        while True:
            timings: Dict[str, float] = {}
            self._process_command_queue()
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
            # tracker는 [class, x_c, y_c, l, w, angle] Nx6 입력을 받게 맞춤 
            det_rows = []
            det_colors: List[Optional[str]] = []
            for det in fused:
                det_rows.append([
                    0,
                    det["cx"],
                    det["cy"],
                    (self.tracker_fixed_length if self.tracker_fixed_length is not None else det["length"]),
                    (self.tracker_fixed_width if self.tracker_fixed_width is not None else det["width"]),
                    det["yaw"],
                ])
                # det_colors.append(normalize_color_label(det.get("color"))) 이미 하고 왔는데 굳이?
                det_colors.append(det.get("color"))
            dets_for_tracker = np.array(det_rows, dtype=float) if det_rows else np.zeros((0,6), dtype=float)
            t2 = time.perf_counter()
            tracks = self.tracker.update(dets_for_tracker, det_colors)  # shape: [N, 8] = [track_id, class, x, y, l, w, yaw]
            timings["track"] = (time.perf_counter() - t2) * 1000.0
            track_attrs = self.tracker.get_track_attributes()
            self._update_track_meta(self.tracker.get_latest_matches(), fused, track_attrs)
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

    def _gather_current(self):
        detections = []
        now = time.time()
        for cam, dq in self.buffer.items():
            if not dq:
                continue
            entry = dq[-1]
            ts = float(entry.get("ts", 0.0) or 0.0)
            if (now - ts) > self.buffer_ttl:
                dq.clear()
                continue
            for det in entry["dets"]:
                det_copy = det.copy()
                det_copy["cam"] = cam
                det_copy["ts"] = ts
                detections.append(det_copy)
        return detections

    def _process_command_queue(self):
        if not self.command_queue:
            return
        while True:
            try:
                item = self.command_queue.get_nowait()
            except queue.Empty:
                break
            response = self._handle_command_item(item)
            resp_q = item.get("response")
            if resp_q:
                try:
                    resp_q.put_nowait(response)
                except queue.Full:
                    pass

    def _handle_command_item(self, item: dict) -> dict:
        cmd = str(item.get("cmd") or "").strip().lower()
        payload = item.get("payload") or {}
        if cmd == "flip_yaw":
            track_id = payload.get("track_id")
            if track_id is None:
                return {"status": "error", "message": "track_id required"}
            try:
                tid = int(track_id)
            except (TypeError, ValueError):
                return {"status": "error", "message": "track_id must be int"}
            delta = payload.get("delta", 180.0)
            try:
                delta = float(delta)
            except (TypeError, ValueError):
                delta = 180.0
            flipped = self.tracker.force_flip_yaw(tid, offset_deg=delta)
            if flipped:
                print(f"[Command] flipped track {tid} yaw by {delta:.1f}°")
                return {"status": "ok", "track_id": tid, "delta": delta}
            return {"status": "error", "message": f"track {tid} not found"}
        if cmd == "set_color":
            track_id = payload.get("track_id")
            if track_id is None:
                return {"status": "error", "message": "track_id required"}
            try:
                tid = int(track_id)
            except (TypeError, ValueError):
                return {"status": "error", "message": "track_id must be int"}
            raw_color = payload.get("color")
            normalized_color = normalize_color_label(raw_color)
            if raw_color is not None and normalized_color is None:
                raw_str = str(raw_color).strip().lower()
                if raw_str and raw_str != "none":
                    return {"status": "error", "message": f"invalid color '{raw_color}'"}
                # empty/none → clear color
                normalized_color = None
            updated = self.tracker.force_set_color(tid, normalized_color)
            if updated:
                print(f"[Command] set track {tid} color -> {normalized_color}")
                return {"status": "ok", "track_id": tid, "color": normalized_color}
            return {"status": "error", "message": f"track {tid} not found"}
        if cmd == "set_yaw":
            track_id = payload.get("track_id")
            if track_id is None:
                return {"status": "error", "message": "track_id required"}
            try:
                tid = int(track_id)
            except (TypeError, ValueError):
                return {"status": "error", "message": "track_id must be int"}
            raw_yaw = payload.get("yaw")
            try:
                yaw_val = float(raw_yaw)
            except (TypeError, ValueError):
                return {"status": "error", "message": "yaw must be float"}
            updated = self.tracker.force_set_yaw(tid, yaw_val)
            if updated:
                print(f"[Command] set track {tid} yaw -> {yaw_val:.1f}°")
                return {"status": "ok", "track_id": tid, "yaw": yaw_val}
            return {"status": "error", "message": f"track {tid} not found"}
        if cmd == "list_tracks":
            tracks = self.tracker.list_tracks()
            return {"status": "ok", "tracks": tracks, "count": len(tracks)}
        return {"status": "error", "message": f"unknown command '{cmd}'"}

    # merge_dist_wbf 의 클러스터링/가중통합 로직을 그대로 사용
    def _fuse_boxes(self, raw_detections: List[dict]) -> List[dict]:
        if not raw_detections:
            return []
        boxes = np.array([[d["cx"], d["cy"], d["length"], d["width"], d["yaw"]] for d in raw_detections], dtype=float)
        cams  = [d.get("cam", "?") for d in raw_detections]
        colors = [normalize_color_label(d.get("color")) for d in raw_detections]
        clusters = cluster_by_aabb_iou( # 색상 같으면 iou 스레쉬홀드 낮추고 다르면 높여서 aabb기준 iou계산 클러스터
            boxes,
            iou_cluster_thr=self.iou_thr,
            color_labels=colors,
            color_bonus=self.color_cluster_bonus,
            color_penalty=self.color_cluster_penalty,
        )
        fused_list = []
        for idxs in clusters: # 클러스터마다 대표값 생성 
            weight_bias = self._color_weight_biases(raw_detections, idxs)
            rep = fuse_cluster_weighted( # 거리기반가중에 바이어스를 넣음
                boxes, cams, idxs, self.cam_xy,
                d0=5.0, p=2.0, extra_weights=weight_bias
            )
            extras = self._aggregate_cluster(raw_detections, idxs, rep) # 얘는일단그냥평균내고잇음 수정필요?
            fused_list.append({
                "cx": float(rep[0]),
                "cy": float(rep[1]),
                "length": float(rep[2]),
                "width": float(rep[3]),
                "yaw": float(rep[4]),
                **extras,
            })
        return fused_list

    def _color_weight_biases(self, detections: List[dict], idxs: List[int]) -> List[float]:
        '''이 클러스터 안에 색상 합의가 됐으면 걔네 바이어스 더 주고 아님 1'''
        normalized = [normalize_color_label(detections[i].get("color")) for i in idxs]
        color_counts = Counter([c for c in normalized if c])
        if not color_counts:
            return [1.0] * len(idxs)
        top_color, top_count = color_counts.most_common(1)[0]
        if top_count < max(self.color_bias_min_votes, 1) or self.color_bias_strength <= 0.0:
            return [1.0] * len(idxs)
        boost = 1.0 + self.color_bias_strength
        penalty = max(1.0 - self.color_bias_strength, 0.1)
        biases = []
        for color in normalized:
            if color == top_color:
                biases.append(boost)
            elif color:
                biases.append(penalty)
            else:
                biases.append(1.0)
        return biases

    def _aggregate_cluster(self, detections: List[dict], idxs: List[int], fused_box: Optional[np.ndarray] = None) -> dict:
        subset = [detections[i] for i in idxs]
        if not subset:
            return {"cz": 0.0, "pitch": 0.0, "roll": 0.0, "score": 0.0, "source_cams": []}
        score = np.mean([float(d.get("score", 0.0)) for d in subset]) # 필요없음
        pitch = np.mean([float(d.get("pitch", 0.0)) for d in subset]) # 얘도 거의 안쓰지 않나
        roll = np.mean([float(d.get("roll", 0.0)) for d in subset]) # 얘도
        cams = [d.get("cam", "?") for d in subset] # 캠 어디어디에서 따왓는지
        normalized_colors = [normalize_color_label(d.get("color")) for d in subset]
        valid_colors = [c for c in normalized_colors if c is not None]  # None/none 은 투표 제외
        color_counts = Counter(valid_colors)
        color = color_counts.most_common(1)[0][0] if color_counts else None # 투표
        color_hex = color_label_to_hex(color) # 헥사코드 6중 1로 변환
        # fused_box: [cx, cy, L, W, yaw]
        if fused_box is not None and len(fused_box) >= 4:
            cx_rep = float(fused_box[0])
            cy_rep = float(fused_box[1])
            cz_default = np.mean([float(d.get("cz", 0.0)) for d in subset])
            cz = self._ground_height_at(cx_rep, cy_rep, default=cz_default)
        else:
            cz = np.mean([float(d.get("cz", 0.0)) for d in subset])
        return {
            "cz": float(cz),
            "pitch": float(pitch),
            "roll": float(roll),
            "score": float(score),
            "source_cams": cams,
            "color": color,
            "color_hex": color_hex,
            "color_votes": dict(color_counts),
        }

    def _update_track_meta(self, matches: List[Tuple[int, int]], fused_list: List[dict], track_attrs: Dict[int, dict]):
        active_ids = set(track_attrs.keys())
        self.track_meta = {tid: meta for tid, meta in self.track_meta.items() if tid in active_ids}
        for tid, attrs in track_attrs.items():
            meta = self.track_meta.setdefault(tid, {})
            color = normalize_color_label(attrs.get("color"))
            if color:
                meta["color"] = color
                hex_color = color_label_to_hex(color)
                if hex_color:
                    meta["color_hex"] = hex_color
            else:
                meta.pop("color", None)
                meta.pop("color_hex", None)
            meta["color_locked"] = bool(attrs.get("color_locked"))
            if "color_confidence" in attrs:
                meta["color_confidence"] = attrs["color_confidence"]
        for tid, det_idx in matches:
            if det_idx < 0 or det_idx >= len(fused_list):
                continue
            det = fused_list[det_idx]
            meta = self.track_meta.setdefault(tid, {})
            meta.update({
                "cz": float(det.get("cz", 0.0)),
                "pitch": float(det.get("pitch", 0.0)),
                "roll": float(det.get("roll", 0.0)),
                "score": float(det.get("score", 0.0)),
                "source_cams": list(det.get("source_cams", [])),
            })
            if not meta.get("color_locked"):
                color = normalize_color_label(det.get("color"))
                if color:
                    meta["color"] = color
                    hex_color = color_label_to_hex(color)
                    if hex_color:
                        meta["color_hex"] = hex_color
            votes = det.get("color_votes")
            if votes:
                meta["color_votes"] = dict(votes)

    def _broadcast_tracks(self, tracks: np.ndarray, ts: float):
        if self.track_tx:
            self.track_tx.send(tracks, self.track_meta, ts)
        if self.carla_tx:
            self.carla_tx.send(tracks, self.track_meta, ts)

    def _serialize_raw(self, raw_detections: List[dict]) -> List[dict]:
        payload = []
        for det in raw_detections:
            cx_val = float(det.get("cx", 0.0))
            cy_val = float(det.get("cy", 0.0))
            cz_val = self._ground_height_at(cx_val, cy_val, default=float(det.get("cz", 0.0)))
            vis = prepare_visual_item(
                class_id=int(det.get("cls", 0)),
                cx=cx_val,
                cy=cy_val,
                cz=cz_val,
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
            vis["cx"] = cx_val
            vis["cy"] = cy_val
            vis["cz"] = cz_val
            color = normalize_color_label(det.get("color"))
            if color:
                vis["color"] = color
                hex_color = color_label_to_hex(color)
                if hex_color:
                    vis["color_hex"] = hex_color
            payload.append(vis)
        return payload

    def _serialize_fused(self, fused_list: List[dict]) -> List[dict]:
        payload = []
        for det in fused_list:
            cx_val = float(det.get("cx", 0.0))
            cy_val = float(det.get("cy", 0.0))
            cz_val = self._ground_height_at(cx_val, cy_val, default=float(det.get("cz", 0.0)))
            vis = prepare_visual_item(
                class_id=int(det.get("cls", 0)),
                cx=cx_val,
                cy=cy_val,
                cz=cz_val,
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
            vis["cx"] = cx_val
            vis["cy"] = cy_val
            vis["cz"] = cz_val
            color = normalize_color_label(det.get("color"))
            if color:
                vis["color"] = color
                hex_color = color_label_to_hex(color)
                if hex_color:
                    vis["color_hex"] = hex_color
            votes = det.get("color_votes")
            if votes:
                vis["color_votes"] = dict(votes)
            payload.append(vis)
        return payload

    def _tracks_to_dicts(self, tracks: np.ndarray) -> List[dict]:
        rows = tracks if (tracks is not None and len(tracks)) else []
        payload = []
        for row in rows:
            tid = int(row[0]); cls = int(row[1])
            cx, cy, L, W, yaw = map(float, row[2:7])
            extra = self.track_meta.get(tid, {})
            cz_val = self._ground_height_at(cx, cy, default=float(extra.get("cz", 0.0)))
            vis = prepare_visual_item(
                class_id=cls,
                cx=cx,
                cy=cy,
                cz=cz_val,
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
            vis["cz"] = cz_val
            color = normalize_color_label(extra.get("color"))
            if color:
                vis["color"] = color
                hex_color = color_label_to_hex(color)
                if hex_color:
                    vis["color_hex"] = hex_color
            if "color_confidence" in extra:
                vis["color_confidence"] = float(extra["color_confidence"])
            if "color_votes" in extra:
                vis["color_votes"] = dict(extra["color_votes"])
            payload.append(vis)
        return payload

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
    ap.add_argument("--cam-ports", default="cam1:50050,cam2:50051")
    ap.add_argument("--cam-positions-json", default="camera_position.json")
    ap.add_argument("--local-ply-dir", default="outputs",
                    help="카메라별 로컬 PLY를 탐색할 디렉토리(패턴: cam_<id>_*.ply)")
    ap.add_argument("--local-lut-dir", default="outputs",
                    help="카메라별 LUT(npz)를 탐색할 디렉토리(패턴: cam_<id>_*.npz)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--iou-thr", type=float, default=0.01)
    ap.add_argument("--roll-secs", type=int, default=60)
    ap.add_argument("--roll-max-rows", type=int, default=1000)
    ap.add_argument("--udp-port", type=int, default=50050) # main에서 받을 때
    ap.add_argument("--tx-host", default=None) # sort 결과 전송
    ap.add_argument("--tx-port", type=int, default=60050)
    ap.add_argument("--tx-protocol", choices=["udp","tcp"], default="udp")
    ap.add_argument("--carla-host", default=None)
    ap.add_argument("--carla-port", type=int, default=61000)
    ap.add_argument("--global-ply", default="pointcloud/real_coshow_map_small.ply")
    ap.add_argument("--vehicle-glb", default="pointcloud/car.glb")
    ap.add_argument("--web-host", default="0.0.0.0")
    ap.add_argument("--web-port", type=int, default=18000)
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--overlay-base-url", type=str, default=None,
                    help="Base URL for camera overlay/video API (typically Edge bridge http://host:port)")
    ap.add_argument("--tracker-fixed-length", type=float, default=None)
    ap.add_argument("--tracker-fixed-width", type=float, default=None)
    ap.add_argument("--size-mode", choices=["bbox","fixed","mesh"], default="mesh")
    ap.add_argument("--fixed-length", type=float, default=4.5)
    ap.add_argument("--fixed-width", type=float, default=1.8)
    ap.add_argument("--height-scale", type=float, default=0.5,
                    help="bbox/fixed 모드일 때 차량 높이 = width * height_scale")
    ap.add_argument("--mesh-scale", type=float, default=1.0,
                    help="size-mode=mesh 일 때 GLB에 곱할 유니폼 스케일")
    ap.add_argument("--mesh-height", type=float, default=0.0,
                    help="size-mode=mesh 일 때 지면 높이 계산용 높이(0이면 mesh-scale 사용)")
    ap.add_argument("--z-offset", type=float, default=0.0,
                    help="모든 박스에 추가할 z 오프셋")
    ap.add_argument("--invert-bev-y", dest="invert_bev_y", action="store_true")
    ap.add_argument("--no-invert-bev-y", dest="invert_bev_y", action="store_false")
    ap.set_defaults(no_invert_bev_y=True)
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
    ap.add_argument("--flip-marker-x", dest="flip_marker_x", action="store_true",
                    help="뷰어에서 카메라 버튼/로컬 클라우드의 X축을 반전")
    ap.add_argument("--no-flip-marker-x", dest="flip_marker_x", action="store_false")
    ap.set_defaults(flip_marker_x=False)
    ap.add_argument("--flip-marker-y", dest="flip_marker_y", action="store_true",
                    help="뷰어에서 카메라 버튼/로컬 클라우드의 Y축을 반전")
    ap.add_argument("--no-flip-marker-y", dest="flip_marker_y", action="store_false")
    ap.set_defaults(flip_marker_y=False)
    ap.add_argument("--cmd-host", default="0.0.0.0", help="yaw 명령 서버 바인드 호스트 (미지정 시 비활성화)")
    ap.add_argument("--cmd-port", type=int, default=18100, help="yaw 명령 서버 포트")

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
        "flipMarkerX": bool(args.flip_marker_x),
        "flipMarkerY": bool(args.flip_marker_y),
    }
    if args.overlay_base_url:
        client_config["overlayBaseUrl"] = args.overlay_base_url

    server = RealtimeFusionServer(
        cam_ports=cam_ports,
        cam_positions_path=args.cam_positions_json,
        local_ply_dir=args.local_ply_dir,
        local_lut_dir=args.local_lut_dir,
        fps=args.fps,
        iou_cluster_thr=args.iou_thr,
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
        command_host=args.cmd_host,
        command_port=args.cmd_port,
    )
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.receiver.stop()
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
