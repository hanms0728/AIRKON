#!/usr/bin/env python3
"""
Local web service that exposes point cloud + inference overlays for browser rendering.

Run:
    python realtime_show_result/server.py \
        --global-ply pointcloud/cloud_rgb_ply/cloud_rgb_9.ply \
        --bev-label-dir dataset_exmple_pointcloud_9/bev_labels \
        --vehicle-glb pointcloud/car.glb \
        --host 0.0.0.0 --port 8000
"""

import argparse
import glob
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# ---------------------- 데이터 모델 ----------------------
@dataclass
class Settings:
    global_ply: str
    bev_label_dir: str
    vehicle_glb: str
    host: str = "127.0.0.1"
    port: int = 8000
    fps: float = 10.0
    height_scale: float = 0.5
    z_offset: float = 0.0
    invert_bev_y: bool = False
    size_mode: str = "bbox"
    fixed_length: float = 4.5
    fixed_width: float = 1.8
    mesh_scale: float = 1.0
    mesh_height: float = 1.5
    normalize_vehicle: bool = True
    vehicle_y_up: bool = True
    flip_ply_y: bool = False
    show_debug_marker: bool = False
    show_scene_axes: bool = False


class RuntimeData:
    """Preloads labels and serves per-frame overlay metadata."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.frames = load_labels_dir(cfg.bev_label_dir)
        self.total_frames = len(self.frames)

    def _prepare_detection_dict(self, row: np.ndarray) -> Dict[str, object]:
        # Input row is either 9 (new format) or 6 (legacy) columns (already normalized)
        cls_id = int(row[0])
        cx = float(row[1])
        cy = float(row[2])
        cz_label = float(row[3])
        length = float(row[4])
        width = float(row[5])
        yaw_deg = float(row[6])
        pitch_deg = float(row[7]) if row.shape[0] > 7 else 0.0
        roll_deg = float(row[8]) if row.shape[0] > 8 else 0.0

        if self.cfg.invert_bev_y:
            cy = -cy
            yaw_deg = -yaw_deg
            pitch_deg = -pitch_deg
            roll_deg = -roll_deg

        scale_override: Optional[Tuple[float, float, float]] = None

        if self.cfg.size_mode == "fixed":
            length_use = float(self.cfg.fixed_length)
            width_use = float(self.cfg.fixed_width)
        else:
            length_use = max(1e-4, length)
            width_use = max(1e-4, width)

        if self.cfg.size_mode == "mesh":
            height = float(self.cfg.mesh_height)
            scale_override = (
                float(self.cfg.mesh_scale),
                float(self.cfg.mesh_scale),
                float(self.cfg.mesh_scale),
            )
        else:
            height = width_use * float(self.cfg.height_scale)

        if scale_override is not None:
            scale_vec = [
                float(scale_override[0]),
                float(scale_override[1]),
                float(scale_override[2]),
            ]
        else:
            scale_vec = [
                float(length_use),
                float(width_use),
                float(height),
            ]

        center_world = np.array(
            [
                cx,
                cy,
                cz_label + float(self.cfg.z_offset) + height * 0.5,
            ],
            dtype=np.float64,
        )

        T = build_unit_to_world_T(
            length_use,
            width_use,
            yaw_deg,
            center_world,
            pitch_deg=float(pitch_deg),
            roll_deg=float(roll_deg),
            up_scale_from_width=float(self.cfg.height_scale),
            scale_override=scale_override,
        )

        transform_col_major = T.T.reshape(-1).tolist()

        return {
            "class_id": cls_id,
            "length": length_use,
            "width": width_use,
            "height": height,
            "center": center_world.tolist(),
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "transform": transform_col_major,
            "scale": scale_vec,
        }

    def get_frame(self, index: int) -> Dict[str, object]:
        if index < 0 or index >= self.total_frames:
            raise IndexError(f"Frame index {index} out of range (total {self.total_frames})")

        frame_path, arr = self.frames[index]
        arr = arr.astype(np.float32)
        detections: List[Dict[str, object]] = []

        if arr.size > 0:
            for row in arr:
                detections.append(self._prepare_detection_dict(row))

        return {
            "index": index,
            "label_file": os.path.relpath(frame_path, start=self.cfg.bev_label_dir),
            "detections": detections,
        }


# ---------------------- 유틸 ----------------------
def load_labels_dir(label_dir: str) -> List[Tuple[str, np.ndarray]]:
    """Load sorted *.txt label files. Supports legacy (6 cols) and new (9 cols)."""
    files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    frames: List[Tuple[str, np.ndarray]] = []
    for f in files:
        try:
            if os.path.getsize(f) == 0:
                frames.append((f, np.zeros((0, 9), dtype=np.float32)))
                continue
            arr = np.loadtxt(f, ndmin=2)
            if arr.size == 0:
                frames.append((f, np.zeros((0, 9), dtype=np.float32)))
                continue
            if arr.shape[1] < 6:
                frames.append((f, np.zeros((0, 9), dtype=np.float32)))
                continue
            arr = arr.astype(np.float32)
            if arr.shape[1] >= 9:
                arr = arr[:, :9]
            else:
                cls_cx_cy = arr[:, :3]
                L = arr[:, 3:4]
                W = arr[:, 4:5]
                yaw = arr[:, 5:6]
                zeros = np.zeros((arr.shape[0], 3), dtype=np.float32)
                arr = np.concatenate([cls_cx_cy, zeros[:, :1], L, W, yaw, zeros[:, 1:]], axis=1)
            frames.append((f, arr))
        except Exception:
            frames.append((f, np.zeros((0, 9), dtype=np.float32)))
    return frames


def build_unit_to_world_T(
    length: float,
    width: float,
    yaw_deg: float,
    center_xyz: np.ndarray,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    up_scale_from_width: float = 0.5,
    scale_override: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """Constructs 4x4 transform from unit mesh space to world coordinates."""
    if scale_override is not None:
        sx = max(1e-4, float(scale_override[0]))
        sy = max(1e-4, float(scale_override[1]))
        sz = max(1e-4, float(scale_override[2]))
    else:
        sx = max(1e-4, float(length))
        sy = max(1e-4, float(width))
        sz = max(1e-4, float(width) * up_scale_from_width)

    S = np.diag([sx, sy, sz, 1.0]).astype(np.float64)

    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))

    cz = math.cos(yaw)
    szn = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cr = math.cos(roll)
    sr = math.sin(roll)

    Rz = np.array(
        [[cz, -szn, 0.0, 0.0], [szn, cz, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    Ry = np.array(
        [[cp, 0.0, sp, 0.0], [0.0, 1.0, 0.0, 0.0], [-sp, 0.0, cp, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    Rx = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, cr, -sr, 0.0], [0.0, sr, cr, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    R = Rz @ (Ry @ Rx)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R[:3, :3]
    T = T @ S
    T[:3, 3] = center_xyz[:3]
    return T


def validate_settings(cfg: Settings) -> None:
    if not os.path.isfile(cfg.global_ply):
        raise FileNotFoundError(f"Global PLY not found: {cfg.global_ply}")
    if not os.path.isdir(cfg.bev_label_dir):
        raise FileNotFoundError(f"BEV label dir not found: {cfg.bev_label_dir}")
    if not os.path.isfile(cfg.vehicle_glb):
        raise FileNotFoundError(f"Vehicle GLB not found: {cfg.vehicle_glb}")
    if cfg.size_mode not in {"bbox", "fixed", "mesh"}:
        raise ValueError(f"Unsupported size_mode: {cfg.size_mode}")


def create_app(cfg: Settings) -> FastAPI:
    validate_settings(cfg)
    runtime = RuntimeData(cfg)

    app = FastAPI(title="AIRKON Real-time Result Viewer", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.runtime = runtime
    app.state.cfg = cfg

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        def index() -> FileResponse:
            index_path = os.path.join(static_dir, "index.html")
            if not os.path.isfile(index_path):
                raise HTTPException(status_code=404, detail="index.html not found in static directory")
            return FileResponse(index_path, media_type="text/html")

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    def config() -> Dict[str, object]:
        return {
            "totalFrames": runtime.total_frames,
            "fps": cfg.fps,
            "heightScale": cfg.height_scale,
            "invertBevY": cfg.invert_bev_y,
            "sizeMode": cfg.size_mode,
            "normalizeVehicle": cfg.normalize_vehicle,
            "vehicleYAxisUp": cfg.vehicle_y_up,
            "meshScale": cfg.mesh_scale,
            "meshHeight": cfg.mesh_height,
            "flipPlyY": cfg.flip_ply_y,
            "showDebugMarker": cfg.show_debug_marker,
            "showSceneAxes": cfg.show_scene_axes,
        }

    @app.get("/api/frames/{frame_index}")
    def get_frame(frame_index: int) -> Dict[str, object]:
        try:
            return runtime.get_frame(frame_index)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/assets/global.ply")
    def get_global_ply() -> FileResponse:
        return FileResponse(cfg.global_ply, media_type="application/octet-stream")

    @app.get("/assets/vehicle.glb")
    def get_vehicle_glb() -> FileResponse:
        return FileResponse(cfg.vehicle_glb, media_type="model/gltf-binary")

    return app


def parse_args(args: Optional[List[str]] = None) -> Settings:
    parser = argparse.ArgumentParser(description="Launch web service for 3D overlay viewer.")
    parser.add_argument("--global-ply", required=True, help="Path to global point cloud PLY file")
    parser.add_argument("--bev-label-dir", required=True, help="Directory containing BEV label txt files")
    parser.add_argument("--vehicle-glb", required=True, help="Path to vehicle GLB mesh file")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--height-scale", type=float, default=0.5, help="Vehicle height = width * height_scale")
    parser.add_argument("--z-offset", type=float, default=0.0, help="Offset added to cz label for ground alignment")
    parser.add_argument("--invert-bev-y", action="store_true", help="Invert BEV Y axis and yaw/pitch/roll sign")
    parser.add_argument(
        "--size-mode",
        choices=["bbox", "fixed", "mesh"],
        default="bbox",
        help="Use bbox dimensions or fixed fallback size for the vehicle mesh",
    )
    parser.add_argument("--fixed-length", type=float, default=4.5, help="Fixed vehicle length when size-mode=fixed")
    parser.add_argument("--fixed-width", type=float, default=1.8, help="Fixed vehicle width when size-mode=fixed")
    parser.add_argument("--mesh-scale", type=float, default=1.0, help="Uniform scale applied when size-mode=mesh")
    parser.add_argument(
        "--mesh-height",
        type=float,
        default=1.5,
        help="Estimated vehicle height when size-mode=mesh (used for ground offset)",
    )
    parser.add_argument(
        "--normalize-vehicle",
        dest="normalize_vehicle",
        action="store_true",
        help="Force normalization of vehicle mesh to unit size on the frontend",
    )
    parser.add_argument(
        "--no-normalize-vehicle",
        dest="normalize_vehicle",
        action="store_false",
        help="Disable vehicle mesh normalization on the frontend",
    )
    parser.set_defaults(normalize_vehicle=None)
    parser.add_argument(
        "--vehicle-z-up",
        dest="vehicle_y_up",
        action="store_false",
        help="Indicate vehicle GLB already uses Z-up (skip +90deg X rotation)",
    )
    parser.add_argument(
        "--vehicle-y-up",
        dest="vehicle_y_up",
        action="store_true",
        help="Indicate vehicle GLB uses Y-up (default: apply +90deg X rotation)",
    )
    parser.set_defaults(vehicle_y_up=True)
    parser.add_argument(
        "--flip-ply-y",
        dest="flip_ply_y",
        action="store_true",
        help="Flip global PLY's Y axis before visualization",
    )
    parser.add_argument(
        "--no-flip-ply-y",
        dest="flip_ply_y",
        action="store_false",
        help="Do not flip global PLY's Y axis (default)",
    )
    parser.set_defaults(flip_ply_y=False)
    parser.add_argument(
        "--show-debug-marker",
        dest="show_debug_marker",
        action="store_true",
        help="Show debug marker sphere at first vehicle position",
    )
    parser.add_argument(
        "--show-scene-axes",
        dest="show_scene_axes",
        action="store_true",
        help="Show grid and axis helpers in the scene",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Playback FPS hint (used by frontend only)")

    parsed = parser.parse_args(args=args)

    if parsed.normalize_vehicle is None:
        normalize_vehicle = parsed.size_mode != "mesh"
    else:
        normalize_vehicle = parsed.normalize_vehicle

    return Settings(
        global_ply=os.path.abspath(parsed.global_ply),
        bev_label_dir=os.path.abspath(parsed.bev_label_dir),
        vehicle_glb=os.path.abspath(parsed.vehicle_glb),
        host=parsed.host,
        port=parsed.port,
        height_scale=parsed.height_scale,
        z_offset=parsed.z_offset,
        invert_bev_y=parsed.invert_bev_y,
        size_mode=parsed.size_mode,
        fixed_length=parsed.fixed_length,
        fixed_width=parsed.fixed_width,
        mesh_scale=parsed.mesh_scale,
        mesh_height=parsed.mesh_height,
        normalize_vehicle=normalize_vehicle,
        vehicle_y_up=parsed.vehicle_y_up,
        flip_ply_y=parsed.flip_ply_y,
        show_debug_marker=parsed.show_debug_marker,
        show_scene_axes=parsed.show_scene_axes,
        fps=parsed.fps,
    )


def run_with_settings(cfg: Settings) -> None:
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


# ---------------------- 엔트리포인트 ----------------------
def load_settings_from_env() -> Optional[Settings]:
    """Allow uvicorn imports with environment variables."""
    global_ply = os.getenv("AIRKON_GLOBAL_PLY")
    bev_dir = os.getenv("AIRKON_BEV_LABEL_DIR")
    vehicle_glb = os.getenv("AIRKON_VEHICLE_GLB")
    if not (global_ply and bev_dir and vehicle_glb):
        return None

    def _env_float(name: str, default: float) -> float:
        value = os.getenv(name)
        try:
            return float(value) if value is not None else default
        except ValueError:
            return default

    size_mode = os.getenv("AIRKON_SIZE_MODE", "bbox")
    invert_flag = os.getenv("AIRKON_INVERT_BEV_Y", "0").lower() in {"1", "true", "yes"}
    mesh_scale = _env_float("AIRKON_MESH_SCALE", 1.0)
    mesh_height = _env_float("AIRKON_MESH_HEIGHT", 1.5)
    normalize_env = os.getenv("AIRKON_NORMALIZE_VEHICLE")
    vehicle_y_up_flag = os.getenv("AIRKON_VEHICLE_Y_UP", "1").lower() in {"1", "true", "yes"}
    flip_ply_y_flag = os.getenv("AIRKON_FLIP_PLY_Y", "0").lower() in {"1", "true", "yes"}
    debug_marker_flag = os.getenv("AIRKON_SHOW_DEBUG_MARKER", "0").lower() in {"1", "true", "yes"}
    show_axes_flag = os.getenv("AIRKON_SHOW_SCENE_AXES", "0").lower() in {"1", "true", "yes"}

    if normalize_env is None:
        normalize_vehicle = size_mode != "mesh"
    else:
        normalize_vehicle = normalize_env.lower() in {"1", "true", "yes"}

    return Settings(
        global_ply=os.path.abspath(global_ply),
        bev_label_dir=os.path.abspath(bev_dir),
        vehicle_glb=os.path.abspath(vehicle_glb),
        host=os.getenv("AIRKON_HOST", "127.0.0.1"),
        port=int(os.getenv("AIRKON_PORT", "8000")),
        height_scale=_env_float("AIRKON_HEIGHT_SCALE", 0.5),
        z_offset=_env_float("AIRKON_Z_OFFSET", 0.0),
        invert_bev_y=invert_flag,
        size_mode=size_mode,
        fixed_length=_env_float("AIRKON_FIXED_LENGTH", 4.5),
        fixed_width=_env_float("AIRKON_FIXED_WIDTH", 1.8),
        mesh_scale=mesh_scale,
        mesh_height=mesh_height,
        normalize_vehicle=normalize_vehicle,
        vehicle_y_up=vehicle_y_up_flag,
        flip_ply_y=flip_ply_y_flag,
        show_debug_marker=debug_marker_flag,
        show_scene_axes=show_axes_flag,
        fps=_env_float("AIRKON_FPS", 10.0),
    )


ENV_SETTINGS = load_settings_from_env()
if ENV_SETTINGS:
    app = create_app(ENV_SETTINGS)
else:
    # Placeholder application when imported without configuration
    app = FastAPI(title="AIRKON Real-time Result Viewer")

    @app.get("/api/config")
    def missing_config() -> Dict[str, str]:
        raise HTTPException(
            status_code=500,
            detail="Server not configured. Run 'python realtime_show_result/server.py --help' "
            "or set AIRKON_* environment variables.",
        )


if __name__ == "__main__":
    settings = parse_args()
    run_with_settings(settings)

"""
python realtime_show_result/server.py   --global-ply pointcloud/cloud_rgb_ply/cloud_rgb_9.ply   --bev-label-dir dataset_exmple_pointcloud_9/bev_labels   --vehicle-glb pointcloud/car.glb   --host 0.0.0.0 --port 8000   --size-mode mesh   --mesh-scale 1.0   --mesh-height 0   --flip-ply-y   --invert-bev-y
"""
