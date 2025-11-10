from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any

import numpy as np


@dataclass
class VizSizeConfig:
    size_mode: str = "mesh"          # "mesh" | "bbox" | "fixed"
    fixed_length: float = 4.5
    fixed_width: float = 1.8
    height_scale: float = 0.5        # height = width * height_scale when bbox/fixed
    mesh_scale: float = 1.0          # uniform scale when size_mode == "mesh"
    mesh_height: float = 0.0         # optional explicit height for mesh ground offset
    z_offset: float = 0.0            # additional world-Z offset for center placement
    invert_bev_y: bool = True
    normalize_vehicle: bool = True   # client hint
    vehicle_y_up: bool = True        # client hint (True => rotate +90° around X)

    def as_client_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_unit_to_world_T(
    length: float,
    width: float,
    yaw_deg: float,
    center_xyz: np.ndarray,
    *,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    up_scale_from_width: float = 0.5,
    scale_override: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """Construct 4x4 transform from unit mesh space to world coordinates."""
    if scale_override is not None:
        sx = max(1e-4, float(scale_override[0]))
        sy = max(1e-4, float(scale_override[1]))
        sz = max(1e-4, float(scale_override[2]))
    else:
        sx = max(1e-4, float(length))
        sy = max(1e-4, float(width))
        sz = max(1e-4, float(width) * float(up_scale_from_width))

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


def _compute_length_width(length: float, width: float, cfg: VizSizeConfig) -> Tuple[float, float]:
    length_use = max(1e-4, float(length))
    width_use = max(1e-4, float(width))
    if cfg.size_mode == "fixed":
        length_use = max(1e-4, float(cfg.fixed_length))
        width_use = max(1e-4, float(cfg.fixed_width))
    return length_use, width_use


def _compute_scale_and_height(width_use: float, cfg: VizSizeConfig) -> Tuple[float, Tuple[float, float, float]]:
    if cfg.size_mode == "mesh":
        height = float(cfg.mesh_height) if cfg.mesh_height > 0 else float(cfg.mesh_scale)
        scale_override = (float(cfg.mesh_scale), float(cfg.mesh_scale), float(cfg.mesh_scale))
    else:
        height = width_use * float(cfg.height_scale)
        scale_override = None
    return float(height), scale_override


def prepare_visual_item(
    *,
    class_id: int,
    cx: float,
    cy: float,
    cz: float,
    length: float,
    width: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    cfg: VizSizeConfig,
    score: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a visualization dictionary entry matching realtime_show_result/server.py output."""
    cx = float(cx)
    cy = float(cy)
    cz = float(cz)
    yaw_deg = float(yaw_deg)
    pitch_deg = float(pitch_deg)
    roll_deg = float(roll_deg)

    if cfg.invert_bev_y:
        cy = -cy
        yaw_deg = -yaw_deg
        pitch_deg = -pitch_deg
        roll_deg = -roll_deg

    length_use, width_use = _compute_length_width(length, width, cfg)
    height, scale_override = _compute_scale_and_height(width_use, cfg)

    height_for_center = height if cfg.size_mode == "mesh" else width_use * float(cfg.height_scale)
    center_world = np.array(
        [
            cx,
            cy,
            cz + float(cfg.z_offset) + 0.5 * max(0.0, height_for_center),
        ],
        dtype=np.float64,
    )

    T = build_unit_to_world_T(
        length_use,
        width_use,
        yaw_deg,
        center_world,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        up_scale_from_width=float(cfg.height_scale),
        scale_override=scale_override,
    )
    transform_col_major = T.T.reshape(-1).tolist()

    if scale_override is not None:
        scale_vec = [float(scale_override[0]), float(scale_override[1]), float(scale_override[2])]
    else:
        scale_vec = [float(length_use), float(width_use), float(height)]

    item = {
        "class_id": int(class_id),
        "length": float(length_use),
        "width": float(width_use),
        "height": float(height),
        "center": center_world.tolist(),
        "yaw_deg": float(yaw_deg),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(roll_deg),
        "transform": transform_col_major,
        "scale": scale_vec,
        "cz": float(cz),
    }
    if score is not None:
        item["score"] = float(score)
    return item
