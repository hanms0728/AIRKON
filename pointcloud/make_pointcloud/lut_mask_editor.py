#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pixel2world_lut.npz valid_mask GUI editor.

- Loads an existing LUT (npz) that contains valid_mask (and optionally ground_valid_mask)
- Shows the mask overlaid on the original RGB image (if provided) or on a checkerboard
- Allows painting invalid pixels (left drag) / valid pixels (right drag) with adjustable brush
- Saves the edited mask back to a new npz (or in-place) so humans can manually curate visible areas
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - cv2 required at runtime
    raise SystemExit("OpenCV (cv2) is required for the LUT mask editor.") from exc


def _checkerboard_bg(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    block = max(12, min(h, w) // 32)
    ys = (np.arange(h) // block) % 2
    xs = (np.arange(w) // block) % 2
    pattern = (ys[:, None] ^ xs[None, :]) * 40 + 40  # 40 or 80
    bg = np.stack([pattern + 20, pattern + 10, pattern], axis=-1).astype(np.uint8)
    return bg


def _load_image_background(img_path: str, target_shape: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    if img.shape[:2] != target_shape:
        print(f"[Info] Background image shape {img.shape[:2]} != LUT mask {target_shape}. Resizing.")
        img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
    return img


def _compute_xy_bounds(X: np.ndarray, Y: np.ndarray, pad: float = 0.0) -> Optional[Tuple[float, float, float, float]]:
    finite = np.isfinite(X) & np.isfinite(Y)
    if not finite.any():
        return None
    xs = X[finite]
    ys = Y[finite]
    xmin = float(xs.min()) - pad
    xmax = float(xs.max()) + pad
    ymin = float(ys.min()) - pad
    ymax = float(ys.max()) + pad
    if xmax - xmin < 1e-6:
        xmax = xmin + 1.0
    if ymax - ymin < 1e-6:
        ymax = ymin + 1.0
    return xmin, xmax, ymin, ymax


def _render_bev_canvas_from_ply(ply_path: str, bounds: Tuple[float, float, float, float],
                                base_res: int, invert_y: bool, voxel: float) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("Open3D is required for BEV background rendering (--global-ply).") from exc

    cloud = o3d.io.read_point_cloud(ply_path)
    if cloud.is_empty():
        raise RuntimeError(f"PLY has no points: {ply_path}")
    if voxel > 1e-6:
        cloud = cloud.voxel_down_sample(voxel)
    pts = np.asarray(cloud.points)
    if pts.size == 0:
        raise RuntimeError(f"PLY has no points after voxel filtering: {ply_path}")
    cols = np.asarray(cloud.colors)
    if cols.shape != pts.shape:
        cols = np.ones_like(pts) * 0.7

    xmin, xmax, ymin, ymax = bounds
    rx = max(1e-6, xmax - xmin)
    ry = max(1e-6, ymax - ymin)
    aspect = ry / rx
    bev_w = max(32, int(base_res))
    bev_h = max(32, int(bev_w * aspect))
    mask = (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
    pts = pts[mask]
    cols = cols[mask]
    if pts.size == 0:
        raise RuntimeError("No PLY points overlap LUT bounds; adjust --bev-pad or check coordinates.")

    sx = (pts[:, 0] - xmin) / rx
    sy = (pts[:, 1] - ymin) / ry
    if invert_y:
        sy = 1.0 - sy
    px = np.clip((sx * (bev_w - 1)).astype(np.int32), 0, bev_w - 1)
    py = np.clip((sy * (bev_h - 1)).astype(np.int32), 0, bev_h - 1)
    flat = py * bev_w + px

    accum = np.zeros((bev_w * bev_h, 3), dtype=np.float64)
    counts = np.zeros((bev_w * bev_h,), dtype=np.float64)
    np.add.at(accum, flat, cols)
    np.add.at(counts, flat, 1.0)
    counts[counts == 0.0] = 1.0
    bev = (accum / counts[:, None]).reshape(bev_h, bev_w, 3)
    bev = (bev * 255.0).clip(0, 255).astype(np.uint8)
    return bev


def _sample_bev_background(X: np.ndarray, Y: np.ndarray,
                           bev_img: np.ndarray, bounds: Tuple[float, float, float, float],
                           invert_y: bool) -> np.ndarray:
    H, W = X.shape
    bg = _checkerboard_bg((H, W))
    if bev_img is None or bounds is None:
        return bg
    finite = np.isfinite(X) & np.isfinite(Y)
    if not finite.any():
        return bg

    xmin, xmax, ymin, ymax = bounds
    rx = max(1e-6, xmax - xmin)
    ry = max(1e-6, ymax - ymin)
    map_x = np.zeros((H, W), dtype=np.float32)
    map_y = np.zeros((H, W), dtype=np.float32)
    map_x[finite] = (X[finite] - xmin) / rx * (bev_img.shape[1] - 1)
    map_y[finite] = (Y[finite] - ymin) / ry
    if invert_y:
        map_y[finite] = 1.0 - map_y[finite]
    map_y[finite] *= (bev_img.shape[0] - 1)

    sampled = cv2.remap(bev_img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
    bg[finite] = sampled[finite]
    return bg


def _build_background(mask_shape: Tuple[int, int], args, X: np.ndarray, Y: np.ndarray):
    mode = args.background
    if mode == "auto":
        if args.global_ply:
            mode = "bev"
        elif args.image:
            mode = "image"
        else:
            mode = "checker"

    if mode == "image":
        if not args.image:
            raise ValueError("--image is required when --background image")
        return _load_image_background(args.image, mask_shape), None

    if mode == "bev":
        if not args.global_ply:
            raise ValueError("--global-ply is required for BEV background mode")
        bounds = _compute_xy_bounds(X, Y, pad=float(args.bev_pad))
        if bounds is None:
            raise RuntimeError("Cannot compute XY bounds from LUT (no finite X/Y).")
        bev_canvas = _render_bev_canvas_from_ply(
            args.global_ply,
            bounds,
            base_res=int(args.bev_resolution),
            invert_y=not args.no_bev_invert_y,
            voxel=float(args.ply_voxel))
        xmin, xmax, ymin, ymax = bounds
        rx = max(1e-6, xmax - xmin)
        ry = max(1e-6, ymax - ymin)
        bev_h, bev_w = bev_canvas.shape[:2]

        map_x = np.full(mask_shape, np.nan, dtype=np.float32)
        map_y = np.full(mask_shape, np.nan, dtype=np.float32)
        finite = np.isfinite(X) & np.isfinite(Y)
        map_x[finite] = (X[finite] - xmin) / rx * (bev_w - 1)
        map_y[finite] = (Y[finite] - ymin) / ry
        if not args.no_bev_invert_y:
            map_y[finite] = 1.0 - map_y[finite]
        map_y[finite] *= (bev_h - 1)
        bev_valid = finite & np.isfinite(map_x) & np.isfinite(map_y)

        px = np.rint(map_x).astype(np.int32)
        py = np.rint(map_y).astype(np.int32)
        inside = bev_valid & (px >= 0) & (px < bev_w) & (py >= 0) & (py < bev_h)

        pix_idx = np.full(X.size, -1, dtype=np.int64)
        pix_idx[inside.reshape(-1)] = (py[inside] * bev_w + px[inside]).reshape(-1)

        bev_meta = {
            "mode": "bev",
            "map_x": map_x,
            "map_y": map_y,
            "valid": bev_valid,
            "pix_idx": pix_idx,
            "canvas_shape": bev_canvas.shape[:2],
            "bounds": bounds,
            "invert_y": not args.no_bev_invert_y,
            "width": bev_w,
            "height": bev_h,
        }
        return bev_canvas, bev_meta

    # checker fallback
    return _checkerboard_bg(mask_shape), None


class MaskEditor:
    def __init__(self, mask: np.ndarray, background: np.ndarray,
                 window: str = "LUT Mask Editor", brush_radius: int = 25, max_history: int = 48,
                 bev_meta: Optional[dict] = None):
        self.window = window
        self.mask = mask.astype(bool).copy()
        self.original_mask = self.mask.copy()
        self.last_saved_mask = self.mask.copy()
        self.background = background.copy()
        self.brush = max(1, int(brush_radius))
        self.max_history = max(1, int(max_history))
        self.history = []
        self.drawing = False
        self.draw_value = False
        self.cursor = None
        self.cursor_mode = None
        self.message = ""
        self.message_until = 0.0
        self.display_mode = "bev" if bev_meta is not None else "image"
        self._bev_meta = bev_meta
        if self.display_mode == "bev":
            if bev_meta is None:
                raise ValueError("BEV metadata required for BEV mode")
            self._bev_map_x = bev_meta["map_x"].astype(np.float32)
            self._bev_map_y = bev_meta["map_y"].astype(np.float32)
            self._bev_valid = bev_meta["valid"].astype(bool)
            self._bev_pix_idx = bev_meta["pix_idx"].astype(np.int64)
            self._bev_map_x_flat = self._bev_map_x.reshape(-1)
            self._bev_map_y_flat = self._bev_map_y.reshape(-1)
            self._bev_valid_flat = self._bev_valid.reshape(-1)
            self._bev_pix_idx_flat = self._bev_pix_idx.reshape(-1)
            self._bev_canvas_shape = bev_meta["canvas_shape"]
            self._bev_width = bev_meta["width"]
            self._bev_height = bev_meta["height"]
        else:
            self._bev_map_x = self._bev_map_y = None
            self._bev_valid = None
            self._bev_pix_idx = None

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        display_w = min(1600, background.shape[1])
        display_h = int(display_w * background.shape[0] / background.shape[1])
        cv2.resizeWindow(self.window, display_w, display_h)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _set_message(self, text: str, duration: float = 2.0) -> None:
        self.message = text
        self.message_until = time.time() + duration

    def _begin_draw(self, value: bool) -> None:
        if not self.drawing:
            self.history.append(self.mask.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
        self.drawing = True
        self.draw_value = bool(value)

    def _apply_brush(self, x: int, y: int) -> None:
        if self.display_mode == "bev":
            self._apply_brush_bev(x, y)
            return
        h, w = self.mask.shape
        r = self.brush
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        patch = self.mask[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        patch[circle] = self.draw_value
        self.mask[y0:y1, x0:x1] = patch

    def _apply_brush_bev(self, x: int, y: int) -> None:
        if self._bev_map_x_flat is None or self._bev_map_y_flat is None:
            return
        r2 = float(self.brush * self.brush)
        bx = float(x)
        by = float(y)
        dx = self._bev_map_x_flat - bx
        dy = self._bev_map_y_flat - by
        dist2 = dx * dx + dy * dy
        hits = self._bev_valid_flat & (dist2 <= r2)
        if not np.any(hits):
            return
        mask_flat = self.mask.reshape(-1)
        mask_flat[hits] = self.draw_value
        self.mask[:] = mask_flat.reshape(self.mask.shape)

    def _on_mouse(self, event, x, y, flags, _param):
        self.cursor = (int(x), int(y))
        if flags & cv2.EVENT_FLAG_RBUTTON:
            self.cursor_mode = "valid"
        elif flags & cv2.EVENT_FLAG_LBUTTON:
            self.cursor_mode = "invalid"
        else:
            self.cursor_mode = None

        if event == cv2.EVENT_LBUTTONDOWN:
            self._begin_draw(False)
            self._apply_brush(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._begin_draw(True)
            self._apply_brush(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self._apply_brush(x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drawing = False

    def undo(self) -> None:
        if self.history:
            self.mask = self.history.pop()
            self._set_message("Undo")

    def reset(self) -> None:
        self.mask = self.original_mask.copy()
        self.history.clear()
        self._set_message("Reset to original")

    def is_dirty(self) -> bool:
        return not np.array_equal(self.mask, self.last_saved_mask)

    def mark_saved(self) -> None:
        self.last_saved_mask = self.mask.copy()
        self._set_message("Saved")

    def adjust_brush(self, delta: int) -> None:
        self.brush = int(np.clip(self.brush + delta, 1, 512))
        self._set_message(f"Brush: {self.brush}px", duration=1.2)

    def render(self) -> np.ndarray:
        if self.display_mode == "bev":
            frame = self._render_bev_frame()
        else:
            frame = self._render_image_frame()

        brush_color = (0, 255, 0) if self.cursor_mode == "valid" else (0, 0, 255)
        if self.cursor is not None:
            cv2.circle(frame, self.cursor, self.brush, brush_color, 1, cv2.LINE_AA)

        hud_base = "(BEV)" if self.display_mode == "bev" else "(IMG)"
        hud_1 = f"{hud_base} {'*' if self.is_dirty() else ' '} valid: {int(self.mask.sum())}/{self.mask.size}  brush={self.brush}px"
        hud_2 = "Left drag=invalid  Right drag=valid  [/] brush  U undo  R reset  S save  ESC quit"
        cv2.putText(frame, hud_1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(frame, hud_1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, hud_2, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(frame, hud_2, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        if time.time() < self.message_until:
            cv2.putText(frame, self.message, (12, frame.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, self.message, (12, frame.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def _render_image_frame(self) -> np.ndarray:
        frame = self.background.copy()
        valid = self.mask
        invalid = ~valid
        if valid.any():
            frame[valid] = (frame[valid] * 0.35 + np.array([60, 190, 90], dtype=np.float32)).astype(np.uint8)
        if invalid.any():
            frame[invalid] = (frame[invalid] * 0.35 + np.array([30, 30, 180], dtype=np.float32)).astype(np.uint8)
        changed = valid ^ self.original_mask
        if changed.any():
            frame[changed] = (frame[changed] * 0.2 + np.array([0, 220, 255], dtype=np.float32)).astype(np.uint8)
        return frame

    def _render_bev_frame(self) -> np.ndarray:
        frame = self.background.copy()
        pix_idx = self._bev_pix_idx_flat
        if pix_idx is None:
            return frame

        mask_flat = self.mask.reshape(-1)
        orig_flat = self.original_mask.reshape(-1)
        valid_map = pix_idx >= 0
        if not np.any(valid_map):
            return frame

        flat = frame.reshape(-1, 3).astype(np.float32)

        def overlay(idx_mask, color_bgr, alpha):
            if not np.any(idx_mask):
                return
            idx = pix_idx[idx_mask]
            base = flat[idx]
            tint = np.array(color_bgr, dtype=np.float32)
            flat[idx] = base * (1.0 - alpha) + tint * alpha

        invalid_mask = valid_map & (~mask_flat)
        valid_mask = valid_map & mask_flat
        changed_mask = valid_map & (mask_flat ^ orig_flat)

        overlay(valid_mask, (70, 200, 120), 0.25)
        overlay(invalid_mask, (40, 40, 200), 0.55)
        overlay(changed_mask, (0, 230, 255), 0.65)

        frame[:] = flat.reshape(frame.shape).clip(0, 255).astype(np.uint8)
        return frame


def save_lut_with_mask(lut_path: str, mask: np.ndarray, out_path: str,
                       sync_ground: bool = True, sync_floor: bool = False) -> None:
    lut = dict(np.load(lut_path, allow_pickle=False))
    mask_u8 = mask.astype(np.uint8)
    lut["valid_mask"] = mask_u8
    if sync_ground and "ground_valid_mask" in lut:
        lut["ground_valid_mask"] = mask_u8
    if sync_floor and "floor_mask" in lut:
        lut["floor_mask"] = mask_u8
    np.savez_compressed(out_path, **lut)
    print(f"[SAVE] Updated LUT: {out_path}")


def parse_args():
    ap = argparse.ArgumentParser("GUI editor for pixel2world_lut valid_mask")
    ap.add_argument("--lut", required=True, help="pixel2world_lut.npz path to edit")
    ap.add_argument("--image", help="Optional RGB image to show under the mask (auto resized to LUT size)")
    ap.add_argument("--global-ply", help="Global PLY to build a BEV-style background (no RGB image needed)")
    ap.add_argument("--background", choices=["auto", "image", "bev", "checker"], default="auto",
                    help="Choose what to display under the mask (default auto: prefer PLY BEV, otherwise image, else checker)")
    ap.add_argument("--bev-resolution", type=int, default=1600,
                    help="Base resolution (width) of the intermediate BEV canvas (default: 1600)")
    ap.add_argument("--bev-pad", type=float, default=0.5,
                    help="Extra XY margin (meters) added around LUT bounds when projecting PLY")
    ap.add_argument("--ply-voxel", type=float, default=0.0,
                    help="Optional voxel size (meters) to downsample the PLY before rendering the BEV background")
    ap.add_argument("--no-bev-invert-y", action="store_true",
                    help="Do not flip the BEV Y-axis (default flips so increasing Y goes downward for image alignment)")
    ap.add_argument("--out", help="Output npz path (default: <lut>_edited.npz)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite the input LUT instead of creating *_edited.npz")
    ap.add_argument("--brush", type=int, default=25, help="Initial brush radius in pixels (default: 25)")
    ap.add_argument("--max-undo", type=int, default=48, help="Undo history size (default: 48)")
    ap.add_argument("--no-sync-ground", dest="sync_ground", action="store_false",
                    help="Do not copy the edited mask to ground_valid_mask")
    ap.add_argument("--sync-ground", dest="sync_ground", action="store_true",
                    help="Force syncing ground_valid_mask with valid_mask (default)")
    ap.set_defaults(sync_ground=True)
    ap.add_argument("--sync-floor-mask", dest="sync_floor", action="store_true",
                    help="Also copy the edited mask to floor_mask")
    return ap.parse_args()


def main():
    args = parse_args()
    lut_path = args.lut
    if not os.path.isfile(lut_path):
        raise FileNotFoundError(f"LUT file not found: {lut_path}")

    lut_data = np.load(lut_path, allow_pickle=False)
    if "valid_mask" not in lut_data:
        raise KeyError(f"'valid_mask' not present in {lut_path}")
    if "X" not in lut_data or "Y" not in lut_data:
        raise KeyError(f"'X'/'Y' maps are required in LUT for BEV rendering: {lut_path}")
    mask = lut_data["valid_mask"].astype(bool)
    X = np.asarray(lut_data["X"], dtype=np.float32)
    Y = np.asarray(lut_data["Y"], dtype=np.float32)
    background, bev_meta = _build_background(mask.shape, args, X, Y)
    editor = MaskEditor(mask, background, brush_radius=args.brush, max_history=args.max_undo, bev_meta=bev_meta)

    out_path: Optional[str]
    if args.out:
        out_path = args.out
    elif args.inplace:
        out_path = lut_path
    else:
        stem = Path(lut_path)
        out_path = str(stem.with_name(stem.stem + "_edited.npz"))

    print("[Info] Controls: Left drag=mask out, Right drag=mask in, S=save, ESC=quit")
    saved_once = False
    try:
        while True:
            frame = editor.render()
            cv2.imshow(editor.window, frame)
            key = cv2.waitKey(15) & 0xFF
            if key == 27 or key == ord('q'):
                break
            if key == ord('s'):
                save_lut_with_mask(lut_path, editor.mask, out_path,
                                   sync_ground=args.sync_ground, sync_floor=args.sync_floor)
                editor.mark_saved()
                saved_once = True
            elif key == ord('u'):
                editor.undo()
            elif key == ord('r'):
                editor.reset()
            elif key == ord('['):
                editor.adjust_brush(-5)
            elif key == ord(']'):
                editor.adjust_brush(+5)
    finally:
        cv2.destroyAllWindows()

    if editor.is_dirty() and not saved_once:
        print("[Warn] Changes discarded (no save).")


if __name__ == "__main__":
    main()
