#!/usr/bin/env python3

import argparse
import json
import math
import os
import socket
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import depthai as dai
import numpy as np

from src.evaluation_utils import decode_predictions
from src.geometry_utils import parallelogram_from_triangle, tiny_filter_on_dets
from src.inference_lstm_onnx_pointcloud_tensorrt import TensorRTTemporalRunner


def preprocess_frame(frame_bgr: np.ndarray, target_hw: Tuple[int, int]) -> dict:
    target_h, target_w = target_hw
    resized_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_np = np.expand_dims(img_np, 0).copy()

    H0, W0 = frame_bgr.shape[:2]
    scale_to_orig_x = float(W0) / float(target_w)
    scale_to_orig_y = float(H0) / float(target_h)

    return {
        "img_np": img_np,
        "resized_bgr": resized_bgr,
        "orig_bgr": frame_bgr,
        "scale_to_orig_x": scale_to_orig_x,
        "scale_to_orig_y": scale_to_orig_y,
    }


def overlay_detections(frame_bgr: np.ndarray,
                       dets,
                       scale_to_orig_x: float,
                       scale_to_orig_y: float) -> np.ndarray:
    vis = frame_bgr.copy()
    for det in dets:
        tri = np.asarray(det["tri"], dtype=np.float32).copy()
        tri[:, 0] *= scale_to_orig_x
        tri[:, 1] *= scale_to_orig_y
        poly = parallelogram_from_triangle(tri[0], tri[1], tri[2]).astype(np.int32)
        cv2.polylines(vis, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        cx, cy = int(tri[0][0]), int(tri[0][1])
        cv2.putText(vis, f"{det['score']:.2f}", (cx, max(0, cy - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def draw_pred_pseudo3d(
    image_bgr: np.ndarray,
    tris_orig: List[np.ndarray],
    *,
    dy: Optional[int] = None,
    height_scale: float = 0.5,
    min_dy: int = 8,
    max_dy: int = 80,
) -> Optional[np.ndarray]:
    """
    원본 이미지 좌표계의 삼각형(tris_orig)을 이용해 pseudo-3D 박스를 생성한다.
    """
    if not tris_orig:
        return None

    img = image_bgr.copy()
    H, W = image_bgr.shape[:2]

    polys: List[np.ndarray] = []
    for tri in tris_orig:
        tri = np.asarray(tri, dtype=np.float32)
        if tri.shape != (3, 2) or not np.all(np.isfinite(tri)):
            continue
        poly4 = parallelogram_from_triangle(tri[0], tri[1], tri[2]).astype(np.float32)
        polys.append(poly4)

    if not polys:
        return None

    polys.sort(key=lambda poly: poly[:, 1].max())

    for poly in polys:
        x_min, x_max = poly[:, 0].min(), poly[:, 0].max()
        y_min, y_max = poly[:, 1].min(), poly[:, 1].max()
        w = max(1.0, float(x_max - x_min))
        h = max(1.0, float(y_max - y_min))
        diag = math.sqrt(w * w + h * h)

        if dy is not None:
            off = int(dy)
        else:
            off = int(np.clip(height_scale * diag, min_dy, max_dy))

        base = poly.astype(int)
        top = base.copy()
        top[:, 1] -= off

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [base], 1)
        cv2.fillPoly(mask, [top], 1)
        n = len(base)
        for i in range(n):
            j = (i + 1) % n
            quad = np.array([base[i], base[j], top[j], top[i]], dtype=np.int32)
            cv2.fillPoly(mask, [quad], 1)

        overlay = img.copy()
        cv2.fillPoly(overlay, [top], (0, 200, 255))
        cv2.fillPoly(overlay, [base], (0, 170, 240))
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        for i in range(n):
            cv2.line(img, tuple(base[i]), tuple(top[i]), (0, 255, 255), 2)
        cv2.polylines(img, [base], True, (0, 255, 255), 2)
        cv2.polylines(img, [top], True, (0, 255, 255), 2)

    return img


COLOR_LABELS = ("red", "pink", "green", "white", "yellow", "purple")
_COLOR_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(COLOR_LABELS)}
_COLOR_RGB_RULES = {
    "purple": {"min": (120.0, 46.0, 184.0), "max": (174.0, 133.0, 237.0)},
    "white": {"min": (218.0, 200.0, 223.0), "max": (218.0, 200.0, 223.0)},
    "red": {"min": (72.0, 27.0, 20.0), "max": (197.0, 67.0, 49.0)},
    "pink": {"min": (199.0, 135.0, 192.0), "max": (237.0, 189.0, 239.0)},
    "green": {"min": (39.0, 196.0, 141.0), "max": (127.0, 229.0, 232.0)},
    "yellow": {"min": (218.0, 203.0, 74.0), "max": (218.0, 203.0, 74.0)},
}


def _hex_to_rgb_unit(hex_color: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if not hex_color:
        return None
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        return None
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return None
    return r / 255.0, g / 255.0, b / 255.0


def _channel_score(value: float, low: float, high: float) -> float:
    if value < low or value > high:
        return 0.0
    span = max(high - low, 1e-3)
    mid = (low + high) * 0.5
    return max(0.0, 1.0 - abs(value - mid) / (0.5 * span))


def _classify_hex_color_strict(hex_color: Optional[str]):
    rgb = _hex_to_rgb_unit(hex_color)
    if rgb is None:
        return None, 0.0, None

    max_rgb = max(rgb)
    min_rgb = min(rgb)
    candidates = []

    if min_rgb >= 0.7 and (max_rgb - min_rgb) <= 0.2:
        candidates.append("white")

    for label, rule in _COLOR_RGB_RULES.items():
        mins = rule["min"]
        maxs = rule["max"]
        if all(lo <= v <= hi for v, lo, hi in zip(rgb, mins, maxs)):
            candidates.append(label)

    if len(candidates) != 1:
        return None, 0.0, None

    label = candidates[0]
    embedding = [0.0] * len(COLOR_LABELS)
    embedding[_COLOR_LABEL_TO_INDEX[label]] = 1.0
    return label, 1.0, embedding


def extract_tris_and_colors(dets,
                            orig_bgr: np.ndarray,
                            scale_to_orig_x: float,
                            scale_to_orig_y: float) -> Tuple[List[np.ndarray], List[Optional[str]]]:
    tris_orig: List[np.ndarray] = []
    color_hexes: List[Optional[str]] = []
    Hc, Wc = orig_bgr.shape[:2]
    for det in dets:
        tri = det.get("tri")
        if tri is None:
            color_hexes.append(None)
            continue
        tri = np.asarray(tri, dtype=np.float32)
        if tri.shape != (3, 2):
            color_hexes.append(None)
            continue
        tri_orig = tri.copy()
        tri_orig[:, 0] *= scale_to_orig_x
        tri_orig[:, 1] *= scale_to_orig_y
        tris_orig.append(tri_orig)

        poly = parallelogram_from_triangle(tri_orig[0], tri_orig[1], tri_orig[2]).astype(np.int32)
        mask = np.zeros((Hc, Wc), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        roi = orig_bgr[mask == 1]
        if roi.size == 0:
            color_hexes.append("000000")
            continue
        mean_bgr = roi.mean(axis=0)
        mb = int(np.clip(mean_bgr[0], 0, 255))
        mg = int(np.clip(mean_bgr[1], 0, 255))
        mr = int(np.clip(mean_bgr[2], 0, 255))
        hexcol = f"{mr:02x}{mg:02x}{mb:02x}"
        color_hexes.append(hexcol)
    return tris_orig, color_hexes
def load_homography_matrix(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    matrix_path = Path(path).expanduser()
    if not matrix_path.exists():
        raise FileNotFoundError(f"Homography file not found: {matrix_path}")
    if matrix_path.suffix.lower() == ".npy":
        H = np.load(str(matrix_path))
    else:
        with open(matrix_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "H" in data:
            H = np.asarray(data["H"], dtype=np.float64)
        elif isinstance(data, dict) and "points" in data:
            img_pts = []
            world_pts = []
            for item in data["points"]:
                img = item.get("image_px", {})
                world = item.get("world_xy", {})
                if not {"u", "v"} <= img.keys() or not {"x", "y"} <= world.keys():
                    continue
                img_pts.append([float(img["u"]), float(img["v"])])
                world_pts.append([float(world["x"]), float(world["y"])])
            if len(img_pts) < 4:
                raise ValueError("Need at least 4 correspondences to compute homography.")
            img_pts = np.asarray(img_pts, dtype=np.float64)
            world_pts = np.asarray(world_pts, dtype=np.float64)
            H, status = cv2.findHomography(img_pts, world_pts, method=0)
            if H is None:
                raise RuntimeError("Failed to compute homography from provided points.")
        else:
            raise ValueError(f"Unsupported homography file structure: {matrix_path}")
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {H.shape}")
    return H


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    pts = pts.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, H.astype(np.float64))
    return projected.reshape(-1, 2)


def dets_to_bev_entries(dets,
                        homography: Optional[np.ndarray],
                        color_hex: Optional[List[Optional[str]]] = None) -> List[dict]:
    if homography is None:
        return []
    entries: List[dict] = []
    for idx, det in enumerate(dets):
        tri = det.get("tri")
        if tri is None:
            continue
        tri = np.asarray(tri, dtype=np.float64)
        if tri.shape != (3, 2):
            continue
        try:
            tri_world = apply_homography(tri, homography)
        except cv2.error:
            continue
        if not np.all(np.isfinite(tri_world)):
            continue
        center = tri_world[0]
        front_left = tri_world[1]
        front_right = tri_world[2]
        front_center = (front_left + front_right) * 0.5
        forward_vec = front_center - center
        length_half = np.linalg.norm(forward_vec)
        width = np.linalg.norm(front_right - front_left)
        if length_half < 1e-6 or width < 1e-6:
            continue
        yaw = math.degrees(math.atan2(forward_vec[1], forward_vec[0]))
        yaw = (yaw + 180.0) % 360.0 - 180.0
        entry = {
            "center": [float(center[0]), float(center[1])],
            "length": float(length_half * 2.0),
            "width": float(width),
            "yaw": float(yaw),
            "score": float(det.get("score", 0.0)),
        }
        if color_hex is not None and idx < len(color_hex):
            color_value = color_hex[idx]
            if color_value:
                entry["color_hex"] = color_value
                label, confidence, _ = _classify_hex_color_strict(color_value)
                if label:
                    entry["color"] = label
                    entry["color_confidence"] = float(confidence)
        entries.append(entry)
    return entries


class UDPSender:
    def __init__(
        self,
        host: str,
        port: int,
        fmt: str = "json",
        max_bytes: int = 65000,
        fixed_length: Optional[float] = None,
        fixed_width: Optional[float] = None,
    ):
        self.addr = (host, int(port))
        self.fmt = fmt
        self.max_bytes = max_bytes
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.fixed_length = float(fixed_length) if fixed_length is not None else None
        self.fixed_width = float(fixed_width) if fixed_width is not None else None

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def _pack_json(self, cam_id: int, ts: float, bev_dets, capture_ts: Optional[float]):
        items = []
        for d in bev_dets or []:
            length = self.fixed_length if self.fixed_length is not None else d["length"]
            width = self.fixed_width if self.fixed_width is not None else d["width"]
            items.append({
                "center": [float(d["center"][0]), float(d["center"][1])],
                "length": float(length),
                "width": float(width),
                "yaw": float(d["yaw"]),
                "score": float(d["score"]),
            })
            color_hex = d.get("color_hex")
            if color_hex:
                items[-1]["color_hex"] = color_hex
            color_label = d.get("color")
            if color_label:
                items[-1]["color"] = color_label
            color_conf = d.get("color_confidence")
            if color_conf is not None:
                items[-1]["color_confidence"] = float(color_conf)
        msg = {
            "type": "bev_labels",
            "camera_id": cam_id,
            "timestamp": ts,
            "capture_ts": capture_ts,
            "items": items,
        }
        return json.dumps(msg, ensure_ascii=False).encode("utf-8")

    def send(self, cam_id: int, ts: float, bev_dets=None, capture_ts: Optional[float] = None):
        bev_list = bev_dets or []
        if self.fmt == "json":
            payload = self._pack_json(cam_id, ts, bev_list, capture_ts)
        else:
            raise ValueError(f"Unsupported UDP payload fmt: {self.fmt}")
        if len(payload) <= self.max_bytes:
            self.sock.sendto(payload, self.addr)


def run_live_inference(args) -> None:
    target_h, target_w = map(int, args.img_size.split(","))
    target_hw = (target_h, target_w)
    strides = [float(s) for s in args.strides.split(",")]

    runner = TensorRTTemporalRunner(
        args.weights,
        state_stride_hint=args.state_stride_hint,
        default_hidden_ch=args.default_hidden_ch
    )

    cam_w, cam_h = map(int, args.camera_size.split(","))

    save_dir = None
    if args.save_dir:
        save_dir = os.path.abspath(args.save_dir)
        os.makedirs(save_dir, exist_ok=True)

    homography = load_homography_matrix(args.homography) if args.homography else None
    if args.udp_enable and homography is None:
        raise ValueError("UDP output requested but no homography (--homography) provided.")
    udp_sender = None
    if args.udp_enable:
        udp_sender = UDPSender(
            args.udp_host,
            args.udp_port,
            fmt=args.udp_format,
            fixed_length=args.udp_fixed_length,
            fixed_width=args.udp_fixed_width,
        )

    resize_mode = None
    if args.resize_mode is not None:
        mode_key = args.resize_mode.strip().upper()
        if mode_key not in ("", "NONE"):
            if not hasattr(dai.ImgResizeMode, mode_key):
                raise ValueError(f"Unsupported resize mode: {args.resize_mode}")
            resize_mode = getattr(dai.ImgResizeMode, mode_key)

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        request_kwargs = {}
        if resize_mode is not None:
            request_kwargs["resizeMode"] = resize_mode
        if args.enable_undistort:
            request_kwargs["enableUndistortion"] = True
        video_queue = (
            cam.requestOutput((cam_w, cam_h), **request_kwargs)
            .createOutputQueue(maxSize=1, blocking=False)
        )

        pipeline.start()
        print(f"[DepthAI] Streaming {cam_w}x{cam_h} → inference size {target_w}x{target_h}")
        print("[Info] Press 'q' to quit, 'r' to reset temporal state.")

        frame_idx = 0
        try:
            while pipeline.isRunning():
                img_packet = video_queue.tryGet()
                if img_packet is None:
                    continue
                cam_latency_ms = (
                    dai.Clock.now() - img_packet.getTimestamp()
                ).total_seconds() * 1000.0
                frame_bgr = img_packet.getCvFrame()

                prep = preprocess_frame(frame_bgr, target_hw)

                t0 = time.time()
                outputs = runner.forward(prep["img_np"])
                infer_ms = (time.time() - t0) * 1000.0

                t_post0 = time.time()
                dets = decode_predictions(
                    outputs,
                    strides,
                    clip_cells=args.clip_cells,
                    conf_th=args.conf,
                    nms_iou=args.nms_iou,
                    topk=args.topk,
                    contain_thr=args.contain_thr,
                    score_mode=args.score_mode,
                    use_gpu_nms=True
                )[0]
                dets = tiny_filter_on_dets(dets, min_area=20.0, min_edge=3.0)
                tris_orig, color_hex_list = [], []
                if dets:
                    tris_orig, color_hex_list = extract_tris_and_colors(
                        dets,
                        prep["orig_bgr"],
                        prep["scale_to_orig_x"],
                        prep["scale_to_orig_y"]
                    )
                bev_entries = dets_to_bev_entries(dets, homography, color_hex_list)
                post_ms = (time.time() - t_post0) * 1000.0

                t_vis0 = time.time()
                vis_ms = 0.0
                key = -1
                vis_frame = None
                if args.show_vis or save_dir is not None:
                    if tris_orig:
                        vis_frame = draw_pred_pseudo3d(
                            prep["orig_bgr"],
                            tris_orig
                        )
                    if vis_frame is None:
                        vis_frame = overlay_detections(
                            prep["orig_bgr"],
                            dets,
                            prep["scale_to_orig_x"],
                            prep["scale_to_orig_y"]
                        )
                    cv2.putText(vis_frame, f"Infer {infer_ms:.1f} ms | dets {len(dets)}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                if args.show_vis and vis_frame is not None:
                    cv2.imshow("DepthAI TRT Live", vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    vis_ms = (time.time() - t_vis0) * 1000.0
                else:
                    key = cv2.waitKey(1) & 0xFF
                    vis_ms = (time.time() - t_vis0) * 1000.0

                if save_dir is not None:
                    filename = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
                    frame_to_save = vis_frame if vis_frame is not None else prep["orig_bgr"]
                    cv2.imwrite(filename, frame_to_save)
                frame_idx += 1

                capture_ts = None
                ts_raw = img_packet.getTimestamp()
                if ts_raw is not None and hasattr(ts_raw, "total_seconds"):
                    capture_ts = ts_raw.total_seconds()

                if udp_sender is not None:
                    try:
                        udp_sender.send(
                            cam_id=args.camera_id,
                            ts=time.time(),
                            bev_dets=bev_entries,
                            capture_ts=capture_ts
                        )
                    except Exception as exc:
                        print(f"[UDP] send error: {exc}")

                total_ms = cam_latency_ms + infer_ms + post_ms + vis_ms
                print(
                    f"[Timing] cam {cam_latency_ms:.1f} ms | infer {infer_ms:.1f} ms | "
                    f"post {post_ms:.1f} ms | vis {vis_ms:.1f} ms | total {total_ms:.1f} ms"
                )

                if key == ord("q"):
                    break
                if key == ord("r"):
                    print("[Info] State reset requested.")
                    runner.reset()
        finally:
            cv2.destroyAllWindows()
            if udp_sender is not None:
                udp_sender.close()


def parse_args():
    parser = argparse.ArgumentParser("DepthAI live inference with TensorRT temporal model")
    parser.add_argument("--weights", type=str, required=True, help="TensorRT engine (.engine) path")
    parser.add_argument("--img-size", type=str, default="864,1536", help="Model input H,W")
    parser.add_argument("--camera-size", type=str, default="1920,1080", help="DepthAI output width,height")
    parser.add_argument("--score-mode", type=str, default="obj*cls", choices=["obj", "cls", "obj*cls"])
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--nms-iou", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--contain-thr", type=float, default=0.85)
    parser.add_argument("--clip-cells", type=float, default=None)
    parser.add_argument("--strides", type=str, default="8,16,32")
    parser.add_argument("--state-stride-hint", type=int, default=32)
    parser.add_argument("--default-hidden-ch", type=int, default=256)
    parser.add_argument("--show-vis", dest="show_vis", action="store_true", default=True,
                        help="Enable OpenCV visualization window (default: on)")
    parser.add_argument("--no-show-vis", dest="show_vis", action="store_false",
                        help="Disable OpenCV visualization window")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Optional directory to save each processed frame (after inference)")
    parser.add_argument("--enable-undistort", dest="enable_undistort", action="store_true",
                        help="Enable on-device undistortion for camera output")
    parser.add_argument("--disable-undistort", dest="enable_undistort", action="store_false",
                        help="Disable on-device undistortion")
    parser.set_defaults(enable_undistort=True)
    parser.add_argument("--resize-mode", type=str, default="LETTERBOX",
                        help="DepthAI resize mode (CROP|STRETCH|LETTERBOX|NONE)")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="Camera identifier included in UDP payloads")
    parser.add_argument("--homography", type=str, default=None,
                        help="Path to 3x3 homography (.npy or JSON with correspondences)")
    parser.add_argument("--udp-enable", action="store_true",
                        help="Enable UDP streaming of BEV detections (same format as main.py)")
    parser.add_argument("--udp-host", type=str, default="192.168.0.165")
    parser.add_argument("--udp-port", type=int, default=50050)
    parser.add_argument("--udp-format", choices=["json", "text"], default="json")
    parser.add_argument("--udp-fixed-length", type=float, default=4.4,
                        help="Override vehicle length in UDP payloads (meters)")
    parser.add_argument("--udp-fixed-width", type=float, default=2.7,
                        help="Override vehicle width in UDP payloads (meters)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_live_inference(args)


if __name__ == "__main__":
    main()
