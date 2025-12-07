#!/usr/bin/env python3

import argparse
import os
import time
from typing import Tuple

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

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        video_queue = (
            cam.requestOutput((cam_w, cam_h))
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
                post_ms = (time.time() - t_post0) * 1000.0

                t_vis0 = time.time()
                vis_ms = 0.0
                key = -1
                vis_frame = None
                if args.show_vis or save_dir is not None:
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
    return parser.parse_args()


def main():
    args = parse_args()
    run_live_inference(args)


if __name__ == "__main__":
    main()
