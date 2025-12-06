#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from src.evaluation_utils import (
    decode_predictions,
    evaluate_single_image,
    compute_detection_metrics,
)


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ONNX parallelogram detector on an image folder."
    )
    parser.add_argument("--onnx", type=str, required=True, help="Exported ONNX path.")
    parser.add_argument("--image-dir", type=str, required=True, help="Folder of images.")
    parser.add_argument(
        "--label-dir",
        type=str,
        default=None,
        help="Folder of GT label txt files (optional).",
    )
    parser.add_argument(
        "--img-h", type=int, default=864, help="Resize height used during training."
    )
    parser.add_argument(
        "--img-w", type=int, default=1536, help="Resize width used during training."
    )
    parser.add_argument(
        "--strides",
        type=float,
        nargs="+",
        default=[8.0, 16.0, 32.0],
        help="Detection head strides (match training).",
    )
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence thresh.")
    parser.add_argument("--nms-iou", type=float, default=0.20, help="NMS IoU thresh.")
    parser.add_argument(
        "--clip-cells",
        type=float,
        default=4.0,
        help="Clip regression offsets (same as training).",
    )
    parser.add_argument("--topk", type=int, default=200, help="Max detections per image.")
    parser.add_argument(
        "--contain-thr",
        type=float,
        default=0.7,
        help="Containment threshold used by CPU NMS.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["obj", "cls", "obj*cls"],
        default="obj*cls",
        help="Scoring strategy.",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.5,
        help="IoU threshold for evaluation when labels exist.",
    )
    parser.add_argument(
        "--cam-id",
        type=int,
        default=0,
        help="Cam id passed into decode (for per-cam thresholds).",
    )
    parser.add_argument(
        "--save-preds-dir",
        type=str,
        default=None,
        help="If set, save per-image prediction txt files in this directory.",
    )
    parser.add_argument(
        "--save-vis-dir",
        type=str,
        default=None,
        help="If set, save visualization images with predicted polygons.",
    )
    return parser.parse_args()


def list_images(image_dir: Path):
    files = [p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def preprocess_image(path: Path, target_size):
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    orig_h, orig_w = img_bgr.shape[:2]
    resized_bgr = cv2.resize(img_bgr, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    arr = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    batched = np.expand_dims(arr, axis=0)
    return batched, (orig_h, orig_w), resized_bgr


def load_gt(label_dir: Path, stem: str, orig_shape, target_size):
    if label_dir is None:
        return None, None
    label_path = label_dir / f"{stem}.txt"
    if not label_path.exists():
        return None, None
    orig_h, orig_w = orig_shape
    tgt_h, tgt_w = target_size
    sx = tgt_w / max(orig_w, 1e-6)
    sy = tgt_h / max(orig_h, 1e-6)
    tris = []
    cls_ids = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 7:
                continue
            cls_id, p0x, p0y, p1x, p1y, p2x, p2y = map(float, parts)
            tris.append(
                [
                    [p0x * sx, p0y * sy],
                    [p1x * sx, p1y * sy],
                    [p2x * sx, p2y * sy],
                ]
            )
            cls_ids.append(int(cls_id))
    if not tris:
        return None, None
    return np.asarray(tris, dtype=np.float32), np.asarray(cls_ids, dtype=np.int64)


def onnx_outputs_to_torch(outputs):
    if len(outputs) % 3 != 0:
        raise RuntimeError("Unexpected ONNX output count.")
    n_heads = len(outputs) // 3
    torch_outs = []
    for i in range(n_heads):
        reg = torch.from_numpy(outputs[i])
        obj = torch.from_numpy(outputs[i + n_heads])
        cls = torch.from_numpy(outputs[i + 2 * n_heads])
        torch_outs.append((reg, obj, cls))
    return torch_outs

def draw_detections(image_bgr: np.ndarray, detections, color=(0, 255, 0)):
    canvas = image_bgr.copy()
    for det in detections:
        tri = np.asarray(det["tri"], dtype=np.int32)
        cv2.polylines(canvas, [tri.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=2)
        score = det.get("score", 0.0)
        cls_id = det.get("class_id", det.get("cls", 0))
        label = f"{cls_id}:{score:.2f}"
        x, y = int(tri[0, 0]), int(tri[0, 1])
        cv2.putText(
            canvas,
            label,
            (x, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    label_dir = Path(args.label_dir).resolve() if args.label_dir else None
    if label_dir and not label_dir.is_dir():
        raise FileNotFoundError(f"Label folder not found: {label_dir}")
    vis_dir = Path(args.save_vis_dir).resolve() if args.save_vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(args.save_preds_dir).resolve() if args.save_preds_dir else None
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)

    session = ort.InferenceSession(
        str(Path(args.onnx)),
        providers=ort.get_available_providers(),
    )
    input_name = session.get_inputs()[0].name
    target_size = (args.img_h, args.img_w)
    strides = [float(s) for s in args.strides]

    images = list_images(image_dir)
    if not images:
        raise RuntimeError("No images found.")

    all_records = []
    total_gt = 0
    pbar = tqdm(images, desc="ONNX inference")
    for img_path in pbar:
        img_tensor, orig_shape, vis_bgr = preprocess_image(img_path, target_size)
        outputs = session.run(None, {input_name: img_tensor})
        torch_outs = onnx_outputs_to_torch(outputs)
        decoded = decode_predictions(
            args.cam_id,
            torch_outs,
            strides,
            clip_cells=args.clip_cells,
            conf_th=args.conf,
            nms_iou=args.nms_iou,
            topk=args.topk,
            contain_thr=args.contain_thr,
            score_mode=args.score_mode,
            use_gpu_nms=False,
        )[0]

        orig_h, orig_w = orig_shape
        tgt_h, tgt_w = target_size
        sx = max(orig_w / float(tgt_w), 1e-6)
        sy = max(orig_h / float(tgt_h), 1e-6)

        dets_orig = []
        for det in decoded:
            tri = np.asarray(det["tri"], dtype=np.float32)
            tri[:, 0] *= sx
            tri[:, 1] *= sy
            dets_orig.append(
                {
                    "score": float(det["score"]),
                    "class_id": int(det.get("class_id", det.get("cls", 0))),
                    "tri": tri.tolist(),
                }
            )

        if pred_dir:
            txt_path = pred_dir / f"{img_path.stem}.txt"
            with open(txt_path, "w") as f:
                for det in dets_orig:
                    tri = det["tri"]
                    cls_id = det["class_id"]
                    score = det["score"]
                    f.write(
                        "{} {:.6f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(
                            cls_id,
                            score,
                            tri[0][0],
                            tri[0][1],
                            tri[1][0],
                            tri[1][1],
                            tri[2][0],
                            tri[2][1],
                        )
                    )

        if vis_dir:
            orig_img = cv2.imread(str(img_path))
            if orig_img is not None:
                vis_img = draw_detections(orig_img, dets_orig)
                out_path = vis_dir / img_path.name
                cv2.imwrite(str(out_path), vis_img)

        if label_dir is not None:
            gt_tris, gt_cls = load_gt(label_dir, img_path.stem, orig_shape, target_size)
            if gt_tris is not None:
                cls_arg = gt_cls if gt_cls is not None else None
                records, _ = evaluate_single_image(
                    decoded,
                    gt_tris,
                    cls_arg,
                    iou_thr=args.iou_thr,
                )
                all_records.extend(records)
                total_gt += gt_tris.shape[0]

    if total_gt > 0 and all_records:
        metrics = compute_detection_metrics(all_records, total_gt)
        print(
            "[Eval] precision={:.4f} recall={:.4f} mAP@50={:.4f} mAOE(deg)={:.2f}".format(
                metrics["precision"],
                metrics["recall"],
                metrics["map50"],
                metrics["mAOE_deg"],
            )
        )
    else:
        print("Inference completed. No ground-truth labels provided, so metrics were skipped.")


if __name__ == "__main__":
    main()
