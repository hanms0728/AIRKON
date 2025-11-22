"""Batch ONNX inference for every subfolder inside a root directory.

This script reuses the single-folder inference utilities from
``inference_onnx_eval.py`` but adds orchestration around a root path that has
multiple subdirectories, each filled with raw images. For every subfolder it
will:

1. Run the ONNX model on all image files in that folder.
2. Write visualization images to ``images_gt`` and label TXT files to ``labels``
   inside the same folder.
3. Move the original input images into an ``images`` subdirectory so the folder
   keeps the inference outputs separated from the processed raws.

Example
-------
python auto_labels/batch_root_inference.py \
  --onnx ./onnx/2_5d_model.onnx \
  --root-dir ./root_dataset \
  --half
"""

import argparse
import os
import shutil
from typing import Iterable, List, Sequence

import cv2
import numpy as np
from tqdm import tqdm

import detection_inference_onnx as det_onnx
from inference_onnx_eval import draw_pred_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Run inference for every subfolder inside a root directory"
    )
    parser.add_argument("--onnx", type=str, required=True, help="ONNX 모델 경로")
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="여러 개의 이미지 폴더가 들어있는 최상위 디렉터리",
    )
    parser.add_argument(
        "--img-size",
        type=str,
        default="832,1440",
        help="Inferencer 내부 letterbox H,W (모델과 일치해야 함)",
    )
    parser.add_argument(
        "--exts", type=str, default=".jpg,.jpeg,.png", help="허용할 이미지 확장자"
    )
    parser.add_argument("--half", action="store_true", help="half precision 모델인지 여부")
    parser.add_argument(
        "--txt-include-score",
        action="store_true",
        help="라벨 TXT에 score까지 저장",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=400,
        help="폴더 처리 중 N장의 이미지마다 진행 로그 출력",
    )
    return parser.parse_args()


def list_subfolders(root_dir: str) -> List[str]:
    subs = []
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            subs.append(path)
    return subs


def collect_images(folder: str, exts: Sequence[str]) -> List[str]:
    names = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(exts):
            names.append(name)
    return names


def to_numpy(preds) -> np.ndarray:
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    return np.asarray(preds, dtype=np.float32)


def move_images(folder: str, names: Iterable[str]) -> None:
    images_dir = os.path.join(folder, "images")
    os.makedirs(images_dir, exist_ok=True)
    for name in names:
        src = os.path.join(folder, name)
        if not os.path.exists(src):  # 이미 이동된 경우 건너뛰기
            continue
        dst = os.path.join(images_dir, name)
        if os.path.exists(dst) and os.path.isfile(dst):
            os.remove(dst)
        shutil.move(src, dst)


def process_folder(
    folder: str,
    inferencer: det_onnx.Inferencer,
    image_names: Sequence[str],
    include_score: bool,
    log_every: int,
) -> int:
    out_img_dir = os.path.join(folder, "images_gt")
    out_lab_dir = os.path.join(folder, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lab_dir, exist_ok=True)

    for idx, name in enumerate(
        tqdm(image_names, desc=os.path.basename(folder), leave=False, ncols=90)
    ):
        img_path = os.path.join(folder, name)
        img0 = cv2.imread(img_path)
        if img0 is None:
            continue

        preds = to_numpy(inferencer.run(img0))
        draw_pred_only(
            img0,
            preds,
            os.path.join(out_img_dir, name),
            os.path.join(out_lab_dir, os.path.splitext(name)[0] + ".txt"),
            include_score=include_score,
        )

        if log_every > 0 and (idx + 1) % log_every == 0:
            print(f"[{os.path.basename(folder)}] processed {idx+1}/{len(image_names)} images")

    move_images(folder, image_names)
    return len(image_names)


def main() -> None:
    args = parse_args()
    H_l, W_l = map(int, args.img_size.split(","))
    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    if not exts:
        raise SystemExit("[Error] No valid extensions provided via --exts")

    if not os.path.isdir(args.root_dir):
        raise SystemExit(f"[Error] root directory not found: {args.root_dir}")
    if not os.path.isfile(args.onnx):
        raise SystemExit(f"[Error] ONNX file not found: {args.onnx}")

    subfolders = list_subfolders(args.root_dir)
    if not subfolders:
        raise SystemExit(f"[Error] No subfolders found under {args.root_dir}")

    inferencer = det_onnx.Inferencer(
        model_path=args.onnx,
        letterbox_size=(H_l, W_l),
        is_half=args.half,
    )

    total_imgs = 0
    processed_folders = 0
    for folder in subfolders:
        image_names = collect_images(folder, exts)
        if not image_names:
            print(f"[Skip] {folder}: no matching images")
            continue

        print(f"[Run] {folder}: {len(image_names)} images")
        processed = process_folder(
            folder,
            inferencer,
            image_names,
            include_score=args.txt_include_score,
            log_every=args.log_every,
        )
        total_imgs += processed
        processed_folders += 1

    print(
        f"[Done] Processed {total_imgs} images across {processed_folders} subfolders"
    )


if __name__ == "__main__":
    main()
