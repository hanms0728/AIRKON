"""Batch runner for src.inference_lstm_onnx_pointcloud.

이 스크립트는 ``batch_root_inference.py``와 동일한 디렉터리 오케스트레이션을
Temporal ONNX 추론 파이프라인(src.inference_lstm_onnx_pointcloud)에 적용한다. 루트
디렉터리 아래의 각 하위 폴더에 대해:

1. 해당 폴더 안의 이미지 파일을 입력으로 ``src.inference_lstm_onnx_pointcloud`` 모듈을 실행한다.
2. 모듈이 생성한 ``images``(시각화)와 ``labels`` 결과를 각각 ``images_gt``와 ``labels``로 옮긴다.
3. 원본 이미지는 ``images`` 하위 폴더로 이동해 결과물과 분리한다.

필수 파라미터는 루트 폴더와 ONNX weight 경로이며, 추가 인자는
``--extra-args``를 통해 원본 추론 스크립트로 그대로 전달할 수 있다.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from typing import Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Run inference_lstm_onnx_pointcloud.py for every subfolder inside a root directory"
    )
    parser.add_argument("--weights", type=str, required=True, help="ONNX weight 경로")
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="이미지 폴더들이 들어있는 최상위 디렉터리",
    )
    parser.add_argument(
        "--lut-path",
        type=str,
        default=None,
        help="pixel2world_lut.npz 경로 (전달 시 BEV까지 출력)",
    )
    parser.add_argument(
        "--exts", type=str, default=".jpg,.jpeg,.png", help="허용할 이미지 확장자"
    )
    parser.add_argument(
        "--temp-dir-name",
        type=str,
        default="__inference_tmp",
        help="각 폴더 내부에서 임시 출력으로 사용할 디렉터리 이름",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="src.inference_lstm_onnx_pointcloud.py로 그대로 전달할 추가 CLI 인자",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 추론 실행 없이 명령만 출력",
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


def move_images(folder: str, names: Iterable[str]) -> None:
    dst_dir = os.path.join(folder, "images")
    os.makedirs(dst_dir, exist_ok=True)
    for name in names:
        src = os.path.join(folder, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            if os.path.isfile(dst):
                os.remove(dst)
            else:
                shutil.rmtree(dst)
        shutil.move(src, dst)


def move_output_dir(src: str, dst: str) -> None:
    if not os.path.exists(src):
        return
    if os.path.isfile(dst):
        os.remove(dst)
    elif os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)


def build_infer_cmd(
    folder: str,
    temp_dir: str,
    weights: str,
    lut_path: str | None,
    extra_args: str,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.inference_lstm_onnx_pointcloud",
        "--input-dir",
        folder,
        "--output-dir",
        temp_dir,
        "--weights",
        weights,
    ]
    if lut_path:
        cmd.extend(["--lut-path", lut_path])
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args))
    return cmd


def process_folder(
    folder: str,
    image_names: Sequence[str],
    args: argparse.Namespace,
) -> int:
    if not image_names:
        return 0

    temp_dir = os.path.join(folder, args.temp_dir_name)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    cmd = build_infer_cmd(
        folder,
        temp_dir,
        args.weights,
        args.lut_path,
        args.extra_args,
    )

    print(f"[Run] {folder}: {len(image_names)} images")
    if args.dry_run:
        print("    dry-run cmd:", " ".join(cmd))
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 0

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[Error] inference failed for {folder}: {exc}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 0

    move_output_dir(os.path.join(temp_dir, "images"), os.path.join(folder, "images_gt"))
    move_output_dir(os.path.join(temp_dir, "labels"), os.path.join(folder, "labels"))
    move_output_dir(
        os.path.join(temp_dir, "images_with_gt"), os.path.join(folder, "images_with_gt")
    )
    move_output_dir(os.path.join(temp_dir, "bev_images"), os.path.join(folder, "bev_images"))
    move_output_dir(
        os.path.join(temp_dir, "bev_images_with_gt"),
        os.path.join(folder, "bev_images_with_gt"),
    )
    move_output_dir(os.path.join(temp_dir, "bev_labels"), os.path.join(folder, "bev_labels"))

    shutil.rmtree(temp_dir, ignore_errors=True)
    move_images(folder, image_names)
    return len(image_names)


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.root_dir):
        raise SystemExit(f"[Error] root directory not found: {args.root_dir}")
    if not os.path.isfile(args.weights):
        raise SystemExit(f"[Error] weights file not found: {args.weights}")
    if args.lut_path and not os.path.isfile(args.lut_path):
        raise SystemExit(f"[Error] LUT file not found: {args.lut_path}")

    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    if not exts:
        raise SystemExit("[Error] No valid extensions provided via --exts")

    subfolders = list_subfolders(args.root_dir)
    if not subfolders:
        raise SystemExit(f"[Error] No subfolders found under {args.root_dir}")

    total_imgs = 0
    processed_folders = 0

    for folder in subfolders:
        image_names = collect_images(folder, exts)
        if not image_names:
            print(f"[Skip] {folder}: no matching images")
            continue

        processed = process_folder(folder, image_names, args)
        if processed > 0:
            processed_folders += 1
        total_imgs += processed

    print(
        f"[Done] Processed {total_imgs} images across {processed_folders} subfolders"
    )


if __name__ == "__main__":
    main()
