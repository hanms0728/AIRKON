"""Batch resize all images under a root directory while mirroring its structure.

Usage example::

    python utils/distort/batch_resize.py \
        --root ./dataset/root \
        --width 1536 --height 864

The script will produce a sibling directory named ``<root>_resize`` (configurable
via ``--suffix``) and recreate every subdirectory structure inside it. Any image
file found in the original tree (filtered by ``--exts``) is resized to the target
resolution and saved to the corresponding path under the resize root.
"""

import argparse
import os
from typing import Tuple

import cv2


VALID_INTERPS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def is_image_file(name: str, exts: Tuple[str, ...]) -> bool:
    lower = name.lower()
    return lower.endswith(exts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize images for every folder under a root")
    parser.add_argument("--root", required=True, help="원본 이미지 루트 폴더")
    parser.add_argument("--width", type=int, default=1536, help="출력 이미지 폭")
    parser.add_argument("--height", type=int, default=864, help="출력 이미지 높이")
    parser.add_argument(
        "--exts",
        type=str,
        default=".jpg,.jpeg,.png,.bmp,.tif,.tiff",
        help="쉼표로 구분된 이미지 확장자 목록",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_resize",
        help="root 폴더 이름 뒤에 붙일 접미사",
    )
    parser.add_argument(
        "--interp",
        choices=list(VALID_INTERPS.keys()),
        default="area",
        help="cv2.resize 보간 방식",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="이미 출력 파일이 있으면 건너뜀",
    )
    parser.add_argument(
        "--check-input-size",
        action="store_true",
        help="입력 이미지 크기가 기대한 1280x720인지 확인하고 다르면 경고",
    )
    return parser.parse_args()


def ensure_paths(src_root: str, suffix: str) -> str:
    parent = os.path.dirname(src_root)
    base = os.path.basename(src_root.rstrip("/\\"))
    dst_root = os.path.join(parent, base + suffix)
    os.makedirs(dst_root, exist_ok=True)
    return dst_root


def resize_image(img, width: int, height: int, interpolation: int):
    return cv2.resize(img, (width, height), interpolation=interpolation)


def main() -> None:
    args = parse_args()

    root_dir = os.path.abspath(args.root)
    if not os.path.isdir(root_dir):
        raise SystemExit(f"[Error] root directory not found: {root_dir}")

    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    if not exts:
        raise SystemExit("[Error] Provide at least one extension via --exts")

    dst_root = ensure_paths(root_dir, args.suffix)
    interp_flag = VALID_INTERPS[args.interp]

    total_imgs = 0
    skipped = 0

    for dirpath, _, filenames in os.walk(root_dir):
        rel = os.path.relpath(dirpath, root_dir)
        dst_dir = os.path.join(dst_root, rel)
        os.makedirs(dst_dir, exist_ok=True)

        for fname in filenames:
            if not is_image_file(fname, exts):
                continue

            src_path = os.path.join(dirpath, fname)
            dst_path = os.path.join(dst_dir, fname)

            if args.skip_existing and os.path.isfile(dst_path):
                skipped += 1
                continue

            img = cv2.imread(src_path)
            if img is None:
                print(f"[WARN] Failed to load image, skipping: {src_path}")
                continue

            h, w = img.shape[:2]
            if args.check_input_size and (w != 1280 or h != 720):
                print(f"[WARN] Unexpected input size {w}x{h} for {src_path}")

            resized = resize_image(img, args.width, args.height, interp_flag)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            cv2.imwrite(dst_path, resized)
            total_imgs += 1

    print(
        f"[Done] resized {total_imgs} images (skipped {skipped}) into {dst_root}"
    )


if __name__ == "__main__":
    main()
