#!/usr/bin/env python3
"""
inspect_npz.py
---------------
간단한 NPZ 뷰어. pointcloud/cloud_rgb_npz/ 같은 LUT 파일의 구조와
통계를 확인할 수 있습니다.

사용 예시:
    python pointcloud/make_pointcloud/inspect_npz.py pointcloud/cloud_rgb_npz/cloud_rgb_1.npz \
        --verbose --percentiles --sample 5
"""

import argparse
import pathlib
from typing import Iterable

import numpy as np


def _human_readable_shape(shape: Iterable[int]) -> str:
    return "x".join(str(dim) for dim in shape) if shape else "()"


def _summarize_array(
    name: str,
    arr: np.ndarray,
    *,
    verbose: bool = False,
    percentiles: bool = False,
    sample: int = 0,
) -> list[str]:
    """Return a list of description lines for a single NPZ entry."""

    lines = []
    shape = _human_readable_shape(arr.shape)
    dtype = str(arr.dtype)
    header = f"{name}: shape={shape}, dtype={dtype}, size={arr.size}"
    lines.append(header)

    if arr.size == 0:
        lines[0] += " (empty)"
        return lines

    kind = arr.dtype.kind
    if kind in ("f", "i", "u"):
        arr_float = arr.astype(np.float64, copy=False)
        finite_mask = np.isfinite(arr_float)
        valid = arr_float[finite_mask]

        if valid.size:
            min_val = float(np.nanmin(valid))
            max_val = float(np.nanmax(valid))
            lines[0] += f", min={min_val:.5g}, max={max_val:.5g}"
        else:
            lines[0] += ", min=nan, max=nan"

        if verbose:
            total = arr.size
            nan_count = int(np.isnan(arr_float).sum()) if kind == "f" else 0
            inf_count = int(np.isinf(arr_float).sum()) if kind == "f" else 0
            finite_count = int(finite_mask.sum())
            lines.append(
                f"    finite={finite_count}/{total}, nan={nan_count}, inf={inf_count}"
            )
            if valid.size:
                mean_val = float(np.nanmean(valid))
                std_val = float(np.nanstd(valid))
                lines.append(f"    mean={mean_val:.5g}, std={std_val:.5g}")
                if percentiles:
                    p5, p50, p95 = np.nanpercentile(valid, [5, 50, 95])
                    lines.append(f"    pct5/50/95={p5:.5g}/{p50:.5g}/{p95:.5g}")
    elif kind == "b":
        true_count = int(arr.sum())
        false_count = int(arr.size - true_count)
        lines[0] += f", true={true_count}, false={false_count}"
    elif kind in ("U", "S", "O"):
        unique_vals = np.unique(arr)
        preview = ", ".join(repr(v) for v in unique_vals[:5])
        lines[0] += f", unique={len(unique_vals)} [{preview}]"

    if sample > 0:
        flat = arr.reshape(-1)
        take = flat[:sample]
        sample_list = take.tolist()
        lines.append(
            f"    sample({len(sample_list)} of {arr.size}): {sample_list}"
        )

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="NPZ 파일 내용을 요약해 출력합니다.")
    parser.add_argument("npz_path", type=pathlib.Path, help="대상 NPZ 파일")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="float/int 배열에 평균·표준편차 등 추가 통계를 출력합니다.",
    )
    parser.add_argument(
        "--percentiles",
        action="store_true",
        help="5/50/95 분위수를 함께 출력합니다 (numeric 배열).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="각 배열의 앞쪽 N개 값을 평탄화해 예시로 출력합니다.",
    )
    args = parser.parse_args()

    if not args.npz_path.is_file():
        raise FileNotFoundError(f"NPZ 파일을 찾을 수 없습니다: {args.npz_path}")

    with np.load(args.npz_path, allow_pickle=True) as data:
        print(f"[INFO] {args.npz_path} (keys={len(data.files)})")
        for key in data.files:
            arr = data[key]
            lines = _summarize_array(
                key,
                arr,
                verbose=args.verbose,
                percentiles=args.percentiles,
                sample=max(0, args.sample),
            )
            for i, line in enumerate(lines):
                prefix = " - " if i == 0 else "   "
                print(prefix + line)


if __name__ == "__main__":
    main()
