import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(description="단일 PLY 포인트클라우드를 시각화합니다.")
    parser.add_argument(
        "--ply",
        required=True,
        help="시각화할 PLY 파일 경로",
    )
    parser.add_argument(
        "--no-flip-y",
        action="store_true",
        help="기본으로 적용되는 Y축 반전을 비활성화합니다.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ply_path = Path(args.ply).expanduser()
    if not ply_path.is_file():
        raise SystemExit(f"PLY 파일을 찾을 수 없습니다: {ply_path}")

    pcd = o3d.io.read_point_cloud(str(ply_path))

    if not args.no_flip_y:
        # 보기용 좌우 반전 행렬 (Y축 부호 반전)
        T_flipY = np.eye(4)
        T_flipY[1, 1] = -1  # y축 뒤집기
        pcd.transform(T_flipY)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"PLY Viewer | {ply_path.name}",
        width=960,
        height=720,
        left=50,
        top=50,
    )


if __name__ == "__main__":
    main()
