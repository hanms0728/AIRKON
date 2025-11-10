import numpy as np
import open3d as o3d

ply_path = "pointcloud/merged_.ply"
pcd = o3d.io.read_point_cloud(ply_path)

# 보기용 좌우 반전 행렬 (Y축 부호 반전)
T_flipY = np.eye(4)
T_flipY[1, 1] = -1  # y축 뒤집기

pcd.transform(T_flipY)

o3d.visualization.draw_geometries(
    [pcd],
    window_name="CARLA-aligned View",
    width=960,
    height=720,
    left=50,
    top=50,
)