import trimesh
import numpy as np
from trimesh.transformations import rotation_matrix

# ====== 1. 바닥 (Base) ======
base_height = 0.3
base_radius = 0.3

base = trimesh.creation.cylinder(
    radius=base_radius,
    height=base_height,
    sections=64
)
# z = 0 ~ 0.3 에 놓이도록
base.apply_translation([0, 0, base_height / 2])


# ====== 2. 기둥 (Pole) ======
pole_height = 3.0
pole_radius = 0.05

pole = trimesh.creation.cylinder(
    radius=pole_radius,
    height=pole_height,
    sections=64
)
# 바닥 위에 올라가도록: 아래가 z = base_height, 위가 base_height + pole_height
pole.apply_translation([0, 0, base_height + pole_height / 2])


# ====== 3. 수평 팔 (Horizontal Arm, ㄷ의 윗부분) ======
arm_length = 0.8      # 팔 길이
arm_radius = 0.03

arm = trimesh.creation.cylinder(
    radius=arm_radius,
    height=arm_length,
    sections=64
)

# 기본 cylinder는 z 방향 → x축 방향으로 눕히기
rot = rotation_matrix(
    angle=np.pi / 2,      # 90도 회전
    direction=[0, 1, 0],  # y축 기준
    point=[0, 0, 0]
)
arm.apply_transform(rot)

# 팔이 기둥 맨 위 근처에서 옆으로 뻗도록 위치 조정
pole_top_z = base_height + pole_height          # 기둥 맨 위 z 3.3
arm_attach_z = pole_top_z - 0.1                 # 기둥 윗부분에서 살짝 아래 3.2

# x 방향으로 기둥 밖으로 나가도록 (기둥 반지름 + 팔 절반)
arm_center_x = pole_radius + arm_length / 2
arm.apply_translation([arm_center_x, 0, arm_attach_z])

# 팔 끝 x 좌표 (drop arm이 내려올 위치)
arm_end_x = pole_radius + arm_length 


# ====== 4. 아래로 내려오는 팔 (Drop Arm, ㄷ의 세로 부분) ======
drop_length = 0.4   # 얼마나 아래로 내릴지
drop_radius = arm_radius

drop_arm = trimesh.creation.cylinder(
    radius=drop_radius,
    height=drop_length,
    sections=64
)
# 이 cylinder는 기본적으로 z 방향으로 서 있음 (위/아래)

# 위쪽 끝이 arm 아래에 붙고, 아래로 내려오게:
# 중심 z = arm_attach_z - drop_length / 2
drop_center_z = arm_attach_z - drop_length / 2 + 0.05

drop_arm.apply_translation([arm_end_x, 0, drop_center_z])


# ====== 5. 동그란 등 (Bulb) - drop arm 끝에서 아래로 매달기 ======
bulb_radius = 0.15

bulb = trimesh.creation.icosphere(
    subdivisions=3,
    radius=bulb_radius
)

# drop arm의 아래 끝보다 살짝 더 아래에 전구 달기
drop_bottom_z = arm_attach_z - drop_length # 3.2 - 0.4 = 2.8
bulb_center_z = drop_bottom_z - bulb_radius * 0.6   # 약간 더 내려서 매달린 느낌 2.8 - 0.09 = 2.71

bulb.apply_translation([arm_end_x, 0, bulb_center_z]) # 0.85, 0, 2.71


# ====== 6. 색 지정 (선택) ======

def paint_mesh(mesh, color_rgba):
    colors = np.tile(color_rgba, (mesh.vertices.shape[0], 1))
    mesh.visual.vertex_colors = colors

# 기둥/팔/베이스: 회색 계열
paint_mesh(base, [150, 150, 150, 255])
paint_mesh(pole, [170, 170, 170, 255])
paint_mesh(arm,  [170, 170, 170, 255])
paint_mesh(drop_arm, [170, 170, 170, 255])

# 전구: 살짝 노란빛
paint_mesh(bulb, [250, 250, 220, 255])


# ====== 7. 전체 가로등 합치기 & glb로 저장 ======
lamp = trimesh.util.concatenate([base, pole, arm, drop_arm, bulb])

lamp.export("utils/glb/street_lamp_hanging.glb")
print("street_lamp_hanging.glb 생성 완료!")
