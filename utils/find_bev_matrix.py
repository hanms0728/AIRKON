import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import argparse
import os

def _annotate_bev_axes(bev_img, x_min, x_max, z_min, z_max, label_x='X', label_z='Z', camera_pos=None):
    """BEV 이미지에 눈금/축 레이블을 보기 좋게 오버레이합니다."""
    annotated = bev_img.copy()
    height, width = annotated.shape[:2]
    overlay = annotated.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    axis_color = (0, 220, 255)
    tick_color = (255, 255, 255)
    grid_color = (80, 80, 80)
    line_thickness = 2

    def to_u(x_world):
        return int(round((x_world - x_min) / (x_max - x_min) * (width - 1)))

    def to_v(z_world):
        return int(round((z_max - z_world) / (z_max - z_min) * (height - 1)))

    # 살짝 어두운 배경 밴드 추가
    cv2.rectangle(overlay, (0, height - 40), (width - 1, height - 1), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, 0), (90, height - 1), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

    # 0축 강조
    if x_min <= 0.0 <= x_max:
        u0 = to_u(0.0)
        cv2.line(annotated, (u0, 0), (u0, height - 1), grid_color, 1, cv2.LINE_AA)
        cv2.putText(annotated, f"{label_x}=0m", (min(u0 + 6, width - 80), min(24, height - 6)), font, 0.5, axis_color, 1, cv2.LINE_AA)

    if z_min <= 0.0 <= z_max:
        v0 = to_v(0.0)
        cv2.line(annotated, (0, v0), (width - 1, v0), grid_color, 1, cv2.LINE_AA)
        cv2.putText(annotated, f"{label_z}=0m", (min(12, width - 80), max(v0 - 8, 20)), font, 0.5, axis_color, 1, cv2.LINE_AA)

    # 하단 X축 눈금
    for x_val in np.linspace(x_min, x_max, num=5):
        u = int(np.clip(to_u(x_val), 0, width - 1))
        cv2.line(annotated, (u, height - 14), (u, height - 2), axis_color, 1, cv2.LINE_AA)
        label = f"{x_val:.1f}m"
        text_size = cv2.getTextSize(label, font, 0.45, 1)[0]
        text_x = int(np.clip(u - text_size[0] // 2, 2, width - text_size[0] - 2))
        text_y = height - 4
        cv2.putText(annotated, label, (text_x, text_y), font, 0.45, tick_color, 1, cv2.LINE_AA)

    # 좌측 Y축(기존 Z) 눈금
    for z_val in np.linspace(z_min, z_max, num=5):
        v = int(np.clip(to_v(z_val), 0, height - 1))
        cv2.line(annotated, (6, v), (20, v), axis_color, 1, cv2.LINE_AA)
        label = f"{-z_val:.1f}m"
        text_size = cv2.getTextSize(label, font, 0.45, 1)[0]
        text_y = int(np.clip(v + text_size[1] // 2, text_size[1] + 4, height - 4))
        cv2.putText(annotated, label, (24, text_y), font, 0.45, tick_color, 1, cv2.LINE_AA)

    # 방향 안내 화살표
    cv2.arrowedLine(annotated, (width - 100, height - 28), (width - 30, height - 28), axis_color, line_thickness, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(annotated, f"+{label_x}", (width - 95, height - 34), font, 0.55, axis_color, 1, cv2.LINE_AA)

    cv2.arrowedLine(annotated, (50, 70), (50, 20), axis_color, line_thickness, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(annotated, f"-{label_z}", (58, 38), font, 0.55, axis_color, 1, cv2.LINE_AA)

    if camera_pos is not None:
        cam_x, cam_z = camera_pos
        if x_min <= cam_x <= x_max and z_min <= cam_z <= z_max:
            u = to_u(cam_x)
            v = to_v(cam_z)
            cv2.circle(annotated, (u, v), 6, (255, 200, 0), -1, cv2.LINE_AA)
            cv2.circle(annotated, (u, v), 9, (0, 0, 0), 1, cv2.LINE_AA)
            label = 'Camera'
            text_size = cv2.getTextSize(label, font, 0.45, 1)[0]
            text_x = int(np.clip(u - text_size[0] // 2, 5, width - text_size[0] - 5))
            text_y = int(np.clip(v - 10, text_size[1] + 4, height - 4))
            cv2.putText(annotated, label, (text_x, text_y), font, 0.45, (255, 220, 150), 1, cv2.LINE_AA)

    return annotated

def create_bev_with_remap_final(
    image_path,
    resolution,
    fov_x_deg,
    camera_pos_world,   # Carla 월드 좌표계 기준 카메라 위치 (X,Y,Z)
    camera_rot_world,   # Carla 월드 좌표계 기준 카메라 회전 (Pitch,Yaw,Roll)
    output_shape,
    output_filepath,
    bev_range_m=None,
    bev_world_bounds=None,
    annotate_axes=True,
    axis_labels=("X", "Z"),
    mark_camera=True
):
    """
    검증된 표준 좌표계 기반의 3D Remapping으로 BEV 이미지를 생성합니다.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"오류: 이미지를 로드할 수 없습니다: {image_path}")
        return

    img_height, img_width = img.shape[:2]
    if resolution is not None and (resolution[0] != img_width or resolution[1] != img_height):
        print(
            f"경고: 제공된 resolution {resolution}과 실제 이미지 크기 {(img_width, img_height)}가 다릅니다. 실제 크기로 내부 파라미터를 계산합니다."
        )

    # 1. 파라미터 준비
    width, height = img_width, img_height
    bev_width_pixels, bev_height_pixels = output_shape

    # 2. 카메라 내부 파라미터 (K)
    fov_x_rad = np.deg2rad(fov_x_deg)
    fx = (width / 2) / np.tan(fov_x_rad / 2)
    fy = fx
    cx = width / 2
    cy = height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # 3. Carla 좌표계 -> 표준 CV 좌표계 변환
    # Carla(UE): X(앞), Y(오른쪽), Z(위) / Pitch(Y축), Yaw(Z축), Roll(X축)
    # 표준 CV: X(오른쪽), Y(아래), Z(앞)
    axes_carla_to_cv = np.array([[0.0, 1.0, 0.0],
                                 [0.0, 0.0, -1.0],
                                 [1.0, 0.0, 0.0]])

    # 입력을 (좌우 X, 앞뒤 Y, 높이 Z) → Carla (X, Y, Z)로 변환
    cam_pos_input = np.array(camera_pos_world, dtype=np.float64)
    cam_pos_carla = np.array([-cam_pos_input[1], cam_pos_input[0], cam_pos_input[2]], dtype=np.float64)
    cam_pos_cv = axes_carla_to_cv @ cam_pos_carla

    yaw_carla = camera_rot_world[1] + 90.0
    pitch_carla = -camera_rot_world[0]
    roll_carla = camera_rot_world[2]

    rotation_ue = Rotation.from_euler(
        'ZYX',
        [yaw_carla, pitch_carla, roll_carla],
        degrees=True
    )

    R_cam_to_world_carla = rotation_ue.as_matrix()
    R_world_to_cam_carla = R_cam_to_world_carla.T
    R_world_to_cam = axes_carla_to_cv @ R_world_to_cam_carla @ axes_carla_to_cv.T

    t_world_to_cam = -R_world_to_cam @ cam_pos_cv

    # 4. 바닥(Y=0) 평면과 이미지 사이의 호모그래피 계산
    r1 = R_world_to_cam[:, 0]
    r3 = R_world_to_cam[:, 2]
    H_ground_to_img = K @ np.column_stack((r1, r3, t_world_to_cam))

    if bev_world_bounds is not None:
        (x_min, x_max), (z_min, z_max) = bev_world_bounds
        bev_width_m = x_max - x_min
        bev_height_m = z_max - z_min
    else:
        if bev_range_m is None:
            raise ValueError("bev_range_m 또는 bev_world_bounds 중 하나는 반드시 제공해야 합니다.")
        bev_width_m, bev_height_m = bev_range_m
        x_min = cam_pos_cv[0] - (bev_width_m / 2.0)
        x_max = x_min + bev_width_m
        z_max = cam_pos_cv[2] + bev_height_m
        z_min = z_max - bev_height_m

    scale_x = bev_width_m / bev_width_pixels
    scale_y = bev_height_m / bev_height_pixels

    H_bev_to_ground = np.array([
        [scale_x, 0.0, x_min],
        [0.0, -scale_y, z_max],
        [0.0, 0.0, 1.0]
    ])

    H_world_to_bev = np.linalg.inv(H_bev_to_ground)
    H_img_to_ground = np.linalg.inv(H_ground_to_img)

    flip_ground_y = np.diag([1.0, -1.0, 1.0])
    H_img_to_ground = flip_ground_y @ H_img_to_ground
    H_world_to_bev = H_world_to_bev @ flip_ground_y

    H_img_to_bev = H_world_to_bev @ H_img_to_ground

    bev_img = cv2.warpPerspective(
        img,
        H_img_to_bev,
        (bev_width_pixels, bev_height_pixels),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    camera_ground = (cam_pos_cv[0], cam_pos_cv[2])

    if annotate_axes:
        bev_img = _annotate_bev_axes(
            bev_img,
            x_min=x_min,
            x_max=x_max,
            z_min=z_min,
            z_max=z_max,
            label_x=axis_labels[0],
            label_z=axis_labels[1],
            camera_pos=camera_ground if mark_camera else None
        )

    cv2.imwrite(output_filepath, bev_img)
    print(f"\n✨ 최종 BEV 이미지가 '{output_filepath}'에 저장되었습니다.")
    return H_img_to_ground, H_img_to_bev


# --- 메인 실행 부분 ---
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="find_bev_matrix.py")
    p.add_argument("--carla_camera_position", required=True,
                   help="Carla 월드 좌표계 기준 카메라 위치 (X_carla, Y_carla, Z_carla)")
    p.add_argument("--carla_camera_rotation", required=True,
                   help="Carla 월드 좌표계 기준 카메라 회전 (Pitch_carla, Yaw_carla, Roll_carla)")
    p.add_argument("--frame_id", type=str, required=True,
                   help="프레임 ID (예: cam1)") 
    p.add_argument("--output_dir", type=str, required=True) 
    args = p.parse_args()
    
    # ==============================================================================
    # ⚙️ 사용자가 수정할 파라미터 영역
    # ==============================================================================
    # Carla 시뮬레이터에서 얻은 '있는 그대로'의 값을 입력하세요.
    # 이제 이 값을 바꾸면 기대하신 대로 정확하게 동작합니다.
    CARLA_CAMERA_POSITION = tuple(map(float, args.carla_camera_position.split(',')))
    CARLA_CAMERA_ROTATION = tuple(map(float, args.carla_camera_rotation.split(','))) 

    # 월드 절대 좌표계에서 사용할 BEV 커버리지 (X, Y)
    # 필요에 맞게 최소/최대 값을 조정하세요.
    BEV_WORLD_BOUNDS = ((-50.0, 50.0), (-50.0, 50.0))

    # 파일 경로
    INPUT_IMAGE_PATH = 'cam8_frame_000020_-45.jpg'
    OUTPUT_BEV_PATH = 'output_bev.jpg'
    OUTPUT_MAT_PATH = args.output_dir
    
    # 기타 설정
    BEV_OUTPUT_RESOLUTION = (800, 800)
    IMG_RESOLUTION = (1920, 1080)
    H_FOV_DEGREES = 80.0
    
    # ==============================================================================
    # 🛠️ 자동 계산 영역 (수정 필요 없음)
    # ==============================================================================
    H_img_to_ground, H_img_to_bev = create_bev_with_remap_final(
        image_path=INPUT_IMAGE_PATH,
        resolution=IMG_RESOLUTION,
        fov_x_deg=H_FOV_DEGREES,
        camera_pos_world=CARLA_CAMERA_POSITION,
        camera_rot_world=CARLA_CAMERA_ROTATION,
        output_shape=BEV_OUTPUT_RESOLUTION,
        output_filepath=OUTPUT_BEV_PATH,
        bev_world_bounds=BEV_WORLD_BOUNDS,
        annotate_axes=True,
        axis_labels=("X", "Y")
    )
print("\nH_img_to_ground (image -> ground):")
for row in H_img_to_ground:
    print(" ".join(f"{val:.8e}" for val in row))

os.makedirs(OUTPUT_MAT_PATH, exist_ok=True)
save_path = os.path.join(OUTPUT_MAT_PATH, f"{args.frame_id}.txt")
np.savetxt(save_path, H_img_to_ground, fmt="%.8e")

#print("\nH_img_to_bev (image -> BEV pixels):\n", H_img_to_bev)
print("\n--- 모든 작업 완료 ---")
