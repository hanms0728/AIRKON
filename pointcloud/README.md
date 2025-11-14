# Point Cloud 도구 모음
CCARLA 기반 시뮬레이션에서 생성한 글로벌 포인트클라우드(`global_fused_small.ply`)와
프레임별 RGB/레이블을 활용해 **가시 영역 포인트클라우드 추출**, **3D 박스/GLB 메쉬 오버레이**,  
**결과 비교**를 수행하기 위한 스크립트와 샘플 데이터가 들어있는 폴더입니다.

## 디렉터리 구성
- `overlay_obj_on_ply.py` : 글로벌 PLY 위에 BEV 라벨과 차량 메쉬(GLB)를 실시간으로 오버레이하는 뷰어.
- `make_pointcloud/`
  - `img_to_ply_with_global_ply.py` : 단일 카메라 뷰에서 보이는 포인트만 추출하고 RGB 텍스처를 입혀 저장.
  - `compare_ply.py` : 두 개의 PLY를 정렬(ICP) 후 RMSE/Chamfer Distance로 품질 비교.
  - `view_pointcloud.py` : 저장된 PLY를 불러 간단히 시각화.
  - `inspect_npz.py` : LUT/포인트클라우드 NPZ의 키와 통계를 요약 출력.
- `cloud_rgb_npz/` : 압축된 xyz/rgb 배열(.npz) 샘플.
- `cloud_rgb_ply/` : 프레임별로 미리 변환된 컬러 포인트클라우드(.ply).
- `global_fused_small.ply` : 전체 환경을 합친 글로벌 포인트클라우드.
- `car.glb` : 오버레이에 사용하는 차량 메쉬.

## 주요 스크립트 사용법

### 1. 글로벌 PLY + 박스/메쉬 오버레이
BEV 박스 라벨(`*.txt`)과 차량 GLB를 글로벌 포인트클라우드 위에 실시간으로 렌더링합니다.

```bash
python pointcloud/overlay_obj_on_ply.py \
  --global-ply pointcloud/cloud_rgb_ply/cloud_rgb_9.ply \
  --bev-label-dir /path/to/bev_labels \
  --vehicle-glb pointcloud/car.glb \
  --fps 15 --play --bg-dark \
  --estimate-z --invert-bev-y \
  --force-legacy 
```

- 좌표계가 반대일 때 `--invert-bev-y`, PLY 축이 다를 때 `--invert-ply-y` 를 사용합니다.

### 2. 단일 카메라 뷰 포인트클라우드 추출
RGB 이미지와 카메라 포즈를 이용해 글로벌 포인트클라우드에서 보이는 점만 필터링하고 색을 입혀 저장합니다.

```bash
python pointcloud/make_pointcloud/img_to_ply_with_global_ply.py \
  --ply pointcloud/global_fused_small.ply \
  --image real_image/cam_1.png \
  --out outputs/visible_cam9.ply \
  --pos="30.0,2.0,10.0" \
  --rot="-55.0,-35.0,0.0" \
  --fov 89 \
  --interactive --intrinsic-npz realtime/disto/cam_calibration_results.npz
```

- `--pos`는 카메라 위치(x,y,z), `--rot`는 yaw,pitch,roll (deg) 입니다.
- `--save-frame carla|cv`로 결과 좌표계를 선택할 수 있습니다.
- `--interactive`를 켜면 키보드 화살표/WASD와 HUD를 통해 뷰를 조정하면서 실시간으로 확인할 수 있습니다.
- `--live-save-dir`를 지정하면 각 키 프레임을 즉시 PLY로 저장합니다.

### 3. 포인트클라우드 정합 및 품질 비교
샘플 포인트클라우드와 기준 데이터를 재정렬한 뒤 RMSE와 Chamfer Distance를 출력합니다.

```bash
python pointcloud/make_pointcloud/compare_ply.py \
  --src outputs/visible_cam9.ply \
  --tgt pointcloud/cloud_rgb_ply/cloud_rgb_9.ply \
  --voxel 0.05 \
  --threshold 0.2
```

- 정렬된 결과는 Open3D 뷰어로 확인할 수 있습니다(빨강=결과, 초록=기준).

### 4. 빠른 PLY 뷰어
간단히 좌표계를 플립하여 결과를 확인할 때 사용합니다.

```bash
python pointcloud/make_pointcloud/view_pointcloud.py
```

스크립트 내부의 `ply_path`를 원하는 파일로 교체하세요.


### 5. NPZ 구조 뷰어
LUT/포인트클라우드 NPZ의 키와 기본 통계를 확인합니다.

```bash
python pointcloud/make_pointcloud/inspect_npz.py pointcloud/cloud_rgb_npz/cloud_rgb_1.npz \
  --verbose --percentiles --sample 5
```

### 6. LUT valid_mask 편집기
기존에 생성된 `pixel2world_lut.npz`의 `valid_mask`를 사람이 직접 GUI에서 수정할 수 있습니다.

```bash
# RGB 이미지 위에서 편집하고 싶다면:
python pointcloud/make_pointcloud/lut_mask_editor.py \
  --lut outputs/cam_1_x-6.50_y10.60_z12.10_yaw-148.00_pit-34.00_rol-6.00_f84.08_pixel2world_lut.npz \
  --image real_image/cam_1.png

# 또는 글로벌 PLY만으로 BEV 배경을 만들고 싶다면(이미지 불필요):
python pointcloud/make_pointcloud/lut_mask_editor.py \
  --lut outputs/cam_1_x-6.50_y10.60_z12.10_yaw-148.00_pit-34.00_rol-6.00_f84.08_pixel2world_lut.npz \
  --global-ply outputs/cam_1_x-6.50_y10.60_z12.10_yaw-148.00_pit-34.00_rol-6.00_f84.08.ply
```

- `Left drag` : 유효하지 않은 영역으로 마스킹 (invalid), `Right drag` : 다시 유효로 되돌리기.
- `[ / ]` 로 브러시 크기를 조절하고, `U` (undo), `R` (reset), `S` (save), `ESC` 로 종료합니다.
- 기본적으로 `valid_mask`와 `ground_valid_mask`에 동일한 결과를 저장하며, `--sync-floor-mask`로 `floor_mask`도 함께 갱신할 수 있습니다.
- `--background` 는 `auto`(기본, 가능하면 PLY→BEV 사용), `image`, `bev`, `checker` 중 선택 가능하며, `--bev-resolution`, `--bev-pad`, `--ply-voxel`로 BEV 품질/다운샘플을 조정할 수 있습니다.
- `--global-ply`를 주면 실제 카메라 이미지는 필요 없고, LUT의 XY를 PLY와 매칭한 **순수 BEV 뷰** 위에서 직접 마스크를 수정할 수 있습니다 (마우스 좌표가 BEV 픽셀에 바로 매핑됩니다).
- `--out` 경로를 명시하지 않으면 입력 파일 이름에 `_edited.npz`가 붙은 새 파일로 저장되고, `--inplace`를 주면 기존 LUT를 덮어씁니다.

## 예제 데이터
`cloud_rgb_npz/`와 `cloud_rgb_ply/`는 파이프라인을 빠르게 검증하기 위한 샘플입니다.  
자체 데이터셋으로 대체할 경우 동일한 파일 구조(프레임별 `.npz` 혹은 `.ply`)와 BEV 라벨 포맷(`class cx cy cz length width yaw pitch roll`)을 맞춰 주세요.

`cloud_rgb_npz/cloud_rgb_*.npz`는 다음 키를 포함합니다.
- `X`, `Y`, `Z` : 각 픽셀의 월드 좌표(float32 H×W).
- `valid_mask`, `floor_mask`, `ground_valid_mask` : 유효 픽셀을 나타내는 0/1 마스크(uint8 H×W, 없으면 코드가 `isfinite(X)&isfinite(Y)`를 사용).
- `K` : 3×3 카메라 내접행렬(float32).
- `cam_pose` : `[x, y, z, yaw, pitch, roll]` float32 벡터.
- `width`, `height` : 원본 해상도(int 스칼라).
- `fov` : 시야각(float32 스칼라).
- `ray_model`, `sem_channel` : 문자열 메타정보.
- `floor_ids` : 바닥 클래스를 나타내는 int 배열.
- `M_c2w` : 카메라→월드 4×4 변환행렬(float64).
