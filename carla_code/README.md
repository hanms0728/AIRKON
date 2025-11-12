# Carla를 실행할때 사용하는 코드 모듬

## 디렉터리 구성
- `carla_uitl/` : 잡다한 유틸 코드모음
  - `change_map.py` : carla 맵 변경
  - `delete_list.py` : carla 맵에 있는 정적 객체 제거
  - `no_rendering_mode.py` : 현재 calra 상태를 bev 조감도로 표시
  - `free_fly_virtual_cam.py` : 현재 calra에서 자유롭게 이동 가능한 카메라 생성하는 코드
- `make_dataset/` : carla로 dataset 만드는 코드모음
  - `ground_point_coshow.py` : carla로 dataset 수집하는 코드, 카메라 위치, 차량, 대수 등 설정해야함
- `make_pointcloud/` : calra 맵에서 ply, npz 생성 코드모음
- `make_pixel2world_lut.py` : 하나의 카메라에 대해서 ply, npz 파일 생성 코드
- `multi_cam_pixel2world_and_fuse_inline.py` : 여러대 카메라에 대해서 ply, npz 파일 생성 및 통합(global_map) ply를 생성하는 코드
