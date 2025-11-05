# AIRKON Realtime Result Viewer

로컬에서 포인트클라우드(`.ply`)와 라벨(`.txt`)을 로드하고, 차량 GLB 모델을 오버레이해 웹 브라우저에서 재생할 수 있는 FastAPI + Three.js 기반 뷰어입니다. 태블릿 등 브라우저만 있으면 별도 클라이언트 설치 없이 확인할 수 있습니다.

## 요구 사항

- Python 3.9+
- `fastapi`, `uvicorn`, `numpy` (동봉된 `requirements.txt` 사용)
- 전역 포인트클라우드 PLY 파일, BEV 라벨 디렉터리, 차량 GLB 파일

## 설치

```bash
pip install -r realtime_show_result/requirements.txt
```

## 실행

```bash
python realtime_show_result/server.py \
    --global-ply pointcloud/cloud_rgb_ply/cloud_rgb_9.ply \
    --bev-label-dir dataset_exmple_pointcloud_9/bev_labels
```

위 명령은 다음을 기본값으로 사용합니다.

- 차량 GLB: `pointcloud/car.glb`
- 호스트: `0.0.0.0`, 포트: `8000`
- 차량 모드: `mesh`, `--mesh-scale 1.0`, `--mesh-height 0`
- 좌표계 보정: `--flip-ply-y`, `--invert-bev-y`

추가 옵션이 필요하면 `python realtime_show_result/server.py --help`로 확인 후 덧붙이면 됩니다.

## 사용 방법

1. 서버 실행 후 터미널에 `Uvicorn running on http://0.0.0.0:8000`가 표시되면 준비 완료입니다.
2. 브라우저에서 `http://<서버 IP>:8000/`에 접속합니다.
3. 포인트클라우드와 차량 오버레이가 로드되면 상단의 재생/일시정지, 슬라이더, 이전/다음 버튼으로 프레임을 탐색할 수 있습니다.
4. 마우스 드래그 및 휠로 3D 장면을 회전/확대/이동합니다.

## 기타 참고 사항

- `--show-debug-marker` 옵션을 사용하면 첫 번째 차량 위치에 디버그용 빨간 구체가 표시됩니다.
- `--show-scene-axes` 옵션으로 격자(Grid)와 축(Axes) 표시를 켜고 끌 수 있습니다.
- 외부 네트워크에서 접속하려면 8000/TCP 포트를 공유기·방화벽에서 열어야 합니다.
