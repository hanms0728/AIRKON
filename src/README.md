# Temporal ONNX Inference (BEV 비교)

`src/` 디렉터리에는 카메라 시퀀스에 ConvLSTM 기반 ONNX 모델을 적용하는 두 가지 진입점이 있습니다.  
두 스크립트 모두 2D 감지 결과를 생성하지만 **BEV 투영 방식과 결과 라벨링**이 다릅니다.

## BEV 출력 차이 요약

| 구분 | `inference_lstm_onnx.py` | `inference_lstm_onnx_pointcloud.py` |
| --- | --- | --- |
| BEV 좌표 변환 | 프레임별 **3x3 Homography**(img→ground)를 곱해 2D 평면으로 투영 | 픽셀당 `(X,Y,Z)`를 담은 **pixel2world LUT(.npz)**를 bilinear 샘플링 |
| 필수 추가 준비물 | `--calib-dir` 안에 이미지 이름과 매칭되는 `*.npy`/`*.txt`/`*.csv` Homography | `--lut-path`로 지정하는 `pixel2world_lut.npz` (필수 키: `X`, `Y`, `Z`, 선택 마스크 `ground_valid_mask` 등) |
| GT BEV 변환 | GT 삼각형도 Homography로 투영 (3×3 행렬 유효 영역 내에서만) | GT 픽셀 좌표를 LUT로 변환, 유효 마스크에 맞춰 필터링 |
| BEV 라벨 포맷 | `class cx cy length width yaw` (6개 값, 2D 평면) | 기본: `class cx cy cz length width yaw pitch roll` (9개 값, 3D) <br>옵션 `--bev-label-3d --no-bev-label-3d`로 6/9값 선택 |
| 높이/기울기 추정 | 제공되지 않음 (cz/pitch/roll 항상 0) | LUT에서 얻은 Z로 차량 높이 중심(`cz`), pitch/roll 계산. `--use-roll`, `--pitch-clamp-deg` 등 제어 |
| 비정상 박스 거르기 | Homography 결과에 대한 별도 필터 없음 | `_sane_dims`로 길이/너비/비율 범위(`--min-length` 등) 확인 후 제외 |

## 준비물 상세

- **Homography 방식 (`inference_lstm_onnx.py`)**
  - `--calib-dir` 예시: `cam01_calib/` 폴더에 `frame_000123.txt` 형태 3x3 행렬 파일.
  - Homography가 없으면 BEV 관련 아웃풋(`bev_labels`, `bev_images*`)은 생성되지 않습니다.

- **LUT 방식 (`inference_lstm_onnx_pointcloud.py`)**
- `pixel2world_lut.npz`는 최소 `X`, `Y`, `Z` 2D 배열이 있어야 하며, 유효 마스크(`ground_valid_mask`, `valid_mask`, `floor_mask`)가 없을 경우 `isfinite(X)&isfinite(Y)`로 자동 대체됩니다.
  - LUT는 카메라별/해상도별로 상이하므로 입력 영상과 동일한 해상도 기준으로 생성해야 합니다.
  - 모든 프레임에서 LUT 하나를 재사용하며, 투영 실패 픽셀은 자동으로 제외됩니다.

## 출력 라벨 차이

- `inference_lstm_onnx.py`
  - `output_dir/bev_labels/<frame>.txt`
  - 각 줄: `class cx cy length width yaw`
  - `cx, cy`: BEV 평면 중심 (Homography로 축척된 단위), `yaw`: [-180,180) 도.

- `inference_lstm_onnx_pointcloud.py`
  - `output_dir/bev_labels/<frame>.txt`
  - 기본(3D): `class cx cy cz length width yaw pitch roll`
    - `cz`: LUT 기반 평균 높이, `pitch/roll`: LUT Z를 이용해 계산.
  - `--no-bev-label-3d` 사용 시 2D 포맷으로 저장.

## 실행 예시 (핵심 옵션만)

```bash
# Homography 기반 BEV
python src/inference_lstm_onnx.py \
  --input-dir data/cam01/frames \
  --weights weights/model.onnx \
  --calib-dir data/cam01/calib_img2ground \
  --output-dir runs/onnx_homo

# LUT 기반 3D BEV
python src/inference_lstm_onnx_pointcloud.py \
  --input-dir data/cam01/frames \
  --weights weights/model.onnx \
  --lut-path data/cam01/pixel2world_lut.npz \
  --output-dir runs/onnx_lut \
  --bev-label-3d \
  --use-roll
```

더 자세한 실행 코드는 각각의 코드 마지막에 적혀있음.

두 실행 모두 2D 검출 시각화는 동일하게 생성되지만, BEV 라벨과 시각화는 위와 같이 상이한 준비물과 좌표 정의에 따라 결과가 달라집니다.
