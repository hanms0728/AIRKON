# YOLO11 2.5D Polygon Trainer (`src/train.py`)

`src/train.py`는 Ultralytics YOLO11 백본 위에 삼각형 기반 2.5D 헤드를 얹어 차량 지붕/물체 평행사변형을 학습하는 스크립트입니다. 멀티 클래스, 혼합 정밀도, 자동 체크포인트/ONNX 내보내기, 선택적 데이터 증강과 Focal Loss 등을 모두 포함한 완전 버전입니다.

---

## 1. 데이터 준비

라벨 파일은 YOLO 스타일의 텍스트 (`class p0x p0y p1x p1y p2x p2y`) 형식이어야 하며, 아래 두 가지 디렉터리 구조 중 하나를 따르면 됩니다.

### A. Flat 구조
```
/dataset_root
 ├── images/
 │    ├── 0001.jpg
 │    └── 0002.jpg
 └── labels/
      ├── 0001.txt   # class p0x p0y p1x p1y p2x p2y
      └── 0002.txt
```

### B. 멀티 폴더 구조 (카메라/시퀀스별 서브폴더)
```
/dataset_root
 ├── cam01/
 │    ├── images/
 │    └── labels/
 ├── cam02/
 │    ├── images/
 │    └── labels/
 └── ...
```

`--train-root`/`--val-root`를 각각 위 구조의 최상위 폴더로 넘기면 스크립트가 자동으로 감지합니다.

---

## 2. 주요 기능

- **멀티 클래스**: 라벨 첫 열(class id)을 그대로 사용하며, `--num-classes`로 모델/손실을 설정합니다.
- **커스텀 손실**: 삼각형 오프셋 + Chamfer distance 기반 `Strict2_5DLoss`, 선택적 분류 Focal Loss(`--cls-focal`).
- **데이터 증강**: `--train-augment` 플래그를 켜면 Albumentations 기반 flip/brightness/crop/perspective/noise 를 적용합니다.
- **검증/로깅**: mAP/precision/recall/mAOE + 클래스별 지표, epoch마다 PTH/CKPT/CSV 저장, 옵션에 따라 ONNX 내보내기.

---

## 3. 실행 예시

### 3.1 단일 클래스 (차량만) 학습
```bash
python -m src.train \
  --train-root /data/car_only/train \
  --val-root /data/car_only/val \
  --yolo-weights yolo11m.pt \
  --save-dir ./runs/car_only \
  --epochs 60 --batch 4 --img-h 864 --img-w 1536
```
- 기본값이 1클래스이므로 `--num-classes`를 줄 필요가 없습니다.
- 증강을 쓰고 싶다면 `--train-augment`를 추가하세요.

### 3.2 멀티 클래스 (예: 차량/콘/박스) 학습
```bash
python -m src.train \
  --train-root /data/multi/train \
  --val-root /data/multi/val \
  --yolo-weights yolo11m.pt \
  --save-dir ./runs/multi_cls \
  --num-classes 3 \
  --cls-focal --focal-gamma 2.0 --focal-alpha 0.25 \
  --train-augment \
  --epochs 80 --batch 4 --img-h 864 --img-w 1536
```
- 라벨의 클래스 ID 범위(0~2)에 맞춰 `--num-classes 3` 지정.
- 소수 클래스 대응을 위해 Focal Loss를 권장합니다.

---

## 4. 주요 CLI 옵션

| 옵션 | 설명 |
| --- | --- |
| `--train-root`, `--val-root` | 데이터셋 경로 (Flat/멀티폴더 자동 감지). `--val-root`를 생략하면 train과 같은 경로에서 `images/labels` 사용 |
| `--yolo-weights` | 초기 YOLO11 가중치 (`yolo11m.pt` 등) |
| `--num-classes` | 클래스 수 (기본 1). 라벨의 최대 class id + 1과 일치시켜야 함 |
| `--train-augment` | Albumentations 증강을 켭니다 (설치 필요) |
| `--cls-focal`, `--focal-gamma`, `--focal-alpha` | 분류 헤드에 Focal Loss 적용 및 파라미터 조정 |
| `--onnx-dir`, `--onnx-opset` | ONNX 저장 경로/OPSET 설정 (매 epoch + best 모델을 자동 저장) |
| 기타 | 배치/에폭/러닝레이트/Freeze epoch 등은 기본 argparse 옵션 참조 |

> **주의**: `--train-augment`를 사용할 경우 `pip install albumentations opencv-python`이 되어 있어야 합니다.

---

## 5. 결과물

`--save-dir` (기본 `./runs/2p5d`) 아래에 다음이 생성됩니다.

- `pth/`: 매 epoch마다 저장되는 PyTorch 가중치 + `*_best.pth`.
- `ckpt/`: 옵티마/스케줄러/Scaler가 포함된 체크포인트.
- `*.csv`: 에폭별 손실/지표 로그.
- `onnx/`: 각 epoch 및 best 모델의 ONNX 내보내기 결과.

---

## 6. 기타 팁

- 증강과 Focal Loss는 옵션이며, 켠/끈 버전을 각각 짧게 검증해 가장 안정적인 조합을 선택하세요.
- 멀티 클래스일 때 검증 로그에 `[clsN: P=..., R=..., mAP=...]` 형태로 클래스별 지표가 출력되므로, 특정 클래스가 부족한지 쉽게 확인할 수 있습니다.
- 라벨 포맷이 잘못되거나 짝이 맞지 않으면 스크립트가 자동으로 해당 샘플을 건너뜁니다. 로그를 통해 매칭된 샘플 수를 확인하세요.
