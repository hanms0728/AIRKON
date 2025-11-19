# auto_labels

자동으로 라벨링을 만들어주는 코드

사용법

```bash
python ./auto_labels/inference_onnx_eval.py \
  --onnx 2_5d_논문모델.onnx \
  --input-dir ./이미지가_들어가_있는_폴더 \
  --output-dir ./결과저장폴더 \
  --half
```

2_5d_논문모델.onnx는 파일이 큰 관계로 google 드라이브 링크로 대체
📦 [2_5d_논문모델.onnx 다운로드 (Google Drive)](https://drive.google.com/file/d/1LIwblnHQqDgwWvYt6Xyo6V6TvTt6o7Ul/view?usp=sharing)



dataset_root_폴더 구조

```bash
     ├─ scene_001/
     │   ├─ images/
     │   └─ labels/
     ├─ scene_002/
     │   ├─ images
     │   └─ labels
```

모델 학습 방법

```bash
python ./src/train_lstm_onnx.py \
  --train-root /dataset_root_경로 --val-root /검증_root_경로 --data-layout auto \
  --val-mode metrics --val-interval 1 --val-batch 2 \
  --yolo-weights yolo11m.pt \
  --weights ./onnx/base_pth/yolo11m_2_5d_epoch_005.pth \
  --temporal lstm --temporal-hidden 256 --temporal-layers 1 \
  --temporal-on-scales last \
  --seq-len 6 --seq-stride 2 --seq-grouping auto \
  --temporal-reset-per-batch --tbptt-detach \
  --start-epoch 5 --batch 6 \
  --save-dir runs/2_5d_lstm_real_coshow_v3
```



## 권장 워크플로
1. 이미지 왜곡 보정: `utils/distort/batch_undistort.py` 실행  
   - 카메라 IP에 맞게 `--calib`를 지정 (예: 21번 IP → `realtime/disto/cam21_calibration_results_params.npz`)  
   - IP별로 이미지를 분리해 각 root 폴더를 나누고, 해당 root를 `--root` 인자로 전달  
   - 보정 결과는 `root__undist`에 생성됨
2. 자동 라벨 생성: `auto_labels/inference_onnx_eval.py`로 추론  
   - 필요 시  `auto_labels/detection_inference_onnx.py`의 `conf_thres` 조정, 부분 라벨 손수 수정
3. 학습: `src/train_lstm_onnx.py` 실행  
   - 학습/검증 데이터 루트를 위 구조에 맞춰 준비 후 학습 진행