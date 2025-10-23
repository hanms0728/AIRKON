아래처럼 더미데이터로 UDP 서버와 리얼타임 추론을 확인할 수 있음
calib/cam1.txt ... 필요
Python udp_server.py --save-dir=./logs --udp-port 50050 --fps 10
python realtime_edge_infer.py --dummy-cam-dirs cam1=val_dataset_2\val_dataset_2\cam1\images,cam2=val_dataset_2\val_dataset_2\cam2\images,cam3=val_dataset_2\val_dataset_2\cam3\images,cam4=val_dataset_2\val_dataset_2\cam4\images,cam5=val_dataset_2\val_dataset_2\cam5\images,cam6=val_dataset_2\val_dataset_2\cam6\images --dummy-fps 1 --udp-enable --udp-host 127.0.0.1 --udp-port 50050 --weights yolo11m_2_5d_epoch_005.onnx --output-dir ./inference_results_realtime --img-size 864,1536 --strides 8,16,32 --conf 0.80 --nms-iou 0.20 --topk 50 --score-mode obj*cls --calib-dir ./calib --every-n 2 --transport tcp
---
실제 실행 방식
python realtime_edge_infer.py --udp-enable --udp-host 127.0.0.1 --udp-port 50050 --udp-format json --weights yolo11m_2_5d_epoch_005.onnx --output-dir ./inference_results_realtime --img-size 864,1536 --strides 8,16,32 --conf 0.80 --nms-iou 0.20 --topk 50 --score-mode obj*cls --calib-dir ./calib --every-n 2 --transport tcp
ㄴ컴1개로 서버엣지돌릴때
ㄴㄴ컴두대면 udp-host에 서버컴 ip넣기