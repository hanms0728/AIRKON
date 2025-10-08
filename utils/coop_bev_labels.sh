# 우리 데이터셋이 100장이라 아래처럼 반복
for i in $(seq -w 0 99)
do
    FRAME=$(printf "%06d_-45" "$i")
    echo "Processing frame ${FRAME}..."

  python coop_bev_labels.py \
    --pred_dir /inference_dataset/bev_labels \
    --gt_dir /val_dataset45/regions \
    --frame "$FRAME" \
    --xlim -120 -30 \
    --ylim -80 40 \
    --out_prefix /coop_bev_global/frame
done

echo "Done."