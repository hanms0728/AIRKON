
for i in $(seq -w 0 499)
do
    FRAME=$(printf "%06d" "$i")
    echo "Processing frame ${FRAME}..."

  python coop_bev_labels.py \
    --pred_dir /inference_dataset/bev_labels \
    --gt_dir /val_dataset_2/regions \
    --frame "$FRAME" \
    --xlim -120 -30 \
    --ylim 40 -80 \
    --out_prefix /coop_bev_global/frame \
    --bg_path 칼라맵.png \
    --bg_alpha 0.6
done

echo "Done."