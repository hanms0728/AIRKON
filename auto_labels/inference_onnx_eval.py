# inference_onnx_eval.py
# ONNX Inferencer로 폴더 단위 추론 + GT와 평가/시각화/라벨 저장
# 좌표계: 예측/GT/시각화 모두 "원본 이미지 해상도" 기준
# - Pred: 빨간색, GT: 초록색, IoU 텍스트: 빨간색
# - 단일 IoU 임계값 평가(Precision/Recall/F1/mean IoU/mAOE)
# - 예측에 score가 있으면 mAP@0.50 계산, 없으면 경고

import os
import cv2
import math
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple

from tqdm import tqdm  # <-- progress bar
import detection_inference_onnx as det_onnx  # 수정된 모듈 (run -> (N,7) 반환)

# ---------------------------
# 라벨 로더 (GT: 원본 해상도 기준)
# ---------------------------
def load_gt_triangles(label_path: str) -> np.ndarray:
    """GT 포맷: 'cls p0x p0y p1x p1y p2x p2y' (한 줄당 1개)"""
    if not os.path.isfile(label_path):
        return np.zeros((0, 3, 2), dtype=np.float32)
    tris = []
    with open(label_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 7:
                continue
            _, x0, y0, x1, y1, x2, y2 = p
            tris.append([[float(x0), float(y0)],
                         [float(x1), float(y1)],
                         [float(x2), float(y2)]])
    if not tris:
        return np.zeros((0, 3, 2), dtype=np.float32)
    return np.asarray(tris, dtype=np.float32)

# ---------------------------
# 폴리곤/IoU/각도 유틸 (원본 좌표계)
# ---------------------------
def order_poly_ccw(poly4: np.ndarray) -> np.ndarray:
    c = poly4.mean(axis=0)
    ang = np.arctan2(poly4[:,1] - c[1], poly4[:,0] - c[0])
    idx = np.argsort(ang)
    return poly4[idx]

def parallelogram_from_pred_triangle(tri_pred: np.ndarray) -> np.ndarray:
    """tri_pred: [cx,cy,f1x,f1y,f2x,f2y,(score?)] -> (4,2) float32(CCW)"""
    coords = tri_pred[:6].astype(np.float32)
    cx, cy, x2, y2, x3, y3 = coords.tolist()
    x2m, y2m = 2*cx - x2, 2*cy - y2
    x3m, y3m = 2*cx - x3, 2*cy - y3
    poly = np.array([[x2, y2],
                     [x3, y3],
                     [x2m, y2m],
                     [x3m, y3m]], dtype=np.float32)
    return order_poly_ccw(poly)

def parallelogram_from_gt_triangle(tri_gt: np.ndarray) -> np.ndarray:
    """GT: (p0,p1,p2) -> pred 방식과 동일하게 p0 중심 대칭으로 구성"""
    p0, p1, p2 = tri_gt[0].astype(np.float32), tri_gt[1].astype(np.float32), tri_gt[2].astype(np.float32)
    p1m = 2.0 * p0 - p1
    p2m = 2.0 * p0 - p2
    poly = np.stack([p1, p2, p1m, p2m], axis=0).astype(np.float32)
    return order_poly_ccw(poly)

def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:,0], poly[:,1]
    return float(abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1))) * 0.5)

def iou_polygon(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    try:
        inter_area, _ = cv2.intersectConvexConvex(poly_a.astype(np.float32),
                                                  poly_b.astype(np.float32))
        if inter_area <= 0 or not np.isfinite(inter_area):
            return 0.0
        ua = polygon_area(poly_a); ub = polygon_area(poly_b)
        union = ua + ub - inter_area
        if union <= 0 or not np.isfinite(union):
            return 0.0
        return float(np.clip(inter_area / union, 0.0, 1.0))
    except Exception:
        xa1, ya1 = poly_a[:,0].min(), poly_a[:,1].min()
        xa2, ya2 = poly_a[:,0].max(), poly_a[:,1].max()
        xb1, yb1 = poly_b[:,0].min(), poly_b[:,1].min()
        xb2, yb2 = poly_b[:,0].max(), poly_b[:,1].max()
        inter = max(0.0, min(xa2, xb2)-max(xa1, xb1)) * max(0.0, min(ya2, yb2)-max(ya1, yb1))
        ua = max(0.0, xa2-xa1) * max(0.0, ya2-ya1)
        ub = max(0.0, xb2-xb1) * max(0.0, yb2-yb1)
        union = ua + ub - inter
        if union <= 0:
            return 0.0
        return float(np.clip(inter / union, 0.0, 1.0))

def angle_deg(v: np.ndarray) -> float:
    return math.degrees(math.atan2(float(v[1]), float(v[0])))

def angular_error_deg(v_pred: np.ndarray, v_gt: np.ndarray) -> float:
    d = abs(angle_deg(v_pred) - angle_deg(v_gt)) % 360.0
    return 360.0 - d if d > 180.0 else d

# ---------------------------
# 시각화/라벨 저장 (원본 이미지에 그림)
# ---------------------------
def draw_pred_only(img_bgr_orig: np.ndarray,
                   preds_tri_orig: np.ndarray,
                   save_path_img: str,
                   save_path_txt: str,
                   include_score: bool = False) -> None:
    os.makedirs(os.path.dirname(save_path_img), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_txt), exist_ok=True)
    img = img_bgr_orig.copy()
    lines = []
    for tri in preds_tri_orig:
        poly = parallelogram_from_pred_triangle(tri).astype(np.int32)
        cv2.polylines(img, [poly], True, (0,0,255), 2)  # red
        cx, cy, f1x, f1y, f2x, f2y = tri[:6].tolist()
        if include_score and tri.shape[0] >= 7:
            score = float(tri[6])
            lines.append(f"0 {cx:.2f} {cy:.2f} {f1x:.2f} {f1y:.2f} {f2x:.2f} {f2y:.2f} {score:.4f}")
        else:
            lines.append(f"0 {cx:.2f} {cy:.2f} {f1x:.2f} {f1y:.2f} {f2x:.2f} {f2y:.2f}")
    cv2.imwrite(save_path_img, img)
    with open(save_path_txt, "w") as f:
        f.write("\n".join(lines))

def draw_pred_with_gt(img_bgr_orig: np.ndarray,
                      preds_tri_orig: np.ndarray,
                      gt_tris_orig: np.ndarray,
                      save_path_img_mix: str,
                      draw_iou_text: bool=True) -> None:
    os.makedirs(os.path.dirname(save_path_img_mix), exist_ok=True)
    img = img_bgr_orig.copy()

    # GT (green)
    for g in gt_tris_orig:
        poly_g = parallelogram_from_gt_triangle(g).astype(np.int32)
        cv2.polylines(img, [poly_g], True, (0,255,0), 2)

    # Pred (red)
    for tri in preds_tri_orig:
        poly_p = parallelogram_from_pred_triangle(tri).astype(np.int32)
        cv2.polylines(img, [poly_p], True, (0,0,255), 2)

    if draw_iou_text and len(gt_tris_orig) > 0 and len(preds_tri_orig) > 0:
        matched = np.zeros((len(gt_tris_orig),), dtype=bool)
        for tri in preds_tri_orig:
            poly_p = parallelogram_from_pred_triangle(tri)
            cx, cy = int(round(tri[0])), int(round(tri[1]))
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gt_tris_orig):
                if matched[j]:
                    continue
                poly_g = parallelogram_from_gt_triangle(g)
                iou = iou_polygon(poly_p, poly_g)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                matched[best_j] = True
            cv2.putText(img, f"IoU {best_iou:.2f}",
                        (cx, cy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)  # red text

    cv2.imwrite(save_path_img_mix, img)

# ---------------------------
# 단일 운영점 평가
# ---------------------------
def evaluate_dataset(preds_by_name: Dict[str, np.ndarray],
                     gts_by_name: Dict[str, np.ndarray],
                     iou_thr: float=0.5) -> Dict[str, float]:
    TP = FP = FN = 0
    ious_all: List[float] = []
    aoe_all:  List[float] = []

    for name, preds in preds_by_name.items():
        gts = gts_by_name.get(name, np.zeros((0,3,2), dtype=np.float32))
        if len(preds) == 0 and len(gts) == 0:
            continue
        matched = np.zeros((len(gts),), dtype=bool)

        for tri in preds:
            poly_p = parallelogram_from_pred_triangle(tri)
            v_pred = np.array([tri[4]-tri[2], tri[5]-tri[3]], dtype=np.float32)  # f2 - f1

            best_iou, best_j, best_v_gt = 0.0, -1, None
            for j, g in enumerate(gts):
                if matched[j]:
                    continue
                poly_g = parallelogram_from_gt_triangle(g)
                iou = iou_polygon(poly_p, poly_g)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
                    best_v_gt = np.array([g[2,0]-g[1,0], g[2,1]-g[1,1]], dtype=np.float32)

            if best_j >= 0 and best_iou >= iou_thr:
                matched[best_j] = True
                TP += 1
                ious_all.append(float(np.clip(best_iou, 0.0, 1.0)))
                if np.linalg.norm(v_pred) > 1e-6 and best_v_gt is not None and np.linalg.norm(best_v_gt) > 1e-6:
                    aoe_all.append(angular_error_deg(v_pred, best_v_gt))
            else:
                FP += 1

        FN += int((~matched).sum())

    precision = TP / max(TP + FP, 1e-9)
    recall    = TP / max(TP + FN, 1e-9)
    f1        = 2*precision*recall / max(precision + recall, 1e-9)
    mean_iou  = float(np.mean(ious_all)) if ious_all else 0.0
    mAOE_deg  = float(np.mean(aoe_all))  if aoe_all else 0.0
    return dict(precision=precision, recall=recall, f1=f1, mean_iou=mean_iou, mAOE_deg=mAOE_deg)

# ---------------------------
# AP@0.50 (점수 필요)
# ---------------------------
def compute_ap50_with_scores(preds_by_name, gts_by_name, iou_thr=0.5):
    """
    preds_by_name[name]: (N,7) [cx,cy,f1x,f1y,f2x,f2y,score]
    gts_by_name[name]:   (M,3,2)
    """
    all_preds = []
    total_gts = 0
    for name, preds in preds_by_name.items():
        gts = gts_by_name.get(name, np.zeros((0,3,2), np.float32))
        total_gts += gts.shape[0]
        if preds.size == 0:
            continue
        if preds.shape[1] >= 7:
            for row in preds:
                coords6 = row[:6].astype(np.float32)
                score = float(row[6])
                all_preds.append((name, score, coords6))

    if total_gts == 0 or len(all_preds) == 0:
        return 0.0

    all_preds.sort(key=lambda e: -e[1])  # by score desc
    matched = {name: np.zeros((gt.shape[0],), dtype=bool) for name, gt in gts_by_name.items()}

    tp, fp = [], []
    for name, score, coords6 in all_preds:
        gts = gts_by_name.get(name, np.zeros((0,3,2), np.float32))
        if gts.shape[0] == 0:
            fp.append(1); tp.append(0)
            continue

        poly_p = parallelogram_from_pred_triangle(coords6)
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if matched[name][j]:
                continue
            poly_g = parallelogram_from_gt_triangle(g)
            iou = iou_polygon(poly_p, poly_g)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_thr and best_j >= 0:
            matched[name][best_j] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)

    tp = np.asarray(tp, np.float32)
    fp = np.asarray(fp, np.float32)
    if tp.size == 0:
        return 0.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(total_gts, 1e-9)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = float(np.trapz(precisions, recalls))
    return ap

# ---------------------------
# Main
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser("ONNX Inference (+Eval/+Vis) with progress")
    ap.add_argument("--onnx", type=str, required=True, help="ONNX 모델 경로")
    ap.add_argument("--input-dir", type=str, required=True, help="이미지 폴더")
    ap.add_argument("--output-dir", type=str, default="./onnx_results", help="결과 저장 루트")
    ap.add_argument("--img-size", type=str, default="832,1440", help="Inferencer 내부 letterbox H,W (모델과 일치)")
    ap.add_argument("--gt-label-dir", type=str, default=None, help="GT 라벨 폴더(원본 해상도 기준)")
    ap.add_argument("--eval-iou-thr", type=float, default=0.5, help="평가 IoU 임계값")
    ap.add_argument("--exts", type=str, default=".jpg,.jpeg,.png")
    ap.add_argument("--half", action="store_true", help="ONNX half-precision 모델인 경우 지정")
    ap.add_argument("--log-every", type=int, default=200, help="N장마다 진행 로그 남김")
    ap.add_argument("--txt-include-score", action="store_true",
                help="TXT 라벨에 score(7번째 값)까지 저장하려면 지정. 미지정 시 저장 안 함")
    return ap.parse_args()

def main():
    args = parse_args()
    H_l, W_l = map(int, args.img_size.split(","))  # for Inferencer letterbox

    # 폴더 준비
    out_img_dir = os.path.join(args.output_dir, "images_gt")
    out_lab_dir = os.path.join(args.output_dir, "labels")
    out_mix_dir = os.path.join(args.output_dir, "images_with_gt")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lab_dir, exist_ok=True)
    do_eval = args.gt_label_dir and os.path.isdir(args.gt_label_dir)
    if do_eval:
        os.makedirs(out_mix_dir, exist_ok=True)

    # 이미지 목록
    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    names = [f for f in sorted(os.listdir(args.input_dir)) if f.lower().endswith(exts)]

    # Inferencer (run -> (N,7): 6좌표+score, 원본 해상도 좌표)
    inferencer = det_onnx.Inferencer(
        model_path=args.onnx,
        letterbox_size=(H_l, W_l),
        is_half=args.half
    )

    preds_by_name: Dict[str, np.ndarray] = {}
    gts_by_name:   Dict[str, np.ndarray] = {}

    print(f"[ONNX] imgs={len(names)}, eval={bool(do_eval)}, vis=original")
    t0 = time.time()

    for idx, name in enumerate(tqdm(names, desc="Infer", ncols=100)):
        path = os.path.join(args.input_dir, name)
        img0 = cv2.imread(path)
        if img0 is None:
            continue

        out = inferencer.run(img0)   # (N,7) or (N,6)
        preds_tri_orig = out.detach().cpu().numpy().astype(np.float32)

        # 저장: pred-only
        save_img = os.path.join(out_img_dir, name)
        save_txt = os.path.join(out_lab_dir, os.path.splitext(name)[0] + ".txt")
        draw_pred_only(img0, preds_tri_orig, save_img, save_txt, include_score=args.txt_include_score)
        preds_by_name[os.path.splitext(name)[0]] = preds_tri_orig

        # GT & 오버레이
        if do_eval:
            lab = os.path.join(args.gt_label_dir, os.path.splitext(name)[0] + ".txt")
            gt_tris_orig = load_gt_triangles(lab)  # (Ng,3,2)
            gts_by_name[os.path.splitext(name)[0]] = gt_tris_orig
            save_img_mix = os.path.join(out_mix_dir, name)
            draw_pred_with_gt(img0, preds_tri_orig, gt_tris_orig, save_img_mix, draw_iou_text=True)

        # 간단 진행 로그 (N장마다)
        if args.log_every > 0 and (idx + 1) % args.log_every == 0:
            print(f"[{idx+1}/{len(names)}] saved to {args.output_dir}")

    # 평가
    if do_eval:
        print("\n[Eval] computing metrics ...")
        metrics = evaluate_dataset(preds_by_name, gts_by_name, iou_thr=args.eval_iou_thr)
        print("== Eval (dataset-wide, single operating point) ==")
        print("Precision:  {:.4f}".format(metrics["precision"]))
        print("Recall:     {:.4f}".format(metrics["recall"]))
        print("F1:         {:.4f}".format(metrics["f1"]))
        print("mean IoU:   {:.4f}".format(metrics["mean_iou"]))
        print("mAOE(deg):  {:.2f}".format(metrics["mAOE_deg"]))

        print("== mAP@0.50 ==")
        has_scores = any(arr.size > 0 and arr.shape[1] >= 7 for arr in preds_by_name.values())
        if has_scores:
            print("[Eval] computing AP@0.50 ...")
            ap50 = compute_ap50_with_scores(preds_by_name, gts_by_name, iou_thr=0.5)
            print("AP@0.50:   {:.4f}".format(ap50))
        else:
            print("WARNING: 모델 점수(confidence)를 Inferencer에서 노출하지 않아 AP를 정식으로 계산할 수 없습니다.")
            print("         (지금 출력된 Precision은 단일 임계점의 값이며, AP와는 개념이 다릅니다.)")

    print("Done. Elapsed: {:.1f}s".format(time.time() - t0))

if __name__ == "__main__":
    main()

"""
python ./2_5d_src/inference_onnx_eval.py \
  --onnx ./checkpoints/model_v2_half.onnx \
  --input-dir ./dataset_example\images \
  --output-dir ./test \
  --half
"""
