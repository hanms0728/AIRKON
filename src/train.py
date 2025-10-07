# train.py
# 논문 "2.5D Object Detection ..." 충실 구현 (YOLOv11 백본)
# - YOLO11 백본/넥 + 커스텀 헤드(6 회귀 + 1 obj + C cls)
# - one-to-many 라벨 어사인: 삼각형 내부 OR 경계거리 ≤ η(px) 이면 양성
# - 손실: p0(MSE, **sum**) + {p1,p2}(Chamfer, **sum**) + BCE(obj) + BCE(cls@pos-only)
# - 오프셋: "셀 단위", 학습/추론 모두 클립 없음(논문 일치)
# - 디코딩 점수: 기본 cls, 옵션으로 obj / obj*cls 선택 가능
# - 검증 가속: metrics 모드에서는 손실 생략 + AMP + GPU NMS 가능
# - 변경사항(이 버전):
#   * η/k-cap 기본값 보수화 (η=3.0, k=96) → 양성 과다 완화
#   * Optimizer: 백본=SGD, 헤드=AdamW (메모리 절약 + 수렴 가속)
#   * 훈련에도 AMP(autocast + GradScaler) 적용 → 메모리/연산량 절감
#   * p0 가중치 워밍업(0.5→1.0, 5epoch)
#   * 평가 기본: obj*cls, conf=0.30, NMS IoU=0.20, topk=100, clip-cells=4.0
#   * obj_pos_weight=1.2 (배경 억제 과상승 방지)
#   * 선택: 초기 몇 에폭 backbone freeze(--freeze-bb-epochs, 기본 1)
#   * 검증단계 tiny 필터 + parallelogram 시그니처 안전 래퍼(safe_parallelogram_from_triangle)
#   * ✅ resume/warm-start 지원(--resume/--weights) + 매 에폭 .pth & .ckpt 저장

import os
import cv2
import math
import argparse
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from tqdm import tqdm

from geometry_utils import parallelogram_from_triangle as _para  # 시그니처 가변 대응은 아래 safe_*가 담당
from geometry_utils import aabb_of_poly4, iou_aabb_xywh
from evaluation_utils import decode_predictions, evaluate_single_image, compute_detection_metrics

torch.backends.cudnn.benchmark = True

# ===========================
# AMP 호환 래퍼 (PyTorch 버전 무관 경고 제거)
# ===========================
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def amp_autocast():
        return _autocast("cuda", enabled=torch.cuda.is_available())
    GradScaler = _GradScaler
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler as GradScaler
    def amp_autocast():
        return _autocast(enabled=torch.cuda.is_available())

# ===========================
# 안전 래퍼 & Tiny 필터 유틸
# ===========================
def safe_parallelogram_from_triangle(tri_or_p0, p1=None, p2=None):
    """
    geometry_utils.parallelogram_from_triangle 이 (p0,p1,p2)만 받는 구현이어도,
    (3,2) tri 하나로도 안전하게 호출할 수 있게 해주는 래퍼.
    """
    if p1 is not None and p2 is not None:  # (p0, p1, p2)
        return _para(tri_or_p0, p1, p2)
    tri = tri_or_p0  # (3,2)
    if isinstance(tri, torch.Tensor):
        tri = tri.detach().cpu().numpy()
    tri = np.asarray(tri, dtype=np.float32)
    assert tri.shape == (3, 2), f"tri must be (3,2), got {tri.shape}"
    return _para(tri[0], tri[1], tri[2])

def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)

def tiny_filter_on_dets(dets_img, min_area=20.0, min_edge=3.0):
    """
    아주 작은 평행사변형(너무 작은 면적/짧은 변)을 제거.
    dets_img: decode_predictions가 반환한 단일 이미지의 detection 리스트.
    각 item은 최소한 'tri' (3,2) 좌표를 포함한다고 가정. 없으면 필터 스킵.
    """
    filtered = []
    for d in dets_img:
        tri = None
        if isinstance(d, dict):
            if 'tri' in d:
                tri = np.asarray(d['tri'], dtype=np.float32)
            elif 'points' in d:
                tri = np.asarray(d['points'], dtype=np.float32)
        if tri is None or tri.shape != (3, 2):
            filtered.append(d)  # 좌표 없으면 필터 불가 → 그대로 통과
            continue
        try:
            poly4 = safe_parallelogram_from_triangle(tri)  # (4,2)
            area = polygon_area(poly4)
            edges = np.linalg.norm(np.roll(poly4, -1, axis=0) - poly4, axis=1)
            if area >= min_area and edges.min() >= min_edge:
                filtered.append(d)
        except Exception:
            filtered.append(d)
    return filtered

# ---------------------------
# 0) Geometry helpers (라벨 어사인용)
# ---------------------------
def _point_in_triangle(px, py, tri):  # tri: (3,2) tensor
    Ax, Ay = tri[0, 0], tri[0, 1]
    Bx, By = tri[1, 0], tri[1, 1]
    Cx, Cy = tri[2, 0], tri[2, 1]
    def sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)
    d1 = sign(px, py, Ax, Ay, Bx, By)
    d2 = sign(px, py, Bx, By, Cx, Cy)
    d3 = sign(px, py, Cx, Cy, Ax, Ay)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    inside = ~(has_neg & has_pos)
    return inside

def _point_to_segment_dist(px, py, x1, y1, x2, y2, eps=1e-9):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    vv = vx * vx + vy * vy + eps
    t = (wx * vx + wy * vy) / vv
    t = torch.clamp(t, 0.0, 1.0)
    projx = x1 + t * vx
    projy = y1 + t * vy
    dx = px - projx
    dy = py - projy
    return torch.sqrt(dx * dx + dy * dy + 1e-12)

def _point_to_triangle_dist(px, py, tri):
    Ax, Ay = tri[0, 0], tri[0, 1]
    Bx, By = tri[1, 0], tri[1, 1]
    Cx, Cy = tri[2, 0], tri[2, 1]
    d1 = _point_to_segment_dist(px, py, Ax, Ay, Bx, By)
    d2 = _point_to_segment_dist(px, py, Bx, By, Cx, Cy)
    d3 = _point_to_segment_dist(px, py, Cx, Cy, Ax, Ay)
    return torch.minimum(d1, torch.minimum(d2, d3))

# ---------------------------
# 1) Dataset
# ---------------------------
class ParallelogramDataset(Dataset):
    def __init__(self, img_dir, label_dir, target_size=(1088, 1920), transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.Ht, self.Wt = target_size  # (H, W)
        self.transform = transform

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image dir not found: {img_dir}")
        if not os.path.isdir(label_dir):
            raise FileNotFoundError(f"Label dir not found: {label_dir}")

        image_filenames = sorted([f for f in os.listdir(img_dir)
                                  if f.lower().endswith(('.jpg','.jpeg','.png'))])
        self.img_files = [f for f in image_filenames
                          if os.path.exists(os.path.join(label_dir, os.path.splitext(f)[0]+'.txt'))]
        print(f"[Dataset] matched pairs = {len(self.img_files)}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, name)
        lab_path = os.path.join(self.label_dir, os.path.splitext(name)[0]+'.txt')

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        H0, W0 = img_bgr.shape[:2]

        img_bgr = cv2.resize(img_bgr, (self.Wt, self.Ht), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).transpose(2,0,1) / 255.0
        img = torch.from_numpy(img).float()

        points_list, labels_list = [], []
        with open(lab_path,'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 7:
                    continue
                _, p0x,p0y,p1x,p1y,p2x,p2y = parts
                sX = self.Wt / W0
                sY = self.Ht / H0
                p0 = [float(p0x)*sX, float(p0y)*sY]
                p1 = [float(p1x)*sX, float(p1y)*sY]
                p2 = [float(p2x)*sX, float(p2y)*sY]
                points_list.append([p0,p1,p2])
                labels_list.append(0)

        targets = {
            "points": torch.tensor(points_list, dtype=torch.float32),  # (N,3,2)
            "labels": torch.tensor(labels_list, dtype=torch.long)      # (N,)
        }
        return img, targets, name

def collate_fn(batch):
    imgs  = torch.stack([b[0] for b in batch],0)
    tgts  = [b[1] for b in batch]
    names = [b[2] for b in batch]
    return imgs, tgts, names

# ---------------------------
# 2) Model (YOLO11 백본 + 커스텀 헤드)
# ---------------------------
class TriHead(nn.Module):
    def __init__(self, in_ch, nc, prior_p=0.20):
        super().__init__()
        self.reg = nn.Conv2d(in_ch, 6, 1, 1, 0)
        self.obj = nn.Conv2d(in_ch, 1, 1, 1, 0)
        self.cls = nn.Conv2d(in_ch, nc, 1, 1, 0)
        prior_b = math.log(prior_p/(1-prior_p))
        nn.init.constant_(self.obj.bias, prior_b)
        nn.init.constant_(self.cls.bias, prior_b)
        nn.init.zeros_(self.reg.weight)
        nn.init.zeros_(self.reg.bias)
    def forward(self, x):
        return self.reg(x), self.obj(x), self.cls(x)

class YOLO11_2_5D(nn.Module):
    def __init__(self, yolo11_path='yolo11m.pt', num_classes=1, img_size=(1088,1920)):
        super().__init__()
        base = YOLO(yolo11_path).model
        self.backbone_neck = base.model[:-1]
        self.save_idx = base.save
        detect = base.model[-1]
        self.strides = detect.stride
        self.f_indices = detect.f

        self.backbone_neck.eval()
        with torch.no_grad():
            dummy = torch.zeros(1,3,img_size[0],img_size[1])
            feats_memory = []
            x = dummy
            for m in self.backbone_neck:
                if m.f != -1:
                    x = feats_memory[m.f] if isinstance(m.f,int) else [x if j==-1 else feats_memory[j] for j in m.f]
                x = m(x)
                feats_memory.append(x if m.i in self.save_idx else None)
            feat_list = [feats_memory[i] for i in self.f_indices]
            in_chs = [f.shape[1] for f in feat_list]
        self.backbone_neck.train()

        self.heads = nn.ModuleList([TriHead(c, num_classes) for c in in_chs])
        self.num_classes = num_classes

    def forward(self, x):
        feats_memory = []
        for m in self.backbone_neck:
            if m.f != -1:
                x = feats_memory[m.f] if isinstance(m.f,int) else [x if j==-1 else feats_memory[j] for j in m.f]
            x = m(x)
            feats_memory.append(x if m.i in self.save_idx else None)
        feat_list = [feats_memory[i] for i in self.f_indices]
        outs = [head(f) for head,f in zip(self.heads, feat_list)]
        return outs  # list of (reg,obj,cls)

# ---------------------------
# 3) Loss (cls@pos-only + 안정 스케일 + pos cap + p0 weight warmup)
# ---------------------------
def chamfer_2pts(pred_2x2, gt_2x2):
    d = torch.cdist(pred_2x2, gt_2x2, p=2)
    d1 = d.min(dim=-1).values.sum(dim=-1)  # pred->gt
    d2 = d.min(dim=-2).values.sum(dim=-1)  # gt->pred
    return (d1 + d2).mean()

class Strict2_5DLoss(nn.Module):
    def __init__(self, num_classes=1, eta_px=3.0, obj_pos_weight=1.2, lambda_cd=1.0, k_pos_cap=96):
        super().__init__()
        self.num_classes = num_classes
        self.eta_px = float(eta_px)
        self.lambda_cd = float(lambda_cd)
        self.k_pos_cap = int(k_pos_cap)
        self.mse = nn.MSELoss(reduction='sum')  # p0 안정 스케일
        self.register_buffer("obj_pos_weight", torch.tensor([obj_pos_weight], dtype=torch.float32))
        # p0 가중치 워밍업
        self.lambda_p0 = 1.0
    def set_p0_weight(self, w: float):
        self.lambda_p0 = float(w)

    @staticmethod
    def _grid_centers(Hs, Ws, stride, device):
        ys = (torch.arange(Hs, device=device) + 0.5) * stride
        xs = (torch.arange(Ws, device=device) + 0.5) * stride
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([xx,yy], dim=-1)

    def forward(self, outputs, targets, strides):
        device = outputs[0][0].device
        L = len(outputs)
        B = outputs[0][0].shape[0]

        reg_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        reg_p0_sum  = torch.tensor(0.0, device=device)
        reg_p12_sum = torch.tensor(0.0, device=device)
        pos_count = 0
        neg_count = 0

        for b in range(B):
            gt = targets[b]['points'].to(device)   # (Ng,3,2)
            Ng = gt.shape[0]

            for l in range(L):
                pred_reg, pred_obj, pred_cls = outputs[l]
                pred_reg_i = pred_reg[b]  # (6,Hs,Ws)
                pred_obj_i = pred_obj[b]  # (1,Hs,Ws)
                pred_cls_i = pred_cls[b]  # (C,Hs,Ws)
                Hs, Ws = pred_obj_i.shape[1:]
                stride = float(strides[l]) if not isinstance(strides[l], torch.Tensor) else float(strides[l].item())

                centers = self._grid_centers(Hs, Ws, stride, device)  # (Hs,Ws,2)
                obj_t = torch.zeros_like(pred_obj_i, device=device)   # (1,Hs,Ws)

                pred_off_full = pred_reg_i.permute(1,2,0).contiguous().view(Hs,Ws,3,2)

                cls_pos_logit_list = []  # cls@pos-only

                if Ng == 0:
                    obj_loss += F.binary_cross_entropy_with_logits(
                        pred_obj_i, obj_t, pos_weight=self.obj_pos_weight, reduction='sum'
                    )
                    neg_count += Hs*Ws
                    continue

                for j in range(Ng):
                    tri_px = gt[j]  # (3,2)

                    inside = _point_in_triangle(centers[...,0], centers[...,1], tri_px)
                    dist_b = _point_to_triangle_dist(centers[...,0], centers[...,1], tri_px)
                    near = dist_b <= self.eta_px
                    pos_mask = inside | near
                    npix = int(pos_mask.sum().item())
                    if npix == 0:
                        continue

                    # cap: 경계까지 가까운 순서 상위 k
                    if npix > self.k_pos_cap:
                        flat_idx = torch.nonzero(pos_mask, as_tuple=False)
                        d_keep = dist_b[pos_mask]
                        topk = torch.topk(-d_keep, k=self.k_pos_cap).indices
                        pos_mask.zero_()
                        sel = flat_idx[topk]
                        pos_mask[sel[:,0], sel[:,1]] = True
                        npix = self.k_pos_cap

                    obj_t[0, pos_mask] = 1.0

                    # cls@pos-only
                    if pred_cls_i.shape[0] == 1:
                        cls_pos_logit_list.append(pred_cls_i[0][pos_mask])
                    else:
                        raise NotImplementedError("multi-class pos-only not implemented")

                    # 회귀 GT (cells)
                    anchor_xy = centers[pos_mask]  # (Np,2)
                    tri_rep   = tri_px[None,:,:].expand(anchor_xy.shape[0], -1, -1)
                    anchor_rep = anchor_xy[:,None,:].expand(-1,3,-1)
                    gt_off_cells = (tri_rep - anchor_rep) / stride

                    pred_off_pos = pred_off_full[pos_mask]  # (Np,3,2)

                    loss_p0 = self.mse(pred_off_pos[:,0,:], gt_off_cells[:,0,:])
                    loss_cd = chamfer_2pts(pred_off_pos[:,1:,:], gt_off_cells[:,1:,:]) * pred_off_pos.shape[0]

                    reg_loss += self.lambda_p0 * loss_p0 + self.lambda_cd * loss_cd
                    reg_p0_sum  += self.lambda_p0 * loss_p0
                    reg_p12_sum += self.lambda_cd * loss_cd
                    pos_count += npix

                # obj BCE (양/음)
                obj_loss += F.binary_cross_entropy_with_logits(
                    pred_obj_i, obj_t, pos_weight=self.obj_pos_weight, reduction='sum'
                )

                # cls@pos-only BCE
                if len(cls_pos_logit_list) > 0:
                    pred_cls_pos = torch.cat(cls_pos_logit_list, dim=0)
                    cls_pos_t = torch.ones_like(pred_cls_pos)
                    cls_loss += F.binary_cross_entropy_with_logits(pred_cls_pos, cls_pos_t, reduction='sum')

                neg_count += (Hs*Ws - int((obj_t>0.5).sum().item()))

        # 정규화
        pos_eps = max(1, pos_count)
        reg_loss = reg_loss / pos_eps
        obj_loss = obj_loss / (pos_eps + max(1,neg_count))
        cls_loss = cls_loss / pos_eps

        reg_p0  = reg_p0_sum  / pos_eps
        reg_p12 = reg_p12_sum / pos_eps

        total = reg_loss + obj_loss + cls_loss
        logs = {
            "loss_total": float(total.item()),
            "loss_reg": float(reg_loss.item()),
            "loss_obj": float(obj_loss.item()),
            "loss_cls": float(cls_loss.item()),
            "loss_p0":    float(reg_p0.item()),
            "loss_p12":   float(reg_p12.item()),
            "pos": pos_count,
            "neg": neg_count
        }
        return total, logs

# ---------------------------
# 4) Save/Resume 유틸
# ---------------------------
def save_ckpt(path, epoch, model, opt_bb, opt_hd, scheduler, scaler, best_val, extra: dict = None):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "opt_bb": opt_bb.state_dict(),
        "opt_hd": opt_hd.state_dict(),
        "sched": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val": best_val,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)

def load_ckpt(path, device, model, opt_bb, opt_hd, scheduler, scaler):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    opt_bb.load_state_dict(ckpt["opt_bb"])
    opt_hd.load_state_dict(ckpt["opt_hd"])
    scheduler.load_state_dict(ckpt["sched"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = ckpt.get("epoch", 0)
    best = ckpt.get("best_val", float('inf'))
    return epoch, best

# ---------------------------
# 5) Train (검증 가속 옵션 포함)
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 2.5D (paper-accurate, cls@pos-only, fast val)")
    parser.add_argument("--train-root", type=str, default="./output_all")
    parser.add_argument("--val-root", type=str, default=None)
    parser.add_argument("--val-mode", choices=["none", "loss", "metrics"], default="metrics")
    parser.add_argument("--val-interval", type=int, default=10)
    parser.add_argument("--val-batch", type=int, default=16, help="Validation batch size")
    parser.add_argument("--val-max-batches", type=int, default=None, help="Validate at most N batches per epoch")

    # 평가 하이퍼(겹침 억제)
    parser.add_argument("--eval-conf", type=float, default=0.80)
    parser.add_argument("--eval-nms-iou", type=float, default=0.20)
    parser.add_argument("--eval-topk", type=int, default=50)
    parser.add_argument("--eval-iou-thr", type=float, default=0.5)
    parser.add_argument("--score-mode", type=str, default="obj*cls", choices=["cls","obj","obj*cls"])
    parser.add_argument("--clip-cells", type=float, default=4.0, help="If set, tanh clip radius (cells) at decode")

    # 라벨/손실 하이퍼
    parser.add_argument("--eta-px", type=float, default=3.0)
    parser.add_argument("--lambda-cd", type=float, default=1.0)
    parser.add_argument("--k-pos-cap", type=int, default=96)

    # 백본 프리즈
    parser.add_argument("--freeze-bb-epochs", type=int, default=1)

    # 초기 가중치/재개
    parser.add_argument("--yolo-weights", type=str, default="yolo11m.pt", help="Ultralytics YOLOv11 weights to start from")
    parser.add_argument("--weights", type=str, default=None, help="Warm-start from a model .pth (model-only)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a full .ckpt (model+opt+sched+scaler)")
    parser.add_argument("--start-epoch", type=int, default=0, help="Start epoch index when warm-starting")

    args = parser.parse_args()

    train_img_dir = os.path.join(args.train_root, "images")
    train_label_dir = os.path.join(args.train_root, "labels")

    if args.val_mode != "none":
        val_root = args.val_root or args.train_root
        val_img_dir = os.path.join(val_root, "images")
        val_label_dir = os.path.join(val_root, "labels")
    else:
        val_img_dir = val_label_dir = None

    # 하이퍼
    IMG_SIZE = (864, 1536)  # (H,W)
    TRAIN_BATCH = 8
    EPOCHS = 60
    LR_MIN = 1e-4
    WD = 1e-4
    NUM_CLASSES = 1

    eval_cfg = None
    if args.val_mode == "metrics":
        eval_cfg = {
            "conf_th": args.eval_conf,
            "nms_iou": args.eval_nms_iou,
            "topk": args.eval_topk,
            "iou_thr": args.eval_iou_thr,
            "score_mode": args.score_mode,
            "clip_cells": args.clip_cells,
        }

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = ParallelogramDataset(train_img_dir, train_label_dir, target_size=IMG_SIZE)
    if len(train_dataset) == 0:
        print("No training data. Exit.")
        return

    val_loader = None
    if args.val_mode != "none":
        val_dataset = ParallelogramDataset(val_img_dir, val_label_dir, target_size=IMG_SIZE)
        if len(val_dataset) == 0:
            print("No validation data. Exit.")
            return
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch, shuffle=False,
                                collate_fn=collate_fn, num_workers=4, pin_memory=True)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = YOLO11_2_5D(yolo11_path=args.yolo_weights, num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(DEVICE)
    if isinstance(model.strides, torch.Tensor):
        model.strides = [float(s.item()) for s in model.strides]
    else:
        model.strides = [float(s) for s in model.strides]

    criterion = Strict2_5DLoss(num_classes=NUM_CLASSES,
                               eta_px=args.eta_px,
                               obj_pos_weight=1.2,
                               lambda_cd=args.lambda_cd,
                               k_pos_cap=args.k_pos_cap).to(DEVICE)

    # 파라미터 그룹(헤드/백본 분리)
    head_params, bb_params = [], []
    for name, p in model.named_parameters():
        if any(k in name for k in ["heads", "reg", "obj", "cls"]):
            head_params.append(p)
        else:
            bb_params.append(p)

    # 옵티마이저: 백본=SGD, 헤드=AdamW
    opt_bb = optim.SGD(bb_params, lr=2e-4, momentum=0.9, weight_decay=WD, nesterov=True)
    opt_hd = optim.AdamW(head_params, lr=1e-3, betas=(0.9, 0.999), weight_decay=WD)

    class DualOptim:
        def __init__(self, opt1, opt2):
            self.opt1, self.opt2 = opt1, opt2
        def zero_grad(self, set_to_none=True):
            self.opt1.zero_grad(set_to_none=set_to_none)
            self.opt2.zero_grad(set_to_none=set_to_none)
        def step(self):
            self.opt1.step(); self.opt2.step()
        @property
        def param_groups(self):
            return self.opt1.param_groups + self.opt2.param_groups

    optimizer = DualOptim(opt_bb, opt_hd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_bb, T_max=EPOCHS, eta_min=LR_MIN)

    # AMP GradScaler (훈련용)
    scaler = GradScaler(enabled=(DEVICE == 'cuda'))

    # (선택) 백본 프리즈 유틸
    def set_requires_grad(module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    # ---- Resume / Warm-start ----
    best_val = float('inf')
    start_ep = 0
    if args.resume is not None:
        start_ep, best_val = load_ckpt(args.resume, DEVICE, model, opt_bb, opt_hd, scheduler, scaler)
        print(f"[Resume] {args.resume} -> start_ep={start_ep}, best_val={best_val:.4f}")
    elif args.weights is not None:
        sd = torch.load(args.weights, map_location=DEVICE)
        if isinstance(sd, dict) and "model" in sd:  # ckpt를 weights로 준 경우
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[Warm-start] {args.weights} (missing={len(missing)}, unexpected={len(unexpected)})")
        start_ep = int(args.start_epoch)

    val_desc = f"{args.val_mode} (interval={args.val_interval})" if args.val_mode != "none" else "disabled"
    print(f"[Train] device={DEVICE}, epochs={EPOCHS}, start_ep={start_ep}, batch={TRAIN_BATCH}, strides={model.strides}, val={val_desc}")

    def run_epoch(loader, train=True, epoch_idx=0, eval_cfg=None, val_mode="metrics"):
        # p0 가중치 워밍업 (0~4 epoch: 0.5→1.0 선형)
        warmup_ep = 5
        w = 0.5 + 0.5 * min(epoch_idx, warmup_ep-1) / (warmup_ep-1)
        criterion.set_p0_weight(w)

        # 백본 프리즈 제어
        if train:
            if epoch_idx < args.freeze_bb_epochs:
                set_requires_grad(model.backbone_neck, False)
            else:
                set_requires_grad(model.backbone_neck, True)

        model.train(mode=train)
        collect_metrics = (not train) and (eval_cfg is not None)
        metric_records = []
        total_gt = 0

        tot = treg = tobj = tcls = tpos = 0.0
        tp0 = tp12 = 0.0
        nb = 0
        batches_seen = 0

        pbar = tqdm(loader, desc=f"{'Train' if train else 'Val'} {epoch_idx+1}/{EPOCHS} (p0_w={w:.2f})")
        for imgs, targets_cpu, _ in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            targets_dev = [{"points": t["points"].to(DEVICE), "labels": t["labels"].to(DEVICE)} for t in targets_cpu]

            if train:
                optimizer.zero_grad(set_to_none=True)
                with amp_autocast():
                    outs = model(imgs)
                    loss, logs = criterion(outs, targets_dev, model.strides)
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                if val_mode == "loss":
                    with torch.inference_mode():
                        outs = model(imgs)
                        loss, logs = criterion(outs, targets_dev, model.strides)
                else:  # metrics
                    with torch.inference_mode(), amp_autocast():
                        outs = model(imgs)
                    loss = torch.tensor(0., device=imgs.device)
                    logs = {"loss_total": 0, "loss_reg":0, "loss_obj":0, "loss_cls":0,
                            "loss_p0":0, "loss_p12":0, "pos":0, "neg":0}

            tot  += logs["loss_total"]
            treg += logs["loss_reg"]
            tobj += logs["loss_obj"]
            tcls += logs["loss_cls"]
            tpos += logs["pos"]
            tp0  += logs["loss_p0"]
            tp12 += logs["loss_p12"]
            nb   += 1

            pbar.set_postfix(loss=(tot/nb if nb>0 else 0), reg=(treg/nb if nb>0 else 0),
                             obj=(tobj/nb if nb>0 else 0), p0=(tp0/nb if nb>0 else 0),
                             p12=(tp12/nb if nb>0 else 0), cls=(tcls/nb if nb>0 else 0),
                             pos=(tpos/nb if nb>0 else 0), lr=optimizer.param_groups[0]['lr'])

            if collect_metrics:
                decoded = decode_predictions(
                    outs, model.strides,
                    clip_cells=eval_cfg.get("clip_cells", None),
                    conf_th=eval_cfg.get("conf_th", 0.30),
                    nms_iou=eval_cfg.get("nms_iou", 0.20),
                    topk=eval_cfg.get("topk", 100),
                    score_mode=eval_cfg.get("score_mode", "obj*cls"),
                    use_gpu_nms=True
                )
                decoded = [tiny_filter_on_dets(dets_img=di, min_area=20.0, min_edge=3.0) for di in decoded]

                for dets_img, tgt_cpu in zip(decoded, targets_cpu):
                    gt_tri = tgt_cpu["points"].cpu().numpy()
                    records, _ = evaluate_single_image(dets_img, gt_tri, iou_thr=eval_cfg.get("iou_thr", 0.5))
                    metric_records.extend(records)
                    total_gt += gt_tri.shape[0]

            batches_seen += 1
            if (not train) and args.val_max_batches is not None and batches_seen >= args.val_max_batches:
                break

        avg_loss = tot/nb if nb > 0 else 0.0
        metrics = compute_detection_metrics(metric_records, total_gt) if collect_metrics else None
        return avg_loss, metrics

    # --------- Loop ---------
    for ep in range(start_ep, EPOCHS):
        train_loss, _ = run_epoch(train_loader, train=True, epoch_idx=ep, eval_cfg=eval_cfg, val_mode=args.val_mode)

        run_val = (
            args.val_mode != "none"
            and val_loader is not None
            and ((ep + 1) % max(1, args.val_interval) == 0 or ep == EPOCHS - 1)
        )
        if run_val:
            val_loss, val_metrics = run_epoch(val_loader, train=False, epoch_idx=ep, eval_cfg=eval_cfg, val_mode=args.val_mode)
            if val_metrics is not None:
                print("  -> metrics P={:.4f} R={:.4f} mAP@50={:.4f} mAOE(deg)={:.2f}".format(
                    val_metrics["precision"], val_metrics["recall"],
                    val_metrics["map50"], val_metrics["mAOE_deg"]
                ))
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), "yolo11m_2_5d_best.pth")
                print(f"  -> best updated {best_val:.4f}")

        scheduler.step()

        # 매 에폭 저장: .pth(모델만) + .ckpt(풀 재개)
        pth_path = f"yolo11m_2_5d_epoch_{ep+1}.pth"
        torch.save(model.state_dict(), pth_path)
        ckpt_path = f"yolo11m_2_5d_epoch_{ep+1}.ckpt"
        save_ckpt(
            ckpt_path, epoch=ep+1, model=model,
            opt_bb=opt_bb, opt_hd=opt_hd, scheduler=scheduler, scaler=scaler,
            best_val=best_val
        )
        print(f"  -> saved {pth_path} and {ckpt_path}")

    print("Done.")

if __name__ == "__main__":
    main()