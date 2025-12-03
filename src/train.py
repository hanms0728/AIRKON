# -*- coding: utf-8 -*-
# train.py — YOLO11 기반 2.5D polygon detector
# 자동 pth 저장 + 자동 ONNX export 포함 완전 버전

import os
import csv
import cv2
import math
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

try:
    import albumentations as A
except ImportError:
    A = None

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from tqdm import tqdm

from src.geometry_utils import parallelogram_from_triangle as _para
from src.geometry_utils import aabb_of_poly4, iou_aabb_xywh
from src.evaluation_utils import decode_predictions, evaluate_single_image, compute_detection_metrics

torch.backends.cudnn.benchmark = True


# ================================================
# 1. 시드 고정
# ================================================
def set_seed(s=42):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

set_seed(42)


# ================================================
# 2. AMP 호환
# ================================================
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def amp_autocast():
        return _autocast("cuda", enabled=torch.cuda.is_available())
    GradScaler = _GradScaler
except:
    from torch.cuda.amp import autocast as _autocast, GradScaler as GradScaler
    def amp_autocast():
        return _autocast(enabled=torch.cuda.is_available())


# ================================================
# 3. Tiny-filter + geometry 헬퍼
# ================================================
def safe_parallelogram_from_triangle(tri_or_p0, p1=None, p2=None):
    if p1 is not None and p2 is not None:
        return _para(tri_or_p0, p1, p2)
    tri = tri_or_p0
    if isinstance(tri, torch.Tensor):
        tri = tri.detach().cpu().numpy()
    tri = np.asarray(tri, dtype=np.float32)
    return _para(tri[0], tri[1], tri[2])


def polygon_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def tiny_filter_on_dets(dets_img, min_area=20.0, min_edge=3.0):
    filtered = []
    for d in dets_img:
        tri = None
        if isinstance(d, dict):
            tri = d.get("tri", d.get("points", None))
        if tri is None:
            filtered.append(d)
            continue
        tri = np.asarray(tri, dtype=np.float32)
        if tri.shape != (3,2):
            filtered.append(d)
            continue
        try:
            poly4 = safe_parallelogram_from_triangle(tri)
            area = polygon_area(poly4)
            edges = np.linalg.norm(np.roll(poly4, -1, axis=0) - poly4, axis=1)
            if area >= min_area and edges.min() >= min_edge:
                filtered.append(d)
        except:
            filtered.append(d)
    return filtered


# ================================================
# 4. Dataset
# ================================================
class ParallelogramDataset(Dataset):
    def __init__(self, img_dir, label_dir, target_size=(1088,1920), transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.Ht, self.Wt = target_size
        self.transform = transform

        # Support both:
        # ① root/images , root/labels     (flat mode)
        # ② root/<seq1>/images , root/<seq1>/labels
        #    root/<seq2>/images , root/<seq2>/labels   (multi-folder mode)

        def has_image_files(path):
            return any(f.lower().endswith(('.jpg','.jpeg','.png')) for f in os.listdir(path))

        # Detect flat mode
        flat_mode = has_image_files(self.img_dir)

        collected = []  # list of (img_path , label_path , name)

        if flat_mode:
            # 기존 방식
            image_filenames = sorted([
                f for f in os.listdir(self.img_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))
            ])
            for fname in image_filenames:
                stem = os.path.splitext(fname)[0]
                lab_path = os.path.join(self.label_dir, stem + '.txt')
                if os.path.exists(lab_path):
                    collected.append((
                        os.path.join(self.img_dir, fname),
                        lab_path,
                        fname
                    ))

        else:
            # multi-folder 구조 (각 subfolder 아래 images/, labels/ 존재)
            for sub in sorted(os.listdir(self.img_dir)):
                sub_root = os.path.join(self.img_dir, sub)
                if not os.path.isdir(sub_root):
                    continue

                sub_img = os.path.join(sub_root, "images")
                sub_lab = os.path.join(sub_root, "labels")
                if (not os.path.isdir(sub_img)) or (not os.path.isdir(sub_lab)):
                    continue
                if not has_image_files(sub_img):
                    continue

                image_filenames = sorted([
                    f for f in os.listdir(sub_img)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])

                for fname in image_filenames:
                    stem = os.path.splitext(fname)[0]
                    lab_path = os.path.join(sub_lab, stem + '.txt')
                    if os.path.exists(lab_path):
                        out_name = f"{sub}__{fname}"
                        collected.append((
                            os.path.join(sub_img, fname),
                            lab_path,
                            out_name
                        ))

        self.items = collected
        print(f"[Dataset] matched pairs = {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, lab_path, name = self.items[idx]

        img_bgr = cv2.imread(img_path)
        H0, W0 = img_bgr.shape[:2]

        img_bgr = cv2.resize(img_bgr, (self.Wt, self.Ht), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        points_list = []
        label_list = []
        with open(lab_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 7:
                    continue
                cls_id, p0x, p0y, p1x, p1y, p2x, p2y = parts
                sX = self.Wt / W0
                sY = self.Ht / H0
                p0 = [float(p0x)*sX, float(p0y)*sY]
                p1 = [float(p1x)*sX, float(p1y)*sY]
                p2 = [float(p2x)*sX, float(p2y)*sY]
                points_list.append([p0,p1,p2])
                label_list.append(int(float(cls_id)))

        targets = {
            "points": torch.tensor(points_list, dtype=torch.float32),
            "labels": torch.tensor(label_list, dtype=torch.long)
        }

        if self.transform is not None:
            flat_points = [tuple(pt) for tri in points_list for pt in tri]
            aug = self.transform(image=img_rgb, keypoints=flat_points)
            img_rgb = aug["image"]
            aug_pts = aug["keypoints"]
            if aug_pts and len(aug_pts) == len(flat_points):
                new_points = []
                for i in range(0, len(aug_pts), 3):
                    new_points.append([
                        list(aug_pts[i + 0]),
                        list(aug_pts[i + 1]),
                        list(aug_pts[i + 2])
                    ])
                points_list = new_points
                targets["points"] = torch.tensor(points_list, dtype=torch.float32)

        img = img_rgb.transpose(2,0,1) / 255.0
        img = torch.from_numpy(img).float()

        return img, targets, name


def build_default_train_augment(target_size):
    if A is None:
        raise ImportError("Albumentations is required for --train-augment. Please install it via `pip install albumentations`.")
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.02, 0.02),
                rotate=(-5, 5),
                shear=(-2, 2),
                p=0.6
            ),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
    )


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], 0)
    tgts = [b[1] for b in batch]
    names = [b[2] for b in batch]
    return imgs, tgts, names


# ================================================
# 5. Model (YOLO11 base + custom TriHead)
# ================================================
class TriHead(nn.Module):
    def __init__(self, in_ch, nc, prior_p=0.20):
        super().__init__()
        self.reg = nn.Conv2d(in_ch, 6, 1)
        self.obj = nn.Conv2d(in_ch, 1, 1)
        self.cls = nn.Conv2d(in_ch, nc, 1)

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

        # compute feature dims
        self.backbone_neck.eval()
        with torch.no_grad():
            dummy = torch.zeros(1,3,img_size[0],img_size[1])
            feats = []
            x = dummy
            for m in self.backbone_neck:
                if m.f != -1:
                    x = feats[m.f] if isinstance(m.f,int) else [x if j==-1 else feats[j] for j in m.f]
                x = m(x)
                feats.append(x if m.i in self.save_idx else None)

            feat_list = [feats[i] for i in self.f_indices]
            in_chs = [f.shape[1] for f in feat_list]

        self.backbone_neck.train()
        self.heads = nn.ModuleList([TriHead(c, num_classes) for c in in_chs])

    def forward(self, x):
        feats = []
        for m in self.backbone_neck:
            if m.f != -1:
                x = feats[m.f] if isinstance(m.f,int) else [x if j==-1 else feats[j] for j in m.f]
            x = m(x)
            feats.append(x if m.i in self.save_idx else None)

        feat_list = [feats[i] for i in self.f_indices]
        outs = [head(f) for head,f in zip(self.heads, feat_list)]
        return outs    # list[(reg,obj,cls), ...]

# ================================================
# 6. Loss용 Geometry 헬퍼
# ================================================
def _point_in_triangle(px, py, tri):
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
    vv = vx*vx + vy*vy + eps
    t = (wx*vx + wy*vy) / vv
    t = torch.clamp(t, 0.0, 1.0)
    projx = x1 + t * vx
    projy = y1 + t * vy
    dx = px - projx
    dy = py - projy
    return torch.sqrt(dx*dx + dy*dy + 1e-12)


def _point_to_triangle_dist(px, py, tri):
    Ax, Ay = tri[0, 0], tri[0, 1]
    Bx, By = tri[1, 0], tri[1, 1]
    Cx, Cy = tri[2, 0], tri[2, 1]
    d1 = _point_to_segment_dist(px, py, Ax, Ay, Bx, By)
    d2 = _point_to_segment_dist(px, py, Bx, By, Cx, Cy)
    d3 = _point_to_segment_dist(px, py, Cx, Cy, Ax, Ay)
    return torch.minimum(d1, torch.minimum(d2, d3))


# ================================================
# 7. Loss 정의
# ================================================
def chamfer_2pts(pred_2x2, gt_2x2):
    d = torch.cdist(pred_2x2, gt_2x2, p=2)
    d1 = d.min(dim=-1).values.sum(dim=-1)
    d2 = d.min(dim=-2).values.sum(dim=-1)
    return (d1 + d2).mean()


def focal_loss_binary(logits, targets, alpha=0.25, gamma=2.0):
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    else:
        alpha_t = 1.0
    loss = alpha_t * torch.pow(1.0 - p_t, gamma) * ce
    return loss.sum()


def focal_loss_multiclass(logits, targets, gamma=2.0, alpha=None):
    if logits.numel() == 0:
        return logits.sum()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    gather_indices = torch.arange(targets.shape[0], device=logits.device)
    p_t = probs[gather_indices, targets]
    ce = F.nll_loss(log_probs, targets, reduction="none")
    modulating = torch.pow(1.0 - p_t, gamma)
    loss = modulating * ce
    if alpha is not None:
        if isinstance(alpha, (list, tuple)):
            alpha_tensor = logits.new_tensor(alpha)
            alpha_factor = alpha_tensor[targets]
        else:
            alpha_factor = logits.new_full(targets.shape, float(alpha))
        loss = loss * alpha_factor
    return loss.sum()


class Strict2_5DLoss(nn.Module):
    def __init__(self, num_classes=1, eta_px=3.0, obj_pos_weight=1.2,
                 lambda_cd=1.0, k_pos_cap=96,
                 use_focal_cls=False, focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.eta_px = float(eta_px)
        self.lambda_cd = float(lambda_cd)
        self.k_pos_cap = int(k_pos_cap)

        self.mse = nn.MSELoss(reduction="sum")
        self.register_buffer("obj_pos_weight",
                             torch.tensor([obj_pos_weight], dtype=torch.float32))

        # p0(roof 중심)에 대한 가중치
        self.lambda_p0 = 1.0
        self.use_focal_cls = bool(use_focal_cls)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = None if focal_alpha is None else float(focal_alpha)

    def set_p0_weight(self, w: float):
        self.lambda_p0 = float(w)

    @staticmethod
    def _grid_centers(Hs, Ws, stride, device):
        ys = (torch.arange(Hs, device=device) + 0.5) * stride
        xs = (torch.arange(Ws, device=device) + 0.5) * stride
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([xx, yy], dim=-1)

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
            gt_pts = targets[b]["points"].to(device)
            gt_lbl = targets[b].get("labels")
            if gt_lbl is not None:
                gt_lbl = gt_lbl.to(device)
            Ng = gt_pts.shape[0]

            for l in range(L):
                pred_reg, pred_obj, pred_cls = outputs[l]
                pred_reg_i = pred_reg[b]
                pred_obj_i = pred_obj[b]
                pred_cls_i = pred_cls[b]

                Hs, Ws = pred_obj_i.shape[1:]
                stride = float(strides[l]) if not isinstance(strides[l], torch.Tensor) else float(strides[l].item())

                centers = self._grid_centers(Hs, Ws, stride, device)
                obj_t = torch.zeros_like(pred_obj_i, device=device)

                # (C=6, H, W) -> (H, W, 3, 2)
                pred_off_full = pred_reg_i.permute(1,2,0).contiguous().view(Hs, Ws, 3, 2)

                if pred_cls_i.shape[0] == 1:
                    cls_pos_logit_list = []
                    cls_target_map = None
                else:
                    cls_pos_logit_list = None
                    cls_target_map = torch.full((Hs, Ws), -1, dtype=torch.long, device=device)
                    cls_target_dist = torch.full((Hs, Ws), float("inf"), dtype=torch.float32, device=device)

                if Ng == 0:
                    obj_loss += F.binary_cross_entropy_with_logits(
                        pred_obj_i, obj_t, pos_weight=self.obj_pos_weight, reduction="sum"
                    )
                    neg_count += Hs * Ws
                    continue

                for j in range(Ng):
                    tri_px = gt_pts[j]
                    if gt_lbl is not None and gt_lbl.shape[0] > j:
                        cls_idx = int(gt_lbl[j].item())
                    else:
                        cls_idx = 0

                    inside = _point_in_triangle(centers[...,0], centers[...,1], tri_px)
                    dist_b = _point_to_triangle_dist(centers[...,0], centers[...,1], tri_px)
                    near = dist_b <= self.eta_px
                    pos_mask = inside | near
                    npix = int(pos_mask.sum().item())
                    if npix == 0:
                        continue

                    # positive cell cap
                    if npix > self.k_pos_cap:
                        flat_idx = torch.nonzero(pos_mask, as_tuple=False)
                        d_keep = dist_b[pos_mask]
                        topk = torch.topk(-d_keep, k=self.k_pos_cap).indices  # 거리 작은 순
                        pos_mask.zero_()
                        sel = flat_idx[topk]
                        pos_mask[sel[:,0], sel[:,1]] = True
                        npix = self.k_pos_cap

                    obj_t[0, pos_mask] = 1.0

                    if cls_target_map is not None:
                        candidate_dist = dist_b[pos_mask]
                        best_dist = cls_target_dist[pos_mask]
                        better = candidate_dist < best_dist
                        if bool(better.any()):
                            mask_indices = torch.nonzero(pos_mask, as_tuple=False)
                            chosen = mask_indices[better]
                            cls_target_map[chosen[:,0], chosen[:,1]] = cls_idx
                            cls_target_dist[chosen[:,0], chosen[:,1]] = candidate_dist[better]
                    elif pred_cls_i.shape[0] == 1:
                        cls_pos_logit_list.append(pred_cls_i[0][pos_mask])

                    anchor_xy = centers[pos_mask]
                    tri_rep   = tri_px[None,:,:].expand(anchor_xy.shape[0], -1, -1)
                    anchor_rep = anchor_xy[:,None,:].expand(-1,3,-1)
                    gt_off_cells = (tri_rep - anchor_rep) / stride

                    pred_off_pos = pred_off_full[pos_mask]

                    loss_p0 = self.mse(pred_off_pos[:,0,:], gt_off_cells[:,0,:])
                    loss_cd = chamfer_2pts(pred_off_pos[:,1:,:], gt_off_cells[:,1:,:]) * pred_off_pos.shape[0]

                    reg_loss += self.lambda_p0 * loss_p0 + self.lambda_cd * loss_cd
                    reg_p0_sum  += self.lambda_p0 * loss_p0
                    reg_p12_sum += self.lambda_cd * loss_cd
                    pos_count += npix

                obj_loss += F.binary_cross_entropy_with_logits(
                    pred_obj_i, obj_t, pos_weight=self.obj_pos_weight, reduction="sum"
                )

                if cls_target_map is not None:
                    pos_mask_all = obj_t[0] > 0.5
                    if bool(pos_mask_all.any()):
                        cls_targets = cls_target_map[pos_mask_all]
                        valid_mask = cls_targets >= 0
                        if bool(valid_mask.any()):
                            logits = pred_cls_i[:, pos_mask_all].permute(1, 0).contiguous()
                            logits = logits[valid_mask]
                            targets_ce = cls_targets[valid_mask]
                            if self.use_focal_cls:
                                cls_loss += focal_loss_multiclass(
                                    logits, targets_ce,
                                    gamma=self.focal_gamma,
                                    alpha=self.focal_alpha
                                )
                            else:
                                cls_loss += F.cross_entropy(logits, targets_ce, reduction="sum")
                elif cls_pos_logit_list and len(cls_pos_logit_list) > 0:
                    pred_cls_pos = torch.cat(cls_pos_logit_list, dim=0)
                    cls_pos_t = torch.ones_like(pred_cls_pos)
                    if self.use_focal_cls:
                        cls_loss += focal_loss_binary(
                            pred_cls_pos, cls_pos_t,
                            alpha=self.focal_alpha,
                            gamma=self.focal_gamma
                        )
                    else:
                        cls_loss += F.binary_cross_entropy_with_logits(
                            pred_cls_pos, cls_pos_t, reduction="sum"
                        )

                neg_count += (Hs*Ws - int((obj_t > 0.5).sum().item()))

        pos_eps = max(1, pos_count)
        reg_loss = reg_loss / pos_eps
        obj_loss = obj_loss / (pos_eps + max(1, neg_count))
        cls_loss = cls_loss / pos_eps

        reg_p0  = reg_p0_sum  / pos_eps
        reg_p12 = reg_p12_sum / pos_eps

        total = reg_loss + obj_loss + cls_loss

        logs = {
            "loss_total": float(total.item()),
            "loss_reg": float(reg_loss.item()),
            "loss_obj": float(obj_loss.item()),
            "loss_cls": float(cls_loss.item()),
            "loss_p0":  float(reg_p0.item()),
            "loss_p12": float(reg_p12.item()),
            "pos": pos_count,
            "neg": neg_count,
        }
        return total, logs


# ================================================
# 8. Checkpoint & ONNX Export 유틸
# ================================================
def save_ckpt(path, epoch, model, opt_all, sched_all ,scaler, best_val, extra: dict = None):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "opt_all": opt_all.state_dict(),
        "sched_all": sched_all.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val": best_val,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_ckpt(path, device, model, opt_all, sched_all, scaler):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt_all.load_state_dict(ckpt["opt_all"])
    sched_all.load_state_dict(ckpt["sched_all"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt.get("best_val", 0)


def export_onnx(model, onnx_path: Path, img_size, device, opset=12):
    """
    model: YOLO11_2_5D
    output: (reg_0, reg_1, reg_2, obj_0, obj_1, obj_2, cls_0, cls_1, cls_2) 튜플
    """
    class ExportWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            outs = self.m(x)
            regs = [o[0] for o in outs]
            objs = [o[1] for o in outs]
            clss = [o[2] for o in outs]
            return tuple(regs + objs + clss)

    was_training = model.training
    model.eval()
    wrapper = ExportWrapper(model).to(device)

    dummy = torch.zeros(1, 3, img_size[0], img_size[1], device=device)

    n_heads = len(model.heads)
    output_names = (
        [f"reg_{i}" for i in range(n_heads)] +
        [f"obj_{i}" for i in range(n_heads)] +
        [f"cls_{i}" for i in range(n_heads)]
    )

    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["images"],
        output_names=output_names,
        opset_version=opset,
        dynamic_axes={"images": {0: "batch"}}
    )
    print(f"[ONNX] Exported best model to: {onnx_path}")
    model.train(was_training)


# ================================================
# 9. Train main
# ================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO11 2.5D (pure, cls@pos-only, mAP 기준 best + ONNX export)"
    )
    parser.add_argument("--train-root", type=str, default="./output_all")
    parser.add_argument("--val-root", type=str, default=None)
    parser.add_argument("--val-mode", choices=["none","loss","metrics"], default="metrics")
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--val-batch", type=int, default=4)
    parser.add_argument("--val-max-batches", type=int, default=None)

    # 평가 하이퍼
    parser.add_argument("--eval-conf", type=float, default=0.30)
    parser.add_argument("--eval-nms-iou", type=float, default=0.20)
    parser.add_argument("--eval-topk", type=int, default=100)
    parser.add_argument("--eval-iou-thr", type=float, default=0.5)
    parser.add_argument("--score-mode", type=str, default="obj*cls",
                        choices=["cls","obj","obj*cls"])
    parser.add_argument("--clip-cells", type=float, default=4.0)

    # 라벨/손실 하이퍼
    parser.add_argument("--eta-px", type=float, default=3.0)
    parser.add_argument("--lambda-cd", type=float, default=1.0)
    parser.add_argument("--k-pos-cap", type=int, default=96)
    parser.add_argument("--train-augment", action="store_true",
                        help="Apply default Albumentations pipeline to training samples.")
    parser.add_argument("--cls-focal", action="store_true",
                        help="Use focal loss for the classification head.")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)

    # 백본 프리즈
    parser.add_argument("--freeze-bb-epochs", type=int, default=1)

    # 가중치 관련
    parser.add_argument("--yolo-weights", type=str, default="yolo11m.pt")
    parser.add_argument("--weights", type=str, default=None)   # warm-start pth
    parser.add_argument("--resume", type=str, default=None)    # ckpt resume
    parser.add_argument("--start-epoch", type=int, default=0)

    # 공통 하이퍼
    parser.add_argument("--img-h", type=int, default=864)
    parser.add_argument("--img-w", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--lr-bb", type=float, default=2e-4)
    parser.add_argument("--lr-hd", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=1)

    # 저장 경로 & CSV
    parser.add_argument("--save-dir", type=str, default="./runs/2p5d")
    parser.add_argument("--log-csv", type=str, default=None)

    # ONNX export 옵션
    parser.add_argument("--export-onnx", action="store_true",
                        help="best pth 갱신 시 ONNX도 같이 export")
    parser.add_argument("--onnx-dir", type=str, default=None)
    parser.add_argument("--onnx-opset", type=int, default=12)

    args = parser.parse_args()

    IMG_SIZE = (args.img_h, args.img_w)
    TRAIN_BATCH = args.batch
    EPOCHS = args.epochs
    LR_MIN = args.lr_min
    WD = args.wd
    NUM_CLASSES = max(1, int(args.num_classes))

    # --------------------------
    # 경로 설정
    # --------------------------
    model_stem = Path(args.yolo_weights).stem
    save_dir = Path(args.save_dir).expanduser().resolve()
    pth_dir = save_dir / "pth"
    ckpt_dir = save_dir / "ckpt"
    pth_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ONNX 디렉토리
    if args.export_onnx:
        onnx_dir = Path(args.onnx_dir) if args.onnx_dir else (save_dir / "onnx")
        onnx_dir.mkdir(parents=True, exist_ok=True)
    else:
        onnx_dir = None

    # CSV 경로
    csv_path = Path(args.log_csv) if args.log_csv else (save_dir / f"{model_stem}_train_log.csv")
    need_header = (not csv_path.exists())

    # 데이터 경로 및 Dataset 생성
    # Detect flat vs multi-folder mode
    flat_img = os.path.join(args.train_root, "images")
    flat_lab = os.path.join(args.train_root, "labels")
    train_transform = None
    if args.train_augment:
        try:
            train_transform = build_default_train_augment(IMG_SIZE)
            print("[Augment] Training augmentation pipeline enabled.")
        except ImportError as exc:
            raise RuntimeError("Albumentations is required for --train-augment but is not installed.") from exc

    if os.path.isdir(flat_img) and os.path.isdir(flat_lab):
        # Flat mode
        train_dataset = ParallelogramDataset(
            img_dir=flat_img,
            label_dir=flat_lab,
            target_size=IMG_SIZE,
            transform=train_transform
        )
    else:
        # Multi-folder mode (root contains multiple cam folders)
        train_dataset = ParallelogramDataset(
            img_dir=args.train_root,
            label_dir=args.train_root,
            target_size=IMG_SIZE,
            transform=train_transform
        )

    if args.val_mode != "none":
        val_root = args.val_root or args.train_root
        val_img_dir = os.path.join(val_root, "images")
        val_label_dir = os.path.join(val_root, "labels")
    else:
        val_img_dir = val_label_dir = None


    # 평가 config
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

    # --------------------------
    # Dataset / DataLoader
    # --------------------------
    if len(train_dataset) == 0:
        print("No training data. Exit.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = None
    if args.val_mode != "none":
        val_dataset = ParallelogramDataset(val_img_dir, val_label_dir, target_size=IMG_SIZE)
        if len(val_dataset) == 0:
            print("No validation data. Exit.")
            return
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

    # --------------------------
    # Model / Loss / Optimizer
    # --------------------------
    model = YOLO11_2_5D(
        yolo11_path=args.yolo_weights,
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE
    ).to(DEVICE)

    if isinstance(model.strides, torch.Tensor):
        model.strides = [float(s.item()) for s in model.strides]
    else:
        model.strides = [float(s) for s in model.strides]

    criterion = Strict2_5DLoss(
        num_classes=NUM_CLASSES,
        eta_px=args.eta_px,
        obj_pos_weight=1.2,
        lambda_cd=args.lambda_cd,
        k_pos_cap=args.k_pos_cap,
        use_focal_cls=args.cls_focal,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha
    ).to(DEVICE)

    head_params = list(model.heads.parameters())
    bb_params = [p for n,p in model.named_parameters() if not n.startswith("heads.")]

    opt_all = optim.AdamW(
        [
            {"params": bb_params, "lr": args.lr_bb},
            {"params": head_params, "lr": args.lr_hd},
        ],
        betas=(0.9, 0.999),
        weight_decay=WD
    )

    sched_all = optim.lr_scheduler.CosineAnnealingLR(
        opt_all, T_max=EPOCHS, eta_min=LR_MIN
    )

    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    def set_requires_grad(module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    # CSV header
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow([
                "epoch","train_loss","val_loss",
                "precision","recall","map50","mAOE_deg",
                "lr_all","best_map"
            ])

    # best 값 초기화
    best_val = -1.0 if args.val_mode == "metrics" else float("inf")
    start_ep = 0

    # resume or warm-start
    if args.resume is not None:
        start_ep, best_val = load_ckpt(
            args.resume, DEVICE, model,
            opt_all, sched_all, scaler
        )
        print(f"[Resume] {args.resume} -> start_ep={start_ep}, best_val={best_val:.4f}")
    elif args.weights is not None:
        sd = torch.load(args.weights, map_location=DEVICE)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[Warm-start] {args.weights} (missing={len(missing)}, unexpected={len(unexpected)})")
        start_ep = int(args.start_epoch)

    val_desc = (f"{args.val_mode} (interval={args.val_interval})"
                if args.val_mode != "none" else "disabled")
    print(f"[Train] device={DEVICE}, epochs={EPOCHS}, start_ep={start_ep}, batch={TRAIN_BATCH}, strides={model.strides}, val={val_desc}")
    print(f"[Save] base='{model_stem}', save_dir='{save_dir}'")

    # ---------------------------------
    # 한 epoch 실행 함수
    # ---------------------------------
    def run_epoch(loader, train=True, epoch_idx=0, eval_cfg=None, val_mode="metrics"):
        warmup_ep = 5
        w = 0.5 + 0.5 * min(epoch_idx, warmup_ep-1) / (warmup_ep-1)
        criterion.set_p0_weight(w)

        if train:
            if epoch_idx < args.freeze_bb_epochs:
                set_requires_grad(model.backbone_neck, False)
            else:
                set_requires_grad(model.backbone_neck, True)

        model.train(mode=train)
        collect_metrics = (not train) and (eval_cfg is not None)
        metric_records = []
        metric_records_by_cls = defaultdict(list)
        gt_counts_by_cls = defaultdict(int)
        total_gt = 0

        tot = treg = tobj = tcls = tpos = tp0 = tp12 = 0.0
        nb = 0
        batches_seen = 0

        pbar = tqdm(loader, desc=f"{'Train' if train else 'Val'} {epoch_idx+1}/{EPOCHS} (p0_w={w:.2f})")
        for imgs, targets_cpu, _ in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            targets_dev = [
                {"points": t["points"].to(DEVICE),
                 "labels": t["labels"].to(DEVICE)}
                for t in targets_cpu
            ]

            if train:
                opt_all.zero_grad(set_to_none=True)

                with amp_autocast():
                    outs = model(imgs)
                    loss, logs = criterion(outs, targets_dev, model.strides)
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(opt_all)
                scaler.update()
            else:
                if val_mode == "loss":
                    with torch.inference_mode():
                        outs = model(imgs)
                        loss, logs = criterion(outs, targets_dev, model.strides)
                else:
                    with torch.inference_mode(), amp_autocast():
                        outs = model(imgs)
                    loss = torch.tensor(0., device=imgs.device)
                    logs = {
                        "loss_total":0,"loss_reg":0,"loss_obj":0,"loss_cls":0,
                        "loss_p0":0,"loss_p12":0,"pos":0,"neg":0
                    }

            tot  += logs["loss_total"]
            treg += logs["loss_reg"]
            tobj += logs["loss_obj"]
            tcls += logs["loss_cls"]
            tp0  += logs["loss_p0"]
            tp12 += logs["loss_p12"]
            tpos += logs["pos"]
            nb   += 1

            lr_all = opt_all.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=(tot/nb if nb>0 else 0),
                reg=(treg/nb if nb>0 else 0),
                obj=(tobj/nb if nb>0 else 0),
                p0=(tp0/nb if nb>0 else 0),
                p12=(tp12/nb if nb>0 else 0),
                cls=(tcls/nb if nb>0 else 0),
                pos=(tpos/nb if nb>0 else 0),
                lr_bb=lr_all,
            )

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
                decoded = [
                    tiny_filter_on_dets(dets_img=di, min_area=20.0, min_edge=3.0)
                    for di in decoded
                ]
                for dets_img, tgt_cpu in zip(decoded, targets_cpu):
                    gt_tri = tgt_cpu["points"].cpu().numpy()
                    if "labels" in tgt_cpu:
                        gt_cls = tgt_cpu["labels"].cpu().numpy()
                    else:
                        gt_cls = np.zeros((gt_tri.shape[0],), dtype=np.int64)
                    records, _ = evaluate_single_image(
                        dets_img, gt_tri, gt_cls,
                        iou_thr=eval_cfg.get("iou_thr", 0.5)
                    )
                    metric_records.extend(records)
                    for rec in records:
                        cls_id = rec[4] if len(rec) > 4 else 0
                        metric_records_by_cls[cls_id].append(rec)
                    for c in gt_cls:
                        gt_counts_by_cls[int(c)] += 1
                    total_gt += gt_tri.shape[0]

            batches_seen += 1
            if (not train) and args.val_max_batches is not None and batches_seen >= args.val_max_batches:
                break

        avg_loss = float(tot/nb) if nb>0 else 0.0
        metrics = None
        if collect_metrics:
            metrics = compute_detection_metrics(metric_records, total_gt)
            per_class_metrics = {}
            all_cls_ids = set(gt_counts_by_cls.keys()) | set(metric_records_by_cls.keys())
            for cls_id in sorted(all_cls_ids):
                recs = metric_records_by_cls.get(cls_id, [])
                per_class_metrics[cls_id] = compute_detection_metrics(
                    recs, gt_counts_by_cls.get(cls_id, 0)
                )
            metrics["per_class"] = per_class_metrics
            metrics["gt_counts_by_cls"] = dict(gt_counts_by_cls)
        return avg_loss, metrics

    # ---------------------------------
    # Train Loop
    # ---------------------------------
    for ep in range(start_ep, EPOCHS):
        train_loss, _ = run_epoch(
            train_loader, train=True, epoch_idx=ep,
            eval_cfg=eval_cfg, val_mode=args.val_mode
        )

        run_val = (
            args.val_mode != "none"
            and val_loader is not None
            and (((ep+1) % max(1,args.val_interval) == 0) or (ep == EPOCHS-1))
        )

        val_loss = None
        curr_map = curr_prec = curr_rec = curr_maoe = None

        if run_val:
            val_loss, val_metrics = run_epoch(
                val_loader, train=False, epoch_idx=ep,
                eval_cfg=eval_cfg, val_mode=args.val_mode
            )
            if val_metrics is not None:
                curr_prec = float(val_metrics["precision"])
                curr_rec  = float(val_metrics["recall"])
                curr_map  = float(val_metrics["map50"])
                curr_maoe = float(val_metrics["mAOE_deg"])
                per_cls = val_metrics.get("per_class", {})
                msg = "  -> metrics P={:.4f} R={:.4f} mAP@50={:.4f} mAOE(deg)={:.2f}".format(
                    curr_prec, curr_rec, curr_map, curr_maoe
                )
                if per_cls:
                    parts = []
                    for cid, m in sorted(per_cls.items()):
                        parts.append(f"[cls{cid}:P={m['precision']:.3f},R={m['recall']:.3f},mAP={m['map50']:.3f}]")
                    msg += " " + " ".join(parts)
                print(msg)
                # mAP 기준 best 갱신
                if curr_map > best_val:
                    best_val = curr_map
                    best_pth = pth_dir / f"{model_stem}_2_5d_best.pth"
                    torch.save(model.state_dict(), best_pth)
                    print(f"  -> best (mAP@50) updated to {best_val:.4f}, saved: {best_pth}")

                    # ONNX export (옵션)
                    if args.export_onnx and onnx_dir is not None:
                        best_onnx = onnx_dir / f"{model_stem}_2_5d_best.onnx"
                        export_onnx(model, best_onnx, IMG_SIZE, DEVICE, opset=args.onnx_opset)
            else:
                # val_loss 기준 best
                if (val_loss is not None) and (args.val_mode == "loss") and (val_loss < best_val):
                    best_val = val_loss
                    best_pth = pth_dir / f"{model_stem}_2_5d_best.pth"
                    torch.save(model.state_dict(), best_pth)
                    print(f"  -> best (val_loss) updated {best_val:.4f}, saved: {best_pth}")
                    if args.export_onnx and onnx_dir is not None:
                        best_onnx = onnx_dir / f"{model_stem}_2_5d_best.onnx"
                        export_onnx(model, best_onnx, IMG_SIZE, DEVICE, opset=args.onnx_opset)

        # 스케줄러
        sched_all.step()

        # epoch별 pth & ckpt 저장
        pth_path  = pth_dir  / f"{model_stem}_2_5d_epoch_{ep+1:03d}.pth"
        ckpt_path = ckpt_dir / f"{model_stem}_2_5d_epoch_{ep+1:03d}.ckpt"
        torch.save(model.state_dict(), pth_path)
        save_ckpt(
            ckpt_path, epoch=ep+1, model=model,
            opt_all=opt_all, sched_all=sched_all,
            scaler=scaler, best_val=best_val
        )
        print(f"  -> saved {pth_path} and {ckpt_path}")

        # CSV 로깅
        lr_all = opt_all.param_groups[0]["lr"]
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep+1,
                f"{train_loss:.6f}",
                "" if val_loss is None else f"{val_loss:.6f}",
                "" if curr_prec is None else f"{curr_prec:.6f}",
                "" if curr_rec  is None else f"{curr_rec:.6f}",
                "" if curr_map  is None else f"{curr_map:.6f}",
                "" if curr_maoe is None else f"{curr_maoe:.6f}",
                f"{lr_all:.6e}",
                f"{best_val:.6f}" if isinstance(best_val,float) else best_val
            ])

    print("Done.")


if __name__ == "__main__":
    main()

"""
python -m src.train --train-root /home/ubuntu24/code/new_start/data/clean/ \
 --val-root /home/ubuntu24/다운로드/dataset_1125_undist_2_5d_conf_1/cam12_37_5 \
 --weights runs/yolo11_2_5d_real_coshow_v5/pth/yolo11m_2_5d_epoch_008.pth \
 --export-onnx --save-dir ./runs/yolo11_2_5d_real_coshow_v5 \
 --start-epoch 9
"""
