import argparse
import glob
import os.path as osp
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
import torchvision
import numpy as np
import cv2
import onnx
import onnxruntime as ort

# =========================================
# Constants
# =========================================
# Letterbox 크기 (H, W)
DEFAULT_LETTERBOX_SIZE = (832, 1440)

# Runtime device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================================
# LetterBox (논문 코드와 동일 로직)
# =========================================
class LetterBox:
    def __init__(self,
                 shape: Tuple[int, int] = DEFAULT_LETTERBOX_SIZE,
                 auto: bool = False,
                 scale_fill: bool = False,
                 scale_up: bool = True,
                 center: bool = True,
                 stride: int = 32):
        self.new_shape = shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scale_up
        self.stride = stride
        self.center = center

    def __call__(self, image: np.ndarray, labels: Dict[str, Any] = None):
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # (H0, W0)
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)  # (H, W)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down
            r = min(r, 1.0)

        # Compute padding
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (W', H')
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # (W - W', H - H')
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])

        if self.center:
            dw /= 2
            dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = (int(round(dh - 0.1)) if self.center else 0), int(round(dh + 0.1))
        left, right = (int(round(dw - 0.1)) if self.center else 0), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img


# =========================================
# Inferencer (ONNX)
# - run() → (N,7) = [cx,cy,f1x,f1y,f2x,f2y, score]  (원본 해상도 좌표)
# =========================================
class Inferencer:
    def __init__(self,
                 model_path: str,
                 letterbox_size: Tuple[int, int] = DEFAULT_LETTERBOX_SIZE,
                 is_half: bool = True):
        self.is_half = is_half
        self.letterbox_size = letterbox_size  # (H, W)

        # ONNX 로드 및 세션 생성 (CUDA 우선, 없으면 CPU)
        self.model = onnx.load(model_path)
        avail = ort.get_available_providers()
        providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in avail else ['CPUExecutionProvider']
        self.use_cuda = ('CUDAExecutionProvider' in providers)
        self.sess = ort.InferenceSession(model_path, providers=providers)

        # 입출력 이름(가정: x, y) — 다른 경우 필요에 따라 바꿔야 함
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def run(self, image: np.ndarray) -> torch.Tensor:
        """
        Args:
            image (np.ndarray, BGR): 원본 이미지
        Returns:
            det_out (torch.Tensor): (N,7) = [cx,cy,f1x,f1y,f2x,f2y, score]  (원본 해상도 좌표)
        """
        H0, W0 = image.shape[:2]

        # 출력 버퍼(고정 크기 모델 가정)
        y = torch.empty((1, 7, 24570),
                        dtype=(torch.float16 if self.is_half else torch.float32),
                        device=(device if self.use_cuda else 'cpu')).contiguous()

        # 전처리(레터박스 + 정규화 + CHW)
        x = self._preprocess(image)  # (1,3,Hl,Wl) torch
        if self.is_half:
            x = x.half()

        if self.use_cuda:
            # CUDA IO Binding
            assert x.is_cuda, "CUDAExecutionProvider 사용 시 입력 텐서는 CUDA 여야 합니다."
            assert y.is_cuda, "CUDAExecutionProvider 사용 시 출력 텐서는 CUDA 여야 합니다."
            binding = self.sess.io_binding()
            binding.bind_input(self.input_name,
                               device_type='cuda',
                               device_id=x.get_device(),
                               element_type=(np.float16 if self.is_half else np.float32),
                               shape=x.shape,
                               buffer_ptr=x.data_ptr())
            binding.bind_output(self.output_name,
                                device_type='cuda',
                                device_id=x.get_device(),
                                element_type=(np.float16 if self.is_half else np.float32),
                                shape=y.shape,
                                buffer_ptr=y.data_ptr())
            self.sess.run_with_iobinding(binding)
            y_nms_in = y  # torch on CUDA
        else:
            # CPU 경로 (디버그/폴백)
            x_np = x.cpu().numpy().astype(np.float16 if self.is_half else np.float32)
            outputs = self.sess.run([self.output_name], {self.input_name: x_np})
            y_np = outputs[0]
            y_nms_in = torch.from_numpy(y_np)

        # NMS (출력 텐서는 (1,84,6300) 같은 형태를 가정; _non_max_suppression 내부에서 전치)
        y_nms = self._non_max_suppression(y_nms_in)[0]  # (N, 8+nm) [cx,cy,f1x,f1y,f2x,f2y, conf, cls]
        if y_nms.numel() == 0:
            # 빈 결과
            return y_nms.new_zeros((0, 7)).to(device if self.use_cuda else 'cpu')

        # 좌표를 원본 해상도로 복원 (letterbox → original)
        tris_scaled = self._scale_triangles(
            y_nms[:, :6],                         # (N,6)
            src_shape=self.letterbox_size,        # (Hl, Wl)
            target_shape=(H0, W0, 3),
            padding=True
        )

        scores = y_nms[:, 6:7]                    # (N,1)
        det_out = torch.cat([tris_scaled, scores], dim=1)  # (N,7)
        return det_out

    # -----------------------------------------------------
    # Helper: NMS에 사용할 AABB 생성
    # -----------------------------------------------------
    def _compute_bounding_boxes(self, triangles: torch.Tensor) -> torch.Tensor:
        """
        triangles: (N,6) = (cx,cy,f1x,f1y,f2x,f2y)
        평행사변형의 AABB를 생성 (cx,cy에서 f1,f2를 미러링)
        """
        cx, cy = triangles[:, 0], triangles[:, 1]
        x2, y2 = triangles[:, 2], triangles[:, 3]
        x3, y3 = triangles[:, 4], triangles[:, 5]

        # mirror points
        x2m, y2m = 2 * cx - x2, 2 * cy - y2
        x3m, y3m = 2 * cx - x3, 2 * cy - y3

        x_coords = torch.stack([cx, x2, x3, x2m, x3m], dim=1)
        y_coords = torch.stack([cy, y2, y3, y2m, y3m], dim=1)

        min_x, _ = torch.min(x_coords, dim=1)
        max_x, _ = torch.max(x_coords, dim=1)
        min_y, _ = torch.min(y_coords, dim=1)
        max_y, _ = torch.max(y_coords, dim=1)
        return torch.stack([min_x, min_y, max_x, max_y], dim=1)  # (N,4)

    def _nms_bboxes(self, triangles: torch.Tensor, confidences: torch.Tensor, iou_threshold: float):
        bboxes = self._compute_bounding_boxes(triangles)
        return torchvision.ops.nms(bboxes, confidences, iou_threshold)

    # -----------------------------------------------------
    # NMS (기존 코드 유지, 좌표 정의만 triangles에 맞춰서 사용)
    # -----------------------------------------------------
    def _non_max_suppression(
        self,
        prediction: torch.Tensor,
        conf_thres: float = 0.01,
        iou_thres: float = 0.5,
        classes: torch.Tensor = None,
        agnostic: bool = False,
        multi_label: bool = False,
        max_det: int = 300,
        nc: int = 0,  # number of classes
        max_nms: int = 30000,
        max_wh: int = 7680,
    ):
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}'

        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        if classes is not None:
            classes = torch.tensor(classes, device=prediction.device)

        bs = prediction.shape[0]
        nc = nc or (prediction.shape[1] - 6)  # num classes
        nm = prediction.shape[1] - nc - 6     # num masks
        mi = 6 + nc
        xc = prediction[:, 6:mi].amax(1) > conf_thres

        # (1,84,6300) -> (1,6300,84)
        prediction = prediction.transpose(-1, -2)

        output = [torch.zeros((0, 8 + nm), device=prediction.device)] * bs

        for xi, x in enumerate(prediction):
            x = x[xc[xi]]

            # split
            box, cls, mask = x.split((6, nc, nm), 1)

            if multi_label:
                i, j = torch.where(cls > conf_thres)
                x = torch.cat((box[i], x[i, 6 + j, None], j[:, None].float(), mask[i]), 1)
            else:
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            if classes is not None:
                x = x[(x[:, 7:8] == classes).any(1)]

            n = x.shape[0]
            if n > 0:
                if n > max_nms:
                    x = x[x[:, 6].argsort(descending=True)[:max_nms]]

                # class offset (여기서는 클래스 1개 가정 가능)
                c = x[:, 7:8] * (0 if agnostic else max_wh)
                scores = x[:, 6]
                # triangles에 class offset을 더할 필요는 없음 (AABB 만들 때만 사용)
                i = self._nms_bboxes(x[:, :6], scores, iou_thres)
                i = i[:max_det]
                output[xi] = x[i]

        return output

    # -----------------------------------------------------
    # Letterbox → Original 좌표 복원
    # -----------------------------------------------------
    def _scale_triangles(self,
                         triangles: torch.Tensor,
                         src_shape: Tuple[int, int],
                         target_shape: Union[Tuple[int, int], Tuple[int, int, int]],
                         padding: bool = True):
        """
        triangles: (N,6) [cx,cy,f1x,f1y,f2x,f2y] (src_shape 기준)
        src_shape: (Hl, Wl) letterbox 크기
        target_shape: (H0, W0) or (H0, W0, C) 원본 크기
        """
        if isinstance(target_shape, tuple) and len(target_shape) == 3:
            Ht, Wt = target_shape[0], target_shape[1]
        else:
            Ht, Wt = target_shape  # type: ignore

        Hl, Wl = src_shape
        tris = triangles.clone()

        # letterbox 비율 및 패딩 (원래 forward에서 original -> letterbox 했던 것의 역변환)
        gain = min(Hl / Ht, Wl / Wt)
        pad_w = round((Wl - Wt * gain) / 2 - 0.1)
        pad_h = round((Hl - Ht * gain) / 2 - 0.1)

        if padding:
            tris[..., 0] -= pad_w  # cx
            tris[..., 1] -= pad_h  # cy
            tris[..., 2] -= pad_w  # f1x
            tris[..., 3] -= pad_h  # f1y
            tris[..., 4] -= pad_w  # f2x
            tris[..., 5] -= pad_h  # f2y

        tris[..., :6] /= gain
        return tris

    # -----------------------------------------------------
    # Preprocess
    # -----------------------------------------------------
    def _preprocess(self, im: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            # Letterbox는 BGR 이미지를 그대로 받음
            im = np.stack([LetterBox(self.letterbox_size, auto=False, stride=32)(im)])
            # BGR -> RGB, BHWC -> BCHW
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)

        im = im.to(device if self.use_cuda else 'cpu')
        im = im.float()
        if not_tensor:
            im /= 255.0
        return im


# =========================================
# (아래 유틸들은 기존 스크립트 호환용 — 필요시 사용)
# =========================================
def visualize_detections(detections: np.ndarray, image: np.ndarray, save_path: Optional[str] = None) -> Image:
    """
    detections: (N, 6) or (N, 7) [cx,cy,f1x,f1y,f2x,f2y,(score)]
    """
    img = image.copy()
    for det in detections:
        tri = det[:6]
        cx, cy, x2, y2, x3, y3 = tri

        # cx,cy를 기준으로 f1,f2 미러링 → 평행사변형
        x2m, y2m = 2 * cx - x2, 2 * cy - y2
        x3m, y3m = 2 * cx - x3, 2 * cy - y3

        poly = np.array([[x2, y2],
                         [x3, y3],
                         [x2m, y2m],
                         [x3m, y3m]], dtype=int)

        cv2.polylines(img, [poly], isClosed=True, color=(0, 0, 255), thickness=2)  # red

    if save_path is not None:
        cv2.imwrite(save_path, img)

    return Image.fromarray(img[..., ::-1])  # PIL은 RGB


def results_to_df(results: torch.Tensor, ts: float = time.time()) -> pd.DataFrame:
    """
    results: torch.Tensor (N,6) or (N,7)
    """
    res = results.detach().cpu().numpy()
    d: Dict[str, Any] = {
        'ts': ts,
        'cx': res[:, 0],
        'cy': res[:, 1],
        'f1x': res[:, 2],
        'f1y': res[:, 3],
        'f2x': res[:, 4],
        'f2y': res[:, 5],
    }
    if res.shape[1] >= 7:
        d['score'] = res[:, 6]
    return pd.DataFrame(d)


def guess_timestamp(filename: str) -> float:
    if match := re.search(r"(\d{10}-\d+)", filename):
        return float(match.group(1).replace('-', '.'))
    return 0.0


# =========================================
# (옵션) 단일 이미지/글롭 실행 스크립트
# =========================================
def parse_args():
    parser = argparse.ArgumentParser(description='Inference script for ONNX model')
    parser.add_argument('image', type=str, nargs='?', default=None,
                        help='Path to a single input image or a glob pattern')
    parser.add_argument('--model', type=str, default='checkpoints/model_v2_half.onnx',
                        help='Path to the ONNX model file')
    parser.add_argument('--ts', type=float, default=time.time(),
                        help='Timestamp to use in CSV output')
    parser.add_argument('--guess-ts', action='store_true',
                        help='Try to infer image timestamp from image name')
    parser.add_argument('--half', action='store_true',
                        help='Provided model is in half-precision (float16)')
    parser.add_argument('--viz', action='store_true',
                        help='Save visualization images')
    return parser.parse_args()


def main():
    print('Initialized')
    args = parse_args()
    if args.image is None:
        print('No image/glob provided; exiting.')
        return

    print('Preparing inferencer')
    inferencer = Inferencer(args.model, letterbox_size=DEFAULT_LETTERBOX_SIZE, is_half=args.half)

    dfs: List[pd.DataFrame] = []

    for filename in glob.glob(args.image):
        print(f'Loading image {filename}')
        image = cv2.imread(filename)

        ts = args.ts
        if args.guess_ts:
            ts = guess_timestamp(filename) or ts

        print('Running inference')
        t0 = time.monotonic()
        results: torch.Tensor = inferencer.run(image)  # (N,7)
        t1 = time.monotonic()

        if args.viz:
            out_file_ext = filename.split('.')[-1]
            out_file = f'{osp.basename(filename).rstrip("." + out_file_ext)}_out.{out_file_ext}'
            visualize_detections(results.detach().cpu().numpy(), image, save_path=out_file)

        print(f'Got {results.shape[0]} detections, took {t1 - t0:.3f} sec')
        dfs.append(results_to_df(results, ts))

    print('Saving output')
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df.to_csv('detections.csv', header=True, index=False)
    print('Done.')


if __name__ == '__main__':
    main()