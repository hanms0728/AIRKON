import argparse
import cv2
import glob
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch

# 기존 추론 파이프라인과 동일한 로직을 재사용하기 위해 가져온다.
from src.inference_lstm_onnx_pointcloud_add_color import (
    decode_predictions,
    draw_pred_only,
    tiny_filter_on_dets,
)

class _ONNXColorRunner:
    """
    color.py용 간단한 ONNX 실행기.
    - 입력: images + (옵션) h_in, c_in
    - 출력: reg/obj/cls + (옵션) h_out, c_out
    - 상태 버퍼는 내부에 보관한다.
    """
    def __init__(
        self,
        onnx_path: str,
        providers: Optional[Sequence[str]] = None,
        *,
        state_stride_hint: int = 32,
        default_hidden_ch: int = 256,
    ):
        self.sess = ort.InferenceSession(
            str(onnx_path),
            providers=providers or ort.get_available_providers()
        )
        self.inputs = {i.name: i for i in self.sess.get_inputs()}
        self.outs = [o.name for o in self.sess.get_outputs()]

        x_cands = [n for n in self.inputs if n.lower() in ("images", "image", "input")]
        self.x_name = x_cands[0] if x_cands else list(self.inputs.keys())[0]
        self.h_name = next((n for n in self.inputs if "h_in" in n.lower()), None)
        self.c_name = next((n for n in self.inputs if "c_in" in n.lower()), None)

        self.ho_name = next((n for n in self.outs if "h_out" in n.lower()), None)
        self.co_name = next((n for n in self.outs if "c_out" in n.lower()), None)
        self.reg_names = [n for n in self.outs if "reg" in n.lower()]
        self.obj_names = [n for n in self.outs if "obj" in n.lower()]
        self.cls_names = [n for n in self.outs if "cls" in n.lower()]

        def _sort_key(s):
            toks = []
            acc = ""
            for ch in s:
                if ch.isdigit():
                    acc += ch
                else:
                    if acc:
                        toks.append(int(acc))
                        acc = ""
                    toks.append(ch)
            if acc:
                toks.append(int(acc))
            return tuple(toks)

        self.reg_names.sort(key=_sort_key)
        self.obj_names.sort(key=_sort_key)
        self.cls_names.sort(key=_sort_key)

        self.state_stride_hint = int(state_stride_hint)
        self.default_hidden_ch = int(default_hidden_ch)
        self.h_shape_meta = self._shape_from_input_meta(self.h_name)
        self.c_shape_meta = self._shape_from_input_meta(self.c_name)

        self.input_hw = self._input_hw_from_shape(self.inputs[self.x_name].shape)
        self.h_buf = None
        self.c_buf = None

    def _input_hw_from_shape(self, shape) -> Tuple[int, int]:
        """입력 텐서 shape로부터 (H, W)를 추정한다. 동적이면 0 반환."""
        if len(shape) >= 4:
            h, w = shape[2], shape[3]
            if isinstance(h, int) and isinstance(w, int):
                return h, w
        return 0, 0

    def _shape_from_input_meta(self, name):
        if name is None:
            return None
        meta = self.inputs[name].shape  # [N,C,Hs,Ws] with possible None

        def _to_int(val, default):
            return int(val) if isinstance(val, (int, np.integer)) else default

        N = _to_int(meta[0], 1)
        C = _to_int(meta[1], self.default_hidden_ch)
        Hs = _to_int(meta[2], 0)
        Ws = _to_int(meta[3], 0)
        return [N, C, Hs, Ws]

    def reset(self):
        self.h_buf = None
        self.c_buf = None

    def _ensure_state(self, img_numpy_chw: np.ndarray):
        # img: (1,3,H,W)
        _, _, H, W = img_numpy_chw.shape
        if self.h_name and self.h_buf is None:
            N, C, Hs, Ws = self.h_shape_meta
            if Hs == 0 or Ws == 0:
                Hs = max(1, H // self.state_stride_hint)
                Ws = max(1, W // self.state_stride_hint)
            self.h_buf = np.zeros((N, C, Hs, Ws), dtype=np.float32)
        if self.c_name and self.c_buf is None:
            N, C, Hs, Ws = self.c_shape_meta
            if Hs == 0 or Ws == 0:
                Hs = max(1, H // self.state_stride_hint)
                Ws = max(1, W // self.state_stride_hint)
            self.c_buf = np.zeros((N, C, Hs, Ws), dtype=np.float32)

    def forward(self, img_numpy_chw: np.ndarray):
        """img_numpy_chw: (1,3,H,W) float32 [0..1]"""
        self._ensure_state(img_numpy_chw)

        feeds = {self.x_name: img_numpy_chw}
        if self.h_name is not None and self.h_buf is not None:
            feeds[self.h_name] = self.h_buf
        if self.c_name is not None and self.c_buf is not None:
            feeds[self.c_name] = self.c_buf

        outs = self.sess.run(self.outs, feeds)
        out_map = {n: v for n, v in zip(self.outs, outs)}

        if self.ho_name:
            self.h_buf = out_map[self.ho_name]
        if self.co_name:
            self.c_buf = out_map[self.co_name]

        pred_list = []
        for rn, on, cn in zip(self.reg_names, self.obj_names, self.cls_names):
            pr = torch.from_numpy(out_map[rn])
            po = torch.from_numpy(out_map[on])
            pc = torch.from_numpy(out_map[cn])
            pred_list.append((pr, po, pc))
        return pred_list


# 간단한 캐시로 매번 ONNX를 다시 읽지 않도록 한다 (상태 포함).
_RUNNERS = {}


def _get_runner(
    onnx_path: str,
    providers: Optional[Sequence[str]] = None,
    *,
    state_stride_hint: int = 32,
    default_hidden_ch: int = 256,
):
    key = (
        str(onnx_path),
        tuple(providers or []),
        int(state_stride_hint),
        int(default_hidden_ch),
    )
    if key in _RUNNERS:
        return _RUNNERS[key]
    runner = _ONNXColorRunner(
        onnx_path,
        providers=providers,
        state_stride_hint=state_stride_hint,
        default_hidden_ch=default_hidden_ch,
    )
    _RUNNERS[key] = runner
    return runner


def _hex_to_rgb(hex_str: str) -> Optional[np.ndarray]:
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) != 6:
        return None
    try:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
    except ValueError:
        return None
    return np.array([r, g, b], dtype=np.float32)


def _parse_comma_list_int(text: str) -> Sequence[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_comma_list_str(text: Optional[str]) -> Optional[Sequence[str]]:
    if text is None:
        return None
    vals = [x.strip() for x in text.split(",") if x.strip()]
    return vals if vals else None


def _detect_hex_colors(
    img_rgb: np.ndarray,
    *,
    onnx_path: str,
    strides: Sequence[int],
    conf: float,
    nms_iou: float,
    topk: int,
    score_mode: str = "obj*cls",
    providers: Optional[Sequence[str]] = None,
    state_stride_hint: int = 32,
    default_hidden_ch: int = 256,
) -> Optional[str]:
    """ONNX 모델로 객체를 추론하고 draw_pred_only 로직을 그대로 사용해 첫 번째 헥사코드를 얻는다."""
    runner = _get_runner(
        onnx_path,
        providers,
        state_stride_hint=state_stride_hint,
        default_hidden_ch=default_hidden_ch,
    )
    H0, W0 = img_rgb.shape[:2]

    target_h, target_w = runner.input_hw
    if target_h <= 0 or target_w <= 0:
        target_h, target_w = H0, W0

    img_rgb_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img_np = img_rgb_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_np = np.expand_dims(img_np, 0)

    outs = runner.forward(img_np)

    dets = decode_predictions(
        outs,
        list(map(float, strides)),
        conf_th=conf,
        nms_iou=nms_iou,
        topk=topk,
        score_mode=score_mode,
        use_gpu_nms=False,
    )[0]
    dets = tiny_filter_on_dets(dets, min_area=20.0, min_edge=3.0)

    # draw_pred_only 내부에서 ROI 평균을 사용해 hex를 계산한다.
    _, hex_list = draw_pred_only(
        cv2.cvtColor(img_rgb_resized, cv2.COLOR_RGB2BGR),
        dets,
        save_path_img=None,
        save_path_txt=None,
        W=target_w,
        H=target_h,
        W0=W0,
        H0=H0,
    )
    if not hex_list:
        return None
    return hex_list[0]


def get_center_mean_color(
    img_rgb,
    crop_ratio=0.5,
    *,
    onnx_path: Optional[str] = None,
    strides: Sequence[int] = (8, 16, 32),
    conf: float = 0.3,
    nms_iou: float = 0.2,
    topk: int = 50,
    score_mode: str = "obj*cls",
    providers: Optional[Sequence[str]] = None,
    state_stride_hint: int = 32,
    default_hidden_ch: int = 256,
):
    """
    추론 파이프라인을 통해 객체 영역에서 얻은 헥사코드를 RGB로 반환한다.
    - onnx_path가 없거나 추론 실패/헥사코드 미탐지 시 None 반환 (더 이상 중앙 크롭 fallback 없음).
    """
    _ = crop_ratio  # 호환성용 인자 (더 이상 사용하지 않음)
    if not onnx_path:
        return None

    hex_code = _detect_hex_colors(
        img_rgb,
        onnx_path=str(Path(onnx_path)),
        strides=strides,
        conf=conf,
        nms_iou=nms_iou,
        topk=topk,
        score_mode=score_mode,
        providers=providers,
        state_stride_hint=state_stride_hint,
        default_hidden_ch=default_hidden_ch,
    )

    rgb_from_model = _hex_to_rgb(hex_code) if hex_code else None
    return rgb_from_model

def extract_object_color_range(
    folder_path,
    crop_ratio=0.5,
    margin=10,
    *,
    onnx_path: Optional[str] = None,
    strides: Sequence[int] = (8, 16, 32),
    conf: float = 0.3,
    nms_iou: float = 0.2,
    topk: int = 50,
    score_mode: str = "obj*cls",
    providers: Optional[Sequence[str]] = None,
    state_stride_hint: int = 32,
    default_hidden_ch: int = 256,
    output_path: Optional[str] = None,
):
    # 이미지 확장자들
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_paths:
        print("이미지가 없습니다. 폴더 경로를 확인해줘.")
        return None
    
    colors = []  # 각 이미지에서 얻은 [R, G, B]
    log_lines = []
    
    for img_path in image_paths:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"이미지를 읽을 수 없음: {img_path}")
            continue
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 추론 기반 색상 추출 (실패 시 None)
        mean_color = get_center_mean_color(
            img_rgb,
            crop_ratio=crop_ratio,
            onnx_path=onnx_path,
            strides=strides,
            conf=conf,
            nms_iou=nms_iou,
            topk=topk,
            score_mode=score_mode,
            providers=providers,
            state_stride_hint=state_stride_hint,
            default_hidden_ch=default_hidden_ch,
        )
        if mean_color is None:
            print(f"{os.path.basename(img_path)} -> 색상 추출 실패 (None)")
            log_lines.append(f"{os.path.basename(img_path)} -> None")
            continue

        colors.append(mean_color)
        msg = f"{os.path.basename(img_path)} -> mean RGB: {mean_color.astype(int)}"
        print(msg)
        log_lines.append(msg)
    
    if not colors:
        print("유효한 이미지에서 색을 추출하지 못했어.")
        return None
    
    colors = np.array(colors)  # shape: (N, 3)
    
    # 채널별 최소/최대
    rgb_min = colors.min(axis=0)  # [R_min, G_min, B_min]
    rgb_max = colors.max(axis=0)  # [R_max, G_max, B_max]
    
    # margin 적용 (0~255 클램프)
    rgb_min_margin = np.clip(rgb_min - margin, 0, 255)
    rgb_max_margin = np.clip(rgb_max + margin, 0, 255)
    
    print("\n==== 객체 RGB 범위 (원값 기준) ====")
    print(f"R: {rgb_min[0]:.1f} ~ {rgb_max[0]:.1f}")
    print(f"G: {rgb_min[1]:.1f} ~ {rgb_max[1]:.1f}")
    print(f"B: {rgb_min[2]:.1f} ~ {rgb_max[2]:.1f}")
    
    print("\n==== 객체 RGB 범위 (margin 포함, threshold용) ====")
    print(f"R: {rgb_min_margin[0]:.0f} ~ {rgb_max_margin[0]:.0f}")
    print(f"G: {rgb_min_margin[1]:.0f} ~ {rgb_max_margin[1]:.0f}")
    print(f"B: {rgb_min_margin[2]:.0f} ~ {rgb_max_margin[2]:.0f}")
    
    result = {
        "per_image_colors": colors,
        "rgb_min": rgb_min,
        "rgb_max": rgb_max,
        "rgb_min_margin": rgb_min_margin,
        "rgb_max_margin": rgb_max_margin,
    }

    # 결과를 파일로 저장
    out_path = Path(output_path) if output_path else Path(folder_path) / "color_ranges.txt"
    out_lines = []
    out_lines.append("==== 객체 RGB 범위 (원값 기준) ====")
    out_lines.append(f"R: {rgb_min[0]:.1f} ~ {rgb_max[0]:.1f}")
    out_lines.append(f"G: {rgb_min[1]:.1f} ~ {rgb_max[1]:.1f}")
    out_lines.append(f"B: {rgb_min[2]:.1f} ~ {rgb_max[2]:.1f}")
    out_lines.append("")
    out_lines.append("==== 객체 RGB 범위 (margin 포함) ====")
    out_lines.append(f"R: {rgb_min_margin[0]:.0f} ~ {rgb_max_margin[0]:.0f}")
    out_lines.append(f"G: {rgb_min_margin[1]:.0f} ~ {rgb_max_margin[1]:.0f}")
    out_lines.append(f"B: {rgb_min_margin[2]:.0f} ~ {rgb_max_margin[2]:.0f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n결과 저장: {out_path}")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="객체 색상 헥사코드→RGB 추출 도구")
    parser.add_argument("--folder", required=True, help="이미지 폴더 경로")
    parser.add_argument("--onnx_path", required=True, help="객체 탐지 ONNX 파일 경로")
    parser.add_argument("--strides", default="8,16,32", help="모델 stride 목록 (콤마 구분)")
    parser.add_argument("--conf", type=float, default=0.3, help="confidence threshold")
    parser.add_argument("--nms_iou", type=float, default=0.2, help="NMS IoU threshold")
    parser.add_argument("--topk", type=int, default=50, help="NMS top-k")
    parser.add_argument("--score_mode", default="obj*cls", help="score_mode (ex: obj*cls)")
    parser.add_argument("--providers", default=None, help="ONNX Runtime providers (콤마 구분, 미지정 시 CUDA>CPU)")
    parser.add_argument("--crop_ratio", type=float, default=0.5, help="(미사용) 기존 중앙 크롭 비율")
    parser.add_argument("--margin", type=int, default=10, help="RGB 범위 출력 시 margin")
    parser.add_argument("--state_stride_hint", type=int, default=32, help="상태 feature map stride 추정값")
    parser.add_argument("--default_hidden_ch", type=int, default=256, help="상태 채널 기본값 (메타 정보 없을 때)")
    parser.add_argument("--output", type=str, default=None, help="결과 저장 파일 경로 (default: <folder>/color_ranges.txt)")

    args = parser.parse_args()

    strides = _parse_comma_list_int(args.strides)
    providers = _parse_comma_list_str(args.providers)
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    extract_object_color_range(
        args.folder,
        crop_ratio=args.crop_ratio,
        margin=args.margin,
        onnx_path=args.onnx_path,
        strides=strides,
        conf=args.conf,
        nms_iou=args.nms_iou,
        topk=args.topk,
        score_mode=args.score_mode,
        providers=providers,
        state_stride_hint=args.state_stride_hint,
        default_hidden_ch=args.default_hidden_ch,
        output_path=args.output,
    )
