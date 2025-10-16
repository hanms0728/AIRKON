# python edge_oak.py --cam-name cam1 --port 50050 \
#   --onnx yolo11m_2_5d_epoch_005.onnx --device cpu \
#   --H "1,0,0,0,1,0,0,0,1"

from evaluation_utils import decode_predictions   # det 디코딩
from geometry_utils import tiny_filter_on_dets 

# edge_oak.py
import os, time, json, math, argparse, socket
import numpy as np
import cv2

# ---------- OAK ----------
import depthai as dai  # pip install depthai

# ---------- ONNX ----------
try:
    import onnxruntime as ort
    HAS_ORT = True
except Exception:
    HAS_ORT = False

# ==============================
# 공통 유틸
# ==============================
def parse_H(csv9: str) -> np.ndarray:
    vals = [float(v) for v in csv9.split(",")]
    if len(vals) != 9:
        raise ValueError("--H 는 콤마로 구분된 9개 실수여야 함")
    return np.array(vals, dtype=np.float32).reshape(3, 3)

def imgpt_to_bevpt(xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    if xy.ndim == 1:
        xy = xy[None, :]
    ones = np.ones((xy.shape[0], 1), dtype=np.float32)
    pts = np.concatenate([xy.astype(np.float32), ones], axis=1)
    t = (H @ pts.T).T
    denom = np.clip(t[:, 2:3], 1e-9, None)
    out = t[:, :2] / denom
    return out

def obb_quad_from_cxy_lw_yaw(cx, cy, L, W, yaw_deg):
    th = math.radians(yaw_deg)
    dx, dy = L/2.0, W/2.0
    corners = np.array([[ dx,  dy],[ dx,-dy],[-dx,-dy],[-dx, dy]], dtype=np.float32)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c,-s],[s,c]], dtype=np.float32)
    quad = corners @ R.T + np.array([cx, cy], dtype=np.float32)
    return quad

def draw_obb(img_bgr, cx, cy, L, W, yaw_deg, color=(0,255,0), lw=2):
    quad = obb_quad_from_cxy_lw_yaw(cx, cy, L, W, yaw_deg)
    poly = np.vstack([quad, quad[0]]).astype(np.int32)
    for i in range(4):
        cv2.line(img_bgr, poly[i], poly[i+1], color, lw)
    return img_bgr

# ==============================
# OAK 스트림 (1920x1080)
# ==============================
class OAKStream:
    def __init__(self, fps=30, mxid=None, video_size=(1920,1080)):
        self.device = None
        self.q_video = None
        self.video_size = video_size
        self.fps = fps
        self.mxid = mxid

    def __enter__(self):
        # 파이프라인 구성
        pipeline = dai.Pipeline()

        cam = pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)  # 센서 1080p
        cam.setVideoSize(*self.video_size)  # 1920x1080 스트림
        cam.setFps(self.fps)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)

        xout = pipeline.createXLinkOut()
        xout.setStreamName("video")
        cam.video.link(xout.input)

        # 디바이스 선택(옵션: mxid)
        if self.mxid:
            self.device = dai.Device(pipeline, dai.DeviceInfo(self.mxid))
        else:
            self.device = dai.Device(pipeline)

        self.q_video = self.device.getOutputQueue(name="video", maxSize=4, blocking=False)
        return self

    def get(self):
        """프레임 하나 가져오기: BGR numpy, (H,W,3)"""
        in_frame = self.q_video.get()  # blocking=False면 get()이 None일 수 있어, 여기선 기본 블로킹
        frame = in_frame.getCvFrame()
        return frame

    def __exit__(self, exc_type, exc, tb):
        if self.device is not None:
            self.device.close()

# ==============================
# ONNX 래퍼 (네 모델에 맞게 전/후처리 수정)
# ==============================
class OnnxDetector:
    def __init__(self, onnx_path: str, device: str="cpu"):
        if not HAS_ORT:
            raise RuntimeError("onnxruntime 미설치. pip install onnxruntime 혹은 onnxruntime-gpu")
        sess_opts = ort.SessionOptions()
        providers = ["CPUExecutionProvider"] if device=="cpu" else ["CUDAExecutionProvider","CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.out_names  = [o.name for o in self.sess.get_outputs()]
        # TODO: inference_lstm_onnx.py와 동일하게 맞춰
        self.inp_wh = (1280, 720)  # (W,H)

    def preprocess(self, frame_bgr: np.ndarray):
        W, H = self.inp_wh  # 예: (1280, 720)
        img_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW
        x = np.expand_dims(x, 0)  # NCHW
        return x
        # img = cv2.resize(frame_bgr, self.inp_wh)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        # x = np.transpose(img, (2,0,1))[None, ...]  # NCHW
        # return x

    def infer(self, frame_bgr: np.ndarray):
        x = self.preprocess(frame_bgr)
        out = self.sess.run(self.out_names, {self.input_name: x})
        return out, self.inp_wh

# ==============================
# 후처리 (모델 출력 → 이미지 좌표 OBB들)  ★여기만 네 코드에 맞춰 수정
# ==============================
def postprocess_to_img_obbs(model_outputs, model_in_wh,
                            strides=(8,16,32),       # 네 모델에 맞게
                            conf_th=0.8,             # --conf
                            nms_iou=0.2,             # --nms-iou
                            topk=50,                 # --topk
                            score_mode="obj_cls",    # --score-mode (너 세팅대로)
                            clip_cells=None,         # --clip-cells (없으면 None)
                            use_gpu_nms=True):
    """
    반환: [{"cx":..,"cy":..,"L":..,"W":..,"yaw_deg":..,"score":..}, ...] (이미지 좌표, 모델 입력 해상도 기준)
    """
    outs = model_outputs  # runner.forward에서 받은 그대로

    # 1) 디코딩 (inference_lstm_onnx.py의 main()과 동일한 호출)
    dets_list = decode_predictions(
        outs, list(map(float, strides)),
        clip_cells=clip_cells,
        conf_th=conf_th,
        nms_iou=nms_iou,
        topk=topk,
        score_mode=score_mode,
        use_gpu_nms=use_gpu_nms
    )
    dets = dets_list[0]  # batch=1 가정

    # 2) 사소한/깨진 삼각형 필터링 (같은 유틸)
    dets = tiny_filter_on_dets(dets, min_area=20.0, min_edge=3.0)

    # 3) 삼각형 → 이미지 OBB (센터/길이/폭/각) 변환
    #    inference_lstm_onnx.py의 compute_bev_properties()는 삼각형에서
    #    center/length/width/yaw를 구하는 동일 로직이야. (이름만 BEV일 뿐, 이미지 좌표에도 그대로 적용 가능)
    obbs = []
    for d in dets:
        tri = np.asarray(d["tri"], dtype=np.float64)  # shape (3,2), 이미지 좌표(모델 입력 크기 기준)
        # ----- center, length, width, yaw 구하기 -----
        # parallelogram_from_triangle(p0,p1,p2)로 4각형 만든 후 변의 길이 체크하는 로직은
        # compute_bev_properties() 안에 들어있어. 동일 계산을 여기서 직접 구현:
        p0, p1, p2 = tri
        front_center = (p1 + p2) / 2.0
        front_vec = front_center - p0
        yaw = math.degrees(math.atan2(front_vec[1], front_vec[0]))
        # 평행사변형 4점
        poly = np.array([
            p0,
            p1,
            p1 + (p2 - p0),
            p2
        ], dtype=np.float64)
        edges = [np.linalg.norm(poly[(i+1) % 4] - poly[i]) for i in range(4)]
        L = float(max(edges)); W = float(min(edges))
        center = poly.mean(axis=0)
        obbs.append({
            "cx": float(center[0]),
            "cy": float(center[1]),
            "L": L,
            "W": W,
            "yaw_deg": float(yaw),
            "score": float(d.get("score", 1.0))
        })
    return obbs

# ==============================
# IMG → BEV 변환 (중심/앞/뒤 3점 투영)
# ==============================
def imgobb_to_bevobb(det, H_img2bev: np.ndarray, px_to_m=0.02):
    cx, cy, L, W, yaw = det["cx"], det["cy"], det["L"], det["W"], det["yaw_deg"]
    front = np.array([cx + (L/2)*math.cos(math.radians(yaw)),
                      cy + (L/2)*math.sin(math.radians(yaw))], dtype=np.float32)
    back  = np.array([cx - (L/2)*math.cos(math.radians(yaw)),
                      cy - (L/2)*math.sin(math.radians(yaw))], dtype=np.float32)
    ctr_bev, front_bev, back_bev = imgpt_to_bevpt(
        np.stack([np.array([cx,cy],np.float32), front, back], axis=0), H_img2bev
    )

    v = front_bev - ctr_bev
    yaw_bev = math.degrees(math.atan2(v[1], v[0]))
    L_bev = float(np.linalg.norm(front_bev - back_bev))
    # 폭은 간단히 픽셀→미터 스케일로 근사(정확 필요시 코너 4점 투영으로 보정)
    W_bev = float(W * px_to_m)
    return [float(ctr_bev[0]), float(ctr_bev[1]), float(L_bev), float(W_bev), float(yaw_bev)]

# ==============================
# 메인 루프: OAK → 추론 → 그리기 + UDP 송신
# ==============================
def run_oak_edge(cam_name, host, port, H, onnx_path, device="cpu", fps=30, mxid=None):
    # UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[{cam_name}] OAK LIVE → udp://{host}:{port}")
    print("[edge] 화면: bbox만, 송신: bev_labels만")

    # 모델
    det = OnnxDetector(onnx_path, device=device) if onnx_path else None

    frame_id = 0
    with OAKStream(fps=fps, mxid=mxid, video_size=(1920,1080)) as stream:
        try:
            while True:
                frame = stream.get()  # BGR (1080p)
                H_img, W_img = frame.shape[:2]

                # 추론
                if det is not None:
                    model_out, model_wh = det.infer(frame)
                    dets_img = postprocess_to_img_obbs(model_out, model_wh, strides=(8,16,32), conf_th=0.8, nms_iou=0.2, topk=50)
                    # 모델입력 기준 → 원본(1920x1080) 좌표 스케일 복원
                    scale_x = W_img / float(model_wh[0])
                    scale_y = H_img / float(model_wh[1])
                    dets_img_scaled = []
                    for d in dets_img:
                        dets_img_scaled.append({
                            "cx": d["cx"]*scale_x, "cy": d["cy"]*scale_y,
                            "L":  d["L"] *scale_x, "W":  d["W"] *scale_y,   # 단순 비율 보정
                            "yaw_deg": d["yaw_deg"], "score": d["score"]
                        })
                else:
                    # 더미
                    dets_img_scaled = postprocess_to_img_obbs(None, (W_img, H_img))

                # (A) 엣지 시각화
                vis = frame.copy()
                for d in dets_img_scaled:
                    draw_obb(vis, d["cx"], d["cy"], d["L"], d["W"], d["yaw_deg"], (0,255,0), 2)
                cv2.imshow(f"[{cam_name}] detections", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

                # (B) BEV 라벨 변환 + 송신
                bev_labels = [imgobb_to_bevobb(d, H) for d in dets_img_scaled]
                pkt = {"cam": cam_name, "frame_id": frame_id, "ts": time.time(), "labels": bev_labels}
                sock.sendto(json.dumps(pkt).encode("utf-8"), (host, port))
                # print(f"[TX] {cam_name} frame={frame_id} nlabel={len(bev_labels)}")

                frame_id += 1
        except KeyboardInterrupt:
            pass
        finally:
            sock.close()
            cv2.destroyAllWindows()

# ==============================
# CLI
# ==============================
def main():
    ap = argparse.ArgumentParser("EDGE (OAK → ONNX inference → draw → send)")
    ap.add_argument("--cam-name", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--H", type=str, default="1,0,0,0,1,0,0,0,1", help="3x3 homography csv(9)")
    ap.add_argument("--onnx", type=str, required=True, help="onnx model path")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mxid", type=str, help="특정 OAK 장치 MXID 지정(옵션)")
    args = ap.parse_args()

    H = parse_H(args.H)
    run_oak_edge(args.cam_name, args.host, args.port, H, args.onnx, device=args.device, fps=args.fps, mxid=args.mxid)

if __name__ == "__main__":
    main()
