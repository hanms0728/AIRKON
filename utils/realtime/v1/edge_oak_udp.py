# edge_oak_udp.py
import os, time, json, socket, argparse, math
import numpy as np

# ====== (A) OAK 없이 디버그용: txt 라벨 읽기 ======
def read_pred_txt(path):
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 6: 
                continue
            _, cx, cy, L, W, yaw = p
            labels.append([float(cx), float(cy), float(L), float(W), float(yaw)])
    return labels

# ====== (B) OAK 있을 때: 이미지 좌표 → H(3x3) → BEV ======
def apply_H_point(xx, yy, H):
    denom = H[2,0]*xx + H[2,1]*yy + H[2,2]
    if abs(denom) < 1e-9: 
        return np.nan, np.nan
    X = (H[0,0]*xx + H[0,1]*yy + H[0,2]) / denom
    Y = (H[1,0]*xx + H[1,1]*yy + H[1,2]) / denom
    return float(X), float(Y)

def tri_to_bev_obb(tri, H):
    # tri: (3,2) image coords (p0 rear-center, p1,p2 front corners 같은 규칙)
    p0 = tri[0]; p1 = tri[1]; p2 = tri[2]
    c0 = np.array(apply_H_point(p0[0], p0[1], H))
    c1 = np.array(apply_H_point(p1[0], p1[1], H))
    c2 = np.array(apply_H_point(p2[0], p2[1], H))
    if not np.all(np.isfinite([*c0, *c1, *c2])): 
        return None
    front_center = (c1 + c2) / 2.0
    center = (c0 + c1 + c2 + front_center) / 4.0
    yaw = math.degrees(math.atan2(front_center[1] - c0[1], front_center[0] - c0[0]))
    # 길이/너비: 간단히 평행사변형 변의 길이로 근사
    edge1 = np.linalg.norm(c1 - c2)          # front edge
    edge2 = np.linalg.norm(front_center - c0) # depth
    L, W = (max(edge1, edge2), min(edge1, edge2))
    return [center[0], center[1], L, W, yaw]

def run_live(cam_name, host, port, H):
    """
    기존 inference_live_oak.py의 디텍션 출력을 가져와서 tri(3,2)들을 얻어온 뒤
    위의 tri_to_bev_obb로 BEV 변환 → UDP 송신.
    이 부분은 네 환경의 탐지 호출부와 연결해서 det_tris를 채워주면 됨.
    """
    # TODO: 여기를 네 모델 추론(inference_live_oak.py)에서 가져온 triangles로 교체
    # det_tris = [{'tri': np.ndarray shape (3,2), 'score': float}, ...]
    # import로 직접 붙이고 싶으면 모듈 import 후 decode 결과를 사용.
    import random
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[{cam_name}] live mode → udp://{host}:{port}")
    frame_id = 0
    try:
        while True:
            # --- 예시 더미: 원형 이동하는 객체 2개 ---
            det_tris = []
            t = time.time()
            for k in range(2):
                cx = 960 + 120*math.cos(t + k*1.1)
                cy = 540 +  60*math.sin(t + k*0.9)
                tri = np.array([[cx, cy],
                                [cx+30, cy-20],
                                [cx+30, cy+20]], dtype=np.float32)
                det_tris.append({"tri": tri, "score": 0.9})
            # -------------------------------

            bev = []
            for d in det_tris:
                obb = tri_to_bev_obb(d["tri"], H)
                if obb is not None:
                    bev.append(obb)
            pkt = {"cam": cam_name, "frame_id": frame_id, "ts": time.time(), "labels": bev}
            sock.sendto(json.dumps(pkt).encode("utf-8"), (host, port))
            frame_id += 1
            time.sleep(1/15.0)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

def run_debug(cam_name, host, port, pred_dir):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[{cam_name}] debug mode → udp://{host}:{port}, dir={pred_dir}")
    # cam*_frame_XXXX.txt를 시간순으로 계속 돌려 송신
    import glob, re
    paths = sorted(glob.glob(os.path.join(pred_dir, f"{cam_name}_frame_*.txt")))
    fid_pat = re.compile(r"_frame_(.+)\.txt$")
    try:
        while True:
            for p in paths:
                m = fid_pat.search(p)
                frame_key = m.group(1) if m else "0"
                labels = read_pred_txt(p)
                pkt = {"cam": cam_name, "frame_id": frame_key, "ts": time.time(), "labels": labels}
                sock.sendto(json.dumps(pkt).encode("utf-8"), (host, port))
                time.sleep(1/15.0)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

def parse_H(csv9):
    vals = [float(v) for v in csv9.split(",")]
    if len(vals) != 9: 
        raise ValueError("--H 은 9개 실수(콤마구분)여야 함")
    return np.array(vals, dtype=float).reshape(3,3)

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Edge(OAK) → UDP sender")
    ap.add_argument("--cam-name", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--live", action="store_true", help="OAK 실시간 추론 모드")
    ap.add_argument("--H", type=str, default="1,0,0,0,1,0,0,0,1", help="3x3 homography (csv 9개)")
    ap.add_argument("--debug-pred-dir", type=str, help="cam*_frame_*.txt 폴더(디버그)")
    args = ap.parse_args()

    H = parse_H(args.H)

    if args.live:
        run_live(args.cam_name, args.host, args.port, H)
    else:
        if not args.debug_pred_dir:
            raise SystemExit("디버그 모드면 --debug-pred-dir 꼭 지정!")
        run_debug(args.cam_name, args.host, args.port, args.debug_pred_dir)
