# # 서버
# python server_sync.py --ports 50050 50051 --bev-limits "-120,-30,40,-80" --min-cams 2 --sync-timeout 0.25

# # 엣지 1
# python edge_oak.py --cam-name cam1 --port 50050 --onnx model.onnx --H "1,0,0,0,1,0,0,0,1"

# # 엣지 2
# python edge_oak.py --cam-name cam2 --port 50051 --onnx model.onnx --H "1,0,0,0,1,0,0,0,1"

# server_sync.py
import argparse, asyncio, json, time, math, os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # WSL이면 X서버 필요. 안되면 Qt5Agg로 바꿔도 OK
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# ====== 외부 모듈(있으면 사용) ======
merge_mod = None
try:
    import merge_dist_wbf as merge_mod  # 같은 폴더 또는 PYTHONPATH에 있어야 함
except Exception:
    pass

trk_mod = None
try:
    import tracker as trk_mod  # tracker.SortTracker 필요
except Exception:
    pass

# ====== 유틸 ======
def aabb_from_obb(cx, cy, L, W, yaw_deg):
    th = math.radians(yaw_deg)
    dx, dy = L/2.0, W/2.0
    corners = np.array([[ dx,  dy],[ dx,-dy],[-dx,-dy],[-dx, dy]], dtype=np.float32)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c,-s],[s,c]], dtype=np.float32)
    quad = corners @ R.T + np.array([cx, cy], dtype=np.float32)
    xmin, ymin = quad[:,0].min(), quad[:,1].min()
    xmax, ymax = quad[:,0].max(), quad[:,1].max()
    return xmin, ymin, xmax, ymax

def iou_aabb(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / max(1e-9, area_a + area_b - inter)

def fuse_simple_mean(boxes):
    """
    boxes: (N,5) = [cx,cy,L,W,yaw]
    대표값: cx,cy 평균 / L,W 중앙값 / yaw 원형평균
    """
    if len(boxes) == 1: return boxes[0]
    xs, ys = boxes[:,0], boxes[:,1]
    Ls, Ws = boxes[:,2], boxes[:,3]
    yaws = boxes[:,4]
    cx = float(xs.mean()); cy = float(ys.mean())
    L  = float(np.median(Ls)); W = float(np.median(Ws))
    th = np.deg2rad(yaws)
    yaw = float(np.rad2deg(math.atan2(np.mean(np.sin(th)), np.mean(np.cos(th)))))
    return np.array([cx, cy, L, W, yaw], dtype=float)

# ====== 프레임 동기화 버퍼 ======
class FrameBuffer:
    """
    frame_id -> {
        'by_cam': {cam: np.ndarray(N,5)},
        'first_ts': float,
    }
    """
    def __init__(self, sync_timeout=0.25, min_cams=1):
        self.sync_timeout = sync_timeout
        self.min_cams = min_cams
        self.buf = {}

    def add(self, frame_id, cam, labels):
        now = time.time()
        if frame_id not in self.buf:
            self.buf[frame_id] = {'by_cam': {}, 'first_ts': now}
        arr = np.array(labels, dtype=float).reshape(-1,5) if labels else np.zeros((0,5), dtype=float)
        self.buf[frame_id]['by_cam'][cam] = arr

    def pop_ready_frames(self):
        """
        조건: (1) 카메라 수 >= min_cams 이거나 (2) first_ts로부터 sync_timeout 초 경과
        반환: list[(frame_id, by_cam_dict)]
        """
        now = time.time()
        ready = []
        for fid, rec in list(self.buf.items()):
            cams = rec['by_cam']
            enough = (len(cams) >= self.min_cams)
            timeout = (now - rec['first_ts'] >= self.sync_timeout)
            if enough or timeout:
                ready.append((fid, cams))
                del self.buf[fid]
        ready.sort(key=lambda x: x[0])  # frame_id 순서 처리
        return ready

# ====== 서버 상태 ======
class Server:
    def __init__(self, xlim, ylim, fuse_iou=0.15, cam_timeout=1.5, loop_hz=15.0,
                 sync_timeout=0.25, min_cams=1, headless=False, dump_dir="server_frames"):
        # 렌더 파라미터
        self.xlim = xlim; self.ylim = ylim
        self.fuse_iou = fuse_iou
        self.loop_dt  = 1.0/loop_hz
        self.cam_timeout = cam_timeout
        self.headless = headless
        self.dump_dir = dump_dir

        # 최신 수신 상태(모니터링용)
        self.latest_cam_ts = {}  # cam -> ts

        # 프레임 동기화 버퍼
        self.fbuf = FrameBuffer(sync_timeout=sync_timeout, min_cams=min_cams)

        # 트래커
        if trk_mod and hasattr(trk_mod, "SortTracker"):
            self.tracker = trk_mod.SortTracker(max_age=6, min_hits=3, iou_threshold=0.3)
        else:
            self.tracker = None

        # 그림
        self.fig, (self.ax_int, self.ax_fuse, self.ax_trk) = plt.subplots(1, 3, figsize=(15, 5))
        for ax, title in zip((self.ax_int, self.ax_fuse, self.ax_trk),
                             ("[1] Integrated (per-cam)", "[2] Fused (dedup)", "[3] Tracked (IDs)")):
            ax.set_aspect("equal", "box"); ax.set_xlim(*self.xlim); ax.set_ylim(*self.ylim)
            ax.grid(True, ls="--", alpha=0.35); ax.set_title(title)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        self.cmap = plt.colormaps.get_cmap("tab20")
        plt.ion(); 
        if not self.headless: plt.show(block=False)

        # 최근 처리된 결과(시각화 갱신용)
        self.last_integrated = {}  # cam -> np.ndarray(N,5)
        self.last_fused = np.zeros((0,5), dtype=float)
        self.last_trk   = np.zeros((0,7), dtype=float)  # [tid,0,cx,cy,L,W,yaw]

    # ---------- 수신 ----------
    def on_packet(self, cam, frame_id, ts, labels):
        self.latest_cam_ts[cam] = ts
        self.fbuf.add(frame_id, cam, labels)

    # ---------- 통합 ----------
    @staticmethod
    def _concat_nonempty(by_cam):
        non_empty = [v for v in by_cam.values() if isinstance(v, np.ndarray) and v.size > 0]
        if not non_empty:
            return np.zeros((0,5), dtype=float)
        return np.concatenate(non_empty, axis=0)

    # ---------- 융합(중복 제거) ----------
    def fuse(self, arr):
        if arr.size == 0:
            return arr
        # 1) partial inclusion suppression & 클러스터링 (가능하면 네 모듈 사용)
        if merge_mod is not None:
            try:
                arr2 = merge_mod.partial_inclusion_suppression_aabb(arr, incl_frac_thr=0.55)
                clusters = merge_mod.cluster_by_aabb_iou(arr2, iou_cluster_thr=self.fuse_iou)
                fused = []
                for idxs in clusters:
                    fused.append(fuse_simple_mean(arr2[idxs]))
                return np.stack(fused, axis=0) if fused else np.zeros((0,5), dtype=float)
            except Exception:
                pass

        # 2) Fallback: 간단 IoU 기반 클러스터링 (AABB로 변환 후 single-linkage)
        aabbs = np.array([aabb_from_obb(*row) for row in arr], dtype=float)
        N = len(arr)
        used = np.zeros(N, dtype=bool)
        out = []
        for i in range(N):
            if used[i]: continue
            group = [i]; used[i] = True
            for j in range(i+1, N):
                if used[j]: continue
                if iou_aabb(aabbs[i], aabbs[j]) >= self.fuse_iou:
                    used[j] = True; group.append(j)
            out.append(fuse_simple_mean(arr[group]))
        return np.stack(out, axis=0) if out else np.zeros((0,5), dtype=float)

    # ---------- 추적 ----------
    def track(self, fused):
        if self.tracker is None or fused.size == 0:
            return np.zeros((0,7), dtype=float)
        dets = []
        for cx, cy, L, W, yaw in fused:
            dets.append(np.array([0, cx, cy, L, W, yaw], dtype=float))  # carla 포맷
        dets = np.stack(dets, axis=0) if dets else np.zeros((0,6), dtype=float)
        out = self.tracker.update(dets)  # (M,7) = [tid,0,cx,cy,L,W,yaw]
        return out

    # ---------- 시각화 ----------
    def _clear_axes(self):
        for ax, title in zip((self.ax_int, self.ax_fuse, self.ax_trk),
                             ("[1] Integrated (per-cam)", "[2] Fused (dedup)", "[3] Tracked (IDs)")):
            ax.cla()
            ax.set_aspect("equal", "box"); ax.set_xlim(*self.xlim); ax.set_ylim(*self.ylim)
            ax.grid(True, ls="--", alpha=0.35); ax.set_title(title)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    @staticmethod
    def _draw_obb(ax, cx, cy, L, W, yaw_deg, color, lw=2):
        th = math.radians(yaw_deg); dx, dy = L/2.0, W/2.0
        corners = np.array([[ dx,  dy],[ dx,-dy],[-dx,-dy],[-dx, dy]], dtype=float)
        c, s = math.cos(th), math.sin(th); R = np.array([[c,-s],[s,c]])
        quad = corners @ R.T + np.array([cx, cy], dtype=float)
        poly = np.vstack([quad, quad[0]])
        ax.plot(poly[:,0], poly[:,1], lw=lw, color=color, alpha=0.95)

    def render(self):
        self._clear_axes()

        # 1) Integrated
        color_map = {}
        for i, (cam, arr) in enumerate(sorted(self.last_integrated.items())):
            color_map[cam] = plt.colormaps.get_cmap("tab20")(i % 20)
            for row in arr:
                self._draw_obb(self.ax_int, *row, color=color_map[cam])
        if color_map:
            self.ax_int.legend(handles=[plt.Line2D([0],[0], color=c, lw=2, label=name)
                                        for name, c in color_map.items()],
                               loc="upper right", fontsize=8, ncol=2)

        # 2) Fused
        for row in self.last_fused:
            self._draw_obb(self.ax_fuse, *row, color="tab:red")

        # 3) Tracked
        for row in self.last_trk:
            tid, _, cx, cy, L, W, yaw = row.tolist()
            col = plt.colormaps.get_cmap("tab20")(int(tid) % 20)
            self._draw_obb(self.ax_trk, cx, cy, L, W, yaw, color=col)
            self.ax_trk.text(cx, cy, f"{int(tid)}", color=col, fontsize=9,
                             ha="center", va="center",
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        if self.headless:
            os.makedirs(self.dump_dir, exist_ok=True)
            out = os.path.join(self.dump_dir, f"{int(time.time()*1000)}.png")
            self.fig.savefig(out, dpi=120)
        else:
            plt.pause(0.001)

    # ---------- 메인 루프(주기적으로 ready frame 처리 + 타임아웃 프루닝) ----------
    def step(self):
        # 오래 죽은 카메라 제거(모니터링용)
        now = time.time()
        dead = [c for c,t0 in self.latest_cam_ts.items() if now - t0 > self.cam_timeout]
        for c in dead:
            self.latest_cam_ts.pop(c, None)

        # 프레임 동기화 버퍼에서 처리할 프레임 꺼내기
        ready = self.fbuf.pop_ready_frames()
        for frame_id, by_cam in ready:
            # [통합] 카메라별 최신으로 보관(시각화용)
            self.last_integrated = {cam: by_cam[cam] for cam in by_cam.keys()}

            # [융합] 모든 카메라 박스 concat → dedup
            concat = self._concat_nonempty(by_cam)
            fused = self.fuse(concat)
            self.last_fused = fused

            # [추적]
            self.last_trk = self.track(fused)

        # 시각화 갱신
        self.render()

# ====== UDP 핸들러 ======
class UDPServer(asyncio.DatagramProtocol):
    def __init__(self, server: Server):
        super().__init__()
        self.server = server

    def datagram_received(self, data, addr):
        try:
            msg = json.loads(data.decode("utf-8"))
            cam = str(msg.get("cam", f"{addr[0]}:{addr[1]}"))
            frame_id = msg.get("frame_id", 0)
            ts = float(msg.get("ts", time.time()))
            labels = msg.get("labels", [])
            self.server.on_packet(cam, frame_id, ts, labels)
            # 디버깅 로그 (원하면 주석)
            # print(f"[RX] cam={cam} fid={frame_id} n={len(labels)}")
        except Exception as e:
            print("[RX][ERR]", e)

async def run(ports, srv: Server):
    loop = asyncio.get_running_loop()
    transports = []
    for p in ports:
        t, _ = await loop.create_datagram_endpoint(lambda: UDPServer(srv),
                                                   local_addr=("0.0.0.0", p))
        transports.append(t)
        print(f"[UDP] listening udp://0.0.0.0:{p}")
    try:
        while True:
            srv.step()
            await asyncio.sleep(srv.loop_dt)
    finally:
        for t in transports:
            t.close()

def parse_bev_limits(csv4):
    vals = [float(v) for v in csv4.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("--bev-limits 는 'xmin,xmax,ymin,ymax'")
    return (vals[0], vals[1]), (vals[2], vals[3])

if __name__ == "__main__":
    ap = argparse.ArgumentParser("BEV fusion+tracking server (frame-sync)")
    ap.add_argument("--ports", type=int, nargs="+", required=True)
    ap.add_argument("--bev-limits", type=str, default="-120,-30,40,-80")
    ap.add_argument("--fuse-iou", type=float, default=0.15)
    ap.add_argument("--cam-timeout", type=float, default=1.5)
    ap.add_argument("--loop-hz", type=float, default=15.0)
    ap.add_argument("--sync-timeout", type=float, default=0.25, help="frame 동기화 대기 최대시간(s)")
    ap.add_argument("--min-cams", type=int, default=1, help="하나의 frame을 확정 처리하기 위해 필요한 최소 카메라 수")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--dump-dir", type=str, default="server_frames")
    args = ap.parse_args()

    xlim, ylim = parse_bev_limits(args.bev_limits)
    srv = Server(xlim=xlim, ylim=ylim, fuse_iou=args.fuse_iou,
                 cam_timeout=args.cam_timeout, loop_hz=args.loop_hz,
                 sync_timeout=args.sync_timeout, min_cams=args.min_cams,
                 headless=args.headless, dump_dir=args.dump_dir)
    asyncio.run(run(args.ports, srv))
