# server_hub.py
import argparse, asyncio, json, time, math, threading
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# --- 너의 기존 병합/시각화/트래커 모듈 활용 ---
# 병합 규칙/카메라 위치 등 내부에서 관리 (필요시 CAMERA_SETUPS 편집)
import importlib

import sys
sys.path.append("/home/whth/3d/inference_merge")  # .py가 들어있는 '폴더'를 추가
import merge_dist_wbf as merge_mod
sys.path.append("/home/whth/3d/sort_v2_3")  # .py가 들어있는 '폴더'를 추가
import tracker as trk_mod
# merge_mod = importlib.import_module("merge_dist_wbf")  # oriented_iou, partial_inclusion_suppression_aabb 등 활용
# SortTracker는 tracker.py에서 가져옴 (bbox를 carla 포맷으로 넘겨주는 래퍼 제공)
# trk_mod = importlib.import_module("tracker")

# ========= 파라미터 =========
DEFAULT_XLIM = (-120.0, -30.0)
DEFAULT_YLIM = (40.0, -80.0)  # y축 뒤집힌 좌표계를 쓰면 아래처럼 그대로 설정

# ========= 수신 버퍼 =========
class HubState:
    def __init__(self, cam_timeout=1.0, fuse_iou=0.15, loop_hz=15):
        self.latest_by_cam = {}   # cam -> (ts, frame_id, np.ndarray N×5 [cx,cy,L,W,yaw])
        self.cam_timeout = cam_timeout
        self.fuse_iou = fuse_iou
        self.loop_dt = 1.0/loop_hz
        # 트래커
        self.tracker = trk_mod.SortTracker(max_age=6, min_hits=3, iou_threshold=0.3)
        # 시각화
        self.fig, (self.ax_int, self.ax_fuse, self.ax_trk) = plt.subplots(1, 3, figsize=(15, 5))
        for ax, title in zip((self.ax_int, self.ax_fuse, self.ax_trk),
                             ("[1] Integrated (per-cam colors)",
                              "[2] Fused (dedup)",
                              "[3] Tracked (IDs)")):
            ax.set_aspect("equal", "box")
            ax.set_xlim(*DEFAULT_XLIM)
            ax.set_ylim(*DEFAULT_YLIM)
            ax.grid(True, ls="--", alpha=0.35)
            ax.set_title(title)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        self.cmap = plt.colormaps.get_cmap("tab20")

    def _prune_timeouts(self):
        now = time.time()
        drop = [cam for cam, (ts, _, _) in self.latest_by_cam.items() if (now - ts) > self.cam_timeout]
        for cam in drop:
            self.latest_by_cam.pop(cam, None)

    def update_from_packet(self, cam, frame_id, ts, labels):
        # labels: list of [cx,cy,L,W,yaw]
        arr = np.array(labels, dtype=float).reshape(-1, 5) if labels else np.zeros((0,5), dtype=float)
        self.latest_by_cam[cam] = (ts, frame_id, arr)

    def _draw_obbs(self, ax, obbs, color, lw=1.8, alpha=0.95):
        for cx, cy, L, W, yaw in obbs:
            # 4점으로 외곽선 그리기
            th = math.radians(yaw); dx, dy = L/2.0, W/2.0
            corners = np.array([[ dx,  dy],[ dx, -dy],[-dx, -dy],[-dx,  dy]])
            c, s = math.cos(th), math.sin(th)
            R = np.array([[c,-s],[s,c]])
            quad = corners @ R.T + np.array([cx, cy])
            poly = np.vstack([quad, quad[0]])
            ax.plot(poly[:,0], poly[:,1], lw=lw, alpha=alpha, color=color)

    def _clear_axes(self):
        for ax in (self.ax_int, self.ax_fuse, self.ax_trk):
            ax.cla()
            ax.set_aspect("equal", "box")
            ax.set_xlim(*DEFAULT_XLIM)
            ax.set_ylim(*DEFAULT_YLIM)
            ax.grid(True, ls="--", alpha=0.35)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        self.ax_int.set_title("[1] Integrated (per-cam colors)")
        self.ax_fuse.set_title("[2] Fused (dedup)")
        self.ax_trk.set_title("[3] Tracked (IDs)")

    def _fuse_multi_cam(self, all_boxes):
        """
        all_boxes: dict cam -> np.ndarray (N,5)
        병합 파이프라인(요약):
          1) concat → 부분포함 억제(partial_inclusion_suppression_aabb)
          2) AABB IoU로 클러스터링 → 대표 라벨 선택 (merge_dist_wbf 규칙)
        """
        if not all_boxes:
            return np.zeros((0,5), dtype=float)
        
        non_empty = [v for v in all_boxes.values() if isinstance(v, np.ndarray) and v.size > 0]
        if len(non_empty) == 0:
            return np.zeros((0,5), dtype=float)

        concat = np.concatenate(non_empty, axis=0)
        concat = merge_mod.partial_inclusion_suppression_aabb(concat, incl_frac_thr=0.55)
        clusters = merge_mod.cluster_by_aabb_iou(concat, iou_cluster_thr=self.fuse_iou)
        # concat = np.concatenate([v for v in all_boxes.values() if v.size > 0], axis=0)
        # 간단 클러스터링: merge_mod.cluster_by_aabb_iou 사용
        clusters = merge_mod.cluster_by_aabb_iou(concat, iou_cluster_thr=self.fuse_iou)
        fused = []
        for idxs in clusters:
            members = concat[idxs]
            # 여러 규칙이 merge_dist_wbf에 있으니, 여기서는 간단 평균 + 각도 원형평균
            if len(members) == 1:
                fused.append(members[0])
            else:
                xs, ys, Ls, Ws, yaws = members[:,0], members[:,1], members[:,2], members[:,3], members[:,4]
                cx = float(xs.mean()); cy = float(ys.mean())
                L  = float(np.median(Ls)); W = float(np.median(Ws))
                # yaw 원형평균
                th = np.deg2rad(yaws)
                yaw = float(np.rad2deg(math.atan2(np.mean(np.sin(th)), np.mean(np.cos(th)))))
                fused.append(np.array([cx, cy, L, W, yaw], dtype=float))
        return np.array(fused, dtype=float) if fused else np.zeros((0,5), dtype=float)

    def tracking_step(self, fused_boxes):
        """
        tracker.SortTracker 는 carla 포맷 [0, cx, cy, L, W, yaw] 의 리스트를 기대.
        update()는 np.array([[track_id, 0, cx, cy, L, W, yaw], ...]) 를 반환(네 구현 기준).  :contentReference[oaicite:5]{index=5}
        """
        dets_carla = []
        for cx, cy, L, W, yaw in fused_boxes:
            dets_carla.append(np.array([0, cx, cy, L, W, yaw], dtype=float))
        if len(dets_carla) == 0:
            dets_arr = np.zeros((0,6), dtype=float)
        else:
            dets_arr = np.stack(dets_carla, axis=0)
        out = self.tracker.update(dets_arr)  # (M, 7) = [track_id, 0, cx, cy, L, W, yaw]
        return out

    def render_once(self):
        self._prune_timeouts()
        # 1) 통합
        cam_list = sorted(self.latest_by_cam.keys())
        self._clear_axes()

        if not cam_list:
            for ax in (self.ax_int, self.ax_fuse, self.ax_trk):
                ax.text(0.5, 0.5, "No packets yet", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, alpha=0.6)
            plt.pause(0.001)
            return

        color_map = {cam: self.cmap(i % 20) for i, cam in enumerate(cam_list)}
        for cam in cam_list:
            _, _, arr = self.latest_by_cam[cam]
            self._draw_obbs(self.ax_int, arr, color_map[cam])            
        self.ax_int.legend(handles=[plt.Line2D([0],[0], color=color_map[c], lw=2, label=c) for c in cam_list],
                           loc="upper right", fontsize=8, ncol=2)

        # 2) 융합
        fused = self._fuse_multi_cam({c: self.latest_by_cam[c][2] for c in cam_list})
        if fused.size > 0:
            self._draw_obbs(self.ax_fuse, fused, color="tab:red")

        # 3) 추적
        trk_out = self.tracking_step(fused)  # (M,7)
        for row in trk_out:
            tid, _, cx, cy, L, W, yaw = row.tolist()
            color = self.cmap(int(tid) % 20)
            self._draw_obbs(self.ax_trk, np.array([[cx, cy, L, W, yaw]]), color)
            self.ax_trk.text(cx, cy, f"{int(tid)}", color=color, fontsize=9,
                             ha="center", va="center",
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        plt.pause(0.001)  # non-blocking update

# ========= UDP 서버 (asyncio) =========
class UDPServer(asyncio.DatagramProtocol):
    def __init__(self, hub: HubState):
        super().__init__()
        self.hub = hub

    def datagram_received(self, data, addr):
        try:
            msg = json.loads(data.decode("utf-8"))
            cam = str(msg.get("cam", f"{addr[0]}:{addr[1]}"))
            frame_id = msg.get("frame_id", 0)
            ts = float(msg.get("ts", time.time()))
            labels = msg.get("labels", [])
            self.hub.update_from_packet(cam, frame_id, ts, labels)
        except Exception as e:
            # 필요하면 로그
            pass

async def run_servers(ports, hub: HubState):
    loop = asyncio.get_running_loop()
    transports = []
    for p in ports:
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: UDPServer(hub),
            local_addr=("0.0.0.0", p)
        )
        transports.append(transport)
        print(f"[UDP] listening udp://0.0.0.0:{p}")
    try:
        while True:
            hub.render_once()
            await asyncio.sleep(hub.loop_dt)
    finally:
        for t in transports:
            t.close()

def parse_bev_limits(csv4):
    # "xmin,xmax,ymin,ymax" 형태
    vals = [float(v) for v in csv4.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("--bev-limits 는 'xmin,xmax,ymin,ymax' 형태여야 함 (예: -120,-30,40,-80)")
    return (vals[0], vals[1]), (vals[2], vals[3])

if __name__ == "__main__":
    ap = argparse.ArgumentParser("BEV fusion+tracking server")
    ap.add_argument("--ports", type=int, nargs="+", required=True, help="수신 UDP 포트 (여러 개)")
    ap.add_argument("--bev-limits", type=str, default="-120,-30,40,-80",
                    help="x_min,x_max,y_min,y_max (따옴표로 감싸서 전달)")
    ap.add_argument("--fuse-iou", type=float, default=0.15)
    ap.add_argument("--cam-timeout", type=float, default=1.0)
    ap.add_argument("--loop-hz", type=float, default=15.0)
    args = ap.parse_args()

    xlim, ylim = parse_bev_limits(args.bev_limits)
    # 전역 축 한 번 세팅
    DEFAULT_XLIM = xlim
    DEFAULT_YLIM = ylim

    hub = HubState(cam_timeout=args.cam_timeout, fuse_iou=args.fuse_iou, loop_hz=args.loop_hz)
    plt.ion()
    asyncio.run(run_servers(args.ports, hub))
