from AIRKON.realtime.v1.realtime_edge import IPCameraStreamerUltraLL as Streamer
from AIRKON.realtime.v1.batch_infer import BatchedTemporalRunner
from AIRKON.src.inference_lstm_onnx_pointcloud import (
    decode_predictions,       
    tiny_filter_on_dets,       
    tris_img_to_bev_by_lut, 
    poly_from_tri,
    compute_bev_properties
)
from AIRKON.pointcloud.overlay_obj_on_ply import _flip_y_T
from pathlib import Path
import argparse, signal, time, threading, math
import onnxruntime as ort
import numpy as np
import cv2
import queue
from typing import Optional, Dict
import socket, json
import os
from contextlib import contextmanager
import open3d as o3d

# ---- 3D viewer (legacy O3D) -----------------------------------------------
def _unitize_mesh(mesh):
    # car.glb 같은 메쉬를 중심정렬 + 최대변=1.0, Y-up→Z-up 보정
    mesh = mesh.compute_vertex_normals()
    bb = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bb.get_extent())
    scale = 1.0 / max(1e-9, extent.max())
    mesh.translate(-bb.get_center())
    mesh.scale(scale, center=(0, 0, 0))
    Rx90 = mesh.get_rotation_matrix_from_axis_angle([math.radians(90.0), 0.0, 0.0])
    mesh.rotate(Rx90, center=(0, 0, 0))
    return mesh

def _build_T(length, width, yaw_deg, center_xyz, pitch_deg=0.0, roll_deg=0.0, height_scale=1.0):
    sx = max(1e-4, float(length))
    sy = max(1e-4, float(width))
    sz = max(1e-4, float(width) * float(height_scale))

    S = np.diag([sx, sy, sz, 1.0]).astype(np.float64)
    y, p, r = map(math.radians, [yaw_deg, pitch_deg, roll_deg])
    cz, szn = math.cos(y), math.sin(y)
    cp, sp  = math.cos(p), math.sin(p)
    cr, sr  = math.cos(r), math.sin(r)
    Rz = np.array([[ cz,-szn,0,0],[szn,cz,0,0],[0,0,1,0],[0,0,0,1]])
    Ry = np.array([[ cp,0,sp,0],[0,1,0,0],[-sp,0,cp,0],[0,0,0,1]])
    Rx = np.array([[1,0,0,0],[0,cr,-sr,0],[0,sr,cr,0],[0,0,0,1]])
    R = (Rz @ Ry @ Rx)
    Tt = np.eye(4); Tt[:3,3] = np.asarray(center_xyz, dtype=np.float64)
    return (Tt @ R @ S)

class PerCam3DViewer:
    """
    - 글로벌 PLY + 차량 GLB를 한 번 로드
    - 슬롯 메쉬들을 미리 add
    - update(bev_dets)에서 각 슬롯 transform만 갱신
    """
    def __init__(self, title, global_ply, vehicle_glb, invert_ply_y=True, invert_bev_y=True,
                 size_mode="dynamic", fixed_length=5.0, fixed_width=4.0, height_scale=1.0,
                 estimate_z=False, z_radius=0.8, z_offset=0.0, max_slots=32, window=(1200,800), _O3D_AVAILABLE=False):
        self.enabled = _O3D_AVAILABLE
        if not self.enabled:
            print(f"[3D] Open3D not available → viewer disabled")
            return

        import numpy as np
        self.invert_bev_y = bool(invert_bev_y)
        self.size_mode = size_mode
        self.fixed_length = float(fixed_length)
        self.fixed_width  = float(fixed_width)
        self.height_scale = float(height_scale)
        self.estimate_z   = bool(estimate_z)
        self.z_radius = float(z_radius)
        self.z_offset = float(z_offset)

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=title, width=window[0], height=window[1])
        self.lock = threading.Lock()

        # 글로벌 포인트클라우드
        cloud = o3d.io.read_point_cloud(global_ply)
        if cloud.is_empty():
            raise RuntimeError(f"[3D] Empty PLY: {global_ply}")
        if invert_ply_y:
            cloud = o3d.geometry.PointCloud(cloud); cloud.transform(_flip_y_T())
        self.vis.add_geometry(cloud, reset_bounding_box=True)

        # z 추정용 KDTree
        if self.estimate_z:
            self.kdtree = o3d.geometry.KDTreeFlann(cloud)
            self.xyz_pts = np.asarray(cloud.points)
        else:
            self.kdtree = None; self.xyz_pts = None

        # 차량 유닛 메쉬 & 슬롯 생성
        mesh_ref = o3d.io.read_triangle_mesh(vehicle_glb, enable_post_processing=True)
        if mesh_ref.is_empty():
            raise RuntimeError(f"[3D] Cannot load mesh: {vehicle_glb}")
        mesh_unit = _unitize_mesh(mesh_ref)
        self.base_vertices = np.asarray(mesh_unit.vertices).copy()
        self.base_triangles = np.asarray(mesh_unit.triangles).copy()
        mesh_unit.compute_vertex_normals()
        self.base_normals = np.asarray(mesh_unit.vertex_normals).copy()

        self.FAR_HIDE_T = np.diag([1e-6,1e-6,1e-6,1.0]); self.FAR_HIDE_T[:3,3] = np.array([0,0,-9999.0])
        self.meshes = []
        for _ in range(int(max_slots)):
            m = o3d.geometry.TriangleMesh()
            m.vertices  = o3d.utility.Vector3dVector(self.base_vertices.copy())
            m.triangles = o3d.utility.Vector3iVector(self.base_triangles.copy())
            m.vertex_normals = o3d.utility.Vector3dVector(self.base_normals.copy())
            m.transform(self.FAR_HIDE_T)  # 숨김
            self.vis.add_geometry(m, reset_bounding_box=False)
            self.meshes.append(m)

        # 폴링 스레드
        self._stop = False
        import threading, time
        def _pump():
            while not self._stop:
                with self.lock:
                    self.vis.poll_events()
                    self.vis.update_renderer()
                time.sleep(0.01)
        self._thr = threading.Thread(target=_pump, daemon=True); self._thr.start()

    def close(self):
        if not self.enabled: return
        self._stop = True
        try: self._thr.join(timeout=0.5)
        except: pass
        try:
            with self.lock:
                self.vis.destroy_window()
        except: pass

    def _apply_slot(self, slot_mesh, T, reset=False):
        # 레거시는 transform setter가 없으므로 vertices를 원본으로 되돌리고 transform 적용
        if reset:
            slot_mesh.vertices  = o3d.utility.Vector3dVector(self.base_vertices.copy())
            slot_mesh.triangles = o3d.utility.Vector3iVector(self.base_triangles.copy())
            slot_mesh.vertex_normals = o3d.utility.Vector3dVector(self.base_normals.copy())
        slot_mesh.transform(T)
        self.vis.update_geometry(slot_mesh)

    def update(self, bev_dets):
        """
        bev_dets: [{'center':(x,y), 'length':L, 'width':W, 'yaw':deg, 'cz':..., 'pitch':..., 'roll':...}, ...]
        """
        if (not self.enabled) or bev_dets is None:
            return
        import numpy as np
        with self.lock:
            N = min(len(bev_dets), len(self.meshes))
            # 우선 모두 숨김
            for i in range(len(self.meshes)):
                self._apply_slot(self.meshes[i], self.FAR_HIDE_T, reset=True)

            if N == 0:
                return

            xy = np.array([d["center"] for d in bev_dets[:N]], dtype=np.float64)
            yaw = np.array([d.get("yaw", 0.0) for d in bev_dets[:N]], dtype=np.float64)
            pitch = np.array([d.get("pitch", 0.0) for d in bev_dets[:N]], dtype=np.float64)
            roll  = np.array([d.get("roll", 0.0) for d in bev_dets[:N]], dtype=np.float64)
            cz    = np.array([d.get("cz", 0.0) for d in bev_dets[:N]], dtype=np.float64)
            L     = np.array([d["length"] for d in bev_dets[:N]], dtype=np.float64)
            W     = np.array([d["width"]  for d in bev_dets[:N]], dtype=np.float64)

            # 옵션: BEV 좌표계 Y 플립 (라벨계→월드계 정합) — yaw/pitch/roll도 부호 반전
            if self.invert_bev_y:
                xy[:,1] *= -1.0; yaw *= -1.0; pitch *= -1.0; roll *= -1.0

            # z 결정: (1) 추정, (2) 라벨 cz + 오프셋
            if self.estimate_z and self.kdtree is not None and self.xyz_pts is not None:
                zs = []
                for i in range(N):
                    center = np.array([xy[i,0], xy[i,1], 0.0], dtype=np.float64)
                    [_, idxs, _] = self.kdtree.search_hybrid_vector_3d(center, self.z_radius, 512)
                    if len(idxs) == 0:
                        zs.append(float(cz[i]))
                    else:
                        zs.append(float(np.median(self.xyz_pts[idxs, 2])))
                z_here = np.asarray(zs, dtype=np.float64) + self.z_offset
            else:
                z_here = cz + self.z_offset

            for i in range(N):
                if self.size_mode == "fixed":
                    Li, Wi = self.fixed_length, self.fixed_width
                else:
                    Li, Wi = float(L[i]), float(W[i])

                # 바닥 접지: center z는 “모델 중앙”이므로 (높이/2) 만큼 올려줌
                height = Wi * self.height_scale
                center3 = np.array([xy[i,0], xy[i,1], z_here[i] + height*0.5], dtype=np.float64)
                T = _build_T(Li, Wi, float(yaw[i]), center3,
                             pitch_deg=float(pitch[i]), roll_deg=float(roll[i]),
                             height_scale=self.height_scale)
                self._apply_slot(self.meshes[i], T, reset=True)

# ---
def load_LUT_for_cam(calib_dir: str, cam_id: int) -> Optional[dict]:
    """
    calib_dir 안에서 cam{cam_id} 관련 LUT npz 찾기.
    - 우선순위: cam{cid}_*.npz > {cid}_*.npz > cam{cid}.npz > {cid}.npz
    - np.load 후 dict(...)로 감싸서 반환
    - 필수 키: 'X','Y','Z' (선택: 'ground_valid_mask' 등)
    """
    if not calib_dir:
        return None
    cands = []
    stems = [f"cam{cam_id}", f"{cam_id}"]
    for stem in stems:
        cands += [f for f in Path(calib_dir).glob(f"{stem}_*.npz")]
        cands += [f for f in Path(calib_dir).glob(f"{stem}.npz")]
    for p in cands:
        try:
            lut = dict(np.load(str(p)))
            # 최소 키 체크
            if all(k in lut for k in ("X","Y","Z")):
                print(f"[EdgeInfer] cam{cam_id}: Using LUT {p.name}")
                return lut
        except Exception:
            pass
    print(f"[EdgeInfer] WARN: cam{cam_id} LUT not found in {calib_dir}")
    return None

class InferWorker(threading.Thread):
    """
    - Streamer에서 최신 프레임을 모아 배치 추론
    - UDP 전송(옵션)
    - GUI 큐에 (cam_id, resized_bgr, dets) 전달
    """
    def __init__(self, *, streamer, cam_ids, img_hw, strides, onnx_path,
                 score_mode="obj*cls", conf=0.3, nms_iou=0.2, topk=50,
                 calib_dir=None, bev_scale=1.0, providers=None, viewer3d_map=None,
                 gui_queue=None, udp_sender=None):
        super().__init__(daemon=True)
        self.streamer = streamer
        self.cam_ids = list(cam_ids)
        self.H, self.W = img_hw
        self.strides = list(map(float, strides))
        self.score_mode = score_mode
        self.conf = float(conf)
        self.nms_iou = float(nms_iou)
        self.topk = int(topk)
        self.viewer3d_map = viewer3d_map or {}   # {cam_id: PerCam3DViewer}
        self.gui_queue = gui_queue
        self.udp_sender = udp_sender
        self.stop_evt = threading.Event()
        self.bev_scale = float(bev_scale)
        # 호모그래피대신 LUT
        self.LUT_cache: Dict[int, Optional[dict]] = {cid: load_LUT_for_cam(calib_dir, cid) for cid in self.cam_ids}

        if providers is None:
            providers = ["CUDAExecutionProvider","CPUExecutionProvider"] \
                if (ort.get_device().upper() == "GPU") else ["CPUExecutionProvider"]

        self.runner = BatchedTemporalRunner(
            onnx_path=onnx_path,
            cam_ids=self.cam_ids,
            img_size=(self.H, self.W),
            temporal="lstm", 
            providers=providers,
            state_stride_hint=32,
            default_hidden_ch=256,
        )
        
        for cid in self.cam_ids:
            if self.H_cache[cid] is None:
                    print(f"[EdgeInfer] WARN: cam{cid} H not found in {calib_dir}") 

    def _preprocess(self, frame_bgr):
        """BGR → (H,W) 리사이즈 → RGB CHW float32[0,1] + 리사이즈 BGR(시각화용)"""
        if frame_bgr.shape[:2] != (self.H, self.W):
            bgr = cv2.resize(frame_bgr, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        else:
            bgr = frame_bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2,0,1).astype(np.float32) / 255.0
        return chw, bgr

    def _decode(self, outs):
        dets = decode_predictions(
            outs, self.strides,
            clip_cells=None,
            conf_th=self.conf, nms_iou=self.nms_iou,
            topk=self.topk, score_mode=self.score_mode,
            use_gpu_nms=True
        )[0]
        return tiny_filter_on_dets(dets, min_area=20.0, min_edge=3.0)

    def _make_bev(self, dets, lut_data: Optional[dict]): # LUT 들고와서 bev
        """
        dets: decode된 2D 삼각형 검출들
        lut_data: dict(np.load(npz)) - keys: 'X','Y','Z', (옵션) 유효마스크들
        returns: list of bev dicts (center/length/width/yaw/score/…)
        """
        if lut_data is None or not dets:
            # print("LUT 또는 dets 없음")
            return []

        tris_img = np.asarray([d["tri"] for d in dets], dtype=np.float32)  # (N,3,2) in image pixel
        # LUT 기반 (X,Y,Z) 보간으로 BEV 좌표/높이 얻기
        tris_bev_xy, tris_bev_z, tri_ok = tris_img_to_bev_by_lut(
            tris_img, lut_data, bev_scale=self.bev_scale
        )

        bev = []
        for d, tri_xy, tri_z, ok in zip(dets, tris_bev_xy, tris_bev_z, tri_ok):
            if not ok or (not np.all(np.isfinite(tri_xy))):
                continue

            # 사각형 폴리곤 (시각화/중심/변 길이 계산용)
            poly_bev = poly_from_tri(tri_xy)

            # 3D 속성 계산 (Z 사용 → cz/pitch/roll까지 산출)
            props = compute_bev_properties(tri_xy, tri_z)
            if props is None:
                continue
            center, length, width, yaw, front_edge, cz, pitch_deg, roll_deg = props

            bev.append({
                "score":      float(d["score"]),
                "tri":        tri_xy,           # (3,2) BEV 좌표
                "z3":         tri_z,            # (3,) Z (옵션)
                "poly":       poly_bev,
                "center":     center,
                "length":     float(length),
                "width":      float(width),
                "yaw":        float(yaw),
                "front_edge": front_edge,
                "cz":         float(cz),
                "pitch":      float(pitch_deg),
                "roll":       float(roll_deg),
            })
        return bev


    def run(self):
        wrk = StageTimer(name="WORKER", print_every=5.0)

        while not self.stop_evt.is_set():
            # 1) 최신 프레임 수집
            with wrk.span("grab"):
                ready = True
                imgs_chw = {}
                bgr_for_gui = {}
                for cid in self.cam_ids:
                    fr= self.streamer.get_latest(cid)
                    # ts_capture = time.time() # frame에서 얻어온 시각
                    if fr is None:
                        ready = False
                        break
                    with wrk.span("preproc"):
                        fr, ts_capture = fr
                        chw, bgr = self._preprocess(fr)
                    imgs_chw[cid] = chw
                    bgr_for_gui[cid] = bgr

            if not ready:
                time.sleep(0.002)
                wrk.bump()
                continue

            # 2) 배치 주입
            with wrk.span("enqueue"):
                for cid, chw in imgs_chw.items():
                    self.runner.enqueue_frame(cid, chw)

            # 3) 배치 실행
            with wrk.span("infer"):
                per_cam_outs = self.runner.run_if_ready()
            if per_cam_outs is None:
                time.sleep(0.001)
                wrk.bump()
                continue

            ts = time.time()

            # 4) 디코드/BEV/UDP/GUI 큐
            for cid, outs in per_cam_outs.items():
                with wrk.span("decode"):
                    dets = self._decode(outs)

                with wrk.span("bev"):
                    bev = self._make_bev(dets, self.LUT_cache.get(cid)) 
                
                # 3D 시각화 업데이트
                try:
                    v = self.viewer3d_map.get(cid)
                    if v is not None:
                        v.update(bev)   # center/length/width/yaw/cz/pitch/roll 사용
                except Exception as e:
                    print(f"[3D] update error cam{cid}: {e}")

                if self.udp_sender is not None:
                    with wrk.span("udp"):
                        try:
                            self.udp_sender.send(cam_id=cid, ts=ts, bev_dets=bev)
                            print(bev)
                        except Exception as e:
                            print(f"[UDP] send error cam{cid}: {e}")

                # GUI 큐로 전달(여기서 큐 대기/드롭이 발생할 수 있음)
                with wrk.span("gui.put"):
                    try:
                        self.gui_queue.put_nowait((cid, bgr_for_gui[cid], dets, ts_capture))
                    except queue.Full:
                        try:
                            _ = self.gui_queue.get_nowait()
                            self.gui_queue.put_nowait((cid, bgr_for_gui[cid], dets))
                        except Exception:
                            pass

            wrk.bump()

    def stop(self):
        self.stop_evt.set()

class UDPSender:
    def __init__(self, host: str, port: int, fmt: str = "json", max_bytes: int = 65000):
        """
        host: 수신 서버 IP (로컬 테스트면 127.0.0.1)
        port: 수신 UDP 포트
        fmt : 'json' | 'text'  (보낼 포맷)
        max_bytes: UDP 패킷 최대 크기(분할 전송용 한계)
        """
        self.addr = (host, int(port))
        self.fmt = fmt
        self.max_bytes = max_bytes
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self):
        try: self.sock.close()
        except: pass

    def _pack_json(self, cam_id: int, ts: float, bev_dets):
        """
        bev_dets 항목 예시(필요 최소 필드만): 
        [{"center":[x,y],"length":L,"width":W,"yaw":deg,"score":s}, ...]
        """
        msg = {
            "type": "bev_labels",
            "camera_id": cam_id,
            "timestamp": ts,
            "items": [
                {
                    "center": [float(d["center"][0]), float(d["center"][1])],
                    "length": float(d["length"]),
                    "width": float(d["width"]),
                    "yaw": float(d["yaw"]),
                    "score": float(d["score"]),
                    "cz":     float(d.get("cz", 0.0)), # 나중에 하나로  합쳐서도 3d시각화할거니까 보내는게맞다
                    "pitch":  float(d.get("pitch", 0.0)),
                    "roll":   float(d.get("roll", 0.0)),
                } for d in bev_dets
            ],
        }
        return json.dumps(msg, ensure_ascii=False).encode("utf-8")

    def send(self, cam_id: int, ts: float, bev_dets=None):
        """
        fmt='json'이면 파싱된 bev_dets를 JSON으로,
        fmt='text'면 bev_labels 텍스트 파일 내용을 그대로 보냄.
        큰 페이로드는 조각내서 순차 전송(간단한 분할 헤더 포함).
        """
        if self.fmt == "json":
            payload = self._pack_json(cam_id, ts, bev_dets or [])
        # else:
            # payload = self._pack_text(cam_id, ts, fname, bev_txt_path)

        # 필요 시 분할 전송
        if len(payload) <= self.max_bytes:
            self.sock.sendto(payload, self.addr)
            return

        chunk_id = os.urandom(4).hex()
        total = (len(payload) + self.max_bytes - 1) // self.max_bytes
        for idx in range(total):
            part = payload[idx*self.max_bytes:(idx+1)*self.max_bytes]
            # 1줄짜리 헤더: chunk_id,total,idx\n + part
            prefix = f"CHUNK {chunk_id} {total} {idx}\n".encode("utf-8")
            self.sock.sendto(prefix + part, self.addr)

class FPSTicker:
    """
    GUI 루프의 FPS를 일정하게 유지하고, 'q' 입력으로 종료할 수 있게 도와주는 유틸.
    """
    def __init__(self, target_fps=30, print_every=5.0, name="GUI"):
        self.target_fps = max(1, int(target_fps))
        self.spf = 1.0 / self.target_fps  # seconds per frame
        self.next_deadline = time.time()
        self.running = True
        
        # FPS 측정용
        self._print_every = float(print_every)
        self._name = str(name)
        self._cnt = 0
        self._t0 = time.time()
        
    def set_fps(self, target_fps: int):
        self.target_fps = max(1, int(target_fps))
        self.spf = 1.0 / self.target_fps

    def tick(self):
        """한 프레임마다 호출 — FPS 안정화 + 'q' 입력 감지"""
        if not self.running:
            return False

        now = time.time()

        # FPS 유지 (너무 빠를 때 살짝 sleep)
        if now < self.next_deadline:
            time.sleep(self.next_deadline - now)
        self.next_deadline = time.time() + self.spf

        # GUI 이벤트 처리 + 종료 감지
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[FPSTicker] 'q' pressed → stopping.")
            self.running = False
            return False
        
        # 출력 fps 로그용,, 주석 가능
        self._cnt += 1
        if self._print_every > 0:
            t = time.time()
            if (t - self._t0) >= self._print_every:
                fps_out = self._cnt / (t - self._t0)
                print(f"[FPSTicker-{self._name}] render fps ≈ {fps_out:.2f} (target={self.target_fps})")
                self._cnt = 0
                self._t0 = t

        return True

class StageTimer:
    """스테이지별 누적 시간/횟수 집계 + 주기적 출력"""
    def __init__(self, name="prof", print_every=5.0):
        self.name = name
        self.print_every = float(print_every)
        self.t0 = time.perf_counter()
        self.last_print = self.t0
        self.sum = {}   # stage -> seconds accumulated
        self.cnt = {}   # stage -> count
        self.tmp = {}   # stage -> start_time

    @contextmanager
    def span(self, stage: str):
        t = time.perf_counter()
        self.tmp[stage] = t
        try:
            yield
        finally:
            dt = time.perf_counter() - t
            self.sum[stage] = self.sum.get(stage, 0.0) + dt
            self.cnt[stage] = self.cnt.get(stage, 0) + 1

    def bump(self):
        """주기적으로 평균 ms 출력"""
        now = time.perf_counter()
        if (now - self.last_print) >= self.print_every:
            parts = []
            for k in sorted(self.sum.keys()):
                s = self.sum[k]
                n = max(1, self.cnt.get(k, 1))
                parts.append(f"{k}={1000.0*s/n:.2f}ms")
                # 리셋(롤링)
                self.sum[k] = 0.0
                self.cnt[k] = 0
            if parts:
                print(f"[{self.name}] " + " | ".join(parts))
            self.last_print = now

def main():
    ap = argparse.ArgumentParser("Realtime Edge Infer")
    ap.add_argument("--weights", required=True, type=str)
    ap.add_argument("--img-size", default="864,1536", type=str)
    ap.add_argument("--strides", default="8,16,32", type=str)
    ap.add_argument("--conf", default=0.30, type=float)
    ap.add_argument("--nms-iou", default=0.20, type=float)
    ap.add_argument("--topk", default=50, type=int)
    ap.add_argument("--score-mode", default="obj*cls", choices=["obj","cls","obj*cls"])
    ap.add_argument("--bev-scale", default=1.0, type=float)
    ap.add_argument("--lut-path", default="./calib", type=str)
    ap.add_argument("--transport", default="tcp", choices=["tcp","udp"])
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--udp-enable", action="store_true")
    ap.add_argument("--udp-host", default="127.0.0.1")
    ap.add_argument("--udp-port", type=int, default=50050)
    ap.add_argument("--udp-format", choices=["json","text"], default="json")
    ap.add_argument("--visual-size", default="216,384", type=str)
    ap.add_argument("--target-fps", default=30, type=int)
    # 3d
    ap.add_argument("--global-ply", type=str, default="pointcloud/global_fused_small.ply")
    ap.add_argument("--vehicle-glb", type=str, default="pointcloud/car.glb")
    ap.add_argument("--viewer3d", action="store_true")           # ← 3D 창 켜기
    ap.add_argument("--viewer3d-fixed-size", action="store_true")# ← 객체 스케일 고정 모드
    ap.add_argument("--viewer3d-height-scale", type=float, default=1.0)
    ap.add_argument("--viewer3d-estimate-z", action="store_true")
    ap.add_argument("--viewer3d-z-radius", type=float, default=0.8)
    ap.add_argument("--viewer3d-z-offset", type=float, default=0.0)
    ap.add_argument("--viewer3d-invert-bev-y", action="store_true")
    ap.add_argument("--viewer3d-no-invert-bev-y", dest="viewer3d_invert_bev_y", action="store_false")
    ap.set_defaults(viewer3d_invert_bev_y=True)
    args = ap.parse_args()
    
    # GUI 여부
    def gui_available() -> bool:
        try:
            cv2.namedWindow("TEST"); cv2.destroyWindow("TEST")
            print("GUI 사용 가능")
            return True
        except Exception:
            print("GUI 사용 불가")
            return False
    USE_GUI = gui_available()

    try:
        
        _O3D_AVAILABLE = True
    except Exception:
        _O3D_AVAILABLE = False

    
    H, W = map(int, args.img_size.split(","))
    strides = tuple(float(s) for s in args.strides.split(","))
    vis_H, vis_W = map(int, args.visual_size.split(","))
    Target_fps = args.target_fps
    
    # 카메라에서 프레임 받아오기 
    camera_configs = [
        {
            'ip': '192.168.0.3',
            'port': 554,
            'username': 'admin',
            'password': 'zjsxmfhf',
            'camera_id': 1,
            'width': W,
            'height': H,
            'force_tcp': (args.transport == "tcp"),   # UDP 우선(저지연)
        },
        {
            'ip': '192.168.0.2',
            'port': 554,
            'username': 'admin',
            'password': 'zjsxmfhf',
            'camera_id': 2,
            'width': W,
            'height': H,
            'force_tcp': (args.transport == "tcp"),    # 이 카메라만 TCP 강제(손실/초록깨짐 방지)
        },
    ]
    
    cam_ids  = sorted([cfg["camera_id"] for cfg in camera_configs])
    shutdown_evt = threading.Event()
    
    streamer = Streamer(
        camera_configs,
        show_windows=False, # 미리보기
        target_fps=Target_fps,
        snapshot_dir=None,
        snapshot_interval_sec=None,
        catchup_seconds=0.5,
        overlay_ts=False,
        laytency_check=False
    )
    
    providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
    if args.no_cuda or ort.get_device().upper() != "GPU":
        providers = ["CPUExecutionProvider"]
    
    udp_sender = None
    if args.udp_enable:
        udp_sender = UDPSender(args.udp_host, args.udp_port, fmt=args.udp_format)

    # 3D 뷰어 준비(카메라별 한 개씩)
    viewer3d_map = {}
    if args.viewer3d:
        for cfg in camera_configs:
            cid = cfg["camera_id"]
            try:
                viewer3d_map[cid] = PerCam3DViewer(
                    title=f"cam{cid} • 3D",
                    global_ply=args.global_ply,
                    vehicle_glb=args.vehicle_glb,
                    invert_ply_y=True,
                    invert_bev_y=args.viewer3d_invert_bev_y,
                    size_mode=("fixed" if args.viewer3d_fixed_size else "dynamic"),
                    fixed_length=5.0, fixed_width=4.0,
                    height_scale=args.viewer3d_height_scale,
                    estimate_z=args.viewer3d_estimate_z,
                    z_radius=args.viewer3d_z_radius,
                    z_offset=args.viewer3d_z_offset,
                    max_slots=64,
                    window=(1000, 700),
                    _O3D_AVAILABLE = _O3D_AVAILABLE
                )
            except Exception as e:
                print(f"[3D] cam{cid} viewer init failed: {e}")

    # 시작
    streamer.start()
    # -------- GUI 큐 & 워커 --------
    gui_queue = queue.Queue(maxsize=128)
    worker = InferWorker(
        streamer=streamer, cam_ids=cam_ids, img_hw=(H, W), strides=strides,
        onnx_path=args.weights, score_mode=args.score_mode, conf=args.conf, nms_iou=args.nms_iou, topk=args.topk,
        calib_dir=args.lut_path, bev_scale=args.bev_scale, providers=providers, viewer3d_map=viewer3d_map, 
        gui_queue=gui_queue, udp_sender=udp_sender  
    )
    worker.start()
    # -------- 메인 스레드: 오직 GUI --------
    # 창 준비
    if USE_GUI:
        for cid in cam_ids:
            cv2.namedWindow(f"cam{cid}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"cam{cid}", vis_W, vis_H)

    running = True
    def _sigint(sig, frm):
        # ESC나 Ctrl+C 시 모두 같은 경로로 종료
        try:
            ticker.running = False
        except Exception:
            pass
        shutdown_evt.set()
        try:
            if USE_GUI:
                cv2.waitKey(1)
        except Exception:
            pass

    signal.signal(signal.SIGINT, _sigint)
    ticker = FPSTicker(target_fps=Target_fps)
    gui_timer = StageTimer(name="GUI", print_every=5.0)

    try:
        while ticker.running:
            with gui_timer.span("gui.frame"):
                # GUI 큐에서 결과 꺼내 그리기
                with gui_timer.span("gui.get"):
                    try:
                        cid, bgr, dets, ts_capture = gui_queue.get(timeout=0.005) 
                    except ValueError:
                        cid, bgr, dets = gui_queue.get(timeout=0.005) 
                    except queue.Empty:
                        if USE_GUI: cv2.waitKey(1)
                        if not ticker.tick():
                            break
                        # bump는 루프마다 가볍게 호출해도 부담 거의 없음
                        gui_timer.bump()
                        continue

                if not USE_GUI:
                    if not ticker.tick():
                        break
                    gui_timer.bump()
                    continue

                with gui_timer.span("gui.draw"):
                    vis = bgr.copy()
                    for d in dets:
                        tri = np.asarray(d["tri"], dtype=np.int32)
                        poly4 = poly_from_tri(tri).astype(np.int32)
                        cv2.polylines(vis, [poly4], isClosed=True, color=(0,255,0), thickness=2)
                        s = f"{d.get('score', 0.0):.2f}"
                        p0 = (int(tri[0,0]), int(tri[0,1]))
                        cv2.putText(vis, s, p0, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

                with gui_timer.span("gui.show"):
                    cv2.imshow(f"cam{cid}", vis)
                    e2e_ms = (time.time() - ts_capture) * 1000.0
                    print(f"++++++++++++++캡처시점-렌더시점: {e2e_ms}")
                    if not ticker.tick():
                        break

            gui_timer.bump()
    finally:
        worker.stop()
        worker.join(timeout=1)
        streamer.stop()
        if udp_sender: udp_sender.close()
        if USE_GUI:
            try: cv2.destroyAllWindows()
            except: pass
        print("[Main] stopped.")


if __name__ == "__main__":
    main()