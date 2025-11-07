from realtime.v1.realtime_edge import IPCameraStreamerUltraLL as Streamer
from realtime.v1.batch_infer import BatchedTemporalRunner
from src.inference_lstm_onnx_pointcloud import (
    decode_predictions,       
    tiny_filter_on_dets,       
    tris_img_to_bev_by_lut, 
    poly_from_tri,
    compute_bev_properties
)
from pointcloud.overlay_obj_on_ply import _flip_y_T
from pathlib import Path
from dataclasses import dataclass
import argparse, signal, time, threading, math, io
import onnxruntime as ort
import numpy as np
import cv2
import queue
from typing import Optional, Dict, List, Tuple
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

# ---- Camera assets / helpers ----------------------------------------------
@dataclass
class CameraAssets:
    camera_id: int
    name: str
    lut: Optional[dict]
    lut_path: Optional[str]
    undistort_map: Optional[Tuple[np.ndarray, np.ndarray]]
    visible_cloud: Optional[np.ndarray]
    raw_config: Dict


def _lut_mask(lut: Optional[dict]) -> Optional[np.ndarray]:
    if lut is None:
        return None
    for key in ("ground_valid_mask", "valid_mask", "floor_mask"):
        if key in lut:
            mask = np.asarray(lut[key]).astype(bool)
            if mask.shape == lut["X"].shape:
                return mask
    x = np.asarray(lut["X"])
    y = np.asarray(lut["Y"])
    z = np.asarray(lut["Z"])
    return np.isfinite(x) & np.isfinite(y) & np.isfinite(z)


def _build_visible_cloud(lut: Optional[dict]) -> Optional[np.ndarray]:
    if lut is None:
        return None
    mask = _lut_mask(lut)
    if mask is None:
        return None
    X = np.asarray(lut["X"], dtype=np.float32)
    Y = np.asarray(lut["Y"], dtype=np.float32)
    Z = np.asarray(lut["Z"], dtype=np.float32)
    xyz = np.stack([X[mask], Y[mask], Z[mask]], axis=1)
    if xyz.size == 0:
        return None
    return xyz.astype(np.float32)


def draw_detections(image: Optional[np.ndarray], dets, text_scale: float = 0.6):
    if image is None:
        return None
    vis = image.copy()
    for d in dets or []:
        tri = np.asarray(d.get("tri"), dtype=np.int32)
        if tri.shape != (3, 2):
            continue
        poly4 = poly_from_tri(tri).astype(np.int32)
        cv2.polylines(vis, [poly4], isClosed=True, color=(0, 255, 0), thickness=2)
        s = f"{d.get('score', 0.0):.2f}"
        p0 = (int(tri[0, 0]), int(tri[0, 1]))
        cv2.putText(vis, s, p0, cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def _lut_candidates(calib_dir: Optional[str], cam_id: int) -> List[Path]:
    if not calib_dir:
        return []
    root = Path(calib_dir)
    if not root.exists():
        return []
    stems = [f"cam{cam_id}", f"{cam_id}"]
    cands: List[Path] = []
    for stem in stems:
        cands += sorted(root.glob(f"{stem}_*.npz"))
        cands += sorted(root.glob(f"{stem}.npz"))
    return cands


def load_lut(explicit_path: Optional[str], calib_dir: Optional[str], cam_id: int) -> Tuple[Optional[dict], Optional[str]]:
    path = None
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            path = p
    if path is None:
        cands = _lut_candidates(calib_dir, cam_id)
        path = cands[0] if cands else None
    if path is None:
        print(f"[EdgeInfer] WARN: cam{cam_id} LUT not found")
        return None, None
    try:
        lut = dict(np.load(str(path), allow_pickle=False))
        if not all(k in lut for k in ("X", "Y", "Z")):
            raise ValueError("Missing XYZ keys")
        print(f"[EdgeInfer] cam{cam_id}: Using LUT {path}")
        return lut, str(path)
    except Exception as exc:
        print(f"[EdgeInfer] WARN: cam{cam_id} failed to load LUT {path}: {exc}")
        return None, str(path)


def _pick_matrix(data: dict, keys: List[str]) -> Optional[np.ndarray]:
    for k in keys:
        if k in data:
            arr = np.asarray(data[k])
            if arr.size:
                return arr
    return None


def load_undistort_map(desc, frame_hw: Tuple[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not desc:
        return None
    data = None
    if isinstance(desc, (str, Path)):
        path = Path(desc)
        if not path.exists():
            print(f"[EdgeInfer] WARN: undistort file missing → {path}")
            return None
        try:
            data = dict(np.load(str(path), allow_pickle=False))
        except Exception as exc:
            print(f"[EdgeInfer] WARN: failed to load undistort npz {path}: {exc}")
            return None
    elif isinstance(desc, dict):
        data = desc
    else:
        return None

    K = _pick_matrix(data, ["K", "camera_matrix", "intrinsic", "intrinsics"])
    dist = _pick_matrix(data, ["dist", "distCoeffs", "dist_coeffs", "distortion", "D"])
    if K is None or dist is None:
        print("[EdgeInfer] WARN: undistort config missing K/dist")
        return None
    K = K.astype(np.float32).reshape(3, 3)
    dist = dist.astype(np.float32).ravel()
    newK = _pick_matrix(data, ["new_K", "K_new", "rectified_K"])
    if newK is not None:
        newK = newK.astype(np.float32).reshape(3, 3)
    else:
        newK = K

    height, width = frame_hw
    if "size" in data:
        size = data["size"]
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            width = int(size[0])
            height = int(size[1])
        elif isinstance(size, dict):
            width = int(size.get("width", width))
            height = int(size.get("height", height))
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, newK, (int(width), int(height)), cv2.CV_32FC1
    )
    return map1, map2


def load_camera_config_file(path: str, default_hw: Tuple[int, int]) -> List[Dict]:
    cfg_path = Path(path)
    raw = json.loads(cfg_path.read_text())
    entries = raw.get("cameras") if isinstance(raw, dict) and "cameras" in raw else raw
    if not isinstance(entries, list):
        raise ValueError("camera_config must be a list or contain 'cameras'")
    result = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        cid = item.get("camera_id", item.get("id"))
        if cid is None:
            raise ValueError("camera entry missing camera_id")
        cid = int(cid)
        rtsp = item.get("rtsp", {})
        ip = item.get("ip", rtsp.get("ip"))
        if not ip:
            raise ValueError(f"camera {cid} missing IP")
        port = int(item.get("port", rtsp.get("port", 554)))
        username = item.get("username", rtsp.get("username", "admin"))
        password = item.get("password", rtsp.get("password", ""))
        w = int(item.get("width", default_hw[1]))
        h = int(item.get("height", default_hw[0]))
        lut_rel = item.get("lut")
        undist_rel = item.get("undistort") or item.get("undistort_npz")
        cfg = {
            "ip": ip,
            "port": port,
            "username": username,
            "password": password,
            "camera_id": cid,
            "width": w,
            "height": h,
            "force_tcp": bool(item.get("force_tcp", False) or (rtsp.get("transport") == "tcp")),
            "lut_path": str((cfg_path.parent / lut_rel).resolve()) if lut_rel else None,
            "undistort_path": str((cfg_path.parent / undist_rel).resolve()) if undist_rel else None,
            "name": item.get("name") or item.get("label") or f"cam{cid}",
            "group": item.get("group"),
        }
        result.append(cfg)
    return result


def build_camera_assets(camera_cfgs: List[Dict], lut_root: Optional[str]) -> Dict[int, CameraAssets]:
    assets: Dict[int, CameraAssets] = {}
    for cfg in camera_cfgs:
        cid = int(cfg["camera_id"])
        lut, lut_path = load_lut(cfg.get("lut_path"), lut_root, cid)
        undist = load_undistort_map(cfg.get("undistort_path"), (cfg["height"], cfg["width"]))
        assets[cid] = CameraAssets(
            camera_id=cid,
            name=cfg.get("name", f"cam{cid}"),
            lut=lut,
            lut_path=lut_path,
            undistort_map=undist,
            visible_cloud=_build_visible_cloud(lut),
            raw_config=cfg,
        )
    return assets

# ---
class InferWorker(threading.Thread):
    """
    - Streamer에서 최신 프레임을 모아 배치 추론
    - UDP 전송(옵션)
    - GUI 큐에 (cam_id, resized_bgr, dets) 전달
    """
    def __init__(self, *, streamer, camera_assets: Dict[int, CameraAssets], img_hw, strides, onnx_path,
                 score_mode="obj*cls", conf=0.3, nms_iou=0.2, topk=50,
                 bev_scale=1.0, providers=None, viewer3d_map=None,
                 gui_queue=None, udp_sender=None, web_publisher=None):
        super().__init__(daemon=True)
        self.streamer = streamer
        self.camera_assets = camera_assets
        self.cam_ids = sorted(camera_assets.keys())
        self.H, self.W = img_hw
        self.strides = list(map(float, strides))
        self.score_mode = score_mode
        self.conf = float(conf)
        self.nms_iou = float(nms_iou)
        self.topk = int(topk)
        self.viewer3d_map = viewer3d_map or {}   # {cam_id: PerCam3DViewer}
        self.gui_queue = gui_queue
        self.udp_sender = udp_sender
        self.web_publisher = web_publisher
        self.stop_evt = threading.Event()
        self.bev_scale = float(bev_scale)
        self.LUT_cache: Dict[int, Optional[dict]] = {cid: camera_assets[cid].lut for cid in self.cam_ids}

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
            if self.LUT_cache[cid] is None:
                print(f"[EdgeInfer] WARN: cam{cid} has no LUT – BEV/3D disabled")
        self.undist_timer = StageTimer(name="UND", print_every=5.0)

    def _preprocess(self, cam_id: int, frame_bgr):
        """BGR → (H,W) 리사이즈 → RGB CHW float32[0,1] + 리사이즈 BGR(시각화용)"""
        assets = self.camera_assets.get(cam_id)
        if assets and assets.undistort_map is not None:
            map1, map2 = assets.undistort_map
            with self.undist_timer.span(f"cam{cam_id}"): # 아 왜곡 지연 로그가 안 찍힘 
                frame_bgr = cv2.remap(frame_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)
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
                        chw, bgr = self._preprocess(cid, fr)
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
                overlay_frame = draw_detections(bgr_for_gui[cid], dets)
                
                # 3D 시각화 업데이트
                try:
                    v = self.viewer3d_map.get(cid)
                    if v is not None:
                        v.update(bev)   # center/length/width/yaw/cz/pitch/roll 사용
                except Exception as e:
                    print(f"[3D] update error cam{cid}: {e}")

                if self.web_publisher is not None:
                    try:
                        self.web_publisher.update(
                            cam_id=cid,
                            ts=ts,
                            capture_ts=ts_capture,
                            bev_dets=bev,
                            overlay_bgr=overlay_frame,
                        )
                    except Exception as e:
                        print(f"[WEB] publish error cam{cid}: {e}")

                if self.udp_sender is not None:
                    with wrk.span("udp"):
                        try:
                            self.udp_sender.send(cam_id=cid, ts=ts, bev_dets=bev, capture_ts=ts_capture)
                        except Exception as e:
                            print(f"[UDP] send error cam{cid}: {e}")

                # GUI 큐로 전달(여기서 큐 대기/드롭이 발생할 수 있음)
                if self.gui_queue is not None:
                    with wrk.span("gui.put"):
                        try:
                            self.gui_queue.put_nowait((cid, overlay_frame, dets, ts_capture))
                        except queue.Full:
                            try:
                                _ = self.gui_queue.get_nowait()
                                self.gui_queue.put_nowait((cid, overlay_frame, dets, ts_capture))
                            except Exception:
                                pass

            wrk.bump()
            if self.undist_timer:
                self.undist_timer.bump()

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

    def _pack_json(self, cam_id: int, ts: float, bev_dets, capture_ts: Optional[float]):
        """
        bev_dets 항목 예시(필요 최소 필드만): 
        [{"center":[x,y],"length":L,"width":W,"yaw":deg,"score":s}, ...]
        """
        msg = {
            "type": "bev_labels",
            "camera_id": cam_id,
            "timestamp": ts,
            "capture_ts": capture_ts,
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

    def send(self, cam_id: int, ts: float, bev_dets=None, capture_ts: Optional[float] = None):
        """
        fmt='json'이면 파싱된 bev_dets를 JSON으로,
        fmt='text'면 bev_labels 텍스트 파일 내용을 그대로 보냄.
        큰 페이로드는 조각내서 순차 전송(간단한 분할 헤더 포함).
        """
        if self.fmt == "json":
            payload = self._pack_json(cam_id, ts, bev_dets or [], capture_ts)
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

class EdgeWebBridge:
    def __init__(self, *, host: str, port: int, global_ply: str, vehicle_glb: str,
                 camera_assets: Dict[int, CameraAssets], site_name: str = "edge",
                 jpeg_quality: int = 85, static_root: Optional[str] = None):
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import Response, FileResponse
            from fastapi.staticfiles import StaticFiles
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("FastAPI + uvicorn required for --web-enable") from exc

        self.FastAPI = FastAPI
        self.HTTPException = HTTPException
        self.Response = Response
        self.FileResponse = FileResponse
        self.uvicorn = uvicorn
        self.host = host
        self.port = int(port)
        self.site_name = site_name
        self.jpeg_quality = int(jpeg_quality)
        self.global_ply = str(Path(global_ply).resolve())
        self.vehicle_glb = str(Path(vehicle_glb).resolve())
        self.camera_assets = camera_assets
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._latest: Dict[int, Dict] = {
            cid: {"timestamp": None, "capture_ts": None, "detections": [], "overlay": None}
            for cid in camera_assets.keys()
        }
        self._visible_bytes: Dict[int, Optional[bytes]] = {}
        self._visible_arrays: Dict[int, Optional[np.ndarray]] = {}
        self._visible_meta: Dict[int, Optional[dict]] = {}
        for cid, asset in camera_assets.items():
            if asset.visible_cloud is None:
                self._visible_bytes[cid] = None
                self._visible_arrays[cid] = None
                self._visible_meta[cid] = None
                continue
            arr32 = np.asarray(asset.visible_cloud, dtype=np.float32)
            buf = io.BytesIO()
            np.savez_compressed(buf, xyz=arr32)
            self._visible_bytes[cid] = buf.getvalue()
            self._visible_arrays[cid] = arr32
            self._visible_meta[cid] = {"count": int(arr32.shape[0])}
        self.static_root = Path(static_root) if static_root else (Path(__file__).resolve().parent / "realtime_show_result" / "static")
        self.index_html = None
        self.app = self.FastAPI(title="Edge Web Bridge", version="0.1.0")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        if self.static_root.exists():
            self.index_html = self.static_root / "live.html"
            self.app.mount("/static", StaticFiles(directory=str(self.static_root), html=True), name="static")
        self._configure_routes()
        self._server = None
        self._thread = None

    def _configure_routes(self):
        @self.app.get("/")
        def _root():
            if self.index_html and self.index_html.exists():
                return self.FileResponse(str(self.index_html))
            return {"status": "ok", "message": "Edge Web Bridge"}

        @self.app.get("/healthz")
        def _health():
            return {"ok": True, "since": self.started_at}

        @self.app.get("/api/site")
        def _site():
            return {
                "site": self.site_name,
                "started_at": self.started_at,
                "cameras": list(self.camera_assets.keys()),
            }

        @self.app.get("/api/cameras")
        def _cameras():
            return [
                {
                    "camera_id": cid,
                    "name": asset.name,
                    "lut": asset.lut_path,
                    "resolution": [asset.raw_config.get("width"), asset.raw_config.get("height")],
                }
                for cid, asset in sorted(self.camera_assets.items())
            ]

        @self.app.get("/api/cameras/{camera_id}/detections")
        def _detections(camera_id: int):
            entry = self._latest.get(camera_id)
            if entry is None:
                raise self.HTTPException(status_code=404, detail="unknown camera id")
            payload = {k: v for k, v in entry.items() if k != "overlay"}
            return payload

        @self.app.get("/api/cameras/{camera_id}/overlay.jpg")
        def _overlay(camera_id: int):
            entry = self._latest.get(camera_id)
            if entry is None or entry.get("overlay") is None:
                raise self.HTTPException(status_code=404, detail="overlay not ready")
            return self.Response(content=entry["overlay"], media_type="image/jpeg")

        @self.app.get("/api/cameras/{camera_id}/visible-cloud.npz")
        def _visible(camera_id: int):
            data = self._visible_bytes.get(camera_id)
            if data is None:
                raise self.HTTPException(status_code=404, detail="visible cloud missing")
            return self.Response(content=data, media_type="application/octet-stream")

        @self.app.get("/api/cameras/{camera_id}/visible-meta")
        def _visible_meta(camera_id: int):
            meta = self._visible_meta.get(camera_id)
            if meta is None:
                raise self.HTTPException(status_code=404, detail="visible cloud missing")
            return meta

        @self.app.get("/api/cameras/{camera_id}/visible.bin")
        def _visible_bin(camera_id: int):
            arr = self._visible_arrays.get(camera_id)
            if arr is None:
                raise self.HTTPException(status_code=404, detail="visible cloud missing")
            return self.Response(
                content=arr.tobytes(),
                media_type="application/octet-stream",
                headers={"X-Point-Count": str(arr.shape[0])},
            )

        @self.app.get("/assets/global.ply")
        def _global():
            if not os.path.exists(self.global_ply):
                raise self.HTTPException(status_code=404, detail="global PLY missing")
            return self.FileResponse(self.global_ply, filename="global.ply")

        @self.app.get("/assets/vehicle.glb")
        def _vehicle():
            if not os.path.exists(self.vehicle_glb):
                raise self.HTTPException(status_code=404, detail="vehicle mesh missing")
            return self.FileResponse(self.vehicle_glb, filename="vehicle.glb")

    def _serialize_bev(self, bev_dets):
        items = []
        for d in bev_dets or []:
            center = d.get("center")
            tri = d.get("tri")
            front_edge = d.get("front_edge")
            if hasattr(front_edge, "tolist"):
                front_edge = front_edge.tolist()
            items.append({
                "center": [float(center[0]), float(center[1])] if center is not None else None,
                "length": float(d.get("length", 0.0)),
                "width": float(d.get("width", 0.0)),
                "yaw": float(d.get("yaw", 0.0)),
                "score": float(d.get("score", 0.0)),
                "cz": float(d.get("cz", 0.0)),
                "pitch": float(d.get("pitch", 0.0)),
                "roll": float(d.get("roll", 0.0)),
                "front_edge": front_edge,
                "tri": tri.tolist() if hasattr(tri, "tolist") else tri,
            })
        return items

    def update(self, cam_id: int, ts: float, capture_ts: float, bev_dets, overlay_bgr: Optional[np.ndarray]):
        overlay_bytes = None
        if overlay_bgr is not None:
            ok, buf = cv2.imencode(".jpg", overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                overlay_bytes = buf.tobytes()
        payload = {
            "timestamp": ts,
            "capture_ts": capture_ts,
            "detections": self._serialize_bev(bev_dets),
        }
        with self._lock:
            state = self._latest.setdefault(cam_id, {})
            state.update(payload)
            if overlay_bytes is not None:
                state["overlay"] = overlay_bytes

    def start(self):
        if self._server is not None:
            return
        config = self.uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info", access_log=False)
        self._server = self.uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        print(f"[WEB] serving http://{self.host}:{self.port}")

    def stop(self):
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

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
    ap.add_argument("--lut-dir", dest="lut_path", type=str, help="Alias for --lut-path (directory with cam*.npz)")
    ap.add_argument("--camera-config", type=str, help="JSON file describing RTSP/LUT/undistort per camera")
    ap.add_argument("--transport", default="tcp", choices=["tcp","udp"])
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--udp-enable", action="store_true")
    ap.add_argument("--udp-host", default="127.0.0.1")
    ap.add_argument("--udp-port", type=int, default=50050)
    ap.add_argument("--udp-format", choices=["json","text"], default="json")
    ap.add_argument("--visual-size", default="216,384", type=str)
    ap.add_argument("--target-fps", default=30, type=int)
    # web bridge
    ap.add_argument("--web-enable", action="store_true")
    ap.add_argument("--web-host", default="0.0.0.0")
    ap.add_argument("--web-port", type=int, default=18080)
    ap.add_argument("--web-site-name", default="edge-site")
    ap.add_argument("--web-jpeg-quality", type=int, default=85)
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
    
    # 카메라 설정 불러오기
    if args.camera_config:
        try:
            camera_configs = load_camera_config_file(args.camera_config, (H, W))
        except Exception as exc:
            raise SystemExit(f"[Main] failed to parse camera_config: {exc}")
    else:
        camera_configs = [
            {
                'ip': '192.168.0.210',
                'port': 554,
                'username': 'admin',
                'password': 'zjsxmfhf',
                'camera_id': 1,
                'width': W,
                'height': H,
                'force_tcp': (args.transport == "tcp"),
                "lut": "pointcloud/cloud_rgb_npz/cam1.npz",
                "undistort": "realtime/disto/cam1.npz",
                'name': 'cam1',
            },
            {
                'ip': '192.168.0.2',
                'port': 554,
                'username': 'admin',
                'password': 'zjsxmfhf',
                'camera_id': 2,
                'width': W,
                'height': H,
                'force_tcp': (args.transport == "tcp"),
                "lut": "pointcloud/cloud_rgb_npz/cam2.npz",
                "undistort": "realtime/disto/cam2.npz",
                'name': 'cam2',
            },
        ]
    if not camera_configs:
        raise SystemExit("[Main] no cameras configured")

    cam_cfg_map = {int(cfg["camera_id"]): cfg for cfg in camera_configs}
    camera_assets = build_camera_assets(camera_configs, args.lut_path)
    for cid, asset in camera_assets.items():
        cfg = asset.raw_config
        print(f"[Main] slot cam{cid} ({asset.name}): rtsp={cfg['ip']}:{cfg['port']} size={cfg['width']}x{cfg['height']} LUT={asset.lut_path}")
    
    cam_ids  = sorted(camera_assets.keys())
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
        for cid in cam_ids:
            cfg = cam_cfg_map[cid]
            asset = camera_assets.get(cid)
            try:
                viewer3d_map[cid] = PerCam3DViewer(
                    title=f"{asset.name if asset else f'cam{cid}'} • 3D",
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

    web_bridge = None
    if args.web_enable:
        try:
            web_bridge = EdgeWebBridge(
                host=args.web_host,
                port=args.web_port,
                global_ply=args.global_ply,
                vehicle_glb=args.vehicle_glb,
                camera_assets=camera_assets,
                site_name=args.web_site_name,
                jpeg_quality=args.web_jpeg_quality,
            )
            web_bridge.start()
        except Exception as e:
            print(f"[WEB] disabled: {e}")
            web_bridge = None

    # 시작
    streamer.start()
    # -------- GUI 큐 & 워커 --------
    gui_queue = queue.Queue(maxsize=128)
    worker = InferWorker(
        streamer=streamer,
        camera_assets=camera_assets,
        img_hw=(H, W),
        strides=strides,
        onnx_path=args.weights,
        score_mode=args.score_mode,
        conf=args.conf,
        nms_iou=args.nms_iou,
        topk=args.topk,
        bev_scale=args.bev_scale,
        providers=providers,
        viewer3d_map=viewer3d_map,
        gui_queue=gui_queue,
        udp_sender=udp_sender,
        web_publisher=web_bridge,
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

                vis = bgr
                if vis is None:
                    with gui_timer.span("gui.draw"):
                        blank = np.zeros((H, W, 3), dtype=np.uint8)
                        vis = draw_detections(blank, dets)

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
        if web_bridge:
            web_bridge.stop()
        if USE_GUI:
            try: cv2.destroyAllWindows()
            except: pass
        print("[Main] stopped.")


if __name__ == "__main__":
    main()
