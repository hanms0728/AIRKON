# RTSP 카메라에서 가져오는 거
import cv2
import threading
import time
import signal
import sys
import ffmpeg
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List

class IPCameraStreamerUltraLL:
    def __init__(
        self,
        camera_configs: List[Dict], # 카메라 설정 ㅇㅇ
        show_windows: bool = True, # 창 표시 여부
        target_fps: int = 60, 
        snapshot_dir: Optional[str] = None, # 스냅샷 경로
        snapshot_interval_sec: Optional[float] = None, # 스냅샷 주기
        catchup_seconds: float = 0.5,     # 연결 직후 버릴 시간 (0.3~1.0 권장) - 버퍼 털기
        overlay_ts: bool = True,          # 미리보기 시각 스탬프
        laytency_check: bool = False, # 지연시간 볼지말지
    ):
        self.camera_configs = camera_configs
        self.show_windows = show_windows
        self.target_fps = max(1, int(target_fps))
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.snapshot_interval_sec = snapshot_interval_sec
        self.catchup_seconds = max(0.0, float(catchup_seconds))
        self.overlay_ts = overlay_ts
        self.latency_check = laytency_check

        # 최신 프레임 1장만 유지 + 락
        self.latest = {cfg['camera_id']: deque(maxlen=1) for cfg in self.camera_configs} # 카메라별 최신 프레임 1장 보관용 deque 대기열 쌓여 지연되는 걸 막음
        self.last_served_ts = {cfg['camera_id']: None for cfg in self.camera_configs}    # get_latest로 마지막 전달된 timestamp
        self.locks = {cfg['camera_id']: threading.Lock() for cfg in self.camera_configs} # 각 카메라 프레임 교체 시 쓰는 락

        # 프로세스/스레드 관리
        self.procs: Dict[int, object] = {}
        self.threads: List[threading.Thread] = []
        self.running = True # 루프 유지 플래그 

        # OpenCV 내부 스레드 억제(충돌/과점 방지)
        cv2.setNumThreads(1)

        if self.snapshot_dir: # 근데 우린 안 주고 있긴 해~ 
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            self._last_snapshot_ts = {cfg['camera_id']: 0.0 for cfg in self.camera_configs}

        # ffmpeg 입력 프로필(가장 공격적 → 점진적 완화)
        # 주: 일부 옵션은 빌드/프로토콜에 따라 미지원일 수 있으므로 순차 시도
        # UDP (안정 → 공격 순서)
        self.ff_profiles_udp = [ # 입력 옵션 프리셋,, 저지연세팅 빡세게 넣엇대요 
            dict(rtsp_transport='udp',
                fflags='nobuffer',
                flags='low_delay',
                reorder_queue_size='0', 
                max_delay='0',         
                use_wallclock_as_timestamps='1',
                probesize='32k',
                analyzeduration='0'),
            dict(rtsp_transport='udp',
                fflags='nobuffer+discardcorrupt',   # ← 쉼표(,) 대신 +
                flags='low_delay',
                reorder_queue_size='0',
                max_delay='0',        
                use_wallclock_as_timestamps='1',
                probesize='32k',
                analyzeduration='0'),
            dict(rtsp_transport='udp',
                fflags='nobuffer+discardcorrupt',
                flags='low_delay',
                reorder_queue_size='0',
                max_delay='0',        
                use_wallclock_as_timestamps='1',
                probesize='32k',
                analyzeduration='1000k'),
        ]

        # TCP (안정 → 공격 순서)
        self.ff_profiles_tcp = [
            dict(rtsp_transport='tcp',
                fflags='nobuffer',
                flags='low_delay',
                reorder_queue_size='0',
                max_delay='0',        
                use_wallclock_as_timestamps='1',
                probesize='32k',
                analyzeduration='0'),
            dict(rtsp_transport='tcp',
                fflags='nobuffer+discardcorrupt',   
                flags='low_delay',
                reorder_queue_size='0',
                max_delay='0',        
                use_wallclock_as_timestamps='1',
                probesize='32k',
                analyzeduration='0'),
            dict(rtsp_transport='tcp',
                probesize='32k',
                reorder_queue_size='0',
                max_delay='0',        
                use_wallclock_as_timestamps='1',
                analyzeduration='1000k'),
        ]
    
    @staticmethod
    def _read_exact(stream, n: int) -> Optional[bytes]:
        """ffmpeg stdout에서 정확히 n바이트 읽어오는애(부분읽기 보정). EOF/끊김이면 None."""
        buf = bytearray()
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    @staticmethod
    def _close_proc(proc):
        ''' ffmpeg 프로세스 종료/정리. stdout/stderr 닫고 kill → wait '''
        if proc is None:
            return
        try:
            if getattr(proc, "stdout", None):
                proc.stdout.close()
            if getattr(proc, "stderr", None):
                proc.stderr.close()
            proc.kill()
            proc.wait(timeout=1)
        except Exception:
            pass

    def _make_urls(self, cfg: Dict) -> List[str]:
        """장비별 RTSP 경로에 맞게 필요시 수정."""
        ip, port, u, p = cfg['ip'], cfg['port'], cfg['username'], cfg['password']
        return [
            f"rtsp://{u}:{p}@{ip}:{port}/stream1",
            f"rtsp://{u}:{p}@{ip}:{port}/stream2",
        ]

    def _spawn_with_profiles(self, url: str, width: int, height: int, force_tcp: bool):
        '''해당 url로 ffmpeg 파이프를 띄우는 애'''
        last_err = None
        cand = self.ff_profiles_tcp if force_tcp else (self.ff_profiles_udp + self.ff_profiles_tcp) # tcp강제면 tcp프리셋만 아니면 udp->tcp시도

        # HW → SW 순차 시도
        for pass_idx in (0, 1):  # 0: hwaccel, 1: sw
            for opts in cand:
                try:
                    in_ = ffmpeg.input(url, **opts) # 해당 프리셋 순차시도,,와 해당 url
                    stream = (
                        in_
                        .filter('scale', width, height)
                        .output('pipe:', format='rawvideo', pix_fmt='bgr24', vsync='passthrough')
                    )
                    if pass_idx == 0: # 하드웨어로 먼저 해보겟은 
                        stream = stream.global_args('-loglevel', 'error', '-nostats', '-threads', '1',
                                                    '-hwaccel', 'videotoolbox')
                    else:
                        stream = stream.global_args('-loglevel', 'error', '-nostats', '-threads', '1')

                    proc = stream.run_async(pipe_stdout=True, pipe_stderr=True) # 비동기로 실행하고 출력을 파이프로 받음
                    print(f"[Spawn] {'HW' if pass_idx==0 else 'SW'} OK → {opts}") # 이걸로 스트리밍 연결햇ㅇ요
                    return proc
                except ffmpeg.Error as e: # 실패하면 에러반환 다음프리셋으로 ㄱㄱ, 모든시도 실패시 none반환
                    try:
                        err = e.stderr.decode('utf-8', errors='ignore')
                        print(f"[Spawn] {'HW' if pass_idx==0 else 'SW'} FAIL: {opts}\n{err.strip()}",
                            file=sys.stderr)
                        last_err = err
                    except Exception:
                        last_err = "Unknown ffmpeg error"
                        print(f"[Spawn] FAIL (unknown)", file=sys.stderr)

        print("[Spawn] all profiles failed.", file=sys.stderr)
        if last_err:
            print(last_err, file=sys.stderr)
        return None

    # ----------------- 스레드 -----------------
    def _camera_thread(self, cfg: Dict): # 카메라 하나를 담당하는 루프 - 지연시간 로그추가 
        cam_id = cfg['camera_id']
        width = int(cfg.get('width', 1920))
        height = int(cfg.get('height', 1080))
        bpf = width * height * 3 # 한 프레임의 바이트 수(가로세로3채널)
        force_tcp = bool(cfg.get('force_tcp', False))

        backoff = 0.5 # 재연결 대기시간 - 점점늘어날것임
        while self.running:
            urls = self._make_urls(cfg) # 캠마다 url을 만들겟죠?
            connected = False
            for url in urls: # 캠마다~ 
                if not self.running:
                    break

                proc = self._spawn_with_profiles(url, width, height, force_tcp)
                if not proc:
                    continue
                
                # 스트림 진짜 들어오나 확인용
                first = self._read_exact(proc.stdout, bpf)
                if first is None:
                    # 🔎 여기서 에러 로그 뿜기
                    try:
                        err_txt = proc.stderr.read().decode('utf-8', errors='ignore')
                        if err_txt.strip():
                            sys.stderr.write(f"[Cam{cam_id} FFmpeg stderr@init] {err_txt}\n")
                    except Exception:
                        pass
                    self._close_proc(proc)
                    continue

                # 초기 프레임 동기화(정확히 1프레임) + catch-up 드롭
                first = self._read_exact(proc.stdout, bpf)
                if first is None:
                    self._close_proc(proc)
                    continue

                if self.catchup_seconds > 0.0: # 이만큼 프레임 버릴것임,, 카메라/넷웤 버퍼에 쌓인 과거 프레임 비워서 지금시점으로 맞추는 것
                    deadline = time.time() + self.catchup_seconds
                    drop_cnt = 0
                    while time.time() < deadline:
                        junk = self._read_exact(proc.stdout, bpf)
                        if junk is None:
                            break
                        drop_cnt += 1
                    print(f"[Cam{cam_id}] catch-up dropped {drop_cnt} frames") # 이만큼 버렷어요

                print(f"[Cam{cam_id}] ✅ connected: {url} ({width}x{height})") # 이 캠은 이렇게 연결됏어요~
                self.procs[cam_id] = proc # 캠id의 프로세스를 관리해요~ 
                connected = True
                backoff = 0.5  # 성공 시 백오프 리셋

                # 런루프
                frame_count, last_t = 0, time.time()
                while self.running:
                    if self.latency_check:
                        start_read_time = time.time() # 프레임 읽기 시작 시각
                        data = self._read_exact(proc.stdout, bpf) # 이걸로 프레임 받음,, 
                        read_finish_time = time.time() # 프레임 읽기 완료 시각
                    else: data = self._read_exact(proc.stdout, bpf)
                    ts_capture = time.time()
                    if data is None:
                        # 끊기면 에러내용ㄱㄱ 재연결 
                        try:
                            with self.locks[cam_id]:
                                self.latest[cam_id].clear()
                                self.last_served_ts[cam_id] = None
                            err_txt = proc.stderr.read().decode('utf-8', errors='ignore')
                            if err_txt.strip():
                                sys.stderr.write(f"[Cam{cam_id} FFmpeg stderr] {err_txt}\n")
                        except Exception:
                            pass
                        print(f"[Cam{cam_id}] ⚠️ stream ended → reconnect")
                        break
                    
                    if self.latency_check:
                        # (수정) 프레임 읽기 지연 시간 로그
                        read_latency = (read_finish_time - start_read_time) * 1000 # 밀리초
                        print(f"📢 📢 📢 [Cam{cam_id}] FFmpeg Read Latency(넷웤 지연): {read_latency:.2f}ms") 
                        # # read_latency는 주로 네트워크/파이프 버퍼링 지연을 의미함

                    # writable 프레임
                    # 바이트 -> 세로 가로 3 넘파이 배열로 변환,, 
                    frame = np.frombuffer(data, np.uint8).reshape((height, width, 3)).copy()

                    with self.locks[cam_id]: # 이 캠 락해놓고 
                        dq = self.latest[cam_id] # 이 캠의 뎈
                        dq.clear() # 뎈을 싹 클리어한후
                        dq.append((frame,ts_capture)) # 가장 최신 프레임1장을 넣어둘거에요 당연히 넘파이배열로 
                    
                    if self.latency_check: 
                        frame_receive_time = time.time() # 📢 프레임 처리 완료/저장 시각
                        
                        # 📢 지연 시간 로그 추가
                        # # (프레임 읽기 시작 시각 vs 프레임 처리 완료 시각 비교)
                        # # (단순 numpy 변환 + lock 획득/해제 지연)
                        processing_latency = (frame_receive_time - read_finish_time) * 1000 # 밀리초
                        print(f"[📢📢📢Cam{cam_id}] ➡️ Frame Recv/Proc Latency(frame받기~프레임처리완): {processing_latency:.2f}ms")

                    # 스냅샷(선택) 잇으면 일정주기로 찰칵찰칵 
                    if self.snapshot_dir and self.snapshot_interval_sec:
                        now_ts = time.time()
                        if now_ts - getattr(self, "_last_snapshot_ts", {}).get(cam_id, 0.0) >= self.snapshot_interval_sec:
                            out_path = self.snapshot_dir / f"cam{cam_id}_{int(now_ts)}.jpg"
                            try:
                                cv2.imwrite(str(out_path), frame)
                            except Exception as e:
                                sys.stderr.write(f"[Cam{cam_id}] snapshot save error: {e}\n")
                            self._last_snapshot_ts[cam_id] = now_ts

                    # 5초마다 캠쳐 FPS 평균 측정해서 로그로 찍음 
                    frame_count += 1
                    now = time.time()
                    if self.latency_check:
                            estimated_e2e_latency = (now - frame_receive_time) * 1000 # 밀리초
                            latency_full = (now - start_read_time)* 1000
                            print(f"🕒 frame처리완~now: {estimated_e2e_latency:.2f}ms | **frame읽기시작~now(좀과하게잡은거임): {latency_full:.2f}ms")
                    if now - last_t >= 5.0:
                        fps = frame_count / (now - last_t)
                        print(f"[Cam{cam_id}]가 받는 fps≈{fps:.2f}") # 카메라 스레드에서 프레임이 들어오는 속도
                        
                        frame_count, last_t = 0, now

                # 루프 끊어지면 정리 후 다음 URL/재시도
                self._close_proc(proc)
                self.procs.pop(cam_id, None) # 이 프로세스 팝!
                if not self.running:
                    return

            # 모든 캠실패시 로그 남기고 백오프 지수 증가...해서 맨첨캠부터 다시시도
            if self.running and not connected:
                print(f"[Cam{cam_id}] ❌ all URLs failed, retry in {backoff:.1f}s")
            time.sleep(backoff)
            with self.locks[cam_id]:
                self.latest[cam_id].clear()
                self.last_served_ts[cam_id] = None
            backoff = min(backoff * 2, 5.0)

    # ----------------- API -----------------
    def start(self):
        '''카메라 수만큼 스레드 만들어서 threads변수에 넣어두고'''
        for cfg in self.camera_configs:
            th = threading.Thread(target=self._camera_thread, args=(cfg,), daemon=True)
            th.start()
            self.threads.append(th)
        print("[Main] streamer started")

    def get_latest(self, camera_id: int) -> Optional[np.ndarray]:
        '''이 캠에 대해 락걸고 뎈에서 이 캠의 가장최근프레임1장이 잇으면~ 반환 없으면 none'''
        with self.locks[camera_id]:
            if not self.latest[camera_id]:
                return None
            frame, ts_capture = self.latest[camera_id][-1]
            if self.last_served_ts.get(camera_id) == ts_capture:
                return None
            self.last_served_ts[camera_id] = ts_capture
            return frame, ts_capture
        return None

    def stop(self):
        ''' ffmpeg프로세스 정리, 스레드 스탑'''
        self.running = False
        for _, p in list(self.procs.items()):
            self._close_proc(p)
        for th in self.threads:
            th.join(timeout=1)
        print("[Main] streamer stopped")

    def run_preview_loop(self): # 근데 이거 realtime에서 안쓰긴해
        '''실제 해당 캠에 대해 띄우는 루프'''
        if not self.show_windows:
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            return

        try: # 창 띄워봐요~ 계속해서. 모든캠을 돌며~
            spf = 1.0 / self.target_fps
            while self.running:
                start = time.time()
                for cfg in self.camera_configs:
                    cam_id = cfg['camera_id']
                    latest = self.get_latest(cam_id)
                    if latest is None:
                        continue
                    frame, ts_capture = latest
                    vis = frame  # 이미 copy()된 writable 프레임
                    if self.overlay_ts: # 이거 주지 말라한것같음,, 
                        ts = time.time()
                        txt = time.strftime('%H:%M:%S', time.localtime(ts)) + f".{int((ts%1)*1000):03d}"
                        cv2.putText(vis, txt, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)
                    cv2.imshow(f"Cam{cam_id}", vis) # 최신 거 보여줍시대
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    self.stop()
                    break
                remain = spf - (time.time() - start)
                if remain > 0:
                    time.sleep(remain)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

# ----------------- 실행부 -----------------
def main():
    # 카메라별로 force_tcp 설정 가능(UDP 손실/초록깨짐 발생 카메라엔 True)
    camera_configs = [
        {
            'ip': '192.168.0.3',
            'port': 554,
            'username': 'admin',
            'password': 'zjsxmfhf',
            'camera_id': 1,
            'width': 1536,
            'height': 864,
            'force_tcp': False,   # UDP 우선(저지연)
        },
        {
            'ip': '192.168.0.2',
            'port': 554,
            'username': 'admin',
            'password': 'zjsxmfhf',
            'camera_id': 2,
            'width': 1536,
            'height': 864,
            'force_tcp': False,    # 이 카메라만 TCP 강제(손실/초록깨짐 방지)
        },
    ]

    snapshot_dir = None          # 예: "./snapshots"
    snapshot_interval = None     # 예: 2.0 (초)

    streamer = IPCameraStreamerUltraLL(
        camera_configs,
        show_windows=True,
        target_fps=60,
        snapshot_dir=snapshot_dir,
        snapshot_interval_sec=snapshot_interval,
        catchup_seconds=0.5,
        overlay_ts=False,
        laytency_check=True
    )

    def _sigint_handler(sig, frame): # ctrl c 들어오면 딱 stop` 
        streamer.stop()
    signal.signal(signal.SIGINT, _sigint_handler)

    streamer.start()
    streamer.run_preview_loop()
    

if __name__ == '__main__':
    main()
