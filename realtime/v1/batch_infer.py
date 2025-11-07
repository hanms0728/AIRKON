import numpy as np
import onnxruntime as ort
import torch

class BatchedTemporalRunner:
    """
    - 한 개의 ONNX 세션으로 B개의 카메라를 동시 추론
    - 카메라별 상태(h/c)와 입력 프레임 버퍼를 배치 축으로 관리
    - 모든 카메라 프레임이 도착했을 때만 run()
    """
    def __init__(self, onnx_path, cam_ids, img_size, temporal="lstm",
                 providers=("CUDAExecutionProvider","CPUExecutionProvider"),
                 state_stride_hint=32, default_hidden_ch=256):
        assert len(cam_ids) >= 1
        self.cam_ids = list(cam_ids)  # 고정된 슬롯 순서 음근데 이러면 이클래스에 cam_ids 줄 때도 순서 똑바로 줘야겟다...
        self.id2idx  = {cid:i for i,cid in enumerate(self.cam_ids)} # 아 아니다 cam id랑 순서로 매핑하는구나 딕셔너리에 맨첨에. 이 러너 한번부를거잖아.
        self.B = len(cam_ids) # 배치사이즈 = 캠 개수
        self.H, self.W = img_size # h, w... 이미지 사이즈... 864, 1536겟죠 

        # 세션 생성 + 입출력 이름 수집 onnx템퍼럴 러너에도 있었은 
        self.sess = ort.InferenceSession(onnx_path, providers=list(providers)) # 인퍼런스 세션은 하나만 쓸거이에요
        inputs = {i.name:i for i in self.sess.get_inputs()}
        outs   = [o.name for o in self.sess.get_outputs()]
        # 입력 이름 추론 (onnx템퍼럴ㄱunner의 규칙 재사용) 입력 이미지 텐서 이름을 추정images/image/input
        def pick(keys, fallback=None):
            for k in inputs:
                if any(t in k.lower() for t in keys): return k
            return fallback
        self.x_name = pick(["images","image","input"], list(inputs.keys())[0])
        self.h_name = pick(["h_in"]) # state 입력 잇으면 잡고 없으면 non템퍼럴...
        self.c_name = pick(["c_in"])
        # 출력이름그룹...
        def first_out(keys): # 출력중에 keys라는 토큰으로 분류,, keys:state나 예측,,
            for o in outs:
                if any(t in o.lower() for t in keys): return o
            return None
        self.ho_name = first_out(["h_out"]) # 출력중에 히든state가 잇으면 가져옴  
        self.co_name = first_out(["c_out"])
        self.reg_names = sorted([o for o in outs if "reg" in o.lower()], key=lambda s: [int(t) if t.isdigit() else t for t in splitnum(s)])
        self.obj_names = sorted([o for o in outs if "obj" in o.lower()], key=lambda s: [int(t) if t.isdigit() else t for t in splitnum(s)])
        self.cls_names = sorted([o for o in outs if "cls" in o.lower()], key=lambda s: [int(t) if t.isdigit() else t for t in splitnum(s)])

        # 상태 텐서 초기화 (배치로)
        C = default_hidden_ch
        Hs = max(1, self.H // state_stride_hint) # 864//32 면 27임 상태맵의 Hs가 이정도겟구나~
        Ws = max(1, self.W // state_stride_hint) # 1536//32면 48임
        self.h_buf = np.zeros((self.B, C, Hs, Ws), dtype=np.float32) if self.h_name else None # 실행중에 유지될 state버퍼 -넘파이
        self.c_buf = np.zeros((self.B, C, Hs, Ws), dtype=np.float32) if self.c_name else None

        # 프레임 대기 버퍼
        self.pending = np.zeros((self.B,), dtype=bool) # 프레임 동기화용 pending... 일단 이렇게 했는데 타임스탬프로 같프레임 받는게 더 좋대
        self.img_buf = np.zeros((self.B, 3, self.H, self.W), dtype=np.float32)  # 각 배치별로 3채널짜리(rgb) H*W이미지 하나가 오겟죠 

    def reset(self, cam_id=None): # 리셋할거임. 따로 캠id안주면 다 리셋할거고 주면 걔만 리셋할거임
        """cam_id 없으면 전체 상태 리셋, 있으면 해당 슬롯만 리셋"""
        if self.h_buf is not None:
            if cam_id is None: self.h_buf[:] = 0 # 전부 0으로 리셋 
            else: self.h_buf[self.id2idx[cam_id]] = 0 # h버퍼[해당캠idx] = 0으로 리셋
        if self.c_buf is not None:
            if cam_id is None: self.c_buf[:] = 0
            else: self.c_buf[self.id2idx[cam_id]] = 0
        if cam_id is None:
            self.pending[:] = False # 프레임 동기화용 펜딩(해당캠에 프레임이 왔나요!?)도 전부 F
        else:
            self.pending[self.id2idx[cam_id]] = False # 해당 프레임만 아직 안왓단 의미로 F

    def enqueue_frame(self, cam_id, img_chw_float01):
        # 해상도 고정 스트림 전제임... 다른해상도면 리사이즈해서 이 함수에 넣으셈
        """
        img_chw_float01: (3,H,W) float32 [0..1]
        """
        i = self.id2idx[cam_id] # 캠id의 인덱스
        assert img_chw_float01.shape == (3, self.H, self.W) # 셰잎은 이게 확정이셔
        self.img_buf[i] = img_chw_float01 # 이미지 버퍼에 이번 프레임의 이 캠의 이미지를 넣어요
        self.pending[i] = True # 들어왔어요~ 

    def ready(self): # 펜딩이 전부 트루인가요? 즉 모든 프레임이 도착했나요? 
        return bool(self.pending.all())

    def run_if_ready(self): # onnx템퍼럴 러너의 forward부분
        """
        ready()일 때만 실행. 결과는 카메라ID별 리스트로 반환.
        반환: dict[cam_id] -> list of (reg,obj,cls) (torch.Tensor, CPU)
        """
        if not self.ready():
            return None

        # 피드에 이미지와 이전 state를 넣음 
        feeds = { self.x_name: self.img_buf } # 피드에 image: 이미지버퍼(실제이미지저장된)
        if self.h_name is not None: feeds[self.h_name] = self.h_buf # 히든state: 실제 h버퍼 (B, C, Hs, Ws)
        if self.c_name is not None: feeds[self.c_name] = self.c_buf # 셀 state

        out_vals = self.sess.run(None, feeds) 
        out_names = [o.name for o in self.sess.get_outputs()]
        out_map = {n:v for n,v in zip(out_names, out_vals)} # 출력이름순서대로 추론값 배열 나온거 dict

        # 상태 갱신 (배치 전체)
        if self.ho_name: self.h_buf = out_map[self.ho_name] # 이번 출력에서의 hidden state가 잇으면 그걸로 업뎃
        if self.co_name: self.c_buf = out_map[self.co_name]

        # 스케일별 예측을 배치로 받아서 각 카메라별로 분해
        per_cam = {cid: [] for cid in self.cam_ids}
        for rn,on,cn in zip(self.reg_names, self.obj_names, self.cls_names):
            reg_b = out_map[rn]   # (B,6,Hs,Ws) # 실제 이번 프레임의 추론값
            obj_b = out_map[on]   # (B,1,Hs,Ws)
            cls_b = out_map[cn]   # (B,Cc,Hs,Ws)
            for i,cid in enumerate(self.cam_ids):
                # torch 텐서로 바꿔 PyTorch 디코더 재사용
                pr = torch.from_numpy(reg_b[i:i+1].copy())
                po = torch.from_numpy(obj_b[i:i+1].copy())
                pc = torch.from_numpy(cls_b[i:i+1].copy())
                per_cam[cid].append((pr,po,pc)) # 캠id - 각 예측한거넣어줌 

        # 배치 완료 → pending 클리어
        self.pending[:] = False # 이번 프레임 완료했으니까 펜딩 리셋 
        return per_cam # 캠별로 예측한 값이 나옴

def splitnum(s):
    acc=""
    out=[]
    for ch in s:
        if ch.isdigit():
            acc += ch
        else:
            if acc:
                out.append(acc)
                acc=""
                out.append(ch)
    if acc:
        out.append(acc)
    return out