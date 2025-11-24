"""
TensorRT backend for the temporal batch runner used in realtime inference.

This mirrors the interface of realtime.v1.batch_infer.BatchedTemporalRunner but
uses TensorRT for execution so that inference can leverage Tensor Cores on
NVIDIA GPUs.
"""
from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .batch_infer import splitnum


@dataclass
class _DeviceBuffer:
    ptr: int
    handle: object
    size: int


@dataclass
class _BindingInfo:
    name: str
    index: int
    shape: Tuple[int, ...]
    dtype: np.dtype
    is_input: bool
    device: _DeviceBuffer
    host: np.ndarray


class _CudaRuntime:
    """
    Minimal CUDA runtime wrapper that works with either cuda-python or PyCUDA.
    """

    def __init__(self, device_id: int = 0):
        self.device_id = int(device_id)
        self.impl = None
        self.cudart = None
        self.cuda = None
        self._pycuda_ctx = None
        cuda_python_error = None
        try:
            from cuda import cudart  # type: ignore

            self.impl = "cuda-python"
            self.cudart = cudart
            err = self.cudart.cudaSetDevice(self.device_id)
            self._check(err, "cudaSetDevice")
            return
        except Exception as exc:
            cuda_python_error = exc

        try:
            import pycuda.autoinit  # type: ignore
            import pycuda.driver as cuda  # type: ignore

            self.impl = "pycuda"
            self.cuda = cuda
            self._pycuda_ctx = pycuda.autoinit.context
        except Exception as exc:  # pragma: no cover - import guard
            hint = (
                "TensorRT backend requires the 'cuda-python' package "
                "(providing cuda.cudart) or 'pycuda'."
            )
            if cuda_python_error:
                hint += f" cuda-python import failed with: {cuda_python_error!r}"
            raise RuntimeError(hint) from exc

    def _push_context(self) -> bool:
        if self.impl != "pycuda" or self._pycuda_ctx is None:
            return False
        cur = self.cuda.Context.get_current()
        if cur is self._pycuda_ctx:
            return False
        self._pycuda_ctx.push()
        return True

    def _pop_context(self, pushed: bool):
        if pushed and self._pycuda_ctx is not None:
            self._pycuda_ctx.pop()

    def malloc(self, nbytes: int) -> _DeviceBuffer:
        if self.impl == "cuda-python":
            err, ptr = self.cudart.cudaMalloc(nbytes)
            self._check(err, "cudaMalloc")
            return _DeviceBuffer(ptr=ptr, handle=ptr, size=nbytes)
        pushed = self._push_context()
        try:
            mem = self.cuda.mem_alloc(nbytes)
        finally:
            self._pop_context(pushed)
        return _DeviceBuffer(ptr=int(mem), handle=mem, size=nbytes)

    def free(self, buf: _DeviceBuffer):
        if self.impl == "cuda-python":
            err = self.cudart.cudaFree(buf.ptr)
            self._check(err, "cudaFree")
        else:
            pushed = self._push_context()
            try:
                buf.handle.free()
            finally:
                self._pop_context(pushed)

    def memcpy_htod(self, buf: _DeviceBuffer, host: np.ndarray):
        arr = np.ascontiguousarray(host)
        if arr.nbytes > buf.size:
            raise ValueError(
                f"Host array larger than device buffer ({arr.nbytes} > {buf.size})"
            )
        if self.impl == "cuda-python":
            err = self.cudart.cudaMemcpy(
                buf.ptr,
                arr.ctypes.data,
                arr.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
            self._check(err, "cudaMemcpyHostToDevice")
        else:
            pushed = self._push_context()
            try:
                self.cuda.memcpy_htod(buf.handle, arr)
            finally:
                self._pop_context(pushed)

    def memcpy_dtoh(self, host: np.ndarray, buf: _DeviceBuffer):
        arr = np.ascontiguousarray(host)
        if arr.nbytes > buf.size:
            raise ValueError(
                f"Device buffer smaller than host array ({buf.size} < {arr.nbytes})"
            )
        if self.impl == "cuda-python":
            err = self.cudart.cudaMemcpy(
                arr.ctypes.data,
                buf.ptr,
                arr.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
            self._check(err, "cudaMemcpyDeviceToHost")
        else:
            pushed = self._push_context()
            try:
                self.cuda.memcpy_dtoh(arr, buf.handle)
            finally:
                self._pop_context(pushed)
        if arr.ctypes.data != host.ctypes.data:
            np.copyto(host, arr)

    def _check(self, err_code: int, op: str):
        if err_code != 0:
            name = (
                self.cudart.cudaGetErrorName(err_code)[1]
                if self.impl == "cuda-python"
                else str(err_code)
            )
            raise RuntimeError(f"{op} failed with CUDA error {name}")


class TensorRTBatchedTemporalRunner:
    """
    Drop-in replacement for BatchedTemporalRunner that executes the ONNX graph
    via TensorRT.
    """

    def __init__(
        self,
        *,
        onnx_path: Optional[str],
        cam_ids: Sequence[int],
        img_size: Tuple[int, int],
        temporal: str = "lstm",
        state_stride_hint: int = 32,
        default_hidden_ch: int = 256,
        engine_path: Optional[str] = None,
        save_engine_path: Optional[str] = None,
        workspace_mb: int = 2048,
        fp16: bool = False,
        max_batch_size: Optional[int] = None,
        logger_level: str = "WARNING",
        verbose: bool = False,
    ):
        self.cam_ids = list(cam_ids)
        assert len(self.cam_ids) >= 1, "Need at least one camera feed"
        self.id2idx = {cid: i for i, cid in enumerate(self.cam_ids)}
        self.B = len(self.cam_ids)
        self.H, self.W = img_size
        self.temporal = temporal
        self.state_stride_hint = max(1, int(state_stride_hint))
        self.default_hidden_ch = int(default_hidden_ch)
        self.state_hw = (
            max(1, self.H // self.state_stride_hint),
            max(1, self.W // self.state_stride_hint),
        )
        self.pending = np.zeros((self.B,), dtype=bool)

        self.onnx_path = Path(onnx_path).expanduser() if onnx_path else None
        if self.onnx_path is None and engine_path is None:
            raise ValueError("Either an ONNX path or a TensorRT engine path must be provided.")
        if self.onnx_path is not None and not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX weights not found: {self.onnx_path}")
        self.engine_path = Path(engine_path).expanduser() if engine_path else None
        self.save_engine_path = (
            Path(save_engine_path).expanduser() if save_engine_path else None
        )
        self.max_batch_size = int(max_batch_size or self.B)
        if self.max_batch_size < self.B:
            raise ValueError(
                f"TensorRT profile max batch {self.max_batch_size} "
                f"is smaller than active cameras ({self.B})."
            )

        self.trt = self._import_trt()
        log_level = getattr(self.trt.Logger, logger_level.upper(), None)
        if log_level is None:
            log_level = self.trt.Logger.INFO if verbose else self.trt.Logger.WARNING
        self.logger = self.trt.Logger(log_level)
        self.runtime = self.trt.Runtime(self.logger)
        self.engine = self._load_or_build_engine(
            workspace_mb=workspace_mb, fp16=fp16
        )
        self.context = self.engine.create_execution_context()
        self.cuda = _CudaRuntime()
        self._legacy_bindings = hasattr(self.engine, "num_bindings")
        self.binding_info = self._prepare_bindings()
        self._classify_bindings()

        img_dtype = self.binding_info[self.x_name].dtype
        self.img_buf = np.zeros(self.binding_info[self.x_name].shape, dtype=img_dtype)
        self.h_buf = (
            np.zeros(self.binding_info[self.h_name].shape, dtype=self.binding_info[self.h_name].dtype)
            if self.h_name
            else None
        )
        self.c_buf = (
            np.zeros(self.binding_info[self.c_name].shape, dtype=self.binding_info[self.c_name].dtype)
            if self.c_name
            else None
        )
        max_index = max(info.index for info in self.binding_info.values())
        self.device_bindings = [0] * (max_index + 1)
        for info in self.binding_info.values():
            self.device_bindings[info.index] = int(info.device.ptr)

    def _import_trt(self):
        try:
            import tensorrt as trt  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "TensorRT Python package is not installed. "
                "Install NVIDIA TensorRT >= 8.6 to enable this backend."
            ) from exc
        return trt

    def _load_or_build_engine(self, workspace_mb: int, fp16: bool):
        if self.engine_path and self.engine_path.exists():
            with open(self.engine_path, "rb") as f:
                engine_bytes = f.read()
            engine = self.runtime.deserialize_cuda_engine(engine_bytes)
            if engine is None:
                raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
            return engine
        if self.onnx_path is None:
            raise RuntimeError("TensorRT engine file not found and no ONNX path was supplied.")
        engine = self._build_engine(self.onnx_path, workspace_mb, fp16)
        target_path = self.engine_path or self.save_engine_path
        if target_path:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(engine.serialize())
        return engine

    def _build_engine(self, onnx_path: Path, workspace_mb: int, fp16: bool):
        builder = self.trt.Builder(self.logger)
        network_flags = 1 << int(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = self.trt.OnnxParser(network, self.logger)
        with open(onnx_path, "rb") as f:
            model_bytes = f.read()
        if not parser.parse(model_bytes):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(
                "TensorRT failed to parse ONNX:\n" + "\n".join(str(e) for e in errors)
            )
        config = builder.create_builder_config()
        workspace_bytes = int(workspace_mb) * (1 << 20)
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(self.trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        elif hasattr(config, "max_workspace_size"):
            config.max_workspace_size = workspace_bytes
        else:  # pragma: no cover - safety for unexpected TRT versions
            raise RuntimeError("TensorRT builder config does not expose workspace controls.")
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(self.trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            mins, opts, maxs = self._profile_shapes_for_input(tuple(inp.shape), inp.name)
            profile.set_shape(inp.name, mins, opts, maxs)
        config.add_optimization_profile(profile)
        engine = None
        if hasattr(builder, "build_engine"):
            engine = builder.build_engine(network, config)
        elif hasattr(builder, "build_engine_with_config"):
            engine = builder.build_engine_with_config(network, config)
        else:
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                raise RuntimeError("TensorRT failed to build a serialized network.")
            engine = self.runtime.deserialize_cuda_engine(plan)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed.")
        return engine

    def _profile_shapes_for_input(
        self, dims: Tuple[int, ...], name: str
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        name_l = name.lower()

        def resolve(dim_idx: int, dim_val: int, batch: int) -> int:
            if dim_val >= 0:
                return dim_val
            if dim_idx == 0:
                return batch
            if "h" in name_l or "c" in name_l:
                if dim_idx == 1:
                    return self.default_hidden_ch
                if dim_idx == 2:
                    return self.state_hw[0]
                if dim_idx == 3:
                    return self.state_hw[1]
            raise ValueError(f"Dynamic dimension {dim_idx} in {name} is unsupported.")

        min_batch = 1
        max_batch = self.max_batch_size
        opt_batch = min(self.B, max_batch)
        mins = tuple(resolve(i, d, min_batch) for i, d in enumerate(dims))
        opts = tuple(resolve(i, d, opt_batch) for i, d in enumerate(dims))
        maxs = tuple(resolve(i, d, max_batch) for i, d in enumerate(dims))
        return mins, opts, maxs

    def _prepare_bindings(self) -> Dict[str, _BindingInfo]:
        bindings: Dict[str, _BindingInfo] = {}
        if self._legacy_bindings:
            total = self.engine.num_bindings
        else:
            total = getattr(self.engine, "num_io_tensors", 0)

        for idx in range(total):
            if self._legacy_bindings:
                name = self.engine.get_binding_name(idx)
                raw_shape = tuple(self.engine.get_binding_shape(idx))
                is_input = self.engine.binding_is_input(idx)
                dtype = np.dtype(self.trt.nptype(self.engine.get_binding_dtype(idx)))
            else:
                name = self.engine.get_tensor_name(idx)
                if name is None:
                    continue
                mode = self.engine.get_tensor_mode(name)
                if mode == self.trt.TensorIOMode.NONE:
                    continue
                is_input = mode == self.trt.TensorIOMode.INPUT
                raw_shape = tuple(self.engine.get_tensor_shape(name))
                dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self._resolve_binding_shape(name, raw_shape)
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"Invalid resolved shape {shape} for binding {name}")
            host = np.zeros(shape, dtype=dtype)
            buf = self.cuda.malloc(host.nbytes)
            info = _BindingInfo(
                name=name,
                index=idx,
                shape=shape,
                dtype=dtype,
                is_input=is_input,
                device=buf,
                host=host,
            )
            bindings[name] = info
            if -1 in raw_shape or not self._legacy_bindings:
                if self._legacy_bindings:
                    self.context.set_binding_shape(idx, shape)
                elif is_input:
                    ok = self.context.set_input_shape(name, shape)
                    if not ok:
                        raise RuntimeError(f"Failed to set input shape for tensor {name}")
        if not self._legacy_bindings and hasattr(self.context, "infer_shapes"):
            self.context.infer_shapes()
        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("Failed to set TensorRT binding shapes.")
        return bindings

    def _resolve_binding_shape(
        self, name: str, raw_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        name_l = name.lower()
        resolved: List[int] = []
        for dim_idx, dim in enumerate(raw_shape):
            if dim >= 0:
                resolved.append(dim)
                continue
            if dim_idx == 0:
                resolved.append(self.B)
            elif ("h" in name_l or "c" in name_l) and dim_idx == 1:
                resolved.append(self.default_hidden_ch)
            elif ("h" in name_l or "c" in name_l) and dim_idx == 2:
                resolved.append(self.state_hw[0])
            elif ("h" in name_l or "c" in name_l) and dim_idx == 3:
                resolved.append(self.state_hw[1])
            else:
                raise ValueError(f"Cannot resolve dynamic dim {dim_idx} for {name}")
        return tuple(resolved)

    def _classify_bindings(self):
        names = list(self.binding_info.keys())

        def pick(tokens: Sequence[str], fallback: Optional[str] = None) -> Optional[str]:
            for n in names:
                lower = n.lower()
                if any(tok in lower for tok in tokens):
                    return n
            return fallback

        self.x_name = pick(["images", "image", "input"])
        if self.x_name is None:
            raise RuntimeError("Failed to locate image input binding for TensorRT.")
        self.h_name = pick(["h_in"])
        self.c_name = pick(["c_in"])
        self.ho_name = pick(["h_out"])
        self.co_name = pick(["c_out"])
        outs = [n for n, info in self.binding_info.items() if not info.is_input]
        self.reg_names = sorted(
            [n for n in outs if "reg" in n.lower()],
            key=lambda s: [int(t) if t.isdigit() else t for t in splitnum(s)],
        )
        self.obj_names = sorted(
            [n for n in outs if "obj" in n.lower()],
            key=lambda s: [int(t) if t.isdigit() else t for t in splitnum(s)],
        )
        self.cls_names = sorted(
            [n for n in outs if "cls" in n.lower()],
            key=lambda s: [int(t) if t.isdigit() else t for t in splitnum(s)],
        )
        if not (len(self.reg_names) == len(self.obj_names) == len(self.cls_names)):
            raise RuntimeError("TensorRT bindings do not expose balanced reg/obj/cls outputs.")

    def reset(self, cam_id: Optional[int] = None):
        if self.h_buf is not None:
            if cam_id is None:
                self.h_buf[:] = 0
            else:
                self.h_buf[self.id2idx[cam_id]] = 0
        if self.c_buf is not None:
            if cam_id is None:
                self.c_buf[:] = 0
            else:
                self.c_buf[self.id2idx[cam_id]] = 0
        if cam_id is None:
            self.pending[:] = False
        else:
            self.pending[self.id2idx[cam_id]] = False

    def enqueue_frame(self, cam_id: int, img_chw_float01: np.ndarray):
        i = self.id2idx[cam_id]
        assert img_chw_float01.shape == self.img_buf.shape[1:], "Invalid frame shape"
        self.img_buf[i] = img_chw_float01
        self.pending[i] = True

    def ready(self) -> bool:
        return bool(self.pending.all())

    def _copy_input(self, name: Optional[str], data: Optional[np.ndarray]):
        if name is None or data is None:
            return
        info = self.binding_info[name]
        self.cuda.memcpy_htod(info.device, data.astype(info.dtype, copy=False))

    def _fetch_output(self, name: str) -> np.ndarray:
        info = self.binding_info[name]
        self.cuda.memcpy_dtoh(info.host, info.device)
        return info.host.copy()

    def run_if_ready(self):
        if not self.ready():
            return None
        self._copy_input(self.x_name, self.img_buf)
        self._copy_input(self.h_name, self.h_buf)
        self._copy_input(self.c_name, self.c_buf)
        ok = self.context.execute_v2(self.device_bindings)
        if not ok:
            raise RuntimeError("TensorRT execution failed.")
        out_map: Dict[str, np.ndarray] = {}
        for name, info in self.binding_info.items():
            if info.is_input:
                continue
            out_map[name] = self._fetch_output(name)

        if self.ho_name:
            self.h_buf = out_map[self.ho_name]
        if self.co_name:
            self.c_buf = out_map[self.co_name]

        per_cam: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {
            cid: [] for cid in self.cam_ids
        }
        for rn, on, cn in zip(self.reg_names, self.obj_names, self.cls_names):
            reg_b = np.ascontiguousarray(out_map[rn].astype(np.float32, copy=False))
            obj_b = np.ascontiguousarray(out_map[on].astype(np.float32, copy=False))
            cls_b = np.ascontiguousarray(out_map[cn].astype(np.float32, copy=False))
            for i, cid in enumerate(self.cam_ids):
                pr = torch.from_numpy(reg_b[i : i + 1].copy())
                po = torch.from_numpy(obj_b[i : i + 1].copy())
                pc = torch.from_numpy(cls_b[i : i + 1].copy())
                per_cam[cid].append((pr, po, pc))

        self.pending[:] = False
        return per_cam
