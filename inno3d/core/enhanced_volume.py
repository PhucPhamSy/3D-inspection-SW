"""
BumpVoid ISP Enhancement DLL — Python wrapper
─────────────────────────────────────────────
ctypes wrapper for BumpVoid_ISP_ENH.dll (C# ONNX inference, multi-GPU).
Native exports keep EnhancedVolume_* names for ABI stability.
"""

import ctypes
import os
import sys

# ============================================
# DLL GLOBALS
# ============================================
_dll = None
_dll_path = None

# Preferred ship name, then legacy fallback
_DLL_CANDIDATES = (
    "BumpVoid_ISP_ENH.dll",
    "EnhancedVolumeDLL.dll",  # legacy
)

# Progress callback type  (matches C# delegate)
PROGRESS_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_int
)


def _default_progress(message, current, total):
    msg = message.decode("utf-8") if isinstance(message, bytes) else message
    print(f"  [{current}/{total}] {msg}")


_progress_cb = PROGRESS_CALLBACK(_default_progress)


# ============================================
# LOAD DLL
# ============================================
def load_dll(dll_dir: str):
    """Load BumpVoid_ISP_ENH.dll (or legacy EnhancedVolumeDLL.dll) from dll_dir."""
    global _dll, _dll_path

    dll_dir = os.path.abspath(dll_dir)
    if not os.path.isdir(dll_dir):
        raise FileNotFoundError(f"Directory not found: {dll_dir}")

    dll_path = None
    for name in _DLL_CANDIDATES:
        candidate = os.path.join(dll_dir, name)
        if os.path.isfile(candidate):
            dll_path = candidate
            break

    if dll_path is None:
        raise FileNotFoundError(
            f"BumpVoid_ISP_ENH.dll not found in: {dll_dir}\n"
            f"(also checked legacy EnhancedVolumeDLL.dll)"
        )

    # Python 3.8+: explicitly add DLL search directories
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetDllDirectoryW(dll_dir)
        except Exception:
            pass
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass

        # Add CUDA bin paths
        for key in os.environ:
            if key.startswith("CUDA_PATH"):
                cuda_root = os.environ.get(key)
                if cuda_root and os.path.isdir(cuda_root):
                    cuda_bin = os.path.join(cuda_root, "bin")
                    if os.path.isdir(cuda_bin):
                        try:
                            os.add_dll_directory(cuda_bin)
                        except Exception:
                            pass

        # Register PATH dirs so CUDA / cuDNN / TensorRT resolve
        for path_dir in os.environ.get("PATH", "").split(os.pathsep):
            if path_dir and os.path.isdir(path_dir):
                try:
                    os.add_dll_directory(path_dir)
                except Exception:
                    pass

    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

    try:
        _dll = ctypes.CDLL(dll_path)
        _dll_path = dll_path
        _setup_functions()
    except OSError as e:
        raise OSError(
            f"Failed to load {os.path.basename(dll_path)}: {e}\n"
            "Ensure ONNX Runtime native DLLs are in the same folder (V2)."
        )


def _check_dll():
    if _dll is None:
        raise RuntimeError(
            "BumpVoid_ISP_ENH not loaded. "
            "Call enhanced_volume.load_dll('path/to/V2') first."
        )


# ============================================
# FUNCTION SIGNATURES
# ============================================
def _setup_functions():
    _dll.EnhancedVolume_GetVersion.argtypes = []
    _dll.EnhancedVolume_GetVersion.restype = ctypes.c_char_p

    _dll.EnhancedVolume_GetLastError.argtypes = []
    _dll.EnhancedVolume_GetLastError.restype = ctypes.c_char_p

    _dll.EnhancedVolume_IsInitialized.argtypes = []
    _dll.EnhancedVolume_IsInitialized.restype = ctypes.c_int

    _dll.EnhancedVolume_Init.argtypes = [
        ctypes.c_char_p,  # modelPath
        ctypes.c_char_p,  # trtCachePath
        ctypes.c_int,     # useGpu
        ctypes.c_int,     # gpuDeviceId
    ]
    _dll.EnhancedVolume_Init.restype = ctypes.c_int

    _dll.EnhancedVolume_WarmUp.argtypes = [ctypes.c_int, ctypes.c_int]
    _dll.EnhancedVolume_WarmUp.restype = ctypes.c_int

    _dll.EnhancedVolume_ProcessFolder.argtypes = [
        ctypes.c_char_p,   # inputDir
        ctypes.c_char_p,   # outputDir
        PROGRESS_CALLBACK, # callback
    ]
    _dll.EnhancedVolume_ProcessFolder.restype = ctypes.c_int

    _dll.EnhancedVolume_ProcessSliceRange.argtypes = [
        ctypes.c_char_p,   # inputDir
        ctypes.c_char_p,   # outputDir
        ctypes.c_int,      # startSlice
        ctypes.c_int,      # endSlice
        PROGRESS_CALLBACK, # callback
    ]
    _dll.EnhancedVolume_ProcessSliceRange.restype = ctypes.c_int

    _dll.EnhancedVolume_Dispose.argtypes = []
    _dll.EnhancedVolume_Dispose.restype = None

    # Optional diagnostics (v0.0.0+)
    if hasattr(_dll, "EnhancedVolume_GetGpuInfo"):
        _dll.EnhancedVolume_GetGpuInfo.argtypes = []
        _dll.EnhancedVolume_GetGpuInfo.restype = ctypes.c_char_p
    if hasattr(_dll, "EnhancedVolume_GetActiveProvider"):
        _dll.EnhancedVolume_GetActiveProvider.argtypes = []
        _dll.EnhancedVolume_GetActiveProvider.restype = ctypes.c_char_p

    try:
        from inno3d.core.dll_profiling import _bind_optional
        _bind_optional(
            _dll,
            "EnhancedVolume_SetProfiling",
            "EnhancedVolume_GetProfiling",
            "EnhancedVolume_GetLastTiming",
        )
    except Exception:
        pass


def set_profiling(enable: bool) -> bool:
    from inno3d.core.dll_profiling import set_profiling as _sp
    return _sp(_dll, enable, "EnhancedVolume_SetProfiling")


def get_last_timing():
    from inno3d.core.dll_profiling import get_last_timing as _gt
    return _gt(_dll, "EnhancedVolume_GetLastTiming")


# ============================================
# PYTHON API
# ============================================
def _decode_c_str(raw) -> str:
    """Decode ANSI/UTF-8 C strings from the DLL safely."""
    if not raw:
        return ""
    if isinstance(raw, str):
        return raw
    for enc in ("utf-8", "mbcs", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def get_version() -> str:
    _check_dll()
    return _decode_c_str(_dll.EnhancedVolume_GetVersion())


def get_last_error() -> str:
    _check_dll()
    return _decode_c_str(_dll.EnhancedVolume_GetLastError())


def get_gpu_info() -> str:
    _check_dll()
    if not hasattr(_dll, "EnhancedVolume_GetGpuInfo"):
        return ""
    return _decode_c_str(_dll.EnhancedVolume_GetGpuInfo())


def get_active_provider() -> str:
    _check_dll()
    if not hasattr(_dll, "EnhancedVolume_GetActiveProvider"):
        return ""
    return _decode_c_str(_dll.EnhancedVolume_GetActiveProvider())


def is_initialized() -> bool:
    _check_dll()
    return bool(_dll.EnhancedVolume_IsInitialized())


def init(model_path: str, trt_cache_path: str = "", use_gpu: bool = True, gpu_device_id: int = 0) -> bool:
    """
    Initialize the ONNX model session.
    Returns True on success. On failure, call get_last_error() for details.
    """
    _check_dll()
    ok = _dll.EnhancedVolume_Init(
        model_path.encode("utf-8"),
        trt_cache_path.encode("utf-8") if trt_cache_path else b"",
        1 if use_gpu else 0,
        gpu_device_id,
    )
    return bool(ok)


def warm_up(height: int = 1200, width: int = 1200) -> bool:
    """Run a dummy inference to warm up TensorRT/CUDA engine."""
    _check_dll()
    return bool(_dll.EnhancedVolume_WarmUp(height, width))


def process_folder(input_dir: str, output_dir: str, callback=None) -> int:
    """
    Enhance all TIFF files in input_dir, save to output_dir.
    Returns number of files processed, or -1 on error.
    """
    _check_dll()
    cb = callback if callback else _progress_cb
    return _dll.EnhancedVolume_ProcessFolder(
        input_dir.encode("utf-8"),
        output_dir.encode("utf-8"),
        cb,
    )


def process_slice_range(
    input_dir: str, output_dir: str,
    start_slice: int, end_slice: int,
    callback=None
) -> int:
    """
    Enhance a specific range of TIFF files (0-based index).
    Returns number of files processed, or -1 on error.
    """
    _check_dll()
    cb = callback if callback else _progress_cb
    return _dll.EnhancedVolume_ProcessSliceRange(
        input_dir.encode("utf-8"),
        output_dir.encode("utf-8"),
        start_slice,
        end_slice,
        cb,
    )


def has_process_volume_buffer() -> bool:
    """True when DLL exports EnhancedVolume_ProcessVolumeBuffer (v0.0.1+)."""
    _check_dll()
    return hasattr(_dll, "EnhancedVolume_ProcessVolumeBuffer")


def process_volume_buffer(
    input_u16,
    output_u16=None,
    start_slice: int = -1,
    end_slice: int = -1,
    callback=None,
) -> int:
    """Enhance a contiguous uint16 volume in RAM (ZYX C-order).

    Requires BumpVoid_ISP_ENH.dll >= 0.0.1. Avoids temp TIFF slice export.

    Parameters
    ----------
    input_u16 : np.ndarray
        Shape (Z, Y, X), dtype uint16, C-contiguous preferred.
    output_u16 : np.ndarray or None
        Same shape/dtype. If None, an empty_like array is allocated and
        returned via ``process_volume_buffer.result`` (see return note).
        For in-place, pass the same array as ``input_u16``.
    start_slice, end_slice : int
        Half-open [start, end); -1,-1 = all Z.
    callback : optional ctypes progress callback

    Returns
    -------
    int
        Number of slices enhanced, or -1 on error.

    Notes
    -----
    The enhanced array is always written into ``output_u16`` (or the
    allocated buffer). Callers should keep a reference to that array.
    """
    import numpy as np

    _check_dll()
    if not has_process_volume_buffer():
        raise RuntimeError(
            "EnhancedVolume_ProcessVolumeBuffer not in DLL "
            "(need BumpVoid_ISP_ENH >= 0.0.1). Use process_folder instead."
        )

    src = np.asarray(input_u16)
    if src.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape={src.shape}")
    if src.dtype != np.uint16:
        src = np.ascontiguousarray(src, dtype=np.uint16)
    else:
        src = np.ascontiguousarray(src)

    if output_u16 is None:
        dst = np.empty_like(src)
    else:
        dst = np.asarray(output_u16)
        if dst.shape != src.shape:
            raise ValueError(f"output shape {dst.shape} != input {src.shape}")
        if dst.dtype != np.uint16:
            raise ValueError("output must be uint16")
        dst = np.ascontiguousarray(dst)

    depth, height, width = [int(v) for v in src.shape]
    # Keep references so GC cannot free buffers while native code runs
    process_volume_buffer._pin_src = src
    process_volume_buffer._pin_dst = dst
    process_volume_buffer.result = dst

    fn = _dll.EnhancedVolume_ProcessVolumeBuffer
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        PROGRESS_CALLBACK,
    ]
    fn.restype = ctypes.c_int

    cb = callback if callback else _progress_cb
    return int(
        fn(
            src.ctypes.data_as(ctypes.c_void_p),
            dst.ctypes.data_as(ctypes.c_void_p),
            depth,
            height,
            width,
            int(start_slice),
            int(end_slice),
            cb,
        )
    )


def dispose():
    """Release the ONNX session and free GPU memory."""
    _check_dll()
    _dll.EnhancedVolume_Dispose()


def make_progress_callback(func):
    """
    Create a ctypes-compatible progress callback from a Python function.
    func signature: func(message: str, current: int, total: int)
    """
    def wrapper(message, current, total):
        msg = message.decode("utf-8") if isinstance(message, bytes) else message
        func(msg, current, total)
    # Must keep a reference to prevent garbage collection
    cb = PROGRESS_CALLBACK(wrapper)
    cb._prevent_gc = wrapper
    return cb
