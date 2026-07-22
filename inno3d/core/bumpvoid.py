"""
BumpVoid SEG Python Wrapper — loads BumpVoidSeg.dll (preferred) or legacy BumpVoidDLL.dll.

C API symbols remain BumpVoid_* (ABI stable). Product binary name: BumpVoidSeg.dll
"""

import ctypes
import ctypes.wintypes as wintypes
import os
import sys
import threading
import time

# ============================================
# LOAD DLL
# ============================================
_dll = None
_dll_dir = None
_dll_path = None

# Prefer new ISP name; keep legacy package name for outsource / old V2 drops
_SEG_DLL_NAMES = ("BumpVoidSeg.dll", "BumpVoidDLL.dll")
_SEG_DLL_NAME = _SEG_DLL_NAMES[0]


def _pick_seg_dll_path(dll_dir: str):
    """Return first existing SEG binary path in dll_dir, or None."""
    for name in _SEG_DLL_NAMES:
        path = os.path.join(dll_dir, name)
        if os.path.isfile(path):
            return path
    return None


def load_dll(dll_dir):
    """
    Load BumpVoidSeg.dll (or legacy BumpVoidDLL.dll) from dll_dir.
    """
    global _dll, _dll_dir, _dll_path

    dll_dir = os.path.abspath(dll_dir)
    if not os.path.isdir(dll_dir):
        raise FileNotFoundError(f"Directory not found: {dll_dir}")

    dll_path = _pick_seg_dll_path(dll_dir)
    if not dll_path:
        raise FileNotFoundError(
            f"Neither {' nor '.join(_SEG_DLL_NAMES)} found in: {dll_dir}\n"
            "Copy the full V2 folder next to Inno3D.exe (or browse to a complete V2 package)."
        )

    dll_name = os.path.basename(dll_path)

    # Add directory to DLL search path (Python 3.8+ on Windows)
    # This is required because Python 3.8+ ignores the system PATH for DLL resolution.
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(dll_dir)
        except Exception as e:
            print(f"[bumpvoid] Warning: Failed to add DLL directory {dll_dir}: {e}")

        cuda_paths = []
        for key in os.environ:
            if key.startswith("CUDA_PATH"):
                path = os.environ.get(key)
                if path and os.path.isdir(path):
                    cuda_bin = os.path.join(path, "bin")
                    if os.path.isdir(cuda_bin) and cuda_bin not in cuda_paths:
                        cuda_paths.append(cuda_bin)

        for cuda_bin in cuda_paths:
            try:
                os.add_dll_directory(cuda_bin)
                print(f"[bumpvoid] Added CUDA DLL directory to search path: {cuda_bin}")
            except Exception as e:
                print(f"[bumpvoid] Warning: Failed to add CUDA DLL directory {cuda_bin}: {e}")

    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

    try:
        _dll = ctypes.CDLL(dll_path)
        _dll_dir = dll_dir
        _dll_path = dll_path
        _setup_functions()
        try:
            ver = _dll.BumpVoid_GetVersion()
            if ver:
                print(f"[bumpvoid] Loaded {dll_name} version={ver.decode('utf-8', errors='replace')}")
            else:
                print(f"[bumpvoid] Loaded {dll_name}")
        except Exception:
            print(f"[bumpvoid] Loaded {dll_name}")
    except OSError as e:
        error_msg = f"Failed to load {dll_name}: {e}\n"
        err_l = str(e).lower()
        if (
            "could not find module" in str(e)
            or "specified module could not be found" in err_l
            or "was not found when the application was frozen" in err_l
            or "dynlib/dll" in err_l
        ):
            error_msg += (
                "\nTIP: Native dependency missing next to the DLL (OpenCV / CUDA runtime).\n"
                f"Ensure the full V2 package is present (opencv_world4110.dll, CUDA/NPP DLLs) in:\n"
                f"  {dll_dir}\n"
                "For packaged builds, rebuild with build_final.bat so dist\\Inno3D\\V2 is copied.\n"
            )
        raise OSError(error_msg)


def _check_dll():
    if _dll is None:
        raise RuntimeError(
            "DLL not loaded! Call bumpvoid.load_dll('path/to/dll/folder') first.\n"
            f"The folder must contain {' or '.join(_SEG_DLL_NAMES)} and dependencies (OpenCV, etc.)."
        )

# ============================================
# STRUCTURES
# ============================================
class BumpVoidConfig(ctypes.Structure):
    _fields_ = [
        ("inputPath",           ctypes.c_char * 512),
        ("outputDir",           ctypes.c_char * 512),
        ("testName",            ctypes.c_char * 128),
        ("bumpBgOpenX",         ctypes.c_double),
        ("bumpBgOpenY",         ctypes.c_double),
        ("bumpBgOpenZ",         ctypes.c_double),
        ("bumpThresholdWeight", ctypes.c_double),
        ("bumpCleanOpenX",      ctypes.c_double),
        ("bumpCleanOpenY",      ctypes.c_double),
        ("bumpCleanOpenZ",      ctypes.c_double),
        ("voidThresholdWeight", ctypes.c_double),
        ("openEmVoidX",         ctypes.c_double),
        ("openEmVoidY",         ctypes.c_double),
        ("openEmVoidZ",         ctypes.c_double),
        ("closeResidueX",       ctypes.c_double),
        ("closeResidueY",       ctypes.c_double),
        ("closeResidueZ",       ctypes.c_double),
        ("openSmallDotsX",      ctypes.c_double),
        ("openSmallDotsY",      ctypes.c_double),
        ("openSmallDotsZ",      ctypes.c_double),
        ("erodeBumpX",          ctypes.c_double),
        ("erodeBumpY",          ctypes.c_double),
        ("erodeBumpZ",          ctypes.c_double),
        ("blankStart1",         ctypes.c_int),
        ("blankEnd1",           ctypes.c_int),
        ("blankStart2",         ctypes.c_int),
        ("blankEnd2",           ctypes.c_int),
        ("saveBumpIntermediate",ctypes.c_int),
        ("saveVoidIntermediate",ctypes.c_int),
        ("showResult",          ctypes.c_int),
    ]

class BumpVoidResult(ctypes.Structure):
    _fields_ = [
        ("success",        ctypes.c_int),
        ("errorMessage",   ctypes.c_char * 512),
        ("bumpOutputPath", ctypes.c_char * 512),
        ("voidOutputPath", ctypes.c_char * 512),
        ("bumpTime",       ctypes.c_double),
        ("voidTime",       ctypes.c_double),
        ("totalTime",      ctypes.c_double),
    ]

PROGRESS_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_int)

# ============================================
# WINDOWS EVENT API
# ============================================
_kernel32 = ctypes.windll.kernel32

_kernel32.CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
_kernel32.CreateEventW.restype = wintypes.HANDLE
_kernel32.SetEvent.argtypes = [wintypes.HANDLE]
_kernel32.SetEvent.restype = wintypes.BOOL
_kernel32.ResetEvent.argtypes = [wintypes.HANDLE]
_kernel32.ResetEvent.restype = wintypes.BOOL
_kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_kernel32.WaitForSingleObject.restype = wintypes.DWORD
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.CloseHandle.restype = wintypes.BOOL

INFINITE = 0xFFFFFFFF
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102

# ============================================
# SETUP DLL FUNCTION SIGNATURES
# ============================================
def _setup_functions():
    _dll.BumpVoid_InitConfig.argtypes = [ctypes.POINTER(BumpVoidConfig)]
    _dll.BumpVoid_InitConfig.restype = None

    _dll.BumpVoid_LoadConfig.argtypes = [ctypes.c_char_p, ctypes.POINTER(BumpVoidConfig)]
    _dll.BumpVoid_LoadConfig.restype = ctypes.c_int

    _dll.BumpVoid_Process.argtypes = [
        ctypes.POINTER(BumpVoidConfig), ctypes.POINTER(BumpVoidResult),
        PROGRESS_CALLBACK, wintypes.HANDLE
    ]
    _dll.BumpVoid_Process.restype = ctypes.c_int

    _dll.BumpVoid_ProcessBumpOnly.argtypes = [
        ctypes.POINTER(BumpVoidConfig), ctypes.POINTER(BumpVoidResult),
        PROGRESS_CALLBACK, wintypes.HANDLE
    ]
    _dll.BumpVoid_ProcessBumpOnly.restype = ctypes.c_int

    _dll.BumpVoid_ProcessVoidOnly.argtypes = [
        ctypes.POINTER(BumpVoidConfig), ctypes.c_char_p,
        ctypes.POINTER(BumpVoidResult), PROGRESS_CALLBACK, wintypes.HANDLE
    ]
    _dll.BumpVoid_ProcessVoidOnly.restype = ctypes.c_int

    _dll.BumpVoid_GetVersion.argtypes = []
    _dll.BumpVoid_GetVersion.restype = ctypes.c_char_p

    _dll.BumpVoid_GetThreadCount.argtypes = []
    _dll.BumpVoid_GetThreadCount.restype = ctypes.c_int

    _dll.BumpVoid_SetThreadCount.argtypes = [ctypes.c_int]
    _dll.BumpVoid_SetThreadCount.restype = None
    
    _dll.BumpVoid_HasCUDA.argtypes = []
    _dll.BumpVoid_HasCUDA.restype = ctypes.c_int

    _dll.BumpVoid_GetGPUMemoryMB.argtypes = []
    _dll.BumpVoid_GetGPUMemoryMB.restype = ctypes.c_int

    # CC3D — GPU Connected Component 3D Labeling
    try:
        _dll.BumpVoid_CC3D.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),  # bumpMask (uint8*)
            ctypes.c_int,                     # depth
            ctypes.c_int,                     # height
            ctypes.c_int,                     # width
            ctypes.c_int,                     # connectivity
            ctypes.POINTER(ctypes.c_uint),    # outLabels (uint32*)
            ctypes.c_int,                     # minVoxels
        ]
        _dll.BumpVoid_CC3D.restype = ctypes.c_int
    except AttributeError:
        pass  # DLL built without CC3D support

# ============================================
# WRAPPER FUNCTIONS
# ============================================
def get_version():
    _check_dll()
    return _dll.BumpVoid_GetVersion().decode("utf-8")


def get_dll_name():
    if _dll_path:
        return os.path.basename(_dll_path)
    return _SEG_DLL_NAME


def get_dll_path():
    """Absolute path of loaded SEG DLL, or None."""
    return _dll_path


def get_dll_dir():
    return _dll_dir


def get_product_info():
    """Return dict with name/version/path for Online logging."""
    info = {
        "name": _SEG_DLL_NAME,
        "module": "SEG",
        "version": None,
        "path": _dll_path,
        "dir": _dll_dir,
    }
    if _dll is not None:
        try:
            info["version"] = get_version()
        except Exception:
            pass
        try:
            if hasattr(_dll, "BumpVoid_GetProductName"):
                _dll.BumpVoid_GetProductName.restype = ctypes.c_char_p
                p = _dll.BumpVoid_GetProductName()
                if p:
                    info["product"] = p.decode("utf-8", errors="replace")
        except Exception:
            pass
    return info


def get_thread_count():
    _check_dll()
    return _dll.BumpVoid_GetThreadCount()

def set_thread_count(n):
    _check_dll()
    _dll.BumpVoid_SetThreadCount(n)

def has_cuda():
    _check_dll()
    return bool(_dll.BumpVoid_HasCUDA())

def get_gpu_memory_mb():
    _check_dll()
    return _dll.BumpVoid_GetGPUMemoryMB()

def init_config():
    _check_dll()
    config = BumpVoidConfig()
    _dll.BumpVoid_InitConfig(ctypes.byref(config))
    return config

def load_config(config_path):
    _check_dll()
    config = BumpVoidConfig()
    path_bytes = config_path.encode("utf-8") if isinstance(config_path, str) else config_path
    ok = _dll.BumpVoid_LoadConfig(path_bytes, ctypes.byref(config))
    if not ok:
        print(f"Warning: Could not load '{config_path}', using defaults")
        _dll.BumpVoid_InitConfig(ctypes.byref(config))
    return config

def set_config_paths(config, input_path=None, output_dir=None, test_name=None):
    if input_path:
        config.inputPath = input_path.encode("utf-8") if isinstance(input_path, str) else input_path
    if output_dir:
        config.outputDir = output_dir.encode("utf-8") if isinstance(output_dir, str) else output_dir
    if test_name:
        config.testName = test_name.encode("utf-8") if isinstance(test_name, str) else test_name

def print_config(config):
    print(f"  Input:  {config.inputPath.decode()}")
    print(f"  Output: {config.outputDir.decode()}")
    print(f"  Test:   {config.testName.decode()}")
    print(f"  Bump Threshold Weight: {config.bumpThresholdWeight}")
    print(f"  Void Threshold Weight: {config.voidThresholdWeight}")

def print_result(result):
    if result.success:
        print(f"  Status:     SUCCESS")
        print(f"  Bump Time:  {result.bumpTime:.2f} sec")
        print(f"  Void Time:  {result.voidTime:.2f} sec")
        print(f"  Total Time: {result.totalTime:.2f} sec")
        print(f"  Bump File:  {result.bumpOutputPath.decode()}")
        print(f"  Void File:  {result.voidOutputPath.decode()}")
    else:
        print(f"  Status: FAILED")
        print(f"  Error:  {result.errorMessage.decode()}")

# Default progress callback
def _default_progress(message, progress, total):
    msg = message.decode("utf-8") if isinstance(message, bytes) else message
    print(f"  [{progress}/{total}] {msg}")

_progress_cb = PROGRESS_CALLBACK(_default_progress)

def make_progress_callback(func):
    def wrapper(message, progress, total):
        msg = message.decode("utf-8") if isinstance(message, bytes) else message
        func(msg, progress, total)
    return PROGRESS_CALLBACK(wrapper)

# ============================================
# SIMPLE API (no Event)
# ============================================
def process(config, callback=None):
    _check_dll()
    result = BumpVoidResult()
    cb = callback if callback else _progress_cb
    _dll.BumpVoid_Process(ctypes.byref(config), ctypes.byref(result), cb, None)
    return result

def process_bump_only(config, callback=None):
    _check_dll()
    result = BumpVoidResult()
    cb = callback if callback else _progress_cb
    _dll.BumpVoid_ProcessBumpOnly(ctypes.byref(config), ctypes.byref(result), cb, None)
    return result

def process_void_only(config, bump_path, callback=None):
    _check_dll()
    result = BumpVoidResult()
    cb = callback if callback else _progress_cb
    path_bytes = bump_path.encode("utf-8") if isinstance(bump_path, str) else bump_path
    _dll.BumpVoid_ProcessVoidOnly(
        ctypes.byref(config), path_bytes, ctypes.byref(result), cb, None
    )
    return result

# ============================================
# EVENT API
# ============================================
def create_event():
    handle = _kernel32.CreateEventW(None, True, False, None)
    if not handle:
        raise RuntimeError(f"CreateEvent failed! Error: {ctypes.GetLastError()}")
    return handle

def reset_event(handle):
    _kernel32.ResetEvent(handle)

def wait_event(handle, timeout_ms=INFINITE):
    result = _kernel32.WaitForSingleObject(handle, timeout_ms)
    if result == WAIT_OBJECT_0:
        return "signaled"
    elif result == WAIT_TIMEOUT:
        return "timeout"
    return "failed"

def close_event(handle):
    _kernel32.CloseHandle(handle)

def process_async(config, callback=None):
    _check_dll()
    hEvent = create_event()
    result = BumpVoidResult()
    cb = callback if callback else _progress_cb

    def worker():
        _dll.BumpVoid_Process(ctypes.byref(config), ctypes.byref(result), cb, hEvent)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return result, hEvent, t

# ============================================
# GPU CC3D — Connected Component 3D Labeling
# ============================================

def has_cc3d_gpu():
    """Check if the loaded DLL supports GPU CC3D."""
    _check_dll()
    return hasattr(_dll, 'BumpVoid_CC3D')

def cc3d_gpu(binary_volume, connectivity=26, min_voxels=0):
    """
    Run GPU-accelerated 3D Connected Component Labeling.
    
    Args:
        binary_volume: numpy uint8 array, shape (D, H, W), values 0 or 255
        connectivity: 6 or 26 (default 26)
        min_voxels: filter out components smaller than this (0 = no filter)
    
    Returns:
        (labels, num_labels):
            labels: numpy uint32 array, same shape as input, 1-based labels
            num_labels: int, number of unique labels found
    """
    import numpy as np
    _check_dll()
    
    if not hasattr(_dll, 'BumpVoid_CC3D'):
        raise RuntimeError("DLL does not support CC3D. Rebuild with USE_CUDA and cuda_cc3d.cu")
    
    if binary_volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {binary_volume.shape}")
    
    D, H, W = binary_volume.shape
    
    # Ensure contiguous uint8
    binary_volume = np.ascontiguousarray(binary_volume, dtype=np.uint8)
    
    # Allocate output
    out_labels = np.zeros((D, H, W), dtype=np.uint32)
    
    # Get ctypes pointers
    mask_ptr = binary_volume.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    labels_ptr = out_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
    
    num_labels = _dll.BumpVoid_CC3D(
        mask_ptr, D, H, W,
        connectivity,
        labels_ptr,
        min_voxels
    )
    
    if num_labels < 0:
        error_codes = {-1: "No CUDA GPU available", -2: "Insufficient VRAM", -3: "CUDA malloc failed"}
        raise RuntimeError(f"BumpVoid_CC3D failed: {error_codes.get(num_labels, f'error code {num_labels}')}")
    
    return out_labels, num_labels

