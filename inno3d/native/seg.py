# inno3d/native/seg.py
# -----------------------------------------------------------------------
# Native SEG DLL façade (Phase 2)
#
# Thin re-export of inno3d.core.bumpvoid public API.
# Callers: Online controller, Teaching seg_pipeline, workers.
#
# Usage:
#   from inno3d.native.seg import load_dll, BumpVoidConfig, process
# -----------------------------------------------------------------------
"""SEG (BumpVoidSeg) DLL façade — re-exports from inno3d.core.bumpvoid."""

from inno3d.core.bumpvoid import (
    # Structures
    BumpVoidConfig,
    BumpVoidResult,
    PROGRESS_CALLBACK,
    # Load / version
    load_dll,
    unload_dll,
    get_version,
    get_dll_name,
    get_dll_path,
    get_dll_dir,
    get_product_info,
    # Hardware info
    has_cuda,
    get_gpu_memory_mb,
    has_cc3d_gpu,
    # Thread control
    get_thread_count,
    set_thread_count,
    # Config helpers
    init_config,
    load_config,
    set_config_paths,
    print_config,
    print_result,
    # Processing
    process,
    process_bump_only,
    process_void_only,
    make_progress_callback,
    process_async,
    # Async event helpers
    create_event,
    reset_event,
    wait_event,
    close_event,
    # 3D connected-component (CC3D GPU)
    cc3d_gpu,
)

# Module-level state aliases (read-only — mutated by load_dll inside bumpvoid)
from inno3d.core import bumpvoid as _bv


def get_dll_handle():
    """Return the live ctypes DLL handle (None if not yet loaded)."""
    return _bv._dll


__all__ = [
    # Structures
    "BumpVoidConfig",
    "BumpVoidResult",
    "PROGRESS_CALLBACK",
    # Load / info
    "load_dll",
    "unload_dll",
    "get_version",
    "get_dll_name",
    "get_dll_path",
    "get_dll_dir",
    "get_dll_handle",
    "get_product_info",
    # Hardware
    "has_cuda",
    "get_gpu_memory_mb",
    "has_cc3d_gpu",
    # Threads
    "get_thread_count",
    "set_thread_count",
    # Config
    "init_config",
    "load_config",
    "set_config_paths",
    "print_config",
    "print_result",
    # Processing
    "process",
    "process_bump_only",
    "process_void_only",
    "make_progress_callback",
    "process_async",
    # Events
    "create_event",
    "reset_event",
    "wait_event",
    "close_event",
    # CC3D
    "cc3d_gpu",
]
