# inno3d/native/enhance.py
# -----------------------------------------------------------------------
# Native Enhancement DLL façade (Phase 2)
#
# Thin re-export of inno3d.core.enhanced_volume public API.
# Callers: Online controller (enhance step), Teaching EnhancementThread.
#
# Usage:
#   from inno3d.native.enhance import load_dll, process_folder
# -----------------------------------------------------------------------
"""Enhancement (BumpVoid_ISP_ENH) DLL façade — re-exports from inno3d.core.enhanced_volume."""

from inno3d.core.enhanced_volume import (
    # Callback type
    PROGRESS_CALLBACK,
    # Load / version
    load_dll,
    get_version,
    get_last_error,
    # Hardware info
    get_gpu_info,
    get_active_provider,
    # Lifecycle
    is_initialized,
    init,
    warm_up,
    dispose,
    # Processing
    process_folder,
    process_slice_range,
    # Progress helper
    make_progress_callback,
)

__all__ = [
    # Callback
    "PROGRESS_CALLBACK",
    # Load / info
    "load_dll",
    "get_version",
    "get_last_error",
    # Hardware
    "get_gpu_info",
    "get_active_provider",
    # Lifecycle
    "is_initialized",
    "init",
    "warm_up",
    "dispose",
    # Processing
    "process_folder",
    "process_slice_range",
    # Progress
    "make_progress_callback",
]
