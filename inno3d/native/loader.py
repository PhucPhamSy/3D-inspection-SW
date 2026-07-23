# inno3d/native/loader.py
# -----------------------------------------------------------------------
# Native DLL loader façade (Phase 2)
#
# Central DLL loading hub — delegates to core/* wrappers.
# Call sites import from inno3d.native instead of core/* directly.
#
# Usage:
#   from inno3d.native import load_bumpvoid_dll, load_enhanced_dll
#   dll = load_bumpvoid_dll("/path/to/dll/folder")
# -----------------------------------------------------------------------
"""Centralized DLL loading façade for all native binaries."""

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class DllLoadError(RuntimeError):
    """Raised when a native DLL cannot be found or loaded."""


def get_dll_search_paths(dll_dir: str | None = None) -> list[Path]:
    """Return candidate DLL search directories in priority order.

    Priority:
      1. Explicit dll_dir argument
      2. Directory next to main executable (frozen PyInstaller build)
      3. Project root V2 folder (dev mode)
    """
    import sys
    candidates = []

    if dll_dir:
        candidates.append(Path(dll_dir).resolve())

    # PyInstaller frozen: executable sits next to DLL folder
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir)
        candidates.append(exe_dir / "V2")

    # Dev mode: project root / V2
    project_root = Path(__file__).resolve().parents[2]
    candidates.append(project_root / "V2")
    candidates.append(project_root)

    return [p for p in candidates if p.is_dir()]


def load_bumpvoid_dll(dll_dir: str):
    """Load BumpVoidSeg.dll (or legacy BumpVoidDLL.dll) from dll_dir.

    Delegates to inno3d.core.bumpvoid.load_dll which handles:
      - DLL search path registration (os.add_dll_directory)
      - CUDA path detection
      - ctypes function signature setup

    Returns:
        The loaded ctypes DLL handle (via bumpvoid._dll).

    Raises:
        DllLoadError: if the DLL cannot be found or loaded.
    """
    try:
        from inno3d.core import bumpvoid
        bumpvoid.load_dll(dll_dir)
        log.info("BumpVoid DLL loaded from %s", dll_dir)
        return bumpvoid._dll
    except FileNotFoundError as e:
        raise DllLoadError(str(e)) from e
    except OSError as e:
        raise DllLoadError(f"OS error loading BumpVoid DLL: {e}") from e


def load_enhanced_dll(dll_path: str):
    """Load EnhancedVolume DLL (ONNX runtime helper) from dll_path.

    Delegates to inno3d.core.enhanced_volume.load_dll.

    Returns:
        The loaded ctypes DLL handle.

    Raises:
        DllLoadError: if the DLL cannot be found or loaded.
    """
    try:
        from inno3d.core import enhanced_volume
        enhanced_volume.load_dll(dll_path)
        log.info("EnhancedVolume DLL loaded from %s", dll_path)
        return enhanced_volume._dll
    except FileNotFoundError as e:
        raise DllLoadError(str(e)) from e
    except OSError as e:
        raise DllLoadError(f"OS error loading EnhancedVolume DLL: {e}") from e
    except AttributeError:
        # enhanced_volume may not have load_dll — import directly
        import ctypes
        try:
            dll = ctypes.CDLL(dll_path)
            log.info("Loaded DLL (direct ctypes): %s", dll_path)
            return dll
        except OSError as e:
            raise DllLoadError(f"Failed to load {dll_path}: {e}") from e


def load_mes_dll(dll_dir: str):
    """Load BumpVoidMes.dll from dll_dir.

    Delegates to inno3d.core.bumpvoid_mes.load_dll which handles
    CUDA path detection and ctypes function signature setup.

    Returns:
        True if loaded successfully (bumpvoid_mes tracks state internally).

    Raises:
        DllLoadError: if the DLL cannot be found or loaded.
    """
    try:
        from inno3d.core import bumpvoid_mes
        bumpvoid_mes.load_dll(dll_dir)
        log.info("BumpVoidMes DLL loaded from %s", dll_dir)
        return True
    except FileNotFoundError as e:
        raise DllLoadError(str(e)) from e
    except OSError as e:
        raise DllLoadError(f"OS error loading BumpVoidMes DLL: {e}") from e


def load_b2b_dll(dll_dir: str):
    """Load BoundaryGPU.dll (or legacy B2B DLL) from dll_dir.

    Delegates to inno3d.core.bumpvoid_b2b.load_dll.
    Call sites: Online controller._ensure_b2b_loaded, Teaching boundary thread.

    Returns:
        True if a gap-capable DLL was loaded; False if legacy (no gap support).

    Raises:
        DllLoadError: if no B2B DLL can be found or loaded.
    """
    try:
        from inno3d.core import bumpvoid_b2b
        bumpvoid_b2b.load_dll(dll_dir)
        gap_capable = bumpvoid_b2b.is_gap_capable()
        log.info(
            "B2B DLL loaded from %s (gap_capable=%s)", dll_dir, gap_capable
        )
        return gap_capable
    except FileNotFoundError as e:
        raise DllLoadError(str(e)) from e
    except OSError as e:
        raise DllLoadError(f"OS error loading B2B DLL: {e}") from e
