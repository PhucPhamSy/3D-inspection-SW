# inno3d/native/__init__.py
# -----------------------------------------------------------------------
# Native DLL façade (Phase 2)
#
# Centralizes all ctypes.CDLL loading so the rest of the codebase
# never calls ctypes directly.
#
# Loader-level (convenience one-shot loaders):
#   from inno3d.native import load_bumpvoid_dll, load_mes_dll, ...
#
# Semantic submodule façades (full public API per DLL):
#   from inno3d.native import seg, mes, b2b, enhance
#   from inno3d.native.seg import load_dll, BumpVoidConfig, process
#   from inno3d.native.mes import load_dll, MesConfig, process_memory
#   from inno3d.native.b2b import load_dll, run_boundary_analysis
#   from inno3d.native.enhance import load_dll, process_folder
#
# Benefits:
#   - Single DLL search path logic
#   - Easy to swap real vs mock DLL in tests
#   - Centralized error handling + logging
# -----------------------------------------------------------------------
"""inno3d.native — DLL loading façade for all native binaries."""

from inno3d.native.loader import (
    DllLoadError,
    get_dll_search_paths,
    load_b2b_dll,
    load_bumpvoid_dll,
    load_enhanced_dll,
    load_mes_dll,
)

# Semantic submodule façades — loaded lazily so that unit tests can still
# mock inno3d.core.* via patch.dict('sys.modules') before first use.
# Access via: `import inno3d.native.seg` or `from inno3d.native import seg`.
_SUBMODULES = {"seg", "mes", "b2b", "enhance"}


def __getattr__(name: str):
    if name in _SUBMODULES:
        import importlib
        mod = importlib.import_module(f"inno3d.native.{name}")
        globals()[name] = mod  # cache for subsequent accesses
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Convenience loader functions (loader.py)
    "load_bumpvoid_dll",
    "load_enhanced_dll",
    "load_mes_dll",
    "load_b2b_dll",
    "get_dll_search_paths",
    "DllLoadError",
    # Semantic submodule façades (lazy)
    "seg",
    "mes",
    "b2b",
    "enhance",
]

