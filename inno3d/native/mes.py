# inno3d/native/mes.py
# -----------------------------------------------------------------------
# Native MES DLL façade (Phase 2)
#
# Thin re-export of inno3d.core.bumpvoid_mes public API.
# Callers: Online controller (_ensure_mes_loaded), Teaching seg_pipeline.
#
# Usage:
#   from inno3d.native.mes import load_dll, MesConfig, process_memory
# -----------------------------------------------------------------------
"""MES (BumpVoidMes) DLL façade — re-exports from inno3d.core.bumpvoid_mes."""

from inno3d.core.bumpvoid_mes import (
    # Structures
    MesConfig,
    MesResult,
    MesObject,
    MesProgressCallback,
    # Load / version
    load_dll,
    unload_dll,
    get_version,
    get_dll_name,
    get_dll_path,
    get_product_info,
    is_loaded,
    # Config
    init_config,
    # Processing (in-memory — main Online path)
    process_memory,
    mes_objects_to_stats,
    reindex_grid_top_left,
    # CSV helpers
    write_python_format_csv,
    # Debug / comparison
    compare_stats,
)

__all__ = [
    # Structures
    "MesConfig",
    "MesResult",
    "MesObject",
    "MesProgressCallback",
    # Load / info
    "load_dll",
    "unload_dll",
    "get_version",
    "get_dll_name",
    "get_dll_path",
    "get_product_info",
    "is_loaded",
    # Config
    "init_config",
    # Processing
    "process_memory",
    "mes_objects_to_stats",
    "reindex_grid_top_left",
    # CSV
    "write_python_format_csv",
    # Debug
    "compare_stats",
]
