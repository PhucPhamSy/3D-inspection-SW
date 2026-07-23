# inno3d/native/b2b.py
# -----------------------------------------------------------------------
# Native B2B DLL façade (Phase 2)
#
# Thin re-export of inno3d.core.bumpvoid_b2b public API.
# Callers: Online controller (_ensure_b2b_loaded), Teaching boundary thread.
#
# Usage:
#   from inno3d.native.b2b import load_dll, run_boundary_analysis, is_gap_capable
# -----------------------------------------------------------------------
"""B2B (BoundaryGPU) DLL façade — re-exports from inno3d.core.bumpvoid_b2b."""

from inno3d.core.bumpvoid_b2b import (
    # Load / version
    load_dll,
    unload_dll,
    get_version,
    get_dll_path,
    get_product_info,
    is_loaded,
    # Gap-capability probe (used by Teaching B2B tab + controller)
    is_gap_capable,
    is_gap_capable_file,
    # Core analysis
    run_boundary_analysis,
    # CSV helpers — Teaching B2B table population
    find_layer_csv,
    read_boundary_csv,
    read_boundary_summary,
    aggregate_layer_summaries,
    write_combined_gap,
    write_combined_summary,
)

__all__ = [
    # Load / info
    "load_dll",
    "unload_dll",
    "get_version",
    "get_dll_path",
    "get_product_info",
    "is_loaded",
    # Gap capability
    "is_gap_capable",
    "is_gap_capable_file",
    # Analysis
    "run_boundary_analysis",
    # CSV
    "find_layer_csv",
    "read_boundary_csv",
    "read_boundary_summary",
    "aggregate_layer_summaries",
    "write_combined_gap",
    "write_combined_summary",
]
