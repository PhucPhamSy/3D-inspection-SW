"""
BumpVoid B2B Python wrapper — BoundaryGPU.dll / BumpVoidB2B.dll
(Bump-to-Bump directional surface gap measurement after SEG + MES).

Primary output (golden / Teaching):
  boundary_gap.csv
    Src_row,Src_col,Direction,Dst_row,Dst_col,
    Min_dist_X_um,Min_dist_Y_um,Min_dist_Z_um,Min_dist_Euclidean_um,
    Src_voxel_Z,Src_voxel_Y,Src_voxel_X,Dst_voxel_Z,Dst_voxel_Y,Dst_voxel_X

Legacy (old summary-only rebuilds):
  boundary_summary.csv — min gap per bump without direction (not preferred).
"""

from __future__ import annotations

import csv
import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_dll = None
_dll_dir = None
_dll_path = None
_dll_gap_capable: Optional[bool] = None
_DLL_CANDIDATES = ("BoundaryGPU.dll", "BumpVoidB2B.dll")


def _add_search_dirs(dll_dir: str) -> None:
    if sys.platform != "win32":
        return
    dirs = [dll_dir]
    for key, val in os.environ.items():
        if key.startswith("CUDA_PATH") and val:
            binp = os.path.join(val, "bin")
            if os.path.isdir(binp):
                dirs.append(binp)
    for extra in (
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
    ):
        if os.path.isdir(extra) and extra not in dirs:
            dirs.append(extra)
    if hasattr(os, "add_dll_directory"):
        for d in dirs:
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
    os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")


def _probe_csv_capability(path: str) -> str:
    """Return 'gap' | 'summary' | 'unknown' by scanning PE strings (no load)."""
    try:
        with open(path, "rb") as f:
            blob = f.read(min(os.path.getsize(path), 2 * 1024 * 1024))
    except OSError:
        return "unknown"
    has_gap = b"boundary_gap" in blob or b"Src_row" in blob and b"Direction" in blob
    has_sum = b"boundary_summary" in blob or b"min_gap_euclidean" in blob
    if has_gap and not has_sum:
        return "gap"
    if has_gap:
        return "gap"
    if has_sum:
        return "summary"
    return "unknown"


def is_gap_capable_file(path: str) -> bool:
    return _probe_csv_capability(path) == "gap"


def _pick_dll_path(dll_dir: str) -> Optional[str]:
    """
    Prefer golden gap-capable BoundaryGPU.dll (Src→Dst / boundary_gap.csv).

    Ranking:
      1) writes boundary_gap.csv (not legacy summary-only ~73KB)
      2) app contract name BoundaryGPU.dll over BumpVoidB2B.dll
      3) larger PE (older golden ~345KB > ISP 81KB > broken 73KB)
      4) non-.new over .new when otherwise equal
    """
    dll_dir = os.path.abspath(dll_dir)
    candidates: List[str] = []
    for name in _DLL_CANDIDATES:
        p = os.path.join(dll_dir, name)
        if os.path.isfile(p):
            candidates.append(p)
        p_new = p + ".new"
        if os.path.isfile(p_new):
            candidates.append(p_new)
    # Optional ship-side golden backup (not loaded unless present)
    golden = os.path.join(dll_dir, "BoundaryGPU.dll.golden")
    if os.path.isfile(golden):
        candidates.append(golden)

    if not candidates:
        return None

    def _score(path: str) -> Tuple[int, int, int, int, int]:
        cap = _probe_csv_capability(path)
        is_gap = 1 if cap == "gap" else 0
        base = os.path.basename(path).lower()
        # Prefer app contract name BoundaryGPU.dll over BumpVoidB2B / .golden backup
        is_contract = 1 if base in ("boundarygpu.dll", "boundarygpu.dll.new") else 0
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        is_new = 1 if path.endswith(".new") else 0
        # Higher is better for all except is_new
        return (is_gap, is_contract, size, -is_new, 0)

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def unload_dll() -> None:
    """Drop handle so the next load_dll can pick a newer PE (Windows may keep file lock)."""
    global _dll, _dll_dir, _dll_path, _dll_gap_capable
    _dll = None
    _dll_dir = None
    _dll_path = None
    _dll_gap_capable = None


def load_dll(dll_dir: str, force: bool = False) -> str:
    """Load BoundaryGPU.dll (prefer) or BumpVoidB2B.dll from folder. Returns version string.

    force=True re-picks the PE even if already loaded (use after deploying a new DLL).
    """
    global _dll, _dll_dir, _dll_path, _dll_gap_capable
    dll_dir = os.path.abspath(dll_dir)
    dll_path = _pick_dll_path(dll_dir)
    if not dll_path:
        raise FileNotFoundError(
            f"BoundaryGPU.dll / BumpVoidB2B.dll not found in: {dll_dir}"
        )
    if (
        not force
        and _dll is not None
        and _dll_path
        and os.path.normcase(os.path.abspath(_dll_path))
        == os.path.normcase(os.path.abspath(dll_path))
    ):
        return get_version()

    if force or _dll is not None:
        unload_dll()

    _add_search_dirs(dll_dir)
    _dll = ctypes.CDLL(dll_path)
    _dll_dir = dll_dir
    _dll_path = dll_path
    _dll_gap_capable = is_gap_capable_file(dll_path)
    _setup()
    if not _dll_gap_capable:
        print(
            f"[B2B WARN] Loaded summary-only DLL (no boundary_gap.csv): {dll_path}\n"
            f"           Deploy BumpVoid_ISP_B2B release BoundaryGPU.dll for "
            f"Src→Dst gap format (Teaching / before-B2B UI)."
        )
    else:
        print(f"[B2B] Gap-capable DLL loaded: {dll_path}")
    return get_version()


def _setup():
    _dll.RunBoundaryAnalysisCUDA.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    _dll.RunBoundaryAnalysisCUDA.restype = ctypes.c_int
    for name in ("B2B_GetVersion", "BoundaryGPU_GetVersion", "BumpVoidB2B_GetVersion"):
        if hasattr(_dll, name):
            getattr(_dll, name).restype = ctypes.c_char_p
    for name in ("B2B_GetProductName", "B2B_GetModuleCode"):
        if hasattr(_dll, name):
            getattr(_dll, name).restype = ctypes.c_char_p


def _check():
    if _dll is None:
        raise RuntimeError("B2B DLL not loaded. Call bumpvoid_b2b.load_dll(dll_dir) first.")


def is_loaded() -> bool:
    return _dll is not None


def is_gap_capable() -> bool:
    """True if the currently loaded PE writes boundary_gap.csv (Src→Dst)."""
    if _dll_path and _dll_gap_capable is not None:
        return bool(_dll_gap_capable)
    if _dll_path:
        return is_gap_capable_file(_dll_path)
    return False


def get_version() -> str:
    _check()
    for name in ("B2B_GetVersion", "BoundaryGPU_GetVersion", "BumpVoidB2B_GetVersion"):
        if hasattr(_dll, name):
            try:
                v = getattr(_dll, name)()
                if v:
                    return v.decode("utf-8", errors="replace")
            except Exception:
                pass
    return "?"


def get_dll_path() -> Optional[str]:
    return _dll_path


def get_product_info() -> Dict[str, Any]:
    info = {
        "name": os.path.basename(_dll_path) if _dll_path else "BoundaryGPU.dll",
        "module": "B2B",
        "product": "BumpVoidB2B",
        "version": None,
        "path": _dll_path,
        "dir": _dll_dir,
        "gap_capable": is_gap_capable() if _dll_path else None,
        "csv_format": (
            "boundary_gap" if is_gap_capable() else "boundary_summary"
        )
        if _dll_path
        else None,
    }
    if _dll is not None:
        info["version"] = get_version()
        try:
            if hasattr(_dll, "B2B_GetProductName"):
                p = _dll.B2B_GetProductName()
                if p:
                    info["product"] = p.decode("utf-8", errors="replace")
        except Exception:
            pass
    return info


def run_boundary_analysis(
    bump_mask: np.ndarray,
    labeled_data: np.ndarray,
    *,
    voxel_z: float,
    voxel_y: float,
    voxel_x: float,
    output_dir: str,
    object_stats: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """
    Run B2B on one volume (layer crop or full).
    Returns DLL code: 1=ok, 0=no boundary, <0=error.

    Golden DLL writes boundary_gap.csv (Src→Dst per direction).
    Legacy summary-only builds write boundary_summary.csv.
    """
    _check()
    os.makedirs(output_dir, exist_ok=True)
    bump = np.ascontiguousarray(bump_mask, dtype=np.uint8)
    labels = np.ascontiguousarray(labeled_data, dtype=np.uint32)
    if bump.shape != labels.shape or bump.ndim != 3:
        raise ValueError(f"shape mismatch bump{bump.shape} labels{labels.shape}")

    depth, height, width = bump.shape
    max_lbl = int(labels.max()) if labels.size else 0
    grid_rows = np.full(max_lbl + 1, -1, dtype=np.int32)
    grid_cols = np.full(max_lbl + 1, -1, dtype=np.int32)
    if object_stats:
        for stat in object_stats:
            try:
                lbl_val = int(stat.get("label", 0))
            except Exception:
                continue
            if 0 <= lbl_val <= max_lbl:
                gr, gc = stat.get("grid_row", None), stat.get("grid_col", None)
                try:
                    if gr is not None and gc is not None and gr != "?" and gc != "?":
                        # Pass MES grid indices as-is (unified 0-based top-left XY)
                        grid_rows[lbl_val] = int(gr)
                        grid_cols[lbl_val] = int(gc)
                except Exception:
                    pass

    out_buf = ctypes.create_string_buffer(output_dir.encode("utf-8"))
    out_ptr = ctypes.cast(out_buf, ctypes.c_void_p)
    rc = _dll.RunBoundaryAnalysisCUDA(
        bump.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        int(depth),
        int(height),
        int(width),
        float(voxel_z),
        float(voxel_y),
        float(voxel_x),
        out_ptr,
        grid_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        grid_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    return int(rc)


def find_layer_csv(output_dir: str) -> Optional[str]:
    """Prefer boundary_gap.csv; fall back to boundary_summary.csv."""
    gap = os.path.join(output_dir, "boundary_gap.csv")
    if os.path.isfile(gap):
        return gap
    summary = os.path.join(output_dir, "boundary_summary.csv")
    if os.path.isfile(summary):
        return summary
    return None


def _normalize_gap_row(r: Dict[str, Any], layer_name: str = "") -> Dict[str, Any]:
    """
    Normalize both golden gap format and Source_id/Dest_id variants into:
      Src_row, Src_col, Direction, Dst_row, Dst_col, Min_dist_*, voxels, Layer
    """
    item = dict(r)
    if layer_name:
        item["Layer"] = layer_name

    # Source_id / Dest_id style: R1C6
    if "Src_row" not in item or item.get("Src_row") in (None, ""):
        src = str(item.get("Source_id", "") or "")
        if src.startswith("R") and "C" in src:
            try:
                body = src[1:]
                rs, cs = body.split("C", 1)
                item["Src_row"] = int(rs)
                item["Src_col"] = int(cs)
            except Exception:
                pass
    if "Dst_row" not in item or item.get("Dst_row") in (None, ""):
        dst = str(item.get("Dest_id", "") or "")
        if dst.startswith("R") and "C" in dst:
            try:
                body = dst[1:]
                rs, cs = body.split("C", 1)
                item["Dst_row"] = int(rs)
                item["Dst_col"] = int(cs)
            except Exception:
                pass

    # Alias distance keys
    if "Min_dist_Euclidean_um" not in item and "min_gap_euclidean_um" in item:
        item["Min_dist_Euclidean_um"] = item["min_gap_euclidean_um"]
        item["Min_dist_X_um"] = item.get("min_gap_X_um", "")
        item["Min_dist_Y_um"] = item.get("min_gap_Y_um", "")
        item["Min_dist_Z_um"] = item.get("min_gap_Z_um", "")

    return item


def read_boundary_csv(csv_path: str, layer_name: str = "") -> List[Dict[str, Any]]:
    """Parse boundary_gap.csv or boundary_summary.csv into row dicts."""
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(csv_path):
        return rows
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(_normalize_gap_row(dict(r), layer_name=layer_name))
    return rows


# Backward-compatible alias
def read_boundary_summary(csv_path: str, layer_name: str = "") -> List[Dict[str, Any]]:
    return read_boundary_csv(csv_path, layer_name=layer_name)


def aggregate_layer_summaries(
    layer_csvs: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """
    layer_csvs: list of (layer_name, path_to_boundary_gap.csv | boundary_summary.csv)
    Returns combined rows with Layer column first.
    """
    all_rows: List[Dict[str, Any]] = []
    for layer_name, path in layer_csvs:
        all_rows.extend(read_boundary_csv(path, layer_name=layer_name))
    return all_rows


def write_combined_gap(path: str, rows: List[Dict[str, Any]]) -> str:
    """Write combined multi-layer B2B gap CSV (reference format + Layer)."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "#",
        "Layer",
        "Src_row",
        "Src_col",
        "Direction",
        "Dst_row",
        "Dst_col",
        "Min_dist_X_um",
        "Min_dist_Y_um",
        "Min_dist_Z_um",
        "Min_dist_Euclidean_um",
        "Src_voxel_Z",
        "Src_voxel_Y",
        "Src_voxel_X",
        "Dst_voxel_Z",
        "Dst_voxel_Y",
        "Dst_voxel_X",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(rows):
            out = {k: r.get(k, "") for k in fieldnames}
            out["#"] = i + 1
            w.writerow(out)
    return path


def write_combined_summary(path: str, rows: List[Dict[str, Any]]) -> str:
    """
    Write combined multi-layer B2B CSV.
    If rows look like gap format → write gap; else legacy summary.
    """
    if rows and ("Src_row" in rows[0] or "Direction" in rows[0]):
        # Prefer gap filename when caller still passes summary path
        if path.endswith("boundary_summary_all_layers.csv"):
            path = path.replace(
                "boundary_summary_all_layers.csv", "boundary_gap_all_layers.csv"
            )
        return write_combined_gap(path, rows)

    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "#",
        "Layer",
        "Bump_id",
        "label",
        "boundary_voxels",
        "min_gap_X_um",
        "min_gap_Y_um",
        "min_gap_Z_um",
        "min_gap_euclidean_um",
        "min_gap_voxel_Z",
        "min_gap_voxel_Y",
        "min_gap_voxel_X",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(rows):
            out = {k: r.get(k, "") for k in fieldnames}
            out["#"] = i + 1
            w.writerow(out)
    return path
