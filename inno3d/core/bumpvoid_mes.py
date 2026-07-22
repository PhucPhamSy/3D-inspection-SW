"""
BumpVoidMes Python wrapper — loads BumpVoidMes.dll (measurement / object stats + FAR).

Pair with BumpVoidSeg for Online: SEG → masks → MES(+FAR) → object table / CSV → B2B.
"""

from __future__ import annotations

import ctypes
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_dll = None
_dll_dir = None
_dll_path = None
_MES_DLL_NAME = "BumpVoidMes.dll"


class MesConfig(ctypes.Structure):
    _fields_ = [
        ("bumpPath", ctypes.c_char * 512),
        ("voidPath", ctypes.c_char * 512),
        ("outputCsvPath", ctypes.c_char * 512),
        ("layerName", ctypes.c_char * 128),
        ("voxelX", ctypes.c_double),
        ("voxelY", ctypes.c_double),
        ("voxelZ", ctypes.c_double),
        ("zStretched4x", ctypes.c_int),
        ("connectivity", ctypes.c_int),
        ("minVoxels", ctypes.c_int),
        ("startSlice", ctypes.c_int),
        ("endSlice", ctypes.c_int),
        ("ngThreshold", ctypes.c_double),
        ("exportCsv", ctypes.c_int),
        # FAR (Teaching False Alarm Remover)
        ("applyFar", ctypes.c_int),
        ("bumpMinVoxels", ctypes.c_double),
        ("bumpMaxVoxels", ctypes.c_double),
        ("voidMinVoxels", ctypes.c_double),
        ("voidMaxVoxels", ctypes.c_double),
        ("cleanedBumpPath", ctypes.c_char * 512),
        ("cleanedVoidPath", ctypes.c_char * 512),
        ("writeCleanedMasks", ctypes.c_int),
        ("reserved", ctypes.c_int * 4),
    ]


class MesResult(ctypes.Structure):
    _fields_ = [
        ("success", ctypes.c_int),
        ("errorMessage", ctypes.c_char * 512),
        ("numObjects", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("height", ctypes.c_int),
        ("width", ctypes.c_int),
        ("startSliceUsed", ctypes.c_int),
        ("endSliceUsed", ctypes.c_int),
        ("elapsedSec", ctypes.c_double),
        ("csvPath", ctypes.c_char * 512),
        ("version", ctypes.c_char * 32),
        ("farRemovedBumps", ctypes.c_int),
        ("farClearedVoids", ctypes.c_int),
        ("cleanedBumpPath", ctypes.c_char * 512),
        ("cleanedVoidPath", ctypes.c_char * 512),
    ]


class MesObject(ctypes.Structure):
    _fields_ = [
        ("label", ctypes.c_int),
        ("gridRow", ctypes.c_int),
        ("gridCol", ctypes.c_int),
        ("soh", ctypes.c_double),
        ("bumpVolume", ctypes.c_double),
        ("voidVolume", ctypes.c_double),
        ("ratio", ctypes.c_double),
        ("judgment", ctypes.c_int),
        ("gapX", ctypes.c_double),
        ("gapY", ctypes.c_double),
        ("zMin", ctypes.c_int),
        ("zMax", ctypes.c_int),
        ("yMin", ctypes.c_int),
        ("yMax", ctypes.c_int),
        ("xMin", ctypes.c_int),
        ("xMax", ctypes.c_int),
        ("cz", ctypes.c_double),
        ("cy", ctypes.c_double),
        ("cx", ctypes.c_double),
        ("czGlobal", ctypes.c_double),
    ]


MesProgressCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int)


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


def _is_far_capable_dll(dll_path: str) -> bool:
    """New FAR build is ~77KB; pre-FAR build was ~69KB. Size is a cheap gate."""
    try:
        sz = os.path.getsize(dll_path)
        # FAR-capable MES from 2026-07-19 is 77824 bytes; old was 69120
        return sz >= 75000
    except OSError:
        return False


def _candidate_mes_paths(preferred_dir: str) -> List[str]:
    """Ordered list of dirs to search for a FAR-capable BumpVoidMes.dll."""
    preferred_dir = os.path.abspath(preferred_dir) if preferred_dir else ""
    candidates = []
    if preferred_dir:
        candidates.append(preferred_dir)

    try:
        from inno3d.core.resources import app_runtime_dir, default_dll_dir, project_root

        runtime_v2 = default_dll_dir()
        if runtime_v2:
            candidates.append(runtime_v2)
        candidates.append(str(app_runtime_dir() / "V2"))
        if not getattr(sys, "frozen", False):
            candidates.extend(
                [
                    str(project_root() / "V2"),
                    str(project_root().parent / "V2"),
                    r"E:\semiconductor\HBM_DEV_FOR_PROD\DEV\all_v2\BumpVoid_ISP_MES_v0.0.0\bin\Release",
                    r"E:\semiconductor\HBM_DEV_FOR_PROD\DEV\all_v2\BumpVoid_ISP_MES_v0.0.0\release\0.0.0",
                    r"E:\semiconductor\HBM_DEV_FOR_PROD\DEV\all_v2\V2",
                ]
            )
    except Exception:
        # Fallback if resources import fails during early bootstrap
        here = Path(__file__).resolve()
        candidates.append(str(here.parents[2] / "V2"))

    # unique preserve order
    seen = set()
    out = []
    for c in candidates:
        if not c:
            continue
        c = os.path.abspath(c)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def load_dll(dll_dir: str, force: bool = False):
    """Load BumpVoidMes.dll (FAR-capable build required for Online MES path).

    If ``dll_dir`` still has the pre-FAR binary (access-violation risk), auto-fallback
    to app ``V2`` or ``BumpVoid_ISP_MES_v0.0.0\\bin\\Release``.
    """
    global _dll, _dll_dir, _dll_path
    if _dll is not None and not force and _dll_dir == os.path.abspath(dll_dir):
        return get_version()

    chosen_path = None
    chosen_dir = None
    for d in _candidate_mes_paths(dll_dir):
        p = os.path.join(d, _MES_DLL_NAME)
        # also accept BumpVoidMes.dll.new if locked update left it
        p_new = p + ".new"
        if os.path.isfile(p_new) and _is_far_capable_dll(p_new):
            if not os.path.isfile(p) or not _is_far_capable_dll(p):
                p = p_new
        if not os.path.isfile(p):
            continue
        if _is_far_capable_dll(p):
            chosen_path = p
            chosen_dir = d
            break
        # keep first existing as last resort
        if chosen_path is None:
            chosen_path = p
            chosen_dir = d

    if not chosen_path:
        raise FileNotFoundError(
            f"{_MES_DLL_NAME} not found near {dll_dir}. "
            f"Build BumpVoid_ISP_MES and copy to V2."
        )

    if not _is_far_capable_dll(chosen_path):
        print(
            f"[bumpvoid_mes] WARNING: {chosen_path} looks like pre-FAR build "
            f"({os.path.getsize(chosen_path)} bytes). "
            f"Online may crash; replace with bin\\Release\\BumpVoidMes.dll"
        )

    _add_search_dirs(chosen_dir)
    ocv = os.path.join(chosen_dir, "opencv_world4110.dll")
    if not os.path.isfile(ocv):
        # OpenCV often lives with SEG in preferred dll folder
        for d in _candidate_mes_paths(dll_dir):
            alt = os.path.join(d, "opencv_world4110.dll")
            if os.path.isfile(alt):
                _add_search_dirs(d)
                break
    else:
        try:
            ctypes.CDLL(ocv)
        except OSError:
            pass

    # Windows caches loaded modules by basename — load via unique path when possible
    _dll = ctypes.CDLL(chosen_path)
    _dll_dir = chosen_dir
    _dll_path = chosen_path
    _setup()
    ver = get_version()
    print(
        f"[bumpvoid_mes] Loaded {_MES_DLL_NAME} version={ver} "
        f"size={os.path.getsize(chosen_path)} path={chosen_path}"
    )
    return ver


def unload_dll():
    """Drop handle so a newer DLL on disk can be loaded after app restart (or force)."""
    global _dll, _dll_dir, _dll_path
    _dll = None
    _dll_dir = None
    _dll_path = None


def _setup():
    _dll.Mes_InitConfig.argtypes = [ctypes.POINTER(MesConfig)]
    _dll.Mes_InitConfig.restype = None
    _dll.Mes_GetVersion.argtypes = []
    _dll.Mes_GetVersion.restype = ctypes.c_char_p
    _dll.Mes_Process.argtypes = [
        ctypes.POINTER(MesConfig),
        ctypes.POINTER(MesResult),
        MesProgressCallback,
    ]
    _dll.Mes_Process.restype = ctypes.c_int
    _dll.Mes_ProcessMemory.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(MesConfig),
        ctypes.POINTER(MesResult),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_ubyte),  # bumpOut
        ctypes.POINTER(ctypes.c_ubyte),  # voidOut
        MesProgressCallback,
    ]
    _dll.Mes_ProcessMemory.restype = ctypes.c_int
    _dll.Mes_GetObjects.argtypes = [
        ctypes.POINTER(MesObject),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    _dll.Mes_GetObjects.restype = ctypes.c_int
    _dll.Mes_Clear.argtypes = []
    _dll.Mes_Clear.restype = None


def _check():
    if _dll is None:
        raise RuntimeError("MES DLL not loaded. Call bumpvoid_mes.load_dll(dll_dir) first.")


def get_version() -> str:
    _check()
    v = _dll.Mes_GetVersion()
    return v.decode("utf-8", errors="replace") if v else ""


def get_dll_name() -> str:
    return _MES_DLL_NAME


def get_dll_path() -> Optional[str]:
    return _dll_path


def get_product_info() -> Dict[str, Any]:
    return {
        "name": _MES_DLL_NAME,
        "module": "MES",
        "product": "BumpVoidMes",
        "version": get_version() if _dll is not None else None,
        "path": _dll_path,
        "dir": _dll_dir,
    }


def is_loaded() -> bool:
    return _dll is not None


def init_config() -> MesConfig:
    _check()
    cfg = MesConfig()
    _dll.Mes_InitConfig(ctypes.byref(cfg))
    return cfg


def mes_objects_to_stats(objs, count: int, start_slice_offset: int = 0) -> List[Dict[str, Any]]:
    """Convert MesObject buffer → Viewer object_stats dicts (grid reindexed later)."""
    out: List[Dict[str, Any]] = []
    for i in range(count):
        o = objs[i]
        cz = float(o.czGlobal) if o.czGlobal else float(o.cz) + start_slice_offset
        z0 = int(o.zMin) + start_slice_offset
        z1 = int(o.zMax) + start_slice_offset
        out.append(
            {
                "row_id": f"({int(o.gridRow)},{int(o.gridCol)})",
                "label": int(o.label),
                "grid_row": int(o.gridRow),
                "grid_col": int(o.gridCol),
                "soh": float(o.soh),
                "c1_volume": float(o.bumpVolume),
                "c2_volume": float(o.voidVolume),
                "ratio": float(o.ratio),
                "pitch_x": float(o.gapX),
                "pitch_y": float(o.gapY),
                "z_min": z0,
                "z_max": z1,
                "y_min": int(o.yMin),
                "y_max": int(o.yMax),
                "x_min": int(o.xMin),
                "x_max": int(o.xMax),
                "centroid_z": cz,
                "centroid_y": float(o.cy),
                "centroid_x": float(o.cx),
                "judgment": int(o.judgment),
            }
        )
    return out


def reindex_grid_top_left(
    stats: List[Dict[str, Any]],
    *,
    voxel_x: float = 1.0,
    voxel_y: float = 1.0,
    per_layer: bool = True,
) -> List[Dict[str, Any]]:
    """
    Unified indexing for MES Object Stats + B2B Boundary (Teaching-style, 0-based).

    Convention (user / Online):
      - Index starts at **(0,0)**
      - **(0,0)** = top-left of the **XY** face:
          row increases with image Y (down the axial plane)
          col increases with image X (to the right)
      - Sort: centroid_y ascending (top→bottom), then centroid_x ascending (left→right)
      - Row grouping: |Δcy| < 0.7 * mean(object height)  [Teaching]
      - Pitch Gap X / Gap Y recomputed after index assign

    If ``per_layer`` and ``layer_name`` is set, each layer is indexed independently
    (Teaching multi-layer), so every layer has its own (0,0) at that layer's top-left.
    """
    if not stats:
        return []

    vx = float(voxel_x) if voxel_x > 0 else 1.0
    vy = float(voxel_y) if voxel_y > 0 else 1.0

    # Group by layer
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in stats:
        if per_layer:
            key = str(s.get("layer_name") or "N/A")
        else:
            key = "ALL"
        groups.setdefault(key, []).append(s)

    final: List[Dict[str, Any]] = []

    def _index_one_plane(layer_stats: List[Dict[str, Any]], layer_name: str) -> None:
        if not layer_stats:
            return
        # Top to bottom (cy descending), left to right (cx ascending)
        layer_stats.sort(
            key=lambda s: (-float(s.get("centroid_y", 0)), float(s.get("centroid_x", 0)))
        )
        rows: List[List[Dict[str, Any]]] = []
        current = [layer_stats[0]]
        y_prev = float(layer_stats[0].get("centroid_y", 0))
        heights = [
            max(1, int(s.get("y_max", 0)) - int(s.get("y_min", 0)))
            for s in layer_stats
        ]
        h_mean = float(np.mean(heights)) if len(heights) > 1 else 10.0
        tol = h_mean * 0.7

        for s in layer_stats[1:]:
            cy = float(s.get("centroid_y", 0))
            if abs(cy - y_prev) < tol:
                current.append(s)
            else:
                rows.append(current)
                current = [s]
                y_prev = cy
        rows.append(current)

        for r_idx, row_list in enumerate(rows):
            row_list.sort(key=lambda s: float(s.get("centroid_x", 0)))
            for c_idx, s in enumerate(row_list):
                s["grid_row"] = r_idx  # 0-based, top row = 0
                s["grid_col"] = c_idx  # 0-based, left col = 0
                # Unified display id for MES # and B2B Bump (R,C)
                s["bump_id"] = f"{r_idx},{c_idx}"  # raw for B2B DLL / CSV
                s["pitch_x"] = 0.0
                s["pitch_y"] = 0.0
                if c_idx < len(row_list) - 1:
                    nxt = row_list[c_idx + 1]
                    s["pitch_x"] = max(
                        0.0,
                        (int(nxt.get("x_min", 0)) - int(s.get("x_max", 0))) * vx,
                    )
                if r_idx < len(rows) - 1:
                    next_row = rows[r_idx + 1]
                    w_mean = max(1, int(s.get("x_max", 0)) - int(s.get("x_min", 0)))
                    closest = min(
                        next_row,
                        key=lambda t: abs(
                            float(t.get("centroid_x", 0)) - float(s.get("centroid_x", 0))
                        ),
                    )
                    if abs(
                        float(closest.get("centroid_x", 0)) - float(s.get("centroid_x", 0))
                    ) < w_mean * 1.5:
                        # Rows are ordered top→bottom (high Y → low Y). Gap is the
                        # empty band between this object (above) and the next row (below).
                        s["pitch_y"] = max(
                            0.0,
                            (int(s.get("y_min", 0)) - int(closest.get("y_max", 0))) * vy,
                        )
                final.append(s)

    # Stable layer order: natural sort by name, N/A last
    def _layer_key(k: str):
        return (k in ("N/A", "ALL", ""), k)

    for layer_name in sorted(groups.keys(), key=_layer_key):
        _index_one_plane(groups[layer_name], layer_name)

    # Global table order: by layer, then (row,col)
    final.sort(
        key=lambda s: (
            str(s.get("layer_name") or "N/A"),
            int(s.get("grid_row", 0)),
            int(s.get("grid_col", 0)),
        )
    )
    
    # Assign sequential row_id
    for idx, s in enumerate(final):
        s["row_id"] = idx + 1

    return final


def process_memory(
    bump: np.ndarray,
    void_mask: Optional[np.ndarray],
    *,
    voxel_x: float = 1.0,
    voxel_y: float = 1.0,
    voxel_z: float = 1.0,
    z_stretched_4x: bool = False,
    connectivity: int = 26,
    min_voxels: int = 0,
    start_slice: int = 0,
    end_slice: int = 0,
    ng_threshold: float = 0.05,
    export_csv: bool = False,
    csv_path: str = "",
    layer_name: str = "N/A",
    progress=None,
    return_labels: bool = False,
    apply_far: bool = True,
    bump_min_voxels: float = 0.0,
    bump_max_voxels: float = 1e300,
    void_min_voxels: float = 0.0,
    void_max_voxels: float = 1e300,
    cleaned_bump_path: str = "",
    cleaned_void_path: str = "",
    write_cleaned_masks: bool = False,
    return_cleaned_masks: bool = True,
) -> Tuple:
    """
    Measure from in-memory masks (Z,Y,X) uint8 with optional FAR inside DLL.

    Returns by default:
      (MesResult, stats) or
      (MesResult, stats, labels) if return_labels
      and if return_cleaned_masks: appends (bump_clean, void_clean) uint8 arrays
    """
    _check()
    if bump is None:
        raise ValueError("bump mask required")
    b = np.ascontiguousarray(bump, dtype=np.uint8)
    if b.ndim != 3:
        raise ValueError(f"bump must be 3D ZYX, got {b.shape}")
    d, h, w = b.shape

    v_ptr = None
    v_keep = None
    if void_mask is not None:
        v_keep = np.ascontiguousarray(void_mask, dtype=np.uint8)
        if v_keep.shape != b.shape:
            raise ValueError(f"void shape {v_keep.shape} != bump {b.shape}")
        v_ptr = v_keep.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    cfg = init_config()
    cfg.voxelX = float(voxel_x)
    cfg.voxelY = float(voxel_y)
    cfg.voxelZ = float(voxel_z)
    cfg.zStretched4x = 1 if z_stretched_4x else 0
    cfg.connectivity = int(connectivity)
    cfg.minVoxels = int(min_voxels)
    cfg.startSlice = int(start_slice)
    cfg.endSlice = int(end_slice)
    cfg.ngThreshold = float(ng_threshold)
    cfg.exportCsv = 1 if export_csv and csv_path else 0
    cfg.applyFar = 1 if apply_far else 0
    cfg.bumpMinVoxels = float(bump_min_voxels)
    cfg.bumpMaxVoxels = float(bump_max_voxels) if bump_max_voxels > 0 else 1e300
    cfg.voidMinVoxels = float(void_min_voxels)
    cfg.voidMaxVoxels = float(void_max_voxels) if void_max_voxels > 0 else 1e300
    cfg.writeCleanedMasks = 1 if write_cleaned_masks else 0
    if layer_name:
        cfg.layerName = layer_name.encode("utf-8")[:127]
    if csv_path:
        cfg.outputCsvPath = str(csv_path).encode("utf-8")[:511]
    if cleaned_bump_path:
        cfg.cleanedBumpPath = str(cleaned_bump_path).encode("utf-8")[:511]
    if cleaned_void_path:
        cfg.cleanedVoidPath = str(cleaned_void_path).encode("utf-8")[:511]

    def _noop(msg, pct):
        if progress is not None:
            try:
                s = msg.decode("utf-8", errors="replace") if isinstance(msg, bytes) else str(msg)
                progress(s, int(pct))
            except Exception:
                pass

    cb = MesProgressCallback(_noop)

    labels = None
    labels_ptr = ctypes.POINTER(ctypes.c_uint)()
    if return_labels:
        labels = np.zeros((d, h, w), dtype=np.uint32)
        labels_ptr = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))

    bump_out = None
    void_out = None
    bump_out_ptr = ctypes.POINTER(ctypes.c_ubyte)()
    void_out_ptr = ctypes.POINTER(ctypes.c_ubyte)()
    if return_cleaned_masks or write_cleaned_masks:
        bump_out = np.zeros((d, h, w), dtype=np.uint8)
        void_out = np.zeros((d, h, w), dtype=np.uint8)
        bump_out_ptr = bump_out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        void_out_ptr = void_out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    result = MesResult()
    rc = _dll.Mes_ProcessMemory(
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        v_ptr if v_ptr is not None else ctypes.POINTER(ctypes.c_ubyte)(),
        int(d),
        int(h),
        int(w),
        ctypes.byref(cfg),
        ctypes.byref(result),
        labels_ptr,
        bump_out_ptr,
        void_out_ptr,
        cb,
    )
    if not rc or not result.success:
        err = result.errorMessage.decode("utf-8", errors="replace")
        raise RuntimeError(f"Mes_ProcessMemory failed: {err or 'unknown'}")

    n = max(int(result.numObjects), 0)
    objs = (MesObject * max(n, 1))()
    out_n = ctypes.c_int(0)
    if n > 0:
        _dll.Mes_GetObjects(objs, n, ctypes.byref(out_n))
    stats = mes_objects_to_stats(objs, out_n.value, start_slice_offset=int(result.startSliceUsed))

    out: List[Any] = [result, stats]
    if return_labels:
        out.append(labels)
    if return_cleaned_masks:
        out.append(bump_out)
        out.append(void_out)
    return tuple(out)


def write_python_format_csv(path: str, stats: List[Dict[str, Any]], ng_threshold: float = 0.05) -> str:
    """Write CSV identical columns to Viewer.export_stats_csv."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "#",
                "Layer",
                "Bump ID",
                "B. H",
                "B. V",
                "V. V",
                "Ratio (C2/(C1+C2))",
                "Judgment",
                "Gap X (um)",
                "Gap Y (um)",
                "Z_min",
                "Z_max",
                "Y_min",
                "Y_max",
                "X_min",
                "X_max",
                "Centroid_Z",
                "Centroid_Y",
                "Centroid_X",
            ]
        )
        for s in stats:
            ratio = s.get("ratio", 0.0)
            is_ng = ratio >= ng_threshold if ratio != float("inf") else True
            jud = s.get("judgment")
            if jud in (1, 8):
                judgment_str = "NG" if jud == 1 else "OK"
            else:
                judgment_str = "NG" if is_ng else "OK"
            w.writerow(
                [
                    s.get("row_id", ""),
                    str(s.get("layer_name", "") or ""),
                    f"({s.get('grid_row', '?')},{s.get('grid_col', '?')})",
                    f"{s.get('soh', 0):.3f}",
                    f"{s.get('c1_volume', 0):.3f}",
                    f"{s.get('c2_volume', 0):.3f}",
                    f"{ratio * 100:.3f}%" if ratio != float("inf") else "N/A",
                    judgment_str,
                    f"{s.get('pitch_x', 0):.3f}",
                    f"{s.get('pitch_y', 0):.3f}",
                    s.get("z_min", 0),
                    s.get("z_max", 0),
                    s.get("y_min", 0),
                    s.get("y_max", 0),
                    s.get("x_min", 0),
                    s.get("x_max", 0),
                    f"{s.get('centroid_z', 0):.3f}",
                    f"{s.get('centroid_y', 0):.3f}",
                    f"{s.get('centroid_x', 0):.3f}",
                ]
            )
    return path


def compare_stats(
    mes_stats: List[Dict[str, Any]],
    py_stats: List[Dict[str, Any]],
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Compare MES vs Python object tables (match by grid row/col)."""
    def key(s):
        return (s.get("grid_row", -1), s.get("grid_col", -1))

    mes_map = {key(s): s for s in mes_stats}
    py_map = {key(s): s for s in py_stats}
    common = sorted(set(mes_map) & set(py_map))
    only_m = sorted(set(mes_map) - set(py_map))
    only_p = sorted(set(py_map) - set(mes_map))

    fields = ["soh", "c1_volume", "c2_volume", "ratio", "pitch_x", "pitch_y"]
    max_abs = {f: 0.0 for f in fields}
    n_mismatch = 0
    samples = []
    for k in common:
        m, p = mes_map[k], py_map[k]
        for f in fields:
            d = abs(float(m.get(f, 0)) - float(p.get(f, 0)))
            max_abs[f] = max(max_abs[f], d)
            if d > tol:
                n_mismatch += 1
                if len(samples) < 8:
                    samples.append((k, f, m.get(f), p.get(f), d))

    return {
        "n_mes": len(mes_stats),
        "n_py": len(py_stats),
        "n_common": len(common),
        "only_mes": only_m,
        "only_py": only_p,
        "max_abs": max_abs,
        "n_metric_mismatch": n_mismatch,
        "samples": samples,
        "pass": (
            len(mes_stats) == len(py_stats)
            and not only_m
            and not only_p
            and n_mismatch == 0
        ),
    }
