# inno3d/features/shared/mes_mapping.py
# -----------------------------------------------------------------------
# Host-agnostic MES mapping helpers (label ↔ stats, pick, nav coords).
# Used by Viewer stats_panel and Teaching selection/pick paths.
# Qt table selection stays on the host (Single vs Multi widgets differ).
# -----------------------------------------------------------------------

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# (class_num, label_id) — class 1 = bump / C1
HighlightPair = Tuple[int, int]


def label_set_from_highlights(
    selected_highlight_objects: Optional[Sequence[Sequence[Any]]],
) -> Set[int]:
    """Extract positive C1 labels from ``selected_highlight_objects``."""
    out: Set[int] = set()
    for item in selected_highlight_objects or []:
        try:
            _cls, lab = item[0], int(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if lab > 0:
            out.add(lab)
    return out


def label_at_volume_xyz(
    labeled,
    x: int,
    y: int,
    z: int,
    search_rad: int = 3,
) -> int:
    """Read labeled volume at (x,y,z); neighborhood fallback if empty."""
    if labeled is None:
        return 0
    Z, Y, X = labeled.shape
    x, y, z = int(x), int(y), int(z)
    if not (0 <= z < Z and 0 <= y < Y and 0 <= x < X):
        return 0
    lab = int(labeled[z, y, x])
    if lab > 0:
        return lab
    r = max(1, int(search_rad))
    z0, z1 = max(0, z - r), min(Z, z + r + 1)
    y0, y1 = max(0, y - r), min(Y, y + r + 1)
    x0, x1 = max(0, x - r), min(X, x + r + 1)
    patch = labeled[z0:z1, y0:y1, x0:x1]
    vals = patch[patch > 0]
    if vals.size == 0:
        return 0
    try:
        hits = np.argwhere(patch > 0)
        best_i = 0
        best_d = 1e18
        cz, cy, cx = z - z0, y - y0, x - x0
        for i, (hz, hy, hx) in enumerate(hits):
            d = (hz - cz) ** 2 + (hy - cy) ** 2 + (hx - cx) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        hz, hy, hx = hits[best_i]
        return int(patch[hz, hy, hx])
    except Exception:
        return int(np.bincount(vals.ravel()).argmax())


def label_at_stat_centroid(
    labeled,
    stat: Optional[Dict[str, Any]],
    z_offset: int = 0,
) -> int:
    """Look up labeled id at MES centroid (or bbox center) + neighborhood."""
    if labeled is None or stat is None:
        return 0
    Z, Y, X = labeled.shape
    _z_off = int(z_offset or 0)

    def _at(z, y, x):
        z = int(round(float(z))) + _z_off
        y = int(round(float(y)))
        x = int(round(float(x)))
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            return int(labeled[z, y, x])
        return 0

    lab = _at(
        stat.get("centroid_z", 0),
        stat.get("centroid_y", 0),
        stat.get("centroid_x", 0),
    )
    if lab > 0:
        return lab
    lab = _at(
        (float(stat.get("z_min", 0)) + float(stat.get("z_max", 0))) * 0.5,
        (float(stat.get("y_min", 0)) + float(stat.get("y_max", 0))) * 0.5,
        (float(stat.get("x_min", 0)) + float(stat.get("x_max", 0))) * 0.5,
    )
    if lab > 0:
        return lab
    z0 = int(round(float(stat.get("centroid_z", 0)))) + _z_off
    y0 = int(round(float(stat.get("centroid_y", 0))))
    x0 = int(round(float(stat.get("centroid_x", 0))))
    if not (0 <= z0 < Z):
        return 0
    for rad in (2, 5, 10):
        y1, y2 = max(0, y0 - rad), min(Y, y0 + rad + 1)
        x1, x2 = max(0, x0 - rad), min(X, x0 + rad + 1)
        patch = labeled[z0, y1:y2, x1:x2]
        vals = patch[patch > 0]
        if vals.size:
            return int(np.bincount(vals.ravel()).argmax())
    return 0


def resolve_stat_label(
    stat: Dict[str, Any],
    labeled=None,
    z_offset: int = 0,
) -> int:
    """Return positive label for a stats dict; may mutate ``stat['label']``."""
    try:
        lab = int(stat.get("label", 0))
    except (TypeError, ValueError):
        lab = 0
    if lab > 0:
        return lab
    if labeled is not None:
        lab2 = label_at_stat_centroid(labeled, stat, z_offset=z_offset)
        if lab2 > 0:
            stat["label"] = lab2
            return lab2
    return 0


def stats_index_for_label(
    object_stats: Optional[Sequence[Dict[str, Any]]],
    label: Any,
    *,
    labeled=None,
    z_offset: int = 0,
) -> Optional[int]:
    """Map labeled_class1 id → index in ``object_stats``."""
    if not label or not object_stats:
        return None
    try:
        label = int(label)
    except (TypeError, ValueError):
        return None
    if label <= 0:
        return None
    for i, st in enumerate(object_stats):
        try:
            lab = int(st.get("label", 0))
        except (TypeError, ValueError):
            lab = 0
        if lab == label:
            return i
        if lab <= 0 and labeled is not None:
            lab2 = label_at_stat_centroid(labeled, st, z_offset=z_offset)
            if lab2 == label:
                st["label"] = lab2
                return i
    return None


def centroid_nav_xyz(
    stat: Optional[Dict[str, Any]],
    z_offset: int = 0,
) -> Optional[Tuple[int, int, int]]:
    """Return ``(cx, cy, cz)`` for ``updatePoint`` / slider nav, or None."""
    if stat is None:
        return None
    try:
        cz = int(round(float(stat.get("centroid_z", 0)))) + int(z_offset or 0)
        cy = int(round(float(stat.get("centroid_y", 0))))
        cx = int(round(float(stat.get("centroid_x", 0))))
        return cx, cy, cz
    except (TypeError, ValueError):
        return None


def mes_bbox_for_labels(
    object_stats: Optional[Sequence[Dict[str, Any]]],
    label_set: Set[int],
    z_offset: int = 0,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Union bbox of selected labels from object_stats.

    Returns (z0,z1,y0,y1,x0,x1) inclusive, or None.
    """
    if not label_set or not object_stats:
        return None
    _z_off = int(z_offset or 0)
    z0 = y0 = x0 = 10**9
    z1 = y1 = x1 = -1
    hit = False
    for st in object_stats:
        try:
            lab = int(st.get("label", 0))
        except (TypeError, ValueError):
            continue
        if lab not in label_set:
            continue
        try:
            z0 = min(z0, int(st["z_min"]) + _z_off)
            z1 = max(z1, int(st["z_max"]) + _z_off)
            y0 = min(y0, int(st["y_min"]))
            y1 = max(y1, int(st["y_max"]))
            x0 = min(x0, int(st["x_min"]))
            x1 = max(x1, int(st["x_max"]))
            hit = True
        except (KeyError, TypeError, ValueError):
            continue
    if not hit:
        return None
    return z0, z1, y0, y1, x0, x1


def highlights_from_stats_indices(
    object_stats: Optional[Sequence[Dict[str, Any]]],
    stats_indices: Iterable[int],
    *,
    labeled=None,
    z_offset: int = 0,
) -> List[HighlightPair]:
    """Build ``selected_highlight_objects`` list from stats indices."""
    if not object_stats:
        return []
    out: List[HighlightPair] = []
    n = len(object_stats)
    for idx in stats_indices:
        try:
            i = int(idx)
        except (TypeError, ValueError):
            continue
        if i < 0 or i >= n:
            continue
        lab = resolve_stat_label(object_stats[i], labeled=labeled, z_offset=z_offset)
        if lab > 0:
            out.append((1, lab))
    return out


def first_centroid_world(
    object_stats: Optional[Sequence[Dict[str, Any]]],
    label_set: Set[int],
    spacing: Tuple[float, float, float],
    z_offset: int = 0,
) -> Optional[Tuple[float, float, float]]:
    """First selected object's centroid in world coords (sx,sy,sz spacing)."""
    if not object_stats or not label_set:
        return None
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    _z_off = int(z_offset or 0)
    for st in object_stats:
        try:
            lab = int(st.get("label", 0))
        except (TypeError, ValueError):
            continue
        if lab not in label_set:
            continue
        try:
            cz = (float(st.get("centroid_z", 0)) + _z_off) * sz
            cy = float(st.get("centroid_y", 0)) * sy
            cx = float(st.get("centroid_x", 0)) * sx
            return cx, cy, cz
        except (TypeError, ValueError):
            continue
    return None


class MesPickDebounce:
    """Debounce dual Qt+VTK pick paths (Multi ON toggle-twice guard)."""

    def __init__(self, window_s: float = 0.12):
        self.window_s = float(window_s)
        self._last: Optional[Tuple[float, int]] = None

    def is_duplicate(self, lab: Any) -> bool:
        try:
            lab = int(lab)
        except (TypeError, ValueError):
            return False
        if lab <= 0:
            return False
        now = time.monotonic()
        if self._last is not None:
            t0, last_lab = self._last
            if int(last_lab) == lab and (now - float(t0)) < self.window_s:
                return True
        self._last = (now, lab)
        return False
