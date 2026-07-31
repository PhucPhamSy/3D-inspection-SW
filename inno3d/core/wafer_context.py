"""
Wafer → Chip → FOV hierarchy for Online inspection context.

Data is intended to be supplied by the 3D Reconstruction CT PC / host DB.
Inno3D (Inspection) reads Wafer / Chip / FOV metadata and will write
inspection outcomes under each FOV's Results folder.

Host mass-production volume path (canonical):

    yy_mm_dd/LotID_FoupID/WaferID/ChipLocation/FOVLocation/Input.tiff

Results:

    yy_mm_dd/LotID_FoupID/WaferID/ChipLocation/FOVLocation/Results/

Bin codes (from wafer-map email / draw code):
  0 = Outside wafer
  1 = Bad  (NG)          — FinalBin after inspection
  8 = Good (OK)          — FinalBin after inspection

App-only display / rollup (not sent by host map protocol):
  2 = Pending            — die/FOV not inspected yet (no FinalBin)

Host FinalBin after inspect is only 1 or 8. Pending is the correct
default when Online opens a new wafer before any FOV inspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import copy
import json
import re
from pathlib import Path

import numpy as np

BIN_OUTSIDE = 0
BIN_NG = 1
BIN_PENDING = 2  # not inspected yet (UI / rollup only)
BIN_OK = 8

# Circular die inclusion — MUST match Online CONTEXT + Batch Review wafer maps.
# Die cell-centers in normalized space [-0.5, 0.5]; r² = 0.24 (r ≈ 0.49).
# Older Online empty_geometry used index-radius n*0.48 which only left Col 13
# on Row 1 while Batch Review still painted chips on both sides.
WAFER_MAP_RADIUS2 = 0.24


def die_inside_wafer(col0: int, row0: int, map_size: int) -> bool:
    """True if die at 0-based (col, row) lies inside the circular wafer map."""
    n = max(int(map_size), 1)
    cx = (int(col0) + 0.5) / n - 0.5
    cy = (int(row0) + 0.5) / n - 0.5
    return (cx * cx + cy * cy) <= WAFER_MAP_RADIUS2


def circular_bin_mask(map_size: int, inside_bin: int = BIN_PENDING) -> "np.ndarray":
    """n×n bin map: Outside outside the circle, ``inside_bin`` for dies inside."""
    n = int(map_size)
    bins = np.full((n, n), BIN_OUTSIDE, dtype=np.uint8)
    for r in range(n):
        for c in range(n):
            if die_inside_wafer(c, r, n):
                bins[r, c] = int(inside_bin)
    return bins


# Default 9-point layout (row-major P1..P9), offsets in mm relative to chip center.
# Real offsets come from host DB; these are placeholders for UI / offline demo.
_DEFAULT_POINT_OFFSETS_MM = [
    (-0.25, 0.25), (0.0, 0.25), (0.25, 0.25),
    (-0.25, 0.0),  (0.0, 0.0),  (0.25, 0.0),
    (-0.25, -0.25), (0.0, -0.25), (0.25, -0.25),
]


@dataclass
class HostVolumePathInfo:
    """Parsed host path for one FOV volume (mass-production layout).

    Expected::

        {date}/{LotID_FoupID}/{WaferID}/{ChipLoc}/{FOVLoc}/Input.tiff
        → results: .../{FOVLoc}/Results/
    """
    raw_path: str = ""
    date_folder: str = ""
    lot_foup_id: str = ""  # combined folder name LotID_FoupID
    lot_id: str = ""
    foup_id: str = ""
    wafer_id: str = ""
    chip_col: int = 0  # 1-based X
    chip_row: int = 0  # 1-based Y
    chip_folder: str = ""
    fov_index: int = 0  # 1..9
    fov_folder: str = ""
    input_path: str = ""  # full path to Input.tiff (or host file)
    fov_dir: str = ""  # folder containing Input.tiff
    results_dir: str = ""  # sibling Results/ under FOV folder
    parsed_ok: bool = False

    def breadcrumb(self) -> str:
        parts = []
        if self.date_folder:
            parts.append(self.date_folder)
        if self.lot_foup_id:
            parts.append(self.lot_foup_id)
        if self.wafer_id:
            parts.append(self.wafer_id)
        if self.chip_col > 0 and self.chip_row > 0:
            parts.append(f"Chip ({self.chip_col},{self.chip_row})")
        elif self.chip_folder:
            parts.append(self.chip_folder)
        if self.fov_index > 0:
            parts.append(f"FOV P{self.fov_index}")
        elif self.fov_folder:
            parts.append(self.fov_folder)
        return "  ›  ".join(parts) if parts else (self.raw_path or "—")


def _split_lot_foup(name: str) -> Tuple[str, str]:
    """Split ``LotID_FoupID`` on the *last* underscore when possible.

    Delegates to :func:`inno3d.core.lot_foup.split_lot_foup` (canonical).
    """
    from inno3d.core.lot_foup import split_lot_foup

    return split_lot_foup(name)


def _parse_chip_location(name: str) -> Tuple[int, int]:
    """Parse chip folder → (col, row) 1-based. Returns (0,0) if unknown.

    Accepts: Chip_17_16, chip17_16, 17_16, C17_R16, X17Y16, (17,16)
    """
    s = (name or "").strip()
    if not s:
        return 0, 0
    # (17,16) or 17,16
    m = re.search(r"[\(\[]?\s*(\d+)\s*[,_\-xX]\s*(\d+)\s*[\)\]]?", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    # C17_R16 / X17_Y16
    m = re.search(r"[CcXx](\d+)\s*[_-]?\s*[RrYy](\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Chip_17_16
    m = re.search(r"(?i)chip[_-]?(\d+)[_-](\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    return 0, 0


def _parse_fov_location(name: str) -> int:
    """Parse FOV folder → point index 1..9. Returns 0 if unknown.

    Accepts: FOV_P5, FOV_P05, P5, Point5, FOV5, Point_5
    """
    s = (name or "").strip()
    if not s:
        return 0
    m = re.search(r"(?i)(?:fov|point)?[_-]*p(?:oint)?[_-]*(\d+)", s)
    if m:
        idx = int(m.group(1))
        return idx if 1 <= idx <= 9 else 0
    m = re.search(r"(?i)^p(\d+)$", s)
    if m:
        idx = int(m.group(1))
        return idx if 1 <= idx <= 9 else 0
    nums = re.findall(r"\d+", s)
    if nums:
        idx = int(nums[-1])
        return idx if 1 <= idx <= 9 else 0
    return 0


def parse_host_volume_path(path: str) -> HostVolumePathInfo:
    """Parse host CT path into Lot/Wafer/Chip/FOV + Results dir.

    Canonical layout (any drive prefix allowed)::

        yy_mm_dd / LotID_FoupID / WaferID /
            ChipLocationinWafer / FOVLocationinChip / Input.tiff

    Results sibling::

        ... / FOVLocationinChip / Results/
    """
    info = HostVolumePathInfo(raw_path=str(path or ""))
    if not path:
        return info

    p = Path(path)
    # If path is FOV folder (no file), treat as directory
    if p.exists() and p.is_dir():
        fov_dir = p
        # Prefer Input.tiff / Input.tif
        input_file = None
        for cand in ("Input.tiff", "Input.tif", "input.tiff", "input.tif"):
            c = fov_dir / cand
            if c.is_file():
                input_file = c
                break
        if input_file is None:
            tiffs = sorted(
                [f for f in fov_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in (".tif", ".tiff")
                 and "bump" not in f.name.lower()
                 and "void" not in f.name.lower()]
            )
            input_file = tiffs[0] if tiffs else None
        info.input_path = str(input_file) if input_file else str(fov_dir)
        info.fov_dir = str(fov_dir)
    else:
        # File path (may not exist yet on disk during unit tests)
        if p.suffix.lower() in (".tif", ".tiff") or p.name:
            info.input_path = str(p)
            info.fov_dir = str(p.parent)
        else:
            info.fov_dir = str(p)
            info.input_path = str(p)

    fov_dir = Path(info.fov_dir)
    info.results_dir = str(fov_dir / "Results")
    info.fov_folder = fov_dir.name
    info.fov_index = _parse_fov_location(info.fov_folder)

    chip_dir = fov_dir.parent
    info.chip_folder = chip_dir.name
    info.chip_col, info.chip_row = _parse_chip_location(info.chip_folder)

    # Walk up: Wafer / Lot_Foup / date  (no RecipeID folder)
    wafer_dir = chip_dir.parent
    lot_dir = wafer_dir.parent
    date_dir = lot_dir.parent

    info.wafer_id = wafer_dir.name
    info.lot_foup_id = lot_dir.name
    info.lot_id, info.foup_id = _split_lot_foup(info.lot_foup_id)
    info.date_folder = date_dir.name if date_dir and date_dir.name not in ("", "/", "\\") else ""

    # Heuristic: if chip/fov parse failed, try alternate (folder depth short)
    # e.g. path ends with .../Chip_x_y/FOV_Pn/Input.tiff only (no date/lot)
    parts = [x for x in Path(info.raw_path).parts if x not in ("/", "\\")]
    # Drop drive letter on Windows
    if parts and re.match(r"^[A-Za-z]:$", parts[0]):
        parts = parts[1:]

    info.parsed_ok = bool(
        info.fov_dir
        and (info.fov_index > 0 or info.chip_col > 0)
    )
    return info


@dataclass
class FovPoint:
    """One FOV (inspection point) inside a chip — maps to a CT volume."""
    index: int  # 1..9
    x_offset_mm: float = 0.0
    y_offset_mm: float = 0.0
    # Default Pending — not OK until InspectPoint / MES writes a final judge
    judge: int = BIN_PENDING  # 0 / 1 / 2 / 8
    ct_result: str = ""
    ai_score: float = 0.0
    volume_path: str = ""  # path to CT volume for this FOV (when known)
    results_dir: str = ""  # e.g. .../FOV_P5/Results

    def is_ok(self) -> bool:
        return int(self.judge) == BIN_OK

    def is_ng(self) -> bool:
        return int(self.judge) == BIN_NG

    def is_pending(self) -> bool:
        return int(self.judge) not in (BIN_OK, BIN_NG)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FovPoint":
        return cls(
            index=int(d.get("index", 1)),
            x_offset_mm=float(d.get("x_offset_mm", d.get("XOffset", 0.0))),
            y_offset_mm=float(d.get("y_offset_mm", d.get("YOffset", 0.0))),
            judge=int(d.get("judge", d.get("Judge", BIN_PENDING))),
            ct_result=str(d.get("ct_result", d.get("CTResult", ""))),
            ai_score=float(d.get("ai_score", d.get("AIScore", 0.0))),
            volume_path=str(d.get("volume_path", "")),
            results_dir=str(d.get("results_dir", d.get("results", ""))),
        )


@dataclass
class NgCluster:
    """Connected group of FinalBin=NG dies (email: Cluster Boundary / Centroid).

    Spec reference (Innometry wafer-map email)::

        · Cluster = 8-connected component of die bin == 1 (NG)
        · Centroid Chip (X,Y) = member die nearest geometric mean of members
        · Max Radius (Die) = max Euclidean distance from geometric center
          to any member die (in die-pitch units, col/row space)
        · Boundary = axis-aligned box covering member dies (dashed on map)
    """
    id: int  # 1-based cluster id (sorted by size desc, then centroid)
    members: List[Tuple[int, int]] = field(default_factory=list)  # (col,row) 1-based
    centroid_col: int = 0  # 1-based assigned chip X
    centroid_row: int = 0  # 1-based assigned chip Y
    size: int = 0
    max_radius: float = 0.0
    # Bounding box in 1-based inclusive col/row
    min_col: int = 0
    max_col: int = 0
    min_row: int = 0
    max_row: int = 0
    # Geometric mean (float, 1-based) before snap to member die
    geo_col: float = 0.0
    geo_row: float = 0.0

    def contains(self, col: int, row: int) -> bool:
        return (int(col), int(row)) in {(int(c), int(r)) for c, r in self.members}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "members": [[int(c), int(r)] for c, r in self.members],
            "centroid_col": self.centroid_col,
            "centroid_row": self.centroid_row,
            "size": self.size,
            "max_radius": self.max_radius,
            "min_col": self.min_col,
            "max_col": self.max_col,
            "min_row": self.min_row,
            "max_row": self.max_row,
            "geo_col": self.geo_col,
            "geo_row": self.geo_row,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NgCluster":
        mem = d.get("members") or []
        members = []
        for m in mem:
            if isinstance(m, (list, tuple)) and len(m) >= 2:
                members.append((int(m[0]), int(m[1])))
        return cls(
            id=int(d.get("id", 0)),
            members=members,
            centroid_col=int(d.get("centroid_col", 0)),
            centroid_row=int(d.get("centroid_row", 0)),
            size=int(d.get("size", len(members))),
            max_radius=float(d.get("max_radius", 0.0)),
            min_col=int(d.get("min_col", 0)),
            max_col=int(d.get("max_col", 0)),
            min_row=int(d.get("min_row", 0)),
            max_row=int(d.get("max_row", 0)),
            geo_col=float(d.get("geo_col", 0.0)),
            geo_row=float(d.get("geo_row", 0.0)),
        )


def compute_ng_clusters(bins: Any, connectivity: int = 8) -> List[NgCluster]:
    """Find NG die clusters on a FinalBin map (0=Out, 1=NG, 2=Pend, 8=OK).

    Parameters
    ----------
    bins : (n,n) array, row-major, 0-based indices
    connectivity : 8 (default, matches email figure) or 4
    """
    arr = np.asarray(bins)
    if arr.ndim != 2:
        return []
    n_r, n_c = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    if connectivity == 4:
        neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        neigh = (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        )

    raw: List[List[Tuple[int, int]]] = []  # list of member lists as 0-based (c,r)

    for r in range(n_r):
        for c in range(n_c):
            if visited[r, c] or int(arr[r, c]) != BIN_NG:
                continue
            # BFS / flood fill
            stack = [(c, r)]
            visited[r, c] = True
            members0: List[Tuple[int, int]] = []
            while stack:
                cc, rr = stack.pop()
                members0.append((cc, rr))
                for dc, dr in neigh:
                    nc, nr = cc + dc, rr + dr
                    if 0 <= nr < n_r and 0 <= nc < n_c and not visited[nr, nc]:
                        if int(arr[nr, nc]) == BIN_NG:
                            visited[nr, nc] = True
                            stack.append((nc, nr))
            if members0:
                raw.append(members0)

    clusters: List[NgCluster] = []
    for members0 in raw:
        # 1-based members (col, row)
        members = [(c + 1, r + 1) for c, r in members0]
        cols = [m[0] for m in members]
        rows = [m[1] for m in members]
        geo_c = float(sum(cols)) / len(cols)
        geo_r = float(sum(rows)) / len(rows)
        # Assigned centroid chip = member nearest geometric mean
        best = members[0]
        best_d = 1e18
        for mc, mr in members:
            d = (mc - geo_c) ** 2 + (mr - geo_r) ** 2
            if d < best_d:
                best_d = d
                best = (mc, mr)
        # Max radius from geometric center to member centers
        max_r = 0.0
        for mc, mr in members:
            d = ((mc - geo_c) ** 2 + (mr - geo_r) ** 2) ** 0.5
            if d > max_r:
                max_r = d
        clusters.append(
            NgCluster(
                id=0,  # assigned after sort
                members=sorted(members, key=lambda t: (t[1], t[0])),
                centroid_col=int(best[0]),
                centroid_row=int(best[1]),
                size=len(members),
                max_radius=float(max_r),
                min_col=min(cols),
                max_col=max(cols),
                min_row=min(rows),
                max_row=max(rows),
                geo_col=geo_c,
                geo_row=geo_r,
            )
        )

    # Sort: larger first, then upper-left centroid — stable IDs for summary table
    clusters.sort(
        key=lambda cl: (-cl.size, cl.centroid_row, cl.centroid_col)
    )
    for i, cl in enumerate(clusters, start=1):
        cl.id = i
    return clusters


@dataclass
class ChipCell:
    """One die / chip on the wafer map."""
    col: int  # 1-based column (X)
    row: int  # 1-based row (Y)
    original_bin: int = BIN_OUTSIDE
    final_bin: int = BIN_OUTSIDE
    points: List[FovPoint] = field(default_factory=list)
    good_count: int = 0
    bad_count: int = 0
    chip_ng: bool = False

    def ensure_points(self) -> None:
        if len(self.points) >= 9:
            return
        existing = {p.index: p for p in self.points}
        pts: List[FovPoint] = []
        for i in range(1, 10):
            if i in existing:
                pts.append(existing[i])
            else:
                ox, oy = _DEFAULT_POINT_OFFSETS_MM[i - 1]
                pts.append(
                    FovPoint(index=i, x_offset_mm=ox, y_offset_mm=oy, judge=BIN_PENDING)
                )
        self.points = pts
        self.recompute_counts()

    def recompute_counts(self) -> None:
        """Roll up FOV judges → Chip FinalBin (email rule).

        Email: ChipNG = (BadCount > 0) among inspected points.
        Until FOVs are judged, chip stays **Pending** — never auto-OK.
        """
        if self.original_bin == BIN_OUTSIDE and self.final_bin == BIN_OUTSIDE and not self.points:
            return
        self.good_count = sum(1 for p in self.points if p.is_ok())
        self.bad_count = sum(1 for p in self.points if p.is_ng())
        pending = sum(1 for p in self.points if p.is_pending())
        self.chip_ng = self.bad_count > 0
        if self.chip_ng:
            self.final_bin = BIN_NG  # ≥1 FOV NG → chip NG
        elif pending == 0 and self.good_count > 0 and self.bad_count == 0:
            self.final_bin = BIN_OK  # all 9 FOVs inspected OK
        else:
            self.final_bin = BIN_PENDING  # not finished / nothing judged

    def point(self, index: int) -> Optional[FovPoint]:
        for p in self.points:
            if p.index == index:
                return p
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "col": self.col,
            "row": self.row,
            "original_bin": self.original_bin,
            "final_bin": self.final_bin,
            "points": [p.to_dict() for p in self.points],
            "good_count": self.good_count,
            "bad_count": self.bad_count,
            "chip_ng": self.chip_ng,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChipCell":
        pts = [FovPoint.from_dict(p) for p in d.get("points", [])]
        cell = cls(
            col=int(d.get("col", d.get("x", 1))),
            row=int(d.get("row", d.get("y", 1))),
            original_bin=int(d.get("original_bin", d.get("OriginalBin", BIN_OK))),
            final_bin=int(d.get("final_bin", d.get("FinalBin", BIN_OK))),
            points=pts,
            good_count=int(d.get("good_count", 0)),
            bad_count=int(d.get("bad_count", 0)),
            chip_ng=bool(d.get("chip_ng", d.get("ChipNG", False))),
        )
        cell.ensure_points()
        return cell


@dataclass
class WaferContext:
    """Full wafer map + selection state for the CONTEXT panel."""
    lot_id: str = ""
    foup_id: str = ""
    lot_foup_id: str = ""  # combined folder LotID_FoupID when known
    wafer_id: str = ""
    date_folder: str = ""  # yy_mm_dd host root segment
    wafer_size_mm: float = 300.0
    die_size_mm: float = 12.0
    map_size: int = 25
    bins: Any = None  # np.ndarray [row, col] 0-based, values 0/1/8
    chips: Dict[Tuple[int, int], ChipCell] = field(default_factory=dict)
    # Selection (1-based chip col/row; FOV point 1..9)
    selected_col: int = 0
    selected_row: int = 0
    selected_fov: int = 5  # default center P5
    yield_pct: float = 0.0
    # Last host path applied (for UI breadcrumb / Results link)
    last_host_path: str = ""
    last_results_dir: str = ""
    # NG clusters (email Cluster Boundary / Centroid) — recomputed from bins
    clusters: List[NgCluster] = field(default_factory=list)
    selected_cluster_id: int = 0  # 0 = none; else NgCluster.id

    def __post_init__(self) -> None:
        n = int(self.map_size)
        if self.bins is None:
            self.bins = np.zeros((n, n), dtype=np.uint8)
        else:
            self.bins = np.asarray(self.bins, dtype=np.uint8)
            if self.bins.shape != (n, n):
                raise ValueError(f"bins shape {self.bins.shape} != ({n},{n})")
        if self.clusters is None:
            self.clusters = []

    def key(self, col: int, row: int) -> Tuple[int, int]:
        return (int(col), int(row))

    def chip_at(self, col: int, row: int) -> Optional[ChipCell]:
        return self.chips.get(self.key(col, row))

    def selected_chip(self) -> Optional[ChipCell]:
        if self.selected_col <= 0 or self.selected_row <= 0:
            return None
        return self.chip_at(self.selected_col, self.selected_row)

    def selected_point(self) -> Optional[FovPoint]:
        chip = self.selected_chip()
        if chip is None:
            return None
        return chip.point(self.selected_fov)

    def selected_cluster(self) -> Optional[NgCluster]:
        if self.selected_cluster_id <= 0:
            return None
        for cl in self.clusters:
            if cl.id == self.selected_cluster_id:
                return cl
        return None

    def cluster_at(self, col: int, row: int) -> Optional[NgCluster]:
        """Return the NG cluster containing die (col,row) 1-based, if any."""
        for cl in self.clusters:
            if cl.contains(col, row):
                return cl
        return None

    def select_chip(self, col: int, row: int) -> bool:
        cell = self.chip_at(col, row)
        if cell is None:
            return False
        if cell.final_bin == BIN_OUTSIDE and cell.original_bin == BIN_OUTSIDE:
            return False
        self.selected_col = int(col)
        self.selected_row = int(row)
        # Email: selecting a chip that sits in a cluster assigns that cluster
        cl = self.cluster_at(col, row)
        self.selected_cluster_id = int(cl.id) if cl is not None else 0
        return True

    def select_fov(self, index: int) -> bool:
        if index < 1 or index > 9:
            return False
        self.selected_fov = int(index)
        return True

    def select_cluster(self, cluster_id: int) -> bool:
        """Select cluster by id and jump selection to its centroid chip."""
        cl = None
        for c in self.clusters:
            if c.id == int(cluster_id):
                cl = c
                break
        if cl is None:
            self.selected_cluster_id = 0
            return False
        self.selected_cluster_id = int(cl.id)
        # Assign centroid as selected chip (email Cluster Centroid Setting)
        if cl.centroid_col > 0 and cl.centroid_row > 0:
            if self.chip_at(cl.centroid_col, cl.centroid_row) is not None:
                self.selected_col = int(cl.centroid_col)
                self.selected_row = int(cl.centroid_row)
            else:
                # synthesize soft selection even if ChipCell missing
                self.selected_col = int(cl.centroid_col)
                self.selected_row = int(cl.centroid_row)
        return True

    def recompute_clusters(self) -> List[NgCluster]:
        """Rebuild NG clusters from current FinalBin map."""
        prev = int(self.selected_cluster_id or 0)
        self.clusters = compute_ng_clusters(self.bins, connectivity=8)
        # Keep selection if still valid
        if prev > 0 and any(c.id == prev for c in self.clusters):
            self.selected_cluster_id = prev
        else:
            # Re-bind from selected chip if it still sits in a cluster
            cl = self.cluster_at(self.selected_col, self.selected_row)
            self.selected_cluster_id = int(cl.id) if cl is not None else 0
        return self.clusters

    def recompute_yield(self) -> None:
        """Yield = Good / (Good+Bad) among *inspected* dies only.

        Pending dies do not count as OK (new wafer starts ~0% until inspect).
        """
        flat = self.bins.ravel()
        inspected = flat[(flat == BIN_OK) | (flat == BIN_NG)]
        if inspected.size == 0:
            self.yield_pct = 0.0
            return
        good = int(np.sum(inspected == BIN_OK))
        self.yield_pct = 100.0 * good / float(inspected.size)

    def stats(self) -> Dict[str, int]:
        flat = self.bins.ravel()
        return {
            "total_die": int(self.map_size * self.map_size),
            "inside": int(np.sum(flat != BIN_OUTSIDE)),
            "good": int(np.sum(flat == BIN_OK)),
            "bad": int(np.sum(flat == BIN_NG)),
            "pending": int(np.sum(flat == BIN_PENDING)),
            "outside": int(np.sum(flat == BIN_OUTSIDE)),
            "cluster_count": int(len(self.clusters or [])),
        }

    def breadcrumb(self) -> str:
        parts = []
        if self.date_folder:
            parts.append(self.date_folder)
        if self.lot_foup_id:
            parts.append(self.lot_foup_id)
        elif self.lot_id:
            parts.append(self.lot_id)
        parts.append(self.wafer_id or "WAFER")
        if self.selected_col > 0 and self.selected_row > 0:
            parts.append(f"Chip ({self.selected_col},{self.selected_row})")
            parts.append(f"FOV P{self.selected_fov}")
        return "  ›  ".join(parts)

    def results_dir_for_selection(self, root: Optional[str] = None) -> str:
        """Results path for active FOV.

        Prefer host-canonical path stored on the FOV point::

            .../ChipLocation/FOVLocation/Results

        Fallback (no host path yet)::

            {root}/Chip_{c}_{r}/FOV_P{n}/Results
        """
        if self.last_results_dir:
            return self.last_results_dir
        chip = self.selected_chip()
        pt = self.selected_point()
        if chip is None or pt is None:
            return ""
        if pt.results_dir:
            return pt.results_dir
        base = Path(root) if root else Path("Results")
        path = base / f"Chip_{chip.col}_{chip.row}" / f"FOV_P{pt.index}" / "Results"
        return str(path)

    def ensure_chip(self, col: int, row: int, mark_inside: bool = True) -> ChipCell:
        """Get or create a chip cell at 1-based (col,row) for host path selection."""
        col, row = int(col), int(row)
        cell = self.chip_at(col, row)
        if cell is not None:
            cell.ensure_points()
            return cell
        cell = ChipCell(
            col=col,
            row=row,
            original_bin=BIN_PENDING if mark_inside else BIN_OUTSIDE,
            final_bin=BIN_PENDING if mark_inside else BIN_OUTSIDE,
        )
        cell.ensure_points()
        self.chips[self.key(col, row)] = cell
        n = int(self.map_size)
        if mark_inside and 1 <= row <= n and 1 <= col <= n:
            # Do not invent OK — only open geometry as Pending until inspect
            if int(self.bins[row - 1, col - 1]) == BIN_OUTSIDE:
                self.bins[row - 1, col - 1] = BIN_PENDING
        return cell

    def apply_host_path(self, info: HostVolumePathInfo) -> bool:
        """Apply parsed host volume path → metadata + Wafer/Chip/FOV selection.

        Links Input.tiff location to CONTEXT maps and stores Results dir on FOV.
        Returns True if chip+FOV selection was applied.
        """
        if info is None:
            return False
        self.last_host_path = info.raw_path or info.input_path or ""
        if info.date_folder:
            self.date_folder = info.date_folder
        if info.lot_foup_id:
            self.lot_foup_id = info.lot_foup_id
        if info.lot_id:
            self.lot_id = info.lot_id
        if info.foup_id:
            self.foup_id = info.foup_id
        if info.wafer_id:
            self.wafer_id = info.wafer_id
        if info.results_dir:
            self.last_results_dir = info.results_dir

        col, row = int(info.chip_col), int(info.chip_row)
        fov = int(info.fov_index) if info.fov_index else 5
        if col <= 0 or row <= 0:
            # Keep previous selection; still store path on selected FOV if any
            pt = self.selected_point()
            if pt is not None:
                if info.input_path:
                    pt.volume_path = info.input_path
                if info.results_dir:
                    pt.results_dir = info.results_dir
            return False

        cell = self.ensure_chip(col, row, mark_inside=True)
        self.selected_col = col
        self.selected_row = row
        if 1 <= fov <= 9:
            self.selected_fov = fov
        else:
            self.selected_fov = 5
            fov = 5

        pt = cell.point(fov)
        if pt is None:
            cell.ensure_points()
            pt = cell.point(fov)
        if pt is not None:
            if info.input_path:
                pt.volume_path = info.input_path
            if info.results_dir:
                pt.results_dir = info.results_dir
        return True

    def set_fov_judge(
        self,
        col: int,
        row: int,
        fov_index: int,
        judge: int,
        ct_result: str = "",
        ai_score: Optional[float] = None,
    ) -> None:
        """Update one FOV judge and roll up chip FinalBin + wafer map bin."""
        cell = self.ensure_chip(col, row, mark_inside=True)
        pt = cell.point(int(fov_index))
        if pt is None:
            cell.ensure_points()
            pt = cell.point(int(fov_index))
        if pt is None:
            return
        pt.judge = int(judge)
        if ct_result:
            pt.ct_result = ct_result
        elif int(judge) == BIN_NG:
            pt.ct_result = "NG"
        elif int(judge) == BIN_OK:
            pt.ct_result = "OK"
        if ai_score is not None:
            pt.ai_score = float(ai_score)
        cell.recompute_counts()
        n = int(self.map_size)
        if 1 <= row <= n and 1 <= col <= n:
            # Outside stays outside unless we explicitly inspect
            if int(self.bins[row - 1, col - 1]) == BIN_OUTSIDE:
                self.bins[row - 1, col - 1] = cell.final_bin
            else:
                self.bins[row - 1, col - 1] = cell.final_bin
        self.recompute_yield()
        self.recompute_clusters()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lot_id": self.lot_id,
            "foup_id": self.foup_id,
            "lot_foup_id": self.lot_foup_id,
            "wafer_id": self.wafer_id,
            "date_folder": self.date_folder,
            "wafer_size_mm": self.wafer_size_mm,
            "die_size_mm": self.die_size_mm,
            "map_size": self.map_size,
            "bins": self.bins.tolist(),
            "chips": [c.to_dict() for c in self.chips.values()],
            "selected_col": self.selected_col,
            "selected_row": self.selected_row,
            "selected_fov": self.selected_fov,
            "yield_pct": self.yield_pct,
            "last_host_path": self.last_host_path,
            "last_results_dir": self.last_results_dir,
            "clusters": [c.to_dict() for c in (self.clusters or [])],
            "selected_cluster_id": int(self.selected_cluster_id or 0),
        }

    def to_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WaferContext":
        n = int(d.get("map_size", 25))
        bins = np.asarray(d.get("bins", np.zeros((n, n))), dtype=np.uint8)
        if bins.shape != (n, n):
            bins = np.zeros((n, n), dtype=np.uint8)
        ctx = cls(
            lot_id=str(d.get("lot_id", d.get("LotID", ""))),
            foup_id=str(d.get("foup_id", d.get("FoupID", ""))),
            lot_foup_id=str(d.get("lot_foup_id", d.get("LotFoupID", ""))),
            wafer_id=str(d.get("wafer_id", d.get("WaferID", ""))),
            date_folder=str(d.get("date_folder", d.get("Date", ""))),
            wafer_size_mm=float(d.get("wafer_size_mm", 300.0)),
            die_size_mm=float(d.get("die_size_mm", d.get("DieSize", 12.0))),
            map_size=n,
            bins=bins,
            selected_col=int(d.get("selected_col", 0)),
            selected_row=int(d.get("selected_row", 0)),
            selected_fov=int(d.get("selected_fov", 5)),
            yield_pct=float(d.get("yield_pct", 0.0)),
            last_host_path=str(d.get("last_host_path", "")),
            last_results_dir=str(d.get("last_results_dir", "")),
            selected_cluster_id=int(d.get("selected_cluster_id", 0) or 0),
        )
        if not ctx.lot_foup_id and (ctx.lot_id or ctx.foup_id):
            ctx.lot_foup_id = "_".join(x for x in (ctx.lot_id, ctx.foup_id) if x)
        chips: Dict[Tuple[int, int], ChipCell] = {}
        for item in d.get("chips", []):
            cell = ChipCell.from_dict(item)
            chips[ctx.key(cell.col, cell.row)] = cell
        # If only bins provided, synthesize chips
        if not chips:
            for r in range(n):
                for c in range(n):
                    b = int(bins[r, c])
                    if b == BIN_OUTSIDE:
                        continue
                    cell = ChipCell(
                        col=c + 1,
                        row=r + 1,
                        original_bin=b,
                        final_bin=b,
                    )
                    cell.ensure_points()
                    # Mark all points with chip bin for simple maps
                    for p in cell.points:
                        p.judge = BIN_OK if b == BIN_OK else BIN_NG
                    cell.recompute_counts()
                    chips[ctx.key(cell.col, cell.row)] = cell
        ctx.chips = chips
        ctx.recompute_yield()
        ctx.recompute_clusters()
        # Prefer payload cluster selection if still valid after recompute
        scid = int(d.get("selected_cluster_id", 0) or 0)
        if scid > 0 and any(c.id == scid for c in ctx.clusters):
            ctx.selected_cluster_id = scid
        return ctx

    @classmethod
    def from_json(cls, path: str) -> "WaferContext":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def empty_geometry(cls, map_size: int = 25, wafer_size_mm: float = 300.0) -> "WaferContext":
        """New wafer: circular die grid only — all inside dies **Pending**.

        No OK/NG until FOV inspection writes FinalBin (email draw code).
        Use this when Online starts / host path arrives without a FinalBin map.

        Die circle matches Batch Review (``die_inside_wafer`` / r²=0.24).
        """
        n = int(map_size)
        bins = circular_bin_mask(n, BIN_PENDING)

        ctx = cls(
            lot_id="",
            wafer_id="",
            wafer_size_mm=float(wafer_size_mm),
            die_size_mm=12.0,
            map_size=n,
            bins=bins,
        )
        for r in range(n):
            for c in range(n):
                if int(bins[r, c]) == BIN_OUTSIDE:
                    continue
                cell = ChipCell(
                    col=c + 1,
                    row=r + 1,
                    original_bin=BIN_PENDING,
                    final_bin=BIN_PENDING,
                )
                cell.ensure_points()  # 9 FOVs Pending
                cell.recompute_counts()
                bins[r, c] = cell.final_bin  # still Pending
                ctx.chips[ctx.key(cell.col, cell.row)] = cell

        ctx.bins = bins
        ctx.recompute_yield()  # 0% until inspected
        ctx.recompute_clusters()
        # No default “selected bad chip” — pick first inside die if any
        for r in range(n):
            for c in range(n):
                if bins[r, c] != BIN_OUTSIDE:
                    ctx.select_chip(c + 1, r + 1)
                    ctx.select_fov(5)
                    return ctx
        return ctx

    @classmethod
    def demo(cls) -> "WaferContext":
        """Synthetic map *with* fake OK/NG (email figure) — UI mock only.

        Prefer :meth:`empty_geometry` for real Online (uninspected wafer).
        """
        n = 25
        bins = circular_bin_mask(n, BIN_OK)

        # NG clusters (approx. from email figure)
        ng_sets = [
            {(16, 3), (16, 4), (17, 3), (17, 4), (15, 4)},
            {(6, 7), (7, 7), (7, 8), (6, 8), (7, 6)},
            {(10, 17), (11, 18)},
            {
                (17, 15), (17, 16), (18, 15), (18, 16), (16, 16),
                (17, 17), (18, 17), (19, 16), (16, 15), (19, 15), (18, 18),
            },
            {(4, 19), (5, 19), (5, 20), (4, 20), (5, 21), (3, 20)},
        ]
        for cluster in ng_sets:
            for c0, r0 in cluster:  # stored as (col0, row0) 0-based
                if 0 <= r0 < n and 0 <= c0 < n and bins[r0, c0] != BIN_OUTSIDE:
                    bins[r0, c0] = BIN_NG

        ctx = cls(
            lot_id="LOT240701",
            wafer_id="WAFER05",
            wafer_size_mm=300.0,
            die_size_mm=12.0,
            map_size=n,
            bins=bins,
        )
        for r in range(n):
            for c in range(n):
                b = int(bins[r, c])
                if b == BIN_OUTSIDE:
                    continue
                cell = ChipCell(
                    col=c + 1,
                    row=r + 1,
                    original_bin=b,
                    final_bin=b,
                )
                cell.ensure_points()
                if b == BIN_NG:
                    for p in cell.points:
                        p.judge = BIN_NG if p.index in (3, 6, 9) else BIN_OK
                else:
                    for p in cell.points:
                        p.judge = BIN_OK
                cell.recompute_counts()
                cell.final_bin = BIN_NG if cell.chip_ng else BIN_OK
                bins[r, c] = cell.final_bin
                ctx.chips[ctx.key(cell.col, cell.row)] = cell

        ctx.bins = bins
        ctx.recompute_yield()
        ctx.recompute_clusters()
        if ctx.chip_at(17, 16) is not None:
            ctx.select_chip(17, 16)
            ctx.select_fov(5)
        else:
            for r in range(n):
                for c in range(n):
                    if bins[r, c] != BIN_OUTSIDE:
                        ctx.select_chip(c + 1, r + 1)
                        ctx.select_fov(5)
                        break
                if ctx.selected_col:
                    break
        return ctx


def merge_host_payload(payload: Dict[str, Any]) -> WaferContext:
    """Parse a host/DB payload into WaferContext (tolerant key names)."""
    return WaferContext.from_dict(payload)
