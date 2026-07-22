"""
Batch Production Review tab — browse inspection DB, replay FOV results.

UI layout mirrors docs/mockups/batch_review_tab.html:
  left  = production browser (lot / wafer / FOV)
  center = wafer map + MES table + actions
  right  = analytics + run timeline
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QBrush, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from inno3d.core.inspection_db import InspectionDB, get_db
from inno3d.core.styles import SemiconductorTheme
from inno3d.core.wafer_context import (
    BIN_NG,
    BIN_OK,
    BIN_OUTSIDE,
    BIN_PENDING,
    ChipCell,
    FovPoint,
    WaferContext,
    circular_bin_mask,
    die_inside_wafer,
)
# Same canvases as 3D Viewer Online CONTEXT panel (visual parity)
from inno3d.widgets.context_map_panel import _ChipMapCanvas, _WaferMapCanvas


def build_wafer_context_from_db(
    db: InspectionDB,
    wafer_key: str,
    selected_col: int = 0,
    selected_row: int = 0,
    selected_fov: int = 0,
    map_size: int = 25,
) -> Optional[WaferContext]:
    """Build Online-style WaferContext from inspection DB FOV runs."""
    if not wafer_key:
        return None
    runs = db.list_fov_runs(wafer_key=wafer_key, limit=5000)
    if not runs and not wafer_key:
        return None

    # Meta from first run (or wafer_key parts: date|lot_foup|wafer)
    meta = runs[0] if runs else {}
    parts = (wafer_key or "").split("|")
    date_folder = meta.get("date_folder") or (parts[0] if len(parts) > 0 else "")
    lot_foup = meta.get("lot_foup_id") or (parts[1] if len(parts) > 1 else "")
    wafer_id = meta.get("wafer_id") or (parts[2] if len(parts) > 2 else "")
    lot_id, foup_id = "", ""
    if "_" in (lot_foup or ""):
        lot_id, foup_id = lot_foup.rsplit("_", 1)
    else:
        lot_id = lot_foup or ""

    n = int(map_size or 25)
    # Same circular die set as Online empty_geometry / CONTEXT wafer map
    bins = circular_bin_mask(n, BIN_PENDING)

    chips: Dict[tuple, ChipCell] = {}
    # Group latest judgment per (chip, fov)
    # runs ordered finished_at DESC → first is latest
    seen_fov: set = set()
    for run in runs:
        c = int(run.get("chip_col") or 0)
        r = int(run.get("chip_row") or 0)
        fi = int(run.get("fov_index") or 0)
        if c < 1 or r < 1 or fi < 1 or fi > 9:
            continue
        key = (c, r, fi)
        if key in seen_fov:
            continue
        seen_fov.add(key)
        jud = str(run.get("judgment") or "").upper()
        if jud == "NG" or int(run.get("final_bin") or 0) == BIN_NG:
            fj = BIN_NG
        elif jud == "OK" or int(run.get("final_bin") or 0) == BIN_OK:
            fj = BIN_OK
        else:
            fj = BIN_PENDING
        cell = chips.get((c, r))
        if cell is None:
            cell = ChipCell(col=c, row=r, original_bin=BIN_PENDING, final_bin=BIN_PENDING)
            cell.ensure_points()
            chips[(c, r)] = cell
        pt = cell.point(fi)
        if pt is not None:
            pt.judge = fj
            pt.volume_path = str(run.get("input_path") or "")
            pt.results_dir = str(run.get("results_dir") or "")
        cell.recompute_counts()
        # chip final_bin onto wafer bins
        if 1 <= c <= n and 1 <= r <= n:
            bins[r - 1, c - 1] = int(cell.final_bin)

    ctx = WaferContext(
        lot_id=lot_id or "",
        foup_id=foup_id or "",
        lot_foup_id=lot_foup or "",
        wafer_id=wafer_id or "",
        date_folder=date_folder or "",
        map_size=n,
        bins=bins,
        chips=chips,
        selected_col=int(selected_col or 0),
        selected_row=int(selected_row or 0),
        selected_fov=int(selected_fov or 5) or 5,
    )
    ctx.recompute_yield()
    ctx.recompute_clusters()
    # Ensure selection chip exists if requested
    if ctx.selected_col > 0 and ctx.selected_row > 0:
        if ctx.chip_at(ctx.selected_col, ctx.selected_row) is None:
            # allow pending empty chip for map selection
            cell = ChipCell(
                col=ctx.selected_col,
                row=ctx.selected_row,
                original_bin=BIN_PENDING,
                final_bin=BIN_PENDING,
            )
            cell.ensure_points()
            ctx.chips[(ctx.selected_col, ctx.selected_row)] = cell
            if 1 <= ctx.selected_col <= n and 1 <= ctx.selected_row <= n:
                if bins[ctx.selected_row - 1, ctx.selected_col - 1] == BIN_OUTSIDE:
                    bins[ctx.selected_row - 1, ctx.selected_col - 1] = BIN_PENDING
    return ctx



class WaferMapWidget(QWidget):
    """Simple grid wafer map colored by die final_bin."""

    die_clicked = pyqtSignal(int, int)  # col, row (1-based)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(140, 140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cols = 25
        self._rows = 25
        self._bins: Dict[tuple, int] = {}  # (col,row) -> bin
        self._selected: Optional[tuple] = None
        self.setToolTip("Click a die to filter FOV list")

    def set_map(self, cols: int, rows: int, bins: Dict[tuple, int], selected=None):
        self._cols = max(1, int(cols or 25))
        self._rows = max(1, int(rows or 25))
        self._bins = bins or {}
        self._selected = selected
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        side = min(w, h) - 16
        ox = (w - side) // 2
        oy = (h - side) // 2
        # wafer circle background
        p.setBrush(QColor("#0b1220"))
        p.setPen(QPen(QColor("#334155"), 2))
        p.drawEllipse(ox, oy, side, side)

        pad = side * 0.12
        grid = side - 2 * pad
        cw = grid / self._cols
        ch = grid / self._rows
        colors = {
            BIN_OK: QColor("#059669"),
            BIN_NG: QColor("#dc2626"),
            BIN_PENDING: QColor("#374151"),
            BIN_OUTSIDE: QColor("#111827"),
        }
        for r in range(1, self._rows + 1):
            for c in range(1, self._cols + 1):
                # Same circle as Online CONTEXT / build_wafer_context_from_db
                if not die_inside_wafer(c - 1, r - 1, self._cols):
                    b = BIN_OUTSIDE
                else:
                    b = self._bins.get((c, r), BIN_PENDING)
                x = ox + pad + (c - 1) * cw
                y = oy + pad + (r - 1) * ch
                p.setBrush(QBrush(colors.get(b, QColor("#374151"))))
                if self._selected == (c, r):
                    p.setPen(QPen(QColor(SemiconductorTheme.ACCENT_PRIMARY), 2))
                else:
                    p.setPen(QPen(QColor("#1f2937"), 0.5))
                p.drawRect(int(x + 1), int(y + 1), max(1, int(cw - 2)), max(1, int(ch - 2)))
        p.end()

    def mousePressEvent(self, event):
        w, h = self.width(), self.height()
        side = min(w, h) - 16
        ox = (w - side) // 2
        oy = (h - side) // 2
        pad = side * 0.12
        grid = side - 2 * pad
        x = event.x() - ox - pad
        y = event.y() - oy - pad
        if x < 0 or y < 0 or x >= grid or y >= grid:
            return
        c = int(x / grid * self._cols) + 1
        r = int(y / grid * self._rows) + 1
        self._selected = (c, r)
        self.update()
        self.die_clicked.emit(c, r)


class FovMapWidget(QWidget):
    """9-point FOV map on selected chip (P1..P9 row-major)."""

    fov_clicked = pyqtSignal(int)  # 1..9

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(140, 140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._status: Dict[int, str] = {}  # fov_index -> OK/NG/PENDING
        self._selected = 0
        self.setToolTip("Click FOV point P1–P9 (selected chip)")

    def set_fov_status(self, status: Dict[int, str], selected: int = 0):
        self._status = {int(k): str(v).upper() for k, v in (status or {}).items()}
        self._selected = int(selected or 0)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        side = min(w, h) - 12
        ox, oy = (w - side) // 2, (h - side) // 2
        p.setBrush(QColor("#0b1220"))
        p.setPen(QPen(QColor("#334155"), 2))
        p.drawRoundedRect(ox, oy, side, side, 10, 10)

        pad = side * 0.12
        cell = (side - 2 * pad) / 3
        colors = {
            "OK": QColor("#059669"),
            "NG": QColor("#dc2626"),
            "PENDING": QColor("#374151"),
            "ERROR": QColor("#b45309"),
            "": QColor("#1f2937"),
        }
        for i in range(9):
            r, c = divmod(i, 3)
            idx = i + 1
            x = ox + pad + c * cell
            y = oy + pad + r * cell
            st = self._status.get(idx, "PENDING" if idx in self._status else "")
            if idx not in self._status:
                st = "PENDING"
            p.setBrush(QBrush(colors.get(st, colors[""])))
            if self._selected == idx:
                p.setPen(QPen(QColor(SemiconductorTheme.ACCENT_PRIMARY), 2))
            else:
                p.setPen(QPen(QColor("#1f2937"), 1))
            p.drawRoundedRect(int(x + 3), int(y + 3), int(cell - 6), int(cell - 6), 6, 6)
            p.setPen(QColor("#e5e7eb"))
            p.setFont(QFont("Segoe UI", 10, QFont.Bold))
            p.drawText(int(x), int(y), int(cell), int(cell), Qt.AlignCenter, f"P{idx}")
        p.end()

    def mousePressEvent(self, event):
        w, h = self.width(), self.height()
        side = min(w, h) - 12
        ox, oy = (w - side) // 2, (h - side) // 2
        pad = side * 0.12
        cell = (side - 2 * pad) / 3
        x = event.x() - ox - pad
        y = event.y() - oy - pad
        if x < 0 or y < 0 or x >= 3 * cell or y >= 3 * cell:
            return
        c = int(x / cell)
        r = int(y / cell)
        if 0 <= c < 3 and 0 <= r < 3:
            idx = r * 3 + c + 1
            self._selected = idx
            self.update()
            self.fov_clicked.emit(idx)


def _to_uint8_gray(plane: np.ndarray) -> np.ndarray:
    a = np.asarray(plane)
    if a.ndim > 2:
        a = a[..., 0] if a.shape[-1] <= 4 else a[0]
    a = a.astype(np.float32)
    lo, hi = float(np.percentile(a, 1)), float(np.percentile(a, 99))
    if hi <= lo:
        hi = lo + 1.0
    g = np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return g


def _overlay_masks(gray: np.ndarray, bump=None, void=None) -> np.ndarray:
    """RGB uint8: gray + green bump + red void."""
    g = gray
    rgb = np.stack([g, g, g], axis=-1).astype(np.float32)
    if bump is not None:
        m = np.asarray(bump) > 0
        if m.shape == g.shape:
            rgb[m, 1] = np.minimum(255, rgb[m, 1] * 0.35 + 180)
            rgb[m, 0] = rgb[m, 0] * 0.35
            rgb[m, 2] = rgb[m, 2] * 0.35
    if void is not None:
        m = np.asarray(void) > 0
        if m.shape == g.shape:
            rgb[m, 0] = np.minimum(255, rgb[m, 0] * 0.35 + 200)
            rgb[m, 1] = rgb[m, 1] * 0.35
            rgb[m, 2] = rgb[m, 2] * 0.45
    return rgb.astype(np.uint8)


def _numpy_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    if rgb.ndim == 2:
        h, w = rgb.shape
        img = QImage(rgb.data, w, h, w, QImage.Format_Grayscale8).copy()
    else:
        h, w, _ = rgb.shape
        cont = np.ascontiguousarray(rgb)
        img = QImage(cont.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(img)


class SliceViewLabel(QLabel):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self.setMinimumSize(100, 90)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"background:#050a14;border:1px solid {SemiconductorTheme.BORDER_DEFAULT};border-radius:6px;"
        )
        self.setText(title)
        self._pix = None
        self._resize_timer = None
        try:
            from PyQt5.QtCore import QTimer
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.setInterval(40)
            self._resize_timer.timeout.connect(self._update)
        except Exception:
            pass

    def set_plane(self, pixmap: Optional[QPixmap]):
        self._pix = pixmap
        self._update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Debounce scale-on-resize to avoid splitter-drag jitter
        if self._resize_timer is not None:
            self._resize_timer.start()
        else:
            self._update()

    def _update(self):
        if self._pix is None or self._pix.isNull():
            self.setText(self._title)
            return
        # Fast transform while dragging; Smooth only when idle (timer)
        mode = Qt.SmoothTransformation if (
            self._resize_timer is None or not self._resize_timer.isActive()
        ) else Qt.FastTransformation
        # If called from timer timeout, timer is not active → Smooth
        if self._resize_timer is not None and self.sender() is self._resize_timer:
            mode = Qt.SmoothTransformation
        scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatio, mode)
        self.setPixmap(scaled)


class VolumeLoadThread(QThread):
    finished = pyqtSignal(object, object, object, str)  # vol, bump, void, err

    def __init__(self, vol_path, bump_path, void_path):
        super().__init__()
        self.vol_path = vol_path
        self.bump_path = bump_path
        self.void_path = void_path

    def run(self):
        try:
            import tifffile

            vol = bump = void = None
            if self.vol_path and os.path.isfile(self.vol_path):
                vol = np.asarray(tifffile.imread(self.vol_path))
            if self.bump_path and os.path.isfile(self.bump_path):
                bump = np.asarray(tifffile.imread(self.bump_path))
            if self.void_path and os.path.isfile(self.void_path):
                void = np.asarray(tifffile.imread(self.void_path))
            if vol is not None and vol.ndim == 2:
                vol = vol[np.newaxis]
            if vol is not None and vol.ndim == 4:
                vol = vol[..., 0] if vol.shape[-1] <= 4 else vol[:, 0]
            if bump is not None and bump.ndim == 2:
                bump = bump[np.newaxis]
            if void is not None and void.ndim == 2:
                void = void[np.newaxis]
            self.finished.emit(vol, bump, void, "")
        except Exception as e:
            self.finished.emit(None, None, None, str(e))


class MprReplayPanel(QFrame):
    """Lightweight MPR: XY + XZ + YZ with independent Z/Y/X sliders (no 3D/MIP)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("BatchPanel")
        self.setMinimumWidth(280)
        self._vol = None
        self._bump = None
        self._void = None
        self._z = 0
        self._y = 0
        self._x = 0
        self._load_thread = None
        self._render_timer = None
        try:
            from PyQt5.QtCore import QTimer
            self._render_timer = QTimer(self)
            self._render_timer.setSingleShot(True)
            self._render_timer.setInterval(30)
            self._render_timer.timeout.connect(self._render_now)
        except Exception:
            pass

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        head = QHBoxLayout()
        self.title = QLabel("FOV REPLAY · XY / XZ / YZ + MASK")
        self.title.setObjectName("BatchSection")
        head.addWidget(self.title)
        head.addStretch()
        self.chk_bump = QCheckBox("Bump")
        self.chk_bump.setChecked(True)
        self.chk_void = QCheckBox("Void")
        self.chk_void.setChecked(True)
        self.chk_bump.stateChanged.connect(self._schedule_render)
        self.chk_void.stateChanged.connect(self._schedule_render)
        head.addWidget(self.chk_bump)
        head.addWidget(self.chk_void)
        lay.addLayout(head)

        self.status = QLabel("Select a FOV run to load Results")
        self.status.setObjectName("BatchReviewMuted")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        # 1 row × 3 panels (classic MPR strip — no MIP/3D)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.view_xy = SliceViewLabel("XY (Z-slice)")
        self.view_xz = SliceViewLabel("XZ (Y-slice)")
        self.view_yz = SliceViewLabel("YZ (X-slice)")
        for v in (self.view_xy, self.view_xz, self.view_yz):
            v.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            row.addWidget(v, 1)
        lay.addLayout(row, 1)

        # Independent sliders for each plane
        for axis, attr, handler in (
            ("Z · axial", "slider_z", self._on_z),
            ("Y · coronal", "slider_y", self._on_y),
            ("X · sagittal", "slider_x", self._on_x),
        ):
            sl = QHBoxLayout()
            sl.setSpacing(6)
            lab = QLabel(axis)
            lab.setFixedWidth(72)
            lab.setObjectName("BatchReviewMuted")
            sl.addWidget(lab)
            slider = QSlider(Qt.Horizontal)
            slider.valueChanged.connect(handler)
            setattr(self, attr, slider)
            sl.addWidget(slider, 1)
            lbl = QLabel("0 / 0")
            lbl.setMinimumWidth(64)
            setattr(self, attr.replace("slider_", "lbl_"), lbl)
            sl.addWidget(lbl)
            lay.addLayout(sl)

    def clear(self):
        self._vol = self._bump = self._void = None
        self.view_xy.set_plane(None)
        self.view_yz.set_plane(None)
        self.view_xz.set_plane(None)
        self.status.setText("No volume loaded")

    def load_from_paths(self, vol_path: str, bump_path: str = "", void_path: str = ""):
        self.status.setText("Loading volume / masks…")
        if self._load_thread and self._load_thread.isRunning():
            try:
                self._load_thread.finished.disconnect(self._on_loaded)
            except Exception:
                pass
        self._load_thread = VolumeLoadThread(vol_path, bump_path, void_path)
        self._load_thread.finished.connect(self._on_loaded)
        self._load_thread.start()

    def _on_loaded(self, vol, bump, void, err):
        if err:
            self.status.setText(f"Load error: {err}")
            self.clear()
            return
        if vol is None:
            self.status.setText("Volume file not found on disk (path only in DB)")
            self.clear()
            return
        self._vol = np.ascontiguousarray(vol)
        self._bump = None if bump is None else np.ascontiguousarray(bump)
        self._void = None if void is None else np.ascontiguousarray(void)
        # Align mask shapes once
        self._bump = self._align_mask(self._bump)
        self._void = self._align_mask(self._void)

        z, y, x = self._vol.shape[:3]
        self._z, self._y, self._x = z // 2, y // 2, x // 2
        for slider, val, hi in (
            (self.slider_z, self._z, z - 1),
            (self.slider_y, self._y, y - 1),
            (self.slider_x, self._x, x - 1),
        ):
            slider.blockSignals(True)
            slider.setRange(0, max(0, hi))
            slider.setValue(val)
            slider.blockSignals(False)
        mask_txt = []
        if self._bump is not None:
            mask_txt.append("bump")
        if self._void is not None:
            mask_txt.append("void")
        self.status.setText(
            f"Volume Z×Y×X = {z}×{y}×{x}"
            + (f" · masks: {', '.join(mask_txt)}" if mask_txt else " · no masks")
            + "  ·  drag Z / Y / X sliders below"
        )
        self._render_now()

    def _align_mask(self, mask):
        if mask is None or self._vol is None:
            return mask
        m = np.asarray(mask)
        if m.shape == self._vol.shape[:3]:
            return m
        try:
            out = np.zeros(self._vol.shape[:3], dtype=m.dtype)
            z = min(m.shape[0], out.shape[0])
            y = min(m.shape[1], out.shape[1])
            x = min(m.shape[2], out.shape[2])
            out[:z, :y, :x] = m[:z, :y, :x]
            return out
        except Exception:
            return None

    def _on_z(self, v):
        self._z = int(v)
        self._schedule_render()

    def _on_y(self, v):
        self._y = int(v)
        self._schedule_render()

    def _on_x(self, v):
        self._x = int(v)
        self._schedule_render()

    def _schedule_render(self):
        if self._render_timer is not None:
            self._render_timer.start()
        else:
            self._render_now()

    def _plane_mask(self, mask, axis, idx):
        if mask is None:
            return None
        m = np.asarray(mask)
        if axis == 0:
            return m[idx]
        if axis == 1:
            return m[:, idx, :]
        return m[:, :, idx]

    def _render_now(self):
        if self._vol is None:
            return
        z, y, x = self._vol.shape[:3]
        self._z = int(np.clip(self._z, 0, z - 1))
        self._y = int(np.clip(self._y, 0, y - 1))
        self._x = int(np.clip(self._x, 0, x - 1))
        self.lbl_z.setText(f"{self._z} / {z - 1}")
        self.lbl_y.setText(f"{self._y} / {y - 1}")
        self.lbl_x.setText(f"{self._x} / {x - 1}")
        show_b = self.chk_bump.isChecked()
        show_v = self.chk_void.isChecked()

        def pack(plane, mb, mv):
            g = _to_uint8_gray(plane)
            rgb = _overlay_masks(
                g,
                mb if show_b else None,
                mv if show_v else None,
            )
            return _numpy_to_qpixmap(rgb)

        # XY: fixed Z  → shape (Y, X)
        self.view_xy.set_plane(
            pack(
                self._vol[self._z],
                self._plane_mask(self._bump, 0, self._z),
                self._plane_mask(self._void, 0, self._z),
            )
        )
        # XZ: fixed Y  → shape (Z, X)  — use slider Y
        self.view_xz.set_plane(
            pack(
                self._vol[:, self._y, :],
                self._plane_mask(self._bump, 1, self._y),
                self._plane_mask(self._void, 1, self._y),
            )
        )
        # YZ: fixed X  → shape (Z, Y)  — use slider X
        self.view_yz.set_plane(
            pack(
                self._vol[:, :, self._x],
                self._plane_mask(self._bump, 2, self._x),
                self._plane_mask(self._void, 2, self._x),
            )
        )


class MiniBarChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self._values: List[float] = []
        self._labels: List[str] = []

    def set_data(self, values: List[float], labels: Optional[List[str]] = None):
        self._values = list(values or [])
        self._labels = list(labels or [])
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#050a14"))
        if not self._values:
            p.setPen(QColor("#64748b"))
            p.drawText(self.rect(), Qt.AlignCenter, "No data")
            p.end()
            return
        n = len(self._values)
        mx = max(self._values) or 1.0
        margin = 8
        gap = 4
        usable_w = self.width() - 2 * margin
        usable_h = self.height() - 20
        bw = max(4, (usable_w - gap * (n - 1)) / n)
        for i, v in enumerate(self._values):
            h = int(usable_h * (v / mx))
            x = margin + i * (bw + gap)
            y = self.height() - 14 - h
            color = QColor("#3b82f6") if v >= 90 else (QColor("#f87171") if v < 80 else QColor("#22d3ee"))
            p.fillRect(int(x), int(y), int(bw), h, color)
            if i < len(self._labels):
                p.setPen(QColor("#64748b"))
                p.setFont(QFont("Segoe UI", 7))
                p.drawText(int(x), self.height() - 2, int(bw), 12, Qt.AlignCenter, self._labels[i])
        p.end()


class MiniHistChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self._counts: List[int] = []

    def set_counts(self, counts: List[int]):
        self._counts = list(counts or [])
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#050a14"))
        if not self._counts or max(self._counts) == 0:
            p.setPen(QColor("#64748b"))
            p.drawText(self.rect(), Qt.AlignCenter, "No objects")
            p.end()
            return
        mx = max(self._counts)
        n = len(self._counts)
        margin = 6
        gap = 2
        usable_w = self.width() - 2 * margin
        usable_h = self.height() - 10
        bw = max(2, (usable_w - gap * (n - 1)) / n)
        for i, c in enumerate(self._counts):
            h = int(usable_h * (c / mx))
            x = margin + i * (bw + gap)
            y = self.height() - 4 - h
            # last bins (high void ratio) red-ish
            color = QColor("#f87171") if i >= n * 0.7 else QColor("#22d3ee")
            p.fillRect(int(x), int(y), max(1, int(bw)), h, color)
        p.end()


class BatchReviewTab(QWidget):
    """Production batch review against SQLite inspection catalog."""

    open_in_viewer = pyqtSignal(dict)  # run payload for future deep-link

    def __init__(self, parent=None, db: Optional[InspectionDB] = None):
        super().__init__(parent)
        self.db = db or get_db()
        self._current_wafer_key = ""
        self._current_run_id = ""
        self._runs_cache: List[Dict[str, Any]] = []
        self._build_ui()
        self._apply_theme()
        self.refresh_all()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header strip
        header = QFrame()
        header.setObjectName("BatchReviewHeader")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(12, 8, 12, 8)
        title = QLabel("BATCH PRODUCTION REVIEW")
        title.setObjectName("BatchReviewTitle")
        hl.addWidget(title)
        self.db_path_label = QLabel("")
        self.db_path_label.setObjectName("BatchReviewMuted")
        hl.addStretch()
        hl.addWidget(self.db_path_label)
        root.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.setChildrenCollapsible(False)
        splitter.setOpaqueResize(True)
        self._main_splitter = splitter
        root.addWidget(splitter, 1)

        # LEFT
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(8, 8, 8, 8)
        left_l.setSpacing(8)

        left_l.addWidget(self._section_label("PRODUCTION BROWSER"))

        self.lot_combo = QComboBox()
        self.lot_combo.currentIndexChanged.connect(self._on_lot_changed)
        left_l.addWidget(self._field("Date / Lot", self.lot_combo))

        self.wafer_combo = QComboBox()
        self.wafer_combo.currentIndexChanged.connect(self._on_wafer_changed)
        left_l.addWidget(self._field("Wafer", self.wafer_combo))

        self.judge_combo = QComboBox()
        self.judge_combo.addItem("All judgments", "")
        self.judge_combo.addItem("OK only", "OK")
        self.judge_combo.addItem("NG only", "NG")
        self.judge_combo.addItem("ERROR", "ERROR")
        self.judge_combo.currentIndexChanged.connect(self._reload_fov_list)
        left_l.addWidget(self._field("Judgment filter", self.judge_combo))

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Chip col,row / path / FOV…")
        self.search_edit.returnPressed.connect(self._reload_fov_list)
        left_l.addWidget(self._field("Search", self.search_edit))

        # KPIs
        kpi_grid = QGridLayout()
        self.kpi_fov = self._kpi_card("0", "FOV inspected")
        self.kpi_yield = self._kpi_card("—", "Yield (OK FOV)")
        self.kpi_ng = self._kpi_card("0", "NG FOV")
        self.kpi_obj = self._kpi_card("0", "Objects (MES)")
        kpi_grid.addWidget(self.kpi_fov, 0, 0)
        kpi_grid.addWidget(self.kpi_yield, 0, 1)
        kpi_grid.addWidget(self.kpi_ng, 1, 0)
        kpi_grid.addWidget(self.kpi_obj, 1, 1)
        left_l.addLayout(kpi_grid)

        self.wafer_list = QListWidget()
        self.wafer_list.itemClicked.connect(self._on_wafer_list_clicked)
        left_l.addWidget(self.wafer_list, 1)

        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_all)
        self.btn_seed = QPushButton("Seed demo")
        self.btn_seed.setToolTip("Insert demo FOV runs if DB is empty")
        self.btn_seed.clicked.connect(self._seed_demo)
        self.btn_open_db = QPushButton("DB folder")
        self.btn_open_db.clicked.connect(self._open_db_folder)
        btn_row.addWidget(self.btn_refresh)
        btn_row.addWidget(self.btn_seed)
        btn_row.addWidget(self.btn_open_db)
        left_l.addLayout(btn_row)

        splitter.addWidget(left)

        # CENTER
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.setSpacing(8)

        self.breadcrumb = QLabel("REVIEW  ›  (select a wafer / FOV)")
        self.breadcrumb.setObjectName("BatchReviewBreadcrumb")
        self.breadcrumb.setWordWrap(True)
        cl.addWidget(self.breadcrumb)

        # Row 1: Wafer map | FOV map | MPR replay  (matches mockup)
        maps_row = QSplitter(Qt.Horizontal)
        maps_row.setChildrenCollapsible(False)
        maps_row.setOpaqueResize(True)
        maps_row.setHandleWidth(3)
        self._maps_splitter = maps_row

        # Same canvases as 3D Viewer Online CONTEXT (Innometry 25×25 + P1–P9)
        map_panel = QFrame()
        map_panel.setObjectName("BatchPanel")
        map_panel.setMinimumWidth(200)
        mpl = QVBoxLayout(map_panel)
        mpl.setContentsMargins(6, 6, 6, 6)
        mpl.addWidget(self._section_label("WAFER MAP"))
        self.wafer_map = _WaferMapCanvas()
        self.wafer_map.chip_clicked.connect(self._on_die_clicked)
        mpl.addWidget(self.wafer_map, 1)
        maps_row.addWidget(map_panel)

        fovmap_panel = QFrame()
        fovmap_panel.setObjectName("BatchPanel")
        fovmap_panel.setMinimumWidth(180)
        fml = QVBoxLayout(fovmap_panel)
        fml.setContentsMargins(6, 6, 6, 6)
        fml.addWidget(self._section_label("CHIP FOV MAP · P1–P9"))
        self.chip_label = QLabel("Select a die on wafer map")
        self.chip_label.setObjectName("BatchReviewMuted")
        fml.addWidget(self.chip_label)
        self.fov_map = _ChipMapCanvas()
        self.fov_map.fov_clicked.connect(self._on_fov_map_clicked)
        fml.addWidget(self.fov_map, 1)
        maps_row.addWidget(fovmap_panel)
        self._wafer_ctx: Optional[WaferContext] = None

        self.mpr_panel = MprReplayPanel()
        maps_row.addWidget(self.mpr_panel)
        # Prefer MPR wide; maps compact
        maps_row.setStretchFactor(0, 1)
        maps_row.setStretchFactor(1, 1)
        maps_row.setStretchFactor(2, 3)
        maps_row.setSizes([180, 180, 480])
        cl.addWidget(maps_row, 4)

        # Row 2: FOV run table
        cl.addWidget(self._section_label("FOV RUNS · SELECT TO REPLAY"))
        self.fov_table = QTableWidget(0, 8)
        self.fov_table.setHorizontalHeaderLabels(
            ["Run", "Chip", "FOV", "Judgment", "Objects", "NG", "Total s", "Finished"]
        )
        self.fov_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.fov_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.fov_table.setAlternatingRowColors(True)
        self.fov_table.verticalHeader().setVisible(False)
        self.fov_table.horizontalHeader().setStretchLastSection(True)
        self.fov_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.fov_table.itemSelectionChanged.connect(self._on_fov_selected)
        self.fov_table.setMaximumHeight(160)
        cl.addWidget(self.fov_table)

        cl.addWidget(self._section_label("MES OBJECTS · SELECTED FOV"))
        self.mes_table = QTableWidget(0, 10)
        self.mes_table.setHorizontalHeaderLabels(
            ["#", "Layer", "Bump ID", "B.H", "B.V", "V.V", "Ratio%", "Judgment", "Gap X", "Gap Y"]
        )
        self.mes_table.setAlternatingRowColors(True)
        self.mes_table.verticalHeader().setVisible(False)
        self.mes_table.horizontalHeader().setStretchLastSection(True)
        self.mes_table.setMaximumHeight(160)
        cl.addWidget(self.mes_table)

        actions = QHBoxLayout()
        self.btn_reload_mpr = QPushButton("Reload MPR")
        self.btn_reload_mpr.clicked.connect(self._reload_mpr_for_current)
        self.btn_open_results = QPushButton("Open Results folder")
        self.btn_open_results.clicked.connect(self._open_results)
        self.btn_open_csv = QPushButton("Open MES CSV")
        self.btn_open_csv.clicked.connect(self._open_mes_csv)
        self.btn_export = QPushButton("Export wafer FOV list…")
        self.btn_export.clicked.connect(self._export_wafer_csv)
        self.btn_emit_viewer = QPushButton("Open full Viewer")
        self.btn_emit_viewer.setToolTip(
            "Load into 3D VIEWER with Online parity:\n"
            "· Volume + bump/void mask overlay\n"
            "· MES Object Stats table (from DB / object_statistics.csv)\n"
            "· B2B Boundary table (from boundary_gap CSV)\n"
            "· Click MES/B2B row → jump / highlight on MPR & 3D"
        )
        self.btn_emit_viewer.clicked.connect(self._emit_open_viewer)
        actions.addWidget(self.btn_reload_mpr)
        actions.addWidget(self.btn_open_results)
        actions.addWidget(self.btn_open_csv)
        actions.addWidget(self.btn_export)
        actions.addWidget(self.btn_emit_viewer)
        actions.addStretch()
        note = QLabel(
            "MPR = quick replay · Open full Viewer = volume+mask+MES+B2B+mapping (Online parity)"
        )
        note.setObjectName("BatchReviewMuted")
        actions.addWidget(note)
        cl.addLayout(actions)

        splitter.addWidget(center)

        # RIGHT
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)
        rl.setSpacing(8)
        rl.addWidget(self._section_label("LOT / WAFER ANALYTICS"))

        rl.addWidget(QLabel("Yield by FOV point (P1–P9)"))
        self.bar_chart = MiniBarChart()
        rl.addWidget(self.bar_chart)

        rl.addWidget(QLabel("Void ratio distribution (%)"))
        self.hist_chart = MiniHistChart()
        rl.addWidget(self.hist_chart)

        rl.addWidget(self._section_label("RUN TIMELINE · THIS FOV"))
        self.timeline_list = QListWidget()
        rl.addWidget(self.timeline_list, 1)

        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setObjectName("BatchReviewMuted")
        rl.addWidget(self.detail_label)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 780, 300])
        left.setMinimumWidth(220)
        left.setMaximumWidth(360)
        right.setMinimumWidth(220)
        right.setMaximumWidth(380)

    def _section_label(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("BatchSection")
        return lab

    def _field(self, label: str, widget: QWidget) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(2)
        lab = QLabel(label)
        lab.setObjectName("BatchReviewMuted")
        l.addWidget(lab)
        l.addWidget(widget)
        return w

    def _kpi_card(self, value: str, label: str) -> QFrame:
        f = QFrame()
        f.setObjectName("BatchKpi")
        lay = QVBoxLayout(f)
        lay.setContentsMargins(8, 8, 8, 8)
        v = QLabel(value)
        v.setObjectName("BatchKpiValue")
        lab = QLabel(label)
        lab.setObjectName("BatchReviewMuted")
        lay.addWidget(v)
        lay.addWidget(lab)
        f._value_label = v  # type: ignore[attr-defined]
        return f

    def _set_kpi(self, card: QFrame, value: str, ok_color: bool = False, ng_color: bool = False):
        lab = getattr(card, "_value_label", None)
        if lab is None:
            return
        lab.setText(value)
        # Compact KPIs — large digits caused left panel thrash/jitter when resizing
        if ng_color:
            lab.setStyleSheet(f"color: {SemiconductorTheme.ACCENT_ERROR}; font-size: 13pt; font-weight: 700;")
        elif ok_color:
            lab.setStyleSheet(f"color: {SemiconductorTheme.ACCENT_SUCCESS}; font-size: 13pt; font-weight: 700;")
        else:
            lab.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 13pt; font-weight: 700;")

    def _apply_theme(self):
        self.setStyleSheet(
            f"""
            QWidget {{ background: {SemiconductorTheme.BG_DARK}; color: {SemiconductorTheme.TEXT_PRIMARY}; }}
            QFrame#BatchReviewHeader {{
                background: {SemiconductorTheme.BG_MEDIUM};
                border-bottom: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QLabel#BatchReviewTitle {{
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                font-weight: 800; font-size: 11pt; letter-spacing: 1px;
            }}
            QLabel#BatchSection {{
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                font-size: 9pt; font-weight: 700; letter-spacing: 0.6px;
            }}
            QLabel#BatchReviewMuted, QLabel#BatchReviewBreadcrumb {{
                color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt;
            }}
            QFrame#BatchPanel, QFrame#BatchKpi {{
                background: {SemiconductorTheme.BG_PANEL};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 8px;
            }}
            QFrame#BatchKpi {{ padding: 2px; }}
            QComboBox, QLineEdit, QListWidget, QTableWidget {{
                background: {SemiconductorTheme.BG_MEDIUM};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 6px;
                padding: 3px;
                font-size: 9pt;
            }}
            QPushButton {{
                background: {SemiconductorTheme.BG_LIGHT};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 6px;
                padding: 5px 8px;
                font-weight: 600;
                font-size: 8.5pt;
            }}
            QPushButton:hover {{
                border-color: {SemiconductorTheme.ACCENT_PRIMARY};
                color: {SemiconductorTheme.ACCENT_PRIMARY};
            }}
            QHeaderView::section {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_SECONDARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                padding: 3px;
                font-size: 8pt;
            }}
            QSplitter::handle {{
                background: {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QSplitter::handle:hover {{
                background: {SemiconductorTheme.ACCENT_PRIMARY};
            }}
            """
        )
        self.db_path_label.setText(f"DB · {self.db.db_path}")

    # ------------------------------------------------------------------ data
    def refresh_all(self):
        """Full DB reload. Avoid calling during splitter drag."""
        self.setUpdatesEnabled(False)
        try:
            self.db_path_label.setText(f"DB · {self.db.db_path}")
            self._reload_lots()
            self._reload_wafer_list()
            self._reload_fov_list()
            self._reload_kpis()
            self._reload_charts()
        finally:
            self.setUpdatesEnabled(True)

    def _reload_lots(self):
        self.lot_combo.blockSignals(True)
        cur = self.lot_combo.currentData()
        self.lot_combo.clear()
        self.lot_combo.addItem("All lots", ("", ""))
        for lot in self.db.list_lots():
            label = f"{lot.get('date_folder') or '—'} · {lot.get('lot_foup_id') or '—'}"
            self.lot_combo.addItem(label, (lot.get("date_folder") or "", lot.get("lot_foup_id") or ""))
        # restore
        if cur:
            for i in range(self.lot_combo.count()):
                if self.lot_combo.itemData(i) == cur:
                    self.lot_combo.setCurrentIndex(i)
                    break
        self.lot_combo.blockSignals(False)
        self._reload_wafer_combo()

    def _reload_wafer_combo(self):
        date_folder, lot = ("", "")
        data = self.lot_combo.currentData()
        if data:
            date_folder, lot = data
        self.wafer_combo.blockSignals(True)
        self.wafer_combo.clear()
        wafers = self.db.list_wafers(date_folder=date_folder, lot_foup_id=lot)
        for w in wafers:
            y = w.get("yield_pct") or 0
            label = f"{w.get('wafer_id') or '?'} · {w.get('n_fov', 0)} FOV · {y:.1f}%"
            self.wafer_combo.addItem(label, w.get("wafer_key"))
        self.wafer_combo.blockSignals(False)
        if self.wafer_combo.count():
            self._current_wafer_key = self.wafer_combo.currentData() or ""
        else:
            self._current_wafer_key = ""

    def _reload_wafer_list(self):
        self.wafer_list.clear()
        date_folder, lot = ("", "")
        data = self.lot_combo.currentData()
        if data:
            date_folder, lot = data
        for w in self.db.list_wafers(date_folder=date_folder, lot_foup_id=lot):
            y = w.get("yield_pct") or 0
            text = (
                f"{w.get('wafer_id') or '?'}\n"
                f"{w.get('n_fov', 0)} FOV · Yield {y:.1f}% · NG {w.get('n_ng', 0)}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, w.get("wafer_key"))
            if w.get("wafer_key") == self._current_wafer_key:
                item.setSelected(True)
            self.wafer_list.addItem(item)

    def _reload_kpis(self):
        k = self.db.global_kpis(self._current_wafer_key)
        self._set_kpi(self.kpi_fov, str(k.get("n_fov", 0)))
        self._set_kpi(self.kpi_yield, f"{k.get('yield_pct', 0):.1f}%", ok_color=True)
        self._set_kpi(self.kpi_ng, str(k.get("n_ng", 0)), ng_color=True)
        self._set_kpi(self.kpi_obj, f"{int(k.get('n_objects') or 0):,}")

    def _reload_fov_list(self):
        judge = self.judge_combo.currentData() or ""
        search = self.search_edit.text().strip()
        # optional die filter from map
        die = getattr(self, "_die_filter", None)
        runs = self.db.list_fov_runs(
            wafer_key=self._current_wafer_key or "",
            judgment=judge or "",
            search=search,
            limit=800,
        )
        if die:
            c, r = die
            runs = [x for x in runs if int(x.get("chip_col") or 0) == c and int(x.get("chip_row") or 0) == r]
        self._runs_cache = runs
        self.fov_table.setRowCount(len(runs))
        for i, run in enumerate(runs):
            cells = [
                str(run.get("run_id") or "")[:10],
                f"({run.get('chip_col')},{run.get('chip_row')})",
                f"P{run.get('fov_index')}" if run.get("fov_index") else (run.get("fov_folder") or "—"),
                str(run.get("judgment") or "—"),
                str(run.get("n_objects") or 0),
                str(run.get("n_ng") or 0),
                f"{float(run.get('total_sec') or 0):.1f}",
                str(run.get("finished_at") or "")[:19],
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col == 0:
                    item.setData(Qt.UserRole, run.get("run_id"))
                if col == 3:
                    j = str(run.get("judgment") or "").upper()
                    if j == "NG":
                        item.setForeground(QColor(SemiconductorTheme.ACCENT_ERROR))
                    elif j == "OK":
                        item.setForeground(QColor(SemiconductorTheme.ACCENT_SUCCESS))
                self.fov_table.setItem(i, col, item)

        # Rebuild Online-style WaferContext (shared canvas with 3D Viewer)
        sc = die[0] if die else 0
        sr = die[1] if die else 0
        self._refresh_context_maps(selected_col=sc, selected_row=sr, selected_fov=0)

        ctx = self._wafer_ctx
        if ctx is not None:
            crumb = ctx.breadcrumb() if hasattr(ctx, "breadcrumb") else ""
            self.breadcrumb.setText(
                f"REVIEW  ›  {crumb or self._current_wafer_key or '—'}  ·  "
                f"{len(runs)} FOV run(s)  ·  yield {ctx.yield_pct:.1f}%"
            )
        else:
            self.breadcrumb.setText(
                f"REVIEW  ›  {self._current_wafer_key or '—'}  ·  {len(runs)} FOV run(s)"
            )

    def _reload_charts(self):
        pts = self.db.fov_point_yield(self._current_wafer_key or "")
        vals = [0.0] * 9
        labels = [f"P{i}" for i in range(1, 10)]
        for p in pts:
            idx = int(p.get("fov_index") or 0)
            if 1 <= idx <= 9:
                vals[idx - 1] = float(p.get("yield_pct") or 0)
        self.bar_chart.set_data(vals, labels)
        hist = self.db.ratio_histogram(self._current_wafer_key or "", bins=12)
        self.hist_chart.set_counts([c for _, c in hist])

    def _on_lot_changed(self):
        self._reload_wafer_combo()
        self._reload_wafer_list()
        self._die_filter = None
        self._reload_fov_list()
        self._reload_kpis()
        self._reload_charts()

    def _on_wafer_changed(self):
        self._current_wafer_key = self.wafer_combo.currentData() or ""
        self._die_filter = None
        self._reload_wafer_list()
        self._reload_fov_list()
        self._reload_kpis()
        self._reload_charts()

    def _on_wafer_list_clicked(self, item: QListWidgetItem):
        key = item.data(Qt.UserRole)
        if not key:
            return
        self._current_wafer_key = key
        for i in range(self.wafer_combo.count()):
            if self.wafer_combo.itemData(i) == key:
                self.wafer_combo.blockSignals(True)
                self.wafer_combo.setCurrentIndex(i)
                self.wafer_combo.blockSignals(False)
                break
        self._die_filter = None
        self._reload_fov_list()
        self._reload_kpis()
        self._reload_charts()

    def _refresh_context_maps(
        self,
        selected_col: int = 0,
        selected_row: int = 0,
        selected_fov: int = 0,
    ):
        """Push DB-derived WaferContext into Online-identical map canvases."""
        die = getattr(self, "_die_filter", None)
        if selected_col <= 0 and die:
            selected_col, selected_row = die
        ctx = build_wafer_context_from_db(
            self.db,
            self._current_wafer_key or "",
            selected_col=selected_col,
            selected_row=selected_row,
            selected_fov=selected_fov or 5,
        )
        self._wafer_ctx = ctx
        self.wafer_map.set_context(ctx)
        self.fov_map.set_context(ctx)
        if ctx and ctx.selected_col > 0:
            chip = ctx.selected_chip()
            if chip is None:
                fb_txt = "—"
            elif chip.final_bin == BIN_NG:
                fb_txt = "NG"
            elif chip.final_bin == BIN_OK:
                fb_txt = "OK (9/9)"
            else:
                fb_txt = f"Pend ({chip.good_count}/9 OK)"
            self.chip_label.setText(
                f"Chip Col {ctx.selected_col} · Row {ctx.selected_row}  ·  Final {fb_txt}"
            )
        elif not self._current_wafer_key:
            self.chip_label.setText("Select a wafer, then a die")
        else:
            self.chip_label.setText("Click a die on wafer map")

    def _on_die_clicked(self, col: int, row: int):
        self._die_filter = (col, row)
        self.search_edit.setText(f"{col},{row}")
        self.chip_label.setText(f"Chip Col {col} · Row {row}")
        self._refresh_context_maps(selected_col=col, selected_row=row, selected_fov=5)
        self._reload_fov_list()

    def _on_fov_map_clicked(self, fov_index: int):
        """Select latest run for this FOV point on the active die filter."""
        die = getattr(self, "_die_filter", None)
        if self._wafer_ctx is not None:
            self._wafer_ctx.select_fov(int(fov_index))
            self.fov_map.set_context(self._wafer_ctx)
            self.wafer_map.set_context(self._wafer_ctx)
        for i, run in enumerate(self._runs_cache):
            if int(run.get("fov_index") or 0) != int(fov_index):
                continue
            if die:
                c, r = die
                if int(run.get("chip_col") or 0) != c or int(run.get("chip_row") or 0) != r:
                    continue
            self.fov_table.selectRow(i)
            rid = run.get("run_id")
            if rid:
                self._load_run_detail(rid)
            return

    def _on_fov_selected(self):
        rows = self.fov_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.fov_table.item(rows[0].row(), 0)
        if not item:
            return
        run_id = item.data(Qt.UserRole)
        self._load_run_detail(run_id)

    def _load_run_detail(self, run_id: str):
        self._current_run_id = run_id or ""
        run = self.db.get_run(run_id) if run_id else None
        self.timeline_list.clear()
        self.mes_table.setRowCount(0)
        if not run:
            self.detail_label.setText("No run selected")
            return

        self.breadcrumb.setText(
            f"REVIEW  ›  {run.get('date_folder') or ''}  ›  {run.get('lot_foup_id') or ''}  ›  "
            f"<b>{run.get('wafer_id') or ''}</b>  ›  Chip ({run.get('chip_col')},{run.get('chip_row')})  ›  "
            f"FOV P{run.get('fov_index')}  ·  run_id <span style='color:{SemiconductorTheme.ACCENT_PRIMARY}'>{run_id}</span>"
        )
        # Qt rich text
        self.breadcrumb.setTextFormat(Qt.RichText)
        self.breadcrumb.setText(
            f"REVIEW &nbsp;›&nbsp; {run.get('date_folder') or '—'} &nbsp;›&nbsp; "
            f"{run.get('lot_foup_id') or '—'} &nbsp;›&nbsp; <b>{run.get('wafer_id') or '—'}</b> "
            f"&nbsp;›&nbsp; Chip ({run.get('chip_col')},{run.get('chip_row')}) "
            f"&nbsp;›&nbsp; <b>FOV P{run.get('fov_index') or '?'}</b> "
            f"&nbsp;·&nbsp; run_id <span style='color:#22d3ee'>{run_id}</span> "
            f"&nbsp;·&nbsp; {run.get('config_path') or ''}"
        )

        objs = run.get("mes_objects") or []
        self.mes_table.setRowCount(len(objs))
        for i, s in enumerate(objs):
            ratio = float(s.get("ratio") or 0)
            ratio_pct = ratio * 100 if ratio < 1e8 else float("nan")
            jud = str(s.get("judgment") or "")
            cells = [
                str(s.get("row_id") or i + 1),
                str(s.get("layer_name") or ""),
                f"({s.get('grid_row')},{s.get('grid_col')})",
                f"{float(s.get('soh') or 0):.2f}",
                f"{float(s.get('c1_volume') or 0):.1f}",
                f"{float(s.get('c2_volume') or 0):.1f}",
                f"{ratio_pct:.2f}" if ratio_pct == ratio_pct else "N/A",
                jud,
                f"{float(s.get('pitch_x') or 0):.3f}",
                f"{float(s.get('pitch_y') or 0):.3f}",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col == 7:
                    if jud.upper() == "NG":
                        item.setForeground(QColor(SemiconductorTheme.ACCENT_ERROR))
                    elif jud.upper() == "OK":
                        item.setForeground(QColor(SemiconductorTheme.ACCENT_SUCCESS))
                self.mes_table.setItem(i, col, item)

        for t in run.get("timeline") or []:
            ts = str(t.get("ts") or "")[11:19]
            msg = t.get("message") or t.get("step") or ""
            dur = t.get("duration_sec")
            extra = f" ({float(dur):.1f}s)" if dur else ""
            self.timeline_list.addItem(f"{ts}  ·  {msg}{extra}")

        arts = run.get("artifacts") or []
        art_s = ", ".join(f"{a.get('kind')}" for a in arts[:6])
        self.detail_label.setText(
            f"Results: {run.get('results_dir') or '—'}\n"
            f"Input: {run.get('input_path') or '—'}\n"
            f"Enhance: {run.get('enhance_provider') or '—'} · "
            f"t={float(run.get('total_sec') or 0):.1f}s · artifacts: {art_s or '—'}"
        )

        # Sync Online-style wafer / FOV maps
        c, r = int(run.get("chip_col") or 0), int(run.get("chip_row") or 0)
        fi = int(run.get("fov_index") or 0)
        if c > 0 and r > 0:
            self._die_filter = (c, r)
        self._refresh_context_maps(
            selected_col=c, selected_row=r, selected_fov=fi or 5
        )

        # Load MPR replay from Results paths
        self._load_mpr_for_run(run)

    def _resolve_replay_paths(self, run: Dict[str, Any]):
        """Pick volume + mask paths from artifacts / Results conventions."""
        results = run.get("results_dir") or ""
        arts = {a.get("kind"): a.get("path") for a in (run.get("artifacts") or []) if a.get("path")}

        vol = (
            arts.get("enhanced_volume")
            or run.get("input_path")
            or run.get("host_path")
            or ""
        )
        # Prefer enhanced multipage under Results/enhanced_volume
        if results:
            for cand in (
                os.path.join(results, "enhanced_volume", "Enhanced_Volume.tif"),
                os.path.join(results, "enhanced_volume", "Enhanced_Volume.tiff"),
            ):
                if os.path.isfile(cand):
                    vol = cand
                    break
            if not (vol and os.path.isfile(vol)):
                # raw input
                raw = run.get("input_path") or run.get("host_path") or ""
                if raw and os.path.isfile(raw):
                    vol = raw

        bump = arts.get("mask_bump_far") or arts.get("mask_bump") or ""
        void = arts.get("mask_void_far") or arts.get("mask_void") or ""
        if results:
            if not (bump and os.path.isfile(bump)):
                for cand in (
                    os.path.join(results, "online_combined_bump3D_FAR.tif"),
                    os.path.join(results, "online_combined_bump3D.tif"),
                ):
                    if os.path.isfile(cand):
                        bump = cand
                        break
            if not (void and os.path.isfile(void)):
                for cand in (
                    os.path.join(results, "online_combined_voidsOnly_FAR.tif"),
                    os.path.join(results, "online_combined_voidsOnly.tif"),
                ):
                    if os.path.isfile(cand):
                        void = cand
                        break
        return vol, bump, void

    def _load_mpr_for_run(self, run: Dict[str, Any]):
        vol, bump, void = self._resolve_replay_paths(run)
        if not vol or not os.path.isfile(vol):
            self.mpr_panel.clear()
            self.mpr_panel.status.setText(
                f"Volume not on disk: {vol or '(empty path)'}"
            )
            return
        self.mpr_panel.load_from_paths(vol, bump, void)

    def _reload_mpr_for_current(self):
        run = self._selected_run()
        if not run:
            QMessageBox.information(self, "MPR", "Select a FOV run first.")
            return
        self._load_mpr_for_run(run)

    # ------------------------------------------------------------------ actions
    def _seed_demo(self):
        n = self.db.seed_demo_if_empty()
        if n:
            QMessageBox.information(self, "Demo data", f"Inserted {n} demo FOV runs.")
        else:
            QMessageBox.information(
                self, "Demo data", "DB already has runs — demo seed skipped."
            )
        self.refresh_all()

    def _open_db_folder(self):
        folder = str(self.db.db_path.parent)
        self._reveal_path(folder)

    def _selected_run(self) -> Optional[Dict[str, Any]]:
        if not self._current_run_id:
            return None
        return self.db.get_run(self._current_run_id)

    def _open_results(self):
        run = self._selected_run()
        if not run:
            QMessageBox.information(self, "Results", "Select a FOV run first.")
            return
        path = run.get("results_dir") or ""
        if path and os.path.isdir(path):
            self._reveal_path(path)
        else:
            QMessageBox.warning(
                self,
                "Results",
                f"Results folder not found on disk:\n{path or '(empty)'}\n\n"
                "DB only stores the path — file may be on another machine.",
            )

    def _open_mes_csv(self):
        run = self._selected_run()
        if not run:
            return
        # prefer artifact
        for a in run.get("artifacts") or []:
            if a.get("kind") in ("csv_mes", "mes_csv") and a.get("path"):
                p = a["path"]
                if os.path.isfile(p):
                    self._reveal_path(p)
                    return
        # fallback convention
        rd = run.get("results_dir") or ""
        cand = os.path.join(rd, "object_statistics.csv")
        if os.path.isfile(cand):
            self._reveal_path(cand)
            return
        QMessageBox.information(self, "MES CSV", "No MES CSV path found for this run.")

    def _export_wafer_csv(self):
        if not self._runs_cache:
            QMessageBox.information(self, "Export", "No FOV runs to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export FOV list", "wafer_fov_runs.csv", "CSV (*.csv)"
        )
        if not path:
            return
        keys = [
            "run_id", "date_folder", "lot_foup_id", "wafer_id",
            "chip_col", "chip_row", "fov_index", "judgment", "n_objects", "n_ng",
            "total_sec", "results_dir", "input_path", "finished_at",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in self._runs_cache:
                w.writerow(r)
        QMessageBox.information(self, "Export", f"Wrote {len(self._runs_cache)} rows to:\n{path}")

    def _emit_open_viewer(self):
        """Hand off selected FOV to 3D VIEWER (volume + mask + MES + B2B)."""
        run = self._selected_run()
        if not run:
            QMessageBox.information(self, "Viewer", "Select a FOV run first.")
            return
        # Ensure MES objects are present (get_run should include them)
        if not run.get("mes_objects") and run.get("run_id"):
            try:
                full = self.db.get_run(run["run_id"])
                if full:
                    run = full
            except Exception:
                pass
        # MainWindow → Viewer.load_batch_review_run (full Online parity)
        self.open_in_viewer.emit(run)

    @staticmethod
    def _reveal_path(path: str):
        path = os.path.abspath(path)
        try:
            if sys.platform.startswith("win"):
                if os.path.isfile(path):
                    subprocess.Popen(["explorer", "/select,", path])
                else:
                    os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            print(f"[BatchReview] open path failed: {e}")

    def refresh_theme(self):
        self._apply_theme()
