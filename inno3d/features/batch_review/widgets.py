# inno3d/features/batch_review/widgets.py
# -----------------------------------------------------------------------
# Batch Review UI widgets — extracted from inno3d/tabs/batch_review.py
#
# build_wafer_context_from_db, WaferMapWidget, FovMapWidget,
# _to_uint8_gray, _overlay_masks, _numpy_to_qpixmap, SliceViewLabel,
# VolumeLoadThread, MprReplayPanel, MiniBarChart, MiniHistChart
# -----------------------------------------------------------------------

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


