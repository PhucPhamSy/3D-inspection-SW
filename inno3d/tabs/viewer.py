import os
import glob
import re
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support
from PyQt5.QtCore import QEvent, QPoint, QRect, QSize, QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import (
    QColor, QFont, QFontInfo, QFontMetrics, QIcon, QImage, QKeySequence,
    QPainter, QPainterPath, QPen, QPixmap, QPolygon, QBrush,
)
from PyQt5.QtWidgets import *
from scipy import ndimage
from skimage import io, measure
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from inno3d.core.styles import SemiconductorTheme, get_class_colors
from inno3d.core.tf_widgets import (
    ColorGradientWidget,
    HistogramWLWidget,
    TransferFunctionWidget,
    OPACITY_SHAPES,
    OPACITY_SHAPE_TIPS,
    make_opacity_shape_icon,
    make_color_map_icon,
    make_icon_toolbutton,
    sample_opacity_curve,
)
from inno3d.core.view_support import (
    Dragonfly3DInteractorStyle,
    LoadVolumeThread,
    SliceBarWheelFilter,
    SliceControlOverlay,
    StepOneSliceSlider,
    apply_dragonfly_volume_zoom,
    push_camera_outside_aabb,
    volume_world_aabb,
)
from inno3d.core.wafer_context import WaferContext
from inno3d.widgets.context_map_panel import ContextMapPanel


def _ui_icon(name, widget=None, fallback=None):
    icon_path = Path(__file__).resolve().parents[2] / "assets" / "icons" / name
    if icon_path.exists():
        return QIcon(str(icon_path))
    if widget is not None and fallback is not None:
        return widget.style().standardIcon(fallback)
    return QIcon()


def _natural_sort_key(text):
    """Split 'Layer 12' → ['layer ', 12, ''] so Layer 2 < Layer 10."""
    s = "" if text is None else str(text)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


class NumericSortItem(QTableWidgetItem):
    """Table cell that sorts by numeric range/value (UserRole+1), not display text."""

    def __lt__(self, other):
        try:
            a = self.data(Qt.UserRole + 1)
            b = other.data(Qt.UserRole + 1) if other is not None else None
            if a is None:
                a = float("-inf")
            if b is None:
                b = float("-inf")
            return float(a) < float(b)
        except (TypeError, ValueError):
            return super().__lt__(other)


class CategorySortItem(QTableWidgetItem):
    """Table cell that sorts as a category with natural ordering (Layer 2 < Layer 10)."""

    def __lt__(self, other):
        try:
            other_text = other.text() if other is not None else ""
            return _natural_sort_key(self.text()) < _natural_sort_key(other_text)
        except Exception:
            return super().__lt__(other)


def _numeric_item(text, value, editable=False):
    item = NumericSortItem(str(text))
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    try:
        item.setData(Qt.UserRole + 1, float(value) if value is not None else float("-inf"))
    except (TypeError, ValueError):
        item.setData(Qt.UserRole + 1, float("-inf"))
    return item


def _category_item(text, editable=False):
    item = CategorySortItem("" if text is None else str(text))
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


# ── Dragonfly-style view grid layout presets ─────────────────────────────────
# placement: orientation -> (row, col, rowspan, colspan)
# view_3d is the volume pane; icons mark it with a cube glyph.
VIEW_LAYOUT_PRESETS = [
    {
        "id": "mpr_2x2",
        "label": "2×2 MPR + 3D",
        "tooltip": "Standard 2×2: XY | YZ / XZ | 3D Volume",
        "rows": 2,
        "cols": 2,
        "placements": {
            "axial": (0, 0, 1, 1),
            "sagittal": (0, 1, 1, 1),
            "coronal": (1, 0, 1, 1),
            "view_3d": (1, 1, 1, 1),
        },
    },
    {
        "id": "single_3d",
        "label": "3D only",
        "tooltip": "Single full-screen 3D volume",
        "rows": 1,
        "cols": 1,
        "placements": {
            "view_3d": (0, 0, 1, 1),
        },
    },
    {
        "id": "single_axial",
        "label": "XY only",
        "tooltip": "Single axial (XY) slice view",
        "rows": 1,
        "cols": 1,
        "placements": {
            "axial": (0, 0, 1, 1),
        },
    },
    {
        "id": "h_2",
        "label": "2 horizontal",
        "tooltip": "Side-by-side: Axial | 3D Volume",
        "rows": 1,
        "cols": 2,
        "placements": {
            "axial": (0, 0, 1, 1),
            "view_3d": (0, 1, 1, 1),
        },
    },
    {
        "id": "v_2",
        "label": "2 vertical",
        "tooltip": "Stacked: Axial / 3D Volume",
        "rows": 2,
        "cols": 1,
        "placements": {
            "axial": (0, 0, 1, 1),
            "view_3d": (1, 0, 1, 1),
        },
    },
    {
        "id": "h_3mpr",
        "label": "3 MPR row",
        "tooltip": "Three MPR views in a row (no 3D)",
        "rows": 1,
        "cols": 3,
        "placements": {
            "axial": (0, 0, 1, 1),
            "sagittal": (0, 1, 1, 1),
            "coronal": (0, 2, 1, 1),
        },
    },
    {
        "id": "v_3mpr_3d",
        "label": "3 MPR | 3D",
        "tooltip": "Left: stacked XY/YZ/XZ · Right: large 3D volume",
        "rows": 3,
        "cols": 2,
        "placements": {
            "axial": (0, 0, 1, 1),
            "sagittal": (1, 0, 1, 1),
            "coronal": (2, 0, 1, 1),
            "view_3d": (0, 1, 3, 1),
        },
    },
    {
        "id": "v_3d_3mpr",
        "label": "3D | 3 MPR",
        "tooltip": "Left: large 3D volume · Right: stacked XY/YZ/XZ",
        "rows": 3,
        "cols": 2,
        "placements": {
            "view_3d": (0, 0, 3, 1),
            "axial": (0, 1, 1, 1),
            "sagittal": (1, 1, 1, 1),
            "coronal": (2, 1, 1, 1),
        },
    },
    {
        "id": "h_3mpr_3d",
        "label": "3 MPR / 3D",
        "tooltip": "Top: XY | YZ | XZ · Bottom: wide 3D volume",
        "rows": 2,
        "cols": 3,
        "placements": {
            "axial": (0, 0, 1, 1),
            "sagittal": (0, 1, 1, 1),
            "coronal": (0, 2, 1, 1),
            "view_3d": (1, 0, 1, 3),
        },
    },
    {
        "id": "h_3d_3mpr",
        "label": "3D / 3 MPR",
        "tooltip": "Top: wide 3D volume · Bottom: XY | YZ | XZ",
        "rows": 2,
        "cols": 3,
        "placements": {
            "view_3d": (0, 0, 1, 3),
            "axial": (1, 0, 1, 1),
            "sagittal": (1, 1, 1, 1),
            "coronal": (1, 2, 1, 1),
        },
    },
]


def _draw_mini_cube(painter, cx, cy, s):
    """Isometric-ish cube glyph — marks the 3D volume cell (Dragonfly-style)."""
    s = max(5, float(s))
    dx = s * 0.45
    dy = s * 0.26
    h = s * 0.55
    top = QPolygon([
        QPoint(int(cx), int(cy - h * 0.55)),
        QPoint(int(cx + dx), int(cy - h * 0.55 + dy)),
        QPoint(int(cx), int(cy - h * 0.55 + 2 * dy)),
        QPoint(int(cx - dx), int(cy - h * 0.55 + dy)),
    ])
    left = QPolygon([
        top[3],
        top[2],
        QPoint(int(cx), int(cy + h * 0.45)),
        QPoint(int(cx - dx), int(cy + h * 0.45 - dy)),
    ])
    right = QPolygon([
        top[1],
        top[2],
        QPoint(int(cx), int(cy + h * 0.45)),
        QPoint(int(cx + dx), int(cy + h * 0.45 - dy)),
    ])
    painter.setPen(QPen(QColor(20, 30, 40, 210), 0.9))
    painter.setBrush(QColor(140, 220, 240, 230))
    painter.drawPolygon(top)
    painter.setBrush(QColor(50, 140, 170, 220))
    painter.drawPolygon(left)
    painter.setBrush(QColor(90, 185, 210, 230))
    painter.drawPolygon(right)


def _make_view_layout_icon(preset, size=36, selected=False):
    """Paint a Dragonfly-like layout thumbnail; cube marks the 3D volume cell."""
    # ARGB32_Premultiplied is reliable across DPI/styles (avoids blank/white icons)
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)

    rows = max(1, int(preset["rows"]))
    cols = max(1, int(preset["cols"]))
    pad = 2.5
    gap = 1.6
    avail = size - 2 * pad
    cw = (avail - gap * (cols - 1)) / cols
    ch = (avail - gap * (rows - 1)) / rows

    # Soft card background
    bg = QColor(SemiconductorTheme.BG_LIGHT)
    if selected:
        bg = QColor(SemiconductorTheme.BG_PANEL)
    p.setPen(Qt.NoPen)
    p.setBrush(bg)
    p.drawRoundedRect(0, 0, size, size, 5, 5)

    # Occupancy map so spanning cells draw once
    covered = [[False] * cols for _ in range(rows)]
    cells = []
    for key, (r, c, rs, cs) in preset["placements"].items():
        cells.append((key, r, c, rs, cs, key == "view_3d"))

    # Draw back-to-front by area (larger first)
    cells.sort(key=lambda t: -(t[3] * t[4]))

    for key, r, c, rs, cs, is_3d in cells:
        if r >= rows or c >= cols:
            continue
        x = pad + c * (cw + gap)
        y = pad + r * (ch + gap)
        w = cs * cw + max(0, cs - 1) * gap
        h = rs * ch + max(0, rs - 1) * gap

        if is_3d:
            fill = QColor(SemiconductorTheme.PRIMARY_DEFAULT)
            fill.setAlpha(120 if not selected else 160)
            border = QColor(SemiconductorTheme.PRIMARY_HOVER)
            border.setAlpha(240)
        else:
            fill = QColor(150, 170, 190, 110 if not selected else 150)
            border = QColor(190, 205, 220, 200)

        p.setPen(QPen(border, 1.0))
        p.setBrush(fill)
        radius = 2.0 if min(w, h) > 8 else 1.0
        p.drawRoundedRect(int(x), int(y), max(2, int(w)), max(2, int(h)), radius, radius)

        if is_3d:
            cube_s = min(w, h) * 0.48
            _draw_mini_cube(p, x + w * 0.5, y + h * 0.52, cube_s)

        for rr in range(r, min(rows, r + rs)):
            for cc in range(c, min(cols, c + cs)):
                covered[rr][cc] = True

    # Empty slots (if any) as faint outlines
    for rr in range(rows):
        for cc in range(cols):
            if covered[rr][cc]:
                continue
            x = pad + cc * (cw + gap)
            y = pad + rr * (ch + gap)
            p.setPen(QPen(QColor(100, 115, 130, 80), 1.0, Qt.DotLine))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(int(x), int(y), max(2, int(cw)), max(2, int(ch)), 2, 2)

    # Selection ring
    if selected:
        p.setPen(QPen(QColor(SemiconductorTheme.PRIMARY_DEFAULT), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(1, 1, size - 2, size - 2, 5, 5)

    p.end()
    return QIcon(QPixmap.fromImage(img))


class StatsRowDelegate(QStyledItemDelegate):
    """Paint BackgroundRole / ForegroundRole even when a QSS is set on the view.

    Qt stylesheets on ``QTableView::item`` otherwise swallow item backgrounds,
    which made frozen columns look un-tinted while scrolling columns looked OK.
    """

    def paint(self, painter, option, index):
        painter.save()
        rect = option.rect

        bg = index.data(Qt.BackgroundRole)
        if isinstance(bg, QBrush) and bg.style() != Qt.NoBrush:
            painter.fillRect(rect, bg)
        elif isinstance(bg, QColor) and bg.isValid():
            painter.fillRect(rect, bg)
        else:
            base = option.palette.base().color()
            if option.features & QStyleOptionViewItem.Alternate:
                base = option.palette.alternateBase().color()
            painter.fillRect(rect, base)

        if option.state & QStyle.State_Selected:
            painter.fillRect(rect, QColor(34, 174, 209, 75))
        elif option.state & QStyle.State_MouseOver:
            painter.fillRect(rect, QColor(255, 255, 255, 18))

        # Subtle right grid
        painter.setPen(QPen(QColor(SemiconductorTheme.BORDER_DEFAULT)))
        painter.drawLine(rect.topRight(), rect.bottomRight())

        text = index.data(Qt.DisplayRole)
        text = "" if text is None else str(text)

        fg = index.data(Qt.ForegroundRole)
        if isinstance(fg, QBrush) and fg.style() != Qt.NoBrush:
            pen_color = fg.color()
        elif isinstance(fg, QColor) and fg.isValid():
            pen_color = fg
        else:
            pen_color = QColor(SemiconductorTheme.TEXT_PRIMARY)
        painter.setPen(QPen(pen_color))

        font = index.data(Qt.FontRole)
        if isinstance(font, QFont):
            painter.setFont(font)
        else:
            painter.setFont(option.font)

        align = index.data(Qt.TextAlignmentRole)
        if align is None:
            align = int(Qt.AlignLeft | Qt.AlignVCenter)
        else:
            align = int(align)

        painter.drawText(rect.adjusted(6, 0, -4, 0), align, text)
        painter.restore()


class FrozenTableWidget(QTableWidget):
    """QTableWidget with left frozen columns (Excel-style).

    Dual-pane design (fixes H-scroll header/body mismatch):
      - Frozen overlay includes **its own header + body** for cols 0..N-1
      - Stays fixed on the left while the main table scrolls horizontally
      - Main header under the frozen region is covered → labels always match cells
      - Column widths / row heights / v-scroll stay in lock-step
      - Excel filter state is shared when both headers are ExcelFilterHeader
      - StatsRowDelegate paints OK/NG backgrounds on both panes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_cols = 0
        self._frozen_ready = False
        self._row_h = 24
        self._syncing_vscroll = False
        self._syncing_col_width = False
        self._syncing_sort = False

        self.verticalHeader().setDefaultSectionSize(self._row_h)
        self.verticalHeader().setVisible(False)
        self.setWordWrap(False)
        self.setShowGrid(True)
        self.setAlternatingRowColors(True)

        # Custom delegate so BackgroundRole shows under QSS (main + frozen)
        self._row_delegate = StatsRowDelegate(self)
        self.setItemDelegate(self._row_delegate)

        self.frozenTableView = QTableView(self)
        self.frozenTableView.setFocusPolicy(Qt.NoFocus)
        self.frozenTableView.verticalHeader().hide()
        # Own header — critical so labels stay put when main table H-scrolls
        self.frozenTableView.horizontalHeader().show()
        self.frozenTableView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frozenTableView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frozenTableView.setShowGrid(True)
        self.frozenTableView.setAlternatingRowColors(True)
        self.frozenTableView.setWordWrap(False)
        self.frozenTableView.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.frozenTableView.verticalHeader().setDefaultSectionSize(self._row_h)
        self.frozenTableView.setFrameShape(QFrame.NoFrame)
        self.frozenTableView.setItemDelegate(self._row_delegate)
        self.viewport().stackUnder(self.frozenTableView)
        self.frozenTableView.hide()

        self.frozenTableView.verticalScrollBar().valueChanged.connect(self._on_frozen_vscroll)
        self.verticalScrollBar().valueChanged.connect(self._on_main_vscroll)

    def set_frozen_columns(self, count):
        self.frozen_cols = max(0, int(count))

    def _on_main_vscroll(self, value):
        if self._syncing_vscroll:
            return
        self._syncing_vscroll = True
        self.frozenTableView.verticalScrollBar().setValue(value)
        self._syncing_vscroll = False

    def _on_frozen_vscroll(self, value):
        if self._syncing_vscroll:
            return
        self._syncing_vscroll = True
        self.verticalScrollBar().setValue(value)
        self._syncing_vscroll = False

    def _bind_header_signals(self):
        """(Re)connect resize/sort signals after headers may have been replaced."""
        main_h = self.horizontalHeader()
        frozen_h = self.frozenTableView.horizontalHeader()

        for h, slot in (
            (main_h, self.updateSectionWidth),
            (frozen_h, self._on_frozen_section_resized),
        ):
            try:
                h.sectionResized.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
            h.sectionResized.connect(slot)

        try:
            self.verticalHeader().sectionResized.disconnect(self.updateSectionHeight)
        except (TypeError, RuntimeError):
            pass
        self.verticalHeader().sectionResized.connect(self.updateSectionHeight)

        # Sort: either header can drive; keep indicators mirrored
        for h, slot in (
            (main_h, self._on_main_sort_changed),
            (frozen_h, self._on_frozen_sort_changed),
        ):
            try:
                h.sortIndicatorChanged.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
            try:
                h.sortIndicatorChanged.connect(slot)
            except (TypeError, RuntimeError):
                pass

    def setup_frozen(self):
        """Bind frozen view to this table model. Call after headers/columns exist."""
        if self.frozen_cols <= 0:
            self.frozenTableView.hide()
            self._frozen_ready = False
            return

        self.frozenTableView.setModel(self.model())
        self.frozenTableView.setSelectionModel(self.selectionModel())
        self.frozenTableView.setSelectionBehavior(self.selectionBehavior())
        self.frozenTableView.setSelectionMode(self.selectionMode())
        self.frozenTableView.setAlternatingRowColors(self.alternatingRowColors())

        # Frozen header: reuse ExcelFilterHeader when main has one (shared filters)
        main_h = self.horizontalHeader()
        if isinstance(main_h, ExcelFilterHeader):
            frozen_h = self.frozenTableView.horizontalHeader()
            if not isinstance(frozen_h, ExcelFilterHeader):
                frozen_h = ExcelFilterHeader(self)
                self.frozenTableView.setHorizontalHeader(frozen_h)
            # Share filter dict so funnels stay consistent
            frozen_h._filters = main_h._filters
            try:
                frozen_h.filter_applied.disconnect()
            except (TypeError, RuntimeError):
                pass

            def _relay_frozen_filter(col, _main=main_h):
                _main.filter_applied.emit(col)
                try:
                    _main.viewport().update()
                except Exception:
                    pass

            frozen_h.filter_applied.connect(_relay_frozen_filter)
        else:
            frozen_h = self.frozenTableView.horizontalHeader()

        frozen_h.setSectionsClickable(True)
        frozen_h.setHighlightSections(True)
        frozen_h.setSortIndicatorShown(main_h.isSortIndicatorShown())
        frozen_h.setMinimumSectionSize(max(40, main_h.minimumSectionSize()))
        frozen_h.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        frozen_h.setStretchLastSection(False)
        frozen_h.setSectionResizeMode(QHeaderView.Interactive)
        # Match main header height so body rows line up
        hdr_h = max(28, main_h.height(), main_h.sizeHint().height())
        main_h.setFixedHeight(hdr_h)
        frozen_h.setFixedHeight(hdr_h)

        # Do NOT style QTableView::item — that makes QSS hide BackgroundRole tints.
        self.frozenTableView.setStyleSheet(f"""
            QTableView {{
                border: none;
                border-right: 2px solid {SemiconductorTheme.ACCENT_PRIMARY};
                background-color: {SemiconductorTheme.BG_LIGHT};
                outline: none;
            }}
            QHeaderView::section {{
                background-color: {SemiconductorTheme.BG_PANEL};
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                border: none;
                border-bottom: 2px solid {SemiconductorTheme.ACCENT_PRIMARY};
                border-right: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                padding: 4px 6px;
                padding-right: 20px;
                font-weight: 700;
                font-size: 8pt;
            }}
        """)
        # Keep delegate (in case setStyleSheet / setModel reset anything)
        self.frozenTableView.setItemDelegate(self._row_delegate)

        cols = self.model().columnCount() if self.model() else self.columnCount()
        for col in range(cols):
            self.frozenTableView.setColumnHidden(col, col >= self.frozen_cols)

        self._bind_header_signals()

        try:
            self.frozenTableView.clicked.disconnect(self._forward_frozen_click)
        except (TypeError, RuntimeError):
            pass
        self.frozenTableView.clicked.connect(self._forward_frozen_click)

        self.frozenTableView.show()
        self._frozen_ready = True
        self.refresh_frozen()

    def _forward_frozen_click(self, index):
        """Map frozen-view click → main table cellClicked + current cell."""
        if not index.isValid():
            return
        self.setCurrentIndex(index)
        try:
            self.cellClicked.emit(index.row(), index.column())
        except Exception:
            pass

    def _on_main_sort_changed(self, logical_index, order):
        if self._syncing_sort or not self._frozen_ready:
            return
        self._syncing_sort = True
        try:
            fh = self.frozenTableView.horizontalHeader()
            if 0 <= logical_index < self.frozen_cols:
                fh.setSortIndicator(logical_index, order)
            else:
                # Sorted on a scrolling column — clear frozen sort glyph
                fh.setSortIndicator(-1, order)
            QTimer.singleShot(0, self.refresh_frozen)
        finally:
            self._syncing_sort = False

    def _on_frozen_sort_changed(self, logical_index, order):
        if self._syncing_sort or not self._frozen_ready:
            return
        if logical_index < 0 or logical_index >= self.frozen_cols:
            return
        self._syncing_sort = True
        try:
            self.horizontalHeader().setSortIndicator(logical_index, order)
            if self.isSortingEnabled():
                self.sortByColumn(logical_index, order)
            QTimer.singleShot(0, self.refresh_frozen)
        finally:
            self._syncing_sort = False

    def _on_frozen_section_resized(self, logical_index, old_size, new_size):
        if not self._frozen_ready or self._syncing_col_width:
            return
        if logical_index < 0 or logical_index >= self.frozen_cols:
            return
        self._syncing_col_width = True
        try:
            self.setColumnWidth(logical_index, new_size)
            self.updateFrozenTableGeometry()
        finally:
            self._syncing_col_width = False

    def refresh_frozen(self):
        """Sync widths, row heights, geometry after data/sort/filter changes."""
        if not self._frozen_ready or self.frozen_cols <= 0:
            return

        main_h = self.horizontalHeader()
        frozen_h = self.frozenTableView.horizontalHeader()
        hdr_h = max(28, main_h.height(), main_h.sizeHint().height())
        if main_h.height() != hdr_h:
            main_h.setFixedHeight(hdr_h)
        if frozen_h.height() != hdr_h:
            frozen_h.setFixedHeight(hdr_h)

        # Mirror sort indicator on frozen cols
        sec = main_h.sortIndicatorSection()
        order = main_h.sortIndicatorOrder()
        if not self._syncing_sort:
            self._syncing_sort = True
            try:
                if 0 <= sec < self.frozen_cols:
                    frozen_h.setSortIndicator(sec, order)
                else:
                    frozen_h.setSortIndicator(-1, order)
            finally:
                self._syncing_sort = False

        # Row heights + visibility
        for r in range(self.rowCount()):
            h = self.rowHeight(r)
            if h <= 0:
                h = self._row_h
                self.setRowHeight(r, h)
            self.frozenTableView.setRowHeight(r, h)
            self.frozenTableView.setRowHidden(r, self.isRowHidden(r))

        # Column widths of frozen region (main is source of truth)
        self._syncing_col_width = True
        try:
            for c in range(self.frozen_cols):
                w = self.columnWidth(c)
                if w <= 0:
                    w = 60
                    self.setColumnWidth(c, w)
                self.frozenTableView.setColumnWidth(c, w)
        finally:
            self._syncing_col_width = False

        # Keep vertical scroll locked
        self.frozenTableView.verticalScrollBar().setValue(self.verticalScrollBar().value())
        self.updateFrozenTableGeometry()

    def updateSectionWidth(self, logicalIndex, oldSize, newSize):
        if not self._frozen_ready or self._syncing_col_width:
            return
        if logicalIndex < self.frozen_cols:
            self._syncing_col_width = True
            try:
                self.frozenTableView.setColumnWidth(logicalIndex, newSize)
                self.updateFrozenTableGeometry()
            finally:
                self._syncing_col_width = False

    def updateSectionHeight(self, logicalIndex, oldSize, newSize):
        if not self._frozen_ready:
            return
        self.frozenTableView.setRowHeight(logicalIndex, newSize)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateFrozenTableGeometry()

    def moveCursor(self, cursorAction, modifiers):
        current = super().moveCursor(cursorAction, modifiers)
        if current.isValid() and self._frozen_ready:
            frozen_width = self._frozen_width()
            if (
                current.column() >= self.frozen_cols
                and self.visualRect(current).topLeft().x() < frozen_width
            ):
                new_value = (
                    self.horizontalScrollBar().value()
                    + self.visualRect(current).topLeft().x()
                    - frozen_width
                )
                self.horizontalScrollBar().setValue(max(0, new_value))
        return current

    def scrollTo(self, index, hint=QAbstractItemView.EnsureVisible):
        if index.isValid() and index.column() >= self.frozen_cols:
            super().scrollTo(index, hint)
        elif index.isValid():
            super().scrollTo(index, hint)

    def _frozen_width(self):
        w = sum(
            self.frozenTableView.columnWidth(c) for c in range(self.frozen_cols)
        )
        if w <= 0:
            w = sum(max(40, self.columnWidth(c)) for c in range(self.frozen_cols))
        return w

    def updateFrozenTableGeometry(self):
        """Pin frozen pane (header + body) to the left; covers main header for frozen cols."""
        if not self._frozen_ready or self.frozen_cols <= 0:
            return
        x = self.frameWidth()
        if not self.verticalHeader().isHidden():
            x += self.verticalHeader().width()
        y = self.frameWidth()
        frozen_width = self._frozen_width()
        # Include header height so labels stay fixed while main H-scrolls
        total_h = self.horizontalHeader().height() + self.viewport().height()
        self.frozenTableView.setGeometry(x, y, frozen_width, total_h)
        self.frozenTableView.raise_()


class AlignVolumeThread(QThread):
    """Enhanced auto-alignment using Otsu, CSD, LTIC, and iterative PCA refinement."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object, object)  # (aligned_vol, best_R, error)
    
    def __init__(self, volume_data):
        super().__init__()
        self.volume_data = volume_data
        
    def _otsu_threshold(self, data_flat):
        """Compute Otsu's optimal threshold for foreground/background separation."""
        hist, bin_edges = np.histogram(data_flat, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        total = hist.sum()
        if total == 0:
            return float(data_flat.mean())
        
        sum_total = np.dot(hist, bin_centers)
        sum_bg, weight_bg = 0.0, 0.0
        max_var, threshold = 0.0, bin_centers[0]
        
        for i in range(len(hist)):
            weight_bg += hist[i]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += hist[i] * bin_centers[i]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if var_between > max_var:
                max_var = var_between
                threshold = bin_centers[i]
        return threshold
    
    def _compute_csd(self, pts, weights, axis_normal):
        """Compute Continuous Symmetry Distance for a mirror plane through centroid.
        
        CSD measures how symmetric the point cloud is about the given plane.
        Lower CSD = more symmetric. Returns normalized distance [0, 1].
        """
        centroid = np.average(pts, axis=0, weights=weights)
        pts_c = pts - centroid
        # Project each point onto the plane normal
        projections = np.dot(pts_c, axis_normal)
        # Reflect points across the plane
        reflected = pts_c - 2.0 * projections[:, np.newaxis] * axis_normal[np.newaxis, :]
        # For each reflected point, find the closest original point (approximate via binning)
        # Use a fast approximation: average squared distance from reflected to nearest original
        from scipy.spatial import cKDTree
        tree = cKDTree(pts_c)
        dists, _ = tree.query(reflected, k=1)
        # Weighted average distance
        csd = np.average(dists ** 2, weights=weights)
        # Normalize by the bounding sphere radius squared
        max_r = np.max(np.linalg.norm(pts_c, axis=1)) + 1e-8
        return csd / (max_r ** 2)
    
    def _compute_ltic(self, v_ds, axis_direction, n_bins=64):
        """Compute Local Translational Invariance Cost along a direction.
        
        Measures how uniform the cross-sectional shape is along the given axis.
        Higher LTIC = more translational invariance = better alignment axis.
        Uses the Vector Shape Description (F_k) from the paper.
        """
        shape = np.array(v_ds.shape, dtype=float)
        center = (shape - 1.0) / 2.0
        
        # Create coordinate grid for the downsampled volume
        zz, yy, xx = np.mgrid[0:v_ds.shape[0], 0:v_ds.shape[1], 0:v_ds.shape[2]]
        coords = np.stack([zz - center[0], yy - center[1], xx - center[2]], axis=-1)
        
        # Project coordinates onto the axis direction
        proj = np.tensordot(coords, axis_direction, axes=([-1], [0]))
        
        # Bin along projection axis
        proj_min, proj_max = proj.min(), proj.max()
        if proj_max - proj_min < 1e-6:
            return 0.0
        
        bin_width = (proj_max - proj_min) / n_bins
        bin_indices = np.clip(((proj - proj_min) / bin_width).astype(int), 0, n_bins - 1)
        
        # For each bin, compute a shape descriptor (total intensity)
        descriptors = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.any():
                descriptors[b] = v_ds[mask].sum()
        
        # Normalize descriptors
        desc_max = descriptors.max()
        if desc_max > 0:
            descriptors /= desc_max
        
        # LTIC = sum of lengths of intervals where descriptor is approximately constant
        # An interval is "constant" if |f(i) - f(i+1)| < tolerance
        tolerance = 0.05
        ltic_score = 0.0
        run_length = 0
        for i in range(len(descriptors) - 1):
            if abs(descriptors[i] - descriptors[i + 1]) < tolerance:
                run_length += 1
            else:
                if run_length > 1:
                    ltic_score += run_length
                run_length = 0
        if run_length > 1:
            ltic_score += run_length
        
        return ltic_score / n_bins  # Normalize by total bins
        
    def run(self):
        try:
            import itertools
            import math
            from scipy.ndimage import affine_transform
            
            self.progress.emit(5, "Downsampling volume for analysis...")
            shape = self.volume_data.shape
            
            # Choose downsampling factor dynamically
            ds = 2
            if max(shape) > 256:
                ds = 4
            v_ds = self.volume_data[::ds, ::ds, ::ds].astype(np.float64)
            
            # === Step 1: Otsu's thresholding (replaces fixed percentile) ===
            self.progress.emit(10, "Computing Otsu threshold for foreground...")
            flat = v_ds.ravel()
            thresh = self._otsu_threshold(flat)
            
            z_coords, y_coords, x_coords = np.where(v_ds > thresh)
            if len(z_coords) < 100:
                # Fallback: use anything above minimum
                z_coords, y_coords, x_coords = np.where(v_ds > flat.min())
                
            if len(z_coords) < 10:
                raise ValueError("Not enough foreground voxels found to align volume.")
            
            # Scale coordinates back to original resolution
            pts = np.vstack([z_coords, y_coords, x_coords]).T.astype(np.float64) * ds
            intensities = v_ds[z_coords, y_coords, x_coords]
            intensities -= intensities.min()
            intensities /= (intensities.max() + 1e-8)
            
            # === Step 2: Weighted PCA (Pass 1) ===
            self.progress.emit(20, "Computing PCA (Pass 1)...")
            w_sum = np.sum(intensities)
            mean = np.sum(pts * intensities[:, np.newaxis], axis=0) / w_sum
            pts_centered = pts - mean
            cov = np.dot((pts_centered * intensities[:, np.newaxis]).T, pts_centered) / w_sum
            
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # === Step 3: CSD - Assess symmetry quality per axis ===
            self.progress.emit(35, "Computing symmetry distances (CSD)...")
            csd_scores = np.zeros(3)
            for k in range(3):
                csd_scores[k] = self._compute_csd(pts, intensities, eigenvectors[:, k])
            
            # === Step 4: Detect degenerate eigenvalues & apply LTIC ===
            self.progress.emit(45, "Checking eigenvalue degeneracy & LTIC...")
            # Eigenvalues are sorted descending. Check if any pair is "degenerate"
            # (ratio close to 1.0 means ambiguous axis assignment)
            ev_ratios = []
            for i in range(2):
                ratio = eigenvalues[i+1] / (eigenvalues[i] + 1e-8)
                ev_ratios.append(ratio)
            
            # If ratio > 0.85, the two axes are ambiguous → use LTIC to disambiguate
            degenerate_pair = None
            if ev_ratios[0] > 0.85:
                degenerate_pair = (0, 1)
            elif ev_ratios[1] > 0.85:
                degenerate_pair = (1, 2)
            
            if degenerate_pair is not None:
                self.progress.emit(50, f"Axes {degenerate_pair} ambiguous, applying LTIC refinement...")
                i, j = degenerate_pair
                # Test LTIC for both candidate directions
                ltic_i = self._compute_ltic(v_ds, eigenvectors[:, i])
                ltic_j = self._compute_ltic(v_ds, eigenvectors[:, j])
                
                # The axis with HIGHER LTIC should be the one along which shape is uniform
                # (e.g., the long axis of a rectangular chip). Swap if needed.
                if ltic_j > ltic_i:
                    eigenvectors[:, [i, j]] = eigenvectors[:, [j, i]]
                    eigenvalues[[i, j]] = eigenvalues[[j, i]]
                    csd_scores[[i, j]] = csd_scores[[j, i]]
            
            # === Step 5: Find optimal rotation via trace maximization ===
            self.progress.emit(60, "Finding optimal rotation matrix...")
            best_trace = -100.0
            best_R = None
            for perm in itertools.permutations([0, 1, 2]):
                for signs in itertools.product([-1, 1], repeat=3):
                    R_candidate = np.zeros((3, 3))
                    for k in range(3):
                        R_candidate[:, k] = eigenvectors[:, perm[k]] * signs[k]
                    if np.linalg.det(R_candidate) > 0:
                        trace_val = np.trace(R_candidate)
                        if trace_val > best_trace:
                            best_trace = trace_val
                            best_R = R_candidate
                            
            if best_R is None:
                raise ValueError("Failed to compute valid rotation matrix.")
            
            # === Step 6: Iterative refinement (Pass 2) ===
            self.progress.emit(70, "Iterative PCA refinement (Pass 2)...")
            # Apply R to point cloud, recompute PCA, refine R
            pts_rotated = np.dot(pts_centered, best_R.T)
            cov2 = np.dot((pts_rotated * intensities[:, np.newaxis]).T, pts_rotated) / w_sum
            eigenvalues2, eigenvectors2 = np.linalg.eigh(cov2)
            idx2 = np.argsort(eigenvalues2)[::-1]
            eigenvectors2 = eigenvectors2[:, idx2]
            
            # Find refinement rotation
            best_trace2 = -100.0
            refine_R = None
            for perm in itertools.permutations([0, 1, 2]):
                for signs in itertools.product([-1, 1], repeat=3):
                    R2 = np.zeros((3, 3))
                    for k in range(3):
                        R2[:, k] = eigenvectors2[:, perm[k]] * signs[k]
                    if np.linalg.det(R2) > 0:
                        t2 = np.trace(R2)
                        if t2 > best_trace2:
                            best_trace2 = t2
                            refine_R = R2
            
            if refine_R is not None and best_trace2 > 2.95:
                # Only apply refinement if it's a small correction
                best_R = np.dot(refine_R, best_R)
                # Re-orthogonalize via SVD to prevent drift
                U, _, Vt = np.linalg.svd(best_R)
                best_R = np.dot(U, Vt)
                if np.linalg.det(best_R) < 0:
                    U[:, -1] *= -1
                    best_R = np.dot(U, Vt)
                
            # === Step 7: Apply affine transform to full volume ===
            self.progress.emit(82, "Rotating 3D volume (affine transform)...")
            c_old = (np.array(self.volume_data.shape) - 1.0) / 2.0
            offset = c_old - np.dot(best_R, c_old)
            
            aligned_vol = affine_transform(
                self.volume_data,
                matrix=best_R,
                offset=offset,
                order=1,
                mode='constant',
                cval=float(self.volume_data.min())
            )
            
            self.progress.emit(100, "Alignment complete!")
            self.finished.emit(aligned_vol, best_R, None)
            
        except Exception as e:
            import traceback
            self.finished.emit(None, None, f"{str(e)}\n{traceback.format_exc()}")


class ManualAlignThread(QThread):
    """Apply user-specified rotation angles (roll, pitch, yaw) to the volume."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object, object)  # (rotated_vol, R_matrix, error)
    
    def __init__(self, volume_data, roll_deg, pitch_deg, yaw_deg):
        super().__init__()
        self.volume_data = volume_data
        self.roll = np.radians(roll_deg)
        self.pitch = np.radians(pitch_deg)
        self.yaw = np.radians(yaw_deg)
    
    def run(self):
        try:
            from scipy.ndimage import affine_transform
            
            self.progress.emit(20, "Building rotation matrix from angles...")
            
            # Rotation matrices for each axis (ZYX intrinsic = extrinsic XYZ)
            # Roll (X-axis)
            cr, sr = np.cos(self.roll), np.sin(self.roll)
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            # Pitch (Y-axis)
            cp, sp = np.cos(self.pitch), np.sin(self.pitch)
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            # Yaw (Z-axis)
            cy, sy = np.cos(self.yaw), np.sin(self.yaw)
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            
            # Combined rotation: R = Rz @ Ry @ Rx
            R = Rz @ Ry @ Rx
            
            self.progress.emit(40, "Rotating 3D volume...")
            c_old = (np.array(self.volume_data.shape) - 1.0) / 2.0
            offset = c_old - np.dot(R, c_old)
            
            rotated_vol = affine_transform(
                self.volume_data,
                matrix=R,
                offset=offset,
                order=1,
                mode='constant',
                cval=float(self.volume_data.min())
            )
            
            self.progress.emit(100, "Manual rotation complete!")
            self.finished.emit(rotated_vol, R, None)
            
        except Exception as e:
            import traceback
            self.finished.emit(None, None, f"{str(e)}\n{traceback.format_exc()}")


class ExcelFilterMenu(QMenu):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QMenu {{ background: {SemiconductorTheme.BG_PANEL}; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; }}
            QListWidget {{ background: {SemiconductorTheme.BG_DARK}; color: {SemiconductorTheme.TEXT_PRIMARY}; border: none; outline: none; padding: 4px; font-size: 10pt; }}
            QListWidget::item {{ padding: 4px; }}
            QListWidget::item:hover {{ background: {SemiconductorTheme.BG_LIGHT}; }}
            QLineEdit {{ background: {SemiconductorTheme.BG_DARK}; color: {SemiconductorTheme.TEXT_PRIMARY}; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; padding: 4px; border-radius: 4px; font-size: 10pt; }}
            QPushButton {{ background: {SemiconductorTheme.ACCENT_PRIMARY}; color: #000; border: none; padding: 6px; font-weight: bold; border-radius: 4px; }}
            QPushButton:hover {{ background: {SemiconductorTheme.ACCENT_SECONDARY}; }}
            QPushButton[cancel="true"] {{ background: {SemiconductorTheme.BG_LIGHT}; color: {SemiconductorTheme.TEXT_PRIMARY}; }}
            QPushButton[cancel="true"]:hover {{ background: {SemiconductorTheme.BORDER_DEFAULT}; }}
        """)

class ExcelFilterHeader(QHeaderView):
    """Horizontal header with always-visible funnel filter icons + click-to-sort."""

    filter_applied = pyqtSignal(int)
    FILTER_ICON_W = 16
    FILTER_PAD = 4

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setSectionsClickable(True)
        self.setHighlightSections(True)
        self.setMinimumHeight(30)
        self.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._filters = {}
        self._mouse_over_col = -1
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover, True)
        # Room for label + funnel chip
        self.setMinimumSectionSize(56)

    @staticmethod
    def _draw_funnel(painter, rect, active=False, hover=False):
        """Draw a high-contrast funnel chip (always visible on dark headers)."""
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Chip background so icon never blends into header
        if active:
            chip_bg = QColor(0, 200, 230, 55)
            color = QColor(0, 240, 255)
        elif hover:
            chip_bg = QColor(255, 255, 255, 28)
            color = QColor(200, 230, 245)
        else:
            chip_bg = QColor(0, 0, 0, 70)
            color = QColor(120, 210, 230)  # muted cyan — still visible on dark

        painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 90), 1.0))
        painter.setBrush(QBrush(chip_bg))
        painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 3, 3)

        # Funnel glyph
        painter.setPen(QPen(color, 1.6))
        painter.setBrush(QBrush(color if active else Qt.NoBrush))
        cx = rect.center().x()
        top = rect.top() + 3
        bot = rect.bottom() - 3
        mid_y = top + (bot - top) * 0.52
        half_top = max(3.5, rect.width() * 0.28)
        half_neck = max(1.2, rect.width() * 0.07)
        path = QPainterPath()
        path.moveTo(cx - half_top, top)
        path.lineTo(cx + half_top, top)
        path.lineTo(cx + half_neck, mid_y)
        path.lineTo(cx + half_neck, bot)
        path.lineTo(cx - half_neck, bot)
        path.lineTo(cx - half_neck, mid_y)
        path.closeSubpath()
        painter.drawPath(path)
        painter.restore()

    def paintSection(self, painter, rect, logicalIndex):
        if not rect.isValid():
            return
        # Leave a reserved strip on the right for the funnel (label won't cover it)
        super().paintSection(painter, rect, logicalIndex)

        is_filtered = (
            logicalIndex in self._filters
            and self._filters[logicalIndex]
            and len(self._filters[logicalIndex]) > 0
        )
        hover = logicalIndex == self._mouse_over_col
        icon_h = 16
        icon_w = self.FILTER_ICON_W
        icon_rect = QRect(
            rect.right() - icon_w - self.FILTER_PAD,
            rect.top() + (rect.height() - icon_h) // 2,
            icon_w,
            icon_h,
        )
        # Clip to section so funnel always sits inside the header cell
        painter.save()
        painter.setClipRect(rect)
        self._draw_funnel(painter, icon_rect, active=is_filtered, hover=hover)
        painter.restore()

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        col = self.logicalIndexAt(e.pos())
        if self._mouse_over_col != col:
            self._mouse_over_col = col
            self.viewport().update()

    def leaveEvent(self, e):
        super().leaveEvent(e)
        self._mouse_over_col = -1
        self.viewport().update()

    def mousePressEvent(self, e):
        col = self.logicalIndexAt(e.pos())
        if col < 0:
            super().mousePressEvent(e)
            return
        rect_left = self.sectionViewportPosition(col)
        width = self.sectionSize(col)
        # Right strip (funnel chip) → filter menu; rest of section → sort
        funnel_left = rect_left + width - self.FILTER_ICON_W - self.FILTER_PAD - 2
        if e.pos().x() >= funnel_left:
            self.show_filter_menu(col)
            return
        super().mousePressEvent(e)

    def _resolve_table(self):
        """Find owning QTableWidget (main table or FrozenTableWidget).

        Frozen pane installs this header on a child QTableView; walk parents
        until the real QTableWidget is found so item()/rowCount() work.
        """
        w = self.parent()
        while w is not None:
            if isinstance(w, QTableWidget):
                return w
            w = w.parent()
        return None

    def show_filter_menu(self, col):
        table = self._resolve_table()
        if table is None:
            return

        unique_vals = set()
        for r in range(table.rowCount()):
            item = table.item(r, col)
            if item:
                unique_vals.add(item.text())
                
        def parse_num(v):
            import re
            m = re.search(r"[-+]?\d*\.\d+|\d+", v)
            return float(m.group()) if m else float('-inf')
            
        try:
            unique_vals = sorted(list(unique_vals), key=parse_num)
        except Exception:
            unique_vals = sorted(list(unique_vals))
            
        menu = ExcelFilterMenu(self)
        widget_action = QWidgetAction(menu)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        
        search = QLineEdit()
        search.setPlaceholderText("Search...")
        layout.addWidget(search)
        
        list_widget = QListWidget()
        item_all = QListWidgetItem("(Select All)")
        item_all.setFlags(item_all.flags() | Qt.ItemIsUserCheckable)
        item_all.setCheckState(Qt.Checked if col not in self._filters or not self._filters[col] else Qt.Unchecked)
        list_widget.addItem(item_all)
        
        unchecked = self._filters.get(col, set())
        
        items = []
        for val in unique_vals:
            li = QListWidgetItem(val if val else "(Blank)")
            li.setFlags(li.flags() | Qt.ItemIsUserCheckable)
            li.setCheckState(Qt.Unchecked if val in unchecked else Qt.Checked)
            li.setData(Qt.UserRole, val)
            list_widget.addItem(li)
            items.append(li)
            
        layout.addWidget(list_widget)
        
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setProperty("cancel", True)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
        widget_action.setDefaultWidget(container)
        menu.addAction(widget_action)
        
        def on_search(text):
            text = text.lower()
            for i in range(1, list_widget.count()):
                li = list_widget.item(i)
                li.setHidden(text not in li.text().lower())
                
        search.textChanged.connect(on_search)
        
        def on_item_changed(it):
            list_widget.blockSignals(True)
            if it == item_all:
                st = it.checkState()
                for i in range(1, list_widget.count()):
                    if not list_widget.item(i).isHidden():
                        list_widget.item(i).setCheckState(st)
            else:
                all_chk = all(list_widget.item(i).checkState() == Qt.Checked 
                              for i in range(1, list_widget.count()) if not list_widget.item(i).isHidden())
                item_all.setCheckState(Qt.Checked if all_chk else Qt.Unchecked)
            list_widget.blockSignals(False)
            
        list_widget.itemChanged.connect(on_item_changed)
        
        def apply_filter():
            new_unchecked = set()
            for i in range(1, list_widget.count()):
                it = list_widget.item(i)
                if it.checkState() == Qt.Unchecked:
                    new_unchecked.add(it.data(Qt.UserRole))
            self._filters[col] = new_unchecked
            menu.close()
            self.filter_applied.emit(col)
            self.viewport().update()
            
        btn_ok.clicked.connect(apply_filter)
        btn_cancel.clicked.connect(menu.close)
        
        rect = self.sectionViewportPosition(col)
        pos = self.viewport().mapToGlobal(QPoint(rect, self.height()))
        menu.exec_(pos)

    def is_row_hidden(self, table, row):
        for c, unchk in self._filters.items():
            if not unchk: continue
            it = table.item(row, c)
            val = it.text() if it else ""
            if val in unchk: return True
        return False

from inno3d.features.viewer.mpr_nav import MprNavMixin
from inno3d.features.viewer.crosshair import CrosshairMixin


class MultiPlanarView(MprNavMixin, CrosshairMixin, QWidget):
    """Multi-Planar View with Crosshair Tracking.

    Navigation (zoom/pan/scroll) behaviour lives in MprNavMixin.
    Crosshair hit-test / drag / updatePoint lives in CrosshairMixin.
    This class retains the render pipeline, 3D volume, UI setup, and
    all methods not yet extracted.
    """

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.volume_data = None
        self.segmentation_data = None
        self.class1_data = None  # ThÃƒÂªm
        self.class2_data = None  # ThÃƒÂªm
        
        # Backup for alignment reset
        self.original_volume_data = None
        self.original_class1_data = None
        self.original_class2_data = None
        self.original_labeled_class1_data = None
        self.original_labeled_class2_data = None
       
        self.spacing = [1.0, 1.0, 1.0]
        self.custom_spacing = None
        self.current_slices = {'sagittal': 0, 'coronal': 0, 'axial': 0}
        
        # Advanced 3D Controls
        self.auto_rotate_timer = QTimer(self)
        self.auto_rotate_timer.timeout.connect(self.auto_rotate_step)
        self.adv_mpr_actors = {'x': None, 'y': None, 'z': None}
        self.adv_clip_planes = {'x': None, 'y': None, 'z': None}
        self.bbox_actor = None

        # ── Dragonfly-style clip box (2×2 right sidebar — linked MPR + 3D) ──
        # Independent from fullscreen "Advanced MPR & Clipping" axis planes.
        # Bounds are percentages 0..1000 of world extent per axis (min, max).
        self._df_clip_enabled = False
        self._df_clip_bounds = {
            'x': [0, 1000],
            'y': [0, 1000],
            'z': [0, 1000],
        }
        self._df_clip_show_borders = True
        self._df_clip_show_grid = True
        self._df_clip_show_axes = False
        self._df_clip_keep_when_hidden = True
        self._df_clip_display_grid_on_object = True
        self._df_clip_grid_size = 50.0  # world units (µm when spacing is µm)
        self._df_clip_3d_actors = []
        self._df_clip_mpr_actors = {'axial': [], 'coronal': [], 'sagittal': []}
        # Interactive handles (MPR drag like Dragonfly)
        self._df_clip_hover = None   # dict hit or None
        self._df_clip_drag = None    # active drag state or None
        # 3D viewport face drag / hover
        self._df_clip_3d_hover = None  # {'axis': 'x'|'y'|'z', 'side': 'lo'|'hi'}
        self._df_clip_3d_drag = None   # active 3D face drag state
        
        self.window_level = {'axial': None, 'coronal': None, 'sagittal': None}
        self.camera_state = {'axial': None, 'coronal': None, 'sagittal': None}
        
        self.seg_colors = {
            128: [1.0, 1.0, 0.0],  # Default Yellow for Class 1
            255: [1.0, 0.0, 0.0]   # Default Red for Class 2
        }
        
        self.volume_color = [0.95, 0.95, 0.95] # Default Silver/Grayscale (Dragonfly style)
        self.volume_actor = None
        self.load_thread = None
        
        # Slices widgets and overlays
        self.axial_widget = None
        self.coronal_widget = None
        self.sagittal_widget = None
        self.axial_overlay_group = None
        self.coronal_overlay_group = None
        self.sagittal_overlay_group = None
        self.axial_renderer = None
        self.coronal_renderer = None
        self.sagittal_renderer = None
        self.axial_pixel_label = None
        self.coronal_pixel_label = None
        self.sagittal_pixel_label = None
        
        # Crosshair
        self.crosshair_enabled = False
        self.crosshair_position = [0, 0, 0]  # X, Y, Z in volume coordinates
        self.crosshair_actors = {'axial': [], 'coronal': [], 'sagittal': []}
        self.crosshair_color = [1.0, 1.0, 0.0]  # Default yellow
        self.ruler_color = [0.2, 0.9, 0.2]      # Default green
        self.reverse_z = False  # Z direction toggle
        # Dragonfly-style hover / grab:
        #   _crosshair_hover = (orientation, 'center'|'h'|'v') or None
        #   _crosshair_drag_mode = 'center'|'h'|'v' while dragging
        self._crosshair_hover = None
        self._crosshair_drag_mode = None
        self._is_dragging_crosshair = False
        self._crosshair_click_placed = False  # click-to-place vs handle drag
        # P0/P1 nav: throttle linked re-render while dragging crosshair
        self._ch_sync_timer = QTimer(self)
        self._ch_sync_timer.setSingleShot(True)
        self._ch_sync_timer.setInterval(20)  # ~50 Hz full sync max
        self._ch_sync_timer.timeout.connect(self._flush_crosshair_sync)
        self._ch_sync_pending = False
        # MPR pan: (orientation, last QPoint) while middle / Shift+LMB
        self._mpr_pan = None
        # Link zoom/pan across the three MPR panes (optional)
        self._mpr_link_nav = False
        
        # Object measurement & highlight
        self.labeled_class1_data = None  # labeled array from skimage.measure.label
        self.labeled_class2_data = None
        self.object_stats = []  # list of dicts with measurement data
        self.selected_highlight_objects = []  # list of (class_num, obj_id) tuples
        self.highlight_color = [0.0, 1.0, 1.0]  # cyan highlight
        self._mes_3d_highlight_actors = []  # surface actors for MES pick on 3D volume
        self._mes_3d_hl_cache_key = None  # skip rebuild when same labels/bbox
        self.projection_mode = "perspective"
        
        self.rotate_mode = False
        self.last_mouse_pos = None
        
        # Oblique MPR rotation
        self.oblique_angles = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
        self._oblique_hover_handle = None
        self._oblique_dragging = None
        self._manual_align_active = False
        self.oblique_handles_enabled = False  # Dragonfly-style rotation arrows toggle
        
        self.init_ui()
        
        # ESC shortcut to exit in-grid fullscreen
        self._esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self._esc_shortcut.activated.connect(self.exit_fullscreen)
        # P1: view navigation shortcuts (active MPR pane)
        self._fit_shortcut = QShortcut(QKeySequence("Ctrl+Shift+0"), self)
        self._fit_shortcut.activated.connect(self._shortcut_fit_active_pane)
        self._one_to_one_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        self._one_to_one_shortcut.activated.connect(self._shortcut_one_to_one_active_pane)
        self._reset_view_shortcut = QShortcut(QKeySequence(Qt.Key_R), self)
        self._reset_view_shortcut.activated.connect(self._shortcut_reset_active_pane)
        self._reset_view_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        
    def init_ui(self):
        layout = QVBoxLayout()
        # Tight outer margins so the canvas reads as one linked viewport
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        self.info_label = QLabel("No data loaded")
        self.info_label.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-style: italic; font-size: 8pt;")
        # Note: top_widget buttons moved to MainWindow sidebar
        
        # --- Multi-pane Grid View (Dragonfly-style linked panes) ---
        # Technique: 1px grid spacing + divider-colored background creates
        # a continuous hairline cross between black panes (no card gaps).
        self._grid_divider_color = "#1a2430"
        grid_widget = QWidget()
        grid_widget.setObjectName("PlanesGrid")
        grid_widget.setStyleSheet(
            f"#PlanesGrid {{"
            f"  background-color: {self._grid_divider_color};"
            f"  border: 1px solid {self._grid_divider_color};"
            f"}}"
        )
        grid_layout = QGridLayout()
        grid_layout.setSpacing(1)  # hairline divider thickness
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)
        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 1)
        
        axial_widget = self.create_slice_view("XY", "axial")
        grid_layout.addWidget(axial_widget, 0, 0, 1, 1)
        
        sagittal_widget = self.create_slice_view("YZ", "sagittal") 
        grid_layout.addWidget(sagittal_widget, 0, 1, 1, 1)
        
        coronal_widget = self.create_slice_view("XZ", "coronal")
        grid_layout.addWidget(coronal_widget, 1, 0, 1, 1)
        
        view_3d_widget = self.create_3d_view()
        grid_layout.addWidget(view_3d_widget, 1, 1, 1, 1)
        
        grid_widget.setLayout(grid_layout)
        self._grid_widget = grid_widget
        
        # Store references for layout switching + in-grid fullscreen
        self._grid_layout = grid_layout
        self._grid_widgets = {
            'axial': axial_widget,
            'sagittal': sagittal_widget,
            'coronal': coronal_widget,
            'view_3d': view_3d_widget,
        }
        # (widget, row, col, rowspan, colspan)
        self._grid_cells = {
            'axial':    (axial_widget,    0, 0, 1, 1),
            'sagittal': (sagittal_widget, 0, 1, 1, 1),
            'coronal':  (coronal_widget,  1, 0, 1, 1),
            'view_3d':  (view_3d_widget,  1, 1, 1, 1),
        }
        self._layout_visible_keys = set(self._grid_cells.keys())
        self._grid_nrows = 2
        self._grid_ncols = 2
        self._current_layout_id = "mpr_2x2"
        self._fullscreen_view = None  # Track which view is maximized
        self._active_grid_pane = None
        # Default focus ring on XY (axial) — subtle Dragonfly-style active pane
        self._set_active_grid_pane('axial')

        # Left vertical layout picker — hidden until SYSTEM → LAYOUT toggle
        self.layout_sidebar = self._create_layout_sidebar()
        self.layout_sidebar.setVisible(False)

        # Online CONTEXT: Wafer + Chip (FOV) — left of MPR, shown only in Online mode
        self.context_map_panel = ContextMapPanel(self)
        self.context_map_panel.setVisible(False)
        self.context_map_panel.chip_selected.connect(self._on_context_chip_selected)
        self.context_map_panel.fov_selected.connect(self._on_context_fov_selected)
        self.context_map_panel.selection_changed.connect(self._on_context_selection_changed)
        self._wafer_context = None  # type: Optional[WaferContext]
        
        # Wrap grid inside left side of a splitter
        self.stats_panel = self.create_object_stats_panel()
        self.stats_panel.setVisible(False) # Hidden by default, shown in Online mode
        
        # Right tools host: full Window Leveling sidebar OR thin reopen rail
        # Online mode auto-collapses to rail so MPR + CONTEXT + Statistics get space.
        self._online_viewer_mode = False
        self._align_tools_expanded = True
        self._align_tools_user_expanded = False  # session pref while Online is ON
        self.align_tools_host = self._create_align_tools_host()
        self.align_tools_host.setVisible(False)  # until volume is loaded

        # Body: [layout icons | CONTEXT wafer/chip | MPR+tools+stats]
        # MPR grid is untouched — CONTEXT sits to its left when Online is ON.
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self.layout_sidebar, 0)
        body_layout.addWidget(self.context_map_panel, 0)
        
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setStyleSheet(f"QSplitter::handle {{ background: {SemiconductorTheme.BORDER_DEFAULT}; }}")
        main_splitter.addWidget(grid_widget)
        main_splitter.addWidget(self.align_tools_host)
        main_splitter.addWidget(self.stats_panel)
        main_splitter.setSizes([1400, 240, 300])
        self._main_splitter = main_splitter
        body_layout.addWidget(main_splitter, 1)
        
        layout.addWidget(body, 1)  # Stretch factor 1
        self.setLayout(layout)
    
    def toggle_layout_sidebar(self, checked=False):
        """Show/hide the left Dragonfly layout-preset strip (SYSTEM toggle)."""
        if not hasattr(self, "layout_sidebar") or self.layout_sidebar is None:
            return
        self.layout_sidebar.setVisible(bool(checked))
        # Reflow VTK overlays after width change — do not reset cameras
        QTimer.singleShot(40, self._reposition_visible_overlays)

    def _reposition_visible_overlays(self):
        """Reposition / re-show slice/3D overlay tool strips (C1, C2, FS, Reset).

        Always visits every known overlay — not only layout-visible keys — so
        Online progress dialogs cannot leave strips permanently hidden.
        """
        try:
            QApplication.processEvents()
        except Exception:
            pass
        visible = getattr(self, "_layout_visible_keys", None)
        if not visible:
            visible = set(getattr(self, "_grid_cells", {}).keys()) or {
                "axial", "coronal", "sagittal", "view_3d"
            }
        # Always include all standard panes even if layout set is partial
        keys = set(visible) | {"axial", "coronal", "sagittal", "view_3d"}
        for key in keys:
            if key == "view_3d":
                overlay = getattr(self, "view_3d_overlay_group", None)
                vtk_w = getattr(self, "view_3d_widget", None)
            else:
                overlay = getattr(self, f"{key}_overlay_group", None)
                vtk_w = getattr(self, f"{key}_widget", None)
            if overlay is not None:
                try:
                    overlay.update_position()
                except Exception:
                    pass
            if vtk_w is not None:
                try:
                    rw = vtk_w.GetRenderWindow()
                    if rw is not None:
                        rw.Render()
                except Exception:
                    pass

    def is_layout_sidebar_visible(self):
        return bool(
            hasattr(self, "layout_sidebar")
            and self.layout_sidebar is not None
            and self.layout_sidebar.isVisible()
        )

    # ── Online CONTEXT (Wafer → Chip → FOV) ─────────────────────────────
    def set_online_context_visible(self, visible: bool):
        """Show/hide Wafer+Chip CONTEXT panel (Online mode only). Does not touch MPR."""
        panel = getattr(self, "context_map_panel", None)
        if panel is None:
            return
        show = bool(visible)
        panel.setVisible(show)
        if show and panel.context() is None:
            # New wafer: geometry only (Pending) — no fake OK/NG before inspect
            self.load_wafer_context_empty()
        QTimer.singleShot(40, self._reposition_visible_overlays)

    def is_online_context_visible(self) -> bool:
        panel = getattr(self, "context_map_panel", None)
        return bool(panel is not None and panel.isVisible())

    def load_wafer_context_empty(self):
        """Load uninspected wafer map (circle of Pending dies)."""
        if not hasattr(self, "context_map_panel") or self.context_map_panel is None:
            return None
        if hasattr(self.context_map_panel, "load_empty"):
            ctx = self.context_map_panel.load_empty()
        else:
            ctx = self.context_map_panel.load_demo()
        self._wafer_context = ctx
        return ctx

    def load_wafer_context_demo(self):
        """Load synthetic OK/NG map (email figure) — mock/debug only."""
        if not hasattr(self, "context_map_panel") or self.context_map_panel is None:
            return None
        ctx = self.context_map_panel.load_demo()
        self._wafer_context = ctx
        return ctx

    def load_wafer_context(self, data):
        """Load Wafer/Chip/FOV map from host payload (dict) or WaferContext."""
        if not hasattr(self, "context_map_panel") or self.context_map_panel is None:
            return None
        if isinstance(data, WaferContext):
            self.context_map_panel.set_context(data)
            self._wafer_context = data
            return data
        if isinstance(data, dict):
            ctx = self.context_map_panel.load_from_dict(data)
            self._wafer_context = ctx
            return ctx
        return None

    def get_wafer_context(self):
        """Return current WaferContext (selection + map), or None."""
        panel = getattr(self, "context_map_panel", None)
        if panel is not None and panel.context() is not None:
            return panel.context()
        return getattr(self, "_wafer_context", None)

    def get_active_fov_results_dir(self, root=None) -> str:
        """Path where inspection results for the active FOV should be written.

        Host mass-production layout::

            .../ChipLocation/FOVLocation/Results
        """
        ctx = self.get_wafer_context()
        if ctx is None:
            return ""
        return ctx.results_dir_for_selection(root)

    def apply_host_volume_path(self, path: str):
        """Parse host Input.tiff path and select Chip + FOV on CONTEXT maps."""
        from inno3d.core.wafer_context import parse_host_volume_path
        info = parse_host_volume_path(path)
        panel = getattr(self, "context_map_panel", None)
        if panel is not None and hasattr(panel, "apply_host_path_info"):
            ctx = panel.apply_host_path_info(info)
            self._wafer_context = ctx
            return info
        ctx = self.get_wafer_context()
        if ctx is None:
            self.load_wafer_context_demo()
            ctx = self.get_wafer_context()
        if ctx is not None:
            ctx.apply_host_path(info)
            self.load_wafer_context(ctx)
        return info

    def _on_context_chip_selected(self, col: int, row: int):
        """Chip die selected on wafer map — volume remains FOV of that chip's active point."""
        ctx = self.get_wafer_context()
        if ctx is None:
            return
        self._wafer_context = ctx
        # Future: request/load FOV volume for (col,row, selected_fov) from CT PC / local cache

    def _on_context_fov_selected(self, index: int):
        """FOV point P1..P9 selected — this is the volume identity in MPR/3D."""
        ctx = self.get_wafer_context()
        if ctx is None:
            return
        self._wafer_context = ctx
        # Future: swap loaded volume to this FOV's CT path when host provides volume_path

    def _on_context_selection_changed(self):
        """Keep internal ref in sync after any CONTEXT selection change."""
        panel = getattr(self, "context_map_panel", None)
        if panel is not None:
            self._wafer_context = panel.context()

    def _create_layout_sidebar(self):
        """Left vertical strip: Dragonfly-style view layout presets.

        Icons are mini grid thumbnails; cells with a cube glyph are the
        3D volume pane so users can see where volume rendering sits.
        Hidden by default — toggled from SYSTEM → LAYOUT in the app sidebar.
        """
        panel = QWidget()
        panel.setObjectName("LayoutSidebar")
        panel.setFixedWidth(52)
        panel.setStyleSheet(f"""
            QWidget#LayoutSidebar {{
                background-color: {SemiconductorTheme.BG_MEDIUM};
                border-right: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QPushButton#LayoutPresetBtn {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 2px;
                margin: 0px;
            }}
            QPushButton#LayoutPresetBtn:hover {{
                background-color: {SemiconductorTheme.BG_LIGHT};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QPushButton#LayoutPresetBtn:checked {{
                background-color: {SemiconductorTheme.BG_PANEL};
                border: 1px solid {SemiconductorTheme.PRIMARY_DEFAULT};
            }}
            QScrollArea#LayoutSidebarScroll {{
                background: transparent;
                border: none;
            }}
        """)

        root = QVBoxLayout(panel)
        root.setContentsMargins(4, 8, 4, 8)
        root.setSpacing(4)

        title = QLabel("LAYOUT")
        title.setAlignment(Qt.AlignHCenter)
        title.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_DISABLED}; font-weight: 700;"
            f" font-size: 6.5pt; letter-spacing: 1px;"
        )
        title.setToolTip(
            "View grid layout (Dragonfly-style).\n"
            "Cube icon = 3D volume pane; plain cells = MPR slices."
        )
        root.addWidget(title)

        legend = QLabel("cube\n=3D")
        legend.setAlignment(Qt.AlignHCenter)
        legend.setStyleSheet(
            f"color: {SemiconductorTheme.PRIMARY_DEFAULT}; font-size: 6.5pt; font-weight: 600;"
        )
        legend.setToolTip("Icon cells with a cube glyph = 3D volume pane.\nPlain cells = MPR slices (XY / YZ / XZ).")
        root.addWidget(legend)

        scroll = QScrollArea()
        scroll.setObjectName("LayoutSidebarScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 2, 0, 2)
        inner_layout.setSpacing(4)
        inner_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        self._layout_btn_group = QButtonGroup(self)
        self._layout_btn_group.setExclusive(True)
        self._layout_preset_btns = {}

        icon_size = 36
        # Strong local stylesheet so global QPushButton padding/min-height
        # cannot collapse/bleach these thumbnail icons.
        layout_btn_qss = f"""
            QPushButton#LayoutPresetBtn {{
                background-color: {SemiconductorTheme.BG_LIGHT};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 6px;
                padding: 0px;
                margin: 0px;
                min-height: 0px;
                min-width: 0px;
                max-height: {icon_size + 8}px;
                max-width: {icon_size + 8}px;
            }}
            QPushButton#LayoutPresetBtn:hover {{
                background-color: {SemiconductorTheme.BG_PANEL};
                border: 1px solid {SemiconductorTheme.PRIMARY_DEFAULT};
            }}
            QPushButton#LayoutPresetBtn:checked {{
                background-color: {SemiconductorTheme.BG_PANEL};
                border: 1px solid {SemiconductorTheme.PRIMARY_DEFAULT};
            }}
        """
        for preset in VIEW_LAYOUT_PRESETS:
            btn = QPushButton()
            btn.setObjectName("LayoutPresetBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedSize(icon_size + 8, icon_size + 8)
            btn.setIconSize(QSize(icon_size, icon_size))
            # Multi-color grid thumbnails — never monochrome-tint via MainWindow
            btn.setProperty("preserve_icon_color", True)
            btn.setProperty("theme_icon", False)
            btn.setStyleSheet(layout_btn_qss)
            btn.setIcon(_make_view_layout_icon(preset, size=icon_size, selected=False))
            tip = (
                f"{preset['label']}\n{preset['tooltip']}\n\n"
                "Cube glyph = 3D volume · plain cells = MPR (XY/YZ/XZ)"
            )
            btn.setToolTip(tip)
            btn.setProperty("layout_id", preset["id"])
            lid = preset["id"]
            btn.clicked.connect(lambda checked=False, i=lid: self._on_layout_preset_clicked(i))
            self._layout_btn_group.addButton(btn)
            self._layout_preset_btns[preset["id"]] = btn
            inner_layout.addWidget(btn, 0, Qt.AlignHCenter)

        # Default: classic 2×2
        default_btn = self._layout_preset_btns.get("mpr_2x2")
        if default_btn is not None:
            default_btn.setChecked(True)
            default_btn.setIcon(
                _make_view_layout_icon(VIEW_LAYOUT_PRESETS[0], size=icon_size, selected=True)
            )

        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)
        return panel

    def _on_layout_preset_clicked(self, layout_id):
        """Apply a layout preset and refresh selected icon chrome."""
        self.apply_view_layout(layout_id)
        icon_size = 36
        for preset in VIEW_LAYOUT_PRESETS:
            btn = self._layout_preset_btns.get(preset["id"])
            if btn is None:
                continue
            selected = preset["id"] == layout_id
            btn.setChecked(selected)
            btn.setIcon(_make_view_layout_icon(preset, size=icon_size, selected=selected))

    def apply_view_layout(self, layout_id):
        """Rearrange axial/sagittal/coronal/3D panes into a named grid layout."""
        if not hasattr(self, "_grid_layout") or not hasattr(self, "_grid_widgets"):
            return

        preset = next((p for p in VIEW_LAYOUT_PRESETS if p["id"] == layout_id), None)
        if preset is None:
            return

        # Exit in-grid fullscreen first so placements are consistent
        if self._fullscreen_view is not None:
            self.exit_fullscreen()

        placements = preset["placements"]
        nrows = int(preset["rows"])
        ncols = int(preset["cols"])

        # Detach every plane widget from the grid
        for key, widget in self._grid_widgets.items():
            self._grid_layout.removeWidget(widget)
            widget.setVisible(False)
            widget.hide()

        # Clear old stretch factors (up to a safe max)
        for r in range(8):
            self._grid_layout.setRowStretch(r, 0)
            self._grid_layout.setRowMinimumHeight(r, 0)
        for c in range(8):
            self._grid_layout.setColumnStretch(c, 0)
            self._grid_layout.setColumnMinimumWidth(c, 0)

        for r in range(nrows):
            self._grid_layout.setRowStretch(r, 1)
        for c in range(ncols):
            self._grid_layout.setColumnStretch(c, 1)

        # Prefer a larger 3D pane ONLY when it explicitly spans multiple cells
        # (e.g. tall 3D next to stacked MPR). Equal grids like 2×2 / 1×2 must
        # stay uniform — the old "any shared grid" rule made row1/col1 stretch=2
        # and produced the uneven 2×2 (3D bottom-right oversized).
        if "view_3d" in placements and len(placements) > 1:
            r3, c3, rs3, cs3 = placements["view_3d"]
            if rs3 > 1 or cs3 > 1:
                for c in range(c3, c3 + cs3):
                    if 0 <= c < ncols:
                        self._grid_layout.setColumnStretch(c, 2)
                for r in range(r3, r3 + rs3):
                    if 0 <= r < nrows:
                        self._grid_layout.setRowStretch(r, 2)

        new_cells = {}
        for key, (r, c, rs, cs) in placements.items():
            widget = self._grid_widgets.get(key)
            if widget is None:
                continue
            # Expanding + zero min so VTK sizeHint cannot skew equal cells
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            widget.setMinimumSize(0, 0)
            widget.setVisible(True)
            widget.show()
            self._grid_layout.addWidget(widget, r, c, rs, cs)
            new_cells[key] = (widget, r, c, rs, cs)

        # Keep hidden widgets tracked (for fullscreen restore / active ring)
        for key, widget in self._grid_widgets.items():
            if key not in new_cells:
                new_cells[key] = (widget, 0, 0, 1, 1)

        self._grid_cells = new_cells
        self._layout_visible_keys = set(placements.keys())
        self._grid_nrows = nrows
        self._grid_ncols = ncols
        self._current_layout_id = layout_id

        # Focus a visible pane
        preferred = None
        for cand in ("axial", "view_3d", "sagittal", "coronal"):
            if cand in self._layout_visible_keys:
                preferred = cand
                break
        if preferred:
            self._set_active_grid_pane(preferred)

        # Let VTK / overlays reflow after geometry settles, then reset each pane
        QTimer.singleShot(40, self._refresh_layout_viewports)
        # Second pass once Qt has finished resizing native VTK HWNDs
        QTimer.singleShot(120, self._activate_layout_pane_resets)

    def _iter_layout_overlays(self):
        """Yield (key, overlay) for all plane overlays (visible or not)."""
        for key in ("axial", "sagittal", "coronal", "view_3d"):
            if key == "view_3d":
                overlay = getattr(self, "view_3d_overlay_group", None)
            else:
                overlay = getattr(self, f"{key}_overlay_group", None)
            if overlay is not None:
                yield key, overlay

    def _activate_layout_pane_resets(self):
        """Enable + fire Reset on every visible cell (layout-change contract)."""
        visible = getattr(self, "_layout_visible_keys", set())
        for key, overlay in self._iter_layout_overlays():
            if hasattr(overlay, "reset_btn"):
                try:
                    overlay.reset_btn.setEnabled(True)
                    overlay.reset_btn.setVisible(True)
                except Exception:
                    pass
            if key not in visible:
                continue
            try:
                if key == "view_3d":
                    self.reset_3d_view()
                else:
                    self.reset_view(key)
            except Exception:
                pass
            try:
                overlay.update_position()
            except Exception:
                pass

    def _refresh_layout_viewports(self):
        """Reflow VTK widgets after layout change and reset each visible pane.

        After the grid geometry settles, activate each cell's Reset (zoom/pan)
        so cameras fit the new viewport sizes — same action as the per-pane
        refresh button on the overlay strip.
        """
        # Ensure new cell sizes are committed before ResetCamera
        try:
            QApplication.processEvents()
        except Exception:
            pass

        self._activate_layout_pane_resets()

        for key in getattr(self, "_layout_visible_keys", set()):
            vtk_w = None
            if key == "view_3d":
                vtk_w = getattr(self, "view_3d_widget", None)
            else:
                vtk_w = getattr(self, f"{key}_widget", None)
            if vtk_w is not None:
                try:
                    rw = vtk_w.GetRenderWindow()
                    if rw is not None:
                        rw.Render()
                except Exception:
                    pass

    def _make_sidebar_section(self, title, parent_layout, expanded=True, tooltip=None):
        """Collapsible group for the right align sidebar.

        Returns ``body`` (QVBoxLayout host widget) — put section content there.
        Header shows ▼ when open / ▶ when collapsed; click header to toggle.
        """
        section = QWidget()
        section.setObjectName("AlignSidebarSection")
        section_lay = QVBoxLayout(section)
        section_lay.setContentsMargins(0, 0, 0, 0)
        section_lay.setSpacing(4)

        header = QPushButton()
        header.setObjectName("AlignSectionHeader")
        header.setCheckable(True)
        header.setChecked(bool(expanded))
        header.setCursor(Qt.PointingHandCursor)
        header.setFixedHeight(26)
        if tooltip:
            header.setToolTip(tooltip)
        else:
            header.setToolTip(f"Click to expand / collapse · {title}")
        header.setStyleSheet(f"""
            QPushButton#AlignSectionHeader {{
                text-align: left;
                padding: 2px 4px 2px 2px;
                border: none;
                border-bottom: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 0px;
                background: transparent;
                color: {SemiconductorTheme.TEXT_DISABLED};
                font-weight: 700;
                font-size: 7.5pt;
                letter-spacing: 1.5px;
            }}
            QPushButton#AlignSectionHeader:hover {{
                color: {SemiconductorTheme.PRIMARY_DEFAULT};
                background-color: {SemiconductorTheme.BG_LIGHT};
            }}
            QPushButton#AlignSectionHeader:checked {{
                color: {SemiconductorTheme.TEXT_SECONDARY};
            }}
        """)

        def _sync_header_text(is_open):
            arrow = "▼" if is_open else "▶"
            header.setText(f"  {arrow}  {title}")

        body = QWidget()
        body.setObjectName("AlignSectionBody")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(2, 2, 0, 4)
        body_lay.setSpacing(6)
        body.setVisible(bool(expanded))
        _sync_header_text(bool(expanded))

        def _on_toggled(checked):
            body.setVisible(bool(checked))
            _sync_header_text(bool(checked))

        header.toggled.connect(_on_toggled)

        section_lay.addWidget(header)
        section_lay.addWidget(body)
        parent_layout.addWidget(section)

        # Keep refs so expand state can be driven programmatically if needed
        if not hasattr(self, "_align_sidebar_sections"):
            self._align_sidebar_sections = {}
        self._align_sidebar_sections[title] = {
            "header": header,
            "body": body,
            "section": section,
        }
        return body, body_lay

    def _create_align_tools_host(self):
        """Host for VIEW TOOLS: full sidebar (expanded) or thin reopen rail (collapsed)."""
        host = QWidget()
        host.setObjectName("AlignToolsHost")
        host.setStyleSheet(f"""
            QWidget#AlignToolsHost {{
                background: {SemiconductorTheme.BG_MEDIUM};
                border-left: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """)
        lay = QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.align_tools_rail = self._create_align_tools_rail()
        self.align_sidebar = self._create_align_sidebar()
        lay.addWidget(self.align_tools_rail, 0)
        lay.addWidget(self.align_sidebar, 1)

        # Default: full tools (offline). Rail hidden until collapse / Online.
        self.align_tools_rail.setVisible(False)
        self.align_sidebar.setVisible(True)
        host.setMinimumWidth(240)
        host.setMaximumWidth(240)
        return host

    def _create_align_tools_rail(self):
        """Slim strip shown when VIEW TOOLS is collapsed — one click to reopen."""
        rail = QFrame()
        rail.setObjectName("AlignToolsRail")
        rail.setFixedWidth(28)
        rail.setCursor(Qt.PointingHandCursor)
        rail.setToolTip(
            "Show VIEW TOOLS\n"
            "Window Leveling · Clip · Oblique MPR · Alignment"
        )
        rail.setStyleSheet(f"""
            QFrame#AlignToolsRail {{
                background: {SemiconductorTheme.BG_MEDIUM};
                border: none;
                border-right: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QFrame#AlignToolsRail:hover {{
                background: {SemiconductorTheme.BG_LIGHT};
            }}
            QToolButton#AlignToolsRailBtn {{
                background: transparent;
                border: none;
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                padding: 4px 0;
            }}
            QToolButton#AlignToolsRailBtn:hover {{
                background: rgba(34, 174, 209, 0.15);
                border-radius: 4px;
            }}
            QLabel#AlignToolsRailLabel {{
                color: {SemiconductorTheme.TEXT_SECONDARY};
                font-size: 8pt;
                font-weight: 700;
                letter-spacing: 1px;
            }}
        """)
        v = QVBoxLayout(rail)
        v.setContentsMargins(2, 8, 2, 8)
        v.setSpacing(6)

        open_btn = QToolButton()
        open_btn.setObjectName("AlignToolsRailBtn")
        open_btn.setArrowType(Qt.LeftArrow)
        open_btn.setToolTip("Show VIEW TOOLS (Window Leveling, Clip, Align…)")
        open_btn.setFixedSize(24, 24)
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.clicked.connect(lambda: self.set_align_tools_expanded(True))
        v.addWidget(open_btn, 0, Qt.AlignHCenter)

        gear = QToolButton()
        gear.setObjectName("AlignToolsRailBtn")
        gear.setIcon(_ui_icon("settings.svg", gear))
        gear.setIconSize(QSize(16, 16))
        gear.setToolTip("Show VIEW TOOLS")
        gear.setFixedSize(24, 24)
        gear.setCursor(Qt.PointingHandCursor)
        gear.clicked.connect(lambda: self.set_align_tools_expanded(True))
        v.addWidget(gear, 0, Qt.AlignHCenter)

        # Vertical-ish label (stacked letters — readable in 28px rail)
        for ch in ("T", "O", "O", "L", "S"):
            lab = QLabel(ch)
            lab.setObjectName("AlignToolsRailLabel")
            lab.setAlignment(Qt.AlignCenter)
            v.addWidget(lab, 0, Qt.AlignHCenter)

        v.addStretch(1)

        # Click empty rail area also expands
        def _rail_click(_event):
            self.set_align_tools_expanded(True)

        rail.mousePressEvent = _rail_click  # type: ignore[method-assign]
        return rail

    def set_align_tools_expanded(self, expanded, remember=True):
        """Expand (full 240px sidebar) or collapse (28px reopen rail) VIEW TOOLS.

        Online mode defaults to collapsed so MPR / CONTEXT / Statistics are larger.
        ``remember`` stores the choice while Online is ON (session preference).
        """
        expanded = bool(expanded)
        self._align_tools_expanded = expanded
        if remember and getattr(self, "_online_viewer_mode", False):
            self._align_tools_user_expanded = expanded

        if not hasattr(self, "align_tools_host"):
            return

        # Host stays hidden until a volume exists (tools need volume context)
        if self.volume_data is None and not expanded:
            # Still allow rail geometry updates if host already shown
            pass

        if hasattr(self, "align_sidebar") and self.align_sidebar is not None:
            self.align_sidebar.setVisible(expanded)
        if hasattr(self, "align_tools_rail") and self.align_tools_rail is not None:
            self.align_tools_rail.setVisible(not expanded)

        # Collapse button only relevant when panel is open
        if hasattr(self, "align_tools_collapse_btn") and self.align_tools_collapse_btn is not None:
            self.align_tools_collapse_btn.setVisible(expanded)

        if expanded:
            self.align_tools_host.setMinimumWidth(240)
            self.align_tools_host.setMaximumWidth(240)
        else:
            self.align_tools_host.setMinimumWidth(28)
            self.align_tools_host.setMaximumWidth(28)

        self._rebalance_main_splitter_for_tools(expanded)
        QTimer.singleShot(40, self._reposition_visible_overlays)

    def toggle_align_tools(self):
        """Toggle VIEW TOOLS expanded/collapsed."""
        self.set_align_tools_expanded(not getattr(self, "_align_tools_expanded", True))

    def set_online_viewer_layout(self, online):
        """Online ON → auto-collapse VIEW TOOLS (more room for MPR + stats).

        User can reopen via the thin TOOLS rail; choice is remembered for the
        Online session. Online OFF → restore full sidebar when volume is loaded.
        """
        online = bool(online)
        self._online_viewer_mode = online

        if online:
            # Fresh Online session: start collapsed unless already opened this session
            # (toggle ON always resets to collapsed for max inspection space)
            self._align_tools_user_expanded = False
            self._3d_render_force_on_online = False
            if self.volume_data is not None and hasattr(self, "align_tools_host"):
                self.align_tools_host.setVisible(True)
                self.set_align_tools_expanded(False, remember=True)
            elif hasattr(self, "align_tools_host") and self.align_tools_host.isVisible():
                self.set_align_tools_expanded(False, remember=True)
            # Large Online volumes: skip GPU 3D by default (user can re-enable via 3D btn)
            self.set_3d_volume_render_enabled(False, sync_button=True)
        else:
            # Manual mode: full tools when volume present
            if self.volume_data is not None and hasattr(self, "align_tools_host"):
                self.align_tools_host.setVisible(True)
                self.set_align_tools_expanded(True, remember=False)
            elif hasattr(self, "align_sidebar"):
                # No volume — keep hidden
                if hasattr(self, "align_tools_host"):
                    self.align_tools_host.setVisible(False)
            # Offline: restore 3D volume so B2B gap / MES surface review works
            self.set_3d_volume_render_enabled(True, sync_button=True)

        QTimer.singleShot(50, self._reposition_visible_overlays)

    def is_3d_volume_render_enabled(self) -> bool:
        return bool(getattr(self, "_3d_volume_render_enabled", True))

    def _on_3d_render_btn_toggled(self, checked):
        """User toggled the 3D strip button under C2."""
        if getattr(self, "_online_viewer_mode", False):
            # Remember explicit choice during Online session
            self._3d_render_force_on_online = bool(checked)
        self.set_3d_volume_render_enabled(bool(checked), sync_button=False)

    def set_3d_volume_render_enabled(self, enabled, sync_button=True):
        """Enable/disable GPU 3D volume + C1/C2 3D overlays (keeps MPR free).

        When OFF: tear down volume actors (VRAM) and show a short placeholder.
        When ON: full ``render_3d()`` if volume is loaded.
        """
        enabled = bool(enabled)
        prev = bool(getattr(self, "_3d_volume_render_enabled", True))
        self._3d_volume_render_enabled = enabled

        btn = None
        og = getattr(self, "view_3d_overlay_group", None)
        if og is not None:
            btn = getattr(og, "render_3d_btn", None)
        if sync_button and btn is not None:
            btn.blockSignals(True)
            btn.setChecked(enabled)
            btn.blockSignals(False)

        # C1/C2 3D overlays only make sense when volume render is ON
        if og is not None:
            for name in ("c1_btn", "c2_btn", "opacity_slider"):
                w = getattr(og, name, None)
                if w is not None:
                    w.setEnabled(enabled)

        if enabled == prev and enabled and getattr(self, "volume_actor", None) is not None:
            return

        if not enabled:
            self._teardown_3d_volume_actors()
            self._show_3d_disabled_placeholder()
            return

        self._hide_3d_disabled_placeholder()
        if self.volume_data is not None:
            self.render_3d()

    def _teardown_3d_volume_actors(self):
        """Remove GPU volumes / MES surfaces to free VRAM when 3D is OFF."""
        ren = getattr(self, "view_3d_renderer", None)
        if ren is None:
            return
        if getattr(self, "volume_actor", None) is not None:
            try:
                ren.RemoveVolume(self.volume_actor)
            except Exception:
                pass
            self.volume_actor = None
        for name in ("c1_actor_3d", "c2_actor_3d"):
            act = getattr(self, name, None)
            if act is not None:
                try:
                    ren.RemoveVolume(act)
                except Exception:
                    pass
                setattr(self, name, None)
        self._clear_mes_3d_highlight(render=False)
        # Keep B2B actors? Clear them when 3D off (no volume to attach to)
        self._clear_b2b_gap_actors()
        self._volume_mapper = None
        widget = getattr(self, "view_3d_widget", None)
        if widget is not None:
            try:
                ren.ResetCameraClippingRange()
                widget.GetRenderWindow().Render()
            except Exception:
                pass

    def _show_3d_disabled_placeholder(self):
        """Dark 3D pane message when volume rendering is OFF."""
        ren = getattr(self, "view_3d_renderer", None)
        widget = getattr(self, "view_3d_widget", None)
        if ren is None:
            return
        self._hide_3d_disabled_placeholder()
        try:
            text = vtk.vtkTextActor()
            text.SetInput(
                "3D OFF\n"
                "Toggle  3D  under C2 to render volume\n"
                "(B2B gap / MES pick auto-enable when needed)"
            )
            tp = text.GetTextProperty()
            tp.SetFontSize(16)
            tp.SetBold(True)
            tp.SetColor(0.55, 0.72, 0.85)
            tp.SetJustificationToCentered()
            tp.SetVerticalJustificationToCentered()
            tp.SetBackgroundColor(0.04, 0.06, 0.10)
            tp.SetBackgroundOpacity(0.55)
            # Center in viewport
            try:
                text.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
                text.SetPosition(0.5, 0.52)
            except Exception:
                text.SetDisplayPosition(40, 120)
            ren.AddActor2D(text)
            self._3d_disabled_text_actor = text
            if widget is not None:
                widget.GetRenderWindow().Render()
        except Exception:
            self._3d_disabled_text_actor = None

    def _hide_3d_disabled_placeholder(self):
        ren = getattr(self, "view_3d_renderer", None)
        act = getattr(self, "_3d_disabled_text_actor", None)
        if ren is not None and act is not None:
            try:
                ren.RemoveActor(act)
            except Exception:
                try:
                    ren.RemoveActor2D(act)
                except Exception:
                    pass
        self._3d_disabled_text_actor = None

    def ensure_3d_volume_render_for_overlay(self, reason=""):
        """Turn 3D ON if needed (B2B gap / MES surface) then rebuild volume once.

        Returns True when 3D is ready for overlay actors.
        """
        if self.volume_data is None:
            return False
        if self.is_3d_volume_render_enabled() and getattr(self, "volume_actor", None) is not None:
            return True
        if getattr(self, "_online_viewer_mode", False):
            self._3d_render_force_on_online = True
        self.set_3d_volume_render_enabled(True, sync_button=True)
        return (
            self.is_3d_volume_render_enabled()
            and getattr(self, "volume_actor", None) is not None
        )

    def _rebalance_main_splitter_for_tools(self, tools_expanded):
        """Give space freed by collapsing tools to MPR grid (+ stats when Online)."""
        if not hasattr(self, "_main_splitter") or self._main_splitter is None:
            return
        sp = self._main_splitter
        sizes = sp.sizes()
        if len(sizes) < 3:
            return
        total = sum(sizes)
        if total <= 0:
            total = max(sp.width(), 1600)

        tools_w = 240 if tools_expanded else 28
        stats_visible = (
            hasattr(self, "stats_panel")
            and self.stats_panel is not None
            and self.stats_panel.isVisible()
        )
        if stats_visible:
            # Prefer keeping stats readable; surplus goes to MPR
            stats_w = max(sizes[2], 340) if getattr(self, "_online_viewer_mode", False) else max(sizes[2], 280)
            # Cap stats so grid stays usable
            stats_w = min(stats_w, max(280, total // 3))
            grid_w = max(480, total - tools_w - stats_w)
            # If total shrank, shrink stats first
            if grid_w + tools_w + stats_w > total:
                stats_w = max(260, total - tools_w - 480)
                grid_w = max(400, total - tools_w - stats_w)
        else:
            stats_w = 0 if sizes[2] == 0 else max(0, sizes[2])
            grid_w = max(400, total - tools_w - stats_w)

        sp.setSizes([grid_w, tools_w, stats_w])

    def _create_align_sidebar(self):
        """Create a dedicated right sidebar panel for alignment controls.
        
        Design: Carl Zeiss ZEN Industrial — consistent with left sidebar.
        Uses centralized CSS classes: system-text-btn, system-toggle-btn.

        Groups (WINDOW LEVELING / CLIP / OBLIQUE MPR / ALIGNMENT) are
        independently collapsible via arrow headers (▼ open · ▶ closed).
        Top bar has a collapse control (especially useful in Online mode).
        """
        panel = QWidget()
        panel.setObjectName("AlignSidebar")
        panel.setFixedWidth(240)
        panel.setStyleSheet(f"""
            QWidget#AlignSidebar {{
                background-color: {SemiconductorTheme.BG_MEDIUM};
                border: none;
            }}
            QScrollArea#AlignSidebarScroll {{
                background: transparent;
                border: none;
            }}
            QWidget#AlignSidebarTitleBar {{
                background: {SemiconductorTheme.BG_DARK};
                border-bottom: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QToolButton#AlignToolsCollapseBtn {{
                background: transparent;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px;
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                padding: 2px 6px;
                font-weight: 700;
                font-size: 8pt;
            }}
            QToolButton#AlignToolsCollapseBtn:hover {{
                background: rgba(34, 174, 209, 0.15);
                border-color: {SemiconductorTheme.ACCENT_PRIMARY};
            }}
        """)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Title bar: VIEW TOOLS + collapse (free space for Online) ──
        title_bar = QWidget()
        title_bar.setObjectName("AlignSidebarTitleBar")
        title_bar.setFixedHeight(32)
        title_row = QHBoxLayout(title_bar)
        title_row.setContentsMargins(8, 0, 6, 0)
        title_row.setSpacing(6)

        title_lbl = QLabel("VIEW TOOLS")
        title_lbl.setStyleSheet(
            f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-size: 8.5pt;"
            f" font-weight: 700; letter-spacing: 0.8px; background: transparent;"
        )
        title_row.addWidget(title_lbl, 0, Qt.AlignVCenter)
        title_row.addStretch(1)

        self.align_tools_collapse_btn = QToolButton()
        self.align_tools_collapse_btn.setObjectName("AlignToolsCollapseBtn")
        self.align_tools_collapse_btn.setText("Hide ›")
        self.align_tools_collapse_btn.setToolTip(
            "Hide VIEW TOOLS to enlarge MPR / Statistics.\n"
            "Reopen anytime from the thin TOOLS rail on the right of the views."
        )
        self.align_tools_collapse_btn.setCursor(Qt.PointingHandCursor)
        self.align_tools_collapse_btn.clicked.connect(lambda: self.set_align_tools_expanded(False))
        title_row.addWidget(self.align_tools_collapse_btn, 0, Qt.AlignVCenter)
        outer.addWidget(title_bar)

        scroll = QScrollArea()
        scroll.setObjectName("AlignSidebarScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        content.setObjectName("AlignSidebarContent")
        content.setStyleSheet(f"background-color: {SemiconductorTheme.BG_MEDIUM};")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._align_sidebar_sections = {}

        # ── Section: WINDOW LEVELING for 3 MPR faces (XY / YZ / XZ) ──
        _wl_body, wl_lay = self._make_sidebar_section(
            "WINDOW LEVELING",
            layout,
            expanded=True,
            tooltip=(
                "Window/Level for the three MPR views (XY / YZ / XZ).\n"
                "Click header to collapse / expand."
            ),
        )

        mpr_wl_hint = QLabel("MPR · XY / YZ / XZ (linked)")
        mpr_wl_hint.setStyleSheet(
            f"color: {SemiconductorTheme.PRIMARY_DEFAULT}; font-size: 7.5pt; font-weight: 600;"
        )
        wl_lay.addWidget(mpr_wl_hint)

        self.mpr_hist_widget = HistogramWLWidget()
        self.mpr_hist_widget.setMinimumHeight(100)
        self.mpr_hist_widget.setMaximumHeight(120)
        self.mpr_hist_widget.setToolTip(
            "Drag Min/Max markers to adjust window width and level.\n"
            "Applies simultaneously to all three MPR planes."
        )
        self.mpr_hist_widget.rangeChanged.connect(self._on_mpr_wl_hist_changed)
        wl_lay.addWidget(self.mpr_hist_widget)

        sel_lbl = QLabel("Selected range")
        sel_lbl.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt;"
            f" font-weight: 600; letter-spacing: 1px; margin-top: 2px;"
        )
        wl_lay.addWidget(sel_lbl)

        mpr_wl_row = QHBoxLayout()
        mpr_wl_row.setSpacing(4)
        self.mpr_wl_min_spin = QSpinBox()
        self.mpr_wl_min_spin.setRange(0, 65535)
        self.mpr_wl_min_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.mpr_wl_min_spin.setAlignment(Qt.AlignCenter)
        self.mpr_wl_min_spin.setToolTip("Window min intensity (lower black level)")
        self.mpr_wl_min_spin.valueChanged.connect(self._on_mpr_wl_spin_changed)
        mpr_wl_row.addWidget(self.mpr_wl_min_spin)

        self.mpr_wl_max_spin = QSpinBox()
        self.mpr_wl_max_spin.setRange(0, 65535)
        self.mpr_wl_max_spin.setValue(65535)
        self.mpr_wl_max_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.mpr_wl_max_spin.setAlignment(Qt.AlignCenter)
        self.mpr_wl_max_spin.setToolTip("Window max intensity (upper white level)")
        self.mpr_wl_max_spin.valueChanged.connect(self._on_mpr_wl_spin_changed)
        mpr_wl_row.addWidget(self.mpr_wl_max_spin)
        wl_lay.addLayout(mpr_wl_row)

        mpr_wl_info_row = QHBoxLayout()
        mpr_wl_info_row.setContentsMargins(0, 0, 0, 0)
        mpr_wl_info_row.setSpacing(4)
        self.mpr_wl_info_label = QLabel("W: 65535   L: 32767.5")
        self.mpr_wl_info_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 7pt;"
        )
        self.mpr_wl_info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        mpr_wl_info_row.addWidget(self.mpr_wl_info_label, 1)

        self.btn_mpr_wl_reset = QPushButton("Reset")
        self.btn_mpr_wl_reset.setFixedWidth(52)
        self.btn_mpr_wl_reset.setFixedHeight(22)
        self.btn_mpr_wl_reset.setCursor(Qt.PointingHandCursor)
        self.btn_mpr_wl_reset.setToolTip("Reset MPR window/level to full data range")
        self.btn_mpr_wl_reset.clicked.connect(self._reset_mpr_wl_to_data_range)
        mpr_wl_info_row.addWidget(self.btn_mpr_wl_reset)
        wl_lay.addLayout(mpr_wl_info_row)

        # ── Section: CLIP (Dragonfly-style clip box, linked MPR↔3D) ──
        _clip_body, clip_lay = self._make_sidebar_section(
            "CLIP",
            layout,
            expanded=True,
            tooltip="Dragonfly-style clip box (MPR + 3D). Click to collapse / expand.",
        )

        # Toolbar: Enable clip + Reset  |  Grid size
        clip_toolbar = QHBoxLayout()
        clip_toolbar.setSpacing(4)
        clip_toolbar.setContentsMargins(0, 0, 0, 0)

        self.btn_df_clip_enable = QPushButton("  CLIP BOX")
        self.btn_df_clip_enable.setProperty("class", "system-toggle-btn")
        self.btn_df_clip_enable.setCheckable(True)
        self.btn_df_clip_enable.setCursor(Qt.PointingHandCursor)
        self.btn_df_clip_enable.setIcon(_ui_icon("layers.svg", self.btn_df_clip_enable))
        self.btn_df_clip_enable.setIconSize(QSize(14, 14))
        self.btn_df_clip_enable.setToolTip(
            "Dragonfly-style clip box:\n"
            "• ON: show interactive box (MPR + 3D) and edit crop\n"
            "• OFF: hide the box UI but KEEP the current crop\n"
            "• Use Reset to restore the full uncropped volume\n"
            "• Drag faces on 3D / edges on MPR while the tool is ON"
        )
        self.btn_df_clip_enable.clicked.connect(self._df_clip_on_enable_toggled)
        clip_toolbar.addWidget(self.btn_df_clip_enable, 1)

        self.btn_df_clip_reset = QPushButton()
        self.btn_df_clip_reset.setProperty("class", "icon-button")
        self.btn_df_clip_reset.setIcon(_ui_icon("refresh.svg", self.btn_df_clip_reset))
        self.btn_df_clip_reset.setIconSize(QSize(14, 14))
        self.btn_df_clip_reset.setFixedSize(28, 26)
        self.btn_df_clip_reset.setCursor(Qt.PointingHandCursor)
        self.btn_df_clip_reset.setToolTip(
            "Reset clip to full volume (clears crop).\n"
            "Unlike turning CLIP BOX off, this restores the original extent."
        )
        self.btn_df_clip_reset.clicked.connect(self._df_clip_reset)
        clip_toolbar.addWidget(self.btn_df_clip_reset)
        clip_lay.addLayout(clip_toolbar)

        grid_size_row = QHBoxLayout()
        grid_size_row.setSpacing(4)
        grid_size_lbl = QLabel("Grid size:")
        grid_size_lbl.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;"
        )
        grid_size_row.addWidget(grid_size_lbl)
        self.df_clip_grid_size_spin = QDoubleSpinBox()
        self.df_clip_grid_size_spin.setRange(1.0, 10000.0)
        self.df_clip_grid_size_spin.setValue(self._df_clip_grid_size)
        self.df_clip_grid_size_spin.setDecimals(1)
        self.df_clip_grid_size_spin.setSingleStep(10.0)
        self.df_clip_grid_size_spin.setSuffix(" µm")
        self.df_clip_grid_size_spin.setToolTip("Spacing of grid lines on clip faces (world units)")
        self.df_clip_grid_size_spin.valueChanged.connect(self._df_clip_on_options_changed)
        grid_size_row.addWidget(self.df_clip_grid_size_spin, 1)
        clip_lay.addLayout(grid_size_row)

        self.chk_df_clip_keep = QCheckBox("Keep box when volume hidden")
        self.chk_df_clip_keep.setChecked(self._df_clip_keep_when_hidden)
        self.chk_df_clip_keep.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt;")
        self.chk_df_clip_keep.setToolTip("Keep the clip box visible even if the volume is turned off")
        self.chk_df_clip_keep.stateChanged.connect(self._df_clip_on_options_changed)
        clip_lay.addWidget(self.chk_df_clip_keep)

        self.chk_df_clip_grid_on_obj = QCheckBox("Display grid on object")
        self.chk_df_clip_grid_on_obj.setChecked(self._df_clip_display_grid_on_object)
        self.chk_df_clip_grid_on_obj.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt;")
        self.chk_df_clip_grid_on_obj.setToolTip("Draw grid lines on the clipped faces of the 3D volume")
        self.chk_df_clip_grid_on_obj.stateChanged.connect(self._df_clip_on_options_changed)
        clip_lay.addWidget(self.chk_df_clip_grid_on_obj)

        clip_opts = QHBoxLayout()
        clip_opts.setSpacing(6)
        self.chk_df_clip_grid_lines = QCheckBox("Grid lines")
        self.chk_df_clip_grid_lines.setChecked(self._df_clip_show_grid)
        self.chk_df_clip_grid_lines.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt;")
        self.chk_df_clip_grid_lines.stateChanged.connect(self._df_clip_on_options_changed)
        clip_opts.addWidget(self.chk_df_clip_grid_lines)

        self.chk_df_clip_borders = QCheckBox("Borders")
        self.chk_df_clip_borders.setChecked(self._df_clip_show_borders)
        self.chk_df_clip_borders.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt;")
        self.chk_df_clip_borders.stateChanged.connect(self._df_clip_on_options_changed)
        clip_opts.addWidget(self.chk_df_clip_borders)

        self.chk_df_clip_axes = QCheckBox("Axes")
        self.chk_df_clip_axes.setChecked(self._df_clip_show_axes)
        self.chk_df_clip_axes.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt;")
        self.chk_df_clip_axes.setToolTip("Show RGB axis tripod at the clip-box origin")
        self.chk_df_clip_axes.stateChanged.connect(self._df_clip_on_options_changed)
        clip_opts.addWidget(self.chk_df_clip_axes)
        clip_opts.addStretch()
        clip_lay.addLayout(clip_opts)

        # Bound sliders: X / Y / Z  min–max (percent of extent)
        self._df_clip_bound_ctrls = {}
        _axis_colors = {
            'x': ('#FF5555', 'X (Sag)'),
            'y': ('#55FF55', 'Y (Cor)'),
            'z': ('#5588FF', 'Z (Axi)'),
        }
        for axis, (color, label) in _axis_colors.items():
            axis_lbl = QLabel(label)
            axis_lbl.setStyleSheet(
                f"color: {color}; font-weight: 700; font-size: 8pt; margin-top: 4px;"
            )
            clip_lay.addWidget(axis_lbl)

            row = QHBoxLayout()
            row.setSpacing(3)
            lo = QSlider(Qt.Horizontal)
            lo.setRange(0, 1000)
            lo.setValue(0)
            lo.setStyleSheet(
                f"QSlider::groove:horizontal {{ height: 3px; background: #555; }}"
                f"QSlider::handle:horizontal {{ background: {color}; width: 10px; "
                f"margin: -4px 0; border-radius: 5px; }}"
            )
            hi = QSlider(Qt.Horizontal)
            hi.setRange(0, 1000)
            hi.setValue(1000)
            hi.setStyleSheet(
                f"QSlider::groove:horizontal {{ height: 3px; background: #555; }}"
                f"QSlider::handle:horizontal {{ background: {color}; width: 10px; "
                f"margin: -4px 0; border-radius: 5px; }}"
            )
            pct = QLabel("0–100%")
            pct.setFixedWidth(48)
            pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pct.setStyleSheet(f"color: {color}; font-size: 7.5pt;")
            lo.valueChanged.connect(lambda v, a=axis: self._df_clip_on_bound_slider(a, 'lo', v))
            hi.valueChanged.connect(lambda v, a=axis: self._df_clip_on_bound_slider(a, 'hi', v))
            row.addWidget(lo, 1)
            row.addWidget(hi, 1)
            row.addWidget(pct)
            clip_lay.addLayout(row)
            self._df_clip_bound_ctrls[axis] = {'lo': lo, 'hi': hi, 'pct': pct}

        # ── Section: OBLIQUE MPR ──────────────────────────────
        _obl_body, obl_lay = self._make_sidebar_section(
            "OBLIQUE MPR",
            layout,
            expanded=True,
            tooltip="Oblique MPR rotation controls. Click to collapse / expand.",
        )

        # ── Toggle: Rotation Handles (Dragonfly-style) ────────
        self.btn_oblique_handles = QPushButton("  ROTATION HANDLES")
        self.btn_oblique_handles.setProperty("class", "system-toggle-btn")
        self.btn_oblique_handles.setCheckable(True)
        self.btn_oblique_handles.setCursor(Qt.PointingHandCursor)
        self.btn_oblique_handles.setToolTip(
            "Show Dragonfly-style rotation arrow handles on crosshair.\n"
            "Hover over an arrow to highlight it, then drag to rotate\n"
            "the oblique slicing plane around that axis."
        )
        self.btn_oblique_handles.clicked.connect(self._toggle_oblique_handles)
        obl_lay.addWidget(self.btn_oblique_handles)

        # ── Toggle: 2D ROTATE (moved from left SYSTEM sidebar) ──
        self.rotate2d_btn = QPushButton("  2D ROTATE")
        self.rotate2d_btn.setProperty("class", "system-toggle-btn")
        self.rotate2d_btn.setCheckable(True)
        self.rotate2d_btn.setCursor(Qt.PointingHandCursor)
        self.rotate2d_btn.setIcon(_ui_icon("rotate-clockwise-2.svg", self.rotate2d_btn))
        self.rotate2d_btn.setIconSize(QSize(16, 16))
        self.rotate2d_btn.setToolTip(
            "Toggle 2D rotation mode on crosshair views.\n"
            "Drag on a 2D view to roll the camera for that plane."
        )
        self.rotate2d_btn.clicked.connect(self.toggle_rotate_mode)
        obl_lay.addWidget(self.rotate2d_btn)

        # ── Section: VIEW NAV (P0/P1 MPR interaction) ─────
        _nav_body, nav_lay = self._make_sidebar_section(
            "VIEW NAV",
            layout,
            expanded=True,
            tooltip="MPR zoom / pan / crosshair behaviour. Click to collapse / expand.",
        )
        self.btn_link_mpr_nav = QPushButton("  LINK MPR ZOOM/PAN")
        self.btn_link_mpr_nav.setProperty("class", "system-toggle-btn")
        self.btn_link_mpr_nav.setCheckable(True)
        self.btn_link_mpr_nav.setCursor(Qt.PointingHandCursor)
        self.btn_link_mpr_nav.setToolTip(
            "When ON, zoom and pan on one MPR pane are mirrored to the other panes.\n"
            "Crosshair is always linked. Default OFF (Dragonfly-style independent views).\n\n"
            "Mouse:\n"
            "  Scroll = zoom at cursor\n"
            "  Ctrl+Scroll = change slice\n"
            "  Middle-drag or Shift+Left-drag = pan\n"
            "  Left = crosshair (click to place / drag handles)\n"
            "  Double-click = fit pane\n"
            "  Ctrl+0 = 1:1  ·  R = reset view  ·  Ctrl+Shift+0 = fit"
        )
        self.btn_link_mpr_nav.toggled.connect(self._toggle_link_mpr_nav)
        nav_lay.addWidget(self.btn_link_mpr_nav)

        nav_hint = QLabel(
            "Scroll zoom · Mid/Shift-pan · LMB crosshair\n"
            "Dbl-click fit · Ctrl+0 1:1 · R reset"
        )
        nav_hint.setWordWrap(True)
        nav_hint.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt; margin-top: 2px;"
        )
        nav_lay.addWidget(nav_hint)

        # ── Section: ALIGNMENT ───────────────────────────
        _align_body, align_lay = self._make_sidebar_section(
            "ALIGNMENT",
            layout,
            expanded=True,
            tooltip="Auto / manual volume alignment. Click to collapse / expand.",
        )

        # ── Auto Align: symmetric 2-column grid ────────────────
        auto_lbl = QLabel("Auto")
        auto_lbl.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt;"
            f" font-weight: 600; letter-spacing: 1px; margin-top: 4px;"
        )
        align_lay.addWidget(auto_lbl)

        auto_grid = QGridLayout()
        auto_grid.setContentsMargins(0, 0, 0, 0)
        auto_grid.setSpacing(6)
        auto_grid.setColumnStretch(0, 1)
        auto_grid.setColumnStretch(1, 1)

        self.btn_auto_align = QPushButton("  ALIGN")
        self.btn_auto_align.setProperty("class", "system-text-btn")
        self.btn_auto_align.setToolTip("Automatically align HBM chip volume using PCA & Symmetry analysis")
        self.btn_auto_align.setCursor(Qt.PointingHandCursor)
        self.btn_auto_align.clicked.connect(self.align_hbm_volume)
        auto_grid.addWidget(self.btn_auto_align, 0, 0)

        self.btn_reset_align = QPushButton("  RESET")
        self.btn_reset_align.setProperty("class", "system-text-btn")
        self.btn_reset_align.setToolTip("Reset volume and masks to their original unaligned states")
        self.btn_reset_align.setCursor(Qt.PointingHandCursor)
        self.btn_reset_align.clicked.connect(self.reset_hbm_alignment)
        self.btn_reset_align.setEnabled(False)
        auto_grid.addWidget(self.btn_reset_align, 0, 1)

        align_lay.addLayout(auto_grid)

        # ── Manual Align ───────────────────────────────────────
        manual_lbl = QLabel("Manual")
        manual_lbl.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt;"
            f" font-weight: 600; letter-spacing: 1px; margin-top: 6px;"
        )
        align_lay.addWidget(manual_lbl)

        self.btn_manual_align_mode = QPushButton("  Manual Align: OFF")
        self.btn_manual_align_mode.setProperty("class", "system-toggle-btn")
        self.btn_manual_align_mode.setCheckable(True)
        self.btn_manual_align_mode.setCursor(Qt.PointingHandCursor)
        self.btn_manual_align_mode.setToolTip(
            "Enable Manual Align Mode:\n"
            "1. Drag crosshair handles on any 2D view\n"
            "2. Only crosshair rotates (no reslicing)\n"
            "3. Click 'Commit' to apply rotation"
        )
        self.btn_manual_align_mode.clicked.connect(self._toggle_manual_align_mode)
        align_lay.addWidget(self.btn_manual_align_mode)

        # Live angle display
        angle_grid = QGridLayout()
        angle_grid.setSpacing(2)
        angle_grid.setContentsMargins(4, 4, 4, 4)

        _axis_lbl_style = f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;"

        lbl_x = QLabel("Roll (X):")
        lbl_x.setStyleSheet(_axis_lbl_style)
        angle_grid.addWidget(lbl_x, 0, 0)
        self.manual_angle_x_label = QLabel("+0.00°")
        self.manual_angle_x_label.setStyleSheet("font-weight: bold; color: #FF3333; font-size: 9pt;")
        angle_grid.addWidget(self.manual_angle_x_label, 0, 1)

        lbl_y = QLabel("Pitch (Y):")
        lbl_y.setStyleSheet(_axis_lbl_style)
        angle_grid.addWidget(lbl_y, 1, 0)
        self.manual_angle_y_label = QLabel("+0.00°")
        self.manual_angle_y_label.setStyleSheet("font-weight: bold; color: #33FF33; font-size: 9pt;")
        angle_grid.addWidget(self.manual_angle_y_label, 1, 1)

        lbl_z = QLabel("Yaw (Z):")
        lbl_z.setStyleSheet(_axis_lbl_style)
        angle_grid.addWidget(lbl_z, 2, 0)
        self.manual_angle_z_label = QLabel("+0.00°")
        self.manual_angle_z_label.setStyleSheet("font-weight: bold; color: #5588FF; font-size: 9pt;")
        angle_grid.addWidget(self.manual_angle_z_label, 2, 1)

        align_lay.addLayout(angle_grid)

        # Commit + Cancel: symmetric 2-column
        commit_grid = QGridLayout()
        commit_grid.setContentsMargins(0, 0, 0, 0)
        commit_grid.setSpacing(6)
        commit_grid.setColumnStretch(0, 1)
        commit_grid.setColumnStretch(1, 1)

        self.btn_commit_manual_align = QPushButton("  COMMIT")
        self.btn_commit_manual_align.setProperty("class", "system-text-btn")
        self.btn_commit_manual_align.setProperty("role", "success")
        self.btn_commit_manual_align.setToolTip("Apply the current rotation permanently to the volume data")
        self.btn_commit_manual_align.setCursor(Qt.PointingHandCursor)
        self.btn_commit_manual_align.clicked.connect(self.commit_manual_alignment)
        self.btn_commit_manual_align.setEnabled(False)
        commit_grid.addWidget(self.btn_commit_manual_align, 0, 0)

        self.btn_cancel_manual_align = QPushButton("  CANCEL")
        self.btn_cancel_manual_align.setProperty("class", "system-text-btn")
        self.btn_cancel_manual_align.setToolTip("Discard the preview rotation and reset angles to 0")
        self.btn_cancel_manual_align.setCursor(Qt.PointingHandCursor)
        self.btn_cancel_manual_align.clicked.connect(self._cancel_manual_align)
        self.btn_cancel_manual_align.setEnabled(False)
        commit_grid.addWidget(self.btn_cancel_manual_align, 0, 1)

        align_lay.addLayout(commit_grid)

        # ── Export ─────────────────────────────────────────────
        export_lbl = QLabel("Export")
        export_lbl.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt;"
            f" font-weight: 600; letter-spacing: 1px; margin-top: 6px;"
        )
        align_lay.addWidget(export_lbl)

        self.btn_save_aligned = QPushButton("  SAVE VOLUME")
        self.btn_save_aligned.setProperty("class", "system-text-btn")
        self.btn_save_aligned.setToolTip("Save the current volume data to a TIFF stack or RAW file")
        self.btn_save_aligned.setCursor(Qt.PointingHandCursor)
        self.btn_save_aligned.clicked.connect(self.save_aligned_volume)
        align_lay.addWidget(self.btn_save_aligned)

        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return panel
    
    def create_object_stats_panel(self):
        """Create stats panel with MES (Object Stats) + B2B (Boundary) tabs.

        Layout zones (top → bottom), each full-width and non-overlapping:
          1) Title bar  — title | status chip | actions (fixed icons)
          2) Filter bar — search field expands; never collides with actions
          3) Tabs + table / online log (splitter)
        """
        from PyQt5.QtWidgets import QHeaderView, QTabWidget, QSizePolicy, QFrame

        panel = QWidget()
        panel.setObjectName("statsPanel")
        panel.setStyleSheet(f"""
            QWidget#statsPanel {{
                background: {SemiconductorTheme.BG_PANEL};
                border-radius: 6px;
            }}
        """)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 6, 8, 6)
        panel_layout.setSpacing(6)
        panel.setMinimumWidth(340)

        # ── Zone 1: Title bar ──────────────────────────────────────────
        title_zone = QFrame()
        title_zone.setObjectName("statsTitleZone")
        title_zone.setFixedHeight(36)
        title_zone.setStyleSheet(f"""
            QFrame#statsTitleZone {{
                background: {SemiconductorTheme.BG_MEDIUM};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 5px;
            }}
        """)
        title_row = QHBoxLayout(title_zone)
        title_row.setContentsMargins(10, 0, 6, 0)
        title_row.setSpacing(8)

        header_label = QLabel("STATISTICS")
        header_label.setStyleSheet(
            f"font-size: 10pt; font-weight: 700; color: {SemiconductorTheme.ACCENT_PRIMARY};"
            f" letter-spacing: 0.5px; background: transparent; border: none;"
        )
        header_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        title_row.addWidget(header_label, 0, Qt.AlignVCenter)

        # Status chip — elides when panel is narrow; full text on tooltip
        self.stats_info_label = QLabel("Waiting for class data...")
        self.stats_info_label.setObjectName("statsStatusChip")
        self.stats_info_label.setToolTip("Waiting for class data...")
        self.stats_info_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.stats_info_label.setMinimumWidth(48)
        self.stats_info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.stats_info_label.setStyleSheet(f"""
            QLabel#statsStatusChip {{
                color: {SemiconductorTheme.TEXT_SECONDARY};
                font-size: 8.5pt;
                background: {SemiconductorTheme.BG_LIGHT};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 10px;
                padding: 3px 10px;
            }}
        """)
        title_row.addWidget(self.stats_info_label, 1, Qt.AlignVCenter)

        self.show_indices_check = QCheckBox("Index")
        self.show_indices_check.setToolTip(
            "Show Bump ID (Row,Col) on MPR — same as MES Bump ID column.\n"
            "Selected table rows highlight amber. No separate R#/C# edge labels."
        )
        self.show_indices_check.setStyleSheet(
            f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-weight: 600;"
            f" background: transparent; border: none;"
        )
        self.show_indices_check.setChecked(False)
        self.show_indices_check.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.show_indices_check.stateChanged.connect(lambda: self._refresh_all_index_views())
        title_row.addWidget(self.show_indices_check, 0, Qt.AlignVCenter)

        self.export_csv_btn = QPushButton()
        self.export_csv_btn.setProperty("class", "icon-button")
        self.export_csv_btn.setIcon(_ui_icon("download.svg", self))
        self.export_csv_btn.setIconSize(QSize(15, 15))
        self.export_csv_btn.setToolTip("Export statistics CSV (active tab)")
        self.export_csv_btn.setFixedSize(28, 28)
        self.export_csv_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.export_csv_btn.clicked.connect(self.export_stats_csv)
        self.export_csv_btn.setEnabled(False)
        title_row.addWidget(self.export_csv_btn, 0, Qt.AlignVCenter)

        self.clear_highlight_btn = QPushButton()
        self.clear_highlight_btn.setProperty("class", "icon-button")
        self.clear_highlight_btn.setIcon(_ui_icon("x.svg", self))
        self.clear_highlight_btn.setIconSize(QSize(14, 14))
        self.clear_highlight_btn.setToolTip("Clear selection")
        self.clear_highlight_btn.setFixedSize(28, 28)
        self.clear_highlight_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.clear_highlight_btn.clicked.connect(self.clear_object_selection)
        title_row.addWidget(self.clear_highlight_btn, 0, Qt.AlignVCenter)

        panel_layout.addWidget(title_zone)

        # ── Zone 2: Filter bar (own row — never shares space with title/actions) ──
        filter_zone = QFrame()
        filter_zone.setObjectName("statsFilterZone")
        filter_zone.setFixedHeight(34)
        filter_zone.setStyleSheet(f"""
            QFrame#statsFilterZone {{
                background: {SemiconductorTheme.BG_DARK};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 5px;
            }}
        """)
        filter_row = QHBoxLayout(filter_zone)
        filter_row.setContentsMargins(8, 0, 6, 0)
        filter_row.setSpacing(6)

        filter_hint = QLabel("Filter")
        filter_hint.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 8pt; font-weight: 600;"
            f" background: transparent; border: none;"
        )
        filter_hint.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        filter_row.addWidget(filter_hint, 0, Qt.AlignVCenter)

        self.stats_filter_input = QLineEdit()
        self.stats_filter_input.setPlaceholderText("NG, C1, Layer 2, Bump ID…")
        self.stats_filter_input.setClearButtonEnabled(True)
        self.stats_filter_input.setMinimumWidth(80)
        self.stats_filter_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stats_filter_input.setFixedHeight(26)
        self.stats_filter_input.setStyleSheet(f"""
            QLineEdit {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 9pt;
            }}
            QLineEdit:focus {{
                border: 1px solid {SemiconductorTheme.ACCENT_PRIMARY};
            }}
        """)
        self.stats_filter_input.textChanged.connect(self.on_stats_filter_changed)
        filter_row.addWidget(self.stats_filter_input, 1, Qt.AlignVCenter)

        panel_layout.addWidget(filter_zone)

        # ── Zone 3: Tabs MES | B2B ─────────────────────────────────────
        self.stats_tabs = QTabWidget()
        self.stats_tabs.setDocumentMode(True)
        self.stats_tabs.setUsesScrollButtons(True)
        self.stats_tabs.setElideMode(Qt.ElideRight)
        # Fill entire tab strip (avoids black gaps left/right of tab labels)
        self.stats_tabs.setStyleSheet(f"""
            QTabWidget {{
                background: {SemiconductorTheme.BG_MEDIUM};
                border: none;
            }}
            QTabWidget::pane {{
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 0 0 5px 5px;
                background: {SemiconductorTheme.BG_PANEL};
                top: -1px;
                padding: 0px;
            }}
            QTabWidget::tab-bar {{
                alignment: left;
                left: 0px;
            }}
            QTabBar {{
                background: {SemiconductorTheme.BG_MEDIUM};
                qproperty-drawBase: 0;
            }}
            QTabBar::tab {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_SECONDARY};
                min-width: 120px;
                max-width: 220px;
                padding: 7px 16px;
                margin: 0px 1px 0px 0px;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-weight: 600;
                font-size: 9pt;
            }}
            QTabBar::tab:selected {{
                background: {SemiconductorTheme.BG_PANEL};
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                border-color: {SemiconductorTheme.BORDER_DEFAULT};
                border-bottom: 2px solid {SemiconductorTheme.ACCENT_PRIMARY};
            }}
            QTabBar::tab:hover:!selected {{
                color: {SemiconductorTheme.TEXT_PRIMARY};
                background: {SemiconductorTheme.BG_PANEL};
            }}
            QTabBar::scroller {{
                width: 24px;
                background: {SemiconductorTheme.BG_MEDIUM};
            }}
        """)
        # Ensure empty tab-bar region uses panel color (not window black)
        try:
            self.stats_tabs.tabBar().setDrawBase(False)
            self.stats_tabs.tabBar().setAutoFillBackground(True)
            pal = self.stats_tabs.tabBar().palette()
            pal.setColor(pal.Window, QColor(SemiconductorTheme.BG_MEDIUM))
            pal.setColor(pal.Button, QColor(SemiconductorTheme.BG_MEDIUM))
            self.stats_tabs.tabBar().setPalette(pal)
            self.stats_tabs.setAutoFillBackground(True)
            tw_pal = self.stats_tabs.palette()
            tw_pal.setColor(tw_pal.Window, QColor(SemiconductorTheme.BG_MEDIUM))
            self.stats_tabs.setPalette(tw_pal)
        except Exception:
            pass

        # Tab 1 — MES Object Stats
        mes_page = QWidget()
        mes_layout = QVBoxLayout(mes_page)
        mes_layout.setContentsMargins(0, 4, 0, 0)
        # Match Teaching Stats panel columns (teaching.py)
        self.object_stats_table = FrozenTableWidget()
        self.mes_filter_header = ExcelFilterHeader(self.object_stats_table)
        self.object_stats_table.setHorizontalHeader(self.mes_filter_header)
        self.mes_filter_header.filter_applied.connect(self._apply_excel_filters)
        self.mes_filter_header.sortIndicatorChanged.connect(lambda idx, order: self._apply_excel_filters(idx))
        self.object_stats_table.setColumnCount(19)
        self.object_stats_table.setHorizontalHeaderLabels([
            "#", "Layer", "Bump ID", "B. H", "B. V", "V. V", "Ratio (%)",
            "Judgment", "Gap X", "Gap Y",
            "Z_min", "Z_max", "Y_min", "Y_max",
            "X_min", "X_max", "Ctr_Z", "Ctr_Y", "Ctr_X",
        ])
        self.object_stats_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.object_stats_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.object_stats_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.object_stats_table.setSelectionMode(QTableWidget.MultiSelection)
        # Off: full-row OK/NG tint must not fight alternate-row stylesheet
        self.object_stats_table.setAlternatingRowColors(False)
        self.object_stats_table.setShowGrid(True)
        self.object_stats_table.setStyleSheet(SemiconductorTheme.table_stylesheet())
        self.object_stats_table.horizontalHeader().setStretchLastSection(False)
        self.object_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.object_stats_table.verticalHeader().setVisible(False)
        self.object_stats_table.verticalHeader().setDefaultSectionSize(24)
        self.object_stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.object_stats_table.setSortingEnabled(True)
        self.object_stats_table.horizontalHeader().setSortIndicatorShown(True)
        self.object_stats_table.itemSelectionChanged.connect(self.on_object_selection_changed)
        # Freeze first 3 cols (#, Layer, Bump ID) — body-only overlay under shared header
        self.object_stats_table.set_frozen_columns(3)
        self.object_stats_table.setup_frozen()
        mes_layout.addWidget(self.object_stats_table, 1)
        self.stats_tabs.addTab(mes_page, "MES · Object Stats")

        # Tab 2 — B2B Boundary (Teaching format + Layer for multi-layer merge)
        b2b_page = QWidget()
        b2b_layout = QVBoxLayout(b2b_page)
        b2b_layout.setContentsMargins(0, 4, 0, 0)
        self.b2b_stats = []  # list of row dicts
        self.b2b_info_label = QLabel("No B2B results")
        self.b2b_info_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;"
        )
        b2b_layout.addWidget(self.b2b_info_label)
        self.b2b_table = FrozenTableWidget()
        self.b2b_filter_header = ExcelFilterHeader(self.b2b_table)
        self.b2b_table.setHorizontalHeader(self.b2b_filter_header)
        self.b2b_filter_header.filter_applied.connect(self._apply_excel_filters)
        self.b2b_filter_header.sortIndicatorChanged.connect(lambda idx, order: self._apply_excel_filters(idx))
        # Teaching boundary_gap columns + Layer (Src→Dst per direction, 12 layers)
        self.b2b_table.setColumnCount(11)
        self.b2b_table.setHorizontalHeaderLabels([
            "#", "Layer",
            "Source (R,C)", "Dir", "Dest (R,C)",
            "Gap X (µm)", "Gap Y (µm)", "Gap Z (µm)",
            "Euclidean (µm)", "Src Voxel", "Dst Voxel",
        ])
        self.b2b_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.b2b_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.b2b_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.b2b_table.setSelectionMode(QTableWidget.SingleSelection)
        self.b2b_table.setAlternatingRowColors(True)
        self.b2b_table.setShowGrid(True)
        self.b2b_table.verticalHeader().setDefaultSectionSize(24)
        self.b2b_table.setStyleSheet(SemiconductorTheme.table_stylesheet())
        self.b2b_table.horizontalHeader().setStretchLastSection(False)
        self.b2b_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.b2b_table.verticalHeader().setVisible(False)
        self.b2b_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.b2b_table.setSortingEnabled(True)
        self.b2b_table.horizontalHeader().setSortIndicatorShown(True)
        # Freeze #, Layer, Source — body-only under shared header with filter funnels
        self.b2b_table.set_frozen_columns(3)
        self.b2b_table.setup_frozen()
        self.b2b_table.cellClicked.connect(self._on_b2b_row_clicked)
        self._b2b_gap_actors = []
        self.b2b_table.setToolTip(
            "B2B gap on 3D Volume:\n"
            "  · Click row = show · click same row again = hide\n"
            "  · Wireframe cube = exact surface voxel (Z,Y,X)\n"
            "  · Cyan/yellow tint = local object surface at contact\n"
            "  · Tube cyan→yellow + red arrow = gap direction\n"
            "  · Callout = distance + surface voxel coordinates"
        )
        b2b_layout.addWidget(self.b2b_table, 1)
        self.stats_tabs.addTab(b2b_page, "B2B · Boundary")

        # Use a Vertical Splitter to cleanly separate the Tabs and the Online Log
        self.stats_splitter = QSplitter(Qt.Vertical)
        self.stats_splitter.setStyleSheet(f"QSplitter::handle {{ background: {SemiconductorTheme.BORDER_DEFAULT}; height: 4px; border-radius: 2px; margin: 4px 0px; }}")
        self.stats_splitter.addWidget(self.stats_tabs)
        
        # --- Online Log (embedded, hidden by default) ---
        self.online_log_container = QWidget()
        self.online_log_container.setVisible(False)
        log_container_layout = QVBoxLayout(self.online_log_container)
        log_container_layout.setContentsMargins(0, 4, 0, 0)
        log_container_layout.setSpacing(2)

        log_header = QHBoxLayout()
        log_title = QLabel("<b>ONLINE LOG</b>")
        log_title.setStyleSheet(f"font-size: 9pt; color: {SemiconductorTheme.ACCENT_PRIMARY};")
        log_header.addWidget(log_title)
        log_header.addStretch()

        log_clear_btn = QPushButton()
        log_clear_btn.setFixedSize(20, 20)
        log_clear_btn.setToolTip("Clear Log")
        log_clear_btn.setProperty("class", "icon-button")
        log_clear_btn.setIcon(_ui_icon("x.svg", self))
        log_clear_btn.setIconSize(QSize(12, 12))
        log_clear_btn.clicked.connect(lambda: self.online_log_text.clear())
        log_header.addWidget(log_clear_btn)
        log_container_layout.addLayout(log_header)

        self.online_log_text = QTextEdit()
        self.online_log_text.setReadOnly(True)
        self.online_log_text.setMinimumHeight(100)
        self.online_log_text.setStyleSheet(f"""
            QTextEdit {{
                background: {SemiconductorTheme.BG_DARK};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 8.5pt;
                padding: 4px;
            }}
        """)
        log_container_layout.addWidget(self.online_log_text)
        self.stats_splitter.addWidget(self.online_log_container)
        self.stats_splitter.setSizes([600, 200]) # Allocate more space to the table by default
        
        panel_layout.addWidget(self.stats_splitter, 1)
        
        return panel

    def set_online_log_visible(self, visible):
        """Show or hide the embedded online log widget."""
        self.online_log_container.setVisible(visible)

    def set_stats_info(self, text):
        """Update status chip text + tooltip (full text when elided on narrow layout)."""
        if not hasattr(self, "stats_info_label") or self.stats_info_label is None:
            return
        msg = "" if text is None else str(text)
        self.stats_info_label.setText(msg)
        self.stats_info_label.setToolTip(msg)

    def append_online_log(self, msg, color=None):
        """Append a timestamped message to the embedded online log.

        Also ``print`` so main.py StreamToLogger writes into ``Inno3D_Logs/*.txt``.
        """
        import time as _time
        if color is None:
            color = SemiconductorTheme.TEXT_PRIMARY
        ts = _time.strftime('%H:%M:%S')
        # UI panel
        self.online_log_text.append(
            f'<span style="color:{SemiconductorTheme.TEXT_SECONDARY}">[{ts}]</span> '
            f'<span style="color:{color}">{msg}</span>'
        )
        sb = self.online_log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
        # File log (main.py redirects stdout → Inno3D_Logs)
        try:
            print(f"[ONLINE-UI {ts}] {msg}")
        except Exception:
            pass
    
    def run_object_analysis(self):
        """Run skimage-based per-object measurement.
        
        Class 2 objects are always INSIDE class 1 objects.
        We label class 1 objects, then for each C1 object find the C2 volume within it.
        Volume unit: voxels * spacing (1 um^3 per voxel with 1um spacing).
        """
        if self.class1_data is None:
            return
        
        progress = QProgressDialog("Running object analysis...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        
        try:
            self.object_stats = []
            
            # Use configured voxel sizes from SegmentationTab if available
            vx, vy, vz = 1.0, 1.0, 1.0
            z_stretched_4x = False
            main_window = self.window()
            if hasattr(main_window, 'segmentation_tab') and hasattr(main_window.segmentation_tab, 'param_widgets'):
                sw = main_window.segmentation_tab.param_widgets
                if 'voxel_size_x' in sw:
                    vx = sw['voxel_size_x'].value()
                    vy = sw['voxel_size_y'].value()
                    vz = sw['voxel_size_z'].value()
                if sw.get('z_stretched_4x') and sw['z_stretched_4x'].isChecked():
                    z_stretched_4x = True
            
            # If Z-Stretched 4x mode is enabled, divide effective Z voxel size by 4
            if z_stretched_4x:
                vz = vz / 4.0
                print(f"[VIEWER MEASUREMENT] Z-Stretched 4x mode: effective vz = {vz} Âµm")
            
            voxel_volume = vx * vy * vz
            
            # Determine measurement range from blank slices config
            start_slice = 0
            end_slice = self.class1_data.shape[0]
            # Since MultiPlanarView might not have direct access to config, try to get it from main window
            main_window = self.window()
            if hasattr(main_window, 'seg_tab') and hasattr(main_window.seg_tab, 'config') and main_window.seg_tab.config:
                config = main_window.seg_tab.config
                start_slice = max(0, config.blankEnd1)
                end_slice = min(self.class1_data.shape[0], config.blankStart2)
                if end_slice <= start_slice:
                    end_slice = self.class1_data.shape[0]

            # Save start_slice for navigation (slider offset)
            self._measurement_start_slice = start_slice

            # --- Label Class 1 (Bump) objects ---
            progress.setLabelText("Labeling Class 1 (Bump) objects...")
            progress.setValue(10)
            QApplication.processEvents()
            
            c1_subset = self.class1_data[start_slice:end_slice]
            c1_binary = (c1_subset > 0).astype(np.uint8)
            labeled_subset = measure.label(c1_binary)
            c1_props = measure.regionprops(labeled_subset)
            
            self.labeled_class1_data = np.zeros(self.class1_data.shape, dtype=np.uint32)
            self.labeled_class1_data[start_slice:end_slice] = labeled_subset
            
            # Class 2 is used as a binary mask for volume calculation inside Class 1.
            # We don't need to label Class 2, which saves ~7.5GB of memory.
            if self.class2_data is not None:
                pass
            
            progress.setValue(30)
            QApplication.processEvents()
            
            # --- For each C1 object, find C2 volume inside it ---
            c2_subset = self.class2_data[start_slice:end_slice] if self.class2_data is not None else None
            c2_mask_all = (c2_subset > 0) if c2_subset is not None else None

            # TÃƒÂ­nh toÃƒÂ¡n thÃ¡Â»Æ’ tÃƒÂ­ch C2 cho TÃ¡ÂºÂ¤T CÃ¡ÂºÂ¢ cÃƒÂ¡c Ã„â€˜Ã¡Â»â€˜i tÃ†Â°Ã¡Â»Â£ng trong 1 lÃ¡ÂºÂ§n quÃƒÂ©t (CÃ¡Â»Â±c nhanh)
            from scipy import ndimage
            progress.setLabelText("Calculating volumes (optimized)...")
            progress.setValue(80)

            # labels_list lÃƒÂ  danh sÃƒÂ¡ch cÃƒÂ¡c ID Ã„â€˜Ã¡Â»â€˜i tÃ†Â°Ã¡Â»Â£ng [1, 2, 3, ...]
            labels_list = [prop.label for prop in c1_props]

            # ndimage.sum sÃ¡ÂºÂ½ tÃƒÂ­nh tÃ¡Â»â€¢ng sÃ¡Â»â€˜ pixel C2 nÃ¡ÂºÂ±m trong tÃ¡Â»Â«ng vÃƒÂ¹ng cÃ¡Â»Â§a labeled_subset
            if c2_mask_all is not None:
                c2_pixel_counts = ndimage.sum(c2_mask_all, labeled_subset, index=labels_list)
            else:
                c2_pixel_counts = np.zeros(len(labels_list))

            # BÃƒÂ¢y giÃ¡Â»Â vÃƒÂ²ng lÃ¡ÂºÂ·p chÃ¡Â»â€° dÃƒÂ¹ng Ã„â€˜Ã¡Â»Æ’ Ã„â€˜Ã¡Â»â€¹nh dÃ¡ÂºÂ¡ng dÃ¡Â»Â¯ liÃ¡Â»â€¡u, khÃƒÂ´ng tÃƒÂ­nh toÃƒÂ¡n nÃ¡ÂºÂ·ng nÃ¡Â»Â¯a
            for idx, prop in enumerate(c1_props):
                c1_volume = prop.area * voxel_volume
                c2_volume = float(c2_pixel_counts[idx]) * voxel_volume # LÃ¡ÂºÂ¥y kÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ Ã„â€˜ÃƒÂ£ tÃƒÂ­nh sÃ¡ÂºÂµn
                
                ratio = c2_volume / (c1_volume + c2_volume) if (c1_volume + c2_volume) > 0 else 0
                
                bbox = prop.bbox
                centroid = prop.centroid
                
                soh = (bbox[3] - bbox[0]) * vz
                
                self.object_stats.append({
                    'row_id': idx + 1,
                    'label': prop.label,
                    'soh': soh,
                    'c1_volume': c1_volume,
                    'c2_volume': c2_volume,
                    'ratio': ratio,
                    'z_min': bbox[0], 'y_min': bbox[1], 'x_min': bbox[2],
                    'z_max': bbox[3], 'y_max': bbox[4], 'x_max': bbox[5],
                    'centroid_z': centroid[0], 'centroid_y': centroid[1], 'centroid_x': centroid[2],
                })

            
            # --- Apply Automatic FAR (False Alarm Remover) ---
            self.apply_far(voxel_volume)

            # Unified (0,0)=top-left XY indexing (same as Online MES / B2B)
            try:
                from inno3d.core.bumpvoid_mes import reindex_grid_top_left
                self.object_stats = reindex_grid_top_left(
                    self.object_stats, voxel_x=vx, voxel_y=vy, per_layer=True
                )
            except Exception as _ix_err:
                print(f"[VIEWER] reindex_grid_top_left failed: {_ix_err}")
            
            # --- Populate table ---
            progress.setLabelText("Populating table...")
            progress.setValue(92)
            QApplication.processEvents()

            ng_threshold = self._resolve_ng_threshold()
            self._fill_mes_stats_table_rows(ng_threshold=ng_threshold)

            # Count summary
            total_c1 = sum(s['c1_volume'] for s in self.object_stats)
            total_c2 = sum(s['c2_volume'] for s in self.object_stats)
            self.set_stats_info(
                f"{len(self.object_stats)} objects | "
                f"Total C1: {total_c1:,.0f} \u03bcm\u00b3 | Total C2: {total_c2:,.0f} \u03bcm\u00b3"
            )
            self.export_csv_btn.setEnabled(True)
            
            progress.setValue(100)
            progress.close()
            
        except Exception as e:
            progress.close()
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Analysis Error", f"Error during analysis:\n{str(e)}")

    def _resolve_ng_threshold(self, ng_threshold=None):
        if ng_threshold is not None:
            return ng_threshold
        ng_threshold = 0.05
        main_window = self.window()
        if hasattr(main_window, "segmentation_tab") and hasattr(
            main_window.segmentation_tab, "ng_threshold_spin"
        ):
            return main_window.segmentation_tab.ng_threshold_spin.value() / 100.0
        if hasattr(main_window, "seg_tab") and hasattr(main_window.seg_tab, "ng_threshold_spin"):
            return main_window.seg_tab.ng_threshold_spin.value() / 100.0
        return ng_threshold

    def _fill_mes_stats_table_rows(self, ng_threshold=None):
        """Populate MES table with sort-aware items + UserRole stats index mapping.

        Column 0 stores ``Qt.UserRole = stats_idx`` so selection still maps to the
        correct ``object_stats`` entry after the user sorts any column.
        """
        if not getattr(self, "object_stats", None):
            return
        ng_threshold = self._resolve_ng_threshold(ng_threshold)
        table = self.object_stats_table
        from PyQt5.QtWidgets import QHeaderView

        if table.columnCount() != 19:
            table.setColumnCount(19)
            table.setHorizontalHeaderLabels([
                "#", "Layer", "Bump ID", "B. H", "B. V", "V. V", "Ratio (%)",
                "Judgment", "Gap X", "Gap Y",
                "Z_min", "Z_max", "Y_min", "Y_max",
                "X_min", "X_max", "Ctr_Z", "Ctr_Y", "Ctr_X",
            ])

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)

        was_sorting = table.isSortingEnabled()
        table.setSortingEnabled(False)
        table.blockSignals(True)
        table.setRowCount(0)
        table.setRowCount(len(self.object_stats))

        for row, stat in enumerate(self.object_stats):
            is_ng = (
                stat["ratio"] >= ng_threshold
                if stat.get("ratio") != float("inf")
                else True
            )
            judgment = stat.get("judgment")
            if judgment in (1, 8):
                judgment_str = "NG" if judgment == 1 else "OK"
            else:
                judgment_str = "NG" if is_ng else "OK"
            ratio_val = stat.get("ratio", 0)
            ratio_str = (
                f"{ratio_val * 100:.3f}"
                if ratio_val != float("inf")
                else "N/A"
            )
            try:
                row_id_num = float(stat.get("row_id", row + 1))
            except (TypeError, ValueError):
                row_id_num = float(row + 1)

            gr = stat.get("grid_row", "?")
            gc = stat.get("grid_col", "?")
            try:
                bump_sort = float(gr) * 10000.0 + float(gc)
            except (TypeError, ValueError):
                bump_sort = float("-inf")

            items = [
                _numeric_item(str(stat.get("row_id", "")), row_id_num),
                _category_item(str(stat.get("layer_name", "") or "")),
                _numeric_item(f"({gr},{gc})", bump_sort),
                _numeric_item(f"{stat.get('soh', 0):.3f}", stat.get("soh", 0)),
                _numeric_item(f"{stat['c1_volume']:,.3f}", stat.get("c1_volume", 0)),
                _numeric_item(f"{stat['c2_volume']:,.3f}", stat.get("c2_volume", 0)),
                _numeric_item(
                    ratio_str,
                    (ratio_val * 100.0) if ratio_val != float("inf") else float("inf"),
                ),
                _category_item(judgment_str),
                _numeric_item(f"{stat.get('pitch_x', 0):.3f}", stat.get("pitch_x", 0)),
                _numeric_item(f"{stat.get('pitch_y', 0):.3f}", stat.get("pitch_y", 0)),
                _numeric_item(str(stat["z_min"]), stat.get("z_min", 0)),
                _numeric_item(str(stat["z_max"]), stat.get("z_max", 0)),
                _numeric_item(str(stat["y_min"]), stat.get("y_min", 0)),
                _numeric_item(str(stat["y_max"]), stat.get("y_max", 0)),
                _numeric_item(str(stat["x_min"]), stat.get("x_min", 0)),
                _numeric_item(str(stat["x_max"]), stat.get("x_max", 0)),
                _numeric_item(f"{stat['centroid_z']:.3f}", stat.get("centroid_z", 0)),
                _numeric_item(f"{stat['centroid_y']:.3f}", stat.get("centroid_y", 0)),
                _numeric_item(f"{stat['centroid_x']:.3f}", stat.get("centroid_x", 0)),
            ]

            # Full-row judgment tint (main + frozen panes via StatsRowDelegate)
            is_ng_row = judgment_str == "NG"
            if is_ng_row:
                row_bg = QColor(150, 28, 28, 160)       # solid-enough red
                row_fg_default = QColor(255, 228, 228)
            else:
                row_bg = QColor(16, 105, 58, 145)       # solid-enough green
                row_fg_default = QColor(220, 255, 232)
            row_brush = QBrush(row_bg)

            for col, item in enumerate(items):
                # stats_idx for selection→MPR mapping (survives column sort)
                item.setData(Qt.UserRole, row)
                item.setData(Qt.BackgroundRole, row_brush)
                item.setBackground(row_brush)
                item.setForeground(row_fg_default)
                if col == 6:  # Ratio (%)
                    item.setFont(QFont("Arial", 9, QFont.Bold))
                    if is_ng_row:
                        item.setForeground(QColor(255, 200, 120))
                    else:
                        item.setForeground(QColor(255, 220, 140))
                if col == 7:  # Judgment
                    item.setFont(QFont("Arial", 9, QFont.Bold))
                    item.setForeground(
                        QColor(255, 140, 140) if is_ng_row else QColor(130, 255, 170)
                    )
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                table.setItem(row, col, item)
            table.setRowHeight(row, 24)

        # Fixed readable widths (avoid ResizeToContents thrashing freeze overlay)
        header.setSectionResizeMode(QHeaderView.Interactive)
        _mes_widths = {
            0: 44, 1: 72, 2: 70, 3: 64, 4: 78, 5: 72, 6: 72, 7: 70,
            8: 64, 9: 64, 10: 58, 11: 58, 12: 58, 13: 58, 14: 58, 15: 58,
            16: 70, 17: 70, 18: 70,
        }
        for c, w in _mes_widths.items():
            if c < table.columnCount():
                table.setColumnWidth(c, w)

        table.blockSignals(False)
        table.setSortingEnabled(was_sorting if was_sorting is not None else True)
        if hasattr(table, "refresh_frozen"):
            table.refresh_frozen()
        elif hasattr(table, "setup_frozen"):
            table.setup_frozen()

    def populate_object_stats_table(self, stats=None, ng_threshold=None, source_label=""):
        """Fill MES Object Stats table (Teaching-compatible columns)."""
        if stats is not None:
            self.object_stats = list(stats)
        if not getattr(self, "object_stats", None):
            return

        ng_threshold = self._resolve_ng_threshold(ng_threshold)
        self._fill_mes_stats_table_rows(ng_threshold=ng_threshold)

        total_c1 = sum(s["c1_volume"] for s in self.object_stats)
        total_c2 = sum(s["c2_volume"] for s in self.object_stats)
        src = f" [{source_label}]" if source_label else ""
        self.set_stats_info(
            f"{len(self.object_stats)} objects{src} | "
            f"Total C1: {total_c1:,.0f} \u03bcm\u00b3 | Total C2: {total_c2:,.0f} \u03bcm\u00b3"
        )
        self.export_csv_btn.setEnabled(True)
        if hasattr(self, "stats_tabs"):
            try:
                self.stats_tabs.setCurrentIndex(0)
            except Exception:
                pass

    def populate_b2b_table(self, rows=None, source_label="B2B"):
        """Fill B2B Boundary tab like Teaching boundary_gap panel (+ Layer).

        Primary format (golden DLL / reference CSV):
          Src_row, Src_col, Direction, Dst_row, Dst_col, Min_dist_*, voxels
        Legacy summary format still supported if rows lack Direction.
        """
        if rows is not None:
            self.b2b_stats = list(rows)
        rows = getattr(self, "b2b_stats", None) or []
        table = getattr(self, "b2b_table", None)
        if table is None:
            return

        from PyQt5.QtWidgets import QHeaderView

        def _get(r, *keys, default=""):
            for k in keys:
                if k in r and r[k] not in (None, ""):
                    return r[k]
            return default

        is_gap = bool(rows) and (
            "Direction" in rows[0]
            or "Src_row" in rows[0]
            or "Source_id" in rows[0]
        )

        table.setSortingEnabled(False)
        table.blockSignals(True)
        table.setRowCount(0)

        if is_gap:
            # Teaching gap format + Layer
            if table.columnCount() != 11:
                table.setColumnCount(11)
            table.setHorizontalHeaderLabels([
                "#", "Layer",
                "Source (R,C)", "Dir", "Dest (R,C)",
                "Gap X (µm)", "Gap Y (µm)", "Gap Z (µm)",
                "Euclidean (µm)", "Src Voxel", "Dst Voxel",
            ])
            table.setRowCount(len(rows))
            for i, r in enumerate(rows):
                eucl_str = str(_get(
                    r, "Min_dist_Euclidean_um", "min_gap_euclidean_um", default="NaN"
                ))
                try:
                    eucl_val = float(eucl_str)
                    is_close = eucl_val < 1.5
                except (ValueError, TypeError):
                    eucl_val = float("nan")
                    is_close = False

                def _fnum(*keys, default=float("nan")):
                    raw = _get(r, *keys, default="")
                    try:
                        return float(raw)
                    except (TypeError, ValueError):
                        return default

                row_bg = QColor(180, 40, 40, 50) if is_close else QColor(40, 180, 60, 30)
                src_str = f"R{_get(r, 'Src_row', default='?')},C{_get(r, 'Src_col', default='?')}"
                dst_str = f"R{_get(r, 'Dst_row', default='?')},C{_get(r, 'Dst_col', default='?')}"
                direction = str(_get(r, "Direction", default="?"))
                src_pos = (
                    f"Z={_get(r, 'Src_voxel_Z')},"
                    f"Y={_get(r, 'Src_voxel_Y')},"
                    f"X={_get(r, 'Src_voxel_X')}"
                )
                dst_pos = (
                    f"Z={_get(r, 'Dst_voxel_Z')},"
                    f"Y={_get(r, 'Dst_voxel_Y')},"
                    f"X={_get(r, 'Dst_voxel_X')}"
                )
                gx = _fnum("Min_dist_X_um", "min_gap_X_um")
                gy = _fnum("Min_dist_Y_um", "min_gap_Y_um")
                gz = _fnum("Min_dist_Z_um", "min_gap_Z_um")
                try:
                    src_sort = (
                        float(_get(r, "Src_row", default=-1)) * 10000.0
                        + float(_get(r, "Src_col", default=-1))
                    )
                except (TypeError, ValueError):
                    src_sort = float("-inf")
                try:
                    dst_sort = (
                        float(_get(r, "Dst_row", default=-1)) * 10000.0
                        + float(_get(r, "Dst_col", default=-1))
                    )
                except (TypeError, ValueError):
                    dst_sort = float("-inf")

                items = [
                    _numeric_item(str(i + 1), i + 1),
                    _category_item(str(_get(r, "Layer", default=""))),
                    _numeric_item(src_str, src_sort),
                    _category_item(direction),
                    _numeric_item(dst_str, dst_sort),
                    _numeric_item(str(_get(r, "Min_dist_X_um", "min_gap_X_um", default="NaN")), gx),
                    _numeric_item(str(_get(r, "Min_dist_Y_um", "min_gap_Y_um", default="NaN")), gy),
                    _numeric_item(str(_get(r, "Min_dist_Z_um", "min_gap_Z_um", default="NaN")), gz),
                    _numeric_item(eucl_str, eucl_val if eucl_val == eucl_val else float("inf")),
                    _category_item(src_pos),
                    _category_item(dst_pos),
                ]
                for col, item in enumerate(items):
                    item.setData(Qt.UserRole, i)
                    item.setBackground(row_bg)
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                    if col == 8:  # Euclidean
                        item.setFont(QFont("Arial", 9, QFont.Bold))
                        if is_close:
                            item.setForeground(QColor(255, 100, 100))
                        else:
                            item.setForeground(QColor(100, 255, 120))
                    if col == 3:  # Direction
                        dir_colors = {
                            "Up": QColor(100, 200, 255),
                            "Down": QColor(100, 200, 255),
                            "Left": QColor(255, 200, 100),
                            "Right": QColor(255, 200, 100),
                        }
                        item.setForeground(dir_colors.get(direction, QColor(200, 200, 200)))
                    table.setItem(i, col, item)
                table.setRowHeight(i, 24)
            fmt_name = "boundary_gap"
        else:
            # Legacy summary format
            if table.columnCount() != 10:
                table.setColumnCount(10)
            table.setHorizontalHeaderLabels([
                "#", "Layer",
                "Bump (R,C)", "Label", "Bnd Voxels",
                "Gap X (µm)", "Gap Y (µm)", "Gap Z (µm)",
                "Euclidean (µm)", "Min Gap Pos",
            ])
            table.setRowCount(len(rows))
            for i, r in enumerate(rows):
                eucl_str = str(_get(r, "min_gap_euclidean_um", "min_gap_eucl", default="NaN"))
                try:
                    eucl_val = float(eucl_str)
                    is_close = eucl_val < 1.0
                except (ValueError, TypeError):
                    eucl_val = float("nan")
                    is_close = False

                def _fnum2(*keys, default=float("nan")):
                    raw = _get(r, *keys, default="")
                    try:
                        return float(raw)
                    except (TypeError, ValueError):
                        return default

                row_bg = QColor(180, 40, 40, 50) if is_close else QColor(40, 180, 60, 30)
                pos_str = (
                    f"Z={_get(r, 'min_gap_voxel_Z', 'vox_Z')},"
                    f"Y={_get(r, 'min_gap_voxel_Y', 'vox_Y')},"
                    f"X={_get(r, 'min_gap_voxel_X', 'vox_X')}"
                )
                bump_id = str(_get(r, "Bump_id", "Bump (R,C)", "bump_id", default=""))
                if bump_id.startswith('"') and bump_id.endswith('"'):
                    bump_id = bump_id[1:-1]
                bump_id = bump_id.strip()
                if bump_id and not bump_id.startswith("(") and "," in bump_id and not bump_id.startswith("L"):
                    bump_id = f"({bump_id})"

                label_text = str(_get(r, "label", "Label", default=""))
                gx = _fnum2("min_gap_X_um", "min_gap_X")
                gy = _fnum2("min_gap_Y_um", "min_gap_Y")
                gz = _fnum2("min_gap_Z_um", "min_gap_Z")
                bv = _fnum2("boundary_voxels", default=0)

                items = [
                    _numeric_item(str(i + 1), i + 1),
                    _category_item(str(_get(r, "Layer", default=""))),
                    _category_item(bump_id),
                    _category_item(label_text),
                    _numeric_item(str(_get(r, "boundary_voxels", default="")), bv),
                    _numeric_item(str(_get(r, "min_gap_X_um", "min_gap_X", default="NaN")), gx),
                    _numeric_item(str(_get(r, "min_gap_Y_um", "min_gap_Y", default="NaN")), gy),
                    _numeric_item(str(_get(r, "min_gap_Z_um", "min_gap_Z", default="NaN")), gz),
                    _numeric_item(eucl_str, eucl_val if eucl_val == eucl_val else float("inf")),
                    _category_item(pos_str),
                ]
                for col, item in enumerate(items):
                    item.setData(Qt.UserRole, i)
                    item.setBackground(row_bg)
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                    if col == 8:
                        item.setFont(QFont("Arial", 9, QFont.Bold))
                        if is_close:
                            item.setForeground(QColor(255, 100, 100))
                        else:
                            item.setForeground(QColor(100, 255, 120))
                    if col == 3:  # Label
                        if "OK" in label_text.upper():
                            item.setForeground(QColor(100, 255, 120))
                        else:
                            item.setForeground(QColor(200, 200, 200))
                    table.setItem(i, col, item)
                table.setRowHeight(i, 24)
            fmt_name = "boundary_summary"

        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.horizontalHeader().setStretchLastSection(False)
        # Stable column widths for freeze alignment
        if is_gap:
            _b2b_w = {
                0: 40, 1: 72, 2: 78, 3: 48, 4: 78,
                5: 72, 6: 72, 7: 72, 8: 88, 9: 110, 10: 110,
            }
        else:
            _b2b_w = {
                0: 40, 1: 72, 2: 78, 3: 56, 4: 72,
                5: 72, 6: 72, 7: 72, 8: 88, 9: 120,
            }
        for c, w in _b2b_w.items():
            if c < table.columnCount():
                table.setColumnWidth(c, w)

        table.blockSignals(False)
        table.setSortingEnabled(True)

        if hasattr(table, "refresh_frozen"):
            table.refresh_frozen()
        elif hasattr(table, "setup_frozen"):
            table.setup_frozen()

        # New data → drop previous 3D gap overlay
        self._clear_b2b_gap_actors()

        n_layers = len({str(r.get("Layer", "")) for r in rows if r.get("Layer")})
        self.export_csv_btn.setEnabled(True)
        note = (
            f"{len(rows)} rows | {fmt_name} | {n_layers} layers | {source_label} "
            f"| click row → 3D gap"
        )
        if hasattr(self, "b2b_info_label"):
            self.b2b_info_label.setText(note)
        base = self.stats_info_label.text() if hasattr(self, "stats_info_label") else ""
        if base and "objects" in base.lower():
            self.set_stats_info(f"{base}  |  B2B: {len(rows)} gaps")
        else:
            self.set_stats_info(f"B2B: {len(rows)} gaps · {n_layers} layers")

    # ── B2B gap → 3D Volume (spatial distance; not MPR) ──────────────────

    def _b2b_stats_index_from_visual_row(self, visual_row):
        if visual_row is None or visual_row < 0:
            return None
        id_item = self.b2b_table.item(visual_row, 0)
        if id_item is None:
            return None
        idx = id_item.data(Qt.UserRole)
        if idx is None:
            idx = visual_row
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return None
        rows = getattr(self, "b2b_stats", None) or []
        if idx < 0 or idx >= len(rows):
            return None
        return idx

    def _b2b_layer_z_offset(self, row_dict):
        """Local B2B layer Z → full-volume Z."""
        if not row_dict:
            return 0
        for key in ("z_start", "Z_start", "layer_z_start"):
            if key in row_dict and row_dict[key] not in (None, ""):
                try:
                    return int(row_dict[key])
                except (TypeError, ValueError):
                    pass
        ln = str(row_dict.get("Layer", "") or "")
        mw = self.window()
        bands = None
        if hasattr(mw, "_online_layer_bands"):
            try:
                bands = mw._online_layer_bands()
            except Exception:
                bands = None
        if bands:
            key = ln.replace(" ", "_").lower()
            for L in bands:
                name = str(L.get("name", ""))
                if name == ln or name.replace(" ", "_").lower() == key:
                    return int(L.get("z_start", 0) or 0)
        return 0

    def _get_3d_world_spacing(self):
        spacing = getattr(self, "custom_spacing", None)
        if spacing is None:
            spacing = getattr(self, "spacing", [1.0, 1.0, 1.0])
        try:
            return (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        except Exception:
            return (1.0, 1.0, 1.0)

    def _clear_b2b_gap_actors(self):
        ren = getattr(self, "view_3d_renderer", None)
        actors = getattr(self, "_b2b_gap_actors", None) or []
        if ren is not None:
            for actor in actors:
                try:
                    ren.RemoveActor(actor)
                except Exception:
                    pass
                try:
                    ren.RemoveActor2D(actor)
                except Exception:
                    pass
        self._b2b_gap_actors = []
        self._b2b_active_stats_idx = None
        widget = getattr(self, "view_3d_widget", None)
        if widget is not None and ren is not None:
            try:
                widget.GetRenderWindow().Render()
            except Exception:
                pass

    def _b2b_add_gap_actor(self, ren, actor, is_2d=False):
        """Track + add a B2B overlay actor (3D or 2D caption)."""
        if is_2d:
            ren.AddActor2D(actor)
        else:
            ren.AddActor(actor)
        self._b2b_gap_actors.append(actor)

    def _b2b_add_surface_voxel_marker(self, ren, z, y, x, sx, sy, sz, color,
                                     label_tag=""):
        """Mark the **exact discrete surface voxel** (not a floating ball).

        · Wireframe cube = 1 voxel cell centered on sample (Z,Y,X)
        · Filled translucent cube (same cell)
        · Tiny solid sphere at voxel sample center
        """
        # Cell bounds around sample point (index * spacing)
        x0, x1 = (float(x) - 0.5) * sx, (float(x) + 0.5) * sx
        y0, y1 = (float(y) - 0.5) * sy, (float(y) + 0.5) * sy
        z0, z1 = (float(z) - 0.5) * sz, (float(z) + 0.5) * sz
        cx, cy, cz = float(x) * sx, float(y) * sy, float(z) * sz
        min_sp = max(1e-6, min(sx, sy, sz))

        # Filled voxel cell
        cube = vtk.vtkCubeSource()
        cube.SetBounds(x0, x1, y0, y1, z0, z1)
        cube.Update()
        fill_m = vtk.vtkPolyDataMapper()
        fill_m.SetInputConnection(cube.GetOutputPort())
        fill_a = vtk.vtkActor()
        fill_a.SetMapper(fill_m)
        fill_a.GetProperty().SetColor(*color)
        fill_a.GetProperty().SetOpacity(0.55)
        fill_a.GetProperty().SetLighting(False)
        fill_a.GetProperty().SetAmbient(1.0)
        fill_a.GetProperty().SetDiffuse(0.0)
        self._b2b_add_gap_actor(ren, fill_a)

        # Wireframe outline (exact cell edges)
        wire_m = vtk.vtkPolyDataMapper()
        wire_m.SetInputConnection(cube.GetOutputPort())
        wire_a = vtk.vtkActor()
        wire_a.SetMapper(wire_m)
        wire_a.GetProperty().SetRepresentationToWireframe()
        wire_a.GetProperty().SetColor(1.0, 1.0, 1.0)
        wire_a.GetProperty().SetLineWidth(2.0)
        wire_a.GetProperty().SetOpacity(1.0)
        wire_a.GetProperty().SetLighting(False)
        wire_a.GetProperty().SetAmbient(1.0)
        self._b2b_add_gap_actor(ren, wire_a)

        # Sample-center bead (true voxel coordinate used by DLL)
        bead_r = max(0.35, min_sp * 0.35)
        sph = vtk.vtkSphereSource()
        sph.SetCenter(cx, cy, cz)
        sph.SetRadius(bead_r)
        sph.SetPhiResolution(14)
        sph.SetThetaResolution(14)
        sm = vtk.vtkPolyDataMapper()
        sm.SetInputConnection(sph.GetOutputPort())
        sa = vtk.vtkActor()
        sa.SetMapper(sm)
        sa.GetProperty().SetColor(*color)
        sa.GetProperty().SetOpacity(1.0)
        sa.GetProperty().SetLighting(False)
        sa.GetProperty().SetAmbient(1.0)
        self._b2b_add_gap_actor(ren, sa)

        if label_tag:
            try:
                tag = vtk.vtkBillboardTextActor3D()
                tag.SetInput(str(label_tag))
                tag.SetPosition(cx, cy, cz + max(sz, min_sp) * 1.2)
                tp = tag.GetTextProperty()
                tp.SetFontSize(11)
                tp.SetColor(*color)
                tp.BoldOn()
                tp.SetBackgroundColor(0.0, 0.0, 0.0)
                tp.SetBackgroundOpacity(0.7)
                tp.SetJustificationToCentered()
                self._b2b_add_gap_actor(ren, tag)
            except Exception:
                pass

    def _b2b_add_local_object_surface(self, ren, z, y, x, sx, sy, sz, color,
                                     pad=10):
        """Show local isosurface of the object that owns this surface voxel.

        Makes it obvious the marker sits on the **bump surface**, not mid-air.
        Uses labeled_class1_data (or binary class1) around the voxel.
        """
        labeled = getattr(self, "labeled_class1_data", None)
        c1 = getattr(self, "class1_data", None)
        if labeled is None and c1 is None:
            return
        try:
            if labeled is not None:
                Z, Y, X = labeled.shape
            else:
                Z, Y, X = c1.shape
            z = int(z)
            y = int(y)
            x = int(x)
            if not (0 <= z < Z and 0 <= y < Y and 0 <= x < X):
                return

            lab = 0
            if labeled is not None:
                lab = int(labeled[z, y, x])
            # Crop around contact voxel
            z0, z1 = max(0, z - pad), min(Z, z + pad + 1)
            y0, y1 = max(0, y - pad), min(Y, y + pad + 1)
            x0, x1 = max(0, x - pad), min(X, x + pad + 1)

            if labeled is not None and lab > 0:
                crop = labeled[z0:z1, y0:y1, x0:x1]
                mask = (crop == lab)
            else:
                # Binary bump fallback
                src = c1 if c1 is not None else labeled
                crop = src[z0:z1, y0:y1, x0:x1]
                mask = crop > 0
            if not np.any(mask):
                return

            mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))
            vtk_img = vtk.vtkImageData()
            dz, dy, dx = mask_u8.shape
            vtk_img.SetDimensions(dx, dy, dz)
            vtk_img.SetSpacing(float(sx), float(sy), float(sz))
            # Origin at crop corner (same convention as MES highlight)
            vtk_img.SetOrigin(float(x0) * sx, float(y0) * sy, float(z0) * sz)
            flat = np.ascontiguousarray(
                np.transpose(mask_u8, (2, 1, 0)).ravel(order="F")
            )
            vtk_arr = numpy_support.numpy_to_vtk(
                flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
            )
            vtk_img.GetPointData().SetScalars(vtk_arr)

            try:
                contour = vtk.vtkFlyingEdges3D()
            except Exception:
                contour = vtk.vtkMarchingCubes()
            contour.SetInputData(vtk_img)
            contour.SetValue(0, 0.5)
            contour.ComputeNormalsOn()
            try:
                contour.ComputeScalarsOff()
            except Exception:
                pass
            contour.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(contour.GetOutputPort())
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(*color)
            prop.SetOpacity(0.38)
            prop.SetAmbient(0.45)
            prop.SetDiffuse(0.55)
            prop.EdgeVisibilityOff()
            self._b2b_add_gap_actor(ren, actor)
        except Exception as e:
            print(f"[B2B] local surface patch skipped: {e}")

    def _world_to_normalized_viewport(self, ren, wx, wy, wz):
        """Project world point → normalized viewport [0..1]² (origin bottom-left).

        Returns (nx, ny) or None if projection fails / off-renderer.
        """
        if ren is None:
            return None
        try:
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToWorld()
            coord.SetValue(float(wx), float(wy), float(wz))
            # Display pixels relative to the full render window
            dx, dy = coord.GetComputedDisplayValue(ren)
            # Convert display → viewport-normalized for this renderer
            origin = ren.GetOrigin()  # (ox, oy) bottom-left of viewport in display px
            size = ren.GetSize()      # (w, h)
            w = float(size[0]) if size and size[0] else 0.0
            h = float(size[1]) if size and size[1] else 0.0
            if w <= 1.0 or h <= 1.0:
                return None
            nx = (float(dx) - float(origin[0])) / w
            ny = (float(dy) - float(origin[1])) / h
            return (nx, ny)
        except Exception:
            return None

    def _b2b_caption_viewport_pos(self, ren, mid_x, mid_y, mid_z,
                                  box_w=0.30, box_h=0.08):
        """Place callout box near the gap in screen space (short leader).

        Prefer a small offset from the projected midpoint so the leader is short
        even on multi-layer stacks (avoids fixed top-of-viewport placement).
        """
        proj = self._world_to_normalized_viewport(ren, mid_x, mid_y, mid_z)
        if proj is None:
            # Safe default: mid-right of viewport
            return (0.55, 0.50)

        nx, ny = proj
        # If point is behind camera / wildly off-screen, park mid-right
        if nx < -0.15 or nx > 1.15 or ny < -0.15 or ny > 1.15:
            return (0.55, 0.50)

        # Offset box so it doesn't cover the gap itself (right + slightly up)
        # Position is bottom-left of caption box.
        cap_x = nx + 0.06
        cap_y = ny + 0.04

        # If that would go off the right edge, flip to the left of the point
        if cap_x + box_w > 0.98:
            cap_x = nx - box_w - 0.04
        # If off the top, place below
        if cap_y + box_h > 0.96:
            cap_y = ny - box_h - 0.04
        # Clamp into viewport with a small margin
        cap_x = max(0.02, min(cap_x, 0.98 - box_w))
        cap_y = max(0.02, min(cap_y, 0.96 - box_h))
        return (cap_x, cap_y)

    def _draw_b2b_gap_line(self, src_voxel, dst_voxel, label_text=""):
        """Draw Src→Dst gap on 3D Volume — exact surface voxels + callout.

        Visual design:
          · Local isosurface patches of the two objects at contact (context)
          · **1-voxel wireframe cubes** at DLL boundary indices (true surface voxels)
          · Thin gradient tube cyan→yellow + red arrow (gap direction)
          · Screen-space caption near gap with distance + (Z,Y,X)

        Args:
            src_voxel / dst_voxel: (z, y, x) in **global volume** voxel indices
        """
        ren = getattr(self, "view_3d_renderer", None)
        widget = getattr(self, "view_3d_widget", None)
        if ren is None or self.volume_data is None:
            print("[B2B] 3D gap skipped: no renderer/volume")
            return

        # Auto-enable 3D if user is reviewing B2B while render is OFF
        if not self.ensure_3d_volume_render_for_overlay(reason="b2b_gap"):
            print("[B2B] 3D gap skipped: could not enable 3D volume")
            return

        self._clear_b2b_gap_actors()

        sx, sy, sz = self._get_3d_world_spacing()
        # VTK volume world = index * spacing (origin 0)
        src_x = float(src_voxel[2]) * sx
        src_y = float(src_voxel[1]) * sy
        src_z = float(src_voxel[0]) * sz
        dst_x = float(dst_voxel[2]) * sx
        dst_y = float(dst_voxel[1]) * sy
        dst_z = float(dst_voxel[0]) * sz
        mid_x = 0.5 * (src_x + dst_x)
        mid_y = 0.5 * (src_y + dst_y)
        mid_z = 0.5 * (src_z + dst_z)

        dx = dst_x - src_x
        dy = dst_y - src_y
        dz = dst_z - src_z
        gap_len = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if gap_len < 1e-9:
            gap_len = 1e-9
            dx, dy, dz = gap_len, 0.0, 0.0
        ux, uy, uz = dx / gap_len, dy / gap_len, dz / gap_len

        min_sp = max(1e-6, min(sx, sy, sz))
        # Thin gap tube + modest arrow (markers are exact 1-voxel cubes)
        tube_r = max(0.35, min_sp * 0.9)
        arrow_h = max(min_sp * 2.5, tube_r * 4.0)
        arrow_r = max(tube_r * 1.8, min_sp * 1.4)

        src_z_i, src_y_i, src_x_i = (
            int(src_voxel[0]), int(src_voxel[1]), int(src_voxel[2])
        )
        dst_z_i, dst_y_i, dst_x_i = (
            int(dst_voxel[0]), int(dst_voxel[1]), int(dst_voxel[2])
        )
        col_src = (0.0, 0.92, 1.0)   # cyan
        col_dst = (1.0, 0.88, 0.15)  # yellow

        # ── 0) Local object surfaces at SRC/DST (context: on bump surface) ─
        self._b2b_add_local_object_surface(
            ren, src_z_i, src_y_i, src_x_i, sx, sy, sz, col_src, pad=12
        )
        self._b2b_add_local_object_surface(
            ren, dst_z_i, dst_y_i, dst_x_i, sx, sy, sz, col_dst, pad=12
        )

        # ── 1) Directional gradient tube (cyan SRC → yellow DST) ──────────
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(src_x, src_y, src_z)
        pts.InsertNextPoint(dst_x, dst_y, dst_z)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        colors.InsertNextTuple3(0, 230, 255)    # SRC cyan
        colors.InsertNextTuple3(255, 220, 40)   # DST yellow
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetLines(lines)
        poly.GetPointData().SetScalars(colors)

        tube = vtk.vtkTubeFilter()
        tube.SetInputData(poly)
        tube.SetRadius(tube_r)
        tube.SetNumberOfSides(16)
        tube.CappingOn()
        tube.SetVaryRadiusToVaryRadiusOff()
        tube.Update()

        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputConnection(tube.GetOutputPort())
        line_mapper.SetScalarModeToUsePointData()
        line_mapper.ScalarVisibilityOn()
        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetOpacity(1.0)
        line_actor.GetProperty().SetLighting(False)
        line_actor.GetProperty().SetAmbient(1.0)
        line_actor.GetProperty().SetDiffuse(0.0)
        self._b2b_add_gap_actor(ren, line_actor)

        # ── 2) Exact surface-voxel cells (1×1×1 wireframe cubes) ──────────
        # These ARE the DLL boundary voxels on each object surface.
        self._b2b_add_surface_voxel_marker(
            ren, src_z_i, src_y_i, src_x_i, sx, sy, sz, col_src,
            label_tag=f"SRC Z{src_z_i},Y{src_y_i},X{src_x_i}",
        )
        self._b2b_add_surface_voxel_marker(
            ren, dst_z_i, dst_y_i, dst_x_i, sx, sy, sz, col_dst,
            label_tag=f"DST Z{dst_z_i},Y{dst_y_i},X{dst_x_i}",
        )

        # ── 3) Arrow head at DST (direction Src→Dst) ─────────────────────
        try:
            cone = vtk.vtkConeSource()
            cone.SetRadius(arrow_r)
            cone.SetHeight(arrow_h)
            cone.SetResolution(20)
            cone.SetDirection(ux, uy, uz)
            # Tip near DST sample; keep cone mostly on the gap segment
            back = min(arrow_h * 0.55, gap_len * 0.35)
            cone.SetCenter(
                dst_x - ux * back,
                dst_y - uy * back,
                dst_z - uz * back,
            )
            cone.Update()
            cone_mapper = vtk.vtkPolyDataMapper()
            cone_mapper.SetInputConnection(cone.GetOutputPort())
            cone_actor = vtk.vtkActor()
            cone_actor.SetMapper(cone_mapper)
            cone_actor.GetProperty().SetColor(1.0, 0.25, 0.12)
            cone_actor.GetProperty().SetOpacity(1.0)
            cone_actor.GetProperty().SetLighting(False)
            cone_actor.GetProperty().SetAmbient(1.0)
            cone_actor.GetProperty().SetDiffuse(0.0)
            self._b2b_add_gap_actor(ren, cone_actor)
        except Exception as _arr_e:
            print(f"[B2B] DST arrow skipped: {_arr_e}")

        # ── 5) Screen-space callout NEAR the gap (short leader, not top-of-view) ─
        if label_text:
            caption_ok = False
            try:
                # Two-line caption (distance + surface voxel coords) needs more height
                box_w, box_h = 0.38, 0.11
                # Project gap midpoint → place box a few % away (short leader line)
                cap_x, cap_y = self._b2b_caption_viewport_pos(
                    ren, mid_x, mid_y, mid_z, box_w=box_w, box_h=box_h
                )

                caption = vtk.vtkCaptionActor2D()
                caption.SetCaption(str(label_text))
                caption.SetAttachmentPoint(mid_x, mid_y, mid_z)
                caption.BorderOn()
                caption.LeaderOn()
                # 2D leader tracks attachment while orbiting; stays short because
                # box is next to the projected midpoint (not fixed top-left).
                try:
                    caption.ThreeDimensionalLeaderOff()
                except Exception:
                    pass
                try:
                    caption.SetPadding(3)
                except Exception:
                    pass

                caption.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                caption.SetPosition(float(cap_x), float(cap_y))
                caption.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
                caption.SetWidth(box_w)
                caption.SetHeight(box_h)

                cprop = caption.GetCaptionTextProperty()
                cprop.SetFontSize(13)
                cprop.SetBold(1)
                cprop.SetColor(1.0, 1.0, 1.0)
                cprop.SetBackgroundColor(0.05, 0.08, 0.12)
                cprop.SetBackgroundOpacity(0.88)
                cprop.ShadowOn()
                cprop.SetJustificationToLeft()
                cprop.SetVerticalJustificationToCentered()

                caption.GetProperty().SetColor(1.0, 0.45, 0.25)
                caption.GetProperty().SetLineWidth(2.0)
                try:
                    caption.GetAttachmentPointCoordinate().SetCoordinateSystemToWorld()
                except Exception:
                    pass

                self._b2b_add_gap_actor(ren, caption, is_2d=True)
                caption_ok = True
                print(
                    f"[B2B] caption viewport pos=({cap_x:.3f},{cap_y:.3f}) "
                    f"attach=({mid_x:.1f},{mid_y:.1f},{mid_z:.1f})"
                )
            except Exception as _cap_e:
                print(f"[B2B] CaptionActor2D failed ({_cap_e}); fallback billboard")

            if not caption_ok:
                # Fallback: short 3D leader offset ~few voxels beside the gap (not AABB corner)
                try:
                    # Offset perpendicular-ish to gap, short distance
                    off = max(sphere_r * 6.0, min_sp * 12.0)
                    # Prefer camera right direction if available; else +X world
                    ox, oy, oz = off, 0.0, off * 0.35
                    try:
                        cam = ren.GetActiveCamera()
                        if cam is not None:
                            # Camera view-up × view-plane-normal ≈ camera right
                            vpn = list(cam.GetViewPlaneNormal())
                            vup = list(cam.GetViewUp())
                            rx = vup[1] * vpn[2] - vup[2] * vpn[1]
                            ry = vup[2] * vpn[0] - vup[0] * vpn[2]
                            rz = vup[0] * vpn[1] - vup[1] * vpn[0]
                            rl = float(np.sqrt(rx * rx + ry * ry + rz * rz)) or 1.0
                            ox, oy, oz = off * rx / rl, off * ry / rl, off * rz / rl
                    except Exception:
                        pass
                    label_x = mid_x + ox
                    label_y = mid_y + oy
                    label_z = mid_z + oz

                    lead_pts = vtk.vtkPoints()
                    lead_pts.InsertNextPoint(mid_x, mid_y, mid_z)
                    lead_pts.InsertNextPoint(label_x, label_y, label_z)
                    lead_lines = vtk.vtkCellArray()
                    lead_lines.InsertNextCell(2)
                    lead_lines.InsertCellPoint(0)
                    lead_lines.InsertCellPoint(1)
                    lead_pd = vtk.vtkPolyData()
                    lead_pd.SetPoints(lead_pts)
                    lead_pd.SetLines(lead_lines)
                    lead_tube = vtk.vtkTubeFilter()
                    lead_tube.SetInputData(lead_pd)
                    lead_tube.SetRadius(max(0.3, tube_r * 0.4))
                    lead_tube.SetNumberOfSides(8)
                    lead_tube.Update()
                    lead_map = vtk.vtkPolyDataMapper()
                    lead_map.SetInputConnection(lead_tube.GetOutputPort())
                    lead_act = vtk.vtkActor()
                    lead_act.SetMapper(lead_map)
                    lead_act.GetProperty().SetColor(1.0, 0.5, 0.25)
                    lead_act.GetProperty().SetOpacity(0.85)
                    lead_act.GetProperty().SetLighting(False)
                    self._b2b_add_gap_actor(ren, lead_act)

                    text_actor = vtk.vtkBillboardTextActor3D()
                    text_actor.SetInput(str(label_text))
                    text_actor.SetPosition(label_x, label_y, label_z)
                    tp = text_actor.GetTextProperty()
                    tp.SetFontSize(14)
                    tp.SetColor(1.0, 1.0, 1.0)
                    tp.BoldOn()
                    tp.SetBackgroundColor(0.05, 0.08, 0.12)
                    tp.SetBackgroundOpacity(0.85)
                    self._b2b_add_gap_actor(ren, text_actor)
                except Exception as _fb_e:
                    print(f"[B2B] label fallback failed: {_fb_e}")

        if widget is not None:
            try:
                ren.ResetCameraClippingRange()
                widget.GetRenderWindow().Render()
            except Exception:
                pass
        print(
            f"[B2B] 3D gap drawn: {label_text}  "
            f"src={src_voxel} dst={dst_voxel} spacing=({sx:.3f},{sy:.3f},{sz:.3f}) "
            f"len={gap_len:.3f}"
        )

    # ── MES object highlight on 3D Volume (bbox-cropped surface) ─────────

    def _clear_mes_3d_highlight(self, render=True):
        """Remove MES selection surfaces from the 3D renderer."""
        ren = getattr(self, "view_3d_renderer", None)
        actors = getattr(self, "_mes_3d_highlight_actors", None) or []
        if ren is not None:
            for actor in actors:
                try:
                    ren.RemoveActor(actor)
                except Exception:
                    pass
        self._mes_3d_highlight_actors = []
        self._mes_3d_hl_cache_key = None
        if render:
            widget = getattr(self, "view_3d_widget", None)
            if widget is not None and ren is not None:
                try:
                    widget.GetRenderWindow().Render()
                except Exception:
                    pass

    def _mes_3d_bbox_for_labels(self, label_set):
        """Union bbox of selected labels from object_stats (fast — no full-volume scan).

        Returns (z0,z1,y0,y1,x0,x1) inclusive in labeled-volume indices, or None.
        """
        if not label_set or not getattr(self, "object_stats", None):
            return None
        _z_off = int(getattr(self, "_measurement_start_slice", 0) or 0)
        z0 = y0 = x0 = 10 ** 9
        z1 = y1 = x1 = -1
        hit = False
        for st in self.object_stats:
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

    def _update_mes_3d_highlight(self):
        """Show selected MES bump(s) on 3D volume as a cyan surface.

        Optimized path:
          · Only the **bbox crop** of labeled_class1 is contoured (not full volume)
          · Rebuild skipped when label set + bbox unchanged
          · vtkFlyingEdges3D (fallback MarchingCubes) on uint8 mask
        """
        ren = getattr(self, "view_3d_renderer", None)
        widget = getattr(self, "view_3d_widget", None)
        labeled = getattr(self, "labeled_class1_data", None)
        if ren is None or labeled is None or self.volume_data is None:
            self._clear_mes_3d_highlight(render=False)
            return

        label_set = set()
        for _cls, lab in (self.selected_highlight_objects or []):
            try:
                lab = int(lab)
            except (TypeError, ValueError):
                continue
            if lab > 0:
                label_set.add(lab)

        if not label_set:
            self._clear_mes_3d_highlight(render=True)
            return

        # Need 3D volume present for surface placement (auto-on if user picks MES)
        if not self.is_3d_volume_render_enabled() or getattr(self, "volume_actor", None) is None:
            if not self.ensure_3d_volume_render_for_overlay(reason="mes_pick"):
                return

        bbox = self._mes_3d_bbox_for_labels(label_set)
        if bbox is None:
            # No stats bbox (edge case) — skip heavy full-volume path
            self._clear_mes_3d_highlight(render=True)
            return

        z0, z1, y0, y1, x0, x1 = bbox
        Z, Y, X = labeled.shape
        pad = 2
        z0 = max(0, z0 - pad)
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        z1 = min(Z - 1, z1 + pad)
        y1 = min(Y - 1, y1 + pad)
        x1 = min(X - 1, x1 + pad)
        if z1 < z0 or y1 < y0 or x1 < x0:
            self._clear_mes_3d_highlight(render=True)
            return

        cache_key = (frozenset(label_set), z0, z1, y0, y1, x0, x1)
        if (
            cache_key == getattr(self, "_mes_3d_hl_cache_key", None)
            and getattr(self, "_mes_3d_highlight_actors", None)
        ):
            # Already showing the same selection — just re-render
            if widget is not None:
                try:
                    widget.GetRenderWindow().Render()
                except Exception:
                    pass
            return

        crop = labeled[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
        if len(label_set) == 1:
            lab0 = next(iter(label_set))
            mask = (crop == lab0)
        else:
            mask = np.isin(crop, list(label_set))
        if not np.any(mask):
            self._clear_mes_3d_highlight(render=True)
            return

        # Binary uint8 for isosurface (0/1) — small crop only
        mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))

        # Clear previous actors (no intermediate render)
        self._clear_mes_3d_highlight(render=False)

        sx, sy, sz = self._get_3d_world_spacing()
        # VTK image: dims (x,y,z), Fortran order scalars, origin at crop corner
        vtk_img = vtk.vtkImageData()
        dz, dy, dx = mask_u8.shape
        vtk_img.SetDimensions(dx, dy, dz)
        vtk_img.SetSpacing(float(sx), float(sy), float(sz))
        vtk_img.SetOrigin(float(x0) * sx, float(y0) * sy, float(z0) * sz)
        # (z,y,x) → VTK (x,y,z) Fortran
        flat = np.ascontiguousarray(np.transpose(mask_u8, (2, 1, 0)).ravel(order="F"))
        vtk_arr = numpy_support.numpy_to_vtk(
            flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
        )
        vtk_arr.SetNumberOfComponents(1)
        vtk_img.GetPointData().SetScalars(vtk_arr)

        # Isosurface at 0.5
        try:
            contour = vtk.vtkFlyingEdges3D()
        except Exception:
            contour = vtk.vtkMarchingCubes()
        contour.SetInputData(vtk_img)
        contour.SetValue(0, 0.5)
        contour.ComputeNormalsOn()
        try:
            contour.ComputeScalarsOff()
        except Exception:
            pass

        # Light smooth for nicer edges (cheap on small polydata)
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.SetNumberOfIterations(8)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetPassBand(0.1)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        mapper.ScalarVisibilityOff()
        mapper.SetResolveCoincidentTopologyToPolygonOffset()

        hl = self.highlight_color or [0.0, 1.0, 1.0]
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(float(hl[0]), float(hl[1]), float(hl[2]))
        prop.SetOpacity(0.42)
        prop.SetSpecular(0.35)
        prop.SetSpecularPower(20.0)
        prop.SetAmbient(0.35)
        prop.SetDiffuse(0.75)
        prop.EdgeVisibilityOn()
        prop.SetEdgeColor(1.0, 1.0, 1.0)
        prop.SetLineWidth(1.0)

        ren.AddActor(actor)
        self._mes_3d_highlight_actors = [actor]
        self._mes_3d_hl_cache_key = cache_key

        # Centroid marker (one sphere) for quick depth cue
        try:
            # Prefer primary target centroid from first matching stat
            cx = cy = cz = None
            for st in self.object_stats or []:
                try:
                    lab = int(st.get("label", 0))
                except (TypeError, ValueError):
                    continue
                if lab not in label_set:
                    continue
                _z_off = int(getattr(self, "_measurement_start_slice", 0) or 0)
                cz = (float(st.get("centroid_z", 0)) + _z_off) * sz
                cy = float(st.get("centroid_y", 0)) * sy
                cx = float(st.get("centroid_x", 0)) * sx
                break
            if cx is not None:
                min_sp = max(1e-6, min(sx, sy, sz))
                sph = vtk.vtkSphereSource()
                sph.SetCenter(cx, cy, cz)
                sph.SetRadius(max(1.0, min_sp * 3.0))
                sph.SetPhiResolution(12)
                sph.SetThetaResolution(12)
                sm = vtk.vtkPolyDataMapper()
                sm.SetInputConnection(sph.GetOutputPort())
                sa = vtk.vtkActor()
                sa.SetMapper(sm)
                sa.GetProperty().SetColor(1.0, 1.0, 0.2)
                sa.GetProperty().SetOpacity(0.95)
                sa.GetProperty().SetLighting(False)
                ren.AddActor(sa)
                self._mes_3d_highlight_actors.append(sa)
        except Exception:
            pass

        try:
            ren.ResetCameraClippingRange()
        except Exception:
            pass
        if widget is not None:
            try:
                widget.GetRenderWindow().Render()
            except Exception:
                pass

    def _on_b2b_row_clicked(self, row, col):
        """Click B2B row → draw/toggle 3D surface-voxel gap on Volume (not MPR).

        UX:
          · 1st click on a row → show gap overlay
          · 2nd click on the **same** row → hide overlay (toggle off)
          · click a different row → switch to that gap
        """
        idx = self._b2b_stats_index_from_visual_row(row)
        if idx is None:
            print(f"[B2B] click row={row}: no stats index")
            return

        # Toggle OFF if the same gap is already shown
        active = getattr(self, "_b2b_active_stats_idx", None)
        if active is not None and int(active) == int(idx):
            self._clear_b2b_gap_actors()
            if hasattr(self, "b2b_info_label"):
                n = len(getattr(self, "b2b_stats", None) or [])
                self.b2b_info_label.setText(
                    f"{n} rows | click row → show 3D gap · click again → hide"
                )
            try:
                self.b2b_table.clearSelection()
            except Exception:
                pass
            print(f"[B2B] toggle OFF stats_idx={idx}")
            return

        r = self.b2b_stats[idx]
        z_off = self._b2b_layer_z_offset(r)

        def _iv(key, default=0):
            try:
                return int(float(r.get(key, default)))
            except (TypeError, ValueError):
                return default

        has_src_dst = (
            r.get("Src_voxel_Z") not in (None, "")
            and r.get("Dst_voxel_Z") not in (None, "")
        )
        if has_src_dst:
            src = (_iv("Src_voxel_Z") + z_off, _iv("Src_voxel_Y"), _iv("Src_voxel_X"))
            dst = (_iv("Dst_voxel_Z") + z_off, _iv("Dst_voxel_Y"), _iv("Dst_voxel_X"))
            eucl = r.get("Min_dist_Euclidean_um", r.get("min_gap_euclidean_um", "?"))
            direction = r.get("Direction", "")
            src_rc = f"R{r.get('Src_row', '?')}C{r.get('Src_col', '?')}"
            dst_rc = f"R{r.get('Dst_row', '?')}C{r.get('Dst_col', '?')}"
            layer = str(r.get("Layer", "") or "")
            # Caption: grid + exact surface voxels (DLL boundary indices)
            label = (
                f"{layer}  SRC {src_rc}→DST {dst_rc} ({direction})  {eucl} µm\n"
                f"surface voxels  "
                f"SRC(Z,Y,X)=({src[0]},{src[1]},{src[2]})  "
                f"DST(Z,Y,X)=({dst[0]},{dst[1]},{dst[2]})"
            ).strip()
            try:
                self._draw_b2b_gap_line(src, dst, label_text=label)
                self._b2b_active_stats_idx = int(idx)
                if hasattr(self, "b2b_info_label"):
                    self.b2b_info_label.setText(
                        f"3D gap ON · {layer} {src_rc}→{dst_rc} ({direction}) "
                        f"{eucl}µm · wireframe cube = surface voxel · "
                        f"click same row to hide"
                    )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[B2B] draw failed: {e}")
            return

        # Legacy summary: marker at min-gap voxel
        vz = _iv("min_gap_voxel_Z", _iv("vox_Z")) + z_off
        vy = _iv("min_gap_voxel_Y", _iv("vox_Y"))
        vx = _iv("min_gap_voxel_X", _iv("vox_X"))
        if not any(k in r for k in ("min_gap_voxel_Z", "vox_Z", "Src_voxel_Z")):
            print("[B2B] row has no voxel coordinates")
            return
        eucl = r.get("min_gap_euclidean_um", r.get("Min_dist_Euclidean_um", "?"))
        bump = r.get("Bump_id", r.get("bump_id", ""))
        label = (
            f"{r.get('Layer', '')}  min-gap @ {bump}  {eucl} µm\n"
            f"surface voxel (Z,Y,X)=({vz},{vy},{vx})"
        ).strip()
        try:
            self._draw_b2b_gap_line((vz, vy, vx), (vz, vy, vx + 1), label_text=label)
            self._b2b_active_stats_idx = int(idx)
            if hasattr(self, "b2b_info_label"):
                self.b2b_info_label.setText(
                    f"3D gap ON · min-gap {eucl}µm @ ({vz},{vy},{vx}) · "
                    f"click same row to hide"
                )
        except Exception as e:
            print(f"[B2B] marker draw failed: {e}")

    def apply_far(self, voxel_volume):
        """Automatically run FAR after measurement in online mode."""
        def parse_size(text):
            try:
                parts = [float(p.strip()) for p in str(text).lower().split('x') if p.strip()]
                if not parts: return 0.0
                res = 1.0
                for p in parts: res *= p
                return res
            except:
                return 0.0

        main_window = self.window()
        tgv_min, tgv_max = 0.0, 1000000000.0
        void_min, void_max = 0.0, 1000000000.0
        
        if hasattr(main_window, 'segmentation_tab') and hasattr(main_window.segmentation_tab, 'param_widgets'):
            sw = main_window.segmentation_tab.param_widgets
            if 'bump_minimum_size' in sw:
                tgv_min = parse_size(sw['bump_minimum_size'].text())
                tgv_max = parse_size(sw['bump_maximum_size'].text())
                void_min = parse_size(sw['void_minimum_size'].text())
                void_max = parse_size(sw['void_maximum_size'].text())

        if not self.object_stats:
            return

        tgv_labels_to_remove = []
        void_labels_to_remove = []
        rows_to_remove = []

        for row_idx, stat in enumerate(self.object_stats):
            c1_voxels = stat['c1_volume'] / voxel_volume
            c2_voxels = stat['c2_volume'] / voxel_volume
            
            remove_bump = False
            remove_void = False
            
            if c1_voxels < tgv_min or c1_voxels > tgv_max:
                remove_bump = True
            if c2_voxels > 0 and (c2_voxels < void_min or c2_voxels > void_max):
                remove_void = True
                
            if remove_bump:
                tgv_labels_to_remove.append(stat['label'])
                rows_to_remove.append(row_idx)
            elif remove_void:
                void_labels_to_remove.append(stat['label'])
                stat['c2_volume'] = 0.0
                stat['ratio'] = 0.0

        for row_idx in sorted(rows_to_remove, reverse=True):
            self.object_stats.pop(row_idx)

        needs_rerender = False
        if self.labeled_class1_data is not None:
            if tgv_labels_to_remove:
                 mask_bump = np.isin(self.labeled_class1_data, tgv_labels_to_remove)
                 self.labeled_class1_data[mask_bump] = 0
                 if self.class1_data is not None: self.class1_data[mask_bump] = 0
                 if self.class2_data is not None: self.class2_data[mask_bump] = 0
                 if self.segmentation_data is not None: self.segmentation_data[mask_bump] = 0
                 needs_rerender = True
            if void_labels_to_remove:
                 mask_void = np.isin(self.labeled_class1_data, void_labels_to_remove)
                 if self.class2_data is not None: self.class2_data[mask_void] = 0
                 if self.segmentation_data is not None:
                     mask_for_seg = np.logical_and(mask_void, self.segmentation_data == 255)
                     self.segmentation_data[mask_for_seg] = 128
                 needs_rerender = True

        if needs_rerender:
            for ori in ['axial', 'coronal', 'sagittal']:
                self.render_slice(ori, preserve_camera=True)
            self.render_3d()
    
    def export_stats_csv(self, output_path=None):
        """Export active tab statistics to CSV (MES object stats or B2B boundary)."""
        # In PyQt, clicked signal passes a boolean. If it's not a string, it's manual export
        is_auto_export = isinstance(output_path, str) and bool(output_path)

        # Active tab: 0=MES, 1=B2B
        active_b2b = (
            hasattr(self, "stats_tabs")
            and self.stats_tabs.currentIndex() == 1
            and getattr(self, "b2b_stats", None)
        )
        if active_b2b and not is_auto_export:
            if not is_auto_export:
                output_path, _ = QFileDialog.getSaveFileName(
                    self, "Save B2B Boundary CSV", "boundary_summary_all_layers.csv",
                    "CSV Files (*.csv);;All Files (*.*)"
                )
            if not output_path:
                return
            try:
                from inno3d.core import bumpvoid_b2b
                bumpvoid_b2b.write_combined_summary(output_path, self.b2b_stats)
                if not is_auto_export:
                    self.set_stats_info(f"B2B CSV exported: {output_path}")
                    QMessageBox.information(self, "Export Successful", f"B2B CSV exported to:\n{output_path}")
                print(f"[STATS] B2B CSV exported: {output_path}")
                return output_path
            except Exception as e:
                if not is_auto_export:
                    QMessageBox.critical(self, "Export Error", str(e))
                else:
                    print(f"[EXPORT ERROR] B2B: {e}")
                return None

        if not self.object_stats:
            return
        
        if not is_auto_export:
            output_path, _ = QFileDialog.getSaveFileName(
                self, "Save Statistics CSV", "object_statistics.csv",
                "CSV Files (*.csv);;All Files (*.*)"
            )
        
        if not output_path:
            return
        
        try:
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Get NG threshold from main window
                ng_threshold = 0.05
                main_window = self.window()
                if hasattr(main_window, 'segmentation_tab') and hasattr(main_window.segmentation_tab, 'ng_threshold_spin'):
                    ng_threshold = main_window.segmentation_tab.ng_threshold_spin.value() / 100.0
                elif hasattr(main_window, 'seg_tab') and hasattr(main_window.seg_tab, 'ng_threshold_spin'):
                    ng_threshold = main_window.seg_tab.ng_threshold_spin.value() / 100.0
                    
                writer.writerow([
                    '#', 'Layer', 'Bump ID', 'B. H', 'B. V', 'V. V', 'Ratio (C2/(C1+C2))',
                    'Judgment',
                    'Gap X (um)', 'Gap Y (um)',
                    'Z_min', 'Z_max', 'Y_min', 'Y_max', 'X_min', 'X_max',
                    'Centroid_Z', 'Centroid_Y', 'Centroid_X'
                ])
                for stat in self.object_stats:
                    ratio_str = f"{stat['ratio']*100:.3f}%" if stat['ratio'] != float('inf') else "N/A"
                    is_ng = stat['ratio'] >= ng_threshold if stat['ratio'] != float('inf') else True
                    judgment_str = "NG" if is_ng else "OK"
                    
                    writer.writerow([
                        stat.get('row_id', ''),
                        str(stat.get('layer_name', '') or ''),
                        f"({stat.get('grid_row', '?')},{stat.get('grid_col', '?')})",
                        f"{stat.get('soh', 0):.3f}",
                        f"{stat['c1_volume']:.3f}", f"{stat['c2_volume']:.3f}", ratio_str,
                        judgment_str,
                        f"{stat.get('pitch_x', 0):.3f}", f"{stat.get('pitch_y', 0):.3f}",
                        stat['z_min'], stat['z_max'],
                        stat['y_min'], stat['y_max'],
                        stat['x_min'], stat['x_max'],
                        f"{stat['centroid_z']:.3f}", f"{stat['centroid_y']:.3f}", f"{stat['centroid_x']:.3f}"
                    ])

            if not is_auto_export:
                self.set_stats_info(f"CSV exported: {output_path}")
                QMessageBox.information(self, "Export Successful", f"Statistics exported to:\n{output_path}")
            
            print(f"[STATS] CSV exported: {output_path}")
            return output_path
            
        except Exception as e:
            if not is_auto_export:
                QMessageBox.critical(self, "Export Error", f"Failed to export CSV:\n{str(e)}")
            else:
                print(f"[EXPORT ERROR] {e}")
            return None
    
    def _stats_index_from_visual_row(self, visual_row):
        """Map a sorted visual table row → index into ``self.object_stats`` via UserRole."""
        if visual_row is None or visual_row < 0:
            return None
        id_item = self.object_stats_table.item(visual_row, 0)
        if id_item is None:
            return None
        stats_idx = id_item.data(Qt.UserRole)
        if stats_idx is None:
            # Fallback for legacy rows without UserRole
            stats_idx = visual_row
        try:
            stats_idx = int(stats_idx)
        except (TypeError, ValueError):
            return None
        if stats_idx < 0 or stats_idx >= len(self.object_stats or []):
            return None
        return stats_idx

    def _label_at_stat_centroid(self, stat) -> int:
        """Look up labeled_class1 id at MES object centroid (or bbox center)."""
        labeled = getattr(self, "labeled_class1_data", None)
        if labeled is None or stat is None:
            return 0
        import numpy as np

        Z, Y, X = labeled.shape
        _z_off = int(getattr(self, "_measurement_start_slice", 0) or 0)

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
        # Neighborhood mode on centroid Z
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

    def _navigate_to_object_stat(self, stat):
        """Jump MPR sliders + 3D crosshair to object centroid (volume coordinates).

        Online MES uses absolute Z (and sets ``_measurement_start_slice = 0`` after
        restore). Python subset measurement stores local Z and relies on the offset.
        """
        if stat is None or self.volume_data is None:
            return
        _z_offset = int(getattr(self, "_measurement_start_slice", 0) or 0)
        cz = int(round(float(stat.get("centroid_z", 0)))) + _z_offset
        cy = int(round(float(stat.get("centroid_y", 0))))
        cx = int(round(float(stat.get("centroid_x", 0))))
        # updatePoint expects (X, Y, Z) and syncs MPR + 3D crosshair
        self.updatePoint(cx, cy, cz)

    # ── Excel Filter Support ──────────────────────────
    def _apply_excel_filters(self, col_index):
        """Handle visual hiding of rows based on ExcelFilterHeader."""
        header = self.sender()
        if not header:
            return

        table = header.parent()
        if not isinstance(table, QTableWidget):
            return

        for row in range(table.rowCount()):
            hidden = header.is_row_hidden(table, row)
            table.setRowHidden(row, hidden)
        # Keep frozen overlay row visibility / heights in lock-step
        if hasattr(table, "refresh_frozen"):
            table.refresh_frozen()
            
    def on_object_selection_changed(self):
        """Handle MES table selection → highlight + navigate MPR/3D to that object."""
        selected_visual = sorted(
            set(index.row() for index in self.object_stats_table.selectedIndexes())
        )

        self.selected_highlight_objects = []
        target_stat = None

        # Prefer the current item's row for navigation (matches user click)
        current_item = self.object_stats_table.currentItem()
        target_visual = current_item.row() if current_item is not None else -1

        for visual_row in selected_visual:
            stats_idx = self._stats_index_from_visual_row(visual_row)
            if stats_idx is None:
                continue
            stat = self.object_stats[stats_idx]
            # All measured bumps are Class-1 labels
            try:
                label = int(stat.get("label", 0))
            except (TypeError, ValueError):
                label = 0
            # Batch-replay fallback: resolve label from labeled mask at centroid
            if label <= 0 and getattr(self, "labeled_class1_data", None) is not None:
                try:
                    lab = self._label_at_stat_centroid(stat)
                    if lab > 0:
                        label = lab
                        stat["label"] = lab
                except Exception:
                    pass
            if label > 0:
                self.selected_highlight_objects.append((1, label))
            if visual_row == target_visual or target_stat is None:
                target_stat = stat

        if target_stat is not None:
            self._navigate_to_object_stat(target_stat)
        elif self.volume_data is not None:
            # Clear-only path: refresh highlights off current slices + 3D
            for ori in ["axial", "coronal", "sagittal"]:
                self.render_slice(ori, preserve_camera=True)
            self._update_mes_3d_highlight()
            return

        # Re-render MPR highlight overlay + 3D surface for selected labels
        if self.volume_data is not None:
            for ori in ["axial", "coronal", "sagittal"]:
                self.render_slice(ori, preserve_camera=True)
            self._update_mes_3d_highlight()
    
    def clear_object_selection(self):
        """Clear MES selection/highlights and B2B 3D gap overlay."""
        self.object_stats_table.blockSignals(True)
        self.object_stats_table.clearSelection()
        self.object_stats_table.blockSignals(False)
        self.selected_highlight_objects = []
        if getattr(self, "b2b_table", None) is not None:
            self.b2b_table.blockSignals(True)
            self.b2b_table.clearSelection()
            self.b2b_table.blockSignals(False)
        self._clear_b2b_gap_actors()
        self._clear_mes_3d_highlight(render=True)
        if self.volume_data is not None:
            for ori in ['axial', 'coronal', 'sagittal']:
                self.render_slice(ori, preserve_camera=True)
                
    def on_stats_filter_changed(self, text):
        """Filter both MES and B2B tables based on search text."""
        search_term = text.lower()
        
        # Filter Object Stats Table
        for row in range(self.object_stats_table.rowCount()):
            match = False
            for col in range(self.object_stats_table.columnCount()):
                item = self.object_stats_table.item(row, col)
                if item and search_term in item.text().lower():
                    match = True
                    break
            self.object_stats_table.setRowHidden(row, not match if search_term else False)
            
        # Filter B2B Table
        for row in range(self.b2b_table.rowCount()):
            match = False
            for col in range(self.b2b_table.columnCount()):
                item = self.b2b_table.item(row, col)
                if item and search_term in item.text().lower():
                    match = True
                    break
            self.b2b_table.setRowHidden(row, not match if search_term else False)
    
    
    
    def _restore_interactor_style(self, orientation):
        """Restore the original VTK interactor style if it was swapped to dummy.

        This is a safety-net called from BOTH the Qt eventFilter AND the VTK
        observer release path.  It is idempotent — if the style was already
        restored (or was never swapped), this is a no-op.
        """
        if not hasattr(self, 'active_styles'):
            return
        saved = self.active_styles.pop(orientation, None)
        if saved is None:
            return
        widget = getattr(self, f'{orientation}_widget', None)
        if widget is None:
            return
        interactor = widget.GetRenderWindow().GetInteractor()
        # Only restore if we're currently on the dummy style
        if interactor.GetInteractorStyle() is getattr(self, '_dummy_style', None):
            interactor.SetInteractorStyle(saved)

    def setup_vtk_observers(self, vtk_widget, orientation):
        interactor = vtk_widget.GetRenderWindow().GetInteractor()
        
        if not hasattr(self, 'active_styles'):
            self.active_styles = {}
        if not hasattr(self, '_dummy_style'):
            self._dummy_style = vtk.vtkInteractorStyle()
            
        def on_left_button_press(caller, event):
            if self.volume_data is None:
                return

            x, y = caller.GetEventPosition()
            size = vtk_widget.GetRenderWindow().GetSize()
            qt_y = size[1] - y
            qt_pos = QPoint(x, qt_y)

            # 1) Dragonfly clip-box handles (priority over crosshair when CLIP BOX on)
            if getattr(self, '_df_clip_enabled', False):
                clip_hit = self._get_df_clip_hit(qt_pos, orientation)
                if clip_hit is not None:
                    if orientation not in self.active_styles:
                        self.active_styles[orientation] = caller.GetInteractorStyle()
                    caller.SetInteractorStyle(self._dummy_style)
                    self._df_clip_begin_drag(orientation, qt_pos, clip_hit)
                    return

            if not self.crosshair_enabled and not getattr(self, 'rotate_mode', False):
                return

            if not self.crosshair_enabled:
                # rotate_mode only — block default VTK interaction
                if orientation not in self.active_styles:
                    self.active_styles[orientation] = caller.GetInteractorStyle()
                caller.SetInteractorStyle(self._dummy_style)
                self.last_mouse_pos = qt_pos
                return

            # Dragonfly: only grab when near center / axis (or oblique handle)
            if getattr(self, '_oblique_hover_handle', None) and getattr(self, '_oblique_hover_handle')[0] == orientation:
                if orientation not in self.active_styles:
                    self.active_styles[orientation] = caller.GetInteractorStyle()
                caller.SetInteractorStyle(self._dummy_style)
                self._oblique_dragging = self._oblique_hover_handle
                return

            ch_hit = self._get_crosshair_hit(qt_pos, orientation)
            if orientation not in self.active_styles:
                self.active_styles[orientation] = caller.GetInteractorStyle()
            caller.SetInteractorStyle(self._dummy_style)
            if ch_hit is None:
                # Click-to-place on empty space
                self._crosshair_drag_mode = 'place'
                self._crosshair_hover = (orientation, 'center')
                self._is_dragging_crosshair = True
                self.handle_crosshair_click(orientation, qt_pos, mode='place', preview_only=False)
                return

            self._crosshair_drag_mode = ch_hit[1]
            self._crosshair_hover = ch_hit
            self._is_dragging_crosshair = True
            self.handle_crosshair_click(orientation, qt_pos, mode=ch_hit[1], preview_only=False)
            self.update_2d_crosshair(orientation)

        def on_mouse_move(caller, event):
            if self.volume_data is None:
                return

            x, y = caller.GetEventPosition()
            size = vtk_widget.GetRenderWindow().GetSize()
            qt_y = size[1] - y
            qt_pos = QPoint(x, qt_y)

            self.update_pixel_value(orientation, qt_pos)

            # Active clip-box drag
            if getattr(self, '_df_clip_drag', None):
                self._df_clip_handle_drag(qt_pos, orientation)
                return

            # Hover: clip handles first when CLIP BOX is on
            if getattr(self, '_df_clip_enabled', False) and not (getattr(self, '_is_dragging_crosshair', False)
                    or getattr(self, '_oblique_dragging', None)):
                clip_hit = self._get_df_clip_hit(qt_pos, orientation)
                prev = getattr(self, '_df_clip_hover', None)
                if self._df_clip_hit_key(clip_hit) != self._df_clip_hit_key(prev):
                    self._df_clip_hover = clip_hit
                    self._df_clip_update_mpr_overlay(orientation)
                    vtk_widget.GetRenderWindow().Render()
                if clip_hit is not None:
                    vtk_widget.setCursor(self._df_clip_cursor_for_hit(clip_hit))
                    return
                elif prev is not None:
                    self._df_clip_hover = None
                    self._df_clip_update_mpr_overlay(orientation)
                    vtk_widget.GetRenderWindow().Render()

            if not self.crosshair_enabled:
                return

            if getattr(self, '_oblique_dragging', None):
                self._handle_oblique_rotate(qt_pos, orientation)
                return

            if getattr(self, '_is_dragging_crosshair', False):
                mode = getattr(self, '_crosshair_drag_mode', 'center') or 'center'
                self.handle_crosshair_click(
                    orientation, qt_pos, mode=mode, preview_only=True
                )
                return

            # Hover priority: oblique handles first, then crosshair center/lines
            obl_hit = self._get_oblique_hit(qt_pos, orientation)
            if obl_hit != getattr(self, '_oblique_hover_handle', None):
                self._oblique_hover_handle = obl_hit
                if obl_hit:
                    self._crosshair_hover = None
                    self.render_slice(orientation, preserve_camera=True)
                    vtk_widget.setCursor(Qt.SizeAllCursor)
                    return
                self.render_slice(orientation, preserve_camera=True)

            if obl_hit:
                vtk_widget.setCursor(Qt.SizeAllCursor)
                return

            ch_hit = self._get_crosshair_hit(qt_pos, orientation)
            if ch_hit != getattr(self, '_crosshair_hover', None):
                self._crosshair_hover = ch_hit
                self.update_2d_crosshair(orientation)
            cur = self._cursor_for_crosshair_hit(ch_hit)
            if cur is not None:
                vtk_widget.setCursor(cur)
            else:
                vtk_widget.setCursor(Qt.OpenHandCursor)

        def on_left_button_release(caller, event):
            if getattr(self, '_df_clip_drag', None):
                self._df_clip_end_drag()
                x, y = caller.GetEventPosition()
                size = vtk_widget.GetRenderWindow().GetSize()
                qt_y = size[1] - y
                qt_pos = QPoint(x, qt_y)
                hit = self._get_df_clip_hit(qt_pos, orientation) if getattr(self, '_df_clip_enabled', False) else None
                self._df_clip_hover = hit
                if hit is not None:
                    vtk_widget.setCursor(self._df_clip_cursor_for_hit(hit))
                else:
                    vtk_widget.unsetCursor()
                self._restore_interactor_style(orientation)
                return

            if getattr(self, '_is_dragging_crosshair', False):
                self._is_dragging_crosshair = False
                self._crosshair_drag_mode = None
                self._crosshair_drag_end()

            if getattr(self, '_oblique_dragging', None):
                self._oblique_dragging = None

                # Full-quality render of all views on release
                self._oblique_drag_finish()

                x, y = caller.GetEventPosition()
                size = vtk_widget.GetRenderWindow().GetSize()
                qt_y = size[1] - y
                qt_pos = QPoint(x, qt_y)

                hit = self._get_oblique_hit(qt_pos, orientation)
                self._oblique_hover_handle = hit
                if hit:
                    vtk_widget.setCursor(Qt.SizeAllCursor)
                else:
                    vtk_widget.unsetCursor()

            # Restore the active style (centralized helper — idempotent)
            self._restore_interactor_style(orientation)

        interactor.AddObserver("LeftButtonPressEvent", on_left_button_press, 10.0)
        interactor.AddObserver("MouseMoveEvent", on_mouse_move, 10.0)
        interactor.AddObserver("LeftButtonReleaseEvent", on_left_button_release, 10.0)

        # ── Fiji-style scroll wheel zoom at cursor position ──────────────
        # These observers fire at high priority (100.0) so they intercept
        # the wheel event BEFORE the interactor style (including dummy style
        # during crosshair mode) can consume or block it.
        def on_mouse_wheel_forward(caller, event):
            if self.volume_data is None:
                return
            # Qt eventFilter is primary; this is a backup path
            if caller.GetControlKey():
                self._scroll_change_slice(orientation, delta=+1)
            else:
                self._fiji_zoom_at_cursor(
                    vtk_widget, orientation, zoom_in=True, strength=1.0
                )
            caller.SetEventInformation(
                caller.GetEventPosition()[0], caller.GetEventPosition()[1],
                caller.GetControlKey(), caller.GetShiftKey())

        def on_mouse_wheel_backward(caller, event):
            if self.volume_data is None:
                return
            if caller.GetControlKey():
                self._scroll_change_slice(orientation, delta=-1)
            else:
                self._fiji_zoom_at_cursor(
                    vtk_widget, orientation, zoom_in=False, strength=1.0
                )
            caller.SetEventInformation(
                caller.GetEventPosition()[0], caller.GetEventPosition()[1],
                caller.GetControlKey(), caller.GetShiftKey())

        interactor.AddObserver("MouseWheelForwardEvent", on_mouse_wheel_forward, 100.0)
        interactor.AddObserver("MouseWheelBackwardEvent", on_mouse_wheel_backward, 100.0)

        def on_interaction(caller, event):
            if self.volume_data is None:
                return
            renderer = getattr(self, f'{orientation}_renderer')
            z, y, x = self.volume_data.shape
            h, w = (y, x) if orientation == 'axial' else (z, x) if orientation == 'coronal' else (z, y)
            self.add_ruler_overlay(renderer, orientation, (h, w))
            vtk_widget.GetRenderWindow().Render()

        interactor.AddObserver("InteractionEvent", on_interaction, 10.0)
        interactor.AddObserver("EndInteractionEvent", on_interaction, 10.0)

    # ── Fiji-style zoom helpers ──────────────────────────────────────────
    def _fiji_zoom_at_cursor(self, vtk_widget, orientation, zoom_in=True, strength=1.0):
        """Zoom the 2D parallel-projection camera at the cursor (Fiji-style).

        strength scales the step (trackpad / high-res wheel). Linked MPR
        optionally mirrors the scale ratio to other panes.
        """
        import math

        renderer = getattr(self, f'{orientation}_renderer', None)
        if renderer is None:
            return
        camera = renderer.GetActiveCamera()
        if not camera.GetParallelProjection():
            return

        strength = max(0.15, min(float(strength), 3.0))
        base = 1.12  # ~12 % per notch (smoother than 15 %)
        zoom_factor = base ** strength

        interactor = vtk_widget.GetRenderWindow().GetInteractor()
        mx, my = interactor.GetEventPosition()

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToDisplay()
        coord.SetValue(float(mx), float(my), 0.0)
        world_before = list(coord.GetComputedWorldValue(renderer))

        old_scale = float(camera.GetParallelScale())
        if zoom_in:
            new_scale = old_scale / zoom_factor
        else:
            new_scale = old_scale * zoom_factor

        s_min, s_max = self._mpr_zoom_limits(orientation)
        new_scale = max(s_min, min(new_scale, s_max))
        if abs(new_scale - old_scale) < 1e-9:
            return

        camera.SetParallelScale(new_scale)

        coord.SetValue(float(mx), float(my), 0.0)
        world_after = list(coord.GetComputedWorldValue(renderer))
        pos = list(camera.GetPosition())
        fp = list(camera.GetFocalPoint())
        for i in range(3):
            delta = world_before[i] - world_after[i]
            pos[i] += delta
            fp[i] += delta
        camera.SetPosition(*pos)
        camera.SetFocalPoint(*fp)

        renderer.ResetCameraClippingRange()
        self.save_camera_state(orientation)

        if self.volume_data is not None:
            h, w = self._mpr_image_size(orientation)
            self.add_ruler_overlay(renderer, orientation, (h, w))
        vtk_widget.GetRenderWindow().Render()

        # Optional linked zoom: same scale ratio on other panes (about crosshair)
        if getattr(self, "_mpr_link_nav", False) and old_scale > 1e-12:
            ratio = new_scale / old_scale
            for other in ("axial", "coronal", "sagittal"):
                if other == orientation:
                    continue
                o_ren = getattr(self, f"{other}_renderer", None)
                o_w = getattr(self, f"{other}_widget", None)
                if o_ren is None or o_w is None:
                    continue
                o_cam = o_ren.GetActiveCamera()
                if not o_cam.GetParallelProjection():
                    continue
                o_old = float(o_cam.GetParallelScale())
                o_min, o_max = self._mpr_zoom_limits(other)
                o_new = max(o_min, min(o_old * ratio, o_max))
                self._zoom_parallel_about_point(other, o_new, about_crosshair=True)

    def _scroll_change_slice(self, orientation, delta):
        """Change the current slice index by *delta* for the given orientation.

        Used when Ctrl+scroll is pressed — allows navigating through the
        volume stack without a slider.
        """
        if self.volume_data is None:
            return

        vol_z, vol_y, vol_x = self.volume_data.shape
        newX, newY, newZ = self.crosshair_position

        if orientation == 'axial':
            max_idx = vol_z - 1
            newZ = max(0, min(newZ + delta, max_idx))
        elif orientation == 'coronal':
            max_idx = vol_y - 1
            newY = max(0, min(newY + delta, max_idx))
        else:  # sagittal
            max_idx = vol_x - 1
            newX = max(0, min(newX + delta, max_idx))

        self.updatePoint(newX, newY, newZ)

    def _ensure_crosshair_visible(self, orientation):
        """Auto-pan the 2D camera so the crosshair is visible in the viewport.

        Only activates when the user has zoomed-in (parallelScale < 85 % of
        the fit-to-view default).  At default zoom the entire image fits in
        the viewport, so the crosshair is always visible — no pan needed.
        """
        if self.volume_data is None:
            return

        renderer = getattr(self, f'{orientation}_renderer', None)
        if renderer is None:
            return
        camera = renderer.GetActiveCamera()
        if not camera.GetParallelProjection():
            return

        vol_z, vol_y, vol_x = self.volume_data.shape

        # Image dimensions for this orientation (same as render_slice)
        if orientation == 'axial':
            h, w = vol_y, vol_x
        elif orientation == 'coronal':
            h, w = vol_z, vol_x
        else:  # sagittal
            h, w = vol_y, vol_z

        # Compute the "fit-to-view" default parallel scale
        vtk_widget = getattr(self, f'{orientation}_widget')
        vp_size = vtk_widget.GetRenderWindow().GetSize()
        vp_w_px = max(vp_size[0], 1)
        vp_h_px = max(vp_size[1], 1)
        vp_aspect = vp_w_px / vp_h_px
        img_aspect = w / max(h, 1)

        if img_aspect > vp_aspect:
            default_scale = (w / vp_aspect) * 0.5
        else:
            default_scale = h * 0.5

        # Only auto-pan if the user has zoomed in significantly
        current_scale = camera.GetParallelScale()
        if current_scale > default_scale * 0.85:
            return  # Not zoomed in — crosshair is always visible

        # Determine the crosshair world-space position for this orientation
        if orientation == 'axial':
            cx_w = float(self.crosshair_position[0])
            cy_w = float(self.crosshair_position[1])
        elif orientation == 'coronal':
            cx_w = float(self.crosshair_position[0])
            cy_w = float((vol_z - 1) - self.crosshair_position[2])
        else:  # sagittal
            cx_w = float(self.crosshair_position[2])
            cy_w = float(self.crosshair_position[1])

        # Convert crosshair world coords → display coords
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        coord.SetValue(cx_w, cy_w, 0.0)
        dp = coord.GetComputedDisplayValue(renderer)
        dx, dy = dp[0], dp[1]

        # Viewport pixel dimensions
        vp = renderer.GetViewport()
        vp_x0 = vp[0] * vp_size[0]
        vp_y0 = vp[1] * vp_size[1]
        vp_w = (vp[2] - vp[0]) * vp_size[0]
        vp_h = (vp[3] - vp[1]) * vp_size[1]

        # Check if crosshair display position is within the viewport
        # (no margin — only pan when truly off-screen)
        in_view = (
            vp_x0 <= dx <= (vp_x0 + vp_w) and
            vp_y0 <= dy <= (vp_y0 + vp_h)
        )

        if in_view:
            return  # Crosshair is visible — nothing to do

        # Pan camera to center on the crosshair world position
        pos = list(camera.GetPosition())
        fp = list(camera.GetFocalPoint())

        delta_x = cx_w - fp[0]
        delta_y = cy_w - fp[1]

        pos[0] += delta_x
        pos[1] += delta_y
        fp[0] += delta_x
        fp[1] += delta_y

        camera.SetPosition(*pos)
        camera.SetFocalPoint(*fp)
        renderer.ResetCameraClippingRange()

        # Persist the new camera state
        self.save_camera_state(orientation)

        vtk_widget.GetRenderWindow().Render()

    def _plane_cell_stylesheet(self, active=False):
        """Dragonfly-style pane chrome: seamless black fill, optional focus ring."""
        border = (
            SemiconductorTheme.ACCENT_WARNING if active
            else "transparent"
        )
        # 1px transparent border reserves space so activating a pane does not reflow
        return (
            f"QWidget#PlaneCell {{"
            f"  background-color: #000000;"
            f"  border: 1px solid {border};"
            f"}}"
        )

    def _set_active_grid_pane(self, orientation):
        """Highlight the active MPR/3D pane with a thin focus ring (Dragonfly-like)."""
        if not hasattr(self, '_grid_cells'):
            return
        self._active_grid_pane = orientation
        for key, cell in self._grid_cells.items():
            widget = cell[0]
            is_active = (key == orientation)
            widget.setProperty("active", "true" if is_active else "false")
            widget.setStyleSheet(self._plane_cell_stylesheet(active=is_active))

    def create_slice_view(self, title, orientation):
        widget = QWidget()
        widget.setObjectName("PlaneCell")
        widget.setStyleSheet(self._plane_cell_stylesheet(active=False))
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        widget.setMinimumSize(0, 0)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # --- COMPACT HEADER (Title & Pixel Info) — soft overlay bar, no hard seam ---
        header_widget = QWidget()
        header_widget.setMaximumHeight(26)
        header_widget.setStyleSheet(
            "background: rgba(0, 0, 0, 0.45); border: none;"
        )
        
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(8, 0, 8, 0)
        
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setStyleSheet(f"font-size: 9pt; color: {SemiconductorTheme.ACCENT_PRIMARY};")
        header_layout.addWidget(title_label)
        
        # Apply shadow effect for readability
        title_shadow = QGraphicsDropShadowEffect()
        title_shadow.setBlurRadius(3)
        title_shadow.setOffset(1, 1)
        title_shadow.setColor(QColor(0, 0, 0, 220))
        title_label.setGraphicsEffect(title_shadow)

        header_layout.addStretch()
        
        pixel_label = QLabel("Pixel: --")
        pixel_label.setStyleSheet(f"font-size: 8pt; color: white;")
        header_layout.addWidget(pixel_label)
        
        pixel_shadow = QGraphicsDropShadowEffect()
        pixel_shadow.setBlurRadius(3)
        pixel_shadow.setOffset(1, 1)
        pixel_shadow.setColor(QColor(0, 0, 0, 220))
        pixel_label.setGraphicsEffect(pixel_shadow)
        
        layout.addWidget(header_widget)
        
        # VTK container — pure black so hairline grid dividers read cleanly
        vtk_container = QWidget()
        vtk_container.setStyleSheet("background-color: #000000; border: none;")
        vtk_grid = QGridLayout(vtk_container)
        vtk_grid.setContentsMargins(0, 0, 0, 0)
        vtk_grid.setSpacing(0)
        
        vtk_widget = QVTKRenderWindowInteractor()
        vtk_grid.addWidget(vtk_widget, 0, 0)
        
        style = vtk.vtkInteractorStyleImage()
        rw = vtk_widget.GetRenderWindow()
        # Need alpha bit-planes so RGBA segmentation overlays composite correctly
        # (otherwise mask RGBA may look solid yellow / ignore per-pixel alpha).
        try:
            rw.SetAlphaBitPlanes(1)
            rw.SetMultiSamples(0)
        except Exception:
            pass
        rw.GetInteractor().SetInteractorStyle(style)
        self.setup_vtk_observers(vtk_widget, orientation)
        vtk_widget.installEventFilter(self)
        
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0, 0, 0)
        try:
            renderer.SetUseDepthPeeling(0)
        except Exception:
            pass
        rw.AddRenderer(renderer)
        
        # --- VERTICAL OVERLAY ---
        on_fs = lambda checked=False, o=orientation: self.toggle_view_fullscreen(o)
        on_reset = lambda checked=False, o=orientation: self.reset_view(o)
        on_rev_z = self.toggle_reverse_z if orientation == 'axial' else None
        
        overlay = SliceControlOverlay(vtk_widget, orientation, on_fs, on_reset, on_rev_z)
        setattr(self, f'{orientation}_overlay_group', overlay)
        QTimer.singleShot(100, overlay.update_position) # Delay to ensure widget is laid out
        
        layout.addWidget(vtk_container, 1)
        
        # Bottom Slider — soft bar so panes stay visually linked
        bottom_widget = QWidget()
        bottom_widget.setMaximumHeight(28)
        bottom_widget.setStyleSheet(
            "background: rgba(0, 0, 0, 0.45); border: none;"
        )
        
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(8, 0, 8, 0)
        
        slice_text = QLabel("SLICE:")
        slice_text.setStyleSheet(f"font-size: 8pt; color: {SemiconductorTheme.ACCENT_PRIMARY}; font-weight: bold;")
        bottom_layout.addWidget(slice_text)
        
        slice_text_shadow = QGraphicsDropShadowEffect()
        slice_text_shadow.setBlurRadius(3)
        slice_text_shadow.setOffset(1, 1)
        slice_text_shadow.setColor(QColor(0, 0, 0, 220))
        slice_text.setGraphicsEffect(slice_text_shadow)
        
        slice_slider = StepOneSliceSlider(Qt.Horizontal)
        slice_slider.setMinimum(0)
        slice_slider.setMaximum(100)
        slice_slider.setValue(50)
        slice_slider.setMaximumHeight(16)
        slice_slider.setToolTip("Drag or scroll (step 1) to change slice")
        slice_slider.valueChanged.connect(lambda value, o=orientation: self.update_slice(o, value))
        bottom_layout.addWidget(slice_slider, 1)
        
        slice_label = QLabel("50 / 100")
        slice_label.setStyleSheet(f"font-size: 8pt; color: white; min-width: 60px;")
        slice_slider.valueChanged.connect(lambda value, lbl=slice_label, s=slice_slider: lbl.setText(f"{value} / {s.maximum()}"))
        bottom_layout.addWidget(slice_label)
        
        slice_label_shadow = QGraphicsDropShadowEffect()
        slice_label_shadow.setBlurRadius(3)
        slice_label_shadow.setOffset(1, 1)
        slice_label_shadow.setColor(QColor(0, 0, 0, 220))
        slice_label.setGraphicsEffect(slice_label_shadow)

        # Wheel on entire slice bar (label + groove + value) → step 1
        wheel_filter = SliceBarWheelFilter(slice_slider, bottom_widget)
        bottom_widget.installEventFilter(wheel_filter)
        slice_slider.installEventFilter(wheel_filter)
        slice_text.installEventFilter(wheel_filter)
        slice_label.installEventFilter(wheel_filter)
        # Keep filter alive with the bar
        bottom_widget._slice_wheel_filter = wheel_filter
        
        layout.addWidget(bottom_widget)
        widget.setLayout(layout)
        
        # Store refs
        setattr(self, f'{orientation}_widget', vtk_widget)
        setattr(self, f'{orientation}_renderer', renderer)
        setattr(self, f'{orientation}_slice_slider', slice_slider)
        setattr(self, f'{orientation}_slice_label', slice_label)
        setattr(self, f'{orientation}_overlay_c1', overlay.c1_btn)
        setattr(self, f'{orientation}_overlay_c2', overlay.c2_btn)
        setattr(self, f'{orientation}_opacity_slider', overlay.opacity_slider)
        if orientation == 'axial' and hasattr(overlay, 'z_btn'):
            self.reverse_z_btn = overlay.z_btn
        
        # Use toggled/valueChanged with default-arg capture so each MPR pane
        # always refreshes the correct orientation (and checkable clicked(bool)
        # never confuses the slot).
        overlay.c1_btn.toggled.connect(
            lambda _checked=False, o=orientation: self.update_overlay_visibility(o)
        )
        overlay.c2_btn.toggled.connect(
            lambda _checked=False, o=orientation: self.update_overlay_visibility(o)
        )
        overlay.opacity_slider.valueChanged.connect(
            lambda _val=0, o=orientation: self.update_overlay_visibility(o)
        )
        
        setattr(self, f'{orientation}_pixel_label', pixel_label)
        setattr(self, f'{orientation}_container', widget)
        
        return widget

    def toggle_view_fullscreen(self, orientation):
        """Toggle in-grid fullscreen: expand one view to fill the current layout grid."""
        if self._fullscreen_view is not None:
            # Already fullscreen — exit
            self.exit_fullscreen()
            return

        if orientation not in getattr(self, "_layout_visible_keys", set(self._grid_cells.keys())):
            # Pane not in current layout — nothing to maximize
            return

        # ── Enter fullscreen ──
        self._fullscreen_view = orientation
        self._set_active_grid_pane(orientation)

        # Hide the other views in the current layout
        for key, cell in self._grid_cells.items():
            widget = cell[0]
            if key != orientation:
                widget.setVisible(False)

        # Make the target view span the full current grid
        cell = self._grid_cells[orientation]
        target_widget = cell[0]
        nrows = max(1, int(getattr(self, "_grid_nrows", 2)))
        ncols = max(1, int(getattr(self, "_grid_ncols", 2)))
        self._grid_layout.removeWidget(target_widget)
        self._grid_layout.addWidget(target_widget, 0, 0, nrows, ncols)

        # For 3D: show the advanced sidebar (reserve width so dense rows
        # like MPR Show/Clip/Flip are not clipped when the v-scrollbar appears)
        if orientation == 'view_3d':
            self.advanced_3d_controls.setVisible(True)
            self.advanced_3d_controls.setMinimumWidth(360)
            self.advanced_3d_controls.setMaximumWidth(420)
            self.advanced_3d_controls.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Keep overlay strip identical to non-fullscreen (same maximize icon,
        # tooltip, and vertical button column). Only re-anchor after grid span.
        overlay = getattr(self, f'{orientation}_overlay_group', None)
        if not overlay:
            overlay = getattr(self, 'view_3d_overlay_group', None)
        if overlay:
            overlay.set_fullscreen(True)
            QTimer.singleShot(50, overlay.update_position)

    def exit_fullscreen(self):
        """Restore the current view layout from in-grid fullscreen."""
        if self._fullscreen_view is None:
            return

        orientation = self._fullscreen_view

        # Remove the spanning widget and place it back at its layout position
        cell = self._grid_cells.get(orientation)
        if cell is None:
            self._fullscreen_view = None
            return
        target_widget, orig_row, orig_col = cell[0], cell[1], cell[2]
        rowspan = cell[3] if len(cell) > 3 else 1
        colspan = cell[4] if len(cell) > 4 else 1
        self._grid_layout.removeWidget(target_widget)
        self._grid_layout.addWidget(target_widget, orig_row, orig_col, rowspan, colspan)

        # Show only panes that belong to the active layout preset
        visible = getattr(self, "_layout_visible_keys", set(self._grid_cells.keys()))
        for key, cell in self._grid_cells.items():
            widget = cell[0]
            widget.setVisible(key in visible)

        # For 3D: hide + collapse advanced sidebar so 2×2 cells stay even
        if orientation == 'view_3d':
            self.advanced_3d_controls.setVisible(False)
            self.advanced_3d_controls.setMinimumWidth(0)
            self.advanced_3d_controls.setMaximumWidth(0)
            self.advanced_3d_controls.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

        # Same strip look as enter — no icon/tooltip swap
        overlay = getattr(self, f'{orientation}_overlay_group', None)
        if not overlay:
            overlay = getattr(self, 'view_3d_overlay_group', None)
        if overlay:
            overlay.set_fullscreen(False)
            QTimer.singleShot(50, overlay.update_position)

        self._fullscreen_view = None
        QTimer.singleShot(40, self._refresh_layout_viewports)

    def set_projection(self, mode):
        self.projection_mode = str(mode).split('#')[0].strip().lower()
        if not hasattr(self, 'view_3d_renderer') or self.view_3d_renderer is None: 
            return
        
        cam = self.view_3d_renderer.GetActiveCamera()
        if self.projection_mode == "perspective":
            if hasattr(self, 'btn_perspective'):
                self.btn_perspective.setChecked(True)
            if hasattr(self, 'btn_ortho'):
                self.btn_ortho.setChecked(False)
            cam.ParallelProjectionOff()
            cam.SetViewAngle(40.0)
            self.view_3d_renderer.ResetCameraClippingRange()
        else:
            if hasattr(self, 'btn_ortho'):
                self.btn_ortho.setChecked(True)
            if hasattr(self, 'btn_perspective'):
                self.btn_perspective.setChecked(False)
            cam.ParallelProjectionOn()
            
        if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
            self.view_3d_widget.GetRenderWindow().Render()

    def create_3d_view(self):
        widget = QWidget()
        widget.setObjectName("PlaneCell")
        widget.setStyleSheet(self._plane_cell_stylesheet(active=False))
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        widget.setMinimumSize(0, 0)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # --- COMPACT HEADER (Matches 2D Views) ---
        header_widget = QWidget()
        header_widget.setMaximumHeight(26)
        header_widget.setStyleSheet(
            "background: rgba(0, 0, 0, 0.45); border: none;"
        )
        
        controls_layout = QHBoxLayout(header_widget)
        controls_layout.setContentsMargins(8, 0, 8, 0)
        
        # Title
        title_label = QLabel("<b>3D Volume</b>")
        title_label.setStyleSheet(f"font-size: 9pt; color: {SemiconductorTheme.ACCENT_PRIMARY};")
        controls_layout.addWidget(title_label)
        
        # Apply shadow effect for readability (matches 2D)
        title_shadow = QGraphicsDropShadowEffect()
        title_shadow.setBlurRadius(3)
        title_shadow.setOffset(1, 1)
        title_shadow.setColor(QColor(0, 0, 0, 220))
        title_label.setGraphicsEffect(title_shadow)
        
        controls_layout.addStretch()
        
        # Pixel info (empty for 3D)
        pixel_label = QLabel("")
        pixel_label.setStyleSheet("font-size: 8pt; color: white;")
        controls_layout.addWidget(pixel_label)
        
        self.controls_3d_container = header_widget
        layout.addWidget(self.controls_3d_container)

        # --- MAIN CONTENT DIVIDER ---
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(content_layout)
        self._3d_content_layout = content_layout  # keep ref for exit_fullscreen
        
        # --- LEFT: VTK WIDGET (75%) ---
        vtk_container = QWidget()
        vtk_container.setObjectName("VTK3DContainer")
        vtk_container.setStyleSheet("#VTK3DContainer { background-color: #000000; border: none; }")
        vtk_layout = QVBoxLayout(vtk_container)
        vtk_layout.setContentsMargins(0, 0, 0, 0)
        vtk_layout.setSpacing(0)
        
        vtk_widget = QVTKRenderWindowInteractor()
        renderer = vtk.vtkRenderer()
        
        # --- 1. Dragonfly-style Gradient Background ---
        try:
            # Try native radial gradient (VTK 9.3+)
            renderer.GradientBackgroundOn()
            renderer.SetGradientMode(renderer.VTK_GRADIENT_RADIAL_FARTHEST_CORNER)
            renderer.SetBackground(0.85, 0.85, 0.85)  # Center: Very Bright (Like Dragonfly empty view)
            renderer.SetBackground2(0.15, 0.15, 0.15) # Edges: Dark Gray
        except AttributeError:
            # Fallback to Textured Background (VTK < 9.3) for perfect radial gradient
            try:
                dim = 256
                xx, yy = np.meshgrid(np.arange(dim), np.arange(dim))
                center = dim / 2.0
                r = np.sqrt((xx - center)**2 + (yy - center)**2)
                r_norm = r / (dim / 2.0)
                # Gaussian-like glow to simulate a light bulb hotspot
                factor = np.exp(-1.8 * (r_norm ** 1.2))
                
                c_color = np.array([240, 240, 240]) # Bright center (almost white)
                e_color = np.array([30, 30, 30])    # Dark edges
                
                img_array = (c_color[np.newaxis, np.newaxis, :] * factor[:, :, np.newaxis] + 
                             e_color[np.newaxis, np.newaxis, :] * (1.0 - factor[:, :, np.newaxis]))
                img_array = img_array.astype(np.uint8)
                
                image_data = vtk.vtkImageData()
                image_data.SetDimensions(dim, dim, 1)
                vtk_array = numpy_support.numpy_to_vtk(img_array.reshape(-1, 3), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
                image_data.GetPointData().SetScalars(vtk_array)
                
                bg_texture = vtk.vtkTexture()
                bg_texture.SetInputData(image_data)
                
                renderer.SetBackgroundTexture(bg_texture)
                renderer.SetTexturedBackground(True)
                # Keep a reference to prevent garbage collection
                self.bg_texture = bg_texture
            except Exception:
                # Ultimate fallback to linear vertical gradient
                renderer.GradientBackgroundOn()
                renderer.SetBackground(0.15, 0.15, 0.15)  # Bottom: Dark
                renderer.SetBackground2(0.85, 0.85, 0.85) # Top: Lighter Gray
            
        vtk_widget.GetRenderWindow().AddRenderer(renderer)
        
        # --- 2. Soft three-point lighting (Dragonfly-like, not stage-spot) ---
        self.light_kit = vtk.vtkLightKit()
        self.light_kit.AddLightsToRenderer(renderer)
        # Lower key + stronger fill → soft form light, fewer specular “caps”
        self.light_kit.SetKeyLightIntensity(0.72)
        self.light_kit.SetKeyToFillRatio(1.8)
        self.light_kit.SetKeyLightWarmth(0.55)
        try:
            self.light_kit.SetKeyLightElevation(45.0)
            self.light_kit.SetKeyLightAzimuth(10.0)
            self.light_kit.SetFillLightWarmth(0.45)
            self.light_kit.SetHeadLightWarmth(0.5)
            self.light_kit.SetKeyToHeadRatio(2.2)
            self.light_kit.SetKeyToBackRatio(3.5)
        except Exception:
            pass
        
        # --- RTX 5090 Performance Optimizations ---
        render_window = vtk_widget.GetRenderWindow()
        
        # 1. Desired update rate during interaction (higher = smoother rotation)
        #    VTK will automatically reduce quality to meet this frame rate target
        render_window.SetDesiredUpdateRate(60.0)  # Target 60 FPS during interaction
        
        # 2. Still render update rate (lower = higher quality when idle)
        if hasattr(render_window, 'SetStillUpdateRate'):
            render_window.SetStillUpdateRate(0.001)  # Max quality when not interacting
        
        # 3. Multi-sample anti-aliasing (MSAA) — RTX 5090 handles this easily
        if hasattr(render_window, 'SetMultiSamples'):
            render_window.SetMultiSamples(4)  # 4x MSAA for smooth edges
        
        # 4. Enable depth peeling for correct transparency compositing
        renderer.SetUseDepthPeeling(1)
        renderer.SetMaximumNumberOfPeels(4)
        renderer.SetOcclusionRatio(0.1)
        
        # Dragonfly-style 3D navigation: Track around pivot, pan, unlimited zoom
        style = Dragonfly3DInteractorStyle()
        style._host = self  # right-drag / VTK wheel backup → same zoom as Qt wheel
        interactor = vtk_widget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(style)
        self._3d_interactor_style = style
        
        # 5. LOD Interaction Observers: reduce quality during mouse drag, refine on release
        #    This prevents lag when rotating/zooming large 10GB volumes
        def _on_interaction_start(caller, event):
            self._start_lod_interaction()
        def _on_interaction_end(caller, event):
            self._end_lod_interaction()
        
        interactor.AddObserver('StartInteractionEvent', _on_interaction_start)
        interactor.AddObserver('EndInteractionEvent', _on_interaction_end)
        
        # 6. Clipping: use VTK default ResetCameraClippingRange (camera stays
        #    outside the volume — no fly-through near-plane override).
        
        # --- 3. Orientation Marker (Dragonfly-style Cube) ---
        # Assembly = labeled cube + per-face hover highlight plates.
        # InteractiveOff: fixed corner (no drag). Click/hover handled by us.
        # Hover color = app primary cyan (Dragonfly uses yellow).
        self._ori_hover_face = None
        self._ori_face_highlights = {}
        self._ori_face_text_props = {}
        marker = self._build_orientation_marker()

        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(marker)
        self.axes_widget.SetInteractor(interactor)
        self.axes_widget.SetViewport(0.82, 0.0, 1.0, 0.18)
        self.axes_widget.EnabledOn()
        self.axes_widget.InteractiveOff()

        interactor.AddObserver(
            "LeftButtonPressEvent", self._on_orientation_cube_click, 10.0
        )
        interactor.AddObserver(
            "MouseMoveEvent", self._on_orientation_cube_hover, 5.0
        )

        # Dragonfly clip-box face interaction (priority > style defaults, < nothing critical)
        # Style overrides alone are unreliable on some VTK/Qt builds — observers are primary.
        interactor.AddObserver(
            "LeftButtonPressEvent", self._on_df_clip_3d_left_press, 25.0
        )
        interactor.AddObserver(
            "MouseMoveEvent", self._on_df_clip_3d_mouse_move, 25.0
        )
        interactor.AddObserver(
            "LeftButtonReleaseEvent", self._on_df_clip_3d_left_release, 25.0
        )
        self._df_clip_3d_face_actors = {}  # id(actor) -> (axis, side)

        vtk_layout.addWidget(vtk_widget)
        
        # --- VERTICAL OVERLAY (Matches 2D Views) ---
        on_fs = lambda checked=False: self.toggle_view_fullscreen('view_3d')
        on_reset = lambda checked=False: self.reset_3d_view()
        
        self.view_3d_overlay_group = SliceControlOverlay(vtk_widget, '3d', on_fs, on_reset)
        # Master 3D render switch (under C2). Default ON offline; Online auto-OFF.
        self._3d_volume_render_enabled = True
        self._3d_render_force_on_online = False
        self._3d_disabled_text_actor = None
        if hasattr(self.view_3d_overlay_group, 'c1_btn'):
            self.view_3d_overlay_group.c1_btn.toggled.connect(
                lambda _c=False: self.update_overlay_visibility('3d')
            )
        if hasattr(self.view_3d_overlay_group, 'c2_btn'):
            self.view_3d_overlay_group.c2_btn.toggled.connect(
                lambda _c=False: self.update_overlay_visibility('3d')
            )
        if hasattr(self.view_3d_overlay_group, 'opacity_slider'):
            self.view_3d_overlay_group.opacity_slider.valueChanged.connect(
                lambda _v=0: self.update_overlay_visibility('3d')
            )
        if getattr(self.view_3d_overlay_group, "render_3d_btn", None) is not None:
            self.view_3d_overlay_group.render_3d_btn.toggled.connect(
                self._on_3d_render_btn_toggled
            )
            
        QTimer.singleShot(100, self.view_3d_overlay_group.update_position)
        # Stretch 1: VTK fills remaining width after the fixed-min sidebar
        content_layout.addWidget(vtk_container, 1)

        self.view_3d_widget = vtk_widget
        self.view_3d_renderer = renderer
        self._vtk_3d_container = vtk_container
        # Focus ring + scroll-zoom: filter on both VTK widget and its container
        # (VTK often eats wheel events before Qt sees them on the interactor alone)
        vtk_widget.installEventFilter(self)
        vtk_container.installEventFilter(self)
        vtk_widget.setMouseTracking(True)
        vtk_container.setMouseTracking(True)
        vtk_widget.setFocusPolicy(Qt.StrongFocus)
        vtk_container.setFocusPolicy(Qt.StrongFocus)

        # --- RIGHT: ADVANCED CONTROLS (stable min width; shown in fullscreen) ---
        self.advanced_3d_controls = QWidget()
        
        adv_main_layout = QVBoxLayout()
        adv_main_layout.setContentsMargins(10, 10, 10, 10)
        adv_main_layout.setSpacing(15) # Group spacing
        

        
        # Controls Panel (Vertical instead of Horizontal)
        controls_panel = QVBoxLayout()
        controls_panel.setSpacing(10)
        
        # --- Rendering Quality Group ---
        qual_group = QGroupBox("Rendering Quality")
        q_lay = QVBoxLayout()
        
        color_row = QHBoxLayout()
        self.color_btn = QPushButton("Volume Color")
        self.color_btn.clicked.connect(self.choose_volume_color)
        color_row.addWidget(self.color_btn)
        self.view_3d_bg_color_btn = QPushButton("Background Color")
        self.view_3d_bg_color_btn.clicked.connect(self.choose_3d_bg_color)
        color_row.addWidget(self.view_3d_bg_color_btn)
        q_lay.addLayout(color_row)
        
        # Quality presets tuned for high-end GPU (RTX 5090 32GB VRAM, 128GB RAM)
        # sample_dist: ray sampling interval (smaller = sharper, heavier)
        # image_sample: pixel sub-sampling (< 1.0 = supersampled, > 1.0 = faster)
        # interact_image_sample: pixel sub-sampling DURING mouse interaction (LOD)
        # interact_sample_dist: ray sampling interval DURING interaction
        # auto_adj: let VTK auto-tune sample distance during interaction
        # jitter: stochastic jittering to eliminate periodic banding artifacts
        # max_mem_fraction: fraction of GPU VRAM to allocate
        # Quality: denser sampling + mild supersample ≈ Dragonfly "Best quality" still view
        self._quality_presets = {
            "Draft":    {"sample_dist": 2.0,  "image_sample": 1.0, "interact_image_sample": 3.0, "interact_sample_dist": 4.0, "auto_adj": True,  "jitter": False, "max_mem_fraction": 0.75},
            "Standard": {"sample_dist": 1.0,  "image_sample": 1.0, "interact_image_sample": 2.0, "interact_sample_dist": 3.0, "auto_adj": True,  "jitter": True,  "max_mem_fraction": 0.80},
            "High":     {"sample_dist": 0.5,  "image_sample": 1.0, "interact_image_sample": 1.5, "interact_sample_dist": 2.0, "auto_adj": True,  "jitter": True,  "max_mem_fraction": 0.85},
            "Ultra":    {"sample_dist": 0.25, "image_sample": 0.75, "interact_image_sample": 1.25, "interact_sample_dist": 1.5, "auto_adj": True,  "jitter": True,  "max_mem_fraction": 0.90},
        }
        self._current_quality = "High"

        # Phong looks that approximate ORS Dragonfly examine mode (soft silver, not plastic)
        self._lighting_look_presets = {
            "Dragonfly": {
                # Soft metallic — form without specular “raised caps”
                "shade": True, "ambient": 40, "diffuse": 58, "specular": 18, "power": 16,
                "edge": 22, "opacity": 100,
                "key_intensity": 0.72, "key_fill": 1.8, "key_warmth": 0.55,
                "vol_shadows": False,
            },
            "Inspection": {
                # Flatter — max surface/void readability for HBM metrology
                "shade": True, "ambient": 48, "diffuse": 45, "specular": 8, "power": 10,
                "edge": 15, "opacity": 100,
                "key_intensity": 0.65, "key_fill": 1.5, "key_warmth": 0.5,
                "vol_shadows": False,
            },
            "Plastic": {
                # Old glossy look (demo only — strong specular pop)
                "shade": True, "ambient": 25, "diffuse": 85, "specular": 80, "power": 50,
                "edge": 0, "opacity": 100,
                "key_intensity": 0.85, "key_fill": 2.5, "key_warmth": 0.6,
                "vol_shadows": True,
            },
            "Flat": {
                "shade": False, "ambient": 100, "diffuse": 0, "specular": 0, "power": 1,
                "edge": 0, "opacity": 100,
                "key_intensity": 0.7, "key_fill": 1.5, "key_warmth": 0.5,
                "vol_shadows": False,
            },
        }
        self._current_lighting_look = "Dragonfly"
        
        # LOD refinement timer: after interaction stops, re-render at full quality
        self._lod_refine_timer = QTimer(self)
        self._lod_refine_timer.setSingleShot(True)
        self._lod_refine_timer.setInterval(300)  # ms delay after interaction ends
        self._lod_refine_timer.timeout.connect(self._refine_after_interaction)
        self._is_interacting_3d = False
        
        # Quality preset selector
        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Draft", "Standard", "High", "Ultra"])
        self.quality_combo.setCurrentText("High")
        self.quality_combo.setToolTip(
            "Draft: Fast preview, lower detail (good for very large volumes)\n"
            "Standard: Balanced quality and performance\n"
            "High: High quality (recommended)\n"
            "Ultra: Dragonfly-like still quality (denser rays + mild supersample)"
        )
        self.quality_combo.currentTextChanged.connect(self._on_quality_preset_changed)
        quality_row.addWidget(self.quality_combo)
        q_lay.addLayout(quality_row)
        
        # Render Mode selector
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(["Composite", "MIP", "MinIP", "Average", "Additive"])
        self.render_mode_combo.setToolTip(
            "Composite: Standard volume rendering\n"
            "MIP: Maximum Intensity Projection\n"
            "MinIP: Minimum Intensity Projection\n"
            "Average: Average Intensity\n"
            "Additive: Additive (X-Ray like)"
        )
        self.render_mode_combo.currentTextChanged.connect(self.on_render_mode_changed)
        mode_row.addWidget(self.render_mode_combo)
        q_lay.addLayout(mode_row)
        
        # Downsample selector
        downsample_row = QHBoxLayout()
        downsample_row.addWidget(QLabel("Downsample:"))
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 8)
        self.downsample_spin.setValue(2)
        downsample_row.addWidget(self.downsample_spin)
        q_lay.addLayout(downsample_row)
        
        qual_group.setLayout(q_lay)
        controls_panel.addWidget(qual_group)
        
        # ═══════════════════════════════════════════════════════════════
        # Window Leveling — Dragonfly-style panel
        # Histogram · LUT · Selected range · Color mapping · Opacity mapping
        # ═══════════════════════════════════════════════════════════════
        hist_group = QGroupBox("Window Leveling")
        hist_lay = QVBoxLayout()
        hist_lay.setContentsMargins(6, 10, 6, 6)
        hist_lay.setSpacing(6)

        # Top: mode W/L | TF + Log Y
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self._tf_mode_group = QButtonGroup(self)
        self.btn_wl_mode = QPushButton("W/L")
        self.btn_wl_mode.setCheckable(True)
        self.btn_wl_mode.setChecked(True)
        self.btn_wl_mode.setToolTip(
            "Window Leveling (Dragonfly): Min/Max bars + opacity curve shapes."
        )
        self.btn_tf_mode = QPushButton("TF")
        self.btn_tf_mode.setCheckable(True)
        self.btn_tf_mode.setToolTip(
            "Transfer Function: free opacity spline (click add / drag / right-click delete)."
        )
        for b in (self.btn_wl_mode, self.btn_tf_mode):
            b.setFixedHeight(22)
            b.setStyleSheet(
                "QPushButton { font-size: 8pt; padding: 2px 10px; }"
                "QPushButton:checked { background: #3a6ea5; color: white; "
                "border: 1px solid #5a9ed5; }"
            )
            mode_row.addWidget(b)
        self._tf_mode_group.addButton(self.btn_wl_mode, 0)
        self._tf_mode_group.addButton(self.btn_tf_mode, 1)
        self._tf_mode_group.setExclusive(True)
        mode_row.addStretch(1)
        self.tf_log_y_check = QCheckBox("Log Y")
        self.tf_log_y_check.setStyleSheet("color: #a0a0a0; font-size: 8pt;")
        mode_row.addWidget(self.tf_log_y_check)
        hist_lay.addLayout(mode_row)

        # Histogram widget
        self.tf_widget = TransferFunctionWidget()
        self.tf_widget.set_mode(TransferFunctionWidget.MODE_WL)
        self.tf_widget.setMinimumHeight(130)
        self.tf_log_y_check.toggled.connect(self.tf_widget.set_log_y)
        self.tf_widget.opacityChanged.connect(self.on_opacity_spline_changed)
        self.tf_widget.rangeChanged.connect(self._on_tf_range_changed)
        self.tf_widget.modeChanged.connect(self._on_tf_mode_changed)
        self.btn_wl_mode.clicked.connect(
            lambda: self.tf_widget.set_mode(TransferFunctionWidget.MODE_WL)
        )
        self.btn_tf_mode.clicked.connect(
            lambda: self.tf_widget.set_mode(TransferFunctionWidget.MODE_TF)
        )
        hist_lay.addWidget(self.tf_widget)

        # Editable colour stops (TF detail; also feeds LUT ramp)
        self.color_grad_widget = ColorGradientWidget()
        self.color_grad_widget.colorChanged.connect(self.on_color_gradient_changed)
        self.color_grad_widget.colorChanged.connect(
            lambda _: (
                self.tf_preset_combo.blockSignals(True),
                self.tf_preset_combo.setCurrentIndex(0),
                self.tf_preset_combo.blockSignals(False),
            )
        )
        hist_lay.addWidget(self.color_grad_widget)

        # ── Lookup table (LUT) ──
        lut_row = QHBoxLayout()
        lut_lbl = QLabel("Lookup table (LUT)")
        lut_lbl.setStyleSheet("color: #c0c0c0; font-size: 8pt; font-weight: bold;")
        lut_row.addWidget(lut_lbl)
        hist_lay.addLayout(lut_row)

        self.tf_preset_combo = QComboBox()
        self.tf_preset_combo.setIconSize(QSize(72, 14))
        lut_names = [
            "Custom", "Default 3D", "Discrete 24", "Discrete 64", "Electronics",
            "Entomology", "Fire", "Grayscale", "Grayscale Solid", "Grayscale Solid2",
            "Green", "Steel", "Steel2", "Temperature", "Thermal", "Vegetal",
            "Vegetal2", "Warm Metal", "White", "Wood4", "Yellow",
        ]
        for name in lut_names:
            icon = self._make_lut_icon(name, 72, 14)
            self.tf_preset_combo.addItem(icon, name)
        self.tf_preset_combo.setCurrentText("Grayscale")
        self.tf_preset_combo.setToolTip("Dragonfly-style Lookup Table preset")
        self.tf_preset_combo.currentTextChanged.connect(self.apply_tf_preset)
        hist_lay.addWidget(self.tf_preset_combo)

        # ── Selected range (Min / Max) ──
        sel_lbl = QLabel("Selected range")
        sel_lbl.setStyleSheet("color: #c0c0c0; font-size: 8pt; font-weight: bold;")
        hist_lay.addWidget(sel_lbl)

        wl_row = QHBoxLayout()
        self.view_3d_min_spin = QSpinBox()
        self.view_3d_min_spin.setRange(0, 65535)
        self.view_3d_min_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.view_3d_min_spin.setAlignment(Qt.AlignCenter)
        self.view_3d_min_spin.valueChanged.connect(self._on_wl_spin_changed)
        wl_row.addWidget(self.view_3d_min_spin)
        self.view_3d_max_spin = QSpinBox()
        self.view_3d_max_spin.setRange(0, 65535)
        self.view_3d_max_spin.setValue(65535)
        self.view_3d_max_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.view_3d_max_spin.setAlignment(Qt.AlignCenter)
        self.view_3d_max_spin.valueChanged.connect(self._on_wl_spin_changed)
        wl_row.addWidget(self.view_3d_max_spin)
        hist_lay.addLayout(wl_row)

        self._wl_info_label = QLabel("W: 65535   L: 32767.5")
        self._wl_info_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 7pt;"
        )
        self._wl_info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hist_lay.addWidget(self._wl_info_label)

        # ── Plotted range / Data range ──
        plot_lbl = QLabel("Plotted range / Data range")
        plot_lbl.setStyleSheet("color: #c0c0c0; font-size: 8pt; font-weight: bold;")
        hist_lay.addWidget(plot_lbl)

        data_row = QHBoxLayout()
        self._data_range_min_spin = QDoubleSpinBox()
        self._data_range_min_spin.setDecimals(2)
        self._data_range_min_spin.setRange(-1e9, 1e9)
        self._data_range_min_spin.setReadOnly(True)
        self._data_range_min_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._data_range_min_spin.setAlignment(Qt.AlignCenter)
        data_row.addWidget(self._data_range_min_spin)

        self.btn_reset_wl_range = QPushButton("Reset")
        self.btn_reset_wl_range.setFixedWidth(52)
        self.btn_reset_wl_range.setToolTip("Reset selected range to full data range")
        self.btn_reset_wl_range.clicked.connect(self._reset_wl_to_data_range)
        data_row.addWidget(self.btn_reset_wl_range)

        self._data_range_max_spin = QDoubleSpinBox()
        self._data_range_max_spin.setDecimals(2)
        self._data_range_max_spin.setRange(-1e9, 1e9)
        self._data_range_max_spin.setReadOnly(True)
        self._data_range_max_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._data_range_max_spin.setAlignment(Qt.AlignCenter)
        data_row.addWidget(self._data_range_max_spin)
        hist_lay.addLayout(data_row)

        # Full data extent readout (like Dragonfly bottom of plotted range)
        self._data_extent_label = QLabel("—")
        self._data_extent_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 7pt;"
        )
        self._data_extent_label.setAlignment(Qt.AlignHCenter)
        hist_lay.addWidget(self._data_extent_label)

        # ── Color mapping (icon buttons) ──
        cm_lbl = QLabel("Color mapping")
        cm_lbl.setStyleSheet("color: #c0c0c0; font-size: 8pt; font-weight: bold;")
        hist_lay.addWidget(cm_lbl)

        cm_row = QHBoxLayout()
        cm_row.setSpacing(4)
        self._color_map_group = QButtonGroup(self)
        self._color_map_group.setExclusive(True)
        self.btn_cmap_normal = make_icon_toolbutton(
            make_color_map_icon("normal", selected=True),
            "Normal color mapping (low → high)",
            self,
        )
        self.btn_cmap_invert = make_icon_toolbutton(
            make_color_map_icon("invert"),
            "Invert color mapping (high → low)",
            self,
        )
        self.btn_cmap_normal.setChecked(True)
        self._color_map_group.addButton(self.btn_cmap_normal, 0)
        self._color_map_group.addButton(self.btn_cmap_invert, 1)
        self.btn_cmap_normal.clicked.connect(lambda: self._set_color_invert(False))
        self.btn_cmap_invert.clicked.connect(lambda: self._set_color_invert(True))
        cm_row.addWidget(self.btn_cmap_normal)
        cm_row.addWidget(self.btn_cmap_invert)
        cm_row.addStretch(1)
        hist_lay.addLayout(cm_row)

        # ── Opacity mapping (slider + shape icons + gamma) ──
        om_lbl = QLabel("Opacity mapping")
        om_lbl.setStyleSheet("color: #c0c0c0; font-size: 8pt; font-weight: bold;")
        hist_lay.addWidget(om_lbl)

        op_map_row = QHBoxLayout()
        op_map_row.addWidget(QLabel("Opacity:"))
        self.wl_opacity_slider = QSlider(Qt.Horizontal)
        self.wl_opacity_slider.setRange(0, 100)
        self.wl_opacity_slider.setValue(100)
        self.wl_opacity_slider.setToolTip("Global opacity / solidity scale (Dragonfly-style)")
        self.wl_opacity_slider.valueChanged.connect(self._on_wl_opacity_changed)
        op_map_row.addWidget(self.wl_opacity_slider, 1)
        self.wl_opacity_value_lbl = QLabel("1.00")
        self.wl_opacity_value_lbl.setFixedWidth(32)
        self.wl_opacity_value_lbl.setStyleSheet("font-size: 8pt; color: #aaa;")
        op_map_row.addWidget(self.wl_opacity_value_lbl)
        hist_lay.addLayout(op_map_row)

        shape_row = QHBoxLayout()
        shape_row.setSpacing(3)
        self._opacity_shape_group = QButtonGroup(self)
        self._opacity_shape_group.setExclusive(True)
        self._opacity_shape_btns = {}
        for i, shape in enumerate(OPACITY_SHAPES):
            btn = make_icon_toolbutton(
                make_opacity_shape_icon(shape, selected=(shape == "linear")),
                OPACITY_SHAPE_TIPS.get(shape, shape),
                self,
            )
            if shape == "linear":
                btn.setChecked(True)
            self._opacity_shape_group.addButton(btn, i)
            self._opacity_shape_btns[shape] = btn
            btn.clicked.connect(lambda checked=False, s=shape: self._set_opacity_shape(s))
            shape_row.addWidget(btn)
        shape_row.addStretch(1)
        hist_lay.addLayout(shape_row)

        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Gamma:"))
        self.wl_gamma_slider = QSlider(Qt.Horizontal)
        # 0.20 … 3.00  (value = gamma * 100)
        self.wl_gamma_slider.setRange(20, 300)
        self.wl_gamma_slider.setValue(100)
        self.wl_gamma_slider.setToolTip("Gamma warp on opacity curve (1.00 = linear response)")
        self.wl_gamma_slider.valueChanged.connect(self._on_wl_gamma_changed)
        gamma_row.addWidget(self.wl_gamma_slider, 1)
        self.wl_gamma_spin = QDoubleSpinBox()
        self.wl_gamma_spin.setRange(0.20, 3.00)
        self.wl_gamma_spin.setSingleStep(0.05)
        self.wl_gamma_spin.setDecimals(2)
        self.wl_gamma_spin.setValue(1.00)
        self.wl_gamma_spin.setFixedWidth(58)
        self.wl_gamma_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.wl_gamma_spin.valueChanged.connect(self._on_wl_gamma_spin_changed)
        gamma_row.addWidget(self.wl_gamma_spin)
        hist_lay.addLayout(gamma_row)

        hist_group.setLayout(hist_lay)
        controls_panel.addWidget(hist_group)

        # Track full data range for Reset / plotted range UI
        self._volume_data_min = 0.0
        self._volume_data_max = 65535.0
        self._color_invert = False
        self._opacity_shape = "linear"


        # --- B. View & Camera Presets ---
        cam_group = QGroupBox("View & Camera Presets")
        cam_lay = QVBoxLayout()
        
        proj_lay = QHBoxLayout()
        self.btn_ortho = QPushButton("Orthographic")
        self.btn_ortho.setCheckable(True)
        self.btn_ortho.clicked.connect(lambda: self.set_projection("orthographic"))
        proj_lay.addWidget(self.btn_ortho)
        
        self.btn_perspective = QPushButton("Perspective")
        self.btn_perspective.setCheckable(True)
        self.btn_perspective.setChecked(True)
        self.btn_perspective.clicked.connect(lambda: self.set_projection("perspective"))
        proj_lay.addWidget(self.btn_perspective)
        cam_lay.addLayout(proj_lay)
        
        grid_lay = QGridLayout()
        grid_lay.setSpacing(5)
        btn_front = QPushButton("Front")
        btn_front.clicked.connect(lambda: self.set_camera_preset("Front"))
        btn_back = QPushButton("Back")
        btn_back.clicked.connect(lambda: self.set_camera_preset("Back"))
        btn_left = QPushButton("Left")
        btn_left.clicked.connect(lambda: self.set_camera_preset("Left"))
        btn_right = QPushButton("Right")
        btn_right.clicked.connect(lambda: self.set_camera_preset("Right"))
        btn_top = QPushButton("Top")
        btn_top.clicked.connect(lambda: self.set_camera_preset("Top"))
        btn_bottom = QPushButton("Bottom")
        btn_bottom.clicked.connect(lambda: self.set_camera_preset("Bottom"))
        
        grid_lay.addWidget(btn_front, 0, 0)
        grid_lay.addWidget(btn_back, 0, 1)
        grid_lay.addWidget(btn_left, 1, 0)
        grid_lay.addWidget(btn_right, 1, 1)
        grid_lay.addWidget(btn_top, 2, 0)
        grid_lay.addWidget(btn_bottom, 2, 1)
        cam_lay.addLayout(grid_lay)
        


        utils_lay = QHBoxLayout()
        btn_reset = QPushButton("Reset View")
        btn_reset.clicked.connect(self.reset_3d_view)
        
        self.btn_autorotate = QPushButton("Auto-Rotate")
        self.btn_autorotate.setCheckable(True)
        self.btn_autorotate.toggled.connect(self.toggle_auto_rotate)
        
        utils_lay.addWidget(btn_reset)
        utils_lay.addWidget(self.btn_autorotate)
        cam_lay.addLayout(utils_lay)
        
        cam_group.setLayout(cam_lay)
        controls_panel.addWidget(cam_group)
        
        # --- C. Advanced MPR & Clipping ---
        mpr_clip_group = QGroupBox("Advanced MPR & Clipping")
        mc_lay = QVBoxLayout()
        mc_lay.setSpacing(8)
        
        # Bounding Box and Global Actions (two rows to avoid horizontal clipping)
        self.bbox_check = QCheckBox("Show Bounding Box")
        self.bbox_check.stateChanged.connect(self.toggle_bounding_box)
        mc_lay.addWidget(self.bbox_check)

        top_mc_lay = QHBoxLayout()
        top_mc_lay.setSpacing(4)
        btn_show_all = QPushButton("Show All")
        btn_show_all.clicked.connect(lambda: self.on_mpr_show_hide_all(True))
        btn_hide_all = QPushButton("Hide All")
        btn_hide_all.clicked.connect(lambda: self.on_mpr_show_hide_all(False))
        btn_reset_all = QPushButton("Reset")
        btn_reset_all.clicked.connect(self.on_mpr_reset_all)
        top_mc_lay.addWidget(btn_show_all, 1)
        top_mc_lay.addWidget(btn_hide_all, 1)
        top_mc_lay.addWidget(btn_reset_all, 1)
        mc_lay.addLayout(top_mc_lay)
        
        # Slab Thickness
        thick_lay = QHBoxLayout()
        thick_lay.addWidget(QLabel("MIP Thickness:"))
        self.mpr_thickness_slider = QSlider(Qt.Horizontal)
        self.mpr_thickness_slider.setRange(0, 50)
        self.mpr_thickness_slider.setValue(0)
        self.mpr_thickness_slider.valueChanged.connect(self.update_mpr_thickness)
        thick_lay.addWidget(self.mpr_thickness_slider)
        
        self.thick_lbl = QLabel("0 px")
        self.thick_lbl.setMinimumWidth(35)
        self.mpr_thickness_slider.valueChanged.connect(lambda v: self.thick_lbl.setText(f"{v} px"))
        thick_lay.addWidget(self.thick_lbl)
        mc_lay.addLayout(thick_lay)
        
        # Axis Controls — Dragonfly-style percentage sliders (0-1000 = 0-100%)
        self.adv_mpr_controls = {}
        # X: Sagittal (Red), Y: Coronal (Green), Z: Axial (Blue)
        axes_info = [('x', 'Sag (X)', '#FF4444'), ('y', 'Cor (Y)', '#44FF44'), ('z', 'Axi (Z)', '#4444FF')]
        
        for axis, label, color in axes_info:
            # Row 1: label + slider + percent (expanding slider uses remaining width)
            row_lay = QHBoxLayout()
            row_lay.setSpacing(4)
            row_lay.setContentsMargins(0, 0, 0, 0)

            lbl = QLabel(label)
            lbl.setFixedWidth(48)
            lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 8pt;")
            row_lay.addWidget(lbl)

            # Slider — percentage-based (0-1000 = 0.0-100.0%)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 1000)
            sl.setValue(500)  # Start at 50%
            sl.setMinimumWidth(60)
            sl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            sl.setStyleSheet(f"""
                QSlider::groove:horizontal {{ height: 4px; background: #555; }}
                QSlider::handle:horizontal {{ background: {color}; width: 12px; margin: -4px 0; border-radius: 6px; }}
            """)
            sl.valueChanged.connect(lambda val, a=axis: self.update_advanced_mpr_clip(a))
            row_lay.addWidget(sl, 1)

            pct_lbl = QLabel("50%")
            pct_lbl.setFixedWidth(32)
            pct_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pct_lbl.setStyleSheet(f"color: {color}; font-size: 8pt;")
            row_lay.addWidget(pct_lbl)
            mc_lay.addLayout(row_lay)

            # Row 2: Show / Clip / Flip — full width so nothing is clipped horizontally
            toggles_lay = QHBoxLayout()
            toggles_lay.setSpacing(6)
            toggles_lay.setContentsMargins(48, 0, 0, 4)  # indent under slider

            chk_show = QCheckBox("Show")
            chk_show.setToolTip("Enable clipping along this axis")
            chk_show.stateChanged.connect(lambda state, a=axis: self.update_advanced_mpr_clip(a))
            toggles_lay.addWidget(chk_show)

            chk_clip = QCheckBox("Clip")
            chk_clip.setToolTip("Show colored indicator plane at clip position")
            chk_clip.stateChanged.connect(lambda state, a=axis: self.update_advanced_mpr_clip(a))
            toggles_lay.addWidget(chk_clip)

            btn_flip = QPushButton()
            btn_flip.setCheckable(True)
            btn_flip.setProperty("class", "icon-button")
            btn_flip.setIcon(_ui_icon("arrows-up-down.svg", btn_flip))
            btn_flip.setIconSize(QSize(14, 14))
            btn_flip.setFixedSize(28, 26)
            btn_flip.setCursor(Qt.PointingHandCursor)
            btn_flip.setToolTip("Flip clip direction (keep other half)")
            btn_flip.clicked.connect(lambda checked, a=axis: self.update_advanced_mpr_clip(a))
            toggles_lay.addWidget(btn_flip)
            toggles_lay.addStretch()
            mc_lay.addLayout(toggles_lay)

            self.adv_mpr_controls[axis] = {
                'slider': sl,
                'pct_label': pct_lbl,
                'show': chk_show,
                'clip': chk_clip,
                'flip': btn_flip
            }
            
        mpr_clip_group.setLayout(mc_lay)
        controls_panel.addWidget(mpr_clip_group)
        
        # --- D. Spacing Control Group ---
        spacing_group = QGroupBox("3D Spacing (X, Y, Z)")
        sp_lay = QVBoxLayout()
        sp_lay.setContentsMargins(5, 10, 5, 5)
        sp_lay.setSpacing(5)
        
        sp_grid = QGridLayout()
        
        self.spacing_x_spin = QDoubleSpinBox()
        self.spacing_x_spin.setRange(0.0001, 100.0)
        self.spacing_x_spin.setValue(self.spacing[0])
        self.spacing_x_spin.setDecimals(4)
        self.spacing_x_spin.setSingleStep(0.1)
        self.spacing_x_spin.setToolTip("X Spacing")
        
        self.spacing_y_spin = QDoubleSpinBox()
        self.spacing_y_spin.setRange(0.0001, 100.0)
        self.spacing_y_spin.setValue(self.spacing[1])
        self.spacing_y_spin.setDecimals(4)
        self.spacing_y_spin.setSingleStep(0.1)
        self.spacing_y_spin.setToolTip("Y Spacing")
        
        self.spacing_z_spin = QDoubleSpinBox()
        self.spacing_z_spin.setRange(0.0001, 100.0)
        self.spacing_z_spin.setValue(self.spacing[2])
        self.spacing_z_spin.setDecimals(4)
        self.spacing_z_spin.setSingleStep(0.1)
        self.spacing_z_spin.setToolTip("Z Spacing")
        
        sp_grid.addWidget(QLabel("X:"), 0, 0)
        sp_grid.addWidget(self.spacing_x_spin, 0, 1)
        sp_grid.addWidget(QLabel("Y:"), 0, 2)
        sp_grid.addWidget(self.spacing_y_spin, 0, 3)
        sp_grid.addWidget(QLabel("Z:"), 0, 4)
        sp_grid.addWidget(self.spacing_z_spin, 0, 5)
        sp_lay.addLayout(sp_grid)
        
        self.btn_apply_spacing = QPushButton("Apply Spacing")
        self.btn_apply_spacing.clicked.connect(self.apply_custom_spacing)
        sp_lay.addWidget(self.btn_apply_spacing)
        
        spacing_group.setLayout(sp_lay)
        controls_panel.addWidget(spacing_group)
        
        adv_main_layout.addLayout(controls_panel)
        
        # --- Lighting Controls Group (Dragonfly-like defaults) ---
        light_group = QGroupBox("Lighting Properties")
        light_layout = QFormLayout()
        light_layout.setContentsMargins(5, 5, 5, 5)

        # One-click look presets (ORS Dragonfly soft silver vs old plastic)
        look_row = QHBoxLayout()
        self.lighting_look_combo = QComboBox()
        self.lighting_look_combo.addItems(["Dragonfly", "Inspection", "Plastic", "Flat"])
        self.lighting_look_combo.setCurrentText("Dragonfly")
        self.lighting_look_combo.setToolTip(
            "Dragonfly: soft metallic (recommended — no plastic highlight caps)\n"
            "Inspection: flatter light for gap / surface metrology\n"
            "Plastic: old glossy demo look (strong specular pop)\n"
            "Flat: shading off (density TF only)"
        )
        look_row.addWidget(self.lighting_look_combo, 1)
        self.btn_apply_lighting_look = QPushButton("Apply Look")
        self.btn_apply_lighting_look.setToolTip(
            "Apply the selected lighting look (Phong + edge + LightKit)"
        )
        self.btn_apply_lighting_look.clicked.connect(self._on_apply_lighting_look_clicked)
        look_row.addWidget(self.btn_apply_lighting_look)
        light_layout.addRow("Look:", look_row)

        # Shade toggle
        self.shade_check = QCheckBox("Enable Shading")
        self.shade_check.setToolTip(
            "Phong shading (Dragonfly uses soft shade, not harsh specular)"
        )
        self.shade_check.setChecked(True)
        self.shade_check.stateChanged.connect(self.on_shade_toggled)
        light_layout.addRow("", self.shade_check)

        # Gradient Opacity (edge enhancement) — subtle default like Dragonfly surface
        self.gradient_opacity_slider = QSlider(Qt.Horizontal)
        self.gradient_opacity_slider.setRange(0, 100)
        self.gradient_opacity_slider.setValue(22)
        self.gradient_opacity_slider.setToolTip(
            "Gradient opacity — soft surface edges (Dragonfly-like).\n"
            "0 = off · 15–30 recommended · high values look noisy"
        )
        self.gradient_opacity_slider.valueChanged.connect(self.on_gradient_opacity_changed)
        light_layout.addRow("Edge:", self.gradient_opacity_slider)

        # Opacity
        self.view_3d_vol_opacity_slider = QSlider(Qt.Horizontal)
        self.view_3d_vol_opacity_slider.setRange(0, 100)
        self.view_3d_vol_opacity_slider.setValue(100)
        self.view_3d_vol_opacity_slider.valueChanged.connect(self.update_3d_lighting_properties)
        self.view_3d_vol_opacity_label = QLabel("1.00")
        self.view_3d_vol_opacity_label.setFixedWidth(30)
        self.view_3d_vol_opacity_slider.valueChanged.connect(
            lambda val: self.view_3d_vol_opacity_label.setText(f"{val/100:.2f}")
        )
        op_row = QHBoxLayout()
        op_row.addWidget(self.view_3d_vol_opacity_slider)
        op_row.addWidget(self.view_3d_vol_opacity_label)
        light_layout.addRow("Opacity:", op_row)

        # Ambient — higher = less black sides (Dragonfly solid)
        self.view_3d_ambient_slider = QSlider(Qt.Horizontal)
        self.view_3d_ambient_slider.setRange(0, 100)
        self.view_3d_ambient_slider.setValue(40)
        self.view_3d_ambient_slider.valueChanged.connect(self.update_3d_lighting_properties)
        self.view_3d_ambient_label = QLabel("0.40")
        self.view_3d_ambient_label.setFixedWidth(30)
        self.view_3d_ambient_slider.valueChanged.connect(
            lambda val: self.view_3d_ambient_label.setText(f"{val/100:.2f}")
        )
        amb_row = QHBoxLayout()
        amb_row.addWidget(self.view_3d_ambient_slider)
        amb_row.addWidget(self.view_3d_ambient_label)
        light_layout.addRow("Ambient:", amb_row)

        # Diffuse — medium form (not stage-spot)
        self.view_3d_diffuse_slider = QSlider(Qt.Horizontal)
        self.view_3d_diffuse_slider.setRange(0, 100)
        self.view_3d_diffuse_slider.setValue(58)
        self.view_3d_diffuse_slider.valueChanged.connect(self.update_3d_lighting_properties)
        self.view_3d_diffuse_label = QLabel("0.58")
        self.view_3d_diffuse_label.setFixedWidth(30)
        self.view_3d_diffuse_slider.valueChanged.connect(
            lambda val: self.view_3d_diffuse_label.setText(f"{val/100:.2f}")
        )
        diff_row = QHBoxLayout()
        diff_row.addWidget(self.view_3d_diffuse_slider)
        diff_row.addWidget(self.view_3d_diffuse_label)
        light_layout.addRow("Diffuse:", diff_row)

        # Specular — soft sheen only (high + high Power = plastic “raised caps”)
        self.view_3d_specular_slider = QSlider(Qt.Horizontal)
        self.view_3d_specular_slider.setRange(0, 100)
        self.view_3d_specular_slider.setValue(18)
        self.view_3d_specular_slider.valueChanged.connect(self.update_3d_lighting_properties)
        self.view_3d_specular_label = QLabel("0.18")
        self.view_3d_specular_label.setFixedWidth(30)
        self.view_3d_specular_slider.valueChanged.connect(
            lambda val: self.view_3d_specular_label.setText(f"{val/100:.2f}")
        )
        spec_row = QHBoxLayout()
        spec_row.addWidget(self.view_3d_specular_slider)
        spec_row.addWidget(self.view_3d_specular_label)
        light_layout.addRow("Specular:", spec_row)

        # Power — low = broad soft highlight (Dragonfly); high = sharp white spots
        self.view_3d_spec_power_slider = QSlider(Qt.Horizontal)
        self.view_3d_spec_power_slider.setRange(1, 128)
        self.view_3d_spec_power_slider.setValue(16)
        self.view_3d_spec_power_slider.valueChanged.connect(self.update_3d_lighting_properties)
        self.view_3d_spec_power_label = QLabel("16")
        self.view_3d_spec_power_label.setFixedWidth(30)
        self.view_3d_spec_power_slider.valueChanged.connect(
            lambda val: self.view_3d_spec_power_label.setText(str(val))
        )
        pow_row = QHBoxLayout()
        pow_row.addWidget(self.view_3d_spec_power_slider)
        pow_row.addWidget(self.view_3d_spec_power_label)
        light_layout.addRow("Power:", pow_row)

        light_group.setLayout(light_layout)
        adv_main_layout.addWidget(light_group)
        

        
        # Add stretch at bottom to push widgets up
        adv_main_layout.addStretch()
        
        # Wrap sidebar in scroll area.
        # Keep a stable minimum width so dense control rows are not clipped
        # when the vertical scrollbar steals horizontal space.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(360)
        scroll_content = QWidget()
        scroll_content.setMinimumWidth(340)
        scroll_content.setLayout(adv_main_layout)
        scroll.setWidget(scroll_content)

        sidebar_layout = QVBoxLayout(self.advanced_3d_controls)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.addWidget(scroll)

        # When hidden, collapse min size so it cannot inflate the 3D grid cell
        # (otherwise 2×2 becomes uneven: 3D pane wider than XY/XZ).
        self.advanced_3d_controls.setMinimumWidth(0)
        self.advanced_3d_controls.setMaximumWidth(0)
        self.advanced_3d_controls.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.advanced_3d_controls.setVisible(False)
        # Stretch 0: panel keeps its preferred/min width; VTK takes remaining space
        content_layout.addWidget(self.advanced_3d_controls, 0)

        widget.setLayout(layout)
        return widget
    
    def on_render_mode_changed(self, mode_text):
        """Switch volume rendering blend mode."""
        if not hasattr(self, 'volume_actor') or self.volume_actor is None:
            return
        mapper = self.volume_actor.GetMapper()

        mode_map = {
            "Composite":  0,   # vtkVolumeMapper::COMPOSITE_BLEND
            "MIP":        1,   # MAXIMUM_INTENSITY_BLEND
            "MinIP":      2,   # MINIMUM_INTENSITY_BLEND
            "Average":    3,   # AVERAGE_INTENSITY_BLEND
            "Additive":   4,   # ADDITIVE_BLEND
        }
        blend = mode_map.get(mode_text, 0)
        mapper.SetBlendMode(blend)
        self.view_3d_widget.GetRenderWindow().Render()
        print(f">>> Render mode: {mode_text} (blend={blend})")

    def on_gradient_opacity_changed(self, value):
        """Apply gradient-based opacity modulation for edge enhancement."""
        if not hasattr(self, 'volume_actor') or self.volume_actor is None:
            return
        prop = self.volume_actor.GetProperty()
        scale = value / 100.0

        if value == 0:
            prop.DisableGradientOpacityOn()
        else:
            prop.DisableGradientOpacityOff()
            grad = vtk.vtkPiecewiseFunction()
            # Soft Dragonfly-like edge (avoid harsh rim glow)
            grad.AddPoint(0, 0.0)
            grad.AddPoint(40 * (1.0 - scale * 0.5), 0.05 * scale)
            grad.AddPoint(120 * (1.0 - scale * 0.4), 0.35 * scale)
            grad.AddPoint(280, 0.75 * scale)
            grad.AddPoint(1000, 1.0)
            prop.SetGradientOpacity(grad)

        self.view_3d_widget.GetRenderWindow().Render()

    def on_shade_toggled(self, state):
        """Enable/disable Phong shading."""
        if not hasattr(self, 'volume_actor') or self.volume_actor is None:
            return
        prop = self.volume_actor.GetProperty()
        if state == Qt.Checked:
            prop.ShadeOn()
        else:
            prop.ShadeOff()
        self.view_3d_widget.GetRenderWindow().Render()

    def _on_apply_lighting_look_clicked(self):
        name = "Dragonfly"
        if hasattr(self, "lighting_look_combo"):
            name = self.lighting_look_combo.currentText()
        self.apply_lighting_look(name)

    def apply_lighting_look(self, name="Dragonfly", render=True):
        """Apply a named lighting look (Dragonfly / Inspection / Plastic / Flat).

        Dragonfly target: soft silver form light, mild edge OP, no plastic specular caps.
        """
        presets = getattr(self, "_lighting_look_presets", None) or {}
        p = presets.get(name) or presets.get("Dragonfly")
        if not p:
            return

        self._current_lighting_look = name
        self._vol_shadows_enabled = bool(p.get("vol_shadows", False))

        def _set_slider(slider, label, value, is_float=True):
            if slider is None:
                return
            slider.blockSignals(True)
            slider.setValue(int(value))
            slider.blockSignals(False)
            if label is not None:
                if is_float:
                    label.setText(f"{int(value) / 100.0:.2f}")
                else:
                    label.setText(str(int(value)))

        if hasattr(self, "shade_check"):
            self.shade_check.blockSignals(True)
            self.shade_check.setChecked(bool(p.get("shade", True)))
            self.shade_check.blockSignals(False)

        _set_slider(
            getattr(self, "view_3d_ambient_slider", None),
            getattr(self, "view_3d_ambient_label", None),
            p.get("ambient", 40),
        )
        _set_slider(
            getattr(self, "view_3d_diffuse_slider", None),
            getattr(self, "view_3d_diffuse_label", None),
            p.get("diffuse", 58),
        )
        _set_slider(
            getattr(self, "view_3d_specular_slider", None),
            getattr(self, "view_3d_specular_label", None),
            p.get("specular", 18),
        )
        _set_slider(
            getattr(self, "view_3d_spec_power_slider", None),
            getattr(self, "view_3d_spec_power_label", None),
            p.get("power", 16),
            is_float=False,
        )
        _set_slider(
            getattr(self, "view_3d_vol_opacity_slider", None),
            getattr(self, "view_3d_vol_opacity_label", None),
            p.get("opacity", 100),
        )
        if hasattr(self, "gradient_opacity_slider"):
            self.gradient_opacity_slider.blockSignals(True)
            self.gradient_opacity_slider.setValue(int(p.get("edge", 22)))
            self.gradient_opacity_slider.blockSignals(False)

        # Soft LightKit (key/fill balance)
        lk = getattr(self, "light_kit", None)
        if lk is not None:
            try:
                lk.SetKeyLightIntensity(float(p.get("key_intensity", 0.72)))
                lk.SetKeyToFillRatio(float(p.get("key_fill", 1.8)))
                lk.SetKeyLightWarmth(float(p.get("key_warmth", 0.55)))
            except Exception:
                pass

        # Volumetric shadows amplify “raised blob” look — off for Dragonfly/Inspection
        mapper = getattr(self, "_volume_mapper", None)
        if mapper is not None and hasattr(mapper, "SetShadows"):
            try:
                mapper.SetShadows(1 if self._vol_shadows_enabled else 0)
            except Exception:
                pass

        if hasattr(self, "lighting_look_combo"):
            self.lighting_look_combo.blockSignals(True)
            idx = self.lighting_look_combo.findText(name)
            if idx >= 0:
                self.lighting_look_combo.setCurrentIndex(idx)
            self.lighting_look_combo.blockSignals(False)

        if getattr(self, "volume_actor", None) is not None:
            prop = self.volume_actor.GetProperty()
            if p.get("shade", True):
                prop.ShadeOn()
            else:
                prop.ShadeOff()
            prop.SetAmbient(float(p.get("ambient", 40)) / 100.0)
            prop.SetDiffuse(float(p.get("diffuse", 58)) / 100.0)
            prop.SetSpecular(float(p.get("specular", 18)) / 100.0)
            prop.SetSpecularPower(float(p.get("power", 16)))
            self.apply_transfer_function(prop)
            # Re-apply gradient opacity from slider
            if hasattr(self, "on_gradient_opacity_changed"):
                self.on_gradient_opacity_changed(
                    self.gradient_opacity_slider.value()
                    if hasattr(self, "gradient_opacity_slider")
                    else 0
                )
            if render and hasattr(self, "view_3d_widget") and self.view_3d_widget:
                try:
                    self.view_3d_widget.GetRenderWindow().Render()
                except Exception:
                    pass
        print(
            f"[3D LOOK] {name}: amb={p.get('ambient')} dif={p.get('diffuse')} "
            f"spec={p.get('specular')} pow={p.get('power')} edge={p.get('edge')} "
            f"shadows={p.get('vol_shadows')}"
        )

    # ────────── Dragonfly-style LUT Presets ──────────

    _TF_PRESETS = {
        "Default 3D": {
            "opacity": [(0.0, 0.0), (0.2, 0.0), (0.5, 0.3), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 0)), (0.3, (100, 100, 90)), (0.6, (200, 200, 180)),
                (1.0, (255, 255, 240)),
            ],
        },
        "Discrete 24": {
            "opacity": [(0.0, 0.0), (1.0, 1.0)],
            "colors": [
                (0.0, (255, 0, 0)), (0.16, (255, 255, 0)), (0.33, (0, 255, 0)),
                (0.5, (0, 255, 255)), (0.66, (0, 0, 255)), (0.83, (255, 0, 255)),
                (1.0, (255, 0, 0)),
            ],
        },
        "Discrete 64": {
            "opacity": [(0.0, 0.0), (1.0, 1.0)],
            "colors": [
                (0.0, (255, 0, 0)), (0.2, (255, 255, 0)), (0.4, (0, 255, 0)),
                (0.6, (0, 255, 255)), (0.8, (0, 0, 255)), (1.0, (255, 0, 255)),
            ],
        },
        "Electronics": {
            "opacity": [(0.0, 0.0), (0.2, 0.0), (0.5, 0.4), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 0)), (0.3, (80, 40, 20)), (0.6, (180, 100, 50)),
                (0.8, (230, 180, 120)), (1.0, (255, 255, 255)),
            ],
        },
        "Entomology": {
            "opacity": [(0.0, 0.0), (0.15, 0.0), (0.5, 0.5), (1.0, 0.95)],
            "colors": [
                (0.0, (0, 0, 0)), (0.3, (60, 30, 0)), (0.6, (180, 100, 20)),
                (0.8, (230, 160, 50)), (1.0, (255, 240, 200)),
            ],
        },
        "Fire": {
            "opacity": [(0.0, 0.0), (0.2, 0.05), (0.5, 0.4), (0.8, 0.8), (1.0, 1.0)],
            "colors": [
                (0.0, (0, 0, 0)), (0.33, (255, 0, 0)), (0.66, (255, 255, 0)),
                (1.0, (255, 255, 255)),
            ],
        },
        "Grayscale Solid": {
            "opacity": [(0.0, 0.0), (0.1, 0.1), (0.5, 0.5), (1.0, 1.0)],
            "colors": [
                (0.0, (64, 64, 64)), (0.5, (160, 160, 160)), (1.0, (255, 255, 255)),
            ],
        },
        "Grayscale Solid2": {
            "opacity": [(0.0, 0.0), (0.1, 0.2), (0.5, 0.6), (1.0, 1.0)],
            "colors": [
                (0.0, (128, 128, 128)), (0.5, (192, 192, 192)), (1.0, (255, 255, 255)),
            ],
        },
        "Green": {
            "opacity": [(0.0, 0.0), (0.2, 0.0), (0.5, 0.3), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 0)), (0.4, (0, 128, 0)), (0.7, (0, 200, 50)),
                (1.0, (200, 255, 200)),
            ],
        },
        "Grayscale": {
            "opacity": [(0.0, 0.0), (0.15, 0.0), (0.3, 0.08), (0.5, 0.25), (0.75, 0.6), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 0)), (0.25, (64, 64, 64)), (0.5, (128, 128, 128)),
                (0.75, (192, 192, 192)), (1.0, (255, 255, 255)),
            ],
        },
        "Steel": {
            "opacity": [(0.0, 0.0), (0.12, 0.0), (0.3, 0.06), (0.5, 0.22), (0.75, 0.6), (1.0, 0.92)],
            "colors": [
                (0.0, (0, 0, 0)), (0.15, (10, 18, 35)), (0.35, (40, 70, 110)),
                (0.55, (100, 140, 180)), (0.75, (170, 195, 215)),
                (0.9, (215, 225, 240)), (1.0, (240, 245, 255)),
            ],
        },
        "Steel2": {
            "opacity": [(0.0, 0.0), (0.1, 0.0), (0.25, 0.04), (0.45, 0.2), (0.7, 0.55), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 5)), (0.2, (25, 35, 55)), (0.4, (65, 90, 125)),
                (0.6, (120, 150, 180)), (0.8, (185, 200, 220)),
                (1.0, (230, 238, 248)),
            ],
        },
        "Temperature": {
            "opacity": [(0.0, 0.0), (0.1, 0.0), (0.25, 0.05), (0.5, 0.25), (0.75, 0.6), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 0)), (0.15, (10, 0, 40)), (0.3, (60, 0, 120)),
                (0.45, (180, 0, 80)), (0.6, (230, 60, 10)), (0.75, (250, 170, 0)),
                (0.9, (255, 240, 100)), (1.0, (255, 255, 220)),
            ],
        },
        "Thermal": {
            "opacity": [(0.0, 0.0), (0.1, 0.0), (0.25, 0.04), (0.5, 0.2), (0.75, 0.55), (1.0, 0.88)],
            "colors": [
                (0.0, (0, 0, 0)), (0.2, (20, 0, 80)), (0.35, (80, 0, 160)),
                (0.5, (200, 30, 30)), (0.65, (240, 120, 0)),
                (0.8, (255, 220, 50)), (1.0, (255, 255, 200)),
            ],
        },
        "Vegetal": {
            "opacity": [(0.0, 0.0), (0.12, 0.0), (0.3, 0.06), (0.5, 0.2), (0.75, 0.55), (1.0, 0.85)],
            "colors": [
                (0.0, (0, 0, 0)), (0.15, (5, 20, 5)), (0.3, (15, 65, 15)),
                (0.5, (40, 140, 30)), (0.7, (100, 200, 60)),
                (0.85, (180, 235, 120)), (1.0, (230, 255, 200)),
            ],
        },
        "Vegetal2": {
            "opacity": [(0.0, 0.0), (0.1, 0.0), (0.25, 0.05), (0.5, 0.22), (0.75, 0.58), (1.0, 0.88)],
            "colors": [
                (0.0, (0, 0, 0)), (0.2, (10, 30, 8)), (0.4, (30, 100, 25)),
                (0.6, (80, 170, 50)), (0.8, (160, 220, 100)),
                (1.0, (220, 250, 180)),
            ],
        },
        "Warm Metal": {
            "opacity": [(0.0, 0.0), (0.1, 0.0), (0.25, 0.03), (0.4, 0.12), (0.6, 0.35), (0.8, 0.65), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 0)), (0.15, (20, 5, 0)), (0.3, (80, 25, 0)),
                (0.45, (160, 60, 5)), (0.6, (220, 120, 15)),
                (0.75, (245, 190, 50)), (0.9, (255, 230, 120)),
                (1.0, (255, 250, 210)),
            ],
        },
        "White": {
            "opacity": [(0.0, 0.0), (0.15, 0.0), (0.3, 0.1), (0.5, 0.35), (0.75, 0.7), (1.0, 1.0)],
            "colors": [
                (0.0, (0, 0, 0)), (0.3, (80, 80, 80)), (0.5, (160, 160, 160)),
                (0.7, (220, 220, 220)), (1.0, (255, 255, 255)),
            ],
        },
        "Wood4": {
            "opacity": [(0.0, 0.0), (0.1, 0.0), (0.25, 0.04), (0.45, 0.18), (0.65, 0.45), (0.85, 0.75), (1.0, 0.92)],
            "colors": [
                (0.0, (0, 0, 0)), (0.15, (25, 12, 5)), (0.3, (80, 45, 15)),
                (0.5, (160, 100, 35)), (0.65, (200, 145, 60)),
                (0.8, (230, 190, 100)), (0.95, (250, 225, 160)),
                (1.0, (255, 245, 210)),
            ],
        },
        "Yellow": {
            "opacity": [(0.0, 0.0), (0.12, 0.0), (0.3, 0.06), (0.5, 0.22), (0.75, 0.58), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 0)), (0.2, (30, 25, 0)), (0.4, (120, 100, 0)),
                (0.6, (200, 180, 10)), (0.8, (240, 225, 60)),
                (1.0, (255, 255, 150)),
            ],
        },
    }

    def apply_tf_preset(self, preset_name):
        """Apply a named LUT / transfer function preset (Dragonfly-style)."""
        if preset_name == "Custom" or preset_name not in self._TF_PRESETS:
            return

        preset = self._TF_PRESETS[preset_name]

        # Removed updating opacity control points from preset (keep existing TF curve)

        # Update color gradient stops
        self.color_grad_widget.color_stops = [
            (pos, QColor(r, g, b)) for pos, (r, g, b) in preset["colors"]
        ]
        self.color_grad_widget.update()

        # Update the TF widget bottom ramp to show the LUT colors
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_lut_stops(self.color_grad_widget.color_stops)

        # Apply to volume
        if hasattr(self, 'volume_actor') and self.volume_actor is not None:
            self.update_transfer_function()

    def _make_lut_icon(self, preset_name, w=60, h=16):
        """Generate a QIcon with a gradient pixmap for the given LUT preset."""
        pixmap = QPixmap(w, h)
        preset = self._TF_PRESETS.get(preset_name)
        if preset is None:
            pixmap.fill(QColor(50, 50, 50))
            return QIcon(pixmap)
        painter = QPainter(pixmap)
        from PyQt5.QtGui import QLinearGradient
        grad = QLinearGradient(0, 0, w, 0)
        for pos, (r, g, b) in preset["colors"]:
            grad.setColorAt(pos, QColor(r, g, b))
        painter.fillRect(0, 0, w, h, grad)
        painter.end()
        return QIcon(pixmap)

    def update_transfer_function(self):
        """Lightweight TF update without rebuilding the volume actor."""
        if self.volume_actor:
            self.apply_transfer_function(self.volume_actor.GetProperty())
            self.view_3d_widget.GetRenderWindow().Render()

    def apply_custom_spacing(self):
        """Apply custom user-specified spacing values to the 3D volume rendering."""
        if self.volume_data is None:
            QMessageBox.warning(self, "Warning", "No volume data loaded.")
            return
            
        x_sp = self.spacing_x_spin.value()
        y_sp = self.spacing_y_spin.value()
        z_sp = self.spacing_z_spin.value()
        
        self.custom_spacing = [x_sp, y_sp, z_sp]
        print(f">>> Applying custom spacing: X={x_sp}, Y={y_sp}, Z={z_sp}")
        
        # Trigger full 3D re-render
        self.render_3d()

    # --- TF / WL Range Handlers ---

    def _on_tf_range_changed(self, norm_min, norm_max):
        """Called when user drags yellow Min/Max bars on the TF widget (0..1)."""
        data_min = getattr(self, '_volume_data_min', float(self.view_3d_min_spin.minimum()))
        data_max = getattr(self, '_volume_data_max', float(self.view_3d_max_spin.maximum()))
        rng = data_max - data_min
        if rng <= 0:
            rng = 1.0
        new_min = int(round(data_min + norm_min * rng))
        new_max = int(round(data_min + norm_max * rng))
        self.view_3d_min_spin.blockSignals(True)
        self.view_3d_max_spin.blockSignals(True)
        self.view_3d_min_spin.setValue(new_min)
        self.view_3d_max_spin.setValue(new_max)
        self.view_3d_min_spin.blockSignals(False)
        self.view_3d_max_spin.blockSignals(False)
        self._update_wl_readout()
        self.update_transfer_function()

    def _on_wl_spin_changed(self):
        """Called when user edits Min/Max spinboxes."""
        v_min = self.view_3d_min_spin.value()
        v_max = self.view_3d_max_spin.value()
        data_min = getattr(self, '_volume_data_min', float(self.view_3d_min_spin.minimum()))
        data_max = getattr(self, '_volume_data_max', float(self.view_3d_max_spin.maximum()))
        rng = data_max - data_min
        if rng > 0 and hasattr(self, 'tf_widget'):
            self.tf_widget.set_range((v_min - data_min) / rng,
                                     (v_max - data_min) / rng)
        self._update_wl_readout()
        self.update_transfer_function()

    # ── MPR Window/Level (right sidebar, linked to all 3 planes) ──────────

    def _on_mpr_wl_hist_changed(self, v_min, v_max):
        """Histogram Min/Max markers dragged → update spins + apply to MPR."""
        if not hasattr(self, 'mpr_wl_min_spin'):
            return
        self.mpr_wl_min_spin.blockSignals(True)
        self.mpr_wl_max_spin.blockSignals(True)
        self.mpr_wl_min_spin.setValue(int(v_min))
        self.mpr_wl_max_spin.setValue(int(v_max))
        self.mpr_wl_min_spin.blockSignals(False)
        self.mpr_wl_max_spin.blockSignals(False)
        self._apply_mpr_window_level(int(v_min), int(v_max))

    def _on_mpr_wl_spin_changed(self):
        """Min/Max spinboxes edited → sync histogram + apply to MPR."""
        if not hasattr(self, 'mpr_wl_min_spin'):
            return
        v_min = self.mpr_wl_min_spin.value()
        v_max = self.mpr_wl_max_spin.value()
        if v_max <= v_min:
            # Keep a non-zero window; nudge the spin that was not the sender
            v_max = v_min + 1
            self.mpr_wl_max_spin.blockSignals(True)
            self.mpr_wl_max_spin.setValue(v_max)
            self.mpr_wl_max_spin.blockSignals(False)
        if hasattr(self, 'mpr_hist_widget'):
            self.mpr_hist_widget.blockSignals(True)
            self.mpr_hist_widget.set_range(v_min, v_max)
            self.mpr_hist_widget.blockSignals(False)
        self._apply_mpr_window_level(v_min, v_max)

    def _update_mpr_wl_readout(self, v_min=None, v_max=None):
        if not hasattr(self, 'mpr_wl_info_label'):
            return
        if v_min is None:
            v_min = self.mpr_wl_min_spin.value() if hasattr(self, 'mpr_wl_min_spin') else 0
        if v_max is None:
            v_max = self.mpr_wl_max_spin.value() if hasattr(self, 'mpr_wl_max_spin') else 65535
        w = max(1, int(v_max) - int(v_min))
        l = (float(v_max) + float(v_min)) / 2.0
        self.mpr_wl_info_label.setText(f"W: {w}   L: {l:.1f}")

    def _mpr_needs_baked_overlay(self, orientation=None):
        """True when MPR displays CT+mask as pre-baked RGB (W/L cannot be a property tweak)."""
        oris = (orientation,) if orientation else ('axial', 'coronal', 'sagittal')
        has_mask = (
            getattr(self, 'class1_data', None) is not None
            or getattr(self, 'class2_data', None) is not None
            or getattr(self, 'segmentation_data', None) is not None
        )
        if not has_mask:
            return False
        for o in oris:
            try:
                c1 = bool(getattr(self, f'{o}_overlay_c1').isChecked())
                c2 = bool(getattr(self, f'{o}_overlay_c2').isChecked())
                op = int(getattr(self, f'{o}_opacity_slider').value())
            except Exception:
                continue
            if (c1 or c2) and op > 0:
                return True
        return False

    def _apply_mpr_window_level(self, v_min, v_max):
        """Push Window/Level to all three MPR planes (linked).

        Fast path: only update vtkImageProperty on cached *grayscale* actors.
        When C1/C2 overlay is baked into RGB pixels, must re-render slices —
        applying data-range W/L (e.g. 15k–51k) onto 0–255 RGB blacks out MPR.
        """
        v_min = int(v_min)
        v_max = int(v_max)
        if v_max <= v_min:
            v_max = v_min + 1
        window = float(v_max - v_min)
        level = (float(v_max) + float(v_min)) / 2.0

        for orientation in ('axial', 'coronal', 'sagittal'):
            self.window_level[orientation] = (window, level)

        self._update_mpr_wl_readout(v_min, v_max)

        cache = getattr(self, '_cached_slice_actors', None) or {}

        # Baked RGB overlay mode → full re-compose with new W/L
        if self._mpr_needs_baked_overlay() and self.volume_data is not None:
            for orientation in ('axial', 'coronal', 'sagittal'):
                self.render_slice(orientation, preserve_camera=True)
            return

        # Also re-render if any cached actor is still tagged as rgb_overlay
        # (e.g. toggles just turned off but cache not yet rebuilt)
        if any(
            (cache.get(o) or {}).get('display_mode') == 'rgb_overlay'
            for o in ('axial', 'coronal', 'sagittal')
        ) and self.volume_data is not None:
            for orientation in ('axial', 'coronal', 'sagittal'):
                self.render_slice(orientation, preserve_camera=True)
            return

        any_fast = False
        for orientation in ('axial', 'coronal', 'sagittal'):
            entry = cache.get(orientation)
            if not entry or 'image_actor' not in entry:
                continue
            if entry.get('display_mode') == 'rgb_overlay':
                continue
            try:
                prop = entry['image_actor'].GetProperty()
                prop.SetColorWindow(window)
                prop.SetColorLevel(level)
                widget = getattr(self, f'{orientation}_widget', None)
                if widget is not None:
                    rw = widget.GetRenderWindow()
                    if rw is not None:
                        rw.Render()
                any_fast = True
            except Exception:
                pass

        # Actors not cached yet (volume just loading / first paint) → full render
        if not any_fast and self.volume_data is not None:
            for orientation in ('axial', 'coronal', 'sagittal'):
                try:
                    self.render_slice(orientation, preserve_camera=True)
                except Exception:
                    pass

    def _sync_mpr_wl_ui(self, v_min, v_max, data_min=None, data_max=None):
        """Update MPR W/L controls without re-emitting apply (used on volume load)."""
        v_min, v_max = int(v_min), int(v_max)
        if data_min is None:
            data_min = getattr(self, '_volume_data_min', v_min)
        if data_max is None:
            data_max = getattr(self, '_volume_data_max', v_max)
        imin, imax = int(data_min), int(data_max)
        if imax <= imin:
            imax = imin + 1

        if hasattr(self, 'mpr_wl_min_spin'):
            self.mpr_wl_min_spin.blockSignals(True)
            self.mpr_wl_max_spin.blockSignals(True)
            self.mpr_wl_min_spin.setRange(imin, imax)
            self.mpr_wl_max_spin.setRange(imin, imax)
            self.mpr_wl_min_spin.setValue(max(imin, min(v_min, imax)))
            self.mpr_wl_max_spin.setValue(max(imin, min(v_max, imax)))
            self.mpr_wl_min_spin.blockSignals(False)
            self.mpr_wl_max_spin.blockSignals(False)

        if hasattr(self, 'mpr_hist_widget'):
            self.mpr_hist_widget.blockSignals(True)
            self.mpr_hist_widget.set_data_range(imin, imax)
            self.mpr_hist_widget.set_range(v_min, v_max)
            self.mpr_hist_widget.blockSignals(False)

        self._update_mpr_wl_readout(v_min, v_max)

    def _reset_mpr_wl_to_data_range(self):
        """Reset MPR W/L selected range to full volume data range."""
        dmin = int(getattr(self, '_volume_data_min', 0))
        dmax = int(getattr(self, '_volume_data_max', 65535))
        if dmax <= dmin:
            dmax = dmin + 1
        if hasattr(self, 'mpr_wl_min_spin'):
            self.mpr_wl_min_spin.blockSignals(True)
            self.mpr_wl_max_spin.blockSignals(True)
            self.mpr_wl_min_spin.setValue(dmin)
            self.mpr_wl_max_spin.setValue(dmax)
            self.mpr_wl_min_spin.blockSignals(False)
            self.mpr_wl_max_spin.blockSignals(False)
        if hasattr(self, 'mpr_hist_widget'):
            self.mpr_hist_widget.blockSignals(True)
            self.mpr_hist_widget.set_range(dmin, dmax)
            self.mpr_hist_widget.blockSignals(False)
        self._apply_mpr_window_level(dmin, dmax)

    def _update_wl_readout(self):
        """Update the W/L readout label."""
        v_min = self.view_3d_min_spin.value()
        v_max = self.view_3d_max_spin.value()
        w = v_max - v_min
        l = (v_max + v_min) / 2.0
        if hasattr(self, '_wl_info_label'):
            self._wl_info_label.setText(f"W: {w}   L: {l:.1f}")

    def _set_data_range_ui(self, data_min, data_max):
        """Sync Plotted range / Data range widgets after volume load."""
        self._volume_data_min = float(data_min)
        self._volume_data_max = float(data_max)
        if hasattr(self, '_data_range_min_spin'):
            self._data_range_min_spin.blockSignals(True)
            self._data_range_max_spin.blockSignals(True)
            self._data_range_min_spin.setValue(self._volume_data_min)
            self._data_range_max_spin.setValue(self._volume_data_max)
            self._data_range_min_spin.blockSignals(False)
            self._data_range_max_spin.blockSignals(False)
        if hasattr(self, '_data_extent_label'):
            self._data_extent_label.setText(
                f"{self._volume_data_min:.2f}  —  {self._volume_data_max:.2f}"
            )
        # Spin box hard limits track full data extent
        imin, imax = int(data_min), int(data_max)
        if imax <= imin:
            imax = imin + 1
        self.view_3d_min_spin.setRange(imin, imax)
        self.view_3d_max_spin.setRange(imin, imax)

    def _reset_wl_to_data_range(self):
        """Reset selected range to full data range (Dragonfly Reset)."""
        dmin = int(getattr(self, '_volume_data_min', 0))
        dmax = int(getattr(self, '_volume_data_max', 65535))
        self.view_3d_min_spin.blockSignals(True)
        self.view_3d_max_spin.blockSignals(True)
        self.view_3d_min_spin.setValue(dmin)
        self.view_3d_max_spin.setValue(dmax)
        self.view_3d_min_spin.blockSignals(False)
        self.view_3d_max_spin.blockSignals(False)
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_range(0.0, 1.0)
        self._update_wl_readout()
        self.update_transfer_function()

    def _set_color_invert(self, invert):
        self._color_invert = bool(invert)
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_color_invert(self._color_invert)
        # Refresh icon selected state
        if hasattr(self, 'btn_cmap_normal'):
            self.btn_cmap_normal.setIcon(make_color_map_icon("normal", selected=not invert))
            self.btn_cmap_invert.setIcon(make_color_map_icon("invert", selected=invert))
        self.update_transfer_function()

    def _set_opacity_shape(self, shape):
        self._opacity_shape = shape if shape in OPACITY_SHAPES else "linear"
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_opacity_shape(self._opacity_shape)
        # Highlight selected icon
        if hasattr(self, '_opacity_shape_btns'):
            for s, btn in self._opacity_shape_btns.items():
                btn.setIcon(make_opacity_shape_icon(s, selected=(s == self._opacity_shape)))
                btn.setChecked(s == self._opacity_shape)
        self.update_transfer_function()

    def _on_wl_opacity_changed(self, val):
        scale = val / 100.0
        if hasattr(self, 'wl_opacity_value_lbl'):
            self.wl_opacity_value_lbl.setText(f"{scale:.2f}")
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_opacity_scale(scale)
        # Keep lighting-panel opacity in sync if present
        if hasattr(self, 'view_3d_vol_opacity_slider'):
            self.view_3d_vol_opacity_slider.blockSignals(True)
            self.view_3d_vol_opacity_slider.setValue(val)
            self.view_3d_vol_opacity_slider.blockSignals(False)
            if hasattr(self, 'view_3d_vol_opacity_label'):
                self.view_3d_vol_opacity_label.setText(f"{scale:.2f}")
        self.update_transfer_function()

    def _on_wl_gamma_changed(self, val):
        gamma = val / 100.0
        if hasattr(self, 'wl_gamma_spin'):
            self.wl_gamma_spin.blockSignals(True)
            self.wl_gamma_spin.setValue(gamma)
            self.wl_gamma_spin.blockSignals(False)
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_opacity_gamma(gamma)
        self.update_transfer_function()

    def _on_wl_gamma_spin_changed(self, gamma):
        if hasattr(self, 'wl_gamma_slider'):
            self.wl_gamma_slider.blockSignals(True)
            self.wl_gamma_slider.setValue(int(round(gamma * 100)))
            self.wl_gamma_slider.blockSignals(False)
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_opacity_gamma(gamma)
        self.update_transfer_function()




    def eventFilter(self, obj, event):
        # 3D pane / container: focus ring + authoritative scroll-zoom
        is_3d = (
            obj is getattr(self, 'view_3d_widget', None)
            or obj is getattr(self, '_vtk_3d_container', None)
        )
        if is_3d:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._set_active_grid_pane('view_3d')
                if obj is getattr(self, 'view_3d_widget', None):
                    try:
                        self.view_3d_widget.setFocus(Qt.MouseFocusReason)
                    except Exception:
                        pass
                # Qt backup path for clip-face grab (if VTK style misses the press)
                if getattr(self, '_df_clip_enabled', False) and self.volume_data is not None:
                    if self._df_clip_3d_try_grab_at_qt_pos(event.pos()):
                        return True
            elif event.type() == QEvent.MouseMove:
                if getattr(self, '_df_clip_3d_drag', None) and (event.buttons() & Qt.LeftButton):
                    self._df_clip_3d_drag_at_qt_pos(event.pos())
                    return True
                if (
                    getattr(self, '_df_clip_enabled', False)
                    and self.volume_data is not None
                    and not (event.buttons() & Qt.LeftButton)
                ):
                    self._df_clip_3d_hover_at_qt_pos(event.pos())
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if getattr(self, '_df_clip_3d_drag', None):
                    self._df_clip_3d_on_left_up()
                    style = getattr(self, '_3d_interactor_style', None)
                    if style is not None:
                        style._df_clip_dragging = False
                    return True
            elif event.type() == QEvent.Wheel:
                if self.volume_data is None:
                    return super().eventFilter(obj, event)
                # Prefer pixelDelta (trackpad / high-res) for continuous feel;
                # fall back to angleDelta (classic mouse, usually ±120 per notch).
                pixel = event.pixelDelta().y()
                angle = event.angleDelta().y()
                if angle == 0:
                    angle = event.angleDelta().x()
                if pixel == 0 and angle == 0:
                    pixel = event.pixelDelta().x()
                if pixel == 0 and angle == 0:
                    return super().eventFilter(obj, event)

                if pixel != 0:
                    # ~40 px ≈ one mild notch; sign = zoom direction
                    strength = abs(pixel) / 40.0
                    zoom_in = pixel > 0
                else:
                    # One wheel notch (120°) → strength 1.0
                    strength = abs(angle) / 120.0
                    zoom_in = angle > 0

                # Clamp so a single event never jumps wildly
                strength = max(0.15, min(strength, 3.0))
                # 3D: dolly about orbit pivot only (no zoom-to-cursor pan).
                # Cursor-based shift was re-enabled briefly and broke Track/pan feel.
                self._dragonfly_3d_zoom(zoom_in=zoom_in, strength=strength, display_xy=None)
                return True
            return super().eventFilter(obj, event)


        # Handle mouse events for pixel value and crosshair
        if obj in [self.axial_widget, self.coronal_widget, self.sagittal_widget]:
            orientation = None
            for ori in ['axial', 'coronal', 'sagittal']:
                if obj == getattr(self, f'{ori}_widget'):
                    orientation = ori
                    break

            if event.type() == QEvent.MouseMove:
                if self.volume_data is not None:
                    self.update_pixel_value(orientation, event.pos())
                    # MPR pan (middle or Shift+LMB)
                    if getattr(self, "_mpr_pan", None) and (
                        (event.buttons() & Qt.MiddleButton)
                        or ((event.buttons() & Qt.LeftButton) and (event.modifiers() & Qt.ShiftModifier))
                        or ((event.buttons() & Qt.LeftButton) and getattr(self, "_mpr_pan", None))
                    ):
                        self._mpr_pan_move(orientation, event.pos())
                        return True
                    # Clip-box drag (Dragonfly handles on MPR)
                    if getattr(self, '_df_clip_drag', None) and (event.buttons() & Qt.LeftButton):
                        self._df_clip_handle_drag(event.pos(), orientation)
                        return True
                    # Oblique rotation drag
                    if getattr(self, '_oblique_dragging', None) and self.crosshair_enabled and (event.buttons() & Qt.LeftButton):
                        self._handle_oblique_rotate(event.pos(), orientation)
                        return True
                    # Drag to rotate camera roll
                    elif getattr(self, 'rotate_mode', False) and (event.buttons() & Qt.LeftButton):
                        self.handle_rotation_drag(orientation, event.pos())
                    # Drag crosshair only while actively grabbing center/axis
                    elif self.crosshair_enabled and getattr(self, '_is_dragging_crosshair', False) and (event.buttons() & Qt.LeftButton):
                        mode = getattr(self, '_crosshair_drag_mode', 'center') or 'center'
                        self.handle_crosshair_click(
                            orientation, event.pos(), mode=mode, preview_only=True
                        )
                        return True

                    # Hover: clip box → oblique → crosshair → open-hand (pan available)
                    if not (event.buttons() & (Qt.LeftButton | Qt.MiddleButton)):
                        widget = getattr(self, f'{orientation}_widget')
                        if getattr(self, '_df_clip_enabled', False):
                            clip_hit = self._get_df_clip_hit(event.pos(), orientation)
                            prev = getattr(self, '_df_clip_hover', None)
                            if self._df_clip_hit_key(clip_hit) != self._df_clip_hit_key(prev):
                                self._df_clip_hover = clip_hit
                                self._df_clip_update_mpr_overlay(orientation)
                                widget.GetRenderWindow().Render()
                            if clip_hit is not None:
                                widget.setCursor(self._df_clip_cursor_for_hit(clip_hit))
                                return False
                            elif prev is not None:
                                self._df_clip_hover = None
                                self._df_clip_update_mpr_overlay(orientation)
                                widget.GetRenderWindow().Render()
                        if self.crosshair_enabled:
                            obl_hit = self._get_oblique_hit(event.pos(), orientation)
                            if obl_hit != getattr(self, '_oblique_hover_handle', None):
                                self._oblique_hover_handle = obl_hit
                                if obl_hit:
                                    self._crosshair_hover = None
                                self.render_slice(orientation, preserve_camera=True)
                            if obl_hit:
                                widget.setCursor(Qt.SizeAllCursor)
                            else:
                                ch_hit = self._get_crosshair_hit(event.pos(), orientation)
                                if ch_hit != getattr(self, '_crosshair_hover', None):
                                    self._crosshair_hover = ch_hit
                                    self.update_2d_crosshair(orientation)
                                cur = self._cursor_for_crosshair_hit(ch_hit)
                                if cur is not None:
                                    widget.setCursor(cur)
                                else:
                                    # Empty space: pan available (middle / shift+lmb)
                                    widget.setCursor(Qt.OpenHandCursor)
                        else:
                            widget.setCursor(Qt.OpenHandCursor)
            
            elif event.type() == QEvent.MouseButtonPress:
                # Middle button → pan
                if event.button() == Qt.MiddleButton and self.volume_data is not None:
                    self._set_active_grid_pane(orientation)
                    self._mpr_pan_begin(orientation, event.pos())
                    return True
                if event.button() == Qt.LeftButton:
                    # Dragonfly-style: focus ring on the pane being interacted with
                    self._set_active_grid_pane(orientation)
                    if self.volume_data is not None:
                        # Shift+LMB → pan (unified with middle)
                        if event.modifiers() & Qt.ShiftModifier:
                            self._mpr_pan_begin(orientation, event.pos())
                            return True
                        # Clip-box handles first
                        if getattr(self, '_df_clip_enabled', False):
                            clip_hit = self._get_df_clip_hit(event.pos(), orientation)
                            if clip_hit is not None:
                                self._df_clip_begin_drag(orientation, event.pos(), clip_hit)
                                return True
                        if getattr(self, 'rotate_mode', False):
                            self.last_mouse_pos = event.pos()
                        elif self.crosshair_enabled:
                            if getattr(self, '_oblique_hover_handle', None) and getattr(self, '_oblique_hover_handle')[0] == orientation:
                                self._oblique_dragging = self._oblique_hover_handle
                                return True
                            ch_hit = self._get_crosshair_hit(event.pos(), orientation)
                            if ch_hit is None:
                                # P0: click-to-place crosshair on empty space
                                self._crosshair_drag_mode = 'place'
                                self._crosshair_hover = (orientation, 'center')
                                self._is_dragging_crosshair = True
                                self.handle_crosshair_click(
                                    orientation, event.pos(), mode='place', preview_only=False
                                )
                                return True
                            self._crosshair_drag_mode = ch_hit[1]
                            self._crosshair_hover = ch_hit
                            self._is_dragging_crosshair = True
                            self.handle_crosshair_click(
                                orientation, event.pos(), mode=ch_hit[1], preview_only=False
                            )
                            self.update_2d_crosshair(orientation)
                            return True

            elif event.type() == QEvent.MouseButtonDblClick:
                if event.button() == Qt.LeftButton and self.volume_data is not None:
                    # P1: double-click → fit pane
                    self._set_active_grid_pane(orientation)
                    self.reset_view(orientation)
                    return True
            
            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.MiddleButton:
                    if getattr(self, "_mpr_pan", None):
                        self._mpr_pan_end(orientation)
                        return True
                if event.button() == Qt.LeftButton:
                    self.last_mouse_pos = None

                    if getattr(self, "_mpr_pan", None):
                        self._mpr_pan_end(orientation)
                        self._restore_interactor_style(orientation)
                        return True

                    # ── Safety: always restore interactor style if it was ──
                    # swapped to dummy during press.  This prevents the style
                    # from getting stuck when eventFilter consumed the press
                    # but VTK observer also ran (or vice-versa).
                    self._restore_interactor_style(orientation)

                    if getattr(self, '_df_clip_drag', None):
                        self._df_clip_end_drag()
                        widget = getattr(self, f'{orientation}_widget')
                        hit = self._get_df_clip_hit(event.pos(), orientation) if getattr(self, '_df_clip_enabled', False) else None
                        self._df_clip_hover = hit
                        if hit is not None:
                            widget.setCursor(self._df_clip_cursor_for_hit(hit))
                        else:
                            widget.unsetCursor()
                        return True

                    if getattr(self, '_is_dragging_crosshair', False):
                        self._is_dragging_crosshair = False
                        self._crosshair_drag_mode = None
                        self._crosshair_drag_end()
                        return True
                if getattr(self, '_oblique_dragging', None):
                    self._oblique_dragging = None

                    self._restore_interactor_style(orientation)

                    hit = self._get_oblique_hit(event.pos(), orientation)
                    self._oblique_hover_handle = hit
                    self.render_slice(orientation, preserve_camera=True)
                    widget = getattr(self, f'{orientation}_widget')
                    if hit:
                        widget.setCursor(Qt.SizeAllCursor)
                    else:
                        widget.unsetCursor()
                    return True
            
            elif event.type() == QEvent.Wheel:
                # ── Fiji-style zoom via Qt wheel event ──────────────────
                # This is the primary handler for scroll-wheel on the 2D views.
                # It fires even when the VTK interactor style is swapped to
                # dummy (crosshair mode), so zoom always works.
                if self.volume_data is not None:
                    modifiers = event.modifiers()
                    angle_delta = event.angleDelta().y()
                    pixel_delta = event.pixelDelta().y() if hasattr(event, "pixelDelta") else 0
                    if angle_delta == 0 and pixel_delta == 0:
                        return super().eventFilter(obj, event)

                    if modifiers & Qt.ControlModifier:
                        # Ctrl+scroll → change slice (step by wheel notches)
                        if pixel_delta != 0:
                            delta = 1 if pixel_delta > 0 else -1
                        else:
                            delta = 1 if angle_delta > 0 else -1
                        self._scroll_change_slice(orientation, delta)
                    else:
                        vtk_widget = getattr(self, f'{orientation}_widget')
                        if pixel_delta != 0:
                            strength = abs(pixel_delta) / 40.0
                            zoom_in = pixel_delta > 0
                        else:
                            strength = abs(angle_delta) / 120.0
                            zoom_in = angle_delta > 0
                        strength = max(0.15, min(strength, 3.0))
                        self._fiji_zoom_at_cursor(
                            vtk_widget, orientation, zoom_in=zoom_in, strength=strength
                        )
                    return True  # consume the event

            elif event.type() == QEvent.Leave:
                widget = getattr(self, f'{orientation}_widget')
                if getattr(self, '_df_clip_hover', None) is not None:
                    self._df_clip_hover = None
                    if getattr(self, '_df_clip_enabled', False):
                        self._df_clip_update_mpr_overlay(orientation)
                        try:
                            widget.GetRenderWindow().Render()
                        except Exception:
                            pass
                if getattr(self, '_oblique_hover_handle', None) and getattr(self, '_oblique_hover_handle')[0] == orientation:
                    self._oblique_hover_handle = None
                    self.render_slice(orientation, preserve_camera=True)
                if getattr(self, '_crosshair_hover', None) and getattr(self, '_crosshair_hover')[0] == orientation:
                    self._crosshair_hover = None
                    if self.crosshair_enabled:
                        self.update_2d_crosshair(orientation)
                widget.unsetCursor()
        
        return super().eventFilter(obj, event)
    
    def update_pixel_value(self, orientation, pos):
        try:
            widget = getattr(self, f'{orientation}_widget')
            renderer = getattr(self, f'{orientation}_renderer')
            
            x, y = pos.x(), pos.y()
            size = widget.GetRenderWindow().GetSize()
            vtk_y = size[1] - y
            
            picker = vtk.vtkWorldPointPicker()
            picker.Pick(x, vtk_y, 0, renderer)
            world_pos = picker.GetPickPosition()
            
            slice_idx = self.current_slices[orientation]
            
            # Bounds check
            vol_z, vol_y, vol_x = self.volume_data.shape
            value = None
            
            if orientation == 'axial':
                # XY plane: VTK X=X, VTK Y=Y
                px = int(world_pos[0])
                py = int(world_pos[1])
                if 0 <= px < vol_x and 0 <= py < vol_y:
                    value = self.volume_data[slice_idx, py, px]
            elif orientation == 'coronal':
                # XZ plane: VTK X=X, VTK Y=Z
                px = int(world_pos[0])
                pz = int(world_pos[1])
                if 0 <= px < vol_x and 0 <= pz < vol_z:
                     value = self.volume_data[pz, slice_idx, px]
            else: # sagittal (YZ plane, flipud+rot90)
                 # After flipud(rot90(M)): VTK X = Z, VTK Y = Y (direct)
                 pz = int(world_pos[0])
                 py = int(world_pos[1])
                 if 0 <= pz < vol_z and 0 <= py < vol_y:
                     value = self.volume_data[pz, py, slice_idx]

            label = getattr(self, f'{orientation}_pixel_label')
            if value is not None:
                label.setText(f"Pixel: {value}")
            else:
                label.setText("Pixel: --")
        except Exception:
            pass
    
    def reset_crosshair(self):
        """Reset crosshair position to the center of the volume and angles to 0."""
        if self.volume_data is None:
            return
        vol_z, vol_y, vol_x = self.volume_data.shape
        self.oblique_angles = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
        self.updatePoint(vol_x // 2, vol_y // 2, vol_z // 2)

    # ══════════════════════════════════════════════════════════════════
    # P0 / P1 — MPR navigation (crosshair / zoom / pan)
    # ══════════════════════════════════════════════════════════════════
    def _toggle_link_mpr_nav(self, checked):
        self._mpr_link_nav = bool(checked)

    def _active_mpr_orientation(self):
        ori = getattr(self, "_active_grid_pane", None)
        if ori in ("axial", "coronal", "sagittal"):
            return ori
        return "axial"

    def _shortcut_fit_active_pane(self):
        if self.volume_data is None:
            return
        self.reset_view(self._active_mpr_orientation())

    def _shortcut_one_to_one_active_pane(self):
        if self.volume_data is None:
            return
        self._set_mpr_one_to_one(self._active_mpr_orientation())

    def _shortcut_reset_active_pane(self):
        if self.volume_data is None:
            return
        active = getattr(self, "_active_grid_pane", None)
        # 3D pane: reset volume camera only (do not touch MPR / crosshair)
        if active == "view_3d":
            if hasattr(self, "reset_3d_view"):
                self.reset_3d_view()
            return
        ori = self._active_mpr_orientation()
        self.reset_view(ori)
        # Also re-center crosshair if enabled (full reset feel on MPR)
        if self.crosshair_enabled:
            self.reset_crosshair()

    def _mpr_image_size(self, orientation):
        """(h, w) of the displayed plane in voxels."""
        vol_z, vol_y, vol_x = self.volume_data.shape
        if orientation == "axial":
            return vol_y, vol_x
        if orientation == "coronal":
            return vol_z, vol_x
        return vol_y, vol_z  # sagittal

    def _mpr_default_parallel_scale(self, orientation):
        """Fit-to-view ParallelScale for this pane (half-height world units)."""
        h, w = self._mpr_image_size(orientation)
        vtk_widget = getattr(self, f"{orientation}_widget", None)
        if vtk_widget is None:
            return max(h, 1) * 0.5
        vp = vtk_widget.GetRenderWindow().GetSize()
        vp_w = max(float(vp[0]), 1.0)
        vp_h = max(float(vp[1]), 1.0)
        vp_aspect = vp_w / vp_h
        img_aspect = float(w) / max(float(h), 1.0)
        if img_aspect > vp_aspect:
            return (float(w) / vp_aspect) * 0.5
        return float(h) * 0.5

    def _mpr_zoom_limits(self, orientation):
        """(s_min, s_max) — deep zoom (~few voxels in view) but not infinite out."""
        h, w = self._mpr_image_size(orientation)
        default = self._mpr_default_parallel_scale(orientation)
        # ParallelScale = half viewport height in voxels. Min ≈ 2 voxels half-height
        # (≈ 4 voxels visible vertically). Max ≈ 3× fit.
        s_min = 2.0
        s_max = max(default * 3.0, float(max(h, w)) * 1.5, s_min + 1.0)
        return s_min, s_max

    def _crosshair_hit_radii(self, orientation):
        """Center / line hit radii in display px — scale gently with zoom."""
        renderer = getattr(self, f"{orientation}_renderer", None)
        base_c, base_l = 16.0, 8.0
        if renderer is None or self.volume_data is None:
            return base_c, base_l
        try:
            cam = renderer.GetActiveCamera()
            cur = float(cam.GetParallelScale())
            default = self._mpr_default_parallel_scale(orientation)
            # Zoomed in → slightly larger grab; zoomed out → clamp
            zoom = default / max(cur, 1e-6)
            scale = max(0.85, min(1.55, 0.75 + 0.25 * zoom))
            return base_c * scale, base_l * scale
        except Exception:
            return base_c, base_l

    def _pick_world(self, orientation, pos):
        """World coords under Qt pos (widget coords)."""
        widget = getattr(self, f"{orientation}_widget")
        renderer = getattr(self, f"{orientation}_renderer")
        x, y = pos.x(), pos.y()
        size = widget.GetRenderWindow().GetSize()
        vtk_y = size[1] - y
        picker = vtk.vtkWorldPointPicker()
        picker.Pick(float(x), float(vtk_y), 0, renderer)
        return list(picker.GetPickPosition())

    def _pick_voxel(self, orientation, pos, mode="center"):
        """Map display pos → integer voxel (X,Y,Z) with grab-mode constraints."""
        if self.volume_data is None:
            return tuple(self.crosshair_position)
        world = self._pick_world(orientation, pos)
        vol_z, vol_y, vol_x = self.volume_data.shape
        newX, newY, newZ = self.crosshair_position

        if orientation == "axial":
            pickX = int(round(world[0]))
            pickY = int(round(world[1]))
            if mode in ("center", "v", "place"):
                newX = pickX
            if mode in ("center", "h", "place"):
                newY = pickY
        elif orientation == "coronal":
            pickX = int(round(world[0]))
            pickZ = (vol_z - 1) - int(round(world[1]))
            if mode in ("center", "v", "place"):
                newX = pickX
            if mode in ("center", "h", "place"):
                newZ = pickZ
        else:  # sagittal
            pickZ = int(round(world[0]))
            pickY = int(round(world[1]))
            if mode in ("center", "v", "place"):
                newZ = pickZ
            if mode in ("center", "h", "place"):
                newY = pickY

        newX = max(0, min(newX, vol_x - 1))
        newY = max(0, min(newY, vol_y - 1))
        newZ = max(0, min(newZ, vol_z - 1))
        return newX, newY, newZ

    def _flush_crosshair_sync(self):
        """Timer callback kept for compatibility — always full linked sync."""
        self._ch_sync_pending = False
        if self.volume_data is None:
            return
        x, y, z = self.crosshair_position
        self.updatePoint(x, y, z)

    def _crosshair_drag_update(self, orientation, pos, mode="center", preview_only=False):
        """Move crosshair from a pointer position.

        Always goes through ``updatePoint`` so crosshair ↔ slice sliders ↔
        linked MPR panes stay in lockstep (pre-P0 behaviour). Selective
        re-render inside updatePoint already avoids full rebuild when only
        the hair moves on an unchanged slice.
        """
        _ = preview_only  # kept for call-site compatibility; never desync
        newX, newY, newZ = self._pick_voxel(orientation, pos, mode=mode)
        self.updatePoint(newX, newY, newZ)

    def _crosshair_drag_end(self):
        """End drag — force one final sync (sliders + all panes + 3D)."""
        if getattr(self, "_ch_sync_timer", None) is not None:
            self._ch_sync_timer.stop()
        self._ch_sync_pending = False
        if self.volume_data is None:
            return
        x, y, z = self.crosshair_position
        self.updatePoint(x, y, z)

    def _cursor_for_crosshair_hit(self, hit):
        if hit is None:
            return None
        mode = hit[1]
        if mode == "center":
            return Qt.SizeAllCursor
        if mode == "h":
            return Qt.SizeVerCursor  # drag H line → move vertically
        return Qt.SizeHorCursor

    def _set_mpr_cursor(self, orientation, cursor):
        widget = getattr(self, f"{orientation}_widget", None)
        if widget is None:
            return
        if cursor is None:
            widget.unsetCursor()
        else:
            widget.setCursor(cursor)

    def _mpr_pan_begin(self, orientation, pos):
        self._mpr_pan = {
            "ori": orientation,
            "last": QPoint(pos),
            "start_scale": None,
        }
        ren = getattr(self, f"{orientation}_renderer", None)
        if ren is not None:
            try:
                self._mpr_pan["start_scale"] = float(ren.GetActiveCamera().GetParallelScale())
            except Exception:
                pass
        self._set_mpr_cursor(orientation, Qt.ClosedHandCursor)

    def _mpr_pan_move(self, orientation, pos):
        state = getattr(self, "_mpr_pan", None)
        if not state or state.get("ori") != orientation:
            return
        last = state["last"]
        dx_px = float(pos.x() - last.x())
        dy_px = float(pos.y() - last.y())
        state["last"] = QPoint(pos)
        if abs(dx_px) < 0.01 and abs(dy_px) < 0.01:
            return
        self._mpr_pan_by_pixels(orientation, dx_px, dy_px)
        if getattr(self, "_mpr_link_nav", False):
            for other in ("axial", "coronal", "sagittal"):
                if other != orientation:
                    self._mpr_pan_by_pixels(other, dx_px, dy_px)

    def _mpr_pan_by_pixels(self, orientation, dx_px, dy_px):
        """Pan parallel camera by display-pixel deltas (Fiji-style)."""
        ren = getattr(self, f"{orientation}_renderer", None)
        widget = getattr(self, f"{orientation}_widget", None)
        if ren is None or widget is None:
            return
        cam = ren.GetActiveCamera()
        if not cam.GetParallelProjection():
            return
        size = widget.GetRenderWindow().GetSize()
        vp_h = max(float(size[1]), 1.0)
        scale = float(cam.GetParallelScale())
        # ParallelScale = half viewport height in world units
        world_per_px = (2.0 * scale) / vp_h
        # Qt Y down, VTK Y up
        dx_w = -dx_px * world_per_px
        dy_w = dy_px * world_per_px

        # Build camera right/up in world
        pos = list(cam.GetPosition())
        fp = list(cam.GetFocalPoint())
        up = list(cam.GetViewUp())
        view = [fp[0] - pos[0], fp[1] - pos[1], fp[2] - pos[2]]
        # right = view × up
        rx = view[1] * up[2] - view[2] * up[1]
        ry = view[2] * up[0] - view[0] * up[2]
        rz = view[0] * up[1] - view[1] * up[0]
        rn = (rx * rx + ry * ry + rz * rz) ** 0.5
        if rn < 1e-12:
            return
        rx, ry, rz = rx / rn, ry / rn, rz / rn
        un = (up[0] ** 2 + up[1] ** 2 + up[2] ** 2) ** 0.5
        if un < 1e-12:
            return
        ux, uy, uz = up[0] / un, up[1] / un, up[2] / un

        shift = [
            rx * dx_w + ux * dy_w,
            ry * dx_w + uy * dy_w,
            rz * dx_w + uz * dy_w,
        ]
        cam.SetPosition(pos[0] + shift[0], pos[1] + shift[1], pos[2] + shift[2])
        cam.SetFocalPoint(fp[0] + shift[0], fp[1] + shift[1], fp[2] + shift[2])
        ren.ResetCameraClippingRange()
        self.save_camera_state(orientation)
        widget.GetRenderWindow().Render()

    def _mpr_pan_end(self, orientation):
        self._mpr_pan = None
        # Restore hover cursor
        if self.crosshair_enabled:
            self._set_mpr_cursor(orientation, Qt.ArrowCursor)
        else:
            self._set_mpr_cursor(orientation, None)

    def _set_mpr_one_to_one(self, orientation):
        """1 voxel ≈ 1 screen pixel (along viewport height)."""
        ren = getattr(self, f"{orientation}_renderer", None)
        widget = getattr(self, f"{orientation}_widget", None)
        if ren is None or widget is None or self.volume_data is None:
            return
        cam = ren.GetActiveCamera()
        if not cam.GetParallelProjection():
            return
        vp_h = max(float(widget.GetRenderWindow().GetSize()[1]), 1.0)
        # half-height in voxels for 1:1
        new_scale = vp_h * 0.5
        s_min, s_max = self._mpr_zoom_limits(orientation)
        new_scale = max(s_min, min(new_scale, s_max))
        # Zoom about crosshair (or image center)
        self._zoom_parallel_about_point(orientation, new_scale, about_crosshair=True)
        if getattr(self, "_mpr_link_nav", False):
            for other in ("axial", "coronal", "sagittal"):
                if other != orientation:
                    o_min, o_max = self._mpr_zoom_limits(other)
                    self._zoom_parallel_about_point(
                        other, max(o_min, min(new_scale, o_max)), about_crosshair=True
                    )

    def _zoom_parallel_about_point(self, orientation, new_scale, about_crosshair=True):
        ren = getattr(self, f"{orientation}_renderer", None)
        widget = getattr(self, f"{orientation}_widget", None)
        if ren is None or widget is None:
            return
        cam = ren.GetActiveCamera()
        if not cam.GetParallelProjection():
            return
        # Anchor: crosshair world or current focal
        if about_crosshair and self.volume_data is not None:
            cx, cy, cz = self.crosshair_position
            if orientation == "axial":
                anchor = [float(cx), float(cy), float(cam.GetFocalPoint()[2])]
            elif orientation == "coronal":
                vol_z = self.volume_data.shape[0]
                anchor = [float(cx), float((vol_z - 1) - cz), float(cam.GetFocalPoint()[2])]
            else:
                anchor = [float(cz), float(cy), float(cam.GetFocalPoint()[2])]
        else:
            anchor = list(cam.GetFocalPoint())

        old_scale = float(cam.GetParallelScale())
        if old_scale <= 1e-12:
            old_scale = new_scale
        cam.SetParallelScale(float(new_scale))
        # Keep anchor fixed in display: shift by scale ratio around focal
        # (simple: move FP/pos so anchor stays — use display mapping)
        try:
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToWorld()
            coord.SetValue(anchor[0], anchor[1], anchor[2])
            # After scale change without moving camera, re-pick is complex;
            # use parallel scale ratio on view plane offset from FP
            fp = list(cam.GetFocalPoint())
            pos = list(cam.GetPosition())
            # Offset of anchor from FP in world, scale by (1 - new/old) wait:
            # When parallel scale shrinks, world extent shrinks; to keep anchor
            # under same pixel, FP moves toward anchor.
            ratio = new_scale / old_scale
            # Camera space: FP' = anchor + (FP - anchor) * ratio
            nfp = [
                anchor[0] + (fp[0] - anchor[0]) * ratio,
                anchor[1] + (fp[1] - anchor[1]) * ratio,
                anchor[2] + (fp[2] - anchor[2]) * ratio,
            ]
            d = [nfp[0] - fp[0], nfp[1] - fp[1], nfp[2] - fp[2]]
            cam.SetFocalPoint(nfp[0], nfp[1], nfp[2])
            cam.SetPosition(pos[0] + d[0], pos[1] + d[1], pos[2] + d[2])
        except Exception:
            pass
        ren.ResetCameraClippingRange()
        self.save_camera_state(orientation)
        if self.volume_data is not None:
            h, w = self._mpr_image_size(orientation)
            self.add_ruler_overlay(ren, orientation, (h, w))
        widget.GetRenderWindow().Render()

    def updatePoint(self, newX, newY, newZ):
        """Single source of truth: crosshair (X,Y,Z) ↔ slice indices ↔ sliders.

        Mapping (volume index order Z,Y,X):
          sagittal slider  ↔ X  ↔ crosshair_position[0]
          coronal  slider  ↔ Y  ↔ crosshair_position[1]
          axial    slider  ↔ Z  ↔ crosshair_position[2]

        Performance (unchanged from pre-P0): only ``render_slice`` when the
        underlying plane index changes; otherwise lightweight crosshair move.
        """
        if self.volume_data is None:
            return
        vol_z, vol_y, vol_x = self.volume_data.shape
        newX = max(0, min(int(round(newX)), vol_x - 1))
        newY = max(0, min(int(round(newY)), vol_y - 1))
        newZ = max(0, min(int(round(newZ)), vol_z - 1))

        oldX, oldY, oldZ = self.crosshair_position
        # No-op if nothing changed (avoids slider/render thrash)
        if (
            newX == oldX and newY == oldY and newZ == oldZ
            and self.current_slices.get('sagittal') == newX
            and self.current_slices.get('coronal') == newY
            and self.current_slices.get('axial') == newZ
        ):
            return

        self.crosshair_position = [newX, newY, newZ]

        # Which displayed planes need new voxel data?
        changed = {
            'sagittal': newX != oldX or self.current_slices.get('sagittal') != newX,
            'coronal': newY != oldY or self.current_slices.get('coronal') != newY,
            'axial': newZ != oldZ or self.current_slices.get('axial') != newZ,
        }

        self.current_slices['sagittal'] = newX
        self.current_slices['coronal'] = newY
        self.current_slices['axial'] = newZ

        # Sync all three sliders + labels (blocked → no re-entrant update_slice)
        self._sync_slice_sliders_from_crosshair(newX, newY, newZ, vol_x, vol_y, vol_z)

        for ori in ['axial', 'coronal', 'sagittal']:
            if changed[ori]:
                self.render_slice(ori, preserve_camera=True)
            elif self.crosshair_enabled:
                self.update_2d_crosshair(ori)

        if self.crosshair_enabled:
            for ori in ['axial', 'coronal', 'sagittal']:
                self._ensure_crosshair_visible(ori)

        self.update_3d_crosshair()

    def _sync_slice_sliders_from_crosshair(self, newX, newY, newZ, vol_x, vol_y, vol_z):
        """Push crosshair indices onto the three slice sliders without feedback loops."""
        pairs = (
            ('sagittal', newX, vol_x - 1),
            ('coronal', newY, vol_y - 1),
            ('axial', newZ, vol_z - 1),
        )
        for ori, val, vmax in pairs:
            slider = getattr(self, f'{ori}_slice_slider', None)
            label = getattr(self, f'{ori}_slice_label', None)
            if slider is None:
                continue
            # Keep range correct if volume was reloaded mid-session
            if slider.maximum() != vmax and vmax >= 0:
                slider.blockSignals(True)
                slider.setMaximum(vmax)
                slider.blockSignals(False)
            if slider.value() != val:
                slider.blockSignals(True)
                slider.setValue(val)
                slider.blockSignals(False)
            if label is not None:
                label.setText(f"{val} / {vmax}")

    def update_2d_crosshair(self, orientation):
        """Lightweight crosshair update that reuses persistent VTK actors.
        
        Instead of calling render_slice (which destroys and rebuilds ALL
        VTK actors including the base image), this method only repositions
        the crosshair line endpoints and re-renders. This is ~10x faster
        and matches v113's smooth crosshair behavior.
        """
        if not self.crosshair_enabled or self.volume_data is None:
            return
        
        # Initialize persistent crosshair actor cache if needed
        if not hasattr(self, '_persistent_crosshair'):
            self._persistent_crosshair = {'axial': {}, 'coronal': {}, 'sagittal': {}}
        
        renderer = getattr(self, f'{orientation}_renderer')
        widget = getattr(self, f'{orientation}_widget')
        actors_dict = self._persistent_crosshair[orientation]
        
        # Initialize actors if not created yet
        if not actors_dict:
            for name in ['h_line1', 'h_line2', 'v_line1', 'v_line2']:
                ls = vtk.vtkLineSource()
                m = vtk.vtkPolyDataMapper2D()
                m.SetInputConnection(ls.GetOutputPort())
                a = vtk.vtkActor2D()
                a.SetMapper(m)
                a.GetProperty().SetLineWidth(1)
                actors_dict[name] = {'source': ls, 'actor': a}

            # 4 edge labels like 3D Teaching: +H / -H / +V / -V
            for name in ['h_pos', 'h_neg', 'v_pos', 'v_neg']:
                t = vtk.vtkTextActor()
                t.GetTextProperty().SetFontSize(11)
                t.GetTextProperty().BoldOn()
                actors_dict[name] = t

        # Remove legacy near-center labels + center handles (clutter object view)
        for legacy in ('center_fill', 'center_ring', 'h_label', 'v_label'):
            if legacy in actors_dict:
                item = actors_dict[legacy]
                act = item.get('actor') if isinstance(item, dict) else item
                if act is not None and renderer.HasViewProp(act):
                    renderer.RemoveActor(act)
                del actors_dict[legacy]

        # Migrate older caches that only had 2 labels
        for name in ['h_pos', 'h_neg', 'v_pos', 'v_neg']:
            if name not in actors_dict:
                t = vtk.vtkTextActor()
                t.GetTextProperty().SetFontSize(11)
                t.GetTextProperty().BoldOn()
                actors_dict[name] = t
        
        vol_z, vol_y, vol_x = self.volume_data.shape
        
        # Axis-based colors + signed labels (match 3D Teaching / create_crosshair_actors)
        # labels: h_pos, h_neg, v_pos, v_neg  →  +H, -H, +V, -V along axis directions
        if orientation == 'axial':
            cx_w = float(self.crosshair_position[0])
            cy_w = float(self.crosshair_position[1])
            h_pos, h_neg = '+X', '-X'
            v_pos, v_neg = '+Y', '-Y'
            h_color = (1.0, 0.192, 0.192)     # Red   = X axis
            v_color = (0.223, 1.0, 0.078)     # Green = Y axis
            h_name, v_name = 'coronal', 'sagittal'
        elif orientation == 'coronal':
            cx_w = float(self.crosshair_position[0])
            cy_w = float((vol_z - 1) - self.crosshair_position[2])
            h_pos, h_neg = '+X', '-X'
            # Display Y is flipped Z: top (higher display Y) is -Z when reverse_z is False
            if self.reverse_z:
                v_pos, v_neg = '+Z', '-Z'
            else:
                v_pos, v_neg = '-Z', '+Z'
            h_color = (1.0, 0.192, 0.192)     # Red  = X axis
            v_color = (0.121, 0.317, 1.0)     # Blue = Z axis
            h_name, v_name = 'axial', 'sagittal'
        else:  # sagittal
            cx_w = float(self.crosshair_position[2])
            cy_w = float(self.crosshair_position[1])
            if self.reverse_z:
                h_pos, h_neg = '-Z', '+Z'
            else:
                h_pos, h_neg = '+Z', '-Z'
            v_pos, v_neg = '+Y', '-Y'
            h_color = (0.121, 0.317, 1.0)     # Blue  = Z axis
            v_color = (0.223, 1.0, 0.078)     # Green = Y axis
            h_name, v_name = 'axial', 'coronal'
        
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        coord.SetValue(cx_w, cy_w, 0.5)
        dp = coord.GetComputedDisplayValue(renderer)
        dx, dy = float(dp[0]), float(dp[1])
        
        win = renderer.GetRenderWindow().GetSize()
        vp = renderer.GetViewport()
        vp_w = (vp[2] - vp[0]) * win[0]
        vp_h = (vp[3] - vp[1]) * win[1]
        gap = 12

        # Dragonfly hover / drag highlight — only the active pane lights up
        hover = getattr(self, '_crosshair_hover', None)
        drag_mode = getattr(self, '_crosshair_drag_mode', None) if getattr(self, '_is_dragging_crosshair', False) else None
        active_part = None
        if hover and hover[0] == orientation:
            active_part = drag_mode if drag_mode else hover[1]

        h_hot = active_part in ('center', 'h')
        v_hot = active_part in ('center', 'v')

        h_draw = self._boost_color(h_color, 1.45 if h_hot else 1.0)
        v_draw = self._boost_color(v_color, 1.45 if v_hot else 1.0)
        h_lw = 2.5 if h_hot else 1.0
        v_lw = 2.5 if v_hot else 1.0
        
        # Configure colors + line widths (brighten on hover/grab)
        for n in ['h_line1', 'h_line2']:
            prop = actors_dict[n]['actor'].GetProperty()
            prop.SetColor(*h_draw)
            prop.SetLineWidth(h_lw)
            prop.SetOpacity(1.0 if h_hot else 0.92)
        for n in ['v_line1', 'v_line2']:
            prop = actors_dict[n]['actor'].GetProperty()
            prop.SetColor(*v_draw)
            prop.SetLineWidth(v_lw)
            prop.SetOpacity(1.0 if v_hot else 0.92)
        
        import math
        if orientation == 'axial':
            theta_h = math.radians(self.oblique_angles.get('coronal', 0.0))
            theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
        elif orientation == 'coronal':
            theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
            theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
        else:  # sagittal
            theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
            theta_v = math.radians(self.oblique_angles.get('coronal', 0.0)) + math.pi / 2
        
        cos_h, sin_h = math.cos(theta_h), math.sin(theta_h)
        cos_v, sin_v = math.cos(theta_v), math.sin(theta_v)
        line_ext = max(vp_w, vp_h) * 2.5
        
        # Update line endpoints (no new objects created)
        actors_dict['h_line1']['source'].SetPoint1(dx - line_ext * cos_h, dy - line_ext * sin_h, 0)
        actors_dict['h_line1']['source'].SetPoint2(dx - gap * cos_h, dy - gap * sin_h, 0)
        actors_dict['h_line2']['source'].SetPoint1(dx + gap * cos_h, dy + gap * sin_h, 0)
        actors_dict['h_line2']['source'].SetPoint2(dx + line_ext * cos_h, dy + line_ext * sin_h, 0)
        
        actors_dict['v_line1']['source'].SetPoint1(dx - line_ext * cos_v, dy - line_ext * sin_v, 0)
        actors_dict['v_line1']['source'].SetPoint2(dx - gap * cos_v, dy - gap * sin_v, 0)
        actors_dict['v_line2']['source'].SetPoint1(dx + gap * cos_v, dy + gap * sin_v, 0)
        actors_dict['v_line2']['source'].SetPoint2(dx + line_ext * cos_v, dy + line_ext * sin_v, 0)

        # Edge labels at viewport borders (Teaching-style: +X/-X, +Y/-Y, +Z/-Z)
        def _border_ends(cos_t, sin_t):
            t_candidates = []
            if abs(cos_t) > 1e-5:
                t = -dx / cos_t
                y = dy + t * sin_t
                if 0 <= y <= vp_h:
                    t_candidates.append((t, 0.0, y))
                t = (vp_w - dx) / cos_t
                y = dy + t * sin_t
                if 0 <= y <= vp_h:
                    t_candidates.append((t, vp_w, y))
            if abs(sin_t) > 1e-5:
                t = -dy / sin_t
                x = dx + t * cos_t
                if 0 <= x <= vp_w:
                    t_candidates.append((t, x, 0.0))
                t = (vp_h - dy) / sin_t
                x = dx + t * cos_t
                if 0 <= x <= vp_w:
                    t_candidates.append((t, x, vp_h))
            unique = []
            seen = set()
            for item in t_candidates:
                key = (round(item[0], 2), round(item[1], 2), round(item[2], 2))
                if key not in seen:
                    seen.add(key)
                    unique.append(item)
            unique.sort(key=lambda it: it[0])
            neg_pt, pos_pt = None, None
            for t, x, y in unique:
                if t < -0.1:
                    neg_pt = (x, y)
                elif t > 0.1:
                    pos_pt = (x, y)
            if neg_pt is None:
                neg_pt = (dx - 200.0 * cos_t, dy - 200.0 * sin_t)
            if pos_pt is None:
                pos_pt = (dx + 200.0 * cos_t, dy + 200.0 * sin_t)
            nx = max(5.0, min(neg_pt[0], vp_w - 40.0))
            ny = max(5.0, min(neg_pt[1], vp_h - 18.0))
            px = max(5.0, min(pos_pt[0], vp_w - 40.0))
            py = max(5.0, min(pos_pt[1], vp_h - 18.0))
            return (nx, ny), (px, py)

        h_neg_pt, h_pos_pt = _border_ends(cos_h, sin_h)
        v_neg_pt, v_pos_pt = _border_ends(cos_v, sin_v)

        # Optional oblique angle hint (same as create_crosshair_actors)
        ang_h = self.oblique_angles.get(h_name, 0.0)
        ang_v = self.oblique_angles.get(v_name, 0.0)
        h_txt_pos = f"{h_pos} ({ang_h:.1f}°)" if abs(ang_h) > 0.5 else h_pos
        h_txt_neg = f"{h_neg} ({ang_h:.1f}°)" if abs(ang_h) > 0.5 else h_neg
        v_txt_pos = f"{v_pos} ({ang_v:.1f}°)" if abs(ang_v) > 0.5 else v_pos
        v_txt_neg = f"{v_neg} ({ang_v:.1f}°)" if abs(ang_v) > 0.5 else v_neg

        for key, text, pos, color in [
            ('h_pos', h_txt_pos, h_pos_pt, h_draw),
            ('h_neg', h_txt_neg, h_neg_pt, h_draw),
            ('v_pos', v_txt_pos, v_pos_pt, v_draw),
            ('v_neg', v_txt_neg, v_neg_pt, v_draw),
        ]:
            actors_dict[key].SetInput(text)
            actors_dict[key].SetPosition(pos[0], pos[1])
            actors_dict[key].GetTextProperty().SetColor(*color)

        # Ensure all core actors are added to renderer
        for k, v in actors_dict.items():
            if k == 'handles': continue
            act = v['actor'] if isinstance(v, dict) and 'actor' in v else v
            if not renderer.HasViewProp(act):
                renderer.AddActor(act)
                
        # Clean up old handle actors
        if 'handles' in actors_dict:
            for act in actors_dict['handles']:
                renderer.RemoveActor(act)
        actors_dict['handles'] = []
        
        # Create and add new handle actors if enabled
        if getattr(self, 'oblique_handles_enabled', False):
            extent = max(vp_w, vp_h) * 0.6
            handle_dist = min(extent * 0.35, 120)
            hover = getattr(self, '_oblique_hover_handle', None)
            dragging = getattr(self, '_oblique_dragging', None)

            if orientation == 'axial':
                h_name, v_name = 'coronal', 'sagittal'
            elif orientation == 'coronal':
                h_name, v_name = 'axial', 'sagittal'
            else:
                h_name, v_name = 'axial', 'coronal'

            def make_circle2d(cx, cy, r, color, fill=False, opacity=1.0, lw=1):
                src = vtk.vtkRegularPolygonSource()
                src.SetNumberOfSides(24)
                src.SetRadius(r)
                src.SetCenter(cx, cy, 0)
                if fill:
                    src.GeneratePolygonOn()
                else:
                    src.GeneratePolygonOff()
                m = vtk.vtkPolyDataMapper2D()
                m.SetInputConnection(src.GetOutputPort())
                a = vtk.vtkActor2D()
                a.SetMapper(m)
                a.GetProperty().SetColor(*color)
                a.GetProperty().SetOpacity(opacity)
                a.GetProperty().SetLineWidth(lw)
                return a

            def make_curved_arrow_2d(cx, cy, radius, start_angle, sweep_deg, color, opacity=1.0, lw=2.0, with_arrowhead=True):
                import math as _m
                n_pts = max(12, int(abs(sweep_deg) / 3))
                pts = vtk.vtkPoints()
                lines = vtk.vtkCellArray()
                for i in range(n_pts + 1):
                    a = _m.radians(start_angle + sweep_deg * i / n_pts)
                    pts.InsertNextPoint(cx + radius * _m.cos(a), cy + radius * _m.sin(a), 0)
                line = vtk.vtkPolyLine()
                line.GetPointIds().SetNumberOfIds(n_pts + 1)
                for i in range(n_pts + 1):
                    line.GetPointIds().SetId(i, i)
                lines.InsertNextCell(line)
                if with_arrowhead:
                    end_a = _m.radians(start_angle + sweep_deg)
                    ex, ey = cx + radius * _m.cos(end_a), cy + radius * _m.sin(end_a)
                    sign = 1 if sweep_deg > 0 else -1
                    tx = -sign * _m.sin(end_a)
                    ty = sign * _m.cos(end_a)
                    arrow_len = max(8.0, radius * 0.35)
                    nx_d = _m.cos(end_a)
                    ny_d = _m.sin(end_a)
                    tip_x = ex + arrow_len * tx
                    tip_y = ey + arrow_len * ty
                    w1_x = ex + arrow_len * 0.35 * nx_d
                    w1_y = ey + arrow_len * 0.35 * ny_d
                    w2_x = ex - arrow_len * 0.35 * nx_d
                    w2_y = ey - arrow_len * 0.35 * ny_d
                    base_id = pts.GetNumberOfPoints()
                    pts.InsertNextPoint(tip_x, tip_y, 0)
                    pts.InsertNextPoint(w1_x, w1_y, 0)
                    pts.InsertNextPoint(w2_x, w2_y, 0)
                    tri_line = vtk.vtkPolyLine()
                    tri_line.GetPointIds().SetNumberOfIds(4)
                    tri_line.GetPointIds().SetId(0, base_id)
                    tri_line.GetPointIds().SetId(1, base_id + 1)
                    tri_line.GetPointIds().SetId(2, base_id + 2)
                    tri_line.GetPointIds().SetId(3, base_id)
                    lines.InsertNextCell(tri_line)
                pd = vtk.vtkPolyData()
                pd.SetPoints(pts)
                pd.SetLines(lines)
                m = vtk.vtkPolyDataMapper2D()
                m.SetInputData(pd)
                a = vtk.vtkActor2D()
                a.SetMapper(m)
                a.GetProperty().SetColor(*color)
                a.GetProperty().SetOpacity(opacity)
                a.GetProperty().SetLineWidth(lw)
                return a

            for cos_val, sin_val, color, which in [
                (cos_h, sin_h, h_color, h_name),
                (-cos_h, -sin_h, h_color, h_name),
                (cos_v, sin_v, v_color, v_name),
                (-cos_v, -sin_v, v_color, v_name),
            ]:
                hx = dx + handle_dist * cos_val
                hy = dy + handle_dist * sin_val
                is_active = (hover == (orientation, which)) or (dragging == (orientation, which))

                base_angle = math.degrees(math.atan2(sin_val, cos_val))
                arc_radius = 16
                arc_sweep = 120
                arc_start = base_angle - arc_sweep / 2 + 90

                if is_active:
                    actors_dict['handles'].append(make_circle2d(hx, hy, 22, color, fill=True, opacity=0.25))
                    actors_dict['handles'].append(make_curved_arrow_2d(hx, hy, arc_radius, arc_start, arc_sweep, color, opacity=1.0, lw=2.5))
                    actors_dict['handles'].append(make_curved_arrow_2d(hx, hy, arc_radius, arc_start + 180, arc_sweep, color, opacity=0.7, lw=2.0))
                else:
                    actors_dict['handles'].append(make_curved_arrow_2d(hx, hy, arc_radius * 0.8, arc_start, arc_sweep * 0.7, color, opacity=0.35, lw=1.5))

            for act in actors_dict['handles']:
                renderer.AddActor(act)

        widget.GetRenderWindow().Render()

    def _get_oblique_hit(self, pos, orientation):
        import math
        if self.volume_data is None:
            return None
        if not getattr(self, 'oblique_handles_enabled', False):
            return None
        widget = getattr(self, f'{orientation}_widget')
        renderer = getattr(self, f'{orientation}_renderer')
        x, y = pos.x(), pos.y()
        size = widget.GetRenderWindow().GetSize()
        vtk_y = size[1] - y

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        vol_z, vol_y, vol_x = self.volume_data.shape
        
        if orientation == 'axial':
            cx_w = float(self.crosshair_position[0])
            cy_w = float(self.crosshair_position[1])
        elif orientation == 'coronal':
            cx_w = float(self.crosshair_position[0])
            cy_w = float((vol_z - 1) - self.crosshair_position[2])
        else:
            cx_w = float(self.crosshair_position[2])
            cy_w = float(self.crosshair_position[1])
            
        coord.SetValue(cx_w, cy_w, 0.5)
        dp = coord.GetComputedDisplayValue(renderer)
        dx, dy = float(dp[0]), float(dp[1])

        vp = renderer.GetViewport()
        win = renderer.GetRenderWindow().GetSize()
        vp_w = (vp[2] - vp[0]) * win[0]
        vp_h = (vp[3] - vp[1]) * win[1]

        if orientation == 'axial':
            theta_h = math.radians(self.oblique_angles.get('coronal', 0.0))
            theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
            h_name, v_name = 'coronal', 'sagittal'
        elif orientation == 'coronal':
            theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
            theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
            h_name, v_name = 'axial', 'sagittal'
        else: # sagittal
            theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
            theta_v = math.radians(self.oblique_angles.get('coronal', 0.0)) + math.pi / 2
            h_name, v_name = 'axial', 'coronal'

        extent = max(vp_w, vp_h) * 0.6
        handle_dist = min(extent * 0.35, 120)
        hit_dist = 35  # px tolerance

        # Check horizontal handles
        cos_h, sin_h = math.cos(theta_h), math.sin(theta_h)
        for sign in [1, -1]:
            ex = dx + sign * handle_dist * cos_h
            ey = dy + sign * handle_dist * sin_h
            if math.hypot(x - ex, vtk_y - ey) < hit_dist:
                return (orientation, h_name)

        # Check vertical handles
        cos_v, sin_v = math.cos(theta_v), math.sin(theta_v)
        for sign in [1, -1]:
            ex = dx + sign * handle_dist * cos_v
            ey = dy + sign * handle_dist * sin_v
            if math.hypot(x - ex, vtk_y - ey) < hit_dist:
                return (orientation, v_name)

        return None

    def _handle_oblique_rotate(self, pos, orientation):
        import math
        if self.volume_data is None:
            return
        widget = getattr(self, f'{orientation}_widget')
        renderer = getattr(self, f'{orientation}_renderer')
        x, y = pos.x(), pos.y()
        size = widget.GetRenderWindow().GetSize()
        vtk_y = size[1] - y

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        vol_z, vol_y, vol_x = self.volume_data.shape
        
        if orientation == 'axial':
            cx_w = float(self.crosshair_position[0])
            cy_w = float(self.crosshair_position[1])
        elif orientation == 'coronal':
            cx_w = float(self.crosshair_position[0])
            cy_w = float((vol_z - 1) - self.crosshair_position[2])
        else:
            cx_w = float(self.crosshair_position[2])
            cy_w = float(self.crosshair_position[1])
            
        coord.SetValue(cx_w, cy_w, 0.5)
        dp = coord.GetComputedDisplayValue(renderer)
        dx, dy = float(dp[0]), float(dp[1])

        angle = math.degrees(math.atan2(vtk_y - dy, x - dx))

        dragging = getattr(self, '_oblique_dragging', None)
        if dragging:
            drag_ori, axis = dragging
            if drag_ori == 'axial':
                h_name, v_name = 'coronal', 'sagittal'
            elif drag_ori == 'coronal':
                h_name, v_name = 'axial', 'sagittal'
            else: # sagittal
                h_name, v_name = 'axial', 'coronal'
                
            if axis == h_name:
                self.oblique_angles[axis] = angle
            else:
                self.oblique_angles[axis] = angle - 90.0

        # Update manual align angle labels immediately (lightweight)
        if getattr(self, '_manual_align_active', False):
            self._update_manual_angle_labels()

        # ── Throttled rendering during drag ──
        # Instead of rendering all 3 views on every mouse move (very slow),
        # we store which orientation needs refresh and use a debounce timer.
        self._oblique_drag_orientation = orientation
        
        if not hasattr(self, '_oblique_throttle_timer'):
            self._oblique_throttle_timer = QTimer(self)
            self._oblique_throttle_timer.setSingleShot(True)
            self._oblique_throttle_timer.timeout.connect(self._oblique_throttle_render)
        
        # Only schedule a new render if one is not already pending
        if not self._oblique_throttle_timer.isActive():
            self._oblique_throttle_timer.start(60)  # max ~16 fps during drag
    
    def _oblique_throttle_render(self):
        """Render callback for throttled oblique drag.
        
        Synchronises ALL views during drag for real-time feedback:
        - Manual align mode: lightweight crosshair overlay refresh on all 3 views.
        - Normal oblique mode: fast reslice on the active view, lightweight
          crosshair refresh on the other two, plus 3D crosshair update.
        """
        ori = getattr(self, '_oblique_drag_orientation', None)
        if ori is None:
            return
        
        if getattr(self, '_manual_align_active', False):
            # Manual align mode: only redraw crosshair lines (ultra-lightweight)
            for o in ['axial', 'coronal', 'sagittal']:
                self._refresh_crosshair_overlay(o)
        else:
            # Normal oblique mode: fast reslice on active view
            self._oblique_drag_fast = True
            self.render_slice(ori, preserve_camera=True)
            self._oblique_drag_fast = False
            
            # Lightweight crosshair overlay refresh on the other two views
            for o in ['axial', 'coronal', 'sagittal']:
                if o != ori:
                    self._refresh_crosshair_overlay(o)
            
            # Update 3D crosshair lines to follow the rotation
            self.update_3d_crosshair()
    
    def _oblique_drag_finish(self):
        """Called on mouse release to do a full-quality render of all views."""
        # Cancel any pending throttled render
        if hasattr(self, '_oblique_throttle_timer') and self._oblique_throttle_timer.isActive():
            self._oblique_throttle_timer.stop()
        
        self._oblique_drag_orientation = None
        self._oblique_drag_fast = False
        
        if getattr(self, '_manual_align_active', False):
            # In manual align mode: just refresh crosshair overlays (no reslicing)
            for ori in ['axial', 'coronal', 'sagittal']:
                self._refresh_crosshair_overlay(ori)
        else:
            # Normal oblique mode: full quality render of ALL orientations
            for ori in ['axial', 'coronal', 'sagittal']:
                self.render_slice(ori, preserve_camera=True)
            self.update_3d_crosshair()
    
    def _refresh_crosshair_overlay(self, orientation):
        """Redraw ONLY the crosshair overlay actors without re-reslicing the volume.
        
        This is extremely lightweight — only repositions 2D line/circle actors
        based on current oblique_angles. Used during manual align drag for
        instant visual feedback without any volume computation.
        """
        if self.volume_data is None:
            return
        if not self.crosshair_enabled:
            return
        
        # Use the unified persistent crosshair system
        self.update_2d_crosshair(orientation)


    def get_oblique_R(self):
        import math
        # axial: yaw (around Z)
        tz = math.radians(self.oblique_angles.get('axial', 0.0))
        cz, sz = math.cos(tz), math.sin(tz)
        Rz = np.array([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ])
        
        # coronal: pitch (around Y)
        ty = math.radians(self.oblique_angles.get('coronal', 0.0))
        cy, sy = math.cos(ty), math.sin(ty)
        Ry = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ])
        
        # sagittal: roll (around X)
        tx = math.radians(self.oblique_angles.get('sagittal', 0.0))
        cx, sx = math.cos(tx), math.sin(tx)
        Rx = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ])
        
        # Combined rotation matrix R
        return np.dot(Rz, np.dot(Ry, Rx))

    def _oblique_reslice_3d(self, orientation, slice_idx):
        from scipy.ndimage import map_coordinates
        import math
        
        vol = self.volume_data
        vol_z, vol_y, vol_x = vol.shape
        
        cx = float(self.crosshair_position[0])
        cy = float(self.crosshair_position[1])
        cz = float(self.crosshair_position[2])
        
        R = self.get_oblique_R()
        
        def _reslice_vol_3d(data, order=1):
            if data is None:
                return None
            bg_val = float(data.min())
            
            if orientation == 'axial':
                yy, xx = np.mgrid[0:vol_y, 0:vol_x].astype(np.float64)
                x_coords = cx + (xx - cx) * R[0, 0] + (yy - cy) * R[0, 1]
                y_coords = cy + (xx - cx) * R[1, 0] + (yy - cy) * R[1, 1]
                z_coords = cz + (xx - cx) * R[2, 0] + (yy - cy) * R[2, 1]
                
                result = map_coordinates(data, [z_coords, y_coords, x_coords],
                                         order=order, mode='constant', cval=bg_val)
                return result
                
            elif orientation == 'coronal':
                zz, xx = np.mgrid[0:vol_z, 0:vol_x].astype(np.float64)
                x_coords = cx + (xx - cx) * R[0, 0] + (zz - cz) * R[0, 2]
                y_coords = cy + (xx - cx) * R[1, 0] + (zz - cz) * R[1, 2]
                z_coords = cz + (xx - cx) * R[2, 0] + (zz - cz) * R[2, 2]
                
                result = map_coordinates(data, [z_coords, y_coords, x_coords],
                                         order=order, mode='constant', cval=bg_val)
                return np.flipud(result)
                
            else: # sagittal
                yy, zz = np.mgrid[0:vol_y, 0:vol_z].astype(np.float64)
                x_coords = cx + (zz - cz) * R[0, 2] + (yy - cy) * R[0, 1]
                y_coords = cy + (zz - cz) * R[1, 2] + (yy - cy) * R[1, 1]
                z_coords = cz + (zz - cz) * R[2, 2] + (yy - cy) * R[2, 1]
                
                result = map_coordinates(data, [z_coords, y_coords, x_coords],
                                         order=order, mode='constant', cval=bg_val)
                return result

        # During fast drag: use order=0 (nearest-neighbor) for speed, skip masks
        is_fast = getattr(self, '_oblique_drag_fast', False)
        vol_order = 0 if is_fast else 1
        
        slice_data = _reslice_vol_3d(vol, order=vol_order)
        
        if is_fast:
            # Skip mask reslicing during drag for performance
            seg_slice = None
            c1_slice = None
            c2_slice = None
        else:
            seg_slice = _reslice_vol_3d(self.segmentation_data, order=0) if self.segmentation_data is not None else None
            c1_slice = _reslice_vol_3d(self.class1_data, order=0) if self.class1_data is not None else None
            c2_slice = _reslice_vol_3d(self.class2_data, order=0) if self.class2_data is not None else None
        
        return slice_data, seg_slice, c1_slice, c2_slice

    def _crosshair_display_geometry(self, orientation):
        """Project crosshair center + axis directions into VTK display pixels."""
        import math
        if self.volume_data is None:
            return None
        renderer = getattr(self, f'{orientation}_renderer', None)
        if renderer is None:
            return None

        vol_z, vol_y, vol_x = self.volume_data.shape
        if orientation == 'axial':
            cx_w = float(self.crosshair_position[0])
            cy_w = float(self.crosshair_position[1])
            theta_h = math.radians(self.oblique_angles.get('coronal', 0.0))
            theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
        elif orientation == 'coronal':
            cx_w = float(self.crosshair_position[0])
            cy_w = float((vol_z - 1) - self.crosshair_position[2])
            theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
            theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
        else:
            cx_w = float(self.crosshair_position[2])
            cy_w = float(self.crosshair_position[1])
            theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
            theta_v = math.radians(self.oblique_angles.get('coronal', 0.0)) + math.pi / 2

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        coord.SetValue(cx_w, cy_w, 0.5)
        dp = coord.GetComputedDisplayValue(renderer)
        dx, dy = float(dp[0]), float(dp[1])
        return {
            'dx': dx, 'dy': dy,
            'cos_h': math.cos(theta_h), 'sin_h': math.sin(theta_h),
            'cos_v': math.cos(theta_v), 'sin_v': math.sin(theta_v),
            'gap': 12,
        }

    def _get_crosshair_hit(self, pos, orientation):
        """Dragonfly-style hit test: center handle, then H/V axes.

        Hit radii scale gently with zoom (P0). Returns (orientation, mode) or None.
        """
        import math
        if not self.crosshair_enabled or self.volume_data is None:
            return None
        geom = self._crosshair_display_geometry(orientation)
        if geom is None:
            return None

        widget = getattr(self, f'{orientation}_widget')
        size = widget.GetRenderWindow().GetSize()
        x = float(pos.x())
        vtk_y = float(size[1] - pos.y())
        dx, dy = geom['dx'], geom['dy']
        gap = geom['gap']
        center_r, line_tol = self._crosshair_hit_radii(orientation)

        dist_center = math.hypot(x - dx, vtk_y - dy)
        if dist_center <= center_r:
            return (orientation, 'center')

        def dist_to_axis(px, py, cos_t, sin_t):
            vx, vy = px - dx, py - dy
            return abs(vx * sin_t - vy * cos_t)

        dh = dist_to_axis(x, vtk_y, geom['cos_h'], geom['sin_h'])
        dv = dist_to_axis(x, vtk_y, geom['cos_v'], geom['sin_v'])

        # Prefer nearer axis when both qualify; require outside center gap
        if dist_center > gap:
            if dh <= line_tol and dv <= line_tol:
                return (orientation, 'h' if dh <= dv else 'v')
            if dh <= line_tol:
                return (orientation, 'h')
            if dv <= line_tol:
                return (orientation, 'v')
        return None

    @staticmethod
    def _boost_color(color, factor=1.35):
        return tuple(min(1.0, float(c) * factor) for c in color)

    def handle_crosshair_click(self, orientation, pos, mode='center', preview_only=False):
        """Handle mouse drag / place for crosshair (Dragonfly grab modes).

        mode:
          - 'center' / 'place': move both in-plane axes
          - 'h': move only the axis normal to the H line
          - 'v': move only the axis normal to the V line

        Always full-sync via updatePoint (crosshair + sliders + linked panes).
        """
        try:
            self._crosshair_drag_update(
                orientation, pos, mode=mode, preview_only=False
            )
        except Exception:
            pass

    def toggle_crosshair(self, state):
        """Enable/disable crosshair"""
        self.crosshair_enabled = (state == Qt.Checked)
        self._crosshair_hover = None
        self._crosshair_drag_mode = None
        self._is_dragging_crosshair = False

        if self.volume_data is not None:
            # Re-render all views
            for orientation in ['axial', 'coronal', 'sagittal']:
                self.render_slice(orientation, preserve_camera=True)
            self.update_3d_crosshair()

    def update_3d_crosshair(self):
        if not hasattr(self, 'view_3d_renderer') or self.volume_data is None:
            return
            
        if not hasattr(self, 'crosshair_3d_actors'):
            self.crosshair_3d_actors = {'x': vtk.vtkLineSource(), 'y': vtk.vtkLineSource(), 'z': vtk.vtkLineSource()}
            self.crosshair_3d_actor_objs = []
            colors = {'x': (1.0, 0.192, 0.192), 'y': (0.223, 1.0, 0.078), 'z': (0.121, 0.317, 1.0)}
            for axis in ['x', 'y', 'z']:
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(self.crosshair_3d_actors[axis].GetOutputPort())
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*colors[axis])
                actor.GetProperty().SetLineWidth(1)
                actor.SetPickable(False)
                self.crosshair_3d_actor_objs.append(actor)
        
        for actor in self.crosshair_3d_actor_objs:
            if self.crosshair_enabled:
                if not self.view_3d_renderer.HasViewProp(actor):
                    self.view_3d_renderer.AddActor(actor)
            else:
                self.view_3d_renderer.RemoveActor(actor)
                
        if self.crosshair_enabled:
            z, y, x = self.volume_data.shape
            cx, cy, cz = self.crosshair_position
            spx, spy, spz = self.spacing
            
            # Center of crosshair in physical coordinates
            pc = np.array([cx * spx, cy * spy, cz * spz])
            
            # Get 3D rotation matrix
            R = self.get_oblique_R()
            
            # Rotated axes (columns of R)
            rx = R[:, 0]
            ry = R[:, 1]
            rz = R[:, 2]
            
            lx = (x - 1) * spx
            ly = (y - 1) * spy
            lz = (z - 1) * spz
            max_len = max(lx, ly, lz) * 2.0
            
            p1_x = pc - max_len * rx
            p2_x = pc + max_len * rx
            
            p1_y = pc - max_len * ry
            p2_y = pc + max_len * ry
            
            p1_z = pc - max_len * rz
            p2_z = pc + max_len * rz
            
            # X line (Sagittal - Red)
            self.crosshair_3d_actors['x'].SetPoint1(p1_x[0], p1_x[1], p1_x[2])
            self.crosshair_3d_actors['x'].SetPoint2(p2_x[0], p2_x[1], p2_x[2])
            
            # Y line (Coronal - Green)
            self.crosshair_3d_actors['y'].SetPoint1(p1_y[0], p1_y[1], p1_y[2])
            self.crosshair_3d_actors['y'].SetPoint2(p2_y[0], p2_y[1], p2_y[2])
            
            # Z line (Axial - Blue)
            self.crosshair_3d_actors['z'].SetPoint1(p1_z[0], p1_z[1], p1_z[2])
            self.crosshair_3d_actors['z'].SetPoint2(p2_z[0], p2_z[1], p2_z[2])
            
        self.view_3d_widget.GetRenderWindow().Render()
    
    def on_crosshair_slider_changed(self, axis_idx, value):
        """Update crosshair position from toolbar slider and re-render all views.
        
        axis_idx: 0=X (sagittal), 1=Y (coronal), 2=Z (axial)
        """
        if self.volume_data is None:
            return
        
        newX, newY, newZ = self.crosshair_position
        if axis_idx == 0:
            newX = value
        elif axis_idx == 1:
            newY = value
        elif axis_idx == 2:
            newZ = value
            
        self.updatePoint(newX, newY, newZ)
    
    def toggle_reverse_z(self):
        """Toggle Z direction (upward/downward)"""
        self.reverse_z = self.reverse_z_btn.isChecked()
        if self.volume_data is not None:
            # Re-render all views (image reloads with reversed Z mapping)
            for orientation in ['axial', 'coronal', 'sagittal']:
                self.render_slice(orientation, preserve_camera=True)
    
    def create_crosshair_actors(self, orientation, renderer, slice_shape):
        """
        Draw full-viewport crosshair in DISPLAY coordinates so it does not
        follow/zoom with the image. Crosshair center is computed by projecting
        the world crosshair position into display (screen-pixel) space.
        """
        actors = []
        try:
            h, w = slice_shape
            vol_z, vol_y, vol_x = self.volume_data.shape
            
            z_sign = '-Z' if self.reverse_z else '+Z'
            
            # Map world crosshair position to the specific plane orientations 
            # Axial: VTK X = X, VTK Y = Y
            # Coronal: VTK X = X, VTK Y = Z_flipped
            # Sagittal: VTK X = Z, VTK Y = Y
            if orientation == 'axial':
                cx_w = float(self.crosshair_position[0])
                cy_w = float(self.crosshair_position[1])
                h_label, v_label = '+X', '+Y'
                h_color = (1.0, 0.192, 0.192)  # X: Red
                v_color = (0.223, 1.0, 0.078)  # Y: Green
            elif orientation == 'coronal':
                cx_w = float(self.crosshair_position[0])
                cy_w = float((vol_z - 1) - self.crosshair_position[2])
                h_label, v_label = '+X', z_sign
                h_color = (1.0, 0.192, 0.192)  # X: Red
                v_color = (0.121, 0.317, 1.0)  # Z: Blue
            else:  # sagittal
                cx_w = float(self.crosshair_position[2])
                cy_w = float(self.crosshair_position[1])
                h_label, v_label = z_sign, '+Y'
                h_color = (0.121, 0.317, 1.0)  # Z: Blue
                v_color = (0.223, 1.0, 0.078)  # Y: Green
            
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToWorld()
            coord.SetValue(cx_w, cy_w, 0.5)
            dp = coord.GetComputedDisplayValue(renderer)
            dx = float(dp[0])   # display X (pixels from left)
            dy = float(dp[1])   # display Y (pixels from bottom)
            
            win = renderer.GetRenderWindow().GetSize()
            vp = renderer.GetViewport()
            vp_w = (vp[2] - vp[0]) * win[0]
            vp_h = (vp[3] - vp[1]) * win[1]
            gap = 12       # fixed-pixel gap at crosshair center
            
            def make_line2d(x1, y1, x2, y2, color, lw=1):
                ls = vtk.vtkLineSource()
                ls.SetPoint1(x1, y1, 0)
                ls.SetPoint2(x2, y2, 0)
                m = vtk.vtkPolyDataMapper2D()
                m.SetInputConnection(ls.GetOutputPort())
                a = vtk.vtkActor2D()
                a.SetMapper(m)
                a.GetProperty().SetColor(*color)
                a.GetProperty().SetLineWidth(lw)
                return a
            
            def make_text2d(text, px, py, color, size=11):
                t = vtk.vtkTextActor()
                t.SetInput(text)
                t.SetPosition(px, py)
                t.GetTextProperty().SetFontSize(size)
                t.GetTextProperty().SetColor(*color)
                t.GetTextProperty().BoldOn()
                return t
            
            def make_circle2d(cx, cy, r, color, fill=False, opacity=1.0, lw=1):
                src = vtk.vtkRegularPolygonSource()
                src.SetNumberOfSides(24)
                src.SetRadius(r)
                src.SetCenter(cx, cy, 0)
                if fill:
                    src.GeneratePolygonOn()
                else:
                    src.GeneratePolygonOff()
                m = vtk.vtkPolyDataMapper2D()
                m.SetInputConnection(src.GetOutputPort())
                a = vtk.vtkActor2D()
                a.SetMapper(m)
                a.GetProperty().SetColor(*color)
                a.GetProperty().SetOpacity(opacity)
                a.GetProperty().SetLineWidth(lw)
                return a
            
            import math
            # Define labels, colors, and angles for each view orientation
            if orientation == 'axial':
                h_name, v_name = 'coronal', 'sagittal'
                h_color = (1.0, 0.192, 0.192)  # X: Red
                v_color = (0.223, 1.0, 0.078)  # Y: Green
                
                h_label_pos, h_label_neg = '+X', '-X'
                v_label_pos, v_label_neg = '-Y', '+Y'
                
                theta_h = math.radians(self.oblique_angles.get('coronal', 0.0))
                theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
                
            elif orientation == 'coronal':
                h_name, v_name = 'axial', 'sagittal'
                h_color = (1.0, 0.192, 0.192)  # X: Red
                v_color = (0.121, 0.317, 1.0)  # Z: Blue
                
                h_label_pos, h_label_neg = '+X', '-X'
                v_label_pos = '+Z' if self.reverse_z else '-Z'
                v_label_neg = '-Z' if self.reverse_z else '+Z'
                
                theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
                theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
                
            else:  # sagittal
                h_name, v_name = 'axial', 'coronal'
                h_color = (0.121, 0.317, 1.0)  # Z: Blue
                v_color = (0.223, 1.0, 0.078)  # Y: Green
                
                h_label_pos = '-Z' if self.reverse_z else '+Z'
                h_label_neg = '+Z' if self.reverse_z else '-Z'
                v_label_pos = '-Y'
                v_label_neg = '+Y'
                
                theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
                theta_v = math.radians(self.oblique_angles.get('coronal', 0.0)) + math.pi / 2

            # Compute line directions
            cos_h, sin_h = math.cos(theta_h), math.sin(theta_h)
            cos_v, sin_v = math.cos(theta_v), math.sin(theta_v)
            extent = max(vp_w, vp_h) * 0.6
            line_extent = max(vp_w, vp_h) * 2.5

            # Draw horizontal-ish line
            actors.append(make_line2d(dx - line_extent * cos_h, dy - line_extent * sin_h, dx - gap * cos_h, dy - gap * sin_h, h_color))
            actors.append(make_line2d(dx + gap * cos_h, dy + gap * sin_h, dx + line_extent * cos_h, dy + line_extent * sin_h, h_color))

            # Draw vertical-ish line
            actors.append(make_line2d(dx - line_extent * cos_v, dy - line_extent * sin_v, dx - gap * cos_v, dy - gap * sin_v, v_color))
            actors.append(make_line2d(dx + gap * cos_v, dy + gap * sin_v, dx + line_extent * cos_v, dy + line_extent * sin_v, v_color))

            # Rotation handles — Dragonfly-style curved arrows
            # Only draw when oblique handles are enabled
            if getattr(self, 'oblique_handles_enabled', False):
                handle_dist = min(extent * 0.35, 120)
                hover = getattr(self, '_oblique_hover_handle', None)
                dragging = getattr(self, '_oblique_dragging', None)

                def make_curved_arrow_2d(cx, cy, radius, start_angle, sweep_deg, color, opacity=1.0, lw=2.0, with_arrowhead=True):
                    """Create a curved arc with an arrowhead at the end (Dragonfly rotation handle)."""
                    import math as _m
                    n_pts = max(12, int(abs(sweep_deg) / 3))
                    pts = vtk.vtkPoints()
                    lines = vtk.vtkCellArray()
                    for i in range(n_pts + 1):
                        a = _m.radians(start_angle + sweep_deg * i / n_pts)
                        pts.InsertNextPoint(cx + radius * _m.cos(a), cy + radius * _m.sin(a), 0)
                    line = vtk.vtkPolyLine()
                    line.GetPointIds().SetNumberOfIds(n_pts + 1)
                    for i in range(n_pts + 1):
                        line.GetPointIds().SetId(i, i)
                    lines.InsertNextCell(line)
                    # Arrowhead triangle at the end of the arc
                    if with_arrowhead:
                        end_a = _m.radians(start_angle + sweep_deg)
                        ex, ey = cx + radius * _m.cos(end_a), cy + radius * _m.sin(end_a)
                        # Tangent direction at end (perpendicular to radius)
                        sign = 1 if sweep_deg > 0 else -1
                        tx = -sign * _m.sin(end_a)
                        ty = sign * _m.cos(end_a)
                        arrow_len = max(8, radius * 0.35)
                        # Normal (outward from center)
                        nx_d = _m.cos(end_a)
                        ny_d = _m.sin(end_a)
                        # Triangle points: tip along tangent, wings along normal
                        tip_x = ex + arrow_len * tx
                        tip_y = ey + arrow_len * ty
                        w1_x = ex + arrow_len * 0.35 * nx_d
                        w1_y = ey + arrow_len * 0.35 * ny_d
                        w2_x = ex - arrow_len * 0.35 * nx_d
                        w2_y = ey - arrow_len * 0.35 * ny_d
                        base_id = pts.GetNumberOfPoints()
                        pts.InsertNextPoint(tip_x, tip_y, 0)
                        pts.InsertNextPoint(w1_x, w1_y, 0)
                        pts.InsertNextPoint(w2_x, w2_y, 0)
                        tri_line = vtk.vtkPolyLine()
                        tri_line.GetPointIds().SetNumberOfIds(4)
                        tri_line.GetPointIds().SetId(0, base_id)
                        tri_line.GetPointIds().SetId(1, base_id + 1)
                        tri_line.GetPointIds().SetId(2, base_id + 2)
                        tri_line.GetPointIds().SetId(3, base_id)
                        lines.InsertNextCell(tri_line)
                    pd = vtk.vtkPolyData()
                    pd.SetPoints(pts)
                    pd.SetLines(lines)
                    m = vtk.vtkPolyDataMapper2D()
                    m.SetInputData(pd)
                    a = vtk.vtkActor2D()
                    a.SetMapper(m)
                    a.GetProperty().SetColor(*color)
                    a.GetProperty().SetOpacity(opacity)
                    a.GetProperty().SetLineWidth(lw)
                    return a

                for cos_val, sin_val, color, which in [
                    (cos_h, sin_h, h_color, h_name),
                    (-cos_h, -sin_h, h_color, h_name),
                    (cos_v, sin_v, v_color, v_name),
                    (-cos_v, -sin_v, v_color, v_name),
                ]:
                    hx = dx + handle_dist * cos_val
                    hy = dy + handle_dist * sin_val
                    is_active = (hover == (orientation, which)) or (dragging == (orientation, which))

                    # Base angle of this handle position relative to crosshair center
                    base_angle = math.degrees(math.atan2(sin_val, cos_val))
                    arc_radius = 16
                    arc_sweep = 120  # degrees of arc
                    arc_start = base_angle - arc_sweep / 2 + 90  # perpendicular to radial direction

                    if is_active:
                        # Glow background circle
                        actors.append(make_circle2d(hx, hy, 22, color, fill=True, opacity=0.25))
                        # Bright curved arrow
                        actors.append(make_curved_arrow_2d(hx, hy, arc_radius, arc_start, arc_sweep, color, opacity=1.0, lw=2.5))
                        # Counter-direction arrow for bidirectional hint
                        actors.append(make_curved_arrow_2d(hx, hy, arc_radius, arc_start + 180, arc_sweep, color, opacity=0.7, lw=2.0))
                    else:
                        # Dim curved arrow — subtle but visible
                        actors.append(make_curved_arrow_2d(hx, hy, arc_radius * 0.8, arc_start, arc_sweep * 0.7, color, opacity=0.35, lw=1.5))

            # Angle labels — axis letter + oblique angle
            ang_h = self.oblique_angles.get(h_name, 0.0)
            ang_v = self.oblique_angles.get(v_name, 0.0)
            
            h_txt_pos = f"{h_label_pos} ({ang_h:.1f}°)" if abs(ang_h) > 0.5 else h_label_pos
            h_txt_neg = f"{h_label_neg} ({ang_h:.1f}°)" if abs(ang_h) > 0.5 else h_label_neg
            v_txt_pos = f"{v_label_pos} ({ang_v:.1f}°)" if abs(ang_v) > 0.5 else v_label_pos
            v_txt_neg = f"{v_label_neg} ({ang_v:.1f}°)" if abs(ang_v) > 0.5 else v_label_neg

            # Calculate boundary intersections for rotated/oblique lines
            def get_border_intersection(dx, dy, cos_val, sin_val, vp_w, vp_h):
                t_candidates = []
                if abs(cos_val) > 1e-5:
                    # Left border (x = 0)
                    t = -dx / cos_val
                    y = dy + t * sin_val
                    if 0 <= y <= vp_h:
                        t_candidates.append((t, 0.0, y))
                    # Right border (x = vp_w)
                    t = (vp_w - dx) / cos_val
                    y = dy + t * sin_val
                    if 0 <= y <= vp_h:
                        t_candidates.append((t, vp_w, y))
                if abs(sin_val) > 1e-5:
                    # Bottom border (y = 0)
                    t = -dy / sin_val
                    x = dx + t * cos_val
                    if 0 <= x <= vp_w:
                        t_candidates.append((t, x, 0.0))
                    # Top border (y = vp_h)
                    t = (vp_h - dy) / sin_val
                    x = dx + t * cos_val
                    if 0 <= x <= vp_w:
                        t_candidates.append((t, x, vp_h))
                
                unique = []
                seen = set()
                for item in t_candidates:
                    rounded = (round(item[0], 2), round(item[1], 2), round(item[2], 2))
                    if rounded not in seen:
                        seen.add(rounded)
                        unique.append(item)
                unique.sort(key=lambda item: item[0])
                
                neg_pt, pos_pt = None, None
                for t, x, y in unique:
                    if t < -0.1:
                        neg_pt = (x, y)
                    elif t > 0.1:
                        pos_pt = (x, y)
                        
                if neg_pt is None:
                    neg_pt = (dx - 200.0 * cos_val, dy - 200.0 * sin_val)
                if pos_pt is None:
                    pos_pt = (dx + 200.0 * cos_val, dy + 200.0 * sin_val)
                    
                nx = max(5.0, min(neg_pt[0], vp_w - 60.0))
                ny = max(5.0, min(neg_pt[1], vp_h - 20.0))
                px = max(5.0, min(pos_pt[0], vp_w - 60.0))
                py = max(5.0, min(pos_pt[1], vp_h - 20.0))
                return (nx, ny), (px, py)

            h_neg_pt, h_pos_pt = get_border_intersection(dx, dy, cos_h, sin_h, vp_w, vp_h)
            v_neg_pt, v_pos_pt = get_border_intersection(dx, dy, cos_v, sin_v, vp_w, vp_h)

            actors.append(make_text2d(h_txt_neg, h_neg_pt[0], h_neg_pt[1], h_color))
            actors.append(make_text2d(h_txt_pos, h_pos_pt[0], h_pos_pt[1], h_color))
            actors.append(make_text2d(v_txt_neg, v_neg_pt[0], v_neg_pt[1], v_color))
            actors.append(make_text2d(v_txt_pos, v_pos_pt[0], v_pos_pt[1], v_color))
            
        except Exception:
            pass
        return actors



    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self._handle_volume_load(file_path)

    def load_volume(self, from_folder=False):
        """Load 16-bit volume data"""
        if from_folder:
            file_path = QFileDialog.getExistingDirectory(
                self, "Select Folder Containing Image Files"
            )
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image Volume", "", "Image Files (*.tif *.tiff *.raw *.bin)"
            )
        
        if file_path:
            self._handle_volume_load(file_path)

    def load_volume_from_path(self, file_path, mask_bump_path=None, mask_void_path=None):
        """Programmatic load (Batch Review / Online deep-link).

        Optionally load combined bump/void masks after volume finishes.
        """
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(
                self,
                "Load volume",
                f"Volume path not found:\n{file_path or '(empty)'}",
            )
            return False
        # Stash masks to apply in on_volume_loaded
        self._pending_mask_bump = mask_bump_path or ""
        self._pending_mask_void = mask_void_path or ""
        self._handle_volume_load(file_path)
        return True

    # ------------------------------------------------------------------
    # Batch Review → Full Viewer parity (volume + masks + MES + B2B + map)
    # ------------------------------------------------------------------
    @staticmethod
    def resolve_batch_run_paths(run: dict):
        """Resolve volume / bump / void paths from a FOV run dict (DB + artifacts)."""
        run = run or {}
        results = run.get("results_dir") or ""
        arts = {
            a.get("kind"): a.get("path")
            for a in (run.get("artifacts") or [])
            if isinstance(a, dict) and a.get("path")
        }

        vol = ""
        for cand in (
            arts.get("enhanced_volume"),
            os.path.join(results, "enhanced_volume", "Enhanced_Volume.tif") if results else "",
            os.path.join(results, "enhanced_volume", "Enhanced_Volume.tiff") if results else "",
            run.get("input_path") or "",
            run.get("host_path") or "",
        ):
            if cand and os.path.isfile(cand):
                vol = cand
                break

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

    @staticmethod
    def _normalize_mes_stat_row(s: dict, row_fallback: int = 0) -> dict:
        """Normalize DB / CSV MES object row → Viewer object_stats schema."""
        if not isinstance(s, dict):
            return {}

        def _f(key, default=0.0):
            try:
                v = s.get(key, default)
                if v is None or v == "":
                    return default
                return float(v)
            except (TypeError, ValueError):
                return default

        def _i(key, default=0):
            try:
                v = s.get(key, default)
                if v is None or v == "":
                    return default
                return int(float(v))
            except (TypeError, ValueError):
                return default

        ratio = _f("ratio", 0.0)
        # CSV sometimes stores percent 0..100; DB/online use fraction 0..1
        if ratio > 1.0 and ratio <= 100.0 and "ratio_pct" not in s:
            # Heuristic: values like 12.5 mean 12.5% if no explicit fraction
            # Prefer fraction if clearly already fraction
            pass
        if ratio > 1.5:  # almost certainly percent
            ratio = ratio / 100.0

        jud = s.get("judgment")
        if jud in (1, 8):
            jud = "NG" if jud == 1 else "OK"
        elif jud is None:
            jud = ""
        else:
            jud = str(jud)

        gr = s.get("grid_row", s.get("Grid_row", -1))
        gc = s.get("grid_col", s.get("Grid_col", -1))
        try:
            gr = int(float(gr)) if gr not in (None, "", "?") else -1
        except (TypeError, ValueError):
            gr = -1
        try:
            gc = int(float(gc)) if gc not in (None, "", "?") else -1
        except (TypeError, ValueError):
            gc = -1

        label = _i("label", 0)
        return {
            "row_id": _i("row_id", row_fallback) or row_fallback,
            "label": label,
            "layer_name": str(s.get("layer_name") or s.get("Layer") or ""),
            "grid_row": gr,
            "grid_col": gc,
            "soh": _f("soh", 0.0),
            "c1_volume": _f("c1_volume", 0.0),
            "c2_volume": _f("c2_volume", 0.0),
            "ratio": ratio,
            "judgment": jud,
            "pitch_x": _f("pitch_x", 0.0),
            "pitch_y": _f("pitch_y", 0.0),
            "z_min": _i("z_min", 0),
            "z_max": _i("z_max", 0),
            "y_min": _i("y_min", 0),
            "y_max": _i("y_max", 0),
            "x_min": _i("x_min", 0),
            "x_max": _i("x_max", 0),
            "centroid_z": _f("centroid_z", 0.0),
            "centroid_y": _f("centroid_y", 0.0),
            "centroid_x": _f("centroid_x", 0.0),
        }

    @classmethod
    def _parse_mes_object_statistics_csv(cls, path: str):
        """Parse Online/Viewer object_statistics.csv into object_stats list."""
        import csv
        import re

        if not path or not os.path.isfile(path):
            return []
        stats = []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            # Normalize header keys
            for i, raw in enumerate(reader):
                # Map flexible headers → internal keys
                lower = {(k or "").strip().lower(): (k or "") for k in raw.keys()}

                def pick(*names, default=""):
                    for n in names:
                        k = lower.get(n.lower())
                        if k is not None and raw.get(k) not in (None, ""):
                            return raw.get(k)
                    return default

                bump = str(pick("Bump ID", "bump id", "bump_id", default=""))
                gr, gc = -1, -1
                m = re.search(r"\(\s*([-\d]+)\s*,\s*([-\d]+)\s*\)", bump)
                if m:
                    gr, gc = int(m.group(1)), int(m.group(2))

                ratio_raw = pick(
                    "Ratio (C2/(C1+C2))", "Ratio (%)", "ratio", "Ratio", default="0"
                )
                ratio_s = str(ratio_raw).replace("%", "").strip()
                try:
                    ratio_v = float(ratio_s) if ratio_s and ratio_s.upper() != "N/A" else 0.0
                except ValueError:
                    ratio_v = 0.0
                # Viewer export writes percent; Online write_python_format_csv writes percent too
                if ratio_v > 1.5:
                    ratio_v = ratio_v / 100.0

                row = {
                    "row_id": pick("#", "row_id", default=i + 1),
                    "layer_name": pick("Layer", "layer_name", default=""),
                    "grid_row": gr,
                    "grid_col": gc,
                    "soh": pick("B. H", "soh", default=0),
                    "c1_volume": pick("B. V", "c1_volume", default=0),
                    "c2_volume": pick("V. V", "c2_volume", default=0),
                    "ratio": ratio_v,
                    "judgment": pick("Judgment", "judgment", default=""),
                    "pitch_x": pick("Gap X (um)", "Gap X", "pitch_x", default=0),
                    "pitch_y": pick("Gap Y (um)", "Gap Y", "pitch_y", default=0),
                    "z_min": pick("Z_min", "z_min", default=0),
                    "z_max": pick("Z_max", "z_max", default=0),
                    "y_min": pick("Y_min", "y_min", default=0),
                    "y_max": pick("Y_max", "y_max", default=0),
                    "x_min": pick("X_min", "x_min", default=0),
                    "x_max": pick("X_max", "x_max", default=0),
                    "centroid_z": pick("Centroid_Z", "centroid_z", default=0),
                    "centroid_y": pick("Centroid_Y", "centroid_y", default=0),
                    "centroid_x": pick("Centroid_X", "centroid_x", default=0),
                    "label": pick("label", "Label", default=0),
                }
                stats.append(cls._normalize_mes_stat_row(row, row_fallback=i + 1))
        return stats

    @classmethod
    def extract_mes_stats_from_run(cls, run: dict):
        """MES objects: prefer DB rows, fallback object_statistics.csv on disk."""
        run = run or {}
        objs = run.get("mes_objects") or []
        stats = []
        if objs:
            for i, s in enumerate(objs):
                if isinstance(s, dict):
                    stats.append(cls._normalize_mes_stat_row(s, row_fallback=i + 1))
            if stats:
                return stats

        results = run.get("results_dir") or ""
        arts = {
            a.get("kind"): a.get("path")
            for a in (run.get("artifacts") or [])
            if isinstance(a, dict) and a.get("path")
        }
        for cand in (
            arts.get("csv_mes"),
            arts.get("mes_csv"),
            os.path.join(results, "object_statistics.csv") if results else "",
        ):
            if cand and os.path.isfile(cand):
                parsed = cls._parse_mes_object_statistics_csv(cand)
                if parsed:
                    return parsed
        return []

    @staticmethod
    def extract_b2b_rows_from_run(run: dict):
        """B2B gap/summary rows from artifacts or Results folder CSV."""
        run = run or {}
        results = run.get("results_dir") or ""
        arts = {
            a.get("kind"): a.get("path")
            for a in (run.get("artifacts") or [])
            if isinstance(a, dict) and a.get("path")
        }
        candidates = [
            arts.get("csv_b2b"),
            arts.get("b2b_csv"),
            os.path.join(results, "boundary_gap_all_layers.csv") if results else "",
            os.path.join(results, "boundary_summary_all_layers.csv") if results else "",
            os.path.join(results, "boundary_gap.csv") if results else "",
            os.path.join(results, "boundary_summary.csv") if results else "",
        ]
        try:
            from inno3d.core.bumpvoid_b2b import read_boundary_csv
        except Exception as e:
            print(f"[Viewer] B2B import failed: {e}")
            return []

        for path in candidates:
            if path and os.path.isfile(path):
                try:
                    rows = read_boundary_csv(path)
                    if rows:
                        return rows
                except Exception as e:
                    print(f"[Viewer] B2B CSV read failed ({path}): {e}")
        return []

    def show_stats_panel_for_review(self, visible: bool = True, collapse_tools: bool = True):
        """Show STATISTICS (MES/B2B) panel — same chrome Online uses.

        Offline/Batch Review default keeps ``stats_panel`` hidden; Open Full Viewer
        must call this or the tables fill but stay invisible.
        """
        visible = bool(visible)
        if hasattr(self, "stats_panel") and self.stats_panel is not None:
            self.stats_panel.setVisible(visible)
        # Online log stays Offline-style (no live TCP log for batch replay)
        if hasattr(self, "set_online_log_visible"):
            try:
                self.set_online_log_visible(False)
            except Exception:
                pass
        if visible and collapse_tools and self.volume_data is not None:
            if hasattr(self, "align_tools_host"):
                self.align_tools_host.setVisible(True)
            # More room for MPR + stats (user can reopen tools via rail)
            if hasattr(self, "set_align_tools_expanded"):
                try:
                    self.set_align_tools_expanded(False, remember=False)
                except TypeError:
                    self.set_align_tools_expanded(False)
        if visible and hasattr(self, "_rebalance_main_splitter_for_tools"):
            expanded = bool(getattr(self, "_align_tools_expanded", True))
            try:
                self._rebalance_main_splitter_for_tools(expanded)
            except Exception:
                pass
            # Force a readable stats width even if splitter had 0
            try:
                sp = getattr(self, "_main_splitter", None)
                if sp is not None and self.stats_panel is not None and self.stats_panel.isVisible():
                    sizes = sp.sizes()
                    if len(sizes) >= 3 and sizes[2] < 260:
                        total = sum(sizes) or max(sp.width(), 1600)
                        tools_w = sizes[1] if sizes[1] > 0 else 28
                        stats_w = max(320, min(420, total // 4))
                        grid_w = max(480, total - tools_w - stats_w)
                        sp.setSizes([grid_w, tools_w, stats_w])
            except Exception:
                pass
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(50, getattr(self, "_reposition_visible_overlays", lambda: None))

    def load_batch_review_run(self, run: dict):
        """Full parity deep-link from Batch Review → volume + masks + MES + B2B.

        Volume load is async; MES/B2B tables and label mapping are applied after
        volume + masks land (``_apply_pending_batch_replay_if_any``).
        """
        run = run or {}
        vol, bump, void = self.resolve_batch_run_paths(run)
        if not vol:
            QMessageBox.warning(
                self,
                "Open full Viewer",
                "No volume file found on disk for this run.\n\n"
                f"input_path: {run.get('input_path') or '—'}\n"
                f"results: {run.get('results_dir') or '—'}\n\n"
                "DB only stores paths — file may be missing or on another PC.",
            )
            return False

        mes_stats = self.extract_mes_stats_from_run(run)
        b2b_rows = self.extract_b2b_rows_from_run(run)

        # Show STATISTICS panel immediately (was hidden offline — looked "no table")
        self.show_stats_panel_for_review(True, collapse_tools=False)
        if hasattr(self, "set_stats_info"):
            self.set_stats_info(
                f"Loading Batch FOV · Chip({run.get('chip_col')},{run.get('chip_row')}) "
                f"P{run.get('fov_index') or '?'} · MES {len(mes_stats)} · B2B {len(b2b_rows)}…"
            )

        # Stash for post-load application (after masks)
        self._pending_batch_replay = {
            "run_id": run.get("run_id") or "",
            "mes_stats": mes_stats,
            "b2b_rows": b2b_rows,
            "source_label": (
                f"Batch · Chip({run.get('chip_col')},{run.get('chip_row')}) "
                f"P{run.get('fov_index') or '?'}"
            ),
            "chip_col": run.get("chip_col"),
            "chip_row": run.get("chip_row"),
            "fov_index": run.get("fov_index"),
        }

        ok = self.load_volume_from_path(
            vol,
            mask_bump_path=bump if bump and os.path.isfile(bump) else None,
            mask_void_path=void if void and os.path.isfile(void) else None,
        )
        if not ok:
            self._pending_batch_replay = None
        return ok

    def _assign_mes_labels_from_mask(self, stats):
        """Rebuild labeled_class1 from bump mask and map each MES row → label.

        Production MES CSV/DB often omit instance ``label``; without it table→
        image highlight cannot work. Match by centroid (then bbox center).
        """
        import numpy as np

        stats = list(stats or [])
        c1 = getattr(self, "class1_data", None)
        if c1 is None or self.volume_data is None or not stats:
            return stats

        try:
            from skimage.measure import label as sk_label
        except Exception:
            try:
                from scipy.ndimage import label as _scipy_label

                def sk_label(mask):
                    lab, _ = _scipy_label(mask)
                    return lab
            except Exception as e:
                print(f"[Viewer] cannot label mask for MES map: {e}")
                return stats

        try:
            binary = (np.asarray(c1) > 0).astype(np.uint8)
            labeled = sk_label(binary).astype(np.uint32)
            self.labeled_class1_data = labeled
            Z, Y, X = labeled.shape

            def _lookup(z, y, x):
                z = int(round(z))
                y = int(round(y))
                x = int(round(x))
                if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
                    return int(labeled[z, y, x])
                return 0

            matched = 0
            for s in stats:
                lab = int(s.get("label") or 0)
                if lab > 0:
                    matched += 1
                    continue
                lab = _lookup(
                    s.get("centroid_z", 0),
                    s.get("centroid_y", 0),
                    s.get("centroid_x", 0),
                )
                if lab <= 0:
                    # Bbox center fallback
                    lab = _lookup(
                        (float(s.get("z_min", 0)) + float(s.get("z_max", 0))) * 0.5,
                        (float(s.get("y_min", 0)) + float(s.get("y_max", 0))) * 0.5,
                        (float(s.get("x_min", 0)) + float(s.get("x_max", 0))) * 0.5,
                    )
                if lab <= 0:
                    # Small search around centroid in XY on that Z
                    z0 = int(round(float(s.get("centroid_z", 0))))
                    y0 = int(round(float(s.get("centroid_y", 0))))
                    x0 = int(round(float(s.get("centroid_x", 0))))
                    if 0 <= z0 < Z:
                        for rad in (2, 5, 10):
                            y1, y2 = max(0, y0 - rad), min(Y, y0 + rad + 1)
                            x1, x2 = max(0, x0 - rad), min(X, x0 + rad + 1)
                            patch = labeled[z0, y1:y2, x1:x2]
                            vals = patch[patch > 0]
                            if vals.size:
                                # mode
                                lab = int(np.bincount(vals.ravel()).argmax())
                                break
                if lab > 0:
                    s["label"] = lab
                    matched += 1
            print(f"[Viewer] MES label map: {matched}/{len(stats)} objects matched")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[Viewer] MES label assign failed: {e}")
        return stats

    def _apply_pending_batch_replay_if_any(self):
        """After volume+masks: fill MES + B2B tables and enable table→image map."""
        pending = getattr(self, "_pending_batch_replay", None)
        self._pending_batch_replay = None
        if not pending:
            return

        try:
            # CRITICAL: panel is hidden offline — must show or user sees no tables
            self.show_stats_panel_for_review(True, collapse_tools=True)

            # Clear prior picks / overlays
            try:
                self.clear_object_selection()
            except Exception:
                pass
            self._measurement_start_slice = 0

            mes_stats = list(pending.get("mes_stats") or [])
            b2b_rows = list(pending.get("b2b_rows") or [])
            src = pending.get("source_label") or "Batch Review"

            if mes_stats:
                mes_stats = self._assign_mes_labels_from_mask(mes_stats)
                self.populate_object_stats_table(
                    mes_stats, source_label=src
                )
            else:
                # Keep empty table honest
                self.object_stats = []
                if getattr(self, "object_stats_table", None) is not None:
                    self.object_stats_table.setRowCount(0)
                if hasattr(self, "set_stats_info"):
                    self.set_stats_info(f"No MES objects in DB/CSV · {src}")

            if b2b_rows:
                self.populate_b2b_table(b2b_rows, source_label=src)
            else:
                self.b2b_stats = []
                if getattr(self, "b2b_table", None) is not None:
                    self.b2b_table.setRowCount(0)
                if getattr(self, "b2b_info_label", None) is not None:
                    self.b2b_info_label.setText(
                        f"No B2B CSV on disk · {src}"
                    )

            # Refresh MPR so grid labels / overlays use new stats
            if self.volume_data is not None:
                for ori in ("axial", "coronal", "sagittal"):
                    try:
                        self.render_slice(ori, preserve_camera=True)
                    except Exception:
                        pass

            n_mes = len(mes_stats)
            n_b2b = len(b2b_rows)
            print(
                f"[Viewer] Batch replay ready · MES={n_mes} B2B={n_b2b} · {src}"
            )
            # Focus STATISTICS panel on MES
            if hasattr(self, "stats_tabs"):
                try:
                    self.stats_tabs.setCurrentIndex(0)
                except Exception:
                    pass
            # Second rebalance after rows populate (table needs width)
            self.show_stats_panel_for_review(True, collapse_tools=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[Viewer] batch replay apply failed: {e}")

    def _apply_pending_masks_if_any(self):
        """Load offline mask TIFFs queued by load_volume_from_path."""
        bump_p = getattr(self, "_pending_mask_bump", None) or ""
        void_p = getattr(self, "_pending_mask_void", None) or ""
        self._pending_mask_bump = ""
        self._pending_mask_void = ""
        if not bump_p and not void_p:
            # Still apply MES/B2B even without masks (navigate-by-centroid works)
            try:
                self._apply_pending_batch_replay_if_any()
            except Exception as e:
                print(f"[Viewer] batch replay (no mask): {e}")
            return
        if self.volume_data is None:
            return
        try:
            import tifffile
            import numpy as np

            def _load_mask(path):
                if not path or not os.path.isfile(path):
                    return None
                m = np.asarray(tifffile.imread(path))
                if m.ndim == 2:
                    m = m[np.newaxis]
                if m.ndim == 4:
                    m = m[..., 0] if m.shape[-1] <= 4 else m[:, 0]
                # Align to volume
                out = np.zeros(self.volume_data.shape[:3], dtype=np.uint8)
                z = min(m.shape[0], out.shape[0])
                y = min(m.shape[1], out.shape[1])
                x = min(m.shape[2], out.shape[2])
                out[:z, :y, :x] = (m[:z, :y, :x] > 0).astype(np.uint8)
                return out

            c1 = _load_mask(bump_p)
            c2 = _load_mask(void_p)
            if c1 is not None:
                self.class1_data = c1
            if c2 is not None:
                self.class2_data = c2
            if c1 is not None or c2 is not None:
                if hasattr(self, "merge_class_masks"):
                    self.merge_class_masks()
                # Enable overlays
                for orientation in ("axial", "coronal", "sagittal"):
                    c1_check = getattr(self, f"{orientation}_overlay_c1", None)
                    if c1_check is not None and c1 is not None:
                        c1_check.blockSignals(True)
                        c1_check.setChecked(True)
                        c1_check.blockSignals(False)
                    c2_check = getattr(self, f"{orientation}_overlay_c2", None)
                    if c2_check is not None and c2 is not None:
                        c2_check.blockSignals(True)
                        c2_check.setChecked(True)
                        c2_check.blockSignals(False)
                if hasattr(self, "update_all_views"):
                    self.update_all_views()
                elif hasattr(self, "render_slice"):
                    for o in ("axial", "coronal", "sagittal"):
                        try:
                            self.render_slice(o)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[Viewer] pending mask load failed: {e}")
        finally:
            # Full parity: MES + B2B after masks (label map needs class1)
            try:
                self._apply_pending_batch_replay_if_any()
            except Exception as e:
                print(f"[Viewer] batch replay after mask: {e}")
            
    def _handle_volume_load(self, file_path):
        from pathlib import Path
        path = Path(file_path)
        raw_shape = None
        raw_dtype = None
        raw_offset = 0
        
        if path.suffix.lower() in ['.raw', '.bin']:
            from inno3d.core.view_support import RawImportDialog
            dlg = RawImportDialog(path.name, self)
            if dlg.exec_():
                raw_shape, raw_dtype, raw_offset = dlg.get_data()
            else:
                return # user cancelled
                
        from inno3d.core.view_support import LoadVolumeThread
        self.pending_load_path = file_path
        self.load_thread = LoadVolumeThread(file_path, downsample_factor=1, raw_shape=raw_shape, raw_dtype=raw_dtype, raw_offset=raw_offset)
        
        self.progress = QProgressDialog("Loading volume...", "Cancel", 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(True)
        self.progress.setMinimumWidth(400)
        self.progress.setStyleSheet(f"""
            QProgressDialog {{
                background: {SemiconductorTheme.BG_PANEL};
                color: {SemiconductorTheme.TEXT_PRIMARY};
            }}
            QProgressBar {{
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 5px;
                background: {SemiconductorTheme.BG_DARK};
                height: 18px;
                text-align: center;
                color: {SemiconductorTheme.TEXT_PRIMARY};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {SemiconductorTheme.ACCENT_PRIMARY},
                    stop:1 #00b4d8);
                border-radius: 4px;
            }}
            QLabel {{ color: {SemiconductorTheme.TEXT_PRIMARY}; }}
            QPushButton {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px;
                padding: 4px 16px;
            }}
            QPushButton:hover {{ background: {SemiconductorTheme.ACCENT_PRIMARY}; }}
        """)
        self.progress.show()
        
        # Smooth progress animation: interpolates between target values
        self._progress_target = 0
        self._progress_current = 0.0
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(30)  # ~33 fps smooth animation
        self._progress_timer.timeout.connect(self._animate_progress)
        self._progress_timer.start()
        
        self.load_thread.progress.connect(self._update_progress_target)
        self.load_thread.finished.connect(self.on_volume_loaded)
        self.load_thread.start()
    
    def _update_progress_target(self, value, message):
        """Receive target progress from loading thread — animation will smoothly catch up."""
        self._progress_target = value
        if hasattr(self, 'progress') and self.progress is not None:
            self.progress.setLabelText(message)
            if self.progress.wasCanceled():
                if hasattr(self, 'load_thread') and self.load_thread is not None:
                    self.load_thread.terminate()
    
    def _animate_progress(self):
        """Smooth progress bar animation — runs at 33fps, interpolates toward target."""
        if not hasattr(self, 'progress') or self.progress is None:
            if hasattr(self, '_progress_timer'):
                self._progress_timer.stop()
            return
        
        target = self._progress_target
        current = self._progress_current
        
        # Ease toward target: move 20% of remaining distance per frame
        # This creates a smooth deceleration effect
        if abs(target - current) < 0.5:
            self._progress_current = float(target)
        else:
            self._progress_current = current + (target - current) * 0.2
        
        self.progress.setValue(int(self._progress_current))

    def on_volume_loaded(self, data, error):
        # Stop animation timer
        if hasattr(self, '_progress_timer'):
            self._progress_timer.stop()
        if hasattr(self, 'progress') and self.progress is not None:
            self.progress.setValue(100)  # Ensure bar reaches 100%
            self.progress.close()
        if error:
            # Drop pending Batch Review replay so next load is clean
            self._pending_batch_replay = None
            self._pending_mask_bump = ""
            self._pending_mask_void = ""
            QMessageBox.critical(self, "Error", f"Failed to load volume: {error}")
            return
            
        # Reset alignment backups and camera init state
        self.original_volume_data = None
        self._camera_initialized = False
        self.original_class1_data = None
        self.original_class2_data = None
        self.original_labeled_class1_data = None
        self.original_labeled_class2_data = None
        
        if hasattr(self, 'btn_reset_align'):
            self.btn_reset_align.setEnabled(False)
        if hasattr(self, 'reset_align_btn'):
            self.reset_align_btn.setEnabled(False)
        
        # Invalidate caches from previous volume
        self._cached_slice_actors = {}
        if hasattr(self, '_persistent_crosshair'):
            self._persistent_crosshair = {'axial': {}, 'coronal': {}, 'sagittal': {}}
            
        self.volume_data = data
        # Handle multi-channel images (e.g., RGBA): convert to single channel
        if data.ndim == 4:
            data = data[:, :, :, 0]
            self.volume_data = data
        z, y, x = data.shape
        
        # Initialize crosshair at center
        self.crosshair_position = [x // 2, y // 2, z // 2]
        
        # Update crosshair position sliders ranges
        for axis, dim, ctr in [('x', x-1, x//2), ('y', y-1, y//2), ('z', z-1, z//2)]:
            sl = getattr(self, f'crosshair_slider_{axis}', None)
            if sl:
                sl.blockSignals(True)
                sl.setMaximum(dim)
                sl.setValue(ctr)
                sl.blockSignals(False)
        
        # Fast min/max: avoid scanning the entire 32GB array.
        # For uint8/uint16, use subsample-based estimation (1M random voxels)
        # instead of full scan which takes 10-20s on massive volumes.
        total_voxels = data.size
        if total_voxels > 10_000_000:  # > 10M voxels: use subsampling
            # Sample up to 2M evenly-spaced voxels for fast min/max estimation
            step = max(1, total_voxels // 2_000_000)
            flat_view = data.ravel()
            sampled = flat_view[::step]
            data_min = float(sampled.min())
            data_max = float(sampled.max())
        else:
            data_min = float(data.min())
            data_max = float(data.max())
        window = data_max - data_min
        level = (data_max + data_min) / 2.0
        
        self.info_label.setText(f"Volume: ({z},{y},{x}) [{data_min}-{data_max}]")
        
        # Update 3D W/L spinboxes + plotted/data range (Dragonfly Window Leveling)
        self._set_data_range_ui(data_min, data_max)

        self.view_3d_min_spin.blockSignals(True)
        self.view_3d_max_spin.blockSignals(True)
        self.view_3d_min_spin.setValue(int(data_min))
        self.view_3d_max_spin.setValue(int(data_max))
        self.view_3d_min_spin.blockSignals(False)
        self.view_3d_max_spin.blockSignals(False)

        # Histogram bars normalised 0..1 → full data range
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_range(0.0, 1.0)
        self._update_wl_readout()

        # MPR W/L (right sidebar): linked to all 3 planes
        for orientation in ['axial', 'coronal', 'sagittal']:
            self.window_level[orientation] = (window, level)
        self._sync_mpr_wl_ui(int(data_min), int(data_max), data_min, data_max)
            
        # Reset custom spacing when new volume is loaded
        self.custom_spacing = None
        if hasattr(self, 'spacing_x_spin'):
            self.spacing_x_spin.setValue(self.spacing[0])
            self.spacing_y_spin.setValue(self.spacing[1])
            self.spacing_z_spin.setValue(self.spacing[2])
            
        for orientation in ['axial', 'coronal', 'sagittal']:
            self.camera_state[orientation] = None
            
        self.update_all_views()

        # Batch Review deep-link: apply mask TIFFs after volume is ready
        try:
            self._apply_pending_masks_if_any()
        except Exception as e:
            print(f"[Viewer] apply pending masks: {e}")


    def update_pixel_value(self, orientation, pos):
        try:
            widget = getattr(self, f'{orientation}_widget')
            renderer = getattr(self, f'{orientation}_renderer')
            
            x, y = pos.x(), pos.y()
            size = widget.GetRenderWindow().GetSize()
            vtk_y = size[1] - y
            
            picker = vtk.vtkWorldPointPicker()
            picker.Pick(x, vtk_y, 0, renderer)
            world_pos = picker.GetPickPosition()
            
            px, py = int(world_pos[0]), int(world_pos[1])
            slice_idx = self.current_slices[orientation]
            
            if orientation == 'axial':
                slice_data = self.volume_data[slice_idx, :, :]
            elif orientation == 'coronal':
                slice_data = self.volume_data[:, slice_idx, :]
            else:
                slice_data = np.rot90(self.volume_data[:, :, slice_idx])
            
            h, w = slice_data.shape
            
            if 0 <= px < w and 0 <= py < h:
                pixel_value = slice_data[py, px]
                label = getattr(self, f'{orientation}_pixel_label')
                label.setText(f"Pixel ({px},{py}): {pixel_value}")
            else:
                label = getattr(self, f'{orientation}_pixel_label')
                label.setText("Pixel: --")
        except:
            pass
    
    def load_segmentation(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Segmentation TIFF", "", "TIFF Files (*.tif *.tiff)"
        )
        
        if file_path:
            try:
                data = io.imread(file_path)
                if data.ndim == 2:
                    directory = Path(file_path).parent
                    all_files = sorted(glob.glob(str(directory / "*.tif*")))
                    if len(all_files) > 1:
                        data = np.array([io.imread(f) for f in all_files])
                    else:
                        data = data[np.newaxis, :, :]
                
                self.segmentation_data = data
                
                if self.volume_data is not None:
                    for orientation in ['axial', 'coronal', 'sagittal']:
                        self.render_slice(orientation)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
    
    def update_all_views(self):
        if self.volume_data is None:
            return
        
        z, y, x = self.volume_data.shape
        
        self.axial_slice_slider.setMaximum(z - 1)
        self.axial_slice_slider.setValue(z // 2)
        self.current_slices['axial'] = z // 2
        
        # Coronal(Y): slices along Y axis, so max = y-1
        self.coronal_slice_slider.setMaximum(y - 1)
        self.coronal_slice_slider.setValue(y // 2)
        self.current_slices['coronal'] = y // 2
        
        # Sagittal(X): slices along X axis, so max = x-1
        self.sagittal_slice_slider.setMaximum(x - 1)
        self.sagittal_slice_slider.setValue(x // 2)
        self.current_slices['sagittal'] = x // 2
        
        # Init crosshair at center
        self.crosshair_position = [x // 2, y // 2, z // 2]
        for orientation in ['axial', 'coronal', 'sagittal']:
            self.render_slice(orientation)
        
        self.render_3d()
        self.update_3d_crosshair()
        
        # Show VIEW TOOLS host when volume is loaded
        # Online → collapsed rail by default (or user session pref); Manual → full sidebar
        if hasattr(self, "align_tools_host"):
            self.align_tools_host.setVisible(True)
            if getattr(self, "_online_viewer_mode", False):
                self.set_align_tools_expanded(
                    getattr(self, "_align_tools_user_expanded", False),
                    remember=False,
                )
            else:
                self.set_align_tools_expanded(True, remember=False)
        elif hasattr(self, "align_sidebar"):
            self.align_sidebar.setVisible(True)

        # Force MPR tool strips (C1/C2/FS/Reset) back after Online dialogs hid them
        QTimer.singleShot(50, self._reposition_visible_overlays)
        QTimer.singleShot(200, self._reposition_visible_overlays)
    
    def update_slice(self, orientation, value):
        """Slider moved → update crosshair axis for that plane (full linked sync).

        axial    slider → Z
        coronal  slider → Y
        sagittal slider → X

        Coalesces ultra-fast drag events (~60 fps) but never leaves crosshair
        and sliders on different indices.
        """
        if self.volume_data is None:
            return

        if not hasattr(self, '_slice_update_pending'):
            self._slice_update_pending = {}
            self._slice_throttle_timer = QTimer(self)
            self._slice_throttle_timer.setSingleShot(True)
            self._slice_throttle_timer.setInterval(16)
            self._slice_throttle_timer.timeout.connect(self._flush_slice_update)

        self._slice_update_pending[orientation] = int(value)

        # First event: apply immediately so UI feels locked to the thumb
        if not self._slice_throttle_timer.isActive():
            self._flush_slice_update()
            # Coalesce any events that arrive in the next frame
            self._slice_throttle_timer.start()
    
    def _flush_slice_update(self):
        """Apply latest pending slider values through updatePoint (single source of truth)."""
        if not hasattr(self, '_slice_update_pending') or not self._slice_update_pending:
            return

        pending = self._slice_update_pending.copy()
        self._slice_update_pending.clear()

        newX, newY, newZ = self.crosshair_position
        for orientation, value in pending.items():
            if orientation == 'axial':
                newZ = value
            elif orientation == 'coronal':
                newY = value
            elif orientation == 'sagittal':
                newX = value

        self.updatePoint(newX, newY, newZ)

        # If more slider events arrived during render, flush again next tick
        if self._slice_update_pending:
            self._slice_throttle_timer.start()
    
    def save_camera_state(self, orientation):
        renderer = getattr(self, f'{orientation}_renderer')
        camera = renderer.GetActiveCamera()
        
        self.camera_state[orientation] = {
            'position': camera.GetPosition(),
            'focal_point': camera.GetFocalPoint(),
            'view_up': camera.GetViewUp(),
            'parallel_scale': camera.GetParallelScale()
        }
    
    def restore_camera_state(self, orientation):
        if self.camera_state[orientation] is not None:
            renderer = getattr(self, f'{orientation}_renderer')
            camera = renderer.GetActiveCamera()
            
            state = self.camera_state[orientation]
            camera.SetPosition(state['position'])
            camera.SetFocalPoint(state['focal_point'])
            camera.SetViewUp(state['view_up'])
            camera.SetParallelScale(state['parallel_scale'])
            
            renderer.ResetCameraClippingRange()
    
    def reset_view(self, orientation):
        self.camera_state[orientation] = None
        self.render_slice(orientation, preserve_camera=False)
    
    def _volume_world_center(self):
        """Center of the loaded volume in VTK world coordinates (spacing-aware)."""
        if self.volume_data is None:
            return (0.0, 0.0, 0.0)
        z, y, x = self.volume_data.shape
        sp = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        return (
            (x - 1) * float(sp[0]) * 0.5,
            (y - 1) * float(sp[1]) * 0.5,
            (z - 1) * float(sp[2]) * 0.5,
        )

    def _volume_world_radius(self):
        """Half-diagonal of the volume AABB — used for zoom distance clamps."""
        if self.volume_data is None:
            return 100.0
        z, y, x = self.volume_data.shape
        sp = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        hx = (x - 1) * float(sp[0]) * 0.5
        hy = (y - 1) * float(sp[1]) * 0.5
        hz = (z - 1) * float(sp[2]) * 0.5
        return max((hx * hx + hy * hy + hz * hz) ** 0.5, 1.0)

    def _volume_world_bounds(self):
        """World AABB of the loaded volume (origin 0 + spacing)."""
        if self.volume_data is None:
            return None
        sp = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        return volume_world_aabb(self.volume_data.shape, sp)

    def _sync_3d_orbit_pivot(self):
        """Set Track pivot = volume center (default). Does not re-anchor camera.

        Track rotates around this pivot without snapping focal point on
        left-drag. Later: Pivot Point tool can call SetOrbitPivot with a
        user-picked world point.
        """
        style = getattr(self, '_3d_interactor_style', None)
        if style is None or not hasattr(style, 'SetOrbitPivot'):
            return
        cx, cy, cz = self._volume_world_center()
        style.SetOrbitPivot(
            cx, cy, cz,
            radius=self._volume_world_radius(),
            bounds=self._volume_world_bounds(),
        )

    def _dragonfly_3d_zoom(self, zoom_in=True, strength=1.0, display_xy=None):
        """ORS Dragonfly object zoom — dolly about orbit pivot (pre-P1 behaviour).

        Scroll wheel and right-drag share this path (host for
        ``Dragonfly3DInteractorStyle``).

        ``display_xy`` is intentionally ignored: volume-render world-under-cursor
        picks are unstable and were shifting FP/position, which made Track / pan
        / zoom feel broken after the P1 experiment.
        """
        ren = getattr(self, 'view_3d_renderer', None)
        widget = getattr(self, 'view_3d_widget', None)
        if ren is None or widget is None or self.volume_data is None:
            return

        try:
            if hasattr(self, '_start_lod_interaction'):
                self._start_lod_interaction()

            apply_dragonfly_volume_zoom(
                ren,
                zoom_in=zoom_in,
                strength=strength,
                volume_center=self._volume_world_center(),
                volume_radius=self._volume_world_radius(),
                volume_bounds=self._volume_world_bounds(),
                display_xy=None,  # restore: no cursor pan on 3D volume
            )

            rw = widget.GetRenderWindow()
            if rw is not None:
                rw.Render()
        except Exception as e:
            print(f">>> 3D zoom error (swallowed): {e}")
        finally:
            if hasattr(self, '_end_lod_interaction'):
                QTimer.singleShot(150, self._end_lod_interaction)

    def reset_3d_view(self):
        """Reset 3D zoom/pan to a tight fit of the volume (keep orbit direction).

        Avoids bare ``ResetCamera()`` which often frames the scene too loose
        (volume becomes a tiny speck) especially after layout changes or when
        only the focal point was re-centered without adjusting distance.
        """
        import math

        if not getattr(self, "volume_actor", None) or self.volume_data is None:
            return
        ren = getattr(self, "view_3d_renderer", None)
        widget = getattr(self, "view_3d_widget", None)
        if ren is None or widget is None:
            return

        cam = ren.GetActiveCamera()
        cx, cy, cz = self._volume_world_center()
        R = max(float(self._volume_world_radius()), 1.0)

        # Keep current viewing direction (reset = zoom/pan, not re-orient)
        pos = cam.GetPosition()
        fp = cam.GetFocalPoint()
        dx, dy, dz = pos[0] - fp[0], pos[1] - fp[1], pos[2] - fp[2]
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist < 1e-9:
            # Fallback: Top view direction
            dx, dy, dz, dist = 0.0, 0.0, 1.0, 1.0
        inv = 1.0 / dist
        dx, dy, dz = dx * inv, dy * inv, dz * inv

        # Volume should fill ~88% of the FOV — readable, not a speck
        fill = 0.88

        if cam.GetParallelProjection():
            # ParallelScale = half-height of the orthographic view in world units
            cam.SetParallelScale(max(R / fill, 1e-6))
            d = max(R * 3.0, 1.0)
        else:
            va = float(cam.GetViewAngle())
            if va < 10.0 or va > 100.0:
                # Match set_projection default when angle is invalid
                va = 40.0
                cam.SetViewAngle(va)
            half_rad = math.radians(va * 0.5)
            # Distance so a sphere of radius R fills `fill` of the vertical FOV
            d = (R / fill) / max(math.tan(half_rad), 1e-6)
            d = max(d, R * 1.15)

        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(cx + dx * d, cy + dy * d, cz + dz * d)
        bounds = self._volume_world_bounds()
        if bounds is not None:
            push_camera_outside_aabb(cam, bounds, pad_frac=0.02)

        # If view-up is nearly collinear with view direction, pick a stable up
        ux, uy, uz = cam.GetViewUp()
        # |view · up| close to 1 → degenerate
        if abs(dx * ux + dy * uy + dz * uz) > 0.95:
            # Prefer Z-up, else Y-up
            if abs(dz) < 0.9:
                cam.SetViewUp(0.0, 0.0, 1.0)
            else:
                cam.SetViewUp(0.0, 1.0, 0.0)

        self._sync_3d_orbit_pivot()
        ren.ResetCameraClippingRange()
        widget.GetRenderWindow().Render()

    def toggle_rotate_mode(self, checked):
        self.rotate_mode = checked
        if not checked:
            self.last_mouse_pos = None

        # Keep button checked state + CSS dynamic property in sync
        btn = getattr(self, 'rotate2d_btn', None)
        if btn is not None:
            if btn.isChecked() != bool(checked):
                btn.blockSignals(True)
                btn.setChecked(bool(checked))
                btn.blockSignals(False)
            btn.setProperty("checked", "true" if checked else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        
        # Auto-disable crosshair when rotate mode is turned on
        if checked and self.crosshair_enabled:
            self.crosshair_enabled = False
            # Update the external crosshair button if accessible
            parent = self.parent()
            while parent is not None:
                if hasattr(parent, 'cross_btn'):
                    parent.cross_btn.setChecked(False)
                    parent.cross_btn.setProperty("checked", "false")
                    parent.cross_btn.style().unpolish(parent.cross_btn)
                    parent.cross_btn.style().polish(parent.cross_btn)
                    break
                parent = parent.parent()
            if self.volume_data is not None:
                for ori in ['axial', 'coronal', 'sagittal']:
                    self.render_slice(ori, preserve_camera=True)
                self.update_3d_crosshair()

    def reset_2d_rotation(self):
        for orientation in ['axial', 'coronal', 'sagittal']:
            renderer = getattr(self, f'{orientation}_renderer')
            camera = renderer.GetActiveCamera()
            
            # Save state
            self.save_camera_state(orientation)
            state = self.camera_state[orientation]
            
            # Reset view-up
            camera.SetViewUp(0, 1, 0)
            
            # Re-save state
            self.save_camera_state(orientation)
            
            widget = getattr(self, f'{orientation}_widget')
            widget.GetRenderWindow().Render()

    def handle_rotation_drag(self, orientation, pos):
        try:
            if not hasattr(self, 'last_mouse_pos') or self.last_mouse_pos is None:
                self.last_mouse_pos = pos
                return
            
            widget = getattr(self, f'{orientation}_widget')
            renderer = getattr(self, f'{orientation}_renderer')
            camera = renderer.GetActiveCamera()
            
            # Viewport center
            size = widget.GetRenderWindow().GetSize()
            cx = size[0] / 2.0
            cy = size[1] / 2.0
            
            # Current and previous mouse positions
            p1 = pos
            p0 = self.last_mouse_pos
            
            # Vectors from center (Qt Y is downwards, so invert y)
            v0_x = p0.x() - cx
            v0_y = cy - p0.y()
            
            v1_x = p1.x() - cx
            v1_y = cy - p1.y()
            
            import math
            # Calculate angles
            a0 = math.atan2(v0_y, v0_x)
            a1 = math.atan2(v1_y, v1_x)
            
            da = a1 - a0
            # Normalize da to [-pi, pi]
            if da > math.pi:
                da -= 2.0 * math.pi
            elif da < -math.pi:
                da += 2.0 * math.pi
                
            da_deg = math.degrees(da)
            
            # Apply Roll to camera
            camera.Roll(da_deg)
            
            # Save camera state so it is preserved
            self.save_camera_state(orientation)
            
            # Render
            widget.GetRenderWindow().Render()
            
            # Save current pos for next drag event
            self.last_mouse_pos = pos
        except Exception as e:
            print("Error in handle_rotation_drag:", e)
    
    def _window_level_to_uint8(self, slice_data, window, level):
        """Map scalar slice through W/L to display uint8 (H,W)."""
        arr = np.asarray(slice_data, dtype=np.float32)
        half = max(float(window), 1e-6) * 0.5
        lo = float(level) - half
        hi = float(level) + half
        if hi <= lo:
            hi = lo + 1.0
        out = (arr - lo) * (255.0 / (hi - lo))
        return np.clip(out, 0, 255).astype(np.uint8)

    def _compose_mpr_overlay_rgb(
        self,
        slice_data,
        window,
        level,
        c1_slice,
        c2_slice,
        seg_slice,
        c1_on,
        c2_on,
        opacity,
        orientation="",
    ):
        """Alpha-blend C1/C2 masks onto W/L-mapped CT → RGB uint8 (H,W,3).

        Background mask pixels keep pure CT gray (no black film).
        C2 overwrites C1 on shared voxels (void-in-bump).
        """
        gray = self._window_level_to_uint8(slice_data, window, level)
        h, w = gray.shape[:2]
        rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        a = float(np.clip(opacity, 0.0, 1.0))

        c1_color = np.array(
            [float(v * 255) for v in self.seg_colors.get(128, [1.0, 1.0, 0.0])],
            dtype=np.float32,
        )
        c2_color = np.array(
            [float(v * 255) for v in self.seg_colors.get(255, [1.0, 0.0, 0.0])],
            dtype=np.float32,
        )

        def _valid_mask(src, exact=None):
            if src is None:
                return None
            arr = np.asarray(src)
            if arr.ndim > 2:
                arr = arr[..., 0]
            if arr.shape[:2] != (h, w):
                print(
                    f"[VIEWER OVERLAY] mask shape {arr.shape} != {(h, w)} ({orientation}) — skipped"
                )
                return None
            return (arr == exact) if exact is not None else (arr > 0)

        if c1_on:
            if c1_slice is not None:
                m1 = _valid_mask(c1_slice)
            else:
                m1 = _valid_mask(seg_slice, exact=128)
            if m1 is not None and np.any(m1):
                rgb[m1] = (1.0 - a) * rgb[m1] + a * c1_color

        if c2_on:
            if c2_slice is not None:
                m2 = _valid_mask(c2_slice)
            else:
                m2 = _valid_mask(seg_slice, exact=255)
            if m2 is not None and np.any(m2):
                rgb[m2] = (1.0 - a) * rgb[m2] + a * c2_color

        return np.clip(rgb, 0, 255).astype(np.uint8)

    def render_slice(self, orientation, preserve_camera=False):
        if self.volume_data is None:
            return
        
        if preserve_camera:
            self.save_camera_state(orientation)
        
        renderer = getattr(self, f'{orientation}_renderer')
                
        slice_idx = self.current_slices[orientation]
        
        is_oblique = (abs(self.oblique_angles.get('axial', 0.0)) > 0.5 or 
                      abs(self.oblique_angles.get('coronal', 0.0)) > 0.5 or 
                      abs(self.oblique_angles.get('sagittal', 0.0)) > 0.5)
        
        # In manual align mode, oblique_angles are only for crosshair preview
        # — don't reslice the volume until the user commits
        if getattr(self, '_manual_align_active', False):
            is_oblique = False
        
        if is_oblique:
            slice_data, seg_slice, c1_slice, c2_slice = self._oblique_reslice_3d(orientation, slice_idx)
        else:
            if orientation == 'axial':
                # Axial (XY): X = horizontal, Y = vertical (so shape: Y, X)
                actual_z = slice_idx
                if self.reverse_z:
                    actual_z = self.volume_data.shape[0] - 1 - slice_idx
                slice_data = self.volume_data[actual_z, :, :]
                seg_slice = self.segmentation_data[actual_z, :, :] if self.segmentation_data is not None else None
                c1_slice = self.class1_data[actual_z, :, :] if self.class1_data is not None else None
                c2_slice = self.class2_data[actual_z, :, :] if self.class2_data is not None else None
                
            elif orientation == 'coronal':
                # Coronal (XZ): X = horizontal, Z = vertical (Z increases downwards -> Z=0 at top)
                slice_data = np.flipud(self.volume_data[:, slice_idx, :])
                seg_slice = np.flipud(self.segmentation_data[:, slice_idx, :]) if self.segmentation_data is not None else None
                c1_slice = np.flipud(self.class1_data[:, slice_idx, :]) if self.class1_data is not None else None
                c2_slice = np.flipud(self.class2_data[:, slice_idx, :]) if self.class2_data is not None else None
                
            else:
                # Sagittal (YZ): Z = horizontal, Y = vertical (90 deg CCW of previous)
                slice_data = np.transpose(self.volume_data[:, :, slice_idx])
                seg_slice = np.transpose(self.segmentation_data[:, :, slice_idx]) if self.segmentation_data is not None else None
                c1_slice = np.transpose(self.class1_data[:, :, slice_idx]) if self.class1_data is not None else None
                c2_slice = np.transpose(self.class2_data[:, :, slice_idx]) if self.class2_data is not None else None
        
        h, w = slice_data.shape
        
        # ── Performance: Reuse VTK image actors instead of recreating ──
        # Cache the base image actor, vtkImageData, and overlay state per orientation.
        # On subsequent calls with same dimensions, we only update the pixel buffer
        # (zero-copy where possible), avoiding expensive VTK object creation.
        if not hasattr(self, '_cached_slice_actors'):
            self._cached_slice_actors = {}
        
        cache = self._cached_slice_actors.get(orientation)
        try:
            _c1_on = bool(getattr(self, f'{orientation}_overlay_c1').isChecked())
            _c2_on = bool(getattr(self, f'{orientation}_overlay_c2').isChecked())
            _op_val = int(getattr(self, f'{orientation}_opacity_slider').value())
        except Exception:
            _c1_on, _c2_on, _op_val = True, True, 70
        has_overlays = (seg_slice is not None or c1_slice is not None or c2_slice is not None)
        _want_rgb = (
            (_c1_on or _c2_on)
            and _op_val > 0
            and has_overlays
        )
        _display_mode = 'rgb_overlay' if _want_rgb else 'grayscale'

        need_full_rebuild = (
            cache is None or 
            cache.get('h') != h or cache.get('w') != w or
            cache.get('dtype') != slice_data.dtype or
            cache.get('display_mode') != _display_mode
        )
        
        # Overlay visibility/opacity must be part of the cache key so C1/C2/opacity
        # toggles always rebuild the actor stack (not a stale RGB film).
        overlay_state = (
            has_overlays,
            _c1_on,
            _c2_on,
            _op_val,
            bool(self.selected_highlight_objects) if self.volume_data is not None else False,
            _display_mode,
        )
        if cache and cache.get('overlay_state') != overlay_state:
            need_full_rebuild = True
        
        if need_full_rebuild:
            # Full rebuild: remove everything and recreate
            renderer.RemoveAllViewProps()
            
            # Invalidate persistent crosshair actors — they were removed from this renderer
            if hasattr(self, '_persistent_crosshair') and orientation in self._persistent_crosshair:
                self._persistent_crosshair[orientation] = {}
            
            vtk_image = vtk.vtkImageData()
            vtk_image.SetDimensions(w, h, 1)
            vtk_image.SetSpacing(1.0, 1.0, 1.0)
            vtk_image.SetOrigin(0.0, 0.0, 0.0)
            
            image_actor = vtk.vtkImageActor()
            image_actor.GetMapper().SetInputData(vtk_image)
            image_actor.GetProperty().SetInterpolationTypeToNearest()
            renderer.AddActor(image_actor)
            
            # Store in cache
            self._cached_slice_actors[orientation] = {
                'h': h, 'w': w, 'dtype': slice_data.dtype,
                'vtk_image': vtk_image,
                'image_actor': image_actor,
                'overlay_state': overlay_state,
                'display_mode': _display_mode,
            }
            cache = self._cached_slice_actors[orientation]
        else:
            # Fast path: reuse existing actors — just need to remove overlays/crosshair/ruler
            # and re-add them after data update
            vtk_image = cache['vtk_image']
            image_actor = cache['image_actor']
            cache['display_mode'] = _display_mode
            cache['overlay_state'] = overlay_state
            
            # Remove only non-base actors (crosshair, ruler, overlay, text labels) — keep image_actor
            actors_to_keep = {image_actor}
            props = renderer.GetViewProps()
            props.InitTraversal()
            props_to_remove = []
            for _ in range(props.GetNumberOfItems()):
                prop = props.GetNextProp()
                if prop and prop not in actors_to_keep:
                    props_to_remove.append(prop)
            for prop in props_to_remove:
                renderer.RemoveViewProp(prop)
            
            # Invalidate persistent crosshair so they get re-added
            if hasattr(self, '_persistent_crosshair') and orientation in self._persistent_crosshair:
                self._persistent_crosshair[orientation] = {}
        
        prop = image_actor.GetProperty()
        
        # Sliders removed, defaulting to 0
        brightness_val = 0
        contrast_val = 0
        
        if self.window_level[orientation] is not None:
            base_window, base_level = self.window_level[orientation]
        else:
            v_min, v_max = float(slice_data.min()), float(slice_data.max())
            base_window = v_max - v_min if v_max > v_min else 1.0
            base_level = v_min + base_window / 2.0
        
        # Brightness shifts the level, contrast scales the window
        level = base_level + brightness_val * (base_window / 200.0)
        # contrast_val: -100 → window*2 (flat), +100 → window*0.2 (sharp)
        contrast_factor = max(0.2, 1.0 - contrast_val / 100.0)
        window = base_window * contrast_factor

        # Resolve overlay toggles early — when any class is on we bake a display RGB
        # image (CT + mask alpha-blend). This avoids VTK ImageActor RGB black-film
        # and broken per-pixel alpha on some OpenGL backends.
        try:
            overlay_c1_enabled = bool(getattr(self, f'{orientation}_overlay_c1').isChecked())
            overlay_c2_enabled = bool(getattr(self, f'{orientation}_overlay_c2').isChecked())
            opacity = float(getattr(self, f'{orientation}_opacity_slider').value()) / 100.0
        except Exception:
            overlay_c1_enabled, overlay_c2_enabled, opacity = False, False, 0.7
        opacity = float(np.clip(opacity, 0.0, 1.0))
        want_overlay = (
            (overlay_c1_enabled or overlay_c2_enabled)
            and opacity > 0.0
            and (c1_slice is not None or c2_slice is not None or seg_slice is not None)
        )

        if want_overlay:
            display_rgb = self._compose_mpr_overlay_rgb(
                slice_data, window, level,
                c1_slice, c2_slice, seg_slice,
                overlay_c1_enabled, overlay_c2_enabled, opacity,
                orientation,
            )
            # display_rgb: (H,W,3) uint8 in same image axes as slice_data
            overlay_transposed = np.transpose(display_rgb, (1, 0, 2))
            flat_rgb = np.ascontiguousarray(overlay_transposed.reshape(-1, 3, order='F'))
            vtk_array = numpy_support.numpy_to_vtk(
                flat_rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
            )
            vtk_array.SetNumberOfComponents(3)
            # Identity W/L — pixels already display-mapped (MUST stay 255/127.5).
            # Data-range W/L (e.g. 15k–51k) on 0–255 RGB blacks out the whole MPR.
            prop.SetColorWindow(255.0)
            prop.SetColorLevel(127.5)
            if cache is not None:
                cache['display_mode'] = 'rgb_overlay'
        else:
            # Grayscale path — keep native dtype + VTK W/L (fast, full dynamic range)
            slice_transposed = np.transpose(slice_data, (1, 0))
            flat_data = np.ascontiguousarray(slice_transposed.flatten('F'))
            if slice_data.dtype == np.uint16:
                vtk_array = numpy_support.numpy_to_vtk(
                    flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_SHORT
                )
            elif slice_data.dtype == np.uint8:
                vtk_array = numpy_support.numpy_to_vtk(
                    flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
                )
            else:
                vtk_array = numpy_support.numpy_to_vtk(
                    flat_data.astype(np.float32), deep=True, array_type=vtk.VTK_FLOAT
                )
            prop.SetColorWindow(window)
            prop.SetColorLevel(level)
            if cache is not None:
                cache['display_mode'] = 'grayscale'

        vtk_image.GetPointData().SetScalars(vtk_array)
        vtk_image.Modified()
        
        # Camera
        if preserve_camera and self.camera_state[orientation] is not None:
            self.restore_camera_state(orientation)
        else:
            camera = renderer.GetActiveCamera()
            camera.ParallelProjectionOn()
            
            center_x = w / 2.0
            center_y = h / 2.0
            
            camera.SetPosition(center_x, center_y, 1000)
            camera.SetFocalPoint(center_x, center_y, 0)
            camera.SetViewUp(0, 1, 0)
            
            # Fit image to viewport: use aspect ratio to determine scale
            widget = getattr(self, f'{orientation}_widget')
            vp_size = widget.GetRenderWindow().GetSize()
            vp_w_px = max(vp_size[0], 1)
            vp_h_px = max(vp_size[1], 1)
            vp_aspect = vp_w_px / vp_h_px
            img_aspect = w / max(h, 1)
            
            if img_aspect > vp_aspect:
                # Image is wider than viewport -> fit by width
                parallel_scale = (w / vp_aspect) * 0.5
            else:
                # Image is taller or same -> fit by height
                parallel_scale = h * 0.5
            
            camera.SetParallelScale(parallel_scale)
            
            renderer.ResetCameraClippingRange()
        
        # Segmentation mask is already baked into the base image when want_overlay
        # (see _compose_mpr_overlay_rgb). No second ImageActor — avoids black film.

        # --- Object Highlight Overlay (for selected objects from stats table) ---
        if self.selected_highlight_objects and self.volume_data is not None:
            # Get the labeled slice for the current orientation
            labeled_data = self.labeled_class1_data  # All objects are C1-based
            if labeled_data is not None:
                slice_idx_hl = self.current_slices[orientation]
                
                if orientation == 'axial':
                    actual_idx = (labeled_data.shape[0] - 1 - slice_idx_hl) if self.reverse_z else slice_idx_hl
                    if actual_idx < labeled_data.shape[0]:
                        labeled_slice = labeled_data[actual_idx, :, :]
                    else:
                        labeled_slice = None
                elif orientation == 'coronal':
                    if slice_idx_hl < labeled_data.shape[1]:
                        labeled_slice = np.flipud(labeled_data[:, slice_idx_hl, :])
                    else:
                        labeled_slice = None
                else:  # sagittal Ã¢â‚¬â€ same flip/rot as main render
                    if slice_idx_hl < labeled_data.shape[2]:
                        labeled_slice = np.transpose(labeled_data[:, :, slice_idx_hl])
                    else:
                        labeled_slice = None
                
                if labeled_slice is not None:
                    highlight_rgb = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA
                    has_highlight = False
                    
                    for class_num, obj_label in self.selected_highlight_objects:
                        obj_mask = (labeled_slice == obj_label)
                        
                        if np.any(obj_mask):
                            # Create outline: dilate - original = border (3px thick)
                            dilated = ndimage.binary_dilation(obj_mask, iterations=3)
                            outline = dilated & ~obj_mask
                            
                            # Fill interior with semi-transparent highlight
                            hl_color = self.highlight_color
                            hl_r = int(hl_color[0] * 255)
                            hl_g = int(hl_color[1] * 255)
                            hl_b = int(hl_color[2] * 255)
                            
                            # Interior fill (low alpha)
                            highlight_rgb[obj_mask, 0] = np.maximum(highlight_rgb[obj_mask, 0], hl_r)
                            highlight_rgb[obj_mask, 1] = np.maximum(highlight_rgb[obj_mask, 1], hl_g)
                            highlight_rgb[obj_mask, 2] = np.maximum(highlight_rgb[obj_mask, 2], hl_b)
                            highlight_rgb[obj_mask, 3] = np.maximum(highlight_rgb[obj_mask, 3], 100)
                            
                            # Bright border (high alpha white-cyan)
                            highlight_rgb[outline, 0] = 255
                            highlight_rgb[outline, 1] = 255
                            highlight_rgb[outline, 2] = 255
                            highlight_rgb[outline, 3] = 220
                            
                            has_highlight = True
                    
                    if has_highlight:
                        hl_rgb_only = highlight_rgb[:, :, :3]
                        
                        hl_transposed = np.transpose(hl_rgb_only, (1, 0, 2))
                        hl_flat = np.ascontiguousarray(hl_transposed.reshape(-1, 3, order='F'))
                        
                        vtk_hl = vtk.vtkImageData()
                        vtk_hl.SetDimensions(w, h, 1)
                        vtk_hl.SetSpacing(1.0, 1.0, 1.0)
                        vtk_hl.SetOrigin(0.0, 0.0, 0.0)
                        
                        vtk_hl_arr = numpy_support.numpy_to_vtk(hl_flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
                        vtk_hl_arr.SetNumberOfComponents(3)
                        vtk_hl.GetPointData().SetScalars(vtk_hl_arr)
                        
                        hl_actor = vtk.vtkImageActor()
                        hl_actor.GetMapper().SetInputData(vtk_hl)
                        hl_actor.SetOpacity(0.7)
                        
                        renderer.AddActor(hl_actor)
        
        if self.crosshair_enabled:
            # Use the unified persistent crosshair system to avoid double crosshair
            self.update_2d_crosshair(orientation)
        
        # Always draw scale bar (updates with zoom)
        self.add_ruler_overlay(renderer, orientation, (h, w))

        # Dragonfly clip-box overlay (re-added after render_slice clears non-base actors)
        self._df_clip_update_mpr_overlay(orientation)
        
        # Draw Object Indices if enabled
        if getattr(self, 'show_indices_check', None) and self.show_indices_check.isChecked() and getattr(self, 'object_stats', None):
            slice_idx = self.current_slices[orientation]
            if orientation == 'axial':
                actual_z = (self.volume_data.shape[0] - 1 - slice_idx) if self.reverse_z else slice_idx
                self.draw_object_labels(renderer, actual_z, orientation='axial', slice_idx=actual_z)
            elif orientation == 'coronal':
                self.draw_object_labels(renderer, 0, orientation='coronal', slice_idx=slice_idx)
            elif orientation == 'sagittal':
                self.draw_object_labels(renderer, 0, orientation='sagittal', slice_idx=slice_idx)
        
        widget = getattr(self, f'{orientation}_widget')
        widget.GetRenderWindow().Render()
    
    def update_overlay_visibility(self, orientation):
        """Refresh C1/C2 visibility and overlay opacity for one pane (or 3D)."""
        if orientation == '3d':
            if not self.is_3d_volume_render_enabled():
                return
            c1_show = self.view_3d_overlay_group.c1_btn.isChecked() if hasattr(self.view_3d_overlay_group, 'c1_btn') else False
            c2_show = self.view_3d_overlay_group.c2_btn.isChecked() if hasattr(self.view_3d_overlay_group, 'c2_btn') else False
            op_val = self.view_3d_overlay_group.opacity_slider.value() / 100.0 if hasattr(self.view_3d_overlay_group, 'opacity_slider') else 0.5
            
            if getattr(self, 'c1_actor_3d', None):
                self.c1_actor_3d.SetVisibility(c1_show)
                self.c1_actor_3d.GetProperty().GetScalarOpacity().RemoveAllPoints()
                self.c1_actor_3d.GetProperty().GetScalarOpacity().AddPoint(0, 0.0)
                self.c1_actor_3d.GetProperty().GetScalarOpacity().AddPoint(1, op_val)
                self.c1_actor_3d.GetProperty().GetScalarOpacity().AddPoint(255, op_val)
                
            if getattr(self, 'c2_actor_3d', None):
                self.c2_actor_3d.SetVisibility(c2_show)
                self.c2_actor_3d.GetProperty().GetScalarOpacity().RemoveAllPoints()
                self.c2_actor_3d.GetProperty().GetScalarOpacity().AddPoint(0, 0.0)
                self.c2_actor_3d.GetProperty().GetScalarOpacity().AddPoint(1, op_val)
                self.c2_actor_3d.GetProperty().GetScalarOpacity().AddPoint(255, op_val)
            
            self.view_3d_widget.GetRenderWindow().Render()
            return

        # Invalidate cached actors so toggle/opacity cannot leave a stale overlay film
        if hasattr(self, '_cached_slice_actors') and orientation in self._cached_slice_actors:
            try:
                del self._cached_slice_actors[orientation]
            except Exception:
                self._cached_slice_actors[orientation] = None
        self.render_slice(orientation, preserve_camera=True)
    
    def _refresh_all_index_views(self):
        """Refresh all three views when Show Index checkbox changes."""
        for ori in ['axial', 'coronal', 'sagittal']:
            self.render_slice(ori, preserve_camera=True)

    def draw_object_labels(self, renderer, actual_z, orientation='axial', slice_idx=0):
        """Draw bump index labels on MPR views (Index checkbox).

        UX rules (avoid left-edge pile-up of R# + Bump (R,C)):
          · **One identity per bump**: ``(R,C)`` matching MES Bump ID — no extra R#/C#
            edge headers that collide with left/top dies.
          · Labels sit on the **centroid** with a dark plate + cyan text.
          · Selected MES rows highlight that label in amber.
          · Dense FOVs use smaller scale automatically.
        """
        if not self.object_stats:
            return

        # Determine z-offset strategy
        is_multi_layer = getattr(self, 'input_mode', 'single') == 'multi_layer'
        ml_z_starts = getattr(self, '_ml_layer_z_starts', {})
        single_z_offset = getattr(self, '_measurement_start_slice', 0) if not is_multi_layer else 0

        # Collect visible stats for this orientation/slice
        visible_stats = []
        for stat in self.object_stats:
            # Multi-layer filter: only filter when a specific layer is selected
            if is_multi_layer and self.current_display_layer is not None:
                if stat.get('layer_name') != self.current_display_layer:
                    continue

            if orientation == 'axial':
                # Compute the per-stat z offset
                if is_multi_layer and self.current_display_layer is None:
                    # "All Layers" mode: volume is combined stack, use per-layer z_start
                    layer_z_start = ml_z_starts.get(stat.get('layer_name', ''), 0)
                else:
                    # Specific layer or single mode
                    layer_z_start = 0
                local_z = actual_z - single_z_offset - layer_z_start
                if stat['z_min'] <= local_z < stat['z_max']:
                    visible_stats.append((stat, layer_z_start))
            elif orientation == 'coronal':
                # Coronal slices along Y axis
                if stat['y_min'] <= slice_idx < stat['y_max']:
                    if is_multi_layer and self.current_display_layer is None:
                        layer_z_start = ml_z_starts.get(stat.get('layer_name', ''), 0)
                    else:
                        layer_z_start = single_z_offset
                    visible_stats.append((stat, layer_z_start))
            elif orientation == 'sagittal':
                # Sagittal slices along X axis
                if stat['x_min'] <= slice_idx < stat['x_max']:
                    if is_multi_layer and self.current_display_layer is None:
                        layer_z_start = ml_z_starts.get(stat.get('layer_name', ''), 0)
                    else:
                        layer_z_start = single_z_offset
                    visible_stats.append((stat, layer_z_start))

        if not visible_stats:
            return

        # Adaptive scale by density
        n_bumps = len(visible_stats)
        if n_bumps > 500:
            label_scale, font_size = 0.16, 13
        elif n_bumps > 200:
            label_scale, font_size = 0.20, 14
        elif n_bumps > 50:
            label_scale, font_size = 0.24, 15
        else:
            label_scale, font_size = 0.28, 17

        stats_only = [s for s, _ in visible_stats]
        has_grid = all(
            s.get("grid_row") is not None and s.get("grid_col") is not None
            for s in stats_only
        )
        selected_labels = {
            int(lab)
            for _c, lab in (getattr(self, "selected_highlight_objects", None) or [])
            if lab
        }

        if orientation == 'axial':
            self._draw_axial_labels(
                renderer, visible_stats, has_grid, label_scale, font_size, selected_labels
            )
        elif orientation == 'coronal':
            self._draw_coronal_labels(
                renderer, visible_stats, has_grid, label_scale, font_size,
                slice_idx, selected_labels,
            )
        elif orientation == 'sagittal':
            self._draw_sagittal_labels(
                renderer, visible_stats, has_grid, label_scale, font_size,
                slice_idx, selected_labels,
            )

    @staticmethod
    def _style_index_text_prop(tprop, font_size, color, selected=False):
        """Readable index plate: bold + dark background (no extra R# clutter)."""
        tprop.SetFontSize(int(font_size))
        tprop.SetBold(True)
        tprop.ShadowOn()
        tprop.SetShadowOffset(1, -1)
        if selected:
            tprop.SetColor(1.0, 0.92, 0.25)  # amber when MES-selected
            tprop.SetBackgroundColor(0.05, 0.05, 0.0)
            tprop.SetBackgroundOpacity(0.72)
            frame = (0.95, 0.75, 0.15)
        else:
            tprop.SetColor(float(color[0]), float(color[1]), float(color[2]))
            tprop.SetBackgroundColor(0.02, 0.04, 0.08)
            tprop.SetBackgroundOpacity(0.62)
            frame = (0.15, 0.55, 0.65)
        try:
            tprop.SetJustificationToCentered()
            tprop.SetVerticalJustificationToCentered()
        except Exception:
            pass
        try:
            tprop.FrameOn()
            tprop.SetFrameColor(*frame)
            tprop.SetFrameWidth(1)
        except Exception:
            pass

    def _index_label_text(self, stat, has_grid):
        """Single identity string: Bump (R,C) or MES # — never both + R# header."""
        if has_grid:
            return f"{stat['grid_row']},{stat['grid_col']}"
        return str(stat.get("row_id", stat.get("label", "")))

    def _draw_axial_labels(
        self, renderer, visible_stats, has_grid, label_scale, font_size, selected_labels=None
    ):
        """XY: one (R,C) plate per bump at centroid — no R#/C# edge rails."""
        selected_labels = selected_labels or set()
        seen = set()
        for stat, _z_off in visible_stats:
            if has_grid:
                key = (stat["grid_row"], stat["grid_col"])
                if key in seen:
                    continue
                seen.add(key)
            try:
                lab = int(stat.get("label", 0))
            except (TypeError, ValueError):
                lab = 0
            is_sel = lab in selected_labels
            text = self._index_label_text(stat, has_grid)
            if not text:
                continue
            cx = float(stat["centroid_x"])
            cy = float(stat["centroid_y"])
            # Slight inset from geometric edge so labels on rim bumps stay on the pad
            caption = vtk.vtkTextActor3D()
            caption.SetInput(text)
            caption.SetPosition(cx, cy, 0.55)
            sc = label_scale * (1.15 if is_sel else 1.0)
            caption.SetScale(sc, sc, sc)
            self._style_index_text_prop(
                caption.GetTextProperty(),
                font_size + (2 if is_sel else 0),
                (0.25, 0.95, 1.0),
                selected=is_sel,
            )
            renderer.AddActor(caption)

    def _draw_coronal_labels(
        self, renderer, visible_stats, has_grid, label_scale, font_size, slice_y,
        selected_labels=None,
    ):
        """XZ: same (R,C) identity — no duplicate C# header row."""
        selected_labels = selected_labels or set()
        vol_z = self.volume_data.shape[0] if self.volume_data is not None else 1
        seen = set()
        for stat, z_off in visible_stats:
            if has_grid:
                key = (stat["grid_row"], stat["grid_col"])
                if key in seen:
                    continue
                seen.add(key)
            try:
                lab = int(stat.get("label", 0))
            except (TypeError, ValueError):
                lab = 0
            is_sel = lab in selected_labels
            text = self._index_label_text(stat, has_grid)
            if not text:
                continue
            cx = float(stat["centroid_x"])
            global_z = float(stat["centroid_z"]) + z_off
            cz_display = (vol_z - 1) - global_z
            caption = vtk.vtkTextActor3D()
            caption.SetInput(text)
            caption.SetPosition(cx, cz_display, 0.55)
            sc = label_scale * (1.15 if is_sel else 1.0)
            caption.SetScale(sc, sc, sc)
            self._style_index_text_prop(
                caption.GetTextProperty(),
                font_size + (2 if is_sel else 0),
                (0.4, 1.0, 0.55),
                selected=is_sel,
            )
            renderer.AddActor(caption)

    def _draw_sagittal_labels(
        self, renderer, visible_stats, has_grid, label_scale, font_size, slice_x,
        selected_labels=None,
    ):
        """YZ: same (R,C) identity — no duplicate R# rail."""
        selected_labels = selected_labels or set()
        seen = set()
        for stat, z_off in visible_stats:
            if has_grid:
                key = (stat["grid_row"], stat["grid_col"])
                if key in seen:
                    continue
                seen.add(key)
            try:
                lab = int(stat.get("label", 0))
            except (TypeError, ValueError):
                lab = 0
            is_sel = lab in selected_labels
            text = self._index_label_text(stat, has_grid)
            if not text:
                continue
            # Sagittal: horizontal = Z, vertical = Y (after view transform)
            cz = float(stat["centroid_z"]) + z_off
            cy = float(stat["centroid_y"])
            caption = vtk.vtkTextActor3D()
            caption.SetInput(text)
            caption.SetPosition(cz, cy, 0.55)
            sc = label_scale * (1.15 if is_sel else 1.0)
            caption.SetScale(sc, sc, sc)
            self._style_index_text_prop(
                caption.GetTextProperty(),
                font_size + (2 if is_sel else 0),
                (1.0, 0.6, 0.35),
                selected=is_sel,
            )
            renderer.AddActor(caption)

    def choose_overlay_color(self, class_num):
        """Open color dialog to pick class overlay color"""
        key = 128 if class_num == 1 else 255
        current = self.seg_colors.get(key, [0.0, 1.0, 0.0])
        init_color = QColor(int(current[0]*255), int(current[1]*255), int(current[2]*255))
        color = QColorDialog.getColor(init_color, self, f"Choose Class {class_num} Color")
        if color.isValid():
            self.seg_colors[key] = [color.redF(), color.greenF(), color.blueF()]
            # Update button color as preview
            btn = self.color_c1_btn if class_num == 1 else self.color_c2_btn
            btn.setStyleSheet(f"background-color: {color.name()}; color: {'black' if color.lightness() > 128 else 'white'};")
            # Re-render all slices
            for ori in ['axial', 'coronal', 'sagittal']:
                self.render_slice(ori, preserve_camera=True)
    
    def add_ruler_overlay(self, renderer, orientation, slice_shape):
        """
        Draw a horizontal scale bar at the bottom-left of the viewport,
        matching the style and behavior of the DEMO app.
        Auto-adjusts length based on current zoom level. Uses display (screen-pixel)
        coordinates so it stays fixed regardless of pan/zoom position.
        """
        try:
            if not hasattr(self, 'ruler_actors'):
                self.ruler_actors = {'axial': [], 'coronal': [], 'sagittal': []}
            for act in self.ruler_actors.get(orientation, []):
                renderer.RemoveActor(act)
            self.ruler_actors[orientation] = []

            h, w = slice_shape
            rc = self.ruler_color  # configurable color
            
            # 1. Compute µm/screen-pixel from camera
            camera = renderer.GetActiveCamera()
            parallel_scale = camera.GetParallelScale()
            win_size = renderer.GetRenderWindow().GetSize()
            viewport = renderer.GetViewport()
            vp_w_px = max((viewport[2] - viewport[0]) * win_size[0], 1)
            vp_h_px = max((viewport[3] - viewport[1]) * win_size[1], 1)
            world_per_px = (2.0 * parallel_scale) / vp_h_px if vp_h_px > 0 else 1.0

            spacing = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing

            # Horizontal axis spacing — correct per orientation
            if orientation == 'axial':       # XY view → horizontal = X
                um_per_world = spacing[0]
            elif orientation == 'coronal':   # XZ view → horizontal = X
                um_per_world = spacing[0]
            else:                            # YZ (sagittal) → horizontal = Z
                um_per_world = spacing[2]

            um_per_px = world_per_px * um_per_world
            if um_per_px <= 0:
                um_per_px = 1.0

            # 2. Choose a "nice" bar length (~18% of viewport width)
            target_um = vp_w_px * 0.18 * um_per_px
            nice_vals = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
            bar_um = nice_vals[-1]
            for nv in nice_vals:
                if nv >= target_um * 0.6:
                    bar_um = nv
                    break
            bar_px = max(bar_um / um_per_px, 20)

            # 3. Layout constants
            mx, my = 18, 16       # margin from bottom-left
            cap_h = 8             # end-cap height

            def _line(x1, y1, x2, y2, color, width):
                src = vtk.vtkLineSource()
                src.SetPoint1(x1, y1, 0)
                src.SetPoint2(x2, y2, 0)
                m = vtk.vtkPolyDataMapper2D()
                m.SetInputConnection(src.GetOutputPort())
                a = vtk.vtkActor2D()
                a.SetMapper(m)
                a.GetProperty().SetColor(*color)
                a.GetProperty().SetLineWidth(width)
                renderer.AddActor(a)
                self.ruler_actors[orientation].append(a)

            # 4. Draw shadow outline (dark) for contrast on bright images
            sc = (0.0, 0.0, 0.0)
            _line(mx, my, mx + bar_px, my, sc, 6)                            # bar shadow
            _line(mx, my - cap_h // 2, mx, my + cap_h // 2, sc, 4)          # left cap shadow
            _line(mx + bar_px, my - cap_h // 2, mx + bar_px, my + cap_h // 2, sc, 4)  # right cap

            # 5. Draw main bar + end caps (colored)
            _line(mx, my, mx + bar_px, my, rc, 3)                            # main bar
            _line(mx, my - cap_h // 2, mx, my + cap_h // 2, rc, 2)          # left cap
            _line(mx + bar_px, my - cap_h // 2, mx + bar_px, my + cap_h // 2, rc, 2)  # right cap

            # 6. Label with unit
            if bar_um >= 1000:
                txt = f"{bar_um / 1000:.0f} mm"
            elif bar_um >= 1:
                txt = f"{int(bar_um)} µm"
            else:
                txt = f"{bar_um:.1f} µm"

            lbl = vtk.vtkTextActor()
            lbl.SetInput(txt)
            tp = lbl.GetTextProperty()
            tp.SetFontSize(10)
            tp.SetColor(*rc)
            tp.BoldOn()
            tp.SetFontFamilyToArial()
            tp.SetShadow(True)
            tp.SetShadowOffset(1, 1)
            # Center label above bar
            lbl.GetPositionCoordinate().SetCoordinateSystemToDisplay()
            lbl.SetPosition(mx + bar_px / 2 - 12, my + cap_h // 2 + 2)
            renderer.AddActor(lbl)
            self.ruler_actors[orientation].append(lbl)

        except Exception as e:
            print("Error in add_ruler_overlay:", e)

    def _on_tf_mode_changed(self, mode):
        """Keep mode buttons in sync; enable W/L-only controls only in W/L mode."""
        is_wl = mode == TransferFunctionWidget.MODE_WL
        if hasattr(self, 'btn_wl_mode') and hasattr(self, 'btn_tf_mode'):
            self.btn_wl_mode.blockSignals(True)
            self.btn_tf_mode.blockSignals(True)
            self.btn_wl_mode.setChecked(is_wl)
            self.btn_tf_mode.setChecked(not is_wl)
            self.btn_wl_mode.blockSignals(False)
            self.btn_tf_mode.blockSignals(False)
        # Opacity-shape icons & gamma are W/L features (TF uses free spline)
        if hasattr(self, '_opacity_shape_btns'):
            for btn in self._opacity_shape_btns.values():
                btn.setEnabled(is_wl)
        for name in ('wl_gamma_slider', 'wl_gamma_spin', 'wl_opacity_slider'):
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(True)  # opacity scale still useful in TF
        if hasattr(self, 'wl_gamma_slider'):
            self.wl_gamma_slider.setEnabled(is_wl)
            self.wl_gamma_spin.setEnabled(is_wl)
        self.update_transfer_function()

    def on_opacity_spline_changed(self, points):
        if not hasattr(self, 'volume_actor') or self.volume_actor is None:
            return
        # Prefer full TF apply so opacity_scale / colour stay consistent
        self.update_transfer_function()

    def on_color_gradient_changed(self, stops):
        if not hasattr(self, 'volume_actor') or self.volume_actor is None:
            return
        prop = self.volume_actor.GetProperty()
        v_min = float(self.view_3d_min_spin.value())
        v_max = float(self.view_3d_max_spin.value())
        if v_max <= v_min: v_max = v_min + 1.0
        
        color = vtk.vtkColorTransferFunction()
        for pos, qcolor in stops:
            intensity = v_min + pos * (v_max - v_min)
            color.AddRGBPoint(intensity, qcolor.redF(), qcolor.greenF(), qcolor.blueF())
            
        prop.SetColor(color)
        self.view_3d_widget.GetRenderWindow().Render()

    # ── Orientation cube (Dragonfly): hover highlight + click to snap ─────

    @staticmethod
    def _theme_primary_rgb():
        """App primary cyan → (r,g,b) floats for VTK."""
        h = SemiconductorTheme.PRIMARY_DEFAULT.lstrip("#")
        try:
            return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except Exception:
            return (0.133, 0.682, 0.820)  # #22aed1 fallback

    @staticmethod
    def _theme_primary_hover_rgb():
        h = SemiconductorTheme.PRIMARY_HOVER.lstrip("#")
        try:
            return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except Exception:
            return (0.235, 0.769, 0.910)  # #3cc4e8

    def _build_orientation_marker(self):
        """
        vtkAssembly: AnnotatedCube (+X… labels) + 6 thin face plates used for
        hover glow (Inno3D primary cyan). Plates start invisible.
        """
        assembly = vtk.vtkAssembly()

        cube = vtk.vtkAnnotatedCubeActor()
        cube.SetXPlusFaceText("+X")
        cube.SetXMinusFaceText("-X")
        cube.SetYPlusFaceText("+Y")
        cube.SetYMinusFaceText("-Y")
        cube.SetZPlusFaceText("+Z")
        cube.SetZMinusFaceText("-Z")
        cube.GetCubeProperty().SetColor(0.93, 0.94, 0.95)
        cube.GetTextEdgesProperty().SetColor(0.15, 0.18, 0.22)
        cube.GetTextEdgesProperty().SetLineWidth(1)
        cube.SetFaceTextScale(0.35)

        face_prop_getters = {
            "+X": cube.GetXPlusFaceProperty,
            "-X": cube.GetXMinusFaceProperty,
            "+Y": cube.GetYPlusFaceProperty,
            "-Y": cube.GetYMinusFaceProperty,
            "+Z": cube.GetZPlusFaceProperty,
            "-Z": cube.GetZMinusFaceProperty,
        }
        self._ori_face_text_props = {}
        for name, getter in face_prop_getters.items():
            prop = getter()
            prop.SetColor(0.12, 0.14, 0.16)
            prop.SetDiffuse(0.8)
            prop.SetAmbient(0.4)
            self._ori_face_text_props[name] = prop

        assembly.AddPart(cube)
        self._axes_cube_actor = cube

        # Thin plates just outside each face — opacity 0 until hover
        # Annotated cube ≈ [-0.5, 0.5]; plates sit slightly outside.
        t = 0.02
        e = 0.50
        o = 0.51
        face_bounds = {
            "+X": (o - t, o + t, -e, e, -e, e),
            "-X": (-o - t, -o + t, -e, e, -e, e),
            "+Y": (-e, e, o - t, o + t, -e, e),
            "-Y": (-e, e, -o - t, -o + t, -e, e),
            "+Z": (-e, e, -e, e, o - t, o + t),
            "-Z": (-e, e, -e, e, -o - t, -o + t),
        }
        self._ori_face_highlights = {}
        pr, pg, pb = self._theme_primary_rgb()
        for name, bounds in face_bounds.items():
            src = vtk.vtkCubeSource()
            src.SetBounds(*bounds)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(src.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(pr, pg, pb)
            prop.SetOpacity(0.0)
            prop.SetAmbient(0.85)
            prop.SetDiffuse(0.5)
            prop.SetSpecular(0.25)
            prop.LightingOn()
            assembly.AddPart(actor)
            self._ori_face_highlights[name] = actor

        return assembly

    def _set_orientation_face_hover(self, face):
        """Highlight one cube face in app primary cyan (or clear if face is None)."""
        if face == getattr(self, "_ori_hover_face", None):
            return
        self._ori_hover_face = face

        pr, pg, pb = self._theme_primary_rgb()
        phr, phg, phb = self._theme_primary_hover_rgb()
        # Default label color (dark on light cube)
        default_text = (0.12, 0.14, 0.16)
        # Text on cyan plate: near-black / deep navy for contrast
        hover_text = (0.04, 0.08, 0.12)

        for name, actor in getattr(self, "_ori_face_highlights", {}).items():
            prop = actor.GetProperty()
            if name == face:
                prop.SetColor(phr, phg, phb)
                prop.SetOpacity(0.72)
            else:
                prop.SetColor(pr, pg, pb)
                prop.SetOpacity(0.0)

        for name, tprop in getattr(self, "_ori_face_text_props", {}).items():
            if name == face:
                tprop.SetColor(*hover_text)
                tprop.SetAmbient(1.0)
                tprop.SetDiffuse(0.2)
            else:
                tprop.SetColor(*default_text)
                tprop.SetAmbient(0.4)
                tprop.SetDiffuse(0.8)

        # Subtle cube body tint while hovering
        cube = getattr(self, "_axes_cube_actor", None)
        if cube is not None:
            if face:
                cube.GetCubeProperty().SetColor(
                    0.75 * 0.93 + 0.25 * pr,
                    0.75 * 0.94 + 0.25 * pg,
                    0.75 * 0.95 + 0.25 * pb,
                )
            else:
                cube.GetCubeProperty().SetColor(0.93, 0.94, 0.95)

        if hasattr(self, "view_3d_widget") and self.view_3d_widget is not None:
            self.view_3d_widget.GetRenderWindow().Render()

    def _on_orientation_cube_hover(self, obj, event):
        """Mouse-move: glow face under cursor (Dragonfly yellow → Inno3D cyan)."""
        if not hasattr(self, "axes_widget") or self.axes_widget is None:
            return
        if not hasattr(self, "view_3d_widget") or self.view_3d_widget is None:
            return
        iren = self.view_3d_widget.GetRenderWindow().GetInteractor()
        if iren is None:
            return
        x, y = iren.GetEventPosition()
        face = self._pick_orientation_cube_face(x, y)

        # Cursor feedback only when over the marker viewport
        over_marker = self._is_over_orientation_viewport(x, y)
        try:
            if face is not None:
                self.view_3d_widget.setCursor(Qt.PointingHandCursor)
            elif over_marker:
                self.view_3d_widget.setCursor(Qt.ArrowCursor)
            # When leaving marker, restore default arrow if we had hover
            elif getattr(self, "_ori_hover_face", None) is not None:
                self.view_3d_widget.unsetCursor()
        except Exception:
            pass

        self._set_orientation_face_hover(face)

    def _is_over_orientation_viewport(self, display_x, display_y):
        if not hasattr(self, "axes_widget") or self.axes_widget is None:
            return False
        ren_win = self.view_3d_widget.GetRenderWindow()
        size = ren_win.GetSize()
        if not size or size[0] <= 0 or size[1] <= 0:
            return False
        w, h = float(size[0]), float(size[1])
        vp = self.axes_widget.GetViewport()
        nx, ny = display_x / w, display_y / h
        return vp[0] <= nx <= vp[2] and vp[1] <= ny <= vp[3]

    def _on_orientation_cube_click(self, obj, event):
        """Left-click on the ±X/±Y/±Z cube → orient volume camera to that face."""
        if not hasattr(self, 'axes_widget') or self.axes_widget is None:
            return
        if not hasattr(self, 'view_3d_widget') or self.view_3d_widget is None:
            return
        iren = self.view_3d_widget.GetRenderWindow().GetInteractor()
        if iren is None:
            return
        x, y = iren.GetEventPosition()
        face = self._pick_orientation_cube_face(x, y)
        if face is None:
            return
        # Consume the click so trackball rotate does not start on the cube
        try:
            iren.SetAbortFlag(1)
        except Exception:
            pass
        # Brief brighter flash on click then keep hover state
        self._set_orientation_face_hover(face)
        self._orient_camera_to_face(face)

    def _pick_orientation_cube_face(self, display_x, display_y):
        """
        Ray-pick the orientation marker cube.

        Returns face label '+X','-X','+Y','-Y','+Z','-Z', or None if the
        click is outside the marker viewport / misses the cube.
        """
        ren_win = self.view_3d_widget.GetRenderWindow()
        size = ren_win.GetSize()
        if not size or size[0] <= 0 or size[1] <= 0:
            return None
        w, h = float(size[0]), float(size[1])
        vp = self.axes_widget.GetViewport()  # xmin, ymin, xmax, ymax in 0..1
        nx = display_x / w
        ny = display_y / h
        if not (vp[0] <= nx <= vp[2] and vp[1] <= ny <= vp[3]):
            return None

        u = (nx - vp[0]) / max(1e-9, (vp[2] - vp[0]))
        v = (ny - vp[1]) / max(1e-9, (vp[3] - vp[1]))
        # NDC in marker viewport
        sx = 2.0 * u - 1.0
        sy = 2.0 * v - 1.0

        cam = self.view_3d_renderer.GetActiveCamera()
        pos = np.array(cam.GetPosition(), dtype=float)
        fp = np.array(cam.GetFocalPoint(), dtype=float)
        # VTK view-plane normal points from focal point toward camera
        vpn = pos - fp
        nrm = np.linalg.norm(vpn)
        if nrm < 1e-12:
            return None
        vpn = vpn / nrm
        vup = np.array(cam.GetViewUp(), dtype=float)
        nrm = np.linalg.norm(vup)
        if nrm < 1e-12:
            return None
        vup = vup / nrm
        vright = np.cross(vup, vpn)
        nrm = np.linalg.norm(vright)
        if nrm < 1e-12:
            return None
        vright = vright / nrm
        vup = np.cross(vpn, vright)  # re-orthogonalise

        # Marker cube sits at origin, axis-aligned; camera shares main orientation.
        # Parallel-style ray through the click (scale covers the unit cube).
        scale = 1.35
        dist = 6.0
        ray_origin = sx * scale * vright + sy * scale * vup + dist * vpn
        ray_dir = -vpn

        return self._ray_hit_unit_cube_face(ray_origin, ray_dir)

    @staticmethod
    def _ray_hit_unit_cube_face(origin, direction):
        """Nearest front-face hit of ray vs cube [-1,1]^3. Returns '+X'… or None."""
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
        nrm = np.linalg.norm(direction)
        if nrm < 1e-12:
            return None
        direction = direction / nrm

        # (outward normal, point on plane, face name)
        planes = [
            (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), "+X"),
            (np.array([-1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), "-X"),
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), "+Y"),
            (np.array([0.0, -1.0, 0.0]), np.array([0.0, -1.0, 0.0]), "-Y"),
            (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), "+Z"),
            (np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, -1.0]), "-Z"),
        ]
        best_t = None
        best_face = None
        eps = 1e-6
        for n, p0, name in planes:
            denom = float(np.dot(n, direction))
            # Only hit when ray approaches from outside (against outward normal)
            if denom >= -1e-12:
                continue
            t = float(np.dot(n, p0 - origin) / denom)
            if t < eps:
                continue
            hit = origin + t * direction
            if (
                abs(hit[0]) <= 1.02
                and abs(hit[1]) <= 1.02
                and abs(hit[2]) <= 1.02
            ):
                if best_t is None or t < best_t:
                    best_t = t
                    best_face = name
        return best_face

    def _orient_camera_to_face(self, face):
        """
        Snap 3D camera so the given cube face points toward the viewer
        (Dragonfly: click +Z → look from +Z, etc.).
        """
        # Map face → existing named preset (world axes)
        face_to_preset = {
            "+X": "Right",
            "-X": "Left",
            "+Y": "Back",
            "-Y": "Front",
            "+Z": "Top",
            "-Z": "Bottom",
        }
        preset = face_to_preset.get(face)
        if preset is None:
            return
        # Works even before volume is loaded (uses default center/distance)
        if self.volume_data is None:
            self._orient_camera_to_face_no_volume(face)
            return
        self.set_camera_preset(preset)

    def _orient_camera_to_face_no_volume(self, face):
        """Fallback orient when no volume is loaded yet."""
        if not hasattr(self, 'view_3d_renderer') or self.view_3d_renderer is None:
            return
        cam = self.view_3d_renderer.GetActiveCamera()
        r = 500.0
        cx = cy = cz = 0.0
        cam.SetFocalPoint(cx, cy, cz)
        dirs = {
            "+X": ((cx + r, cy, cz), (0, 0, 1)),
            "-X": ((cx - r, cy, cz), (0, 0, 1)),
            "+Y": ((cx, cy + r, cz), (0, 0, 1)),
            "-Y": ((cx, cy - r, cz), (0, 0, 1)),
            "+Z": ((cx, cy, cz + r), (0, 1, 0)),
            "-Z": ((cx, cy, cz - r), (0, -1, 0)),
        }
        pos, up = dirs.get(face, ((cx, cy - r, cz), (0, 0, 1)))
        cam.SetPosition(*pos)
        cam.SetViewUp(*up)
        self.view_3d_renderer.ResetCameraClippingRange()
        self.view_3d_widget.GetRenderWindow().Render()

    def set_camera_preset(self, preset):
        if not hasattr(self, 'view_3d_renderer') or self.volume_data is None:
            return
        cam = self.view_3d_renderer.GetActiveCamera()
        cx, cy, cz = self._volume_world_center()
        # Distance from center based on volume extent
        z, y, x = self.volume_data.shape
        sp = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        half = max((x - 1) * sp[0], (y - 1) * sp[1], (z - 1) * sp[2]) * 0.5
        r = max(half, 1.0) * 2.5
        
        cam.SetFocalPoint(cx, cy, cz)
        if preset == "Front":
            cam.SetPosition(cx, cy - r, cz)
            cam.SetViewUp(0, 0, 1)
        elif preset == "Back":
            cam.SetPosition(cx, cy + r, cz)
            cam.SetViewUp(0, 0, 1)
        elif preset == "Right":
            cam.SetPosition(cx + r, cy, cz)
            cam.SetViewUp(0, 0, 1)
        elif preset == "Left":
            cam.SetPosition(cx - r, cy, cz)
            cam.SetViewUp(0, 0, 1)
        elif preset == "Top":
            cam.SetPosition(cx, cy, cz + r)
            cam.SetViewUp(0, 1, 0)
        elif preset == "Bottom":
            cam.SetPosition(cx, cy, cz - r)
            cam.SetViewUp(0, -1, 0)

        self._sync_3d_orbit_pivot()
        self.view_3d_renderer.ResetCameraClippingRange()
        self.view_3d_widget.GetRenderWindow().Render()

    def toggle_auto_rotate(self, checked):
        if checked:
            self.auto_rotate_timer.start(30)
        else:
            self.auto_rotate_timer.stop()
            
    def auto_rotate_step(self):
        if hasattr(self, 'view_3d_renderer') and self.view_3d_renderer:
            cam = self.view_3d_renderer.GetActiveCamera()
            cam.Azimuth(0.5)
            self.view_3d_widget.GetRenderWindow().Render()

    def toggle_bounding_box(self, state):
        if not hasattr(self, 'view_3d_renderer') or self.volume_data is None:
            return
            
        if state == Qt.Checked:
            if self.bbox_actor is None:
                outline = vtk.vtkOutlineFilter()
                z, y, x = self.volume_data.shape
                img = vtk.vtkImageData()
                img.SetDimensions(x, y, z)
                img.SetSpacing(self.spacing[0], self.spacing[1], self.spacing[2])
                outline.SetInputData(img)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(outline.GetOutputPort())
                self.bbox_actor = vtk.vtkActor()
                self.bbox_actor.SetMapper(mapper)
                self.bbox_actor.GetProperty().SetColor(1, 1, 1)
            self.view_3d_renderer.AddActor(self.bbox_actor)
        else:
            if self.bbox_actor:
                self.view_3d_renderer.RemoveActor(self.bbox_actor)
                
        self.view_3d_widget.GetRenderWindow().Render()

    def _get_vtk_world_extent(self):
        """Get world-space extents from VTK data or volume shape.
        
        Returns dict: {'x': (min, max), 'y': (min, max), 'z': (min, max)}
        The VTK volume is always at origin (0,0,0) with:
          X-extent = [0, (dim_x - 1) * spacing_x]
          Y-extent = [0, (dim_y - 1) * spacing_y]
          Z-extent = [0, (dim_z - 1) * spacing_z]
        
        Works correctly regardless of whether the volume is horizontal,
        vertical, tall, or flat — the coordinate system is purely VTK world.
        """
        if self.volume_data is None:
            return {'x': (0.0, 1.0), 'y': (0.0, 1.0), 'z': (0.0, 1.0)}
        
        z, y, x = self.volume_data.shape
        spacing = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        # Spacing indexing: spacing[0]=X, spacing[1]=Y, spacing[2]=Z
        sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        
        return {
            'x': (0.0, max((x - 1) * sx, 1e-6)),
            'y': (0.0, max((y - 1) * sy, 1e-6)),
            'z': (0.0, max((z - 1) * sz, 1e-6)),
        }

    def _build_clip_plane(self, axis, slider_val, flipped=False):
        """Build a vtkPlane for the given axis from a slider value (0..1000).
        
        Dragonfly-style clipping:
        - slider_val 0..1000 maps to 0..100% of the axis extent
        - VTK clips everything where dot(normal, point - origin) < 0
        - Default normal = +1 → keeps everything ABOVE the slider position
        - Flipped normal = -1 → keeps everything BELOW the slider position
        
        This is orientation-agnostic: X, Y, Z always map to VTK world axes
        regardless of volume shape or aspect ratio.
        """
        extent = self._get_vtk_world_extent()
        amin, amax = extent[axis]
        t = max(0.0, min(1.0, slider_val / 1000.0))
        pos = amin + t * (amax - amin)
        
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        normal = [0.0, 0.0, 0.0]
        # +1 = keep positive side, -1 = keep negative side
        normal[axis_idx] = -1.0 if flipped else 1.0
        
        origin = [0.0, 0.0, 0.0]
        origin[axis_idx] = pos
        
        plane = vtk.vtkPlane()
        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)
        return plane

    def _sync_all_clip_planes(self):
        """Apply clipping to all volume mappers using GPU-native cropping.
        
        Uses SetCropping/SetCroppingRegionPlanes for vtkGPUVolumeRayCastMapper
        instead of AddClippingPlane, because VTK's GPU mapper has known
        rendering artifacts with ClippingPlanes from certain viewing angles.
        
        Crop region is the intersection of:
          1) Fullscreen Advanced MPR axis clip planes (if any)
          2) Dragonfly clip-box bounds from the 2×2 right sidebar
        
        Dragonfly behavior: turning CLIP BOX tool OFF only hides the box UI;
        the current crop is KEPT. Reset restores full volume (bounds 0–100%).
        
        Segmentation volume overlays also use GPU cropping when available;
        polydata mappers still receive ClippingPlanes for axis planes.
        """
        extent = self._get_vtk_world_extent()
        
        # Determine crop region from active clip planes
        crop_min = [extent['x'][0], extent['y'][0], extent['z'][0]]
        crop_max = [extent['x'][1], extent['y'][1], extent['z'][1]]
        has_any_clip = False
        
        active_planes = []  # For polydata / legacy clip-plane consumers
        for axis in ['x', 'y', 'z']:
            plane = self.adv_clip_planes.get(axis)
            if plane is None:
                continue
            has_any_clip = True
            active_planes.append(plane)
            
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
            normal = plane.GetNormal()
            origin = plane.GetOrigin()
            pos = origin[axis_idx]
            
            if normal[axis_idx] > 0:
                # Keep positive side (above pos)
                crop_min[axis_idx] = max(crop_min[axis_idx], pos)
            else:
                # Keep negative side (below pos)
                crop_max[axis_idx] = min(crop_max[axis_idx], pos)

        # Dragonfly clip box: apply stored bounds even when the tool UI is off
        # (toggle only shows/hides the interactive box; Reset clears the crop).
        box = self._df_clip_world_bounds()
        if box is not None and self._df_clip_bounds_are_partial():
            has_any_clip = True
            for i, ax in enumerate(['x', 'y', 'z']):
                crop_min[i] = max(crop_min[i], box[ax][0])
                crop_max[i] = min(crop_max[i], box[ax][1])
            # Six half-space planes for polydata consumers
            for ax, idx in (('x', 0), ('y', 1), ('z', 2)):
                lo, hi = box[ax]
                p_lo = vtk.vtkPlane()
                o = [0.0, 0.0, 0.0]
                o[idx] = lo
                n = [0.0, 0.0, 0.0]
                n[idx] = 1.0
                p_lo.SetOrigin(*o)
                p_lo.SetNormal(*n)
                active_planes.append(p_lo)
                p_hi = vtk.vtkPlane()
                o2 = [0.0, 0.0, 0.0]
                o2[idx] = hi
                n2 = [0.0, 0.0, 0.0]
                n2[idx] = -1.0
                p_hi.SetOrigin(*o2)
                p_hi.SetNormal(*n2)
                active_planes.append(p_hi)

        # Ensure non-empty crop region
        for i in range(3):
            if crop_max[i] <= crop_min[i]:
                mid = 0.5 * (extent[['x', 'y', 'z'][i]][0] + extent[['x', 'y', 'z'][i]][1])
                crop_min[i] = mid - 1e-3
                crop_max[i] = mid + 1e-3
        
        # Apply GPU-native cropping to volume + mask volumes
        for actor_name in ['volume_actor', 'c1_actor_3d', 'c2_actor_3d']:
            actor = getattr(self, actor_name, None)
            if actor is None:
                continue
            mapper = actor.GetMapper()
            if mapper is None:
                continue

            if hasattr(mapper, 'SetCropping'):
                if hasattr(mapper, 'RemoveAllClippingPlanes'):
                    mapper.RemoveAllClippingPlanes()
                if has_any_clip:
                    mapper.SetCropping(1)
                    mapper.SetCroppingRegionPlanes(
                        crop_min[0], crop_max[0],
                        crop_min[1], crop_max[1],
                        crop_min[2], crop_max[2]
                    )
                    mapper.SetCroppingRegionFlags(0x2000)  # VTK_CROP_SUBVOLUME
                else:
                    mapper.SetCropping(0)
                mapper.Modified()
            elif hasattr(mapper, 'RemoveAllClippingPlanes'):
                # Polydata / non-GPU path
                mapper.RemoveAllClippingPlanes()
                for p in active_planes:
                    mapper.AddClippingPlane(p)
                mapper.Modified()

    def on_mpr_show_hide_all(self, show):
        for axis, ctrl in self.adv_mpr_controls.items():
            ctrl['show'].blockSignals(True)
            ctrl['show'].setChecked(show)
            ctrl['show'].blockSignals(False)
            self.update_advanced_mpr_clip(axis)

    def on_mpr_reset_all(self):
        for axis, ctrl in self.adv_mpr_controls.items():
            ctrl['slider'].blockSignals(True)
            ctrl['slider'].setValue(500)
            ctrl['pct_label'].setText("50%")
            ctrl['slider'].blockSignals(False)
            
            ctrl['show'].blockSignals(True)
            ctrl['show'].setChecked(False)
            ctrl['show'].blockSignals(False)
            
            ctrl['clip'].blockSignals(True)
            ctrl['clip'].setChecked(False)
            ctrl['clip'].blockSignals(False)
            
            ctrl['flip'].blockSignals(True)
            ctrl['flip'].setChecked(False)
            ctrl['flip'].blockSignals(False)
            
            self.update_advanced_mpr_clip(axis)

    def update_mpr_thickness(self, val):
        self.update_advanced_mpr_clip('x')
        self.update_advanced_mpr_clip('y')
        self.update_advanced_mpr_clip('z')

    def update_advanced_mpr_clip(self, axis):
        """Dragonfly-style clipping: Show = always clip volume.
        
        - Show checked → clip plane applied, volume is cut at slider position
        - Clip checked → additionally show a colored indicator plane
        - Flip → reverse which half is kept
        
        Coordinate logic (orientation-agnostic):
          1. Read world extent from _get_vtk_world_extent() 
          2. Map slider % → position within [0, max_world]
          3. Build vtkPlane at that position with axis-aligned normal
          4. Apply to ALL mappers via _sync_all_clip_planes()
        
        This works correctly regardless of volume orientation (horizontal,
        vertical, tall, flat) because it operates purely in VTK world space.
        """
        if not hasattr(self, 'view_3d_renderer') or self.volume_data is None:
            return
            
        ctrl = self.adv_mpr_controls[axis]
        enabled = ctrl['show'].isChecked()
        show_plane = ctrl['clip'].isChecked()
        flipped = ctrl['flip'].isChecked()
        slider_val = ctrl['slider'].value()
        
        # Update percentage label
        ctrl['pct_label'].setText(f"{slider_val / 10.0:.0f}%")
        
        # 1. Build clip plane — always active when Show is checked
        if enabled:
            self.adv_clip_planes[axis] = self._build_clip_plane(axis, slider_val, flipped)
        else:
            self.adv_clip_planes[axis] = None
        
        # 2. Sync clip planes to ALL volume mappers
        self._sync_all_clip_planes()
            
        # 3. Optional colored indicator plane at the clip position
        if self.adv_mpr_actors[axis]:
            self.view_3d_renderer.RemoveActor(self.adv_mpr_actors[axis])
            self.adv_mpr_actors[axis] = None
            
        if enabled and show_plane:
            extent = self._get_vtk_world_extent()
            t = max(0.0, min(1.0, slider_val / 1000.0))
            pos = extent[axis][0] + t * (extent[axis][1] - extent[axis][0])
            
            # Build a visible rectangle spanning the other two axes
            other_axes = [a for a in ['x', 'y', 'z'] if a != axis]
            a1_min, a1_max = extent[other_axes[0]]
            a2_min, a2_max = extent[other_axes[1]]
            
            plane_src = vtk.vtkPlaneSource()
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
            o1_idx = {'x': 0, 'y': 1, 'z': 2}[other_axes[0]]
            o2_idx = {'x': 0, 'y': 1, 'z': 2}[other_axes[1]]
            
            # Origin point (base corner of the indicator plane)
            origin_pt = [0.0, 0.0, 0.0]
            origin_pt[axis_idx] = pos
            origin_pt[o1_idx] = a1_min
            origin_pt[o2_idx] = a2_min
            
            # Point1: extend along first other axis
            pt1 = list(origin_pt)
            pt1[o1_idx] = a1_max
            
            # Point2: extend along second other axis
            pt2 = list(origin_pt)
            pt2[o2_idx] = a2_max
            
            plane_src.SetOrigin(*origin_pt)
            plane_src.SetPoint1(*pt1)
            plane_src.SetPoint2(*pt2)
                
            p_mapper = vtk.vtkPolyDataMapper()
            p_mapper.SetInputConnection(plane_src.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(p_mapper)
            
            colors = {'x': (1, 0.3, 0.3), 'y': (0.3, 1, 0.3), 'z': (0.3, 0.5, 1)}
            actor.GetProperty().SetColor(*colors[axis])
            actor.GetProperty().SetOpacity(0.35)
            actor.GetProperty().SetLighting(False)
            
            self.adv_mpr_actors[axis] = actor
            self.view_3d_renderer.AddActor(actor)
            
        self.view_3d_widget.GetRenderWindow().Render()

    # ══════════════════════════════════════════════════════════════════════
    # Dragonfly-style clip box (2×2 right sidebar — linked MPR + 3D)
    # Independent from fullscreen Advanced MPR & Clipping axis planes.
    # ══════════════════════════════════════════════════════════════════════

    def _df_clip_world_bounds(self):
        """Return clip-box world bounds from percentage sliders, or None."""
        if self.volume_data is None:
            return None
        extent = self._get_vtk_world_extent()
        out = {}
        for ax in ('x', 'y', 'z'):
            amin, amax = extent[ax]
            lo_t = max(0.0, min(1.0, self._df_clip_bounds[ax][0] / 1000.0))
            hi_t = max(0.0, min(1.0, self._df_clip_bounds[ax][1] / 1000.0))
            if hi_t <= lo_t:
                hi_t = min(1.0, lo_t + 0.001)
            out[ax] = (amin + lo_t * (amax - amin), amin + hi_t * (amax - amin))
        return out

    def _df_clip_bounds_are_partial(self):
        """True if stored clip bounds are smaller than the full volume."""
        b = getattr(self, '_df_clip_bounds', None)
        if not b:
            return False
        for ax in ('x', 'y', 'z'):
            lo, hi = b.get(ax, [0, 1000])
            if int(lo) > 0 or int(hi) < 1000:
                return True
        return False

    def _df_clip_on_enable_toggled(self, checked=False):
        """Toggle clip-box tool UI (Dragonfly): OFF keeps crop, only hides box."""
        self._df_clip_enabled = bool(
            self.btn_df_clip_enable.isChecked() if hasattr(self, 'btn_df_clip_enable') else checked
        )
        if not self._df_clip_enabled:
            # Stop interaction + hide box graphics; do NOT clear crop bounds
            self._df_clip_3d_hover = None
            self._df_clip_3d_drag = None
            self._df_clip_hover = None
            self._df_clip_drag = None
            style = getattr(self, '_3d_interactor_style', None)
            if style is not None:
                style._df_clip_dragging = False
            w = getattr(self, 'view_3d_widget', None)
            if w is not None:
                try:
                    w.unsetCursor()
                except Exception:
                    pass
            self._df_clip_clear_3d_actors()
            self._df_clip_clear_mpr_actors()
            # Re-apply crop from stored bounds (keeps clipped volume)
            self._sync_all_clip_planes()
            if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
                try:
                    self.view_3d_widget.GetRenderWindow().Render()
                except Exception:
                    pass
            for ori in ('axial', 'coronal', 'sagittal'):
                ww = getattr(self, f'{ori}_widget', None)
                if ww is not None:
                    try:
                        ww.GetRenderWindow().Render()
                    except Exception:
                        pass
            return
        # Tool ON: show box at current bounds and re-sync
        self._df_clip_refresh_all()

    def _df_clip_reset(self):
        """Reset clip to full volume — this is what restores the uncropped volume."""
        self._df_clip_bounds = {'x': [0, 1000], 'y': [0, 1000], 'z': [0, 1000]}
        for ax, ctrls in getattr(self, '_df_clip_bound_ctrls', {}).items():
            for key in ('lo', 'hi'):
                ctrls[key].blockSignals(True)
            ctrls['lo'].setValue(0)
            ctrls['hi'].setValue(1000)
            ctrls['pct'].setText("0–100%")
            for key in ('lo', 'hi'):
                ctrls[key].blockSignals(False)
        # Always re-apply crop (full) even if tool UI is off
        self._sync_all_clip_planes()
        if self._df_clip_enabled:
            self._df_clip_update_3d_visuals()
            for ori in ('axial', 'coronal', 'sagittal'):
                self._df_clip_update_mpr_overlay(ori)
        else:
            self._df_clip_clear_3d_actors()
            self._df_clip_clear_mpr_actors()
        if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
            try:
                self.view_3d_widget.GetRenderWindow().Render()
            except Exception:
                pass
        for ori in ('axial', 'coronal', 'sagittal'):
            ww = getattr(self, f'{ori}_widget', None)
            if ww is not None:
                try:
                    ww.GetRenderWindow().Render()
                except Exception:
                    pass

    def _df_clip_on_bound_slider(self, axis, which, value):
        """Update one end of the clip-box axis (0..1000) and re-apply."""
        lo, hi = self._df_clip_bounds[axis]
        if which == 'lo':
            lo = int(value)
            if lo > hi:
                hi = lo
                ctrls = self._df_clip_bound_ctrls.get(axis)
                if ctrls:
                    ctrls['hi'].blockSignals(True)
                    ctrls['hi'].setValue(hi)
                    ctrls['hi'].blockSignals(False)
        else:
            hi = int(value)
            if hi < lo:
                lo = hi
                ctrls = self._df_clip_bound_ctrls.get(axis)
                if ctrls:
                    ctrls['lo'].blockSignals(True)
                    ctrls['lo'].setValue(lo)
                    ctrls['lo'].blockSignals(False)
        self._df_clip_bounds[axis] = [lo, hi]
        ctrls = self._df_clip_bound_ctrls.get(axis)
        if ctrls:
            ctrls['pct'].setText(f"{lo / 10.0:.0f}–{hi / 10.0:.0f}%")
        # Crop always follows bounds (tool ON/OFF only controls box UI)
        if self._df_clip_enabled:
            self._df_clip_refresh_all()
        else:
            self._sync_all_clip_planes()
            if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
                try:
                    self.view_3d_widget.GetRenderWindow().Render()
                except Exception:
                    pass

    def _df_clip_on_options_changed(self, *_args):
        """Sync checkbox / grid-size options and redraw visuals."""
        if hasattr(self, 'chk_df_clip_keep'):
            self._df_clip_keep_when_hidden = self.chk_df_clip_keep.isChecked()
        if hasattr(self, 'chk_df_clip_grid_on_obj'):
            self._df_clip_display_grid_on_object = self.chk_df_clip_grid_on_obj.isChecked()
        if hasattr(self, 'chk_df_clip_grid_lines'):
            self._df_clip_show_grid = self.chk_df_clip_grid_lines.isChecked()
        if hasattr(self, 'chk_df_clip_borders'):
            self._df_clip_show_borders = self.chk_df_clip_borders.isChecked()
        if hasattr(self, 'chk_df_clip_axes'):
            self._df_clip_show_axes = self.chk_df_clip_axes.isChecked()
        if hasattr(self, 'df_clip_grid_size_spin'):
            self._df_clip_grid_size = float(self.df_clip_grid_size_spin.value())
        # Options only affect visuals (crop already applied when enabled)
        if self._df_clip_enabled or self._df_clip_keep_when_hidden:
            self._df_clip_update_3d_visuals()
            for ori in ('axial', 'coronal', 'sagittal'):
                self._df_clip_update_mpr_overlay(ori)
            if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
                self.view_3d_widget.GetRenderWindow().Render()
            for ori in ('axial', 'coronal', 'sagittal'):
                w = getattr(self, f'{ori}_widget', None)
                if w is not None:
                    try:
                        w.GetRenderWindow().Render()
                    except Exception:
                        pass

    def _df_clip_refresh_all(self):
        """Re-apply crop + rebuild 3D/MPR box visuals."""
        self._sync_all_clip_planes()
        self._df_clip_update_3d_visuals()
        for ori in ('axial', 'coronal', 'sagittal'):
            self._df_clip_update_mpr_overlay(ori)
        if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
            try:
                self.view_3d_widget.GetRenderWindow().Render()
            except Exception:
                pass
        for ori in ('axial', 'coronal', 'sagittal'):
            w = getattr(self, f'{ori}_widget', None)
            if w is not None:
                try:
                    w.GetRenderWindow().Render()
                except Exception:
                    pass

    def _df_clip_clear_3d_actors(self):
        ren = getattr(self, 'view_3d_renderer', None)
        if ren is not None:
            for a in getattr(self, '_df_clip_3d_actors', []):
                try:
                    ren.RemoveActor(a)
                except Exception:
                    pass
        self._df_clip_3d_actors = []
        self._df_clip_3d_face_actors = {}

    def _df_clip_clear_mpr_actors(self, orientation=None):
        oris = [orientation] if orientation else ['axial', 'coronal', 'sagittal']
        for ori in oris:
            ren = getattr(self, f'{ori}_renderer', None)
            actors = self._df_clip_mpr_actors.get(ori, [])
            if ren is not None:
                for a in actors:
                    try:
                        ren.RemoveActor(a)
                    except Exception:
                        pass
            self._df_clip_mpr_actors[ori] = []

    def _df_clip_should_show_visuals(self):
        """Whether the clip-box graphics should be drawn."""
        if self.volume_data is None:
            return False
        if self._df_clip_enabled:
            return True
        # Keep box when volume is "hidden" (volume actor invisible / missing)
        if self._df_clip_keep_when_hidden and getattr(self, 'volume_actor', None) is not None:
            try:
                if not self.volume_actor.GetVisibility():
                    return True
            except Exception:
                pass
        return False

    def _df_clip_add_line_actor_3d(self, p1, p2, color, width=1.5):
        src = vtk.vtkLineSource()
        src.SetPoint1(*p1)
        src.SetPoint2(*p2)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(src.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(width)
        actor.GetProperty().SetLighting(False)
        actor.GetProperty().SetOpacity(0.95)
        return actor

    def _df_clip_add_face_plate_3d(self, origin, p1, p2, color, opacity=0.12):
        """Semi-transparent rectangle for one clip-box face."""
        src = vtk.vtkPlaneSource()
        src.SetOrigin(*origin)
        src.SetPoint1(*p1)
        src.SetPoint2(*p2)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(src.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetLighting(False)
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        actor.GetProperty().BackfaceCullingOff()
        return actor

    # ── 3D viewport face drag (Dragonfly) ───────────────────────────────
    # Primary path: high-priority VTK observers + Qt eventFilter backup.
    # Pick: display-space proximity to face-center handles (reliable),
    #       with geometric ray–face as secondary.

    def _df_clip_3d_qt_to_display(self, qt_pos):
        """Qt widget coords (top-left origin) → VTK display (bottom-left)."""
        w = getattr(self, 'view_3d_widget', None)
        if w is None:
            return None
        try:
            size = w.GetRenderWindow().GetSize()
        except Exception:
            return None
        return float(qt_pos.x()), float(size[1] - qt_pos.y())

    def _df_clip_3d_event_display(self):
        """Current interactor display position, or None."""
        w = getattr(self, 'view_3d_widget', None)
        if w is None:
            return None
        iren = w.GetRenderWindow().GetInteractor()
        if iren is None:
            return None
        return iren.GetEventPosition()

    def _on_df_clip_3d_left_press(self, obj, event):
        """VTK observer: grab clip face before camera Track starts."""
        if not getattr(self, '_df_clip_enabled', False) or self.volume_data is None:
            return
        if self._df_clip_3d_on_left_down():
            style = getattr(self, '_3d_interactor_style', None)
            if style is not None:
                style._df_clip_dragging = True
                style._track_active = False
            # Abort further observers / style for this press when possible
            try:
                if hasattr(obj, 'SetAbortFlag'):
                    obj.SetAbortFlag(1)
            except Exception:
                pass

    def _on_df_clip_3d_mouse_move(self, obj, event):
        """VTK observer: face drag or hover highlight."""
        if getattr(self, '_df_clip_3d_drag', None):
            self._df_clip_3d_on_mouse_move()
            return
        if getattr(self, '_df_clip_enabled', False) and self.volume_data is not None:
            self._df_clip_3d_on_hover()

    def _on_df_clip_3d_left_release(self, obj, event):
        """VTK observer: end face drag."""
        if getattr(self, '_df_clip_3d_drag', None):
            self._df_clip_3d_on_left_up()
            style = getattr(self, '_3d_interactor_style', None)
            if style is not None:
                style._df_clip_dragging = False

    def _df_clip_3d_try_grab_at_qt_pos(self, qt_pos):
        """Qt eventFilter press path. Returns True if face grabbed."""
        disp = self._df_clip_3d_qt_to_display(qt_pos)
        if disp is None:
            return False
        return self._df_clip_3d_begin_drag_at_display(disp[0], disp[1])

    def _df_clip_3d_drag_at_qt_pos(self, qt_pos):
        disp = self._df_clip_3d_qt_to_display(qt_pos)
        if disp is None:
            return
        self._df_clip_3d_update_drag_at_display(disp[0], disp[1])

    def _df_clip_3d_hover_at_qt_pos(self, qt_pos):
        disp = self._df_clip_3d_qt_to_display(qt_pos)
        if disp is None:
            return
        self._df_clip_3d_hover_at_display(disp[0], disp[1])

    def _df_clip_3d_display_ray(self, dx, dy):
        """Camera ray through display pixel → (origin, direction) world."""
        ren = getattr(self, 'view_3d_renderer', None)
        if ren is None:
            return None
        try:
            ren.SetDisplayPoint(float(dx), float(dy), 0.0)
            ren.DisplayToWorld()
            n = ren.GetWorldPoint()
            if abs(n[3]) < 1e-12:
                return None
            near = [n[0] / n[3], n[1] / n[3], n[2] / n[3]]

            ren.SetDisplayPoint(float(dx), float(dy), 1.0)
            ren.DisplayToWorld()
            f = ren.GetWorldPoint()
            if abs(f[3]) < 1e-12:
                return None
            far = [f[0] / f[3], f[1] / f[3], f[2] / f[3]]
        except Exception:
            return None

        d = [far[0] - near[0], far[1] - near[1], far[2] - near[2]]
        ln = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) ** 0.5
        if ln < 1e-15:
            return None
        d = [d[0] / ln, d[1] / ln, d[2] / ln]
        return near, d

    def _df_clip_3d_faces(self):
        """Six clip faces with center points for handle picking."""
        box = self._df_clip_world_bounds()
        if box is None:
            return []
        x0, x1 = box['x']
        y0, y1 = box['y']
        z0, z1 = box['z']
        mx, my, mz = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)
        # (axis, side, normal, plane_point, a0,a1, b0,b1, a_idx, b_idx, center)
        return [
            ('x', 'lo', (-1.0, 0.0, 0.0), (x0, y0, z0), y0, y1, z0, z1, 1, 2, (x0, my, mz)),
            ('x', 'hi', (1.0, 0.0, 0.0), (x1, y0, z0), y0, y1, z0, z1, 1, 2, (x1, my, mz)),
            ('y', 'lo', (0.0, -1.0, 0.0), (x0, y0, z0), x0, x1, z0, z1, 0, 2, (mx, y0, mz)),
            ('y', 'hi', (0.0, 1.0, 0.0), (x0, y1, z0), x0, x1, z0, z1, 0, 2, (mx, y1, mz)),
            ('z', 'lo', (0.0, 0.0, -1.0), (x0, y0, z0), x0, x1, y0, y1, 0, 1, (mx, my, z0)),
            ('z', 'hi', (0.0, 0.0, 1.0), (x0, y0, z1), x0, x1, y0, y1, 0, 1, (mx, my, z1)),
        ]

    def _get_df_clip_3d_face_hit(self, dx, dy, pad=0.0):
        """Pick a clip face under display pixel.

        Prefer display-space hit on face-center handles (works when box is full
        volume / faces edge-on). Fall back to geometric ray–face intersection.
        """
        import math
        ren = getattr(self, 'view_3d_renderer', None)
        if ren is None or self.volume_data is None:
            return None
        faces = self._df_clip_3d_faces()
        if not faces:
            return None

        # 1) Display-space proximity to face-center handles
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        handle_tol = 22.0  # pixels — generous for screenshot-scale views
        best_h = None
        best_hd = handle_tol
        for axis, side, normal, p0, a0, a1, b0, b1, ai, bi, center in faces:
            coord.SetValue(float(center[0]), float(center[1]), float(center[2]))
            try:
                dp = coord.GetComputedDisplayValue(ren)
            except Exception:
                continue
            dist = math.hypot(float(dx) - float(dp[0]), float(dy) - float(dp[1]))
            if dist <= best_hd:
                best_hd = dist
                best_h = {
                    'axis': axis,
                    'side': side,
                    'point': list(center),
                    't': 0.0,
                    'normal': normal,
                    'via': 'handle',
                }
        if best_h is not None:
            return best_h

        # 2) Geometric ray–face (full plate pick)
        ray = self._df_clip_3d_display_ray(dx, dy)
        if ray is None:
            return None
        origin, direction = ray
        best = None
        best_t = 1e30
        eps = max(pad, 1e-4)
        for axis, side, normal, p0, a0, a1, b0, b1, ai, bi, center in faces:
            nd = normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2]
            if abs(nd) < 1e-10:
                continue
            w = [p0[0] - origin[0], p0[1] - origin[1], p0[2] - origin[2]]
            t = (normal[0] * w[0] + normal[1] * w[1] + normal[2] * w[2]) / nd
            if t < 0.0:
                continue
            hit = [
                origin[0] + t * direction[0],
                origin[1] + t * direction[1],
                origin[2] + t * direction[2],
            ]
            a = hit[ai]
            b = hit[bi]
            if (a0 - eps) <= a <= (a1 + eps) and (b0 - eps) <= b <= (b1 + eps):
                if t < best_t:
                    best_t = t
                    best = {
                        'axis': axis,
                        'side': side,
                        'point': hit,
                        't': t,
                        'normal': normal,
                        'via': 'ray',
                    }
        return best

    def _df_clip_3d_hover_at_display(self, dx, dy):
        """Highlight face under display pixel; redraw only on change."""
        if not getattr(self, '_df_clip_enabled', False) or self.volume_data is None:
            return
        if getattr(self, '_df_clip_3d_drag', None):
            return
        box = self._df_clip_world_bounds()
        pad = 0.0
        if box is not None:
            span = max(
                box['x'][1] - box['x'][0],
                box['y'][1] - box['y'][0],
                box['z'][1] - box['z'][0],
                1.0,
            )
            pad = span * 0.03
        hit = self._get_df_clip_3d_face_hit(dx, dy, pad=pad)
        key = (hit['axis'], hit['side']) if hit else None
        prev = self._df_clip_3d_hover
        prev_key = (prev['axis'], prev['side']) if prev else None
        w = getattr(self, 'view_3d_widget', None)
        if key == prev_key:
            if w is not None:
                if hit is not None:
                    w.setCursor(Qt.SizeAllCursor)
                elif prev is not None:
                    w.unsetCursor()
            return
        self._df_clip_3d_hover = (
            {'axis': hit['axis'], 'side': hit['side']} if hit else None
        )
        self._df_clip_update_3d_visuals()
        if w is not None:
            if hit is not None:
                w.setCursor(Qt.SizeAllCursor)
            else:
                w.unsetCursor()
            try:
                w.GetRenderWindow().Render()
            except Exception:
                pass

    def _df_clip_3d_on_hover(self):
        """Hover via VTK interactor event position."""
        pos = self._df_clip_3d_event_display()
        if pos is None:
            return
        self._df_clip_3d_hover_at_display(pos[0], pos[1])

    def _df_clip_3d_begin_drag_at_display(self, dx, dy):
        """Start face drag at display pixel. Returns True if consumed."""
        if not getattr(self, '_df_clip_enabled', False) or self.volume_data is None:
            return False
        # Already dragging — ignore re-entrant begin (style + observer double fire)
        if getattr(self, '_df_clip_3d_drag', None):
            return True
        box = self._df_clip_world_bounds()
        pad = 0.0
        if box is not None:
            span = max(
                box['x'][1] - box['x'][0],
                box['y'][1] - box['y'][0],
                box['z'][1] - box['z'][0],
                1.0,
            )
            pad = span * 0.03
        hit = self._get_df_clip_3d_face_hit(dx, dy, pad=pad)
        if hit is None:
            return False

        ren = self.view_3d_renderer
        cam = ren.GetActiveCamera()
        fn = list(hit['normal'])
        pos_c = cam.GetPosition()
        fp = cam.GetFocalPoint()
        view = [fp[0] - pos_c[0], fp[1] - pos_c[1], fp[2] - pos_c[2]]
        vn = (view[0] ** 2 + view[1] ** 2 + view[2] ** 2) ** 0.5
        if vn < 1e-12:
            view = [0.0, 0.0, -1.0]
        else:
            view = [view[0] / vn, view[1] / vn, view[2] / vn]
        side = [
            view[1] * fn[2] - view[2] * fn[1],
            view[2] * fn[0] - view[0] * fn[2],
            view[0] * fn[1] - view[1] * fn[0],
        ]
        sn = (side[0] ** 2 + side[1] ** 2 + side[2] ** 2) ** 0.5
        if sn < 1e-8:
            up = list(cam.GetViewUp())
            side = [
                up[1] * fn[2] - up[2] * fn[1],
                up[2] * fn[0] - up[0] * fn[2],
                up[0] * fn[1] - up[1] * fn[0],
            ]
            sn = (side[0] ** 2 + side[1] ** 2 + side[2] ** 2) ** 0.5
        if sn < 1e-8:
            return False
        side = [side[0] / sn, side[1] / sn, side[2] / sn]
        dpn = [
            fn[1] * side[2] - fn[2] * side[1],
            fn[2] * side[0] - fn[0] * side[2],
            fn[0] * side[1] - fn[1] * side[0],
        ]
        dpn_n = (dpn[0] ** 2 + dpn[1] ** 2 + dpn[2] ** 2) ** 0.5
        if dpn_n < 1e-8:
            return False
        dpn = [dpn[0] / dpn_n, dpn[1] / dpn_n, dpn[2] / dpn_n]

        import copy
        self._df_clip_3d_drag = {
            'axis': hit['axis'],
            'side': hit['side'],
            'start_bounds': copy.deepcopy(self._df_clip_bounds),
            'start_point': list(hit['point']),
            'plane_point': list(hit['point']),
            'plane_normal': dpn,
            'face_normal': fn,
        }
        self._df_clip_3d_hover = {'axis': hit['axis'], 'side': hit['side']}
        style = getattr(self, '_3d_interactor_style', None)
        if style is not None:
            style._df_clip_dragging = True
            style._track_active = False
        self._df_clip_update_3d_visuals()
        w = getattr(self, 'view_3d_widget', None)
        if w is not None:
            w.setCursor(Qt.SizeAllCursor)
            try:
                w.GetRenderWindow().Render()
            except Exception:
                pass
        return True

    def _df_clip_3d_on_left_down(self):
        """Begin face drag from VTK event position. Returns True if consumed."""
        pos = self._df_clip_3d_event_display()
        if pos is None:
            return False
        return self._df_clip_3d_begin_drag_at_display(pos[0], pos[1])

    def _df_clip_3d_update_drag_at_display(self, dx, dy):
        """Update clip bound while dragging a 3D face (display coords)."""
        drag = getattr(self, '_df_clip_3d_drag', None)
        if not drag or self.volume_data is None:
            return
        ray = self._df_clip_3d_display_ray(dx, dy)
        if ray is None:
            return
        origin, direction = ray
        n = drag['plane_normal']
        p0 = drag['plane_point']
        nd = n[0] * direction[0] + n[1] * direction[1] + n[2] * direction[2]
        if abs(nd) < 1e-10:
            return
        w = [p0[0] - origin[0], p0[1] - origin[1], p0[2] - origin[2]]
        t = (n[0] * w[0] + n[1] * w[1] + n[2] * w[2]) / nd
        hit = [
            origin[0] + t * direction[0],
            origin[1] + t * direction[1],
            origin[2] + t * direction[2],
        ]
        axis = drag['axis']
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        world_pos = hit[axis_idx]

        extent = self._get_vtk_world_extent()
        amin, amax = extent[axis]
        span = max(amax - amin, 1e-12)
        pct = int(round(max(0.0, min(1.0, (world_pos - amin) / span)) * 1000.0))

        bounds = {
            'x': list(drag['start_bounds']['x']),
            'y': list(drag['start_bounds']['y']),
            'z': list(drag['start_bounds']['z']),
        }
        lo0, hi0 = drag['start_bounds'][axis]
        if drag['side'] == 'lo':
            lo = max(0, min(pct, hi0 - 1))
            hi = hi0
        else:
            hi = min(1000, max(pct, lo0 + 1))
            lo = lo0
        bounds[axis] = [lo, hi]
        self._df_clip_bounds = bounds
        self._df_clip_sync_bound_sliders()
        self._sync_all_clip_planes()
        self._df_clip_update_3d_visuals()
        for o in ('axial', 'coronal', 'sagittal'):
            self._df_clip_update_mpr_overlay(o)
            ww = getattr(self, f'{o}_widget', None)
            if ww is not None:
                try:
                    ww.GetRenderWindow().Render()
                except Exception:
                    pass
        if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
            try:
                self.view_3d_widget.GetRenderWindow().Render()
            except Exception:
                pass

    def _df_clip_3d_on_mouse_move(self):
        """Update drag from VTK event position."""
        pos = self._df_clip_3d_event_display()
        if pos is None:
            return
        self._df_clip_3d_update_drag_at_display(pos[0], pos[1])

    def _df_clip_3d_on_left_up(self):
        self._df_clip_3d_drag = None
        style = getattr(self, '_3d_interactor_style', None)
        if style is not None:
            style._df_clip_dragging = False
        self._df_clip_refresh_all()
        # Restore hover cursor if still over a face
        self._df_clip_3d_on_hover()

    def _df_clip_update_3d_visuals(self):
        """Draw Dragonfly-style clip box (borders / face grids / axes) in 3D."""
        self._df_clip_clear_3d_actors()
        ren = getattr(self, 'view_3d_renderer', None)
        if ren is None or not self._df_clip_should_show_visuals():
            return
        show_borders = self._df_clip_show_borders or self._df_clip_enabled
        if not (show_borders or self._df_clip_show_grid
                or self._df_clip_display_grid_on_object or self._df_clip_show_axes):
            return

        box = self._df_clip_world_bounds()
        if box is None:
            return
        x0, x1 = box['x']
        y0, y1 = box['y']
        z0, z1 = box['z']

        # Active / hovered face for highlight
        active_face = None
        if getattr(self, '_df_clip_3d_drag', None):
            active_face = (
                self._df_clip_3d_drag.get('axis'),
                self._df_clip_3d_drag.get('side'),
            )
        elif getattr(self, '_df_clip_3d_hover', None):
            active_face = (
                self._df_clip_3d_hover.get('axis'),
                self._df_clip_3d_hover.get('side'),
            )

        # Face plates only when hovered/dragged (no permanent sphere handles)
        if self._df_clip_enabled and active_face is not None:
            face_defs = [
                # axis, side, origin, p1, p2, color
                ('x', 'lo', (x0, y0, z0), (x0, y1, z0), (x0, y0, z1), (1.0, 0.3, 0.3)),
                ('x', 'hi', (x1, y0, z0), (x1, y1, z0), (x1, y0, z1), (1.0, 0.3, 0.3)),
                ('y', 'lo', (x0, y0, z0), (x1, y0, z0), (x0, y0, z1), (0.3, 1.0, 0.3)),
                ('y', 'hi', (x0, y1, z0), (x1, y1, z0), (x0, y1, z1), (0.3, 1.0, 0.3)),
                ('z', 'lo', (x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (0.35, 0.55, 1.0)),
                ('z', 'hi', (x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (0.35, 0.55, 1.0)),
            ]
            for ax, side, origin, p1, p2, col in face_defs:
                if active_face != (ax, side):
                    continue
                plate = self._df_clip_add_face_plate_3d(origin, p1, p2, col, opacity=0.28)
                ren.AddActor(plate)
                self._df_clip_3d_actors.append(plate)

        # 8 corners
        corners = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        # 12 edges (i,j)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        # Edges belonging to each face (for highlight thickness)
        face_edge_map = {
            ('x', 'lo'): {(3, 0), (0, 4), (4, 7), (7, 3), (0, 3), (4, 0), (7, 4), (3, 7)},
            ('x', 'hi'): {(1, 2), (2, 6), (6, 5), (5, 1), (2, 1), (6, 2), (5, 6), (1, 5)},
            ('y', 'lo'): {(0, 1), (1, 5), (5, 4), (4, 0), (1, 0), (5, 1), (4, 5), (0, 4)},
            ('y', 'hi'): {(3, 2), (2, 6), (6, 7), (7, 3), (2, 3), (6, 2), (7, 6), (3, 7)},
            ('z', 'lo'): {(0, 1), (1, 2), (2, 3), (3, 0), (1, 0), (2, 1), (3, 2), (0, 3)},
            ('z', 'hi'): {(4, 5), (5, 6), (6, 7), (7, 4), (5, 4), (6, 5), (7, 6), (4, 7)},
        }
        border_color = (0.85, 0.9, 1.0)
        if show_borders:
            for i, j in edges:
                ekey = (i, j)
                ekey_r = (j, i)
                thick = 2.0
                col = border_color
                if active_face is not None:
                    em = face_edge_map.get(active_face, set())
                    if ekey in em or ekey_r in em:
                        thick = 3.5
                        col = (1.0, 0.95, 0.4)
                actor = self._df_clip_add_line_actor_3d(corners[i], corners[j], col, thick)
                ren.AddActor(actor)
                self._df_clip_3d_actors.append(actor)

        # Face grids (on the six faces of the box) — density capped for perf
        show_grid = self._df_clip_show_grid or self._df_clip_display_grid_on_object
        if show_grid:
            gs = max(float(self._df_clip_grid_size), 1e-3)
            # Cap ~24 lines per axis per face to keep FPS high on large volumes
            max_lines = 24
            dx = max(x1 - x0, 1e-6)
            dy = max(y1 - y0, 1e-6)
            dz = max(z1 - z0, 1e-6)
            gs_x = max(gs, dx / max_lines)
            gs_y = max(gs, dy / max_lines)
            gs_z = max(gs, dz / max_lines)
            grid_color = (0.55, 0.75, 0.95)

            def _grid_range(a0, a1, step):
                vals = []
                t = a0
                while t <= a1 + 1e-9:
                    vals.append(t)
                    t += step
                if not vals or abs(vals[-1] - a1) > 1e-6:
                    vals.append(a1)
                return vals

            # XY faces (z = z0, z1)
            for zf in (z0, z1):
                for x in _grid_range(x0, x1, gs_x):
                    actor = self._df_clip_add_line_actor_3d((x, y0, zf), (x, y1, zf), grid_color, 1.0)
                    ren.AddActor(actor)
                    self._df_clip_3d_actors.append(actor)
                for y in _grid_range(y0, y1, gs_y):
                    actor = self._df_clip_add_line_actor_3d((x0, y, zf), (x1, y, zf), grid_color, 1.0)
                    ren.AddActor(actor)
                    self._df_clip_3d_actors.append(actor)
            # XZ faces (y = y0, y1)
            for yf in (y0, y1):
                for x in _grid_range(x0, x1, gs_x):
                    actor = self._df_clip_add_line_actor_3d((x, yf, z0), (x, yf, z1), grid_color, 1.0)
                    ren.AddActor(actor)
                    self._df_clip_3d_actors.append(actor)
                for z in _grid_range(z0, z1, gs_z):
                    actor = self._df_clip_add_line_actor_3d((x0, yf, z), (x1, yf, z), grid_color, 1.0)
                    ren.AddActor(actor)
                    self._df_clip_3d_actors.append(actor)
            # YZ faces (x = x0, x1)
            for xf in (x0, x1):
                for y in _grid_range(y0, y1, gs_y):
                    actor = self._df_clip_add_line_actor_3d((xf, y, z0), (xf, y, z1), grid_color, 1.0)
                    ren.AddActor(actor)
                    self._df_clip_3d_actors.append(actor)
                for z in _grid_range(z0, z1, gs_z):
                    actor = self._df_clip_add_line_actor_3d((xf, y0, z), (xf, y1, z), grid_color, 1.0)
                    ren.AddActor(actor)
                    self._df_clip_3d_actors.append(actor)

        # RGB axes at box origin (min corner)
        if self._df_clip_show_axes:
            extent = self._get_vtk_world_extent()
            L = 0.15 * max(
                extent['x'][1] - extent['x'][0],
                extent['y'][1] - extent['y'][0],
                extent['z'][1] - extent['z'][0],
                1.0,
            )
            for p2, col in (
                ((x0 + L, y0, z0), (1.0, 0.25, 0.25)),
                ((x0, y0 + L, z0), (0.25, 1.0, 0.25)),
                ((x0, y0, z0 + L), (0.3, 0.45, 1.0)),
            ):
                actor = self._df_clip_add_line_actor_3d((x0, y0, z0), p2, col, 2.5)
                ren.AddActor(actor)
                self._df_clip_3d_actors.append(actor)

    def _df_clip_plane_rect(self, orientation):
        """Clip-box rectangle in MPR display (voxel) coords + axis mapping.

        Returns dict:
          u0,u1,v0,v1  — display-space rect (always u0<=u1, v0<=v1)
          u_axis, v_axis — 'x'|'y'|'z'
          u_lo_is_min, v_lo_is_min — whether display-lo edge is axis min
          h_color, v_color, u_spacing, v_spacing
        """
        if self.volume_data is None:
            return None
        box = self._df_clip_world_bounds()
        if box is None:
            return None
        spacing = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        vol_z, vol_y, vol_x = self.volume_data.shape

        vx0 = box['x'][0] / max(sx, 1e-12)
        vx1 = box['x'][1] / max(sx, 1e-12)
        vy0 = box['y'][0] / max(sy, 1e-12)
        vy1 = box['y'][1] / max(sy, 1e-12)
        vz0 = box['z'][0] / max(sz, 1e-12)
        vz1 = box['z'][1] / max(sz, 1e-12)

        if orientation == 'axial':
            return {
                'u0': vx0, 'u1': vx1, 'v0': vy0, 'v1': vy1,
                'u_axis': 'x', 'v_axis': 'y',
                'u_lo_is_min': True, 'v_lo_is_min': True,
                'h_color': (1.0, 0.25, 0.25), 'v_color': (0.25, 1.0, 0.2),
                'u_spacing': sx, 'v_spacing': sy,
            }
        if orientation == 'coronal':
            # flipud(z): display_v = (z-1) - vz  →  low display = high z
            dv0 = (vol_z - 1) - vz1  # edge at z_max
            dv1 = (vol_z - 1) - vz0  # edge at z_min
            return {
                'u0': vx0, 'u1': vx1,
                'v0': min(dv0, dv1), 'v1': max(dv0, dv1),
                'u_axis': 'x', 'v_axis': 'z',
                'u_lo_is_min': True, 'v_lo_is_min': False,  # display lo = z max
                'h_color': (1.0, 0.25, 0.25), 'v_color': (0.25, 0.45, 1.0),
                'u_spacing': sx, 'v_spacing': sz,
            }
        # sagittal: display x=z, y=y
        return {
            'u0': vz0, 'u1': vz1, 'v0': vy0, 'v1': vy1,
            'u_axis': 'z', 'v_axis': 'y',
            'u_lo_is_min': True, 'v_lo_is_min': True,
            'h_color': (0.25, 0.45, 1.0), 'v_color': (0.25, 1.0, 0.2),
            'u_spacing': sz, 'v_spacing': sy,
        }

    def _df_clip_pick_display_uv(self, pos, orientation):
        """Map Qt mouse pos → MPR display (u,v) in voxel/image coords."""
        widget = getattr(self, f'{orientation}_widget', None)
        renderer = getattr(self, f'{orientation}_renderer', None)
        if widget is None or renderer is None:
            return None
        size = widget.GetRenderWindow().GetSize()
        x = float(pos.x())
        vtk_y = float(size[1] - pos.y())
        picker = vtk.vtkWorldPointPicker()
        picker.Pick(x, vtk_y, 0, renderer)
        wp = picker.GetPickPosition()
        return float(wp[0]), float(wp[1])

    def _df_clip_display_to_pct(self, orientation, u=None, v=None):
        """Convert display u/v (voxel) to clip % (0..1000) for the mapped axes.

        Returns dict partial: {axis: pct_int, ...}
        """
        rect = self._df_clip_plane_rect(orientation)
        if rect is None or self.volume_data is None:
            return {}
        spacing = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        vol_z = self.volume_data.shape[0]
        extent = self._get_vtk_world_extent()
        out = {}

        def _axis_pct(axis, voxel):
            sp = {'x': sx, 'y': sy, 'z': sz}[axis]
            world = float(voxel) * sp
            amin, amax = extent[axis]
            span = max(amax - amin, 1e-12)
            t = (world - amin) / span
            return int(round(max(0.0, min(1.0, t)) * 1000.0))

        if u is not None:
            u_axis = rect['u_axis']
            # display u always maps monotonically for axial/sagittal/coronal-x
            out[u_axis] = _axis_pct(u_axis, u)
        if v is not None:
            v_axis = rect['v_axis']
            if orientation == 'coronal' and v_axis == 'z':
                # display_v = (z-1) - vz  →  vz = (z-1) - display_v
                vz = (vol_z - 1) - float(v)
                out[v_axis] = _axis_pct(v_axis, vz)
            else:
                out[v_axis] = _axis_pct(v_axis, v)
        return out

    @staticmethod
    def _df_clip_hit_key(hit):
        """Stable compare key for hover redraws."""
        if hit is None:
            return None
        return (hit.get('kind'), hit.get('u_side'), hit.get('v_side'), hit.get('orientation'))

    @staticmethod
    def _df_clip_cursor_for_hit(hit):
        if hit is None:
            return Qt.ArrowCursor
        kind = hit.get('kind')
        if kind == 'move':
            return Qt.SizeAllCursor
        if kind == 'corner':
            # Diagonal: lo/lo & hi/hi → FDiag; lo/hi & hi/lo → BDiag
            us, vs = hit.get('u_side'), hit.get('v_side')
            if (us == 'lo' and vs == 'lo') or (us == 'hi' and vs == 'hi'):
                return Qt.SizeFDiagCursor
            return Qt.SizeBDiagCursor
        if kind == 'edge':
            if hit.get('u_side') is not None:
                return Qt.SizeHorCursor
            return Qt.SizeVerCursor
        return Qt.ArrowCursor

    def _get_df_clip_hit(self, pos, orientation):
        """Hit-test clip-box edges / corners / interior on an MPR pane.

        Returns hit dict or None:
          kind: 'corner'|'edge'|'move'
          u_side: 'lo'|'hi'|None
          v_side: 'lo'|'hi'|None
        """
        import math
        if not getattr(self, '_df_clip_enabled', False) or self.volume_data is None:
            return None
        rect = self._df_clip_plane_rect(orientation)
        if rect is None:
            return None
        ren = getattr(self, f'{orientation}_renderer', None)
        widget = getattr(self, f'{orientation}_widget', None)
        if ren is None or widget is None:
            return None

        # Project rect corners to display pixels
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()

        def _to_disp(wu, wv):
            coord.SetValue(float(wu), float(wv), 0.5)
            d = coord.GetComputedDisplayValue(ren)
            return float(d[0]), float(d[1])

        u0, u1 = rect['u0'], rect['u1']
        v0, v1 = rect['v0'], rect['v1']
        # Four corners in display
        c_ll = _to_disp(u0, v0)
        c_hl = _to_disp(u1, v0)
        c_lh = _to_disp(u0, v1)
        c_hh = _to_disp(u1, v1)

        size = widget.GetRenderWindow().GetSize()
        mx = float(pos.x())
        my = float(size[1] - pos.y())  # VTK display Y

        corner_tol = 12.0
        edge_tol = 9.0

        # Corners first
        corners = [
            ('lo', 'lo', c_ll),
            ('hi', 'lo', c_hl),
            ('lo', 'hi', c_lh),
            ('hi', 'hi', c_hh),
        ]
        for us, vs, (cx, cy) in corners:
            if math.hypot(mx - cx, my - cy) <= corner_tol:
                return {'kind': 'corner', 'u_side': us, 'v_side': vs, 'orientation': orientation}

        def _dist_seg(px, py, ax, ay, bx, by):
            abx, aby = bx - ax, by - ay
            apx, apy = px - ax, py - ay
            ab2 = abx * abx + aby * aby
            if ab2 < 1e-12:
                return math.hypot(apx, apy)
            t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
            qx, qy = ax + t * abx, ay + t * aby
            return math.hypot(px - qx, py - qy)

        edges = [
            ('u', 'lo', c_ll, c_lh),  # left  u=u0
            ('u', 'hi', c_hl, c_hh),  # right u=u1
            ('v', 'lo', c_ll, c_hl),  # bottom v=v0
            ('v', 'hi', c_lh, c_hh),  # top    v=v1
        ]
        best = None
        best_d = edge_tol
        for axis_uv, side, (ax, ay), (bx, by) in edges:
            d = _dist_seg(mx, my, ax, ay, bx, by)
            if d <= best_d:
                best_d = d
                if axis_uv == 'u':
                    best = {'kind': 'edge', 'u_side': side, 'v_side': None, 'orientation': orientation}
                else:
                    best = {'kind': 'edge', 'u_side': None, 'v_side': side, 'orientation': orientation}
        if best is not None:
            return best

        # Center handle only (not full interior) so W/L + crosshair stay usable
        mid_x = 0.25 * (c_ll[0] + c_hl[0] + c_lh[0] + c_hh[0])
        mid_y = 0.25 * (c_ll[1] + c_hl[1] + c_lh[1] + c_hh[1])
        if math.hypot(mx - mid_x, my - mid_y) <= 14.0:
            return {'kind': 'move', 'u_side': None, 'v_side': None, 'orientation': orientation}
        return None

    def _df_clip_begin_drag(self, orientation, pos, hit):
        uv = self._df_clip_pick_display_uv(pos, orientation)
        if uv is None:
            return
        import copy
        self._df_clip_drag = {
            'orientation': orientation,
            'hit': dict(hit),
            'start_bounds': copy.deepcopy(self._df_clip_bounds),
            'start_uv': uv,
        }
        self._df_clip_hover = hit
        # Highlight handles
        self._df_clip_update_mpr_overlay(orientation)
        w = getattr(self, f'{orientation}_widget', None)
        if w is not None:
            w.setCursor(self._df_clip_cursor_for_hit(hit))
            try:
                w.GetRenderWindow().Render()
            except Exception:
                pass

    def _df_clip_handle_drag(self, pos, orientation):
        drag = getattr(self, '_df_clip_drag', None)
        if not drag or self.volume_data is None:
            return
        # Stick to the pane that started the drag
        ori = drag.get('orientation', orientation)
        uv = self._df_clip_pick_display_uv(pos, ori)
        if uv is None:
            return
        hit = drag['hit']
        kind = hit.get('kind')
        start_bounds = drag['start_bounds']
        su, sv = drag['start_uv']
        cu, cv = uv

        # Working copy from start (absolute for edges; delta for move)
        bounds = {
            'x': list(start_bounds['x']),
            'y': list(start_bounds['y']),
            'z': list(start_bounds['z']),
        }
        rect = self._df_clip_plane_rect(ori)
        if rect is None:
            return

        def _set_side(axis, side_is_lo, pct):
            lo, hi = bounds[axis]
            if side_is_lo:
                lo = max(0, min(int(pct), hi - 1))
            else:
                hi = min(1000, max(int(pct), lo + 1))
            bounds[axis] = [lo, hi]

        def _apply_u_side(side, pct):
            # side 'lo'/'hi' is display side; map to axis min/max
            if side is None:
                return
            is_min = rect['u_lo_is_min'] if side == 'lo' else (not rect['u_lo_is_min'])
            _set_side(rect['u_axis'], is_min, pct)

        def _apply_v_side(side, pct):
            if side is None:
                return
            is_min = rect['v_lo_is_min'] if side == 'lo' else (not rect['v_lo_is_min'])
            _set_side(rect['v_axis'], is_min, pct)

        if kind == 'move':
            # Delta in display → delta in % for both plane axes
            pct_start = self._df_clip_display_to_pct(ori, u=su, v=sv)
            pct_now = self._df_clip_display_to_pct(ori, u=cu, v=cv)
            for axis in (rect['u_axis'], rect['v_axis']):
                if axis not in pct_start or axis not in pct_now:
                    continue
                d = int(pct_now[axis] - pct_start[axis])
                lo0, hi0 = start_bounds[axis]
                lo = lo0 + d
                hi = hi0 + d
                # Clamp keeping width
                width = hi0 - lo0
                if lo < 0:
                    lo, hi = 0, width
                if hi > 1000:
                    hi, lo = 1000, 1000 - width
                bounds[axis] = [max(0, lo), min(1000, hi)]
        else:
            # Edge / corner: set absolute positions from current pick
            if hit.get('u_side') is not None:
                pcts = self._df_clip_display_to_pct(ori, u=cu, v=None)
                if rect['u_axis'] in pcts:
                    _apply_u_side(hit['u_side'], pcts[rect['u_axis']])
            if hit.get('v_side') is not None:
                pcts = self._df_clip_display_to_pct(ori, u=None, v=cv)
                if rect['v_axis'] in pcts:
                    _apply_v_side(hit['v_side'], pcts[rect['v_axis']])

        self._df_clip_bounds = bounds
        self._df_clip_sync_bound_sliders()
        # Lightweight live update (no full reslice)
        self._sync_all_clip_planes()
        self._df_clip_update_3d_visuals()
        for o in ('axial', 'coronal', 'sagittal'):
            self._df_clip_update_mpr_overlay(o)
            w = getattr(self, f'{o}_widget', None)
            if w is not None:
                try:
                    w.GetRenderWindow().Render()
                except Exception:
                    pass
        if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
            try:
                self.view_3d_widget.GetRenderWindow().Render()
            except Exception:
                pass

    def _df_clip_end_drag(self):
        self._df_clip_drag = None
        # Final full sync
        self._df_clip_refresh_all()

    def _df_clip_sync_bound_sliders(self):
        """Push current bounds into sidebar sliders without re-entrancy."""
        for ax, ctrls in getattr(self, '_df_clip_bound_ctrls', {}).items():
            lo, hi = self._df_clip_bounds[ax]
            ctrls['lo'].blockSignals(True)
            ctrls['hi'].blockSignals(True)
            ctrls['lo'].setValue(int(lo))
            ctrls['hi'].setValue(int(hi))
            ctrls['pct'].setText(f"{lo / 10.0:.0f}–{hi / 10.0:.0f}%")
            ctrls['lo'].blockSignals(False)
            ctrls['hi'].blockSignals(False)

    def _df_clip_update_mpr_overlay(self, orientation):
        """Draw clip-box rectangle, grid, and interactive handles on one MPR pane.

        Coordinates match render_slice display space (voxel units, not world):
          axial    → (x, y)
          coronal  → (x, flipud z)
          sagittal → (z, y) after transpose
        """
        self._df_clip_clear_mpr_actors(orientation)
        ren = getattr(self, f'{orientation}_renderer', None)
        if ren is None or self.volume_data is None:
            return
        if not self._df_clip_should_show_visuals():
            return

        rect = self._df_clip_plane_rect(orientation)
        if rect is None:
            return

        u0, u1 = rect['u0'], rect['u1']
        v0, v1 = rect['v0'], rect['v1']
        h_color = rect['h_color']
        v_color = rect['v_color']
        u_spacing = rect['u_spacing']
        v_spacing = rect['v_spacing']

        # Always show at least a thin box when clip is enabled (for drag targets)
        show_borders = self._df_clip_show_borders or self._df_clip_enabled
        show_grid = self._df_clip_show_grid or self._df_clip_display_grid_on_object
        if not (show_borders or show_grid):
            return

        def _add_line(x1, y1, x2, y2, color, width=1.5):
            src = vtk.vtkLineSource()
            src.SetPoint1(float(x1), float(y1), 0.5)
            src.SetPoint2(float(x2), float(y2), 0.5)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(src.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(width)
            actor.GetProperty().SetLighting(False)
            ren.AddActor(actor)
            self._df_clip_mpr_actors[orientation].append(actor)

        def _add_handle(cx, cy, color, size=3.5, highlight=False):
            # Small square handle via 4 lines (robust, no extra VTK filters)
            s = size * (1.4 if highlight else 1.0)
            col = tuple(min(1.0, c * 1.25) for c in color) if highlight else color
            w = 2.5 if highlight else 1.8
            _add_line(cx - s, cy - s, cx + s, cy - s, col, w)
            _add_line(cx + s, cy - s, cx + s, cy + s, col, w)
            _add_line(cx + s, cy + s, cx - s, cy + s, col, w)
            _add_line(cx - s, cy + s, cx - s, cy - s, col, w)

        hover = getattr(self, '_df_clip_hover', None)
        hover_on = (
            hover is not None
            and hover.get('orientation') == orientation
        ) or (
            getattr(self, '_df_clip_drag', None) is not None
            and self._df_clip_drag.get('orientation') == orientation
        )
        active = self._df_clip_drag['hit'] if getattr(self, '_df_clip_drag', None) else hover

        border_col = (0.9, 0.95, 1.0)
        if show_borders:
            # rectangle
            _add_line(u0, v0, u1, v0, border_col, 2.0)
            _add_line(u1, v0, u1, v1, border_col, 2.0)
            _add_line(u1, v1, u0, v1, border_col, 2.0)
            _add_line(u0, v1, u0, v0, border_col, 2.0)
            # colored edge accents (Dragonfly axis tint); thicken active edge
            def _edge_w(u_side=None, v_side=None):
                if not hover_on or active is None:
                    return 1.5
                if active.get('kind') == 'move':
                    return 2.2
                if u_side and active.get('u_side') == u_side and active.get('v_side') is None:
                    return 3.0
                if v_side and active.get('v_side') == v_side and active.get('u_side') is None:
                    return 3.0
                if active.get('kind') == 'corner':
                    if active.get('u_side') == u_side or active.get('v_side') == v_side:
                        return 2.5
                return 1.5

            _add_line(u0, v0, u1, v0, h_color, _edge_w(v_side='lo'))
            _add_line(u0, v1, u1, v1, h_color, _edge_w(v_side='hi'))
            _add_line(u0, v0, u0, v1, v_color, _edge_w(u_side='lo'))
            _add_line(u1, v0, u1, v1, v_color, _edge_w(u_side='hi'))

            # Corner + center handles (when clip enabled — Dragonfly grab targets)
            if self._df_clip_enabled:
                corners = [
                    (u0, v0, 'lo', 'lo'),
                    (u1, v0, 'hi', 'lo'),
                    (u0, v1, 'lo', 'hi'),
                    (u1, v1, 'hi', 'hi'),
                ]
                for cx, cy, us, vs in corners:
                    hl = (
                        hover_on and active is not None
                        and active.get('kind') == 'corner'
                        and active.get('u_side') == us
                        and active.get('v_side') == vs
                    )
                    _add_handle(cx, cy, border_col, size=3.2, highlight=hl)
                # Center move handle
                muc, mvc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
                hl_move = hover_on and active is not None and active.get('kind') == 'move'
                _add_handle(muc, mvc, (0.4, 0.85, 1.0), size=4.0, highlight=hl_move)

        if show_grid:
            gs = max(float(self._df_clip_grid_size), 1e-3)
            max_lines = 24
            du = max(gs / max(u_spacing, 1e-12), (u1 - u0) / max_lines if u1 > u0 else gs)
            dv = max(gs / max(v_spacing, 1e-12), (v1 - v0) / max_lines if v1 > v0 else gs)
            grid_col = (0.5, 0.7, 0.95)
            u = u0 + du
            while u < u1 - 1e-9:
                _add_line(u, v0, u, v1, grid_col, 1.0)
                u += du
            v = v0 + dv
            while v < v1 - 1e-9:
                _add_line(u0, v, u1, v, grid_col, 1.0)
                v += dv

    def render_3d(self):
        if self.volume_data is None:
            return

        # Master switch (Online default OFF for large volumes)
        if not self.is_3d_volume_render_enabled():
            self._teardown_3d_volume_actors()
            self._show_3d_disabled_placeholder()
            return

        self._hide_3d_disabled_placeholder()

        # NEW: GPU Driver/Vendor Information Check
        # This will print the GPU info to terminal/log to help verify RTX or Intel usage
        win = self.view_3d_widget.GetRenderWindow()
        if hasattr(win, 'ReportCapabilities'):
            caps = win.ReportCapabilities()
            if "NVIDIA" in caps.upper() or "RTX" in caps.upper():
                print(">>> 3D Viewer: High-Performance GPU Detected (NVIDIA/RTX)")
            elif "INTEL" in caps.upper():
                print(">>> 3D Viewer WARNING: Only Integrated Graphics Detected (INTEL)")
                print("Please check Windows Graphics Settings to force NVIDIA GPU.")
            else:
                print(f">>> 3D Viewer Vendor Info: {caps[:100]}...")

        # Optimization: Only create actor if it doesn't exist or data changed
        # For simplicty in this merge, let's recreate if needed but update properties if exists
        # NOTE: To support dynamic updates (quality/clipping) without flickering, we should try to update existing actor if possible.
        # But `render_3d` is often called when data *changes*. 
        # The sliders call specific update methods. 
        # So `render_3d` should do the INITIAL setup or FULL reset.
        
        # If actor exists, remove it to ensure clean state for full render
        if self.volume_actor:
             self.view_3d_renderer.RemoveVolume(self.volume_actor)
        if getattr(self, 'c1_actor_3d', None):
             self.view_3d_renderer.RemoveVolume(self.c1_actor_3d)
             self.c1_actor_3d = None
        if getattr(self, 'c2_actor_3d', None):
             self.view_3d_renderer.RemoveVolume(self.c2_actor_3d)
             self.c2_actor_3d = None

        z, y, x = self.volume_data.shape

        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(x, y, z)
        # Use custom spacing if set, otherwise fallback to global spacing
        spacing = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        vtk_data.SetSpacing(spacing[0], spacing[1], spacing[2])

        data_fortran = np.transpose(self.volume_data, (2, 1, 0))
        flat_data = np.ascontiguousarray(data_fortran.flatten('F'))

        if self.volume_data.dtype == np.uint16:
            vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
        elif self.volume_data.dtype == np.uint8:
            vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        else:
            flat_data = flat_data.astype(np.float32)
            vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_FLOAT)

        vtk_data.GetPointData().SetScalars(vtk_array)

        # Apply optimal settings for GPU Volume Mapper
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_data)
        
        # Setup histogram widgets with volume data
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_histogram(self.volume_data)
        # Sync LUT ramp colors on TF widget
        if hasattr(self, 'tf_widget') and hasattr(self, 'color_grad_widget'):
            self.tf_widget.set_lut_stops(self.color_grad_widget.color_stops)
        # MPR W/L histogram (right sidebar)
        if hasattr(self, 'mpr_hist_widget'):
            dmin = getattr(self, '_volume_data_min', None)
            dmax = getattr(self, '_volume_data_max', None)
            self.mpr_hist_widget.set_histogram(
                self.volume_data, data_min=dmin, data_max=dmax
            )
            if dmin is not None and dmax is not None:
                self.mpr_hist_widget.set_range(int(dmin), int(dmax))
        
        # --- Quality rendering parameters from selected preset ---
        quality_name = getattr(self, '_current_quality', 'High')
        preset = self._quality_presets.get(quality_name) if hasattr(self, '_quality_presets') else None
        if preset is None:
            preset = {"sample_dist": 0.5, "image_sample": 1.0, "auto_adj": True, "jitter": True, "max_mem_fraction": 0.85}
        
        sample_distance = preset["sample_dist"]
        image_sample_dist = preset["image_sample"]
        use_auto_adjust = preset["auto_adj"]
        use_jitter = preset["jitter"]
        max_mem_frac = preset.get("max_mem_fraction", 0.85)
        
        # 1. AutoAdjustSampleDistances: let VTK auto-tune during interaction
        volume_mapper.SetAutoAdjustSampleDistances(1 if use_auto_adjust else 0)
        
        # 2. SampleDistance: distance between samples along the ray
        #    Smaller = higher quality, less banding/stripe artifacts
        volume_mapper.SetSampleDistance(sample_distance)
        
        # 3. ImageSampleDistance: controls pixel-level sub-sampling
        #    1.0 = full resolution, 0.5 = super-sampled (2x), 2.0 = half resolution
        if hasattr(volume_mapper, 'SetImageSampleDistance'):
            volume_mapper.SetImageSampleDistance(image_sample_dist)
        
        # 4. Stochastic jittering to eliminate periodic banding artifacts
        if hasattr(volume_mapper, 'SetUseJittering'):
            volume_mapper.SetUseJittering(1 if use_jitter else 0)
        
        # 5. GPU Memory Optimization — RTX 5090 (32GB VRAM)
        #    Allocate up to 90% of VRAM for volume texture streaming
        if hasattr(volume_mapper, 'SetMaxMemoryFraction'):
            volume_mapper.SetMaxMemoryFraction(max_mem_frac)
        
        # 6. GPU texture streaming for large volumes (> 2GB)
        #    Allows VTK to stream volume data as tiled 3D textures to GPU
        if hasattr(volume_mapper, 'SetMaxMemoryInBytes'):
            # RTX 5090: allow up to 28GB for volume textures
            volume_mapper.SetMaxMemoryInBytes(int(28 * 1024 * 1024 * 1024))
        
        # 7. Blend mode
        if hasattr(self, 'render_mode_combo'):
            mode_text = self.render_mode_combo.currentText()
            mode_map = {
                "Composite":  0,
                "MIP":        1,
                "MinIP":      2,
                "Average":    3,
                "Additive":   4,
            }
            volume_mapper.SetBlendMode(mode_map.get(mode_text, 0))
        elif hasattr(volume_mapper, 'SetBlendModeToComposite'):
            volume_mapper.SetBlendModeToComposite()
        
        # 8. Volumetric shadows — OFF by default (Dragonfly soft look; Plastic look can enable)
        if hasattr(volume_mapper, 'SetShadows'):
            use_sh = bool(getattr(self, "_vol_shadows_enabled", False))
            volume_mapper.SetShadows(1 if use_sh else 0)

        # 9. Store mapper reference for LOD interaction updates
        self._volume_mapper = volume_mapper
                
        volume_property = vtk.vtkVolumeProperty()
        
        # Phong Shading
        if hasattr(self, 'shade_check') and self.shade_check.isChecked():
            volume_property.ShadeOn()
        else:
            volume_property.ShadeOff()
            
        volume_property.SetInterpolationTypeToLinear()
        # Consistent opacity vs spacing (Dragonfly-like solidity across aniso voxels)
        try:
            sp = self.custom_spacing if getattr(self, "custom_spacing", None) is not None else self.spacing
            soud = float(min(sp[0], sp[1], sp[2]))
            if soud > 0:
                volume_property.SetScalarOpacityUnitDistance(soud)
        except Exception:
            pass
        
        # Apply Transfer Function (Color/Opacity)
        self.apply_transfer_function(volume_property)
        
        # Apply Gradient Opacity (soft surface edges — Dragonfly default ~20)
        if hasattr(self, 'gradient_opacity_slider') and self.gradient_opacity_slider.value() > 0:
            scale = self.gradient_opacity_slider.value() / 100.0
            volume_property.DisableGradientOpacityOff()
            grad = vtk.vtkPiecewiseFunction()
            # Gentler curve: avoid harsh edge glow / noisy “raised” rims
            grad.AddPoint(0, 0.0)
            grad.AddPoint(40 * (1.0 - scale * 0.5), 0.05 * scale)
            grad.AddPoint(120 * (1.0 - scale * 0.4), 0.35 * scale)
            grad.AddPoint(280, 0.75 * scale)
            grad.AddPoint(1000, 1.0)
            volume_property.SetGradientOpacity(grad)
        else:
            try:
                volume_property.DisableGradientOpacityOn()
            except Exception:
                pass

        # Lighting — UI sliders (defaults = Dragonfly soft metallic)
        amb = 0.40
        dif = 0.58
        spe = 0.18
        spow = 16.0
        if hasattr(self, 'view_3d_ambient_slider'):
            amb = self.view_3d_ambient_slider.value() / 100.0
            dif = self.view_3d_diffuse_slider.value() / 100.0
            spe = self.view_3d_specular_slider.value() / 100.0
            spow = float(self.view_3d_spec_power_slider.value())
        volume_property.SetAmbient(amb)
        volume_property.SetDiffuse(dif)
        volume_property.SetSpecular(spe)
        volume_property.SetSpecularPower(spow)

        self.volume_actor = vtk.vtkVolume()
        self.volume_actor.SetMapper(volume_mapper)
        self.volume_actor.SetProperty(volume_property)

        self.view_3d_renderer.AddVolume(self.volume_actor)
        
        # --- Add Segmentation Overlays as separate Volume Actors ---
        def create_mask_volume(mask_data, color_val, base_opacity):
            vz, vy, vx = mask_data.shape
            vtk_mask = vtk.vtkImageData()
            vtk_mask.SetDimensions(vx, vy, vz)
            spacing = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
            vtk_mask.SetSpacing(spacing[0], spacing[1], spacing[2])
            
            mask_fortran = np.transpose(mask_data, (2, 1, 0))
            flat_mask = np.ascontiguousarray(mask_fortran.flatten('F'))
            vtk_m_array = numpy_support.numpy_to_vtk(flat_mask, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            vtk_mask.GetPointData().SetScalars(vtk_m_array)
            
            m_mapper = vtk.vtkGPUVolumeRayCastMapper()
            m_mapper.SetInputData(vtk_mask)
            m_mapper.SetBlendModeToComposite()
                    
            m_prop = vtk.vtkVolumeProperty()
            color_tf = vtk.vtkColorTransferFunction()
            color_tf.AddRGBPoint(0, 0, 0, 0)
            color_tf.AddRGBPoint(1, color_val[0], color_val[1], color_val[2])
            color_tf.AddRGBPoint(255, color_val[0], color_val[1], color_val[2])
            m_prop.SetColor(color_tf)
            
            opacity_tf = vtk.vtkPiecewiseFunction()
            opacity_tf.AddPoint(0, 0.0)
            opacity_tf.AddPoint(1, base_opacity)
            opacity_tf.AddPoint(255, base_opacity)
            m_prop.SetScalarOpacity(opacity_tf)
            
            m_prop.ShadeOn()
            m_prop.SetAmbient(0.3)
            m_prop.SetDiffuse(0.7)
            
            m_actor = vtk.vtkVolume()
            m_actor.SetMapper(m_mapper)
            m_actor.SetProperty(m_prop)
            return m_actor
            
        if getattr(self, 'class1_data', None) is not None:
             c1_color = self.seg_colors.get(128, [0.0, 1.0, 0.0])
             self.c1_actor_3d = create_mask_volume(self.class1_data, c1_color, 0.4)
             self.view_3d_renderer.AddVolume(self.c1_actor_3d)
             
        if getattr(self, 'class2_data', None) is not None:
             c2_color = self.seg_colors.get(255, [1.0, 0.0, 0.0])
             self.c2_actor_3d = create_mask_volume(self.class2_data, c2_color, 0.8)
             self.view_3d_renderer.AddVolume(self.c2_actor_3d)
        
        # Sync clip planes to ALL actors (volume + c1 + c2) in one pass
        # (fullscreen Advanced clips + Dragonfly 2×2 clip box)
        self._sync_all_clip_planes()
        # Rebuild Dragonfly clip-box graphics after volume recreation
        self._df_clip_update_3d_visuals()
        
        # Only reset camera if it's a fresh load, otherwise navigation might be jarring
        if not getattr(self, '_camera_initialized', False):
            self.view_3d_renderer.ResetCamera()
            cam = self.view_3d_renderer.GetActiveCamera()
            pmode = str(getattr(self, 'projection_mode', 'perspective')).split('#')[0].strip().lower()
            if pmode == 'perspective':
                cam.ParallelProjectionOff()
                cam.SetViewAngle(40.0)  # Dragonfly examine FOV (not game-wide)
            else:
                cam.ParallelProjectionOn()
                
            self.set_camera_preset("Top")
            self._camera_initialized = True
        else:
            # Just ensure projection mode is correct if switched
            cam = self.view_3d_renderer.GetActiveCamera()
            pmode = str(getattr(self, 'projection_mode', 'perspective')).split('#')[0].strip().lower()
            if pmode == 'perspective':
                cam.ParallelProjectionOff()
            else:
                cam.ParallelProjectionOn()

        # Always refresh orbit pivot (spacing / volume size may have changed)
        self._sync_3d_orbit_pivot()
        # Re-apply MES pick surfaces after volume recreation (cache invalid)
        self._mes_3d_hl_cache_key = None
        if getattr(self, "selected_highlight_objects", None):
            self._update_mes_3d_highlight()
        else:
            self._clear_mes_3d_highlight(render=True)

    def apply_transfer_function(self, volume_property):
        """Apply opacity + colour TF for the current W/L or TF mode (Dragonfly-style)."""
        v_min = float(self.view_3d_min_spin.value())
        v_max = float(self.view_3d_max_spin.value())

        if v_max <= v_min:
            v_max = v_min + 1.0

        # Global opacity / solidity
        opacity_scale = 1.0
        if hasattr(self, 'wl_opacity_slider'):
            opacity_scale *= self.wl_opacity_slider.value() / 100.0
        elif hasattr(self, 'view_3d_vol_opacity_slider'):
            opacity_scale *= self.view_3d_vol_opacity_slider.value() / 100.0

        # Effective opacity curve (shape+gamma for W/L, free spline for TF)
        tw = getattr(self, 'tf_widget', None)
        if tw is not None and tw.is_wl_mode():
            shape = tw.opacity_shape()
            gamma = tw.opacity_gamma()
            # Sample with scale once — do not double-apply
            eff = sample_opacity_curve(shape, gamma=gamma, n=33, opacity_scale=opacity_scale)
        elif tw is not None:
            eff = [(float(x), float(y) * opacity_scale) for x, y in tw.get_points()]
        else:
            shape = getattr(self, '_opacity_shape', 'linear')
            gamma = self.wl_gamma_spin.value() if hasattr(self, 'wl_gamma_spin') else 1.0
            eff = sample_opacity_curve(shape, gamma=gamma, n=33, opacity_scale=opacity_scale)

        cg_stops = (
            getattr(self.color_grad_widget, 'color_stops', None)
            if hasattr(self, 'color_grad_widget') else None
        )
        invert = bool(getattr(self, '_color_invert', False))
        if tw is not None:
            invert = bool(tw.color_invert())

        if eff and len(eff) > 0 and cg_stops and len(cg_stops) > 0:
            opacity = vtk.vtkPiecewiseFunction()
            opacity.AddPoint(v_min - 1.0, 0.0)
            for nx, ny in eff:
                intensity = v_min + float(nx) * (v_max - v_min)
                opacity.AddPoint(intensity, float(ny))
            opacity.AddPoint(v_max + 1.0, float(eff[-1][1]))

            color = vtk.vtkColorTransferFunction()
            stops = list(cg_stops)
            if invert:
                stops = [(1.0 - pos, c) for pos, c in stops]
                stops.sort(key=lambda s: s[0])
            for pos, qcolor in stops:
                intensity = v_min + float(pos) * (v_max - v_min)
                color.AddRGBPoint(
                    intensity, qcolor.redF(), qcolor.greenF(), qcolor.blueF()
                )

            volume_property.SetScalarOpacity(opacity)
            volume_property.SetColor(color)
            return

        # Fallback silver/grayscale (Dragonfly soft metal — avoid blue-white specular tip)
        mid = v_min + 0.55 * (v_max - v_min)
        r, g, b = self.volume_color
        dark = (r * 0.22, g * 0.22, b * 0.22)
        mid_c = (r * 0.72, g * 0.72, b * 0.70)
        # Soft near-white plateau (no cool blue boost that reads as plastic)
        hi = (min(1.0, r * 1.08), min(1.0, g * 1.08), min(1.0, b * 1.06))

        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(v_min, 0.0, 0.0, 0.0)
        color_func.AddRGBPoint(v_min + 0.22 * (v_max - v_min), *dark)
        color_func.AddRGBPoint(mid, *mid_c)
        color_func.AddRGBPoint(v_min + 0.88 * (v_max - v_min), r, g, b)
        color_func.AddRGBPoint(v_max, *hi)

        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(v_min, 0.0)
        opacity_func.AddPoint(v_min + 0.28 * (v_max - v_min), 0.0)
        opacity_func.AddPoint(mid, 0.22 * opacity_scale)
        opacity_func.AddPoint(v_min + 0.85 * (v_max - v_min), 0.72 * opacity_scale)
        opacity_func.AddPoint(v_max, 0.92 * opacity_scale)

        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)




    # --- LOD (Level-of-Detail) Interaction System for Large Volumes ---
    
    def _start_lod_interaction(self):
        """Called when user starts rotating/zooming the 3D view.
        
        Temporarily reduces rendering quality (larger sample distance,
        lower pixel resolution) so that interaction remains smooth even
        with 10GB+ volumes on RTX 5090.
        """
        self._is_interacting_3d = True
        # Cancel any pending refinement
        if hasattr(self, '_lod_refine_timer'):
            self._lod_refine_timer.stop()
        
        mapper = getattr(self, '_volume_mapper', None)
        if mapper is None:
            return
        
        quality_name = getattr(self, '_current_quality', 'High')
        preset = self._quality_presets.get(quality_name, {})
        
        # Switch to interaction-quality settings
        interact_sample = preset.get('interact_sample_dist', 2.0)
        interact_image = preset.get('interact_image_sample', 2.0)
        
        mapper.SetSampleDistance(interact_sample)
        if hasattr(mapper, 'SetImageSampleDistance'):
            mapper.SetImageSampleDistance(interact_image)
        # Always let VTK auto-adjust during interaction for safety
        mapper.SetAutoAdjustSampleDistances(1)
    
    def _end_lod_interaction(self):
        """Called when user stops rotating/zooming.
        
        Schedules a delayed refinement render at full quality after
        a short pause (300ms) to avoid unnecessary re-renders if the
        user immediately starts interacting again.
        """
        self._is_interacting_3d = False
        if hasattr(self, '_lod_refine_timer'):
            self._lod_refine_timer.start()  # Uses the interval set in __init__ (300ms)
    
    def _refine_after_interaction(self):
        """Restore full-quality rendering after interaction stops.
        
        This is called by the LOD refinement timer. It restores the
        original (high-quality) sample distances from the selected
        quality preset and triggers a crisp re-render.
        """
        if self._is_interacting_3d:
            return  # User started interacting again, skip
        
        mapper = getattr(self, '_volume_mapper', None)
        if mapper is None:
            return
        
        quality_name = getattr(self, '_current_quality', 'High')
        preset = self._quality_presets.get(quality_name, {})
        
        # Restore full-quality settings
        mapper.SetSampleDistance(preset.get('sample_dist', 0.5))
        if hasattr(mapper, 'SetImageSampleDistance'):
            mapper.SetImageSampleDistance(preset.get('image_sample', 1.0))
        mapper.SetAutoAdjustSampleDistances(1 if preset.get('auto_adj', True) else 0)
        
        # Re-sync clip planes (defensive — ensures they persist through interaction)
        self._sync_all_clip_planes()
        
        # Trigger crisp re-render
        if hasattr(self, 'view_3d_widget') and self.view_3d_widget:
            self.view_3d_widget.GetRenderWindow().Render()
    
    def _on_quality_preset_changed(self, quality_name):
        """Handle quality preset combo box change.
        
        Updates the mapper parameters in-place without re-creating the
        volume actor, so the change is instant.
        """
        self._current_quality = quality_name
        
        mapper = getattr(self, '_volume_mapper', None)
        if mapper is None:
            return
        
        preset = self._quality_presets.get(quality_name, {})
        
        # Apply static (non-interaction) quality parameters
        mapper.SetSampleDistance(preset.get('sample_dist', 0.5))
        if hasattr(mapper, 'SetImageSampleDistance'):
            mapper.SetImageSampleDistance(preset.get('image_sample', 1.0))
        mapper.SetAutoAdjustSampleDistances(1 if preset.get('auto_adj', True) else 0)
        if hasattr(mapper, 'SetUseJittering'):
            mapper.SetUseJittering(1 if preset.get('jitter', True) else 0)
        if hasattr(mapper, 'SetMaxMemoryFraction'):
            mapper.SetMaxMemoryFraction(preset.get('max_mem_fraction', 0.85))
        
        # Re-render with new settings
        if hasattr(self, 'view_3d_widget') and self.view_3d_widget:
            self.view_3d_widget.GetRenderWindow().Render()
        
        print(f">>> 3D Quality: {quality_name} | SampleDist={preset.get('sample_dist'):.2f} | "
              f"ImageSample={preset.get('image_sample'):.1f} | "
              f"InteractDist={preset.get('interact_sample_dist'):.1f} | "
              f"MaxMem={preset.get('max_mem_fraction', 0.85)*100:.0f}%")

    # --- NEW 3D CONTROL HANDLERS ---
    
    def choose_volume_color(self):
        """Choose color for volume rendering"""
        parent = getattr(self, 'fullscreen_window', None) or self
        init_c = QColor(int(self.volume_color[0]*255), int(self.volume_color[1]*255), int(self.volume_color[2]*255))
        color = QColorDialog.getColor(init_c, parent, "Volume Color")
        if color.isValid():
            self.volume_color = [color.redF(), color.greenF(), color.blueF()]
            
            if hasattr(self, 'color_grad_widget'):
                dark = QColor(int(color.red()*0.35), int(color.green()*0.35), int(color.blue()*0.35))
                mid_c = QColor(int(color.red()*0.85), int(color.green()*0.85), int(color.blue()*0.85))
                # Add a high specular highlight (bright silver/gold light) at the very peak
                highlight_c = QColor(
                    min(255, int(color.red() * 1.25)),
                    min(255, int(color.green() * 1.25 + 12)),
                    min(255, int(color.blue() * 1.25 + 63))
                )
                self.color_grad_widget.color_stops = [
                    (0.0, QColor(0, 0, 0)),
                    (0.3, dark),
                    (0.6, mid_c),
                    (0.95, color),
                    (1.0, highlight_c)
                ]
                self.color_grad_widget.update()
                
            if self.volume_actor:
                self.update_transfer_function()

    def update_3d_lighting_properties(self):
        """Update lighting properties for 3D volume actor"""
        if not self.volume_actor:
            return
        prop = self.volume_actor.GetProperty()
        # Respect Enable Shading checkbox (do not force ShadeOn)
        if hasattr(self, "shade_check") and self.shade_check.isChecked():
            prop.ShadeOn()
        else:
            prop.ShadeOff()
        prop.SetAmbient(self.view_3d_ambient_slider.value() / 100.0)
        prop.SetDiffuse(self.view_3d_diffuse_slider.value() / 100.0)
        prop.SetSpecular(self.view_3d_specular_slider.value() / 100.0)
        prop.SetSpecularPower(float(self.view_3d_spec_power_slider.value()))
        # Opacity is handled via transfer function opacity scale
        self.apply_transfer_function(prop)
        self.view_3d_widget.GetRenderWindow().Render()

    def choose_3d_bg_color(self):
        """Choose background color for 3D volume viewport with a Dragonfly-style gradient"""
        parent = getattr(self, 'fullscreen_window', None) or self
        bg2 = self.view_3d_renderer.GetBackground2()
        init_c = QColor(int(bg2[0]*255), int(bg2[1]*255), int(bg2[2]*255))
        color = QColorDialog.getColor(init_c, parent, "Choose Viewport Top/Center Color")
        if color.isValid():
            r, g, b = color.redF(), color.greenF(), color.blueF()
            
            # Disable texture background if it was on
            self.view_3d_renderer.SetTexturedBackground(False)
            
            self.view_3d_renderer.GradientBackgroundOn()
            try:
                self.view_3d_renderer.SetGradientMode(self.view_3d_renderer.VTK_GRADIENT_RADIAL_FARTHEST_CORNER)
                self.view_3d_renderer.SetBackground(r, g, b)
                self.view_3d_renderer.SetBackground2(r * 0.15, g * 0.15, b * 0.15)
            except AttributeError:
                self.view_3d_renderer.SetBackground2(r, g, b)
                self.view_3d_renderer.SetBackground(r * 0.15, g * 0.15, b * 0.15)
                
            self.view_3d_widget.GetRenderWindow().Render()


    def choose_crosshair_color(self):
        """Pick crosshair color"""
        init_c = QColor(int(self.crosshair_color[0]*255), int(self.crosshair_color[1]*255), int(self.crosshair_color[2]*255))
        color = QColorDialog.getColor(init_c, self, "Crosshair Color")
        if color.isValid():
            self.crosshair_color = [color.redF(), color.greenF(), color.blueF()]
            self.crosshair_color_btn.setStyleSheet(f"background-color: {SemiconductorTheme.BG_LIGHT}; color: {color.name()}; font-weight: bold; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; border-radius: 3px;")
            for ori in ['axial', 'coronal', 'sagittal']:
                self.render_slice(ori, preserve_camera=True)
    
    def choose_ruler_color(self):
        """Pick ruler color"""
        init_c = QColor(int(self.ruler_color[0]*255), int(self.ruler_color[1]*255), int(self.ruler_color[2]*255))
        color = QColorDialog.getColor(init_c, self, "Ruler Color")
        if color.isValid():
            self.ruler_color = [color.redF(), color.greenF(), color.blueF()]
            self.ruler_color_btn.setStyleSheet(f"background-color: {SemiconductorTheme.BG_LIGHT}; color: {color.name()}; font-weight: bold; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; border-radius: 3px;")
            for ori in ['axial', 'coronal', 'sagittal']:
                self.render_slice(ori, preserve_camera=True)

    def load_class_mask(self, class_num):
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select Class {class_num} Binary Mask", "", "TIFF Files (*.tif *.tiff)"
        )
        if file_path:
            try:
                data = io.imread(file_path)
                if data.ndim == 2:
                    directory = Path(file_path).parent
                    all_files = sorted(glob.glob(str(directory / "*.tif*")))
                    if len(all_files) > 1:
                        data = np.array([io.imread(f) for f in all_files])
                    else:
                        data = data[np.newaxis, :, :]
                
                # --- Handle shape mismatch (per-layer sub-volume) ---
                if self.volume_data is not None and data.shape != self.volume_data.shape:
                    vol_z, vol_y, vol_x = self.volume_data.shape
                    mask_z, mask_y, mask_x = data.shape

                    # XY must match
                    if mask_y != vol_y or mask_x != vol_x:
                        QMessageBox.critical(self, "Shape Error",
                            f"Mask XY dimensions ({mask_y}Ã—{mask_x}) do not match "
                            f"volume XY ({vol_y}Ã—{vol_x}).\n\nCannot load this mask.")
                        return

                    # Z is smaller â†’ need layer offset from config
                    if mask_z < vol_z:
                        reply = QMessageBox.question(self, "Layer Mask Detected",
                            f"Mask Z={mask_z} < Volume Z={vol_z}.\n\n"
                            "This appears to be a per-layer mask.\n"
                            "Load a config file to get the layer Z offset?\n\n"
                            "(The config must contain LAYER definitions with z_start values.)",
                            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                        
                        if reply == QMessageBox.Yes:
                            z_start = self._get_layer_offset_from_config(mask_z)
                            if z_start is None:
                                return  # User cancelled or error
                        else:
                            # Fallback: place at start
                            z_start = 0
                        
                        # Create full-size mask and place sub-volume at correct Z
                        full_mask = np.zeros(self.volume_data.shape, dtype=data.dtype)
                        z_end = min(z_start + mask_z, vol_z)
                        actual_len = z_end - z_start
                        full_mask[z_start:z_end] = data[:actual_len]
                        data = full_mask
                        
                        QMessageBox.information(self, "Mask Placed",
                            f"Layer mask placed at Z=[{z_start}:{z_end}] "
                            f"within full volume Z=[0:{vol_z}].")

                    elif mask_z > vol_z:
                        # Mask is larger â€” crop to volume size
                        data = data[:vol_z]
                
                if class_num == 1:
                    self.class1_data = data
                else:
                    self.class2_data = data
                
                self.merge_class_masks()
                
                if self.volume_data is not None:
                     for orientation in ['axial', 'coronal', 'sagittal']:
                        self.render_slice(orientation)
                        
                self.update_info_label()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load mask: {str(e)}")

    def load_mask_folder(self):
        """Load all per-layer masks from a sample output folder.
        
        Expects structure:
            <folder>/Layer_1/*_bump3D.tif   â†’ Class 1 (Bump)
            <folder>/Layer_1/*_voidsOnly.tif â†’ Class 2 (Void)
            ...
        Also requires a config file with LAYER_N definitions for Z offsets.
        """
        if self.volume_data is None:
            QMessageBox.warning(self, "No Volume", "Please load a volume first.")
            return
        
        # 1. Select mask folder
        folder = QFileDialog.getExistingDirectory(
            self, "Select Sample Output Folder (contains Layer_N subfolders)",
            "", QFileDialog.ShowDirsOnly)
        if not folder:
            return
        
        # 2. Discover layer subfolders
        import re
        layer_dirs = []
        for entry in os.listdir(folder):
            sub = os.path.join(folder, entry)
            if os.path.isdir(sub) and re.match(r'Layer_\d+', entry):
                layer_dirs.append((entry, sub))
        
        if not layer_dirs:
            QMessageBox.warning(self, "No Layers",
                f"No Layer_N subfolders found in:\n{folder}")
            return
        
        # Natural sort
        layer_dirs.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() 
                                        for c in re.split(r'(\d+)', x[0])])
        
        # 3. Select config file for Z offsets
        config_path, _ = QFileDialog.getOpenFileName(
            self, "Select Config File (for layer Z offsets)", 
            os.path.dirname(folder),
            "Config Files (*.txt *.cfg *.ini);;All Files (*.*)")
        if not config_path:
            return
        
        # 4. Parse layer definitions from config
        layer_config = {}  # "Layer 1" â†’ (z_start, z_end)
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    match = re.match(r'LAYER_\d+\s*=\s*(.+)', line)
                    if match:
                        parts = match.group(1).rsplit(',', 3)
                        if len(parts) >= 3:
                            try:
                                name = parts[0].strip()
                                z_start = int(parts[1].strip())
                                z_end = int(parts[2].strip())
                                # Also store with underscore variant: "Layer 1" â†’ "Layer_1"
                                layer_config[name] = (z_start, z_end)
                                layer_config[name.replace(' ', '_')] = (z_start, z_end)
                            except ValueError:
                                continue
        except Exception as e:
            QMessageBox.critical(self, "Config Error", f"Failed to parse config:\n{str(e)}")
            return
        
        if not layer_config:
            QMessageBox.warning(self, "No Layers in Config",
                "No LAYER_N definitions found in config file.")
            return
        
        # 5. Build full-volume masks
        vol_z, vol_y, vol_x = self.volume_data.shape
        full_bump = np.zeros((vol_z, vol_y, vol_x), dtype=np.uint8)
        full_void = np.zeros((vol_z, vol_y, vol_x), dtype=np.uint8)
        
        progress = QProgressDialog("Loading per-layer masks...", "Cancel", 0, len(layer_dirs), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        
        loaded_count = 0
        errors = []
        
        for idx, (dir_name, dir_path) in enumerate(layer_dirs):
            if progress.wasCanceled():
                break
            progress.setValue(idx)
            progress.setLabelText(f"Loading {dir_name}...")
            QApplication.processEvents()
            
            # Match dir name to config: "Layer_1" â†’ try "Layer_1" and "Layer 1"
            z_info = layer_config.get(dir_name) or layer_config.get(dir_name.replace('_', ' '))
            if z_info is None:
                errors.append(f"{dir_name}: no matching layer in config")
                continue
            
            z_start, z_end = z_info
            
            # Find bump3D and voidsOnly TIFs
            bump_file = None
            void_file = None
            for fname in os.listdir(dir_path):
                fl = fname.lower()
                if fl.endswith('.tif') or fl.endswith('.tiff'):
                    if 'bump3d' in fl:
                        bump_file = os.path.join(dir_path, fname)
                    elif 'voidsonly' in fl and 'flatten' not in fl:
                        void_file = os.path.join(dir_path, fname)
            
            try:
                if bump_file:
                    bump_data = io.imread(bump_file)
                    mask_z = bump_data.shape[0] if bump_data.ndim == 3 else 1
                    layer_len = min(mask_z, z_end - z_start, vol_z - z_start)
                    if bump_data.ndim == 3:
                        full_bump[z_start:z_start + layer_len] = np.maximum(
                            full_bump[z_start:z_start + layer_len],
                            bump_data[:layer_len])
                    
                if void_file:
                    void_data = io.imread(void_file)
                    mask_z = void_data.shape[0] if void_data.ndim == 3 else 1
                    layer_len = min(mask_z, z_end - z_start, vol_z - z_start)
                    if void_data.ndim == 3:
                        full_void[z_start:z_start + layer_len] = np.maximum(
                            full_void[z_start:z_start + layer_len],
                            void_data[:layer_len])
                
                loaded_count += 1
                
            except Exception as e:
                errors.append(f"{dir_name}: {str(e)}")
        
        progress.setValue(len(layer_dirs))
        progress.close()
        
        # 6. Assign to viewer
        self.class1_data = full_bump
        self.class2_data = full_void
        self.merge_class_masks()
        
        for orientation in ['axial', 'coronal', 'sagittal']:
            self.render_slice(orientation)
        self.update_info_label()
        
        # Report
        msg = f"Loaded {loaded_count}/{len(layer_dirs)} layers successfully."
        if errors:
            msg += f"\n\nWarnings:\n" + "\n".join(errors[:10])
        QMessageBox.information(self, "Mask Folder Loaded", msg)

    def _get_layer_offset_from_config(self, mask_z: int):
        """Parse a config file and let user pick which layer matches the mask Z size.
        
        Config format: LAYER_N = LayerName,z_start,z_end,selected
        Example:       LAYER_0 = Layer 1,666,831,true
        
        Returns the z_start offset for the selected layer, or None if cancelled.
        """
        import re
        
        config_path, _ = QFileDialog.getOpenFileName(
            self, "Select Config File with Layer Definitions", "",
            "Config Files (*.txt *.cfg *.ini);;All Files (*.*)")
        if not config_path:
            return None
        
        try:
            layers = []
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    # Match: LAYER_0 = Layer 1,666,831,true
                    match = re.match(r'LAYER_\d+\s*=\s*(.+)', line)
                    if match:
                        parts = match.group(1).rsplit(',', 3)  # split from right: name may contain commas
                        if len(parts) >= 3:
                            # parts: ['Layer 1', '666', '831', 'true'] or ['Layer 1', '666', '831']
                            selected_str = parts[3].strip().lower() if len(parts) >= 4 else 'true'
                            try:
                                name = parts[0].strip()
                                z_start = int(parts[1].strip())
                                z_end = int(parts[2].strip())
                                z_size = z_end - z_start
                                layers.append((name, z_start, z_end, z_size))
                            except ValueError:
                                continue
            
            if not layers:
                QMessageBox.warning(self, "No Layers",
                    "No LAYER definitions found in config file.\n"
                    "Expected format: LAYER_N = LayerName,z_start,z_end,selected")
                return None
            
            # Filter layers that match the mask Z size
            matching = [(name, z_start, z_end, z_size) 
                       for name, z_start, z_end, z_size in layers if z_size == mask_z]
            
            if len(matching) == 1:
                # Exact single match â€” use it directly
                return matching[0][1]
            
            # Multiple matches or no exact match â€” let user pick
            if not matching:
                matching = layers  # Show all layers if none match exactly
            
            items = [f"{name} (Z={z_start}â†’{z_end}, size={z_size})" 
                    for name, z_start, z_end, z_size in matching]
            
            chosen, ok = QInputDialog.getItem(
                self, "Select Layer",
                f"Mask Z size = {mask_z}. Which layer does this mask belong to?",
                items, 0, False)
            
            if ok and chosen:
                idx = items.index(chosen)
                return matching[idx][1]
            
            return None
            
        except Exception as e:
            QMessageBox.critical(self, "Config Error",
                f"Failed to parse config file:\n{str(e)}")
            return None

    def merge_class_masks(self):
        if self.class1_data is None and self.class2_data is None:
            self.segmentation_data = None
            return

        shape = self.class1_data.shape if self.class1_data is not None else self.class2_data.shape
        self.segmentation_data = np.zeros(shape, dtype=np.uint8)
        
        if self.class1_data is not None:
            self.segmentation_data[self.class1_data > 0] = 128
            
        if self.class2_data is not None:
            self.segmentation_data[self.class2_data > 0] = 255
            
    def clear_masks(self):
        self.class1_data = None
        self.class2_data = None
        self.segmentation_data = None
        self.labeled_class1_data = None
        self.object_stats = []
        if hasattr(self, 'selected_highlight_objects'):
            self.selected_highlight_objects = []
        if hasattr(self, '_refresh_stats_table'):
            self._refresh_stats_table()
        
        if self.volume_data is not None:
            for orientation in ['axial', 'coronal', 'sagittal']:
                self.render_slice(orientation)
        self.update_info_label()

    def clear_all_views(self):
        """Completely reset all 2D and 3D views to a blank state."""
        self.volume_data = None
        self.clear_masks()
        
        # Clear 3D actors
        if getattr(self, 'volume_actor', None):
            self.view_3d_renderer.RemoveVolume(self.volume_actor)
            self.volume_actor = None
        if getattr(self, 'c1_actor_3d', None):
            self.view_3d_renderer.RemoveVolume(self.c1_actor_3d)
            self.c1_actor_3d = None
        if getattr(self, 'c2_actor_3d', None):
            self.view_3d_renderer.RemoveVolume(self.c2_actor_3d)
            self.c2_actor_3d = None
            
        # Clear 2D actors
        for ori in ['axial', 'coronal', 'sagittal']:
            renderer = getattr(self, f'{ori}_renderer')
            renderer.RemoveAllViewProps()
            widget = getattr(self, f'{ori}_widget')
            widget.GetRenderWindow().Render()
            
        if hasattr(self, '_cached_slice_actors'):
            self._cached_slice_actors.clear()
            
        if hasattr(self, '_persistent_crosshair'):
            self._persistent_crosshair.clear()
            
        # Also clear object stats table
        if hasattr(self, 'object_stats_table'):
            self.object_stats_table.blockSignals(True)
            self.object_stats_table.setRowCount(0)
            self.object_stats_table.blockSignals(False)
            
        self.view_3d_widget.GetRenderWindow().Render()
        self.update_info_label()

    def align_hbm_volume(self):
        """Align the HBM volume data using PCA alignment (with LTIC principles)"""
        if self.volume_data is None:
            QMessageBox.warning(self, "No Data", "Please load a volume file first.")
            return
            
        # Show progress dialog
        self.align_progress = QProgressDialog("Aligning HBM volume...", "Cancel", 0, 100, self)
        self.align_progress.setWindowModality(Qt.WindowModal)
        self.align_progress.setMinimumDuration(0)
        self.align_progress.show()
        
        # Start worker thread
        self.align_thread = AlignVolumeThread(self.volume_data)
        
        def on_progress(value, msg):
            self.align_progress.setValue(value)
            self.align_progress.setLabelText(msg)
            
        def on_finished(aligned_vol, best_R, error):
            self.align_progress.close()
            if error:
                QMessageBox.critical(self, "Alignment Error", f"Failed to align volume: {error}")
                return
                
            # Create backup of unaligned data first time alignment is run
            if getattr(self, 'original_volume_data', None) is None:
                self.original_volume_data = self.volume_data.copy()
                self.original_class1_data = self.class1_data.copy() if self.class1_data is not None else None
                self.original_class2_data = self.class2_data.copy() if self.class2_data is not None else None
                self.original_labeled_class1_data = self.labeled_class1_data.copy() if getattr(self, 'labeled_class1_data', None) is not None else None
                self.original_labeled_class2_data = self.labeled_class2_data.copy() if getattr(self, 'labeled_class2_data', None) is not None else None

            # Update volume data
            self.volume_data = aligned_vol
            
            # Apply same rotation to labels and masks if they exist
            import math
            c_old = (np.array(aligned_vol.shape) - 1.0) / 2.0
            offset = c_old - np.dot(best_R, c_old)
            
            # Rotate Class 1 mask
            if self.class1_data is not None:
                self.class1_data = ndimage.affine_transform(
                    self.class1_data,
                    matrix=best_R,
                    offset=offset,
                    order=0,
                    mode='constant',
                    cval=0
                )
                
            # Rotate Class 2 mask
            if self.class2_data is not None:
                self.class2_data = ndimage.affine_transform(
                    self.class2_data,
                    matrix=best_R,
                    offset=offset,
                    order=0,
                    mode='constant',
                    cval=0
                )
                
            # Re-merge masks if either exists
            if self.class1_data is not None or self.class2_data is not None:
                self.merge_class_masks()
                
            # Rotate labeled arrays if present
            if getattr(self, 'labeled_class1_data', None) is not None:
                self.labeled_class1_data = ndimage.affine_transform(
                    self.labeled_class1_data,
                    matrix=best_R,
                    offset=offset,
                    order=0,
                    mode='constant',
                    cval=0
                )
            if getattr(self, 'labeled_class2_data', None) is not None:
                self.labeled_class2_data = ndimage.affine_transform(
                    self.labeled_class2_data,
                    matrix=best_R,
                    offset=offset,
                    order=0,
                    mode='constant',
                    cval=0
                )
                
            # Reset oblique angles to 0
            self.oblique_angles = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
            
            # Recalculate stats if needed
            if hasattr(self, 'update_object_measurements'):
                self.update_object_measurements()
                
            # Re-initialize crosshair position at new center
            z, y, x = aligned_vol.shape
            self.crosshair_position = [x // 2, y // 2, z // 2]
            
            # Update sliders for crosshair
            for axis, dim, ctr in [('x', x-1, x//2), ('y', y-1, y//2), ('z', z-1, z//2)]:
                sl = getattr(self, f'crosshair_slider_{axis}', None)
                if sl:
                    sl.blockSignals(True)
                    sl.setMaximum(dim)
                    sl.setValue(ctr)
                    sl.blockSignals(False)
            
            # Redraw all views
            self.update_all_views()
            
            # Compute angles in degrees for user feedback
            try:
                pitch = math.degrees(math.asin(-best_R[0, 2]))
                cos_pitch = math.cos(math.radians(pitch))
                if abs(cos_pitch) > 1e-4:
                    roll = math.degrees(math.atan2(best_R[1, 2], best_R[2, 2]))
                    yaw = math.degrees(math.atan2(best_R[0, 1], best_R[0, 0]))
                else:
                    roll = 0.0
                    yaw = math.degrees(math.atan2(-best_R[1, 0], best_R[1, 1]))
            except Exception:
                roll, pitch, yaw = 0.0, 0.0, 0.0
                
            # Enable reset buttons
            if hasattr(self, 'btn_reset_align'):
                self.btn_reset_align.setEnabled(True)
            if hasattr(self, 'reset_align_btn'):
                self.reset_align_btn.setEnabled(True)
                
            QMessageBox.information(
                self, 
                "Alignment Success", 
                f"HBM volume aligned successfully!\n\n"
                f"Computed misalignment corrections:\n"
                f"• Roll (X-axis): {roll:+.2f}°\n"
                f"• Pitch (Y-axis): {pitch:+.2f}°\n"
                f"• Yaw (Z-axis): {yaw:+.2f}°"
            )
            
        self.align_thread.progress.connect(on_progress)
        self.align_thread.finished.connect(on_finished)
        self.align_thread.start()

    def reset_hbm_alignment(self):
        """Restore the volume and masks to their original unaligned states"""
        if getattr(self, 'original_volume_data', None) is None:
            QMessageBox.warning(self, "No Backup", "No alignment has been performed yet, or backup is not available.")
            return
            
        self.volume_data = self.original_volume_data.copy()
        
        self.class1_data = self.original_class1_data.copy() if self.original_class1_data is not None else None
        self.class2_data = self.original_class2_data.copy() if self.original_class2_data is not None else None
        
        if self.class1_data is not None or self.class2_data is not None:
            self.merge_class_masks()
        else:
            self.segmentation_data = None
            
        if self.original_labeled_class1_data is not None:
            self.labeled_class1_data = self.original_labeled_class1_data.copy()
        else:
            self.labeled_class1_data = None
            
        if getattr(self, 'original_labeled_class2_data', None) is not None:
            self.labeled_class2_data = self.original_labeled_class2_data.copy()
        else:
            self.labeled_class2_data = None
            
        # Reset oblique angles to 0
        self.oblique_angles = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
        
        # Recalculate stats if needed
        if hasattr(self, 'update_object_measurements'):
            self.update_object_measurements()
            
        # Re-initialize crosshair position at original center
        z, y, x = self.volume_data.shape
        self.crosshair_position = [x // 2, y // 2, z // 2]
        
        # Update sliders for crosshair
        for axis, dim, ctr in [('x', x-1, x//2), ('y', y-1, y//2), ('z', z-1, z//2)]:
            sl = getattr(self, f'crosshair_slider_{axis}', None)
            if sl:
                sl.blockSignals(True)
                sl.setMaximum(dim)
                sl.setValue(ctr)
                sl.blockSignals(False)
        
        # Redraw all views
        self.update_all_views()
        
        # Disable reset buttons
        if hasattr(self, 'btn_reset_align'):
            self.btn_reset_align.setEnabled(False)
        if hasattr(self, 'reset_align_btn'):
            self.reset_align_btn.setEnabled(False)
            
        QMessageBox.information(self, "Reset Success", "Volume and masks restored to original unaligned states.")

    def _toggle_oblique_handles(self, checked):
        """Toggle Dragonfly-style oblique rotation handles on/off.
        
        When ON: shows curved arrow handles on crosshair lines, auto-enables
        crosshair if not already active.
        When OFF: hides handles and resets hover state.
        """
        self.oblique_handles_enabled = checked
        
        if checked:
            # Update button visual
            if hasattr(self, 'btn_oblique_handles'):
                self.btn_oblique_handles.setStyleSheet("background: rgba(0, 200, 100, 0.3); font-weight: bold;")
            
            # Auto-enable crosshair (handles require it)
            if not self.crosshair_enabled:
                self.crosshair_enabled = True
                parent = self.parent()
                while parent is not None:
                    if hasattr(parent, 'cross_btn'):
                        parent.cross_btn.setChecked(True)
                        break
                    parent = parent.parent()
        else:
            # Clear hover/drag state
            self._oblique_hover_handle = None
            self._oblique_dragging = None
            if hasattr(self, 'btn_oblique_handles'):
                self.btn_oblique_handles.setStyleSheet("")
            # Reset cursors
            for ori in ['axial', 'coronal', 'sagittal']:
                w = getattr(self, f'{ori}_widget', None)
                if w:
                    w.unsetCursor()
        
        # Re-render all views to show/hide handles
        if self.volume_data is not None:
            for ori in ['axial', 'coronal', 'sagittal']:
                self.render_slice(ori, preserve_camera=True)
    
    def _toggle_manual_align_mode(self, checked):
        """Toggle manual alignment mode on/off.
        
        When ON: enables crosshair + oblique rotate mode so user can drag
        crosshair handles on 2D views to preview rotation in real-time.
        When OFF: disables the mode and leaves the preview rotation in place.
        """
        self._manual_align_active = checked
        
        if checked:
            # Turn ON: enable crosshair and oblique dragging
            if hasattr(self, 'btn_manual_align_mode'):
                self.btn_manual_align_mode.setText("Manual Align: ON")
                self.btn_manual_align_mode.setStyleSheet("background: rgba(0, 200, 100, 0.3); font-weight: bold;")
            if hasattr(self, 'btn_commit_manual_align'):
                self.btn_commit_manual_align.setEnabled(True)
            if hasattr(self, 'btn_cancel_manual_align'):
                self.btn_cancel_manual_align.setEnabled(True)
            
            # Auto-enable rotation handles for manual align
            if not self.oblique_handles_enabled:
                self.oblique_handles_enabled = True
                if hasattr(self, 'btn_oblique_handles'):
                    self.btn_oblique_handles.setChecked(True)
                    self.btn_oblique_handles.setStyleSheet("background: rgba(0, 200, 100, 0.3); font-weight: bold;")
            
            # Ensure crosshair is visible (needed for handle-based rotation)
            if not self.crosshair_enabled:
                self.crosshair_enabled = True
                parent = self.parent()
                while parent is not None:
                    if hasattr(parent, 'cross_btn'):
                        parent.cross_btn.setChecked(True)
                        break
                    parent = parent.parent()
            
            # Reset oblique angles for a fresh manual align session
            self.oblique_angles = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
            self._update_manual_angle_labels()
            
            if self.volume_data is not None:
                for ori in ['axial', 'coronal', 'sagittal']:
                    self.render_slice(ori, preserve_camera=True)
                self.update_3d_crosshair()
        else:
            # Turn OFF: just update button text, keep preview angles
            if hasattr(self, 'btn_manual_align_mode'):
                self.btn_manual_align_mode.setText("Manual Align: OFF")
                self.btn_manual_align_mode.setStyleSheet("")
        
        # Sync the left sidebar button in MainWindow
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'manual_align_btn'):
                parent.manual_align_btn.blockSignals(True)
                parent.manual_align_btn.setChecked(checked)
                if checked:
                    parent.manual_align_btn.setText("MANUAL ALIGN: ON")
                    parent.manual_align_btn.setStyleSheet("background: rgba(0, 200, 100, 0.3); font-weight: bold;")
                else:
                    parent.manual_align_btn.setText("MANUAL ALIGN")
                    parent.manual_align_btn.setStyleSheet("")
                parent.manual_align_btn.blockSignals(False)
                break
            parent = parent.parent()
    
    def _update_manual_angle_labels(self):
        """Sync the angle display labels with current oblique_angles.
        
        Called after every oblique rotation drag so the user sees the live angles.
        """
        if not hasattr(self, 'manual_angle_x_label'):
            return
        roll = self.oblique_angles.get('sagittal', 0.0)
        pitch = self.oblique_angles.get('coronal', 0.0)
        yaw = self.oblique_angles.get('axial', 0.0)
        self.manual_angle_x_label.setText(f"{roll:+.2f}°")
        self.manual_angle_y_label.setText(f"{pitch:+.2f}°")
        self.manual_angle_z_label.setText(f"{yaw:+.2f}°")
    
    def commit_manual_alignment(self):
        """Apply the current oblique preview rotation permanently to the volume data.
        
        Reads the current oblique_angles (set interactively by dragging crosshair
        handles on 2D views), computes the rotation matrix, applies affine_transform
        to volume + masks, then resets oblique_angles to 0.
        """
        if self.volume_data is None:
            QMessageBox.warning(self, "No Data", "Please load a volume file first.")
            return
        
        roll = self.oblique_angles.get('sagittal', 0.0)
        pitch = self.oblique_angles.get('coronal', 0.0)
        yaw = self.oblique_angles.get('axial', 0.0)
        
        if abs(roll) < 0.01 and abs(pitch) < 0.01 and abs(yaw) < 0.01:
            QMessageBox.information(self, "No Rotation", "No rotation to commit. Drag crosshair handles first.")
            return
        
        # Create backup if this is the first alignment operation
        if getattr(self, 'original_volume_data', None) is None:
            self.original_volume_data = self.volume_data.copy()
            self.original_class1_data = self.class1_data.copy() if self.class1_data is not None else None
            self.original_class2_data = self.class2_data.copy() if self.class2_data is not None else None
            self.original_labeled_class1_data = self.labeled_class1_data.copy() if getattr(self, 'labeled_class1_data', None) is not None else None
            self.original_labeled_class2_data = self.labeled_class2_data.copy() if getattr(self, 'labeled_class2_data', None) is not None else None
        
        # Show progress dialog
        self.manual_align_progress = QProgressDialog("Committing manual alignment...", "Cancel", 0, 100, self)
        self.manual_align_progress.setWindowModality(Qt.WindowModal)
        self.manual_align_progress.setMinimumDuration(0)
        self.manual_align_progress.show()
        
        # Use the oblique_angles as rotation angles
        self.manual_align_thread = ManualAlignThread(self.volume_data, roll, pitch, yaw)
        
        def on_progress(value, msg):
            self.manual_align_progress.setValue(value)
            self.manual_align_progress.setLabelText(msg)
            
        def on_finished(rotated_vol, R_matrix, error):
            self.manual_align_progress.close()
            if error:
                QMessageBox.critical(self, "Manual Alignment Error", f"Failed to apply rotation: {error}")
                return
                
            # Update volume data
            self.volume_data = rotated_vol
            
            # Apply same rotation to masks
            c_old = (np.array(rotated_vol.shape) - 1.0) / 2.0
            offset = c_old - np.dot(R_matrix, c_old)
            
            if self.class1_data is not None:
                self.class1_data = ndimage.affine_transform(
                    self.class1_data, matrix=R_matrix, offset=offset, order=0, mode='constant', cval=0)
            if self.class2_data is not None:
                self.class2_data = ndimage.affine_transform(
                    self.class2_data, matrix=R_matrix, offset=offset, order=0, mode='constant', cval=0)
            if self.class1_data is not None or self.class2_data is not None:
                self.merge_class_masks()
            if getattr(self, 'labeled_class1_data', None) is not None:
                self.labeled_class1_data = ndimage.affine_transform(
                    self.labeled_class1_data, matrix=R_matrix, offset=offset, order=0, mode='constant', cval=0)
            if getattr(self, 'labeled_class2_data', None) is not None:
                self.labeled_class2_data = ndimage.affine_transform(
                    self.labeled_class2_data, matrix=R_matrix, offset=offset, order=0, mode='constant', cval=0)
            
            # Reset oblique angles to 0 (rotation is now baked into the data)
            self.oblique_angles = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
            self._update_manual_angle_labels()
            
            # Recalculate stats
            if hasattr(self, 'update_object_measurements'):
                self.update_object_measurements()
            
            # Update crosshair
            z, y, x = rotated_vol.shape
            self.crosshair_position = [x // 2, y // 2, z // 2]
            for axis, dim, ctr in [('x', x-1, x//2), ('y', y-1, y//2), ('z', z-1, z//2)]:
                sl = getattr(self, f'crosshair_slider_{axis}', None)
                if sl:
                    sl.blockSignals(True)
                    sl.setMaximum(dim)
                    sl.setValue(ctr)
                    sl.blockSignals(False)
            
            self.update_all_views()
            
            # Enable reset buttons
            if hasattr(self, 'btn_reset_align'):
                self.btn_reset_align.setEnabled(True)
            if hasattr(self, 'reset_align_btn'):
                self.reset_align_btn.setEnabled(True)
            
            # Turn off manual align mode
            self.btn_manual_align_mode.setChecked(False)
            self._toggle_manual_align_mode(False)
            self.btn_commit_manual_align.setEnabled(False)
            self.btn_cancel_manual_align.setEnabled(False)
            
            QMessageBox.information(
                self, "Manual Alignment Committed",
                f"Volume rotation applied permanently!\n\n"
                f"Applied rotation angles:\n"
                f"• Roll (X-axis): {roll:+.2f}°\n"
                f"• Pitch (Y-axis): {pitch:+.2f}°\n"
                f"• Yaw (Z-axis): {yaw:+.2f}°"
            )
            
        self.manual_align_thread.progress.connect(on_progress)
        self.manual_align_thread.finished.connect(on_finished)
        self.manual_align_thread.start()
    
    def _cancel_manual_align(self):
        """Cancel the manual alignment preview and reset oblique angles to 0."""
        self.oblique_angles = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
        self._update_manual_angle_labels()
        
        # Turn off manual align mode
        self.btn_manual_align_mode.setChecked(False)
        self._toggle_manual_align_mode(False)
        self.btn_commit_manual_align.setEnabled(False)
        self.btn_cancel_manual_align.setEnabled(False)
        
        # Re-render with no oblique angles (reset to straight views)
        if self.volume_data is not None:
            for ori in ['axial', 'coronal', 'sagittal']:
                self.render_slice(ori, preserve_camera=True)
            self.update_3d_crosshair()
    
    def save_aligned_volume(self):
        """Save the current volume data to a TIFF stack or RAW binary file."""
        if self.volume_data is None:
            QMessageBox.warning(self, "No Data", "No volume data to save.")
            return
        
        from pathlib import Path
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Volume",
            "",
            "TIFF Stack (*.tif *.tiff);;RAW Binary (*.raw);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            ext = Path(file_path).suffix.lower()
            
            progress = QProgressDialog("Saving volume...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            
            if ext in ['.tif', '.tiff']:
                # Save as multi-page TIFF using tifffile or skimage
                progress.setLabelText("Writing TIFF stack...")
                progress.setValue(10)
                
                try:
                    import tifffile
                    tifffile.imwrite(file_path, self.volume_data)
                except ImportError:
                    from skimage import io as skio
                    skio.imsave(file_path, self.volume_data)
                    
                progress.setValue(90)
                
            elif ext == '.raw':
                # Save as flat binary
                progress.setLabelText("Writing RAW binary...")
                progress.setValue(10)
                
                self.volume_data.tofile(file_path)
                progress.setValue(80)
                
                # Write metadata sidecar file
                meta_path = file_path + '.meta.txt'
                z, y, x = self.volume_data.shape
                with open(meta_path, 'w') as f:
                    f.write(f"# Volume Metadata\n")
                    f.write(f"shape_z={z}\n")
                    f.write(f"shape_y={y}\n")
                    f.write(f"shape_x={x}\n")
                    f.write(f"dtype={self.volume_data.dtype}\n")
                    f.write(f"byte_order=little_endian\n")
                    f.write(f"spacing_x={self.spacing[0]}\n")
                    f.write(f"spacing_y={self.spacing[1]}\n")
                    f.write(f"spacing_z={self.spacing[2]}\n")
                progress.setValue(90)
            else:
                # Fallback: try TIFF
                progress.setLabelText("Writing TIFF stack (default)...")
                progress.setValue(10)
                try:
                    import tifffile
                    tifffile.imwrite(file_path, self.volume_data)
                except ImportError:
                    from skimage import io as skio
                    skio.imsave(file_path, self.volume_data)
                progress.setValue(90)
            
            progress.setValue(100)
            progress.close()
            
            z, y, x = self.volume_data.shape
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            
            QMessageBox.information(
                self, "Save Success",
                f"Volume saved successfully!\n\n"
                f"File: {file_path}\n"
                f"Shape: ({z}, {y}, {x})\n"
                f"Dtype: {self.volume_data.dtype}\n"
                f"Size: {file_size_mb:.1f} MB"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save volume:\n{str(e)}")

    def update_info_label(self):
        """Update info label vÃ¡Â»â€ºi status cÃ¡Â»Â§a masks"""
        info_parts = []
        
        if self.volume_data is not None:
            z, y, x = self.volume_data.shape
            info_parts.append(f"Vol:({z},{y},{x})")
        
        if self.class1_data is not None:
            c1_voxels = np.sum(self.class1_data > 127)
            info_parts.append(f"C1:{c1_voxels:,}")
        
        if self.class2_data is not None:
            c2_voxels = np.sum(self.class2_data > 127)
            info_parts.append(f"C2:{c2_voxels:,}")
        
        if info_parts:
            self.info_label.setText(" | ".join(info_parts))
        else:
            self.info_label.setText("No data loaded")

# ==================== INSPECTION RESULT TAB ====================

# ==================== INSPECTION RESULT TAB ====================



