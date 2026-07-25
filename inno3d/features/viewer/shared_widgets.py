# inno3d/features/viewer/shared_widgets.py
# -----------------------------------------------------------------------
# Shared widget helpers — extracted from inno3d/tabs/viewer.py (F11)
#
# Contains helper functions + standalone classes used by multiple mixins:
#   _ui_icon, _natural_sort_key, _numeric_item, _category_item
#   NumericSortItem, CategorySortItem
#   StatsRowDelegate, FrozenTableWidget
#   AlignVolumeThread, ManualAlignThread
#   ExcelFilterMenu, ExcelFilterHeader
# -----------------------------------------------------------------------

import os
import re
from pathlib import Path

import numpy as np

from PyQt5.QtCore import (
    Qt, QThread, QTimer, pyqtSignal, QSize, QPoint, QRect,
    QItemSelection, QItemSelectionModel,
)
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QIcon, QLinearGradient, QPainter, QPainterPath,
    QPen, QPixmap,
)
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMenu, QPushButton, QScrollArea, QScrollBar, QSlider, QStyle,
    QStyledItemDelegate, QStyleOptionViewItem, QTableView, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QWidgetAction,
)

from inno3d.core.styles import SemiconductorTheme
from inno3d.infra.paths import project_root, resource_path


def _ui_icon(name, widget=None, fallback=None):
    """Resolve ``assets/icons/<name>`` for dev + PyInstaller.

    Must NOT use ``Path(__file__).parents[N]`` alone — this module lives under
    ``inno3d/features/viewer/`` (parents[2] = ``inno3d/``, not the project root).
    """
    candidates = [
        Path(resource_path(os.path.join("assets", "icons", name))),
        project_root() / "assets" / "icons" / name,
        # Frozen / alternate layouts
        Path(__file__).resolve().parents[3] / "assets" / "icons" / name,
    ]
    for icon_path in candidates:
        try:
            if icon_path.is_file():
                return QIcon(str(icon_path))
        except OSError:
            continue
    if widget is not None and fallback is not None:
        return widget.style().standardIcon(fallback)
    return QIcon()


def _tinted_ui_icon(name, color=None, size=14, widget=None, fallback=None):
    """Load a monochrome SVG and tint it for theme contrast.

    Many icons hardcode a light stroke (``#DDE7F2``) which disappears on light
    ``icon-button`` backgrounds. Tint with ``TEXT_PRIMARY`` (or an explicit
    color) so the glyph stays visible in both dark and light themes.
    """
    base = _ui_icon(name, widget=widget, fallback=fallback)
    if base is None or base.isNull():
        return base
    if color is None:
        color = QColor(SemiconductorTheme.TEXT_PRIMARY)
    elif not isinstance(color, QColor):
        color = QColor(str(color))
    sz = QSize(int(size), int(size))
    pixmap = base.pixmap(sz)
    if pixmap.isNull():
        return base
    tinted = QPixmap(pixmap.size())
    tinted.fill(Qt.transparent)
    painter = QPainter(tinted)
    painter.drawPixmap(0, 0, pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), color)
    painter.end()
    return QIcon(tinted)


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
            # Strong cyan overlay — must read on solid OK/NG row tints
            painter.fillRect(rect, QColor(30, 200, 255, 130))
            # Left accent bar (selection cue independent of row colour)
            bar = QRect(rect.left(), rect.top(), 3, rect.height())
            painter.fillRect(bar, QColor(0, 230, 255, 255))
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
        """Map frozen-view click → main table selection + cellClicked.

        Frozen shares ``selectionModel`` with the main table, so Qt already
        applied Multi toggle / Single ClearAndSelect on mouse press.  Do **not**
        Toggle again here (that undoes Multi picks). Only ensure full-row
        selection in Single mode and keep the current index.
        """
        if not index.isValid():
            return
        model = self.model()
        sm = self.selectionModel()
        if model is None or sm is None:
            return

        row = index.row()
        left = model.index(row, 0)
        right = model.index(row, max(0, self.columnCount() - 1))
        if not left.isValid():
            return
        sel = QItemSelection(left, right)
        mode = self.selectionMode()

        if mode in (
            QAbstractItemView.MultiSelection,
            QAbstractItemView.ExtendedSelection,
        ):
            # Already toggled by QTableView on press — never Toggle/Select again
            # (would undo Multi ON deselect). Only pin current index.
            sm.setCurrentIndex(left, QItemSelectionModel.NoUpdate)
        else:
            # Single: full-row sticky selection (kept until another row / Clear)
            sm.select(
                sel,
                QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
            )
            sm.setCurrentIndex(left, QItemSelectionModel.NoUpdate)

        try:
            self.cellClicked.emit(row, index.column())
        except Exception:
            pass
        try:
            self.viewport().update()
            if getattr(self, "frozenTableView", None) is not None:
                self.frozenTableView.viewport().update()
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




