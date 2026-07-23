# inno3d/features/viewer/stats_panel.py
# -----------------------------------------------------------------------
# StatsPanelMixin  --  extracted from inno3d/tabs/viewer.py (Phase 3.6)
#
# Object statistics panel UI (create_object_stats_panel), MES/B2B tables,
# object selection, FAR, CSV export, MES 3D highlight, B2B gap actors.
# -----------------------------------------------------------------------

import csv
import math
import os
import re
from pathlib import Path as _Path
from typing import List, Optional

import numpy as np
import vtk
from vtk.util import numpy_support
from skimage import measure

from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap, QPainter, QPen, QBrush
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QToolButton, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QSizePolicy, QGroupBox, QButtonGroup,
    QScrollArea, QFrame, QAbstractSpinBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QLineEdit, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox, QSplitter, QAbstractItemView, QProgressDialog,
    QApplication,
)

from inno3d.core.styles import SemiconductorTheme


def _viewer_shared():
    """Lazy import host helpers defined on tabs.viewer (avoids circular import)."""
    import inno3d.tabs.viewer as _v
    return _v


def _ui_icon(*args, **kwargs):
    return _viewer_shared()._ui_icon(*args, **kwargs)


def _numeric_item(*args, **kwargs):
    return _viewer_shared()._numeric_item(*args, **kwargs)


def _category_item(*args, **kwargs):
    return _viewer_shared()._category_item(*args, **kwargs)


class StatsPanelMixin:
    """Mixin: Object-statistics panel, B2B table, MES highlight, FAR, CSV.

    Extracted from inno3d/tabs/viewer.py Phase 3.6.
    """

    def create_object_stats_panel(self):
        """Create stats panel with MES (Object Stats) + B2B (Boundary) tabs.

        Layout zones (top → bottom), each full-width and non-overlapping:
          1) Title bar  — title | status chip | actions (fixed icons)
          2) Filter bar — search field expands; never collides with actions
          3) Tabs + table / online log (splitter)
        """
        from PyQt5.QtWidgets import QHeaderView, QTabWidget, QSizePolicy, QFrame

        # Shared table helpers still live on tabs.viewer (circular-import safe when lazy)
        _v = _viewer_shared()
        FrozenTableWidget = _v.FrozenTableWidget
        ExcelFilterHeader = _v.ExcelFilterHeader
        _ui_icon = _v._ui_icon

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
            # Compute C2 volume for ALL objects in one vectorized pass (fast)
            c2_subset = self.class2_data[start_slice:end_slice] if self.class2_data is not None else None
            c2_mask_all = (c2_subset > 0) if c2_subset is not None else None

            from scipy import ndimage
            progress.setLabelText("Calculating volumes (optimized)...")
            progress.setValue(80)

            # labels_list: list of object IDs [1, 2, 3, ...]
            labels_list = [prop.label for prop in c1_props]

            # ndimage.sum counts C2 pixels within each labeled C1 region
            if c2_mask_all is not None:
                c2_pixel_counts = ndimage.sum(c2_mask_all, labeled_subset, index=labels_list)
            else:
                c2_pixel_counts = np.zeros(len(labels_list))

            # Loop only to format output data — all heavy computation is already done
            for idx, prop in enumerate(c1_props):
                c1_volume = prop.area * voxel_volume
                c2_volume = float(c2_pixel_counts[idx]) * voxel_volume  # pre-computed above
                
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
    
    
