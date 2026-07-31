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
import time
from pathlib import Path as _Path
from typing import List, Optional

import numpy as np
import vtk
from vtk.util import numpy_support
from skimage import measure

from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal, QItemSelection, QItemSelectionModel
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
from inno3d.features.shared.b2b_gap_3d import (
    B2BGapOverlay,
    format_gap_label as _shared_format_gap_label,
    layer_z_offset as _shared_layer_z_offset,
    world_spacing as _shared_world_spacing,
)
from inno3d.features.shared.mes_highlight_3d import MESHighlightOverlay
from inno3d.features.shared.mes_mapping import (
    MesPickDebounce,
    centroid_nav_xyz as _shared_centroid_nav_xyz,
    label_at_stat_centroid as _shared_label_at_stat_centroid,
    label_at_volume_xyz as _shared_label_at_volume_xyz,
    mes_bbox_for_labels as _shared_mes_bbox_for_labels,
    resolve_stat_label as _shared_resolve_stat_label,
    stats_index_for_label as _shared_stats_index_for_label,
)


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

        # Multi-select toggle (MES rows + MPR/3D bump highlight set)
        self.mes_multi_select_check = QCheckBox("Multi")
        self.mes_multi_select_check.setObjectName("mesMultiSelectToggle")
        self.mes_multi_select_check.setToolTip(
            "Multi-select bumps (MES table ↔ MPR/3D) — sticky both ways:\n"
            "· OFF: click 1 MES row or Ctrl+click MPR/3D → sticky single (kept after release)\n"
            "· ON: click rows / Ctrl+click MPR/3D → multi highlight set (all kept)\n"
            "· Clear (X) to deselect all"
        )
        self.mes_multi_select_check.setChecked(False)
        self.mes_multi_select_check.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.mes_multi_select_check.setStyleSheet(f"""
            QCheckBox#mesMultiSelectToggle {{
                color: {SemiconductorTheme.TEXT_SECONDARY};
                font-size: 8pt;
                font-weight: 700;
                spacing: 4px;
                background: transparent;
                border: none;
                padding: 0 4px;
            }}
            QCheckBox#mesMultiSelectToggle:checked {{
                color: {SemiconductorTheme.ACCENT_PRIMARY};
            }}
            QCheckBox#mesMultiSelectToggle::indicator {{
                width: 14px;
                height: 14px;
            }}
        """)
        self._mes_multi_select = False
        self.mes_multi_select_check.toggled.connect(self._on_mes_multi_select_toggled)
        filter_row.addWidget(self.mes_multi_select_check, 0, Qt.AlignVCenter)

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
        # Default single-select; toggle "Multi" next to Filter enables ExtendedSelection
        self.object_stats_table.setSelectionMode(QAbstractItemView.SingleSelection)
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
        self.object_stats_table.setToolTip(
            "MES object stats (synced with MPR/3D Ctrl+click):\n"
            "  · Multi OFF: click 1 row or Ctrl+click MPR/3D = sticky single\n"
            "  · Multi ON: click rows / Ctrl+click MPR/3D = multi sticky set\n"
            "  · Clear (X) to deselect"
        )
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
            "  · Cyan = SRC (source) · Amber = DST (destination)\n"
            "  · Tags SRC/DST on voxels + caption lists SRC/DST bump IDs\n"
            "  · Tube cyan→amber + red arrow = gap direction SRC→DST\n"
            "  · Caption = distance · direction · SRC/DST IDs"
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

            # Unified (1,1)=top-left XY indexing (same as Online MES / B2B)
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

    # ── B2B gap → 3D Volume (shared module; spatial distance, not MPR) ───

    def _get_b2b_gap_overlay(self) -> B2BGapOverlay:
        """Lazy shared B2B gap actor stack (Viewer + Teaching parity)."""
        ov = getattr(self, "_b2b_gap_overlay", None)
        if ov is None:
            ov = B2BGapOverlay()
            self._b2b_gap_overlay = ov
        return ov

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
        """Local B2B layer Z → full-volume Z (shared helper)."""
        bands = None
        try:
            mw = self.window()
            if hasattr(mw, "_online_layer_bands"):
                bands = mw._online_layer_bands()
        except Exception:
            bands = None
        return _shared_layer_z_offset(row_dict, layer_bands=bands, fallback=0)

    def _get_3d_world_spacing(self):
        return _shared_world_spacing(
            getattr(self, "custom_spacing", None),
            getattr(self, "spacing", [1.0, 1.0, 1.0]),
        )

    def _clear_b2b_gap_actors(self, restore_context=True):
        ren = getattr(self, "view_3d_renderer", None)
        widget = getattr(self, "view_3d_widget", None)
        ov = self._get_b2b_gap_overlay()

        def _restore():
            if restore_context and hasattr(self, "_set_mes_3d_context_dim"):
                try:
                    self._set_mes_3d_context_dim(False)
                except Exception:
                    pass

        ov.clear(
            ren,
            widget,
            render=True,
            on_restore_context=_restore if restore_context else None,
        )
        # Legacy attrs kept in sync for any external readers
        self._b2b_gap_actors = ov.actors
        self._b2b_active_stats_idx = ov.active_stats_idx

    @staticmethod
    def _b2b_format_gap_label(layer, eucl, src_rc, dst_rc, direction,
                              src_zyx=None, dst_zyx=None):
        return _shared_format_gap_label(
            layer, eucl, src_rc, dst_rc, direction, src_zyx, dst_zyx
        )

    def _draw_b2b_gap_line(self, src_voxel, dst_voxel, label_text="", stats_idx=None):
        """Draw Src→Dst gap on 3D Volume via shared B2BGapOverlay (Viewer SoT)."""
        ren = getattr(self, "view_3d_renderer", None)
        widget = getattr(self, "view_3d_widget", None)
        if ren is None or self.volume_data is None:
            print("[B2B] 3D gap skipped: no renderer/volume")
            return

        if not self.ensure_3d_volume_render_for_overlay(reason="b2b_gap"):
            print("[B2B] 3D gap skipped: could not enable 3D volume")
            return

        if hasattr(self, "_clear_mes_3d_highlight"):
            try:
                self._clear_mes_3d_highlight(render=False, restore_context=False)
            except Exception:
                pass

        ov = self._get_b2b_gap_overlay()
        spacing = self._get_3d_world_spacing()
        try:
            R = float(self._volume_world_radius())
        except Exception:
            R = 100.0

        def _dim():
            if hasattr(self, "_set_mes_3d_context_dim"):
                self._set_mes_3d_context_dim(True)

        def _pivot():
            if hasattr(self, "_sync_3d_orbit_pivot"):
                self._sync_3d_orbit_pivot()

        ok = ov.draw(
            ren,
            widget,
            src_voxel,
            dst_voxel,
            spacing,
            label_text=label_text,
            labeled=getattr(self, "labeled_class1_data", None),
            class1_binary=getattr(self, "class1_data", None),
            on_dim_context=_dim,
            volume_world_radius=R,
            sync_orbit_pivot=_pivot,
            frame_camera=True,
            stats_idx=stats_idx,
        )
        self._b2b_gap_actors = ov.actors
        self._b2b_active_stats_idx = ov.active_stats_idx if ok else None

    # ── MES object highlight on 3D Volume (shared MESHighlightOverlay) ───

    def _get_mes_highlight_overlay(self) -> MESHighlightOverlay:
        ov = getattr(self, "_mes_highlight_overlay", None)
        if ov is None:
            ov = MESHighlightOverlay()
            self._mes_highlight_overlay = ov
        return ov

    def _clear_mes_3d_highlight(self, render=True, restore_context=True):
        """Remove MES selection surfaces from the 3D renderer (shared overlay)."""
        ren = getattr(self, "view_3d_renderer", None)
        widget = getattr(self, "view_3d_widget", None)
        ov = self._get_mes_highlight_overlay()

        def _restore():
            if restore_context:
                self._set_mes_3d_context_dim(False)

        ov.clear(
            ren,
            widget,
            render=render,
            on_restore_context=_restore if restore_context else None,
        )
        self._mes_3d_highlight_actors = ov.actors
        self._mes_3d_hl_cache_key = ov.cache_key

    def _set_mes_3d_context_dim(self, dim):
        """Dim global C1/C2 shells so the selected MES object pops on 3D."""
        from inno3d.features.viewer.seg_mask_3d import set_mask_overlay_opacity

        if not dim:
            if hasattr(self, "update_overlay_visibility"):
                try:
                    self.update_overlay_visibility("3d")
                except Exception:
                    pass
            return

        try:
            og = getattr(self, "view_3d_overlay_group", None)
            base = (
                float(og.opacity_slider.value()) / 100.0
                if og is not None and hasattr(og, "opacity_slider")
                else 0.35
            )
        except Exception:
            base = 0.35
        c1 = getattr(self, "c1_actor_3d", None)
        c2 = getattr(self, "c2_actor_3d", None)
        if c1 is not None:
            set_mask_overlay_opacity(c1, max(0.04, base * 0.12))
        if c2 is not None:
            set_mask_overlay_opacity(c2, max(0.05, base * 0.18))

    def _mes_3d_bbox_for_labels(self, label_set):
        """Union bbox of selected labels — shared mes_mapping helper."""
        return _shared_mes_bbox_for_labels(
            getattr(self, "object_stats", None),
            label_set,
            z_offset=int(getattr(self, "_measurement_start_slice", 0) or 0),
        )

    def _update_mes_3d_highlight(self):
        """Show selected MES bump(s) on 3D via shared MESHighlightOverlay."""
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

        if not self.is_3d_volume_render_enabled() or getattr(self, "volume_actor", None) is None:
            if not self.ensure_3d_volume_render_for_overlay(reason="mes_pick"):
                return

        ov = self._get_mes_highlight_overlay()
        ov.update(
            ren,
            widget,
            labeled,
            getattr(self, "object_stats", None),
            self.selected_highlight_objects,
            self._get_3d_world_spacing(),
            z_offset=int(getattr(self, "_measurement_start_slice", 0) or 0),
            highlight_color=getattr(self, "highlight_color", None),
            on_dim_context=lambda: self._set_mes_3d_context_dim(True),
            on_restore_context=lambda: self._set_mes_3d_context_dim(False),
        )
        self._mes_3d_highlight_actors = ov.actors
        self._mes_3d_hl_cache_key = ov.cache_key

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
        if active is None:
            ov = getattr(self, "_b2b_gap_overlay", None)
            if ov is not None:
                active = ov.active_stats_idx
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
            label = self._b2b_format_gap_label(
                layer, eucl, src_rc, dst_rc, direction, src, dst
            )
            try:
                self._draw_b2b_gap_line(src, dst, label_text=label, stats_idx=int(idx))
                if hasattr(self, "b2b_info_label"):
                    self.b2b_info_label.setText(
                        f"3D gap ON · SRC {src_rc} → DST {dst_rc} ({direction}) "
                        f"{eucl}µm · cyan=SRC · amber=DST · click row again to hide"
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
            self._draw_b2b_gap_line(
                (vz, vy, vx), (vz, vy, vx + 1), label_text=label, stats_idx=int(idx)
            )
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
        """Look up labeled_class1 id at MES object centroid (shared helper)."""
        return _shared_label_at_stat_centroid(
            getattr(self, "labeled_class1_data", None),
            stat,
            z_offset=int(getattr(self, "_measurement_start_slice", 0) or 0),
        )

    def _navigate_to_object_stat(self, stat):
        """Jump MPR sliders + 3D crosshair to object centroid (shared coords)."""
        if stat is None or self.volume_data is None:
            return
        xyz = _shared_centroid_nav_xyz(
            stat,
            z_offset=int(getattr(self, "_measurement_start_slice", 0) or 0),
        )
        if xyz is None:
            return
        cx, cy, cz = xyz
        # updatePoint expects (X, Y, Z) and syncs MPR + 3D crosshair
        self.updatePoint(cx, cy, cz)

    def _mpr_pick_volume_xyz(self, orientation, pos):
        """Map Qt pos on an MPR pane → volume voxel (X, Y, Z).

        In-plane axes follow crosshair place (``_pick_voxel``); the fixed axis
        is taken from the pane's current slice so pick matches the image shown.
        """
        if self.volume_data is None or not hasattr(self, "_pick_voxel"):
            return None
        vol_z, vol_y, vol_x = self.volume_data.shape
        x, y, z = self._pick_voxel(orientation, pos, mode="place")
        slices = getattr(self, "current_slices", None) or {}
        if orientation == "axial":
            z = int(slices.get("axial", z))
            # Slider Z is display index; labeled mask is storage order
            if getattr(self, "reverse_z", False):
                z = vol_z - 1 - z
        elif orientation == "coronal":
            y = int(slices.get("coronal", y))
        else:
            x = int(slices.get("sagittal", x))
        x = max(0, min(int(x), vol_x - 1))
        y = max(0, min(int(y), vol_y - 1))
        z = max(0, min(int(z), vol_z - 1))
        return x, y, z

    def _label_at_volume_xyz(self, x, y, z, search_rad=3):
        """Read labeled_class1 at (x,y,z); neighborhood fallback (shared)."""
        return _shared_label_at_volume_xyz(
            getattr(self, "labeled_class1_data", None),
            x, y, z,
            search_rad=search_rad,
        )

    def _visual_row_for_stats_index(self, stats_idx):
        """Find current visual table row for a ``object_stats`` index (sort-safe)."""
        table = getattr(self, "object_stats_table", None)
        if table is None or stats_idx is None:
            return None
        for row in range(table.rowCount()):
            if table.isRowHidden(row):
                continue
            item = table.item(row, 0)
            if item is None:
                continue
            try:
                if int(item.data(Qt.UserRole)) == int(stats_idx):
                    return row
            except (TypeError, ValueError):
                if row == int(stats_idx):
                    return row
        return None

    def _stats_index_for_label(self, label):
        """Map labeled_class1 id → index in ``object_stats`` (shared)."""
        return _shared_stats_index_for_label(
            getattr(self, "object_stats", None),
            label,
            labeled=getattr(self, "labeled_class1_data", None),
            z_offset=int(getattr(self, "_measurement_start_slice", 0) or 0),
        )

    def is_mes_multi_select(self):
        """True when Multi toggle is on (many MES rows / MPR bumps)."""
        # Prefer live checkbox state (source of truth for UI)
        cb = getattr(self, "mes_multi_select_check", None)
        if cb is not None:
            try:
                return bool(cb.isChecked())
            except Exception:
                pass
        return bool(getattr(self, "_mes_multi_select", False))

    def _set_mes_table_selection_mode(self, multi):
        """Apply Single vs MultiSelection on main + frozen MES table.

        Only changes mode when needed — re-applying can clear multi selection
        on some Qt builds and must not run on every Ctrl+pick.
        """
        table = getattr(self, "object_stats_table", None)
        if table is None:
            return
        # MultiSelection: plain click toggles rows (MES table → MPR/3D parity).
        # ExtendedSelection needs Ctrl for multi — felt “broken” vs single.
        mode = (
            QAbstractItemView.MultiSelection
            if multi
            else QAbstractItemView.SingleSelection
        )
        if table.selectionMode() != mode:
            table.setSelectionMode(mode)
        frozen = getattr(table, "frozenTableView", None)
        if frozen is not None:
            try:
                if frozen.selectionMode() != mode:
                    frozen.setSelectionMode(mode)
            except Exception:
                pass

    @staticmethod
    def _mes_row_is_selected(table, visual_row):
        """True if visual row is in the current selection (row-based)."""
        if table is None or visual_row is None or visual_row < 0:
            return False
        sm = table.selectionModel()
        if sm is None:
            return False
        try:
            return any(int(idx.row()) == int(visual_row) for idx in sm.selectedRows())
        except Exception:
            try:
                model = table.model()
                if model is None:
                    return False
                return sm.isSelected(model.index(int(visual_row), 0))
            except Exception:
                return False

    def _on_mes_multi_select_toggled(self, checked):
        """Filter-bar Multi toggle → table selection mode + MPR pick policy."""
        self._mes_multi_select = bool(checked)
        table = getattr(self, "object_stats_table", None)
        if table is None:
            return

        multi = self._mes_multi_select
        self._set_mes_table_selection_mode(multi)

        # Leaving multi → keep only current row (parity with single mode)
        if not multi:
            current = table.currentRow()
            table.blockSignals(True)
            try:
                table.clearSelection()
                if current >= 0:
                    self._select_mes_visual_row(current)
            finally:
                table.blockSignals(False)

        # Rebuild cyan highlights from current table selection
        self._sync_mes_highlights_from_table(navigate=True)

        n = self._mes_selected_visual_row_count()
        if hasattr(self, "set_stats_info"):
            try:
                if multi:
                    self.set_stats_info(
                        f"Multi ON · {n} selected · "
                        f"table click / Ctrl+click MPR/3D toggle (sticky)"
                    )
                else:
                    self.set_stats_info(
                        "Multi OFF · sticky single · "
                        "table click or Ctrl+click MPR/3D (kept after release)"
                    )
            except Exception:
                pass

    def _mes_selected_visual_rows(self):
        """Sorted unique visual rows currently selected (Multi-safe)."""
        table = getattr(self, "object_stats_table", None)
        if table is None:
            return []
        sm = table.selectionModel()
        try:
            if sm is not None:
                rows = sorted({int(idx.row()) for idx in sm.selectedRows()})
                if rows:
                    return rows
        except Exception:
            pass
        # Fallback: cell indexes (still works for SingleSelection)
        try:
            return sorted({int(i.row()) for i in table.selectedIndexes()})
        except Exception:
            return []

    def _mes_selected_visual_row_count(self):
        return len(self._mes_selected_visual_rows())

    def _sync_mes_highlights_from_table(self, navigate=True, navigate_stat=None):
        """Rebuild ``selected_highlight_objects`` from MES table selection.

        Same path for single + multi so Multi feels identical (just more labels).
        Multi ON reference: every selected MES row → cyan on MPR + 3D surfaces.
        """
        table = getattr(self, "object_stats_table", None)
        if table is None:
            return

        selected_visual = self._mes_selected_visual_rows()
        self.selected_highlight_objects = []
        target_stat = None
        current_item = table.currentItem()
        target_visual = current_item.row() if current_item is not None else -1

        for visual_row in selected_visual:
            stats_idx = self._stats_index_from_visual_row(visual_row)
            if stats_idx is None:
                continue
            if stats_idx < 0 or stats_idx >= len(self.object_stats or []):
                continue
            stat = self.object_stats[stats_idx]
            try:
                label = int(stat.get("label", 0))
            except (TypeError, ValueError):
                label = 0
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
            # Prefer current row for camera jump; else first selected
            if visual_row == target_visual or target_stat is None:
                target_stat = stat

        # Explicit navigate target (e.g. last MPR multi-pick)
        if navigate_stat is not None:
            target_stat = navigate_stat

        if navigate and target_stat is not None:
            self._navigate_to_object_stat(target_stat)

        if self.volume_data is not None:
            for ori in ["axial", "coronal", "sagittal"]:
                try:
                    self.render_slice(ori, preserve_camera=True)
                except Exception:
                    pass
            try:
                self._update_mes_3d_highlight()
            except Exception:
                pass
        elif not self.selected_highlight_objects:
            if hasattr(self, "_clear_mes_3d_highlight"):
                try:
                    self._clear_mes_3d_highlight(render=True)
                except Exception:
                    pass

    def _mes_pick_is_duplicate(self, lab):
        """True if same label was picked very recently (shared debounce)."""
        deb = getattr(self, "_mes_pick_debounce_obj", None)
        if deb is None:
            deb = MesPickDebounce(0.12)
            self._mes_pick_debounce_obj = deb
        return deb.is_duplicate(lab)

    def _3d_pick_volume_xyz(self, qt_pos):
        """Ctrl+click on 3D volume → first labeled voxel along camera ray.

        World space matches VTK image: origin (0,0,0), spacing (sx,sy,sz).
        Returns (x, y, z) voxel indices or None.
        """
        labeled = getattr(self, "labeled_class1_data", None)
        if labeled is None or self.volume_data is None:
            return None
        if not hasattr(self, "_df_clip_3d_qt_to_display") or not hasattr(
            self, "_df_clip_3d_display_ray"
        ):
            return None
        disp = self._df_clip_3d_qt_to_display(qt_pos)
        if disp is None:
            return None
        ray = self._df_clip_3d_display_ray(disp[0], disp[1])
        if ray is None:
            return None
        origin, direction = ray
        sp = (
            self.custom_spacing
            if getattr(self, "custom_spacing", None) is not None
            else getattr(self, "spacing", (1.0, 1.0, 1.0))
        )
        try:
            sx, sy, sz = float(sp[0]), float(sp[1]), float(sp[2])
        except (TypeError, ValueError, IndexError):
            sx = sy = sz = 1.0
        sx = sx if abs(sx) > 1e-12 else 1.0
        sy = sy if abs(sy) > 1e-12 else 1.0
        sz = sz if abs(sz) > 1e-12 else 1.0

        Z, Y, X = labeled.shape
        # March in world units; step ~ half min spacing for thin masks
        step = max(min(sx, sy, sz) * 0.5, 0.25)
        # Bounds in world (inclusive voxel centers span)
        x_max_w = max((X - 1) * sx, 0.0)
        y_max_w = max((Y - 1) * sy, 0.0)
        z_max_w = max((Z - 1) * sz, 0.0)
        # Rough ray length through AABB diagonal
        diag = (x_max_w ** 2 + y_max_w ** 2 + z_max_w ** 2) ** 0.5 + step
        n_steps = int(min(max(diag / step, 8), 4000))

        ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
        dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])

        best = None
        for i in range(n_steps + 1):
            t = i * step
            wx = ox + t * dx
            wy = oy + t * dy
            wz = oz + t * dz
            # Outside expanded AABB → keep going (ray may enter later)
            if (
                wx < -sx or wx > x_max_w + sx
                or wy < -sy or wy > y_max_w + sy
                or wz < -sz or wz > z_max_w + sz
            ):
                if best is not None:
                    break
                continue
            ix = int(round(wx / sx))
            iy = int(round(wy / sy))
            iz = int(round(wz / sz))
            if not (0 <= iz < Z and 0 <= iy < Y and 0 <= ix < X):
                continue
            lab = int(labeled[iz, iy, ix])
            if lab > 0:
                return ix, iy, iz
        return None

    def _apply_mes_pick_label(self, lab, xyz=None, source="mpr"):
        """Apply sticky MES table + MPR/3D highlight for a labeled bump.

        Rules:
          · Multi OFF: replace with this one bump — kept after mouse release
          · Multi ON: toggle this bump in/out of the multi set — all kept

        Returns True if selection changed / hit a table row.
        """
        try:
            lab = int(lab)
        except (TypeError, ValueError):
            return False
        if lab <= 0:
            return False

        # Debounce dual Qt+VTK (and accidental double events)
        if self._mes_pick_is_duplicate(lab):
            return True

        if not getattr(self, "object_stats", None):
            if hasattr(self, "set_stats_info"):
                try:
                    self.set_stats_info("No MES objects — run measurement first")
                except Exception:
                    pass
            return False

        stats_idx = self._stats_index_for_label(lab)
        xyz_s = ""
        if xyz is not None:
            try:
                x, y, z = int(xyz[0]), int(xyz[1]), int(xyz[2])
                xyz_s = f" @ X{x} Y{y} Z{z}"
            except (TypeError, ValueError, IndexError):
                xyz_s = ""

        if stats_idx is None:
            if hasattr(self, "set_stats_info"):
                try:
                    self.set_stats_info(
                        f"Ctrl+click label={lab}{xyz_s}: not in MES table"
                    )
                except Exception:
                    pass
            print(f"[MES pick] label={lab} not in object_stats source={source}")
            return False

        visual_row = self._visual_row_for_stats_index(stats_idx)
        if visual_row is None:
            if hasattr(self, "set_stats_info"):
                try:
                    self.set_stats_info(
                        f"MES object label={lab} is filtered/hidden in table"
                    )
                except Exception:
                    pass
            return False

        if hasattr(self, "stats_tabs"):
            try:
                self.stats_tabs.setCurrentIndex(0)
            except Exception:
                pass

        multi = self.is_mes_multi_select()
        table = self.object_stats_table
        st = self.object_stats[stats_idx]
        bump = st.get("bump_id", st.get("Bump ID", lab))

        # Never clear on mouse-up — sticky both Multi ON and OFF
        self._mes_ctrl_pick_active = False
        self._mes_ctrl_pick_armed = False
        self._mes_selection_lock = False

        model = table.model()
        sm = table.selectionModel()
        if model is None or sm is None:
            return False
        left = model.index(int(visual_row), 0)
        right = model.index(int(visual_row), max(0, table.columnCount() - 1))
        if not left.isValid():
            return False
        sel = QItemSelection(left, right)

        # Match MES table Multi ON: MultiSelection row toggle accumulates
        # cyan highlights (see multi-on reference: many rows + many MPR/3D).
        # Critical: never call setCurrentCell in multi mode — Qt treats that as
        # ClearAndSelect and wipes the multi set down to one row.
        table.blockSignals(True)
        try:
            if multi:
                self._set_mes_table_selection_mode(True)
                already = self._mes_row_is_selected(table, visual_row)
                sm.select(
                    sel,
                    QItemSelectionModel.Toggle | QItemSelectionModel.Rows,
                )
                # Pin current without changing selection set (NoUpdate)
                sm.setCurrentIndex(left, QItemSelectionModel.NoUpdate)
                try:
                    # scroll only — do not setCurrentCell (clears multi)
                    table.scrollTo(left, QAbstractItemView.PositionAtCenter)
                except Exception:
                    pass
                action = "removed" if already else "added"
            else:
                # Single sticky: replace selection (same as table click)
                self._set_mes_table_selection_mode(False)
                sm.select(
                    sel,
                    QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
                )
                sm.setCurrentIndex(left, QItemSelectionModel.NoUpdate)
                try:
                    table.setCurrentCell(int(visual_row), 0)
                except Exception:
                    pass
                try:
                    table.scrollTo(left, QAbstractItemView.PositionAtCenter)
                except Exception:
                    pass
                action = "selected"
        finally:
            table.blockSignals(False)

        self._repaint_mes_table(table)
        # Rebuild multi cyan set from full table selection (parity MES→MPR)
        self._sync_mes_highlights_from_table(
            navigate=True,
            navigate_stat=st if action != "removed" else None,
        )

        n = self._mes_selected_visual_row_count()
        if hasattr(self, "set_stats_info"):
            try:
                src = str(source or "pick").upper()
                if multi:
                    self.set_stats_info(
                        f"Multi {action} Bump {bump}  label={lab}"
                        f"{xyz_s}  · {n} selected (sticky · {src})"
                    )
                else:
                    self.set_stats_info(
                        f"{src} pick → MES #{stats_idx + 1}  Bump {bump}  "
                        f"label={lab}{xyz_s}  (sticky)"
                    )
            except Exception:
                pass
        print(
            f"[MES pick] source={source} multi={multi} {action} "
            f"stats_idx={stats_idx} visual_row={visual_row} label={lab} "
            f"bump={bump} xyz={xyz} n={n} sticky"
        )
        return True

    def select_mes_object_from_mpr(self, orientation, pos):
        """Ctrl+click on MPR → MES row + highlight (sticky both ways).

        Rules (synced with table click):
          · Multi OFF: replace selection with this one bump — **kept after release**
          · Multi ON: toggle this bump in/out of the multi set — **all kept**

        Returns True if a MES object was hit, False on miss / no data.
        """
        if self.volume_data is None:
            return False
        if not getattr(self, "object_stats", None):
            if hasattr(self, "set_stats_info"):
                try:
                    self.set_stats_info("No MES objects — run measurement first")
                except Exception:
                    pass
            return False
        if getattr(self, "labeled_class1_data", None) is None:
            if hasattr(self, "set_stats_info"):
                try:
                    self.set_stats_info("No labeled mask — cannot pick MES from MPR")
                except Exception:
                    pass
            return False

        xyz = self._mpr_pick_volume_xyz(orientation, pos)
        if xyz is None:
            return False
        x, y, z = xyz
        lab = self._label_at_volume_xyz(x, y, z)
        if lab <= 0:
            if hasattr(self, "set_stats_info"):
                try:
                    self.set_stats_info(
                        f"Ctrl+click @ X{x} Y{y} Z{z}: no bump label"
                    )
                except Exception:
                    pass
            print(f"[MES pick] MPR miss at ({x},{y},{z})")
            return False
        return self._apply_mes_pick_label(lab, xyz=(x, y, z), source="mpr")

    def select_mes_object_from_3d(self, qt_pos):
        """Ctrl+click on 3D volume → MES row + highlight (same sticky rules as MPR)."""
        if self.volume_data is None:
            return False
        if not getattr(self, "object_stats", None):
            if hasattr(self, "set_stats_info"):
                try:
                    self.set_stats_info("No MES objects — run measurement first")
                except Exception:
                    pass
            return False
        if getattr(self, "labeled_class1_data", None) is None:
            if hasattr(self, "set_stats_info"):
                try:
                    self.set_stats_info("No labeled mask — cannot pick MES from 3D")
                except Exception:
                    pass
            return False

        xyz = self._3d_pick_volume_xyz(qt_pos)
        if xyz is None:
            if hasattr(self, "set_stats_info"):
                try:
                    self.set_stats_info("Ctrl+click 3D: no bump along ray")
                except Exception:
                    pass
            print("[MES pick] 3D ray miss")
            return False
        x, y, z = xyz
        lab = self._label_at_volume_xyz(x, y, z)
        if lab <= 0:
            if hasattr(self, "set_stats_info"):
                try:
                    self.set_stats_info(
                        f"Ctrl+click 3D @ X{x} Y{y} Z{z}: no bump label"
                    )
                except Exception:
                    pass
            return False
        return self._apply_mes_pick_label(lab, xyz=(x, y, z), source="3d")

    def release_mes_ctrl_pick(self):
        """Mouse-up after Ctrl+pick — selection is always sticky (no clear).

        Kept for call-site compatibility; only clears arm flags.
        """
        self._mes_ctrl_pick_armed = False
        self._mes_ctrl_pick_active = False
        self._mes_selection_lock = False

    def _select_mes_visual_row(self, visual_row):
        """Select one MES table row so both main + frozen panes show highlight."""
        table = getattr(self, "object_stats_table", None)
        if table is None or visual_row is None or visual_row < 0:
            return
        if visual_row >= table.rowCount():
            return

        model = table.model()
        sm = table.selectionModel()
        if model is None or sm is None:
            return

        ncols = max(1, table.columnCount())
        left = model.index(int(visual_row), 0)
        right = model.index(int(visual_row), ncols - 1)
        if not left.isValid():
            return

        table.setFocus(Qt.OtherFocusReason)
        sel = QItemSelection(left, right)
        sm.select(
            sel,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )
        table.setCurrentIndex(left)
        try:
            table.setCurrentCell(int(visual_row), 0)
        except Exception:
            pass
        table.scrollTo(left, QAbstractItemView.PositionAtCenter)
        self._repaint_mes_table(table)

    @staticmethod
    def _repaint_mes_table(table):
        if table is None:
            return
        try:
            table.viewport().update()
            frozen = getattr(table, "frozenTableView", None)
            if frozen is not None:
                frozen.viewport().update()
        except Exception:
            pass

    def _apply_mes_selection_from_stats_idx(self, stats_idx):
        """Set sticky highlight + navigate — same end state as clicking a MES row."""
        if stats_idx is None or not getattr(self, "object_stats", None):
            return
        try:
            stats_idx = int(stats_idx)
        except (TypeError, ValueError):
            return
        if stats_idx < 0 or stats_idx >= len(self.object_stats):
            return

        stat = self.object_stats[stats_idx]
        try:
            label = int(stat.get("label", 0))
        except (TypeError, ValueError):
            label = 0
        if label <= 0 and getattr(self, "labeled_class1_data", None) is not None:
            try:
                label = self._label_at_stat_centroid(stat)
                if label > 0:
                    stat["label"] = label
            except Exception:
                pass

        # Highlight for current pick (MES row stays; Ctrl+pick clears on release)
        self.selected_highlight_objects = [(1, label)] if label > 0 else []
        self._navigate_to_object_stat(stat)

        if self.volume_data is not None:
            for ori in ["axial", "coronal", "sagittal"]:
                try:
                    self.render_slice(ori, preserve_camera=True)
                except Exception:
                    pass
            try:
                self._update_mes_3d_highlight()
            except Exception:
                pass

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
        """Handle MES table selection → highlight + navigate MPR/3D.

        Sticky policy (same both directions):
          · Multi OFF: one row → one cyan highlight on MPR/3D (kept until change)
          · Multi ON: all selected rows → multi highlight set (kept until toggle/Clear)
        """
        selected_visual = self._mes_selected_visual_rows()

        # During Ctrl+pick hold (single mode), ignore empty flicker from focus steal
        if not selected_visual and getattr(self, "_mes_selection_lock", False):
            return

        self._sync_mes_highlights_from_table(navigate=True)
    
    def clear_object_selection(self):
        """Clear MES selection/highlights and B2B 3D gap overlay."""
        self._mes_selection_lock = False
        self._mes_ctrl_pick_active = False
        self._mes_sticky_stats_idx = None
        self._mes_sticky_visual_row = None
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
    
    
