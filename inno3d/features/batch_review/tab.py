from __future__ import annotations
# inno3d/features/batch_review/tab.py
# -----------------------------------------------------------------------
# BatchReviewTab — extracted from inno3d/tabs/batch_review.py
#
# Full BatchReviewTab QWidget: DB browser, KPIs, wafer/FOV maps,
# MPR replay, CSV export, context map panel integration.
# -----------------------------------------------------------------------

from inno3d.features.batch_review.map_stage import MapStage
from inno3d.features.batch_review.widgets import (
    build_wafer_context_from_db,
    WaferMapWidget, FovMapWidget,
    MiniBarChart,
)

"""
Batch Production Review tab — browse inspection DB, replay FOV results.

UI layout mirrors docs/mockups/batch_review_tab.html:
  left  = production browser (lot / wafer / FOV)
  center = wafer map + MES table + actions
  right  = analytics + run timeline
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
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
from inno3d.core.lot_foup import (
    format_lot_foup_display,
    format_slot_display,
    join_lot_foup,
    parse_date_folder_display,
    split_lot_foup,
)
from inno3d.core.product_labels import (
    BUMP_RESULTS,
    BUMPS_KPI,
    FOV_DONE_KPI,
    LIVE_REVIEW_NAV,
    LIVE_REVIEW_SUBTITLE,
    LIVE_REVIEW_TITLE,
    MES_HELPER,
    NG_FOV_KPI,
    OPEN_BUMP_STATS_CSV,
    YIELD_KPI,
)
from inno3d.core.styles import SemiconductorTheme
from inno3d.features.viewer.shared_widgets import (
    ExcelFilterHeader,
    _category_item,
    _numeric_item,
)
from inno3d.core.wafer_context import WaferContext


class LiveStrip(QFrame):
    """Operator summary: wafer scope, FOV progress (done/planned), last inspected site."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LinePulseLiveStrip")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 5, 12, 5)
        layout.setSpacing(1)
        self.scope_label = QLabel("● LIVE  ·  Waiting for inspection results")
        self.scope_label.setObjectName("LinePulseLiveScope")
        self.metrics_label = QLabel(
            "FOV 0/—  ░░░░░░░░░░  —%    Yield —   NG FOV 0   Last: —"
        )
        self.metrics_label.setObjectName("LinePulseLiveMetrics")
        layout.addWidget(self.scope_label)
        layout.addWidget(self.metrics_label)

    def set_snapshot(
        self,
        progress: Dict[str, Any],
        focus: Optional[Dict[str, Any]] = None,
    ) -> None:
        progress = progress or {}
        focus = focus or {}
        n_fov = int(progress.get("n_fov") or 0)
        n_expected = int(progress.get("n_expected") or 0)
        n_ng = int(progress.get("n_ng") or 0)
        yield_pct = float(progress.get("yield_pct") or 0)
        progress_pct = float(progress.get("progress_pct") or 0)
        if progress_pct <= 0 and n_expected > 0:
            progress_pct = 100.0 * n_fov / n_expected
        # Clamp display to 100% if n_fov > n_expected (rare, but show raw counts in text)
        progress_pct = min(100.0, progress_pct)

        lot = str(focus.get("lot_id") or "")
        foup = str(focus.get("foup_id") or "")
        wafer = str(focus.get("wafer_id") or "")
        try:
            slot = int(focus.get("slot") or 0)
        except (TypeError, ValueError):
            slot = 0
        if not (lot or foup or wafer):
            parts = str(focus.get("wafer_key") or "").split("|")
            if len(parts) >= 3:
                wafer = parts[2]
            if slot <= 0 and len(parts) >= 4:
                from inno3d.core.lot_foup import parse_slot_token

                slot = parse_slot_token(parts[3])
        slot_disp = format_slot_display(slot)
        scope = "  ·  ".join(
            part
            for part in (
                f"LOT {lot}" if lot else "",
                f"FOUP {foup}" if foup else "",
                f"WAFER {wafer}" if wafer else "",
                slot_disp,
            )
            if part
        )
        self.scope_label.setText(f"● LIVE  {scope or '· Current wafer'}")

        bar_n = min(10, max(0, round(progress_pct / 10)))
        bar = "█" * bar_n + "░" * (10 - bar_n)
        fov_text = f"{n_fov}/{n_expected}" if n_expected else str(n_fov)
        col = focus.get("chip_col")
        row = focus.get("chip_row")
        point = focus.get("fov_index")
        judgment = str(focus.get("judgment") or "").upper()
        if col is not None and row is not None and point:
            last = f"Chip {col},{row} P{point} {judgment or '—'}"
        else:
            last = "—"
        self.metrics_label.setText(
            f"FOV {fov_text}  {bar}  {progress_pct:.1f}%    "
            f"Yield {yield_pct:.1f}%   NG FOV {n_ng}   Last: {last}"
        )


class BatchReviewTab(QWidget):
    """Production batch review against SQLite inspection catalog."""

    open_in_viewer = pyqtSignal(dict)  # run payload for future deep-link

    def __init__(self, parent=None, db: Optional[InspectionDB] = None):
        super().__init__(parent)
        self.db = db or get_db()
        self._current_wafer_key = ""
        self._current_run_id = ""
        self._runs_cache: List[Dict[str, Any]] = []
        self._wafer_runs_cache: List[Dict[str, Any]] = []
        self._wafer_runs_cache_key = ""
        self._last_scope_fingerprint: Optional[tuple] = None
        self._lot_rows: List[Dict[str, Any]] = []
        self._pending_live_payload: Dict[str, Any] = {}
        self._pending_run_id = ""
        self._pending_wafer_key = ""
        self._live_refresh_timer = QTimer(self)
        self._live_refresh_timer.setSingleShot(True)
        self._live_refresh_timer.setInterval(500)
        self._live_refresh_timer.timeout.connect(self._refresh_live_scope)
        self._live_poll_timer = QTimer(self)
        self._live_poll_timer.setInterval(5000)
        self._live_poll_timer.timeout.connect(self._poll_live_scope)
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
        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        title = QLabel(LIVE_REVIEW_TITLE)
        title.setObjectName("BatchReviewTitle")
        title_col.addWidget(title)
        subtitle = QLabel(LIVE_REVIEW_SUBTITLE)
        subtitle.setObjectName("BatchReviewMuted")
        title_col.addWidget(subtitle)
        hl.addLayout(title_col)
        self.db_path_label = QLabel("")
        self.db_path_label.setObjectName("BatchReviewMuted")
        hl.addStretch()
        hl.addWidget(self.db_path_label)
        root.addWidget(header)

        live_row = QFrame()
        live_row.setObjectName("LinePulseLiveRow")
        live_layout = QHBoxLayout(live_row)
        live_layout.setContentsMargins(0, 0, 0, 0)
        self.live_strip = LiveStrip()
        self.live_auto_refresh = QCheckBox("Follow live inspection")
        self.live_auto_refresh.setChecked(True)
        self.live_auto_refresh.setToolTip(
            "When Online finishes a FOV:\n"
            "· Jump to that wafer and select the new result\n"
            "· Refresh progress (done / planned FOV)\n"
            "· Poll every 5 s while this tab is open (backup)"
        )
        self.live_auto_refresh.toggled.connect(self._sync_live_poll_timer)
        live_layout.addWidget(self.live_strip, 1)
        live_layout.addWidget(self.live_auto_refresh)
        root.addWidget(live_row)

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

        self.date_combo = QComboBox()
        self.date_combo.currentIndexChanged.connect(self._on_date_changed)
        left_l.addWidget(self._field("Date", self.date_combo))

        self.lot_combo = QComboBox()
        self.lot_combo.currentIndexChanged.connect(self._on_lot_filter_changed)
        left_l.addWidget(self._field("Lot", self.lot_combo))

        self.foup_combo = QComboBox()
        self.foup_combo.currentIndexChanged.connect(self._on_foup_changed)
        left_l.addWidget(self._field("FOUP", self.foup_combo))

        self.wafer_combo = QComboBox()
        self.wafer_combo.currentIndexChanged.connect(self._on_wafer_changed)
        left_l.addWidget(self._field("Wafer · SLOT", self.wafer_combo))

        self.judge_combo = QComboBox()
        self.judge_combo.addItem("All judgments", "")
        self.judge_combo.addItem("OK only", "OK")
        self.judge_combo.addItem("NG only", "NG")
        self.judge_combo.addItem("ERROR", "ERROR")
        self.judge_combo.currentIndexChanged.connect(self._reload_fov_list)
        left_l.addWidget(self._field("Judgment filter", self.judge_combo))

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Chip row,col / path / FOV…")
        self.search_edit.returnPressed.connect(self._reload_fov_list)
        left_l.addWidget(self._field("Search", self.search_edit))

        # KPIs
        kpi_grid = QGridLayout()
        self.kpi_fov = self._kpi_card("0", FOV_DONE_KPI)
        self.kpi_fov.setToolTip(
            "FOV done / planned\n"
            "Planned = dies inside wafer circle \u00d7 9 FOV points (P1\u2013P9).\n"
            "Counts latest result only per site (re-inspect does not inflate)."
        )
        self.kpi_yield = self._kpi_card("\u2014", YIELD_KPI)
        self.kpi_ng = self._kpi_card("0", NG_FOV_KPI)
        self.kpi_obj = self._kpi_card("0", BUMPS_KPI)
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

        self.breadcrumb = QLabel(f"{LIVE_REVIEW_NAV}  ›  (select a wafer / FOV)")
        self.breadcrumb.setObjectName("BatchReviewBreadcrumb")
        self.breadcrumb.setWordWrap(True)
        cl.addWidget(self.breadcrumb)

        # Row 1: Map Stage — one wafer canvas; zoom-in shows FOV P1–P9 in-die.
        self.map_stage = MapStage()
        self.map_stage.setMinimumHeight(300)
        self.map_stage.dieSelected.connect(self._on_die_clicked)
        self.map_stage.fovSelected.connect(self._on_fov_map_clicked)
        self.map_stage.levelChanged.connect(self._on_map_level_changed)
        # Aliases for any leftover references / chip status text
        self.wafer_map = self.map_stage.wafer_map
        self.fov_map = self.map_stage.fov_map
        self.chip_label = self.map_stage.chip_label
        self._wafer_ctx: Optional[WaferContext] = None
        cl.addWidget(self.map_stage, 4)

        # Row 2: FOV run table (one row per site — latest result)
        cl.addWidget(self._section_label("FOV SITES · LATEST PER POINT"))
        self.fov_table = QTableWidget(0, 8)
        self.fov_table.setHorizontalHeaderLabels(
            ["Run", "Chip", "FOV", "Judgment", "Bumps", "NG", "Total s", "Finished"]
        )
        self.fov_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.fov_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.fov_table.setAlternatingRowColors(True)
        self.fov_table.verticalHeader().setVisible(False)
        self.fov_table.horizontalHeader().setStretchLastSection(True)
        self.fov_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.fov_table.itemSelectionChanged.connect(self._on_fov_selected)
        self.fov_table.setMinimumHeight(120)
        cl.addWidget(self.fov_table, 2)

        cl.addWidget(self._section_label(BUMP_RESULTS))
        mes_helper = QLabel(MES_HELPER)
        mes_helper.setObjectName("BatchReviewMuted")
        mes_helper.setWordWrap(True)
        cl.addWidget(mes_helper)
        self.mes_table = QTableWidget(0, 10)
        self.mes_filter_header = ExcelFilterHeader(self.mes_table)
        self.mes_table.setHorizontalHeader(self.mes_filter_header)
        self.mes_filter_header.filter_applied.connect(self._apply_mes_excel_filters)
        self.mes_filter_header.sortIndicatorChanged.connect(
            lambda _idx, _order: self._apply_mes_excel_filters()
        )
        self.mes_table.setHorizontalHeaderLabels(
            ["#", "Layer", "Bump ID", "B.H", "B.V", "V.V", "Ratio%", "Judgment", "Gap X", "Gap Y"]
        )
        self.mes_table.setAlternatingRowColors(True)
        self.mes_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mes_table.verticalHeader().setVisible(False)
        self.mes_table.horizontalHeader().setStretchLastSection(True)
        self.mes_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.mes_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.mes_table.setSortingEnabled(True)
        self.mes_table.horizontalHeader().setSortIndicatorShown(True)
        self.mes_table.setMinimumHeight(140)
        self.mes_table.setToolTip(
            "BUMP RESULTS — click column header to sort\n"
            "Click funnel icon for Excel-style value filter"
        )
        cl.addWidget(self.mes_table, 3)

        actions = QHBoxLayout()
        self.btn_open_results = QPushButton("Open Results folder")
        self.btn_open_results.clicked.connect(self._open_results)
        self.btn_open_csv = QPushButton(OPEN_BUMP_STATS_CSV)
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
        actions.addWidget(self.btn_open_results)
        actions.addWidget(self.btn_open_csv)
        actions.addWidget(self.btn_export)
        actions.addWidget(self.btn_emit_viewer)
        actions.addStretch()
        note = QLabel(
            "Open full Viewer is the only 3D path · volume + mask + MES + B2B + mapping"
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
            QFrame#LinePulseLiveStrip {{
                background: {SemiconductorTheme.BG_PANEL};
                border-bottom: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QLabel#LinePulseLiveScope {{
                color: {SemiconductorTheme.ACCENT_SUCCESS};
                font-size: 9pt; font-weight: 700;
            }}
            QLabel#LinePulseLiveMetrics {{
                color: {SemiconductorTheme.TEXT_SECONDARY};
                font-family: Consolas, monospace; font-size: 9pt;
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
            self._reload_date_lot_foup_filters()
            self._reload_wafer_list()
            self._wafer_runs_cache_key = ""
            self._reload_fov_list()
            self._reload_kpis()
            self._reload_charts()
            self._update_live_strip()
            self._remember_scope_fingerprint()
        finally:
            self.setUpdatesEnabled(True)

    def _scope_fingerprint(self) -> tuple:
        """Cheap signature — detects new/changed latest-per-FOV results."""
        p = self.db.wafer_progress(self._current_wafer_key)
        return (
            self._current_wafer_key or "",
            int(p.get("n_fov") or 0),
            int(p.get("n_ok") or 0),
            int(p.get("n_ng") or 0),
            int(p.get("n_bump_ng") or 0),
            str(self._pending_run_id or ""),
        )

    def _remember_scope_fingerprint(self) -> None:
        self._last_scope_fingerprint = self._scope_fingerprint()

    def _ensure_wafer_runs_cache(self, force: bool = False) -> None:
        key = self._current_wafer_key or ""
        if not force and key == self._wafer_runs_cache_key and self._wafer_runs_cache:
            return
        self._wafer_runs_cache_key = key
        if not key:
            self._wafer_runs_cache = []
            return
        self._wafer_runs_cache = self.db.list_fov_runs(
            wafer_key=key, limit=5000, latest_only=True
        )

    def on_run_catalogued(self, payload: Optional[Dict[str, Any]] = None) -> None:
        """Online completion → follow wafer, refresh, select the new FOV."""
        if not self.live_auto_refresh.isChecked():
            return
        data = dict(payload or {})
        self._pending_live_payload = data
        self._pending_run_id = str(data.get("run_id") or "")
        self._pending_wafer_key = str(data.get("wafer_key") or "")
        self._live_refresh_timer.start()

    def _refresh_live_scope(self) -> None:
        """Follow the latest Online FOV, or quietly refresh when poll detects change."""
        if not self.live_auto_refresh.isChecked():
            return
        payload = dict(self._pending_live_payload or {})
        run_id = str(payload.get("run_id") or self._pending_run_id or "")
        wafer_key = str(payload.get("wafer_key") or self._pending_wafer_key or "")
        # Only a real Online catalog event may jump wafer / die / selected FOV.
        follow_event = bool(run_id and wafer_key)

        self.setUpdatesEnabled(False)
        try:
            if follow_event:
                self._reload_date_lot_foup_filters()
                self._focus_wafer_key(wafer_key)
                chip_col = int(payload.get("chip_col") or 0)
                chip_row = int(payload.get("chip_row") or 0)
                if chip_col > 0 and chip_row > 0:
                    self._die_filter = (chip_col, chip_row)
                    self.search_edit.blockSignals(True)
                    self.search_edit.setText(f"{chip_row},{chip_col}")
                    self.search_edit.blockSignals(False)
                # Reset judgment filter to "All" so a newly catalogued NG/OK
                # FOV is never hidden by an active filter before the follow-jump.
                self.judge_combo.blockSignals(True)
                self.judge_combo.setCurrentIndex(0)  # index 0 = "All judgments", data=""
                self.judge_combo.blockSignals(False)
            elif not self._current_wafer_key:
                self._reload_date_lot_foup_filters()
                if self.wafer_combo.count():
                    self._current_wafer_key = self.wafer_combo.currentData() or ""

            self._wafer_runs_cache_key = ""
            self._ensure_wafer_runs_cache(force=True)
            self._reload_wafer_list()
            self._reload_fov_list()
            self._reload_kpis()
            self._reload_charts()
            self._update_live_strip(payload if follow_event else None)
            if follow_event:
                self._select_run_by_id(run_id, load_detail=True)
                # Consume one-shot follow target so poll won't re-jump.
                self._pending_run_id = ""
                self._pending_wafer_key = ""
            self._remember_scope_fingerprint()
        finally:
            self.setUpdatesEnabled(True)

    def _focus_wafer_key(self, wafer_key: str) -> bool:
        """Select Date / Lot / FOUP / Wafer combos for a canonical wafer_key."""
        from inno3d.core.inspection_db import normalize_wafer_key
        from inno3d.core.lot_foup import split_lot_foup

        key = normalize_wafer_key(str(wafer_key or ""))
        if not key:
            return False
        parts = key.split("|")
        date_folder = parts[0] if len(parts) > 0 else ""
        lot_foup = parts[1] if len(parts) > 1 else ""
        lot_id, foup_id = split_lot_foup(lot_foup)

        combos = (self.date_combo, self.lot_combo, self.foup_combo, self.wafer_combo)
        for c in combos:
            c.blockSignals(True)
        try:
            if date_folder:
                self._restore_combo(self.date_combo, date_folder)
            else:
                restored = False
                for i in range(self.date_combo.count()):
                    if self.date_combo.itemData(i) == "__EMPTY__":
                        self.date_combo.setCurrentIndex(i)
                        restored = True
                        break
                if not restored:
                    self.date_combo.setCurrentIndex(0)

            self._reload_lot_combo(preserve_lot=lot_id, preserve_foup=foup_id)
            self._reload_wafer_combo()
            self._restore_combo(self.wafer_combo, key)
            self._current_wafer_key = self.wafer_combo.currentData() or key
        finally:
            for c in combos:
                c.blockSignals(False)
        return True

    def _select_run_by_id(self, run_id: str, load_detail: bool = True) -> bool:
        rid = str(run_id or "")
        if not rid:
            return False
        for i, run in enumerate(self._runs_cache):
            if str(run.get("run_id") or "") != rid:
                continue
            # Avoid double _load_run_detail via itemSelectionChanged.
            self.fov_table.blockSignals(True)
            try:
                self.fov_table.selectRow(i)
            finally:
                self.fov_table.blockSignals(False)
            if load_detail:
                self._load_run_detail(rid)
            return True
        return False

    def _update_live_strip(self, focus: Optional[Dict[str, Any]] = None) -> None:
        focus_data = dict(focus if focus is not None else self._pending_live_payload)
        if self._current_wafer_key and not focus_data.get("wafer_key"):
            focus_data.setdefault("wafer_key", self._current_wafer_key)
        self.live_strip.set_snapshot(
            self.db.wafer_progress(self._current_wafer_key),
            focus_data,
        )

    def _poll_live_scope(self) -> None:
        if not (self.live_auto_refresh.isChecked() and self.isVisible()):
            self._sync_live_poll_timer()
            return
        fp = self._scope_fingerprint()
        if fp == self._last_scope_fingerprint:
            return
        self._refresh_live_scope()

    def _sync_live_poll_timer(self, _checked: Optional[bool] = None) -> None:
        if self.live_auto_refresh.isChecked() and self.isVisible():
            if not self._live_poll_timer.isActive():
                self._live_poll_timer.start()
        else:
            self._live_poll_timer.stop()

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_live_poll_timer()

    def hideEvent(self, event):
        self._live_poll_timer.stop()
        self._live_refresh_timer.stop()
        super().hideEvent(event)

    @staticmethod
    def _restore_combo(combo: QComboBox, data: Any) -> None:
        if data is None:
            return
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                combo.setCurrentIndex(i)
                return

    def _selected_date_folder(self):
        """Return date filter: ``None`` = all, ``\"\"`` = legacy empty, else ``yy_mm_dd``."""
        data = self.date_combo.currentData() if self.date_combo.count() else None
        if data is None:
            return None
        if data == "__EMPTY__":
            return ""
        return str(data)

    def _rows_for_date(self, date_folder) -> List[Dict[str, Any]]:
        rows = self._lot_rows
        if date_folder is None:
            return rows
        return [r for r in rows if (r.get("date_folder") or "") == (date_folder or "")]

    def _reload_date_lot_foup_filters(self):
        cur_date = self.date_combo.currentData() if self.date_combo.count() else None
        cur_lot = self.lot_combo.currentData() if self.lot_combo.count() else ""
        cur_foup = self.foup_combo.currentData() if self.foup_combo.count() else ""

        self._lot_rows = self.db.list_lot_foup_parts()

        self.date_combo.blockSignals(True)
        self.date_combo.clear()
        # None = All dates; "__EMPTY__" = legacy Online rows with blank date_folder
        self.date_combo.addItem("All dates", None)
        raw_dates = {r.get("date_folder") or "" for r in self._lot_rows}
        if "" in raw_dates:
            self.date_combo.addItem("(no date · legacy)", "__EMPTY__")
        for d in sorted((x for x in raw_dates if x), reverse=True):
            self.date_combo.addItem(parse_date_folder_display(d), d)
        self._restore_combo(self.date_combo, cur_date)
        self.date_combo.blockSignals(False)

        self._reload_lot_combo(preserve_lot=cur_lot, preserve_foup=cur_foup)
        self._reload_wafer_combo()

    def _reload_lot_combo(self, preserve_lot: Any = None, preserve_foup: Any = None):
        date_folder = self._selected_date_folder()
        cur_lot = preserve_lot if preserve_lot is not None else (self.lot_combo.currentData() or "")

        self.lot_combo.blockSignals(True)
        self.lot_combo.clear()
        self.lot_combo.addItem("All lots", "")
        lots = sorted({r.get("lot_id") or "" for r in self._rows_for_date(date_folder) if r.get("lot_id")})
        for lot in lots:
            self.lot_combo.addItem(lot, lot)
        self._restore_combo(self.lot_combo, cur_lot)
        self.lot_combo.blockSignals(False)

        self._reload_foup_combo(preserve_foup=preserve_foup)

    def _reload_foup_combo(self, preserve_foup: Any = None):
        date_folder = self._selected_date_folder()
        lot_id = self.lot_combo.currentData() or ""
        cur_foup = preserve_foup if preserve_foup is not None else (self.foup_combo.currentData() or "")

        rows = self._rows_for_date(date_folder)
        if lot_id:
            rows = [r for r in rows if (r.get("lot_id") or "") == lot_id]

        self.foup_combo.blockSignals(True)
        self.foup_combo.clear()
        self.foup_combo.addItem("All FOUPs", "")
        foups = sorted({r.get("foup_id") or "" for r in rows if r.get("foup_id")})
        for foup in foups:
            self.foup_combo.addItem(foup, foup)
        self._restore_combo(self.foup_combo, cur_foup)
        self.foup_combo.blockSignals(False)

    def _current_lot_foup_id(self) -> str:
        lot_id = self.lot_combo.currentData() or ""
        foup_id = self.foup_combo.currentData() or ""
        if lot_id and foup_id:
            return join_lot_foup(lot_id, foup_id)
        return ""

    def _list_filtered_wafers(self) -> List[Dict[str, Any]]:
        date_sel = self._selected_date_folder()  # None | "" | yy_mm_dd
        lot_id = self.lot_combo.currentData() or ""
        foup_id = self.foup_combo.currentData() or ""
        lot_foup_id = self._current_lot_foup_id()
        # DB helper: empty date_folder arg means "do not filter by date"
        if date_sel:
            wafers = self.db.list_wafers(date_folder=date_sel, lot_foup_id=lot_foup_id or "")
        else:
            wafers = self.db.list_wafers(date_folder="", lot_foup_id=lot_foup_id or "")
            if date_sel == "":
                wafers = [w for w in wafers if not (w.get("date_folder") or "")]
        if not lot_foup_id:
            if lot_id:
                wafers = [
                    w for w in wafers
                    if split_lot_foup(w.get("lot_foup_id") or "")[0] == lot_id
                ]
            if foup_id:
                wafers = [
                    w for w in wafers
                    if split_lot_foup(w.get("lot_foup_id") or "")[1] == foup_id
                ]
        return wafers

    def _reload_wafer_combo(self):
        self.wafer_combo.blockSignals(True)
        self.wafer_combo.clear()
        wafers = self._list_filtered_wafers()
        for w in wafers:
            y = w.get("yield_pct") or 0
            slot_disp = format_slot_display(int(w.get("slot") or 0))
            wafer_label = w.get("wafer_id") or "?"
            if slot_disp:
                wafer_label = f"{wafer_label} · {slot_disp}"
            n_done = int(w.get("n_fov") or 0)
            n_plan = int(w.get("n_expected") or 0)
            fov_disp = f"{n_done}/{n_plan}" if n_plan else str(n_done)
            label = f"{wafer_label} · FOV {fov_disp} · Yield {y:.1f}%"
            self.wafer_combo.addItem(label, w.get("wafer_key"))
        self.wafer_combo.blockSignals(False)
        if self.wafer_combo.count():
            self._current_wafer_key = self.wafer_combo.currentData() or ""
        else:
            self._current_wafer_key = ""

    def _reload_wafer_list(self):
        self.wafer_list.clear()
        for w in self._list_filtered_wafers():
            y = w.get("yield_pct") or 0
            _, foup_id = split_lot_foup(w.get("lot_foup_id") or "")
            foup_disp = format_lot_foup_display("", foup_id)
            slot_disp = format_slot_display(int(w.get("slot") or 0))
            last = w.get("last_run") or ""
            updated = str(last)[:19] if last else "—"
            meta_bits = [p for p in (foup_disp if foup_disp != "—" else "", slot_disp) if p]
            meta = " · ".join(meta_bits + [f"updated {updated}"])
            wafer_title = w.get("wafer_id") or "?"
            if slot_disp:
                wafer_title = f"{wafer_title} · {slot_disp}"
            n_done = int(w.get("n_fov") or 0)
            n_plan = int(w.get("n_expected") or 0)
            fov_disp = f"{n_done}/{n_plan}" if n_plan else str(n_done)
            text = (
                f"{wafer_title}\n"
                f"FOV {fov_disp} · Yield {y:.1f}% · NG {w.get('n_ng', 0)}\n"
                f"{meta}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, w.get("wafer_key"))
            if w.get("wafer_key") == self._current_wafer_key:
                item.setSelected(True)
            self.wafer_list.addItem(item)

    def _reload_kpis(self):
        p = self.db.wafer_progress(self._current_wafer_key)
        n = int(p.get("n_fov") or 0)
        expected = int(p.get("n_expected") or 0)
        fov_disp = f"{n}/{expected}" if expected else str(n)
        self._set_kpi(self.kpi_fov, fov_disp)
        self._set_kpi(self.kpi_yield, f"{p.get('yield_pct', 0):.1f}%", ok_color=True)
        self._set_kpi(self.kpi_ng, str(p.get("n_ng", 0)), ng_color=True)
        n_bump = int(p.get("n_bump_ng") or 0)
        self._set_kpi(self.kpi_obj, str(n_bump), ng_color=bool(n_bump))

    def _reload_fov_list(self):
        judge = self.judge_combo.currentData() or ""
        search = self.search_edit.text().strip()
        die = getattr(self, "_die_filter", None)
        self._ensure_wafer_runs_cache()
        runs = list(self._wafer_runs_cache)
        if judge:
            runs = [x for x in runs if str(x.get("judgment") or "").upper() == str(judge).upper()]
        if search:
            q = search.lower()
            runs = [
                x for x in runs
                if q in f"{x.get('chip_row')},{x.get('chip_col')}".lower()
                or q in str(x.get("fov_folder") or "").lower()
                or q in str(x.get("results_dir") or "").lower()
                or q in str(x.get("run_id") or "").lower()
            ]
        if die:
            c, r = die
            runs = [x for x in runs if int(x.get("chip_col") or 0) == c and int(x.get("chip_row") or 0) == r]
        self._runs_cache = runs
        self.fov_table.setRowCount(len(runs))
        for i, run in enumerate(runs):
            cells = [
                str(run.get("run_id") or "")[:10],
                f"({run.get('chip_row')},{run.get('chip_col')})",
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
                f"{LIVE_REVIEW_NAV}  ›  {crumb or self._current_wafer_key or '—'}  ·  "
                f"{len(runs)} FOV run(s)  ·  yield {ctx.yield_pct:.1f}%"
            )
        else:
            self.breadcrumb.setText(
                f"{LIVE_REVIEW_NAV}  ›  {self._current_wafer_key or '—'}  ·  {len(runs)} FOV run(s)"
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

    def _on_date_changed(self):
        self._reload_lot_combo()
        self._on_production_filter_changed()

    def _on_lot_filter_changed(self):
        self._reload_foup_combo()
        self._on_production_filter_changed()

    def _on_foup_changed(self):
        self._on_production_filter_changed()

    def _on_production_filter_changed(self):
        self._reload_wafer_combo()
        self._reload_wafer_list()
        self._die_filter = None
        self._wafer_runs_cache_key = ""
        self.map_stage.set_level(0)
        self._reload_fov_list()
        self._reload_kpis()
        self._reload_charts()
        self._update_live_strip({})

    def _on_wafer_changed(self):
        self._current_wafer_key = self.wafer_combo.currentData() or ""
        self._die_filter = None
        self._wafer_runs_cache_key = ""
        self.map_stage.set_level(0)
        self._reload_wafer_list()
        self._reload_fov_list()
        self._reload_kpis()
        self._reload_charts()
        self._update_live_strip({})

    def _on_wafer_list_clicked(self, item: QListWidgetItem):
        key = item.data(Qt.UserRole)
        if not key:
            return
        self._current_wafer_key = key
        self._wafer_runs_cache_key = ""
        for i in range(self.wafer_combo.count()):
            if self.wafer_combo.itemData(i) == key:
                self.wafer_combo.blockSignals(True)
                self.wafer_combo.setCurrentIndex(i)
                self.wafer_combo.blockSignals(False)
                break
        self._die_filter = None
        self.map_stage.set_level(0)
        self._reload_fov_list()
        self._reload_kpis()
        self._reload_charts()
        self._update_live_strip({})

    def _refresh_context_maps(
        self,
        selected_col: int = 0,
        selected_row: int = 0,
        selected_fov: int = 0,
    ):
        """Push DB-derived WaferContext into MapStage (Online-identical canvases)."""
        die = getattr(self, "_die_filter", None)
        if selected_col <= 0 and die:
            selected_col, selected_row = die
        ctx = build_wafer_context_from_db(
            self.db,
            self._current_wafer_key or "",
            selected_col=selected_col,
            selected_row=selected_row,
            selected_fov=selected_fov or 5,
            runs=self._wafer_runs_cache if self._wafer_runs_cache_key == (self._current_wafer_key or "") else None,
        )
        self._wafer_ctx = ctx
        self.map_stage.set_context(ctx)
        # Do not change zoom here — die click / Fit wafer own the view.
        if not self._current_wafer_key and not (ctx and ctx.selected_col > 0):
            self.chip_label.setText("Select a wafer, then a die")

    def _on_die_clicked(self, col: int, row: int):
        self._die_filter = (col, row)
        self.search_edit.setText(f"{row},{col}")
        self._refresh_context_maps(selected_col=col, selected_row=row, selected_fov=5)
        self.map_stage.set_level(1)
        self._reload_fov_list()

    def _on_fov_map_clicked(self, col: int, row: int, fov_index: int):
        """Select latest run for this FOV point (in-die P1–P9 click)."""
        die_changed = getattr(self, "_die_filter", None) != (int(col), int(row))
        self._die_filter = (int(col), int(row))
        self.search_edit.setText(f"{row},{col}")
        if self._wafer_ctx is not None:
            self._wafer_ctx.select_chip(int(col), int(row))
            self._wafer_ctx.select_fov(int(fov_index))
            self.map_stage.set_context(self._wafer_ctx)
        if die_changed:
            self._reload_fov_list()
        for i, run in enumerate(self._runs_cache):
            if int(run.get("fov_index") or 0) != int(fov_index):
                continue
            if int(run.get("chip_col") or 0) != int(col) or int(run.get("chip_row") or 0) != int(row):
                continue
            self.fov_table.selectRow(i)
            rid = run.get("run_id")
            if rid:
                self._load_run_detail(rid)
            return

    def _on_map_level_changed(self, level: int):
        """Back / empty click clears die filter so FOV table returns to wafer scope."""
        if int(level) == 0 and getattr(self, "_die_filter", None):
            self._die_filter = None
            txt = (self.search_edit.text() or "").strip()
            if "," in txt and all(p.strip().lstrip("-").isdigit() for p in txt.split(",", 1)):
                self.search_edit.blockSignals(True)
                self.search_edit.clear()
                self.search_edit.blockSignals(False)
            self._refresh_context_maps(selected_col=0, selected_row=0, selected_fov=0)
            self._reload_fov_list()

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
        self._clear_mes_excel_filters()
        self.mes_table.setRowCount(0)
        if not run:
            self.detail_label.setText("No run selected")
            return

        date_disp = parse_date_folder_display(run.get("date_folder") or "")
        lot_id, foup_id = split_lot_foup(run.get("lot_foup_id") or "")
        lot_foup_disp = format_lot_foup_display(lot_id, foup_id)
        slot_disp = format_slot_display(int(run.get("slot") or 0))
        wafer_disp = str(run.get("wafer_id") or "—")
        if slot_disp:
            wafer_disp = f"{wafer_disp} · {slot_disp}"
        # Qt rich text
        self.breadcrumb.setTextFormat(Qt.RichText)
        self.breadcrumb.setText(
            f"{LIVE_REVIEW_NAV} &nbsp;›&nbsp; {date_disp} &nbsp;›&nbsp; "
            f"{lot_foup_disp} &nbsp;›&nbsp; <b>{wafer_disp}</b> "
            f"&nbsp;›&nbsp; Chip (R{run.get('chip_row')},C{run.get('chip_col')}) "
            f"&nbsp;›&nbsp; <b>P{run.get('fov_index') or '?'}</b> "
            f"&nbsp;·&nbsp; run_id <span style='color:#22d3ee'>{run_id}</span> "
            f"&nbsp;·&nbsp; {run.get('config_path') or ''}"
        )

        objs = list(run.get("mes_objects") or [])
        # Layer 1 → 12 ascending (natural), then bump grid position
        def _mes_row_key(s: dict):
            layer = str(s.get("layer_name") or "")
            layer_key = [
                int(p) if p.isdigit() else p.lower()
                for p in re.split(r"(\d+)", layer)
            ]
            return (
                layer in ("", "N/A", "ALL"),
                layer_key,
                int(s.get("grid_row") or 0),
                int(s.get("grid_col") or 0),
                int(s.get("row_id") or 0),
            )

        objs.sort(key=_mes_row_key)
        table = self.mes_table
        was_sorting = table.isSortingEnabled()
        table.setSortingEnabled(False)
        table.setRowCount(len(objs))
        for i, s in enumerate(objs):
            ratio = float(s.get("ratio") or 0)
            ratio_pct = ratio * 100 if ratio < 1e8 else float("nan")
            jud = str(s.get("judgment") or "")
            try:
                row_id_num = float(s.get("row_id") or i + 1)
            except (TypeError, ValueError):
                row_id_num = float(i + 1)
            gr, gc = s.get("grid_row"), s.get("grid_col")
            try:
                bump_sort = float(gr) * 10000.0 + float(gc)
            except (TypeError, ValueError):
                bump_sort = float("-inf")
            soh = float(s.get("soh") or 0)
            c1 = float(s.get("c1_volume") or 0)
            c2 = float(s.get("c2_volume") or 0)
            px = float(s.get("pitch_x") or 0)
            py = float(s.get("pitch_y") or 0)
            items = [
                _numeric_item(str(s.get("row_id") or i + 1), row_id_num),
                _category_item(str(s.get("layer_name") or "")),
                _numeric_item(f"({gr},{gc})", bump_sort),
                _numeric_item(f"{soh:.2f}", soh),
                _numeric_item(f"{c1:.1f}", c1),
                _numeric_item(f"{c2:.1f}", c2),
                _numeric_item(
                    f"{ratio_pct:.2f}" if ratio_pct == ratio_pct else "N/A",
                    ratio_pct if ratio_pct == ratio_pct else float("inf"),
                ),
                _category_item(jud),
                _numeric_item(f"{px:.3f}", px),
                _numeric_item(f"{py:.3f}", py),
            ]
            for col, item in enumerate(items):
                if col == 7:
                    if jud.upper() == "NG":
                        item.setForeground(QColor(SemiconductorTheme.ACCENT_ERROR))
                    elif jud.upper() == "OK":
                        item.setForeground(QColor(SemiconductorTheme.ACCENT_SUCCESS))
                table.setItem(i, col, item)
        table.setSortingEnabled(was_sorting)
        self._apply_mes_excel_filters()

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

    def _clear_mes_excel_filters(self) -> None:
        header = getattr(self, "mes_filter_header", None)
        if header is None:
            return
        header._filters = {}
        try:
            header.viewport().update()
        except Exception:
            pass
        table = getattr(self, "mes_table", None)
        if table is None:
            return
        for row in range(table.rowCount()):
            table.setRowHidden(row, False)

    def _apply_mes_excel_filters(self, _col_index: int = -1) -> None:
        """Hide rows based on ExcelFilterHeader unchecked values."""
        header = getattr(self, "mes_filter_header", None)
        table = getattr(self, "mes_table", None)
        if header is None or table is None:
            return
        for row in range(table.rowCount()):
            table.setRowHidden(row, header.is_row_hidden(table, row))

    def select_run(self, run_id: str) -> bool:
        """Select and display a DB FOV run, including its wafer context."""
        run = self.db.get_run(run_id) if run_id else None
        if not run:
            return False

        wafer_key = str(run.get("wafer_key") or "")
        self.refresh_all()
        if wafer_key and wafer_key != self._current_wafer_key:
            for i in range(self.wafer_combo.count()):
                if self.wafer_combo.itemData(i) == wafer_key:
                    self.wafer_combo.setCurrentIndex(i)
                    break
            else:
                self._current_wafer_key = wafer_key
                self._reload_fov_list()
                self._reload_kpis()
                self._reload_charts()

        for row, cached_run in enumerate(self._runs_cache):
            if str(cached_run.get("run_id") or "") == str(run_id):
                self.fov_table.selectRow(row)
                break
        self._load_run_detail(str(run_id))
        return True

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
            "run_id", "date_folder", "lot_foup_id", "wafer_id", "slot",
            "chip_row", "chip_col", "fov_index", "judgment", "n_objects", "n_ng",
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
