# inno3d/app/main_window.py
# -----------------------------------------------------------------------
# MainWindow — extracted from main.py (Phase 6)
#
# MainWindow(OnlineModeMixin, QMainWindow): full application shell.
# Wires tabs (Viewer, Teaching, Online, Batch, Analysis, AI, Help).
# -----------------------------------------------------------------------

from inno3d.app.splash import (
    LuxuryLogoLabel,
    CustomTitleBar,
    HardwareTrackingWidget,
    app_ini_path,
    resolve_startup_theme,
)
from inno3d.modes.online import OnlineModeMixin
from inno3d.app.panels import PanelsMixin

from inno3d.core.resources import resource_path
from inno3d.core.styles import SemiconductorTheme, apply_theme, normalize_theme_name
from inno3d.core.ui_system import AnimatedNavButton, GlowProgressBar

from inno3d.tabs.ai import AI3DTab
from inno3d.tabs.teaching import SegmentationTab
from inno3d.tabs.analysis import AnalysisTab
from inno3d.tabs.viewer import MultiPlanarView
from inno3d.tabs.batch_review import BatchReviewTab
from inno3d.tabs.help import HelpTab

import psutil

from inno3d.infra.paths import app_config_ini_path as _app_config_ini_path
from inno3d.app.settings import get_theme as _settings_get_theme, set_theme as _settings_set_theme

import os
import sys
import configparser
from pathlib import Path
import logging
from datetime import datetime

# ==========================================
# Global Logging & Crash Tracking Setup
# ==========================================
from inno3d.infra.logging_setup import init_logging
init_logging(redirect_stdio=True)
# ==========================================

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import time


class OnlineToggleSwitch(QAbstractButton):
    """Compact left/right sliding switch (Offline ← → Online labels live beside it)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(44, 22)
        self._colors = {}
        self.apply_theme()

    def sizeHint(self):
        return QSize(44, 22)

    def apply_theme(self):
        is_light = SemiconductorTheme.is_light()
        self._colors = {
            "off_bg": QColor("#d2deec") if is_light else QColor("#1a2233"),
            "on_bg": QColor("#22b273") if is_light else QColor("#00c878"),
            "off_border": QColor("#7f98b4") if is_light else QColor("#43597a"),
            "on_border": QColor("#10d88a") if not is_light else QColor("#1a8f5f"),
            "knob": QColor("#f7fbff") if is_light else QColor("#ecf4ff"),
            "focus": QColor("#22aed1"),
        }
        self.update()

    def _track_rect(self):
        return QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)

    def _knob_rect(self, track_rect):
        margin = 2.0
        knob_d = max(12.0, track_rect.height() - margin * 2.0)
        x = track_rect.right() - margin - knob_d if self.isChecked() else track_rect.x() + margin
        return QRectF(float(x), float(track_rect.y() + margin), float(knob_d), float(knob_d))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        track_rect = self._track_rect()

        bg = self._colors["on_bg"] if self.isChecked() else self._colors["off_bg"]
        border = self._colors["on_border"] if self.isChecked() else self._colors["off_border"]
        if self.underMouse():
            border = self._colors["focus"]

        radius = track_rect.height() / 2.0
        p.setPen(QPen(border, 1.6))
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(track_rect, radius, radius)

        knob_rect = self._knob_rect(track_rect)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self._colors["knob"]))
        p.drawEllipse(knob_rect)

        if self.hasFocus():
            p.setPen(QPen(self._colors["focus"], 1.0, Qt.DotLine))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(track_rect.adjusted(-1, -1, 1, 1), radius + 1, radius + 1)

        p.end()


class MainWindow(OnlineModeMixin, PanelsMixin, QMainWindow):
    """Main Window - 3D Semiconductor Viewer"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("INNO3D INSPECTION v1.0.0")
        self.setGeometry(50, 50, 1600, 900)
        self.setMinimumSize(1200, 700)
        
        # Set App Icon
        app_icon_path = resource_path("assets/branding/company_logo_v1.png")
        if os.path.exists(app_icon_path):
            self.setWindowIcon(QIcon(app_icon_path))
            
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        
        # Online mode state
        self.online_server = None
        self.online_load_thread = None
        self.online_roi = None
        self.online_roi_active = False
        self.online_config_path = None
        self.online_processing = False
        self.online_config_folder = None
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main Layout: Vertical (Title Bar + Rest)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 1. Custom Title Bar
        self.title_bar = CustomTitleBar(self)
        main_layout.addWidget(self.title_bar)
        
        # 2. Content Layout: Horizontal (Sidebar + Stack)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        main_layout.addLayout(content_layout)

        
        # --- RIGHT CONTENT (Stacked Widget) ---
        self.stack = QStackedWidget()
        
        self.multiplanar_tab = MultiPlanarView()
        self.segmentation_tab = SegmentationTab()
        self.ai_3d_tab = AI3DTab()
        self.analysis_tab = AnalysisTab()
        self.batch_review_tab = BatchReviewTab()
        self.help_tab = HelpTab()
        
        self.stack.addWidget(self.multiplanar_tab)
        self.stack.addWidget(self.segmentation_tab)
        self.stack.addWidget(self.ai_3d_tab)
        self.stack.addWidget(self.analysis_tab)
        self.stack.addWidget(self.batch_review_tab)
        self.stack.addWidget(self.help_tab)
        
        # Connect segmentation signals
        self.segmentation_tab.inspection_done.connect(self._on_seg_inspection_done)
        self.segmentation_tab.apply_online_roi_signal.connect(self._on_apply_online_roi)
        self.help_tab.theme_changed.connect(self.on_theme_changed)
        self.help_tab.fdc_settings_changed.connect(self._on_fdc_settings_changed)
        self.help_tab.set_current_theme(SemiconductorTheme.CURRENT_THEME)
        if hasattr(self.batch_review_tab, "open_in_viewer"):
            self.batch_review_tab.open_in_viewer.connect(self._on_batch_review_open_run)
        if hasattr(self.analysis_tab, "open_in_line_pulse"):
            self.analysis_tab.open_in_line_pulse.connect(self._on_analysis_open_in_line_pulse)
        if hasattr(self.analysis_tab, "open_in_viewer"):
            self.analysis_tab.open_in_viewer.connect(self._on_batch_review_open_run)

        # Tabs Container (Sidebar + Content) uses QSplitter for resizing
        tabs_container = QSplitter(Qt.Horizontal)
        tabs_container.setHandleWidth(2)
        tabs_container.setStyleSheet(f"QSplitter::handle {{ background: {SemiconductorTheme.BORDER_DEFAULT}; }}")
        self.tabs_splitter = tabs_container
        
        # --- LEFT SIDEBAR ---
        # Scoped #AppSidebar rules only — unscoped background/border on the
        # parent used to interfere with child QPushButton hover painting so
        # left-rail buttons looked flatter than right VIEW TOOLS.
        sidebar = QWidget()
        sidebar.setObjectName("AppSidebar")
        sidebar.setMinimumWidth(150)
        sidebar.setMaximumWidth(400)
        sidebar.setStyleSheet(
            f"""
            QWidget#AppSidebar {{
                background-color: {SemiconductorTheme.BG_MEDIUM};
                border-right: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            """
        )
        self.sidebar_widget = sidebar
        
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 15, 0, 20)
        sidebar_layout.setSpacing(5)
        
        # --- App Title (Oxford Scientific Luxury Redesign) ---
        self.app_title = LuxuryLogoLabel()
        sidebar_layout.addWidget(self.app_title)

        lbl_hbm = QLabel()
        lbl_hbm.setAlignment(Qt.AlignCenter)
        lbl_hbm.setTextFormat(Qt.RichText)
        lbl_hbm.setText(
            f"<span style='color: {SemiconductorTheme.TEXT_SECONDARY};'>High-Bandwidth Memory</span>"
            f"<br>"
            f"<span style='color: {SemiconductorTheme.ACCENT_PRIMARY}; font-weight: 600; letter-spacing: 1.2px;'>(HBM)</span>"
        )
        lbl_hbm.setStyleSheet(
            "font-size: 8.5pt; letter-spacing: 0.8px; margin-top: 0px; margin-bottom: 2px;"
        )
        self.sidebar_hbm_label = lbl_hbm
        sidebar_layout.addWidget(lbl_hbm)
        
        lbl_info = QLabel("v0.0.1")
        lbl_info.setAlignment(Qt.AlignCenter)
        lbl_info.setStyleSheet(f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 8pt; margin-top: -12px; margin-bottom: 5px;")
        self.sidebar_version_label = lbl_info
        sidebar_layout.addWidget(lbl_info)
        
        # --- Online Toggle (above navigation) ---
        # Layout: Offline [switch] Online — knob left=Offline, right=Online
        # Outer frame gets cyan glow when Online is ON (#8 chrome)
        online_container = QWidget()
        online_container.setObjectName("onlineModeFrame")
        online_outer = QVBoxLayout(online_container)
        online_outer.setContentsMargins(10, 6, 10, 8)
        online_outer.setSpacing(0)

        online_row = QWidget()
        online_layout = QHBoxLayout(online_row)
        online_layout.setContentsMargins(5, 2, 5, 2)
        online_layout.setSpacing(6)

        self.online_offline_label = QLabel("Offline")
        self.online_online_label = QLabel("Online")
        for lbl in (self.online_offline_label, self.online_online_label):
            lbl.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)

        self.online_toggle = OnlineToggleSwitch()
        self.online_toggle.clicked.connect(self.toggle_online)
        self.online_toggle.toggled.connect(self._sync_online_toggle_hint)
        self._sync_online_toggle_hint(self.online_toggle.isChecked())

        online_layout.addStretch(1)
        online_layout.addWidget(self.online_offline_label)
        online_layout.addWidget(self.online_toggle)
        online_layout.addWidget(self.online_online_label)
        online_layout.addStretch(1)
        online_outer.addWidget(online_row)

        self.online_container = online_container
        self._apply_online_chrome(False)
        sidebar_layout.addWidget(online_container)

        # Online compact controls: expand full menu / collapse to free MPR space
        self._app_sidebar_expanded = True
        self._app_sidebar_user_expanded = False
        self._pre_online_sidebar_sizes = None

        sidebar_nav_row = QHBoxLayout()
        sidebar_nav_row.setContentsMargins(12, 0, 12, 0)
        sidebar_nav_row.setSpacing(6)

        self.sidebar_collapse_btn = QPushButton("Hide menu ‹")
        self.sidebar_collapse_btn.setCursor(Qt.PointingHandCursor)
        self.sidebar_collapse_btn.setToolTip(
            "Collapse left menu to free space for MPR / CONTEXT / Statistics.\n"
            "ONLINE toggle stays visible. Click « Menu to reopen."
        )
        self.sidebar_collapse_btn.setStyleSheet(f"""
            QPushButton {{
                background: {SemiconductorTheme.BG_LIGHT};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 5px;
                color: {SemiconductorTheme.TEXT_SECONDARY};
                font-size: 8pt; font-weight: 600;
                padding: 4px 8px; min-height: 24px;
            }}
            QPushButton:hover {{
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                border-color: {SemiconductorTheme.ACCENT_PRIMARY};
            }}
        """)
        self.sidebar_collapse_btn.clicked.connect(lambda: self.set_app_sidebar_expanded(False))
        self.sidebar_collapse_btn.setVisible(False)  # only while Online + expanded
        sidebar_nav_row.addWidget(self.sidebar_collapse_btn, 1)

        self.sidebar_expand_btn = QPushButton("« Menu")
        self.sidebar_expand_btn.setCursor(Qt.PointingHandCursor)
        self.sidebar_expand_btn.setToolTip(
            "Expand left menu (Input / Analysis / System controls)"
        )
        self.sidebar_expand_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(34, 174, 209, 0.12);
                border: 1px solid {SemiconductorTheme.ACCENT_PRIMARY};
                border-radius: 5px;
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                font-size: 8pt; font-weight: 700;
                padding: 4px 8px; min-height: 24px;
            }}
            QPushButton:hover {{
                background: rgba(34, 174, 209, 0.22);
            }}
        """)
        self.sidebar_expand_btn.clicked.connect(lambda: self.set_app_sidebar_expanded(True))
        self.sidebar_expand_btn.setVisible(False)  # only while Online + collapsed
        sidebar_nav_row.addWidget(self.sidebar_expand_btn, 1)

        sidebar_layout.addLayout(sidebar_nav_row)
        
        # Separator line
        online_sep = QWidget()
        online_sep.setFixedHeight(1)
        online_sep.setStyleSheet(f"background: {SemiconductorTheme.BORDER_DEFAULT};")
        self.online_separator = online_sep
        sidebar_layout.addWidget(online_sep)
        sidebar_layout.addSpacing(4)
        
        # --- DYNAMIC SIDEBAR CONTROLS ---
        self.sidebar_stack = QStackedWidget()
        
        # 1. 3D VIEWER PANEL
        self.viewer_panel = self._create_viewer_panel()
        
        # 2. 3D SEGMENTATION PANEL
        # We need to hide the original toolbar first
        self.segmentation_tab.toolbar.setVisible(False)
        self.segmentation_panel = self._create_segmentation_sidebar_panel()
        
        # 3. 3D AI PANEL
        self.ai_panel = QWidget()
        ai_layout = QVBoxLayout(self.ai_panel)
        ai_layout.setContentsMargins(15, 0, 15, 0)
        ai_layout.setSpacing(8)
        
        lbl_input = QLabel("INPUT DATA")
        lbl_input.setStyleSheet(SemiconductorTheme.sidebar_section_style())
        ai_layout.addWidget(lbl_input)
        
        b_ai_open = QPushButton("  OPEN")
        b_ai_open.setProperty("class", "list-btn")
        b_ai_open.setCursor(Qt.PointingHandCursor)
        b_ai_open.setToolTip("Open a 3D file or a folder of stack images")
        b_ai_open.clicked.connect(lambda: self.ai_3d_tab.load_volume())
        ai_layout.addWidget(b_ai_open)
        
        # --- AI VIEW SECTION ---
        def add_ai_section(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(SemiconductorTheme.sidebar_section_style())
            ai_layout.addWidget(lbl)
        
        add_ai_section("ZOOM")
        
        self.ai_btn_zoom_out = self._create_compact_icon_button(
            "Zoom out",
            lambda: self.ai_3d_tab.zoom_out_view() if hasattr(self.ai_3d_tab, 'zoom_out_view') else None,
            text=" - "
        )
        self.ai_btn_zoom_fit = self._create_compact_icon_button(
            "Reset zoom to fit",
            lambda: self.ai_3d_tab.reset_view_zoom() if hasattr(self.ai_3d_tab, 'reset_view_zoom') else None,
            text=" FIT "
        )
        self.ai_btn_zoom_in = self._create_compact_icon_button(
            "Zoom in",
            lambda: self.ai_3d_tab.zoom_in_view() if hasattr(self.ai_3d_tab, 'zoom_in_view') else None,
            text=" + "
        )
        # Adjust widths for text-only compact buttons
        self.ai_btn_zoom_fit.setFixedSize(44, 26)
        
        zoom_row = QHBoxLayout()
        zoom_row.setContentsMargins(0, 0, 0, 0)
        zoom_row.setSpacing(6)
        zoom_row.addWidget(self.ai_btn_zoom_out)
        zoom_row.addWidget(self.ai_btn_zoom_fit)
        zoom_row.addWidget(self.ai_btn_zoom_in)
        zoom_row.addStretch()
        ai_layout.addLayout(zoom_row)
        
        add_ai_section("VIEW MODE")
        
        self.ai_btn_xy_only = QPushButton("XY ONLY")
        self.ai_btn_xy_only.setProperty("class", "list-btn")
        self.ai_btn_xy_only.setCheckable(True)
        self.ai_btn_xy_only.setChecked(True)
        self.ai_btn_xy_only.setCursor(Qt.PointingHandCursor)
        self.ai_btn_xy_only.clicked.connect(lambda: self._ai_set_view_mode(0))
        ai_layout.addWidget(self.ai_btn_xy_only)
        
        self.ai_btn_multi = QPushButton("MULTI-PLANAR")
        self.ai_btn_multi.setProperty("class", "list-btn")
        self.ai_btn_multi.setCheckable(True)
        self.ai_btn_multi.setCursor(Qt.PointingHandCursor)
        self.ai_btn_multi.clicked.connect(lambda: self._ai_set_view_mode(1))
        ai_layout.addWidget(self.ai_btn_multi)
        
        add_ai_section("OVERLAYS")
        
        self.ai_btn_frames = QPushButton("■ SHOW FRAMES")
        self.ai_btn_frames.setProperty("class", "list-btn")
        self.ai_btn_frames.setCheckable(True)
        self.ai_btn_frames.setChecked(True)
        self.ai_btn_frames.setCursor(Qt.PointingHandCursor)
        self.ai_btn_frames.clicked.connect(self._toggle_ai_frames)
        ai_layout.addWidget(self.ai_btn_frames)
        
        add_ai_section("SYSTEM")
        
        ch_row = QHBoxLayout()
        ch_row.setContentsMargins(0, 0, 0, 0)
        self.ai_cross_btn = QPushButton("CROSSHAIR")
        self.ai_cross_btn.setProperty("class", "crosshair-btn")
        self.ai_cross_btn.setCheckable(True)
        self.ai_cross_btn.setCursor(Qt.PointingHandCursor)
        self.ai_cross_btn.clicked.connect(self._toggle_ai_crosshair)
        ch_row.addWidget(self.ai_cross_btn, 1)
        ai_layout.addLayout(ch_row)
        
        ai_layout.addStretch()
        
        # 4. 3D ANALYSIS PANEL
        self.analysis_panel = self._create_analysis_sidebar_panel()
        self.help_panel = QWidget()
        
        self.batch_panel = self._create_batch_review_sidebar_panel()
        self.sidebar_stack.addWidget(self.viewer_panel)      # Index 0
        self.sidebar_stack.addWidget(self.segmentation_panel) # Index 1
        self.sidebar_stack.addWidget(self.ai_panel)           # Index 2
        self.sidebar_stack.addWidget(self.analysis_panel)     # Index 3
        self.sidebar_stack.addWidget(self.batch_panel)        # Index 4
        self.sidebar_stack.addWidget(self.help_panel)         # Index 5
        
        sidebar_layout.addWidget(self.sidebar_stack)
        sidebar_layout.addStretch()

        # Footer info in sidebar
        logo_label = QLabel()
        logo_path = resource_path("assets/branding/company_logo.webp")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            scaled_pixmap = pixmap.scaledToHeight(30, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        else:
            logo_label.setText('<span style="color: #E0E0E0; font-weight: 300;">INNO</span><span style="color: #00FFFF; font-weight: 900;">3D</span>')
            logo_label.setStyleSheet("font-family: 'Segoe UI Semibold', 'Segoe UI'; font-size: 12pt;")
            
            # Subtle glow for footer as well
            footer_glow = QGraphicsDropShadowEffect()
            footer_glow.setBlurRadius(8)
            footer_glow.setColor(QColor(0, 255, 255, 100))
            footer_glow.setOffset(0, 0)
            logo_label.setGraphicsEffect(footer_glow)
        logo_label.setAlignment(Qt.AlignCenter)
        self.sidebar_footer_logo = logo_label
        
        # Add Hardware Tracking (same structure as DEMO)
        self.hw_monitor = HardwareTrackingWidget()
        sidebar_layout.addWidget(self.hw_monitor)
        
        sidebar_layout.addWidget(logo_label)

        # Widgets hidden when Online collapses the left menu (ONLINE toggle stays)
        self._sidebar_detail_widgets = [
            self.app_title,
            self.sidebar_hbm_label,
            self.sidebar_version_label,
            self.online_separator,
            self.sidebar_stack,
            self.hw_monitor,
            self.sidebar_footer_logo,
        ]
        
        tabs_container.addWidget(sidebar)

        # Main display area takes full height (no bottom status bar)
        tabs_container.addWidget(self.stack)

        # Keep hidden status sink for online/status updates to avoid breaking signal handlers
        self.status_label = QLabel("Ready")
        self.status_label.setVisible(False)
        
        # Set splitter sizes (sidebar ~200px, rest for stack)
        tabs_container.setSizes([200, 1400])
        
        content_layout.addWidget(tabs_container)

        self.current_tab_index = 0
        self.switch_tab(0) # Init first tab
        self._apply_dynamic_theme_styles()
        # FDC heartbeat is always-on (independent from Online toggle state).
        try:
            self._online_start_fdc_monitor_sender()
        except Exception as e:
            print(f"[ONLINE] FDC sender init failed: {e}")

    def _on_fdc_settings_changed(self):
        """Reload always-on FDC sender after Help-tab Save & Apply."""
        try:
            if hasattr(self, "_online_stop_fdc_monitor_sender"):
                self._online_stop_fdc_monitor_sender()
            if hasattr(self, "_online_start_fdc_monitor_sender"):
                self._online_start_fdc_monitor_sender()
            host = getattr(self, "_fdc_monitor_host", "?")
            port = getattr(self, "_fdc_monitor_port", "?")
            if hasattr(self, "status_label") and self.status_label is not None:
                self.status_label.setText(f"FDC settings applied → {host}:{port}")
        except Exception as e:
            print(f"[ONLINE] FDC settings apply failed: {e}")

    def set_app_sidebar_expanded(self, expanded, remember=True):
        """Expand / collapse the left app menu (Online frees space for MPR).

        When collapsed, brand / stack / footer hide — **ONLINE toggle remains**
        so the operator can still turn Online OFF. Use « Menu to reopen.
        """
        expanded = bool(expanded)
        self._app_sidebar_expanded = expanded
        online = bool(getattr(self, "online_toggle", None) and self.online_toggle.isChecked())
        if remember and online:
            self._app_sidebar_user_expanded = expanded

        for w in getattr(self, "_sidebar_detail_widgets", []) or []:
            if w is not None:
                w.setVisible(expanded)

        # Nav buttons
        if hasattr(self, "sidebar_expand_btn"):
            self.sidebar_expand_btn.setVisible((not expanded) and online)
        if hasattr(self, "sidebar_collapse_btn"):
            self.sidebar_collapse_btn.setVisible(expanded and online)

        sidebar = getattr(self, "sidebar_widget", None)
        sp = getattr(self, "tabs_splitter", None)
        if sidebar is None:
            return

        if expanded:
            sidebar.setMinimumWidth(150)
            sidebar.setMaximumWidth(400)
            if sp is not None:
                sizes = sp.sizes()
                total = sum(sizes) if sum(sizes) > 0 else max(sp.width(), 1600)
                left = 200
                if self._pre_online_sidebar_sizes and len(self._pre_online_sidebar_sizes) >= 1:
                    left = max(160, self._pre_online_sidebar_sizes[0])
                left = min(left, max(160, total // 5))
                sp.setSizes([left, max(400, total - left)])
        else:
            # Compact strip: ONLINE label + switch + « Menu
            compact_w = 160
            sidebar.setMinimumWidth(compact_w)
            sidebar.setMaximumWidth(compact_w)
            if sp is not None:
                sizes = sp.sizes()
                total = sum(sizes) if sum(sizes) > 0 else max(sp.width(), 1600)
                sp.setSizes([compact_w, max(400, total - compact_w)])

        # Reflow viewer overlays after width change
        mpv = getattr(self, "multiplanar_tab", None)
        if mpv is not None and hasattr(mpv, "_reposition_visible_overlays"):
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(40, mpv._reposition_visible_overlays)

    def set_online_app_sidebar_layout(self, online):
        """Online ON → keep left menu expanded by default (user can manually collapse with 'Hide menu ‹'); Offline → full menu restored."""
        online = bool(online)
        sp = getattr(self, "tabs_splitter", None)
        if online:
            if sp is not None:
                self._pre_online_sidebar_sizes = list(sp.sizes())
            self.set_app_sidebar_expanded(True, remember=False)
        else:
            self.set_app_sidebar_expanded(True, remember=False)
            # Restore pre-online splitter sizes when possible
            if sp is not None and self._pre_online_sidebar_sizes:
                try:
                    sp.setSizes(self._pre_online_sidebar_sizes)
                except Exception:
                    pass
            self._pre_online_sidebar_sizes = None
            if hasattr(self, "sidebar_expand_btn"):
                self.sidebar_expand_btn.setVisible(False)
            if hasattr(self, "sidebar_collapse_btn"):
                self.sidebar_collapse_btn.setVisible(False)

    def on_theme_changed(self, theme_name):
        mode = normalize_theme_name(theme_name)
        app = QApplication.instance()
        if app is None:
            return

        apply_theme(app, mode)
        self._apply_dynamic_theme_styles()
        self._save_theme_preference(mode)
        self.help_tab.set_current_theme(mode)
        if hasattr(self, "batch_review_tab") and hasattr(self.batch_review_tab, "refresh_theme"):
            try:
                self.batch_review_tab.refresh_theme()
            except Exception:
                pass

    def _apply_online_toggle_style(self):
        if hasattr(self, "online_toggle") and hasattr(self.online_toggle, "apply_theme"):
            self.online_toggle.apply_theme()

    def _sync_online_toggle_hint(self, checked):
        """Highlight active side label and keep tooltip in sync with knob position."""
        if not hasattr(self, "online_toggle"):
            return
        if checked:
            self.online_toggle.setToolTip("Online mode is ON. Slide left for Offline.")
        else:
            self.online_toggle.setToolTip("Online mode is OFF. Slide right for Online.")
        # Side labels: active side bold+accent, inactive muted
        self._style_online_side_labels(checked)

    def _style_online_side_labels(self, online_active):
        """Offline (left) / Online (right) — emphasize the active mode."""
        accent = SemiconductorTheme.ACCENT_PRIMARY
        muted = SemiconductorTheme.TEXT_SECONDARY
        active_ss = f"font-size: 8.5pt; font-weight: bold; color: {accent};"
        inactive_ss = f"font-size: 8.5pt; font-weight: 600; color: {muted};"
        if hasattr(self, "online_offline_label"):
            self.online_offline_label.setStyleSheet(inactive_ss if online_active else active_ss)
        if hasattr(self, "online_online_label"):
            self.online_online_label.setStyleSheet(active_ss if online_active else inactive_ss)

    def _apply_online_chrome(self, active):
        """Cyan glow frame around Offline/Online switch when Online is ON."""
        if not hasattr(self, "online_container"):
            return
        if active:
            self.online_container.setStyleSheet(
                """
                QWidget#onlineModeFrame {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 rgba(34, 174, 209, 0.14),
                        stop:1 rgba(34, 174, 209, 0.03));
                    border: 1px solid rgba(34, 174, 209, 0.45);
                    border-radius: 10px;
                }
                """
            )
        else:
            self.online_container.setStyleSheet(
                """
                QWidget#onlineModeFrame {
                    background: transparent;
                    border: 1px solid transparent;
                    border-radius: 10px;
                }
                """
            )
        self._style_online_side_labels(bool(active))

    def _set_online_input_locked(self, locked):
        """Disable manual OPEN load while Online controls input (#3)."""
        lock_tip = "Input is controlled by Online server (Server.cpp)"
        normal_tip = "Open a 3D file or a folder of stack images"
        btn = getattr(self, "viewer_btn_open", None)
        if btn is not None:
            btn.setEnabled(not locked)
            btn.setToolTip(lock_tip if locked else normal_tip)
            btn.setCursor(Qt.ArrowCursor if locked else Qt.PointingHandCursor)
        # Yellow banner removed — OPEN stays disabled + tooltip is enough
        hint = getattr(self, "online_input_lock_hint", None)
        if hint is not None:
            hint.setVisible(False)

    def _apply_dynamic_theme_styles(self):
        if hasattr(self, "tabs_splitter"):
            self.tabs_splitter.setStyleSheet(
                f"QSplitter::handle {{ background: {SemiconductorTheme.BORDER_DEFAULT}; }}"
            )
        if hasattr(self, "sidebar_widget"):
            self.sidebar_widget.setObjectName("AppSidebar")
            self.sidebar_widget.setStyleSheet(
                f"""
                QWidget#AppSidebar {{
                    background-color: {SemiconductorTheme.BG_MEDIUM};
                    border-right: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                }}
                """
            )
        if hasattr(self, "sidebar_version_label"):
            self.sidebar_version_label.setStyleSheet(
                f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 8pt; margin-top: -12px; margin-bottom: 5px;"
            )
        if hasattr(self, "sidebar_hbm_label"):
            self.sidebar_hbm_label.setStyleSheet(
                "font-size: 8.5pt; letter-spacing: 0.8px; margin-top: 0px; margin-bottom: 2px;"
            )
            self.sidebar_hbm_label.setText(
                f"<span style='color: {SemiconductorTheme.TEXT_SECONDARY};'>High-Bandwidth Memory</span>"
                f"<br>"
                f"<span style='color: {SemiconductorTheme.ACCENT_PRIMARY}; font-weight: 600; letter-spacing: 1.2px;'>(HBM)</span>"
            )
        if hasattr(self, "online_separator"):
            self.online_separator.setStyleSheet(f"background: {SemiconductorTheme.BORDER_DEFAULT};")
        if hasattr(self, "online_toggle"):
            self._apply_online_toggle_style()
            self._apply_online_chrome(self.online_toggle.isChecked())
        if hasattr(self, "title_bar"):
            self.title_bar.refresh_theme()
        if hasattr(self, "hw_monitor"):
            self.hw_monitor.refresh_theme()
        if hasattr(self, "segmentation_tab") and hasattr(self.segmentation_tab, "refresh_theme"):
            self.segmentation_tab.refresh_theme()
        for overlay in self.findChildren(QWidget):
            if overlay.__class__.__name__ == "SliceControlOverlay" and hasattr(overlay, "refresh_theme"):
                overlay.refresh_theme()
        self._refresh_themed_icons()

    def _save_theme_preference(self, theme_name):
        _settings_set_theme(normalize_theme_name(theme_name))

    def _resolve_ui_icon(self, icon_name, fallback_standard_icon=None):
        """Load icon from bundled assets with fallback to Qt standard icon."""
        icon_path = resource_path(os.path.join("assets", "icons", icon_name))
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        if fallback_standard_icon is not None:
            return self.style().standardIcon(fallback_standard_icon)
        return QIcon()

    def _icon_tint_color(self, btn=None):
        if btn is not None and str(btn.property("role") or "").lower() == "danger":
            return QColor(SemiconductorTheme.ACCENT_ERROR)
        if btn is not None and bool(btn.property("loaded")):
            return QColor(SemiconductorTheme.ACCENT_SUCCESS)
        return QColor(SemiconductorTheme.TEXT_PRIMARY if SemiconductorTheme.is_light() else "#dfeaf5")

    def _tint_icon(self, icon, size, color):
        pixmap = icon.pixmap(size)
        if pixmap.isNull():
            return icon
        tinted = QPixmap(pixmap.size())
        tinted.fill(Qt.transparent)
        painter = QPainter(tinted)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), color)
        painter.end()
        return QIcon(tinted)

    def _apply_themed_icon(self, btn):
        if not bool(btn.property("theme_icon")):
            return

        icon_name = btn.property("icon_name")
        std_prop = btn.property("standard_icon")
        fallback_standard_icon = None
        if std_prop is not None and std_prop != "":
            try:
                fallback_standard_icon = QStyle.StandardPixmap(int(std_prop))
            except Exception:
                fallback_standard_icon = None

        base_icon = self._resolve_ui_icon(icon_name, fallback_standard_icon) if icon_name else self.style().standardIcon(fallback_standard_icon) if fallback_standard_icon is not None else QIcon()
        size_w = int(btn.property("icon_size_w") or 16)
        size_h = int(btn.property("icon_size_h") or 16)
        icon_size = QSize(size_w, size_h)
        tinted_icon = self._tint_icon(base_icon, icon_size, self._icon_tint_color(btn))
        btn.setIcon(tinted_icon)
        btn.setIconSize(icon_size)

    def _refresh_themed_icons(self):
        """Retint monochrome sidebar icons for the active theme.

        Only buttons marked ``theme_icon`` (or previously tracked monochrome
        bases) are recolored. Multi-color custom icons (e.g. layout grid
        thumbnails with a 3D cube glyph) must set ``preserve_icon_color`` and
        must never go through SourceIn tinting — that turns them into solid
        white squares on startup.
        """
        for btn in self.findChildren(QAbstractButton):
            if btn.icon().isNull():
                continue
            # Custom multi-color icons (layout presets, etc.)
            if bool(btn.property("preserve_icon_color")):
                continue
            if bool(btn.property("theme_icon")):
                self._apply_themed_icon(btn)
                continue
            # Only re-tint icons we already classified as monochrome theme icons.
            # Do NOT adopt arbitrary button icons as tint bases (that bleached
            # the Dragonfly layout thumbnails to white on first theme polish).
            base_icon = getattr(btn, "_base_theme_icon", None)
            if base_icon is None or base_icon.isNull():
                continue
            size = btn.iconSize()
            if not size.isValid() or size.width() <= 0 or size.height() <= 0:
                size = QSize(16, 16)
            btn.setIcon(self._tint_icon(base_icon, size, self._icon_tint_color(btn)))

    def _create_load_icon_button(self, tooltip, callback, icon_name=None, standard_icon=QStyle.SP_FileIcon):
        """Create unified icon-only load buttons used in sidebars."""
        btn = QPushButton()
        btn.setProperty("class", "load-icon-btn")
        btn.setProperty("role", "secondary")
        btn.setProperty("size", "lg")
        btn.setProperty("is_icon_only", True)
        btn.setProperty("theme_icon", True)
        btn.setProperty("icon_name", icon_name or "")
        btn.setProperty("standard_icon", int(standard_icon) if standard_icon is not None else "")
        btn.setProperty("icon_size_w", 18)
        btn.setProperty("icon_size_h", 18)
        self._apply_themed_icon(btn)
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(callback)
        return btn

    def _create_compact_icon_button(self, tooltip, callback, text="", standard_icon=None, icon_name=None):
        """Create compact utility buttons for color and small tools."""
        btn = QPushButton(text)
        width = 30
        if text and (icon_name or standard_icon is not None):
            width = 44
        btn.setFixedSize(width, 26)
        btn.setProperty("class", "icon-button")
        btn.setProperty("role", "ghost")
        btn.setProperty("size", "sm")
        if icon_name or standard_icon is not None:
            btn.setProperty("theme_icon", True)
            btn.setProperty("icon_name", icon_name or "")
            btn.setProperty("standard_icon", int(standard_icon) if standard_icon is not None else "")
            btn.setProperty("icon_size_w", 14)
            btn.setProperty("icon_size_h", 14)
            self._apply_themed_icon(btn)
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(callback)
        return btn

    def _create_system_icon_button(self, tooltip, callback, standard_icon=None, icon_name=None, fill=False):
        """Create system icon button matching Carl Zeiss design.

        ``fill=False`` → fixed 34×34 square (e.g. color pickers beside MASK C1/C2).
        ``fill=True``  → height 34px, width expands equally in a full-width row
        (utility row under CROSSHAIR — no empty gap on the right).
        """
        btn = QPushButton()
        if fill:
            btn.setProperty("class", "system-icon-btn-fill")
            btn.setFixedHeight(34)
            btn.setMinimumWidth(34)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        else:
            btn.setFixedSize(34, 34)
            btn.setProperty("class", "system-icon-btn")
        if icon_name or standard_icon is not None:
            btn.setProperty("theme_icon", True)
            btn.setProperty("icon_name", icon_name or "")
            btn.setProperty("standard_icon", int(standard_icon) if standard_icon is not None else "")
            btn.setProperty("icon_size_w", 16)
            btn.setProperty("icon_size_h", 16)
            self._apply_themed_icon(btn)
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(callback)
        return btn





    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(200, self.init_vtk_interactors)
        self.status_label.setText("Initializing VTK Rendering Engine...")
    
    def init_vtk_interactors(self):
        """Initialize all VTK interactors once"""
        try:
            # 1. Multi-Planar Tab Interactors
            for orientation in ['axial', 'sagittal', 'coronal']:
                widget = getattr(self.multiplanar_tab, f'{orientation}_widget')
                widget.GetRenderWindow().GetInteractor().Initialize()
            self.multiplanar_tab.view_3d_widget.GetRenderWindow().GetInteractor().Initialize()
            
            # 2. Segmentation Tab Interactors
            self.segmentation_tab.init_vtk_widgets()

            # 4. Visualizer Tab Interactor (if exists)
            if hasattr(self, 'visualizer_tab'):
                self.visualizer_tab.vtk_widget.GetRenderWindow().GetInteractor().Initialize()
                self.visualizer_tab.renderer.SetBackground(0.05, 0.05, 0.08) # Set default BG

            self.status_label.setText("Ready - System Initialized")
            
        except Exception as e:
            self.status_label.setText(f"Initialization Warning: {str(e)}")
    
    def on_tab_changed(self, index):
        """Handle Tab Change - Universal cleanup of VTK widgets to fix ghosting"""
        
        # Helper to hide objects
        def safe_hide(parent, attr):
            if hasattr(parent, attr):
                obj = getattr(parent, attr)
                if hasattr(obj, 'setVisible'):
                    obj.setVisible(False)

        # 1. HIDE ALL possibly floating VTK widgets from any tab
        if hasattr(self, 'multiplanar_tab'):
            for o in ['axial', 'sagittal', 'coronal']: safe_hide(self.multiplanar_tab, f'{o}_widget')
            safe_hide(self.multiplanar_tab, 'view_3d_widget')
            
        if hasattr(self, 'segmentation_tab'):
            for o in ['axial', 'sagittal', 'coronal']: safe_hide(self.segmentation_tab, f'{o}_widget')

        if hasattr(self, 'visualizer_tab'):
            safe_hide(self.visualizer_tab, 'vtk_widget')

        # 2. SHOW current tab's widgets (Render deferred — sync multi-pane Render freezes UI)
        show_widgets = []
        if index == 0: # viewer
            for o in ['axial', 'sagittal', 'coronal']:
                w = getattr(self.multiplanar_tab, f'{o}_widget')
                w.setVisible(True)
                show_widgets.append(w)
            self.multiplanar_tab.view_3d_widget.setVisible(True)
            show_widgets.append(self.multiplanar_tab.view_3d_widget)
        elif index == 1: # segmentation
            for o in ['axial', 'coronal', 'sagittal']:
                w = getattr(self.segmentation_tab, f'{o}_widget')
                w.setVisible(True)
                show_widgets.append(w)
        elif index == 2 and hasattr(self, 'visualizer_tab'):
            w = self.visualizer_tab.vtk_widget
            w.setVisible(True)
            show_widgets.append(w)
        # index == 3: Analysis tab — no VTK widgets, nothing to show/hide

        if show_widgets:
            QTimer.singleShot(0, lambda widgets=show_widgets: self._render_tab_vtk_widgets(widgets))

        # Update labels and state
        tab_names = [
            "3D VIEWER",
            "3D TEACHING",
            "3D AI MODULE",
            "3D ANALYSIS",
            "LINE PULSE",
            "HELP",
        ]
        if index < len(tab_names):
            self.status_label.setText(f"ACTIVE MODULE: {tab_names[index]}")
        if index == 4 and hasattr(self, "batch_review_tab"):
            try:
                if hasattr(self.batch_review_tab, "_sync_live_poll_timer"):
                    self.batch_review_tab._sync_live_poll_timer()
            except Exception:
                pass
        
        self.current_tab_index = index

    def _render_tab_vtk_widgets(self, widgets):
        """Deferred VTK paints after tab switch so the stack can swap first."""
        for w in widgets or []:
            try:
                if w is None or not w.isVisible():
                    continue
                rw = w.GetRenderWindow()
                if rw is not None:
                    rw.Render()
            except Exception:
                pass

    def _ai_set_view_mode(self, mode):
        """Toggle AI tab XY Only / Multi-Planar from sidebar"""
        self.ai_btn_xy_only.setChecked(mode == 0)
        self.ai_btn_multi.setChecked(mode == 1)
        # Sync with the toggle bar inside ai tab
        if hasattr(self.ai_3d_tab, 'btn_xy_only'):
            self.ai_3d_tab.btn_xy_only.setChecked(mode == 0)
        if hasattr(self.ai_3d_tab, 'btn_multi_view'):
            self.ai_3d_tab.btn_multi_view.setChecked(mode == 1)
        self.ai_3d_tab.set_view_mode(mode)

    def _toggle_ai_crosshair(self):
        """Enable/disable crosshair on AI tab viewers, set to center position."""
        checked = self.ai_cross_btn.isChecked()
        ai = self.ai_3d_tab
        
        # If enabling, place crosshair at volume center
        if checked and ai.volume_data is not None:
            z, y, x = ai.volume_data.shape
            cz, cy, cx = z // 2, y // 2, x // 2
            Z_max = z - 1
            # Set slice sliders to center
            if ai.xy_view.current_slice != cz: ai.xy_view.set_current_slice(cz)
            if ai.xz_view.current_slice != cy: ai.xz_view.set_current_slice(cy)
            if ai.yz_view.current_slice != cx: ai.yz_view.set_current_slice(cx)
            # Set crosshair positions in DISPLAY space:
            # axial (no transform):     col=world_x, row=world_y
            ai.xy_view.crosshair_data_pos = (cx, cy)
            # coronal (no flipud):      col=world_x, row=world_z (Z increases downward)
            ai.xz_view.crosshair_data_pos = (cx, cz)
            # sagittal (transpose Z↔):  col=world_z, row=world_y
            ai.yz_view.crosshair_data_pos = (cz, cy)
        
        # Enable/disable crosshair on all viewers
        for v in ai.all_viewers:
            v.show_crosshair = checked
            v.update_view()
        
        # Keep internal btn in sync
        if hasattr(ai, 'btn_crosshair'):
            ai.btn_crosshair.setChecked(checked)


    def _toggle_ai_frames(self):
        """Sync the sidebar SHOW FRAMES toggle with the tab's btn_show_frames."""
        checked = self.ai_btn_frames.isChecked()
        self.ai_btn_frames.setText(f"{'■' if checked else '□'} SHOW FRAMES")
        if hasattr(self.ai_3d_tab, 'btn_show_frames'):
            self.ai_3d_tab.btn_show_frames.setChecked(checked)
        self.ai_3d_tab.sync()


    def _on_batch_review_open_run(self, run: dict):
        """Deep-link from Batch Review → full Viewer parity (vol+mask+MES+B2B)."""
        run = run or {}
        try:
            self.switch_tab(0)
            mpv = self.multiplanar_tab
            # Offline keeps STATISTICS hidden — force show before/while loading
            if hasattr(mpv, "show_stats_panel_for_review"):
                mpv.show_stats_panel_for_review(True, collapse_tools=False)
            elif hasattr(mpv, "stats_panel") and mpv.stats_panel is not None:
                mpv.stats_panel.setVisible(True)
            if hasattr(mpv, "load_batch_review_run"):
                ok = mpv.load_batch_review_run(run)
            else:
                # Legacy fallback: volume + masks only
                vol, bump, void = (
                    mpv.resolve_batch_run_paths(run)
                    if hasattr(mpv, "resolve_batch_run_paths")
                    else ("", "", "")
                )
                if not vol:
                    QMessageBox.warning(
                        self,
                        "Open full Viewer",
                        "No volume file found on disk for this run.",
                    )
                    return
                ok = mpv.load_volume_from_path(
                    vol,
                    mask_bump_path=bump if bump and os.path.isfile(bump) else None,
                    mask_void_path=void if void and os.path.isfile(void) else None,
                )
            if not ok:
                return
            n_mes = len(run.get("mes_objects") or [])
            chip = f"({run.get('chip_col')},{run.get('chip_row')})"
            fov = run.get("fov_index") or "?"
            self.status_label.setText(
                f"LINE PULSE → Viewer full parity · Chip {chip} P{fov} · "
                f"MES {n_mes} · vol/mask/B2B loading…"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Open full Viewer", f"Failed to load:\n{e}")

    def _on_analysis_open_in_line_pulse(self, run_id: str):
        """Deep-link focused Analysis FOV to its Line Pulse run detail."""
        run_id = str(run_id or "")
        self.switch_tab(4)
        selected = False
        if run_id and hasattr(self.batch_review_tab, "select_run"):
            try:
                selected = bool(self.batch_review_tab.select_run(run_id))
            except Exception:
                selected = False
        if selected:
            self.status_label.setText(f"Analysis → Line Pulse · focused run {run_id}")
        else:
            self.status_label.setText(
                f"Analysis → Line Pulse · refresh complete; run {run_id or 'not available'}"
            )


    def closeEvent(self, event):
        """Cleanly shut down VTK and background threads to prevent wglMakeCurrent errors."""
        try:
            if hasattr(self, "_online_stop_fdc_monitor_sender"):
                self._online_stop_fdc_monitor_sender()

            # 1. Clean up MultiPlanarView VTK widgets
            if hasattr(self, 'multiplanar_tab') and self.multiplanar_tab:
                for w_name in ['axial_widget', 'coronal_widget', 'sagittal_widget', 'view_3d_widget']:
                    w = getattr(self.multiplanar_tab, w_name, None)
                    if w is not None and hasattr(w, 'Finalize'):
                        w.Finalize()
            
            # 2. Clean up SegmentationTab VTK widgets
            if hasattr(self, 'segmentation_tab') and self.segmentation_tab:
                for w_name in ['axial_widget', 'coronal_widget', 'sagittal_widget', 'view_3d_widget']:
                    w = getattr(self.segmentation_tab, w_name, None)
                    if w is not None and hasattr(w, 'Finalize'):
                        w.Finalize()

            # 3. Clean up Visualizer VTK widgets
            if hasattr(self, 'visualizer_tab') and self.visualizer_tab:
                if hasattr(self.visualizer_tab, 'vtk_widget') and hasattr(self.visualizer_tab.vtk_widget, 'Finalize'):
                    self.visualizer_tab.vtk_widget.Finalize()

            # 4. Stop background threads
            if hasattr(self, "hw_monitor") and self.hw_monitor is not None:
                if hasattr(self.hw_monitor, "stop"):
                    self.hw_monitor.stop()

            if hasattr(self, 'ai_3d_tab') and self.ai_3d_tab:
                if hasattr(self.ai_3d_tab, 'train_worker') and self.ai_3d_tab.train_worker:
                    self.ai_3d_tab.train_worker.stop()
                    self.ai_3d_tab.train_worker.wait(500)
                
            # Cleanly stop online processor if running
            if hasattr(self, 'online_processor') and self.online_processor:
                self.online_processor.stop()

        except Exception as e:
            print("Error during VTK cleanup:", e)

        super().closeEvent(event)

