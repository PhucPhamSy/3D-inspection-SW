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

# ---- SPLASH SCREEN LOGIC ----
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)

# Determine logo path safely
if getattr(sys, "frozen", False):
    base_dir = Path(sys._MEIPASS)
else:
    base_dir = Path(__file__).resolve().parent
splash_pixmap_path = str(base_dir / "assets" / "branding" / "company_logo_v1.png")

bg_pix = QPixmap(500, 300)
bg_pix.fill(QColor("#030712"))

if os.path.exists(splash_pixmap_path):
    logo = QPixmap(splash_pixmap_path).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    painter = QPainter(bg_pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.TextAntialiasing)
    logo_x = (500 - 150) // 2
    logo_y = 50
    painter.drawPixmap(logo_x, logo_y, 150, 150, logo)
    # Trademark mark near top-right of the app logo
    painter.setPen(QColor("#9ca3af"))
    painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
    painter.drawText(QRect(logo_x + 118, logo_y + 6, 28, 18), Qt.AlignLeft | Qt.AlignVCenter, "™")
    painter.setPen(QColor("#f9fafb"))
    painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
    painter.drawText(QRect(0, 210, 500, 30), Qt.AlignCenter, "INNO3D INSPECTION")
    painter.end()

splash = QSplashScreen(bg_pix, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
progress = QProgressBar(splash)
progress.setGeometry(50, 250, 400, 6)
progress.setStyleSheet("""
    QProgressBar { background: rgba(255, 255, 255, 0.1); border-radius: 3px; border: none; }
    QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #10b981); border-radius: 3px; }
""")
progress.setTextVisible(False)

msg_label = QLabel(splash)
msg_label.setGeometry(0, 265, 500, 20)
msg_label.setAlignment(Qt.AlignCenter)
msg_label.setStyleSheet("color: #9ca3af; font-family: 'Segoe UI'; font-size: 10px;")

splash.show()
app.processEvents()

def update_splash(val, msg):
    progress.setValue(val)
    msg_label.setText(msg)
    app.processEvents()

update_splash(10, "Loading UI Framework...")
# ---- END SPLASH SCREEN LOGIC ----

update_splash(20, "Loading Core Resources...")
from inno3d.core.resources import resource_path
from inno3d.core.styles import SemiconductorTheme, apply_theme, normalize_theme_name
from inno3d.core.ui_system import AnimatedNavButton, GlowProgressBar

update_splash(35, "Loading Online Mode Modules...")
from inno3d.modes.online import OnlineModeMixin

update_splash(50, "Loading AI Engine & Segmentation...")
from inno3d.tabs.ai import AI3DTab
from inno3d.tabs.teaching import SegmentationTab

update_splash(75, "Loading 3D Visualizer (VTK)...")
from inno3d.tabs.analysis import AnalysisTab
from inno3d.tabs.viewer import MultiPlanarView

update_splash(88, "Loading Batch Review...")
from inno3d.tabs.batch_review import BatchReviewTab

update_splash(90, "Loading Help Module...")
from inno3d.tabs.help import HelpTab

import psutil
update_splash(95, "Initializing Application Window...")


from inno3d.infra.paths import app_config_ini_path
from inno3d.app.settings import get_theme as _settings_get_theme, set_theme as _settings_set_theme


def app_ini_path():
    """Resolve app_config.ini location — delegates to infra.paths."""
    return app_config_ini_path()


def resolve_startup_theme():
    """Load preferred theme — delegates to app.settings."""
    return _settings_get_theme()


class LuxuryLogoLabel(QLabel):
    """Custom typographic brand mark for sidebar title."""
    _brand_family = None
    _brand_font_loaded = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(78)
        self.setContentsMargins(0, 8, 0, 8)
        self._ensure_brand_font()

    @classmethod
    def _ensure_brand_font(cls):
        if cls._brand_family:
            return cls._brand_family

        font_path = resource_path(os.path.join("assets", "fonts", "Casanova Scotia.otf"))
        if os.path.exists(font_path):
            font_id = QFontDatabase.addApplicationFont(font_path)
            if font_id != -1:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    cls._brand_family = families[0]
                    cls._brand_font_loaded = True

        if not cls._brand_family:
            cls._brand_family = "Segoe UI Semibold"
            cls._brand_font_loaded = False
        return cls._brand_family
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        rect = self.rect()

        brand_family = self._ensure_brand_font()
        f_main = QFont(brand_family, 30 if self._brand_font_loaded else 28)
        f_main.setLetterSpacing(QFont.AbsoluteSpacing, 0.4 if self._brand_font_loaded else 0.8)
        if not self._brand_font_loaded:
            f_main = QFont("Segoe UI Semibold", 24)
            f_main.setLetterSpacing(QFont.AbsoluteSpacing, 1.0)

        f_suffix = QFont(f_main)
        f_suffix.setPointSize(max(24, f_main.pointSize() - 2))

        txt_left = "INNO"
        txt_right = "3D"
        txt_tm = "™"

        fm_main = QFontMetrics(f_main)
        fm_suffix = QFontMetrics(f_suffix)
        gap = 8
        tm_gap = 2
        w_left = fm_main.horizontalAdvance(txt_left)
        w_right = fm_suffix.horizontalAdvance(txt_right)

        f_tm = QFont("Segoe UI", 9, QFont.Bold)
        fm_tm = QFontMetrics(f_tm)
        w_tm = fm_tm.horizontalAdvance(txt_tm)
        total_w = w_left + gap + w_right + tm_gap + w_tm

        start_x = int((rect.width() - total_w) / 2)
        baseline = int(rect.center().y() + 8)

        painter.setPen(QColor(8, 14, 20, 170))
        painter.setFont(f_main)
        painter.drawText(start_x + 1, baseline + 1, txt_left)
        painter.setFont(f_suffix)
        painter.drawText(start_x + w_left + gap + 1, baseline + 1, txt_right)

        inno_path = QPainterPath()
        inno_path.addText(start_x, baseline, f_main, txt_left)
        grad_inno = QLinearGradient(0, baseline - 26, 0, baseline + 2)
        grad_inno.setColorAt(0.0, QColor("#f2f7ff"))
        grad_inno.setColorAt(1.0, QColor("#a6b3c2"))
        painter.fillPath(inno_path, QBrush(grad_inno))

        right_x = start_x + w_left + gap
        d3_path = QPainterPath()
        d3_path.addText(right_x, baseline, f_suffix, txt_right)
        grad_3d = QLinearGradient(0, baseline - 24, 0, baseline + 2)
        grad_3d.setColorAt(0.0, QColor("#76efff"))
        grad_3d.setColorAt(1.0, QColor("#24b7d8"))
        painter.fillPath(d3_path, QBrush(grad_3d))

        # Trademark symbol after brand mark
        tm_x = right_x + w_right + tm_gap
        tm_baseline = baseline - max(10, f_suffix.pointSize() - 10)
        painter.setPen(QColor("#8a97a8"))
        painter.setFont(f_tm)
        painter.drawText(tm_x, tm_baseline, txt_tm)


class CustomTitleBar(QWidget):
    """Horizontal Title Bar with Navigation and System Controls"""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        self.setFixedHeight(28)
        self.setObjectName("CustomTitleBar")
        self.setAttribute(Qt.WA_StyledBackground)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(0)
        
        # 1. Logo Icon
        self.logo_label = QLabel()
        app_icon_path = resource_path("assets/branding/company_logo_v1.png")
        if os.path.exists(app_icon_path):
            pixmap = QPixmap(app_icon_path).scaled(18, 18, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(pixmap)
        layout.addWidget(self.logo_label)
        layout.addSpacing(20)
        
        # 2. Navigation Tabs
        self.tabs_container = QWidget()
        self.tabs_layout = QHBoxLayout(self.tabs_container)
        self.tabs_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs_layout.setSpacing(0)
        
        self.nav_btns = []
        self.btn_multi = self._create_tab("3D VIEWER", 0)
        self.btn_seg = self._create_tab("3D TEACHING", 1)
        self.btn_ai = self._create_tab("3D AI", 2)
        self.btn_analysis = self._create_tab("3D ANALYSIS", 3)
        self.btn_batch = self._create_tab("BATCH REVIEW", 4)
        self.btn_help = self._create_tab("HELP", 5)
        
        self.tabs_layout.addWidget(self.btn_multi)
        self.tabs_layout.addWidget(self.btn_seg)
        self.tabs_layout.addWidget(self.btn_ai)
        self.tabs_layout.addWidget(self.btn_analysis)
        self.tabs_layout.addWidget(self.btn_batch)
        self.tabs_layout.addWidget(self.btn_help)
        
        layout.addWidget(self.tabs_container)
        layout.addStretch()
        
        # 3. System Buttons
        self._is_custom_maximized = False
        self.btn_min = self._create_sys_btn("−", self.parent_window.showMinimized)
        self.btn_max = self._create_sys_btn("□", self._toggle_maximize)
        self.btn_close = self._create_sys_btn("×", self.parent_window.close, is_close=True)
        
        layout.addWidget(self.btn_min)
        layout.addWidget(self.btn_max)
        layout.addWidget(self.btn_close)
        
        self._drag_pos = None

    def _create_tab(self, text, index):
        btn = AnimatedNavButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda: self.parent_window.switch_tab(index))
        self.nav_btns.append(btn)
        return btn
        
    def _create_sys_btn(self, text, callback, is_close=False):
        btn = QPushButton(text)
        btn.setFixedSize(45, 28)
        btn.setProperty("is_close_btn", bool(is_close))
        self._apply_sys_btn_style(btn)
        btn.clicked.connect(callback)
        return btn

    def _apply_sys_btn_style(self, btn):
        is_close = bool(btn.property("is_close_btn"))
        is_light = SemiconductorTheme.is_light()
        text_color = "rgba(20, 33, 46, 0.80)" if is_light else "rgba(255, 255, 255, 0.72)"
        hover_bg = "rgba(10, 143, 183, 0.14)" if is_light else "rgba(255, 255, 255, 0.15)"
        hover_text = SemiconductorTheme.TEXT_PRIMARY if is_light else "#ffffff"
        close_hover_bg = "#d9363e" if is_light else "#e81123"
        close_hover_text = "#ffffff"
        applied_hover_bg = close_hover_bg if is_close else hover_bg
        applied_hover_text = close_hover_text if is_close else hover_text

        btn.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {text_color};
                font-family: 'Segoe UI Symbol', 'Arial', sans-serif;
                font-size: 11pt;
                padding: 0px;
                margin: 0px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {applied_hover_bg};
                color: {applied_hover_text};
            }}
            """
        )

    def refresh_theme(self):
        for btn in (self.btn_min, self.btn_max, self.btn_close):
            self._apply_sys_btn_style(btn)

    def _toggle_maximize(self):
        if self.parent_window.isMaximized() or self._is_custom_maximized:
            if hasattr(self.parent_window, '_normal_geometry'):
                self.parent_window.setGeometry(self.parent_window._normal_geometry)
            self.parent_window.setWindowState(Qt.WindowNoState)
            self._is_custom_maximized = False
            self.btn_max.setText("□")
        else:
            self.parent_window._normal_geometry = self.parent_window.geometry()
            ag = QApplication.desktop().availableGeometry(self.parent_window)
            self.parent_window.setGeometry(ag)
            self._is_custom_maximized = True
            self.btn_max.setText("❐")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos()
            
    def mouseMoveEvent(self, event):
        if self._drag_pos is not None:
            delta = event.globalPos() - self._drag_pos
            self.parent_window.move(self.parent_window.pos() + delta)
            self._drag_pos = event.globalPos()
            
    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event):
        self._toggle_maximize()


class HardwareTrackingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 15)
        layout.setSpacing(12)

        self.setObjectName("HardwareTracker")
        self.sensor_title_labels = []

        # CPU
        self.cpu_val, self.cpu_bar = self._create_sensor("CPU", layout)
        # RAM
        self.ram_val, self.ram_bar = self._create_sensor("RAM", layout)
        # GPU
        self.gpu_val, self.gpu_bar = self._create_sensor("GPU UTIL", layout)
        self.gpu_vram_val, self.gpu_vram_bar = self._create_sensor("GPU VRAM", layout)

        self.refresh_theme()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_stats)
        self.timer.start(1000)

    def _create_sensor(self, title, parent_layout):
        container = QVBoxLayout()
        container.setSpacing(4)

        header_row = QHBoxLayout()
        title_lbl = QLabel(title)
        self.sensor_title_labels.append(title_lbl)

        val_lbl = QLabel("0.0%")
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        val_lbl.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-family: 'Consolas'; font-size: 8.5pt; font-weight: bold;")

        header_row.addWidget(title_lbl)
        header_row.addStretch()
        header_row.addWidget(val_lbl)

        bar = GlowProgressBar()
        bar.setTextVisible(False)
        bar.setFixedHeight(8)

        container.addLayout(header_row)
        container.addWidget(bar)
        parent_layout.addLayout(container)

        return val_lbl, bar

    def refresh_theme(self):
        is_light = SemiconductorTheme.is_light()
        panel_bg = "rgba(231, 238, 246, 0.92)" if is_light else "rgba(15, 20, 35, 0.6)"
        panel_border = "rgba(133, 155, 178, 0.30)" if is_light else "rgba(255, 255, 255, 0.05)"
        title_color = "rgba(63, 82, 103, 0.88)" if is_light else "rgba(255, 255, 255, 0.4)"
        self.setStyleSheet(
            f"""
            #HardwareTracker {{
                background: {panel_bg};
                border-top: 1px solid {panel_border};
                border-bottom: 1px solid {panel_border};
            }}
            """
        )
        for title_lbl in self.sensor_title_labels:
            title_lbl.setStyleSheet(
                f"color: {title_color}; font-size: 7.5pt; font-weight: 800; letter-spacing: 1px;"
            )

    def _update_stats(self):
        try:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent

            self._apply_val(cpu, self.cpu_val, self.cpu_bar)
            self._apply_val(ram, self.ram_val, self.ram_bar)

            # NVIDIA GPU Fetch
            import subprocess
            try:
                smi = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
                    timeout=0.5, text=True
                )
                parts = smi.strip().split(',')
                gpu = float(parts[0].strip())
                vram = float(parts[1].strip()) / max(float(parts[2].strip()), 1) * 100.0

                self._apply_val(gpu, self.gpu_val, self.gpu_bar)
                self._apply_val(vram, self.gpu_vram_val, self.gpu_vram_bar)
            except Exception:
                pass

        except Exception:
            pass

    def _apply_val(self, val, lbl, bar):
        lbl.setText(f"{val:04.1f}%")
        bar.setValue(int(val))
        bar.setStyleSheet(self._bar_style(val))
        lbl.setStyleSheet(self._text_style(val))

    def _bar_style(self, val):
        is_light = SemiconductorTheme.is_light()
        if val > 85:
            color = SemiconductorTheme.ACCENT_ERROR
        elif val > 65:
            color = SemiconductorTheme.ACCENT_WARNING
        else:
            color = "#1b8e5a" if is_light else "#00ff9d"

        bar_bg = "rgba(255, 255, 255, 0.82)" if is_light else "rgba(0, 0, 0, 0.4)"
        bar_border = "rgba(133, 155, 178, 0.45)" if is_light else "rgba(255, 255, 255, 0.05)"
        return f"""
            QProgressBar {{
                background-color: {bar_bg};
                border-radius: 4px;
                border: 1px solid {bar_border};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {color.replace("1.0", "0.4") if "rgba" in color else color}, stop:1 {color});
                border-radius: 3px;
            }}
        """

    def _text_style(self, val):
        if val > 85:
            color = SemiconductorTheme.ACCENT_ERROR
        elif val > 65:
            color = SemiconductorTheme.ACCENT_WARNING
        else:
            color = SemiconductorTheme.TEXT_SECONDARY
        return f"color: {color}; font-family: 'Consolas'; font-size: 8.5pt; font-weight: bold;"


class MainWindow(OnlineModeMixin, QMainWindow):
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
        self.help_tab.set_current_theme(SemiconductorTheme.CURRENT_THEME)
        if hasattr(self.batch_review_tab, "open_in_viewer"):
            self.batch_review_tab.open_in_viewer.connect(self._on_batch_review_open_run)

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
        # Outer frame gets cyan glow when Online is ON (#8 chrome)
        online_container = QWidget()
        online_container.setObjectName("onlineModeFrame")
        online_outer = QVBoxLayout(online_container)
        online_outer.setContentsMargins(10, 6, 10, 8)
        online_outer.setSpacing(0)

        online_row = QWidget()
        online_layout = QHBoxLayout(online_row)
        online_layout.setContentsMargins(5, 2, 5, 2)
        online_layout.setSpacing(8)
        
        online_label = QLabel("ONLINE")
        online_label.setStyleSheet(f"font-size: 9pt; font-weight: bold; color: {SemiconductorTheme.TEXT_SECONDARY};")
        self.online_label = online_label
        online_layout.addWidget(online_label)
        
        self.online_toggle = QPushButton("OFF")
        self.online_toggle.setCheckable(True)
        self.online_toggle.setFixedSize(64, 30)
        self.online_toggle.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(26, 31, 53, 220),
                    stop:1 rgba(20, 28, 55, 240));
                border: 2px solid rgba(45, 55, 72, 0.6);
                border-radius: 15px;
                font-size: 9pt;
                font-weight: bold;
                padding: 0px 0px 1px 0px;
                color: {SemiconductorTheme.TEXT_DISABLED};
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                border-color: rgba(0, 255, 157, 0.3);
                color: {SemiconductorTheme.TEXT_SECONDARY};
            }}
            QPushButton:checked {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00cc7a, stop:1 #00ff9d);
                border-color: {SemiconductorTheme.ACCENT_SUCCESS};
                color: #050510;
            }}
        """)
        self.online_toggle.clicked.connect(self.toggle_online)
        online_layout.addWidget(self.online_toggle)
        online_layout.addStretch()
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
        
        b_ai_load = QPushButton("  LOAD FILE")
        b_ai_load.setProperty("class", "list-btn")
        b_ai_load.setCursor(Qt.PointingHandCursor)
        b_ai_load.clicked.connect(lambda: self.ai_3d_tab.load_volume(from_folder=False))
        
        b_ai_load_folder = QPushButton("  LOAD FOLDER")
        b_ai_load_folder.setProperty("class", "list-btn")
        b_ai_load_folder.setCursor(Qt.PointingHandCursor)
        b_ai_load_folder.clicked.connect(lambda: self.ai_3d_tab.load_folder(from_folder=True) if hasattr(self.ai_3d_tab, 'load_folder') else self.ai_3d_tab.load_volume(from_folder=True))
        
        ai_load_row = QHBoxLayout()
        ai_load_row.setContentsMargins(0, 0, 0, 0)
        ai_load_row.addWidget(b_ai_load)
        ai_load_row.addWidget(b_ai_load_folder)
        ai_layout.addLayout(ai_load_row)
        
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
            # Compact strip: ONLINE toggle + « Menu (must fit 64px switch + label)
            compact_w = 140
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
        """Online ON → auto-collapse left menu; Offline → full menu restored."""
        online = bool(online)
        sp = getattr(self, "tabs_splitter", None)
        if online:
            if sp is not None:
                self._pre_online_sidebar_sizes = list(sp.sizes())
            self._app_sidebar_user_expanded = False
            self.set_app_sidebar_expanded(False, remember=True)
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
        is_light = SemiconductorTheme.is_light()
        off_grad_start = "rgba(222, 232, 243, 245)" if is_light else "rgba(26, 31, 53, 220)"
        off_grad_end = "rgba(206, 219, 235, 250)" if is_light else "rgba(20, 28, 55, 240)"
        off_border = "rgba(132, 154, 178, 0.72)" if is_light else "rgba(45, 55, 72, 0.6)"
        off_text = SemiconductorTheme.TEXT_DISABLED
        hover_border = "rgba(10, 143, 183, 0.60)" if is_light else "rgba(0, 255, 157, 0.3)"
        hover_text = SemiconductorTheme.TEXT_PRIMARY if is_light else SemiconductorTheme.TEXT_SECONDARY
        on_start = "#20b574" if is_light else "#00cc7a"
        on_end = "#1b8e5a" if is_light else "#00ff9d"
        on_text = "#ffffff" if is_light else "#050510"

        self.online_toggle.setStyleSheet(
            f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {off_grad_start},
                    stop:1 {off_grad_end});
                border: 2px solid {off_border};
                border-radius: 15px;
                font-size: 9pt;
                font-weight: bold;
                padding: 0px 0px 1px 0px;
                color: {off_text};
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                border-color: {hover_border};
                color: {hover_text};
            }}
            QPushButton:checked {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {on_start}, stop:1 {on_end});
                border-color: {SemiconductorTheme.ACCENT_SUCCESS};
                color: {on_text};
            }}
            """
        )

    def _apply_online_chrome(self, active):
        """Cyan glow frame around ONLINE block when Online is ON."""
        if not hasattr(self, "online_container"):
            return
        accent = SemiconductorTheme.ACCENT_PRIMARY
        if active:
            self.online_container.setStyleSheet(
                f"""
                QWidget#onlineModeFrame {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 rgba(34, 174, 209, 0.14),
                        stop:1 rgba(34, 174, 209, 0.03));
                    border: 1px solid rgba(34, 174, 209, 0.45);
                    border-radius: 10px;
                }}
                """
            )
            if hasattr(self, "online_label"):
                self.online_label.setStyleSheet(
                    f"font-size: 9pt; font-weight: bold; color: {accent};"
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
            if hasattr(self, "online_label"):
                self.online_label.setStyleSheet(
                    f"font-size: 9pt; font-weight: bold; color: {SemiconductorTheme.TEXT_SECONDARY};"
                )

    def _set_online_input_locked(self, locked):
        """Disable manual FILE/FOLDER load while Online controls input (#3)."""
        lock_tip = "Input is controlled by Online server (Server.cpp)"
        for btn, normal_tip in (
            (getattr(self, "viewer_btn_file", None), "Load single 3D file (.tif/.tiff/.raw)"),
            (getattr(self, "viewer_btn_folder", None), "Load 3D stack from folder"),
        ):
            if btn is None:
                continue
            btn.setEnabled(not locked)
            btn.setToolTip(lock_tip if locked else normal_tip)
            btn.setCursor(Qt.ArrowCursor if locked else Qt.PointingHandCursor)
        # Yellow banner removed — FILE/FOLDER stay disabled + tooltip is enough
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

    def _create_viewer_panel(self):
        """Panel for 3D Viewer tab (index 0)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 0, 15, 10)
        layout.setSpacing(8) # Tighter spacing -> Increased to 8px
        
        def add_section_title(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(SemiconductorTheme.sidebar_section_style())
            layout.addWidget(lbl)

        # ═══════════════════════════════════════════════════════════
        # 1. INPUT DATA  —  Full-width load buttons with icon + text
        # ═══════════════════════════════════════════════════════════
        add_section_title("INPUT DATA")
        
        load_row = QGridLayout()
        load_row.setContentsMargins(0, 0, 0, 0)
        load_row.setSpacing(6)
        load_row.setColumnStretch(0, 1)
        load_row.setColumnStretch(1, 1)
        
        b_load = QPushButton("  FILE")
        b_load.setProperty("class", "system-text-btn")
        b_load.setProperty("theme_icon", True)
        b_load.setProperty("icon_name", "file.svg")
        b_load.setProperty("standard_icon", int(QStyle.SP_FileIcon))
        b_load.setProperty("icon_size_w", 16)
        b_load.setProperty("icon_size_h", 16)
        b_load.setProperty("is_icon_only", False)
        self._apply_themed_icon(b_load)
        b_load.setCursor(Qt.PointingHandCursor)
        b_load.setToolTip("Load single 3D file (.tif/.tiff/.raw)")
        b_load.clicked.connect(lambda: self.multiplanar_tab.load_volume(from_folder=False))
        load_row.addWidget(b_load, 0, 0)
        self.viewer_btn_file = b_load
        
        b_load_folder = QPushButton("  FOLDER")
        b_load_folder.setProperty("class", "system-text-btn")
        b_load_folder.setProperty("theme_icon", True)
        b_load_folder.setProperty("icon_name", "folder-open.svg")
        b_load_folder.setProperty("standard_icon", int(QStyle.SP_DirIcon))
        b_load_folder.setProperty("icon_size_w", 16)
        b_load_folder.setProperty("icon_size_h", 16)
        b_load_folder.setProperty("is_icon_only", False)
        self._apply_themed_icon(b_load_folder)
        b_load_folder.setCursor(Qt.PointingHandCursor)
        b_load_folder.setToolTip("Load 3D stack from folder")
        b_load_folder.clicked.connect(lambda: self.multiplanar_tab.load_volume(from_folder=True))
        load_row.addWidget(b_load_folder, 0, 1)
        self.viewer_btn_folder = b_load_folder
        
        layout.addLayout(load_row)

        # Online still disables FILE/FOLDER (see _set_online_input_locked); no banner text.
        self.online_input_lock_hint = None

        # ═══════════════════════════════════════════════════════════
        # 2. ANALYSIS TOOLS  —  Bordered buttons with inline icons
        # ═══════════════════════════════════════════════════════════
        add_section_title("ANALYSIS TOOLS")
        
        # MASK C1 + color picker
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(6)
        b_c1 = QPushButton("  MASK C1")
        b_c1.setProperty("class", "system-text-btn")
        b_c1.setProperty("theme_icon", True)
        b_c1.setProperty("icon_name", "mask.svg")
        b_c1.setProperty("standard_icon", "")
        b_c1.setProperty("icon_size_w", 16)
        b_c1.setProperty("icon_size_h", 16)
        self._apply_themed_icon(b_c1)
        b_c1.setCursor(Qt.PointingHandCursor)
        b_c1.clicked.connect(lambda: self.multiplanar_tab.load_class_mask(1))
        b_c1_clr = self._create_system_icon_button(
            "Choose Class 1 overlay color",
            lambda: self.multiplanar_tab.choose_overlay_color(1),
            icon_name="color-picker.svg",
            standard_icon=QStyle.SP_DriveDVDIcon,
        )
        row1.addWidget(b_c1, 1)
        row1.addWidget(b_c1_clr)
        layout.addLayout(row1)
        
        # MASK C2 + color picker
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(6)
        b_c2 = QPushButton("  MASK C2")
        b_c2.setProperty("class", "system-text-btn")
        b_c2.setProperty("theme_icon", True)
        b_c2.setProperty("icon_name", "mask.svg")
        b_c2.setProperty("standard_icon", "")
        b_c2.setProperty("icon_size_w", 16)
        b_c2.setProperty("icon_size_h", 16)
        self._apply_themed_icon(b_c2)
        b_c2.setCursor(Qt.PointingHandCursor)
        b_c2.clicked.connect(lambda: self.multiplanar_tab.load_class_mask(2))
        b_c2_clr = self._create_system_icon_button(
            "Choose Class 2 overlay color",
            lambda: self.multiplanar_tab.choose_overlay_color(2),
            icon_name="color-picker.svg",
            standard_icon=QStyle.SP_DriveDVDIcon,
        )
        row2.addWidget(b_c2, 1)
        row2.addWidget(b_c2_clr)
        layout.addLayout(row2)
        
        # MASK FOLDER (full width)
        b_mask_folder = QPushButton("  MASK FOLDER")
        b_mask_folder.setProperty("class", "system-text-btn")
        b_mask_folder.setProperty("theme_icon", True)
        b_mask_folder.setProperty("icon_name", "layers.svg")
        b_mask_folder.setProperty("standard_icon", "")
        b_mask_folder.setProperty("icon_size_w", 16)
        b_mask_folder.setProperty("icon_size_h", 16)
        self._apply_themed_icon(b_mask_folder)
        b_mask_folder.setCursor(Qt.PointingHandCursor)
        b_mask_folder.setToolTip("Load all per-layer masks (bump + void) from a sample output folder")
        b_mask_folder.clicked.connect(self.multiplanar_tab.load_mask_folder)
        layout.addWidget(b_mask_folder)
        
        # ═══════════════════════════════════════════════════════════
        # 3. SYSTEM  —  Carl Zeiss Industrial Design Language
        #    Layout:
        #    Row 0: [LAYOUT toggle — show/hide grid layout strip]
        #    Row 1: [CROSSHAIR toggle — full width        ]
        #    Row 2: [⟳ Reset CH] [🎯 CH Color] [📏 Ruler] [⟳ Reset Rot]
        #    (2D ROTATE lives on the right OBLIQUE MPR sidebar)
        # ═══════════════════════════════════════════════════════════
        add_section_title("SYSTEM")

        # ── Toggle: LAYOUT (above crosshair) ───────────────────
        # Shows/hides the left Dragonfly-style view-grid icon strip.
        self.layout_grid_btn = QPushButton("  LAYOUT")
        self.layout_grid_btn.setProperty("class", "system-toggle-btn")
        self.layout_grid_btn.setProperty("theme_icon", True)
        self.layout_grid_btn.setProperty("icon_name", "grid.svg")
        self.layout_grid_btn.setProperty("standard_icon", "")
        self.layout_grid_btn.setProperty("icon_size_w", 16)
        self.layout_grid_btn.setProperty("icon_size_h", 16)
        self._apply_themed_icon(self.layout_grid_btn)
        self.layout_grid_btn.setCheckable(True)
        self.layout_grid_btn.setChecked(False)
        self.layout_grid_btn.setCursor(Qt.PointingHandCursor)
        self.layout_grid_btn.setToolTip(
            "Show / hide view grid layout picker (Dragonfly-style).\n"
            "Cube glyph on icons = 3D volume pane; plain cells = MPR."
        )
        self.layout_grid_btn.clicked.connect(
            lambda checked: self.multiplanar_tab.toggle_layout_sidebar(checked)
        )
        layout.addWidget(self.layout_grid_btn)
        
        # ── Toggle: CROSSHAIR (full-width) ─────────────────────
        self.cross_btn = QPushButton("  CROSSHAIR")
        self.cross_btn.setProperty("class", "system-toggle-btn")
        self.cross_btn.setProperty("theme_icon", True)
        self.cross_btn.setProperty("icon_name", "crosshair.svg")
        self.cross_btn.setProperty("standard_icon", "")
        self.cross_btn.setProperty("icon_size_w", 16)
        self.cross_btn.setProperty("icon_size_h", 16)
        self._apply_themed_icon(self.cross_btn)
        self.cross_btn.setCheckable(True)
        self.cross_btn.setCursor(Qt.PointingHandCursor)
        self.cross_btn.setToolTip("Toggle crosshair overlay on 2D views")
        self.cross_btn.clicked.connect(lambda checked: self.multiplanar_tab.toggle_crosshair(Qt.Checked if checked else Qt.Unchecked))
        layout.addWidget(self.cross_btn)
        
        # ── Utility icon row: 4 equal cells, full width (matches LAYOUT/CROSSHAIR) ──
        util_row = QGridLayout()
        util_row.setContentsMargins(0, 0, 0, 0)
        util_row.setHorizontalSpacing(6)
        util_row.setVerticalSpacing(0)
        for c in range(4):
            util_row.setColumnStretch(c, 1)

        b_ch_reset = self._create_system_icon_button(
            "Reset crosshair position & rotation",
            self.multiplanar_tab.reset_crosshair,
            icon_name="refresh.svg",
            standard_icon=QStyle.SP_BrowserReload,
            fill=True,
        )
        b_ch_clr = self._create_system_icon_button(
            "Choose crosshair color",
            self.multiplanar_tab.choose_crosshair_color,
            icon_name="color-picker.svg",
            standard_icon=QStyle.SP_DriveDVDIcon,
            fill=True,
        )
        b_r_clr = self._create_system_icon_button(
            "Choose ruler color",
            self.multiplanar_tab.choose_ruler_color,
            icon_name="ruler-2.svg",
            standard_icon=QStyle.SP_DriveDVDIcon,
            fill=True,
        )
        b_rot_reset = self._create_system_icon_button(
            "Reset 2D Rotation",
            self.multiplanar_tab.reset_2d_rotation,
            icon_name="rotate-clockwise-2.svg",
            standard_icon=QStyle.SP_BrowserReload,
            fill=True,
        )
        util_row.addWidget(b_ch_reset, 0, 0)
        util_row.addWidget(b_ch_clr, 0, 1)
        util_row.addWidget(b_r_clr, 0, 2)
        util_row.addWidget(b_rot_reset, 0, 3)
        layout.addLayout(util_row)
        
        # (2D ROTATE + alignment controls consolidated in 3D viewer right panel)
        
        # ═══════════════════════════════════════════════════════════
        # CLEAR ALL  —  danger zone, icon + text, full width
        # ═══════════════════════════════════════════════════════════
        layout.addSpacing(8)
        b_clear = QPushButton("  CLEAR ALL")
        b_clear.setProperty("class", "clear-btn")
        b_clear.setProperty("theme_icon", True)
        b_clear.setProperty("icon_name", "eraser.svg")
        b_clear.setProperty("standard_icon", "")
        b_clear.setProperty("icon_size_w", 14)
        b_clear.setProperty("icon_size_h", 14)
        b_clear.setProperty("role", "danger")
        self._apply_themed_icon(b_clear)
        b_clear.setCursor(Qt.PointingHandCursor)
        b_clear.clicked.connect(self.multiplanar_tab.clear_masks)
        layout.addWidget(b_clear)
        
        # Bottom stretch to push everything UP
        layout.addStretch()

        # Map to tab
        self.multiplanar_tab.load_volume_btn = b_load
        self.multiplanar_tab.load_volume_folder_btn = b_load_folder
        self.multiplanar_tab.load_class1_btn = b_c1
        self.multiplanar_tab.load_class2_btn = b_c2
        self.multiplanar_tab.clear_masks_btn = b_clear
        self.multiplanar_tab.crosshair_check = self.cross_btn
        
        # Timer to update bullets
        self.bullet_timer = QTimer(self)
        self.bullet_timer.timeout.connect(self._update_sidebar_bullets)
        self.bullet_timer.start(500)
        
        return panel

    def _update_sidebar_bullets(self):
        """Update bullet and styling indicating loaded state."""
        tab = self.multiplanar_tab
        if not hasattr(tab, 'load_volume_btn'): return
        
        def set_bullet(btn, loaded):
            changed = False
            if loaded and not btn.property("loaded"):
                btn.setProperty("loaded", True)
                changed = True
            elif not loaded and btn.property("loaded"):
                btn.setProperty("loaded", False)
                changed = True
                 
            if changed:
                btn.style().unpolish(btn)
                btn.style().polish(btn)
                if bool(btn.property("theme_icon")):
                    self._apply_themed_icon(btn)
                
        # Use getattr safely for C1/C2 because we don't know immediately if they exist
        vol_loaded = hasattr(tab, 'volume_data') and tab.volume_data is not None
        c1_loaded = hasattr(tab, 'class1_data') and tab.class1_data is not None
        c2_loaded = hasattr(tab, 'class2_data') and tab.class2_data is not None
        
        set_bullet(tab.load_volume_btn, vol_loaded)
        if hasattr(tab, 'load_volume_folder_btn'): set_bullet(tab.load_volume_folder_btn, vol_loaded)
        if hasattr(tab, 'load_class1_btn'): set_bullet(tab.load_class1_btn, c1_loaded)
        if hasattr(tab, 'load_class2_btn'): set_bullet(tab.load_class2_btn, c2_loaded)


    def _create_segmentation_sidebar_panel(self):
        """Panel for 3D Segmentation tab (index 1)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 0, 15, 0)
        layout.setSpacing(6)
        
        seg = self.segmentation_tab
        
        def add_section_title(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(SemiconductorTheme.sidebar_section_style())
            layout.addWidget(lbl)

        # Environment
        add_section_title("IO & CONFIG")
        
        # DLL
        layout.addWidget(QLabel("DLL Folder:"))
        self.seg_dll_input = QLineEdit()
        self.seg_dll_input.setReadOnly(True)
        b_dll = self._create_compact_icon_button(
            "Browse DLL folder",
            seg.load_dll_folder,
            standard_icon=QStyle.SP_DirIcon,
            icon_name="folder.svg",
        )
        dll_row = QHBoxLayout()
        dll_row.setContentsMargins(0, 0, 0, 0)
        dll_row.setSpacing(6)
        dll_row.addWidget(self.seg_dll_input, 1)
        dll_row.addWidget(b_dll, 0)
        layout.addLayout(dll_row)
        
        # Config
        layout.addWidget(QLabel("Config File:"))
        self.seg_config_input = QLineEdit()
        self.seg_config_input.setReadOnly(True)
        b_cfg = self._create_compact_icon_button(
            "Browse config file",
            seg.load_config_file,
            standard_icon=QStyle.SP_FileIcon,
            icon_name="settings.svg",
        )
        cfg_row = QHBoxLayout()
        cfg_row.setContentsMargins(0, 0, 0, 0)
        cfg_row.setSpacing(6)
        cfg_row.addWidget(self.seg_config_input, 1)
        cfg_row.addWidget(b_cfg, 0)
        layout.addLayout(cfg_row)
        
        # Input
        layout.addWidget(QLabel("Input Path:"))
        self.seg_input_path = QLineEdit()
        self.seg_input_path.setReadOnly(True)
        
        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(6)
        
        b_input_file = self._create_compact_icon_button(
            "Browse input file",
            lambda: seg.load_volume(from_folder=False),
            standard_icon=QStyle.SP_FileIcon,
            icon_name="file.svg",
        )
        b_input_file.setProperty("is_icon_only", True)
        
        b_input_folder = self._create_compact_icon_button(
            "Browse input folder",
            lambda: seg.load_volume(from_folder=True),
            standard_icon=QStyle.SP_DirIcon,
            icon_name="folder-open.svg",
        )
        b_input_folder.setProperty("is_icon_only", True)
        
        input_row.addWidget(self.seg_input_path, 1)
        input_row.addWidget(b_input_file, 0)
        input_row.addWidget(b_input_folder, 0)
        layout.addLayout(input_row)

        # Output
        layout.addWidget(QLabel("Output Path:"))
        self.seg_output_input = QLineEdit()
        self.seg_output_input.setReadOnly(True)
        b_out = self._create_compact_icon_button(
            "Browse output folder",
            seg.browse_output_path,
            standard_icon=QStyle.SP_DialogOpenButton,
            icon_name="folder-open.svg",
        )
        out_row = QHBoxLayout()
        out_row.setContentsMargins(0, 0, 0, 0)
        out_row.setSpacing(6)
        out_row.addWidget(self.seg_output_input, 1)
        out_row.addWidget(b_out, 0)
        layout.addLayout(out_row)
        
        # Actions
        add_section_title("ACTIONS")
        b_clear = QPushButton("CLEAR MASK")
        b_clear.setProperty("class", "clear-btn")
        b_clear.setProperty("role", "danger")
        b_clear.clicked.connect(seg.clear_masks)
        layout.addWidget(b_clear)
        
        # Overlay & Crosshair
        add_section_title("OVERLAY")
        ch_row = QHBoxLayout()
        self.seg_cross_check = QCheckBox("Crosshair")
        self.seg_cross_check.stateChanged.connect(seg.toggle_crosshair)
        ch_row.addWidget(self.seg_cross_check)
        
        b_c1 = self._create_compact_icon_button(
            "Choose Class 1 color",
            lambda: seg.choose_overlay_color(1),
            text="C1",
            icon_name="color-picker.svg",
            standard_icon=QStyle.SP_DriveDVDIcon,
        )
        b_c2 = self._create_compact_icon_button(
            "Choose Class 2 color",
            lambda: seg.choose_overlay_color(2),
            text="C2",
            icon_name="color-picker.svg",
            standard_icon=QStyle.SP_DriveDVDIcon,
        )
        ch_row.addWidget(b_c1); ch_row.addWidget(b_c2)
        layout.addLayout(ch_row)

        # Info
        layout.addSpacing(10)
        self.seg_info_label = QLabel("Ready")
        self.seg_info_label.setWordWrap(True)
        self.seg_info_label.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt; font-style: italic;")
        layout.addWidget(self.seg_info_label)
        
        # Link back to tab
        seg.dll_path_input = self.seg_dll_input
        seg.config_path_input = self.seg_config_input
        seg.input_path_input = self.seg_input_path
        seg.output_path_input = self.seg_output_input
        seg.crosshair_check = self.seg_cross_check
        seg.info_label = self.seg_info_label
        
        # Sync the text of the new inputs if the tab already loaded default paths during init
        if hasattr(seg, 'dll_path') and seg.dll_path:
            seg.dll_path_input.setText(seg.dll_path)
        if hasattr(seg, 'config_path') and seg.config_path:
            seg.config_path_input.setText(seg.config_path)
        
        layout.addStretch()
        return panel

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

        # 2. SHOW current tab's widgets
        if index == 0: # viewer
            for o in ['axial', 'sagittal', 'coronal']:
                w = getattr(self.multiplanar_tab, f'{o}_widget')
                w.setVisible(True)
                w.GetRenderWindow().Render()
            self.multiplanar_tab.view_3d_widget.setVisible(True)
            self.multiplanar_tab.view_3d_widget.GetRenderWindow().Render()
        elif index == 1: # segmentation
            for o in ['axial', 'coronal', 'sagittal']:
                w = getattr(self.segmentation_tab, f'{o}_widget')
                w.setVisible(True)
                w.GetRenderWindow().Render()
        elif index == 2 and hasattr(self, 'visualizer_tab'):
            w = self.visualizer_tab.vtk_widget
            w.setVisible(True)
            w.GetRenderWindow().Render()
        # index == 3: Analysis tab — no VTK widgets, nothing to show/hide

        # Update labels and state
        tab_names = [
            "3D VIEWER",
            "3D TEACHING",
            "3D AI MODULE",
            "3D ANALYSIS",
            "BATCH REVIEW",
            "HELP",
        ]
        if index < len(tab_names):
            self.status_label.setText(f"ACTIVE MODULE: {tab_names[index]}")
        if index == 4 and hasattr(self, "batch_review_tab"):
            try:
                self.batch_review_tab.refresh_all()
            except Exception:
                pass
        
        self.current_tab_index = index

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

    def _create_batch_review_sidebar_panel(self):
        """Sidebar for Batch Review tab (index 4)."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 0, 15, 0)
        layout.setSpacing(8)

        lbl = QLabel("BATCH REVIEW")
        lbl.setStyleSheet(SemiconductorTheme.sidebar_section_style())
        layout.addWidget(lbl)

        info = QLabel(
            "Browse FOV runs stored in the local inspection database.\n\n"
            "• Online mode appends each completed FOV\n"
            "• DB path: Inno3D_Data/inspection.db\n"
            "• Use Seed demo if empty"
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8.5pt;")
        layout.addWidget(info)

        b_refresh = QPushButton("  REFRESH DB")
        b_refresh.setProperty("class", "list-btn")
        b_refresh.setCursor(Qt.PointingHandCursor)
        b_refresh.clicked.connect(
            lambda: self.batch_review_tab.refresh_all()
            if hasattr(self, "batch_review_tab")
            else None
        )
        layout.addWidget(b_refresh)

        b_seed = QPushButton("  SEED DEMO DATA")
        b_seed.setProperty("class", "list-btn")
        b_seed.setCursor(Qt.PointingHandCursor)
        b_seed.clicked.connect(
            lambda: self.batch_review_tab._seed_demo()
            if hasattr(self, "batch_review_tab")
            else None
        )
        layout.addWidget(b_seed)

        b_folder = QPushButton("  OPEN DB FOLDER")
        b_folder.setProperty("class", "list-btn")
        b_folder.setCursor(Qt.PointingHandCursor)
        b_folder.clicked.connect(
            lambda: self.batch_review_tab._open_db_folder()
            if hasattr(self, "batch_review_tab")
            else None
        )
        layout.addWidget(b_folder)

        layout.addStretch()
        return panel

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
                f"BATCH REVIEW → Viewer full parity · Chip {chip} P{fov} · "
                f"MES {n_mes} · vol/mask/B2B loading…"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Open full Viewer", f"Failed to load:\n{e}")

    def _create_analysis_sidebar_panel(self):
        """Panel for 3D Analysis tab (index 3) — DB-first SOH workbench."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 0, 15, 0)
        layout.setSpacing(8)

        def add_section_title(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(SemiconductorTheme.sidebar_section_style())
            layout.addWidget(lbl)

        add_section_title("DATABASE")

        b_refresh_db = QPushButton("  REFRESH DB")
        b_refresh_db.setProperty("class", "list-btn")
        b_refresh_db.setProperty("theme_icon", True)
        b_refresh_db.setProperty("icon_name", "refresh.svg")
        self._apply_themed_icon(b_refresh_db)
        b_refresh_db.setCursor(Qt.PointingHandCursor)
        b_refresh_db.clicked.connect(self.analysis_tab.refresh_from_db)
        layout.addWidget(b_refresh_db)

        b_load_wafer = QPushButton("  LOAD WAFER")
        b_load_wafer.setProperty("class", "list-btn")
        b_load_wafer.setProperty("theme_icon", True)
        b_load_wafer.setProperty("icon_name", "layers.svg")
        self._apply_themed_icon(b_load_wafer)
        b_load_wafer.setCursor(Qt.PointingHandCursor)
        b_load_wafer.clicked.connect(self.analysis_tab.load_selected_wafer)
        layout.addWidget(b_load_wafer)

        b_load_checked = QPushButton("  LOAD CHECKED FOVS")
        b_load_checked.setProperty("class", "list-btn")
        b_load_checked.setProperty("theme_icon", True)
        b_load_checked.setProperty("icon_name", "check.svg")
        self._apply_themed_icon(b_load_checked)
        b_load_checked.setCursor(Qt.PointingHandCursor)
        b_load_checked.clicked.connect(self.analysis_tab.load_checked_fovs)
        layout.addWidget(b_load_checked)

        add_section_title("ANALYSIS SET")

        remove_row = QHBoxLayout()
        remove_row.setSpacing(4)

        b_remove = QPushButton(" REMOVE")
        b_remove.setProperty("class", "list-btn")
        b_remove.setProperty("theme_icon", True)
        b_remove.setProperty("icon_name", "trash.svg")
        self._apply_themed_icon(b_remove)
        b_remove.setCursor(Qt.PointingHandCursor)
        b_remove.clicked.connect(self.analysis_tab.remove_sample)
        remove_row.addWidget(b_remove)

        b_remove_all = QPushButton(" CLEAR")
        b_remove_all.setProperty("class", "list-btn")
        b_remove_all.setProperty("theme_icon", True)
        b_remove_all.setProperty("icon_name", "x.svg")
        self._apply_themed_icon(b_remove_all)
        b_remove_all.setCursor(Qt.PointingHandCursor)
        b_remove_all.clicked.connect(self.analysis_tab.remove_all_samples)
        b_remove_all.setStyleSheet("color: #ff6b9d;")
        remove_row.addWidget(b_remove_all)

        layout.addLayout(remove_row)

        b_export = QPushButton("  EXPORT REPORT")
        b_export.setProperty("class", "list-btn")
        b_export.setProperty("theme_icon", True)
        b_export.setProperty("icon_name", "device-floppy.svg")
        self._apply_themed_icon(b_export)
        b_export.setCursor(Qt.PointingHandCursor)
        b_export.clicked.connect(self.analysis_tab.export_report)
        layout.addWidget(b_export)

        b_recalc = QPushButton("  RECALCULATE")
        b_recalc.setProperty("class", "list-btn")
        b_recalc.setProperty("theme_icon", True)
        b_recalc.setProperty("icon_name", "refresh.svg")
        self._apply_themed_icon(b_recalc)
        b_recalc.setCursor(Qt.PointingHandCursor)
        b_recalc.clicked.connect(self.analysis_tab._recalculate_all)
        layout.addWidget(b_recalc)

        add_section_title("CSV (LAB)")

        b_import = QPushButton("  IMPORT SAMPLE")
        b_import.setProperty("class", "list-btn")
        b_import.setProperty("theme_icon", True)
        b_import.setProperty("icon_name", "file.svg")
        self._apply_themed_icon(b_import)
        b_import.setCursor(Qt.PointingHandCursor)
        b_import.clicked.connect(self.analysis_tab.import_sample)
        layout.addWidget(b_import)

        b_batch = QPushButton("  IMPORT BATCH")
        b_batch.setProperty("class", "list-btn")
        b_batch.setProperty("theme_icon", True)
        b_batch.setProperty("icon_name", "folder-open.svg")
        self._apply_themed_icon(b_batch)
        b_batch.setCursor(Qt.PointingHandCursor)
        b_batch.clicked.connect(self.analysis_tab.import_batch)
        layout.addWidget(b_batch)

        add_section_title("INFO")
        self.analysis_info_label = QLabel(
            "DB-first SOH · select lot/wafer · Load wafer or checked FOVs")
        self.analysis_info_label.setWordWrap(True)
        self.analysis_info_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt; font-style: italic;")
        layout.addWidget(self.analysis_info_label)

        layout.addStretch()
        return panel

    def closeEvent(self, event):
        """Cleanly shut down VTK and background threads to prevent wglMakeCurrent errors."""
        try:
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

def main():
    # Suppress VTK console warnings to keep CLI outputs clean
    try:
        import vtk
        vtk.vtkObject.GlobalWarningDisplayOff()
    except ImportError:
        pass

    global app, splash
    
    # Apply modern theme
    apply_theme(app, resolve_startup_theme())
    
    # Create and show window
    window = MainWindow()
    
    update_splash(100, "Ready!")
    time.sleep(0.2)
    splash.finish(window)
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

