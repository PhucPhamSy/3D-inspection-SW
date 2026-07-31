# inno3d/app/splash.py
# -----------------------------------------------------------------------
# UI chrome widgets — extracted from main.py (Phase 6)
#
# LuxuryLogoLabel, CustomTitleBar, HardwareTrackingWidget
# + thin helpers: app_ini_path, resolve_startup_theme
#
# IMPORTANT: This module must NOT create QApplication / QSplashScreen or
# import heavy tabs at import time. Splash lifecycle lives only in main.py.
# (Phase 6 COPY left a second broken splash here → stuck / no progress.)
# -----------------------------------------------------------------------

from __future__ import annotations

import os
import subprocess

import psutil
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from inno3d.app.settings import get_theme as _settings_get_theme
from inno3d.core.resources import resource_path
from inno3d.core.styles import SemiconductorTheme
from inno3d.core.ui_system import AnimatedNavButton, GlowProgressBar
from inno3d.infra.paths import app_config_ini_path


def app_ini_path():
    """Resolve app_config.ini location — delegates to infra.paths."""
    return app_config_ini_path()


def resolve_startup_theme():
    """Load preferred theme — delegates to app.settings."""
    return _settings_get_theme()


def update_splash(val, msg):
    """Optional no-op fallback.

    Real progress is driven by ``main.py``'s own ``update_splash`` during
    staged imports. Importers must not rely on this module owning the splash.
    """
    return None


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
        self.btn_batch = self._create_tab("LINE PULSE", 4)
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
            if hasattr(self.parent_window, "_normal_geometry"):
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

        self.cpu_val, self.cpu_bar = self._create_sensor("CPU", layout)
        self.ram_val, self.ram_bar = self._create_sensor("RAM", layout)
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
        val_lbl.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-family: 'Consolas'; "
            f"font-size: 8.5pt; font-weight: bold;"
        )

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

            try:
                smi = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    creationflags=subprocess.CREATE_NO_WINDOW
                    if hasattr(subprocess, "CREATE_NO_WINDOW")
                    else 0,
                    timeout=0.5,
                    text=True,
                )
                parts = smi.strip().split(",")
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
