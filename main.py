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


# ── All classes live in inno3d/app/ (Phase 6 P1b) ─────────────────────────
from inno3d.app.splash import (
    LuxuryLogoLabel,
    CustomTitleBar,
    HardwareTrackingWidget,
)
from inno3d.app.main_window import MainWindow


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

