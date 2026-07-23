import time
from PyQt5.QtCore import Qt, QPropertyAnimation, pyqtProperty, pyqtSignal, QTimer, QRect, QPoint, QEasingCurve
from PyQt5.QtGui import QPainter, QColor, QFont, QBrush, QPen, QLinearGradient, QPainterPath
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QFrame, QLabel, QVBoxLayout, QHBoxLayout,
    QGraphicsDropShadowEffect, QToolTip, QDialog, QApplication,
)

from inno3d.core.styles import SemiconductorTheme

class AnimatedNavButton(QPushButton):
    """Tab navigation button with animated underline and hover glow.

    State machine:
      UNCHECKED + not hovered â†’ no underline, no glow
      UNCHECKED + hovered     â†’ glow fades in
      CHECKED                 â†’ full underline, no glow (hover ignored)
    """

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setProperty("class", "nav-tab")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(36)

        self._glow_intensity = 0.0
        self._underline_width = 0.0

        # Glow animation (hover feedback for unchecked buttons)
        self._glow_anim = QPropertyAnimation(self, b"glow_intensity")
        self._glow_anim.setDuration(150)
        self._glow_anim.setEasingCurve(QEasingCurve.OutCubic)

        # Underline animation (checked indicator)
        self._underline_anim = QPropertyAnimation(self, b"underline_width")
        self._underline_anim.setDuration(200)
        self._underline_anim.setEasingCurve(QEasingCurve.OutQuint)

    # â”€â”€ Qt properties for QPropertyAnimation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @pyqtProperty(float)
    def glow_intensity(self):
        return self._glow_intensity

    @glow_intensity.setter
    def glow_intensity(self, value):
        self._glow_intensity = value
        self.update()

    @pyqtProperty(float)
    def underline_width(self):
        return self._underline_width

    @underline_width.setter
    def underline_width(self, value):
        self._underline_width = value
        self.update()

    # â”€â”€ Public API called by switch_tab â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def setChecked(self, checked):
        was_checked = self.isChecked()
        super().setChecked(checked)

        # Kill any in-flight glow immediately
        self._glow_anim.stop()
        self._glow_intensity = 0.0

        if checked and not was_checked:
            # Animate underline IN (from current width â†’ full)
            self._underline_anim.stop()
            self._underline_anim.setStartValue(self._underline_width)
            self._underline_anim.setEndValue(1.0)
            self._underline_anim.start()
        elif not checked and was_checked:
            # Snap underline OFF instantly (no lingering animation)
            self._underline_anim.stop()
            self._underline_width = 0.0

        self.update()

    # â”€â”€ Hover feedback (only when unchecked) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def enterEvent(self, event):
        super().enterEvent(event)
        if not self.isChecked():
            self._glow_anim.stop()
            self._glow_anim.setStartValue(self._glow_intensity)
            self._glow_anim.setEndValue(1.0)
            self._glow_anim.start()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        # Always fade out glow on leave, regardless of checked state
        self._glow_anim.stop()
        self._glow_anim.setStartValue(self._glow_intensity)
        self._glow_anim.setEndValue(0.0)
        self._glow_anim.start()

    # â”€â”€ Paint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Underline (checked indicator) â€” expands from centre
        if self._underline_width > 0.01:
            w = self.width() * self._underline_width
            x = (self.width() - w) / 2
            y = self.height() - 2
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(SemiconductorTheme.PRIMARY_DEFAULT))
            painter.drawRect(int(x), int(y), int(w), 2)

        # Hover glow (unchecked only)
        if self._glow_intensity > 0.01:
            c = QColor(SemiconductorTheme.PRIMARY_DEFAULT)
            c.setAlphaF(0.12 * self._glow_intensity)
            painter.setPen(Qt.NoPen)
            painter.setBrush(c)
            painter.drawRect(self.rect())


class PulseIndicator(QWidget):
    """Widget trÃ²n nhá» hiá»ƒn thá»‹ tráº¡ng thÃ¡i Online/Offline vá»›i pulse animation"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        
        self.status = "offline" # offline, online, processing, error
        self._pulse_radius = 4.0
        
        self.pulse_anim = QPropertyAnimation(self, b"pulse_radius")
        self.pulse_anim.setDuration(1200)
        self.pulse_anim.setStartValue(4.0)
        self.pulse_anim.setEndValue(8.0)
        self.pulse_anim.setEasingCurve(QEasingCurve.OutSine)
        self.pulse_anim.setLoopCount(-1)
        
        self.setToolTip("System Offline")
        
    @pyqtProperty(float)
    def pulse_radius(self):
        return self._pulse_radius
        
    @pulse_radius.setter
    def pulse_radius(self, value):
        self._pulse_radius = value
        self.update()
        
    def set_status(self, status):
        self.status = status
        if status == "online" or status == "processing":
            if self.pulse_anim.state() != QPropertyAnimation.Running:
                self.pulse_anim.start()
        else:
            self.pulse_anim.stop()
            self._pulse_radius = 4.0
            
        tip_map = {
            "online": "System Online - Listening",
            "offline": "System Offline",
            "processing": "Processing Data...",
            "error": "System Error"
        }
        self.setToolTip(tip_map.get(status, "Unknown Status"))
        self.update()
        
    def get_color(self):
        if self.status == "online": return QColor(SemiconductorTheme.SUCCESS_DEFAULT)
        elif self.status == "processing": return QColor(SemiconductorTheme.WARNING_DEFAULT)
        elif self.status == "error": return QColor(SemiconductorTheme.DANGER_DEFAULT)
        else: return QColor(SemiconductorTheme.ON_SURFACE_DISABLED)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        center = QPoint(self.width() // 2, self.height() // 2)
        base_color = self.get_color()
        
        if self.status in ["online", "processing"]:
            pulse_color = QColor(base_color)
            alpha_f = 1.0 - ((self._pulse_radius - 4) / 4.0)
            pulse_color.setAlphaF(max(0.0, min(1.0, alpha_f * 0.5)))
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(pulse_color)
            painter.drawEllipse(center, int(self._pulse_radius), int(self._pulse_radius))
            
        painter.setPen(Qt.NoPen)
        painter.setBrush(base_color)
        painter.drawEllipse(center, 4, 4)


class AnimatedCard(QFrame):
    """Container card vá»›i hover elevation effect"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            AnimatedCard {{
                background-color: {SemiconductorTheme.EL2};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 8px;
            }}
        """)
        
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(10)
        self.shadow.setColor(QColor(0, 0, 0, 80))
        self.shadow.setOffset(0, 2)
        self.setGraphicsEffect(self.shadow)
        
        self.hover_anim = QPropertyAnimation(self.shadow, b"blurRadius")
        self.hover_anim.setDuration(250)
        self.hover_anim.setEasingCurve(QEasingCurve.OutCubic)
        
    def enterEvent(self, event):
        super().enterEvent(event)
        self.hover_anim.stop()
        self.hover_anim.setEndValue(20)
        self.hover_anim.start()
        
    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.hover_anim.stop()
        self.hover_anim.setEndValue(10)
        self.hover_anim.start()


class GlowProgressBar(QWidget):
    """Custom progress bar vá»›i gradient fill vÃ  subtle shine"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(12)
        self._value = 0
        self._max = 100
        self._color = QColor(SemiconductorTheme.PRIMARY_DEFAULT)
        
    def set_value(self, val):
        self._value = min(self._max, max(0, val))
        self.update()

    # QProgressBar-compatible API
    def setValue(self, val):
        self.set_value(val)

    def setTextVisible(self, visible):
        pass  # We never show text â€” no-op for compatibility

    def setStyleSheet(self, ss):
        """Override to extract color from QProgressBar stylesheet for compatibility."""
        # Try to use the accent color from the stylesheet, or just ignore
        import re
        match = re.search(r'stop:1\s+(#[0-9a-fA-F]{6})', ss)
        if match:
            self._color = QColor(match.group(1))
            self.update()
        
    def set_maximum(self, max_val):
        self._max = max(1, max_val)
        self.update()
        
    def set_color(self, hex_color):
        self._color = QColor(hex_color)
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        
        # Draw background
        painter.setPen(Qt.NoPen)
        b_color = QColor(SemiconductorTheme.EL0)
        painter.setBrush(b_color)
        painter.drawRoundedRect(rect, 4, 4)
        
        # Draw progress
        if self._value > 0:
            prog_w = int((self._value / self._max) * rect.width())
            prog_rect = QRect(0, 0, prog_w, rect.height())
            
            grad = QLinearGradient(0, 0, prog_w, 0)
            c1 = QColor(self._color)
            c2 = QColor(self._color).lighter(120)
            grad.setColorAt(0, c1)
            grad.setColorAt(1, c2)
            
            painter.setBrush(grad)
            painter.drawRoundedRect(prog_rect, 4, 4)


class StatusBadge(QLabel):
    """Badge OK/NG vá»›i color-coded background"""
    def __init__(self, status="OK", parent=None):
        super().__init__(status, parent)
        self.setFont(QFont("Inter", 8, QFont.Bold))
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(40, 20)
        self.set_status(status)
        
    def set_status(self, status):
        status = status.upper()
        self.setText(status)
        color = SemiconductorTheme.SUCCESS_DEFAULT if status == "OK" else SemiconductorTheme.DANGER_DEFAULT
        bg = QColor(color)
        bg.setAlpha(40)
        
        self.setStyleSheet(f"""
            StatusBadge {{
                background-color: {bg.name(QColor.HexArgb)};
                color: {color};
                border: 1px solid {color};
                border-radius: 4px;
                padding: 2px 6px;
            }}
        """)


class SectionHeader(QWidget):
    """Header phÃ¢n chia sidebar sections"""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 4)
        
        self.line = QFrame()
        self.line.setFixedSize(3, 14)
        self.line.setStyleSheet(f"background-color: {SemiconductorTheme.PRIMARY_DEFAULT}; border-radius: 1px;")
        
        self.label = QLabel(title.upper())
        self.label.setStyleSheet(f"""
            color: {SemiconductorTheme.ON_SURFACE_SECONDARY};
            font-size: 8pt;
            font-weight: bold;
            letter-spacing: 1px;
        """)
        
        layout.addWidget(self.line)
        layout.addWidget(self.label)
        layout.addStretch()


class TooltipButton(QPushButton):
    """Icon button with rich tooltip"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("class", "icon-button")
        
    def enterEvent(self, event):
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        super().leaveEvent(event)


class _SmoothBar(QWidget):
    """Painted progress track — reliable width (no QFrame stretch quirks)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(8)
        self.setMinimumWidth(200)
        self._pct = 0.0
        self._color_ok = True

    def set_pct(self, pct, error=False):
        self._pct = max(0.0, min(100.0, float(pct)))
        self._color_ok = not error
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(SemiconductorTheme.EL0))
        p.drawRoundedRect(r, 4, 4)
        if self._pct > 0.05:
            fw = max(6, int(r.width() * self._pct / 100.0))
            fr = QRect(r.x(), r.y(), min(fw, r.width()), r.height())
            if self._color_ok:
                g = QLinearGradient(fr.left(), 0, fr.right(), 0)
                g.setColorAt(0, QColor(SemiconductorTheme.PRIMARY_PRESSED))
                g.setColorAt(0.55, QColor(SemiconductorTheme.PRIMARY_DEFAULT))
                g.setColorAt(1, QColor(SemiconductorTheme.PRIMARY_HOVER))
                p.setBrush(g)
            else:
                p.setBrush(QColor(SemiconductorTheme.DANGER_DEFAULT))
            p.drawRoundedRect(fr, 4, 4)
        p.setPen(QPen(QColor(SemiconductorTheme.BORDER_DEFAULT), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r, 4, 4)


class OnlinePipelineProgress(QDialog):
    """Compact online progress card — stage-only labels + smooth bar.

    Real pipeline callbacks are often sparse (long TIFF decode, DLL work).
    This dialog maps stage callbacks into overall 0..100, smoothly animates
    the displayed value toward the target, and auto-crawls within the current
    stage so the bar never looks frozen. Non-modal so the pipeline is never blocked.
    """

    STAGE_META = {
        "load": ("Loading", "Reading volume data"),
        "enhance": ("Enhancing", "Preprocessing volume"),
        "segment": ("Segmenting", "Running inspection"),
        "measure": ("Measuring", "Computing results"),
        "done": ("Complete", "Inspection finished"),
        "error": ("Failed", "Inspection stopped"),
    }

    def __init__(self, parent=None, has_enhance=False):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setObjectName("OnlinePipelineProgress")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setWindowModality(Qt.NonModal)
        self.setFixedSize(380, 138)

        self._has_enhance = bool(has_enhance)
        self._stages = self._build_stage_list(self._has_enhance)
        self._stage = self._stages[0] if self._stages else "load"
        self._target = 0.0
        self._display = 0.0
        self._closed = False
        self._error = False
        self._crawl = True

        self._build_ui()
        self._apply_style()
        self._refresh_stage_ui()

        self._anim = QTimer(self)
        self._anim.setInterval(33)
        self._anim.timeout.connect(self._tick_anim)

    def configure(self, has_enhance=False):
        self._has_enhance = bool(has_enhance)
        self._stages = self._build_stage_list(self._has_enhance)
        if self._stage not in self._stages and self._stage not in ("done", "error"):
            self._stage = self._stages[0]
        self._refresh_stage_ui()

    def set_stage(self, stage_key, progress=None):
        if stage_key not in self.STAGE_META:
            return
        if stage_key == "enhance" and not self._has_enhance:
            stage_key = "segment"
        self._stage = stage_key
        self._error = stage_key == "error"
        self._crawl = stage_key not in ("done", "error")

        within = 0 if progress is None else int(progress)
        floor = float(self._map_stage_progress(stage_key, within))
        # New FOV reset
        if stage_key == "load" and within <= 0:
            self._target = floor
            self._display = floor
        else:
            self._target = max(self._target, floor)

        self._refresh_stage_ui()
        self._apply_display()
        if not self._anim.isActive() and not self._closed:
            self._anim.start()

    def set_stage_progress(self, percent):
        try:
            pct = max(0, min(100, int(percent)))
        except (TypeError, ValueError):
            pct = 0
        mapped = float(self._map_stage_progress(self._stage, pct))
        self._target = max(self._target, mapped)
        if not self._anim.isActive() and not self._closed:
            self._anim.start()

    def setValue(self, val):
        self.set_stage_progress(val)

    def setLabelText(self, text):
        _ = text

    def setWindowModality(self, modality):
        super().setWindowModality(Qt.NonModal)

    def setMinimumDuration(self, _ms):
        pass

    def setCancelButton(self, _btn):
        pass

    def setRange(self, _minimum, _maximum):
        pass

    def setMaximum(self, _maximum):
        pass

    def setMinimum(self, _minimum):
        pass

    def show(self):
        self._closed = False
        self._error = False
        self._crawl = True
        self._center_on_parent()
        super().show()
        self.raise_()
        if not self._anim.isActive():
            self._anim.start()
        QTimer.singleShot(0, self._apply_display)

    def close(self):
        self._closed = True
        self._crawl = False
        self._anim.stop()
        super().close()

    def finish_ok(self):
        self._crawl = False
        self._stage = "done"
        self._target = 100.0
        self._refresh_stage_ui()
        if not self._anim.isActive():
            self._anim.start()
        QTimer.singleShot(700, self.close)

    def finish_error(self):
        self._crawl = False
        self._error = True
        self._stage = "error"
        self._title.setText("Failed")
        self._subtitle.setText("Inspection stopped")
        self._header.setText("Inspection failed")
        self._apply_display()
        QTimer.singleShot(900, self.close)

    @staticmethod
    def _build_stage_list(has_enhance):
        stages = ["load"]
        if has_enhance:
            stages.append("enhance")
        stages.extend(["segment", "measure"])
        return stages

    def _stage_ranges(self):
        n = max(1, len(self._stages))
        span = 92.0 / n
        ranges = {}
        for i, key in enumerate(self._stages):
            ranges[key] = (i * span, (i + 1) * span)
        ranges["done"] = (92.0, 100.0)
        ranges["error"] = (0.0, 100.0)
        return ranges

    def _map_stage_progress(self, stage_key, within_pct):
        ranges = self._stage_ranges()
        lo, hi = ranges.get(stage_key, (0.0, 100.0))
        within = max(0.0, min(100.0, float(within_pct))) / 100.0
        return lo + (hi - lo) * within

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        self._badge = QLabel("ONLINE")
        self._badge.setObjectName("onlineBadge")
        self._header = QLabel("Inspection in progress")
        self._header.setObjectName("onlineHeader")
        header.addWidget(self._badge, 0, Qt.AlignVCenter)
        header.addWidget(self._header, 0, Qt.AlignVCenter)
        header.addStretch()
        self._pct = QLabel("0%")
        self._pct.setObjectName("onlinePct")
        header.addWidget(self._pct, 0, Qt.AlignVCenter)
        root.addLayout(header)

        self._title = QLabel("Loading")
        self._title.setObjectName("onlineStageTitle")
        self._subtitle = QLabel("Reading volume data")
        self._subtitle.setObjectName("onlineStageSub")
        root.addWidget(self._title)
        root.addWidget(self._subtitle)

        self._bar = _SmoothBar()
        root.addWidget(self._bar)

        self._steps = QLabel("")
        self._steps.setObjectName("onlineSteps")
        self._steps.setAlignment(Qt.AlignCenter)
        root.addWidget(self._steps)

    def _apply_style(self):
        t = SemiconductorTheme
        self.setStyleSheet(f"""
            QDialog#OnlinePipelineProgress {{
                background-color: {t.EL2};
                border: 1px solid {t.BORDER_DEFAULT};
                border-radius: 12px;
            }}
            QLabel#onlineBadge {{
                background-color: rgba(34, 174, 209, 0.18);
                color: {t.PRIMARY_DEFAULT};
                border: 1px solid rgba(34, 174, 209, 0.45);
                border-radius: 4px;
                padding: 2px 7px;
                font-size: 9px;
                font-weight: 700;
                letter-spacing: 1px;
            }}
            QLabel#onlineHeader {{
                color: {t.ON_SURFACE_SECONDARY};
                font-size: 11px;
                font-weight: 500;
            }}
            QLabel#onlinePct {{
                color: {t.PRIMARY_DEFAULT};
                font-size: 12px;
                font-weight: 700;
            }}
            QLabel#onlineStageTitle {{
                color: {t.ON_SURFACE_PRIMARY};
                font-size: 16px;
                font-weight: 700;
            }}
            QLabel#onlineStageSub {{
                color: {t.ON_SURFACE_SECONDARY};
                font-size: 11px;
            }}
            QLabel#onlineSteps {{
                color: {t.ON_SURFACE_DISABLED};
                font-size: 10px;
                font-weight: 600;
            }}
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setColor(QColor(0, 0, 0, 140))
        shadow.setOffset(0, 6)
        self.setGraphicsEffect(shadow)

    def _refresh_stage_ui(self):
        title, sub = self.STAGE_META.get(self._stage, ("Working", ""))
        self._title.setText(title)
        self._subtitle.setText(sub)
        if self._stage == "done":
            self._header.setText("Inspection complete")
        elif self._stage == "error":
            self._header.setText("Inspection failed")
        else:
            self._header.setText("Inspection in progress")

        active_idx = self._stages.index(self._stage) if self._stage in self._stages else -1
        t = SemiconductorTheme
        parts = []
        for i, key in enumerate(self._stages):
            name = self.STAGE_META[key][0]
            if key == self._stage or (self._stage == "done" and i == len(self._stages) - 1):
                parts.append(
                    f'<span style="color:{t.PRIMARY_DEFAULT};font-weight:700;">{name}</span>'
                )
            elif active_idx >= 0 and i < active_idx:
                parts.append(f'<span style="color:{t.SUCCESS_DEFAULT};">{name}</span>')
            else:
                parts.append(f'<span style="color:{t.ON_SURFACE_DISABLED};">{name}</span>')
        self._steps.setText("  ·  ".join(parts))

    def _tick_anim(self):
        if self._closed:
            self._anim.stop()
            return

        # Soft crawl within current stage while waiting for sparse DLL/TIFF callbacks
        if self._crawl and self._stage in self._stage_ranges():
            lo, hi = self._stage_ranges()[self._stage]
            soft_cap = lo + (hi - lo) * 0.88
            if self._target < soft_cap:
                # ~8 overall-% per second while idle in stage
                self._target = min(soft_cap, self._target + 0.28)

        gap = self._target - self._display
        if abs(gap) < 0.05:
            self._display = self._target
        else:
            step = max(0.35, abs(gap) * 0.16)
            if gap > 0:
                self._display = min(self._target, self._display + step)
            else:
                self._display = max(self._target, self._display - step)

        self._apply_display()

        if self._stage == "done" and self._display >= 99.5:
            self._display = 100.0
            self._apply_display()

    def _apply_display(self):
        pct = max(0.0, min(100.0, self._display))
        self._pct.setText(f"{int(round(pct))}%")
        self._bar.set_pct(pct, error=self._error)

    def _center_on_parent(self):
        parent = self.parentWidget()
        if parent is not None:
            top_left = parent.mapToGlobal(parent.rect().topLeft())
            x = top_left.x() + (parent.width() - self.width()) // 2
            y = top_left.y() + 72
            self.move(max(0, x), max(0, y))
        else:
            screen = QApplication.primaryScreen()
            if screen:
                sg = screen.availableGeometry()
                self.move(
                    sg.x() + (sg.width() - self.width()) // 2,
                    sg.y() + 80,
                )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_display()

    def reject(self):
        if self._stage in ("done", "error"):
            super().reject()
