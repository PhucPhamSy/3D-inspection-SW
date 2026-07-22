import numpy as np
from PyQt5.QtWidgets import QWidget, QToolButton, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QSize
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath, QLinearGradient, QPolygonF,
    QIcon, QPixmap,
)
import vtk


# ── Dragonfly-style opacity curve shapes ─────────────────────────────────────
# t ∈ [0, 1] → opacity ∈ [0, 1]

OPACITY_SHAPES = (
    "linear",          # /
    "inverse_linear",  # \
    "s_curve",         # soft S
    "reverse_s",       # reverse S
    "peak",            # inverted-V / tent (high in middle)
    "valley",          # V (high at ends)
)

OPACITY_SHAPE_TIPS = {
    "linear": "Linear ramp (low → high)",
    "inverse_linear": "Inverse linear (high → low)",
    "s_curve": "S-curve (slow start, fast mid)",
    "reverse_s": "Reverse S-curve",
    "peak": "Peak / tent (opaque mid-range)",
    "valley": "Valley (opaque ends)",
}


def eval_opacity_shape(shape, t):
    """Evaluate a named opacity shape at normalised t ∈ [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    if shape == "inverse_linear":
        return 1.0 - t
    if shape == "s_curve":
        # smoothstep-like sigmoid
        x = (t - 0.5) * 8.0
        return float(1.0 / (1.0 + np.exp(-x)))
    if shape == "reverse_s":
        x = (t - 0.5) * 8.0
        return float(1.0 - 1.0 / (1.0 + np.exp(-x)))
    if shape == "peak":
        return float(1.0 - abs(2.0 * t - 1.0))
    if shape == "valley":
        return float(abs(2.0 * t - 1.0))
    # linear (default)
    return t


def sample_opacity_curve(shape="linear", gamma=1.0, n=33, opacity_scale=1.0):
    """
    Sample opacity curve as list[(norm_x, opacity)].
    Gamma warps the shape output: y = shape(t) ** gamma  (gamma>0).
    """
    gamma = max(0.05, float(gamma))
    scale = float(np.clip(opacity_scale, 0.0, 1.0))
    pts = []
    for i in range(n):
        t = i / max(1, n - 1)
        y = eval_opacity_shape(shape, t)
        y = float(y) ** gamma
        pts.append((t, float(np.clip(y * scale, 0.0, 1.0))))
    return pts


def _paint_curve_icon(shape, size=28, selected=False):
    """Paint a Dragonfly-like yellow-on-dark curve icon."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    bg = QColor(40, 42, 48) if not selected else QColor(55, 70, 95)
    border = QColor(255, 210, 0) if selected else QColor(90, 95, 105)
    p.setPen(QPen(border, 1.2))
    p.setBrush(bg)
    p.drawRoundedRect(1, 1, size - 2, size - 2, 3, 3)

    # sample curve
    margin = 5
    w = size - 2 * margin
    h = size - 2 * margin
    path = QPainterPath()
    n = 24
    for i in range(n):
        t = i / (n - 1)
        y = eval_opacity_shape(shape, t)
        px = margin + t * w
        py = margin + (1.0 - y) * h
        if i == 0:
            path.moveTo(px, py)
        else:
            path.lineTo(px, py)
    p.setPen(QPen(QColor(255, 220, 60), 1.8))
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)
    p.end()
    return pm


def make_opacity_shape_icon(shape, size=28, selected=False):
    return QIcon(_paint_curve_icon(shape, size=size, selected=selected))


def _paint_color_map_icon(kind, size=28, selected=False):
    """kind: 'normal' | 'invert' — bar-style color mapping like Dragonfly."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    bg = QColor(40, 42, 48) if not selected else QColor(55, 70, 95)
    border = QColor(255, 210, 0) if selected else QColor(90, 95, 105)
    p.setPen(QPen(border, 1.2))
    p.setBrush(bg)
    p.drawRoundedRect(1, 1, size - 2, size - 2, 3, 3)

    # three vertical bars of increasing/decreasing brightness
    bars = 4
    gap = 2
    total_gap = gap * (bars + 1)
    bw = max(2, (size - total_gap) // bars)
    x0 = (size - (bars * bw + (bars - 1) * gap)) // 2
    for i in range(bars):
        if kind == "invert":
            # tall dark → short bright (inverted emphasis)
            frac = 1.0 - i / (bars - 1)
        else:
            frac = (i + 1) / bars
        bh = max(4, int((size - 10) * (0.35 + 0.65 * frac)))
        gray = int(40 + 200 * (i / (bars - 1) if kind != "invert" else 1.0 - i / (bars - 1)))
        if kind == "invert":
            gray = int(40 + 200 * (1.0 - i / (bars - 1)))
        else:
            gray = int(40 + 200 * (i / (bars - 1)))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(gray, gray, gray))
        x = x0 + i * (bw + gap)
        y = (size - bh) // 2
        p.drawRoundedRect(x, y, bw, bh, 1, 1)
    p.end()
    return pm


def make_color_map_icon(kind, size=28, selected=False):
    return QIcon(_paint_color_map_icon(kind, size=size, selected=selected))


def make_icon_toolbutton(icon, tooltip, parent=None, checkable=True):
    """Compact Dragonfly-style icon tool button."""
    btn = QToolButton(parent)
    btn.setIcon(icon)
    btn.setIconSize(QSize(28, 28))
    btn.setFixedSize(32, 32)
    btn.setCheckable(checkable)
    btn.setAutoRaise(False)
    btn.setToolTip(tooltip)
    btn.setStyleSheet(
        "QToolButton {"
        "  background: #2a2c32; border: 1px solid #555960; border-radius: 3px;"
        "  padding: 1px;"
        "}"
        "QToolButton:hover { border-color: #888; background: #353840; }"
        "QToolButton:checked {"
        "  border: 1px solid #ffd200; background: #3a4558;"
        "}"
    )
    return btn


class TransferFunctionWidget(QWidget):
    """
    Hybrid Transfer Function / Window-Level editor (Dragonfly-inspired).

    Modes
    -----
    "wl"  Window Leveling — yellow Min/Max bars + opacity curve shape
          (linear / S / peak … + gamma). Drag bars, or pan the window.
    "tf"  Transfer Function — full opacity spline with add/move/delete points,
          plus the same Min/Max range bars.

    Signals
    -------
    opacityChanged : list[(norm_x, opacity)]
    rangeChanged   : (norm_min, norm_max) in 0..1
    modeChanged    : "wl" | "tf"
    """
    opacityChanged = pyqtSignal(list)
    rangeChanged   = pyqtSignal(float, float)
    modeChanged    = pyqtSignal(str)

    MODE_WL = "wl"
    MODE_TF = "tf"

    _RAMP_H  = 14   # LUT ramp strip height at bottom
    _BAR_HIT = 8    # half-width hit zone for vertical bars (px)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMinimumWidth(200)
        self.setMouseTracking(True)

        # Histogram (normalised 0..1)
        self.histogram = None
        self._raw_counts = None
        self._log_y = False
        self.hist_max = 1.0

        # Opacity control points in normalised coords [0..1, 0..1]  (TF mode)
        self.points = [QPointF(0.0, 0.0), QPointF(1.0, 1.0)]
        self.active_point_idx = None
        self.point_radius = 6

        # ── Min / Max range bars (normalised 0..1) ──
        self._range_min = 0.0
        self._range_max = 1.0
        self._dragging_bar = None   # 'min', 'max', 'window', or None
        self._window_drag_origin = None  # (x, range_min, range_max) for pan

        # LUT color stops for the bottom ramp [(pos, QColor), ...]
        self._lut_stops = [(0.0, QColor(0, 0, 0)), (1.0, QColor(255, 255, 255))]

        # W/L opacity mapping (Dragonfly)
        self._opacity_shape = "linear"
        self._opacity_gamma = 1.0
        self._opacity_scale = 1.0
        self._color_invert = False

        # Default: Dragonfly-style Window Leveling
        self._mode = self.MODE_WL

    # ───────────────────── public API ─────────────────────

    def mode(self):
        return self._mode

    def set_mode(self, mode):
        """Switch between 'wl' (Window Leveling) and 'tf' (Transfer Function)."""
        mode = (mode or self.MODE_WL).lower()
        if mode not in (self.MODE_WL, self.MODE_TF):
            mode = self.MODE_WL
        if mode == self._mode:
            return
        self._mode = mode
        self.active_point_idx = None
        self._dragging_bar = None
        # Entering TF from W/L: seed a linear ramp if still default 2-point line
        if mode == self.MODE_TF:
            self._ensure_tf_points_match_range()
        self.modeChanged.emit(self._mode)
        self.opacityChanged.emit(self.get_effective_points())
        self.update()

    def is_wl_mode(self):
        return self._mode == self.MODE_WL

    def is_tf_mode(self):
        return self._mode == self.MODE_TF

    def set_histogram(self, volume_data):
        if volume_data is None:
            return
        stride = max(1, min(volume_data.shape) // 20)
        sample = volume_data[::stride, ::stride, ::stride].flatten()
        counts, _ = np.histogram(sample, bins=100, range=(0, np.max(sample)))
        self._raw_counts = counts
        self._update_histogram()

    def set_log_y(self, use_log):
        self._log_y = use_log
        self._update_histogram()

    def _update_histogram(self):
        if self._raw_counts is None:
            return
        counts = self._raw_counts.astype(np.float64)
        if self._log_y:
            counts = np.log1p(counts)
        mx = np.max(counts)
        self.histogram = counts / mx if mx > 0 else np.zeros(len(counts))
        self.hist_max = mx
        self.update()

    def get_points(self):
        """Raw spline control points (TF mode storage)."""
        return [(p.x(), p.y()) for p in sorted(self.points, key=lambda p: p.x())]

    def get_effective_points(self):
        """
        Opacity points actually applied to the volume.
        W/L mode → sampled curve from shape + gamma + opacity scale.
        TF mode  → user spline control points.
        """
        if self._mode == self.MODE_WL:
            return sample_opacity_curve(
                self._opacity_shape,
                gamma=self._opacity_gamma,
                n=33,
                opacity_scale=self._opacity_scale,
            )
        return self.get_points()

    def set_opacity_shape(self, shape):
        if shape not in OPACITY_SHAPES:
            shape = "linear"
        if shape == self._opacity_shape:
            return
        self._opacity_shape = shape
        self.opacityChanged.emit(self.get_effective_points())
        self.update()

    def opacity_shape(self):
        return self._opacity_shape

    def set_opacity_gamma(self, gamma):
        g = max(0.05, float(gamma))
        if abs(g - self._opacity_gamma) < 1e-6:
            return
        self._opacity_gamma = g
        self.opacityChanged.emit(self.get_effective_points())
        self.update()

    def opacity_gamma(self):
        return self._opacity_gamma

    def set_opacity_scale(self, scale):
        s = float(np.clip(scale, 0.0, 1.0))
        if abs(s - self._opacity_scale) < 1e-6:
            return
        self._opacity_scale = s
        self.opacityChanged.emit(self.get_effective_points())
        self.update()

    def opacity_scale(self):
        return self._opacity_scale

    def set_color_invert(self, invert):
        self._color_invert = bool(invert)
        self.update()

    def color_invert(self):
        return self._color_invert

    def add_point(self, norm_x, norm_y):
        self.points.append(QPointF(norm_x, norm_y))
        self.points.sort(key=lambda p: p.x())
        self.opacityChanged.emit(self.get_effective_points())
        self.update()

    def set_range(self, norm_min, norm_max):
        """Set the min/max bar positions (0..1 normalised)."""
        self._range_min = float(np.clip(norm_min, 0.0, 1.0))
        self._range_max = float(np.clip(norm_max, 0.0, 1.0))
        if self._range_max <= self._range_min:
            self._range_max = min(1.0, self._range_min + 0.005)
        self.update()

    def get_range(self):
        return self._range_min, self._range_max

    def set_lut_stops(self, stops):
        """stops = [(pos, QColor), ...]"""
        self._lut_stops = list(stops) if stops else self._lut_stops
        self.update()

    def _ensure_tf_points_match_range(self):
        """When switching to TF, seed spline from current W/L curve if still default."""
        if len(self.points) == 2:
            xs = sorted(p.x() for p in self.points)
            if abs(xs[0] - 0.0) < 1e-6 and abs(xs[1] - 1.0) < 1e-6:
                # Seed a few samples from the W/L curve so TF starts matching look
                samples = sample_opacity_curve(
                    self._opacity_shape, self._opacity_gamma, n=5,
                    opacity_scale=self._opacity_scale,
                )
                self.points = [QPointF(x, y) for x, y in samples]

    # ───────────────────── painting ─────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        ramp_h = self._RAMP_H
        hist_h = h - ramp_h

        # ── background ──
        painter.fillRect(self.rect(), QColor(30, 30, 35))

        # ── histogram bars ──
        if self.histogram is not None:
            n = len(self.histogram)
            bin_w = w / n
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(100, 100, 120, 150))
            for i, val in enumerate(self.histogram):
                bh = val * hist_h * 0.88
                painter.drawRect(int(i * bin_w), int(hist_h - bh),
                                 int(max(1, bin_w)), int(bh))

        # ── grid ──
        painter.setPen(QPen(QColor(60, 60, 60), 1, Qt.DashLine))
        for i in range(1, 4):
            gx = int(w * i / 4.0)
            painter.drawLine(gx, 0, gx, hist_h)
            gy = int(hist_h * i / 4.0)
            painter.drawLine(0, gy, w, gy)

        min_x = int(self._range_min * w)
        max_x = int(self._range_max * w)

        if self._mode == self.MODE_WL:
            self._paint_wl_ramp(painter, w, hist_h, min_x, max_x)
        else:
            self._paint_tf_spline(painter, w, hist_h)

        # ── Min / Max yellow vertical bars (both modes) ──
        for bx in (min_x, max_x):
            painter.setPen(QPen(QColor(255, 210, 0), 2))
            painter.drawLine(bx, 0, bx, hist_h)
            # triangle handle at top
            tri_t = QPolygonF([QPointF(bx, 0),
                               QPointF(bx - 6, 9),
                               QPointF(bx + 6, 9)])
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 210, 0))
            painter.drawPolygon(tri_t)
            # triangle handle at bottom
            tri_b = QPolygonF([QPointF(bx, hist_h),
                               QPointF(bx - 6, hist_h - 9),
                               QPointF(bx + 6, hist_h - 9)])
            painter.drawPolygon(tri_b)

        # Dragonfly-style yellow frame around active window (W/L)
        if self._mode == self.MODE_WL:
            painter.setPen(QPen(QColor(255, 210, 0), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(min_x, 0, max(1, max_x - min_x), hist_h)

        # ── dim regions outside min/max ──
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 100))
        if min_x > 0:
            painter.drawRect(0, 0, min_x, hist_h)
        if max_x < w:
            painter.drawRect(max_x, 0, w - max_x, hist_h)

        # ── LUT colour ramp at bottom ──
        ramp_y = hist_h
        stops = list(self._lut_stops)
        if self._color_invert and stops:
            stops = [(1.0 - pos, c) for pos, c in stops]
            stops.sort(key=lambda s: s[0])
        if self._mode == self.MODE_WL:
            grad = QLinearGradient(min_x, 0, max(min_x + 1, max_x), 0)
        else:
            grad = QLinearGradient(0, 0, w, 0)
        for pos, color in stops:
            grad.setColorAt(float(np.clip(pos, 0.0, 1.0)), color)
        painter.setBrush(QBrush(grad))
        if self._mode == self.MODE_WL:
            painter.drawRect(min_x, ramp_y, max(1, max_x - min_x), ramp_h)
            painter.setBrush(QColor(15, 15, 18))
            if min_x > 0:
                painter.drawRect(0, ramp_y, min_x, ramp_h)
            if max_x < w:
                painter.drawRect(max_x, ramp_y, w - max_x, ramp_h)
        else:
            painter.drawRect(0, ramp_y, w, ramp_h)
            painter.setBrush(QColor(0, 0, 0, 160))
            if min_x > 0:
                painter.drawRect(0, ramp_y, min_x, ramp_h)
            if max_x < w:
                painter.drawRect(max_x, ramp_y, w - max_x, ramp_h)

        painter.end()

    def _paint_wl_ramp(self, painter, w, hist_h, min_x, max_x):
        """Dragonfly Window Leveling: opacity curve across the yellow window."""
        span = max(1, max_x - min_x)
        samples = sample_opacity_curve(
            self._opacity_shape,
            gamma=self._opacity_gamma,
            n=48,
            opacity_scale=self._opacity_scale,
        )
        path = QPainterPath()
        # flat zero left of min
        path.moveTo(0, hist_h)
        path.lineTo(min_x, hist_h)
        for t, y in samples:
            px = min_x + t * span
            py = hist_h - y * (hist_h - 2)
            path.lineTo(px, py)
        # hold final opacity to right edge
        last_y = samples[-1][1] if samples else 1.0
        path.lineTo(w, hist_h - last_y * (hist_h - 2))

        painter.setPen(QPen(QColor(210, 210, 230), 1.8))
        painter.drawPath(path)

        fill = QPainterPath(path)
        fill.lineTo(w, hist_h)
        fill.lineTo(0, hist_h)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(200, 200, 250, 40))
        painter.drawPath(fill)

    def _paint_tf_spline(self, painter, w, hist_h):
        """Full opacity spline with control-point handles."""
        sorted_pts = sorted(self.points, key=lambda p: p.x())
        path = QPainterPath()
        if sorted_pts:
            pt0 = sorted_pts[0]
            path.moveTo(pt0.x() * w, hist_h - pt0.y() * hist_h)
            for pt in sorted_pts[1:]:
                path.lineTo(pt.x() * w, hist_h - pt.y() * hist_h)

        painter.setPen(QPen(QColor(200, 200, 250), 2))
        painter.drawPath(path)

        fill = QPainterPath(path)
        fill.lineTo(w, hist_h)
        fill.lineTo(0, hist_h)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(200, 200, 250, 35))
        painter.drawPath(fill)

        painter.setPen(QPen(QColor(255, 255, 255), 1))
        for i, pt in enumerate(sorted_pts):
            painter.setBrush(QColor(255, 100, 100) if i == self.active_point_idx
                             else QColor(100, 200, 255))
            px = pt.x() * w
            py = hist_h - pt.y() * hist_h
            painter.drawEllipse(QPointF(px, py), self.point_radius, self.point_radius)

    # ───────────────── mouse interaction ─────────────────

    def _bar_at(self, x):
        """Return 'min', 'max', or None."""
        w = max(1, self.width())
        min_x = self._range_min * w
        max_x = self._range_max * w
        if abs(x - min_x) <= self._BAR_HIT:
            return 'min'
        if abs(x - max_x) <= self._BAR_HIT:
            return 'max'
        return None

    def _inside_window(self, x):
        w = max(1, self.width())
        min_x = self._range_min * w
        max_x = self._range_max * w
        return min_x + self._BAR_HIT < x < max_x - self._BAR_HIT

    def get_point_at(self, pos):
        w = self.width()
        h = self.height() - self._RAMP_H
        for i, pt in enumerate(self.points):
            px = pt.x() * w
            py = h - pt.y() * h
            if (pos.x() - px)**2 + (pos.y() - py)**2 <= (self.point_radius * 2)**2:
                return i
        return -1

    def mousePressEvent(self, event):
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return

        hist_h = self.height() - self._RAMP_H
        x = event.pos().x()

        # Priority 1: yellow bars (both modes)
        bar = self._bar_at(x)
        if event.button() == Qt.LeftButton and bar is not None:
            self._dragging_bar = bar
            return

        # W/L: drag middle of window to pan (Level)
        if (self._mode == self.MODE_WL
                and event.button() == Qt.LeftButton
                and event.pos().y() <= hist_h
                and self._inside_window(x)):
            self._dragging_bar = 'window'
            self._window_drag_origin = (x, self._range_min, self._range_max)
            return

        # TF-only: control points
        if self._mode != self.MODE_TF:
            return

        idx = self.get_point_at(event.pos())
        if event.button() == Qt.LeftButton:
            if idx != -1:
                self.active_point_idx = idx
            else:
                if event.pos().y() <= hist_h:
                    nx = np.clip(event.pos().x() / self.width(), 0.0, 1.0)
                    ny = np.clip(1.0 - event.pos().y() / hist_h, 0.0, 1.0)
                    self.add_point(nx, ny)
                    for j, p in enumerate(self.points):
                        if abs(p.x() - nx) < 1e-6 and abs(p.y() - ny) < 1e-6:
                            self.active_point_idx = j
                            break
        elif event.button() == Qt.RightButton:
            if idx != -1 and len(self.points) > 2:
                self.points.pop(idx)
                self.active_point_idx = None
                self.opacityChanged.emit(self.get_effective_points())
                self.update()

    def mouseMoveEvent(self, event):
        w = max(1, self.width())
        hist_h = self.height() - self._RAMP_H

        # ── dragging yellow bars / window ──
        if self._dragging_bar is not None and event.buttons() & Qt.LeftButton:
            if self._dragging_bar == 'window' and self._window_drag_origin is not None:
                ox, o_min, o_max = self._window_drag_origin
                dx = (event.pos().x() - ox) / w
                width = o_max - o_min
                new_min = o_min + dx
                new_max = o_max + dx
                if new_min < 0.0:
                    new_min = 0.0
                    new_max = width
                if new_max > 1.0:
                    new_max = 1.0
                    new_min = 1.0 - width
                self._range_min = float(new_min)
                self._range_max = float(new_max)
            else:
                nx = float(np.clip(event.pos().x() / w, 0.0, 1.0))
                if self._dragging_bar == 'min':
                    nx = min(nx, self._range_max - 0.005)
                    self._range_min = nx
                elif self._dragging_bar == 'max':
                    nx = max(nx, self._range_min + 0.005)
                    self._range_max = nx
            self.update()
            self.rangeChanged.emit(self._range_min, self._range_max)
            # W/L ramp is pure range → re-emit opacity so volume stays in sync
            if self._mode == self.MODE_WL:
                self.opacityChanged.emit(self.get_effective_points())
            return

        # ── dragging an opacity point (TF only) ──
        if (self._mode == self.MODE_TF
                and self.active_point_idx is not None
                and event.buttons() & Qt.LeftButton):
            nx = np.clip(event.pos().x() / w, 0.0, 1.0)
            ny = np.clip(1.0 - event.pos().y() / hist_h, 0.0, 1.0)

            if self.active_point_idx == 0:
                nx = 0.0
            elif self.active_point_idx == len(self.points) - 1:
                nx = 1.0

            if self.active_point_idx > 0:
                nx = max(nx, self.points[self.active_point_idx - 1].x())
            if self.active_point_idx < len(self.points) - 1:
                nx = min(nx, self.points[self.active_point_idx + 1].x())

            self.points[self.active_point_idx].setX(nx)
            self.points[self.active_point_idx].setY(ny)
            self.opacityChanged.emit(self.get_effective_points())
            self.update()
            return

        # ── hover cursor ──
        bar = self._bar_at(event.pos().x())
        if bar:
            self.setCursor(Qt.SizeHorCursor)
        elif (self._mode == self.MODE_WL
              and event.pos().y() <= hist_h
              and self._inside_window(event.pos().x())):
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        self._dragging_bar = None
        self._window_drag_origin = None
        self.active_point_idx = None
        self.update()


class ColorGradientWidget(QWidget):
    colorChanged = pyqtSignal(list) # Emits list of (pos, QColor)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setMinimumWidth(200)
        
        # List of (normalized_pos, QColor)
        self.color_stops = [
            (0.0, QColor(0, 0, 0)),
            (0.3, QColor(64, 64, 64)),
            (0.6, QColor(191, 191, 191)),
            (0.95, QColor(242, 242, 242)),
            (1.0, QColor(255, 255, 255))
        ]
        self.active_stop = None

    def get_stops(self):
        return sorted(self.color_stops, key=lambda x: x[0])

    def add_stop(self, pos, color):
        self.color_stops.append((pos, color))
        self.color_stops.sort(key=lambda x: x[0])
        self.colorChanged.emit(self.get_stops())
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        w = rect.width()
        
        grad = QLinearGradient(0, 0, w, 0)
        for pos, color in self.color_stops:
            grad.setColorAt(pos, color)
            
        painter.fillRect(rect, grad)
        
        # Draw handles
        painter.setRenderHint(QPainter.Antialiasing)
        for i, (pos, color) in enumerate(self.color_stops):
            x = int(pos * w)
            # Draw triangle pointing up
            poly = QPolygonF([QPointF(x, 15), QPointF(x-5, 30), QPointF(x+5, 30)])
            if i == self.active_stop:
                painter.setPen(QPen(Qt.white, 2))
            else:
                painter.setPen(QPen(Qt.black, 1))
            painter.setBrush(color)
            painter.drawPolygon(poly)

    def get_stop_at(self, x):
        w = self.width()
        for i, (pos, _) in enumerate(self.color_stops):
            px = pos * w
            if abs(x - px) <= 6:
                return i
        return -1

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            idx = self.get_stop_at(event.pos().x())
            if idx != -1:
                self.active_stop = idx
            else:
                from PyQt5.QtWidgets import QColorDialog
                color = QColorDialog.getColor()
                if color.isValid():
                    nx = np.clip(event.pos().x() / self.width(), 0.0, 1.0)
                    self.add_stop(nx, color)
                    # Find new index
                    self.active_stop = next(i for i, (p, c) in enumerate(self.color_stops) if p == nx and c == color)
            self.update()
        elif event.button() == Qt.RightButton:
            idx = self.get_stop_at(event.pos().x())
            # Keep at least two stops
            if idx != -1 and len(self.color_stops) > 2:
                # Don't delete end stops at exactly 0 and 1
                if self.color_stops[idx][0] not in [0.0, 1.0]:
                    self.color_stops.pop(idx)
                    self.active_stop = None
                    self.colorChanged.emit(self.get_stops())
                    self.update()

    def mouseDoubleClickEvent(self, event):
        idx = self.get_stop_at(event.pos().x())
        if idx != -1:
            from PyQt5.QtWidgets import QColorDialog
            color = QColorDialog.getColor(self.color_stops[idx][1])
            if color.isValid():
                pos = self.color_stops[idx][0]
                self.color_stops[idx] = (pos, color)
                self.colorChanged.emit(self.get_stops())
                self.update()

    def mouseMoveEvent(self, event):
        if self.active_stop is not None and event.buttons() & Qt.LeftButton:
            # Don't move endpoints
            if self.color_stops[self.active_stop][0] in [0.0, 1.0] and (self.active_stop == 0 or self.active_stop == len(self.color_stops)-1):
                return
                
            nx = np.clip(event.pos().x() / self.width(), 0.0, 1.0)
            
            # Prevent passing neighbors
            if self.active_stop > 0:
                nx = max(nx, self.color_stops[self.active_stop - 1][0] + 0.01)
            if self.active_stop < len(self.color_stops) - 1:
                nx = min(nx, self.color_stops[self.active_stop + 1][0] - 0.01)
                
            color = self.color_stops[self.active_stop][1]
            self.color_stops[self.active_stop] = (nx, color)
            self.colorChanged.emit(self.get_stops())
            self.update()

    def mouseReleaseEvent(self, event):
        self.active_stop = None
        self.update()


class HistogramWLWidget(QWidget):
    """Dragonfly-style histogram with draggable Min/Max markers and LUT ramp overlay.
    
    Emits rangeChanged(min_val, max_val) when the user drags the markers.
    The markers are displayed as vertical bars with triangular handles.
    """
    rangeChanged = pyqtSignal(int, int)  # (min_intensity, max_intensity)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setMinimumWidth(200)
        self.setCursor(Qt.ArrowCursor)

        # Data range
        self._data_min = 0
        self._data_max = 65535
        self._wl_min = 0
        self._wl_max = 65535

        # Histogram bins (normalized 0..1)
        self._hist_bins = None
        self._hist_max_count = 1.0

        # Dragging state: 'min', 'max', 'window', or None
        self._dragging = None
        self._handle_w = 8  # half-width of the handle hot zone
        self._window_drag_origin = None  # (x_px, wl_min, wl_max) when panning

        # LUT color stops for the bottom ramp (default grayscale)
        self._lut_stops = [(0.0, QColor(0,0,0)), (1.0, QColor(255,255,255))]

        # Transfer function line (opacity spline, normalized 0..1)
        self._tf_line = [(0.0, 1.0), (1.0, 0.0)]  # default: ramp down

    # ── Public API ──────────────────────────────────────────────

    def set_histogram(self, volume_data, data_min=None, data_max=None):
        """Compute histogram from volume data.

        Optional data_min/data_max fix the plotted intensity axis (useful when
        the official volume range was estimated separately from a subsample).
        """
        if volume_data is None:
            return
        stride = max(1, min(volume_data.shape) // 20)
        sample = volume_data[::stride, ::stride, ::stride].flatten()
        if data_min is None:
            data_min = float(sample.min())
        if data_max is None:
            data_max = float(sample.max())
        self._data_min = int(data_min)
        self._data_max = int(data_max)
        if self._data_max <= self._data_min:
            self._data_max = self._data_min + 1
        counts, _ = np.histogram(
            sample, bins=128, range=(self._data_min, self._data_max)
        )
        mx = counts.max()
        self._hist_bins = counts / mx if mx > 0 else np.zeros(128)
        self._hist_max_count = mx
        # Keep selected window inside the new data extent
        self._wl_min = int(np.clip(self._wl_min, self._data_min, self._data_max - 1))
        self._wl_max = int(np.clip(self._wl_max, self._wl_min + 1, self._data_max))
        self.update()

    def set_data_range(self, data_min, data_max):
        """Set the plotted intensity axis without recomputing the histogram."""
        self._data_min = int(data_min)
        self._data_max = int(data_max)
        if self._data_max <= self._data_min:
            self._data_max = self._data_min + 1
        self._wl_min = int(np.clip(self._wl_min, self._data_min, self._data_max - 1))
        self._wl_max = int(np.clip(self._wl_max, self._wl_min + 1, self._data_max))
        self.update()

    def set_range(self, v_min, v_max):
        """Set the W/L min and max (in data intensity units)."""
        self._wl_min = int(v_min)
        self._wl_max = int(v_max)
        if self._wl_max <= self._wl_min:
            self._wl_max = self._wl_min + 1
        self.update()

    def set_lut_stops(self, stops):
        """Set LUT color stops for the bottom ramp. stops = [(pos, QColor), ...]"""
        self._lut_stops = stops
        self.update()

    def set_tf_line(self, points):
        """Set the TF opacity line. points = [(norm_x, norm_y), ...]"""
        self._tf_line = points
        self.update()

    def get_range(self):
        return self._wl_min, self._wl_max

    # ── Internal helpers ────────────────────────────────────────

    def _val_to_x(self, val):
        """Map intensity value to pixel x."""
        rng = self._data_max - self._data_min
        if rng <= 0:
            rng = 1
        return int((val - self._data_min) / rng * (self.width() - 1))

    def _x_to_val(self, x):
        """Map pixel x to intensity value."""
        rng = self._data_max - self._data_min
        if rng <= 0:
            rng = 1
        val = self._data_min + x / max(1, self.width() - 1) * rng
        return int(np.clip(val, self._data_min, self._data_max))

    # ── Painting ────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        ramp_h = 16  # height of LUT ramp at bottom
        hist_h = h - ramp_h - 4  # histogram area height

        # Background
        painter.fillRect(self.rect(), QColor(25, 28, 32))

        # ── Histogram bars ──
        if self._hist_bins is not None:
            n = len(self._hist_bins)
            bin_w = w / n
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(120, 125, 135, 160))
            for i, val in enumerate(self._hist_bins):
                bh = val * hist_h * 0.88
                painter.drawRect(int(i * bin_w), int(hist_h - bh), int(max(1, bin_w)), int(bh))

        # ── Transfer function line (diagonal from min→max) ──
        min_x = self._val_to_x(self._wl_min)
        max_x = self._val_to_x(self._wl_max)
        tf_pen = QPen(QColor(160, 190, 220, 200), 1.5)
        painter.setPen(tf_pen)
        # Draw the TF ramp line from (min_x, hist_h) to (max_x, 0) with fill
        path = QPainterPath()
        path.moveTo(0, hist_h)
        path.lineTo(min_x, hist_h)
        path.lineTo(max_x, 4)
        path.lineTo(w, 4)
        painter.drawPath(path)
        # Fill under the TF line
        fill_path = QPainterPath(path)
        fill_path.lineTo(w, hist_h)
        fill_path.lineTo(0, hist_h)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(140, 180, 220, 35))
        painter.drawPath(fill_path)

        # ── Min marker (cyan/teal) ──
        self._draw_marker(painter, min_x, hist_h, QColor(0, 200, 220),
                          f"Min {self._wl_min}", align_left=True)

        # ── Max marker (orange) ──
        self._draw_marker(painter, max_x, hist_h, QColor(255, 165, 0),
                          f"Max {self._wl_max}", align_left=False)

        # ── LUT color ramp at bottom ──
        ramp_y = h - ramp_h
        grad = QLinearGradient(min_x, 0, max_x, 0)
        for pos, color in self._lut_stops:
            grad.setColorAt(pos, color)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(grad))
        painter.drawRect(min_x, ramp_y, max(1, max_x - min_x), ramp_h)
        # Dark fill outside the ramp
        painter.setBrush(QColor(15, 15, 18))
        if min_x > 0:
            painter.drawRect(0, ramp_y, min_x, ramp_h)
        if max_x < w:
            painter.drawRect(max_x, ramp_y, w - max_x, ramp_h)

        painter.end()

    def _draw_marker(self, painter, x, hist_h, color, label, align_left):
        """Draw a vertical marker bar with label badge."""
        # Vertical line
        painter.setPen(QPen(color, 2))
        painter.drawLine(x, 0, x, hist_h)

        # Triangle handle at bottom
        tri = QPolygonF([
            QPointF(x, hist_h),
            QPointF(x - 5, hist_h + 8),
            QPointF(x + 5, hist_h + 8),
        ])
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(tri)

        # Label badge
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label) + 8
        th = fm.height() + 4
        if align_left:
            lx = max(0, x - 2)
        else:
            lx = min(self.width() - tw, x - tw + 2)
        ly = 2
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 180))
        painter.drawRoundedRect(lx, ly, tw, th, 3, 3)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(lx + 4, ly + fm.ascent() + 1, label)

    # ── Mouse interaction ───────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        x = event.pos().x()
        min_x = self._val_to_x(self._wl_min)
        max_x = self._val_to_x(self._wl_max)
        if abs(x - min_x) <= self._handle_w:
            self._dragging = 'min'
            self._window_drag_origin = None
        elif abs(x - max_x) <= self._handle_w:
            self._dragging = 'max'
            self._window_drag_origin = None
        elif min_x < x < max_x:
            # Drag middle of window → pan Level (keep Width)
            self._dragging = 'window'
            self._window_drag_origin = (x, self._wl_min, self._wl_max)
        else:
            self._dragging = None
            self._window_drag_origin = None

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            # Update cursor hint
            x = event.pos().x()
            min_x = self._val_to_x(self._wl_min)
            max_x = self._val_to_x(self._wl_max)
            if abs(x - min_x) <= self._handle_w or abs(x - max_x) <= self._handle_w:
                self.setCursor(Qt.SizeHorCursor)
            elif min_x < x < max_x:
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        if self._dragging == 'window' and self._window_drag_origin is not None:
            ox, omin, omax = self._window_drag_origin
            dx = event.pos().x() - ox
            dval = self._x_to_val(ox + dx) - self._x_to_val(ox)
            width = omax - omin
            new_min = int(np.clip(omin + dval, self._data_min, self._data_max - width))
            new_max = new_min + width
            self._wl_min = new_min
            self._wl_max = new_max
        else:
            val = self._x_to_val(event.pos().x())
            if self._dragging == 'min':
                val = min(val, self._wl_max - 1)
                self._wl_min = val
            elif self._dragging == 'max':
                val = max(val, self._wl_min + 1)
                self._wl_max = val
        self.update()
        self.rangeChanged.emit(self._wl_min, self._wl_max)

    def mouseReleaseEvent(self, event):
        self._dragging = None
        self._window_drag_origin = None
