"""
Online-mode CONTEXT panel: Wafer map (chip-in-wafer) + Chip map (FOV-in-chip).

Visual design reference: Innometry wafer-map email / 25×25 draw code
  · Bin 0 = Outside (grey) · 1 = Bad NG (red) · 8 = Good OK (green)
  · Selected chip = reticle (cluster-centroid style)
  · Active FOV on 9-point chip map = amber highlight

Does not modify MPR/3D panes — sits left of the plane grid when Online is ON.
"""
from __future__ import annotations

from typing import Optional, Tuple
from pathlib import Path

from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QBrush,
    QWheelEvent,
)
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from inno3d.core.styles import SemiconductorTheme
from inno3d.core.wafer_context import (
    BIN_NG,
    BIN_OK,
    BIN_OUTSIDE,
    BIN_PENDING,
    HostVolumePathInfo,
    NgCluster,
    WaferContext,
)


def _c(hex_or_name: str, alpha: int = 255) -> QColor:
    col = QColor(hex_or_name)
    col.setAlpha(alpha)
    return col


# Palette — high contrast for dark UI (Online CONTEXT + Batch Review share this canvas)
# Outside vs Pending were too close (#5a6570 / #3a4a5c) → redesigned.
_CLR_OUT = "#1a1f28"          # outside die (0) — near-black, barely a cell
_CLR_OUT_BORDER = "#2a3340"   # faint edge for outside only
_CLR_PENDING = "#5b7a9a"      # not inspected (2) — cool blue-slate, clearly "inside"
_CLR_PENDING_BORDER = "#7a9bb8"
_CLR_OK = "#2fd67b"           # final OK (8) — bright green
_CLR_OK_DIM = "#1fa85c"
_CLR_NG = "#f04444"           # final NG (1) — bright red
_CLR_NG_DIM = "#b83030"
_CLR_WAFER_BG = "#0c121c"     # substrate inside circle
_CLR_WAFER_RIM = "#3dd6f5"    # bright cyan wafer rim (orientation readable)
_CLR_WAFER_RIM_DIM = "#1a4a5c"
_CLR_RETICLE = "#22d3ee"      # selected chip reticle (cyan)
_CLR_FOV_MARK = "#f5a623"     # FOV / amber
_CLR_CLUSTER = "#3dd6f5"      # cluster boundary + centroid (email cyan)
_CLR_CLUSTER_SEL = "#f5d76e"  # selected cluster highlight


def _draw_reticle(
    p: QPainter,
    cx: float,
    cy: float,
    r: float,
    color: QColor,
    width: float = 1.8,
    *,
    open_center: bool = False,
):
    """Cluster-centroid style reticle (PDF: ⊕ on assigned chip).

    open_center=True  → arms leave a gap in the middle so die OK/NG color +
                        bin glyph stay visible (important when zoomed in).
    open_center=False → classic solid hub (fit / low-zoom overview).
    """
    p.save()
    pen = QPen(color, width)
    pen.setCosmetic(True)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(cx, cy), r, r)
    arm = r * 1.35
    if open_center:
        # Leave ~45% of radius clear so inspected die surface shows through
        gap = max(2.5, r * 0.45)
        p.drawLine(QPointF(cx - arm, cy), QPointF(cx - gap, cy))
        p.drawLine(QPointF(cx + gap, cy), QPointF(cx + arm, cy))
        p.drawLine(QPointF(cx, cy - arm), QPointF(cx, cy - gap))
        p.drawLine(QPointF(cx, cy + gap), QPointF(cx, cy + arm))
        # Thin ring only — no filled hub covering the die
        p.drawEllipse(QPointF(cx, cy), max(1.2, r * 0.12), max(1.2, r * 0.12))
    else:
        p.drawLine(QPointF(cx - arm, cy), QPointF(cx + arm, cy))
        p.drawLine(QPointF(cx, cy - arm), QPointF(cx, cy + arm))
        p.setBrush(QBrush(color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), max(1.5, r * 0.18), max(1.5, r * 0.18))
    p.restore()


def _draw_corner_brackets(
    p: QPainter,
    rect: QRectF,
    color: QColor,
    width: float = 2.0,
    arm_frac: float = 0.28,
) -> None:
    """Selection brackets at four corners — never covers die center (zoom-in mode)."""
    p.save()
    pen = QPen(color, width)
    pen.setCosmetic(True)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    L = min(rect.width(), rect.height()) * max(0.12, min(0.4, arm_frac))
    x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
    # TL
    p.drawLine(QPointF(x0, y0 + L), QPointF(x0, y0))
    p.drawLine(QPointF(x0, y0), QPointF(x0 + L, y0))
    # TR
    p.drawLine(QPointF(x1 - L, y0), QPointF(x1, y0))
    p.drawLine(QPointF(x1, y0), QPointF(x1, y0 + L))
    # BL
    p.drawLine(QPointF(x0, y1 - L), QPointF(x0, y1))
    p.drawLine(QPointF(x0, y1), QPointF(x0 + L, y1))
    # BR
    p.drawLine(QPointF(x1 - L, y1), QPointF(x1, y1))
    p.drawLine(QPointF(x1, y1), QPointF(x1, y1 - L))
    p.restore()


def _draw_coord_badge(
    p: QPainter,
    cx: float,
    cy: float,
    text: str,
    *,
    color: Optional[QColor] = None,
    font: Optional[QFont] = None,
) -> None:
    """Dark pill with Col,Row — drawn *after* reticle so coordinates stay readable."""
    if not text:
        return
    p.save()
    f = font or QFont("Segoe UI", 7)
    f.setBold(True)
    p.setFont(f)
    fm = QFontMetrics(f)
    tw = fm.horizontalAdvance(text) + 8
    th = max(12, fm.height() + 2)
    rect = QRectF(cx - tw * 0.5, cy - th * 0.5, tw, th)
    # Dark fill + cyan rim so badge pops over reticle / die color
    p.setPen(QPen(color or _c(_CLR_RETICLE, 230), 1.0))
    p.setBrush(_c("#0a121c", 230))
    p.drawRoundedRect(rect, 3.0, 3.0)
    p.setPen(color or _c("#e8f4ff"))
    p.drawText(rect, Qt.AlignCenter, text)
    p.restore()


def _reticle_radius(cell: float, frac: float = 0.38, cap_px: float = 14.0) -> float:
    """Scale reticle with cell but cap in screen px so zoom-in never fills the die."""
    return max(3.5, min(cell * frac, cap_px))


class _WaferMapCanvas(QWidget):
    """Circular die map with zoom / pan and live Col·Row tracking.

    Interaction
    -----------
    · Wheel …………… zoom toward cursor (0.6× – 16×)
    · Left-drag …… pan
    · Left-click … select die (if not dragged)
    · Middle / Right-drag … pan (same as left-drag)
    · Double-click … fit view (reset zoom + pan)
    · Hover ……… Col / Row badge + tooltip

    Same widget is used by Online CONTEXT and Batch Review.
    """

    chip_clicked = pyqtSignal(int, int)  # col, row 1-based
    cluster_clicked = pyqtSignal(int)    # NgCluster.id

    # Axis band (col top / row left) — tight so wafer circle grows
    _AXIS_L = 22   # left for row numbers
    _AXIS_T = 18   # top for col numbers
    _AXIS_R = 6
    _AXIS_B = 24   # bottom legend strip

    _ZOOM_MIN = 0.6
    _ZOOM_MAX = 16.0
    _ZOOM_STEP = 1.15
    _DRAG_THRESH_PX = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctx: Optional[WaferContext] = None
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._hover: Optional[Tuple[int, int]] = None  # 0-based c,r

        # View transform (zoom about base center + pan offset in screen px)
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0

        # Interaction state
        self._press_pos: Optional[QPoint] = None
        self._press_pan: Tuple[float, float] = (0.0, 0.0)
        self._press_hit: Optional[Tuple[int, int]] = None
        self._dragging: bool = False
        self._pan_button: Optional[int] = None

        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip(
            "Wheel = zoom  ·  Drag = pan  ·  Click = select die\n"
            "Double-click = fit view  ·  Hover = Col / Row"
        )

    def set_context(self, ctx: Optional[WaferContext]) -> None:
        self._ctx = ctx
        self.update()

    def reset_view(self) -> None:
        """Fit wafer to panel (zoom=1, pan=0)."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def zoom_to_selection(self, col: int = 0, row: int = 0, zoom: float = 3.0) -> None:
        """Center view on a die (1-based). Used after Online path selection."""
        if self._ctx is None:
            return
        n = int(self._ctx.map_size)
        c = int(col or self._ctx.selected_col) - 1
        r = int(row or self._ctx.selected_row) - 1
        if not (0 <= c < n and 0 <= r < n):
            return
        self._zoom = max(self._ZOOM_MIN, min(self._ZOOM_MAX, float(zoom)))
        # Place die center at view center: pan offsets base center so die maps to mid
        base_cx, base_cy, _side = self._base_center_and_side()
        # die center in base (zoom=1, pan=0) coordinates
        base_R = _side * 0.495
        base_cell = (base_R * 1.92) / max(n, 1)
        base_ox = base_cx - (n * base_cell) * 0.5
        base_oy = base_cy - (n * base_cell) * 0.5
        die_bx = base_ox + (c + 0.5) * base_cell
        die_by = base_oy + (r + 0.5) * base_cell
        # screen = base_center + (world - base_center)*zoom + pan
        # want die at (base_cx, base_cy):
        # base_cx = base_cx + (die_bx - base_cx)*zoom + pan_x  → pan_x = -(die_bx-base_cx)*zoom
        self._pan_x = -(die_bx - base_cx) * self._zoom
        self._pan_y = -(die_by - base_cy) * self._zoom
        self._clamp_pan()
        self.update()

    # ── Geometry ─────────────────────────────────────────────────
    def _base_center_and_side(self) -> Tuple[float, float, float]:
        """Fit-box center + square side (zoom=1 frame)."""
        x0 = self._AXIS_L
        y0 = self._AXIS_T
        x1 = self.width() - self._AXIS_R
        y1 = self.height() - self._AXIS_B
        uw = max(40.0, x1 - x0)
        uh = max(40.0, y1 - y0)
        side = min(uw, uh)
        cx = x0 + uw * 0.5
        cy = y0 + uh * 0.5
        return cx, cy, side

    def _layout_geom(self):
        """Return (n, cx, cy, R, cell, ox, oy) after zoom + pan."""
        n = int(self._ctx.map_size) if self._ctx else 25
        base_cx, base_cy, side = self._base_center_and_side()
        z = max(self._ZOOM_MIN, min(self._ZOOM_MAX, float(self._zoom)))
        base_R = side * 0.495
        R = base_R * z
        cell = (R * 1.92) / max(n, 1)
        cx = base_cx + self._pan_x
        cy = base_cy + self._pan_y
        ox = cx - (n * cell) * 0.5
        oy = cy - (n * cell) * 0.5
        return n, cx, cy, R, cell, ox, oy

    def _clamp_pan(self) -> None:
        """Keep at least ~25% of wafer radius inside the view box."""
        if self.width() < 10 or self.height() < 10:
            return
        _bcx, _bcy, side = self._base_center_and_side()
        z = max(self._ZOOM_MIN, min(self._ZOOM_MAX, float(self._zoom)))
        R = side * 0.495 * z
        # Allow pan so wafer can be moved but not completely lost
        max_pan = R * 0.85 + side * 0.15
        self._pan_x = max(-max_pan, min(max_pan, self._pan_x))
        self._pan_y = max(-max_pan, min(max_pan, self._pan_y))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), _c(SemiconductorTheme.BG_DARK))

        if self._ctx is None:
            p.setPen(_c(SemiconductorTheme.TEXT_DISABLED))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(self.rect(), Qt.AlignCenter, "No wafer map")
            p.end()
            return

        n, cx, cy, R, cell, ox, oy = self._layout_geom()
        bins = self._ctx.bins
        sel_c = self._ctx.selected_col - 1
        sel_r = self._ctx.selected_row - 1

        # Clip content to map area (keep axis/legend bands clean)
        map_clip = QRectF(
            self._AXIS_L - 1,
            self._AXIS_T - 1,
            max(1.0, self.width() - self._AXIS_L - self._AXIS_R + 2),
            max(1.0, self.height() - self._AXIS_T - self._AXIS_B + 2),
        )

        # ── Wafer substrate ──
        p.save()
        p.setClipRect(map_clip)
        p.setPen(Qt.NoPen)
        p.setBrush(_c(_CLR_WAFER_BG))
        p.drawEllipse(QPointF(cx, cy), R, R)

        # Clip dies to wafer circle
        wafer_clip = QPainterPath()
        wafer_clip.addEllipse(QPointF(cx, cy), R * 0.998, R * 0.998)
        p.setClipPath(wafer_clip, Qt.IntersectClip)

        show_glyph = cell >= 7.0
        show_colrow = cell >= 14.0  # zoomed-in: print Col,Row on die
        glyph_font = QFont("Segoe UI", max(5, min(11, int(cell * 0.55))))
        glyph_font.setBold(True)
        cr_font = QFont("Segoe UI", max(5, min(8, int(cell * 0.28))))
        cr_font.setBold(True)

        # Only iterate dies that can intersect the viewport (perf when zoomed)
        if cell > 0.5:
            c0 = max(0, int((map_clip.left() - ox) / cell) - 1)
            c1 = min(n - 1, int((map_clip.right() - ox) / cell) + 1)
            r0 = max(0, int((map_clip.top() - oy) / cell) - 1)
            r1 = min(n - 1, int((map_clip.bottom() - oy) / cell) + 1)
        else:
            c0, c1, r0, r1 = 0, n - 1, 0, n - 1

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                b = int(bins[r, c])
                if b == BIN_OUTSIDE:
                    continue

                gap = max(0.4, cell * 0.08)
                rect = QRectF(
                    ox + c * cell + gap * 0.5,
                    oy + r * cell + gap * 0.5,
                    cell - gap,
                    cell - gap,
                )
                rad = 1.4 if cell > 8 else 0.6

                if b == BIN_NG:
                    p.setPen(QPen(_c("#7a1515"), 0.9))
                    p.setBrush(_c(_CLR_NG))
                    p.drawRoundedRect(rect, rad, rad)
                    glyph = "1"
                    gcol = _c("#1a0505")
                elif b == BIN_OK:
                    p.setPen(QPen(_c("#0d5c32"), 0.9))
                    p.setBrush(_c(_CLR_OK))
                    p.drawRoundedRect(rect, rad, rad)
                    glyph = "8"
                    gcol = _c("#04180c")
                else:
                    p.setPen(QPen(_c(_CLR_PENDING_BORDER, 200), 0.9))
                    p.setBrush(_c(_CLR_PENDING))
                    p.drawRoundedRect(rect, rad, rad)
                    glyph = "·"
                    gcol = _c("#d0e4f5")

                is_selected_die = (c == sel_c and r == sel_r)
                if show_glyph and glyph:
                    p.setFont(glyph_font)
                    p.setPen(gcol)
                    if show_colrow:
                        # Bin glyph top-half, Col·Row bottom-half when zoomed.
                        # Selected die: in-die Col·Row is skipped here; a floating
                        # badge is painted last (after reticle) so it is never covered.
                        p.drawText(
                            rect.adjusted(0, 0, 0, -rect.height() * 0.28 if not is_selected_die else 0),
                            Qt.AlignHCenter | Qt.AlignVCenter,
                            glyph,
                        )
                        if not is_selected_die:
                            p.setFont(cr_font)
                            p.setPen(_c("#e8f4ff" if b == BIN_PENDING else gcol))
                            p.drawText(
                                rect.adjusted(1, rect.height() * 0.52, -1, -1),
                                Qt.AlignHCenter | Qt.AlignTop,
                                f"{c + 1},{r + 1}",
                            )
                    else:
                        p.drawText(rect, Qt.AlignCenter, glyph)

                if self._hover == (c, r):
                    p.setPen(QPen(_c(_CLR_RETICLE, 220), 1.6))
                    p.setBrush(Qt.NoBrush)
                    p.drawRoundedRect(rect.adjusted(-0.5, -0.5, 0.5, 0.5), rad, rad)

        # When zoomed in enough to show Col·Row, markers must not fill the die:
        # use corner brackets + small open reticle (capped px) instead of solid ⊕.
        zoom_hi = show_colrow or cell >= 12.0

        # Selected chip marker (cyan) — always visible, never covers die face/text
        sel_badge_pos: Optional[Tuple[float, float, str]] = None
        if 0 <= sel_c < n and 0 <= sel_r < n:
            b_sel = int(bins[sel_r, sel_c]) if bins is not None else BIN_OUTSIDE
            if b_sel != BIN_OUTSIDE:
                sx = ox + (sel_c + 0.5) * cell
                sy = oy + (sel_r + 0.5) * cell
                sel_rect = QRectF(
                    ox + sel_c * cell,
                    oy + sel_r * cell,
                    cell,
                    cell,
                ).adjusted(-1.0, -1.0, 1.0, 1.0)
                if zoom_hi:
                    # Corner brackets keep OK/NG color + glyph + Col·Row free
                    _draw_corner_brackets(p, sel_rect, _c(_CLR_RETICLE), 2.2, 0.26)
                    ret_r = _reticle_radius(cell, frac=0.18, cap_px=11.0)
                    _draw_reticle(
                        p, sx, sy, ret_r, _c(_CLR_RETICLE, 200), 1.4, open_center=True
                    )
                    # FOV mark: small open ring only (no second solid cross)
                    if self._ctx.selected_fov > 0:
                        _draw_reticle(
                            p,
                            sx,
                            sy,
                            max(2.0, ret_r * 0.55),
                            _c(_CLR_FOV_MARK, 220),
                            1.2,
                            open_center=True,
                        )
                else:
                    p.setPen(QPen(_c(_CLR_RETICLE), 2.0))
                    p.setBrush(Qt.NoBrush)
                    p.drawRoundedRect(sel_rect, 2, 2)
                    ret_r = _reticle_radius(cell, frac=0.40, cap_px=16.0)
                    _draw_reticle(p, sx, sy, ret_r, _c(_CLR_RETICLE), 1.8)
                    if self._ctx.selected_fov > 0:
                        _draw_reticle(
                            p,
                            sx,
                            sy,
                            max(2.5, ret_r * 0.5),
                            _c(_CLR_FOV_MARK),
                            1.3,
                        )
                # Badge floats above die — painted last after clusters
                if zoom_hi or cell >= 10.0:
                    badge_y = sy - max(ret_r + 10.0, cell * 0.42)
                    sel_badge_pos = (sx, badge_y, f"{sel_c + 1},{sel_r + 1}")

        # ── NG Cluster boundaries + centroids (email Cluster Boundary / ⊕) ──
        # Drawn after dies so dashed boxes and reticles sit on top.
        clusters = list(getattr(self._ctx, "clusters", None) or [])
        sel_cl_id = int(getattr(self._ctx, "selected_cluster_id", 0) or 0)
        for cl in clusters:
            if not isinstance(cl, NgCluster) or cl.size <= 0:
                continue
            # Bounding rect covering member dies (axis-aligned, email dashed box)
            pad = max(0.6, cell * 0.06)
            brect = QRectF(
                ox + (cl.min_col - 1) * cell - pad,
                oy + (cl.min_row - 1) * cell - pad,
                (cl.max_col - cl.min_col + 1) * cell + 2 * pad,
                (cl.max_row - cl.min_row + 1) * cell + 2 * pad,
            )
            is_sel = cl.id == sel_cl_id
            col = _c(_CLR_CLUSTER_SEL if is_sel else _CLR_CLUSTER, 240 if is_sel else 200)
            pen = QPen(col, 2.2 if is_sel else 1.5, Qt.DashLine)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(brect, 2.5, 2.5)

            # Centroid reticle on assigned chip (1-based → 0-based)
            cc0 = int(cl.centroid_col) - 1
            cr0 = int(cl.centroid_row) - 1
            if 0 <= cc0 < n and 0 <= cr0 < n:
                cx_cl = ox + (cc0 + 0.5) * cell
                cy_cl = oy + (cr0 + 0.5) * cell
                # Skip extra centroid reticle when it is the already-selected die
                # (selected marker is enough — double ⊕ was hiding Col·Row)
                same_as_sel = (cc0 == sel_c and cr0 == sel_r)
                if not same_as_sel:
                    if zoom_hi:
                        # Small open reticle — leave Col·Row / die color readable
                        _draw_reticle(
                            p,
                            cx_cl,
                            cy_cl,
                            _reticle_radius(cell, frac=0.16, cap_px=10.0),
                            col,
                            1.5 if is_sel else 1.3,
                            open_center=True,
                        )
                    else:
                        _draw_reticle(
                            p,
                            cx_cl,
                            cy_cl,
                            _reticle_radius(
                                cell, frac=0.42 if is_sel else 0.34, cap_px=14.0
                            ),
                            col,
                            2.0 if is_sel else 1.5,
                        )
                # Small cluster ID label near centroid (outside die center)
                if cell >= 6.0:
                    id_font = QFont("Segoe UI", max(6, min(9, int(cell * 0.28))))
                    id_font.setBold(True)
                    p.setFont(id_font)
                    p.setPen(col)
                    p.drawText(
                        QRectF(
                            cx_cl + cell * 0.28,
                            cy_cl - cell * 0.50,
                            cell * 1.2,
                            cell * 0.40,
                        ),
                        Qt.AlignLeft | Qt.AlignVCenter,
                        f"C{cl.id}",
                    )

        # Col·Row badge on selected die — last paint so text is never under reticle
        if sel_badge_pos is not None:
            bx, by, btxt = sel_badge_pos
            badge_font = QFont("Segoe UI", max(6, min(9, int(cell * 0.30))))
            badge_font.setBold(True)
            _draw_coord_badge(p, bx, by, btxt, font=badge_font)

        p.setClipping(False)
        p.setClipRect(map_clip)

        # Rim
        p.setPen(QPen(_c(_CLR_WAFER_RIM_DIM), max(2.0, cell * 0.18)))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), R + 1.0, R + 1.0)
        p.setPen(QPen(_c(_CLR_WAFER_RIM, 230), max(1.5, cell * 0.13)))
        p.drawEllipse(QPointF(cx, cy), R, R)

        # Notch (orientation) at top of wafer
        notch_r = max(3.0, cell * 0.38)
        p.setPen(Qt.NoPen)
        p.setBrush(_c(SemiconductorTheme.BG_DARK))
        p.drawEllipse(QPointF(cx, cy - R + 1), notch_r, notch_r * 0.75)
        p.setPen(QPen(_c(_CLR_WAFER_RIM, 180), 1.2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy - R + 1), notch_r * 0.85, notch_r * 0.55)
        p.restore()

        # ── Axis ticks (track pan/zoom — only labels whose tick is in view) ──
        tick_font = QFont("Segoe UI", 6)
        p.setFont(tick_font)
        step = 5 if n >= 20 else (2 if n >= 10 else 1)
        # denser ticks when zoomed so Col/Row stay trackable
        if self._zoom >= 2.5:
            step = max(1, step // 2)
        if self._zoom >= 5.0:
            step = 1
        tick_idxs = sorted(set(
            [0, n - 1]
            + list(range(0, n, step))
            + ([sel_c] if 0 <= sel_c < n else [])
            + ([sel_r] if 0 <= sel_r < n else [])
            + ([self._hover[0], self._hover[1]] if self._hover else [])
        ))

        # Column numbers (top band) — x follows pan/zoom
        col_band = QRectF(self._AXIS_L, 0, self.width() - self._AXIS_L - self._AXIS_R, self._AXIS_T)
        for c in tick_idxs:
            if c < 0 or c >= n:
                continue
            tx = ox + (c + 0.5) * cell
            if tx < col_band.left() - 4 or tx > col_band.right() + 4:
                continue
            label = f"{c + 1:02d}" if n >= 10 else str(c + 1)
            if c == sel_c or (self._hover and self._hover[0] == c):
                p.setPen(_c(_CLR_RETICLE if c == sel_c else _CLR_FOV_MARK))
                f2 = QFont(tick_font)
                f2.setBold(True)
                p.setFont(f2)
            else:
                p.setPen(_c(SemiconductorTheme.TEXT_DISABLED))
                p.setFont(tick_font)
            p.drawText(
                QRectF(tx - 12, 1, 24, self._AXIS_T - 2),
                Qt.AlignHCenter | Qt.AlignVCenter,
                label,
            )

        # Row numbers (left band) — y follows pan/zoom
        for r in tick_idxs:
            if r < 0 or r >= n:
                continue
            ty = oy + (r + 0.5) * cell
            if ty < self._AXIS_T - 4 or ty > self.height() - self._AXIS_B + 4:
                continue
            label = f"{r + 1:02d}" if n >= 10 else str(r + 1)
            if r == sel_r or (self._hover and self._hover[1] == r):
                p.setPen(_c(_CLR_RETICLE if r == sel_r else _CLR_FOV_MARK))
                f2 = QFont(tick_font)
                f2.setBold(True)
                p.setFont(f2)
            else:
                p.setPen(_c(SemiconductorTheme.TEXT_DISABLED))
                p.setFont(tick_font)
            p.drawText(
                QRectF(1, ty - 7, self._AXIS_L - 2, 14),
                Qt.AlignRight | Qt.AlignVCenter,
                label,
            )

        # ── Bottom legend + badges ──
        leg_y = self.height() - self._AXIS_B + 3
        leg_font = QFont("Segoe UI", 6)
        leg_font.setBold(True)
        p.setFont(leg_font)
        items = [
            (_CLR_OUT, _CLR_OUT_BORDER, "Out"),
            (_CLR_PENDING, _CLR_PENDING_BORDER, "Pend"),
            (_CLR_OK, "#0d5c32", "OK"),
            (_CLR_NG, "#7a1515", "NG"),
        ]
        lx = 4.0
        for fill, border, text in items:
            p.setPen(QPen(_c(border), 1.0))
            p.setBrush(_c(fill))
            p.drawRoundedRect(QRectF(lx, leg_y + 2, 8, 8), 1.5, 1.5)
            p.setPen(_c(SemiconductorTheme.TEXT_SECONDARY))
            p.drawText(QRectF(lx + 10, leg_y, 28, 12), Qt.AlignLeft | Qt.AlignVCenter, text)
            lx += 38
        # Cluster legend swatch (dashed box icon)
        p.setPen(QPen(_c(_CLR_CLUSTER), 1.2, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(lx, leg_y + 2, 8, 8), 1.5, 1.5)
        p.setPen(_c(SemiconductorTheme.TEXT_SECONDARY))
        p.drawText(QRectF(lx + 10, leg_y, 36, 12), Qt.AlignLeft | Qt.AlignVCenter, "Clus")
        lx += 48

        # Zoom badge (left of status)
        z_txt = f"{self._zoom:.1f}×"
        z_font = QFont("Segoe UI", 7)
        z_font.setBold(True)
        p.setFont(z_font)
        zfm = QFontMetrics(z_font)
        zw = zfm.horizontalAdvance(z_txt) + 10
        zh = zfm.height() + 4
        zx = 4.0
        zy = self._AXIS_T + 2
        p.setPen(QPen(_c(SemiconductorTheme.BORDER_DEFAULT), 1))
        p.setBrush(_c(SemiconductorTheme.BG_PANEL, 220))
        p.drawRoundedRect(QRectF(zx, zy, zw, zh), 3, 3)
        p.setPen(_c(SemiconductorTheme.PRIMARY_DEFAULT))
        p.drawText(QRectF(zx, zy, zw, zh), Qt.AlignCenter, z_txt)

        # Selection / hover coordinate badge
        badge = None
        if self._hover is not None:
            hc, hr = self._hover
            b = int(bins[hr, hc]) if 0 <= hr < n and 0 <= hc < n else -1
            bin_s = {0: "Out", 1: "NG", 2: "Pend", 8: "OK"}.get(b, "?")
            badge = f"Col {hc + 1} · Row {hr + 1}  ({bin_s})"
        elif sel_c >= 0 and sel_r >= 0:
            b = int(bins[sel_r, sel_c]) if 0 <= sel_r < n and 0 <= sel_c < n else -1
            bin_s = {0: "Out", 1: "NG", 2: "Pend", 8: "OK"}.get(b, "?")
            fov = self._ctx.selected_fov
            extra = f"  ·  FOV P{fov}" if fov > 0 else ""
            badge = f"Sel ({sel_c + 1},{sel_r + 1}) {bin_s}{extra}"
        if badge:
            badge_font = QFont("Segoe UI", 7)
            badge_font.setBold(True)
            p.setFont(badge_font)
            fm = QFontMetrics(badge_font)
            tw = fm.horizontalAdvance(badge) + 12
            th = fm.height() + 6
            bx = self.width() - tw - 4
            by = leg_y - 1
            p.setPen(QPen(_c(SemiconductorTheme.BORDER_DEFAULT), 1))
            p.setBrush(_c(SemiconductorTheme.BG_PANEL, 235))
            p.drawRoundedRect(QRectF(bx, by, tw, th), 4, 4)
            p.setPen(_c(_CLR_FOV_MARK if self._hover else _CLR_RETICLE))
            p.drawText(QRectF(bx, by, tw, th), Qt.AlignCenter, badge)

        p.end()

    def _hit(self, pos) -> Optional[Tuple[int, int]]:
        if self._ctx is None:
            return None
        n, cx, cy, R, cell, ox, oy = self._layout_geom()
        if cell <= 1e-6:
            return None
        x, y = pos.x(), pos.y()
        if (x - cx) ** 2 + (y - cy) ** 2 > (R * 0.985) ** 2:
            return None
        c = int((x - ox) / cell)
        r = int((y - oy) / cell)
        if 0 <= c < n and 0 <= r < n:
            if int(self._ctx.bins[r, c]) == BIN_OUTSIDE:
                return None
            return (c, r)
        return None

    def wheelEvent(self, event: QWheelEvent):
        if self._ctx is None:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            # trackpad / pixel scroll
            delta = event.pixelDelta().y()
        if delta == 0:
            event.ignore()
            return

        old_z = max(self._ZOOM_MIN, min(self._ZOOM_MAX, float(self._zoom)))
        factor = self._ZOOM_STEP if delta > 0 else (1.0 / self._ZOOM_STEP)
        new_z = max(self._ZOOM_MIN, min(self._ZOOM_MAX, old_z * factor))
        if abs(new_z - old_z) < 1e-6:
            event.accept()
            return

        # Zoom toward cursor: keep world point under mouse fixed
        base_cx, base_cy, _ = self._base_center_and_side()
        mx = float(event.pos().x())
        my = float(event.pos().y())
        # world offset from base center at old zoom
        ox = (mx - base_cx - self._pan_x) / old_z
        oy = (my - base_cy - self._pan_y) / old_z
        self._zoom = new_z
        self._pan_x = mx - base_cx - ox * new_z
        self._pan_y = my - base_cy - oy * new_z
        self._clamp_pan()
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            self._press_pos = QPoint(event.pos())
            self._press_pan = (self._pan_x, self._pan_y)
            self._press_hit = self._hit(event.pos()) if event.button() == Qt.LeftButton else None
            self._dragging = False
            self._pan_button = int(event.button())
            if event.button() in (Qt.MiddleButton, Qt.RightButton):
                self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Pan while button held
        if self._press_pos is not None and self._pan_button is not None:
            dx = event.pos().x() - self._press_pos.x()
            dy = event.pos().y() - self._press_pos.y()
            if (not self._dragging
                    and (abs(dx) >= self._DRAG_THRESH_PX or abs(dy) >= self._DRAG_THRESH_PX)):
                self._dragging = True
                self.setCursor(Qt.ClosedHandCursor)
            if self._dragging or self._pan_button in (int(Qt.MiddleButton), int(Qt.RightButton)):
                self._dragging = True
                self._pan_x = self._press_pan[0] + dx
                self._pan_y = self._press_pan[1] + dy
                self._clamp_pan()
                self.update()
                super().mouseMoveEvent(event)
                return

        # Hover tracking
        h = self._hit(event.pos())
        if h != self._hover:
            self._hover = h
            if h is not None:
                c0, r0 = h
                b = int(self._ctx.bins[r0, c0]) if self._ctx is not None else -1
                bin_s = {0: "Outside", 1: "NG", 2: "Pending", 8: "OK"}.get(b, "?")
                self.setToolTip(
                    f"Chip  Col {c0 + 1}  ·  Row {r0 + 1}  (X,Y 1-based)\n"
                    f"Final bin: {bin_s} ({b})\n"
                    f"Wheel zoom · drag pan · double-click fit"
                )
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.setToolTip(
                    "Wheel = zoom  ·  Drag = pan  ·  Click = select die\n"
                    "Double-click = fit view"
                )
                self.setCursor(Qt.OpenHandCursor)
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._press_pos is not None and event.button() == Qt.LeftButton:
            # Click (no drag) → select die. Cluster highlight follows via
            # WaferContext.select_chip → selected_cluster_id (stay on clicked die).
            if not self._dragging and self._press_hit is not None:
                c0, r0 = self._press_hit
                self.chip_clicked.emit(c0 + 1, r0 + 1)
        self._press_pos = None
        self._press_hit = None
        self._dragging = False
        self._pan_button = None
        # restore cursor based on hover
        if self._hit(event.pos()) is not None:
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event):
        self._hover = None
        self.setToolTip(
            "Wheel = zoom  ·  Drag = pan  ·  Click = select die\n"
            "Double-click = fit view"
        )
        if not self._dragging:
            self.setCursor(Qt.OpenHandCursor)
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        # +/- zoom, 0 = fit, arrows nudge pan
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self._zoom_by(self._ZOOM_STEP)
            event.accept()
            return
        if event.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            self._zoom_by(1.0 / self._ZOOM_STEP)
            event.accept()
            return
        if event.key() == Qt.Key_0 or (event.key() == Qt.Key_Home):
            self.reset_view()
            event.accept()
            return
        step = 24.0
        if event.key() == Qt.Key_Left:
            self._pan_x += step
            self._clamp_pan()
            self.update()
            event.accept()
            return
        if event.key() == Qt.Key_Right:
            self._pan_x -= step
            self._clamp_pan()
            self.update()
            event.accept()
            return
        if event.key() == Qt.Key_Up:
            self._pan_y += step
            self._clamp_pan()
            self.update()
            event.accept()
            return
        if event.key() == Qt.Key_Down:
            self._pan_y -= step
            self._clamp_pan()
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def _zoom_by(self, factor: float) -> None:
        old_z = max(self._ZOOM_MIN, min(self._ZOOM_MAX, float(self._zoom)))
        new_z = max(self._ZOOM_MIN, min(self._ZOOM_MAX, old_z * factor))
        if abs(new_z - old_z) < 1e-6:
            return
        # zoom about view center
        base_cx, base_cy, _ = self._base_center_and_side()
        mx, my = base_cx, base_cy
        ox = (mx - base_cx - self._pan_x) / old_z
        oy = (my - base_cy - self._pan_y) / old_z
        self._zoom = new_z
        self._pan_x = mx - base_cx - ox * new_z
        self._pan_y = my - base_cy - oy * new_z
        self._clamp_pan()
        self.update()


class _ChipMapCanvas(QWidget):
    """3×3 FOV points (P1–P9); amber = active FOV volume in MPR."""

    fov_clicked = pyqtSignal(int)  # 1..9

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctx: Optional[WaferContext] = None
        self.setMinimumHeight(168)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self._hover: Optional[int] = None  # 1..9

    def set_context(self, ctx: Optional[WaferContext]) -> None:
        self._ctx = ctx
        self.update()

    def _cell_rects(self):
        margin = 6
        gap = 5
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        cell_w = (w - 2 * gap) / 3.0
        cell_h = (h - 2 * gap) / 3.0
        side = min(cell_w, cell_h)
        total = 3 * side + 2 * gap
        ox = (self.width() - total) * 0.5
        oy = (self.height() - total) * 0.5
        rects = {}
        idx = 1
        for r in range(3):
            for c in range(3):
                x = ox + c * (side + gap)
                y = oy + r * (side + gap)
                rects[idx] = QRectF(x, y, side, side)
                idx += 1
        return rects

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), _c(SemiconductorTheme.BG_DARK))

        chip = self._ctx.selected_chip() if self._ctx else None
        if chip is None:
            p.setPen(_c(SemiconductorTheme.TEXT_DISABLED))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(self.rect(), Qt.AlignCenter, "Click a die on wafer")
            p.end()
            return

        chip.ensure_points()
        active = self._ctx.selected_fov if self._ctx else 0
        label_font = QFont("Segoe UI", 7)
        label_font.setBold(True)

        for idx, rect in self._cell_rects().items():
            pt = chip.point(idx)
            judge = pt.judge if pt else BIN_PENDING
            is_active = idx == active

            if is_active:
                fill = _c("#2a5a70")
                border = _c(_CLR_FOV_MARK)
                bw = 2.6
            elif judge == BIN_NG:
                fill = _c("#6a2020")
                border = _c(_CLR_NG, 200)
                bw = 1.4
            elif judge == BIN_OK:
                fill = _c("#1a4a32")
                border = _c(_CLR_OK, 180)
                bw = 1.4
            else:
                # Pending FOV — cooler slate, clearly not Outside/empty
                fill = _c("#3d556e")
                border = _c(_CLR_PENDING_BORDER, 180)
                bw = 1.3

            p.setPen(QPen(border, bw))
            p.setBrush(fill)
            p.drawRoundedRect(rect, 6, 6)

            # P# corner
            p.setFont(label_font)
            p.setPen(_c(SemiconductorTheme.TEXT_SECONDARY))
            p.drawText(rect.adjusted(5, 3, -4, -4), Qt.AlignLeft | Qt.AlignTop, f"P{idx}")

            # Bin 8 / 1 / · (pending) — host final is only 8 or 1 after inspect
            if judge == BIN_OK:
                bin_txt, bin_col = "8", _c("#6dff9a")
            elif judge == BIN_NG:
                bin_txt, bin_col = "1", _c("#ff7a7a")
            else:
                bin_txt, bin_col = "·", _c(SemiconductorTheme.TEXT_DISABLED)
            if is_active and judge in (BIN_OK, BIN_NG):
                bin_col = _c(_CLR_FOV_MARK)
            elif is_active:
                bin_col = _c(_CLR_FOV_MARK)
            big = QFont("Segoe UI", max(12, int(rect.height() * 0.38)))
            big.setBold(True)
            p.setFont(big)
            p.setPen(bin_col)
            p.drawText(rect, Qt.AlignCenter, bin_txt)

            if is_active:
                # FOV pill at bottom
                pill_h = 14
                pill = QRectF(
                    rect.left() + 6,
                    rect.bottom() - pill_h - 5,
                    rect.width() - 12,
                    pill_h,
                )
                p.setPen(Qt.NoPen)
                p.setBrush(_c(_CLR_FOV_MARK, 220))
                p.drawRoundedRect(pill, 3, 3)
                p.setFont(label_font)
                p.setPen(_c("#1a1200"))
                p.drawText(pill, Qt.AlignCenter, "FOV · MPR")

            if self._hover == idx and not is_active:
                p.setPen(QPen(_c(SemiconductorTheme.PRIMARY_HOVER), 1.4))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 5, 5)

        p.end()

    def _hit(self, pos) -> Optional[int]:
        for idx, rect in self._cell_rects().items():
            if rect.contains(QPointF(pos)):
                return idx
        return None

    def mouseMoveEvent(self, event):
        h = self._hit(event.pos())
        if h != self._hover:
            self._hover = h
            if h is not None and self._ctx is not None:
                chip = self._ctx.selected_chip()
                pt = chip.point(h) if chip else None
                j = "OK" if (pt and pt.judge == BIN_OK) else (
                    "NG" if (pt and pt.judge == BIN_NG) else "?"
                )
                self.setToolTip(f"FOV P{h}  ·  Judge {j}\nClick to open volume in MPR")
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hover = None
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            h = self._hit(event.pos())
            if h is not None:
                self.fov_clicked.emit(h)
        super().mousePressEvent(event)


class ContextMapPanel(QWidget):
    """Left strip: Wafer map + Chip FOV map for Online inspection context."""

    chip_selected = pyqtSignal(int, int)  # col, row
    fov_selected = pyqtSignal(int)  # 1..9
    cluster_selected = pyqtSignal(int)  # NgCluster.id
    selection_changed = pyqtSignal()  # any selection update

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ContextMapPanel")
        # Wider strip so zoomed dies + Col/Row axis stay readable
        self.setFixedWidth(280)
        self.setMinimumWidth(240)
        self._ctx: Optional[WaferContext] = None
        self._cluster_table_block = False
        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QLabel("CONTEXT  ·  WAFER › CHIP › FOV")
        header.setObjectName("ContextHeader")
        header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setFixedHeight(28)
        root.addWidget(header)

        # Breadcrumb
        self.breadcrumb = QLabel("—")
        self.breadcrumb.setObjectName("ContextBreadcrumb")
        self.breadcrumb.setWordWrap(True)
        self.breadcrumb.setMinimumHeight(26)
        root.addWidget(self.breadcrumb)

        # Meta
        self.meta_label = QLabel("")
        self.meta_label.setObjectName("ContextMeta")
        self.meta_label.setWordWrap(True)
        root.addWidget(self.meta_label)

        # Wafer section
        w_title = QLabel("WAFER MAP  ·  zoom/pan · NG clusters")
        w_title.setObjectName("ContextSection")
        w_title.setToolTip(
            "Wheel = zoom  ·  Drag = pan  ·  Click die = select\n"
            "Double-click = fit whole wafer\n"
            "Dashed cyan box = NG Cluster Boundary\n"
            "⊕ = Cluster Centroid (assigned chip)\n"
            "Click Cluster Summary row → jump to centroid"
        )
        root.addWidget(w_title)

        self.wafer_canvas = _WaferMapCanvas()
        self.wafer_canvas.chip_clicked.connect(self._on_chip_clicked)
        root.addWidget(self.wafer_canvas, 5)  # more stretch → bigger wafer

        # Stats strip (Good / Bad / Yield / Cluster Count) — PDF-style summary
        self.wafer_stats = QLabel("")
        self.wafer_stats.setObjectName("ContextStats")
        self.wafer_stats.setWordWrap(True)
        self.wafer_stats.setTextFormat(Qt.RichText)
        root.addWidget(self.wafer_stats)

        # Cluster Summary (email table: ID · Size · Centroid · MaxR)
        cl_title = QLabel("CLUSTER SUMMARY  ·  NG groups")
        cl_title.setObjectName("ContextSection")
        cl_title.setToolTip(
            "8-connected NG (bin=1) components.\n"
            "Centroid Chip = member die nearest geometric mean.\n"
            "Max Radius in die-pitch units."
        )
        root.addWidget(cl_title)

        self.cluster_table = QTableWidget(0, 4)
        self.cluster_table.setObjectName("ContextClusterTable")
        self.cluster_table.setHorizontalHeaderLabels(["ID", "Size", "Ctr (X,Y)", "MaxR"])
        self.cluster_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cluster_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.cluster_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cluster_table.verticalHeader().setVisible(False)
        self.cluster_table.horizontalHeader().setStretchLastSection(True)
        self.cluster_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cluster_table.setAlternatingRowColors(True)
        self.cluster_table.setMaximumHeight(110)
        self.cluster_table.setMinimumHeight(52)
        self.cluster_table.verticalHeader().setDefaultSectionSize(18)
        self.cluster_table.itemSelectionChanged.connect(self._on_cluster_table_selection)
        root.addWidget(self.cluster_table)

        self.cluster_setting = QLabel("")
        self.cluster_setting.setObjectName("ContextMeta")
        self.cluster_setting.setWordWrap(True)
        self.cluster_setting.setTextFormat(Qt.RichText)
        root.addWidget(self.cluster_setting)

        # Chip section
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setObjectName("ContextSep")
        root.addWidget(sep)

        self.chip_title = QLabel("CHIP MAP  ·  9-POINT FOV")
        self.chip_title.setObjectName("ContextSection")
        root.addWidget(self.chip_title)

        self.chip_meta = QLabel("")
        self.chip_meta.setObjectName("ContextMeta")
        self.chip_meta.setWordWrap(True)
        self.chip_meta.setTextFormat(Qt.RichText)
        root.addWidget(self.chip_meta)

        self.chip_canvas = _ChipMapCanvas()
        self.chip_canvas.fov_clicked.connect(self._on_fov_clicked)
        root.addWidget(self.chip_canvas, 3)

        self.fov_note = QLabel("Amber FOV = volume open in MPR")
        self.fov_note.setObjectName("ContextNote")
        self.fov_note.setWordWrap(True)
        root.addWidget(self.fov_note)

        footer = QLabel("Results → …/Chip/FOV/Results")
        footer.setObjectName("ContextFooter")
        self.footer_label = footer
        root.addWidget(footer)

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QWidget#ContextMapPanel {{
                background-color: {SemiconductorTheme.BG_MEDIUM};
                border-right: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QLabel#ContextHeader {{
                background-color: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.PRIMARY_DEFAULT};
                font-weight: 700;
                font-size: 8pt;
                padding: 4px 10px;
                border-bottom: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QLabel#ContextBreadcrumb {{
                background-color: {SemiconductorTheme.BG_PANEL};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                font-size: 8pt;
                padding: 4px 10px;
                margin: 4px 6px 2px 6px;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px;
            }}
            QLabel#ContextSection {{
                color: {SemiconductorTheme.TEXT_SECONDARY};
                font-weight: 700;
                font-size: 7.5pt;
                letter-spacing: 0.8px;
                padding: 6px 10px 2px 10px;
            }}
            QLabel#ContextMeta {{
                color: {SemiconductorTheme.TEXT_DISABLED};
                font-size: 7.5pt;
                padding: 0px 10px 4px 10px;
            }}
            QLabel#ContextStats {{
                color: {SemiconductorTheme.TEXT_SECONDARY};
                font-size: 8pt;
                padding: 4px 10px 6px 10px;
                background: {SemiconductorTheme.BG_DARK};
                margin: 0 6px 2px 6px;
                border-radius: 4px;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QLabel#ContextNote {{
                color: {SemiconductorTheme.ACCENT_WARNING};
                font-size: 7.5pt;
                padding: 4px 10px;
            }}
            QLabel#ContextFooter {{
                color: {SemiconductorTheme.TEXT_DISABLED};
                font-size: 7pt;
                padding: 4px 10px 8px 10px;
            }}
            QFrame#ContextSep {{
                background-color: {SemiconductorTheme.BORDER_DEFAULT};
                margin: 4px 8px;
            }}
            QTableWidget#ContextClusterTable {{
                background: {SemiconductorTheme.BG_DARK};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px;
                margin: 0 6px 2px 6px;
                gridline-color: {SemiconductorTheme.BORDER_DEFAULT};
                font-size: 7.5pt;
            }}
            QTableWidget#ContextClusterTable::item:selected {{
                background: {SemiconductorTheme.PRIMARY_DEFAULT};
                color: {SemiconductorTheme.TEXT_ON_ACCENT};
            }}
            QHeaderView::section {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_SECONDARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                padding: 2px 4px;
                font-size: 7pt;
                font-weight: 700;
            }}
        """)

    # ── Public API ───────────────────────────────────────────────
    def set_context(self, ctx: Optional[WaferContext]) -> None:
        self._ctx = ctx
        if ctx is not None:
            # Always rebuild NG clusters from FinalBin map (email Cluster Count)
            try:
                ctx.recompute_clusters()
            except Exception:
                pass
        self.wafer_canvas.set_context(ctx)
        self.chip_canvas.set_context(ctx)
        self._refresh_labels()
        self._refresh_cluster_table()

    def context(self) -> Optional[WaferContext]:
        return self._ctx

    def load_demo(self) -> WaferContext:
        """Email-style fake OK/NG map — mock UI only."""
        ctx = WaferContext.demo()
        self.set_context(ctx)
        return ctx

    def load_empty(self) -> WaferContext:
        """Uninspected wafer geometry (all inside dies Pending)."""
        ctx = WaferContext.empty_geometry()
        self.set_context(ctx)
        return ctx

    def load_from_dict(self, data: dict) -> WaferContext:
        ctx = WaferContext.from_dict(data)
        self.set_context(ctx)
        return ctx

    def apply_host_path_info(self, info: HostVolumePathInfo) -> Optional[WaferContext]:
        """Select Chip/FOV from host volume path and refresh maps."""
        if self._ctx is None:
            # Real Online: empty geometry, not pre-baked demo NG/OK
            self._ctx = WaferContext.empty_geometry()
        self._ctx.apply_host_path(info)
        self.set_context(self._ctx)
        # Auto-zoom onto the host die so operator finds Col/Row immediately
        if info.chip_col > 0 and info.chip_row > 0:
            try:
                self.wafer_canvas.zoom_to_selection(
                    int(info.chip_col), int(info.chip_row), zoom=3.2
                )
            except Exception:
                pass
            self.chip_selected.emit(int(info.chip_col), int(info.chip_row))
        if info.fov_index > 0:
            self.fov_selected.emit(int(info.fov_index))
        self.selection_changed.emit()
        return self._ctx

    def refresh(self) -> None:
        """Repaint maps + labels after external WaferContext mutation (e.g. judge)."""
        self.set_context(self._ctx)

    def _refresh_labels(self) -> None:
        ctx = self._ctx
        if ctx is None:
            self.breadcrumb.setText("—")
            self.meta_label.setText("Waiting for host Wafer / Chip / FOV data")
            self.wafer_stats.setText("")
            self.chip_title.setText("CHIP MAP  ·  9-POINT FOV")
            self.chip_meta.setText("")
            self.fov_note.setText("Amber FOV = volume open in MPR")
            if hasattr(self, "cluster_setting"):
                self.cluster_setting.setText("")
            if hasattr(self, "footer_label"):
                self.footer_label.setText("Results → …/Chip/FOV/Results")
            return

        self.breadcrumb.setText(ctx.breadcrumb())
        lot_line = ctx.lot_foup_id or (
            f"{ctx.lot_id}_{ctx.foup_id}" if ctx.foup_id else (ctx.lot_id or "—")
        )
        self.meta_label.setText(
            f"Lot/Foup {lot_line}\n"
            f"Wafer {ctx.wafer_id or '—'}  ·  {ctx.wafer_size_mm:.0f} mm  ·  "
            f"{ctx.map_size}×{ctx.map_size}"
        )
        st = ctx.stats()
        n_cl = int(st.get("cluster_count", len(getattr(ctx, "clusters", None) or [])))
        # Pending dies are not Good — new wafer shows Pend + Yield 0% until inspect
        self.wafer_stats.setText(
            f"<span style='color:#8fa3b8'><b>Pend {st.get('pending', 0)}</b></span>"
            f"  ·  <span style='color:#6dff9a'><b>OK {st['good']}</b></span>"
            f"  ·  <span style='color:#ff7a7a'><b>NG {st['bad']}</b></span>"
            f"  ·  Yield <span style='color:#22d3ee'><b>{ctx.yield_pct:.2f}%</b></span>"
            f"<br><span style='color:#8fa3b8;font-size:7.5pt'>"
            f"Inside {st.get('inside', 0)}  ·  Outside {st.get('outside', 0)}  ·  "
            f"<span style='color:#3dd6f5'><b>Cluster {n_cl}</b></span></span>"
        )

        chip = ctx.selected_chip()
        if chip is None:
            self.chip_title.setText("CHIP MAP  ·  9-POINT FOV")
            self.chip_meta.setText("Click a die on the wafer (⊕ reticle = selection)")
            self.fov_note.setText("Amber FOV = volume open in MPR")
        else:
            # Match wafer-map FinalBin: NG if any FOV NG; OK only when all 9 FOVs OK;
            # otherwise Pending (partial inspect must not look like chip OK).
            if chip.chip_ng or chip.final_bin == BIN_NG:
                judge, jcol = "NG", "#ff7a7a"
            elif chip.final_bin == BIN_OK:
                judge, jcol = "OK", "#6dff9a"
            else:
                judge, jcol = "Pend", "#8fa3b8"
            self.chip_title.setText(
                f"CHIP MAP  ·  Col {chip.col}  ·  Row {chip.row}"
            )
            self.chip_meta.setText(
                f"Chip (X={chip.col}, Y={chip.row})  ·  Final: "
                f"<span style='color:{jcol}'><b>{judge}</b></span><br>"
                f"9-Point  ·  Good {chip.good_count}/9  ·  Bad {chip.bad_count}/9"
                f"{'  ·  Wafer green only after 9/9 OK' if judge == 'Pend' and chip.good_count > 0 else ''}"
            )
            pt = ctx.selected_point()
            results_path = ctx.results_dir_for_selection() or "…/FOV/Results"
            if pt is not None:
                vol = Path(pt.volume_path).name if pt.volume_path else "Input.tiff"
                self.fov_note.setText(
                    f"FOV P{pt.index}  ·  {vol}\n"
                    f"offset X {pt.x_offset_mm:+.2f} mm  Y {pt.y_offset_mm:+.2f} mm\n"
                    f"Results → {results_path}"
                )
            else:
                self.fov_note.setText("Amber FOV = volume open in MPR")
            if hasattr(self, "footer_label"):
                self.footer_label.setText(f"Results → {results_path}")

        # Cluster Centroid Setting (email panel)
        if hasattr(self, "cluster_setting"):
            cl = ctx.selected_cluster()
            if cl is not None:
                self.cluster_setting.setText(
                    f"Selected Cluster : <span style='color:#3dd6f5'><b>{cl.id}</b></span>"
                    f"  ·  Centroid Chip (X,Y) : "
                    f"<span style='color:#f5d76e'><b>({cl.centroid_col},{cl.centroid_row})</b></span>"
                    f"<br><span style='color:#8fa3b8;font-size:7.5pt'>"
                    f"Size {cl.size} die  ·  MaxR {cl.max_radius:.2f}  ·  "
                    f"⊕ = cluster centroid (assigned chip)</span>"
                )
            elif (ctx.clusters or []):
                self.cluster_setting.setText(
                    f"<span style='color:#8fa3b8'>{len(ctx.clusters)} NG cluster(s) — "
                    f"click a red die or a table row</span>"
                )
            else:
                self.cluster_setting.setText(
                    "<span style='color:#8fa3b8'>No NG clusters (no FinalBin=1 groups)</span>"
                )

    def _refresh_cluster_table(self) -> None:
        """Fill Cluster Summary table (ID · Size · Centroid · MaxR)."""
        if not hasattr(self, "cluster_table"):
            return
        ctx = self._ctx
        self._cluster_table_block = True
        self.cluster_table.setRowCount(0)
        if ctx is None:
            self._cluster_table_block = False
            return
        clusters = list(getattr(ctx, "clusters", None) or [])
        self.cluster_table.setRowCount(len(clusters))
        sel_id = int(getattr(ctx, "selected_cluster_id", 0) or 0)
        sel_row = -1
        for i, cl in enumerate(clusters):
            vals = [
                str(cl.id),
                str(cl.size),
                f"({cl.centroid_col},{cl.centroid_row})",
                f"{cl.max_radius:.2f}",
            ]
            for j, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if j == 0:
                    item.setData(Qt.UserRole, int(cl.id))
                self.cluster_table.setItem(i, j, item)
            if cl.id == sel_id:
                sel_row = i
        if sel_row >= 0:
            self.cluster_table.selectRow(sel_row)
        else:
            self.cluster_table.clearSelection()
        self._cluster_table_block = False

    def _on_cluster_table_selection(self) -> None:
        if self._cluster_table_block or self._ctx is None:
            return
        rows = self.cluster_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.cluster_table.item(rows[0].row(), 0)
        if item is None:
            return
        cid = item.data(Qt.UserRole)
        if cid is None:
            try:
                cid = int(item.text())
            except ValueError:
                return
        self._select_cluster(int(cid), from_table=True)

    def _select_cluster(self, cluster_id: int, from_table: bool = False) -> None:
        """Select cluster → jump map selection to centroid chip (email setting)."""
        if self._ctx is None:
            return
        if not self._ctx.select_cluster(int(cluster_id)):
            return
        self.wafer_canvas.set_context(self._ctx)
        self.chip_canvas.set_context(self._ctx)
        self._refresh_labels()
        if not from_table:
            self._refresh_cluster_table()
        else:
            # Keep table selection; still sync highlight row
            pass
        # Zoom onto centroid for operator focus
        cl = self._ctx.selected_cluster()
        if cl is not None:
            try:
                self.wafer_canvas.zoom_to_selection(
                    cl.centroid_col, cl.centroid_row, zoom=3.5
                )
            except Exception:
                pass
            self.chip_selected.emit(int(cl.centroid_col), int(cl.centroid_row))
        self.cluster_selected.emit(int(cluster_id))
        self.selection_changed.emit()

    def _on_chip_clicked(self, col: int, row: int) -> None:
        if self._ctx is None:
            return
        if self._ctx.select_chip(col, row):
            self.wafer_canvas.set_context(self._ctx)
            self.chip_canvas.set_context(self._ctx)
            self._refresh_labels()
            self._refresh_cluster_table()
            self.chip_selected.emit(col, row)
            if self._ctx.selected_cluster_id > 0:
                self.cluster_selected.emit(int(self._ctx.selected_cluster_id))
            self.selection_changed.emit()

    def focus_wafer_selection(self, zoom: float = 3.2) -> None:
        """Public: zoom/pan wafer map onto current selected die."""
        if self._ctx is None:
            return
        if self._ctx.selected_col > 0 and self._ctx.selected_row > 0:
            self.wafer_canvas.zoom_to_selection(
                self._ctx.selected_col, self._ctx.selected_row, zoom=zoom
            )

    def _on_fov_clicked(self, index: int) -> None:
        if self._ctx is None:
            return
        if self._ctx.select_fov(index):
            self.chip_canvas.set_context(self._ctx)
            self.wafer_canvas.set_context(self._ctx)
            self._refresh_labels()
            self.fov_selected.emit(index)
            self.selection_changed.emit()
