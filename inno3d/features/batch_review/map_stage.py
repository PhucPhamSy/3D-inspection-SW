# inno3d/features/batch_review/map_stage.py
# -----------------------------------------------------------------------
# MapStage — unified Wafer (L0) → Chip FOV overlay (L1) for Line Pulse.
# Composes Online-identical canvases; does not rewrite paint logic.
# -----------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from inno3d.core.styles import SemiconductorTheme
from inno3d.core.wafer_context import BIN_NG, BIN_OK, WaferContext
from inno3d.widgets.context_map_panel import _ChipMapCanvas, _WaferMapCanvas

# Bin colors — parity with Online CONTEXT (_WaferMapCanvas legend)
_CLR_OUT = "#1a1f28"
_CLR_OUT_BORDER = "#2a3340"
_CLR_PENDING = "#5b7a9a"
_CLR_PENDING_BORDER = "#7a9bb8"
_CLR_OK = "#2fd67b"
_CLR_NG = "#f04444"


class _BinLegendBar(QWidget):
    """Compact OK/NG/Pend/Out swatches (Online wafer-map palette)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(18)
        self.setMinimumWidth(160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        items = [
            (_CLR_OUT, _CLR_OUT_BORDER, "Out"),
            (_CLR_PENDING, _CLR_PENDING_BORDER, "Pend"),
            (_CLR_OK, "#0d5c32", "OK"),
            (_CLR_NG, "#7a1515", "NG"),
        ]
        font = QFont("Segoe UI", 7)
        font.setBold(True)
        p.setFont(font)
        lx = 0.0
        cy = (self.height() - 10) * 0.5
        for fill, border, text in items:
            p.setPen(QPen(QColor(border), 1.0))
            p.setBrush(QColor(fill))
            p.drawRoundedRect(int(lx), int(cy), 9, 9, 1.5, 1.5)
            p.setPen(QColor(SemiconductorTheme.TEXT_SECONDARY))
            p.drawText(int(lx + 12), int(cy + 9), text)
            lx += 42
        p.end()


class MapStage(QWidget):
    """Level0 wafer die grid → Level1 selected die + P1–P9 overlay."""

    dieSelected = pyqtSignal(int, int)  # col, row
    fovSelected = pyqtSignal(int, int, int)  # col, row, fov_index
    levelChanged = pyqtSignal(int)  # 0 or 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctx: Optional[WaferContext] = None
        self._level = 0
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        panel = QFrame()
        panel.setObjectName("BatchPanel")
        panel.setMinimumHeight(280)
        panel.setMinimumWidth(280)
        outer.addWidget(panel, 1)

        lay = QVBoxLayout(panel)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # Header: title + chip status + Back
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        self._title = QLabel("MAP STAGE · Wafer")
        self._title.setObjectName("BatchSection")
        header.addWidget(self._title)

        self.chip_label = QLabel("Click a die on wafer map")
        self.chip_label.setObjectName("BatchReviewMuted")
        self.chip_label.setWordWrap(False)
        header.addWidget(self.chip_label, 1)

        self._btn_back = QPushButton("◀ Back")
        self._btn_back.setToolTip("Return to wafer map (Level 0)")
        self._btn_back.setFixedHeight(24)
        self._btn_back.setVisible(False)
        self._btn_back.clicked.connect(self.back_to_wafer)
        header.addWidget(self._btn_back)
        lay.addLayout(header)

        # Legend always visible
        self._legend = _BinLegendBar()
        lay.addWidget(self._legend)

        # Stacked canvases (Online paint parity)
        self._stack = QStackedWidget()
        self.wafer_map = _WaferMapCanvas()
        self.wafer_map.chip_clicked.connect(self._on_wafer_die)
        self._stack.addWidget(self.wafer_map)

        self.fov_map = _ChipMapCanvas()
        self.fov_map.fov_clicked.connect(self._on_chip_fov)
        self.fov_map.installEventFilter(self)
        self._stack.addWidget(self.fov_map)

        lay.addWidget(self._stack, 1)

    # ── Public API ────────────────────────────────────────────────
    def set_context(self, ctx: Optional[WaferContext]) -> None:
        self._ctx = ctx
        self.wafer_map.set_context(ctx)
        self.fov_map.set_context(ctx)
        self._update_chip_label()

    def set_level(self, level: int) -> None:
        level = 1 if int(level) == 1 else 0
        if level == 1 and (self._ctx is None or int(getattr(self._ctx, "selected_col", 0) or 0) <= 0):
            level = 0
        prev = self._level
        self._level = level
        self._stack.setCurrentIndex(level)
        self._btn_back.setVisible(level == 1)
        if level == 0:
            self._title.setText("MAP STAGE · Wafer")
        else:
            col = int(self._ctx.selected_col) if self._ctx else 0
            row = int(self._ctx.selected_row) if self._ctx else 0
            self._title.setText(f"MAP STAGE · Chip {col},{row}")
        self._update_chip_label()
        if prev != self._level:
            self.levelChanged.emit(self._level)

    def back_to_wafer(self) -> None:
        self.set_level(0)

    @property
    def level(self) -> int:
        return self._level

    # ── Internals ─────────────────────────────────────────────────
    def _update_chip_label(self) -> None:
        ctx = self._ctx
        if ctx is None or int(ctx.selected_col or 0) <= 0:
            self.chip_label.setText(
                "Click a die on wafer map" if self._level == 0 else "No die selected"
            )
            return
        chip = ctx.selected_chip()
        if chip is None:
            fb_txt = "—"
        elif chip.final_bin == BIN_NG:
            fb_txt = "NG"
        elif chip.final_bin == BIN_OK:
            fb_txt = "OK (9/9)"
        else:
            fb_txt = f"Pend ({chip.good_count}/9 OK)"
        self.chip_label.setText(
            f"Chip Col {ctx.selected_col} · Row {ctx.selected_row}  ·  Final {fb_txt}"
        )

    def _on_wafer_die(self, col: int, row: int) -> None:
        self.dieSelected.emit(int(col), int(row))
        # Tab refreshes WaferContext; still advance to L1 for standalone use
        self.set_level(1)

    def _on_chip_fov(self, fov_index: int) -> None:
        col = int(self._ctx.selected_col) if self._ctx else 0
        row = int(self._ctx.selected_row) if self._ctx else 0
        if col <= 0 or row <= 0:
            return
        if self._ctx is not None:
            self._ctx.select_fov(int(fov_index))
            self.fov_map.set_context(self._ctx)
            self.wafer_map.set_context(self._ctx)
        self.fovSelected.emit(col, row, int(fov_index))

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        # Click empty region on chip FOV canvas → Level 0
        if obj is self.fov_map and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:  # type: ignore[attr-defined]
                hit = self.fov_map._hit(event.pos())  # type: ignore[attr-defined]
                if hit is None:
                    self.back_to_wafer()
                    return False
        return super().eventFilter(obj, event)
