# inno3d/features/batch_review/map_stage.py
# -----------------------------------------------------------------------
# MapStage — single wafer canvas for Live Review.
# Zoom into a die to see P1–P9 FOV cells in-place (no page switch).
# -----------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from inno3d.core.wafer_context import BIN_CT_OK, BIN_OK, BIN_PENDING, WaferContext
from inno3d.widgets.context_map_panel import _WaferMapCanvas


class MapStage(QWidget):
    """Wafer die grid; zoom-in reveals FOV P1–P9 inside each die."""

    dieSelected = pyqtSignal(int, int)  # col, row
    fovSelected = pyqtSignal(int, int, int)  # col, row, fov_index
    levelChanged = pyqtSignal(int)  # 0 = fit wafer, 1 = focused die

    _DIE_FOCUS_ZOOM = 5.5

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

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        self._title = QLabel("MAP STAGE · Wafer")
        self._title.setObjectName("BatchSection")
        header.addWidget(self._title)

        self.chip_label = QLabel("Click a die · zoom in to see FOV P1–P9")
        self.chip_label.setObjectName("BatchReviewMuted")
        self.chip_label.setWordWrap(False)
        header.addWidget(self.chip_label, 1)

        self._btn_back = QPushButton("Fit wafer")
        self._btn_back.setToolTip("Zoom out to full wafer (double-click also fits)")
        self._btn_back.setFixedHeight(24)
        self._btn_back.setVisible(False)
        self._btn_back.clicked.connect(self.back_to_wafer)
        header.addWidget(self._btn_back)
        lay.addLayout(header)

        self.wafer_map = _WaferMapCanvas()
        self.wafer_map.chip_clicked.connect(self._on_wafer_die)
        self.wafer_map.fov_clicked.connect(self._on_wafer_fov)
        lay.addWidget(self.wafer_map, 1)
        # Alias kept for tab.py leftover references
        self.fov_map = self.wafer_map

    def set_context(self, ctx: Optional[WaferContext]) -> None:
        self._ctx = ctx
        self.wafer_map.set_context(ctx)
        self._update_chip_label()

    def set_level(self, level: int) -> None:
        """0 = fit whole wafer; 1 = focus selected die (never zoom out)."""
        level = 1 if int(level) == 1 else 0
        if level == 1 and (self._ctx is None or int(getattr(self._ctx, "selected_col", 0) or 0) <= 0):
            level = 0
        prev = self._level
        self._level = level
        self._btn_back.setVisible(level == 1)
        if level == 0:
            self._title.setText("MAP STAGE · Wafer")
            self.wafer_map.reset_view()
        else:
            col = int(self._ctx.selected_col) if self._ctx else 0
            row = int(self._ctx.selected_row) if self._ctx else 0
            self._title.setText(f"MAP STAGE · Chip {col},{row}")
            # Keep user's zoom if already in; only bump from overview → focus.
            cur = float(getattr(self.wafer_map, "_zoom", 1.0) or 1.0)
            z = self._DIE_FOCUS_ZOOM if cur <= 1.05 else cur
            self.wafer_map.zoom_to_selection(col, row, zoom=z)
        self._update_chip_label()
        if prev != self._level:
            self.levelChanged.emit(self._level)

    def back_to_wafer(self) -> None:
        self.set_level(0)

    @property
    def level(self) -> int:
        return self._level

    def _update_chip_label(self) -> None:
        ctx = self._ctx
        if ctx is None or int(ctx.selected_col or 0) <= 0:
            self.chip_label.setText("Click a die · zoom in to see FOV P1–P9")
            return
        chip = ctx.selected_chip()
        if chip is None:
            fb_txt = "—"
        else:
            fb_txt = chip.pair_label()
            if int(chip.final_bin) == BIN_PENDING:
                fb_txt = f"{fb_txt} ({chip.good_count}/9 OK)"
            elif int(chip.final_bin) in (BIN_OK, BIN_CT_OK):
                fb_txt = f"{fb_txt} (9/9)"
            fov = int(ctx.selected_fov or 0)
            if fov > 0:
                fb_txt = f"{fb_txt}  ·  FOV P{fov}"
        self.chip_label.setText(
            f"Chip Col {ctx.selected_col} · Row {ctx.selected_row}  ·  {fb_txt}"
        )

    def _on_wafer_die(self, col: int, row: int) -> None:
        if self._ctx is not None:
            self._ctx.select_chip(int(col), int(row))
            self.wafer_map.set_context(self._ctx)
        self.dieSelected.emit(int(col), int(row))
        self.set_level(1)

    def _on_wafer_fov(self, col: int, row: int, fov_index: int) -> None:
        if self._ctx is not None:
            self._ctx.select_chip(int(col), int(row))
            self._ctx.select_fov(int(fov_index))
            self.wafer_map.set_context(self._ctx)
            self._update_chip_label()
        self.fovSelected.emit(int(col), int(row), int(fov_index))
        if self._level != 1:
            self.set_level(1)
