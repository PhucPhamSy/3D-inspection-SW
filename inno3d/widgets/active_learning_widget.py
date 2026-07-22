"""
Active Learning Widget — Smart Labeling Dashboard.

Displays:
- Recommended slices list with uncertainty scores
- Per-slice entropy heatmap preview
- Annotation efficiency gauge
- "Go to slice" navigation
- Progress tracking
"""

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QProgressBar, QGroupBox,
    QSpinBox, QFrame, QSizePolicy, QAbstractItemView,
)
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QFont, QBrush, QLinearGradient,
    QImage, QPixmap, QPainterPath,
)

from inno3d.core.styles import SemiconductorTheme


# ═══════════════════════════════════════════════════════════════════════════════
# Entropy Mini-Map
# ═══════════════════════════════════════════════════════════════════════════════

class EntropyMiniMap(QWidget):
    """Vertical strip showing per-slice uncertainty as a heatmap bar."""

    slice_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(40)
        self.setMinimumHeight(80)
        self._entropy_per_slice = None   # (D,) float
        self._labeled = set()
        self._recommended = set()
        self._total = 0

    def set_data(self, entropy_per_slice, labeled, recommended):
        self._entropy_per_slice = entropy_per_slice
        self._labeled = set(labeled)
        self._recommended = set(recommended)
        self._total = len(entropy_per_slice) if entropy_per_slice is not None else 0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        p.fillRect(0, 0, w, h, QColor(SemiconductorTheme.BG_DARK))

        if self._entropy_per_slice is None or self._total == 0:
            p.setPen(QColor(SemiconductorTheme.TEXT_DISABLED))
            p.setFont(QFont("Segoe UI", 7))
            p.drawText(self.rect(), Qt.AlignCenter, "No\ndata")
            p.end()
            return

        bar_x = 6
        bar_w = w - 12
        bar_h = h - 4

        # Draw entropy bar
        e = self._entropy_per_slice
        e_max = max(e.max(), 1e-8)

        for z in range(self._total):
            y = 2 + int(z / self._total * bar_h)
            y_next = 2 + int((z + 1) / self._total * bar_h)
            row_h = max(1, y_next - y)

            val = e[z] / e_max

            if z in self._labeled:
                color = QColor(76, 175, 80, 200)  # Green = labeled
            elif z in self._recommended:
                color = QColor(255, 152, 0, 220)  # Orange = recommended
            else:
                # Blue → Red gradient by uncertainty
                r = int(val * 255)
                b = int((1 - val) * 200)
                color = QColor(r, 40, b, 150)

            p.fillRect(bar_x, y, bar_w, row_h, color)

        # Border
        p.setPen(QPen(QColor(SemiconductorTheme.BORDER_DEFAULT), 1))
        p.drawRect(bar_x, 2, bar_w, bar_h)

        p.end()

    def mousePressEvent(self, event):
        if self._total == 0:
            return
        h = self.height() - 4
        y = event.pos().y() - 2
        z = int(y / h * self._total)
        z = max(0, min(z, self._total - 1))
        self.slice_clicked.emit(z)


# ═══════════════════════════════════════════════════════════════════════════════
# Efficiency Gauge
# ═══════════════════════════════════════════════════════════════════════════════

class EfficiencyGauge(QWidget):
    """Shows annotation efficiency: labeled/total ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(90, 90)
        self._labeled = 0
        self._total = 1
        self._recommended = 0

    def set_values(self, labeled, total, recommended=0):
        self._labeled = labeled
        self._total = max(1, total)
        self._recommended = recommended
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 8

        # Background ring
        p.setPen(QPen(QColor(40, 45, 60), 6))
        p.drawArc(cx - r, cy - r, 2 * r, 2 * r, 225 * 16, -270 * 16)

        # Labeled arc (green)
        ratio = self._labeled / self._total
        span_labeled = int(-270 * ratio * 16)
        p.setPen(QPen(QColor(76, 175, 80), 6, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(cx - r, cy - r, 2 * r, 2 * r, 225 * 16, span_labeled)

        # Center text
        p.setPen(QColor(SemiconductorTheme.TEXT_PRIMARY))
        p.setFont(QFont("Segoe UI", 12, QFont.Bold))
        p.drawText(self.rect().adjusted(0, -8, 0, 0), Qt.AlignCenter,
                   f"{self._labeled}")

        p.setPen(QColor(SemiconductorTheme.TEXT_DISABLED))
        p.setFont(QFont("Segoe UI", 7))
        p.drawText(self.rect().adjusted(0, 16, 0, 0), Qt.AlignCenter,
                   f"/ {self._total} slices")

        p.end()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Widget
# ═══════════════════════════════════════════════════════════════════════════════

class ActiveLearningWidget(QWidget):
    """
    Smart Labeling Dashboard for Active Learning integration.
    """

    run_requested = pyqtSignal()             # Start AL scan
    goto_slice = pyqtSignal(int)             # Navigate to slice Z
    label_slice = pyqtSignal(int)            # Mark slice for labeling

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header
        header = QLabel("🧠 SMART LABELING")
        header.setStyleSheet(f"""
            font-size: 10pt; font-weight: bold;
            color: #b388ff;
            letter-spacing: 2px;
        """)
        layout.addWidget(header)

        # Top row: efficiency gauge + mini-map + stats
        top_row = QHBoxLayout()

        self.gauge = EfficiencyGauge()
        top_row.addWidget(self.gauge)

        self.minimap = EntropyMiniMap()
        self.minimap.slice_clicked.connect(self.goto_slice.emit)
        top_row.addWidget(self.minimap)

        # Stats column
        stats_col = QVBoxLayout()
        stats_col.setSpacing(3)

        self._stat_labels = {}
        stat_items = [
            ("Labeled:", "n_labeled"),
            ("Remaining:", "n_remaining"),
            ("MC Passes:", "mc_passes"),
            ("Top Uncertainty:", "top_uncertainty"),
        ]
        for name, key in stat_items:
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt;")
            val = QLabel("—")
            val.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 7.5pt; font-weight: bold;")
            row.addWidget(lbl)
            row.addWidget(val)
            row.addStretch()
            stats_col.addLayout(row)
            self._stat_labels[key] = val

        stats_col.addStretch()
        top_row.addLayout(stats_col, 1)
        layout.addLayout(top_row)

        # MC passes config
        mc_row = QHBoxLayout()
        mc_lbl = QLabel("MC Passes:")
        mc_lbl.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        mc_row.addWidget(mc_lbl)
        self.spin_mc_passes = QSpinBox()
        self.spin_mc_passes.setRange(3, 20)
        self.spin_mc_passes.setValue(5)
        self.spin_mc_passes.setFixedWidth(55)
        self.spin_mc_passes.setStyleSheet(f"""
            QSpinBox {{
                background: {SemiconductorTheme.BG_MEDIUM};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 3px;
                padding: 2px;
                font-size: 8pt;
            }}
        """)
        mc_row.addWidget(self.spin_mc_passes)

        rec_lbl = QLabel("Recommend:")
        rec_lbl.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        mc_row.addWidget(rec_lbl)
        self.spin_n_recommend = QSpinBox()
        self.spin_n_recommend.setRange(3, 50)
        self.spin_n_recommend.setValue(10)
        self.spin_n_recommend.setFixedWidth(55)
        self.spin_n_recommend.setStyleSheet(self.spin_mc_passes.styleSheet())
        mc_row.addWidget(self.spin_n_recommend)

        mc_row.addStretch()
        layout.addLayout(mc_row)

        # Run button
        self.btn_run = QPushButton("🧠 Find Best Slices to Label")
        self.btn_run.setFixedHeight(32)
        self.btn_run.setStyleSheet(f"""
            QPushButton {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: #b388ff;
                border: 1px solid #b388ff;
                border-radius: 4px;
                font-size: 9pt;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #b388ff;
                color: {SemiconductorTheme.BG_DARK};
            }}
            QPushButton:disabled {{
                background: {SemiconductorTheme.BG_MEDIUM};
                color: {SemiconductorTheme.TEXT_DISABLED};
                border-color: {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """)
        self.btn_run.clicked.connect(self.run_requested.emit)
        layout.addWidget(self.btn_run)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {SemiconductorTheme.BG_DARK};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px;
                text-align: center;
                color: {SemiconductorTheme.TEXT_SECONDARY};
                font-size: 7.5pt;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #7c4dff, stop:1 #b388ff);
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress_bar)

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 7.5pt; font-style: italic;"
        )
        layout.addWidget(self._lbl_status)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {SemiconductorTheme.BORDER_DEFAULT};")
        layout.addWidget(sep)

        # Recommended slices list
        rec_header = QLabel("📋 Recommended Slices (label these next)")
        rec_header.setStyleSheet(
            f"color: #FFB74D; font-size: 8.5pt; font-weight: bold;"
        )
        layout.addWidget(rec_header)

        self.slice_list = QListWidget()
        self.slice_list.setMinimumHeight(80)
        self.slice_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.slice_list.setStyleSheet(f"""
            QListWidget {{
                background: {SemiconductorTheme.BG_DARK};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px;
                font-size: 8pt;
            }}
            QListWidget::item {{
                padding: 4px 8px;
                border-bottom: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QListWidget::item:selected {{
                background: rgba(179, 136, 255, 0.2);
                color: #b388ff;
            }}
            QListWidget::item:hover {{
                background: rgba(179, 136, 255, 0.1);
            }}
        """)
        self.slice_list.itemDoubleClicked.connect(self._on_item_double_click)
        layout.addWidget(self.slice_list, 1)

        # Go to slice button
        self.btn_goto = QPushButton("▶ Go To Selected Slice")
        self.btn_goto.setFixedHeight(28)
        self.btn_goto.setStyleSheet(f"""
            QPushButton {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.ACCENT_SUCCESS};
                border: 1px solid {SemiconductorTheme.ACCENT_SUCCESS};
                border-radius: 4px;
                font-size: 8.5pt;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {SemiconductorTheme.ACCENT_SUCCESS};
                color: {SemiconductorTheme.BG_DARK};
            }}
        """)
        self.btn_goto.clicked.connect(self._on_goto_clicked)
        layout.addWidget(self.btn_goto)

        # Container style
        self.setStyleSheet(f"""
            ActiveLearningWidget {{
                background: {SemiconductorTheme.BG_MEDIUM};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 6px;
            }}
        """)

    # ── Event handlers ──

    def _on_item_double_click(self, item):
        z = item.data(Qt.UserRole)
        if z is not None:
            self.goto_slice.emit(z)

    def _on_goto_clicked(self):
        item = self.slice_list.currentItem()
        if item:
            z = item.data(Qt.UserRole)
            if z is not None:
                self.goto_slice.emit(z)

    # ── Public update methods ──

    def set_progress(self, value, total, message=""):
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(value)
        self._lbl_status.setText(message)

    def set_running(self, running):
        self.btn_run.setEnabled(not running)
        self.btn_run.setText(
            "⏳ Scanning..." if running else "🧠 Find Best Slices to Label"
        )
        self.progress_bar.setVisible(running)

    def update_state(self, state):
        """Update from ActiveLearningState."""
        n_labeled = len(state.labeled_slices)
        n_total = state.total_slices
        n_rec = len(state.recommended_slices)

        self.gauge.set_values(n_labeled, n_total, n_rec)

        self._stat_labels['n_labeled'].setText(f"{n_labeled}")
        self._stat_labels['n_remaining'].setText(f"{n_total - n_labeled}")
        self._stat_labels['mc_passes'].setText(f"{state.mc_passes}")

        # Top uncertainty
        if state.all_uncertainties:
            unlabeled = [u for u in state.all_uncertainties if not u.is_labeled]
            if unlabeled:
                top = max(unlabeled, key=lambda u: u.combined_score)
                self._stat_labels['top_uncertainty'].setText(
                    f"Z={top.slice_index} ({top.mean_entropy:.3f})"
                )

        # Entropy mini-map
        if state.entropy_map is not None:
            entropy_per_slice = np.array([
                state.entropy_map[z].mean()
                for z in range(state.entropy_map.shape[0])
            ])
            self.minimap.set_data(
                entropy_per_slice, state.labeled_slices, state.recommended_slices
            )

        # Recommended list
        self.slice_list.clear()
        for rank, z in enumerate(state.recommended_slices):
            u = next(
                (u for u in state.all_uncertainties if u.slice_index == z),
                None
            )
            if u:
                text = (
                    f"#{rank+1}  Z={z:>4d}  |  "
                    f"Entropy={u.mean_entropy:.3f}  "
                    f"Var={u.prediction_variance:.4f}  "
                    f"Score={u.combined_score:.3f}"
                )
            else:
                text = f"#{rank+1}  Z={z}"

            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, z)
            if rank < 3:
                item.setForeground(QColor('#FFB74D'))  # Top 3 highlighted
            self.slice_list.addItem(item)

        self._lbl_status.setText(
            f"✓ {n_rec} slices recommended — double-click or press 'Go To' to navigate"
        )
        self._lbl_status.setStyleSheet(
            f"color: #b388ff; font-size: 7.5pt;"
        )

        self.progress_bar.setVisible(False)
