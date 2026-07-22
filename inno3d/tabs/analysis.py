"""
3D Analysis Tab — SOH (State of Height of Bump) Analytics Workbench

DB-first: load FOV runs / MES objects from inspection.db, aggregate SOH by
layer / FOV / chip. CSV folder import remains as lab fallback.

Advanced modes (Cross-σ, Per-Bump σ, 3D-SIM, LOO-VD) operate on the loaded
analysis set (each FOV run ≈ one SampleData unit).
"""
import os
import csv
import glob
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set

import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# matplotlib for embedded charts
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from inno3d.core.styles import SemiconductorTheme
from inno3d.core.inspection_db import InspectionDB, get_db
from inno3d.core.soh_data import (
    SampleData,
    load_samples_from_db,
    aggregate_layers_from_samples,
    aggregate_fov_from_samples,
)
from matplotlib.patches import Rectangle as MplRect
from matplotlib.patches import Patch


def _natural_sort_key(s: str):
    """Sort key for natural ordering: 'Layer 2' < 'Layer 12'."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]


# ──────────────────────────── Data Models ────────────────────────────

@dataclass
class LayerStats:
    """SOH statistics for a single layer."""
    layer_name: str
    object_count: int = 0
    ng_count: int = 0
    ng_rate: float = 0.0
    mean_z_height: float = 0.0
    std_z_height: float = 0.0
    min_z_height: float = 0.0
    max_z_height: float = 0.0
    mean_bump_volume: float = 0.0
    total_bump_volume: float = 0.0
    total_void_volume: float = 0.0
    mean_void_ratio: float = 0.0
    max_void_ratio: float = 0.0


# ──────────────────────────── CSV Parser ─────────────────────────────

def parse_measurement_csv(csv_path: str) -> List[dict]:
    """Parse a measurement CSV exported from 3D Teaching tab.
    
    Returns list of dicts, one per row.
    Handles both per-layer CSVs and combined object_statistics.csv.
    """
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            # Normalize column names (strip whitespace)
            for key, val in row.items():
                key = key.strip()
                val = val.strip() if val else ''
                parsed[key] = val
            rows.append(parsed)
    return rows


def _safe_float(val: str) -> float:
    """Convert string to float, handling commas and percentages."""
    if not val or val == 'N/A':
        return 0.0
    val = val.replace(',', '').replace('%', '').strip()
    try:
        return float(val)
    except ValueError:
        return 0.0


def compute_layer_stats(rows: List[dict], layer_name: str, ng_threshold: float) -> LayerStats:
    """Compute SOH statistics for a set of rows belonging to one layer."""
    stats = LayerStats(layer_name=layer_name)
    if not rows:
        return stats

    z_heights = []
    bump_volumes = []
    void_volumes = []
    ratios = []

    for row in rows:
        # Try multiple possible column name variants for each field
        bump_v = _safe_float(
            row.get('Bump Volume (um3)',
            row.get('Bump Vol(µm³)',
            row.get('Bump Vol(um3)', '0'))))
        void_v = _safe_float(
            row.get('Void Volume (um3)',
            row.get('Void Vol(µm³)',
            row.get('Void Vol(um3)', '0'))))
        ratio = _safe_float(
            row.get('Ratio (Void/(TGV+Void))',
            row.get('Ratio', '0')))

        # Ratio in CSV is like "5.123%" → _safe_float strips '%' → 5.123 → convert to fraction
        if ratio > 1.0:
            ratio = ratio / 100.0

        # Z Height: try dedicated column first, otherwise compute from Z_min/Z_max
        z_h = _safe_float(row.get('SOH (um)', row.get('SOH(µm)', 
              row.get('Z Height (um)', row.get('Z Height(µm)', '')))))
        if z_h == 0.0:
            z_min = _safe_float(row.get('Z_min', '0'))
            z_max = _safe_float(row.get('Z_max', '0'))
            z_h = abs(z_max - z_min)

        z_heights.append(z_h)
        bump_volumes.append(bump_v)
        void_volumes.append(void_v)
        ratios.append(ratio)

    z_arr = np.array(z_heights)
    ratio_arr = np.array(ratios)

    stats.object_count = len(rows)
    stats.ng_count = int(np.sum(ratio_arr >= ng_threshold))
    stats.ng_rate = stats.ng_count / stats.object_count if stats.object_count > 0 else 0.0

    if len(z_arr) > 0:
        stats.mean_z_height = float(np.mean(z_arr))
        stats.std_z_height = float(np.std(z_arr))
        stats.min_z_height = float(np.min(z_arr))
        stats.max_z_height = float(np.max(z_arr))

    stats.mean_bump_volume = float(np.mean(bump_volumes)) if bump_volumes else 0.0
    stats.total_bump_volume = float(np.sum(bump_volumes))
    stats.total_void_volume = float(np.sum(void_volumes))
    stats.mean_void_ratio = float(np.mean(ratio_arr)) if len(ratio_arr) > 0 else 0.0
    stats.max_void_ratio = float(np.max(ratio_arr)) if len(ratio_arr) > 0 else 0.0

    return stats


# ──────────────────────────── Chart Widget ───────────────────────────

class EmbeddedChart(FigureCanvas):
    """Matplotlib chart embedded in Qt."""

    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0a0e1a')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear(self):
        self.fig.clear()
        self.draw()

    def plot_intra_sample(self, sample: SampleData, metric: str = 'mean_z_height'):
        """Bar chart: compare layers within a single sample."""
        self.fig.clear()
        if not sample.layers:
            self.draw()
            return

        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0f1320')

        layer_names = sorted(sample.layers.keys(), key=_natural_sort_key)
        values = [getattr(sample.layers[ln], metric, 0.0) for ln in layer_names]

        # Color bars — highlight NG-heavy layers in red
        colors = []
        for ln in layer_names:
            ng_rate = sample.layers[ln].ng_rate
            if ng_rate >= 0.1:
                colors.append('#ff4444')
            elif ng_rate >= 0.05:
                colors.append('#ff8800')
            else:
                colors.append('#00e5ff')

        bars = ax.bar(range(len(layer_names)), values, color=colors, edgecolor='#1a2040', linewidth=0.8)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, color='#e0e0e0')

        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, fontsize=8, color='#b0b0b0', rotation=30, ha='right')
        ax.tick_params(axis='y', colors='#b0b0b0', labelsize=8)

        metric_labels = {
            'mean_z_height': 'Mean Z Height (µm)',
            'ng_rate': 'NG Rate',
            'mean_void_ratio': 'Mean Void Ratio',
            'max_void_ratio': 'Max Void Ratio',
            'total_void_volume': 'Total Void Volume (µm³)',
            'mean_bump_volume': 'Mean Bump Volume (µm³)',
        }
        title = f"{sample.name} — {metric_labels.get(metric, metric)}"
        ax.set_title(title, fontsize=10, color='#00e5ff', pad=10)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=9, color='#b0b0b0')

        # Grid
        ax.grid(axis='y', color='#1a2040', linewidth=0.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2a3050')
        ax.spines['left'].set_color('#2a3050')

        self.fig.tight_layout()
        self.draw()

    def plot_inter_sample(self, samples: List[SampleData], metric: str = 'mean_z_height'):
        """Grouped bar chart: compare corresponding layers across all samples."""
        self.fig.clear()
        if not samples:
            self.draw()
            return

        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0f1320')

        # Collect all unique layer names across all samples (natural sorted)
        all_layers = []
        for s in samples:
            for ln in s.layers:
                if ln not in all_layers:
                    all_layers.append(ln)

        all_layers.sort(key=_natural_sort_key)

        if not all_layers:
            self.draw()
            return

        n_samples = len(samples)
        n_layers = len(all_layers)
        bar_width = 0.8 / max(n_samples, 1)
        x = np.arange(n_layers)

        # Color palette for samples
        sample_colors = ['#00e5ff', '#ff6b9d', '#76ff03', '#ffc107', '#e040fb',
                         '#00bfa5', '#ff5252', '#448aff', '#ffab40', '#69f0ae']

        for s_idx, sample in enumerate(samples):
            values = []
            for ln in all_layers:
                if ln in sample.layers:
                    values.append(getattr(sample.layers[ln], metric, 0.0))
                else:
                    values.append(0.0)
            offset = (s_idx - n_samples / 2 + 0.5) * bar_width
            color = sample_colors[s_idx % len(sample_colors)]
            ax.bar(x + offset, values, bar_width * 0.9, label=sample.name,
                   color=color, alpha=0.85, edgecolor='#1a2040', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(all_layers, fontsize=8, color='#b0b0b0', rotation=30, ha='right')
        ax.tick_params(axis='y', colors='#b0b0b0', labelsize=8)

        metric_labels = {
            'mean_z_height': 'Mean Z Height (µm)',
            'ng_rate': 'NG Rate',
            'mean_void_ratio': 'Mean Void Ratio',
            'max_void_ratio': 'Max Void Ratio',
            'total_void_volume': 'Total Void Volume (µm³)',
            'mean_bump_volume': 'Mean Bump Volume (µm³)',
        }
        ax.set_title(f"Cross-Sample Comparison — {metric_labels.get(metric, metric)}",
                     fontsize=10, color='#00e5ff', pad=10)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=9, color='#b0b0b0')

        ax.legend(fontsize=8, loc='upper right', framealpha=0.5,
                  facecolor='#0f1320', edgecolor='#2a3050', labelcolor='#e0e0e0')
        ax.grid(axis='y', color='#1a2040', linewidth=0.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2a3050')
        ax.spines['left'].set_color('#2a3050')

        self.fig.tight_layout()
        self.draw()

    def plot_scope_layers(self, layers: Dict[str, LayerStats], metric: str = 'mean_z_height',
                          title: str = "SOH by Layer (scope)"):
        """Bar chart of scope-level layer aggregates (not a single FOV)."""
        self.fig.clear()
        if not layers:
            self.draw()
            return
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0f1320')
        layer_names = sorted(layers.keys(), key=_natural_sort_key)
        values = [getattr(layers[ln], metric, 0.0) for ln in layer_names]
        colors = []
        for ln in layer_names:
            ng_rate = layers[ln].ng_rate
            if ng_rate >= 0.1:
                colors.append('#ff4444')
            elif ng_rate >= 0.05:
                colors.append('#ff8800')
            else:
                colors.append('#00e5ff')
        bars = ax.bar(range(len(layer_names)), values, color=colors,
                      edgecolor='#1a2040', linewidth=0.8)
        for bar, val in zip(bars, values):
            if metric == 'ng_rate':
                lbl = f'{val * 100:.1f}%'
            elif metric in ('mean_void_ratio', 'max_void_ratio'):
                lbl = f'{val * 100:.2f}%'
            else:
                lbl = f'{val:.2f}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    lbl, ha='center', va='bottom', fontsize=7, color='#e0e0e0')
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, fontsize=8, color='#b0b0b0', rotation=30, ha='right')
        ax.tick_params(axis='y', colors='#b0b0b0', labelsize=8)
        ax.set_title(title, fontsize=10, color='#00e5ff', pad=10)
        ax.grid(axis='y', color='#1a2040', linewidth=0.5, alpha=0.7)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color('#2a3050')
        ax.spines['left'].set_color('#2a3050')
        self.fig.tight_layout()
        self.draw()

    def plot_fov_heatmap(self, fov_stats: List[dict], metric: str = 'mean_soh',
                         title: str = "SOH by FOV Position (P1–P9)"):
        """3×3 FOV heatmap (row-major P1..P9)."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0f1320')

        by_idx = {}
        for d in fov_stats or []:
            try:
                fi = int(d.get('key') if 'key' in d else d.get('fov_index', 0))
            except (TypeError, ValueError):
                continue
            if 1 <= fi <= 9:
                by_idx[fi] = d

        grid = np.full((3, 3), np.nan)
        annot = [[''] * 3 for _ in range(3)]
        for i in range(9):
            r, c = divmod(i, 3)
            fi = i + 1
            d = by_idx.get(fi)
            if not d:
                annot[r][c] = f'P{fi}\n—'
                continue
            if metric in ('mean_soh', 'mean_z_height'):
                val = float(d.get('mean_soh', d.get('mean_z_height', 0)) or 0)
                annot[r][c] = f'P{fi}\n{val:.2f}'
            elif metric == 'ng_rate':
                val = float(d.get('ng_rate', 0) or 0)
                annot[r][c] = f'P{fi}\n{val * 100:.1f}%'
            else:
                val = float(d.get(metric, d.get('mean_soh', 0)) or 0)
                annot[r][c] = f'P{fi}\n{val:.2f}'
            grid[r, c] = val

        if np.all(np.isnan(grid)):
            ax.text(0.5, 0.5, 'No FOV data in scope', ha='center', va='center',
                    color='#808080', fontsize=11, transform=ax.transAxes)
            ax.set_axis_off()
            self.draw()
            return

        try:
            cmap = matplotlib.colormaps['coolwarm'].copy()
        except Exception:
            cmap = matplotlib.cm.get_cmap('coolwarm')
            if hasattr(cmap, 'copy'):
                cmap = cmap.copy()
        if hasattr(cmap, 'set_bad'):
            cmap.set_bad(color='#1a2040')
        im = ax.imshow(grid, cmap=cmap, aspect='equal')
        for r in range(3):
            for c in range(3):
                ax.text(c, r, annot[r][c], ha='center', va='center',
                        color='#f0f0f0', fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, color='#00e5ff', pad=10)
        for sp in ax.spines.values():
            sp.set_color('#2a3050')
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.fig.tight_layout()
        self.draw()


# ──────────────────────────── Main Tab ───────────────────────────────

class AnalysisTab(QWidget):
    """3D Analysis Tab — DB-first SOH workbench."""

    def __init__(self):
        super().__init__()
        self.samples: List[SampleData] = []
        self._selected_sample_idx: Optional[int] = None
        self._db: InspectionDB = get_db()
        self._scope_wafer_key: str = ""
        self._scope_lot: str = ""
        self._scope_run_ids: Set[str] = set()
        self._db_lots: List[Dict[str, Any]] = []
        self._db_wafers: List[Dict[str, Any]] = []
        self._auto_loaded = False
        self.init_ui()
        # Defer DB browse until first show (avoids slow import path)
        QTimer.singleShot(0, self.refresh_from_db)

    # ────────── UI ──────────

    def init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Main splitter ──
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {SemiconductorTheme.BORDER_DEFAULT}; }}")

        left = self._create_left_panel()
        splitter.addWidget(left)

        right = self._create_right_panel()
        splitter.addWidget(right)

        splitter.setSizes([340, 900])
        root.addWidget(splitter, 1)

        self.status_label = QLabel("Ready — Loading inspection database…")
        self.status_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt; padding: 4px;")
        root.addWidget(self.status_label)

    # ── Left panel: scope browser + params + analysis set ──

    def _create_left_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(300)
        panel.setStyleSheet(f"""
            QWidget {{
                background: {SemiconductorTheme.BG_PANEL};
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Scope header ──
        scope_hdr = QLabel("<b>SCOPE BROWSER</b>  ·  DB-first")
        scope_hdr.setStyleSheet(
            f"font-size: 10pt; color: {SemiconductorTheme.ACCENT_PRIMARY}; padding: 2px 0;")
        layout.addWidget(scope_hdr)

        # Filters
        filt_grid = QGridLayout()
        filt_grid.setSpacing(4)
        lbl_ss = f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt; font-weight: bold;"

        lot_lbl = QLabel("Lot")
        lot_lbl.setStyleSheet(lbl_ss)
        self.lot_combo = QComboBox()
        self.lot_combo.setToolTip("Date · Lot/FOUP")
        self.lot_combo.currentIndexChanged.connect(self._on_lot_filter_changed)
        filt_grid.addWidget(lot_lbl, 0, 0)
        filt_grid.addWidget(self.lot_combo, 0, 1)

        waf_lbl = QLabel("Wafer")
        waf_lbl.setStyleSheet(lbl_ss)
        self.wafer_combo = QComboBox()
        self.wafer_combo.currentIndexChanged.connect(self._on_wafer_filter_changed)
        filt_grid.addWidget(waf_lbl, 1, 0)
        filt_grid.addWidget(self.wafer_combo, 1, 1)
        layout.addLayout(filt_grid)

        # Scope KPI strip
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(4)
        self.kpi_labels = {}
        for key, title in (
            ("fov", "FOV"), ("obj", "Obj"), ("yield", "Yld"), ("soh", "SOH"),
        ):
            box = QFrame()
            box.setStyleSheet(
                f"background: {SemiconductorTheme.BG_DARK}; border: 1px solid "
                f"{SemiconductorTheme.BORDER_DEFAULT}; border-radius: 4px; padding: 2px;")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(4, 2, 4, 2)
            bl.setSpacing(0)
            v = QLabel("—")
            v.setAlignment(Qt.AlignCenter)
            v.setStyleSheet(
                f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-weight: bold; font-size: 9pt;")
            t = QLabel(title)
            t.setAlignment(Qt.AlignCenter)
            t.setStyleSheet(f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 7pt;")
            bl.addWidget(v)
            bl.addWidget(t)
            kpi_row.addWidget(box)
            self.kpi_labels[key] = v
        layout.addLayout(kpi_row)

        # Scope actions
        act_row = QHBoxLayout()
        act_row.setSpacing(4)
        self.btn_load_wafer = QPushButton("Load wafer")
        self.btn_load_wafer.setToolTip("Load all FOV runs on selected wafer into analysis set")
        self.btn_load_wafer.setCursor(Qt.PointingHandCursor)
        self.btn_load_wafer.clicked.connect(self.load_selected_wafer)
        act_row.addWidget(self.btn_load_wafer)

        self.btn_load_checked = QPushButton("Load ✓")
        self.btn_load_checked.setToolTip("Load only checked FOV runs")
        self.btn_load_checked.setCursor(Qt.PointingHandCursor)
        self.btn_load_checked.clicked.connect(self.load_checked_fovs)
        act_row.addWidget(self.btn_load_checked)

        self.btn_refresh_db = QPushButton("↻")
        self.btn_refresh_db.setFixedWidth(32)
        self.btn_refresh_db.setToolTip("Refresh lot/wafer list from DB")
        self.btn_refresh_db.setCursor(Qt.PointingHandCursor)
        self.btn_refresh_db.clicked.connect(self.refresh_from_db)
        act_row.addWidget(self.btn_refresh_db)
        layout.addLayout(act_row)

        # ── NG + Metric ──
        params_row = QHBoxLayout()
        params_row.setSpacing(4)
        ng_label = QLabel("NG:")
        ng_label.setStyleSheet(lbl_ss)
        params_row.addWidget(ng_label)

        self.ng_threshold_spin = QDoubleSpinBox()
        self.ng_threshold_spin.setRange(0.0, 100.0)
        self.ng_threshold_spin.setValue(5.0)
        self.ng_threshold_spin.setSingleStep(0.5)
        self.ng_threshold_spin.setSuffix(" %")
        self.ng_threshold_spin.setDecimals(2)
        self.ng_threshold_spin.setFixedWidth(72)
        self.ng_threshold_spin.setToolTip("Objects with Void Ratio ≥ threshold → NG")
        params_row.addWidget(self.ng_threshold_spin)

        btn_apply = QPushButton("Apply")
        btn_apply.setFixedWidth(48)
        btn_apply.setCursor(Qt.PointingHandCursor)
        btn_apply.setToolTip("Recalculate NG metrics")
        btn_apply.clicked.connect(self._recalculate_all)
        params_row.addWidget(btn_apply)

        metric_label = QLabel("Metric:")
        metric_label.setStyleSheet(lbl_ss)
        params_row.addWidget(metric_label)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems([
            "Mean Z Height", "NG Rate", "Mean Void Ratio",
            "Max Void Ratio", "Total Void Volume", "Mean Bump Volume"
        ])
        self.metric_combo.currentIndexChanged.connect(self._refresh_charts)
        params_row.addWidget(self.metric_combo, 1)
        layout.addLayout(params_row)

        # ── %Tol ──
        tol_row = QHBoxLayout()
        tol_row.setSpacing(4)
        tol_label = QLabel("%Tol:")
        tol_label.setStyleSheet(lbl_ss)
        tol_label.setToolTip("%Tol Excellent / Acceptable thresholds")
        tol_row.addWidget(tol_label)
        self.tol_excellent_spin = QDoubleSpinBox()
        self.tol_excellent_spin.setRange(0.1, 50.0)
        self.tol_excellent_spin.setValue(5.0)
        self.tol_excellent_spin.setSingleStep(0.5)
        self.tol_excellent_spin.setSuffix(" %")
        self.tol_excellent_spin.setDecimals(1)
        self.tol_excellent_spin.setFixedWidth(65)
        self.tol_excellent_spin.valueChanged.connect(self._refresh_charts)
        tol_row.addWidget(self.tol_excellent_spin)
        dash_label = QLabel("-")
        dash_label.setStyleSheet(lbl_ss)
        tol_row.addWidget(dash_label)
        self.tol_acceptable_spin = QDoubleSpinBox()
        self.tol_acceptable_spin.setRange(0.1, 100.0)
        self.tol_acceptable_spin.setValue(15.0)
        self.tol_acceptable_spin.setSingleStep(1.0)
        self.tol_acceptable_spin.setSuffix(" %")
        self.tol_acceptable_spin.setDecimals(1)
        self.tol_acceptable_spin.setFixedWidth(65)
        self.tol_acceptable_spin.valueChanged.connect(self._refresh_charts)
        tol_row.addWidget(self.tol_acceptable_spin, 1)
        layout.addLayout(tol_row)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet(f"color: {SemiconductorTheme.BORDER_DEFAULT};")
        layout.addWidget(sep1)

        # ── VIEW MODE ──
        view_label = QLabel("VIEW MODE")
        view_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 7pt; "
            f"font-weight: bold; letter-spacing: 1px; padding: 2px 0;")
        layout.addWidget(view_label)

        view_row = QHBoxLayout()
        view_row.setSpacing(2)
        self._view_mode_buttons = []
        view_configs = [
            (" Overview", 0, "bar-chart-2.svg"),
            (" Cross-σ", 1, "trending-up.svg"),
            (" Bump σ", 2, "grid.svg"),
            (" Spatial", 3, "zap.svg"),
            (" LOO", 4, "crosshair.svg"),
        ]
        for text, mode_idx, icon_name in view_configs:
            btn = QPushButton(text)
            btn.setProperty("role", "toggle")
            btn.setProperty("theme_icon", True)
            btn.setProperty("icon_name", icon_name)
            btn.setProperty("icon_size_w", 14)
            btn.setProperty("icon_size_h", 14)
            icon_path = os.path.join("assets", "icons", icon_name)
            if os.path.exists(icon_path):
                btn.setIcon(QIcon(icon_path))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setCheckable(True)
            btn.setChecked(mode_idx == 0)
            btn.setStyleSheet("font-size: 8pt; padding: 3px 4px;")
            btn.clicked.connect(lambda checked, idx=mode_idx: self._set_view_mode(idx))
            view_row.addWidget(btn)
            self._view_mode_buttons.append(btn)
        layout.addLayout(view_row)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"color: {SemiconductorTheme.BORDER_DEFAULT};")
        layout.addWidget(sep2)

        # ── Scope tree (DB chips/FOVs) + Analysis set tree ──
        header = QLabel("<b>WAFER TREE</b>  <span style='color:#808080;font-size:8pt'>(check FOVs)</span>")
        header.setStyleSheet(f"font-size: 9pt; color: {SemiconductorTheme.ACCENT_PRIMARY}; padding: 2px;")
        layout.addWidget(header)

        self.scope_tree = QTreeWidget()
        self.scope_tree.setHeaderLabels(["Chip / FOV", "Obj", "Jdg"])
        self.scope_tree.setColumnCount(3)
        self.scope_tree.setAlternatingRowColors(True)
        self.scope_tree.setRootIsDecorated(True)
        self.scope_tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.scope_tree.setStyleSheet(f"""
            QTreeWidget {{
                background: {SemiconductorTheme.BG_DARK};
                alternate-background-color: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px; font-size: 8pt;
            }}
            QTreeWidget::item {{ padding: 2px 3px; }}
            QTreeWidget::item:selected {{
                background-color: rgba(0, 229, 255, 0.2);
            }}
            QHeaderView::section {{
                background: {SemiconductorTheme.BG_PANEL};
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                font-weight: bold; font-size: 7pt; padding: 3px;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """)
        self.scope_tree.header().setStretchLastSection(False)
        self.scope_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 3):
            self.scope_tree.header().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.scope_tree.itemChanged.connect(self._on_scope_item_changed)
        self.scope_tree.itemSelectionChanged.connect(self._on_scope_selection_changed)
        layout.addWidget(self.scope_tree, 1)

        set_hdr = QLabel("<b>ANALYSIS SET</b>  <span style='color:#808080;font-size:8pt'>(loaded FOVs)</span>")
        set_hdr.setStyleSheet(f"font-size: 9pt; color: {SemiconductorTheme.ACCENT_PRIMARY}; padding: 2px;")
        layout.addWidget(set_hdr)

        # Keep sample_tree name for advanced modes / remove / legacy
        self.sample_tree = QTreeWidget()
        self.sample_tree.setHeaderLabels(["FOV / Layer", "Obj", "NG", "NG%"])
        self.sample_tree.setColumnCount(4)
        self.sample_tree.setAlternatingRowColors(True)
        self.sample_tree.setRootIsDecorated(True)
        self.sample_tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.sample_tree.setStyleSheet(self.scope_tree.styleSheet())
        self.sample_tree.header().setStretchLastSection(False)
        self.sample_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 4):
            self.sample_tree.header().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.sample_tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self.sample_tree.header().setSectionsClickable(True)
        self.sample_tree.header().sectionClicked.connect(self._on_tree_header_clicked)
        self.sample_tree.setMaximumHeight(180)
        layout.addWidget(self.sample_tree)

        # Detail info
        detail_group = QGroupBox("Focus")
        detail_group.setStyleSheet(f"""
            QGroupBox {{
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                font-weight: bold; font-size: 9pt;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px; margin-top: 6px; padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 10px; padding: 0 6px;
            }}
        """)
        detail_layout = QVBoxLayout(detail_group)
        detail_layout.setContentsMargins(6, 6, 6, 6)
        detail_layout.setSpacing(2)
        self.detail_labels = {}
        for key, label_text in [
            ('sample', 'Unit:'), ('path', 'Path:'), ('layer', 'Layer:'),
            ('objects', 'Objects:'), ('ng_count', 'NG:'), ('ng_rate', 'NG%:'),
            ('mean_zh', 'Mean SOH:'), ('std_zh', 'σ SOH:'),
            ('min_zh', 'Min:'), ('max_zh', 'Max:'),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setStyleSheet(
                f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt; min-width: 70px;")
            val = QLabel("—")
            val.setStyleSheet(
                f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt; font-weight: bold;")
            val.setWordWrap(True)
            row.addWidget(lbl)
            row.addWidget(val, 1)
            detail_layout.addLayout(row)
            self.detail_labels[key] = val
        layout.addWidget(detail_group)
        return panel


    def get_tol_thresholds(self):
        """Get the %Tol thresholds (excellent, acceptable)."""
        if hasattr(self, 'tol_excellent_spin') and hasattr(self, 'tol_acceptable_spin'):
            return self.tol_excellent_spin.value(), self.tol_acceptable_spin.value()
        return 5.0, 15.0

    def _set_view_mode(self, mode_idx: int):
        """Switch between SOH(0), Cross-σ(1), Per-Bump σ(2), 3D-SIM(3), LOO-VD(4)."""
        self._current_view_mode = mode_idx
        for i, btn in enumerate(self._view_mode_buttons):
            btn.setChecked(i == mode_idx)

        # Update QStackedWidget to show the correct page
        if hasattr(self, '_right_vsplit') and isinstance(self._right_vsplit, QStackedWidget):
            self._right_vsplit.setCurrentIndex(mode_idx)


    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Use QStackedWidget for 3 view modes
        self._right_vsplit = QStackedWidget()

        # ── PAGE 0: SOH Summary + Charts ──
        soh_page = QWidget()
        soh_layout = QVBoxLayout(soh_page)
        soh_layout.setContentsMargins(0, 0, 0, 0)
        soh_layout.setSpacing(6)

        # KPI strip (scope)
        self.overview_kpi_row = QHBoxLayout()
        self.overview_kpi_row.setSpacing(6)
        self.overview_kpis = {}
        for key, title in (
            ("layers", "Layers"), ("mean_soh", "Mean SOH"), ("sigma", "σ SOH"),
            ("ng", "NG rate"), ("void", "Mean void"),
        ):
            card = QFrame()
            card.setStyleSheet(
                f"background: {SemiconductorTheme.BG_PANEL}; border: 1px solid "
                f"{SemiconductorTheme.BORDER_DEFAULT}; border-radius: 8px;")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(10, 6, 10, 6)
            cl.setSpacing(0)
            vv = QLabel("—")
            vv.setStyleSheet(
                f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-size: 14pt; font-weight: bold;")
            tt = QLabel(title.upper())
            tt.setStyleSheet(
                f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 8pt; letter-spacing: 0.5px;")
            cl.addWidget(vv)
            cl.addWidget(tt)
            self.overview_kpi_row.addWidget(card)
            self.overview_kpis[key] = vv
        soh_layout.addLayout(self.overview_kpi_row)

        # Summary table
        table_header = QLabel("<b>SOH SUMMARY</b>  ·  by Layer (analysis set)")
        table_header.setStyleSheet(
            f"font-size: 10pt; color: {SemiconductorTheme.ACCENT_PRIMARY}; padding: 4px 8px;")
        soh_layout.addWidget(table_header)

        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(12)
        self.summary_table.setHorizontalHeaderLabels([
            "Idx.", "FOV / Unit", "Layer", "Objects", "NG", "NG%",
            "Mean SOH", "σ SOH", "Min", "Max",
            "Mean Void%", "Max Void%"
        ])
        self.summary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.setStyleSheet(f"""
            QTableWidget {{
                background: {SemiconductorTheme.BG_DARK};
                alternate-background-color: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                gridline-color: {SemiconductorTheme.BORDER_DEFAULT};
                font-size: 9pt;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QTableWidget::item:selected {{
                background-color: rgba(0, 229, 255, 0.15);
                color: white;
            }}
            QHeaderView::section {{
                background: {SemiconductorTheme.BG_PANEL};
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                font-weight: bold; font-size: 8pt; padding: 3px;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """)
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setMaximumHeight(250)
        soh_layout.addWidget(self.summary_table)

        # Charts area — Two charts side by side
        chart_splitter = QSplitter(Qt.Horizontal)
        chart_splitter.setHandleWidth(3)
        chart_splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {SemiconductorTheme.BORDER_DEFAULT}; }}")

        # SOH by Layer (scope aggregate)
        intra_container = QWidget()
        intra_layout = QVBoxLayout(intra_container)
        intra_layout.setContentsMargins(4, 4, 4, 4)
        intra_layout.setSpacing(2)
        self.intra_label = QLabel("<b>SOH BY LAYER</b>  ·  analysis set aggregate")
        self.intra_label.setStyleSheet(
            f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-size: 9pt; padding: 2px;")
        intra_layout.addWidget(self.intra_label)
        self.intra_chart = EmbeddedChart(parent=intra_container, width=5, height=3.5)
        intra_layout.addWidget(self.intra_chart, 1)
        chart_splitter.addWidget(intra_container)

        # SOH by FOV position (P1–P9 heatmap)
        inter_container = QWidget()
        inter_layout = QVBoxLayout(inter_container)
        inter_layout.setContentsMargins(4, 4, 4, 4)
        inter_layout.setSpacing(2)
        self.inter_label = QLabel("<b>SOH BY FOV</b>  ·  P1–P9 position map")
        self.inter_label.setStyleSheet(
            f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-size: 9pt; padding: 2px;")
        inter_layout.addWidget(self.inter_label)
        self.inter_chart = EmbeddedChart(parent=inter_container, width=5, height=3.5)
        inter_layout.addWidget(self.inter_chart, 1)
        chart_splitter.addWidget(inter_container)

        chart_splitter.setSizes([500, 500])
        soh_layout.addWidget(chart_splitter, 1)
        self._right_vsplit.addWidget(soh_page)

        # ── PAGE 1: Cross-Sample Sigma Analysis ──
        sigma_page = QWidget()
        sigma_layout = QVBoxLayout(sigma_page)
        sigma_layout.setContentsMargins(0, 0, 0, 0)
        sigma_layout.setSpacing(6)

        sigma_header_row = QHBoxLayout()
        sigma_title = QLabel(
            "<b>CROSS-SAMPLE SIGMA ANALYSIS</b>  "
            "<span style='color: #808080;'>Repeatability / Tolerance Test</span>")
        sigma_title.setStyleSheet("font-size: 10pt; color: #ff6b9d; padding: 4px 8px;")
        sigma_header_row.addWidget(sigma_title)
        sigma_header_row.addStretch()

        sigma_metric_label = QLabel("Sigma Metric:")
        sigma_metric_label.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-weight: bold;")
        sigma_header_row.addWidget(sigma_metric_label)

        self.sigma_metric_combo = QComboBox()
        self.sigma_metric_combo.addItems(["SOH / Z Height", "Void Ratio", "Bump Volume"])
        self.sigma_metric_combo.setFixedWidth(150)
        self.sigma_metric_combo.setStyleSheet(f"""
            QComboBox {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 3px; padding: 4px 8px; font-size: 9pt;
            }}
        """)
        self.sigma_metric_combo.currentIndexChanged.connect(self._refresh_sigma)
        sigma_header_row.addWidget(self.sigma_metric_combo)
        sigma_layout.addLayout(sigma_header_row)

        # Sigma content: table (top) + chart (bottom)
        sigma_splitter = QSplitter(Qt.Vertical)
        sigma_splitter.setHandleWidth(3)
        sigma_splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {SemiconductorTheme.BORDER_DEFAULT}; }}")

        self.sigma_table = QTableWidget()
        self.sigma_table.setColumnCount(8)
        self.sigma_table.setHorizontalHeaderLabels([
            "Layer", "N Samples", "Mean", "StdDev.S",
            "%Tol", "Min", "Max", "Range"
        ])
        self.sigma_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sigma_table.setAlternatingRowColors(True)
        self.sigma_table.setStyleSheet(f"""
            QTableWidget {{
                background: {SemiconductorTheme.BG_DARK};
                alternate-background-color: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                gridline-color: {SemiconductorTheme.BORDER_DEFAULT};
                font-size: 9pt;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QTableWidget::item:selected {{
                background-color: rgba(255, 107, 157, 0.15);
                color: white;
            }}
            QHeaderView::section {{
                background: {SemiconductorTheme.BG_PANEL};
                color: #ff6b9d;
                font-weight: bold; font-size: 8pt; padding: 3px;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """)
        self.sigma_table.horizontalHeader().setStretchLastSection(True)
        self.sigma_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sigma_table.verticalHeader().setVisible(False)
        sigma_splitter.addWidget(self.sigma_table)

        sigma_chart_container = QWidget()
        sigma_chart_layout = QVBoxLayout(sigma_chart_container)
        sigma_chart_layout.setContentsMargins(4, 4, 4, 4)
        sigma_chart_layout.setSpacing(2)
        sigma_chart_label = QLabel("<b>σ ACROSS LAYERS</b>  (lower = better repeatability)")
        sigma_chart_label.setStyleSheet("color: #ff6b9d; font-size: 9pt; padding: 2px;")
        sigma_chart_layout.addWidget(sigma_chart_label)
        self.sigma_chart = EmbeddedChart(parent=sigma_chart_container, width=5, height=3)
        sigma_chart_layout.addWidget(self.sigma_chart, 1)
        sigma_splitter.addWidget(sigma_chart_container)

        sigma_splitter.setSizes([250, 450])
        sigma_layout.addWidget(sigma_splitter, 1)
        self._right_vsplit.addWidget(sigma_page)

        # ── PAGE 2: Per-Bump Sigma Analysis ──
        bump_sigma_page = self._create_per_bump_sigma_section()
        self._right_vsplit.addWidget(bump_sigma_page)

        # ── PAGE 3: 3D Structural Integrity Map ──
        sim_page = self._create_3d_sim_section()
        self._right_vsplit.addWidget(sim_page)

        # ── PAGE 4: LOO Variance Decomposition ──
        loo_page = self._create_loo_vd_section()
        self._right_vsplit.addWidget(loo_page)

        # Default to page 0
        self._right_vsplit.setCurrentIndex(0)
        self._current_view_mode = 0

        layout.addWidget(self._right_vsplit, 1)
        return panel



    def _create_per_bump_sigma_section(self):
        """Build the Per-Bump Sigma Analysis UI section."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        combo_ss = f"""
            QComboBox {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 3px; padding: 4px 8px; font-size: 9pt;
            }}
        """

        # ── Header row with controls ──
        header_row = QHBoxLayout()
        title = QLabel(
            "<b>PER-BUMP SIGMA ANALYSIS</b>  "
            "<span style='color: #808080;'>Same bump index across samples</span>")
        title.setStyleSheet("font-size: 10pt; color: #76ff03; padding: 4px 8px;")
        header_row.addWidget(title)
        header_row.addStretch()

        lbl_ss = f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-weight: bold;"

        # Layer filter
        layer_lbl = QLabel("Layer:")
        layer_lbl.setStyleSheet(lbl_ss)
        header_row.addWidget(layer_lbl)

        self.bump_sigma_layer_combo = QComboBox()
        self.bump_sigma_layer_combo.addItem("— Select Layer —")
        self.bump_sigma_layer_combo.setFixedWidth(160)
        self.bump_sigma_layer_combo.setStyleSheet(combo_ss)
        self.bump_sigma_layer_combo.currentIndexChanged.connect(self._on_bump_layer_changed)
        header_row.addWidget(self.bump_sigma_layer_combo)

        # Bump Index filter
        bump_lbl = QLabel("Bump:")
        bump_lbl.setStyleSheet(lbl_ss)
        header_row.addWidget(bump_lbl)

        self.bump_sigma_index_combo = QComboBox()
        self.bump_sigma_index_combo.addItem("All Bumps")
        self.bump_sigma_index_combo.setFixedWidth(120)
        self.bump_sigma_index_combo.setStyleSheet(combo_ss)
        self.bump_sigma_index_combo.currentIndexChanged.connect(self._refresh_per_bump_sigma)
        header_row.addWidget(self.bump_sigma_index_combo)

        # Metric selector
        metric_lbl = QLabel("Metric:")
        metric_lbl.setStyleSheet(lbl_ss)
        header_row.addWidget(metric_lbl)

        self.bump_sigma_metric_combo = QComboBox()
        self.bump_sigma_metric_combo.addItems([
            "SOH / Z Height", "Void Ratio", "Bump Volume"
        ])
        self.bump_sigma_metric_combo.setFixedWidth(150)
        self.bump_sigma_metric_combo.setStyleSheet(combo_ss)
        self.bump_sigma_metric_combo.currentIndexChanged.connect(self._refresh_per_bump_sigma)
        header_row.addWidget(self.bump_sigma_metric_combo)

        layout.addLayout(header_row)

        # ── Sigma summary label ──
        self.bump_sigma_summary_label = QLabel("Select a Layer and Bump to view per-sample detail")
        self.bump_sigma_summary_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt; "
            f"padding: 2px 8px; font-style: italic;")
        layout.addWidget(self.bump_sigma_summary_label)

        # ── Content: table (top) + chart (bottom) ──
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {SemiconductorTheme.BORDER_DEFAULT}; }}")

        table_ss = f"""
            QTableWidget {{
                background: {SemiconductorTheme.BG_DARK};
                alternate-background-color: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                gridline-color: {SemiconductorTheme.BORDER_DEFAULT};
                font-size: 9pt;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QTableWidget::item:selected {{
                background-color: rgba(118, 255, 3, 0.15);
                color: white;
            }}
            QHeaderView::section {{
                background: {SemiconductorTheme.BG_PANEL};
                color: #76ff03;
                font-weight: bold;
                font-size: 8pt;
                padding: 3px;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """

        # Per-bump sigma table (dynamic columns based on mode)
        self.bump_sigma_table = QTableWidget()
        self.bump_sigma_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.bump_sigma_table.setAlternatingRowColors(True)
        self.bump_sigma_table.setSortingEnabled(True)
        self.bump_sigma_table.setStyleSheet(table_ss)
        self.bump_sigma_table.horizontalHeader().setStretchLastSection(True)
        self.bump_sigma_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.bump_sigma_table.verticalHeader().setVisible(False)
        # Set initial columns (overview mode)
        self._set_bump_table_overview_mode()
        splitter.addWidget(self.bump_sigma_table)

        # Per-bump sigma chart
        chart_container = QWidget()
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(4, 4, 4, 4)
        chart_layout.setSpacing(2)
        self.bump_sigma_chart_label = QLabel("<b>σ PER BUMP INDEX</b>  (sorted by worst σ)")
        self.bump_sigma_chart_label.setStyleSheet("color: #76ff03; font-size: 9pt; padding: 2px;")
        chart_layout.addWidget(self.bump_sigma_chart_label)
        self.bump_sigma_chart = EmbeddedChart(parent=chart_container, width=5, height=3)
        chart_layout.addWidget(self.bump_sigma_chart, 1)
        splitter.addWidget(chart_container)

        splitter.setSizes([250, 450])
        layout.addWidget(splitter, 1)

        return container

    def _set_bump_table_overview_mode(self):
        """Configure table columns for overview (all bumps) mode."""
        self.bump_sigma_table.setColumnCount(10)
        self.bump_sigma_table.setHorizontalHeaderLabels([
            "#", "Layer", "Grid(R,C)", "N Samples", "Mean", "StdDev.S",
            "%Tol", "Min", "Max", "Range"
        ])

    def _set_bump_table_detail_mode(self):
        """Configure table columns for detail (single bump, per-sample) mode."""
        self.bump_sigma_table.setColumnCount(4)
        self.bump_sigma_table.setHorizontalHeaderLabels([
            "#", "Sample", "Value", "Δ from Mean"
        ])

    # ────────── Database scope ──────────

    def refresh_from_db(self):
        """Reload lot/wafer filters and rebuild scope tree from inspection.db."""
        try:
            self._db = get_db()
            self._db_lots = self._db.list_lots()
        except Exception as e:
            self.status_label.setText(f"DB error: {e}")
            return

        self.lot_combo.blockSignals(True)
        self.lot_combo.clear()
        if not self._db_lots:
            self.lot_combo.addItem("(no lots in DB)")
            self.lot_combo.blockSignals(False)
            self.wafer_combo.clear()
            self.scope_tree.clear()
            self.status_label.setText(
                "Database empty — run Online inspections or Batch Review → Seed demo")
            self._update_scope_kpis()
            return

        for lot in self._db_lots:
            date_f = lot.get("date_folder") or ""
            lf = lot.get("lot_foup_id") or ""
            n_w = lot.get("n_wafers") or 0
            n_f = lot.get("n_fov") or 0
            label = f"{date_f} · {lf}  ({n_w}W / {n_f} FOV)"
            self.lot_combo.addItem(label, (date_f, lf))
        self.lot_combo.blockSignals(False)

        self._on_lot_filter_changed()
        # Auto-load first wafer once
        if not self._auto_loaded and self._scope_wafer_key:
            self._auto_loaded = True
            self.load_selected_wafer()
        else:
            self.status_label.setText(
                f"DB ready · {len(self._db_lots)} lot(s) · select wafer and Load")

    def _on_lot_filter_changed(self):
        data = self.lot_combo.currentData()
        self.wafer_combo.blockSignals(True)
        self.wafer_combo.clear()
        if not data:
            self.wafer_combo.blockSignals(False)
            return
        date_f, lf = data
        self._scope_lot = f"{date_f}|{lf}"
        try:
            self._db_wafers = self._db.list_wafers(date_folder=date_f, lot_foup_id=lf)
        except Exception:
            self._db_wafers = []
        for w in self._db_wafers:
            wid = w.get("wafer_id") or "?"
            yld = w.get("yield_pct") or 0.0
            n_f = w.get("n_fov") or 0
            label = f"{wid}  ·  {n_f} FOV  ·  {yld:.0f}%"
            self.wafer_combo.addItem(label, w.get("wafer_key") or "")
        self.wafer_combo.blockSignals(False)
        self._on_wafer_filter_changed()

    def _on_wafer_filter_changed(self):
        key = self.wafer_combo.currentData() or ""
        self._scope_wafer_key = str(key)
        self._rebuild_scope_tree()
        self._update_scope_kpis()

    def _update_scope_kpis(self):
        ng_thr = self.ng_threshold_spin.value() / 100.0
        try:
            if self._scope_wafer_key:
                k = self._db.soh_kpis(wafer_key=self._scope_wafer_key, ng_threshold=ng_thr)
            else:
                k = {"n_fov": 0, "n_objects": 0, "yield_pct": 0.0, "mean_soh": 0.0}
        except Exception:
            k = {"n_fov": 0, "n_objects": 0, "yield_pct": 0.0, "mean_soh": 0.0}
        self.kpi_labels["fov"].setText(str(int(k.get("n_fov") or 0)))
        nob = int(k.get("n_objects") or 0)
        self.kpi_labels["obj"].setText(f"{nob/1000:.1f}k" if nob >= 1000 else str(nob))
        self.kpi_labels["yield"].setText(f"{float(k.get('yield_pct') or 0):.0f}%")
        self.kpi_labels["soh"].setText(f"{float(k.get('mean_soh') or 0):.2f}")

    def _rebuild_scope_tree(self):
        """Build Chip → FOV tree for selected wafer (with checkboxes)."""
        self.scope_tree.blockSignals(True)
        self.scope_tree.clear()
        if not self._scope_wafer_key:
            self.scope_tree.blockSignals(False)
            return
        try:
            runs = self._db.list_fov_runs(wafer_key=self._scope_wafer_key, limit=5000)
            chips = self._db.list_chips_for_wafer(self._scope_wafer_key)
        except Exception as e:
            self.status_label.setText(f"Scope tree error: {e}")
            self.scope_tree.blockSignals(False)
            return

        # Group runs by chip
        by_chip: Dict[tuple, List[dict]] = {}
        for run in runs:
            key = (int(run.get("chip_col") or 0), int(run.get("chip_row") or 0))
            by_chip.setdefault(key, []).append(run)

        chip_judg = {
            (int(c.get("chip_col") or 0), int(c.get("chip_row") or 0)): c
            for c in chips
        }

        for (cc, cr) in sorted(by_chip.keys(), key=lambda t: (t[1], t[0])):
            cinfo = chip_judg.get((cc, cr), {})
            jud = str(cinfo.get("judgment") or "")
            n_obj = int(cinfo.get("n_objects") or 0)
            chip_item = QTreeWidgetItem([
                f"Chip {cc},{cr}",
                str(n_obj) if n_obj else "",
                jud or "—",
            ])
            chip_item.setData(0, Qt.UserRole, ("chip", cc, cr))
            chip_item.setFlags(chip_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsTristate)
            chip_item.setCheckState(0, Qt.Unchecked)
            chip_item.setFont(0, QFont("Segoe UI", 8, QFont.Bold))
            if jud == "NG":
                chip_item.setForeground(2, QColor("#f87171"))
            elif jud == "OK":
                chip_item.setForeground(2, QColor("#34d399"))

            # Dedupe FOV by latest finished_at (list is DESC)
            seen_fov = set()
            for run in by_chip[(cc, cr)]:
                fi = int(run.get("fov_index") or 0)
                if fi < 1 or fi in seen_fov:
                    continue
                seen_fov.add(fi)
                rid = str(run.get("run_id") or "")
                rjud = str(run.get("judgment") or "—")
                n_o = int(run.get("n_objects") or 0)
                fov_item = QTreeWidgetItem([
                    f"FOV P{fi}",
                    str(n_o),
                    rjud,
                ])
                fov_item.setData(0, Qt.UserRole, ("fov", rid, fi, cc, cr))
                fov_item.setFlags(fov_item.flags() | Qt.ItemIsUserCheckable)
                fov_item.setCheckState(0, Qt.Unchecked)
                if rjud.upper() == "NG":
                    fov_item.setForeground(2, QColor("#f87171"))
                elif rjud.upper() == "OK":
                    fov_item.setForeground(2, QColor("#34d399"))
                # Mark if already in analysis set
                if rid in self._scope_run_ids or any(s.run_id == rid for s in self.samples):
                    fov_item.setCheckState(0, Qt.Checked)
                    fov_item.setForeground(0, QColor(SemiconductorTheme.ACCENT_PRIMARY))
                chip_item.addChild(fov_item)

            self.scope_tree.addTopLevelItem(chip_item)
            chip_item.setExpanded(True)

        self.scope_tree.blockSignals(False)

    def _on_scope_item_changed(self, item: QTreeWidgetItem, column: int):
        # Parent tristate handles cascade; nothing required
        pass

    def _on_scope_selection_changed(self):
        items = self.scope_tree.selectedItems()
        if not items:
            return
        data = items[0].data(0, Qt.UserRole)
        if not data:
            return
        if data[0] == "fov":
            rid = data[1]
            # Focus matching loaded sample if present
            for i, s in enumerate(self.samples):
                if s.run_id == rid:
                    self._selected_sample_idx = i
                    self._update_detail_sample(s)
                    self._refresh_charts()
                    break

    def _checked_run_ids(self) -> List[str]:
        ids: List[str] = []
        root = self.scope_tree.invisibleRootItem()
        for i in range(root.childCount()):
            chip = root.child(i)
            for j in range(chip.childCount()):
                fov = chip.child(j)
                if fov.checkState(0) == Qt.Checked:
                    data = fov.data(0, Qt.UserRole)
                    if data and data[0] == "fov" and data[1]:
                        ids.append(str(data[1]))
        return ids

    def load_selected_wafer(self):
        """Load all FOV runs for the selected wafer into the analysis set."""
        if not self._scope_wafer_key:
            QMessageBox.information(self, "Load", "Select a wafer first.")
            return
        self.status_label.setText("Loading MES objects from DB…")
        QApplication.processEvents()
        ng_thr = self.ng_threshold_spin.value() / 100.0
        try:
            samples = load_samples_from_db(
                self._db,
                wafer_key=self._scope_wafer_key,
                ng_threshold=ng_thr,
                compute_layer_stats_fn=compute_layer_stats,
            )
        except Exception as e:
            QMessageBox.critical(self, "DB Load Error", str(e))
            self.status_label.setText(f"Load failed: {e}")
            return
        self._replace_samples(samples)
        self._scope_run_ids = {s.run_id for s in samples if s.run_id}
        self._rebuild_scope_tree()
        n_obj = sum(len(s.raw_rows) for s in samples)
        self.status_label.setText(
            f"Loaded wafer · {len(samples)} FOV · {n_obj} objects · "
            f"{sum(len(s.layers) for s in samples)} layer-slots")

    def load_checked_fovs(self):
        """Load only checked FOV runs."""
        ids = self._checked_run_ids()
        if not ids:
            QMessageBox.information(
                self, "Load", "Check one or more FOV items in the wafer tree first.")
            return
        self.status_label.setText(f"Loading {len(ids)} FOV run(s)…")
        QApplication.processEvents()
        ng_thr = self.ng_threshold_spin.value() / 100.0
        try:
            samples = load_samples_from_db(
                self._db,
                run_ids=ids,
                ng_threshold=ng_thr,
                compute_layer_stats_fn=compute_layer_stats,
            )
        except Exception as e:
            QMessageBox.critical(self, "DB Load Error", str(e))
            return
        self._replace_samples(samples)
        self._scope_run_ids = set(ids)
        n_obj = sum(len(s.raw_rows) for s in samples)
        self.status_label.setText(
            f"Loaded set · {len(samples)} FOV · {n_obj} objects")

    def _replace_samples(self, samples: List[SampleData]):
        self.samples = list(samples)
        self._selected_sample_idx = 0 if samples else None
        self._rebuild_tree()
        self._refresh_summary_table()
        self._refresh_overview_kpis()
        self._refresh_charts()
        if samples:
            self._update_detail_sample(samples[0])
        else:
            self._clear_detail()

    # ────────── Import Logic (CSV fallback) ──────────

    def import_sample(self):
        """Import a single sample — browse for a folder containing measurement CSV(s)."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Sample Output Folder",
            "", QFileDialog.ShowDirsOnly)
        if not folder:
            return

        sample = self._load_sample_from_folder(folder)
        if sample:
            self.samples.append(sample)
            self._rebuild_tree()
            self._refresh_summary_table()
            self._refresh_overview_kpis()
            self._refresh_charts()
            self.status_label.setText(
                f"CSV imported: {sample.name} ({len(sample.layers)} layers, {len(sample.raw_rows)} objects)")

    def import_batch(self):
        """Import multiple samples — browse for a parent folder where each subfolder is a sample."""
        parent = QFileDialog.getExistingDirectory(
            self, "Select Parent Folder (each subfolder = 1 sample)",
            "", QFileDialog.ShowDirsOnly)
        if not parent:
            return

        count = 0
        for entry in sorted(os.listdir(parent)):
            sub = os.path.join(parent, entry)
            if os.path.isdir(sub):
                sample = self._load_sample_from_folder(sub)
                if sample:
                    self.samples.append(sample)
                    count += 1

        if count > 0:
            self._rebuild_tree()
            self._refresh_summary_table()
            self._refresh_overview_kpis()
            self._refresh_charts()
            self.status_label.setText(f"CSV batch imported: {count} samples from {parent}")
        else:
            QMessageBox.information(self, "No Data",
                                   "No measurement CSV files found in subfolders.")

    def _load_sample_from_folder(self, folder: str) -> Optional[SampleData]:
        """Load all measurement CSVs from a folder into a SampleData object.
        
        Searches for:
        1. Per-layer CSVs: <folder>/<LayerName>/measurement_<LayerName>.csv
        2. Combined CSV: <folder>/object_statistics.csv
        """
        sample_name = os.path.basename(folder)
        # If the user selected the /out or /measurement_results folder directly, use the parent folder's name instead
        parent_name = os.path.basename(os.path.dirname(folder))
        if sample_name.lower() in ["out", "measurement_results", "measurements"]:
            sample_name = parent_name
        elif parent_name.lower() == "out" and os.path.basename(os.path.dirname(os.path.dirname(folder))).startswith("PKg"):
            # They selected 'measurement_results' which is inside 'out'
            sample_name = os.path.basename(os.path.dirname(os.path.dirname(folder)))
        # Check for duplicate
        for existing in self.samples:
            if os.path.normpath(existing.path) == os.path.normpath(folder):
                QMessageBox.warning(self, "Duplicate", f"Sample already loaded:\n{folder}")
                return None

        all_rows = []
        ng_threshold = self.ng_threshold_spin.value() / 100.0

        # Strategy 1: Look for new measurement_results structure
        mr_path = os.path.join(folder, "out", "measurement_results")
        if not os.path.isdir(mr_path):
            mr_path = os.path.join(folder, "measurement_results")
            
        if os.path.isdir(mr_path):
            csv_files = glob.glob(os.path.join(mr_path, "*.csv"))
            # Filter out summary CSV to avoid duplicating data
            per_layer_csvs = [f for f in csv_files if "summary" not in os.path.basename(f).lower()]
            if not per_layer_csvs:
                per_layer_csvs = csv_files # Fallback
                
            for csv_path in sorted(per_layer_csvs):
                rows = parse_measurement_csv(csv_path)
                fname = os.path.basename(csv_path)
                layer_name = fname.replace(".csv", "").replace("_", " ")
                if rows and 'Layer' in rows[0]:
                    csv_layer = rows[0]['Layer'].strip()
                    if csv_layer and csv_layer != 'N/A':
                        layer_name = csv_layer
                for r in rows:
                    r['_layer_resolved'] = layer_name
                all_rows.extend(rows)
        else:
            # Strategy 2: Legacy subfolder/measurement_*.csv
            per_layer_csvs = glob.glob(os.path.join(folder, "*", "measurement_*.csv"))
            if per_layer_csvs:
                for csv_path in sorted(per_layer_csvs):
                    rows = parse_measurement_csv(csv_path)
                    fname = os.path.basename(csv_path)
                    layer_name = fname.replace("measurement_", "").replace(".csv", "").replace("_", " ")
                    if rows and 'Layer' in rows[0]:
                        csv_layer = rows[0]['Layer'].strip()
                        if csv_layer and csv_layer != 'N/A':
                            layer_name = csv_layer
                    for r in rows:
                        r['_layer_resolved'] = layer_name
                    all_rows.extend(rows)
            else:
                # Strategy 3: Legacy combined CSV
                combined_csv = os.path.join(folder, "object_statistics.csv")
                if os.path.exists(combined_csv):
                    rows = parse_measurement_csv(combined_csv)
                    for r in rows:
                        layer_key = r.get('Layer', '').strip()
                        r['_layer_resolved'] = layer_key if layer_key and layer_key != 'N/A' else 'Default'
                    all_rows.extend(rows)

        if not all_rows:
            return None

        # Group rows by layer
        layer_groups: Dict[str, List[dict]] = {}
        for row in all_rows:
            ln = row.get('_layer_resolved', 'Default')
            layer_groups.setdefault(ln, []).append(row)

        # Compute stats per layer
        layers = {}
        for ln, rows in layer_groups.items():
            layers[ln] = compute_layer_stats(rows, ln, ng_threshold)

        sample = SampleData(
            name=sample_name, path=folder, layers=layers, raw_rows=all_rows, source="csv")
        return sample

    def remove_sample(self):
        """Remove selected units from the analysis set."""
        selected_items = self.sample_tree.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self, "No Selection", "Select at least one FOV/unit in the analysis set.")
            return

        indices_to_remove = set()
        for item in selected_items:
            data = item.data(0, Qt.UserRole)
            if data and data[0] in ('sample', 'path', 'layer'):
                indices_to_remove.add(data[1])

        if not indices_to_remove:
            return

        names = [self.samples[idx].name for idx in sorted(indices_to_remove)]

        reply = QMessageBox.question(
            self, "Confirm Remove",
            f"Remove {len(names)} unit(s) from the analysis set?\n\n"
            + "\n".join(names[:5])
            + ("\n..." if len(names) > 5 else ""),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            for idx in sorted(indices_to_remove, reverse=True):
                rid = getattr(self.samples[idx], "run_id", "")
                if rid:
                    self._scope_run_ids.discard(rid)
                self.samples.pop(idx)

            self._selected_sample_idx = 0 if self.samples else None
            self._rebuild_tree()
            self._refresh_summary_table()
            self._refresh_overview_kpis()
            self._refresh_charts()
            if self.samples:
                self._update_detail_sample(self.samples[0])
            else:
                self._clear_detail()
            self.status_label.setText(f"Removed {len(names)} unit(s).")

    def remove_all_samples(self):
        """Clear the analysis set."""
        if not self.samples:
            return

        reply = QMessageBox.question(
            self, "Confirm Clear",
            f"Clear ALL {len(self.samples)} unit(s) from the analysis set?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.samples.clear()
            self._scope_run_ids.clear()
            self._selected_sample_idx = None
            self._rebuild_tree()
            self._refresh_summary_table()
            self._refresh_overview_kpis()
            self._refresh_charts()
            self._clear_detail()
            self.status_label.setText("Analysis set cleared.")

    # ────────── Tree ──────────

    def _rebuild_tree(self):
        """Rebuild the analysis-set tree (loaded FOV / CSV units)."""
        self.sample_tree.blockSignals(True)
        self.sample_tree.clear()

        for s_idx, sample in enumerate(self.samples):
            total_obj = sum(ls.object_count for ls in sample.layers.values())
            total_ng = sum(ls.ng_count for ls in sample.layers.values())
            ng_pct = f"{total_ng / total_obj * 100:.1f}%" if total_obj > 0 else "—"

            src = "DB" if getattr(sample, "source", "") == "db" else "CSV"
            if getattr(sample, "fov_index", 0):
                title = (
                    f"P{sample.fov_index} · C{sample.chip_col},{sample.chip_row}"
                    f" · {sample.wafer_id or sample.name}"
                )
            else:
                title = sample.name

            item = QTreeWidgetItem([
                f"[{src}] {title}", str(total_obj), str(total_ng), ng_pct
            ])
            item.setData(0, Qt.UserRole, ('sample', s_idx))
            tip = sample.path or sample.run_id or sample.name
            if sample.judgment:
                tip = f"{tip}\njudgment={sample.judgment}"
            item.setToolTip(0, tip)
            item.setForeground(0, QColor(SemiconductorTheme.ACCENT_PRIMARY))
            item.setFont(0, QFont("Segoe UI", 8, QFont.Bold))

            for ln in sorted(sample.layers.keys(), key=_natural_sort_key):
                ls = sample.layers[ln]
                ng_pct_layer = f"{ls.ng_rate * 100:.1f}%"
                child = QTreeWidgetItem([
                    f"  {ln}", str(ls.object_count), str(ls.ng_count), ng_pct_layer
                ])
                child.setData(0, Qt.UserRole, ('layer', s_idx, ln))
                if ls.ng_rate >= 0.1:
                    child.setForeground(3, QColor('#ff4444'))
                elif ls.ng_rate >= 0.05:
                    child.setForeground(3, QColor('#ff8800'))
                else:
                    child.setForeground(3, QColor('#76ff03'))
                item.addChild(child)

            self.sample_tree.addTopLevelItem(item)
            # Keep collapsed when many FOVs
            item.setExpanded(len(self.samples) <= 6)

        self.sample_tree.blockSignals(False)
    def _on_tree_header_clicked(self, logicalIndex):
        """Sort samples when clicking the tree header."""
        if logicalIndex == 0:
            if not hasattr(self, '_sort_asc'):
                self._sort_asc = True
            
            # Sort self.samples by name
            self.samples.sort(key=lambda s: _natural_sort_key(s.name), reverse=not self._sort_asc)
            self._sort_asc = not self._sort_asc
            
            self._rebuild_tree()
            self._refresh_summary_table()
            self._refresh_charts()
            self.status_label.setText("Samples sorted by Name.")

    def _on_tree_selection_changed(self):
        """Handle tree item selection — update detail info and intra chart."""
        items = self.sample_tree.selectedItems()
        if not items:
            return

        data = items[0].data(0, Qt.UserRole)
        if not data:
            return

        if data[0] == 'sample' or data[0] == 'path':
            s_idx = data[1]
            self._selected_sample_idx = s_idx
            sample = self.samples[s_idx]
            self._update_detail_sample(sample)
            self._refresh_charts()

        elif data[0] == 'layer':
            s_idx = data[1]
            layer_name = data[2]
            self._selected_sample_idx = s_idx
            sample = self.samples[s_idx]
            layer_stats = sample.layers.get(layer_name)
            if layer_stats:
                self._update_detail_layer(sample, layer_stats)
            self._refresh_charts()

    def _update_detail_sample(self, sample: SampleData):
        """Show sample-level summary in detail panel."""
        self.detail_labels['sample'].setText(sample.name)
        self.detail_labels['path'].setText(sample.path)
        self.detail_labels['layer'].setText(f"{len(sample.layers)} layer(s)")

        total_obj = sum(ls.object_count for ls in sample.layers.values())
        total_ng = sum(ls.ng_count for ls in sample.layers.values())
        self.detail_labels['objects'].setText(str(total_obj))
        self.detail_labels['ng_count'].setText(str(total_ng))
        self.detail_labels['ng_rate'].setText(
            f"{total_ng / total_obj * 100:.2f}%" if total_obj > 0 else "—")

        all_zh = [ls.mean_z_height for ls in sample.layers.values() if ls.object_count > 0]
        if all_zh:
            self.detail_labels['mean_zh'].setText(f"{np.mean(all_zh):.2f} µm")
            self.detail_labels['std_zh'].setText(f"{np.std(all_zh):.2f} µm")
            self.detail_labels['min_zh'].setText(
                f"{min(ls.min_z_height for ls in sample.layers.values()):.2f} µm")
            self.detail_labels['max_zh'].setText(
                f"{max(ls.max_z_height for ls in sample.layers.values()):.2f} µm")
        else:
            for k in ('mean_zh', 'std_zh', 'min_zh', 'max_zh'):
                self.detail_labels[k].setText("—")

    def _update_detail_layer(self, sample: SampleData, ls: LayerStats):
        """Show layer-level detail in detail panel."""
        self.detail_labels['sample'].setText(sample.name)
        self.detail_labels['path'].setText(sample.path)
        self.detail_labels['layer'].setText(ls.layer_name)
        self.detail_labels['objects'].setText(str(ls.object_count))
        self.detail_labels['ng_count'].setText(str(ls.ng_count))
        self.detail_labels['ng_rate'].setText(f"{ls.ng_rate * 100:.2f}%")
        self.detail_labels['mean_zh'].setText(f"{ls.mean_z_height:.2f} µm")
        self.detail_labels['std_zh'].setText(f"{ls.std_z_height:.2f} µm")
        self.detail_labels['min_zh'].setText(f"{ls.min_z_height:.2f} µm")
        self.detail_labels['max_zh'].setText(f"{ls.max_z_height:.2f} µm")

    def _clear_detail(self):
        for val in self.detail_labels.values():
            val.setText("—")

    # ────────── Summary Table ──────────

    def _refresh_summary_table(self):
        """Populate summary: scope-level by layer when many FOVs; else unit×layer."""
        self.summary_table.setRowCount(0)
        ng_threshold = self.ng_threshold_spin.value() / 100.0
        if not self.samples:
            return

        # Prefer compact scope aggregate when analysis set is large (DB wafer load)
        use_scope = len(self.samples) > 3 or any(
            getattr(s, "source", "") == "db" for s in self.samples)

        rows_src = []  # list of (unit_label, LayerStats)
        if use_scope:
            scope_layers = aggregate_layers_from_samples(
                self.samples, ng_threshold, compute_layer_stats)
            unit = f"SET ({len(self.samples)} FOV)"
            for ln in sorted(scope_layers.keys(), key=_natural_sort_key):
                rows_src.append((unit, scope_layers[ln]))
        else:
            for sample in self.samples:
                for ln in sorted(sample.layers.keys(), key=_natural_sort_key):
                    rows_src.append((sample.name, sample.layers[ln]))

        for row_idx, (unit_label, ls) in enumerate(rows_src):
            self.summary_table.insertRow(row_idx)
            items_data = [
                str(row_idx + 1),
                unit_label,
                ls.layer_name,
                str(ls.object_count),
                str(ls.ng_count),
                f"{ls.ng_rate * 100:.1f}%",
                f"{ls.mean_z_height:.2f}",
                f"{ls.std_z_height:.2f}",
                f"{ls.min_z_height:.2f}",
                f"{ls.max_z_height:.2f}",
                f"{ls.mean_void_ratio * 100:.3f}%",
                f"{ls.max_void_ratio * 100:.3f}%",
            ]
            if ls.ng_rate >= 0.1:
                bg = QColor(180, 40, 40, 50)
            elif ls.ng_rate >= 0.05:
                bg = QColor(180, 120, 0, 40)
            else:
                bg = QColor(40, 180, 60, 30)

            for col, text in enumerate(items_data):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setBackground(bg)
                if col == 5:
                    item.setFont(QFont("Arial", 9, QFont.Bold))
                    if ls.ng_rate >= 0.1:
                        item.setForeground(QColor(255, 80, 80))
                    elif ls.ng_rate >= 0.05:
                        item.setForeground(QColor(255, 160, 0))
                    else:
                        item.setForeground(QColor(100, 255, 120))
                self.summary_table.setItem(row_idx, col, item)

    # ────────── Charts ──────────

    def _get_selected_metric(self) -> str:
        """Map combo text to attribute name."""
        mapping = {
            "Mean Z Height": "mean_z_height",
            "NG Rate": "ng_rate",
            "Mean Void Ratio": "mean_void_ratio",
            "Max Void Ratio": "max_void_ratio",
            "Total Void Volume": "total_void_volume",
            "Mean Bump Volume": "mean_bump_volume",
        }
        return mapping.get(self.metric_combo.currentText(), "mean_z_height")

    def _refresh_overview_kpis(self):
        """Update Overview KPI cards from current analysis set."""
        if not hasattr(self, "overview_kpis"):
            return
        if not self.samples:
            for v in self.overview_kpis.values():
                v.setText("—")
            return
        ng_thr = self.ng_threshold_spin.value() / 100.0
        scope_layers = aggregate_layers_from_samples(
            self.samples, ng_thr, compute_layer_stats)
        all_soh = []
        total_n = total_ng = 0
        void_acc = 0.0
        for ls in scope_layers.values():
            total_n += ls.object_count
            total_ng += ls.ng_count
            void_acc += ls.mean_void_ratio * ls.object_count
            if ls.object_count:
                all_soh.extend([ls.mean_z_height] * max(1, ls.object_count // 10))
        # better mean SOH from raw
        soh_vals = []
        for s in self.samples:
            for row in s.raw_rows:
                try:
                    soh_vals.append(float(row.get("SOH (um)") or row.get("SOH(µm)") or 0))
                except (TypeError, ValueError):
                    pass
        mean_soh = float(np.mean(soh_vals)) if soh_vals else 0.0
        std_soh = float(np.std(soh_vals, ddof=1)) if len(soh_vals) > 1 else 0.0
        ng_rate = (total_ng / total_n) if total_n else 0.0
        mean_void = (void_acc / total_n) if total_n else 0.0

        self.overview_kpis["layers"].setText(str(len(scope_layers)))
        self.overview_kpis["mean_soh"].setText(f"{mean_soh:.2f} µm")
        self.overview_kpis["sigma"].setText(f"{std_soh:.2f} µm")
        self.overview_kpis["ng"].setText(f"{ng_rate * 100:.1f}%")
        self.overview_kpis["void"].setText(f"{mean_void * 100:.2f}%")

    def _refresh_charts(self):
        """Redraw Overview charts (layer aggregate + FOV map) + advanced pages."""
        metric = self._get_selected_metric()
        ng_thr = self.ng_threshold_spin.value() / 100.0

        if self.samples:
            # Prefer scope-level layer aggregate; if a single unit selected, still show set
            scope_layers = aggregate_layers_from_samples(
                self.samples, ng_thr, compute_layer_stats)
            # If user selected one FOV, overlay that FOV's layers when metric is local
            if (
                self._selected_sample_idx is not None
                and 0 <= self._selected_sample_idx < len(self.samples)
                and len(self.samples) > 1
            ):
                sel = self.samples[self._selected_sample_idx]
                self.intra_label.setText(
                    f"<b>SOH BY LAYER</b>  ·  set aggregate  "
                    f"<span style='color:#808080'>(focus: {sel.name})</span>")
            else:
                self.intra_label.setText(
                    "<b>SOH BY LAYER</b>  ·  analysis set aggregate")
            self.intra_chart.plot_scope_layers(
                scope_layers, metric,
                title=f"SOH by Layer — {self.metric_combo.currentText()}")

            # FOV heatmap when DB samples present
            has_fov = any(int(getattr(s, "fov_index", 0) or 0) > 0 for s in self.samples)
            if has_fov:
                fov_stats = aggregate_fov_from_samples(self.samples)
                # Use ng threshold for ng_rate recompute lightly
                if metric == "ng_rate":
                    # recompute ng with current threshold
                    from collections import defaultdict
                    buckets = defaultdict(list)
                    for s in self.samples:
                        fi = int(s.fov_index or 0)
                        if fi <= 0:
                            continue
                        for row in s.raw_rows:
                            val = _safe_float(row.get("Ratio") or row.get(
                                "Ratio (Void/(TGV+Void))", "0"))
                            if val > 1.0:
                                val /= 100.0
                            buckets[fi].append(val)
                    fov_stats = []
                    for fi, ratios in sorted(buckets.items()):
                        n = len(ratios)
                        n_ng = sum(1 for r in ratios if r >= ng_thr)
                        fov_stats.append({
                            "key": fi, "n": n,
                            "ng_rate": (n_ng / n) if n else 0.0,
                            "mean_soh": 0.0,
                        })
                    self.inter_chart.plot_fov_heatmap(
                        fov_stats, metric="ng_rate",
                        title="NG Rate by FOV Position (P1–P9)")
                else:
                    self.inter_chart.plot_fov_heatmap(
                        fov_stats, metric="mean_soh",
                        title="Mean SOH by FOV Position (P1–P9)")
                self.inter_label.setText("<b>SOH BY FOV</b>  ·  P1–P9 position map")
            else:
                # CSV fallback: cross-sample layer comparison
                self.inter_label.setText(
                    "<b>LAYERS ACROSS UNITS</b>  ·  cross-sample")
                self.inter_chart.plot_inter_sample(self.samples, metric)
        else:
            self.intra_chart.clear()
            self.inter_chart.clear()

        self._refresh_sigma()
        self._refresh_per_bump_sigma()

    # ────────── Recalculate ──────────

    def _recalculate_all(self):
        """Recompute all SOH metrics using current NG threshold."""
        ng_threshold = self.ng_threshold_spin.value() / 100.0

        for sample in self.samples:
            layer_groups: Dict[str, List[dict]] = {}
            for row in sample.raw_rows:
                ln = row.get('_layer_resolved', 'Default')
                layer_groups.setdefault(ln, []).append(row)

            for ln, rows in layer_groups.items():
                sample.layers[ln] = compute_layer_stats(rows, ln, ng_threshold)

        self._rebuild_tree()
        self._refresh_summary_table()
        self._refresh_overview_kpis()
        self._update_scope_kpis()
        self._refresh_charts()

        if self._selected_sample_idx is not None and self._selected_sample_idx < len(self.samples):
            self._update_detail_sample(self.samples[self._selected_sample_idx])

    # ────────── Export ──────────

    def export_report(self):
        """Export the summary table as CSV."""
        if not self.samples:
            QMessageBox.information(
                self, "Export", "No data to export. Load a wafer from DB or import CSV first.")
            return

        default_name = "soh_analysis_report.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export SOH Report", default_name,
            "CSV Files (*.csv);;All Files (*.*)")
        if not path:
            return

        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Sample', 'Path', 'Layer', 'Objects', 'NG Count', 'NG Rate (%)',
                    'Mean Z Height (um)', 'Std Z Height', 'Min Z Height', 'Max Z Height',
                    'Mean Bump Vol (um3)', 'Total Bump Vol (um3)',
                    'Total Void Vol (um3)', 'Mean Void Ratio (%)', 'Max Void Ratio (%)'
                ])
                for sample in self.samples:
                    for ln, ls in sample.layers.items():
                        writer.writerow([
                            sample.name, sample.path, ln,
                            ls.object_count, ls.ng_count, f"{ls.ng_rate * 100:.2f}",
                            f"{ls.mean_z_height:.2f}", f"{ls.std_z_height:.2f}",
                            f"{ls.min_z_height:.2f}", f"{ls.max_z_height:.2f}",
                            f"{ls.mean_bump_volume:.0f}", f"{ls.total_bump_volume:.0f}",
                            f"{ls.total_void_volume:.0f}",
                            f"{ls.mean_void_ratio * 100:.3f}", f"{ls.max_void_ratio * 100:.3f}",
                        ])

            QMessageBox.information(self, "Export Successful",
                                    f"SOH report exported to:\n{path}")
            self.status_label.setText(f"Report exported: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{str(e)}")

    # ────────── Cross-Sample Sigma Analysis ──────────

    def _get_sigma_metric_key(self) -> str:
        """Map sigma combo selection to LayerStats attribute."""
        mapping = {
            "SOH / Z Height": "mean_z_height",
            "Void Ratio": "mean_void_ratio",
            "Bump Volume": "mean_bump_volume",
        }
        return mapping.get(self.sigma_metric_combo.currentText(), "mean_z_height")

    def _compute_sigma_data(self) -> List[dict]:
        """Compute cross-sample sigma for each layer.
        
        For each layer that appears in multiple samples, collect the metric
        value from each sample and compute:
            - mean: average across samples
            - sigma (σ): standard deviation across samples
            - CV%: coefficient of variation = (σ / mean) × 100
            - min, max, range
        
        Low σ = consistent measurements across samples = good repeatability.
        High σ = inconsistent = system instability or sample variation.
        """
        if len(self.samples) < 2:
            return []

        metric_key = self._get_sigma_metric_key()

        # Collect all unique layer names across all samples
        all_layers = set()
        for sample in self.samples:
            all_layers.update(sample.layers.keys())
        all_layers = sorted(all_layers, key=_natural_sort_key)

        results = []
        for layer_name in all_layers:
            values = []
            for sample in self.samples:
                if layer_name in sample.layers:
                    val = getattr(sample.layers[layer_name], metric_key, None)
                    if val is not None:
                        values.append(val)

            if len(values) < 2:
                continue

            arr = np.array(values)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1))  # Sample std dev (ddof=1)
            # Advisor's formula for %Tol
            cv_pct = (5.15 * std_val / 3.0) * 100
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            range_val = max_val - min_val

            results.append({
                'layer': layer_name,
                'n_samples': len(values),
                'mean': mean_val,
                'sigma': std_val,
                'cv_pct': cv_pct,
                'min': min_val,
                'max': max_val,
                'range': range_val,
                'values': values,  # raw for per-position drill-down
            })

        return results

    def _refresh_sigma(self):
        """Refresh sigma table and chart."""
        sigma_data = self._compute_sigma_data()
        metric_key = self._get_sigma_metric_key()

        # ── Populate sigma table ──
        self.sigma_table.setRowCount(0)

        # Determine units
        unit_map = {
            "mean_z_height": "µm",
            "mean_void_ratio": "%",
            "mean_bump_volume": "µm³",
        }
        unit = unit_map.get(metric_key, "")
        is_ratio = (metric_key == "mean_void_ratio")

        for row_idx, sd in enumerate(sigma_data):
            self.sigma_table.insertRow(row_idx)

            # Scale ratios to percentage for display
            scale = 100.0 if is_ratio else 1.0

            items = [
                sd['layer'],
                str(sd['n_samples']),
                f"{sd['mean'] * scale:.3f}",
                f"{sd['sigma'] * scale:.4f}",
                f"{sd['cv_pct']:.1f}%",
                f"{sd['min'] * scale:.3f}",
                f"{sd['max'] * scale:.3f}",
                f"{sd['range'] * scale:.4f}",
            ]

            exc_th, acc_th = self.get_tol_thresholds()
            cv = sd['cv_pct']
            if cv < exc_th:
                bg = QColor(40, 180, 60, 30)
                sigma_color = QColor(100, 255, 120)
            elif cv < acc_th:
                bg = QColor(180, 120, 0, 30)
                sigma_color = QColor(255, 200, 0)
            else:
                bg = QColor(180, 40, 40, 40)
                sigma_color = QColor(255, 80, 80)

            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setBackground(bg)
                # Bold sigma and CV columns
                if col in (3, 4):
                    item.setFont(QFont("Consolas", 9, QFont.Bold))
                    item.setForeground(sigma_color)
                self.sigma_table.setItem(row_idx, col, item)

        # ── Sigma chart ──
        self._plot_sigma_chart(sigma_data, metric_key)

    def _plot_sigma_chart(self, sigma_data: list, metric_key: str):
        """Plot sigma values per layer as a bar chart with mean ± σ error bars."""
        self.sigma_chart.fig.clear()

        if not sigma_data:
            self.sigma_chart.draw()
            return

        fig = self.sigma_chart.fig
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0f1320')

        is_ratio = (metric_key == "mean_void_ratio")
        scale = 100.0 if is_ratio else 1.0

        layer_names = [sd['layer'] for sd in sigma_data]
        means = [sd['mean'] * scale for sd in sigma_data]
        sigmas = [sd['sigma'] * scale for sd in sigma_data]
        cvs = [sd['cv_pct'] for sd in sigma_data]

        x = np.arange(len(layer_names))

        exc_th, acc_th = self.get_tol_thresholds()
        # Color bars by CV%
        colors = []
        for cv in cvs:
            if cv < exc_th:
                colors.append('#76ff03')
            elif cv < acc_th:
                colors.append('#ffc107')
            else:
                colors.append('#ff4444')

        # Bar chart of means with sigma error bars
        bars = ax.bar(x, means, color=colors, alpha=0.7,
                      edgecolor='#1a2040', linewidth=0.8)
        ax.errorbar(x, means, yerr=sigmas, fmt='none', ecolor='#ff6b9d',
                    elinewidth=2, capsize=5, capthick=2)

        # Annotate sigma values on top
        for i, (bar, sigma, cv) in enumerate(zip(bars, sigmas, cvs)):
            y_pos = bar.get_height() + sigma + (max(means) * 0.02)
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"σ={sigma:.3f}\n%Tol={cv:.1f}%",
                    ha='center', va='bottom', fontsize=7,
                    color='#e0e0e0', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, fontsize=8, color='#b0b0b0',
                           rotation=30, ha='right')
        ax.tick_params(axis='y', colors='#b0b0b0', labelsize=8)

        metric_labels = {
            "mean_z_height": "Mean SOH / Z Height (µm)",
            "mean_void_ratio": "Mean Void Ratio (%)",
            "mean_bump_volume": "Mean Bump Volume (µm³)",
        }
        title = f"Cross-Sample Repeatability — {metric_labels.get(metric_key, metric_key)}"
        ax.set_title(title, fontsize=10, color='#ff6b9d', pad=10)
        ax.set_ylabel(metric_labels.get(metric_key, metric_key),
                       fontsize=9, color='#b0b0b0')

        # Grid
        ax.grid(axis='y', color='#1a2040', linewidth=0.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2a3050')
        ax.spines['left'].set_color('#2a3050')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#76ff03', edgecolor='#1a2040', label=f'%Tol < {exc_th}% (Excellent)'),
            Patch(facecolor='#ffc107', edgecolor='#1a2040', label=f'%Tol {exc_th}–{acc_th}% (Acceptable)'),
            Patch(facecolor='#ff4444', edgecolor='#1a2040', label=f'%Tol > {acc_th}% (Poor)'),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc='upper right',
                  framealpha=0.5, facecolor='#0f1320', edgecolor='#2a3050',
                  labelcolor='#e0e0e0')

        fig.tight_layout()
        self.sigma_chart.draw()

    # ────────── Per-Bump Sigma Analysis ──────────

    def _get_bump_sigma_metric_key(self) -> str:
        """Map per-bump sigma combo to CSV column key."""
        mapping = {
            "SOH / Z Height": "soh",
            "Void Ratio": "ratio",
            "Bump Volume": "bump_vol",
        }
        return mapping.get(self.bump_sigma_metric_combo.currentText(), "soh")

    def _extract_bump_index(self, row: dict) -> Optional[str]:
        """Extract bump grid index from a CSV row. Returns '(R,C)' string or None."""
        grid_val = row.get('Grid(R,C)', '').strip()
        if grid_val and grid_val != '(?,?)':
            return grid_val
        idx_val = row.get('#', '').strip()
        if idx_val:
            return idx_val
        return None

    def _extract_bump_metric(self, row: dict, metric_key: str) -> float:
        """Extract a metric value from a raw CSV row."""
        if metric_key == 'soh':
            return _safe_float(row.get('SOH (um)', row.get('SOH(µm)', '0')))
        elif metric_key == 'ratio':
            val = _safe_float(row.get('Ratio (Void/(TGV+Void))', row.get('Ratio', '0')))
            if val > 1.0:
                val = val / 100.0
            return val
        elif metric_key == 'bump_vol':
            return _safe_float(row.get('Bump Volume (um3)',
                               row.get('Bump Vol(µm³)',
                               row.get('Bump Vol(um3)', '0'))))
        return 0.0

    def _get_row_layer(self, row: dict) -> str:
        """Get layer name from a raw CSV row."""
        return row.get('_layer_resolved', row.get('Layer', 'Default')).strip() or 'Default'

    def _on_bump_layer_changed(self):
        """When layer combo changes, repopulate bump index combo."""
        selected_layer = self.bump_sigma_layer_combo.currentText()
        self.bump_sigma_index_combo.blockSignals(True)
        self.bump_sigma_index_combo.clear()
        self.bump_sigma_index_combo.addItem("All Bumps")

        if selected_layer and selected_layer != "— Select Layer —":
            # Collect all bump indices for this layer
            bump_indices = set()
            for sample in self.samples:
                for row in sample.raw_rows:
                    layer = self._get_row_layer(row)
                    if layer == selected_layer:
                        idx = self._extract_bump_index(row)
                        if idx:
                            bump_indices.add(idx)
            # Sort naturally: (1,1) < (1,2) < (2,1) etc.
            for bi in sorted(bump_indices, key=_natural_sort_key):
                self.bump_sigma_index_combo.addItem(bi)

        self.bump_sigma_index_combo.blockSignals(False)
        self._refresh_per_bump_sigma()

    def _refresh_per_bump_sigma(self):
        """Refresh the per-bump sigma table and chart.
        
        Two modes:
        - 'All Bumps': overview table (σ per bump, sorted worst-first)
        - Specific bump selected: detail table (1 row per sample)
        """
        # Update layer combo options (preserve current selection)
        current_layer = self.bump_sigma_layer_combo.currentText()
        self.bump_sigma_layer_combo.blockSignals(True)
        old_layers = [self.bump_sigma_layer_combo.itemText(i)
                      for i in range(self.bump_sigma_layer_combo.count())]
        all_layers_set = set()
        for sample in self.samples:
            for row in sample.raw_rows:
                all_layers_set.add(self._get_row_layer(row))
        new_layers = ["— Select Layer —"] + sorted(all_layers_set, key=_natural_sort_key)
        if new_layers != old_layers:
            self.bump_sigma_layer_combo.clear()
            self.bump_sigma_layer_combo.addItems(new_layers)
            idx = self.bump_sigma_layer_combo.findText(current_layer)
            if idx >= 0:
                self.bump_sigma_layer_combo.setCurrentIndex(idx)
        self.bump_sigma_layer_combo.blockSignals(False)

        selected_layer = self.bump_sigma_layer_combo.currentText()
        selected_bump = self.bump_sigma_index_combo.currentText()

        # Determine mode
        if (selected_layer and selected_layer != "— Select Layer —"
                and selected_bump and selected_bump != "All Bumps"):
            self._refresh_bump_detail_mode(selected_layer, selected_bump)
        else:
            self._refresh_bump_overview_mode(selected_layer)

    def _refresh_bump_overview_mode(self, selected_layer: str):
        """Overview: show σ for each bump across samples (sorted worst-first)."""
        self._set_bump_table_overview_mode()
        self.bump_sigma_chart_label.setText("<b>ACROSS BUMPS</b>  (mean ± σ, lower σ = better repeatability)")

        if len(self.samples) < 2:
            self.bump_sigma_table.setRowCount(0)
            self.bump_sigma_summary_label.setText("Import ≥2 samples to compute per-bump sigma")
            self.bump_sigma_chart.clear()
            return

        metric_key = self._get_bump_sigma_metric_key()
        is_ratio = (metric_key == 'ratio')
        scale = 100.0 if is_ratio else 1.0

        # Build bump_map: {(layer, bump_idx): {sample_name: value}}
        bump_map: Dict[tuple, Dict[str, float]] = {}
        for sample in self.samples:
            for row in sample.raw_rows:
                layer = self._get_row_layer(row)
                if selected_layer not in ("— Select Layer —", "") and layer != selected_layer:
                    continue
                bump_idx = self._extract_bump_index(row)
                if bump_idx is None:
                    continue
                val = self._extract_bump_metric(row, metric_key)
                key = (layer, bump_idx)
                if key not in bump_map:
                    bump_map[key] = {}
                bump_map[key][sample.name] = val

        # Compute stats
        results = []
        for (layer, bump_idx), sample_vals in bump_map.items():
            if len(sample_vals) < 2:
                continue
            arr = np.array(list(sample_vals.values()))
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1))
            # Advisor's formula for %Tol
            cv_pct = (5.15 * std_val / 3.0) * 100
            results.append({
                'layer': layer, 'bump_idx': bump_idx,
                'n_samples': len(sample_vals),
                'mean': mean_val, 'sigma': std_val, 'cv_pct': cv_pct,
                'min': float(np.min(arr)), 'max': float(np.max(arr)),
                'range': float(np.max(arr) - np.min(arr)),
            })
        results.sort(key=lambda x: x['sigma'], reverse=True)

        self.bump_sigma_summary_label.setText(
            f"{len(results)} bump(s) with ≥2 samples | "
            f"Select a specific Layer → Bump to drill down")

        # Populate table
        self.bump_sigma_table.setSortingEnabled(False)
        self.bump_sigma_table.setRowCount(0)
        for row_idx, bd in enumerate(results):
            self.bump_sigma_table.insertRow(row_idx)
            items = [
                str(row_idx + 1), bd['layer'], bd['bump_idx'], str(bd['n_samples']),
                f"{bd['mean'] * scale:.4f}", f"{bd['sigma'] * scale:.4f}",
                f"{bd['cv_pct']:.1f}%", f"{bd['min'] * scale:.4f}",
                f"{bd['max'] * scale:.4f}", f"{bd['range'] * scale:.4f}",
            ]
            exc_th, acc_th = self.get_tol_thresholds()
            cv = bd['cv_pct']
            bg = (QColor(40, 180, 60, 30) if cv < exc_th
                  else QColor(180, 120, 0, 30) if cv < acc_th
                  else QColor(180, 40, 40, 40))
            s_color = (QColor(100, 255, 120) if cv < exc_th
                       else QColor(255, 200, 0) if cv < acc_th
                       else QColor(255, 80, 80))
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setBackground(bg)
                if col in (5, 6):
                    item.setFont(QFont("Consolas", 9, QFont.Bold))
                    item.setForeground(s_color)
                self.bump_sigma_table.setItem(row_idx, col, item)
        self.bump_sigma_table.setSortingEnabled(True)

        # Chart: top 20 worst
        self._plot_bump_overview_chart(results[:20], metric_key)

    def _refresh_bump_detail_mode(self, layer: str, bump_idx: str):
        """Detail: show per-sample values for one specific (layer, bump)."""
        self._set_bump_table_detail_mode()
        self.bump_sigma_chart_label.setText(
            f"<b>{layer} — Bump {bump_idx}</b>  (per-sample values)")

        metric_key = self._get_bump_sigma_metric_key()
        is_ratio = (metric_key == 'ratio')
        scale = 100.0 if is_ratio else 1.0
        unit_map = {'soh': 'µm', 'ratio': '%', 'bump_vol': 'µm³'}
        unit = unit_map.get(metric_key, '')

        # Collect values per sample
        sample_values = []  # list of (sample_name, value)
        for sample in self.samples:
            for row in sample.raw_rows:
                if self._get_row_layer(row) != layer:
                    continue
                if self._extract_bump_index(row) != bump_idx:
                    continue
                val = self._extract_bump_metric(row, metric_key)
                sample_values.append((sample.name, val))
                break  # one bump per sample per layer

        if not sample_values:
            self.bump_sigma_table.setRowCount(0)
            self.bump_sigma_summary_label.setText("No data for this bump/layer combination")
            self.bump_sigma_chart.clear()
            return

        arr = np.array([v for _, v in sample_values])
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        cv_pct = (5.15 * std_val / 3.0) * 100

        self.bump_sigma_summary_label.setText(
            f"<b>{layer} — Bump {bump_idx}</b>  |  "
            f"Mean = {mean_val * scale:.3f} {unit}  |  "
            f"<b style='color: #76ff03;'>StdDev.S = {std_val * scale:.4f} {unit}</b>  |  "
            f"%Tol = {cv_pct:.1f}%  |  "
            f"N = {len(sample_values)} samples")

        # Populate detail table
        self.bump_sigma_table.setSortingEnabled(False)
        self.bump_sigma_table.setRowCount(0)
        for row_idx, (s_name, val) in enumerate(sample_values):
            self.bump_sigma_table.insertRow(row_idx)
            delta = val - mean_val
            items = [
                str(row_idx + 1),
                s_name,
                f"{val * scale:.3f}",
                f"{delta * scale:+.4f}",
            ]
            # Color by deviation from mean
            abs_delta = abs(delta)
            if std_val > 0 and abs_delta > 2 * std_val:
                bg = QColor(180, 40, 40, 40)
                delta_color = QColor(255, 80, 80)
            elif std_val > 0 and abs_delta > std_val:
                bg = QColor(180, 120, 0, 30)
                delta_color = QColor(255, 200, 0)
            else:
                bg = QColor(40, 180, 60, 30)
                delta_color = QColor(100, 255, 120)

            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setBackground(bg)
                if col == 3:
                    item.setFont(QFont("Consolas", 9, QFont.Bold))
                    item.setForeground(delta_color)
                self.bump_sigma_table.setItem(row_idx, col, item)
        self.bump_sigma_table.setSortingEnabled(True)

        # Chart: per-sample bars with mean ± σ lines
        self._plot_bump_detail_chart(sample_values, mean_val, std_val, metric_key, layer, bump_idx)

    def _plot_bump_overview_chart(self, bump_data: list, metric_key: str):
        """Bar chart of mean ± σ per bump, matching cross-sample sigma chart style."""
        self.bump_sigma_chart.fig.clear()
        if not bump_data:
            self.bump_sigma_chart.draw()
            return

        fig = self.bump_sigma_chart.fig
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0f1320')

        is_ratio = (metric_key == 'ratio')
        scale = 100.0 if is_ratio else 1.0

        labels = [f"{bd['bump_idx']}" for bd in bump_data]
        means = [bd['mean'] * scale for bd in bump_data]
        sigmas = [bd['sigma'] * scale for bd in bump_data]
        cvs = [bd['cv_pct'] for bd in bump_data]

        exc_th, acc_th = self.get_tol_thresholds()
        x = np.arange(len(labels))
        colors = ['#ff4444' if cv >= acc_th else '#ffc107' if cv >= exc_th else '#76ff03' for cv in cvs]

        # Bar = Mean, Error bar = ±σ (same style as cross-sample sigma)
        bars = ax.bar(x, means, color=colors, alpha=0.7,
                      edgecolor='#1a2040', linewidth=0.8)
        ax.errorbar(x, means, yerr=sigmas, fmt='none', ecolor='#ff6b9d',
                    elinewidth=2, capsize=5, capthick=2)

        # Annotate σ and CV on top
        for i, (bar, sigma, cv) in enumerate(zip(bars, sigmas, cvs)):
            y_pos = bar.get_height() + sigma + (max(means) * 0.02 if means else 0)
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"σ={sigma:.3f}\n%Tol={cv:.1f}%",
                    ha='center', va='bottom', fontsize=7,
                    color='#e0e0e0', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, color='#b0b0b0', rotation=30, ha='right')
        ax.tick_params(axis='y', colors='#b0b0b0', labelsize=8)

        unit_map = {'soh': 'µm', 'ratio': '%', 'bump_vol': 'µm³'}
        unit = unit_map.get(metric_key, '')
        metric_labels = {'soh': 'Mean SOH / Z Height', 'ratio': 'Mean Void Ratio',
                         'bump_vol': 'Mean Bump Volume'}
        ax.set_title(f"Per-Bump Repeatability — {metric_labels.get(metric_key, '')} ({unit})",
                     fontsize=10, color='#76ff03', pad=10)
        ax.set_ylabel(f"{metric_labels.get(metric_key, '')} ({unit})",
                      fontsize=9, color='#b0b0b0')

        ax.grid(axis='y', color='#1a2040', linewidth=0.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2a3050')
        ax.spines['left'].set_color('#2a3050')

        from matplotlib.patches import Patch
        import matplotlib.lines as mlines
        legend_elements = [
            Patch(facecolor='#76ff03', edgecolor='#1a2040', label=f'%Tol < {exc_th}% (Excellent)'),
            Patch(facecolor='#ffc107', edgecolor='#1a2040', label=f'%Tol {exc_th}–{acc_th}% (Acceptable)'),
            Patch(facecolor='#ff4444', edgecolor='#1a2040', label=f'%Tol > {acc_th}% (Poor)'),
            mlines.Line2D([], [], color='#ff6b9d', linewidth=2, label='±1σ error bar'),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc='upper right',
                  framealpha=0.5, facecolor='#0f1320', edgecolor='#2a3050',
                  labelcolor='#e0e0e0')

        fig.tight_layout()
        self.bump_sigma_chart.draw()

    def _plot_bump_detail_chart(self, sample_values, mean_val, std_val, metric_key,
                                layer, bump_idx):
        """Bar chart: one bar per sample, with mean ± σ lines (matching cross-sample style)."""
        self.bump_sigma_chart.fig.clear()
        if not sample_values:
            self.bump_sigma_chart.draw()
            return

        fig = self.bump_sigma_chart.fig
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0f1320')

        is_ratio = (metric_key == 'ratio')
        scale = 100.0 if is_ratio else 1.0
        unit_map = {'soh': 'µm', 'ratio': '%', 'bump_vol': 'µm³'}
        unit = unit_map.get(metric_key, '')

        names = [sv[0] for sv in sample_values]
        values = [sv[1] * scale for sv in sample_values]
        mean_s = mean_val * scale
        std_s = std_val * scale

        x = np.arange(len(names))

        # Color bars by deviation from mean
        colors = []
        for v in values:
            delta = abs(v - mean_s)
            if std_s > 0 and delta > 2 * std_s:
                colors.append('#ff4444')
            elif std_s > 0 and delta > std_s:
                colors.append('#ffc107')
            else:
                colors.append('#76ff03')

        bars = ax.bar(x, values, color=colors, alpha=0.7,
                      edgecolor='#1a2040', linewidth=0.8)

        # Value labels on bars (including Z-score and % tolerance)
        for bar, val in zip(bars, values):
            z_score = (val - mean_s) / std_s if std_s > 0 else 0
            tol_pct = (val - mean_s) / abs(mean_s) * 100 if mean_s != 0 else 0
            
            label_text = f'{val:.3f}\n({z_score:+.1f}σ)\n{tol_pct:+.1f}%'
            
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    label_text, ha='center', va='bottom',
                    fontsize=7, color='#e0e0e0', fontweight='bold')

        # Mean + ±1σ reference lines
        ax.axhline(y=mean_s, color='#00e5ff', linewidth=2, linestyle='-',
                   label=f'Mean = {mean_s:.3f}', alpha=0.9)
        if std_s > 0:
            ax.axhline(y=mean_s + std_s, color='#ff6b9d', linewidth=1.5,
                       linestyle='--', label=f'+1σ = {mean_s + std_s:.3f}', alpha=0.7)
            ax.axhline(y=mean_s - std_s, color='#ff6b9d', linewidth=1.5,
                       linestyle='--', label=f'−1σ = {mean_s - std_s:.3f}', alpha=0.7)
            ax.axhspan(mean_s - std_s, mean_s + std_s,
                       alpha=0.06, color='#00e5ff')

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, color='#b0b0b0', rotation=30, ha='right')
        ax.tick_params(axis='y', colors='#b0b0b0', labelsize=8)

        ax.set_title(f"{layer} — Bump {bump_idx}  ({unit})",
                     fontsize=10, color='#76ff03', pad=10)
        ax.set_ylabel(f"Value ({unit})", fontsize=9, color='#b0b0b0')

        ax.grid(axis='y', color='#1a2040', linewidth=0.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2a3050')
        ax.spines['left'].set_color('#2a3050')

        ax.legend(fontsize=7, loc='upper right', framealpha=0.5,
                  facecolor='#0f1320', edgecolor='#2a3050', labelcolor='#e0e0e0')

        fig.tight_layout()
        self.bump_sigma_chart.draw()


    # ══════════════════════════════════════════════════════════════
    # PAGE 3: 3D Structural Integrity Map (3D-SIM)
    # ══════════════════════════════════════════════════════════════

    def _create_3d_sim_section(self):
        """Build the 3D-SIM Analysis UI section."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        combo_ss = f"""
            QComboBox {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 3px; padding: 4px 8px; font-size: 9pt;
            }}
        """
        lbl_ss = f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-weight: bold;"

        # ── Header row ──
        header_row = QHBoxLayout()
        title = QLabel(
            "<b>3D STRUCTURAL INTEGRITY MAP</b>  "
            "<span style='color: #808080;'>Spatial Autocorrelation + Cross-Layer Propagation</span>")
        title.setStyleSheet("font-size: 10pt; color: #e040fb; padding: 4px 8px;")
        header_row.addWidget(title)
        header_row.addStretch()

        layer_lbl = QLabel("Layer:")
        layer_lbl.setStyleSheet(lbl_ss)
        header_row.addWidget(layer_lbl)

        self.sim_layer_combo = QComboBox()
        self.sim_layer_combo.addItem("— All Layers —")
        self.sim_layer_combo.setFixedWidth(160)
        self.sim_layer_combo.setStyleSheet(combo_ss)
        self.sim_layer_combo.currentIndexChanged.connect(self._refresh_3d_sim)
        header_row.addWidget(self.sim_layer_combo)

        metric_lbl = QLabel("Metric:")
        metric_lbl.setStyleSheet(lbl_ss)
        header_row.addWidget(metric_lbl)

        self.sim_metric_combo = QComboBox()
        self.sim_metric_combo.addItems(["Void Ratio", "SOH / Z Height"])
        self.sim_metric_combo.setFixedWidth(140)
        self.sim_metric_combo.setStyleSheet(combo_ss)
        self.sim_metric_combo.currentIndexChanged.connect(self._refresh_3d_sim)
        header_row.addWidget(self.sim_metric_combo)

        btn_run = QPushButton(" Analyze")
        btn_run.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #e040fb,stop:1 #7c4dff);
                color: white; font-weight: bold; font-size: 9pt;
                border: none; border-radius: 4px; padding: 6px 16px;
            }}
            QPushButton:hover {{ background: #e040fb; }}
        """)
        btn_run.setCursor(Qt.PointingHandCursor)
        btn_run.clicked.connect(self._refresh_3d_sim)
        header_row.addWidget(btn_run)

        layout.addLayout(header_row)

        # ── Summary label ──
        self.sim_summary_label = QLabel("Import samples and click Analyze to compute 3D-SIM")
        self.sim_summary_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt; "
            f"padding: 2px 8px; font-style: italic;")
        layout.addWidget(self.sim_summary_label)

        # ── Content: 2×2 grid of charts ──
        grid_splitter = QSplitter(Qt.Vertical)
        grid_splitter.setHandleWidth(3)
        grid_splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {SemiconductorTheme.BORDER_DEFAULT}; }}")

        # TOP ROW: SII Heatmap (left) + Correlation Matrix (right)
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setHandleWidth(3)
        top_splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {SemiconductorTheme.BORDER_DEFAULT}; }}")

        # SII Heatmap
        hm_container = QWidget()
        hm_layout = QVBoxLayout(hm_container)
        hm_layout.setContentsMargins(4, 4, 4, 4)
        hm_layout.setSpacing(2)
        hm_label = QLabel("<b>STRUCTURAL INTEGRITY INDEX (SII) MAP</b>")
        hm_label.setStyleSheet("color: #e040fb; font-size: 9pt; padding: 2px;")
        hm_layout.addWidget(hm_label)
        self.sim_heatmap_chart = EmbeddedChart(parent=hm_container, width=5, height=4)
        hm_layout.addWidget(self.sim_heatmap_chart, 1)
        top_splitter.addWidget(hm_container)

        # Correlation Matrix
        cm_container = QWidget()
        cm_layout = QVBoxLayout(cm_container)
        cm_layout.setContentsMargins(4, 4, 4, 4)
        cm_layout.setSpacing(2)
        cm_label = QLabel("<b>CROSS-LAYER PROPAGATION MATRIX</b>")
        cm_label.setStyleSheet("color: #e040fb; font-size: 9pt; padding: 2px;")
        cm_layout.addWidget(cm_label)
        self.sim_corr_chart = EmbeddedChart(parent=cm_container, width=4, height=4)
        cm_layout.addWidget(self.sim_corr_chart, 1)
        top_splitter.addWidget(cm_container)

        top_splitter.setSizes([550, 450])
        grid_splitter.addWidget(top_splitter)

        # BOTTOM ROW: Moran's I bar chart (left) + Weak Zone table (right)
        bot_splitter = QSplitter(Qt.Horizontal)
        bot_splitter.setHandleWidth(3)
        bot_splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {SemiconductorTheme.BORDER_DEFAULT}; }}")

        # Moran's I chart
        mi_container = QWidget()
        mi_layout = QVBoxLayout(mi_container)
        mi_layout.setContentsMargins(4, 4, 4, 4)
        mi_layout.setSpacing(2)
        mi_label = QLabel("<b>MORAN'S I PER LAYER</b>  (spatial clustering index)")
        mi_label.setStyleSheet("color: #e040fb; font-size: 9pt; padding: 2px;")
        mi_layout.addWidget(mi_label)
        self.sim_moran_chart = EmbeddedChart(parent=mi_container, width=5, height=3)
        mi_layout.addWidget(self.sim_moran_chart, 1)
        bot_splitter.addWidget(mi_container)

        # Weak Zone & Stats table
        wz_container = QWidget()
        wz_layout = QVBoxLayout(wz_container)
        wz_layout.setContentsMargins(4, 4, 4, 4)
        wz_layout.setSpacing(2)
        wz_label = QLabel("<b>ANALYSIS RESULTS</b>")
        wz_label.setStyleSheet("color: #e040fb; font-size: 9pt; padding: 2px;")
        wz_layout.addWidget(wz_label)

        self.sim_results_table = QTableWidget()
        self.sim_results_table.setColumnCount(5)
        self.sim_results_table.setHorizontalHeaderLabels([
            "Layer", "Moran's I", "Pattern", "Mean SII", "Risk Bumps"
        ])
        self.sim_results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sim_results_table.setAlternatingRowColors(True)
        self.sim_results_table.setStyleSheet(f"""
            QTableWidget {{
                background: {SemiconductorTheme.BG_DARK};
                alternate-background-color: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                gridline-color: {SemiconductorTheme.BORDER_DEFAULT};
                font-size: 9pt;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QHeaderView::section {{
                background: {SemiconductorTheme.BG_PANEL};
                color: #e040fb;
                font-weight: bold; font-size: 8pt; padding: 3px;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """)
        self.sim_results_table.horizontalHeader().setStretchLastSection(True)
        self.sim_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sim_results_table.verticalHeader().setVisible(False)
        wz_layout.addWidget(self.sim_results_table, 1)
        bot_splitter.addWidget(wz_container)

        bot_splitter.setSizes([550, 450])
        grid_splitter.addWidget(bot_splitter)
        grid_splitter.setSizes([500, 300])

        layout.addWidget(grid_splitter, 1)
        return container

    def _refresh_3d_sim(self):
        """Run the 3D-SIM analysis pipeline and update all charts."""
        try:
            from inno3d.core.spatial_analysis import (
                run_full_analysis, _natural_sort_key as sim_sort_key)
        except ImportError:
            from inno3d.core.spatial_analysis import run_full_analysis
            sim_sort_key = _natural_sort_key

        if not self.samples:
            self.sim_summary_label.setText("No samples loaded. Import data first.")
            return

        # Use first selected sample, or first sample
        idx = self._selected_sample_idx if self._selected_sample_idx is not None else 0
        if idx >= len(self.samples):
            idx = 0
        sample = self.samples[idx]

        if not sample.raw_rows:
            self.sim_summary_label.setText(f"Sample '{sample.name}' has no measurement data.")
            return

        # Determine metric
        metric_idx = self.sim_metric_combo.currentIndex()
        metric = 'void_ratio' if metric_idx == 0 else 'soh'

        # Run analysis
        try:
            results = run_full_analysis(
                sample.raw_rows, sample_name=sample.name, metric=metric)
        except Exception as e:
            self.sim_summary_label.setText(f"Analysis error: {e}")
            return

        moran = results['moran']
        prop = results['propagation']
        sii = results['sii']
        weak_zones = results['weak_zones']
        fp = results['fingerprint']

        # ── Update summary label ──
        n_layers = len(results['layer_bumps'])
        total_bumps = sum(len(b) for b in results['layer_bumps'].values())
        total_risk = sum(s.risk_count for s in sii.values())
        total_wz = sum(len(z) for z in weak_zones.values())
        prop_dir = prop.dominant_direction if prop else 'N/A'
        self.sim_summary_label.setText(
            f"<b>{sample.name}</b> — {n_layers} layers, {total_bumps} bumps | "
            f"Mean SII: {fp.mean_sii:.3f} | Risk bumps: {total_risk} | "
            f"Weak zones: {total_wz} | Propagation: {prop_dir}")

        # ── Populate results table ──
        layer_names = sorted(results['layer_bumps'].keys(), key=_natural_sort_key)
        self.sim_results_table.setRowCount(len(layer_names))
        for row_i, ln in enumerate(layer_names):
            m_result = moran.get(ln)
            s_result = sii.get(ln)
            items = [
                ln,
                f"{m_result.I_global:.3f}" if m_result else "N/A",
                m_result.pattern if m_result else "N/A",
                f"{s_result.mean_sii:.3f}" if s_result else "N/A",
                f"{s_result.risk_count}" if s_result else "0"
            ]
            for col_i, txt in enumerate(items):
                item = QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                # Color code pattern
                if col_i == 2:
                    if txt == 'Clustered':
                        item.setForeground(QBrush(QColor('#ff4444')))
                    elif txt == 'Dispersed':
                        item.setForeground(QBrush(QColor('#76ff03')))
                    else:
                        item.setForeground(QBrush(QColor('#808080')))
                # Color code SII
                if col_i == 3 and s_result:
                    if s_result.mean_sii < 0.4:
                        item.setForeground(QBrush(QColor('#ff4444')))
                    elif s_result.mean_sii < 0.7:
                        item.setForeground(QBrush(QColor('#ffc107')))
                    else:
                        item.setForeground(QBrush(QColor('#76ff03')))
                self.sim_results_table.setItem(row_i, col_i, item)

        # ── Plot 1: SII Heatmap ──
        self._plot_sii_heatmap(sii, weak_zones, layer_names, sample.name)

        # ── Plot 2: Correlation Matrix ──
        self._plot_corr_matrix(prop)

        # ── Plot 3: Moran's I bar chart ──
        self._plot_moran_bars(moran, layer_names)

        # Update layer combo for SIM page
        self.sim_layer_combo.blockSignals(True)
        self.sim_layer_combo.clear()
        self.sim_layer_combo.addItem("— All Layers —")
        for ln in layer_names:
            self.sim_layer_combo.addItem(ln)
        self.sim_layer_combo.blockSignals(False)

    def _plot_sii_heatmap(self, sii_results, weak_zones, layer_names, sample_name):
        """Plot SII scatter heatmap with weak zone overlays."""
        fig = self.sim_heatmap_chart.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0f1320')

        # Determine which layer to show
        sel_idx = self.sim_layer_combo.currentIndex()
        if sel_idx > 0 and sel_idx - 1 < len(layer_names):
            show_layers = [layer_names[sel_idx - 1]]
        else:
            show_layers = layer_names

        all_cx, all_cy, all_sii = [], [], []
        for ln in show_layers:
            if ln in sii_results:
                s = sii_results[ln]
                all_cx.extend(s.cx.tolist())
                all_cy.extend(s.cy.tolist())
                all_sii.extend(s.sii_values.tolist())

        if not all_cx:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                    color='#808080', fontsize=12, transform=ax.transAxes)
            fig.tight_layout()
            self.sim_heatmap_chart.draw()
            return

        cx = np.array(all_cx)
        cy = np.array(all_cy)
        sii_vals = np.array(all_sii)

        # Scatter plot with color = SII
        sc = ax.scatter(cx, cy, c=sii_vals, cmap='RdYlGn', vmin=0, vmax=1,
                        s=40, edgecolors='#1a2040', linewidth=0.5, alpha=0.85, zorder=2)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('SII', fontsize=8, color='#b0b0b0')
        cbar.ax.tick_params(colors='#b0b0b0', labelsize=7)

        # Draw weak zone rectangles
        for ln in show_layers:
            if ln in weak_zones:
                for wz in weak_zones[ln]:
                    x0, y0, x1, y1 = wz.bbox
                    pad = 5
                    color = '#ff4444' if wz.severity == 'Critical' else (
                            '#ffc107' if wz.severity == 'Warning' else '#ff8800')
                    rect = MplRect((x0 - pad, y0 - pad), x1 - x0 + 2 * pad,
                                   y1 - y0 + 2 * pad,
                                   linewidth=2, edgecolor=color,
                                   facecolor=color, alpha=0.12, zorder=1)
                    ax.add_patch(rect)
                    ax.text(x0, y1 + pad + 2,
                            f"{wz.severity}\n{wz.zone_type}",
                            fontsize=6, color=color, fontweight='bold')

        layer_str = show_layers[0] if len(show_layers) == 1 else f"All ({len(show_layers)} layers)"
        ax.set_title(f"{sample_name} — {layer_str}",
                     fontsize=10, color='#e040fb', pad=8)
        ax.set_xlabel("X (px)", fontsize=8, color='#b0b0b0')
        ax.set_ylabel("Y (px)", fontsize=8, color='#b0b0b0')
        ax.tick_params(colors='#b0b0b0', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2a3050')
        ax.spines['left'].set_color('#2a3050')
        ax.invert_yaxis()

        # Legend
        legend_patches = [
            Patch(facecolor='#ff4444', alpha=0.3, label='Critical Zone'),
            Patch(facecolor='#ffc107', alpha=0.3, label='Warning Zone'),
        ]
        ax.legend(handles=legend_patches, fontsize=7, loc='upper right',
                  framealpha=0.5, facecolor='#0f1320', edgecolor='#2a3050',
                  labelcolor='#e0e0e0')

        fig.tight_layout()
        self.sim_heatmap_chart.draw()

    def _plot_corr_matrix(self, prop_result):
        """Plot cross-layer correlation matrix as heatmap."""
        fig = self.sim_corr_chart.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0f1320')

        if not prop_result or len(prop_result.layer_names) < 2:
            ax.text(0.5, 0.5, "Need ≥ 2 layers", ha='center', va='center',
                    color='#808080', fontsize=12, transform=ax.transAxes)
            fig.tight_layout()
            self.sim_corr_chart.draw()
            return

        matrix = prop_result.correlation_matrix
        names = prop_result.layer_names
        n = len(names)

        # Truncate long layer names
        short_names = [ln[:12] for ln in names]

        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Pearson r', fontsize=8, color='#b0b0b0')
        cbar.ax.tick_params(colors='#b0b0b0', labelsize=7)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_names, fontsize=7, color='#b0b0b0', rotation=45, ha='right')
        ax.set_yticklabels(short_names, fontsize=7, color='#b0b0b0')

        # Annotate values
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                color = 'white' if abs(val) > 0.5 else '#b0b0b0'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')

        dir_label = prop_result.dominant_direction.replace('-', '→').replace('none', 'No significant')
        ax.set_title(f"Propagation: {dir_label} (score={prop_result.overall_score:.2f})",
                     fontsize=9, color='#e040fb', pad=8)

        fig.tight_layout()
        self.sim_corr_chart.draw()

    def _plot_moran_bars(self, moran_results, layer_names):
        """Plot Moran's I values per layer as bar chart."""
        fig = self.sim_moran_chart.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0f1320')

        if not moran_results:
            ax.text(0.5, 0.5, "No spatial data", ha='center', va='center',
                    color='#808080', fontsize=12, transform=ax.transAxes)
            fig.tight_layout()
            self.sim_moran_chart.draw()
            return

        layers = [ln for ln in layer_names if ln in moran_results]
        values = [moran_results[ln].I_global for ln in layers]
        patterns = [moran_results[ln].pattern for ln in layers]
        p_vals = [moran_results[ln].p_value for ln in layers]

        colors = []
        for pat, pv in zip(patterns, p_vals):
            if pat == 'Clustered':
                colors.append('#ff4444')
            elif pat == 'Dispersed':
                colors.append('#76ff03')
            else:
                colors.append('#607080')

        x = range(len(layers))
        bars = ax.bar(x, values, color=colors, edgecolor='#1a2040', linewidth=0.8, alpha=0.85)

        # Value labels
        for bar, val, pv in zip(bars, values, p_vals):
            sig = "**" if pv < 0.01 else ("*" if pv < 0.05 else "")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}{sig}", ha='center', va='bottom',
                    fontsize=7, color='#e0e0e0', fontweight='bold')

        # Reference lines
        ax.axhline(y=0, color='#808080', linewidth=0.5, linestyle='-')
        ax.axhline(y=0.3, color='#ff4444', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axhline(y=-0.3, color='#76ff03', linewidth=0.5, linestyle='--', alpha=0.5)

        short_names = [ln[:12] for ln in layers]
        ax.set_xticks(list(x))
        ax.set_xticklabels(short_names, fontsize=7, color='#b0b0b0', rotation=30, ha='right')
        ax.tick_params(axis='y', colors='#b0b0b0', labelsize=7)
        ax.set_ylabel("Moran's I", fontsize=8, color='#b0b0b0')
        ax.set_title("Spatial Autocorrelation (* p<0.05, ** p<0.01)",
                     fontsize=9, color='#e040fb', pad=8)

        legend_patches = [
            Patch(facecolor='#ff4444', label='Clustered (I > 0)'),
            Patch(facecolor='#76ff03', label='Dispersed (I < 0)'),
            Patch(facecolor='#607080', label='Random'),
        ]
        ax.legend(handles=legend_patches, fontsize=7, loc='upper right',
                  framealpha=0.5, facecolor='#0f1320', edgecolor='#2a3050',
                  labelcolor='#e0e0e0')

        ax.grid(axis='y', color='#1a2040', linewidth=0.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2a3050')
        ax.spines['left'].set_color('#2a3050')

        fig.tight_layout()
        self.sim_moran_chart.draw()

    # ══════════════════════════════════════════════════════════════
    # PAGE 4: LOO Variance Decomposition (LOO-VD)
    # ══════════════════════════════════════════════════════════════

    def _create_loo_vd_section(self):
        """Build the LOO-VD Analysis UI section."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        ACCENT = '#ff9100'  # orange accent for LOO-VD

        combo_ss = f"""
            QComboBox {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 3px; padding: 4px 8px; font-size: 9pt;
            }}
        """
        lbl_ss = f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-weight: bold;"

        # ── Header row ──
        header_row = QHBoxLayout()
        title = QLabel(
            "<b>LEAVE-ONE-OUT VARIANCE DECOMPOSITION</b>  "
            "<span style='color: #808080;'>Root-Cause σ Localization</span>")
        title.setStyleSheet(f"font-size: 10pt; color: {ACCENT}; padding: 4px 8px;")
        header_row.addWidget(title)
        header_row.addStretch()

        layer_lbl = QLabel("Layer:")
        layer_lbl.setStyleSheet(lbl_ss)
        header_row.addWidget(layer_lbl)

        self.loo_layer_combo = QComboBox()
        self.loo_layer_combo.addItem("— Select Layer —")
        self.loo_layer_combo.setFixedWidth(160)
        self.loo_layer_combo.setStyleSheet(combo_ss)
        self.loo_layer_combo.currentIndexChanged.connect(self._refresh_loo_vd)
        header_row.addWidget(self.loo_layer_combo)

        metric_lbl = QLabel("Metric:")
        metric_lbl.setStyleSheet(lbl_ss)
        header_row.addWidget(metric_lbl)

        self.loo_metric_combo = QComboBox()
        self.loo_metric_combo.addItems(["SOH / Z Height", "Void Ratio", "Bump Volume"])
        self.loo_metric_combo.setFixedWidth(140)
        self.loo_metric_combo.setStyleSheet(combo_ss)
        self.loo_metric_combo.currentIndexChanged.connect(self._refresh_loo_vd)
        header_row.addWidget(self.loo_metric_combo)

        btn_run = QPushButton(" Decompose")
        btn_run.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #ff9100,stop:1 #ff5722);
                color: white; font-weight: bold; font-size: 9pt;
                border: none; border-radius: 4px; padding: 6px 16px;
            }}
            QPushButton:hover {{ background: #ff9100; }}
        """)
        btn_run.setCursor(Qt.PointingHandCursor)
        btn_run.clicked.connect(self._refresh_loo_vd)
        header_row.addWidget(btn_run)
        layout.addLayout(header_row)

        # ── Summary ──
        self.loo_summary_label = QLabel("Select a sample and layer, then click Decompose")
        self.loo_summary_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt; "
            f"padding: 2px 8px; font-style: italic;")
        layout.addWidget(self.loo_summary_label)

        # ── Content: table (left) + charts (right) ──
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(3)
        main_splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {SemiconductorTheme.BORDER_DEFAULT}; }}")

        # LEFT: Impact ranking table
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(4, 4, 4, 4)
        table_layout.setSpacing(2)
        table_lbl = QLabel("<b>BUMP IMPACT RANKING</b>  (sorted by LOO impact)")
        table_lbl.setStyleSheet(f"color: {ACCENT}; font-size: 9pt; padding: 2px;")
        table_layout.addWidget(table_lbl)

        self.loo_table = QTableWidget()
        self.loo_table.setColumnCount(7)
        self.loo_table.setHorizontalHeaderLabels([
            "Rank", "Grid", "Value", "Δ Mean",
            "σ w/o", "Impact%", "Contrib%"
        ])
        self.loo_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.loo_table.setAlternatingRowColors(True)
        self.loo_table.setStyleSheet(f"""
            QTableWidget {{
                background: {SemiconductorTheme.BG_DARK};
                alternate-background-color: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                gridline-color: {SemiconductorTheme.BORDER_DEFAULT};
                font-size: 9pt;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QHeaderView::section {{
                background: {SemiconductorTheme.BG_PANEL};
                color: {ACCENT};
                font-weight: bold; font-size: 8pt; padding: 3px;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """)
        self.loo_table.horizontalHeader().setStretchLastSection(True)
        self.loo_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.loo_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.loo_table, 1)
        main_splitter.addWidget(table_container)

        # RIGHT: Charts stacked vertically
        chart_splitter = QSplitter(Qt.Vertical)
        chart_splitter.setHandleWidth(3)
        chart_splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {SemiconductorTheme.BORDER_DEFAULT}; }}")

        # Contribution heatmap (XY scatter)
        hm_container = QWidget()
        hm_layout = QVBoxLayout(hm_container)
        hm_layout.setContentsMargins(4, 4, 4, 4)
        hm_layout.setSpacing(2)
        hm_lbl = QLabel("<b>VARIANCE CONTRIBUTION MAP</b>  (XY spatial)")
        hm_lbl.setStyleSheet(f"color: {ACCENT}; font-size: 9pt; padding: 2px;")
        hm_layout.addWidget(hm_lbl)
        self.loo_heatmap_chart = EmbeddedChart(parent=hm_container, width=5, height=3.5)
        hm_layout.addWidget(self.loo_heatmap_chart, 1)
        chart_splitter.addWidget(hm_container)

        # What-if bar chart
        wf_container = QWidget()
        wf_layout = QVBoxLayout(wf_container)
        wf_layout.setContentsMargins(4, 4, 4, 4)
        wf_layout.setSpacing(2)
        wf_lbl = QLabel("<b>WHAT-IF σ REDUCTION</b>  (remove top outliers)")
        wf_lbl.setStyleSheet(f"color: {ACCENT}; font-size: 9pt; padding: 2px;")
        wf_layout.addWidget(wf_lbl)
        self.loo_whatif_chart = EmbeddedChart(parent=wf_container, width=5, height=2.5)
        wf_layout.addWidget(self.loo_whatif_chart, 1)
        chart_splitter.addWidget(wf_container)

        chart_splitter.setSizes([400, 250])
        main_splitter.addWidget(chart_splitter)
        main_splitter.setSizes([450, 550])

        layout.addWidget(main_splitter, 1)
        return container

    def _refresh_loo_vd(self):
        """Run LOO-VD analysis and update charts."""
        from inno3d.core.spatial_analysis import compute_loo_variance

        if not self.samples:
            self.loo_summary_label.setText("No samples loaded.")
            return

        idx = self._selected_sample_idx if self._selected_sample_idx is not None else 0
        if idx >= len(self.samples):
            idx = 0
        sample = self.samples[idx]

        if not sample.raw_rows:
            self.loo_summary_label.setText(f"No data in '{sample.name}'.")
            return

        # Metric mapping
        metric_map = {0: 'soh', 1: 'void_ratio', 2: 'bump_volume'}
        metric_key = metric_map.get(self.loo_metric_combo.currentIndex(), 'soh')
        metric_labels = {0: 'SOH (µm)', 1: 'Void Ratio', 2: 'Bump Vol (µm³)'}
        metric_label = metric_labels.get(self.loo_metric_combo.currentIndex(), 'SOH')

        # Run LOO
        try:
            loo_results = compute_loo_variance(sample.raw_rows, metric_key=metric_key)
        except Exception as e:
            self.loo_summary_label.setText(f"LOO error: {e}")
            return

        # Update layer combo
        layer_names = sorted(loo_results.keys(), key=_natural_sort_key)
        self.loo_layer_combo.blockSignals(True)
        current_text = self.loo_layer_combo.currentText()
        self.loo_layer_combo.clear()
        self.loo_layer_combo.addItem("— Select Layer —")
        for ln in layer_names:
            self.loo_layer_combo.addItem(ln)
        # Restore selection
        restore_idx = self.loo_layer_combo.findText(current_text)
        if restore_idx >= 0:
            self.loo_layer_combo.setCurrentIndex(restore_idx)
        self.loo_layer_combo.blockSignals(False)

        # Get selected layer
        sel_layer = self.loo_layer_combo.currentText()
        if sel_layer.startswith("—") or sel_layer not in loo_results:
            if layer_names:
                sel_layer = layer_names[0]
                self.loo_layer_combo.blockSignals(True)
                self.loo_layer_combo.setCurrentText(sel_layer)
                self.loo_layer_combo.blockSignals(False)
            else:
                self.loo_summary_label.setText("No layers with ≥3 bumps found.")
                return

        lr = loo_results[sel_layer]

        # ── Summary ──
        top_bump = lr.bumps[0] if lr.bumps else None
        top_str = (f"Top outlier: {top_bump.grid_label} "
                   f"(impact={top_bump.impact_pct:.1f}%)") if top_bump else ""
        self.loo_summary_label.setText(
            f"<b>{sample.name} / {sel_layer}</b> — "
            f"{lr.n_bumps} bumps | σ={lr.sigma_full:.4f} | "
            f"%Tol={lr.pct_tol:.1f}% | {top_str}")

        # ── Populate table ──
        self.loo_table.setRowCount(len(lr.bumps))
        for row_i, bump in enumerate(lr.bumps):
            items = [
                f"#{row_i+1}",
                bump.grid_label,
                f"{bump.value:.4f}",
                f"{bump.deviation:+.4f}",
                f"{bump.sigma_without:.4f}",
                f"{bump.impact_pct:.1f}%",
                f"{bump.contribution_pct:.1f}%"
            ]
            for col_i, txt in enumerate(items):
                item = QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                # Color code impact
                if col_i == 5:
                    val = bump.impact_pct
                    if val > 20:
                        item.setForeground(QBrush(QColor('#ff4444')))
                    elif val > 10:
                        item.setForeground(QBrush(QColor('#ffc107')))
                    elif val < 0:
                        item.setForeground(QBrush(QColor('#76ff03')))
                self.loo_table.setItem(row_i, col_i, item)

        # ── Plot contribution heatmap ──
        self._plot_loo_heatmap(lr, metric_label, sample.name)

        # ── Plot what-if chart ──
        self._plot_loo_whatif(lr, metric_label)

    def _plot_loo_heatmap(self, lr, metric_label, sample_name):
        """XY scatter colored by variance contribution %."""
        fig = self.loo_heatmap_chart.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0f1320')

        if not lr.bumps:
            fig.tight_layout()
            self.loo_heatmap_chart.draw()
            return

        cx = np.array([b.cx for b in lr.bumps])
        cy = np.array([b.cy for b in lr.bumps])
        contrib = np.array([b.contribution_pct for b in lr.bumps])
        impact = np.array([b.impact_pct for b in lr.bumps])

        # Use contribution for color
        sc = ax.scatter(cx, cy, c=contrib, cmap='YlOrRd', vmin=0,
                        vmax=max(np.max(contrib), 1.0),
                        s=80, edgecolors='#1a2040', linewidth=0.8, alpha=0.9, zorder=2)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Contribution %', fontsize=8, color='#b0b0b0')
        cbar.ax.tick_params(colors='#b0b0b0', labelsize=7)

        # Annotate top outliers with grid labels
        for b in lr.bumps[:5]:
            if b.impact_pct > 5:
                ax.annotate(f"{b.grid_label}\n{b.impact_pct:.0f}%",
                            (b.cx, b.cy), textcoords="offset points",
                            xytext=(8, 8), fontsize=7, color='#ff9100',
                            fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='#ff9100',
                                            lw=0.8))

        ax.set_title(f"{sample_name} / {lr.layer} — {metric_label}",
                     fontsize=9, color='#ff9100', pad=8)
        ax.set_xlabel("X (px)", fontsize=8, color='#b0b0b0')
        ax.set_ylabel("Y (px)", fontsize=8, color='#b0b0b0')
        ax.tick_params(colors='#b0b0b0', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2a3050')
        ax.spines['left'].set_color('#2a3050')
        ax.invert_yaxis()

        fig.tight_layout()
        self.loo_heatmap_chart.draw()

    def _plot_loo_whatif(self, lr, metric_label):
        """Bar chart: current σ vs what-if σ after removing top outliers."""
        fig = self.loo_whatif_chart.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0f1320')

        labels = ['Current σ', 'Remove Top-1', 'Remove Top-2', 'Remove Top-3']
        values = [lr.sigma_full, lr.whatif_1, lr.whatif_2, lr.whatif_3]

        # Filter out zeros for display
        valid = [(l, v) for l, v in zip(labels, values) if v > 0 or l == 'Current σ']
        labels = [v[0] for v in valid]
        values = [v[1] for v in valid]

        colors = ['#ff5252'] + ['#ff9100'] * (len(values) - 1)
        bars = ax.bar(range(len(labels)), values, color=colors,
                      edgecolor='#1a2040', linewidth=0.8, alpha=0.85)

        # Value + reduction % labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            label = f"{val:.4f}"
            if i > 0 and lr.sigma_full > 1e-12:
                reduction = (1 - val / lr.sigma_full) * 100
                label += f"\n(-{reduction:.0f}%)"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha='center', va='bottom', fontsize=8,
                    color='#e0e0e0', fontweight='bold')

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, color='#b0b0b0')
        ax.tick_params(axis='y', colors='#b0b0b0', labelsize=7)
        ax.set_ylabel(f"σ ({metric_label})", fontsize=8, color='#b0b0b0')
        ax.set_title(f"What-If σ Reduction — {lr.layer}",
                     fontsize=9, color='#ff9100', pad=8)

        ax.grid(axis='y', color='#1a2040', linewidth=0.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2a3050')
        ax.spines['left'].set_color('#2a3050')

        fig.tight_layout()
        self.loo_whatif_chart.draw()
