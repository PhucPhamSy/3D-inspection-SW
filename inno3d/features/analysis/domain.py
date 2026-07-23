# inno3d/features/analysis/domain.py
# -----------------------------------------------------------------------
# Analysis domain — extracted from inno3d/tabs/analysis.py (Phase 7)
#
# Pure logic + chart widget (no Qt tabs):
#   LayerStats, parse_measurement_csv, compute_layer_stats
#   EmbeddedChart (matplotlib FigureCanvas)
# -----------------------------------------------------------------------

"""
3D Analysis Tab — SOH (State of Height of Bump) Analytics Workbench

DB-first: load FOV runs / MES objects from inspection.db, aggregate SOH by
layer / FOV / chip. CSV folder import remains as lab fallback.

Advanced modes (Cross-σ, Per-Bump σ, 3D-SIM, LOO-VD) operate on the loaded
analysis set (each FOV run ≈ one SampleData unit).
"""
import csv
from dataclasses import dataclass
import re
from typing import Dict, List

import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# matplotlib for embedded charts
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from inno3d.core.soh_data import (
    SampleData,
)


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
        for bar, val in zip(bars, values, strict=False):
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
        for bar, val in zip(bars, values, strict=False):
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

