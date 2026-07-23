"""
3D Analysis Tab — SOH (State of Height of Bump) Analytics Workbench
────────────────────────────────────────────────────────────────────
Thin façade (Phase 7). All logic in inno3d/features/analysis/:
  domain.py   — pure data models, helpers
  tab_ui.py   — AnalysisTab QWidget

This file re-exports AnalysisTab so existing call sites
  from inno3d.tabs.analysis import AnalysisTab
continue to work without modification.
"""
# inno3d/tabs/analysis.py  ─── thin façade (Phase 7)

from inno3d.features.analysis.domain import (
    LayerStats,
    EmbeddedChart,
    _natural_sort_key,
    parse_measurement_csv,
    compute_layer_stats,
)
from inno3d.features.analysis.tab_ui import AnalysisTab

__all__ = [
    "AnalysisTab",
    "LayerStats",
    "EmbeddedChart",
    "_natural_sort_key",
    "parse_measurement_csv",
    "compute_layer_stats",
]
