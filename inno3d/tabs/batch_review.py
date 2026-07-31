"""
Batch Production Review Tab
───────────────────────────
Thin façade (Phase 7). All logic in inno3d/features/batch_review/:
  widgets.py — helper widgets (WaferMapWidget, MprReplayPanel, …)
  tab.py     — BatchReviewTab QWidget

This file re-exports BatchReviewTab so existing call sites
  from inno3d.tabs.batch_review import BatchReviewTab
continue to work without modification.
"""
# inno3d/tabs/batch_review.py  ─── thin façade (Phase 7)

from inno3d.features.batch_review.map_stage import MapStage
from inno3d.features.batch_review.widgets import (
    WaferMapWidget,
    FovMapWidget,
    SliceViewLabel,
    VolumeLoadThread,
    MprReplayPanel,
    MiniBarChart,
    MiniHistChart,
    build_wafer_context_from_db,
)
from inno3d.features.batch_review.tab import BatchReviewTab

__all__ = [
    "BatchReviewTab",
    "MapStage",
    "WaferMapWidget",
    "FovMapWidget",
    "SliceViewLabel",
    "VolumeLoadThread",
    "MprReplayPanel",
    "MiniBarChart",
    "MiniHistChart",
    "build_wafer_context_from_db",
]
