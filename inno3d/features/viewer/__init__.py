# inno3d/features/viewer/__init__.py
# -----------------------------------------------------------------------
# features.viewer — MultiPlanarView mixin sub-modules (Phase 3 extracts)
#
# MRO order (must match MultiPlanarView base list in tabs/viewer.py):
#
#   MultiPlanarView(
#       VolumeIOMixin,       # drag-drop / load TIFF / batch / align tools
#       AlignSidebarMixin,   # right VIEW TOOLS panel builders
#       SliceViewMixin,      # 2×2 MPR grid layout + fullscreen toggle
#       StatsPanelMixin,     # object-stats panel, MES/B2B tables, FAR, export
#       WindowLevelMixin,    # W/L slider + histogram widget
#       ClipBoxMixin,        # 3D clip-box interactive UI
#       Volume3dMixin,       # 3D volume render (VTK ray-cast, TF, lighting)
#       MprInputMixin,       # mouse/wheel event observers for 2D panes
#       MprRenderMixin,      # render_slice, reset_view, ruler overlay
#       MprNavMixin,         # slice navigation, oblique, cursor tracking
#       CrosshairMixin,      # crosshair draw + updatePoint sync
#       QWidget,
#   )
#
# Rules:
#   - Every name used inside a mixin must be imported in THAT module.
#   - Do not import heavy tabs (AI, Teaching) here — only viewer stack.
#   - After any extract: run tools/p0b_import_scan.py + pytest tests/unit.
# -----------------------------------------------------------------------
"""Viewer feature mixins — see MRO comment above for correct base order."""

from .volume_io import VolumeIOMixin
from .align_sidebar import AlignSidebarMixin
from .slice_view import SliceViewMixin
from .stats_panel import StatsPanelMixin
from .window_level import WindowLevelMixin
from .clip_box import ClipBoxMixin
from .volume_3d import Volume3dMixin
from .mpr_input import MprInputMixin
from .mpr_render import MprRenderMixin
from .mpr_nav import MprNavMixin
from .crosshair import CrosshairMixin
from .shared_widgets import (
    NumericSortItem,
    CategorySortItem,
    StatsRowDelegate,
    FrozenTableWidget,
    AlignVolumeThread,
    ManualAlignThread,
    ExcelFilterMenu,
    ExcelFilterHeader,
)

__all__ = [
    # Mixins (MRO order)
    "VolumeIOMixin",
    "AlignSidebarMixin",
    "SliceViewMixin",
    "StatsPanelMixin",
    "WindowLevelMixin",
    "ClipBoxMixin",
    "Volume3dMixin",
    "MprInputMixin",
    "MprRenderMixin",
    "MprNavMixin",
    "CrosshairMixin",
    # Shared widgets
    "NumericSortItem",
    "CategorySortItem",
    "StatsRowDelegate",
    "FrozenTableWidget",
    "AlignVolumeThread",
    "ManualAlignThread",
    "ExcelFilterMenu",
    "ExcelFilterHeader",
]
