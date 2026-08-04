import os
import glob
import re
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support
from PyQt5.QtCore import QEvent, QPoint, QRect, QSize, QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import (
    QColor, QFont, QFontInfo, QFontMetrics, QIcon, QImage, QKeySequence,
    QPainter, QPainterPath, QPen, QPixmap, QPolygon, QBrush,
)
from PyQt5.QtWidgets import *
from scipy import ndimage
from skimage import io, measure
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from inno3d.core.styles import SemiconductorTheme, get_class_colors
from inno3d.core.tf_widgets import (
    ColorGradientWidget,
    HistogramWLWidget,
    TransferFunctionWidget,
    OPACITY_SHAPES,
    OPACITY_SHAPE_TIPS,
    make_opacity_shape_icon,
    make_color_map_icon,
    make_icon_toolbutton,
    sample_opacity_curve,
)
from inno3d.core.view_support import (
    Dragonfly3DInteractorStyle,
    LoadVolumeThread,
    SliceBarWheelFilter,
    SliceControlOverlay,
    StepOneSliceSlider,
    apply_dragonfly_volume_zoom,
    push_camera_outside_aabb,
    volume_world_aabb,
)
from inno3d.core.wafer_context import WaferContext
from inno3d.widgets.context_map_panel import ContextMapPanel
from inno3d.features.viewer.mpr_nav import MprNavMixin
from inno3d.features.viewer.crosshair import CrosshairMixin
from inno3d.features.viewer.mpr_input import MprInputMixin
from inno3d.features.viewer.mpr_render import MprRenderMixin
from inno3d.features.viewer.volume_3d import Volume3dMixin
from inno3d.features.viewer.window_level import WindowLevelMixin
from inno3d.features.viewer.clip_box import ClipBoxMixin
from inno3d.features.viewer.stats_panel import StatsPanelMixin
from inno3d.features.viewer.slice_view import SliceViewMixin
from inno3d.features.viewer.volume_io import VolumeIOMixin
from inno3d.features.viewer.align_sidebar import AlignSidebarMixin
from inno3d.features.viewer.shared_widgets import (
    _ui_icon, _natural_sort_key,
    NumericSortItem, CategorySortItem, _numeric_item, _category_item,
    StatsRowDelegate, FrozenTableWidget,
    AlignVolumeThread, ManualAlignThread,
    ExcelFilterMenu, ExcelFilterHeader,
)



# ── Dragonfly-style view grid layout presets ─────────────────────────────────
# placement: orientation -> (row, col, rowspan, colspan)
# view_3d is the volume pane; icons mark it with a cube glyph.
VIEW_LAYOUT_PRESETS = [
    {
        "id": "mpr_2x2",
        "label": "2×2 MPR + 3D",
        "tooltip": "Standard 2×2: XY | YZ / XZ | 3D Volume",
        "rows": 2,
        "cols": 2,
        "placements": {
            "axial": (0, 0, 1, 1),
            "sagittal": (0, 1, 1, 1),
            "coronal": (1, 0, 1, 1),
            "view_3d": (1, 1, 1, 1),
        },
    },
    {
        "id": "single_3d",
        "label": "3D only",
        "tooltip": "Single full-screen 3D volume",
        "rows": 1,
        "cols": 1,
        "placements": {
            "view_3d": (0, 0, 1, 1),
        },
    },
    {
        "id": "single_axial",
        "label": "XY only",
        "tooltip": "Single axial (XY) slice view",
        "rows": 1,
        "cols": 1,
        "placements": {
            "axial": (0, 0, 1, 1),
        },
    },
    {
        "id": "h_2",
        "label": "2 horizontal",
        "tooltip": "Side-by-side: Axial | 3D Volume",
        "rows": 1,
        "cols": 2,
        "placements": {
            "axial": (0, 0, 1, 1),
            "view_3d": (0, 1, 1, 1),
        },
    },
    {
        "id": "v_2",
        "label": "2 vertical",
        "tooltip": "Stacked: Axial / 3D Volume",
        "rows": 2,
        "cols": 1,
        "placements": {
            "axial": (0, 0, 1, 1),
            "view_3d": (1, 0, 1, 1),
        },
    },
    {
        "id": "h_3mpr",
        "label": "3 MPR row",
        "tooltip": "Three MPR views in a row (no 3D)",
        "rows": 1,
        "cols": 3,
        "placements": {
            "axial": (0, 0, 1, 1),
            "sagittal": (0, 1, 1, 1),
            "coronal": (0, 2, 1, 1),
        },
    },
    {
        "id": "v_3mpr_3d",
        "label": "3 MPR | 3D",
        "tooltip": "Left: stacked XY/YZ/XZ · Right: large 3D volume",
        "rows": 3,
        "cols": 2,
        "placements": {
            "axial": (0, 0, 1, 1),
            "sagittal": (1, 0, 1, 1),
            "coronal": (2, 0, 1, 1),
            "view_3d": (0, 1, 3, 1),
        },
    },
    {
        "id": "v_3d_3mpr",
        "label": "3D | 3 MPR",
        "tooltip": "Left: large 3D volume · Right: stacked XY/YZ/XZ",
        "rows": 3,
        "cols": 2,
        "placements": {
            "view_3d": (0, 0, 3, 1),
            "axial": (0, 1, 1, 1),
            "sagittal": (1, 1, 1, 1),
            "coronal": (2, 1, 1, 1),
        },
    },
    {
        "id": "h_3mpr_3d",
        "label": "3 MPR / 3D",
        "tooltip": "Top: XY | YZ | XZ · Bottom: wide 3D volume",
        "rows": 2,
        "cols": 3,
        "placements": {
            "axial": (0, 0, 1, 1),
            "sagittal": (0, 1, 1, 1),
            "coronal": (0, 2, 1, 1),
            "view_3d": (1, 0, 1, 3),
        },
    },
    {
        "id": "h_3d_3mpr",
        "label": "3D / 3 MPR",
        "tooltip": "Top: wide 3D volume · Bottom: XY | YZ | XZ",
        "rows": 2,
        "cols": 3,
        "placements": {
            "view_3d": (0, 0, 1, 3),
            "axial": (1, 0, 1, 1),
            "sagittal": (1, 1, 1, 1),
            "coronal": (1, 2, 1, 1),
        },
    },
]


def _draw_mini_cube(painter, cx, cy, s):
    """Isometric-ish cube glyph — marks the 3D volume cell (Dragonfly-style)."""
    s = max(5, float(s))
    dx = s * 0.45
    dy = s * 0.26
    h = s * 0.55
    top = QPolygon([
        QPoint(int(cx), int(cy - h * 0.55)),
        QPoint(int(cx + dx), int(cy - h * 0.55 + dy)),
        QPoint(int(cx), int(cy - h * 0.55 + 2 * dy)),
        QPoint(int(cx - dx), int(cy - h * 0.55 + dy)),
    ])
    left = QPolygon([
        top[3],
        top[2],
        QPoint(int(cx), int(cy + h * 0.45)),
        QPoint(int(cx - dx), int(cy + h * 0.45 - dy)),
    ])
    right = QPolygon([
        top[1],
        top[2],
        QPoint(int(cx), int(cy + h * 0.45)),
        QPoint(int(cx + dx), int(cy + h * 0.45 - dy)),
    ])
    painter.setPen(QPen(QColor(20, 30, 40, 210), 0.9))
    painter.setBrush(QColor(140, 220, 240, 230))
    painter.drawPolygon(top)
    painter.setBrush(QColor(50, 140, 170, 220))
    painter.drawPolygon(left)
    painter.setBrush(QColor(90, 185, 210, 230))
    painter.drawPolygon(right)


def _make_view_layout_icon(preset, size=36, selected=False):
    """Paint a Dragonfly-like layout thumbnail; cube marks the 3D volume cell."""
    # ARGB32_Premultiplied is reliable across DPI/styles (avoids blank/white icons)
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)

    rows = max(1, int(preset["rows"]))
    cols = max(1, int(preset["cols"]))
    pad = 2.5
    gap = 1.6
    avail = size - 2 * pad
    cw = (avail - gap * (cols - 1)) / cols
    ch = (avail - gap * (rows - 1)) / rows

    # Soft card background
    bg = QColor(SemiconductorTheme.BG_LIGHT)
    if selected:
        bg = QColor(SemiconductorTheme.BG_PANEL)
    p.setPen(Qt.NoPen)
    p.setBrush(bg)
    p.drawRoundedRect(0, 0, size, size, 5, 5)

    # Occupancy map so spanning cells draw once
    covered = [[False] * cols for _ in range(rows)]
    cells = []
    for key, (r, c, rs, cs) in preset["placements"].items():
        cells.append((key, r, c, rs, cs, key == "view_3d"))

    # Draw back-to-front by area (larger first)
    cells.sort(key=lambda t: -(t[3] * t[4]))

    for key, r, c, rs, cs, is_3d in cells:
        if r >= rows or c >= cols:
            continue
        x = pad + c * (cw + gap)
        y = pad + r * (ch + gap)
        w = cs * cw + max(0, cs - 1) * gap
        h = rs * ch + max(0, rs - 1) * gap

        if is_3d:
            fill = QColor(SemiconductorTheme.PRIMARY_DEFAULT)
            fill.setAlpha(120 if not selected else 160)
            border = QColor(SemiconductorTheme.PRIMARY_HOVER)
            border.setAlpha(240)
        else:
            fill = QColor(150, 170, 190, 110 if not selected else 150)
            border = QColor(190, 205, 220, 200)

        p.setPen(QPen(border, 1.0))
        p.setBrush(fill)
        radius = 2.0 if min(w, h) > 8 else 1.0
        p.drawRoundedRect(int(x), int(y), max(2, int(w)), max(2, int(h)), radius, radius)

        if is_3d:
            cube_s = min(w, h) * 0.48
            _draw_mini_cube(p, x + w * 0.5, y + h * 0.52, cube_s)

        for rr in range(r, min(rows, r + rs)):
            for cc in range(c, min(cols, c + cs)):
                covered[rr][cc] = True

    # Empty slots (if any) as faint outlines
    for rr in range(rows):
        for cc in range(cols):
            if covered[rr][cc]:
                continue
            x = pad + cc * (cw + gap)
            y = pad + rr * (ch + gap)
            p.setPen(QPen(QColor(100, 115, 130, 80), 1.0, Qt.DotLine))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(int(x), int(y), max(2, int(cw)), max(2, int(ch)), 2, 2)

    # Selection ring
    if selected:
        p.setPen(QPen(QColor(SemiconductorTheme.PRIMARY_DEFAULT), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(1, 1, size - 2, size - 2, 5, 5)

    p.end()
    return QIcon(QPixmap.fromImage(img))


class MultiPlanarView(VolumeIOMixin, AlignSidebarMixin, SliceViewMixin, StatsPanelMixin, WindowLevelMixin, ClipBoxMixin, Volume3dMixin, MprInputMixin, MprRenderMixin, MprNavMixin, CrosshairMixin, QWidget):
    """Multi-Planar View with Crosshair Tracking.

    Navigation (zoom/pan/scroll) behaviour lives in MprNavMixin.
    Crosshair hit-test / drag / updatePoint lives in CrosshairMixin.
    Mouse/keyboard input dispatch lives in MprInputMixin.
    Slice rendering + ruler overlay lives in MprRenderMixin.
    This class retains the 3D volume, UI setup, and all methods not yet extracted.
    """

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.volume_data = None
        self.segmentation_data = None
        self.class1_data = None  # Added: SEG class 1 array
        self.class2_data = None  # Added: SEG class 2 array
        
        # Backup for alignment reset
        self.original_volume_data = None
        self.original_class1_data = None
        self.original_class2_data = None
        self.original_labeled_class1_data = None
        self.original_labeled_class2_data = None
       
        self.spacing = [1.0, 1.0, 1.0]
        self.custom_spacing = None
        self.current_slices = {'sagittal': 0, 'coronal': 0, 'axial': 0}
        
        # Advanced 3D Controls
        self.auto_rotate_timer = QTimer(self)
        self.auto_rotate_timer.timeout.connect(self.auto_rotate_step)
        self.adv_mpr_actors = {'x': None, 'y': None, 'z': None}
        self.adv_clip_planes = {'x': None, 'y': None, 'z': None}
        self.bbox_actor = None

        # ── Dragonfly-style clip box (2×2 right sidebar — linked MPR + 3D) ──
        # Independent from fullscreen "Advanced MPR & Clipping" axis planes.
        # Bounds are percentages 0..1000 of world extent per axis (min, max).
        self._df_clip_enabled = False
        self._df_clip_bounds = {
            'x': [0, 1000],
            'y': [0, 1000],
            'z': [0, 1000],
        }
        self._df_clip_show_borders = True
        self._df_clip_show_grid = True
        self._df_clip_show_axes = False
        self._df_clip_keep_when_hidden = True
        self._df_clip_display_grid_on_object = True
        self._df_clip_grid_size = 50.0  # world units (µm when spacing is µm)
        self._df_clip_3d_actors = []
        self._df_clip_mpr_actors = {'axial': [], 'coronal': [], 'sagittal': []}
        # Interactive handles (MPR drag like Dragonfly)
        self._df_clip_hover = None   # dict hit or None
        self._df_clip_drag = None    # active drag state or None
        # 3D viewport face drag / hover
        self._df_clip_3d_hover = None  # {'axis': 'x'|'y'|'z', 'side': 'lo'|'hi'}
        self._df_clip_3d_drag = None   # active 3D face drag state
        
        self.window_level = {'axial': None, 'coronal': None, 'sagittal': None}
        self.camera_state = {'axial': None, 'coronal': None, 'sagittal': None}
        
        self.seg_colors = {
            128: [1.0, 1.0, 0.0],  # Default Yellow for Class 1
            255: [1.0, 0.0, 0.0]   # Default Red for Class 2
        }
        
        self.volume_color = [0.95, 0.95, 0.95] # Default Silver/Grayscale (Dragonfly style)
        self.volume_actor = None
        self.load_thread = None
        
        # Slices widgets and overlays
        self.axial_widget = None
        self.coronal_widget = None
        self.sagittal_widget = None
        self.axial_overlay_group = None
        self.coronal_overlay_group = None
        self.sagittal_overlay_group = None
        self.axial_renderer = None
        self.coronal_renderer = None
        self.sagittal_renderer = None
        self.axial_pixel_label = None
        self.coronal_pixel_label = None
        self.sagittal_pixel_label = None
        
        # Crosshair
        self.crosshair_enabled = False
        self.crosshair_position = [0, 0, 0]  # X, Y, Z in volume coordinates
        self.crosshair_actors = {'axial': [], 'coronal': [], 'sagittal': []}
        self.crosshair_color = [1.0, 1.0, 0.0]  # Default yellow
        self.ruler_color = [0.2, 0.9, 0.2]      # Default green
        self.reverse_z = False  # Z direction toggle
        # Dragonfly-style hover / grab:
        #   _crosshair_hover = (orientation, 'center'|'h'|'v') or None
        #   _crosshair_drag_mode = 'center'|'h'|'v' while dragging
        self._crosshair_hover = None
        self._crosshair_drag_mode = None
        self._is_dragging_crosshair = False
        self._crosshair_click_placed = False  # click-to-place vs handle drag
        # P0/P1 nav: throttle linked re-render while dragging crosshair
        self._ch_sync_timer = QTimer(self)
        self._ch_sync_timer.setSingleShot(True)
        self._ch_sync_timer.setInterval(20)  # ~50 Hz full sync max
        self._ch_sync_timer.timeout.connect(self._flush_crosshair_sync)
        self._ch_sync_pending = False
        # MPR pan: (orientation, last QPoint) while middle / Shift+LMB
        self._mpr_pan = None
        # Link zoom/pan across the three MPR panes (optional)
        self._mpr_link_nav = False
        
        # Object measurement & highlight
        self.labeled_class1_data = None  # labeled array from skimage.measure.label
        self.labeled_class2_data = None
        self.object_stats = []  # list of dicts with measurement data
        self.selected_highlight_objects = []  # list of (class_num, obj_id) tuples
        self.highlight_color = [0.0, 1.0, 1.0]  # cyan highlight
        self._mes_3d_highlight_actors = []  # surface actors for MES pick on 3D volume
        self._mes_3d_hl_cache_key = None  # skip rebuild when same labels/bbox
        self.projection_mode = "perspective"
        
        self.rotate_mode = False
        self.last_mouse_pos = None
        
        # Oblique MPR rotation
        self.oblique_angles = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
        self._oblique_hover_handle = None
        self._oblique_dragging = None
        self._manual_align_active = False
        self.oblique_handles_enabled = False  # Dragonfly-style rotation arrows toggle
        
        self.init_ui()
        
        # ESC shortcut to exit in-grid fullscreen
        self._esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self._esc_shortcut.activated.connect(self.exit_fullscreen)
        # P1: view navigation shortcuts (active MPR pane)
        self._fit_shortcut = QShortcut(QKeySequence("Ctrl+Shift+0"), self)
        self._fit_shortcut.activated.connect(self._shortcut_fit_active_pane)
        self._one_to_one_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        self._one_to_one_shortcut.activated.connect(self._shortcut_one_to_one_active_pane)
        self._reset_view_shortcut = QShortcut(QKeySequence(Qt.Key_R), self)
        self._reset_view_shortcut.activated.connect(self._shortcut_reset_active_pane)
        self._reset_view_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        
    def init_ui(self):
        layout = QVBoxLayout()
        # Tight outer margins so the canvas reads as one linked viewport
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        self.info_label = QLabel("No data loaded")
        self.info_label.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-style: italic; font-size: 8pt;")
        # Note: top_widget buttons moved to MainWindow sidebar
        
        # --- Multi-pane Grid View (Dragonfly-style linked panes) ---
        # Technique: 1px grid spacing + divider-colored background creates
        # a continuous hairline cross between black panes (no card gaps).
        self._grid_divider_color = "#1a2430"
        grid_widget = QWidget()
        grid_widget.setObjectName("PlanesGrid")
        grid_widget.setStyleSheet(
            f"#PlanesGrid {{"
            f"  background-color: {self._grid_divider_color};"
            f"  border: 1px solid {self._grid_divider_color};"
            f"}}"
        )
        grid_layout = QGridLayout()
        grid_layout.setSpacing(1)  # hairline divider thickness
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)
        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 1)
        
        axial_widget = self.create_slice_view("XY", "axial")
        grid_layout.addWidget(axial_widget, 0, 0, 1, 1)
        
        sagittal_widget = self.create_slice_view("YZ", "sagittal") 
        grid_layout.addWidget(sagittal_widget, 0, 1, 1, 1)
        
        coronal_widget = self.create_slice_view("XZ", "coronal")
        grid_layout.addWidget(coronal_widget, 1, 0, 1, 1)
        
        view_3d_widget = self.create_3d_view()
        grid_layout.addWidget(view_3d_widget, 1, 1, 1, 1)
        
        grid_widget.setLayout(grid_layout)
        self._grid_widget = grid_widget
        
        # Store references for layout switching + in-grid fullscreen
        self._grid_layout = grid_layout
        self._grid_widgets = {
            'axial': axial_widget,
            'sagittal': sagittal_widget,
            'coronal': coronal_widget,
            'view_3d': view_3d_widget,
        }
        # (widget, row, col, rowspan, colspan)
        self._grid_cells = {
            'axial':    (axial_widget,    0, 0, 1, 1),
            'sagittal': (sagittal_widget, 0, 1, 1, 1),
            'coronal':  (coronal_widget,  1, 0, 1, 1),
            'view_3d':  (view_3d_widget,  1, 1, 1, 1),
        }
        self._layout_visible_keys = set(self._grid_cells.keys())
        self._grid_nrows = 2
        self._grid_ncols = 2
        self._current_layout_id = "mpr_2x2"
        self._fullscreen_view = None  # Track which view is maximized
        self._active_grid_pane = None
        # Default focus ring on XY (axial) — subtle Dragonfly-style active pane
        self._set_active_grid_pane('axial')

        # Left vertical layout picker — hidden until SYSTEM → LAYOUT toggle
        self.layout_sidebar = self._create_layout_sidebar()
        self.layout_sidebar.setVisible(False)

        # Online CONTEXT: Wafer + Chip (FOV) — left of MPR, shown only in Online mode
        self.context_map_panel = ContextMapPanel(self)
        self.context_map_panel.setVisible(False)
        self.context_map_panel.chip_selected.connect(self._on_context_chip_selected)
        self.context_map_panel.fov_selected.connect(self._on_context_fov_selected)
        self.context_map_panel.selection_changed.connect(self._on_context_selection_changed)
        self._wafer_context = None  # type: Optional[WaferContext]
        
        # Wrap grid inside left side of a splitter
        self.stats_panel = self.create_object_stats_panel()
        self.stats_panel.setVisible(False) # Hidden by default, shown in Online mode
        
        # Right tools host: full Window Leveling sidebar OR thin reopen rail
        # Online mode auto-collapses to rail so MPR + CONTEXT + Statistics get space.
        self._online_viewer_mode = False
        self._align_tools_expanded = True
        self._align_tools_user_expanded = False  # session pref while Online is ON
        self.align_tools_host = self._create_align_tools_host()
        self.align_tools_host.setVisible(False)  # until volume is loaded

        # Body: [layout icons | CONTEXT wafer/chip | MPR+tools+stats]
        # MPR grid is untouched — CONTEXT sits to its left when Online is ON.
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self.layout_sidebar, 0)
        body_layout.addWidget(self.context_map_panel, 0)
        
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setStyleSheet(f"QSplitter::handle {{ background: {SemiconductorTheme.BORDER_DEFAULT}; }}")
        main_splitter.addWidget(grid_widget)
        main_splitter.addWidget(self.align_tools_host)
        main_splitter.addWidget(self.stats_panel)
        main_splitter.setSizes([1400, 240, 300])
        self._main_splitter = main_splitter
        body_layout.addWidget(main_splitter, 1)
        
        layout.addWidget(body, 1)  # Stretch factor 1
        self.setLayout(layout)
    
    def toggle_layout_sidebar(self, checked=False):
        """Show/hide the left Dragonfly layout-preset strip (SYSTEM toggle)."""
        if not hasattr(self, "layout_sidebar") or self.layout_sidebar is None:
            return
        self.layout_sidebar.setVisible(bool(checked))
        # Reflow VTK overlays after width change — do not reset cameras
        QTimer.singleShot(40, self._reposition_visible_overlays)

    def _reposition_visible_overlays(self):
        """Reposition / re-show slice/3D overlay tool strips (C1, C2, FS, Reset).

        Always visits every known overlay — not only layout-visible keys — so
        Online progress dialogs cannot leave strips permanently hidden.
        """
        visible = getattr(self, "_layout_visible_keys", None)
        if not visible:
            visible = set(getattr(self, "_grid_cells", {}).keys()) or {
                "axial", "coronal", "sagittal", "view_3d"
            }
        # Always include all standard panes even if layout set is partial
        keys = set(visible) | {"axial", "coronal", "sagittal", "view_3d"}
        for key in keys:
            if key == "view_3d":
                overlay = getattr(self, "view_3d_overlay_group", None)
                vtk_w = getattr(self, "view_3d_widget", None)
            else:
                overlay = getattr(self, f"{key}_overlay_group", None)
                vtk_w = getattr(self, f"{key}_widget", None)
            if overlay is not None:
                try:
                    overlay.update_position()
                except Exception:
                    pass
            should_render = key in visible
            if (
                should_render
                and key == "view_3d"
                and hasattr(self, "is_3d_volume_render_enabled")
                and not self.is_3d_volume_render_enabled()
            ):
                # 3D OFF placeholder is static; skip expensive render churn while
                # users resize splitters/layout in Online mode.
                should_render = False
            if should_render and vtk_w is not None:
                try:
                    rw = vtk_w.GetRenderWindow()
                    if rw is not None:
                        rw.Render()
                except Exception:
                    pass

    def is_layout_sidebar_visible(self):
        return bool(
            hasattr(self, "layout_sidebar")
            and self.layout_sidebar is not None
            and self.layout_sidebar.isVisible()
        )

    # ── Online CONTEXT (Wafer → Chip → FOV) ─────────────────────────────
    def set_online_context_visible(self, visible: bool):
        """Show/hide Wafer+Chip CONTEXT panel (Online mode only). Does not touch MPR."""
        panel = getattr(self, "context_map_panel", None)
        if panel is None:
            return
        show = bool(visible)
        panel.setVisible(show)
        if show and panel.context() is None:
            # New wafer: geometry only (Pending) — no fake OK/NG before inspect
            self.load_wafer_context_empty()
        QTimer.singleShot(40, self._reposition_visible_overlays)

    def is_online_context_visible(self) -> bool:
        panel = getattr(self, "context_map_panel", None)
        return bool(panel is not None and panel.isVisible())

    def load_wafer_context_empty(self):
        """Load uninspected wafer map (circle of Pending dies)."""
        if not hasattr(self, "context_map_panel") or self.context_map_panel is None:
            return None
        if hasattr(self.context_map_panel, "load_empty"):
            ctx = self.context_map_panel.load_empty()
        else:
            ctx = self.context_map_panel.load_demo()
        self._wafer_context = ctx
        return ctx

    def load_wafer_context_demo(self):
        """Load synthetic OK/NG map (email figure) — mock/debug only."""
        if not hasattr(self, "context_map_panel") or self.context_map_panel is None:
            return None
        ctx = self.context_map_panel.load_demo()
        self._wafer_context = ctx
        return ctx

    def load_wafer_context(self, data):
        """Load Wafer/Chip/FOV map from host payload (dict) or WaferContext."""
        if not hasattr(self, "context_map_panel") or self.context_map_panel is None:
            return None
        if isinstance(data, WaferContext):
            self.context_map_panel.set_context(data)
            self._wafer_context = data
            return data
        if isinstance(data, dict):
            ctx = self.context_map_panel.load_from_dict(data)
            self._wafer_context = ctx
            return ctx
        return None

    def get_wafer_context(self):
        """Return current WaferContext (selection + map), or None."""
        panel = getattr(self, "context_map_panel", None)
        if panel is not None and panel.context() is not None:
            return panel.context()
        return getattr(self, "_wafer_context", None)

    def get_active_fov_results_dir(self, root=None) -> str:
        """Path where inspection results for the active FOV should be written.

        Host mass-production layout::

            .../ChipLocation/FOVLocation/Results
        """
        ctx = self.get_wafer_context()
        if ctx is None:
            return ""
        return ctx.results_dir_for_selection(root)

    def apply_host_volume_path(self, path: str):
        """Parse host Input.tiff path and select Chip + FOV on CONTEXT maps."""
        from inno3d.core.wafer_context import parse_host_volume_path
        info = parse_host_volume_path(path)
        panel = getattr(self, "context_map_panel", None)
        if panel is not None and hasattr(panel, "apply_host_path_info"):
            ctx = panel.apply_host_path_info(info)
            self._wafer_context = ctx
            return info
        ctx = self.get_wafer_context()
        if ctx is None:
            self.load_wafer_context_demo()
            ctx = self.get_wafer_context()
        if ctx is not None:
            ctx.apply_host_path(info)
            self.load_wafer_context(ctx)
        return info

    def _on_context_chip_selected(self, col: int, row: int):
        """Chip die selected on wafer map — volume remains FOV of that chip's active point."""
        ctx = self.get_wafer_context()
        if ctx is None:
            return
        self._wafer_context = ctx
        # Future: request/load FOV volume for (col,row, selected_fov) from CT PC / local cache

    def _on_context_fov_selected(self, index: int):
        """FOV point P1..P9 selected — this is the volume identity in MPR/3D."""
        ctx = self.get_wafer_context()
        if ctx is None:
            return
        self._wafer_context = ctx
        # Future: swap loaded volume to this FOV's CT path when host provides volume_path

    def _on_context_selection_changed(self):
        """Keep internal ref in sync after any CONTEXT selection change."""
        panel = getattr(self, "context_map_panel", None)
        if panel is not None:
            self._wafer_context = panel.context()

    def _create_layout_sidebar(self):
        """Left vertical strip: Dragonfly-style view layout presets.

        Icons are mini grid thumbnails; cells with a cube glyph are the
        3D volume pane so users can see where volume rendering sits.
        Hidden by default — toggled from SYSTEM → LAYOUT in the app sidebar.
        """
        panel = QWidget()
        panel.setObjectName("LayoutSidebar")
        panel.setFixedWidth(52)
        panel.setStyleSheet(f"""
            QWidget#LayoutSidebar {{
                background-color: {SemiconductorTheme.BG_MEDIUM};
                border-right: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QPushButton#LayoutPresetBtn {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 2px;
                margin: 0px;
            }}
            QPushButton#LayoutPresetBtn:hover {{
                background-color: {SemiconductorTheme.BG_LIGHT};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QPushButton#LayoutPresetBtn:checked {{
                background-color: {SemiconductorTheme.BG_PANEL};
                border: 1px solid {SemiconductorTheme.PRIMARY_DEFAULT};
            }}
            QScrollArea#LayoutSidebarScroll {{
                background: transparent;
                border: none;
            }}
        """)

        root = QVBoxLayout(panel)
        root.setContentsMargins(4, 8, 4, 8)
        root.setSpacing(4)

        title = QLabel("LAYOUT")
        title.setAlignment(Qt.AlignHCenter)
        title.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_DISABLED}; font-weight: 700;"
            f" font-size: 6.5pt; letter-spacing: 1px;"
        )
        title.setToolTip(
            "View grid layout (Dragonfly-style).\n"
            "Cube icon = 3D volume pane; plain cells = MPR slices."
        )
        root.addWidget(title)

        legend = QLabel("cube\n=3D")
        legend.setAlignment(Qt.AlignHCenter)
        legend.setStyleSheet(
            f"color: {SemiconductorTheme.PRIMARY_DEFAULT}; font-size: 6.5pt; font-weight: 600;"
        )
        legend.setToolTip("Icon cells with a cube glyph = 3D volume pane.\nPlain cells = MPR slices (XY / YZ / XZ).")
        root.addWidget(legend)

        scroll = QScrollArea()
        scroll.setObjectName("LayoutSidebarScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 2, 0, 2)
        inner_layout.setSpacing(4)
        inner_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        self._layout_btn_group = QButtonGroup(self)
        self._layout_btn_group.setExclusive(True)
        self._layout_preset_btns = {}

        icon_size = 36
        # Strong local stylesheet so global QPushButton padding/min-height
        # cannot collapse/bleach these thumbnail icons.
        layout_btn_qss = f"""
            QPushButton#LayoutPresetBtn {{
                background-color: {SemiconductorTheme.BG_LIGHT};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 6px;
                padding: 0px;
                margin: 0px;
                min-height: 0px;
                min-width: 0px;
                max-height: {icon_size + 8}px;
                max-width: {icon_size + 8}px;
            }}
            QPushButton#LayoutPresetBtn:hover {{
                background-color: {SemiconductorTheme.BG_PANEL};
                border: 1px solid {SemiconductorTheme.PRIMARY_DEFAULT};
            }}
            QPushButton#LayoutPresetBtn:checked {{
                background-color: {SemiconductorTheme.BG_PANEL};
                border: 1px solid {SemiconductorTheme.PRIMARY_DEFAULT};
            }}
        """
        for preset in VIEW_LAYOUT_PRESETS:
            btn = QPushButton()
            btn.setObjectName("LayoutPresetBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedSize(icon_size + 8, icon_size + 8)
            btn.setIconSize(QSize(icon_size, icon_size))
            # Multi-color grid thumbnails — never monochrome-tint via MainWindow
            btn.setProperty("preserve_icon_color", True)
            btn.setProperty("theme_icon", False)
            btn.setStyleSheet(layout_btn_qss)
            btn.setIcon(_make_view_layout_icon(preset, size=icon_size, selected=False))
            tip = (
                f"{preset['label']}\n{preset['tooltip']}\n\n"
                "Cube glyph = 3D volume · plain cells = MPR (XY/YZ/XZ)"
            )
            btn.setToolTip(tip)
            btn.setProperty("layout_id", preset["id"])
            lid = preset["id"]
            btn.clicked.connect(lambda checked=False, i=lid: self._on_layout_preset_clicked(i))
            self._layout_btn_group.addButton(btn)
            self._layout_preset_btns[preset["id"]] = btn
            inner_layout.addWidget(btn, 0, Qt.AlignHCenter)

        # Default: classic 2×2
        default_btn = self._layout_preset_btns.get("mpr_2x2")
        if default_btn is not None:
            default_btn.setChecked(True)
            default_btn.setIcon(
                _make_view_layout_icon(VIEW_LAYOUT_PRESETS[0], size=icon_size, selected=True)
            )

        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)
        return panel

    def _on_layout_preset_clicked(self, layout_id):
        """Apply a layout preset and refresh selected icon chrome."""
        self.apply_view_layout(layout_id)
        icon_size = 36
        for preset in VIEW_LAYOUT_PRESETS:
            btn = self._layout_preset_btns.get(preset["id"])
            if btn is None:
                continue
            selected = preset["id"] == layout_id
            btn.setChecked(selected)
            btn.setIcon(_make_view_layout_icon(preset, size=icon_size, selected=selected))

    def apply_view_layout(self, layout_id):
        """Rearrange axial/sagittal/coronal/3D panes into a named grid layout."""
        if not hasattr(self, "_grid_layout") or not hasattr(self, "_grid_widgets"):
            return

        preset = next((p for p in VIEW_LAYOUT_PRESETS if p["id"] == layout_id), None)
        if preset is None:
            return

        # Exit in-grid fullscreen first so placements are consistent
        if self._fullscreen_view is not None:
            self.exit_fullscreen()

        placements = preset["placements"]
        nrows = int(preset["rows"])
        ncols = int(preset["cols"])

        # Detach every plane widget from the grid
        for key, widget in self._grid_widgets.items():
            self._grid_layout.removeWidget(widget)
            widget.setVisible(False)
            widget.hide()

        # Clear old stretch factors (up to a safe max)
        for r in range(8):
            self._grid_layout.setRowStretch(r, 0)
            self._grid_layout.setRowMinimumHeight(r, 0)
        for c in range(8):
            self._grid_layout.setColumnStretch(c, 0)
            self._grid_layout.setColumnMinimumWidth(c, 0)

        for r in range(nrows):
            self._grid_layout.setRowStretch(r, 1)
        for c in range(ncols):
            self._grid_layout.setColumnStretch(c, 1)

        # Prefer a larger 3D pane ONLY when it explicitly spans multiple cells
        # (e.g. tall 3D next to stacked MPR). Equal grids like 2×2 / 1×2 must
        # stay uniform — the old "any shared grid" rule made row1/col1 stretch=2
        # and produced the uneven 2×2 (3D bottom-right oversized).
        if "view_3d" in placements and len(placements) > 1:
            r3, c3, rs3, cs3 = placements["view_3d"]
            if rs3 > 1 or cs3 > 1:
                for c in range(c3, c3 + cs3):
                    if 0 <= c < ncols:
                        self._grid_layout.setColumnStretch(c, 2)
                for r in range(r3, r3 + rs3):
                    if 0 <= r < nrows:
                        self._grid_layout.setRowStretch(r, 2)

        new_cells = {}
        for key, (r, c, rs, cs) in placements.items():
            widget = self._grid_widgets.get(key)
            if widget is None:
                continue
            # Expanding + zero min so VTK sizeHint cannot skew equal cells
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            widget.setMinimumSize(0, 0)
            widget.setVisible(True)
            widget.show()
            self._grid_layout.addWidget(widget, r, c, rs, cs)
            new_cells[key] = (widget, r, c, rs, cs)

        # Keep hidden widgets tracked (for fullscreen restore / active ring)
        for key, widget in self._grid_widgets.items():
            if key not in new_cells:
                new_cells[key] = (widget, 0, 0, 1, 1)

        self._grid_cells = new_cells
        self._layout_visible_keys = set(placements.keys())
        self._grid_nrows = nrows
        self._grid_ncols = ncols
        self._current_layout_id = layout_id

        # Focus a visible pane
        preferred = None
        for cand in ("axial", "view_3d", "sagittal", "coronal"):
            if cand in self._layout_visible_keys:
                preferred = cand
                break
        if preferred:
            self._set_active_grid_pane(preferred)

        # Let VTK / overlays reflow after geometry settles, then reset each pane
        QTimer.singleShot(40, self._refresh_layout_viewports)
        # Second pass once Qt has finished resizing native VTK HWNDs
        QTimer.singleShot(120, self._activate_layout_pane_resets)

    def _iter_layout_overlays(self):
        """Yield (key, overlay) for all plane overlays (visible or not)."""
        for key in ("axial", "sagittal", "coronal", "view_3d"):
            if key == "view_3d":
                overlay = getattr(self, "view_3d_overlay_group", None)
            else:
                overlay = getattr(self, f"{key}_overlay_group", None)
            if overlay is not None:
                yield key, overlay

    def _activate_layout_pane_resets(self):
        """Enable + fire Reset on every visible cell (layout-change contract)."""
        visible = getattr(self, "_layout_visible_keys", set())
        for key, overlay in self._iter_layout_overlays():
            if hasattr(overlay, "reset_btn"):
                try:
                    overlay.reset_btn.setEnabled(True)
                    overlay.reset_btn.setVisible(True)
                except Exception:
                    pass
            if key not in visible:
                continue
            try:
                if key == "view_3d":
                    self.reset_3d_view()
                else:
                    self.reset_view(key)
            except Exception:
                pass
            try:
                overlay.update_position()
            except Exception:
                pass

    def _refresh_layout_viewports(self):
        """Reflow VTK widgets after layout change and reset each visible pane.

        After the grid geometry settles, activate each cell's Reset (zoom/pan)
        so cameras fit the new viewport sizes — same action as the per-pane
        refresh button on the overlay strip.
        """
        self._activate_layout_pane_resets()

        for key in getattr(self, "_layout_visible_keys", set()):
            vtk_w = None
            if key == "view_3d":
                vtk_w = getattr(self, "view_3d_widget", None)
            else:
                vtk_w = getattr(self, f"{key}_widget", None)
            should_render = True
            if (
                key == "view_3d"
                and hasattr(self, "is_3d_volume_render_enabled")
                and not self.is_3d_volume_render_enabled()
            ):
                should_render = False
            if should_render and vtk_w is not None:
                try:
                    rw = vtk_w.GetRenderWindow()
                    if rw is not None:
                        rw.Render()
                except Exception:
                    pass

    def _make_sidebar_section(self, title, parent_layout, expanded=True, tooltip=None):
        """Collapsible group for the right align sidebar.

        Returns ``body`` (QVBoxLayout host widget) — put section content there.
        Header shows ▼ when open / ▶ when collapsed; click header to toggle.
        """
        section = QWidget()
        section.setObjectName("AlignSidebarSection")
        section_lay = QVBoxLayout(section)
        section_lay.setContentsMargins(0, 0, 0, 0)
        section_lay.setSpacing(4)

        header = QPushButton()
        header.setObjectName("AlignSectionHeader")
        header.setCheckable(True)
        header.setChecked(bool(expanded))
        header.setCursor(Qt.PointingHandCursor)
        header.setFixedHeight(26)
        if tooltip:
            header.setToolTip(tooltip)
        else:
            header.setToolTip(f"Click to expand / collapse · {title}")
        header.setStyleSheet(f"""
            QPushButton#AlignSectionHeader {{
                text-align: left;
                padding: 2px 4px 2px 2px;
                border: none;
                border-bottom: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 0px;
                background: transparent;
                color: {SemiconductorTheme.TEXT_DISABLED};
                font-weight: 700;
                font-size: 7.5pt;
                letter-spacing: 1.5px;
            }}
            QPushButton#AlignSectionHeader:hover {{
                color: {SemiconductorTheme.PRIMARY_DEFAULT};
                background-color: {SemiconductorTheme.BG_LIGHT};
            }}
            QPushButton#AlignSectionHeader:checked {{
                color: {SemiconductorTheme.TEXT_SECONDARY};
            }}
        """)

        def _sync_header_text(is_open):
            arrow = "▼" if is_open else "▶"
            header.setText(f"  {arrow}  {title}")

        body = QWidget()
        body.setObjectName("AlignSectionBody")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(2, 2, 0, 4)
        body_lay.setSpacing(6)
        body.setVisible(bool(expanded))
        _sync_header_text(bool(expanded))

        def _on_toggled(checked):
            body.setVisible(bool(checked))
            _sync_header_text(bool(checked))

        header.toggled.connect(_on_toggled)

        section_lay.addWidget(header)
        section_lay.addWidget(body)
        parent_layout.addWidget(section)

        # Keep refs so expand state can be driven programmatically if needed
        if not hasattr(self, "_align_sidebar_sections"):
            self._align_sidebar_sections = {}
        self._align_sidebar_sections[title] = {
            "header": header,
            "body": body,
            "section": section,
        }
        return body, body_lay

    
