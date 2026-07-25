# inno3d/features/teaching/seg_ui.py
# -----------------------------------------------------------------------
# SegmentationUIMixin — extracted from inno3d/tabs/teaching.py (Phase 4.2)
#
# UI building: __init__, layout sidebar, parameter panel, action bar,
# enhancement panel, boundary panel, 3D controls + view, ROI overlay.
# -----------------------------------------------------------------------

import csv
import os
import re
import traceback
from pathlib import Path
from pathlib import Path as _Path

import numpy as np
import vtk
from vtk.util import numpy_support

from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import (
    QColor, QCursor, QFont, QIcon, QImage, QLinearGradient, QPainter, QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QAbstractItemView, QAbstractScrollArea, QButtonGroup, QCheckBox, QColorDialog,
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMenu, QMenuBar, QMessageBox, QProgressBar, QProgressDialog,
    QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QSplitter, QStackedWidget, QStyle, QStyledItemDelegate, QStyleOptionViewItem,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QToolButton,
    QVBoxLayout, QWidget,
)
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from inno3d.core.roi_manager import ROIManager, ExportCropThread
from inno3d.core.styles import SemiconductorTheme
from inno3d.core.view_support import (
    Dragonfly3DInteractorStyle,
    SliceBarWheelFilter,
    StepOneSliceSlider,
    apply_dragonfly_volume_zoom,
    push_camera_outside_aabb,
    volume_world_aabb,
)
from inno3d.features.shared.b2b_gap_3d import (
    B2BGapOverlay,
    format_gap_label as _shared_format_gap_label,
    layer_z_offset as _shared_layer_z_offset,
    world_spacing as _shared_world_spacing,
)
from inno3d.features.shared.mes_highlight_3d import MESHighlightOverlay
from inno3d.features.teaching.workers import (
    NoScrollDoubleSpinBox, NoScrollSpinBox,
    _ExportLayerVolumesThread, DistanceTransformThread, BoundaryAnalysisThread,
    SegmentationPreviewThread, SegmentationInspectionThread, EnhancementThread,
)


class SegmentationUIMixin:
    """Tab for interactive Bump/Void segmentation with real-time parameter adjustment"""
    # Signal emitted when inspection finishes: (volume_data, bump_data, void_data)
    inspection_done = pyqtSignal(object, object, object)
    
    # Signal to apply current ROI + Config to Online mode in MainWindow
    apply_online_roi_signal = pyqtSignal(dict, object, str)
    
    def _configure_symbol_button(self, button, icon, tooltip, text="", size=(30, 28)):
        button.setIcon(self.style().standardIcon(icon))
        button.setIconSize(QSize(16, 16))
        button.setToolTip(tooltip)
        if text:
            button.setText(text)
            button.setMinimumHeight(size[1])
        else:
            button.setText("")
            button.setFixedSize(*size)

    def _load_ui_icon(self, icon_name, fallback_standard_icon=None):
        """Load icon from project assets with safe fallback."""
        icon_path = Path(__file__).resolve().parents[2] / "assets" / "icons" / icon_name
        if icon_path.exists():
            return QIcon(str(icon_path))
        if fallback_standard_icon is not None:
            return self.style().standardIcon(fallback_standard_icon)
        return QIcon()

    def _theme_register(self, collection_name, widget):
        collection = getattr(self, collection_name, None)
        if collection is not None and widget is not None and widget not in collection:
            collection.append(widget)
        return widget

    def _secondary_button_qss(self):
        return f"""
            QPushButton {{
                background: {SemiconductorTheme.BTN_SECONDARY_BG};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 8pt;
                font-weight: 600;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QPushButton:hover {{
                background: {SemiconductorTheme.BTN_SECONDARY_HOVER};
                border-color: {SemiconductorTheme.BORDER_HOVER};
            }}
            QPushButton:pressed {{
                background: {SemiconductorTheme.BG_LIGHT};
            }}
            QPushButton:disabled {{
                color: {SemiconductorTheme.TEXT_DISABLED};
            }}
        """

    def _primary_button_qss(self):
        return f"""
            QPushButton {{
                background-color: {SemiconductorTheme.ACCENT_PRIMARY};
                color: {SemiconductorTheme.TEXT_ON_ACCENT};
                font-weight: bold;
                padding: 6px 14px;
                border-radius: 4px;
                font-size: 8.5pt;
                border: 1px solid {SemiconductorTheme.ACCENT_PRIMARY};
            }}
            QPushButton:hover {{
                background-color: {SemiconductorTheme.BORDER_HOVER};
                border-color: {SemiconductorTheme.BORDER_HOVER};
            }}
            QPushButton:disabled {{
                background-color: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_SECONDARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """

    def _success_button_qss(self):
        return f"""
            QPushButton {{
                background-color: {SemiconductorTheme.ACCENT_SUCCESS};
                color: {SemiconductorTheme.TEXT_ON_ACCENT};
                font-weight: bold;
                padding: 6px 14px;
                border-radius: 4px;
                font-size: 8.5pt;
                border: 1px solid {SemiconductorTheme.ACCENT_SUCCESS};
            }}
            QPushButton:hover {{
                background-color: #27ae60;
                border-color: #27ae60;
            }}
            QPushButton:disabled {{
                background-color: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_SECONDARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """

    def _danger_button_qss(self):
        return f"""
            QPushButton {{
                background-color: {SemiconductorTheme.ACCENT_ERROR};
                color: {SemiconductorTheme.TEXT_ON_ACCENT};
                font-weight: bold;
                padding: 5px 12px;
                border-radius: 4px;
                font-size: 8.5pt;
                border: 1px solid {SemiconductorTheme.ACCENT_ERROR};
            }}
            QPushButton:hover {{
                background-color: #bf4a46;
                border-color: #bf4a46;
            }}
            QPushButton:disabled {{
                background-color: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_SECONDARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """

    def _warning_button_qss(self):
        return f"""
            QPushButton {{
                background-color: {SemiconductorTheme.ACCENT_WARNING};
                color: {SemiconductorTheme.TEXT_ON_ACCENT};
                font-weight: bold;
                padding: 5px 12px;
                border-radius: 4px;
                font-size: 8.5pt;
                border: 1px solid {SemiconductorTheme.ACCENT_WARNING};
            }}
            QPushButton:hover {{
                background-color: #b57e25;
                border-color: #b57e25;
            }}
            QPushButton:disabled {{
                background-color: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_SECONDARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """

    def _input_qss(self, accent=False):
        border_color = SemiconductorTheme.ACCENT_PRIMARY if accent else SemiconductorTheme.BORDER_DEFAULT
        return f"""
            background: {SemiconductorTheme.BG_LIGHT};
            color: {SemiconductorTheme.TEXT_PRIMARY};
            border: 1px solid {border_color};
            border-radius: 3px;
            padding: 2px 6px;
        """

    def _status_label_qss(self, color=None):
        tone = color or SemiconductorTheme.TEXT_SECONDARY
        return f"""
            background: {SemiconductorTheme.BG_LIGHT};
            color: {tone};
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 8pt;
            border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
        """

    def _value_label_qss(self):
        return f"""
            background: {SemiconductorTheme.BG_LIGHT};
            color: {SemiconductorTheme.ACCENT_PRIMARY};
            border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            border-radius: 3px;
            padding: 3px 6px;
            font-size: 9pt;
            font-weight: bold;
        """

    def _table_qss(self):
        return SemiconductorTheme.table_stylesheet()

    def _list_qss(self):
        return f"""
            QListWidget {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                font-size: 9pt;
            }}
            QListWidget::item:selected {{
                background: {SemiconductorTheme.ACCENT_PRIMARY};
                color: {SemiconductorTheme.TEXT_ON_ACCENT};
            }}
        """

    def _group_box_qss(self):
        return f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 9pt;
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: {SemiconductorTheme.TEXT_SECONDARY};
            }}
        """

    def _radio_button_qss(self):
        return f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt;"

    def _accent_check_qss(self):
        return f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-weight: bold; margin-left: 10px;"

    def _warning_label_qss(self):
        return f"color: {SemiconductorTheme.ACCENT_WARNING}; font-size: 8pt; font-weight: bold; padding: 2px;"

    class _TeachingTabBarShim:
        """Compatibility shim so existing `teaching_tab_bar.setCurrentIndex(i)` call sites keep working."""

        def __init__(self, owner):
            self._owner = owner

        def setCurrentIndex(self, index):
            self._owner._set_teaching_tab(index)

        def currentIndex(self):
            return self._owner._teaching_tab_index

    def _create_teaching_tab_nav(self):
        """Two-row equal-width tab button grid for the teaching sidebar."""
        nav = QWidget()
        nav.setObjectName("TeachingTabNav")
        grid = QGridLayout(nav)
        grid.setContentsMargins(6, 6, 6, 4)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)

        self._tab_buttons = []
        self._tab_group = QButtonGroup(self)
        self._tab_group.setExclusive(True)
        self._teaching_tab_index = 0

        # 4 columns → row0: Layers ROI Params Enhance | row1: MES B2B 3D
        cols = 4
        for i, (icon_name, label, tip, fallback) in enumerate(self._teaching_tab_defs):
            btn = QToolButton()
            btn.setObjectName("TeachingTabBtn")
            btn.setCheckable(True)
            btn.setAutoRaise(False)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setIcon(self._load_ui_icon(icon_name, fallback))
            btn.setIconSize(QSize(13, 13))
            btn.setText(label)
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            btn.setToolTip(tip)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumHeight(28)
            btn.setMinimumWidth(0)
            btn.setStyleSheet(self._teaching_tab_btn_qss())
            self._tab_group.addButton(btn, i)
            self._tab_buttons.append(btn)
            r, c = divmod(i, cols)
            grid.addWidget(btn, r, c)

        for c in range(cols):
            grid.setColumnStretch(c, 1)

        # Fill remaining cells so row 2 keeps equal column widths
        n = len(self._teaching_tab_defs)
        for c in range(n % cols, cols):
            if n % cols == 0:
                break
            spacer = QWidget()
            spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            spacer.setFixedHeight(28)
            grid.addWidget(spacer, n // cols, c)

        # buttonClicked(QAbstractButton) — works across PyQt5 versions
        self._tab_group.buttonClicked.connect(
            lambda btn: self._on_teaching_tab_clicked(self._tab_group.id(btn))
        )
        if self._tab_buttons:
            self._tab_buttons[0].setChecked(True)
        return nav

    def _on_teaching_tab_clicked(self, index):
        self._teaching_tab_index = index
        if hasattr(self, "teaching_pane"):
            self.teaching_pane.setCurrentIndex(index)

    def _set_teaching_tab(self, index):
        """Programmatically select a teaching sidebar tab (0..6)."""
        if not hasattr(self, "_tab_buttons") or not self._tab_buttons:
            return
        if index < 0 or index >= len(self._tab_buttons):
            return
        btn = self._tab_buttons[index]
        if not btn.isChecked():
            btn.setChecked(True)
        self._on_teaching_tab_clicked(index)

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # --- Top Toolbar ---
        self.toolbar = self.create_toolbar()
        layout.addWidget(self.toolbar)
        
        # --- Main Content: 2x2 Resizable Splitters ---
        self.main_splitter = QSplitter(Qt.Vertical)
        
        self.top_splitter = QSplitter(Qt.Horizontal)
        axial_widget = self.create_single_plane_view("XY", "axial")
        sagittal_widget = self.create_single_plane_view("YZ", "sagittal")
        self.top_splitter.addWidget(axial_widget)
        self.top_splitter.addWidget(sagittal_widget)
        
        self.bottom_splitter = QSplitter(Qt.Horizontal)
        coronal_widget = self.create_single_plane_view("XZ", "coronal")
        
        # 4th cell: Container for the whole QUADRANT (was stacked)
        
        # Custom Header for Teaching Space:
        #   1) Tab navigation — 2-row equal-width button grid
        #   2) Pipeline action strip — equal-width run buttons
        self.teaching_header = QWidget()
        self.teaching_header.setObjectName("TeachingHeader")
        self.teaching_header.setStyleSheet(self._teaching_header_qss())
        header_layout = QVBoxLayout(self.teaching_header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        # --- Tab navigation (2-row grid, equal cell sizes) ---
        # Index order is fixed for _set_teaching_tab / setCurrentIndex callers:
        #   0 Layers, 1 ROI, 2 Params, 3 Enhance, 4 MES, 5 B2B, 6 3D
        self._teaching_tab_defs = [
            ("layers.svg", "Layers", "Define Layers", QStyle.SP_FileDialogListView),
            ("crosshair.svg", "ROI", "Define ROI", QStyle.SP_DialogYesButton),
            ("settings.svg", "Params", "Config. Params", QStyle.SP_FileDialogDetailedView),
            ("sparkles.svg", "Enhance", "AI Volume Enhancement (ONNX)", QStyle.SP_ComputerIcon),
            ("bar-chart-2.svg", "MES", "Object Statistics / Measurement", QStyle.SP_FileDialogInfoView),
            ("ruler-2.svg", "B2B", "3D Boundary Analysis Results (Bump-to-Bump)", QStyle.SP_FileDialogContentsView),
            ("maximize.svg", "3D", "3D Rendering Controls", QStyle.SP_DesktopIcon),
        ]
        self.teaching_tab_nav = self._create_teaching_tab_nav()
        header_layout.addWidget(self.teaching_tab_nav)

        # Subtle divider between nav and pipeline
        self._header_divider = QFrame()
        self._header_divider.setFrameShape(QFrame.HLine)
        self._header_divider.setFixedHeight(1)
        self._header_divider.setStyleSheet(f"background: {SemiconductorTheme.BORDER_DEFAULT}; border: none;")
        header_layout.addWidget(self._header_divider)

        # --- Pipeline actions ---
        self.action_bar = self.create_action_bar()
        header_layout.addWidget(self.action_bar)

        # teaching_pane will hold the actual content widgets
        self.teaching_pane = QStackedWidget()
        self.layer_panel = self.create_define_layers_panel()
        self.roi_panel = self.create_define_roi_panel()
        self.param_widget = self.create_parameter_panel()
        self.enhancement_panel = self.create_enhancement_panel()
        self.stats_widget = self.create_object_stats_panel()
        self.boundary_panel = self._create_boundary_panel()
        self._3d_controls_panel = self._create_3d_controls_panel()

        for panel in (
            self.layer_panel,
            self.roi_panel,
            self.param_widget,
            self.enhancement_panel,
            self.stats_widget,
            self.boundary_panel,
            self._3d_controls_panel,
        ):
            self.teaching_pane.addWidget(panel)

        # Keep a thin QTabBar-compatible shim for legacy setCurrentIndex call sites
        self.teaching_tab_bar = self._TeachingTabBarShim(self)

        # Container for the whole QUADRANT
        self.teaching_quadrant = QWidget()
        quad_layout = QVBoxLayout(self.teaching_quadrant)
        quad_layout.setContentsMargins(0, 0, 0, 0)
        quad_layout.setSpacing(0)
        quad_layout.addWidget(self.teaching_header)
        quad_layout.addWidget(self.teaching_pane, 1)
        
        # --- Bottom-right cell: 3D view (hidden by default) or info ---
        self._3d_view_active = False
        self.teaching_3d_container = self._create_teaching_3d_view()
        self.teaching_3d_container.setVisible(False)
        
        # Info placeholder when 3D is off
        self._bottom_right_info = QWidget()
        self._bottom_right_info.setStyleSheet(f"background: {SemiconductorTheme.BG_DARK};")
        info_lay = QVBoxLayout(self._bottom_right_info)
        info_lay.setAlignment(Qt.AlignCenter)
        info_lbl = QLabel("💡 Enable 3D Rendering\nfrom the [3D] sidebar tab")
        info_lbl.setAlignment(Qt.AlignCenter)
        info_lbl.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 10pt;")
        info_lay.addWidget(info_lbl)
        
        # === ASSEMBLE LAYOUT: [Views 2x2] | [Sidebar] ===
        self.bottom_right_container = QWidget()
        br_layout = QVBoxLayout(self.bottom_right_container)
        br_layout.setContentsMargins(0, 0, 0, 0)
        br_layout.addWidget(self._bottom_right_info)
        br_layout.addWidget(self.teaching_3d_container)
        
        self.bottom_splitter.addWidget(coronal_widget)
        self.bottom_splitter.addWidget(self.bottom_right_container)
        
        self.main_splitter.addWidget(self.top_splitter)
        self.main_splitter.addWidget(self.bottom_splitter)
        
        # Synchronize horizontal splitters to act like a grid
        def sync_top_to_bottom(pos, index):
            self.bottom_splitter.blockSignals(True)
            self.bottom_splitter.moveSplitter(pos, index)
            self.bottom_splitter.blockSignals(False)
            
        def sync_bottom_to_top(pos, index):
            self.top_splitter.blockSignals(True)
            self.top_splitter.moveSplitter(pos, index)
            self.top_splitter.blockSignals(False)
            
        self.top_splitter.splitterMoved.connect(sync_top_to_bottom)
        self.bottom_splitter.splitterMoved.connect(sync_bottom_to_top)
        
        # Ensure widgets have small minimum widths
        axial_widget.setMinimumWidth(100)
        coronal_widget.setMinimumWidth(100)
        sagittal_widget.setMinimumWidth(100)
        
        # Outer splitter: [Views] | [Sidebar]
        self.outer_splitter = QSplitter(Qt.Horizontal)
        self.outer_splitter.addWidget(self.main_splitter)
        self.outer_splitter.addWidget(self.teaching_quadrant)
        # Wide enough for 2-row equal tabs + single-column params without H-scroll
        self.teaching_quadrant.setMinimumWidth(320)
        self.teaching_quadrant.setMaximumWidth(560)
        
        # Set initial sizes
        self.top_splitter.setSizes([800, 800])
        self.bottom_splitter.setSizes([800, 800])
        self.main_splitter.setSizes([600, 600])
        self.outer_splitter.setSizes([1150, 380])
        
        layout.addWidget(self.outer_splitter, 1)
        
        self.setLayout(layout)
        self.refresh_theme()
        
        # Load default DLL and Config (portable: next to exe when frozen, project V2 in dev)
        from inno3d.core.resources import default_config_path, default_dll_dir

        default_dll = default_dll_dir()
        default_config = default_config_path()

        if default_dll and os.path.exists(default_dll):
            self.load_dll_from_path(default_dll)
        if default_config and os.path.exists(default_config):
            self.load_config_from_path(default_config)
        
    def create_toolbar(self):
        """Create top toolbar with load buttons"""
        toolbar = QWidget()
        toolbar.setObjectName("Toolbar")
        toolbar.setStyleSheet(f"""
            .QWidget#Toolbar {{
                background: {SemiconductorTheme.BG_PANEL};
                border-radius: 6px;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        def make_browse_btn(icon_name, tooltip, callback, fallback_icon):
            btn = QToolButton()
            btn.setProperty("class", "load-icon-btn")
            btn.setIcon(self._load_ui_icon(icon_name, fallback_icon))
            btn.setIconSize(QSize(16, 16))
            btn.setToolTip(tooltip)
            btn.setFixedSize(34, 30)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(callback)
            return btn
        
        # DLL Path
        dll_label = QLabel("DLL Folder:")
        layout.addWidget(dll_label)
        
        self.dll_path_input = QLineEdit()
        self.dll_path_input.setReadOnly(True)
        self.dll_path_input.setPlaceholderText("Select folder containing BumpVoidSeg.dll...")
        layout.addWidget(self.dll_path_input, 2)
        
        dll_btn = make_browse_btn("folder.svg", "Browse DLL folder", self.load_dll_folder, QStyle.SP_DirIcon)
        layout.addWidget(dll_btn)
        
        layout.addSpacing(20)
        
        # Config recipe (auto-listed from app config/ folder)
        config_label = QLabel("Config:")
        layout.addWidget(config_label)
        
        self.config_recipe_combo = QComboBox()
        self.config_recipe_combo.setMinimumWidth(160)
        self.config_recipe_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.register_config_recipe_combo(self.config_recipe_combo)
        self.refresh_config_recipe_list()
        layout.addWidget(self.config_recipe_combo, 2)
        
        config_btn = make_browse_btn(
            "settings.svg",
            "Browse recipe outside config/ folder",
            self.load_config_file,
            QStyle.SP_FileIcon,
        )
        layout.addWidget(config_btn)
        
        layout.addSpacing(20)
        
        # Input Path (read-only, auto-updated when volume loads)
        input_label = QLabel("Input Path:")
        layout.addWidget(input_label)
        
        self.input_path_input = QLineEdit()
        self.input_path_input.setReadOnly(True)
        self.input_path_input.setPlaceholderText("Auto-filled when volume is loaded...")
        self.input_path_input.setStyleSheet(
            f"background: {SemiconductorTheme.BG_LIGHT}; color: {SemiconductorTheme.TEXT_PRIMARY}; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};"
        )
        layout.addWidget(self.input_path_input, 2)
        
        layout.addSpacing(20)
        
        # Custom Browse Button for Input Path (Replaces Load Volume main button)
        self.load_volume_btn = make_browse_btn("file.svg", "Browse input file", lambda: self.load_volume(from_folder=False), QStyle.SP_FileIcon)
        self.load_volume_btn.setProperty("is_icon_only", True)
        layout.addWidget(self.load_volume_btn)
        
        self.load_volume_folder_btn = make_browse_btn("folder-open.svg", "Browse input folder", lambda: self.load_volume(from_folder=True), QStyle.SP_DirIcon)
        self.load_volume_folder_btn.setProperty("is_icon_only", True)
        layout.addWidget(self.load_volume_folder_btn)
        
        # Clear Masks button
        self.clear_mask_btn = QPushButton("Clear Mask")
        self.clear_mask_btn.clicked.connect(self.clear_masks)
        layout.addWidget(self.clear_mask_btn)
        
        layout.addSpacing(20)
        
        # Output Path
        output_label = QLabel("Output Path:")
        layout.addWidget(output_label)
        
        self.output_path_input = QLineEdit()
        self.output_path_input.setPlaceholderText("Select output folder for results...")
        layout.addWidget(self.output_path_input, 2)
        
        output_btn = make_browse_btn("folder-open.svg", "Browse output folder", self.browse_output_path, QStyle.SP_DialogOpenButton)
        layout.addWidget(output_btn)
        
        layout.addSpacing(20)
        
        # Crosshair toggle
        self.crosshair_check = QCheckBox("Crosshair")
        self.crosshair_check.stateChanged.connect(self.toggle_crosshair)
        layout.addWidget(self.crosshair_check)
        
        # Color pickers for Class 1 & 2
        self.color_c1_btn = QPushButton("C1 #")
        self.color_c1_btn.setFixedWidth(60)
        self.color_c1_btn.setToolTip("Choose Class 1 overlay color")
        self.color_c1_btn.clicked.connect(lambda: self.choose_overlay_color(1))
        layout.addWidget(self.color_c1_btn)

        self.color_c2_btn = QPushButton("C2 #")
        self.color_c2_btn.setFixedWidth(60)
        self.color_c2_btn.setToolTip("Choose Class 2 overlay color")
        self.color_c2_btn.clicked.connect(lambda: self.choose_overlay_color(2))
        layout.addWidget(self.color_c2_btn)
        
        layout.addStretch()
        
        # Info label
        self.info_label = QLabel("Ready - Load DLL, Config, and Volume to begin")
        self._theme_register("_theme_secondary_labels", self.info_label)
        self.info_label.setStyleSheet(f"""
            color: {SemiconductorTheme.TEXT_SECONDARY};
            font-size: 9pt;
            padding: 5px;
        """)
        layout.addWidget(self.info_label)
        
        toolbar.setLayout(layout)
        return toolbar

    def refresh_theme(self):
        """Re-apply dynamic styles when app theme changes at runtime."""
        panel_qss = f"background: {SemiconductorTheme.BG_PANEL}; border-radius: 6px;"

        if hasattr(self, "teaching_quadrant"):
            self.teaching_quadrant.setStyleSheet(f"background: {SemiconductorTheme.BG_PANEL};")
        if hasattr(self, "teaching_pane"):
            self.teaching_pane.setStyleSheet(f"background: {SemiconductorTheme.BG_PANEL};")
        for panel in self._theme_panels:
            panel.setStyleSheet(panel_qss)

        if hasattr(self, "teaching_header"):
            self.teaching_header.setStyleSheet(self._teaching_header_qss())
        if hasattr(self, "_tab_buttons"):
            for btn in self._tab_buttons:
                btn.setStyleSheet(self._teaching_tab_btn_qss())
        if hasattr(self, "_header_divider"):
            self._header_divider.setStyleSheet(
                f"background: {SemiconductorTheme.BORDER_DEFAULT}; border: none;"
            )
        if hasattr(self, "action_bar"):
            self.action_bar.setStyleSheet(
                f"QWidget#TeachingActionBar {{"
                f"  background: {SemiconductorTheme.BG_DARK};"
                f"  border: none;"
                f"}}"
            )
        if hasattr(self, "_pipeline_run_label"):
            self._pipeline_run_label.setStyleSheet(self._pipeline_run_label_qss())
        # Equal-width pipeline buttons (role-based accents stay current after theme switch)
        if hasattr(self, "_pipeline_buttons"):
            for btn, role in self._pipeline_buttons:
                btn.setStyleSheet(self._pipeline_btn_qss(role))

        if hasattr(self, "toolbar"):
            self.toolbar.setStyleSheet(
                f"""
                .QWidget#Toolbar {{
                    background: {SemiconductorTheme.BG_PANEL};
                    border-radius: 6px;
                    border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                }}
                """
            )
        if hasattr(self, "input_path_input"):
            self.input_path_input.setStyleSheet(
                f"background: {SemiconductorTheme.BG_LIGHT}; color: {SemiconductorTheme.TEXT_PRIMARY}; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};"
            )
        if hasattr(self, "info_label"):
            self.info_label.setStyleSheet(
                f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt; padding: 5px;"
            )

        for label in self._theme_accent_labels:
            label.setStyleSheet(f"font-size: 10pt; color: {SemiconductorTheme.ACCENT_PRIMARY};")
        for label in self._theme_secondary_labels:
            label.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        for label in self._theme_value_labels:
            label.setStyleSheet(self._value_label_qss())
        for button in self._theme_secondary_buttons:
            button.setStyleSheet(self._secondary_button_qss())
        for button in self._theme_primary_buttons:
            button.setStyleSheet(self._primary_button_qss())
        for button in self._theme_success_buttons:
            button.setStyleSheet(self._success_button_qss())
        for button in self._theme_danger_buttons:
            button.setStyleSheet(self._danger_button_qss())
        for button in self._theme_warning_buttons:
            button.setStyleSheet(self._warning_button_qss())
        for label in self._theme_warning_labels:
            label.setStyleSheet(self._warning_label_qss())
        for label in self._theme_status_labels:
            label.setStyleSheet(self._status_label_qss())
        for widget in self._theme_tables:
            widget.setStyleSheet(self._table_qss())
        for widget in self._theme_lists:
            widget.setStyleSheet(self._list_qss())
        for widget in self._theme_inputs:
            widget.setStyleSheet(self._input_qss())
        for widget in self._theme_group_boxes:
            widget.setStyleSheet(self._group_box_qss())
        if hasattr(self, "_param_sticky_frame"):
            self._param_sticky_frame.setStyleSheet(self._param_sticky_header_qss())
        for widget in self._theme_radio_buttons:
            widget.setStyleSheet(self._radio_button_qss())
        for widget in self._theme_accent_checks:
            widget.setStyleSheet(self._accent_check_qss())
        # MES toolbar uses its own compact styles (do not use generic button padding)
        if hasattr(self, "_style_mes_toolbar"):
            self._style_mes_toolbar()
        if hasattr(self, "_mes_toolbar_divider") and self._mes_toolbar_divider is not None:
            self._mes_toolbar_divider.setStyleSheet(
                f"background: {SemiconductorTheme.BORDER_DEFAULT}; border: none; max-height: 1px;"
            )
        for button in self.findChildren(QPushButton):
            if button.text() in ("Browse...", "Browse...", "Remove Selected"):
                button.setStyleSheet(self._secondary_button_qss())

        for orientation in ["axial", "coronal", "sagittal"]:
            header_widget = getattr(self, f"{orientation}_header_widget", None)
            if header_widget is not None:
                header_widget.setStyleSheet(self._mpr_header_qss())
            title_label = getattr(self, f"{orientation}_title_label", None)
            if title_label is not None:
                title_label.setStyleSheet(self._mpr_plane_badge_qss())
            tool_cluster = getattr(self, f"{orientation}_tool_cluster", None)
            if tool_cluster is not None:
                tool_cluster.setStyleSheet(self._mpr_action_pair_qss())
            slice_bar = getattr(self, f"{orientation}_slice_bar", None)
            if slice_bar is not None:
                slice_bar.setStyleSheet(
                    f"QFrame#MprSliceBar {{"
                    f"  background: {SemiconductorTheme.BG_PANEL};"
                    f"  border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};"
                    f"  border-radius: 4px;"
                    f"}}"
                )
            pixel_label = getattr(self, f"{orientation}_pixel_label", None)
            if pixel_label is not None:
                pixel_label.setStyleSheet(
                    f"font-size: 8pt; color: {SemiconductorTheme.TEXT_SECONDARY}; padding-left: 4px;"
                )
            overlay_bump = getattr(self, f"{orientation}_overlay_bump", None)
            overlay_void = getattr(self, f"{orientation}_overlay_void", None)
            if overlay_bump is not None:
                overlay_bump.setStyleSheet(
                    f"font-size: 8pt; color: {SemiconductorTheme.TEXT_PRIMARY}; spacing: 3px;"
                )
            if overlay_void is not None:
                overlay_void.setStyleSheet(
                    f"font-size: 8pt; color: {SemiconductorTheme.TEXT_PRIMARY}; spacing: 3px;"
                )

        if hasattr(self, "inspect_btn"):
            self.inspect_btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {SemiconductorTheme.ACCENT_SUCCESS};
                    color: {SemiconductorTheme.TEXT_ON_ACCENT};
                    font-weight: bold;
                    padding: 5px 15px;
                    border-radius: 4px;
                    font-size: 8.5pt;
                }}
                QPushButton:hover {{
                    background-color: #27ae60;
                }}
                QPushButton:disabled {{
                    background-color: {SemiconductorTheme.BG_LIGHT};
                    color: {SemiconductorTheme.TEXT_SECONDARY};
                    border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                }}
                """
            )

        if hasattr(self, "measure_btn"):
            self.measure_btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {SemiconductorTheme.ACCENT_SUCCESS};
                    color: {SemiconductorTheme.TEXT_ON_ACCENT};
                    font-weight: bold;
                    padding: 5px 15px;
                    border-radius: 4px;
                    font-size: 8.5pt;
                }}
                QPushButton:hover {{
                    background-color: #27ae60;
                }}
                QPushButton:disabled {{
                    background-color: {SemiconductorTheme.BG_LIGHT};
                    color: {SemiconductorTheme.TEXT_SECONDARY};
                    border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                }}
                """
            )

        if hasattr(self, "far_btn"):
            self.far_btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {SemiconductorTheme.ACCENT_PRIMARY};
                    color: {SemiconductorTheme.TEXT_ON_ACCENT};
                    font-weight: bold;
                    padding: 5px 15px;
                    border-radius: 4px;
                    font-size: 8.5pt;
                }}
                QPushButton:hover {{
                    opacity: 0.9;
                }}
                QPushButton:disabled {{
                    background-color: {SemiconductorTheme.BG_LIGHT};
                    color: {SemiconductorTheme.TEXT_SECONDARY};
                    border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                }}
                """
            )

        if hasattr(self, "dt_btn"):
            self.dt_btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {SemiconductorTheme.ACCENT_PRIMARY};
                    color: {SemiconductorTheme.TEXT_ON_ACCENT};
                    font-weight: bold;
                    padding: 5px 15px;
                    border-radius: 4px;
                    font-size: 8.5pt;
                }}
                QPushButton:hover {{
                    opacity: 0.9;
                }}
                QPushButton:disabled {{
                    background-color: {SemiconductorTheme.BG_LIGHT};
                    color: {SemiconductorTheme.TEXT_SECONDARY};
                    border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                }}
                """
            )

        if hasattr(self, "bnd_btn"):
            self.bnd_btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {SemiconductorTheme.ACCENT_WARNING};
                    color: {SemiconductorTheme.TEXT_ON_ACCENT};
                    font-weight: bold;
                    padding: 5px 15px;
                    border-radius: 4px;
                    font-size: 8.5pt;
                }}
                QPushButton:hover {{
                    background-color: #b57e25;
                }}
                QPushButton:disabled {{
                    background-color: {SemiconductorTheme.BG_LIGHT};
                    color: {SemiconductorTheme.TEXT_SECONDARY};
                    border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                }}
                """
            )

        if hasattr(self, "layer_filter_combo"):
            self.layer_filter_combo.setStyleSheet(
                f"""
                QComboBox {{
                    background: {SemiconductorTheme.BG_LIGHT};
                    color: {SemiconductorTheme.TEXT_PRIMARY};
                    border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                    border-radius: 3px;
                    padding: 2px 10px;
                    min-width: 100px;
                }}
                """
            )
        if hasattr(self, "roi_select_btn"):
            self.roi_select_btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {SemiconductorTheme.ACCENT_PRIMARY};
                    color: {SemiconductorTheme.TEXT_ON_ACCENT};
                    font-weight: bold; font-size: 10pt;
                    padding: 8px 16px;
                    border-radius: 5px;
                    border: none;
                }}
                QPushButton:checked {{
                    background-color: {SemiconductorTheme.ACCENT_ERROR};
                    color: {SemiconductorTheme.TEXT_ON_ACCENT};
                }}
                QPushButton:hover {{
                    opacity: 0.9;
                }}
                """
            )

    def _mes_compact_btn_qss(self, role="secondary"):
        """Compact toolbar button styles for the narrow MES sidebar."""
        if role == "primary":
            bg, hover, text, border = (
                SemiconductorTheme.ACCENT_PRIMARY,
                SemiconductorTheme.PRIMARY_HOVER,
                SemiconductorTheme.TEXT_ON_ACCENT,
                SemiconductorTheme.ACCENT_PRIMARY,
            )
        elif role == "danger":
            bg, hover, text, border = (
                SemiconductorTheme.ACCENT_ERROR,
                "#bf4a46",
                "#ffffff",
                SemiconductorTheme.ACCENT_ERROR,
            )
        elif role == "warning":
            bg, hover, text, border = (
                SemiconductorTheme.ACCENT_WARNING,
                "#b57e25",
                SemiconductorTheme.TEXT_ON_ACCENT,
                SemiconductorTheme.ACCENT_WARNING,
            )
        else:
            bg, hover, text, border = (
                SemiconductorTheme.BTN_SECONDARY_BG,
                SemiconductorTheme.BTN_SECONDARY_HOVER,
                SemiconductorTheme.TEXT_PRIMARY,
                SemiconductorTheme.BORDER_DEFAULT,
            )
        return f"""
            QPushButton {{
                background: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 0 8px;
                font-size: 8pt;
                font-weight: 700;
                min-height: 26px;
                max-height: 26px;
            }}
            QPushButton:hover {{
                background: {hover};
                border-color: {hover};
            }}
            QPushButton:pressed {{
                background: {SemiconductorTheme.BG_DARK};
            }}
            QPushButton:disabled {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_DISABLED};
                border-color: {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """

    def _mes_toolbar_frame_qss(self):
        return f"""
            QFrame#MesToolbar {{
                background: {SemiconductorTheme.BG_LIGHT};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 6px;
            }}
        """

    def _mes_ng_spin_qss(self):
        return f"""
            QDoubleSpinBox {{
                background: {SemiconductorTheme.BG_DARK};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.ACCENT_PRIMARY};
                border-radius: 4px;
                padding: 1px 4px;
                font-size: 8pt;
                font-weight: 700;
                min-height: 24px;
                max-height: 26px;
            }}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width: 14px;
            }}
        """

    def _style_mes_toolbar(self):
        """Apply/refresh MES toolbar chrome (theme-safe compact controls)."""
        if hasattr(self, "mes_toolbar") and self.mes_toolbar is not None:
            self.mes_toolbar.setStyleSheet(self._mes_toolbar_frame_qss())
        if hasattr(self, "mes_title_label") and self.mes_title_label is not None:
            self.mes_title_label.setStyleSheet(
                f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-size: 9.5pt; "
                f"font-weight: 800; letter-spacing: 0.4px; background: transparent; border: none;"
            )
        if hasattr(self, "stats_info_label") and self.stats_info_label is not None:
            self.stats_info_label.setStyleSheet(
                f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt; "
                f"background: transparent; border: none;"
            )
        if hasattr(self, "mes_ng_label") and self.mes_ng_label is not None:
            self.mes_ng_label.setStyleSheet(
                f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt; "
                f"font-weight: 700; background: transparent; border: none;"
            )
        if hasattr(self, "show_indices_check") and self.show_indices_check is not None:
            self.show_indices_check.setStyleSheet(
                f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt; font-weight: 600; "
                f"spacing: 4px; background: transparent; border: none;"
            )
        if hasattr(self, "ng_threshold_spin") and self.ng_threshold_spin is not None:
            self.ng_threshold_spin.setStyleSheet(self._mes_ng_spin_qss())

        btn_roles = (
            ("export_csv_btn", "secondary"),
            ("clear_highlight_btn", "secondary"),
            ("bnd_clear_btn", "secondary"),
            ("apply_ng_btn", "primary"),
            ("delete_selected_btn", "danger"),
            ("delete_edge_btn", "warning"),
        )
        for attr, role in btn_roles:
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setStyleSheet(self._mes_compact_btn_qss(role))

    def _make_mes_tool_button(self, text, tooltip, role, on_click, icon_name=None, fallback_icon=None):
        """Uniform compact action button for MES toolbar."""
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(tooltip)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setMinimumWidth(0)
        btn.setFixedHeight(26)
        btn.setStyleSheet(self._mes_compact_btn_qss(role))
        if icon_name or fallback_icon is not None:
            icon = self._load_ui_icon(icon_name, fallback_icon) if icon_name else self.style().standardIcon(fallback_icon)
            if not icon.isNull():
                btn.setIcon(icon)
                btn.setIconSize(QSize(13, 13))
        btn.clicked.connect(on_click)
        return btn

    def create_object_stats_panel(self):
        """Create the object statistics panel with table and controls"""
        panel = QWidget()
        panel.setObjectName("MesStatsPanel")
        self._theme_register("_theme_panels", panel)
        panel.setStyleSheet(
            f"QWidget#MesStatsPanel {{"
            f"  background: {SemiconductorTheme.BG_PANEL};"
            f"  border-radius: 6px;"
            f"}}"
        )
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(6, 4, 6, 4)
        panel_layout.setSpacing(6)

        # ── Structured toolbar (3 rows, narrow-sidebar safe) ──
        self.mes_toolbar = QFrame()
        self.mes_toolbar.setObjectName("MesToolbar")
        self.mes_toolbar.setStyleSheet(self._mes_toolbar_frame_qss())
        tb = QVBoxLayout(self.mes_toolbar)
        tb.setContentsMargins(8, 6, 8, 6)
        tb.setSpacing(5)

        # Row 1 — title + status
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)
        self.mes_title_label = QLabel("MES")
        self.mes_title_label.setObjectName("MesTitle")
        title_row.addWidget(self.mes_title_label, 0, Qt.AlignVCenter)

        self.stats_info_label = QLabel("Waiting for measurement…")
        self.stats_info_label.setObjectName("MesStatus")
        self.stats_info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.stats_info_label.setWordWrap(True)
        self.stats_info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        title_row.addWidget(self.stats_info_label, 1)
        tb.addLayout(title_row)

        # Subtle divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFixedHeight(1)
        divider.setStyleSheet(
            f"background: {SemiconductorTheme.BORDER_DEFAULT}; border: none; max-height: 1px;"
        )
        self._mes_toolbar_divider = divider
        tb.addWidget(divider)

        # Row 2 — export / selection / view
        tools_row = QHBoxLayout()
        tools_row.setContentsMargins(0, 0, 0, 0)
        tools_row.setSpacing(4)

        self.export_csv_btn = self._make_mes_tool_button(
            "CSV",
            "Export object statistics to CSV",
            "secondary",
            self.export_stats_csv,
            icon_name="download.svg",
            fallback_icon=QStyle.SP_DialogSaveButton,
        )
        self.export_csv_btn.setEnabled(False)
        tools_row.addWidget(self.export_csv_btn, 1)

        self.clear_highlight_btn = self._make_mes_tool_button(
            "Clear",
            "Clear object selection / highlight",
            "secondary",
            self.clear_object_selection,
            icon_name="x.svg",
            fallback_icon=QStyle.SP_DialogResetButton,
        )
        tools_row.addWidget(self.clear_highlight_btn, 1)

        self.show_indices_check = QCheckBox("Index")
        self.show_indices_check.setToolTip("Show object index labels on MPR views")
        self.show_indices_check.setChecked(False)
        self.show_indices_check.setCursor(Qt.PointingHandCursor)
        self.show_indices_check.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.show_indices_check.stateChanged.connect(lambda: self._refresh_all_index_views())
        tools_row.addWidget(self.show_indices_check, 0, Qt.AlignVCenter)
        tb.addLayout(tools_row)

        # Row 3 — NG threshold + destructive actions (one compact control strip)
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(4)

        self.mes_ng_label = QLabel("NG")
        self.mes_ng_label.setToolTip("NG Threshold (void ratio %)")
        action_row.addWidget(self.mes_ng_label, 0, Qt.AlignVCenter)

        self.ng_threshold_spin = NoScrollDoubleSpinBox()
        self.ng_threshold_spin.setRange(0.0, 100.0)
        self.ng_threshold_spin.setValue(5.0)
        self.ng_threshold_spin.setSingleStep(0.5)
        self.ng_threshold_spin.setSuffix("%")
        self.ng_threshold_spin.setDecimals(2)
        self.ng_threshold_spin.setMinimumWidth(64)
        self.ng_threshold_spin.setMaximumWidth(84)
        self.ng_threshold_spin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ng_threshold_spin.setToolTip(
            "Objects with Ratio ≥ this threshold are classified as NG (Not Good)"
        )
        action_row.addWidget(self.ng_threshold_spin, 0)

        self.apply_ng_btn = self._make_mes_tool_button(
            "Apply",
            "Apply NG threshold to judgment column",
            "primary",
            self._on_threshold_changed,
            icon_name="check.svg",
            fallback_icon=QStyle.SP_DialogApplyButton,
        )
        action_row.addWidget(self.apply_ng_btn, 1)

        self.delete_selected_btn = self._make_mes_tool_button(
            "Del",
            "Delete selected objects from stats / masks",
            "danger",
            self.delete_selected_objects,
            icon_name="trash.svg",
            fallback_icon=QStyle.SP_TrashIcon,
        )
        action_row.addWidget(self.delete_selected_btn, 1)

        self.delete_edge_btn = self._make_mes_tool_button(
            "Edge",
            "Delete objects touching image edges (incomplete objects)",
            "warning",
            self.delete_edge_objects,
            icon_name="eraser.svg",
            fallback_icon=QStyle.SP_DialogDiscardButton,
        )
        action_row.addWidget(self.delete_edge_btn, 1)
        tb.addLayout(action_row)

        self._style_mes_toolbar()
        panel_layout.addWidget(self.mes_toolbar)

        self.object_stats_table = QTableWidget()
        self._theme_register("_theme_tables", self.object_stats_table)
        self.object_stats_table.setColumnCount(19)
        self.object_stats_table.setHorizontalHeaderLabels([
            "#", "Layer", "Bump ID", "B. H", "B. V", "V. V", "Ratio (%)",
            "Judgment", "Gap X", "Gap Y",
            "Z_min", "Z_max", "Y_min", "Y_max",
            "X_min", "X_max", "Ctr_Z", "Ctr_Y", "Ctr_X"
        ])
        self.object_stats_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.object_stats_table.setSelectionMode(QTableWidget.MultiSelection)
        self.object_stats_table.setAlternatingRowColors(True)
        self.object_stats_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.object_stats_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.object_stats_table.setStyleSheet(SemiconductorTheme.table_stylesheet())
        self.object_stats_table.horizontalHeader().setStretchLastSection(False)
        # Interactive only — ResizeToContents freezes UI when filling 400+ rows
        # (Qt remeasures every setItem). Widths set once in _refresh_stats_table.
        self.object_stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.object_stats_table.verticalHeader().setVisible(False)
        self.object_stats_table.setSortingEnabled(True)
        self.object_stats_table.itemSelectionChanged.connect(self.on_object_selection_changed)
        panel_layout.addWidget(self.object_stats_table, 1)

        return panel
        
    def _mpr_header_qss(self):
        return (
            f"QFrame#MprViewHeader {{"
            f"  background: {SemiconductorTheme.BG_PANEL};"
            f"  border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};"
            f"  border-radius: 4px;"
            f"}}"
        )

    def _mpr_plane_badge_qss(self):
        return (
            f"QLabel#MprPlaneBadge {{"
            f"  background: {SemiconductorTheme.BG_LIGHT};"
            f"  color: {SemiconductorTheme.ACCENT_PRIMARY};"
            f"  border: 1px solid {SemiconductorTheme.ACCENT_PRIMARY};"
            f"  border-radius: 4px;"
            f"  font-size: 8.5pt;"
            f"  font-weight: 800;"
            f"  letter-spacing: 0.4px;"
            f"  padding: 0 6px;"
            f"}}"
        )

    def _mpr_action_pair_qss(self):
        """Segmented Fullscreen | Reset control — matches plane-badge height."""
        return f"""
            QFrame#MprActionPair {{
                background: {SemiconductorTheme.BG_LIGHT};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 5px;
            }}
            QPushButton#MprActionBtn {{
                background: transparent;
                border: none;
                border-radius: 0px;
                padding: 0px;
                margin: 0px;
                color: {SemiconductorTheme.TEXT_PRIMARY};
            }}
            QPushButton#MprActionBtn:hover {{
                background: {SemiconductorTheme.EL4};
            }}
            QPushButton#MprActionBtn:pressed {{
                background: {SemiconductorTheme.BG_DARK};
            }}
            QFrame#MprActionSplit {{
                background: {SemiconductorTheme.BORDER_DEFAULT};
                border: none;
                max-width: 1px;
                min-width: 1px;
            }}
        """

    def _mpr_tool_cluster_qss(self):
        # Back-compat alias used by theme refresh
        return self._mpr_action_pair_qss()

    def _mpr_tool_btn_qss(self):
        # Back-compat alias — real styles live on the pair container
        return ""

    def _mpr_vdivider_qss(self):
        return (
            f"QFrame#MprVDivider {{"
            f"  background: {SemiconductorTheme.BORDER_DEFAULT};"
            f"  border: none;"
            f"  max-width: 1px;"
            f"  min-width: 1px;"
            f"}}"
        )

    def _make_mpr_action_button(self, icon_name, tooltip, fallback_icon):
        """Equal square action button for the MPR segmented pair."""
        btn = QPushButton()
        btn.setObjectName("MprActionBtn")
        btn.setIcon(self._load_ui_icon(icon_name, fallback_icon))
        btn.setIconSize(QSize(15, 15))
        btn.setFixedSize(28, 26)
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFocusPolicy(Qt.NoFocus)
        btn.setFlat(True)
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return btn

    def _make_mpr_action_pair(self, orientation):
        """New balanced Fullscreen | Reset segmented control for one MPR pane."""
        pair = QFrame()
        pair.setObjectName("MprActionPair")
        pair.setStyleSheet(self._mpr_action_pair_qss())
        pair.setFixedHeight(28)
        pair.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        lay = QHBoxLayout(pair)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Fullscreen — maximize icon (symmetric glyph)
        fs_btn = self._make_mpr_action_button(
            "maximize.svg", "Fullscreen view", QStyle.SP_TitleBarMaxButton
        )
        fs_btn.clicked.connect(lambda: self.toggle_view_fullscreen(orientation))
        lay.addWidget(fs_btn)

        split = QFrame()
        split.setObjectName("MprActionSplit")
        split.setFixedWidth(1)
        split.setFixedHeight(16)
        lay.addWidget(split, 0, Qt.AlignVCenter)

        # Reset — balanced dual-arrow refresh (cleaner than rotate-clockwise)
        reset_btn = self._make_mpr_action_button(
            "refresh.svg", "Reset zoom & pan", QStyle.SP_BrowserReload
        )
        reset_btn.clicked.connect(lambda: self.reset_view(orientation))
        lay.addWidget(reset_btn)

        return pair, fs_btn, reset_btn

    def _make_mpr_vdivider(self):
        div = QFrame()
        div.setObjectName("MprVDivider")
        div.setFrameShape(QFrame.NoFrame)
        div.setFixedWidth(1)
        div.setFixedHeight(18)
        div.setStyleSheet(self._mpr_vdivider_qss())
        return div

    def create_single_plane_view(self, title, orientation):
        """Create one MPR pane: structured header (identity | tools | overlays | status) + VTK + slice."""
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        widget.setMinimumWidth(200)
        widget.setMinimumHeight(140)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(3)

        # ── Header: zones that do not collapse into each other ──
        # [Plane] [⛶ ↻] | [Bump] [Void] [OP ──] [Z?] ........ [Pixel]
        header_widget = QFrame()
        header_widget.setObjectName("MprViewHeader")
        header_widget.setFixedHeight(34)
        header_widget.setStyleSheet(self._mpr_header_qss())
        header_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        controls = QHBoxLayout(header_widget)
        controls.setContentsMargins(6, 4, 6, 4)
        controls.setSpacing(8)

        # Zone A — plane identity (same height as action pair for visual balance)
        title_label = QLabel(title)
        title_label.setObjectName("MprPlaneBadge")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFixedHeight(28)
        title_label.setMinimumWidth(32)
        title_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        title_label.setStyleSheet(self._mpr_plane_badge_qss())
        title_label.setToolTip(f"{title} plane view")
        controls.addWidget(title_label, 0, Qt.AlignVCenter)

        # Zone B — new segmented control: [ Fullscreen | Reset ]
        tool_cluster, fullscreen_btn, reset_btn = self._make_mpr_action_pair(orientation)
        controls.addWidget(tool_cluster, 0, Qt.AlignVCenter)

        # Divider between tools and overlays
        controls.addWidget(self._make_mpr_vdivider(), 0, Qt.AlignVCenter)

        # Zone C — overlays + opacity (compact, can shrink slightly)
        overlay_wrap = QWidget()
        overlay_wrap.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        overlay_lay = QHBoxLayout(overlay_wrap)
        overlay_lay.setContentsMargins(0, 0, 0, 0)
        overlay_lay.setSpacing(6)

        overlay_bump_check = QCheckBox("Bump")
        overlay_bump_check.setToolTip("Show Bump / Class 1 overlay")
        overlay_bump_check.setChecked(True)
        overlay_bump_check.setStyleSheet(
            f"font-size: 8pt; color: {SemiconductorTheme.TEXT_PRIMARY}; spacing: 3px;"
        )
        overlay_bump_check.stateChanged.connect(
            lambda state, o=orientation: self.update_plane_view(o, preserve_camera=True)
        )
        overlay_lay.addWidget(overlay_bump_check)

        overlay_void_check = QCheckBox("Void")
        overlay_void_check.setToolTip("Show Void / Class 2 overlay")
        overlay_void_check.setChecked(True)
        overlay_void_check.setStyleSheet(
            f"font-size: 8pt; color: {SemiconductorTheme.TEXT_PRIMARY}; spacing: 3px;"
        )
        overlay_void_check.stateChanged.connect(
            lambda state, o=orientation: self.update_plane_view(o, preserve_camera=True)
        )
        overlay_lay.addWidget(overlay_void_check)

        opacity_label = QLabel("OP")
        opacity_label.setStyleSheet(
            f"font-size: 7.5pt; font-weight: 700; color: {SemiconductorTheme.TEXT_SECONDARY};"
        )
        opacity_label.setToolTip("Overlay opacity")
        overlay_lay.addWidget(opacity_label)

        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setRange(0, 100)
        opacity_slider.setValue(70)
        opacity_slider.setFixedWidth(56)
        opacity_slider.setFixedHeight(16)
        opacity_slider.setToolTip("Overlay opacity")
        opacity_slider.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        opacity_slider.valueChanged.connect(
            lambda value, o=orientation: self.update_plane_view(o, preserve_camera=True)
        )
        overlay_lay.addWidget(opacity_slider)

        controls.addWidget(overlay_wrap, 0, Qt.AlignVCenter)

        # Flexible spacer — keeps left tools stable when pane resizes
        controls.addStretch(1)

        # Zone D — readout (right, elidable)
        pixel_label = QLabel("Pixel: --")
        pixel_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        pixel_label.setMinimumWidth(64)
        pixel_label.setMaximumWidth(160)
        pixel_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        pixel_label.setStyleSheet(
            f"font-size: 8pt; color: {SemiconductorTheme.TEXT_SECONDARY}; padding-left: 4px;"
        )
        controls.addWidget(pixel_label, 0, Qt.AlignVCenter)

        # Legacy alias (theme refresh used separator_label)
        separator = None

        layout.addWidget(header_widget)

        # ── VTK view ──
        vtk_widget = QVTKRenderWindowInteractor()
        style = vtk.vtkInteractorStyleImage()
        vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
        vtk_widget.installEventFilter(self)
        vtk_widget.setMouseTracking(True)  # hover highlight without button held

        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0, 0, 0)
        vtk_widget.GetRenderWindow().AddRenderer(renderer)
        layout.addWidget(vtk_widget, 1)

        # ── Slice bar ──
        slice_widget = QFrame()
        slice_widget.setObjectName("MprSliceBar")
        slice_widget.setFixedHeight(26)
        slice_widget.setStyleSheet(
            f"QFrame#MprSliceBar {{"
            f"  background: {SemiconductorTheme.BG_PANEL};"
            f"  border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};"
            f"  border-radius: 4px;"
            f"}}"
        )
        slice_layout = QHBoxLayout(slice_widget)
        slice_layout.setContentsMargins(8, 2, 8, 2)
        slice_layout.setSpacing(8)

        slice_text = QLabel("Slice")
        slice_text.setStyleSheet(
            f"font-size: 7.5pt; font-weight: 700; color: {SemiconductorTheme.TEXT_SECONDARY};"
        )
        slice_layout.addWidget(slice_text)

        slice_slider = StepOneSliceSlider(Qt.Horizontal)
        slice_slider.setRange(0, 100)
        slice_slider.setValue(50)
        slice_slider.setFixedHeight(16)
        slice_slider.setToolTip("Drag or scroll (step 1) to change slice")
        slice_slider.valueChanged.connect(lambda value, o=orientation: self.update_slice(o, value))
        slice_layout.addWidget(slice_slider, 1)

        slice_label = QLabel("50 / 100")
        slice_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slice_label.setMinimumWidth(56)
        slice_label.setStyleSheet(
            f"font-size: 8pt; font-weight: 600; color: {SemiconductorTheme.TEXT_PRIMARY};"
        )
        slice_slider.valueChanged.connect(
            lambda value, lbl=slice_label, s=slice_slider: lbl.setText(f"{value} / {s.maximum()}")
        )
        slice_layout.addWidget(slice_label)

        # Wheel on entire slice bar → step 1 (not only the thin groove)
        wheel_filter = SliceBarWheelFilter(slice_slider, slice_widget)
        slice_widget.installEventFilter(wheel_filter)
        slice_slider.installEventFilter(wheel_filter)
        slice_text.installEventFilter(wheel_filter)
        slice_label.installEventFilter(wheel_filter)
        slice_widget._slice_wheel_filter = wheel_filter

        layout.addWidget(slice_widget)

        # Store refs
        setattr(self, f"{orientation}_widget", vtk_widget)
        setattr(self, f"{orientation}_renderer", renderer)
        setattr(self, f"{orientation}_slice_slider", slice_slider)
        setattr(self, f"{orientation}_slice_label", slice_label)
        setattr(self, f"{orientation}_overlay_bump", overlay_bump_check)
        setattr(self, f"{orientation}_overlay_void", overlay_void_check)
        setattr(self, f"{orientation}_opacity_slider", opacity_slider)
        setattr(self, f"{orientation}_pixel_label", pixel_label)
        setattr(self, f"{orientation}_container", widget)
        setattr(self, f"{orientation}_header_widget", header_widget)
        setattr(self, f"{orientation}_title_label", title_label)
        setattr(self, f"{orientation}_tool_cluster", tool_cluster)
        setattr(self, f"{orientation}_fullscreen_btn", fullscreen_btn)
        setattr(self, f"{orientation}_reset_btn", reset_btn)
        setattr(self, f"{orientation}_separator_label", separator)
        setattr(self, f"{orientation}_slice_bar", slice_widget)

        return widget
    # ==================== DEFINE LAYERS PANEL ====================

    def create_define_layers_panel(self):
        """Create the Define Layers panel for splitting chip volume into Z-ranges."""
        panel = QWidget()
        self._theme_register("_theme_panels", panel)
        panel.setStyleSheet(f"background: {SemiconductorTheme.BG_PANEL}; border-radius: 6px;")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(6)

        # --- Header ---
        header_label = QLabel("<b>DEFINE LAYERS</b>")
        self._theme_register("_theme_accent_labels", header_label)
        header_label.setStyleSheet(f"font-size: 10pt; color: {SemiconductorTheme.ACCENT_PRIMARY};")
        panel_layout.addWidget(header_label)

        # --- Mode Switcher ---
        mode_row = QHBoxLayout()
        mode_row.setSpacing(16)
        self.mode_test1_radio  = QRadioButton("Test Volume (1 Layer)")
        self.mode_single_radio = QRadioButton("Single Volume (Multi-Layer)")
        self.mode_multi_radio  = QRadioButton("Multi-Layer (sep_layer/)")
        self._theme_register("_theme_radio_buttons", self.mode_test1_radio)
        self._theme_register("_theme_radio_buttons", self.mode_single_radio)
        self._theme_register("_theme_radio_buttons", self.mode_multi_radio)
        self.mode_test1_radio.setChecked(True)
        self.mode_test1_radio.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt;")
        self.mode_single_radio.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt;")
        self.mode_multi_radio.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt;")
        mode_row.addWidget(self.mode_test1_radio)
        mode_row.addWidget(self.mode_single_radio)
        mode_row.addWidget(self.mode_multi_radio)
        mode_row.addStretch()
        panel_layout.addLayout(mode_row)

        self.mode_test1_radio.toggled.connect(self._on_input_mode_changed)
        self.mode_single_radio.toggled.connect(self._on_input_mode_changed)
        self.mode_multi_radio.toggled.connect(self._on_input_mode_changed)

        # --- Stacked widget: page 0 = test 1-layer, page 1 = single multi-layer, page 2 = multi-layer sep ---
        self.layer_mode_stack = QStackedWidget()

        # ---- PAGE 0: Test Volume (1 Layer) mode ----
        test1_page = QWidget()
        test1_layout = QVBoxLayout(test1_page)
        test1_layout.setContentsMargins(0, 0, 0, 0)
        test1_layout.setSpacing(6)

        test1_desc = QLabel(
            "Run the full DLL pipeline on the entire loaded volume as a single layer.\n"
            "No splitting is needed — ideal for quick testing and validation."
        )
        self._theme_register("_theme_secondary_labels", test1_desc)
        test1_desc.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        test1_desc.setWordWrap(True)
        test1_layout.addWidget(test1_desc)

        # Volume info group
        vol_info_group = QGroupBox("Volume Info")
        self._theme_register("_theme_group_boxes", vol_info_group)
        vol_info_group.setStyleSheet(self._group_box_qss())
        vol_info_inner = QVBoxLayout(vol_info_group)
        vol_info_inner.setContentsMargins(8, 14, 8, 8)
        vol_info_inner.setSpacing(4)

        self.test1_shape_lbl = QLabel("Shape: — (load a volume)")
        self._theme_register("_theme_secondary_labels", self.test1_shape_lbl)
        self.test1_shape_lbl.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt;")
        vol_info_inner.addWidget(self.test1_shape_lbl)

        self.test1_zrange_lbl = QLabel("Z Range: —")
        self._theme_register("_theme_secondary_labels", self.test1_zrange_lbl)
        self.test1_zrange_lbl.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt;")
        vol_info_inner.addWidget(self.test1_zrange_lbl)

        self.test1_mem_lbl = QLabel("Memory: —")
        self._theme_register("_theme_secondary_labels", self.test1_mem_lbl)
        self.test1_mem_lbl.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt;")
        vol_info_inner.addWidget(self.test1_mem_lbl)

        self.test1_dtype_lbl = QLabel("Data Type: —")
        self._theme_register("_theme_secondary_labels", self.test1_dtype_lbl)
        self.test1_dtype_lbl.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt;")
        vol_info_inner.addWidget(self.test1_dtype_lbl)

        test1_layout.addWidget(vol_info_group)

        # Status label
        self.test1_status_lbl = QLabel("Ready — Load DLL, Config, and Volume, then click SEGMENTATION.")
        self._theme_register("_theme_status_labels", self.test1_status_lbl)
        self.test1_status_lbl.setStyleSheet(self._status_label_qss())
        self.test1_status_lbl.setWordWrap(True)
        test1_layout.addWidget(self.test1_status_lbl)

        test1_layout.addStretch()

        # Apply to Online
        self.test1_apply_online_btn = QPushButton("Apply for Online Mode")
        self._theme_register("_theme_primary_buttons", self.test1_apply_online_btn)
        self.test1_apply_online_btn.setStyleSheet(
            f"background-color: {SemiconductorTheme.ACCENT_PRIMARY}; color: {SemiconductorTheme.BG_DARK}; "
            f"font-weight: bold; padding: 6px; border-radius: 4px;"
        )
        self.test1_apply_online_btn.clicked.connect(self.roi_apply_online)
        test1_layout.addWidget(self.test1_apply_online_btn)

        self.layer_mode_stack.addWidget(test1_page)   # index 0

        # ---- PAGE 1: Single Volume (Multi-Layer) mode (existing UI) ----
        single_page = QWidget()
        single_layout = QVBoxLayout(single_page)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.setSpacing(6)

        desc_label = QLabel("Split the entire volume into layers for detailed analysis.")
        self._theme_register("_theme_secondary_labels", desc_label)
        desc_label.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        desc_label.setWordWrap(True)
        single_layout.addWidget(desc_label)

        # --- Generation Row ---
        gen_row = QHBoxLayout()
        gen_row.setSpacing(8)
        num_lbl = QLabel("Layers:")
        self._theme_register("_theme_secondary_labels", num_lbl)
        num_lbl.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        gen_row.addWidget(num_lbl)
        
        self.layer_spinbox = NoScrollSpinBox()
        self.layer_spinbox.setRange(1, 20)
        self.layer_spinbox.setValue(4)
        gen_row.addWidget(self.layer_spinbox)
        
        self.layer_gen_btn = QPushButton("Auto-Split")
        self._theme_register("_theme_secondary_buttons", self.layer_gen_btn)
        self.layer_gen_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {SemiconductorTheme.BTN_SECONDARY_BG};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                font-weight: bold; font-size: 8pt;
                padding: 4px 10px;
                border-radius: 4px; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QPushButton:disabled {{
                opacity: 0.5; color: {SemiconductorTheme.TEXT_DISABLED};
            }}
            QPushButton:hover {{ background-color: {SemiconductorTheme.BG_LIGHT}; }}
        """)
        self.layer_gen_btn.clicked.connect(self.generate_layers)
        gen_row.addWidget(self.layer_gen_btn)
        gen_row.addStretch()
        single_layout.addLayout(gen_row)

        # --- Table ---
        self.layer_table = QTableWidget()
        self._theme_register("_theme_tables", self.layer_table)
        self.layer_table.setColumnCount(4)
        self.layer_table.setHorizontalHeaderLabels(["", "Name", "Z Start", "Z End"])
        self.layer_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.layer_table.setColumnWidth(0, 30)
        self.layer_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.layer_table.verticalHeader().setVisible(False)
        self.layer_table.setStyleSheet(f"""
            QTableWidget {{
                background: {SemiconductorTheme.BG_DARK}; color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; font-size: 9pt;
            }}
            QHeaderView::section {{
                background: {SemiconductorTheme.BG_PANEL}; color: {SemiconductorTheme.ACCENT_PRIMARY};
                font-size: 8pt; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """)
        self.layer_table.itemChanged.connect(self._on_layer_table_changed)
        single_layout.addWidget(self.layer_table, 1)

        # --- Actions Row ---
        action_row = QHBoxLayout()
        self.layer_sel_all = QPushButton("All")
        self.layer_sel_all.clicked.connect(lambda: self._set_all_layers(True))
        self.layer_sel_none = QPushButton("None")
        self.layer_sel_none.clicked.connect(lambda: self._set_all_layers(False))
        self.layer_clear = QPushButton("Clear")
        self.layer_clear.clicked.connect(self.clear_layers)
        self.layer_export_btn = QPushButton("Export Volumes")
        self._theme_register("_theme_secondary_buttons", self.layer_sel_all)
        self._theme_register("_theme_secondary_buttons", self.layer_sel_none)
        self._theme_register("_theme_secondary_buttons", self.layer_clear)
        self._theme_register("_theme_primary_buttons", self.layer_export_btn)
        self.layer_export_btn.clicked.connect(self.export_volumes_by_layer)
        self.layer_export_btn.setStyleSheet(f"""
            QPushButton {{
                background: {SemiconductorTheme.ACCENT_PRIMARY};
                color: {SemiconductorTheme.BG_DARK};
                border-radius: 3px;
                font-size: 8pt;
                font-weight: bold;
                padding: 2px 8px;
            }}
            QPushButton:hover {{ opacity: 0.85; }}
            QPushButton:disabled {{ background: {SemiconductorTheme.BG_LIGHT}; color: {SemiconductorTheme.TEXT_DISABLED}; }}
        """)
        
        for b in [self.layer_sel_all, self.layer_sel_none, self.layer_clear]:
            b.setStyleSheet(f"background: {SemiconductorTheme.BTN_SECONDARY_BG}; border-radius: 3px; font-size: 8pt;")
            action_row.addWidget(b)
        action_row.addStretch()
        action_row.addWidget(self.layer_export_btn)
        single_layout.addLayout(action_row)

        # --- Status & Apply Online ---
        self.layer_status_lbl = QLabel("0 layers defined  |  ~0.0 GB")
        self._theme_register("_theme_secondary_labels", self.layer_status_lbl)
        self.layer_status_lbl.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        single_layout.addWidget(self.layer_status_lbl)

        # Apply to Online
        self.apply_layers_online_btn = QPushButton("Apply Layers for Online Mode")
        self._theme_register("_theme_primary_buttons", self.apply_layers_online_btn)
        self.apply_layers_online_btn.setStyleSheet(f"background-color: {SemiconductorTheme.ACCENT_PRIMARY}; color: {SemiconductorTheme.BG_DARK}; font-weight: bold; padding: 6px; border-radius: 4px;")
        self.apply_layers_online_btn.clicked.connect(self.roi_apply_online)
        single_layout.addWidget(self.apply_layers_online_btn)

        self.layer_mode_stack.addWidget(single_page)   # index 1

        # ---- PAGE 2: Multi-Layer (sep_layer/) mode ----
        multi_page = QWidget()
        multi_layout = QVBoxLayout(multi_page)
        multi_layout.setContentsMargins(0, 0, 0, 0)
        multi_layout.setSpacing(6)

        ml_desc = QLabel(
            "Load multiple pre-split TIFF files from a <b>sep_layer/</b> folder.\n"
            "Each file becomes one layer. Files are sorted alphabetically."
        )
        self._theme_register("_theme_secondary_labels", ml_desc)
        ml_desc.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        ml_desc.setWordWrap(True)
        multi_layout.addWidget(ml_desc)

        # Browse row
        browse_row = QHBoxLayout()
        self.ml_folder_input = QLineEdit()
        self._theme_register("_theme_inputs", self.ml_folder_input)
        self.ml_folder_input.setReadOnly(True)
        self.ml_folder_input.setPlaceholderText("Select folder containing sep_layer/ or the sep_layer/ folder itself...")
        self.ml_folder_input.setStyleSheet(f"background: {SemiconductorTheme.BG_DARK}; color: {SemiconductorTheme.TEXT_PRIMARY}; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; border-radius: 3px; padding: 2px 6px;")
        browse_row.addWidget(self.ml_folder_input, 1)

        ml_browse_btn = QPushButton("Browse...")
        ml_browse_btn.setStyleSheet(f"background: {SemiconductorTheme.BTN_SECONDARY_BG}; color: {SemiconductorTheme.TEXT_PRIMARY}; border-radius: 3px; font-size: 8pt; padding: 4px 10px;")
        ml_browse_btn.clicked.connect(self._browse_sep_layer_folder)
        browse_row.addWidget(ml_browse_btn)
        multi_layout.addLayout(browse_row)

        # File list
        ml_list_lbl = QLabel("Detected layer files (drag to reorder):")
        self._theme_register("_theme_secondary_labels", ml_list_lbl)
        ml_list_lbl.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        multi_layout.addWidget(ml_list_lbl)

        self.multi_layer_list = QListWidget()
        self._theme_register("_theme_lists", self.multi_layer_list)
        self.multi_layer_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.multi_layer_list.setStyleSheet(f"""
            QListWidget {{
                background: {SemiconductorTheme.BG_DARK};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                font-size: 9pt;
            }}
            QListWidget::item:selected {{ background: {SemiconductorTheme.ACCENT_PRIMARY}; color: {SemiconductorTheme.BG_DARK}; }}
        """)
        multi_layout.addWidget(self.multi_layer_list, 1)
        self.multi_layer_list.currentItemChanged.connect(self._on_ml_layer_clicked)

        # Multi-layer action row
        ml_action_row = QHBoxLayout()
        ml_remove_btn = QPushButton("Remove Selected")
        ml_remove_btn.setStyleSheet(f"background: {SemiconductorTheme.BTN_SECONDARY_BG}; border-radius: 3px; font-size: 8pt;")
        ml_remove_btn.clicked.connect(self._remove_selected_ml_file)
        ml_action_row.addWidget(ml_remove_btn)
        ml_action_row.addStretch()

        self.ml_status_lbl = QLabel("No files loaded")
        self._theme_register("_theme_secondary_labels", self.ml_status_lbl)
        self.ml_status_lbl.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        ml_action_row.addWidget(self.ml_status_lbl)
        multi_layout.addLayout(ml_action_row)

        self.layer_mode_stack.addWidget(multi_page)    # index 2

        panel_layout.addWidget(self.layer_mode_stack, 1)
        panel_layout.addStretch()
        return panel


    # ==================== MULTI-LAYER INPUT MODE HANDLERS ====================

    def _on_input_mode_changed(self, checked):
        """Toggle between test-1-layer, single volume, and multi-layer input pages."""
        if not checked:
            return  # Ignore unchecked signals
        if self.mode_test1_radio.isChecked():
            self.input_mode = 'test_1layer'
            self.layers_active = False
            self.layer_mode_stack.setCurrentIndex(0)
            self._update_test1_volume_info()
        elif self.mode_single_radio.isChecked():
            self.input_mode = 'single'
            self.layer_mode_stack.setCurrentIndex(1)
        else:
            self.input_mode = 'multi_layer'
            self.layer_mode_stack.setCurrentIndex(2)
        self.check_ready_state()

    def _update_test1_volume_info(self):
        """Update the volume info labels in the Test Volume (1 Layer) panel."""
        if self.volume_data is None:
            self.test1_shape_lbl.setText("Shape: — (load a volume)")
            self.test1_zrange_lbl.setText("Z Range: —")
            self.test1_mem_lbl.setText("Memory: —")
            self.test1_dtype_lbl.setText("Data Type: —")
            self.test1_status_lbl.setText("Ready — Load DLL, Config, and Volume, then click SEGMENTATION.")
            return
        z, y, x = self.volume_data.shape
        mem_bytes = self.volume_data.nbytes
        mem_gb = mem_bytes / (1024 ** 3)
        self.test1_shape_lbl.setText(f"Shape: {z} × {y} × {x}  (Z × Y × X)")
        self.test1_zrange_lbl.setText(f"Z Range: 0 → {z}  ({z} slices)")
        self.test1_mem_lbl.setText(f"Memory: ~{mem_gb:.2f} GB  ({mem_bytes:,} bytes)")
        self.test1_dtype_lbl.setText(f"Data Type: {self.volume_data.dtype}")
        checklist = []
        if self.dll_path: checklist.append("✓ DLL")
        else: checklist.append("✗ DLL")
        if self.config: checklist.append("✓ Config")
        else: checklist.append("✗ Config")
        checklist.append("✓ Volume")
        all_ok = self.dll_path is not None and self.config is not None
        if all_ok:
            self.test1_status_lbl.setText(f"{'  |  '.join(checklist)}  —  Ready! Click SEGMENTATION to run.")
        else:
            self.test1_status_lbl.setText(f"{'  |  '.join(checklist)}  —  Missing prerequisites.")

    def _browse_sep_layer_folder(self):
        """Open folder dialog and scan for TIFF files in sep_layer/ subfolder."""
        start_dir = ""
        if hasattr(self, 'config_path') and self.config_path:
            start_dir = os.path.dirname(self.config_path)

        folder = QFileDialog.getExistingDirectory(
            self, "Select sep_layer/ folder or its parent folder", start_dir
        )
        if not folder:
            return

        # Auto-detect sep_layer/ subfolder
        sep_path = os.path.join(folder, "sep_layer")
        if os.path.isdir(sep_path):
            target = sep_path
        elif os.path.basename(folder).lower() == "sep_layer":
            target = folder
        else:
            # Show contents and let user know
            target = folder

        self.ml_folder_input.setText(target)

        # Scan for TIFF files
        tif_files = sorted(
            [f for f in os.listdir(target) if f.lower().endswith(('.tif', '.tiff'))]
        )

        if not tif_files:
            QMessageBox.warning(self, "No TIFF Files",
                f"No .tif/.tiff files found in:\n{target}")
            return

        # Populate list widget and internal store
        self.multi_layer_list.clear()
        self.multi_layer_files = []
        for fname in tif_files:
            fpath = os.path.join(target, fname)
            name = os.path.splitext(fname)[0]
            self.multi_layer_files.append({'name': name, 'path': fpath})
            item = QListWidgetItem(f"{name}    {fname}")
            item.setData(Qt.UserRole, fpath)
            self.multi_layer_list.addItem(item)

        self.ml_status_lbl.setText(f"{len(tif_files)} layer files detected")
        self.info_label.setText(f"Multi-layer mode: {len(tif_files)} files from {target}")
        self.check_ready_state()

    def _remove_selected_ml_file(self):
        """Remove selected items from the multi-layer file list."""
        for item in self.multi_layer_list.selectedItems():
            row = self.multi_layer_list.row(item)
            self.multi_layer_list.takeItem(row)
            if row < len(self.multi_layer_files):
                self.multi_layer_files.pop(row)
        self.ml_status_lbl.setText(f"{self.multi_layer_list.count()} layer files")
        self.check_ready_state()

    def _get_ordered_multi_layer_files(self):
        """Return multi_layer_files in the current list order (user may have reordered via drag)."""
        result = []
        for i in range(self.multi_layer_list.count()):
            item = self.multi_layer_list.item(i)
            fpath = item.data(Qt.UserRole)
            name = item.text().split("    ")[0].strip()
            result.append({'name': name, 'path': fpath})
        return result

    def _on_ml_layer_clicked(self, current, previous):
        """Handle user clicking a layer in the multi-layer file list."""
        if current is None:
            return
        if not hasattr(self, '_ml_layer_cache') or not self._ml_layer_cache:
            return  # No segmentation results yet
        layer_name = current.text().split("    ")[0].strip()
        self._switch_to_ml_layer(layer_name)

    def _switch_to_ml_layer(self, layer_name):
        """Instantly swap displayed data to a single cached layer (no file I/O)."""
        cache = getattr(self, '_ml_layer_cache', {})
        if layer_name not in cache:
            return

        entry = cache[layer_name]
        self.volume_data = entry['input']
        self.bump_segmentation = entry['bump']
        self.void_segmentation = entry['void']
        self.labeled_class1_data = entry.get('labeled_class1_data')
        
        # Sync the dropdown
        if hasattr(self, 'layer_filter_combo'):
            idx = self.layer_filter_combo.findData(layer_name)
            if idx >= 0 and self.layer_filter_combo.currentIndex() != idx:
                self.layer_filter_combo.setCurrentIndex(idx)
        self.current_display_layer = layer_name

        if self.volume_data is not None:
            z, y, x = self.volume_data.shape
        elif self.bump_segmentation is not None:
            z, y, x = self.bump_segmentation.shape
        elif self.void_segmentation is not None:
            z, y, x = self.void_segmentation.shape
        else:
            return

        # Update sliders range and position
        self.current_slices = {'axial': z // 2, 'coronal': y // 2, 'sagittal': x // 2}

        self.axial_slice_slider.setMaximum(z - 1)
        self.axial_slice_slider.setValue(z // 2)
        self.axial_slice_label.setText(f"{z // 2} / {z - 1}")

        self.coronal_slice_slider.setMaximum(y - 1)
        self.coronal_slice_slider.setValue(y // 2)
        self.coronal_slice_label.setText(f"{y // 2} / {y - 1}")

        self.sagittal_slice_slider.setMaximum(x - 1)
        self.sagittal_slice_slider.setValue(x // 2)
        self.sagittal_slice_label.setText(f"{x // 2} / {x - 1}")

        # Refresh all 3 plane views
        for orientation in ['axial', 'coronal', 'sagittal']:
            self.update_plane_view(orientation)

        # Filter measurement table to show only this layer's objects
        self._filter_stats_table_by_layer(layer_name)

        self.info_label.setText(f"Viewing: {layer_name}  ({z}{y}{x})")

    def _filter_stats_table_by_layer(self, layer_name):
        """Show/hide rows in the stats table to display only the given layer.
        Uses row visibility for O(n) performance  no table rebuild needed."""
        if not hasattr(self, 'object_stats') or not self.object_stats:
            return
        # Store full stats for Measurement/CSV export (never modified)
        if not hasattr(self, '_ml_all_object_stats'):
            self._ml_all_object_stats = self.object_stats

        visible_count = 0
        for row in range(self.object_stats_table.rowCount()):
            layer_item = self.object_stats_table.item(row, 1)  # Column 1 = Layer
            if layer_item and layer_item.text() == layer_name:
                self.object_stats_table.setRowHidden(row, False)
                visible_count += 1
            else:
                self.object_stats_table.setRowHidden(row, True)

        self.stats_info_label.setText(
            f"Layer: {layer_name} | {visible_count} objects"
        )

    def generate_layers(self):
        if self.volume_data is None:
            QMessageBox.warning(self, "Warning", "Please load a volume first.")
            return
            
        n = self.layer_spinbox.value()
        z_max = self.volume_data.shape[0]
        step = z_max // n
        
        self.layer_definitions.clear()
        for i in range(n):
            z_s = i * step
            z_e = z_max if i == n - 1 else (i + 1) * step
            self.layer_definitions.append({
                'id': i, 'name': f"Layer {i+1}", 'z_start': z_s, 'z_end': z_e, 'selected': True
            })
            
        self.layers_active = True
        self._populate_layer_table()
        
    def _populate_layer_table(self):
        self.layer_table.blockSignals(True)
        self.layer_table.setRowCount(len(self.layer_definitions))
        
        for r, l in enumerate(self.layer_definitions):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked if l['selected'] else Qt.Unchecked)
            self.layer_table.setItem(r, 0, chk)
            
            self.layer_table.setItem(r, 1, QTableWidgetItem(l['name']))
            
            z_s = QTableWidgetItem(str(l['z_start']))
            self.layer_table.setItem(r, 2, z_s)
            
            z_e = QTableWidgetItem(str(l['z_end']))
            self.layer_table.setItem(r, 3, z_e)
            
        self.layer_table.blockSignals(False)
        self._update_layer_status()

    def _on_layer_table_changed(self, item):
        row = item.row()
        col = item.column()
        if row >= len(self.layer_definitions): return
        
        layer = self.layer_definitions[row]
        if col == 0:
            layer['selected'] = (item.checkState() == Qt.Checked)
        elif col == 1:
            layer['name'] = item.text()
        elif col == 2 or col == 3:
            try:
                val = int(item.text())
                if col == 2: layer['z_start'] = max(0, val)
                else: layer['z_end'] = val
            except ValueError:
                pass # Invalid int, keep old value silently
        
        self._update_layer_status()

    def _set_all_layers(self, state):
        for l in self.layer_definitions:
            l['selected'] = state
        self._populate_layer_table()

    def clear_layers(self):
        self.layer_definitions.clear()
        self.layers_active = False
        self._populate_layer_table()

    def export_volumes_by_layer(self):
        """Export sub-volume slices for each selected layer as individual multi-page TIFF files."""
        if self.volume_data is None:
            QMessageBox.warning(self, "Export Volumes", "No volume loaded. Please load a volume first.")
            return
        if not self.layer_definitions:
            QMessageBox.information(self, "Export Volumes", "No layers defined. Please generate layers first.")
            return

        selected_layers = [l for l in self.layer_definitions if l['selected']]
        if not selected_layers:
            QMessageBox.information(self, "Export Volumes", "No layers selected. Please check at least one layer.")
            return

        # Choose output directory
        default_dir = ""
        if hasattr(self, 'config_path') and self.config_path:
            default_dir = os.path.dirname(self.config_path)

        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Layer Volumes", default_dir
        )
        if not out_dir:
            return

        # Run export in background thread to avoid UI freeze
        self._layer_export_thread = _ExportLayerVolumesThread(
            self.volume_data, selected_layers, out_dir
        )

        self._export_progress = QProgressDialog(
            "Exporting volumes...", "Cancel", 0, len(selected_layers), self
        )
        self._export_progress.setWindowTitle("Export Volumes by Layer")
        self._export_progress.setWindowModality(Qt.WindowModal)
        self._export_progress.setMinimumDuration(0)
        self._export_progress.canceled.connect(self._layer_export_thread.cancel)
        self._export_progress.show()

        self._layer_export_thread.progress.connect(
            lambda i, name: (
                self._export_progress.setValue(i),
                self._export_progress.setLabelText(f"Saving: {name}")
            )
        )
        self._layer_export_thread.finished.connect(self._on_layer_export_done)
        self._layer_export_thread.start()

    def _on_layer_export_done(self, saved_files, errors, out_dir):
        """Called when _ExportLayerVolumesThread finishes."""
        if hasattr(self, '_export_progress') and self._export_progress:
            self._export_progress.close()

        if errors:
            err_msg = "\n".join(errors[:5])
            QMessageBox.warning(self, "Export Volumes",
                f"Export finished with {len(errors)} error(s):\n{err_msg}")
        elif saved_files:
            names = "\n".join(os.path.basename(f) for f in saved_files)
            QMessageBox.information(self, "Export Volumes Complete",
                f"Saved {len(saved_files)} volume(s) to:\n{out_dir}\n\nFiles:\n{names}")
        else:
            QMessageBox.information(self, "Export Volumes", "Export was cancelled.")

    def _update_layer_status(self):
        sel_count = sum(1 for l in self.layer_definitions if l['selected'])
        total_layers = len(self.layer_definitions)
        
        mem_mb = 0
        if self.volume_data is not None:
            dtype = self.volume_data.dtype
            _, h, w = self.volume_data.shape
            for l in self.layer_definitions:
                if l['selected']:
                    dz = l['z_end'] - l['z_start']
                    mem_mb += (w * h * dz * np.dtype(dtype).itemsize) / (1024 * 1024)
                    
        self.layer_status_lbl.setText(f"Selected: {sel_count}/{total_layers} layers  |  ~{mem_mb/1024:.2f} GB")

    # ==================== DEFINE ROI PANEL ====================

    def create_define_roi_panel(self):
        """Create the Define ROI panel with interactive rectangle selection controls."""
        panel = QWidget()
        self._theme_register("_theme_panels", panel)
        panel.setStyleSheet(f"background: {SemiconductorTheme.BG_PANEL}; border-radius: 6px;")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(6)

        # --- Header ---
        header_label = QLabel("<b>DEFINE ROI</b>")
        self._theme_register("_theme_accent_labels", header_label)
        header_label.setStyleSheet(f"font-size: 10pt; color: {SemiconductorTheme.ACCENT_PRIMARY};")
        panel_layout.addWidget(header_label)

        desc_label = QLabel("Draw a rectangle on the XY view to define a Region of Interest.")
        self._theme_register("_theme_secondary_labels", desc_label)
        desc_label.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        desc_label.setWordWrap(True)
        panel_layout.addWidget(desc_label)

        # --- ROI Selection Button ---
        self.roi_select_btn = QPushButton("ROI START")
        self._theme_register("_theme_primary_buttons", self.roi_select_btn)
        self.roi_select_btn.setCheckable(True)
        self.roi_select_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {SemiconductorTheme.ACCENT_PRIMARY};
                color: {SemiconductorTheme.BG_DARK};
                font-weight: bold; font-size: 10pt;
                padding: 8px 16px;
                border-radius: 5px;
                border: none;
            }}
            QPushButton:checked {{
                background-color: #e74c3c;
                color: white;
            }}
            QPushButton:hover {{
                opacity: 0.9;
            }}
        """)
        self.roi_select_btn.toggled.connect(self.toggle_roi_selection)
        panel_layout.addWidget(self.roi_select_btn)

        # --- ROI Coordinates (XY) + Z Range SIDE-BY-SIDE ---
        coord_z_row = QHBoxLayout()
        coord_z_row.setSpacing(6)

        group_style = f"""
            QGroupBox {{
                font-weight: bold; font-size: 9pt;
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
        """
        lbl_style = f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;"
        val_style = f"""
            background: {SemiconductorTheme.BG_DARK};
            color: {SemiconductorTheme.ACCENT_PRIMARY};
            border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            border-radius: 3px;
            padding: 3px 6px;
            font-size: 9pt; font-weight: bold;
        """

        # Left column: XY Coordinates
        coord_group = QGroupBox("XY Coordinates")
        self._theme_register("_theme_group_boxes", coord_group)
        coord_group.setStyleSheet(group_style)
        coord_layout = QGridLayout()
        coord_layout.setSpacing(4)
        for lbl_text, row, col in [("X1:", 0, 0), ("Y1:", 0, 2), ("X2:", 1, 0), ("Y2:", 1, 2)]:
            lbl = QLabel(lbl_text)
            self._theme_register("_theme_secondary_labels", lbl)
            lbl.setStyleSheet(lbl_style)
            coord_layout.addWidget(lbl, row, col)

        self.roi_x1_label = QLabel("--")
        self._theme_register("_theme_value_labels", self.roi_x1_label)
        self.roi_x1_label.setStyleSheet(val_style)
        coord_layout.addWidget(self.roi_x1_label, 0, 1)
        self.roi_y1_label = QLabel("--")
        self._theme_register("_theme_value_labels", self.roi_y1_label)
        self.roi_y1_label.setStyleSheet(val_style)
        coord_layout.addWidget(self.roi_y1_label, 0, 3)
        self.roi_x2_label = QLabel("--")
        self._theme_register("_theme_value_labels", self.roi_x2_label)
        self.roi_x2_label.setStyleSheet(val_style)
        coord_layout.addWidget(self.roi_x2_label, 1, 1)
        self.roi_y2_label = QLabel("--")
        self._theme_register("_theme_value_labels", self.roi_y2_label)
        self.roi_y2_label.setStyleSheet(val_style)
        coord_layout.addWidget(self.roi_y2_label, 1, 3)
        coord_group.setLayout(coord_layout)
        coord_z_row.addWidget(coord_group, 1)

        # Right column: Z Range
        z_group = QGroupBox("Z Slice Range")
        self._theme_register("_theme_group_boxes", z_group)
        z_group.setStyleSheet(group_style)
        z_layout = QGridLayout()
        z_layout.setSpacing(4)
        z_lbl_start = QLabel("Z Start:")
        self._theme_register("_theme_secondary_labels", z_lbl_start)
        z_lbl_start.setStyleSheet(lbl_style)
        z_layout.addWidget(z_lbl_start, 0, 0)
        self.roi_z_start_spin = NoScrollSpinBox()
        self.roi_z_start_spin.setRange(0, 0)
        self.roi_z_start_spin.setValue(0)
        self.roi_z_start_spin.valueChanged.connect(self.on_roi_z_changed)
        z_layout.addWidget(self.roi_z_start_spin, 0, 1)
        z_lbl_end = QLabel("Z End:")
        self._theme_register("_theme_secondary_labels", z_lbl_end)
        z_lbl_end.setStyleSheet(lbl_style)
        z_layout.addWidget(z_lbl_end, 1, 0)
        self.roi_z_end_spin = NoScrollSpinBox()
        self.roi_z_end_spin.setRange(0, 0)
        self.roi_z_end_spin.setValue(0)
        self.roi_z_end_spin.valueChanged.connect(self.on_roi_z_changed)
        z_layout.addWidget(self.roi_z_end_spin, 1, 1)
        z_group.setLayout(z_layout)
        coord_z_row.addWidget(z_group, 1)

        panel_layout.addLayout(coord_z_row)

        # --- Action Buttons Row: Undo / Redo / Clear / Full Volume ---
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        btn_style_secondary = f"""
            QPushButton {{
                background: {SemiconductorTheme.BTN_SECONDARY_BG};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 8pt;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QPushButton:hover {{
                background: {SemiconductorTheme.BG_LIGHT};
            }}
            QPushButton:disabled {{
                opacity: 0.4;
                color: {SemiconductorTheme.TEXT_SECONDARY};
            }}
        """
        self.roi_undo_btn = QPushButton("UNDO")
        self._theme_register("_theme_secondary_buttons", self.roi_undo_btn)
        self.roi_undo_btn.setStyleSheet(btn_style_secondary)
        self.roi_undo_btn.setEnabled(False)
        self.roi_undo_btn.clicked.connect(self.roi_undo)
        self._configure_symbol_button(self.roi_undo_btn, QStyle.SP_ArrowBack, "Undo ROI change")
        btn_layout.addWidget(self.roi_undo_btn)

        self.roi_redo_btn = QPushButton("REDO")
        self._theme_register("_theme_secondary_buttons", self.roi_redo_btn)
        self.roi_redo_btn.setStyleSheet(btn_style_secondary)
        self.roi_redo_btn.setEnabled(False)
        self.roi_redo_btn.clicked.connect(self.roi_redo)
        self._configure_symbol_button(self.roi_redo_btn, QStyle.SP_ArrowForward, "Redo ROI change")
        btn_layout.addWidget(self.roi_redo_btn)

        self.roi_clear_btn = QPushButton("CLEAR")
        self._theme_register("_theme_secondary_buttons", self.roi_clear_btn)
        self.roi_clear_btn.setStyleSheet(btn_style_secondary)
        self.roi_clear_btn.setEnabled(False)
        self.roi_clear_btn.clicked.connect(self.roi_clear)
        self._configure_symbol_button(self.roi_clear_btn, QStyle.SP_TrashIcon, "Clear ROI")
        btn_layout.addWidget(self.roi_clear_btn)

        self.roi_full_volume_btn = QPushButton("FULL VOL")
        self._theme_register("_theme_secondary_buttons", self.roi_full_volume_btn)
        self.roi_full_volume_btn.setStyleSheet(btn_style_secondary)
        self.roi_full_volume_btn.setEnabled(False)
        self.roi_full_volume_btn.setToolTip("Restore full volume (undo crop)")
        self.roi_full_volume_btn.clicked.connect(self.roi_restore_full_volume)
        self._configure_symbol_button(self.roi_full_volume_btn, QStyle.SP_BrowserReload, "Restore full volume")
        btn_layout.addWidget(self.roi_full_volume_btn)

        self.roi_full_xy_btn = QPushButton("FULL XY")
        self._theme_register("_theme_secondary_buttons", self.roi_full_xy_btn)
        self.roi_full_xy_btn.setStyleSheet(btn_style_secondary)
        self.roi_full_xy_btn.setToolTip("Set ROI to full XY plane of current volume")
        self.roi_full_xy_btn.clicked.connect(self.roi_set_full_xy)
        btn_layout.addWidget(self.roi_full_xy_btn)

        panel_layout.addLayout(btn_layout)

        # --- Apply Crop button ---
        self.roi_apply_crop_btn = QPushButton("Apply Crop to Views")
        self._theme_register("_theme_primary_buttons", self.roi_apply_crop_btn)
        self.roi_apply_crop_btn.setEnabled(False)
        self.roi_apply_crop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #2196F3;
                color: white;
                font-weight: bold; font-size: 9pt;
                padding: 7px 16px;
                border-radius: 5px;
                border: none;
            }}
            QPushButton:disabled {{
                background-color: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_SECONDARY};
            }}
            QPushButton:hover {{
                background-color: #1976D2;
            }}
        """)
        self.roi_apply_crop_btn.clicked.connect(self.roi_apply_crop)
        self._configure_symbol_button(self.roi_apply_crop_btn, QStyle.SP_DialogApplyButton, "Apply crop to views", text="Crop", size=(78, 30))
        panel_layout.addWidget(self.roi_apply_crop_btn)

        # --- Export Crop button ---
        self.roi_export_btn = QPushButton("EXPORT CROP")
        self._theme_register("_theme_success_buttons", self.roi_export_btn)
        self.roi_export_btn.setEnabled(False)
        self.roi_export_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {SemiconductorTheme.ACCENT_SUCCESS};
                color: black;
                font-weight: bold; font-size: 9pt;
                padding: 7px 16px;
                border-radius: 5px;
                border: none;
            }}
            QPushButton:disabled {{
                background-color: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_SECONDARY};
            }}
            QPushButton:hover {{
                opacity: 0.9;
            }}
        """)
        self.roi_export_btn.clicked.connect(self.roi_export_crop)
        self._configure_symbol_button(self.roi_export_btn, QStyle.SP_DialogSaveButton, "Export cropped volume", text="Save", size=(78, 30))
        panel_layout.addWidget(self.roi_export_btn)

        # --- Apply for Online Mode button ---
        self.roi_apply_online_btn = QPushButton("Apply for Online Mode")
        self._theme_register("_theme_primary_buttons", self.roi_apply_online_btn)
        self.roi_apply_online_btn.setEnabled(False)
        self.roi_apply_online_btn.setToolTip("Lock this ROI and current parameters for standard Online processing")
        self.roi_apply_online_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #00897B;
                color: white;
                font-weight: bold; font-size: 10pt;
                padding: 10px 16px;
                border-radius: 6px;
                border: 2px solid #004D40;
                margin-top: 5px;
            }}
            QPushButton:disabled {{
                background-color: {SemiconductorTheme.BG_LIGHT};
                border-color: {SemiconductorTheme.BORDER_DEFAULT};
                color: {SemiconductorTheme.TEXT_SECONDARY};
            }}
            QPushButton:hover {{
                background-color: #00796B;
            }}
        """)
        self.roi_apply_online_btn.clicked.connect(self.roi_apply_online)
        self._configure_symbol_button(self.roi_apply_online_btn, QStyle.SP_DialogApplyButton, "Apply ROI for Online mode", text="Online", size=(92, 34))
        panel_layout.addWidget(self.roi_apply_online_btn)

        # --- Status Label ---
        self.roi_status_label = QLabel("No ROI defined")
        self._theme_register("_theme_status_labels", self.roi_status_label)
        self.roi_status_label.setStyleSheet(f"""
            background: {SemiconductorTheme.BG_DARK};
            color: {SemiconductorTheme.TEXT_SECONDARY};
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 8pt;
            border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
        """)
        self.roi_status_label.setWordWrap(True)
        panel_layout.addWidget(self.roi_status_label)

        panel_layout.addStretch()
        return panel

    # ==================== ROI INTERACTION LOGIC ====================

    def _create_large_cross_cursor(self):
        """Creates a custom large crosshair cursor for better visibility."""
        size = 48  
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, False)
        
        mid = size // 2
        
        # Black outer stroke for visibility on light backgrounds
        pen_bg = QPen(QColor(0, 0, 0), 3)
        painter.setPen(pen_bg)
        painter.drawLine(0, mid, size, mid)
        painter.drawLine(mid, 0, mid, size)
        
        # Bright neon primary color for the center
        color = QColor(SemiconductorTheme.ACCENT_PRIMARY)
        pen_fg = QPen(color, 1)
        painter.setPen(pen_fg)
        painter.drawLine(0, mid, size, mid)
        painter.drawLine(mid, 0, mid, size)
        
        painter.end()
        return QCursor(pixmap, mid, mid)

    def toggle_roi_selection(self, checked):
        """Toggle ROI selection mode on/off."""
        self.roi_selection_active = checked
        if checked:
            self.roi_select_btn.setText("ROI STOP")
            self.set_roi_status("ROI mode active - click and drag on XY view to draw a region", "info")
            if hasattr(self, 'axial_widget'):
                if not hasattr(self, '_large_cross_cursor'):
                    self._large_cross_cursor = self._create_large_cross_cursor()
                self.axial_widget.setCursor(self._large_cross_cursor)
        else:
            self.roi_select_btn.setText("ROI START")
            self.roi_drawing = False
            self.roi_start_point = None
            self.roi_temp_end = None
            if hasattr(self, 'axial_widget'):
                self.axial_widget.setCursor(Qt.ArrowCursor)
            if self.roi_manager.has_valid_roi:
                self.update_roi_status()
            else:
                self.set_roi_status("ROI selection stopped", "default")

    def set_roi_status(self, text, style="default"):
        """Set the ROI status label text with the given style type."""
        self.roi_status_label.setText(text)
        if style == "info":
            color = SemiconductorTheme.ACCENT_PRIMARY
            border = SemiconductorTheme.ACCENT_PRIMARY
        elif style == "success":
            color = SemiconductorTheme.ACCENT_SUCCESS
            border = SemiconductorTheme.ACCENT_SUCCESS
        elif style == "warning":
            color = "#e67e22"
            border = "#e67e22"
        else:
            color = SemiconductorTheme.TEXT_SECONDARY
            border = SemiconductorTheme.BORDER_DEFAULT
        self.roi_status_label.setStyleSheet(f"""
            background: {SemiconductorTheme.BG_LIGHT};
            color: {color};
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 8pt; font-weight: bold;
            border: 1px solid {border};
        """)

    def pick_world_coords(self, orientation, qt_pos):
        """Convert Qt mouse position to world coordinates on the given VTK view."""
        try:
            widget = getattr(self, f'{orientation}_widget')
            renderer = getattr(self, f'{orientation}_renderer')
            x, y = qt_pos.x(), qt_pos.y()
            size = widget.GetRenderWindow().GetSize()
            vtk_y = size[1] - y

            picker = vtk.vtkWorldPointPicker()
            picker.Pick(x, vtk_y, 0, renderer)
            world_pos = picker.GetPickPosition()
            return (world_pos[0], world_pos[1])
        except Exception:
            return None

    def get_resize_handle(self, world_coords):
        """Identify if a point is near an ROI edge or corner for resizing."""
        if self.roi_manager.current_roi is None:
            return None
            
        roi = self.roi_manager.current_roi
        x, y = world_coords
        x1, x2 = sorted([roi['x1'], roi['x2']])
        y1, y2 = sorted([roi['y1'], roi['y2']])
        
        # Threshold for grabbing (in world/pixel units)
        margin = 8 
        
        on_left = abs(x - x1) < margin
        on_right = abs(x - x2) < margin
        on_bottom = abs(y - y1) < margin
        on_top = abs(y - y2) < margin
        
        in_x = (x1 - margin) <= x <= (x2 + margin)
        in_y = (y1 - margin) <= y <= (y2 + margin)
        
        if on_left and on_top: return 'nw'
        if on_right and on_top: return 'ne'
        if on_left and on_bottom: return 'sw'
        if on_right and on_bottom: return 'se'
        if on_left and in_y: return 'w'
        if on_right and in_y: return 'e'
        if on_top and in_x: return 'n'
        if on_bottom and in_x: return 's'
        
        return None

    def handle_roi_mouse_press(self, pos):
        """Handle mouse press during ROI selection on axial (XY) view."""
        coords = self.pick_world_coords('axial', pos)
        if coords is None:
            return
            
        # Check if we are clicking on an existing ROI handle
        handle = self.get_resize_handle(coords)
        if handle and not self.roi_drawing:
            self.roi_resizing = True
            self.roi_resize_handle = handle
            roi = self.roi_manager.current_roi
            self.roi_resize_vars = []
            if 'n' in handle: self.roi_resize_vars.append('y2' if roi['y2'] >= roi['y1'] else 'y1')
            if 's' in handle: self.roi_resize_vars.append('y1' if roi['y1'] <= roi['y2'] else 'y2')
            if 'w' in handle: self.roi_resize_vars.append('x1' if roi['x1'] <= roi['x2'] else 'x2')
            if 'e' in handle: self.roi_resize_vars.append('x2' if roi['x2'] >= roi['x1'] else 'x1')
            self.set_roi_status(f"Resizing ROI...", "info")
            return

        # Otherwise, start drawing a new ROI
        self.roi_drawing = True
        self.roi_resizing = False
        self.roi_start_point = coords
        self.roi_temp_end = coords
        self.roi_x1_label.setText(str(int(coords[0])))
        self.roi_y1_label.setText(str(int(coords[1])))
        self.roi_x2_label.setText(str(int(coords[0])))
        self.roi_y2_label.setText(str(int(coords[1])))

    def handle_roi_mouse_move(self, pos, is_drag=False):
        """Handle mouse move/drag during ROI drawing or resizing."""
        coords = self.pick_world_coords('axial', pos)
        if coords is None:
            return

        if self.roi_drawing and is_drag:
            self.roi_temp_end = coords
            x1, y1 = self.roi_start_point
            x2, y2 = coords
            self.roi_x1_label.setText(str(int(min(x1, x2))))
            self.roi_y1_label.setText(str(int(min(y1, y2))))
            self.roi_x2_label.setText(str(int(max(x1, x2))))
            self.roi_y2_label.setText(str(int(max(y1, y2))))
            self.update_plane_view('axial', preserve_camera=True)
            
        elif self.roi_resizing and is_drag:
            roi = self.roi_manager.current_roi
            if not roi: return
            
            x, y = coords
            for var in self.roi_resize_vars:
                if 'x' in var: roi[var] = x
                if 'y' in var: roi[var] = y
                
            rx1, rx2 = sorted([roi['x1'], roi['x2']])
            ry1, ry2 = sorted([roi['y1'], roi['y2']])
            self.roi_x1_label.setText(str(int(rx1)))
            self.roi_y1_label.setText(str(int(ry1)))
            self.roi_x2_label.setText(str(int(rx2)))
            self.roi_y2_label.setText(str(int(ry2)))
            self.update_plane_view('axial', preserve_camera=True)
            
        else:
            # Hover state cursor update
            handle = self.get_resize_handle(coords)
            if handle:
                if handle in ['n', 's']: self.axial_widget.setCursor(Qt.SizeVerCursor)
                elif handle in ['e', 'w']: self.axial_widget.setCursor(Qt.SizeHorCursor)
                elif handle in ['nw', 'se']: self.axial_widget.setCursor(Qt.SizeFDiagCursor)
                elif handle in ['ne', 'sw']: self.axial_widget.setCursor(Qt.SizeBDiagCursor)
            else:
                if not hasattr(self, '_large_cross_cursor'):
                    self._large_cross_cursor = self._create_large_cross_cursor()
                self.axial_widget.setCursor(self._large_cross_cursor)

    def handle_roi_mouse_release(self, pos):
        """Handle mouse release to finalize ROI on axial (XY) view."""
        if self.roi_resizing:
            self.roi_resizing = False
            self.roi_resize_handle = None
            roi = self.roi_manager.current_roi
            # Commit the change to the manager to push to undo stack
            if roi:
                self.roi_manager.set_roi(
                    roi['x1'], roi['y1'], roi['x2'], roi['y2'],
                    roi['z_start'], roi['z_end']
                )
            self.update_roi_status()
            return

        if not self.roi_drawing or self.roi_start_point is None:
            self.roi_drawing = False
            return

        coords = self.pick_world_coords('axial', pos)
        if coords is None:
            self.roi_drawing = False
            return

        x1, y1 = self.roi_start_point
        x2, y2 = coords
        self.roi_drawing = False
        self.roi_start_point = None
        self.roi_temp_end = None

        # Validate minimum size
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            self.set_roi_status("ROI too small - please draw a larger rectangle", "warning")
            return

        # Determine source volume bounds
        source_vol = self._original_volume_data if hasattr(self, '_original_volume_data') and self._original_volume_data is not None else self.volume_data
        if source_vol is not None:
            vol_z, vol_y, vol_x = source_vol.shape
            x1_c = max(0, min(int(min(x1, x2)), vol_x))
            x2_c = max(0, min(int(max(x1, x2)), vol_x))
            y1_c = max(0, min(int(min(y1, y2)), vol_y))
            y2_c = max(0, min(int(max(y1, y2)), vol_y))
            z_start = self.roi_z_start_spin.value()
            z_end = self.roi_z_end_spin.value()
        else:
            x1_c, x2_c = int(min(x1, x2)), int(max(x1, x2))
            y1_c, y2_c = int(min(y1, y2)), int(max(y1, y2))
            z_start, z_end = 0, 0

        # Set ROI via manager (handles undo stack)
        self.roi_manager.set_roi(x1_c, y1_c, x2_c, y2_c, z_start, z_end)
        self.update_roi_ui_from_manager()
        # Re-render to show the persistent ROI overlay
        self.update_plane_view('axial', preserve_camera=True)
        roi = self.roi_manager.current_roi
        dx = roi['x2'] - roi['x1']
        dy = roi['y2'] - roi['y1']
        dz = roi['z_end'] - roi['z_start']
        dtype = self.volume_data.dtype if self.volume_data is not None else np.uint16
        mem_mb = self.roi_manager.estimate_memory_mb(dtype)
        self.set_roi_status(
            f"ROI defined: X[{roi['x1']}:{roi['x2']}] Y[{roi['y1']}:{roi['y2']}] Z[{roi['z_start']}:{roi['z_end']}]"
            f"  |  {dx}x{dy}x{dz}  |  ~{mem_mb:.1f} MB",
            "info"
        )

    def on_roi_z_changed(self):
        """Handle Z Start/End spinbox changes - uses debouncing to avoid excessive Undo steps."""
        # Reset and restart the timer (wait 500ms after last change to save to manager)
        self.roi_z_timer.start(500)
        
        # Immediate UI feedback for current user input
        self.update_roi_status()
        self.update_plane_view('axial', preserve_camera=True)

    def _commit_roi_z_change(self):
        """Commit current Z Start/End from UI to ROI Manager (saving to history)."""
        if self.roi_manager.current_roi is None:
            return
        
        z1 = self.roi_z_start_spin.value()
        z2 = self.roi_z_end_spin.value()
        
        # Update manager state -- this clears redo and pushes to undo stack
        self.roi_manager.update_z_range(z1, z2)
        
        # Update UI buttons state without forcing a data refresh
        self.roi_undo_btn.setEnabled(self.roi_manager.can_undo)
        self.roi_redo_btn.setEnabled(self.roi_manager.can_redo)
        self.roi_clear_btn.setEnabled(True)

    def roi_set_full_xy(self):
        """Set ROI to cover the entire current XY plane."""
        if self.volume_data is None: return
        vz, vy, vx = self.volume_data.shape
        z1 = self.roi_z_start_spin.value()
        z2 = self.roi_z_end_spin.value()
        self.roi_manager.set_roi(0, 0, vx, vy, z1, z2)
        self.update_roi_ui_from_manager() # Force sync to update status/buttons
        self.update_plane_view('axial', preserve_camera=True)

    def update_roi_ui_from_manager(self):
        """Sync the UI elements with the current ROI state from the manager."""
        roi = self.roi_manager.current_roi
        if roi:
            self.roi_x1_label.setText(str(roi['x1']))
            self.roi_y1_label.setText(str(roi['y1']))
            self.roi_x2_label.setText(str(roi['x2']))
            self.roi_y2_label.setText(str(roi['y2']))
            self.roi_z_start_spin.blockSignals(True)
            self.roi_z_end_spin.blockSignals(True)
            self.roi_z_start_spin.setValue(roi['z_start'])
            self.roi_z_end_spin.setValue(roi['z_end'])
            self.roi_z_start_spin.blockSignals(False)
            self.roi_z_end_spin.blockSignals(False)
        else:
            for lbl in [self.roi_x1_label, self.roi_y1_label, self.roi_x2_label, self.roi_y2_label]:
                lbl.setText("--")
            
            self.roi_z_start_spin.blockSignals(True)
            self.roi_z_end_spin.blockSignals(True)
            self.roi_z_start_spin.setValue(0)
            
            # Default to full volume height if no ROI is set
            if self.volume_data is not None:
                self.roi_z_end_spin.setValue(self.volume_data.shape[0])
            else:
                self.roi_z_end_spin.setValue(0)
                
            self.roi_z_start_spin.blockSignals(False)
            self.roi_z_end_spin.blockSignals(False)

        self.roi_undo_btn.setEnabled(self.roi_manager.can_undo)
        self.roi_redo_btn.setEnabled(self.roi_manager.can_redo)
        has_roi = self.roi_manager.has_valid_roi
        has_vol = self.volume_data is not None
        self.roi_clear_btn.setEnabled(has_roi or self.roi_manager.current_roi is not None)
        self.roi_apply_crop_btn.setEnabled(has_roi and has_vol)
        self.roi_export_btn.setEnabled(has_roi and has_vol)
        self.roi_apply_online_btn.setEnabled((has_roi or self.last_applied_roi_raw is not None) and self.config is not None)
        self.roi_full_volume_btn.setEnabled(hasattr(self, '_original_volume_data') and self._original_volume_data is not None)
        self.update_roi_status()

    def update_roi_status(self):
        """Update the ROI status label based on current UI input or manager state."""
        # Use current spinbox values for Z (more immediate while typing)
        z1 = self.roi_z_start_spin.value()
        z2 = self.roi_z_end_spin.value()
        dz = abs(z2 - z1)

        if self.roi_manager.current_roi is None:
            if self.volume_data is not None:
                vz, vy, vx = self.volume_data.shape
                self.set_roi_status(f"Z Range: {z1}-{z2} ({dz}) | Full Vol: {vz}x{vy}x{vx}", "default")
            else:
                self.set_roi_status(f"Z Range: {z1}-{z2} | No volume loaded", "default")
            return

        roi = self.roi_manager.current_roi
        dx = abs(roi['x2'] - roi['x1'])
        dy = abs(roi['y2'] - roi['y1'])
        
        # Check validity based on current UI + manager
        if dx == 0 or dy == 0:
            self.set_roi_status("ROI Invalid: XY region is empty. Draw on XY view.", "warning")
            return
        elif dz == 0:
            self.set_roi_status("ROI Invalid: Z range is 0. Adjust Start/End.", "warning")
            return
            
        # Valid ROI - calculate memory
        dtype = self.volume_data.dtype if self.volume_data is not None else np.uint16
        itemsize = np.dtype(dtype).itemsize
        mem_mb = (dx * dy * dz * itemsize) / (1024 * 1024)
        
        parts = [f"ROI: {dx}x{dy}x{dz}", f"~{mem_mb:.1f} MB"]
        if self.volume_data is not None:
            vz, vy, vx = self.volume_data.shape
            parts.append(f"View: {vz}x{vy}x{vx}")
        self.set_roi_status("  |  ".join(parts), "info")

        # Enable/Disable sync-dependent buttons
        has_vol = self.volume_data is not None
        is_valid = (dx > 0 and dy > 0 and dz > 0)
        self.roi_apply_crop_btn.setEnabled(is_valid and has_vol)
        self.roi_export_btn.setEnabled(is_valid and has_vol)

    def roi_undo(self):
        """Undo to previous ROI."""
        result = self.roi_manager.undo()
        if result is not None:
            self.update_roi_ui_from_manager()
            self.update_plane_view('axial', preserve_camera=True)
            self.set_roi_status(
                f"Undo: ROI X[{result['x1']}:{result['x2']}] Y[{result['y1']}:{result['y2']}]", "info"
            )

    def roi_redo(self):
        """Redo to next ROI."""
        result = self.roi_manager.redo()
        if result is not None:
            self.update_roi_ui_from_manager()
            self.update_plane_view('axial', preserve_camera=True)
            self.set_roi_status(
                f"Redo: ROI X[{result['x1']}:{result['x2']}] Y[{result['y1']}:{result['y2']}]", "info"
            )

    def roi_clear(self):
        """Clear all ROI state and overlay."""
        self.roi_manager.clear()
        self.roi_drawing = False
        self.roi_start_point = None
        self.roi_temp_end = None
        self.last_applied_roi_raw = None
        
        # Reset Z spins to full volume if available
        if self.volume_data is not None:
            self.roi_z_start_spin.blockSignals(True)
            self.roi_z_end_spin.blockSignals(True)
            self.roi_z_start_spin.setValue(0)
            self.roi_z_end_spin.setValue(self.volume_data.shape[0])
            self.roi_z_start_spin.blockSignals(False)
            self.roi_z_end_spin.blockSignals(False)
            
        self.update_roi_ui_from_manager()
        self.set_roi_status("ROI cleared", "default")
        self.update_plane_view('axial', preserve_camera=True)

    def roi_apply_online(self):
        """Apply current ROI and/or Layer recipe + Config to Online mode.

        Mass-production Online can lock:
          - XY/Z ROI crop (optional)
          - LAYER_N bands for INPUT_MODE=single (recommended for HBM stacks)
        Layers-only is allowed (no XY ROI) so full-FOV Z-band inspection works.
        """
        roi = self.roi_manager.current_roi
        
        # If current ROI is empty/invalid, fall back to last crop ROI if any
        has_xy_roi = bool(self.roi_manager.has_valid_roi)
        if not has_xy_roi and self.last_applied_roi_raw:
            roi = self.last_applied_roi_raw
            has_xy_roi = bool(roi and 'x1' in roi)
            
        if self.config is None:
            QMessageBox.warning(self, "Config Warning", "No configuration found. Please load a config file first.")
            return

        selected_layers = []
        if self.layers_active:
            selected_layers = [l for l in self.layer_definitions if l.get('selected')]

        if not has_xy_roi and not selected_layers:
            QMessageBox.warning(
                self, "Warning",
                "Nothing to lock for Online.\n\n"
                "Define an XY/Z ROI and/or select LAYER bands (INPUT_MODE=single),\n"
                "then click Apply again."
            )
            return

        # 1. Confirm application
        parts = []
        if has_xy_roi:
            parts.append("XY/Z ROI crop")
        if selected_layers:
            parts.append(f"{len(selected_layers)} layer band(s)")
        reply = QMessageBox.question(
            self, 'Confirm Online Recipe',
            "Apply this recipe to Online Mode?\n\n"
            f"  · {', '.join(parts)}\n\n"
            "Host volumes will use this recipe for crop + segmentation.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply == QMessageBox.No:
            return

        # Collect recipe
        roi_to_send = roi.copy() if (has_xy_roi and roi) else {}
        if selected_layers:
            roi_to_send['layers'] = [dict(l) for l in selected_layers]
                
        if not roi_to_send:
            QMessageBox.warning(self, "Warning", "No ROI or Layers defined to apply.")
            return
        
        self.apply_online_roi_signal.emit(roi_to_send, self.config, self.config_path or "")
        
        status = "ONLINE recipe locked"
        if selected_layers:
            status += f" · {len(selected_layers)} layers"
        if has_xy_roi:
            status += " · ROI crop"
        self.set_roi_status(status, "success")
        QMessageBox.information(
            self, "Success!",
            "Recipe locked for Online Mode.\n"
            "Enable the ONLINE toggle in the sidebar to begin receiving data."
        )

    def roi_apply_crop(self):
        """Apply the ROI crop to all 3 views  replaces volume_data with cropped sub-volume."""
        if not self.roi_manager.has_valid_roi or self.volume_data is None:
            QMessageBox.warning(self, "Warning", "No valid ROI or volume to crop.")
            return

        # Save original volume if not yet saved
        if not hasattr(self, '_original_volume_data') or self._original_volume_data is None:
            self._original_volume_data = self.volume_data

        # Crop from the original volume
        source = self._original_volume_data
        roi = self.roi_manager.current_roi
        
        # Ensure coordinates are sorted and converted to integers for slicing
        z_coords = sorted([roi['z_start'], roi['z_end']])
        y_coords = sorted([roi['y1'], roi['y2']])
        x_coords = sorted([roi['x1'], roi['x2']])
        
        z1 = int(max(0, z_coords[0]))
        z2 = int(min(source.shape[0], z_coords[1]))
        y1 = int(max(0, y_coords[0]))
        y2 = int(min(source.shape[1], y_coords[1]))
        x1 = int(max(0, x_coords[0]))
        x2 = int(min(source.shape[2], x_coords[1]))

        cropped = source[z1:z2, y1:y2, x1:x2].copy()
        if cropped.size == 0:
            QMessageBox.warning(self, "Warning", "Cropped region is empty.")
            return

        # Replace current volume_data
        self.volume_data = cropped
        cz, cy, cx = cropped.shape

        # Update sliders
        self.current_slices = {'axial': cz // 2, 'coronal': cy // 2, 'sagittal': cx // 2}
        for orientation, dim in [('axial', cz), ('coronal', cy), ('sagittal', cx)]:
            slider = getattr(self, f'{orientation}_slice_slider')
            label = getattr(self, f'{orientation}_slice_label')
            slider.blockSignals(True)
            slider.setMaximum(dim - 1)
            slider.setValue(dim // 2)
            slider.blockSignals(False)
            label.setText(f"{dim // 2} / {dim - 1}")

        # Recalculate window/level for cropped data; first CACHE the original W/L for instant restore
        # Must cache BEFORE overwriting self.window_level
        if not hasattr(self, '_original_window_level') or self._original_window_level is None:
            self._original_window_level = {o: self.window_level.get(o, (255, 127.5))
                                            for o in ['axial', 'coronal', 'sagittal']}

        if cropped.dtype == np.uint16:
            data_min = float(np.percentile(cropped, 1))
            data_max = float(np.percentile(cropped, 99))
            window = data_max - data_min
            level = (data_max + data_min) / 2
            wl = (window, level)
        else:
            wl = (255, 127.5)
        for orientation in ['axial', 'coronal', 'sagittal']:
            self.window_level[orientation] = wl

        # Clear segmentation masks (they no longer match)
        self.bump_segmentation = None
        self.void_segmentation = None

        # --- Save cropped volume to a temp TIFF so DLL run_inspection can use it ---
        import tempfile, os
        try:
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, 'roi_crop_volume.tif')
            from skimage import io as skio
            skio.imsave(tmp_path, cropped)
            # Patch config inputPath so the DLL reads the cropped volume
            if self.config is not None:
                self._original_config_input_path = self.config.inputPath
                self.config.inputPath = tmp_path.encode('utf-8')
        except Exception as e:
            print(f"[ROI] Warning: failed to write temp TIFF for DLL: {e}")

        # Cache the current ROI as the 'last applied' before clearing, so Online mode can use it
        self.last_applied_roi_raw = self.roi_manager.current_roi.copy()
        
        # Clear ROI overlay so it doesn't appear on the cropped views
        # (ROI coords are in original-volume space; after crop they don't map correctly)
        self.roi_manager.clear()
        # Re-init the ROI z-range for the cropped volume
        if hasattr(self, 'roi_z_start_spin'):
            self.roi_z_start_spin.setRange(0, cz - 1)
            self.roi_z_start_spin.setValue(0)
        if hasattr(self, 'roi_z_end_spin'):
            self.roi_z_end_spin.setRange(0, cz)
            self.roi_z_end_spin.setValue(cz)

        # Refresh all views
        for orientation in ['axial', 'coronal', 'sagittal']:
            self.update_plane_view(orientation)

        self.roi_full_volume_btn.setEnabled(True)
        self.roi_apply_crop_btn.setEnabled(False)  # Can't re-apply until new ROI drawn
        self.roi_export_btn.setEnabled(True)        # Export button uses _original_volume_data
        self.info_label.setText(f"Cropped volume: {cz}x{cy}x{cx}")

        oz, oy, ox = self._original_volume_data.shape
        self.set_roi_status(
            f"Crop applied: {cz}x{cy}x{cx}  |  "
            f"X[{x1}:{x2}] Y[{y1}:{y2}] Z[{z1}:{z2}]  |  "
            f"Original: {oz}x{oy}x{ox}",
            "success"
        )

    def roi_restore_full_volume(self):
        """Restore the original full volume (undo crop). Uses cached window/level for speed."""
        if not hasattr(self, '_original_volume_data') or self._original_volume_data is None:
            return

        self.volume_data = self._original_volume_data
        self._original_volume_data = None
        z, y, x = self.volume_data.shape

        self.current_slices = {'axial': z // 2, 'coronal': y // 2, 'sagittal': x // 2}
        for orientation, dim in [('axial', z), ('coronal', y), ('sagittal', x)]:
            slider = getattr(self, f'{orientation}_slice_slider')
            label = getattr(self, f'{orientation}_slice_label')
            slider.blockSignals(True)
            slider.setMaximum(dim - 1)
            slider.setValue(dim // 2)
            slider.blockSignals(False)
            label.setText(f"{dim // 2} / {dim - 1}")

        # Restore cached window/level instead of re-computing percentiles (much faster)
        if hasattr(self, '_original_window_level') and self._original_window_level:
            for orientation in ['axial', 'coronal', 'sagittal']:
                self.window_level[orientation] = self._original_window_level.get(
                    orientation, self.window_level.get(orientation, (255, 127.5))
                )
            self._original_window_level = None
        elif self.volume_data.dtype == np.uint16:
            # Fallback: only compute if no cache
            data_min = float(np.percentile(self.volume_data, 1))
            data_max = float(np.percentile(self.volume_data, 99))
            window = data_max - data_min
            level = (data_max + data_min) / 2
            for orientation in ['axial', 'coronal', 'sagittal']:
                self.window_level[orientation] = (window, level)

        # Restore original DLL input path if it was patched
        if hasattr(self, '_original_config_input_path') and self._original_config_input_path is not None:
            if self.config is not None:
                self.config.inputPath = self._original_config_input_path
            self._original_config_input_path = None

        # Clean up temp TIFF if it exists
        import tempfile, os
        tmp_path = os.path.join(tempfile.gettempdir(), 'roi_crop_volume.tif')
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

        if hasattr(self, 'roi_z_start_spin'):
            self.roi_z_start_spin.setRange(0, z - 1)
            self.roi_z_start_spin.setValue(0)
        if hasattr(self, 'roi_z_end_spin'):
            self.roi_z_end_spin.setRange(0, z)
            self.roi_z_end_spin.setValue(z)

        self.bump_segmentation = None
        self.void_segmentation = None

        for orientation in ['axial', 'coronal', 'sagittal']:
            self.update_plane_view(orientation)

        self.roi_full_volume_btn.setEnabled(False)
        self.info_label.setText(f"Full volume restored: {z}x{y}x{x}")
        self.set_roi_status(f"Full volume restored: {z}x{y}x{x}", "success")
        self.update_roi_ui_from_manager()

    def roi_export_crop(self):
        """Export the cropped volume to .npy or .tiff file."""
        if not self.roi_manager.has_valid_roi or self.volume_data is None:
            QMessageBox.warning(self, "Warning", "No valid ROI defined or no volume loaded.")
            return

        save_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Cropped Volume", "",
            "NumPy Array (*.npy);;TIFF Stack (*.tif *.tiff)"
        )
        if not save_path:
            return

        if save_path.lower().endswith('.npy'):
            save_format = 'npy'
        elif save_path.lower().endswith(('.tif', '.tiff')):
            save_format = 'tiff'
        else:
            if 'npy' in selected_filter.lower():
                save_path += '.npy'
                save_format = 'npy'
            else:
                save_path += '.tif'
                save_format = 'tiff'

        # Use original volume if available, otherwise current
        source = self._original_volume_data if hasattr(self, '_original_volume_data') and self._original_volume_data is not None else self.volume_data
        dtype = source.dtype
        mem_mb = self.roi_manager.estimate_memory_mb(dtype)

        if mem_mb > 500:
            self.set_roi_status(f"Exporting {mem_mb:.0f} MB in background...", "warning")
            self.export_thread = ExportCropThread(
                source, self.roi_manager.current_roi.copy(),
                save_path, save_format
            )
            self.export_thread.finished.connect(self.on_export_finished)
            self.export_thread.start()
            self.roi_export_btn.setEnabled(False)
        else:
            try:
                sub_vol = self.roi_manager.get_crop(source)
                if sub_vol is None:
                    QMessageBox.warning(self, "Error", "Failed to crop volume.")
                    return

                if save_format == 'npy':
                    np.save(save_path, sub_vol)
                else:
                    from skimage import io as skio
                    skio.imsave(save_path, sub_vol)

                size_mb = sub_vol.nbytes / (1024 * 1024)
                self.set_roi_status(
                    f"Exported {Path(save_path).name} | Shape: {sub_vol.shape} | {size_mb:.1f} MB",
                    "success"
                )
                QMessageBox.information(self, "Export Complete",
                    f"Cropped volume saved!\n"
                    f"Shape: {sub_vol.shape}\n"
                    f"Size: {size_mb:.1f} MB\n"
                    f"Path: {save_path}")
            except Exception as e:
                self.set_roi_status(f"Export failed: {e}", "warning")
                QMessageBox.critical(self, "Export Error", str(e))

    def on_export_finished(self, success, message):
        """Handle export thread completion."""
        self.roi_export_btn.setEnabled(self.roi_manager.has_valid_roi)
        if success:
            self.set_roi_status(message, "success")
            QMessageBox.information(self, "Export Complete", message)
        else:
            self.set_roi_status("Export failed", "warning")
            QMessageBox.critical(self, "Export Error", message)

    def draw_roi_overlay(self, renderer, h, w):
        """Draw the ROI rectangle overlay on the axial (XY) view.
        Persists after drawing finishes as long as ROI is defined in manager."""
        if self.roi_drawing and self.roi_start_point is not None and self.roi_temp_end is not None:
            x1, y1 = self.roi_start_point
            x2, y2 = self.roi_temp_end
            # When drawing, we always show it on the current slice
        elif self.roi_manager.current_roi is not None:
            roi = self.roi_manager.current_roi
            # Use SpinBox values for Z visibility check (more immediate feedback while typing)
            z_start = self.roi_z_start_spin.value()
            z_end = self.roi_z_end_spin.value()
            z_min, z_max = sorted([z_start, z_end])
            
            curr_z = self.current_slices.get('axial', 0)
            if not (z_min <= curr_z < z_max):
                return
            x1, y1 = roi['x1'], roi['y1']
            x2, y2 = roi['x2'], roi['y2']
        else:
            return

        rx1, rx2 = min(x1, x2), max(x1, x2)
        ry1, ry2 = min(y1, y2), max(y1, y2)

        roi_h = int(ry2 - ry1)
        roi_w = int(rx2 - rx1)
        if roi_h < 1 or roi_w < 1:
            return

        overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        iy1 = max(0, int(ry1))
        iy2 = min(h, int(ry2))
        ix1 = max(0, int(rx1))
        ix2 = min(w, int(rx2))
        
        overlay_rgba[iy1:iy2, ix1:ix2, 0] = 50
        overlay_rgba[iy1:iy2, ix1:ix2, 1] = 150
        overlay_rgba[iy1:iy2, ix1:ix2, 2] = 255
        overlay_rgba[iy1:iy2, ix1:ix2, 3] = 60
        
        border_thickness = max(1, min(3, roi_h // 20, roi_w // 20))
        for t in range(border_thickness):
            if iy1 + t < h:
                overlay_rgba[iy1 + t, ix1:ix2, :3] = [0, 200, 255]
                overlay_rgba[iy1 + t, ix1:ix2, 3] = 220
            if iy2 - 1 - t >= 0 and iy2 - 1 - t < h:
                overlay_rgba[iy2 - 1 - t, ix1:ix2, :3] = [0, 200, 255]
                overlay_rgba[iy2 - 1 - t, ix1:ix2, 3] = 220
            if ix1 + t < w:
                overlay_rgba[iy1:iy2, ix1 + t, :3] = [0, 200, 255]
                overlay_rgba[iy1:iy2, ix1 + t, 3] = 220
            if ix2 - 1 - t >= 0 and ix2 - 1 - t < w:
                overlay_rgba[iy1:iy2, ix2 - 1 - t, :3] = [0, 200, 255]
                overlay_rgba[iy1:iy2, ix2 - 1 - t, 3] = 220
        
        # Draw corner handles for resizing feedback
        if not self.roi_drawing:
            h_size = 4
            for py, px in [(iy1, ix1), (iy1, ix2), (iy2, ix1), (iy2, ix2)]:
                y_s = max(0, py - h_size)
                y_e = min(h, py + h_size)
                x_s = max(0, px - h_size)
                x_e = min(w, px + h_size)
                overlay_rgba[y_s:y_e, x_s:x_e, :3] = [255, 255, 255]
                overlay_rgba[y_s:y_e, x_s:x_e, 3] = 255

        overlay_transposed = np.transpose(overlay_rgba, (1, 0, 2))
        flat_rgba = np.ascontiguousarray(overlay_transposed.reshape(-1, 4, order='F'))
        
        vtk_overlay = vtk.vtkImageData()
        vtk_overlay.SetDimensions(w, h, 1)
        vtk_overlay.SetOrigin(0.0, 0.0, 0.0)
        
        vtk_rgba = numpy_support.numpy_to_vtk(flat_rgba, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_rgba.SetNumberOfComponents(4)
        vtk_overlay.GetPointData().SetScalars(vtk_rgba)
        
        overlay_actor = vtk.vtkImageActor()
        overlay_actor.GetMapper().SetInputData(vtk_overlay)
        overlay_actor.SetPosition(0, 0, 0.2)
        # Nearest — keep ROI edges crisp (match MPR base image / 3D Viewer)
        try:
            overlay_actor.GetProperty().SetInterpolationTypeToNearest()
        except Exception:
            pass
        renderer.AddActor(overlay_actor)

    def _refresh_all_index_views(self):
        """Refresh all three views when Show Index checkbox changes."""
        for ori in ['axial', 'coronal', 'sagittal']:
            self.update_plane_view(ori, preserve_camera=True)

    def draw_object_labels(self, renderer, actual_z, orientation='axial', slice_idx=0):
        """Draw bump index labels on MPR — Viewer Online parity.

        UX (match 3D Viewer):
          · One identity per bump: ``(R,C)`` — no R#/C# edge rails that pile up
          · Dark plate + cyan text at centroid; amber when MES-selected
        """
        if not self.object_stats:
            return

        is_multi_layer = getattr(self, 'input_mode', 'single') == 'multi_layer'
        ml_z_starts = getattr(self, '_ml_layer_z_starts', {})
        single_z_offset = getattr(self, '_measurement_start_slice', 0) if not is_multi_layer else 0

        visible_stats = []
        for stat in self.object_stats:
            if is_multi_layer and self.current_display_layer is not None:
                if stat.get('layer_name') != self.current_display_layer:
                    continue

            if orientation == 'axial':
                if is_multi_layer and self.current_display_layer is None:
                    layer_z_start = ml_z_starts.get(stat.get('layer_name', ''), 0)
                else:
                    layer_z_start = 0
                local_z = actual_z - single_z_offset - layer_z_start
                if stat['z_min'] <= local_z < stat['z_max']:
                    visible_stats.append((stat, layer_z_start))
            elif orientation == 'coronal':
                if stat['y_min'] <= slice_idx < stat['y_max']:
                    if is_multi_layer and self.current_display_layer is None:
                        layer_z_start = ml_z_starts.get(stat.get('layer_name', ''), 0)
                    else:
                        layer_z_start = single_z_offset
                    visible_stats.append((stat, layer_z_start))
            elif orientation == 'sagittal':
                if stat['x_min'] <= slice_idx < stat['x_max']:
                    if is_multi_layer and self.current_display_layer is None:
                        layer_z_start = ml_z_starts.get(stat.get('layer_name', ''), 0)
                    else:
                        layer_z_start = single_z_offset
                    visible_stats.append((stat, layer_z_start))

        if not visible_stats:
            return

        n_bumps = len(visible_stats)
        if n_bumps > 500:
            label_scale, font_size = 0.16, 13
        elif n_bumps > 200:
            label_scale, font_size = 0.20, 14
        elif n_bumps > 50:
            label_scale, font_size = 0.24, 15
        else:
            label_scale, font_size = 0.28, 17

        stats_only = [s for s, _ in visible_stats]
        has_grid = all(
            s.get("grid_row") is not None and s.get("grid_col") is not None
            for s in stats_only
        )
        selected_labels = {
            int(lab)
            for _c, lab in (getattr(self, "selected_highlight_objects", None) or [])
            if lab
        }

        if orientation == 'axial':
            self._draw_axial_labels(
                renderer, visible_stats, has_grid, label_scale, font_size, selected_labels
            )
        elif orientation == 'coronal':
            self._draw_coronal_labels(
                renderer, visible_stats, has_grid, label_scale, font_size,
                slice_idx, selected_labels,
            )
        elif orientation == 'sagittal':
            self._draw_sagittal_labels(
                renderer, visible_stats, has_grid, label_scale, font_size,
                slice_idx, selected_labels,
            )

    @staticmethod
    def _style_index_text_prop(tprop, font_size, color, selected=False):
        """Readable index plate — match Viewer Online."""
        tprop.SetFontSize(int(font_size))
        tprop.SetBold(True)
        tprop.ShadowOn()
        tprop.SetShadowOffset(1, -1)
        if selected:
            tprop.SetColor(1.0, 0.92, 0.25)
            tprop.SetBackgroundColor(0.05, 0.05, 0.0)
            tprop.SetBackgroundOpacity(0.72)
            frame = (0.95, 0.75, 0.15)
        else:
            tprop.SetColor(float(color[0]), float(color[1]), float(color[2]))
            tprop.SetBackgroundColor(0.02, 0.04, 0.08)
            tprop.SetBackgroundOpacity(0.62)
            frame = (0.15, 0.55, 0.65)
        try:
            tprop.SetJustificationToCentered()
            tprop.SetVerticalJustificationToCentered()
        except Exception:
            pass
        try:
            tprop.FrameOn()
            tprop.SetFrameColor(*frame)
            tprop.SetFrameWidth(1)
        except Exception:
            pass

    def _index_label_text(self, stat, has_grid):
        if has_grid:
            return f"{stat['grid_row']},{stat['grid_col']}"
        return str(stat.get("row_id", stat.get("label", "")))

    def _draw_axial_labels(
        self, renderer, visible_stats, has_grid, label_scale, font_size, selected_labels=None
    ):
        """XY: one (R,C) plate per bump at centroid — no R#/C# edge rails."""
        selected_labels = selected_labels or set()
        seen = set()
        for stat, _z_off in visible_stats:
            if has_grid:
                key = (stat["grid_row"], stat["grid_col"])
                if key in seen:
                    continue
                seen.add(key)
            try:
                lab = int(stat.get("label", 0))
            except (TypeError, ValueError):
                lab = 0
            is_sel = lab in selected_labels
            text = self._index_label_text(stat, has_grid)
            if not text:
                continue
            cx = float(stat["centroid_x"])
            cy = float(stat["centroid_y"])
            caption = vtk.vtkTextActor3D()
            caption.SetInput(text)
            caption.SetPosition(cx, cy, 0.55)
            sc = label_scale * (1.15 if is_sel else 1.0)
            caption.SetScale(sc, sc, sc)
            self._style_index_text_prop(
                caption.GetTextProperty(),
                font_size + (2 if is_sel else 0),
                (0.25, 0.95, 1.0),
                selected=is_sel,
            )
            renderer.AddActor(caption)

    def _draw_coronal_labels(
        self, renderer, visible_stats, has_grid, label_scale, font_size, slice_y,
        selected_labels=None,
    ):
        """XZ: same (R,C) identity as Viewer — no C# edge headers."""
        selected_labels = selected_labels or set()
        vol_z = self.volume_data.shape[0] if self.volume_data is not None else 1
        seen = set()
        for stat, z_off in visible_stats:
            if has_grid:
                key = (stat["grid_row"], stat["grid_col"])
                if key in seen:
                    continue
                seen.add(key)
            try:
                lab = int(stat.get("label", 0))
            except (TypeError, ValueError):
                lab = 0
            is_sel = lab in selected_labels
            text = self._index_label_text(stat, has_grid)
            if not text:
                continue
            cx = float(stat["centroid_x"])
            global_z = float(stat["centroid_z"]) + z_off
            cz_display = (vol_z - 1) - global_z
            caption = vtk.vtkTextActor3D()
            caption.SetInput(text)
            caption.SetPosition(cx, cz_display, 0.55)
            sc = label_scale * (1.15 if is_sel else 1.0)
            caption.SetScale(sc, sc, sc)
            self._style_index_text_prop(
                caption.GetTextProperty(),
                font_size + (2 if is_sel else 0),
                (0.4, 1.0, 0.55),
                selected=is_sel,
            )
            renderer.AddActor(caption)

    def _draw_sagittal_labels(
        self, renderer, visible_stats, has_grid, label_scale, font_size, slice_x,
        selected_labels=None,
    ):
        """YZ: same (R,C) identity as Viewer — no R# edge rail."""
        selected_labels = selected_labels or set()
        seen = set()
        for stat, z_off in visible_stats:
            if has_grid:
                key = (stat["grid_row"], stat["grid_col"])
                if key in seen:
                    continue
                seen.add(key)
            try:
                lab = int(stat.get("label", 0))
            except (TypeError, ValueError):
                lab = 0
            is_sel = lab in selected_labels
            text = self._index_label_text(stat, has_grid)
            if not text:
                continue
            cz = float(stat["centroid_z"]) + z_off
            cy = float(stat["centroid_y"])
            caption = vtk.vtkTextActor3D()
            caption.SetInput(text)
            caption.SetPosition(cz, cy, 0.55)
            sc = label_scale * (1.15 if is_sel else 1.0)
            caption.SetScale(sc, sc, sc)
            self._style_index_text_prop(
                caption.GetTextProperty(),
                font_size + (2 if is_sel else 0),
                (1.0, 0.6, 0.35),
                selected=is_sel,
            )
            renderer.AddActor(caption)

    def get_active_volume(self):
        """Return the active volume data  cropped if ROI is defined, else full volume."""
        if self.roi_manager.has_valid_roi and self.volume_data is not None:
            source = self._original_volume_data if hasattr(self, '_original_volume_data') and self._original_volume_data is not None else self.volume_data
            return self.roi_manager.get_crop(source)
        return self.volume_data

    # ==================== BOUNDARY PANEL ====================
    
    def _create_boundary_panel(self):
        """Create the 3D Boundary results panel with table and click-to-navigate."""
        panel = QWidget()
        self._theme_register("_theme_panels", panel)
        panel.setStyleSheet(f"background: {SemiconductorTheme.BG_PANEL}; border-radius: 6px;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("<b>3D BOUNDARY ANALYSIS RESULTS</b>")
        self._theme_register("_theme_accent_labels", title)
        title.setStyleSheet(f"font-size: 10pt; color: {SemiconductorTheme.ACCENT_PRIMARY};")
        header.addWidget(title)
        header.addStretch()
        
        self.bnd_info_label = QLabel("No results loaded")
        self._theme_register("_theme_secondary_labels", self.bnd_info_label)
        self.bnd_info_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;"
        )
        header.addWidget(self.bnd_info_label)

        # Clear selection — same role as MES Clear (table + 3D gap overlay)
        self.bnd_clear_btn = self._make_mes_tool_button(
            "Clear",
            "Clear B2B row selection and 3D gap overlay",
            "secondary",
            self.clear_boundary_selection,
            icon_name="x.svg",
            fallback_icon=QStyle.SP_DialogResetButton,
        )
        header.addWidget(self.bnd_clear_btn)

        layout.addLayout(header)
        
        # Table (Viewer parity: no vertical row header — avoids dual "#" index look)
        self.boundary_table = QTableWidget()
        self.boundary_table.setColumnCount(11)
        self.boundary_table.setHorizontalHeaderLabels([
            "#", "Layer",
            "Source (R,C)", "Dir", "Dest (R,C)",
            "Gap X (µm)", "Gap Y (µm)", "Gap Z (µm)",
            "Euclidean (µm)", "Src Voxel", "Dst Voxel",
        ])
        self.boundary_table.horizontalHeader().setStretchLastSection(False)
        self.boundary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.boundary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.boundary_table.setSelectionMode(QTableWidget.SingleSelection)
        self.boundary_table.setAlternatingRowColors(True)
        self.boundary_table.verticalHeader().setVisible(False)
        self.boundary_table.verticalHeader().setDefaultSectionSize(22)
        self.boundary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.boundary_table.setStyleSheet(f"""
            QTableWidget {{
                background: {SemiconductorTheme.BG_DARK};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                gridline-color: {SemiconductorTheme.BORDER_DEFAULT};
                font-size: 8pt;
            }}
            QTableWidget::item:selected {{
                background: {SemiconductorTheme.ACCENT_PRIMARY};
                color: {SemiconductorTheme.TEXT_ON_ACCENT};
            }}
            QHeaderView::section {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                padding: 2px 4px;
                font-size: 8pt;
                font-weight: bold;
            }}
        """)
        self.boundary_table.cellClicked.connect(self._on_boundary_row_clicked)
        layout.addWidget(self.boundary_table, 1)
        
        self.boundary_results = []  # list of dicts from CSV
        return panel

    def clear_boundary_selection(self):
        """Clear B2B table selection + 3D gap actors (MES Clear parity)."""
        table = getattr(self, "boundary_table", None)
        if table is not None:
            try:
                table.blockSignals(True)
                table.clearSelection()
                table.blockSignals(False)
            except Exception:
                pass
        if hasattr(self, "_clear_boundary_actors"):
            try:
                self._clear_boundary_actors()
            except Exception:
                pass
        # Drop MES 3D pick only if it was cleared for gap spotlight — restore clean 3D
        n = len(getattr(self, "boundary_results", None) or [])
        if hasattr(self, "bnd_info_label"):
            try:
                if n > 0:
                    self.bnd_info_label.setText(
                        f"{n} rows | click row → 3D gap · Clear to hide"
                    )
                else:
                    self.bnd_info_label.setText("No results loaded")
            except Exception:
                pass
        if hasattr(self, "_3d_bnd_info"):
            try:
                self._3d_bnd_info.setText("")
            except Exception:
                pass

    def _find_boundary_csv_paths(self, roots):
        """Discover B2B result CSVs under output folders (gap preferred over summary).

        Supports files written by Teaching BoundaryAnalysisThread / BoundaryGPU.dll
        and Online-style combined names.
        """
        prefer = (
            "boundary_gap_all_layers.csv",
            "boundary_gap.csv",
            "boundary_summary.csv",
        )
        found = []
        seen = set()
        for root in roots or []:
            if not root:
                continue
            root = os.path.abspath(root)
            if os.path.isfile(root) and root.lower().endswith(".csv"):
                key = root.lower()
                if key not in seen:
                    seen.add(key)
                    found.append(root)
                continue
            if not os.path.isdir(root):
                continue
            for dirpath, _dirs, files in os.walk(root):
                lower_map = {f.lower(): f for f in files}
                for name in prefer:
                    real = lower_map.get(name.lower())
                    if real:
                        p = os.path.join(dirpath, real)
                        key = os.path.abspath(p).lower()
                        if key not in seen:
                            seen.add(key)
                            found.append(p)
        # Prefer gap-format paths first
        found.sort(
            key=lambda p: (
                0 if "gap" in os.path.basename(p).lower() else 1,
                0 if "all_layers" in os.path.basename(p).lower() else 1,
                p.lower(),
            )
        )
        return found

    def _merge_extra_boundary_csvs(self, paths):
        """Append additional gap-format CSVs into the current boundary table."""
        import csv as _csv
        if not paths or not getattr(self, "boundary_results", None):
            return
        first = self.boundary_results[0] if self.boundary_results else {}
        if "Src_row" not in first:
            return
        extra = []
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    for row in _csv.DictReader(f):
                        if "Src_row" in row:
                            extra.append(row)
            except Exception as e:
                print(f"[BOUNDARY] merge skip {p}: {e}")
        if not extra:
            return
        self.boundary_results.extend(extra)
        # Re-render table from merged rows via temporary rewrite of items
        # (reuse loader path: write is heavy — call populate helper)
        self._populate_boundary_table_from_results()

    def _sync_3d_spacing_from_viewer(self):
        """Copy world spacing from 3D Viewer (Online FOV) so B2B/mask align."""
        try:
            mw = self.window()
            mpv = (
                getattr(mw, "multi_planar_view", None)
                or getattr(mw, "viewer", None)
                or getattr(mw, "mpv", None)
            )
            if mpv is None:
                return
            sp = None
            if hasattr(mpv, "_get_3d_world_spacing"):
                sp = mpv._get_3d_world_spacing()
            elif getattr(mpv, "custom_spacing", None) is not None:
                sp = mpv.custom_spacing
            elif getattr(mpv, "spacing", None) is not None:
                sp = mpv.spacing
            if sp is not None:
                self.custom_spacing = [float(sp[0]), float(sp[1]), float(sp[2])]
                self.spacing = list(self.custom_spacing)
            # Also reuse Viewer labeled mask for B2B surface patches when Teaching empty
            if getattr(self, "labeled_class1_data", None) is None:
                lab = getattr(mpv, "labeled_class1_data", None)
                if lab is not None:
                    self.labeled_class1_data = lab
        except Exception as e:
            print(f"[TEACHING 3D] spacing sync from Viewer skipped: {e}")

    def _get_3d_world_spacing(self):
        """Same contract as Viewer — prefer live VTK image spacing when present."""
        try:
            vd = getattr(self, "_3d_vtk_data", None)
            if vd is not None:
                sp = vd.GetSpacing()
                return (float(sp[0]), float(sp[1]), float(sp[2]))
        except Exception:
            pass
        custom = getattr(self, "custom_spacing", None)
        if custom is None:
            custom = getattr(self, "_3d_original_spacing", None)
        if custom is None:
            custom = getattr(self, "_3d_spacing", None)
        return _shared_world_spacing(custom, getattr(self, "spacing", [1.0, 1.0, 1.0]))

    def _b2b_layer_z_offset(self, row_dict):
        """Local B2B layer Z → full-volume Z (shared helper + Teaching bands)."""
        fallback = int(getattr(self, "_boundary_z_offset", 0) or 0)
        bands = None
        try:
            mw = self.window()
            if hasattr(mw, "_online_layer_bands"):
                bands = mw._online_layer_bands()
        except Exception:
            bands = None
        # Prefer Teaching layer_definitions as bands when Online bands empty
        if not bands:
            bands = [
                {"name": L.get("name", ""), "z_start": L.get("z_start", 0)}
                for L in (getattr(self, "layer_definitions", None) or [])
                if L.get("name")
            ] or None
        # Inject Layer from folder detect when row lacks it
        row = row_dict
        if row and not row.get("Layer") and getattr(self, "_boundary_layer_name", ""):
            row = dict(row)
            row["Layer"] = self._boundary_layer_name
        return _shared_layer_z_offset(row, layer_bands=bands, fallback=fallback)

    def _populate_boundary_table_from_results(self):
        """Fill boundary_table from self.boundary_results (gap or summary)."""
        if not self.boundary_results:
            return
        first_row = self.boundary_results[0]
        is_gap_format = (
            "Src_row" in first_row
            or "Direction" in first_row
            or "Source_id" in first_row
        )
        self._boundary_csv_format = "gap" if is_gap_format else "summary"
        table = self.boundary_table
        table.setSortingEnabled(False)
        table.blockSignals(True)
        table.verticalHeader().setVisible(False)
        if is_gap_format:
            table.setColumnCount(11)
            table.setHorizontalHeaderLabels([
                "#", "Layer",
                "Source (R,C)", "Dir", "Dest (R,C)",
                "Gap X (µm)", "Gap Y (µm)", "Gap Z (µm)",
                "Euclidean (µm)", "Src Voxel", "Dst Voxel",
            ])
            table.setRowCount(len(self.boundary_results))
            for i, r in enumerate(self.boundary_results):
                eucl_str = r.get("Min_dist_Euclidean_um", r.get("min_gap_euclidean_um", "NaN"))
                try:
                    eucl_val = float(eucl_str)
                    is_close = eucl_val < 1.5
                except (ValueError, TypeError):
                    is_close = False
                row_bg = QColor(180, 40, 40, 50) if is_close else QColor(40, 180, 60, 30)
                src_str = f"R{r.get('Src_row', '?')},C{r.get('Src_col', '?')}"
                dst_str = f"R{r.get('Dst_row', '?')},C{r.get('Dst_col', '?')}"
                direction = str(r.get("Direction", "?"))
                src_pos = (
                    f"Z={r.get('Src_voxel_Z', '')},Y={r.get('Src_voxel_Y', '')},"
                    f"X={r.get('Src_voxel_X', '')}"
                )
                dst_pos = (
                    f"Z={r.get('Dst_voxel_Z', '')},Y={r.get('Dst_voxel_Y', '')},"
                    f"X={r.get('Dst_voxel_X', '')}"
                )
                layer = str(r.get("Layer", "") or "")
                items_data = [
                    str(i + 1), layer, src_str, direction, dst_str,
                    r.get("Min_dist_X_um", r.get("min_gap_X_um", "NaN")),
                    r.get("Min_dist_Y_um", r.get("min_gap_Y_um", "NaN")),
                    r.get("Min_dist_Z_um", r.get("min_gap_Z_um", "NaN")),
                    eucl_str, src_pos, dst_pos,
                ]
                for col, text in enumerate(items_data):
                    item = QTableWidgetItem(str(text))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setBackground(row_bg)
                    if col == 0:
                        item.setData(Qt.UserRole, i)  # only col0 — sort-safe index
                    if col == 8:
                        item.setFont(QFont("Arial", 9, QFont.Bold))
                        item.setForeground(
                            QColor(255, 100, 100) if is_close else QColor(100, 255, 120)
                        )
                    if col == 3:
                        dir_colors = {
                            "Up": QColor(100, 200, 255),
                            "Down": QColor(100, 200, 255),
                            "Left": QColor(255, 200, 100),
                            "Right": QColor(255, 200, 100),
                        }
                        item.setForeground(dir_colors.get(direction, QColor(200, 200, 200)))
                    table.setItem(i, col, item)
        else:
            table.setColumnCount(9)
            table.setHorizontalHeaderLabels([
                "#", "Bump (R,C)", "Label", "Bnd Voxels",
                "Gap X (µm)", "Gap Y (µm)", "Gap Z (µm)",
                "Euclidean (µm)", "Min Gap Pos",
            ])
            table.setRowCount(len(self.boundary_results))
            for i, r in enumerate(self.boundary_results):
                eucl_str = r.get("min_gap_euclidean_um", "NaN")
                try:
                    eucl_val = float(eucl_str)
                    is_close = eucl_val < 1.0
                except (ValueError, TypeError):
                    is_close = False
                row_bg = QColor(180, 40, 40, 50) if is_close else QColor(40, 180, 60, 30)
                pos_str = (
                    f"Z={r.get('min_gap_voxel_Z', '')},Y={r.get('min_gap_voxel_Y', '')},"
                    f"X={r.get('min_gap_voxel_X', '')}"
                )
                items_data = [
                    str(i + 1),
                    r.get("Bump_id", ""),
                    r.get("label", ""),
                    r.get("boundary_voxels", ""),
                    r.get("min_gap_X_um", "NaN"),
                    r.get("min_gap_Y_um", "NaN"),
                    r.get("min_gap_Z_um", "NaN"),
                    eucl_str, pos_str,
                ]
                for col, text in enumerate(items_data):
                    item = QTableWidgetItem(str(text))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setBackground(row_bg)
                    if col == 0:
                        item.setData(Qt.UserRole, i)
                    if col == 7:
                        item.setFont(QFont("Arial", 9, QFont.Bold))
                        item.setForeground(
                            QColor(255, 100, 100) if is_close else QColor(100, 255, 120)
                        )
                    table.setItem(i, col, item)
        table.blockSignals(False)
        table.setSortingEnabled(True)
        try:
            table.resizeColumnsToContents()
            for c in range(table.columnCount()):
                w = table.columnWidth(c)
                if w > 160:
                    table.setColumnWidth(c, 160)
        except Exception:
            pass

    def _load_boundary_csv(self, csv_path):
        """Load boundary CSV (supports both boundary_summary.csv and boundary_gap.csv formats)."""
        import csv
        self.boundary_results = []
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.boundary_results.append(row)
        except Exception as e:
            print(f"[BOUNDARY] Failed to load {csv_path}: {e}")
            QMessageBox.critical(self, "Load Error", f"Failed to load CSV:\n{e}")
            return
        
        if not self.boundary_results:
            QMessageBox.information(self, "Empty", "CSV has no data rows.")
            return
        
        # Auto-detect layer Z-offset from CSV folder path
        # CSV lives in e.g. .../oiaoia/Layer_1/boundary_gap.csv
        # Layer_1 maps to config "Layer 1" with z_start=20
        self._boundary_z_offset = 0
        self._boundary_layer_name = ""
        try:
            csv_dir = os.path.dirname(os.path.abspath(csv_path))
            folder_name = os.path.basename(csv_dir)  # e.g. "Layer_1"
            
            if hasattr(self, 'layer_definitions') and self.layer_definitions:
                for layer_def in self.layer_definitions:
                    # Match: folder "Layer_1" ↔ config name "Layer 1" (underscore ↔ space)
                    cfg_name = layer_def.get('name', '')
                    folder_canonical = folder_name.replace('_', ' ').strip().lower()
                    cfg_canonical = cfg_name.strip().lower()
                    
                    if folder_canonical == cfg_canonical or folder_name.lower() == cfg_name.replace(' ', '_').lower():
                        self._boundary_z_offset = int(layer_def.get('z_start', 0))
                        self._boundary_layer_name = cfg_name
                        print(f"[BOUNDARY] Layer '{cfg_name}' detected → Z offset = {self._boundary_z_offset}")
                        break
        except Exception as e:
            print(f"[BOUNDARY] Could not detect layer offset: {e}")

        self._populate_boundary_table_from_results()
        is_gap_format = getattr(self, "_boundary_csv_format", "summary") == "gap"
        fmt_name = "boundary_gap" if is_gap_format else "boundary_summary"
        z_info = (
            f" | Z+{self._boundary_z_offset} ({self._boundary_layer_name})"
            if getattr(self, "_boundary_z_offset", 0) > 0
            else ""
        )
        self.bnd_info_label.setText(
            f"{len(self.boundary_results)} rows | {fmt_name} | {os.path.basename(csv_path)}"
            f"{z_info} | click row → 3D gap"
        )
        self.boundary_table.setToolTip(
            "Click a row to map the 3D boundary gap (Src→Dst) on the Teaching 3D Volume view."
        )

        # Switch to B2B tab (index 5)
        if hasattr(self, "_set_teaching_tab"):
            self._set_teaching_tab(5)

    def _on_boundary_row_clicked(self, row, col):
        """Map selected boundary gap onto the 3D Volume only (not MPR).

        Gap is a true 3D spatial distance — MPR slices make it hard to interpret.
        Auto-enables Teaching 3D view when needed.

        Coordinates: CSV Src/Dst voxel Z are **layer-local** (same as Viewer Online).
        Must add ``_b2b_layer_z_offset(row)`` before drawing or markers sit at wrong Z.
        """
        id_item = self.boundary_table.item(row, 0)
        if id_item is None:
            return
        stats_idx = id_item.data(Qt.UserRole)
        if stats_idx is None:
            stats_idx = row
        try:
            stats_idx = int(stats_idx)
        except (TypeError, ValueError):
            return
        if stats_idx < 0 or stats_idx >= len(self.boundary_results):
            return
        r = self.boundary_results[stats_idx]

        # Viewer-identical: per-row Layer → z_start (not folder-only offset)
        z_offset = self._b2b_layer_z_offset(r)

        fmt = getattr(self, '_boundary_csv_format', 'summary')

        def _iv(key, default=0):
            try:
                return int(float(r.get(key, default)))
            except (TypeError, ValueError):
                return default

        # Ensure 3D volume panel is on + spacing synced from Viewer (B2B world coords)
        self._sync_3d_spacing_from_viewer()
        if not getattr(self, '_3d_view_active', False):
            if hasattr(self, '_3d_enable_check'):
                self._3d_enable_check.blockSignals(True)
                self._3d_enable_check.setChecked(True)
                self._3d_enable_check.blockSignals(False)
            self._toggle_teaching_3d(True)
        else:
            # Re-render so vtk spacing / Seg Overlay masks match current masks
            try:
                self._render_teaching_3d()
            except Exception as re:
                print(f"[B2B] re-render before gap: {re}")

        if fmt == 'gap' or (
            r.get('Src_voxel_Z') not in (None, '')
            and r.get('Dst_voxel_Z') not in (None, '')
        ):
            try:
                src = (
                    _iv('Src_voxel_Z') + z_offset,
                    _iv('Src_voxel_Y'),
                    _iv('Src_voxel_X'),
                )
                dst = (
                    _iv('Dst_voxel_Z') + z_offset,
                    _iv('Dst_voxel_Y'),
                    _iv('Dst_voxel_X'),
                )
                eucl = r.get('Min_dist_Euclidean_um', r.get('min_gap_euclidean_um', '?'))
                direction = r.get('Direction', '')
                src_rc = f"R{r.get('Src_row','?')}C{r.get('Src_col','?')}"
                dst_rc = f"R{r.get('Dst_row','?')}C{r.get('Dst_col','?')}"
                layer = str(r.get('Layer', '') or getattr(self, '_boundary_layer_name', '') or '')
                # Same 3-line callout as Viewer (shared format_gap_label)
                label = _shared_format_gap_label(
                    layer, eucl, src_rc, dst_rc, direction, src, dst
                )
                print(
                    f"[B2B] Teaching gap Layer={layer!r} z_off={z_offset} "
                    f"localSRC=({_iv('Src_voxel_Z')},{_iv('Src_voxel_Y')},{_iv('Src_voxel_X')}) "
                    f"→ globalSRC={src} globalDST={dst}"
                )

                self._draw_boundary_gap_line(
                    src_voxel=src,
                    dst_voxel=dst,
                    label_text=label,
                )
                if hasattr(self, 'bnd_info_label'):
                    self.bnd_info_label.setText(
                        f"3D gap ON · {layer} {src_rc}→{dst_rc} ({direction}) "
                        f"{eucl}µm · Z+{z_offset} · click another row to switch"
                    )
            except (ValueError, TypeError) as e:
                print(f"[BOUNDARY] Could not draw 3D line: {e}")
            return

        # Summary format: single min-gap voxel → marker (short segment)
        try:
            vz = _iv('min_gap_voxel_Z') + z_offset
            vy = _iv('min_gap_voxel_Y')
            vx = _iv('min_gap_voxel_X')
            eucl = r.get('min_gap_euclidean_um', '?')
            bump = r.get('Bump_id', '')
            label = f"{bump} min-gap {eucl}µm".strip()
            self._draw_boundary_gap_line(
                src_voxel=(vz, vy, vx),
                dst_voxel=(vz, vy, vx + 1),
                label_text=label,
            )
            if hasattr(self, 'bnd_info_label'):
                self.bnd_info_label.setText(f"3D gap: {label} (Z+{z_offset})")
        except (ValueError, TypeError) as e:
            print(f"[BOUNDARY] Could not draw 3D marker: {e}")

    def _browse_boundary_csv(self):
        """Browse for a boundary_summary.csv file."""
        start_dir = self.output_path_input.text().strip() or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Boundary Summary CSV", start_dir,
            "CSV Files (*.csv);;All Files (*)")
        if path and os.path.isfile(path):
            self._load_boundary_csv(path)

    def _autodetect_boundary_csv(self):
        """Auto-detect boundary_summary.csv from output path and sub-directories."""
        output_base = self.output_path_input.text().strip()
        if not output_base:
            output_base = os.path.dirname(self.config_path) if self.config_path else os.getcwd()
        
        found = []
        # Check root
        root_csv = os.path.join(output_base, "boundary_summary.csv")
        if os.path.isfile(root_csv):
            found.append(root_csv)
        # Check sub-directories
        if os.path.isdir(output_base):
            for sub in os.listdir(output_base):
                sub_csv = os.path.join(output_base, sub, "boundary_summary.csv")
                if os.path.isfile(sub_csv):
                    found.append(sub_csv)
        
        if not found:
            QMessageBox.information(self, "Auto-detect",
                f"No boundary_summary.csv found in:\n{output_base}\n\nRun 'B2B' analysis first, or use 'Load CSV' to browse manually.")
            return
        
        # If multiple found, load the first one (could show a selection dialog)
        self._load_boundary_csv(found[0])
        if len(found) > 1:
            self.bnd_info_label.setText(
                self.bnd_info_label.text() + f" (+{len(found)-1} more)")

    # ==================== 3D CONTROLS PANEL ====================

    def _create_3d_controls_panel(self):
        """Create the 3D rendering controls panel for the sidebar."""
        panel = QWidget()
        self._theme_register("_theme_panels", panel)
        panel.setStyleSheet(f"background: {SemiconductorTheme.BG_PANEL}; border-radius: 6px;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        
        title = QLabel("<b>3D VOLUME RENDERING</b>")
        self._theme_register("_theme_accent_labels", title)
        title.setStyleSheet(f"font-size: 10pt; color: {SemiconductorTheme.ACCENT_PRIMARY};")
        layout.addWidget(title)
        
        # Toggle 3D rendering
        self._3d_enable_check = QCheckBox("Enable 3D Rendering")
        self._3d_enable_check.setStyleSheet(f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-weight: bold; font-size: 10pt;")
        self._3d_enable_check.setChecked(False)
        self._3d_enable_check.stateChanged.connect(self._toggle_teaching_3d)
        layout.addWidget(self._3d_enable_check)
        
        # Render mode
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._3d_mode_combo = QComboBox()
        self._3d_mode_combo.addItems(["Volume", "Seg Overlay", "Seg Only"])
        self._3d_mode_combo.setToolTip("Volume: raw data\nSeg Overlay: volume + bump mask\nSeg Only: only show segmentation")
        self._3d_mode_combo.currentTextChanged.connect(self._on_3d_mode_changed)
        mode_row.addWidget(self._3d_mode_combo)
        layout.addLayout(mode_row)
        
        # Quality preset
        qual_row = QHBoxLayout()
        qual_row.addWidget(QLabel("Quality:"))
        self._3d_quality_combo = QComboBox()
        self._3d_quality_combo.addItems(["Draft", "Standard", "High"])
        self._3d_quality_combo.setCurrentText("Draft")
        self._3d_quality_combo.currentTextChanged.connect(self._on_3d_quality_changed)
        qual_row.addWidget(self._3d_quality_combo)
        layout.addLayout(qual_row)
        
        # Opacity
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacity:"))
        self._3d_opacity_slider = QSlider(Qt.Horizontal)
        self._3d_opacity_slider.setRange(0, 100)
        self._3d_opacity_slider.setValue(30)
        self._3d_opacity_slider.valueChanged.connect(self._on_3d_opacity_changed)
        op_row.addWidget(self._3d_opacity_slider)
        layout.addLayout(op_row)
        
        # Intensity range
        int_group = QGroupBox("Intensity Range")
        int_lay = QVBoxLayout()
        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Min:"))
        self._3d_min_slider = QSlider(Qt.Horizontal)
        self._3d_min_slider.setRange(0, 65535)
        self._3d_min_slider.setValue(0)
        self._3d_min_slider.valueChanged.connect(self._update_3d_transfer_function)
        min_row.addWidget(self._3d_min_slider)
        int_lay.addLayout(min_row)
        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max:"))
        self._3d_max_slider = QSlider(Qt.Horizontal)
        self._3d_max_slider.setRange(0, 65535)
        self._3d_max_slider.setValue(65535)
        self._3d_max_slider.valueChanged.connect(self._update_3d_transfer_function)
        max_row.addWidget(self._3d_max_slider)
        int_lay.addLayout(max_row)
        int_group.setLayout(int_lay)
        layout.addWidget(int_group)
        
        # Camera presets
        cam_group = QGroupBox("Camera Presets")
        cam_lay = QGridLayout()
        cam_lay.setSpacing(4)
        for i, (name, args) in enumerate([
            ("Front", (0, 0, 1, 0, 1, 0)), ("Back", (0, 0, -1, 0, 1, 0)),
            ("Left", (-1, 0, 0, 0, 1, 0)), ("Right", (1, 0, 0, 0, 1, 0)),
            ("Top", (0, 1, 0, 0, 0, -1)), ("Bottom", (0, -1, 0, 0, 0, 1)),
        ]):
            btn = QPushButton(name)
            btn.setMaximumHeight(24)
            btn.clicked.connect(lambda checked, a=args: self._set_3d_camera_preset(a))
            cam_lay.addWidget(btn, i // 2, i % 2)
        cam_group.setLayout(cam_lay)
        layout.addWidget(cam_group)

        # ── Boundary clipping (cut through volume/seg to inspect gaps) ──
        clip_group = QGroupBox("Boundary Clipping")
        self._theme_register("_theme_group_boxes", clip_group)
        clip_group.setStyleSheet(self._group_box_qss())
        clip_group.setToolTip(
            "Slice through the 3D volume/segmentation along X, Y or Z\n"
            "so boundary gap lines between bumps are easier to see."
        )
        clip_lay = QVBoxLayout(clip_group)
        clip_lay.setContentsMargins(6, 12, 6, 6)
        clip_lay.setSpacing(6)

        hint = QLabel("Cut planes to inspect boundary gaps")
        hint.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt;"
        )
        hint.setWordWrap(True)
        clip_lay.addWidget(hint)

        # State (also initialized in 3D view create)
        self._3d_clip_enabled = {"x": False, "y": False, "z": False}
        self._3d_clip_flip = {"x": False, "y": False, "z": False}
        self._3d_clip_planes = {"x": None, "y": None, "z": None}
        self._3d_clip_controls = {}

        # X red, Y green, Z blue — match orientation axes
        axes_ui = [
            ("x", "X", "#e85d5d"),
            ("y", "Y", "#5dce6a"),
            ("z", "Z", "#5d8de8"),
        ]
        for axis, label, color in axes_ui:
            row = QHBoxLayout()
            row.setSpacing(4)

            chk = QCheckBox(label)
            chk.setFixedWidth(28)
            chk.setStyleSheet(f"color: {color}; font-weight: 800; font-size: 9pt;")
            chk.setToolTip(f"Enable {label}-axis clip plane")
            chk.stateChanged.connect(
                lambda state, a=axis: self._on_3d_clip_toggled(a, bool(state))
            )
            row.addWidget(chk)

            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 1000)  # 0.1% steps
            sl.setValue(500)     # mid
            sl.setEnabled(False)
            sl.setStyleSheet(f"""
                QSlider::groove:horizontal {{
                    height: 4px; background: {SemiconductorTheme.BG_DARK};
                    border-radius: 2px;
                }}
                QSlider::handle:horizontal {{
                    background: {color}; width: 12px; margin: -5px 0;
                    border-radius: 6px;
                }}
                QSlider::sub-page:horizontal {{
                    background: {color}; border-radius: 2px;
                }}
            """)
            sl.valueChanged.connect(lambda val, a=axis: self._on_3d_clip_moved(a, val))
            row.addWidget(sl, 1)

            val_lbl = QLabel("50%")
            val_lbl.setFixedWidth(36)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setStyleSheet(
                f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;"
            )
            row.addWidget(val_lbl)

            flip = QPushButton("Flip")
            flip.setCheckable(True)
            flip.setFixedWidth(40)
            flip.setEnabled(False)
            flip.setToolTip(f"Flip {label} clip direction (keep other half)")
            flip.setStyleSheet(self._secondary_button_qss())
            flip.clicked.connect(lambda checked, a=axis: self._on_3d_clip_flip(a, checked))
            row.addWidget(flip)

            clip_lay.addLayout(row)
            self._3d_clip_controls[axis] = {
                "check": chk,
                "slider": sl,
                "label": val_lbl,
                "flip": flip,
            }

        reset_row = QHBoxLayout()
        reset_clip_btn = QPushButton("Reset Clips")
        reset_clip_btn.setStyleSheet(self._secondary_button_qss())
        reset_clip_btn.setToolTip("Disable all clip planes")
        reset_clip_btn.clicked.connect(self._reset_3d_clips)
        reset_row.addWidget(reset_clip_btn)
        reset_row.addStretch(1)
        clip_lay.addLayout(reset_row)

        layout.addWidget(clip_group)
        
        # Render button
        self._3d_render_btn = QPushButton("🔄 Render Now")
        self._3d_render_btn.setStyleSheet(f"""
            QPushButton {{
                background: {SemiconductorTheme.ACCENT_PRIMARY};
                color: {SemiconductorTheme.BG_DARK};
                font-weight: bold; padding: 6px; border-radius: 4px;
            }}
        """)
        self._3d_render_btn.clicked.connect(self._render_teaching_3d)
        layout.addWidget(self._3d_render_btn)
        
        layout.addStretch()
        return panel

    # ==================== 3D VIEW WIDGET ====================

    def _create_teaching_3d_view(self):
        """Create a VTK 3D rendering widget for the teaching tab (Viewer-grade quality)."""
        container = QWidget()
        container.setStyleSheet(f"background: {SemiconductorTheme.BG_DARK};")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QWidget()
        header.setMaximumHeight(24)
        header.setStyleSheet("background: rgba(0,0,0,0.5);")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(8, 0, 8, 0)
        lbl = QLabel("<b>3D Volume</b>")
        lbl.setStyleSheet(f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-size: 9pt;")
        h_lay.addWidget(lbl)
        h_lay.addStretch()
        # Boundary line info label
        self._3d_bnd_info = QLabel("")
        self._3d_bnd_info.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        h_lay.addWidget(self._3d_bnd_info)
        layout.addWidget(header)
        
        # VTK widget
        self._3d_vtk_widget = QVTKRenderWindowInteractor()
        self._3d_renderer = vtk.vtkRenderer()
        
        # --- Dragonfly-style Gradient Background ---
        try:
            self._3d_renderer.GradientBackgroundOn()
            self._3d_renderer.SetGradientMode(self._3d_renderer.VTK_GRADIENT_RADIAL_FARTHEST_CORNER)
            self._3d_renderer.SetBackground(0.85, 0.85, 0.85)
            self._3d_renderer.SetBackground2(0.15, 0.15, 0.15)
        except AttributeError:
            self._3d_renderer.GradientBackgroundOn()
            self._3d_renderer.SetBackground(0.15, 0.15, 0.15)
            self._3d_renderer.SetBackground2(0.85, 0.85, 0.85)
        
        render_window = self._3d_vtk_widget.GetRenderWindow()
        render_window.AddRenderer(self._3d_renderer)
        
        # --- Advanced Three-Point Lighting (LightKit) ---
        self._3d_light_kit = vtk.vtkLightKit()
        self._3d_light_kit.AddLightsToRenderer(self._3d_renderer)
        self._3d_light_kit.SetKeyLightIntensity(0.85)
        self._3d_light_kit.SetKeyToFillRatio(2.5)
        self._3d_light_kit.SetKeyLightWarmth(0.6)
        
        # --- Performance Optimizations ---
        render_window.SetDesiredUpdateRate(60.0)
        if hasattr(render_window, 'SetMultiSamples'):
            render_window.SetMultiSamples(4)
        
        # Depth peeling for correct transparency
        self._3d_renderer.SetUseDepthPeeling(1)
        self._3d_renderer.SetMaximumNumberOfPeels(4)
        self._3d_renderer.SetOcclusionRatio(0.1)
        
        # Dragonfly-style 3D navigation (same as 3D Viewer tab)
        style = Dragonfly3DInteractorStyle()
        style._host = self  # right-drag / VTK wheel → _dragonfly_3d_zoom
        interactor = render_window.GetInteractor()
        interactor.SetInteractorStyle(style)
        self._3d_interactor_style = style
        
        # LOD Interaction Observers (same as viewer.py)
        def _on_3d_interaction_start(caller, event):
            self._start_3d_lod_interaction()
        def _on_3d_interaction_end(caller, event):
            self._end_3d_lod_interaction()
        interactor.AddObserver('StartInteractionEvent', _on_3d_interaction_start)
        interactor.AddObserver('EndInteractionEvent', _on_3d_interaction_end)
        self._3d_is_interacting = False
        
        # LOD refinement timer
        self._3d_lod_refine_timer = QTimer(self)
        self._3d_lod_refine_timer.setSingleShot(True)
        self._3d_lod_refine_timer.setInterval(300)
        self._3d_lod_refine_timer.timeout.connect(self._refine_3d_after_interaction)
        
        # Clipping: VTK default (camera stays outside volume — no fly-through)
        
        # Orientation cube — Viewer SoT (hover glow + click-to-orient)
        from inno3d.features.shared.orientation_cube import build_orientation_marker
        self._ori_hover_face = None
        marker, ori_state = build_orientation_marker()
        self._ori_cube_state = ori_state
        self._axes_cube_actor = ori_state.get("cube")
        self._ori_face_highlights = ori_state.get("face_highlights") or {}
        self._ori_face_text_props = ori_state.get("face_text_props") or {}
        self._3d_axes_widget = vtk.vtkOrientationMarkerWidget()
        self._3d_axes_widget.SetOrientationMarker(marker)
        self._3d_axes_widget.SetInteractor(interactor)
        # Same viewport family as Viewer
        self._3d_axes_widget.SetViewport(0.82, 0.0, 1.0, 0.18)
        self._3d_axes_widget.EnabledOn()
        self._3d_axes_widget.InteractiveOff()
        interactor.AddObserver(
            "LeftButtonPressEvent", self._on_orientation_cube_click, 10.0
        )
        interactor.AddObserver(
            "MouseMoveEvent", self._on_orientation_cube_hover, 5.0
        )
        
        self._3d_volume_actor = None
        self._3d_volume_mapper = None  # Store mapper ref for LOD updates
        self._3d_boundary_actors = []  # Store boundary line actors
        self._3d_seg_actor = None
        self._3d_volume_color = [0.95, 0.95, 0.95]  # Silver/Grayscale (Dragonfly style)
        self._3d_needs_initial_camera = True  # Only reset camera on first render
        # Clip state defaults (UI may already have set these)
        if not hasattr(self, "_3d_clip_enabled"):
            self._3d_clip_enabled = {"x": False, "y": False, "z": False}
            self._3d_clip_flip = {"x": False, "y": False, "z": False}
            self._3d_clip_planes = {"x": None, "y": None, "z": None}
        
        # Quality presets (same as viewer — includes interaction LOD params)
        self._3d_quality_presets = {
            "Draft":    {"sample_dist": 2.0,  "image_sample": 1.0, "interact_image_sample": 3.0, "interact_sample_dist": 4.0, "auto_adj": True,  "jitter": False, "max_mem_fraction": 0.75},
            "Standard": {"sample_dist": 1.0,  "image_sample": 1.0, "interact_image_sample": 2.0, "interact_sample_dist": 3.0, "auto_adj": True,  "jitter": True,  "max_mem_fraction": 0.80},
            "High":     {"sample_dist": 0.5,  "image_sample": 1.0, "interact_image_sample": 1.5, "interact_sample_dist": 2.0, "auto_adj": True,  "jitter": True,  "max_mem_fraction": 0.85},
        }

        # Qt wheel zoom + focus (same path as 3D Viewer)
        self._3d_vtk_container = container
        self._3d_vtk_widget.installEventFilter(self)
        container.installEventFilter(self)
        self._3d_vtk_widget.setMouseTracking(True)
        container.setMouseTracking(True)
        self._3d_vtk_widget.setFocusPolicy(Qt.StrongFocus)
        
        layout.addWidget(self._3d_vtk_widget, 1)
        return container

    def _toggle_teaching_3d(self, state):
        """Toggle 3D rendering view visibility."""
        enabled = bool(state)
        self._3d_view_active = enabled
        self.teaching_3d_container.setVisible(enabled)
        self._bottom_right_info.setVisible(not enabled)
        if enabled:
            self._render_teaching_3d()  # includes update_3d_crosshair

    def _render_teaching_3d(self):
        """Render the 3D volume in the teaching tab (Viewer-grade quality).
        
        Mirrors viewer.py render_3d: full-resolution data, Dragonfly metallic TF,
        LOD mapper ref, camera reset only on first load.
        """
        if self.volume_data is None or not self._3d_view_active:
            return
        
        renderer = self._3d_renderer
        # Remove old volume actor but keep boundary lines
        if self._3d_volume_actor is not None:
            renderer.RemoveVolume(self._3d_volume_actor)
            self._3d_volume_actor = None
            self._3d_volume_mapper = None
        # Remove old seg actor (legacy MarchingCubes, kept for 'Seg Only' path)
        if getattr(self, '_3d_seg_actor', None):
            renderer.RemoveActor(self._3d_seg_actor)
            self._3d_seg_actor = None
        # Remove Seg Overlay mask shells (surface actors; legacy volumes OK too)
        from inno3d.features.viewer.seg_mask_3d import remove_mask_overlay_from_renderer
        if getattr(self, '_3d_c1_actor', None) is not None:
            remove_mask_overlay_from_renderer(renderer, self._3d_c1_actor)
            self._3d_c1_actor = None
        if getattr(self, '_3d_c2_actor', None) is not None:
            remove_mask_overlay_from_renderer(renderer, self._3d_c2_actor)
            self._3d_c2_actor = None

        vol = self.volume_data
        mode = (
            self._3d_mode_combo.currentText()
            if hasattr(self, "_3d_mode_combo")
            else "Volume"
        )
        # After SEG, prefer Seg Overlay once so yellow mask voxels show (Viewer parity)
        if (
            getattr(self, "bump_segmentation", None) is not None
            and mode == "Volume"
            and not getattr(self, "_3d_user_forced_volume_mode", False)
        ):
            try:
                self._3d_mode_combo.blockSignals(True)
                self._3d_mode_combo.setCurrentText("Seg Overlay")
                self._3d_mode_combo.blockSignals(False)
                mode = "Seg Overlay"
            except Exception:
                mode = "Seg Overlay"

        # Determine which data to render as the *base* grey volume
        if mode == "Seg Only" and self.bump_segmentation is not None:
            render_data = (self.bump_segmentation > 0).astype(np.uint8) * 255
        else:
            render_data = vol

        # Full resolution — same as viewer.py (no downsample)
        z, y, x = render_data.shape

        # Spacing: identical to 3D Viewer MultiPlanarView (never invent voxel-size scale)
        self._sync_3d_spacing_from_viewer()
        spacing = self._get_3d_world_spacing()

        # Convert to VTK (using Fortran-order transpose like viewer)
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(x, y, z)
        vtk_data.SetSpacing(spacing[0], spacing[1], spacing[2])
        vtk_data.SetOrigin(0.0, 0.0, 0.0)

        data_fortran = np.transpose(render_data, (2, 1, 0))
        flat = np.ascontiguousarray(data_fortran.flatten("F"))
        if render_data.dtype == np.uint16:
            vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
        elif render_data.dtype == np.uint8:
            vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        else:
            flat = flat.astype(np.float32)
            vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_data.GetPointData().SetScalars(vtk_arr)
        self._3d_vtk_data = vtk_data

        # GPU Volume Mapper (same as viewer.py) — prefer High when overlaying masks
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        mapper.SetInputData(vtk_data)

        quality_name = (
            self._3d_quality_combo.currentText()
            if hasattr(self, "_3d_quality_combo")
            else "High"
        )
        if mode == "Seg Overlay" and quality_name == "Draft":
            quality_name = "High"
            try:
                self._3d_quality_combo.blockSignals(True)
                self._3d_quality_combo.setCurrentText("High")
                self._3d_quality_combo.blockSignals(False)
            except Exception:
                pass
        preset = self._3d_quality_presets.get(
            quality_name, self._3d_quality_presets.get("High", {})
        )

        mapper.SetAutoAdjustSampleDistances(1 if preset.get("auto_adj", True) else 0)
        mapper.SetSampleDistance(preset.get("sample_dist", 0.5))
        if hasattr(mapper, "SetImageSampleDistance"):
            mapper.SetImageSampleDistance(preset.get("image_sample", 1.0))
        if hasattr(mapper, "SetUseJittering"):
            mapper.SetUseJittering(1 if preset.get("jitter", True) else 0)
        if hasattr(mapper, "SetMaxMemoryFraction"):
            mapper.SetMaxMemoryFraction(preset.get("max_mem_fraction", 0.85))
        if hasattr(mapper, "SetBlendModeToComposite"):
            mapper.SetBlendModeToComposite()

        self._3d_volume_mapper = mapper

        # Volume property — Viewer defaults (less “blown white cloud”)
        vol_prop = vtk.vtkVolumeProperty()
        vol_prop.ShadeOn()
        vol_prop.SetInterpolationTypeToLinear()
        vol_prop.SetAmbient(0.40)
        vol_prop.SetDiffuse(0.58)
        vol_prop.SetSpecular(0.18)
        vol_prop.SetSpecularPower(16.0)
        try:
            soud = float(min(spacing[0], spacing[1], spacing[2]))
            if soud > 0:
                vol_prop.SetScalarOpacityUnitDistance(soud)
        except Exception:
            pass

        self._apply_3d_transfer_function(vol_prop)
        # Seg Overlay uses glass shells — only light CT dim so structure stays readable
        if mode == "Seg Overlay":
            try:
                op = vol_prop.GetScalarOpacity()
                if op is not None and op.GetSize() > 0:
                    self._3d_opacity_slider.blockSignals(True)
                    prev = self._3d_opacity_slider.value()
                    self._3d_opacity_slider.setValue(max(12, int(prev * 0.72)))
                    self._apply_3d_transfer_function(vol_prop)
                    self._3d_opacity_slider.setValue(prev)
                    self._3d_opacity_slider.blockSignals(False)
            except Exception:
                pass

        volume_actor = vtk.vtkVolume()
        volume_actor.SetMapper(mapper)
        volume_actor.SetProperty(vol_prop)

        renderer.AddVolume(volume_actor)
        self._3d_volume_actor = volume_actor
        self._3d_ds = 1
        self._3d_spacing = (spacing[0], spacing[1], spacing[2])
        self._3d_original_spacing = (spacing[0], spacing[1], spacing[2])

        # ── Seg Overlay: medical glass shells (shared with 3D Viewer) ──
        if mode == "Seg Overlay":
            from inno3d.features.viewer.seg_mask_3d import (
                create_mask_surface_actor,
                add_mask_overlay_to_renderer,
            )

            c1_color = list(getattr(self, "class1_color", None) or [1.0, 1.0, 0.0])
            c2_color = list(getattr(self, "class2_color", None) or [1.0, 0.0, 0.0])
            bump_seg = getattr(self, "bump_segmentation", None)
            void_seg = getattr(self, "void_segmentation", None)

            if bump_seg is not None:
                c1_actor = create_mask_surface_actor(
                    bump_seg, spacing, c1_color, base_opacity=0.35, smooth=True
                )
                if c1_actor is not None:
                    add_mask_overlay_to_renderer(renderer, c1_actor)
                    self._3d_c1_actor = c1_actor
            else:
                self._3d_c1_actor = None

            if void_seg is not None:
                c2_actor = create_mask_surface_actor(
                    void_seg, spacing, c2_color, base_opacity=0.48, smooth=True
                )
                if c2_actor is not None:
                    add_mask_overlay_to_renderer(renderer, c2_actor)
                    self._3d_c2_actor = c2_actor
            else:
                self._3d_c2_actor = None

            print(
                f"[TEACHING 3D] Seg Overlay glass shells "
                f"c1={'yes' if self._3d_c1_actor else 'no'} "
                f"c2={'yes' if self._3d_c2_actor else 'no'} "
                f"spacing={spacing}"
            )

        # Re-apply clip planes to boundary actors
        for actor in getattr(self, "_3d_boundary_actors", []) or []:
            try:
                m = actor.GetMapper() if hasattr(actor, "GetMapper") else None
                self._apply_clip_to_mapper(m, self._collect_3d_clip_planes())
            except Exception:
                pass
        
        # Only reset camera on first render or data change
        if getattr(self, '_3d_needs_initial_camera', True):
            renderer.ResetCamera()
            cam = renderer.GetActiveCamera()
            cam.Elevation(20)
            cam.Azimuth(-32)
            renderer.ResetCameraClippingRange()
            self._3d_needs_initial_camera = False

        # Orbit pivot for Dragonfly navigation (volume center + radius)
        self._sync_3d_orbit_pivot()
        
        # Apply GPU-native cropping to all mappers
        self._apply_3d_clipping(render=False)

        # RGB crosshair axes (after volume/masks; final Render below)
        if hasattr(self, "update_3d_crosshair"):
            try:
                self.update_3d_crosshair(render=False)
            except Exception:
                pass
        
        self._3d_vtk_widget.GetRenderWindow().Render()

    # ── Boundary clipping helpers ──────────────────────────────────────────

    def _get_3d_world_extent(self):
        """Get world-space extents for each VTK axis.
        
        Returns dict: {'x': (min, max), 'y': (min, max), 'z': (min, max)}
        
        The VTK volume sits at origin (0,0,0) with extents:
          X: [0, (dim_x - 1) * spacing_x]
          Y: [0, (dim_y - 1) * spacing_y]
          Z: [0, (dim_z - 1) * spacing_z]
        
        Primary source: read directly from vtkImageData (accounts for any
        downsampling or re-spacing applied during render_teaching_3d).
        Fallback: compute from volume shape x spacing.
        
        Works correctly regardless of whether the volume is horizontal,
        vertical, tall, or flat -- the coordinate system is purely VTK world.
        """
        if getattr(self, "_3d_vtk_data", None) is not None:
            dims = self._3d_vtk_data.GetDimensions()  # (nx, ny, nz)
            sp = self._3d_vtk_data.GetSpacing()
            return {
                "x": (0.0, max((dims[0] - 1) * sp[0], 1e-6)),
                "y": (0.0, max((dims[1] - 1) * sp[1], 1e-6)),
                "z": (0.0, max((dims[2] - 1) * sp[2], 1e-6)),
            }
        # Fallback: full volume * spacing
        if self.volume_data is not None:
            z, y, x = self.volume_data.shape
            spacing = getattr(self, "custom_spacing", None) or getattr(
                self, "spacing", (1.0, 1.0, 1.0)
            )
            sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
            return {
                "x": (0.0, max((x - 1) * sx, 1e-6)),
                "y": (0.0, max((y - 1) * sy, 1e-6)),
                "z": (0.0, max((z - 1) * sz, 1e-6)),
            }
        return {"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)}

    def _build_3d_clip_plane(self, axis, slider_val):
        """Build a vtkPlane for the given axis from a slider value (0..1000).
        
        Dragonfly-style clipping -- orientation-agnostic:
        - slider_val 0..1000 maps to 0..100% of the axis world extent
        - VTK clips everything where dot(normal, point - origin) < 0
        - Default normal = +1: keeps everything ABOVE the slider position
        - Flipped normal = -1: keeps everything BELOW the slider position
        
        This works correctly regardless of volume shape or aspect ratio
        because it operates purely in VTK world coordinates.
        """
        extent = self._get_3d_world_extent()
        amin, amax = extent[axis]
        t = max(0.0, min(1.0, slider_val / 1000.0))
        pos = amin + t * (amax - amin)

        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        normal = [0.0, 0.0, 0.0]
        # +1 = keep positive side, -1 = keep negative side
        sign = -1.0 if self._3d_clip_flip.get(axis, False) else 1.0
        normal[axis_idx] = sign

        origin = [0.0, 0.0, 0.0]
        origin[axis_idx] = pos

        plane = vtk.vtkPlane()
        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)
        return plane

    def _collect_3d_clip_planes(self):
        planes = []
        for ax in ("x", "y", "z"):
            if self._3d_clip_enabled.get(ax) and self._3d_clip_planes.get(ax) is not None:
                planes.append(self._3d_clip_planes[ax])
        return planes

    def _apply_clip_to_mapper(self, mapper, planes):
        if mapper is None or not hasattr(mapper, "RemoveAllClippingPlanes"):
            return
        mapper.RemoveAllClippingPlanes()
        for p in planes:
            mapper.AddClippingPlane(p)
        mapper.Modified()  # Force VTK pipeline to recognize clip changes

    def _apply_3d_clipping(self, render=True):
        """Apply clipping using GPU-native cropping for volume mapper.
        
        Uses SetCropping/SetCroppingRegionPlanes for vtkGPUVolumeRayCastMapper
        to avoid rendering artifacts from certain viewing angles.
        Seg mesh and boundary actors use standard ClippingPlanes.
        """
        planes = self._collect_3d_clip_planes()
        extent = self._get_3d_world_extent()
        
        # GPU-native cropping for volume mapper
        if getattr(self, "_3d_volume_actor", None) is not None:
            mapper = self._3d_volume_actor.GetMapper()
            if mapper is not None and hasattr(mapper, 'SetCropping'):
                mapper.RemoveAllClippingPlanes()
                if planes:
                    crop_min = [extent['x'][0], extent['y'][0], extent['z'][0]]
                    crop_max = [extent['x'][1], extent['y'][1], extent['z'][1]]
                    for p in planes:
                        normal = p.GetNormal()
                        origin = p.GetOrigin()
                        for i in range(3):
                            if abs(normal[i]) > 0.5:
                                if normal[i] > 0:
                                    crop_min[i] = origin[i]
                                else:
                                    crop_max[i] = origin[i]
                    mapper.SetCropping(1)
                    mapper.SetCroppingRegionPlanes(
                        crop_min[0], crop_max[0],
                        crop_min[1], crop_max[1],
                        crop_min[2], crop_max[2]
                    )
                    mapper.SetCroppingRegionFlags(0x2000)  # VTK_CROP_SUBVOLUME
                else:
                    mapper.SetCropping(0)
                mapper.Modified()

        # Seg mesh + boundary actors use standard ClippingPlanes
        if getattr(self, "_3d_seg_actor", None) is not None:
            self._apply_clip_to_mapper(self._3d_seg_actor.GetMapper(), planes)

        for actor in getattr(self, "_3d_boundary_actors", []) or []:
            try:
                mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
                self._apply_clip_to_mapper(mapper, planes)
            except Exception:
                pass

        if render and getattr(self, "_3d_vtk_widget", None) is not None:
            self._3d_vtk_widget.GetRenderWindow().Render()

    def _on_3d_clip_toggled(self, axis, enabled):
        self._3d_clip_enabled[axis] = enabled
        ctrl = self._3d_clip_controls.get(axis, {})
        if "slider" in ctrl:
            ctrl["slider"].setEnabled(enabled)
        if "flip" in ctrl:
            ctrl["flip"].setEnabled(enabled)

        if enabled:
            val = ctrl["slider"].value() if "slider" in ctrl else 500
            self._3d_clip_planes[axis] = self._build_3d_clip_plane(axis, val)
        else:
            self._3d_clip_planes[axis] = None

        self._apply_3d_clipping(render=True)

    def _on_3d_clip_moved(self, axis, value):
        ctrl = self._3d_clip_controls.get(axis, {})
        if "label" in ctrl:
            ctrl["label"].setText(f"{value / 10.0:.0f}%")
        if not self._3d_clip_enabled.get(axis):
            return
        self._3d_clip_planes[axis] = self._build_3d_clip_plane(axis, value)
        self._apply_3d_clipping(render=True)

    def _on_3d_clip_flip(self, axis, flipped):
        self._3d_clip_flip[axis] = bool(flipped)
        if not self._3d_clip_enabled.get(axis):
            return
        ctrl = self._3d_clip_controls.get(axis, {})
        val = ctrl["slider"].value() if "slider" in ctrl else 500
        self._3d_clip_planes[axis] = self._build_3d_clip_plane(axis, val)
        self._apply_3d_clipping(render=True)

    def _reset_3d_clips(self):
        """Disable all boundary clip planes."""
        for axis in ("x", "y", "z"):
            self._3d_clip_enabled[axis] = False
            self._3d_clip_flip[axis] = False
            self._3d_clip_planes[axis] = None
            ctrl = self._3d_clip_controls.get(axis, {})
            if "check" in ctrl:
                ctrl["check"].blockSignals(True)
                ctrl["check"].setChecked(False)
                ctrl["check"].blockSignals(False)
            if "slider" in ctrl:
                ctrl["slider"].blockSignals(True)
                ctrl["slider"].setValue(500)
                ctrl["slider"].setEnabled(False)
                ctrl["slider"].blockSignals(False)
            if "label" in ctrl:
                ctrl["label"].setText("50%")
            if "flip" in ctrl:
                ctrl["flip"].blockSignals(True)
                ctrl["flip"].setChecked(False)
                ctrl["flip"].setEnabled(False)
                ctrl["flip"].blockSignals(False)
        self._apply_3d_clipping(render=True)

    def _get_b2b_gap_overlay(self) -> B2BGapOverlay:
        """Shared B2B gap actor stack (same module as Viewer stats_panel)."""
        ov = getattr(self, "_b2b_gap_overlay", None)
        if ov is None:
            ov = B2BGapOverlay()
            self._b2b_gap_overlay = ov
        return ov

    # ── MES 3D highlight (shared MESHighlightOverlay — Viewer SoT) ───────

    def _get_mes_highlight_overlay(self) -> MESHighlightOverlay:
        ov = getattr(self, "_mes_highlight_overlay", None)
        if ov is None:
            ov = MESHighlightOverlay()
            self._mes_highlight_overlay = ov
        return ov

    def _clear_mes_3d_highlight(self, render=True, restore_context=True):
        """Remove MES pick surfaces from Teaching 3D renderer."""
        ren = getattr(self, "_3d_renderer", None)
        widget = getattr(self, "_3d_vtk_widget", None)
        ov = self._get_mes_highlight_overlay()
        ov.clear(ren, widget, render=render)
        self._mes_3d_highlight_actors = ov.actors

    def _update_mes_3d_highlight(self):
        """Cyan MES pick surfaces on Teaching 3D (shared overlay).

        Only draws when 3D view is already active — does not force-enable 3D
        (VRAM-conscious Teaching default).
        """
        if not getattr(self, "_3d_view_active", False):
            return
        ren = getattr(self, "_3d_renderer", None)
        widget = getattr(self, "_3d_vtk_widget", None)
        labeled = getattr(self, "labeled_class1_data", None)
        if ren is None or labeled is None or self.volume_data is None:
            self._clear_mes_3d_highlight(render=False)
            return
        if not getattr(self, "selected_highlight_objects", None):
            self._clear_mes_3d_highlight(render=True)
            return

        # Drop B2B gap so MES pick is the spotlight (same as Viewer)
        if hasattr(self, "_clear_boundary_actors"):
            try:
                self._clear_boundary_actors()
            except Exception:
                pass

        ov = self._get_mes_highlight_overlay()
        ov.update(
            ren,
            widget,
            labeled,
            getattr(self, "object_stats", None),
            self.selected_highlight_objects,
            self._get_3d_world_spacing(),
            z_offset=int(getattr(self, "_measurement_start_slice", 0) or 0),
            highlight_color=getattr(self, "highlight_color", [0.0, 1.0, 1.0]),
            on_dim_context=None,
            on_restore_context=None,
        )
        self._mes_3d_highlight_actors = ov.actors

    def _clear_boundary_actors(self):
        """Remove B2B gap actors from Teaching 3D renderer (shared overlay)."""
        ren = getattr(self, "_3d_renderer", None)
        widget = getattr(self, "_3d_vtk_widget", None)
        ov = self._get_b2b_gap_overlay()
        ov.clear(ren, widget, render=True)
        # Legacy lists kept empty so re-render paths do not double-remove
        self._3d_boundary_actors = []
        self._3d_b2b_2d_actors = []

    def _draw_boundary_gap_line(self, src_voxel, dst_voxel, label_text=""):
        """Draw Src→Dst B2B gap via shared B2BGapOverlay (Viewer SoT)."""
        if not getattr(self, "_3d_view_active", False) or not hasattr(self, "_3d_renderer"):
            return
        ren = self._3d_renderer
        widget = getattr(self, "_3d_vtk_widget", None)
        if self.volume_data is None:
            print("[B2B] 3D gap skipped: no volume")
            return

        ov = self._get_b2b_gap_overlay()
        spacing = self._get_3d_world_spacing()
        # Drop MES pick stack so B2B is the only spotlight (Viewer parity)
        if hasattr(self, "_clear_mes_3d_highlight"):
            try:
                self._clear_mes_3d_highlight(render=False)
            except Exception:
                pass
        # Teaching uses bump_segmentation as class1 binary fallback
        c1 = getattr(self, "bump_segmentation", None)
        if c1 is None:
            c1 = getattr(self, "class1_data", None)

        ok = ov.draw(
            ren,
            widget,
            src_voxel,
            dst_voxel,
            spacing,
            label_text=label_text,
            labeled=getattr(self, "labeled_class1_data", None),
            class1_binary=c1,
            on_dim_context=None,
            volume_world_radius=None,
            sync_orbit_pivot=None,
            frame_camera=True,
        )
        # Keep legacy actor list in sync for re-render paths
        self._3d_boundary_actors = list(ov.actors)
        self._3d_b2b_2d_actors = []
        if hasattr(self, "_3d_bnd_info") and ok:
            try:
                self._3d_bnd_info.setText(f"Gap: {label_text}")
            except Exception:
                pass

    # ── Viewer-grade Transfer Function (Dragonfly metallic) ───────────────

    def _apply_3d_transfer_function(self, vol_prop):
        """Apply Dragonfly-style metallic grayscale TF — mirrors viewer.py apply_transfer_function."""
        v_min = float(self._3d_min_slider.value()) if hasattr(self, '_3d_min_slider') else 0.0
        v_max = float(self._3d_max_slider.value()) if hasattr(self, '_3d_max_slider') else 65535.0
        if v_max <= v_min:
            v_max = v_min + 1.0

        opacity_scale = self._3d_opacity_slider.value() / 100.0 if hasattr(self, '_3d_opacity_slider') else 0.3

        # Dragonfly Silver/Grayscale metallic ramp (same as viewer.py fallback)
        r, g, b = getattr(self, '_3d_volume_color', [0.95, 0.95, 0.95])
        dark = (r * 0.25, g * 0.25, b * 0.25)
        mid_c = (r * 0.75, g * 0.75, b * 0.75)
        mid = v_min + 0.6 * (v_max - v_min)
        highlight = (min(1.0, r * 1.25), min(1.0, g * 1.25 + 0.05), min(1.0, b * 1.25 + 0.25))

        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(v_min, 0.0, 0.0, 0.0)
        color_func.AddRGBPoint(v_min + 0.3 * (v_max - v_min), *dark)
        color_func.AddRGBPoint(mid, *mid_c)
        color_func.AddRGBPoint(v_min + 0.95 * (v_max - v_min), r, g, b)
        color_func.AddRGBPoint(v_max, *highlight)

        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(v_min, 0.0)
        opacity_func.AddPoint(v_min + 0.3 * (v_max - v_min), 0.0)
        opacity_func.AddPoint(mid, 0.25 * opacity_scale)
        opacity_func.AddPoint(v_max, 0.9 * opacity_scale)

        vol_prop.SetColor(color_func)
        vol_prop.SetScalarOpacity(opacity_func)

    # ── In-place property updates (no actor rebuild, no camera reset) ───

    def _update_3d_transfer_function(self, *args):
        """Update TF on existing actor — mirrors viewer.py update_transfer_function."""
        if not self._3d_view_active or self._3d_volume_actor is None:
            return
        self._apply_3d_transfer_function(self._3d_volume_actor.GetProperty())
        self._3d_vtk_widget.GetRenderWindow().Render()

    def _on_3d_mode_changed(self, mode):
        """Mode change requires full rebuild (different data source)."""
        # Remember if user explicitly chose grey-only Volume
        self._3d_user_forced_volume_mode = str(mode) == "Volume"
        if self._3d_view_active:
            self._3d_needs_initial_camera = False  # Keep current camera
            self._render_teaching_3d()

    def _on_3d_quality_changed(self, quality):
        """Update mapper quality in-place — mirrors viewer.py _on_quality_preset_changed."""
        mapper = getattr(self, '_3d_volume_mapper', None)
        if not self._3d_view_active or mapper is None:
            return
        preset = self._3d_quality_presets.get(quality, {})
        mapper.SetSampleDistance(preset.get('sample_dist', 0.5))
        if hasattr(mapper, 'SetImageSampleDistance'):
            mapper.SetImageSampleDistance(preset.get('image_sample', 1.0))
        mapper.SetAutoAdjustSampleDistances(1 if preset.get('auto_adj', True) else 0)
        if hasattr(mapper, 'SetUseJittering'):
            mapper.SetUseJittering(1 if preset.get('jitter', True) else 0)
        if hasattr(mapper, 'SetMaxMemoryFraction'):
            mapper.SetMaxMemoryFraction(preset.get('max_mem_fraction', 0.85))
        self._3d_vtk_widget.GetRenderWindow().Render()

    def _on_3d_opacity_changed(self, value):
        """Update opacity on existing actor — no rebuild."""
        if not self._3d_view_active or self._3d_volume_actor is None:
            return
        self._apply_3d_transfer_function(self._3d_volume_actor.GetProperty())
        self._3d_vtk_widget.GetRenderWindow().Render()

    # ── LOD Interaction (same as viewer.py) ─────────────────────────────

    def _start_3d_lod_interaction(self):
        """Reduce quality during mouse rotation/zoom for smooth interaction."""
        self._3d_is_interacting = True
        if hasattr(self, '_3d_lod_refine_timer'):
            self._3d_lod_refine_timer.stop()
        mapper = getattr(self, '_3d_volume_mapper', None)
        if mapper is None:
            return
        quality_name = self._3d_quality_combo.currentText() if hasattr(self, '_3d_quality_combo') else "Draft"
        preset = self._3d_quality_presets.get(quality_name, {})
        mapper.SetSampleDistance(preset.get('interact_sample_dist', 2.0))
        if hasattr(mapper, 'SetImageSampleDistance'):
            mapper.SetImageSampleDistance(preset.get('interact_image_sample', 2.0))
        mapper.SetAutoAdjustSampleDistances(1)

    def _end_3d_lod_interaction(self):
        """Schedule quality refinement after interaction stops."""
        self._3d_is_interacting = False
        if hasattr(self, '_3d_lod_refine_timer'):
            self._3d_lod_refine_timer.start()

    def _refine_3d_after_interaction(self):
        """Restore full-quality rendering after interaction ends."""
        if getattr(self, '_3d_is_interacting', False):
            return
        mapper = getattr(self, '_3d_volume_mapper', None)
        if mapper is None:
            return
        quality_name = self._3d_quality_combo.currentText() if hasattr(self, '_3d_quality_combo') else "Draft"
        preset = self._3d_quality_presets.get(quality_name, {})
        mapper.SetSampleDistance(preset.get('sample_dist', 0.5))
        if hasattr(mapper, 'SetImageSampleDistance'):
            mapper.SetImageSampleDistance(preset.get('image_sample', 1.0))
        mapper.SetAutoAdjustSampleDistances(1 if preset.get('auto_adj', True) else 0)
        
        # Re-sync clip planes (defensive — ensures they persist through interaction)
        self._apply_3d_clipping(render=False)
        
        if hasattr(self, '_3d_vtk_widget') and self._3d_vtk_widget:
            self._3d_vtk_widget.GetRenderWindow().Render()

    def _volume_world_center(self):
        """Center of the loaded volume in VTK world coordinates (spacing-aware)."""
        if self.volume_data is None:
            return (0.0, 0.0, 0.0)
        z, y, x = self.volume_data.shape
        sp = getattr(self, 'custom_spacing', None) or getattr(self, 'spacing', [1.0, 1.0, 1.0])
        return (
            (x - 1) * float(sp[0]) * 0.5,
            (y - 1) * float(sp[1]) * 0.5,
            (z - 1) * float(sp[2]) * 0.5,
        )

    def _volume_world_radius(self):
        """Half-diagonal of the volume AABB — used for zoom distance clamps."""
        if self.volume_data is None:
            return 100.0
        z, y, x = self.volume_data.shape
        sp = getattr(self, 'custom_spacing', None) or getattr(self, 'spacing', [1.0, 1.0, 1.0])
        hx = (x - 1) * float(sp[0]) * 0.5
        hy = (y - 1) * float(sp[1]) * 0.5
        hz = (z - 1) * float(sp[2]) * 0.5
        return max((hx * hx + hy * hy + hz * hz) ** 0.5, 1.0)

    def _volume_world_bounds(self):
        """World AABB of the loaded volume (origin 0 + spacing)."""
        if self.volume_data is None:
            return None
        sp = getattr(self, 'custom_spacing', None) or getattr(self, 'spacing', [1.0, 1.0, 1.0])
        return volume_world_aabb(self.volume_data.shape, sp)

    def _sync_3d_orbit_pivot(self):
        """Keep Dragonfly interactor orbiting the volume center (same as viewer)."""
        style = getattr(self, '_3d_interactor_style', None)
        if style is None or not hasattr(style, 'SetOrbitPivot'):
            return
        cx, cy, cz = self._volume_world_center()
        style.SetOrbitPivot(
            cx, cy, cz,
            radius=self._volume_world_radius(),
            bounds=self._volume_world_bounds(),
        )

    # ── Orientation cube (shared Viewer SoT: hover + click-to-orient) ─────

    def _orientation_cube_render(self):
        widget = getattr(self, "_3d_vtk_widget", None)
        if widget is not None:
            try:
                widget.GetRenderWindow().Render()
            except Exception:
                pass

    def _set_orientation_face_hover(self, face):
        """Highlight one cube face in app primary cyan (or clear if face is None)."""
        from inno3d.features.shared.orientation_cube import set_orientation_face_hover

        state = getattr(self, "_ori_cube_state", None)
        if state is None:
            # Legacy attrs fallback
            state = {
                "cube": getattr(self, "_axes_cube_actor", None),
                "face_highlights": getattr(self, "_ori_face_highlights", {}) or {},
                "face_text_props": getattr(self, "_ori_face_text_props", {}) or {},
                "hover_face": getattr(self, "_ori_hover_face", None),
            }
            self._ori_cube_state = state
        if set_orientation_face_hover(state, face):
            self._ori_hover_face = face
            self._orientation_cube_render()

    def _is_over_orientation_viewport(self, display_x, display_y):
        from inno3d.features.shared.orientation_cube import is_over_orientation_viewport

        axes = getattr(self, "_3d_axes_widget", None)
        widget = getattr(self, "_3d_vtk_widget", None)
        if axes is None or widget is None:
            return False
        size = widget.GetRenderWindow().GetSize()
        if not size or size[0] <= 0 or size[1] <= 0:
            return False
        return is_over_orientation_viewport(
            display_x, display_y, size[0], size[1], axes.GetViewport()
        )

    def _pick_orientation_cube_face(self, display_x, display_y):
        from inno3d.features.shared.orientation_cube import pick_orientation_cube_face

        axes = getattr(self, "_3d_axes_widget", None)
        widget = getattr(self, "_3d_vtk_widget", None)
        ren = getattr(self, "_3d_renderer", None)
        if axes is None or widget is None or ren is None:
            return None
        size = widget.GetRenderWindow().GetSize()
        if not size or size[0] <= 0 or size[1] <= 0:
            return None
        return pick_orientation_cube_face(
            display_x,
            display_y,
            size[0],
            size[1],
            axes.GetViewport(),
            ren.GetActiveCamera(),
        )

    def _on_orientation_cube_hover(self, obj, event):
        """Mouse-move: glow face under cursor (Viewer parity)."""
        if not getattr(self, "_3d_view_active", False):
            return
        axes = getattr(self, "_3d_axes_widget", None)
        widget = getattr(self, "_3d_vtk_widget", None)
        if axes is None or widget is None:
            return
        iren = widget.GetRenderWindow().GetInteractor()
        if iren is None:
            return
        x, y = iren.GetEventPosition()
        face = self._pick_orientation_cube_face(x, y)
        over_marker = self._is_over_orientation_viewport(x, y)
        try:
            if face is not None:
                widget.setCursor(Qt.PointingHandCursor)
            elif over_marker:
                widget.setCursor(Qt.ArrowCursor)
            elif getattr(self, "_ori_hover_face", None) is not None:
                widget.unsetCursor()
        except Exception:
            pass
        self._set_orientation_face_hover(face)

    def _on_orientation_cube_click(self, obj, event):
        """Left-click ±X/±Y/±Z cube → orient Teaching 3D camera to that face."""
        if not getattr(self, "_3d_view_active", False):
            return
        axes = getattr(self, "_3d_axes_widget", None)
        widget = getattr(self, "_3d_vtk_widget", None)
        if axes is None or widget is None:
            return
        iren = widget.GetRenderWindow().GetInteractor()
        if iren is None:
            return
        x, y = iren.GetEventPosition()
        face = self._pick_orientation_cube_face(x, y)
        if face is None:
            return
        try:
            iren.SetAbortFlag(1)
        except Exception:
            pass
        self._set_orientation_face_hover(face)
        self._orient_camera_to_face(face)

    def _orient_camera_to_face(self, face):
        """Snap 3D camera so the given cube face points toward the viewer."""
        from inno3d.features.shared.orientation_cube import apply_camera_face_preset

        ren = getattr(self, "_3d_renderer", None)
        widget = getattr(self, "_3d_vtk_widget", None)
        if ren is None:
            return
        cam = ren.GetActiveCamera()
        if self.volume_data is not None:
            cx, cy, cz = self._volume_world_center()
            r = max(self._volume_world_radius() * 2.5, 1.0)
        else:
            cx = cy = cz = 0.0
            r = 500.0
        if not apply_camera_face_preset(
            cam, face, center=(cx, cy, cz), radius=r
        ):
            return
        try:
            self._sync_3d_orbit_pivot()
        except Exception:
            pass
        ren.ResetCameraClippingRange()
        if widget is not None:
            try:
                widget.GetRenderWindow().Render()
            except Exception:
                pass

    def _dragonfly_3d_zoom(self, zoom_in=True, strength=1.0, display_xy=None):
        """ORS Dragonfly object zoom — same algorithm as 3D Viewer tab.

        Dolly only; camera stays outside volume (no free-flight through center).
        """
        ren = getattr(self, '_3d_renderer', None)
        widget = getattr(self, '_3d_vtk_widget', None)
        if ren is None or widget is None or self.volume_data is None:
            return
        if not getattr(self, '_3d_view_active', False):
            return

        try:
            self._start_3d_lod_interaction()
            apply_dragonfly_volume_zoom(
                ren,
                zoom_in=zoom_in,
                strength=strength,
                volume_center=self._volume_world_center(),
                volume_radius=self._volume_world_radius(),
                volume_bounds=self._volume_world_bounds(),
                display_xy=None,  # cursor-pan disabled (QVTK multi-pane crash risk)
            )
            rw = widget.GetRenderWindow()
            if rw is not None:
                rw.Render()
        except Exception as e:
            print(f">>> Teaching 3D zoom error (swallowed): {e}")
        finally:
            QTimer.singleShot(150, self._end_3d_lod_interaction)

    def _set_3d_camera_preset(self, direction):
        """Set the 3D camera to a preset view direction (volume-center based)."""
        if not hasattr(self, '_3d_renderer') or self.volume_data is None:
            return
        cam = self._3d_renderer.GetActiveCamera()
        cx, cy, cz = self._volume_world_center()
        R = self._volume_world_radius()
        dist = max(cam.GetDistance(), R * 2.5)

        # direction = (dx, dy, dz, ux, uy, uz) unit look-from offset + view-up
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(
            cx + direction[0] * dist,
            cy + direction[1] * dist,
            cz + direction[2] * dist,
        )
        cam.SetViewUp(direction[3], direction[4], direction[5])
        self._sync_3d_orbit_pivot()
        self._3d_renderer.ResetCameraClippingRange()
        if self._3d_view_active and hasattr(self, '_3d_vtk_widget'):
            self._3d_vtk_widget.GetRenderWindow().Render()


    def _param_label(self, text, tooltip=None):
        """Parameter name label (above value row)."""
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8.5pt; font-weight: 600;"
        )
        if tooltip:
            lbl.setToolTip(tooltip)
        return lbl

    def _param_section_layout(self):
        """Vertical stack inside a param group (name-first rows)."""
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(8)
        return layout

    def _style_param_value_widget(self, widget, width=88):
        """Constrain spin/edit width so labels stay readable in sidebar."""
        widget.setMinimumWidth(width)
        widget.setMaximumWidth(max(width, 120))
        widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return widget

    def _param_axis_chip_qss(self):
        return (
            f"color: {SemiconductorTheme.TEXT_SECONDARY};"
            f"font-size: 7.5pt; font-weight: 800;"
            f"min-width: 10px;"
        )

    def _make_xyz_spin_row(
        self,
        key_prefix,
        defaults,
        *,
        key_style="suffix",  # "suffix" → keyPrefixX / "lower" → key_prefix_x
        is_int=False,
        range_min=0.0,
        range_max=100.0,
        step=0.5,
        decimals=2,
        unit_tooltip="voxels",
    ):
        """One parameter name → X Y Z spins evenly spanning the sidebar width."""
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        # Wider gaps so X/Y/Z cells breathe across the full panel width
        lay.setSpacing(10)

        axes = ["X", "Y", "Z"]
        for i, axis in enumerate(axes):
            # Each axis is an equal-width cell: chip + expanding spin
            cell = QWidget()
            cell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            cell_lay = QHBoxLayout(cell)
            cell_lay.setContentsMargins(0, 0, 0, 0)
            cell_lay.setSpacing(4)

            chip = QLabel(axis)
            chip.setAlignment(Qt.AlignCenter)
            chip.setStyleSheet(self._param_axis_chip_qss())
            chip.setFixedWidth(12)
            cell_lay.addWidget(chip)

            if is_int:
                spin = NoScrollSpinBox()
                spin.setRange(int(range_min), int(range_max))
                spin.setValue(int(defaults[i]))
                spin.setSingleStep(int(step) if step >= 1 else 1)
            else:
                spin = NoScrollDoubleSpinBox()
                spin.setRange(float(range_min), float(range_max))
                spin.setDecimals(decimals)
                spin.setSingleStep(step)
                spin.setValue(float(defaults[i]))

            spin.setMinimumWidth(56)
            spin.setMinimumHeight(22)
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            spin.setToolTip(f"{axis} ({unit_tooltip})")
            spin.valueChanged.connect(self.on_parameter_changed)
            cell_lay.addWidget(spin, 1)

            if key_style == "lower":
                self.param_widgets[f"{key_prefix}_{axis.lower()}"] = spin
            else:
                self.param_widgets[f"{key_prefix}{axis}"] = spin

            lay.addWidget(cell, 1)  # equal horizontal stretch for X, Y, Z

        return row

    def _add_named_param_block(self, layout, name, value_widget, tooltip=None):
        """Name on its own line, then the value widget (XYZ row or single control)."""
        layout.addWidget(self._param_label(name, tooltip))
        layout.addWidget(value_widget)

    def _add_single_spin_block(
        self,
        layout,
        name,
        widget_key,
        *,
        default,
        is_int=False,
        range_min=0.0,
        range_max=100.0,
        step=0.1,
        decimals=2,
        tooltip=None,
    ):
        """Name + single spin on one row (for non-XYZ scalars)."""
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self._param_label(name, tooltip), 1)

        if is_int:
            spin = NoScrollSpinBox()
            spin.setRange(int(range_min), int(range_max))
            spin.setValue(int(default))
            spin.setSingleStep(int(step) if step >= 1 else 1)
        else:
            spin = NoScrollDoubleSpinBox()
            spin.setRange(float(range_min), float(range_max))
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setValue(float(default))

        self._style_param_value_widget(spin, width=88)
        spin.valueChanged.connect(self.on_parameter_changed)
        lay.addWidget(spin, 0, Qt.AlignRight)
        self.param_widgets[widget_key] = spin
        layout.addWidget(row)
        return spin

    def _param_sticky_header_qss(self):
        return f"""
            QFrame#ParamStickyHeader {{
                background: {SemiconductorTheme.BG_LIGHT};
                border: 1px solid {SemiconductorTheme.ACCENT_PRIMARY};
                border-radius: 4px;
            }}
            QLabel#ParamStickyTitle {{
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                font-size: 9pt;
                font-weight: 800;
                letter-spacing: 0.3px;
            }}
            QLabel#ParamStickyHint {{
                color: {SemiconductorTheme.TEXT_SECONDARY};
                font-size: 7.5pt;
            }}
        """

    def _update_param_sticky_section(self):
        """Show which param group is currently under the scroll viewport."""
        if not hasattr(self, "_param_scroll") or not hasattr(self, "_param_sections"):
            return
        if not self._param_sections:
            return

        scroll = self._param_scroll
        content = scroll.widget()
        if content is None:
            return

        # Viewport top in content coordinates
        y_top = scroll.verticalScrollBar().value()
        # Prefer the last section whose top is at/above the sticky threshold
        threshold = y_top + 12
        current_name = self._param_sections[0][0]
        for name, widget in self._param_sections:
            # mapTo(content) handles nested layout positions
            top = widget.mapTo(content, QPoint(0, 0)).y()
            if top <= threshold:
                current_name = name
            else:
                break

        if hasattr(self, "_param_sticky_title"):
            self._param_sticky_title.setText(current_name)

    def create_parameter_panel(self):
        """Scrollable params: name-first XYZ rows + sticky current-group header."""
        panel = QWidget()
        self._theme_register("_theme_panels", panel)
        panel.setStyleSheet(f"background: {SemiconductorTheme.BG_PANEL}; border-radius: 6px;")
        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(6, 6, 6, 6)
        panel_layout.setSpacing(6)

        header_label = QLabel("<b>CONFIGURATION PARAMETERS</b>")
        self._theme_register("_theme_accent_labels", header_label)
        header_label.setStyleSheet(f"font-size: 10pt; color: {SemiconductorTheme.ACCENT_PRIMARY};")
        panel_layout.addWidget(header_label)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(4)

        self.back_to_stats_btn = QPushButton("MES")
        self._theme_register("_theme_secondary_buttons", self.back_to_stats_btn)
        self.back_to_stats_btn.setStyleSheet(self._secondary_button_qss())
        self.back_to_stats_btn.setToolTip("Back to MES (Object Statistics)")
        self.back_to_stats_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.back_to_stats_btn.clicked.connect(lambda: self._set_teaching_tab(4))
        actions.addWidget(self.back_to_stats_btn, 1)

        self.save_config_btn = QPushButton("Save")
        self._theme_register("_theme_secondary_buttons", self.save_config_btn)
        self.save_config_btn.setStyleSheet(self._secondary_button_qss())
        self.save_config_btn.setToolTip("Save configuration to file")
        self.save_config_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.save_config_btn.clicked.connect(self.save_config)
        actions.addWidget(self.save_config_btn, 1)

        self.reset_config_btn = QPushButton("Reset")
        self._theme_register("_theme_secondary_buttons", self.reset_config_btn)
        self.reset_config_btn.setStyleSheet(self._secondary_button_qss())
        self.reset_config_btn.setToolTip("Reset config to defaults")
        self.reset_config_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.reset_config_btn.clicked.connect(self.reset_config)
        actions.addWidget(self.reset_config_btn, 1)

        panel_layout.addLayout(actions)

        # Sticky section indicator (always visible while scrolling groups below)
        sticky = QFrame()
        sticky.setObjectName("ParamStickyHeader")
        sticky.setStyleSheet(self._param_sticky_header_qss())
        sticky.setFixedHeight(30)
        sticky_lay = QHBoxLayout(sticky)
        sticky_lay.setContentsMargins(8, 2, 8, 2)
        sticky_lay.setSpacing(8)

        pin = QLabel("§")
        pin.setStyleSheet(
            f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-weight: 900; font-size: 10pt;"
        )
        sticky_lay.addWidget(pin)

        self._param_sticky_title = QLabel("Bump Detection")
        self._param_sticky_title.setObjectName("ParamStickyTitle")
        sticky_lay.addWidget(self._param_sticky_title, 1)

        hint = QLabel("current group")
        hint.setObjectName("ParamStickyHint")
        sticky_lay.addWidget(hint)

        self._param_sticky_frame = sticky
        panel_layout.addWidget(sticky)

        # Scrollable area — vertical only
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self._param_scroll = scroll

        scroll_content = QWidget()
        scroll_content.setMinimumWidth(0)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 4, 0)
        scroll_layout.setSpacing(10)

        self.warning_label = QLabel("Note: Changing parameters will affect segmentation.")
        self._theme_register("_theme_warning_labels", self.warning_label)
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(
            f"color: {SemiconductorTheme.ACCENT_WARNING}; font-size: 8pt; font-weight: bold; padding: 2px;"
        )
        self.warning_label.hide()
        scroll_layout.addWidget(self.warning_label)

        # Sections: name-first layout + registered for sticky tracking
        self._param_sections = []  # list of (title, widget)
        section_builders = [
            ("Bump Detection", self.create_bump_parameter_group),
            ("Void Detection", self.create_void_parameter_group),
            ("Blank Slices (Air)", self.create_blank_slices_group),
            ("Measurement (Voxel & FAR)", self.create_measurement_parameter_group),
            ("Debug Options", self.create_options_group),
        ]
        for title, builder in section_builders:
            group = builder()
            self._theme_register("_theme_group_boxes", group)
            group.setStyleSheet(self._group_box_qss())
            group.setProperty("param_section_title", title)
            scroll_layout.addWidget(group)
            self._param_sections.append((title, group))

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_content)

        scroll.verticalScrollBar().valueChanged.connect(
            lambda _v: self._update_param_sticky_section()
        )
        # Initial sticky after first layout pass
        QTimer.singleShot(0, self._update_param_sticky_section)

        panel_layout.addWidget(scroll, 1)
        panel.setLayout(panel_layout)
        return panel

    def create_bump_parameter_group(self):
        """Bump params — each parameter name, then X/Y/Z on one row."""
        group = QGroupBox("Bump Detection")
        layout = self._param_section_layout()

        self._add_single_spin_block(
            layout,
            "Threshold Weight",
            "bumpThresholdWeight",
            default=1.5,
            range_min=0.0,
            range_max=100.0,
            step=0.1,
            decimals=2,
        )

        self._add_named_param_block(
            layout,
            "Max Noise Size (vox)",
            self._make_xyz_spin_row(
                "bumpCleanRadius", [2.0, 2.0, 2.0], step=0.5, decimals=2
            ),
        )
        self._add_named_param_block(
            layout,
            "Max Bump Radius (vox)",
            self._make_xyz_spin_row(
                "bumpFillHoleRadius", [3.0, 3.0, 3.0], step=0.5, decimals=2
            ),
        )

        group.setLayout(layout)
        return group

    def create_void_parameter_group(self):
        """Void params — name first, X/Y/Z together."""
        group = QGroupBox("Void Detection")
        layout = self._param_section_layout()

        self._add_single_spin_block(
            layout,
            "Threshold Weight",
            "voidThresholdWeight",
            default=0.6,
            range_min=0.0,
            range_max=100.0,
            step=0.1,
            decimals=2,
        )

        void_params = [
            ("openEmVoid", "Enlarge void size (vox)", [1.0, 1.0, 3.0]),
            ("closeResidue", "Max void radius (vox)", [7.0, 7.0, 3.0]),
            ("openSmallDots", "Particle noise (vox)", [1.0, 1.0, 0.5]),
            ("erodeBump", "Void margin from bump (vox)", [5.0, 5.0, 5.0]),
        ]
        for key_prefix, label, defaults in void_params:
            self._add_named_param_block(
                layout,
                label,
                self._make_xyz_spin_row(key_prefix, defaults, step=0.5, decimals=2),
            )

        group.setLayout(layout)
        return group

    def create_measurement_parameter_group(self):
        """Measurement — Voxel Size as X/Y/Z row; other fields by name."""
        group = QGroupBox("Measurement (Voxel & FAR)")
        layout = self._param_section_layout()

        self._add_named_param_block(
            layout,
            "Voxel Size (µm)",
            self._make_xyz_spin_row(
                "voxel_size",
                [1.0, 1.0, 1.0],
                key_style="lower",
                range_min=0.0001,
                range_max=1000.0,
                step=0.1,
                decimals=4,
                unit_tooltip="µm",
            ),
        )

        z4x_check = QCheckBox("Z-Stretched 4x Data")
        z4x_check.setChecked(False)
        z4x_check.setToolTip(
            "Enable this if your input data has Z spacing stretched by 4x.\n"
            "When enabled, the effective Z voxel size used for measurement\n"
            "will be automatically divided by 4 to correct for the stretch."
        )
        z4x_check.setStyleSheet(f"""
            QCheckBox {{
                color: {SemiconductorTheme.ACCENT_WARNING};
                font-weight: bold;
                font-size: 8.5pt;
                padding: 2px 0px;
            }}
            QCheckBox::indicator {{
                width: 14px; height: 14px;
            }}
        """)
        z4x_check.stateChanged.connect(self.on_parameter_changed)
        layout.addWidget(z4x_check)
        self.param_widgets["z_stretched_4x"] = z4x_check

        size_fields = [
            ("bump_minimum_size", "Min Bump Size (XxYxZ)", "0x0x0"),
            ("bump_maximum_size", "Max Bump Size (XxYxZ)", "1000x1000x1000"),
            ("void_minimum_size", "Min Void Size (XxYxZ)", "0x0x0"),
            ("void_maximum_size", "Max Void Size (XxYxZ)", "1000x1000x1000"),
        ]
        for key, label, default in size_fields:
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(6)
            lay.addWidget(self._param_label(label), 1)
            edit = QLineEdit(default)
            edit.setMinimumWidth(96)
            edit.setMaximumWidth(140)
            edit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            edit.textChanged.connect(self.on_parameter_changed)
            lay.addWidget(edit, 0, Qt.AlignRight)
            self.param_widgets[key] = edit
            layout.addWidget(row)

        # cc3d connectivity
        conn_row = QWidget()
        conn_lay = QHBoxLayout(conn_row)
        conn_lay.setContentsMargins(0, 0, 0, 0)
        conn_lay.setSpacing(6)
        conn_lay.addWidget(self._param_label("cc3d Connectivity"), 1)
        combo = QComboBox()
        combo.setMinimumWidth(96)
        combo.setMaximumWidth(120)
        combo.addItems(["6 (Face)", "18 (Edge)", "26 (Vertex)"])
        combo.setCurrentIndex(2)
        combo.currentIndexChanged.connect(self.on_parameter_changed)
        conn_lay.addWidget(combo, 0, Qt.AlignRight)
        self.param_widgets["cc3d_connectivity"] = combo
        layout.addWidget(conn_row)

        self._add_single_spin_block(
            layout,
            "cc3d Min Voxels",
            "cc3d_min_voxels",
            default=0,
            is_int=True,
            range_min=0,
            range_max=1000000,
            step=10,
            tooltip=(
                "Minimum voxel count for a connected component to be kept.\n"
                "Components with fewer voxels than this threshold are removed.\n"
                "Set to 0 to disable filtering (keep all components)."
            ),
        )

        cc3d_gpu_check = QCheckBox("Use GPU CC3D (CUDA)")
        cc3d_gpu_check.setChecked(True)
        cc3d_gpu_check.setToolTip(
            "When enabled, connected component labeling runs on the GPU\n"
            "via the DLL's CUDA CC3D kernel (much faster for large volumes).\n"
            "Falls back to Python cc3d if DLL doesn't support it."
        )
        cc3d_gpu_check.stateChanged.connect(self.on_parameter_changed)
        layout.addWidget(cc3d_gpu_check)
        self.param_widgets["cc3d_use_gpu"] = cc3d_gpu_check

        group.setLayout(layout)
        return group

    def create_blank_slices_group(self):
        """Blank slices — Top/Bottom Air grouped by name, Start/End on one row."""
        group = QGroupBox("Blank Slices (Air)")
        layout = self._param_section_layout()

        def _air_pair(name, start_key, end_key, start_default, end_default):
            layout.addWidget(self._param_label(name))
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(10)

            for chip_text, key, default in (
                ("Start", start_key, start_default),
                ("End", end_key, end_default),
            ):
                cell = QWidget()
                cell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                cell_lay = QHBoxLayout(cell)
                cell_lay.setContentsMargins(0, 0, 0, 0)
                cell_lay.setSpacing(4)

                chip = QLabel(chip_text)
                chip.setStyleSheet(
                    f"color: {SemiconductorTheme.TEXT_SECONDARY};"
                    f"font-size: 7.5pt; font-weight: 700;"
                )
                cell_lay.addWidget(chip)
                spin = NoScrollSpinBox()
                spin.setRange(0, 10000)
                spin.setValue(default)
                spin.setMinimumWidth(56)
                spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                spin.valueChanged.connect(self.on_parameter_changed)
                cell_lay.addWidget(spin, 1)
                self.param_widgets[key] = spin
                lay.addWidget(cell, 1)

            layout.addWidget(row)

        _air_pair("Top Air", "blankStart1", "blankEnd1", 0, 1)
        _air_pair("Bottom Air", "blankStart2", "blankEnd2", 499, 501)

        group.setLayout(layout)
        return group

    def create_options_group(self):
        """Debug options group"""
        group = QGroupBox("Debug Options")
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(4)

        bump_debug_check = QCheckBox("Save Bump Intermediate Images")
        bump_debug_check.stateChanged.connect(self.on_parameter_changed)
        layout.addWidget(bump_debug_check)
        self.param_widgets["saveBumpIntermediate"] = bump_debug_check

        void_debug_check = QCheckBox("Save Void Intermediate Images")
        void_debug_check.stateChanged.connect(self.on_parameter_changed)
        layout.addWidget(void_debug_check)
        self.param_widgets["saveVoidIntermediate"] = void_debug_check

        show_result_check = QCheckBox("Show Flattened Result")
        show_result_check.stateChanged.connect(self.on_parameter_changed)
        layout.addWidget(show_result_check)
        self.param_widgets["showResult"] = show_result_check

        group.setLayout(layout)
        return group
        
    def create_enhancement_panel(self):
        """Create the Enhancement panel for ONNX-based volume enhancement."""
        panel = QWidget()
        self._theme_register("_theme_panels", panel)
        panel.setStyleSheet(f"background: {SemiconductorTheme.BG_PANEL}; border-radius: 6px;")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(6)

        # --- Header ---
        header_label = QLabel("<b>VOLUME ENHANCEMENT</b>")
        self._theme_register("_theme_accent_labels", header_label)
        header_label.setStyleSheet(f"font-size: 10pt; color: {SemiconductorTheme.ACCENT_PRIMARY};")
        panel_layout.addWidget(header_label)

        desc = QLabel(
            "Pre-process volume slices with an ONNX denoising model (RestormerNano, etc.)\n"
            "before running segmentation. Enhancement improves bump/void detection accuracy."
        )
        self._theme_register("_theme_secondary_labels", desc)
        desc.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        desc.setWordWrap(True)
        panel_layout.addWidget(desc)

        # --- Enable Checkbox ---
        self.enh_enable_check = QCheckBox("Enable Enhancement (pre-process before segmentation)")
        self._theme_register("_theme_accent_checks", self.enh_enable_check)
        self.enh_enable_check.setStyleSheet(self._accent_check_qss())
        self.enh_enable_check.stateChanged.connect(self._on_enhancement_toggled)
        panel_layout.addWidget(self.enh_enable_check)

        # --- Settings Group ---
        settings_group = QGroupBox("Enhancement Settings")
        self._theme_register("_theme_group_boxes", settings_group)
        settings_group.setStyleSheet(self._group_box_qss())
        settings_layout = QGridLayout(settings_group)
        settings_layout.setContentsMargins(8, 14, 8, 8)
        settings_layout.setSpacing(6)

        row = 0

        # Model Path
        settings_layout.addWidget(QLabel("ONNX Model Path:"), row, 0)
        model_row = QHBoxLayout()
        self.enh_model_input = QLineEdit()
        self.enh_model_input.setPlaceholderText("Path to .onnx model file...")
        self.enh_model_input.setStyleSheet(self._input_qss())
        self._theme_register("_theme_inputs", self.enh_model_input)
        model_row.addWidget(self.enh_model_input, 1)
        enh_browse_btn = QPushButton("Browse...")
        enh_browse_btn.setStyleSheet(self._secondary_button_qss())
        self._theme_register("_theme_secondary_buttons", enh_browse_btn)
        enh_browse_btn.clicked.connect(self._browse_enhancement_model)
        model_row.addWidget(enh_browse_btn)
        settings_layout.addLayout(model_row, row, 1)
        row += 1

        # TRT Cache Path
        settings_layout.addWidget(QLabel("TensorRT Cache:"), row, 0)
        self.enh_trt_input = QLineEdit("./TRT_Cache")
        self.enh_trt_input.setStyleSheet(self._input_qss())
        self._theme_register("_theme_inputs", self.enh_trt_input)
        settings_layout.addWidget(self.enh_trt_input, row, 1)
        row += 1

        # GPU toggle
        gpu_row = QHBoxLayout()
        self.enh_gpu_check = QCheckBox("Use GPU")
        self.enh_gpu_check.setChecked(True)
        self.enh_gpu_check.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY};")
        gpu_row.addWidget(self.enh_gpu_check)

        gpu_row.addWidget(QLabel("GPU ID:"))
        self.enh_gpuid_spin = NoScrollSpinBox()
        self.enh_gpuid_spin.setRange(0, 7)
        self.enh_gpuid_spin.setValue(0)
        self.enh_gpuid_spin.setStyleSheet(self._input_qss())
        self._theme_register("_theme_inputs", self.enh_gpuid_spin)
        gpu_row.addWidget(self.enh_gpuid_spin)
        gpu_row.addStretch()

        settings_layout.addWidget(QLabel("GPU:"), row, 0)
        settings_layout.addLayout(gpu_row, row, 1)
        row += 1

        panel_layout.addWidget(settings_group)

        # --- Mode Group ---
        mode_group = QGroupBox("Enhancement Mode")
        self._theme_register("_theme_group_boxes", mode_group)
        mode_group.setStyleSheet(self._group_box_qss())
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setContentsMargins(8, 14, 8, 8)

        self.enh_mode_full = QRadioButton("Full Volume — enhance all slices")
        self.enh_mode_layers = QRadioButton("Selected Layers Only — enhance per-layer ranges")
        self._theme_register("_theme_radio_buttons", self.enh_mode_full)
        self._theme_register("_theme_radio_buttons", self.enh_mode_layers)
        self.enh_mode_full.setChecked(True)
        self.enh_mode_full.setStyleSheet(self._radio_button_qss())
        self.enh_mode_layers.setStyleSheet(self._radio_button_qss())
        self.enh_mode_full.toggled.connect(self._on_enhancement_mode_changed)
        self.enh_mode_layers.toggled.connect(self._on_enhancement_mode_changed)
        mode_layout.addWidget(self.enh_mode_full)
        mode_layout.addWidget(self.enh_mode_layers)

        self.enh_save_check = QCheckBox("Save enhanced images to output folder")
        self.enh_save_check.setChecked(True)
        self.enh_save_check.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY};")
        self.enh_save_check.stateChanged.connect(self._on_enhancement_save_toggled)
        mode_layout.addWidget(self.enh_save_check)

        panel_layout.addWidget(mode_group)

        # --- Status ---
        self.enh_status_label = QLabel("Enhancement: Disabled")
        self._theme_register("_theme_status_labels", self.enh_status_label)
        self.enh_status_label.setStyleSheet(self._status_label_qss())
        panel_layout.addWidget(self.enh_status_label)

        # --- Run Enhancement Only button ---
        self.enh_run_btn = QPushButton("Run Enhancement Only")
        self.enh_run_btn.setStyleSheet(self._primary_button_qss())
        self._theme_register("_theme_primary_buttons", self.enh_run_btn)
        self.enh_run_btn.setEnabled(False)
        self.enh_run_btn.clicked.connect(self._run_enhancement_only)
        panel_layout.addWidget(self.enh_run_btn)

        panel_layout.addStretch()
        return panel

    def _on_enhancement_toggled(self, state):
        """Handle enhancement checkbox toggle."""
        self.enhancement_enabled = bool(state)
        enabled_text = "Enabled" if self.enhancement_enabled else "Disabled"
        color = SemiconductorTheme.ACCENT_SUCCESS if self.enhancement_enabled else SemiconductorTheme.TEXT_SECONDARY
        self.enh_status_label.setText(f"Enhancement: {enabled_text}")
        self.enh_status_label.setStyleSheet(self._status_label_qss(color))
        self.enh_run_btn.setEnabled(self.enhancement_enabled)

    def _on_enhancement_mode_changed(self, _checked=False):
        """Sync radio buttons → enhancement_mode state."""
        if hasattr(self, "enh_mode_layers") and self.enh_mode_layers.isChecked():
            self.enhancement_mode = "layers"
        else:
            self.enhancement_mode = "full"

    def _on_enhancement_save_toggled(self, state):
        self.enhancement_save_output = bool(state)

    def _sync_enhancement_state_from_ui(self):
        """Pull latest values from Enhance-panel widgets into state fields."""
        if hasattr(self, "enh_enable_check"):
            self.enhancement_enabled = self.enh_enable_check.isChecked()
        if hasattr(self, "enh_model_input"):
            self.enhancement_model_path = self.enh_model_input.text().strip()
        if hasattr(self, "enh_trt_input"):
            self.enhancement_trt_cache = self.enh_trt_input.text().strip() or "./TRT_Cache"
        if hasattr(self, "enh_gpu_check"):
            self.enhancement_use_gpu = self.enh_gpu_check.isChecked()
        if hasattr(self, "enh_gpuid_spin"):
            self.enhancement_gpu_id = int(self.enh_gpuid_spin.value())
        if hasattr(self, "enh_mode_layers") and self.enh_mode_layers.isChecked():
            self.enhancement_mode = "layers"
        else:
            self.enhancement_mode = "full"
        if hasattr(self, "enh_save_check"):
            self.enhancement_save_output = self.enh_save_check.isChecked()

    def get_enhancement_slice_range(self):
        """Return (start_slice, end_slice) for DLL process.

        - full mode  → (-1, -1) = all slices
        - layers mode → [min(z_start), max(z_end)) over selected layers
          (end is exclusive, matching EnhancedVolume_ProcessSliceRange)
        """
        self._sync_enhancement_state_from_ui()
        if self.enhancement_mode != "layers":
            return -1, -1
        layers = getattr(self, "layer_definitions", None) or []
        selected = [l for l in layers if l.get("selected")]
        if not selected:
            return -1, -1
        z_starts = [int(l["z_start"]) for l in selected]
        z_ends = [int(l["z_end"]) for l in selected]
        # Layer z_end in config is treated as exclusive upper bound (same as prior pipeline)
        return min(z_starts), max(z_ends)

    def _apply_enhancement_from_config_text(self, content):
        """Parse ENHANCEMENT_* / ENHANCED_* keys from config text into UI + state.

        ENHANCEMENT_ENABLED controls the Enhance-tab checkbox.
        ENHANCEMENT_MODE = full | layers  (aliases: full_volume, selected_layers)
        ENHANCEMENT_SAVE_OUTPUT = true | false
        Backward compat: if ENABLED is missing, enable when ENHANCED_MODEL_PATH is non-empty.
        """
        enh_enable_match = re.search(
            r'ENHANCEMENT_ENABLED\s*=\s*(true|false)', content, re.IGNORECASE
        )
        enh_model_match = re.search(r'ENHANCED_MODEL_PATH\s*=\s*(.*)', content)
        enh_trt_match = re.search(r'ENHANCED_TRT_CACHE_PATH\s*=\s*(.*)', content)
        enh_gpu_match = re.search(
            r'ENHANCED_USE_GPU\s*=\s*(true|false)', content, re.IGNORECASE
        )
        enh_gpuid_match = re.search(r'ENHANCED_GPU_DEVICE_ID\s*=\s*(\d+)', content)
        enh_mode_match = re.search(
            r'ENHANCEMENT_MODE\s*=\s*([A-Za-z_]+)', content, re.IGNORECASE
        )
        enh_save_match = re.search(
            r'ENHANCEMENT_SAVE_OUTPUT\s*=\s*(true|false)', content, re.IGNORECASE
        )

        model_val = ""
        if enh_model_match:
            model_val = enh_model_match.group(1).strip().strip('"').strip("'")
            self.enhancement_model_path = model_val
            if hasattr(self, "enh_model_input"):
                self.enh_model_input.setText(model_val)

        if enh_trt_match:
            trt_val = enh_trt_match.group(1).strip().strip('"').strip("'")
            self.enhancement_trt_cache = trt_val
            if hasattr(self, "enh_trt_input"):
                self.enh_trt_input.setText(trt_val)

        if enh_gpu_match:
            self.enhancement_use_gpu = enh_gpu_match.group(1).lower() == "true"
            if hasattr(self, "enh_gpu_check"):
                self.enh_gpu_check.setChecked(self.enhancement_use_gpu)

        if enh_gpuid_match:
            self.enhancement_gpu_id = int(enh_gpuid_match.group(1))
            if hasattr(self, "enh_gpuid_spin"):
                self.enh_gpuid_spin.setValue(self.enhancement_gpu_id)

        # Enhancement mode: full | layers
        if enh_mode_match:
            raw = enh_mode_match.group(1).strip().lower()
            if raw in ("layers", "selected_layers", "selected_layer", "layer", "selected"):
                self.enhancement_mode = "layers"
            else:
                self.enhancement_mode = "full"
        # else keep current default

        if hasattr(self, "enh_mode_full") and hasattr(self, "enh_mode_layers"):
            is_layers = self.enhancement_mode == "layers"
            self.enh_mode_full.blockSignals(True)
            self.enh_mode_layers.blockSignals(True)
            self.enh_mode_full.setChecked(not is_layers)
            self.enh_mode_layers.setChecked(is_layers)
            self.enh_mode_full.blockSignals(False)
            self.enh_mode_layers.blockSignals(False)

        if enh_save_match:
            self.enhancement_save_output = enh_save_match.group(1).lower() == "true"
            if hasattr(self, "enh_save_check"):
                self.enh_save_check.blockSignals(True)
                self.enh_save_check.setChecked(self.enhancement_save_output)
                self.enh_save_check.blockSignals(False)

        # Explicit flag wins; otherwise legacy: non-empty model path ⇒ enable
        if enh_enable_match:
            enabled = enh_enable_match.group(1).lower() == "true"
        else:
            enabled = bool(model_val)

        self.enhancement_enabled = enabled
        if hasattr(self, "enh_enable_check"):
            self.enh_enable_check.blockSignals(True)
            self.enh_enable_check.setChecked(enabled)
            self.enh_enable_check.blockSignals(False)
            self._on_enhancement_toggled(enabled)

    def _browse_enhancement_model(self):
        """Browse for ONNX model file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX Model", "", "ONNX Models (*.onnx);;All Files (*)"
        )
        if path:
            self.enh_model_input.setText(path)
            self.enhancement_model_path = path

    def _prepare_enhancement_input_dir(self):
        """Build a folder of **per-slice** TIFFs for BumpVoid_ISP_ENH.

        The enhance DLL treats each file as one 2D image. A multipage TIFF in a
        parent folder is only **one file** → only the first page is enhanced
        (classic “opened result has 1 slice” bug).

        Prefer in-memory ``volume_data`` (full Z). Fall back to extracting a
        multipage file, or use a directory of already-split slices as-is.

        Returns (input_dir, temp_dir_or_None). Caller must clean temp_dir.
        """
        import tempfile
        import tifffile

        input_path = (self.input_path_input.text() or "").strip()
        vol = getattr(self, "volume_data", None)

        # 1) Prefer loaded 3D volume in memory (most reliable Z count)
        if vol is not None and getattr(vol, "ndim", 0) == 3 and vol.shape[0] > 0:
            temp_dir = tempfile.mkdtemp(prefix="inno3d_enh_input_")
            n = int(vol.shape[0])
            print(f"[ENHANCE] Exporting {n} slices from volume_data → {temp_dir}")
            for idx in range(n):
                tifffile.imwrite(
                    os.path.join(temp_dir, f"{idx:04d}.tif"),
                    np.ascontiguousarray(vol[idx]),
                )
            return temp_dir, temp_dir

        # 2) Folder of slice TIFFs (already 2D-per-file)
        if input_path and os.path.isdir(input_path):
            print(f"[ENHANCE] Using slice folder as-is: {input_path}")
            return input_path, None

        # 3) Multipage / single TIFF file → explode pages to temp
        if input_path and os.path.isfile(input_path):
            temp_dir = tempfile.mkdtemp(prefix="inno3d_enh_input_")
            try:
                stack = tifffile.imread(input_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read volume for enhancement:\n{input_path}\n{e}"
                ) from e
            stack = np.asarray(stack)
            if stack.ndim == 2:
                # Single plane file
                tifffile.imwrite(
                    os.path.join(temp_dir, "0000.tif"),
                    np.ascontiguousarray(stack),
                )
                n = 1
            elif stack.ndim >= 3:
                # (Z,Y,X) or (Z,Y,X,C)
                n = int(stack.shape[0])
                for idx in range(n):
                    plane = stack[idx]
                    if plane.ndim > 2:
                        plane = plane[..., 0]
                    tifffile.imwrite(
                        os.path.join(temp_dir, f"{idx:04d}.tif"),
                        np.ascontiguousarray(plane),
                    )
            else:
                raise RuntimeError(f"Unsupported volume ndim={stack.ndim} for enhance")
            print(
                f"[ENHANCE] Split multipage file into {n} slices → {temp_dir} "
                f"(NOT parent folder — avoids 1-file/1-slice bug)"
            )
            return temp_dir, temp_dir

        raise RuntimeError(
            "No volume loaded. Load a 3D volume (or multipage TIFF) before Run Enhancement."
        )

    def _run_enhancement_only(self):
        """Run enhancement as a standalone step (without segmentation)."""
        model_path = self.enh_model_input.text().strip()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Warning", "Please select a valid ONNX model file.")
            return

        if not self.dll_path:
            QMessageBox.warning(self, "Warning", "Please load the DLL folder first.")
            return

        input_path = self.input_path_input.text().strip()
        if not input_path and getattr(self, "volume_data", None) is None:
            QMessageBox.warning(self, "Warning", "Please load a volume first.")
            return

        output_path = self.output_path_input.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Warning", "Please specify output path.")
            return

        # Per-slice folder for DLL (must NOT be multipage-parent dirname)
        try:
            input_dir, temp_dir = self._prepare_enhancement_input_dir()
        except Exception as e:
            QMessageBox.critical(self, "Enhancement", str(e))
            return
        self._enhancement_temp_dir = temp_dir

        enhanced_output = os.path.join(output_path, "enhanced_volume")
        os.makedirs(enhanced_output, exist_ok=True)

        progress = QProgressDialog("Running Volume Enhancement...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        start_slice, end_slice = self.get_enhancement_slice_range()
        mode_txt = "selected layers" if start_slice >= 0 else "full volume"
        progress.setLabelText(f"Enhancing ({mode_txt})...")

        self.enhancement_thread = EnhancementThread(
            dll_dir=self.dll_path,
            model_path=model_path,
            trt_cache_path=self.enh_trt_input.text().strip(),
            use_gpu=self.enh_gpu_check.isChecked(),
            gpu_device_id=self.enh_gpuid_spin.value(),
            input_dir=input_dir,
            output_dir=enhanced_output,
            start_slice=start_slice,
            end_slice=end_slice,
        )
        self.enhancement_thread.progress.connect(
            lambda v, m: (progress.setValue(v), progress.setLabelText(m))
        )
        self.enhancement_thread.finished.connect(
            lambda ok, err: self._on_enhancement_finished(
                ok, err, progress, enhanced_output=enhanced_output
            )
        )
        self.enhancement_thread.start()

    def _cleanup_enhancement_temp_dir(self):
        temp = getattr(self, "_enhancement_temp_dir", None)
        if not temp:
            return
        try:
            import shutil
            shutil.rmtree(temp, ignore_errors=True)
            print(f"[ENHANCE] Cleaned temp input dir: {temp}")
        except Exception as e:
            print(f"[ENHANCE] Temp cleanup failed: {e}")
        self._enhancement_temp_dir = None

    def _on_enhancement_finished(self, success, error, progress, enhanced_output=None):
        """Handle enhancement completion."""
        if progress:
            progress.close()
        self._cleanup_enhancement_temp_dir()

        if success:
            n_out = 0
            if enhanced_output and os.path.isdir(enhanced_output):
                try:
                    n_out = len(
                        [
                            f
                            for f in os.listdir(enhanced_output)
                            if f.lower().endswith((".tif", ".tiff"))
                        ]
                    )
                except Exception:
                    n_out = 0
            self.enh_status_label.setText(
                f"Enhancement: Complete ✓ ({n_out} slices)"
                if n_out
                else "Enhancement: Complete ✓"
            )
            self.enh_status_label.setStyleSheet(
                self._status_label_qss(SemiconductorTheme.ACCENT_SUCCESS)
            )
            msg = (
                "Volume enhancement finished successfully.\n"
                f"Enhanced images saved to:\n{enhanced_output or 'enhanced_volume/'}"
            )
            if n_out:
                msg += f"\n\n{n_out} slice file(s) written (one TIFF per Z)."
            QMessageBox.information(self, "Enhancement Complete", msg)
        else:
            self.enh_status_label.setText("Enhancement: Failed ✗")
            self.enh_status_label.setStyleSheet(
                self._status_label_qss(SemiconductorTheme.ACCENT_ERROR)
            )
            QMessageBox.critical(self, "Enhancement Failed", f"Enhancement error:\n{error}")

    def _teaching_header_qss(self):
        return (
            f"QWidget#TeachingHeader {{"
            f"  background: {SemiconductorTheme.BG_DARK};"
            f"  border-bottom: 1px solid {SemiconductorTheme.BORDER_DEFAULT};"
            f"}}"
        )

    def _teaching_tab_btn_qss(self):
        """Equal-size 2-row teaching tab buttons."""
        return f"""
            QToolButton#TeachingTabBtn {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_SECONDARY};
                padding: 4px 4px;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px;
                font-size: 8pt;
                font-weight: 600;
                text-align: center;
            }}
            QToolButton#TeachingTabBtn:hover:!checked {{
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border-color: {SemiconductorTheme.BORDER_HOVER};
                background: {SemiconductorTheme.EL4};
            }}
            QToolButton#TeachingTabBtn:checked {{
                background: {SemiconductorTheme.BG_PANEL};
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                border-color: {SemiconductorTheme.ACCENT_PRIMARY};
                font-weight: 700;
            }}
            QToolButton#TeachingTabBtn:pressed {{
                background: {SemiconductorTheme.EL4};
            }}
        """

    def _pipeline_run_label_qss(self):
        return (
            f"color: {SemiconductorTheme.TEXT_SECONDARY};"
            f"font-size: 7pt; font-weight: 700; letter-spacing: 0.6px;"
            f"padding: 0 0 2px 2px; background: transparent;"
        )

    def _pipeline_accent_color(self, role):
        """Resolve pipeline button accent from a stable role name."""
        return {
            "success": SemiconductorTheme.ACCENT_SUCCESS,
            "primary": SemiconductorTheme.ACCENT_PRIMARY,
            "warning": SemiconductorTheme.ACCENT_WARNING,
        }.get(role, SemiconductorTheme.ACCENT_PRIMARY)

    def _pipeline_btn_qss(self, role):
        """Equal-width pipeline action button style (role: success|primary|warning)."""
        accent_color = self._pipeline_accent_color(role)
        text_color = SemiconductorTheme.TEXT_ON_ACCENT
        return f"""
            QPushButton {{
                background-color: {accent_color};
                color: {text_color};
                font-weight: 700;
                padding: 5px 4px;
                border-radius: 4px;
                font-size: 8pt;
                border: 1px solid {accent_color};
                min-height: 24px;
            }}
            QPushButton:hover {{
                border-color: {SemiconductorTheme.TEXT_PRIMARY};
            }}
            QPushButton:disabled {{
                background-color: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_DISABLED};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """

    def create_action_bar(self):
        """Pipeline strip under the teaching tabs — equal-width step buttons."""
        bar = QWidget()
        bar.setObjectName("TeachingActionBar")
        bar.setStyleSheet(
            f"QWidget#TeachingActionBar {{"
            f"  background: {SemiconductorTheme.BG_DARK};"
            f"  border: none;"
            f"}}"
        )

        root = QVBoxLayout(bar)
        root.setContentsMargins(6, 4, 6, 6)
        root.setSpacing(4)

        self._pipeline_run_label = QLabel("PIPELINE")
        self._pipeline_run_label.setObjectName("PipelineRunLabel")
        self._pipeline_run_label.setStyleSheet(self._pipeline_run_label_qss())
        root.addWidget(self._pipeline_run_label)

        # Equal-width button row
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(4)

        self._pipeline_buttons = []  # list of (btn, role) for theme refresh

        def _make_pipeline_btn(text, tooltip, role, on_click):
            btn = QPushButton(text)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(tooltip)
            btn.setStyleSheet(self._pipeline_btn_qss(role))
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumWidth(0)
            self._pipeline_buttons.append((btn, role))
            btn.clicked.connect(on_click)
            btn.setEnabled(False)
            return btn

        # Ordered pipeline: SEG → MES → FAR → B2B → 3DT
        self.inspect_btn = _make_pipeline_btn(
            "SEG",
            "1. Run Segmentation (Bump + Void)",
            "success",
            self.run_inspection,
        )
        self.measure_btn = _make_pipeline_btn(
            "MES",
            "2. Run Measurement / Object Stats",
            "success",
            self.run_measurement,
        )
        self.far_btn = _make_pipeline_btn(
            "FAR",
            "3. False Alarm Remover",
            "primary",
            self.run_false_alarm_remover,
        )
        self.bnd_btn = _make_pipeline_btn(
            "B2B",
            "4. Bump-to-Bump Boundary Analysis (requires Measurement)",
            "warning",
            self.run_boundary_analysis,
        )
        self.dt_btn = _make_pipeline_btn(
            "3DT",
            "5. 3D Distance Transform",
            "primary",
            self.run_distance_transform,
        )

        for btn in (
            self.inspect_btn,
            self.measure_btn,
            self.far_btn,
            self.bnd_btn,
            self.dt_btn,
        ):
            btn_row.addWidget(btn, 1)

        root.addLayout(btn_row)

        # Layer filter (shown only when multi-layer data is available)
        self.layer_filter_combo = QComboBox()
        self._theme_register("_theme_inputs", self.layer_filter_combo)
        self.layer_filter_combo.addItem("All Layers", userData=None)
        self.layer_filter_combo.setStyleSheet(self._input_qss())
        self.layer_filter_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.layer_filter_combo.setVisible(False)
        self.layer_filter_combo.currentIndexChanged.connect(self._on_layer_filter_changed)
        root.addWidget(self.layer_filter_combo)

        return bar
        
    def _populate_layer_filter(self):
        self.layer_filter_combo.blockSignals(True)
        self.layer_filter_combo.clear()
        
        # Test Volume (1 Layer) mode: no filter needed
        if getattr(self, 'input_mode', 'test_1layer') == 'test_1layer':
            self.layer_filter_combo.addItem("Volume (1 Layer)", userData=None)
            self.layer_filter_combo.setVisible(False)
        elif getattr(self, 'input_mode', 'test_1layer') == 'multi_layer':
            import re
            ml_files = self._get_ordered_multi_layer_files()
            if ml_files:
                for mf in ml_files:
                    short_name = mf['name']
                    match = re.search(r'Layer\s*(\d+)', short_name, re.IGNORECASE)
                    if match:
                        short_name = f"Layer {match.group(1)}"
                    self.layer_filter_combo.addItem(short_name, userData=mf['name'])
                self.layer_filter_combo.setVisible(True)
            else:
                self.layer_filter_combo.setVisible(False)
        else:
            # Single Volume (Multi-Layer) mode
            self.layer_filter_combo.addItem("All Layers", userData=None)
            if self.layers_active:
                for l in self.layer_definitions:
                    if l['selected']:
                        self.layer_filter_combo.addItem(l['name'], userData=l['name'])
                self.layer_filter_combo.setVisible(True)
            else:
                self.layer_filter_combo.setVisible(False)
            
        if self.layer_filter_combo.count() > 0:
            self.layer_filter_combo.setCurrentIndex(0)
            self.current_display_layer = self.layer_filter_combo.itemData(0)
        else:
            self.current_display_layer = None
            
        self.layer_filter_combo.blockSignals(False)

    def _on_layer_filter_changed(self, index):
        layer_name = self.layer_filter_combo.currentData()
        if layer_name:
            self._switch_to_ml_layer(layer_name)
        else:
            self.current_display_layer = None
            for orientation in ['axial', 'coronal', 'sagittal']:
                self.update_plane_view(orientation, preserve_camera=True)
        
    # ==================== CORE FUNCTIONALITY ====================
    
    def clear_masks(self):
        """Clear all segmentation results"""
        self.bump_segmentation = None
        self.void_segmentation = None
        if hasattr(self, 'labeled_class1_data'):
            self.labeled_class1_data = None
        if hasattr(self, 'object_stats'):
            self.object_stats = []
        if hasattr(self, 'selected_highlight_objects'):
            self.selected_highlight_objects = []
        if hasattr(self, '_refresh_stats_table'):
            self._refresh_stats_table()

        if hasattr(self, 'measure_btn'):
            self.measure_btn.setEnabled(False)
        if hasattr(self, 'far_btn'):
            self.far_btn.setEnabled(False)
        if hasattr(self, 'dt_btn'):
            self.dt_btn.setEnabled(False)
        if hasattr(self, 'bnd_btn'):
            self.bnd_btn.setEnabled(False)
        
        # Clear 3D views
        for orientation in ['axial', 'coronal', 'sagittal']:
            renderer = getattr(self, f'{orientation}_renderer')
            # Remove all actors but keep the slice image
            # Since we don't track them explicitly in a dict here for 2D, we might need to rely on update_plane_view logic
            pass
            
        self.info_label.setText("Masks cleared")
        if self.volume_data is not None:
             # Refresh views to clear overlays
             for orientation in ['axial', 'coronal', 'sagittal']:
                self.update_plane_view(orientation)
    
