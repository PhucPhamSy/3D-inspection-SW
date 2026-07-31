"""
Segmentation Tab - Real-time Bump/Void Segmentation with Parameter Adjustment
"""
import os
import sys
import numpy as np
from pathlib import Path
from vtk.util import numpy_support
from scipy import ndimage
from skimage import io, measure
import glob
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import re
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from inno3d.core import bumpvoid
from inno3d.core import enhanced_volume
from inno3d.core.roi_manager import ROIManager, ExportCropThread
from inno3d.core.view_support import (
    Dragonfly3DInteractorStyle,
    SliceBarWheelFilter,
    StepOneSliceSlider,
    apply_dragonfly_volume_zoom,
    push_camera_outside_aabb,
    volume_world_aabb,
)
from inno3d.features.teaching.workers import (
    NoScrollDoubleSpinBox, NoScrollSpinBox,
    _ExportLayerVolumesThread, DistanceTransformThread, BoundaryAnalysisThread,
    SegmentationPreviewThread, SegmentationInspectionThread, EnhancementThread,
)
from inno3d.features.teaching.seg_ui import SegmentationUIMixin
from inno3d.features.teaching.seg_pipeline import SegmentationPipelineMixin
from inno3d.features.teaching.seg_mpr import SegmentationMPRMixin
from inno3d.features.teaching.layer_define import LayerDefineMixin

class SegmentationTab(LayerDefineMixin, SegmentationPipelineMixin, SegmentationMPRMixin, SegmentationUIMixin, QWidget):
    """Tab for interactive Bump/Void segmentation with real-time parameter adjustment"""
    # Signal emitted when inspection finishes: (volume_data, bump_data, void_data)
    inspection_done = pyqtSignal(object, object, object)
    
    # Signal to apply current ROI + Config to Online mode in MainWindow
    apply_online_roi_signal = pyqtSignal(dict, object, str)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        
        # Core data
        self.dll_path = None
        self.config_path = None
        self.config = None
        self.volume_data = None
        self.bump_segmentation = None
        self.void_segmentation = None
        
        # Layer handling
        self.layer_definitions = []  # List of dicts: {'id': int, 'name': str, 'z_start': int, 'z_end': int, 'selected': bool}
        self.layers_active = False
        self.input_mode = 'test_1layer'    # 'test_1layer' | 'single' | 'multi_layer'
        self.multi_layer_files = []        # [{'name': str, 'path': str}, ...]
        
        # Track last applied ROI for Online Mode fallback
        self.last_applied_roi_raw = None
        
        # Viewer state
        self.current_slices = {'axial': 0, 'coronal': 0, 'sagittal': 0}
        self.spacing = [1.0, 1.0, 1.0]
        self.window_level = {'axial': None, 'coronal': None, 'sagittal': None}
        
        # Enhancement state
        self.enhancement_enabled = False
        self.enhancement_model_path = ""
        self.enhancement_trt_cache = "./TRT_Cache"
        self.enhancement_use_gpu = True
        self.enhancement_gpu_id = 0
        # "full" = all slices; "layers" = selected layer z-ranges only
        self.enhancement_mode = "full"
        self.enhancement_save_output = True
        self.enhancement_thread = None

        # Threading
        self.preview_thread = None
        self.inspection_thread = None
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.run_preview_segmentation)
        
        # Debounce timer for ROI Z changes (avoid flooding the undo stack)
        self.roi_z_timer = QTimer()
        self.roi_z_timer.setSingleShot(True)
        self.roi_z_timer.timeout.connect(self._commit_roi_z_change)
        
        # Parameter widgets storage
        self.param_widgets = {}
        self._pending_input_path = None  # Store input path if loaded before config
        
        # Crosshair state (Dragonfly-style — same as 3D Viewer)
        self.crosshair_enabled = False
        self.crosshair_position = [0, 0, 0]  # [x, y, z]
        self.crosshair_color = [1.0, 1.0, 0.0]  # Yellow
        # Hover / grab: (orientation, 'center'|'h'|'v') while over/dragging axes
        self._crosshair_hover = None
        self._crosshair_drag_mode = None
        self._is_dragging_crosshair = False
        self._persistent_crosshair = {'axial': {}, 'coronal': {}, 'sagittal': {}}
        # Same default as 3D Viewer scale-bar ruler
        self.ruler_color = [0.2, 0.9, 0.2]
        self.ruler_actors = {'axial': [], 'coronal': [], 'sagittal': []}
        self.reverse_z = False

        self.class1_color = [1.0, 1.0, 0.0]  # Yellow
        self.class2_color = [1.0, 0.0, 0.0]  # Red
        self.camera_states = {}  # Store per-orientation camera zoom/pan
        
        # Object measurement & highlight
        self.labeled_class1_data = None  # labeled array from skimage.measure.label
        self.labeled_class2_data = None
        self.object_stats = []  # list of dicts with measurement data
        self.selected_highlight_objects = []  # list of (class_num, obj_id) tuples
        self.highlight_color = [0.0, 1.0, 1.0]  # cyan highlight
        self.current_display_layer = None # None means All Layers
        
        # ROI Manager
        self.roi_manager = ROIManager()
        self.roi_selection_active = False
        self.roi_drawing = False  # True while mouse is held down during drawing
        self.roi_resizing = False # True while mouse is held down during resizing
        self.roi_resize_handle = None # 'n', 's', 'e', 'w', 'nw', 'ne', 'sw', 'se'
        self.roi_resize_vars = []  # List of keys from ROI dict being updated
        self.roi_start_point = None  # (x, y) world coords of initial click
        self.roi_temp_end = None    # (x, y) world coords during drag
        self.export_thread = None

        # Layer Define Mode (interactive bar placement on sagittal view)
        self.layer_define_active = False
        self._layer_define_dragging = False
        self._layer_define_drag_idx = None   # (layer_index, 'start'|'end')
        self._layer_define_hover_idx = None  # currently hovered bar

        self._theme_panels = []
        self._theme_accent_labels = []
        self._theme_secondary_labels = []
        self._theme_value_labels = []
        self._theme_secondary_buttons = []
        self._theme_primary_buttons = []
        self._theme_success_buttons = []
        self._theme_danger_buttons = []
        self._theme_warning_buttons = []
        self._theme_warning_labels = []
        self._theme_status_labels = []
        self._theme_tables = []
        self._theme_lists = []
        self._theme_inputs = []
        self._theme_group_boxes = []
        self._theme_radio_buttons = []
        self._theme_accent_checks = []
        
        self.init_ui()

