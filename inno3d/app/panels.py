# inno3d/app/panels.py
# -----------------------------------------------------------------------
# PanelsMixin — sidebar and viewer panel builders for MainWindow
#
# Extracted from app/main_window.py (Phase 6 continuation) — MOVE not COPY.
# Contains: _create_viewer_panel, _create_segmentation_sidebar_panel,
#            _create_batch_review_sidebar_panel, _create_analysis_sidebar_panel,
#            _update_sidebar_bullets
#
# All names used below must be imported in THIS module (define-module rule).
# -----------------------------------------------------------------------
"""PanelsMixin — sidebar + viewer panel builder methods for MainWindow."""

from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap, QPainter
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QLabel, QPushButton, QToolButton, QCheckBox, QComboBox,
    QScrollArea, QFrame, QSizePolicy, QTabWidget, QStackedWidget,
    QSpinBox, QDoubleSpinBox, QSlider, QGroupBox, QButtonGroup,
    QLineEdit, QProgressBar, QMessageBox, QApplication, QStyle,
)

from inno3d.core.styles import SemiconductorTheme, apply_theme, normalize_theme_name
from inno3d.core.ui_system import AnimatedNavButton, GlowProgressBar
from inno3d.core.resources import resource_path
from inno3d.tabs.viewer import MultiPlanarView
from inno3d.tabs.teaching import SegmentationTab
from inno3d.tabs.analysis import AnalysisTab
from inno3d.tabs.batch_review import BatchReviewTab
from inno3d.tabs.ai import AI3DTab
from inno3d.tabs.help import HelpTab


class PanelsMixin:
    """Mixin providing sidebar and viewer panel builder methods for MainWindow."""

    def _create_viewer_panel(self):
        """Panel for 3D Viewer tab (index 0)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 0, 15, 10)
        layout.setSpacing(8) # Tighter spacing -> Increased to 8px
        
        def add_section_title(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(SemiconductorTheme.sidebar_section_style())
            layout.addWidget(lbl)

        # ═══════════════════════════════════════════════════════════
        # 1. INPUT DATA  —  Full-width load buttons with icon + text
        # ═══════════════════════════════════════════════════════════
        add_section_title("INPUT DATA")
        
        b_open = QPushButton("  OPEN")
        b_open.setProperty("class", "system-text-btn")
        b_open.setProperty("theme_icon", True)
        b_open.setProperty("icon_name", "folder-open.svg")
        b_open.setProperty("standard_icon", int(QStyle.SP_DirOpenIcon))
        b_open.setProperty("icon_size_w", 16)
        b_open.setProperty("icon_size_h", 16)
        b_open.setProperty("is_icon_only", False)
        self._apply_themed_icon(b_open)
        b_open.setCursor(Qt.PointingHandCursor)
        b_open.setToolTip("Open a 3D file or a folder of stack images")
        b_open.clicked.connect(lambda: self.multiplanar_tab.load_volume())
        layout.addWidget(b_open)
        self.viewer_btn_open = b_open

        # Online still disables OPEN (see _set_online_input_locked); no banner text.
        self.online_input_lock_hint = None

        # ═══════════════════════════════════════════════════════════
        # 2. SEG MASKS  —  Bordered buttons with inline icons
        # ═══════════════════════════════════════════════════════════
        add_section_title("SEG MASKS")
        
        # SINGLE LAYER C1 + color picker
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(6)
        b_c1 = QPushButton("  SINGLE LAYER C1")
        b_c1.setProperty("class", "system-text-btn")
        b_c1.setProperty("theme_icon", True)
        b_c1.setProperty("icon_name", "mask.svg")
        b_c1.setProperty("standard_icon", "")
        b_c1.setProperty("icon_size_w", 16)
        b_c1.setProperty("icon_size_h", 16)
        self._apply_themed_icon(b_c1)
        b_c1.setCursor(Qt.PointingHandCursor)
        b_c1.clicked.connect(lambda: self.multiplanar_tab.load_class_mask(1))
        b_c1_clr = self._create_system_icon_button(
            "Choose Class 1 overlay color",
            lambda: self.multiplanar_tab.choose_overlay_color(1),
            icon_name="color-picker.svg",
            standard_icon=QStyle.SP_DriveDVDIcon,
        )
        row1.addWidget(b_c1, 1)
        row1.addWidget(b_c1_clr)
        layout.addLayout(row1)
        
        # SINGLE LAYER C2 + color picker
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(6)
        b_c2 = QPushButton("  SINGLE LAYER C2")
        b_c2.setProperty("class", "system-text-btn")
        b_c2.setProperty("theme_icon", True)
        b_c2.setProperty("icon_name", "mask.svg")
        b_c2.setProperty("standard_icon", "")
        b_c2.setProperty("icon_size_w", 16)
        b_c2.setProperty("icon_size_h", 16)
        self._apply_themed_icon(b_c2)
        b_c2.setCursor(Qt.PointingHandCursor)
        b_c2.clicked.connect(lambda: self.multiplanar_tab.load_class_mask(2))
        b_c2_clr = self._create_system_icon_button(
            "Choose Class 2 overlay color",
            lambda: self.multiplanar_tab.choose_overlay_color(2),
            icon_name="color-picker.svg",
            standard_icon=QStyle.SP_DriveDVDIcon,
        )
        row2.addWidget(b_c2, 1)
        row2.addWidget(b_c2_clr)
        layout.addLayout(row2)
        
        # ALL LAYERS (full width)
        b_mask_folder = QPushButton("  ALL LAYERS")
        b_mask_folder.setProperty("class", "system-text-btn")
        b_mask_folder.setProperty("theme_icon", True)
        b_mask_folder.setProperty("icon_name", "layers.svg")
        b_mask_folder.setProperty("standard_icon", "")
        b_mask_folder.setProperty("icon_size_w", 16)
        b_mask_folder.setProperty("icon_size_h", 16)
        self._apply_themed_icon(b_mask_folder)
        b_mask_folder.setCursor(Qt.PointingHandCursor)
        b_mask_folder.setToolTip("Load all per-layer masks (bump + void) from a sample output folder")
        b_mask_folder.clicked.connect(self.multiplanar_tab.load_mask_folder)
        layout.addWidget(b_mask_folder)

        # CLEAR ALL (under SEG MASKS)
        b_clear = QPushButton("  CLEAR ALL")
        b_clear.setProperty("class", "clear-btn")
        b_clear.setProperty("theme_icon", True)
        b_clear.setProperty("icon_name", "eraser.svg")
        b_clear.setProperty("standard_icon", "")
        b_clear.setProperty("icon_size_w", 14)
        b_clear.setProperty("icon_size_h", 14)
        b_clear.setProperty("role", "danger")
        self._apply_themed_icon(b_clear)
        b_clear.setCursor(Qt.PointingHandCursor)
        b_clear.clicked.connect(self.multiplanar_tab.clear_masks)
        layout.addWidget(b_clear)
        
        # ═══════════════════════════════════════════════════════════
        # 3. SYSTEM  —  Carl Zeiss Industrial Design Language
        #    Layout:
        #    Row 0: [LAYOUT toggle — show/hide grid layout strip]
        #    Row 1: [CROSSHAIR toggle — full width        ]
        #    Row 2: [⟳ Reset CH] [🎯 CH Color] [📏 Ruler] [⟳ Reset Rot]
        #    (2D ROTATE lives on the right OBLIQUE MPR sidebar)
        # ═══════════════════════════════════════════════════════════
        add_section_title("SYSTEM")

        # ── Toggle: LAYOUT (above crosshair) ───────────────────
        # Shows/hides the left Dragonfly-style view-grid icon strip.
        self.layout_grid_btn = QPushButton("  LAYOUT")
        self.layout_grid_btn.setProperty("class", "system-toggle-btn")
        self.layout_grid_btn.setProperty("theme_icon", True)
        self.layout_grid_btn.setProperty("icon_name", "grid.svg")
        self.layout_grid_btn.setProperty("standard_icon", "")
        self.layout_grid_btn.setProperty("icon_size_w", 16)
        self.layout_grid_btn.setProperty("icon_size_h", 16)
        self._apply_themed_icon(self.layout_grid_btn)
        self.layout_grid_btn.setCheckable(True)
        self.layout_grid_btn.setChecked(False)
        self.layout_grid_btn.setCursor(Qt.PointingHandCursor)
        self.layout_grid_btn.setToolTip(
            "Show / hide view grid layout picker (Dragonfly-style).\n"
            "Cube glyph on icons = 3D volume pane; plain cells = MPR."
        )
        self.layout_grid_btn.clicked.connect(
            lambda checked: self.multiplanar_tab.toggle_layout_sidebar(checked)
        )
        layout.addWidget(self.layout_grid_btn)
        
        # ── Toggle: CROSSHAIR (full-width) ─────────────────────
        self.cross_btn = QPushButton("  CROSSHAIR")
        self.cross_btn.setProperty("class", "system-toggle-btn")
        self.cross_btn.setProperty("theme_icon", True)
        self.cross_btn.setProperty("icon_name", "crosshair.svg")
        self.cross_btn.setProperty("standard_icon", "")
        self.cross_btn.setProperty("icon_size_w", 16)
        self.cross_btn.setProperty("icon_size_h", 16)
        self._apply_themed_icon(self.cross_btn)
        self.cross_btn.setCheckable(True)
        self.cross_btn.setCursor(Qt.PointingHandCursor)
        self.cross_btn.setToolTip("Toggle crosshair overlay on 2D views")
        self.cross_btn.clicked.connect(lambda checked: self.multiplanar_tab.toggle_crosshair(Qt.Checked if checked else Qt.Unchecked))
        layout.addWidget(self.cross_btn)
        
        # ── Utility icon row: 4 equal cells, full width (matches LAYOUT/CROSSHAIR) ──
        util_row = QGridLayout()
        util_row.setContentsMargins(0, 0, 0, 0)
        util_row.setHorizontalSpacing(6)
        util_row.setVerticalSpacing(0)
        for c in range(4):
            util_row.setColumnStretch(c, 1)

        b_ch_reset = self._create_system_icon_button(
            "Reset crosshair position & rotation",
            self.multiplanar_tab.reset_crosshair,
            icon_name="refresh.svg",
            standard_icon=QStyle.SP_BrowserReload,
            fill=True,
        )
        b_ch_clr = self._create_system_icon_button(
            "Choose crosshair color",
            self.multiplanar_tab.choose_crosshair_color,
            icon_name="color-picker.svg",
            standard_icon=QStyle.SP_DriveDVDIcon,
            fill=True,
        )
        b_r_clr = self._create_system_icon_button(
            "Choose ruler color",
            self.multiplanar_tab.choose_ruler_color,
            icon_name="ruler-2.svg",
            standard_icon=QStyle.SP_DriveDVDIcon,
            fill=True,
        )
        b_rot_reset = self._create_system_icon_button(
            "Reset 2D Rotation",
            self.multiplanar_tab.reset_2d_rotation,
            icon_name="rotate-clockwise-2.svg",
            standard_icon=QStyle.SP_BrowserReload,
            fill=True,
        )
        util_row.addWidget(b_ch_reset, 0, 0)
        util_row.addWidget(b_ch_clr, 0, 1)
        util_row.addWidget(b_r_clr, 0, 2)
        util_row.addWidget(b_rot_reset, 0, 3)
        layout.addLayout(util_row)
        
        # Bottom stretch to push everything UP
        layout.addStretch()

        # Map to tab
        self.multiplanar_tab.load_volume_btn = b_open
        self.multiplanar_tab.load_volume_folder_btn = b_open  # unified — same button
        self.multiplanar_tab.load_class1_btn = b_c1
        self.multiplanar_tab.load_class2_btn = b_c2
        self.multiplanar_tab.color_c1_btn = b_c1_clr
        self.multiplanar_tab.color_c2_btn = b_c2_clr
        self.multiplanar_tab.clear_masks_btn = b_clear
        self.multiplanar_tab.crosshair_check = self.cross_btn
        
        # Timer to update bullets (slow poll; polish only when load-state changes)
        self.bullet_timer = QTimer(self)
        self.bullet_timer.timeout.connect(self._update_sidebar_bullets)
        self.bullet_timer.start(2000)
        
        return panel

    def _update_sidebar_bullets(self):
        """Update bullet and styling indicating loaded state."""
        tab = self.multiplanar_tab
        if not hasattr(tab, 'load_volume_btn'): return
        
        def set_bullet(btn, loaded):
            changed = False
            if loaded and not btn.property("loaded"):
                btn.setProperty("loaded", True)
                changed = True
            elif not loaded and btn.property("loaded"):
                btn.setProperty("loaded", False)
                changed = True
                 
            if changed:
                btn.style().unpolish(btn)
                btn.style().polish(btn)
                if bool(btn.property("theme_icon")):
                    self._apply_themed_icon(btn)
                
        # Use getattr safely for C1/C2 because we don't know immediately if they exist
        vol_loaded = hasattr(tab, 'volume_data') and tab.volume_data is not None
        c1_loaded = hasattr(tab, 'class1_data') and tab.class1_data is not None
        c2_loaded = hasattr(tab, 'class2_data') and tab.class2_data is not None
        
        set_bullet(tab.load_volume_btn, vol_loaded)
        if hasattr(tab, 'load_volume_folder_btn'): set_bullet(tab.load_volume_folder_btn, vol_loaded)
        if hasattr(tab, 'load_class1_btn'): set_bullet(tab.load_class1_btn, c1_loaded)
        if hasattr(tab, 'load_class2_btn'): set_bullet(tab.load_class2_btn, c2_loaded)

    def _create_segmentation_sidebar_panel(self):
        """Panel for 3D Segmentation tab (index 1)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 0, 15, 0)
        layout.setSpacing(6)
        
        seg = self.segmentation_tab
        
        def add_section_title(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(SemiconductorTheme.sidebar_section_style())
            layout.addWidget(lbl)

        # Environment
        add_section_title("IO & CONFIG")
        
        # DLL
        layout.addWidget(QLabel("DLL Folder:"))
        self.seg_dll_input = QLineEdit()
        self.seg_dll_input.setReadOnly(True)
        b_dll = self._create_compact_icon_button(
            "Browse DLL folder",
            seg.load_dll_folder,
            standard_icon=QStyle.SP_DirIcon,
            icon_name="folder.svg",
        )
        dll_row = QHBoxLayout()
        dll_row.setContentsMargins(0, 0, 0, 0)
        dll_row.setSpacing(6)
        dll_row.addWidget(self.seg_dll_input, 1)
        dll_row.addWidget(b_dll, 0)
        layout.addLayout(dll_row)
        
        # Config recipe — list all recipes from app config/ (shipped after build)
        layout.addWidget(QLabel("Config Recipe:"))
        self.seg_config_combo = QComboBox()
        self.seg_config_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.seg_config_combo.setToolTip(
            "Recipes from the config/ folder next to the app.\n"
            "Use Browse for a file outside that folder."
        )
        seg.register_config_recipe_combo(self.seg_config_combo)
        b_cfg = self._create_compact_icon_button(
            "Browse recipe outside config/ folder",
            seg.load_config_file,
            standard_icon=QStyle.SP_FileIcon,
            icon_name="settings.svg",
        )
        cfg_row = QHBoxLayout()
        cfg_row.setContentsMargins(0, 0, 0, 0)
        cfg_row.setSpacing(6)
        cfg_row.addWidget(self.seg_config_combo, 1)
        cfg_row.addWidget(b_cfg, 0)
        layout.addLayout(cfg_row)
        
        # Input
        layout.addWidget(QLabel("Input Path:"))
        self.seg_input_path = QLineEdit()
        self.seg_input_path.setReadOnly(True)
        
        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(6)
        
        b_input_file = self._create_compact_icon_button(
            "Browse input file",
            lambda: seg.load_volume(from_folder=False),
            standard_icon=QStyle.SP_FileIcon,
            icon_name="file.svg",
        )
        b_input_file.setProperty("is_icon_only", True)
        
        b_input_folder = self._create_compact_icon_button(
            "Browse input folder",
            lambda: seg.load_volume(from_folder=True),
            standard_icon=QStyle.SP_DirIcon,
            icon_name="folder-open.svg",
        )
        b_input_folder.setProperty("is_icon_only", True)
        
        input_row.addWidget(self.seg_input_path, 1)
        input_row.addWidget(b_input_file, 0)
        input_row.addWidget(b_input_folder, 0)
        layout.addLayout(input_row)

        # Output
        layout.addWidget(QLabel("Output Path:"))
        self.seg_output_input = QLineEdit()
        self.seg_output_input.setReadOnly(True)
        b_out = self._create_compact_icon_button(
            "Browse output folder",
            seg.browse_output_path,
            standard_icon=QStyle.SP_DialogOpenButton,
            icon_name="folder-open.svg",
        )
        out_row = QHBoxLayout()
        out_row.setContentsMargins(0, 0, 0, 0)
        out_row.setSpacing(6)
        out_row.addWidget(self.seg_output_input, 1)
        out_row.addWidget(b_out, 0)
        layout.addLayout(out_row)
        
        # Actions
        add_section_title("ACTIONS")
        b_clear = QPushButton("CLEAR MASK")
        b_clear.setProperty("class", "clear-btn")
        b_clear.setProperty("role", "danger")
        b_clear.clicked.connect(seg.clear_masks)
        layout.addWidget(b_clear)
        
        # Overlay & Crosshair
        add_section_title("OVERLAY")
        ch_row = QHBoxLayout()
        self.seg_cross_check = QCheckBox("Crosshair")
        self.seg_cross_check.stateChanged.connect(seg.toggle_crosshair)
        ch_row.addWidget(self.seg_cross_check)
        
        b_c1 = self._create_compact_icon_button(
            "Choose Class 1 color",
            lambda: seg.choose_overlay_color(1),
            text="C1",
            icon_name="color-picker.svg",
            standard_icon=QStyle.SP_DriveDVDIcon,
        )
        b_c2 = self._create_compact_icon_button(
            "Choose Class 2 color",
            lambda: seg.choose_overlay_color(2),
            text="C2",
            icon_name="color-picker.svg",
            standard_icon=QStyle.SP_DriveDVDIcon,
        )
        ch_row.addWidget(b_c1); ch_row.addWidget(b_c2)
        layout.addLayout(ch_row)

        # Info
        layout.addSpacing(10)
        self.seg_info_label = QLabel("Ready")
        self.seg_info_label.setWordWrap(True)
        self.seg_info_label.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt; font-style: italic;")
        layout.addWidget(self.seg_info_label)
        
        # Link back to tab
        seg.dll_path_input = self.seg_dll_input
        seg.config_path_input = self.seg_config_combo  # recipe combo (legacy attr name)
        seg.input_path_input = self.seg_input_path
        seg.output_path_input = self.seg_output_input
        seg.crosshair_check = self.seg_cross_check
        seg.info_label = self.seg_info_label
        
        # Sync DLL path on both sidebar + Teaching toolbar
        if hasattr(seg, 'dll_path') and seg.dll_path:
            if hasattr(seg, '_sync_dll_path_displays'):
                seg._sync_dll_path_displays(seg.dll_path)
            else:
                seg.dll_path_input.setText(seg.dll_path)
        # Populate recipe list from config/ and select current path if already loaded
        seg.refresh_config_recipe_list(select_path=getattr(seg, "config_path", None))
        
        layout.addStretch()
        return panel

    def _create_batch_review_sidebar_panel(self):
        """Sidebar for Batch Review tab (index 4)."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 0, 15, 0)
        layout.setSpacing(8)

        lbl = QLabel("LINE PULSE")
        lbl.setStyleSheet(SemiconductorTheme.sidebar_section_style())
        layout.addWidget(lbl)

        info = QLabel(
            "Live yield · FOV progress · open Viewer on demand.\n\n"
            "• Online mode appends each completed FOV\n"
            "• DB path: Inno3D_Data/inspection.db\n"
            "• Use Seed demo if empty (advanced)"
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8.5pt;")
        layout.addWidget(info)

        b_refresh = QPushButton("  REFRESH DB")
        b_refresh.setProperty("class", "list-btn")
        b_refresh.setCursor(Qt.PointingHandCursor)
        b_refresh.clicked.connect(
            lambda: self.batch_review_tab.refresh_all()
            if hasattr(self, "batch_review_tab")
            else None
        )
        layout.addWidget(b_refresh)

        b_seed = QPushButton("  SEED DEMO DATA")
        b_seed.setProperty("class", "list-btn")
        b_seed.setCursor(Qt.PointingHandCursor)
        b_seed.clicked.connect(
            lambda: self.batch_review_tab._seed_demo()
            if hasattr(self, "batch_review_tab")
            else None
        )
        layout.addWidget(b_seed)

        b_folder = QPushButton("  OPEN DB FOLDER")
        b_folder.setProperty("class", "list-btn")
        b_folder.setCursor(Qt.PointingHandCursor)
        b_folder.clicked.connect(
            lambda: self.batch_review_tab._open_db_folder()
            if hasattr(self, "batch_review_tab")
            else None
        )
        layout.addWidget(b_folder)

        layout.addStretch()
        return panel

    def _create_analysis_sidebar_panel(self):
        """Panel for 3D Analysis tab (index 3) — DB-first SOH workbench."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 0, 15, 0)
        layout.setSpacing(8)

        def add_section_title(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(SemiconductorTheme.sidebar_section_style())
            layout.addWidget(lbl)

        add_section_title("DATABASE")

        b_refresh_db = QPushButton("  REFRESH DB")
        b_refresh_db.setProperty("class", "list-btn")
        b_refresh_db.setProperty("theme_icon", True)
        b_refresh_db.setProperty("icon_name", "refresh.svg")
        self._apply_themed_icon(b_refresh_db)
        b_refresh_db.setCursor(Qt.PointingHandCursor)
        b_refresh_db.clicked.connect(self.analysis_tab.refresh_from_db)
        layout.addWidget(b_refresh_db)

        b_load_wafer = QPushButton("  LOAD WAFER")
        b_load_wafer.setProperty("class", "list-btn")
        b_load_wafer.setProperty("theme_icon", True)
        b_load_wafer.setProperty("icon_name", "layers.svg")
        self._apply_themed_icon(b_load_wafer)
        b_load_wafer.setCursor(Qt.PointingHandCursor)
        b_load_wafer.clicked.connect(self.analysis_tab.load_selected_wafer)
        layout.addWidget(b_load_wafer)

        b_load_checked = QPushButton("  LOAD CHECKED FOVS")
        b_load_checked.setProperty("class", "list-btn")
        b_load_checked.setProperty("theme_icon", True)
        b_load_checked.setProperty("icon_name", "check.svg")
        self._apply_themed_icon(b_load_checked)
        b_load_checked.setCursor(Qt.PointingHandCursor)
        b_load_checked.clicked.connect(self.analysis_tab.load_checked_fovs)
        layout.addWidget(b_load_checked)

        add_section_title("ANALYSIS SET")

        remove_row = QHBoxLayout()
        remove_row.setSpacing(4)

        b_remove = QPushButton(" REMOVE")
        b_remove.setProperty("class", "list-btn")
        b_remove.setProperty("theme_icon", True)
        b_remove.setProperty("icon_name", "trash.svg")
        self._apply_themed_icon(b_remove)
        b_remove.setCursor(Qt.PointingHandCursor)
        b_remove.clicked.connect(self.analysis_tab.remove_sample)
        remove_row.addWidget(b_remove)

        b_remove_all = QPushButton(" CLEAR")
        b_remove_all.setProperty("class", "list-btn")
        b_remove_all.setProperty("theme_icon", True)
        b_remove_all.setProperty("icon_name", "x.svg")
        self._apply_themed_icon(b_remove_all)
        b_remove_all.setCursor(Qt.PointingHandCursor)
        b_remove_all.clicked.connect(self.analysis_tab.remove_all_samples)
        b_remove_all.setStyleSheet("color: #ff6b9d;")
        remove_row.addWidget(b_remove_all)

        layout.addLayout(remove_row)

        b_export = QPushButton("  EXPORT REPORT")
        b_export.setProperty("class", "list-btn")
        b_export.setProperty("theme_icon", True)
        b_export.setProperty("icon_name", "device-floppy.svg")
        self._apply_themed_icon(b_export)
        b_export.setCursor(Qt.PointingHandCursor)
        b_export.clicked.connect(self.analysis_tab.export_report)
        layout.addWidget(b_export)

        b_recalc = QPushButton("  RECALCULATE")
        b_recalc.setProperty("class", "list-btn")
        b_recalc.setProperty("theme_icon", True)
        b_recalc.setProperty("icon_name", "refresh.svg")
        self._apply_themed_icon(b_recalc)
        b_recalc.setCursor(Qt.PointingHandCursor)
        b_recalc.clicked.connect(self.analysis_tab._recalculate_all)
        layout.addWidget(b_recalc)

        add_section_title("CSV (LAB)")

        b_import = QPushButton("  IMPORT SAMPLE")
        b_import.setProperty("class", "list-btn")
        b_import.setProperty("theme_icon", True)
        b_import.setProperty("icon_name", "file.svg")
        self._apply_themed_icon(b_import)
        b_import.setCursor(Qt.PointingHandCursor)
        b_import.clicked.connect(self.analysis_tab.import_sample)
        layout.addWidget(b_import)

        b_batch = QPushButton("  IMPORT BATCH")
        b_batch.setProperty("class", "list-btn")
        b_batch.setProperty("theme_icon", True)
        b_batch.setProperty("icon_name", "folder-open.svg")
        self._apply_themed_icon(b_batch)
        b_batch.setCursor(Qt.PointingHandCursor)
        b_batch.clicked.connect(self.analysis_tab.import_batch)
        layout.addWidget(b_batch)

        add_section_title("INFO")
        self.analysis_info_label = QLabel(
            "DB-first SOH · select lot/wafer · Load wafer or checked FOVs")
        self.analysis_info_label.setWordWrap(True)
        self.analysis_info_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt; font-style: italic;")
        layout.addWidget(self.analysis_info_label)

        layout.addStretch()
        return panel

