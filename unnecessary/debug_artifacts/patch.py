import os
import re

file_path = r"e:\semiconductor\DEMO\src_frontend_backend_v15\segmentation_tab.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Fix save_results_btn bug
content = re.sub(
    r"self\.save_results_btn\.setEnabled\(True\)",
    r"# self.save_results_btn.setEnabled(True)  # Removed",
    content
)

# 2. Add crosshair properties to __init__ (after crosshair_enabled)
init_vars = """        self.crosshair_enabled = False
        self.crosshair_position = [0, 0, 0]  # [x, y, z]
        self.crosshair_color = [1.0, 1.0, 0.0]  # Yellow
        self.ruler_color = [0.0, 1.0, 0.0]  # Green
        self.reverse_z = False
"""
content = re.sub(
    r"        self.crosshair_enabled = False",
    init_vars,
    content
)


# 3. Add reverse_z_btn definition logic + new control layout in create_single_plane_view
# We will just replace create_single_plane_view completely with the one from main.py adapted for TGV/Void
new_create_single_plane_view = '''
    def create_single_plane_view(self, title, orientation):
        """Create a single plane view with controls matching MultiPlanarView"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Header panel with controls
        header_widget = QWidget()
        header_widget.setMaximumHeight(30)
        header_widget.setStyleSheet(f"background: {SemiconductorTheme.BG_PANEL}; border-radius: 3px;")
        
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(5, 2, 5, 2)
        controls_layout.setSpacing(5)
        
        # Title label
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setStyleSheet(f"font-size: 9pt; color: {SemiconductorTheme.ACCENT_PRIMARY};")
        title_label.setMinimumWidth(100)
        controls_layout.addWidget(title_label)
        
        # Button group
        button_layout = QHBoxLayout()
        button_layout.setSpacing(2)
        
        fullscreen_btn = QToolButton()
        fullscreen_btn.setText("⛶")
        fullscreen_btn.setToolTip("Fullscreen view")
        fullscreen_btn.setMaximumSize(22, 22)
        fullscreen_btn.setMinimumSize(22, 22)
        fullscreen_btn.setStyleSheet(f"""
            QToolButton {{ background: {SemiconductorTheme.BG_LIGHT}; border: 1px solid {SemiconductorTheme.ACCENT_PRIMARY}; border-radius: 3px; color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 11pt; }}
            QToolButton:hover {{ background: {SemiconductorTheme.ACCENT_PRIMARY}; border-color: {SemiconductorTheme.ACCENT_PRIMARY}; color: {SemiconductorTheme.BG_DARK}; }}
        """)
        fullscreen_btn.clicked.connect(lambda: self.toggle_view_fullscreen(orientation))
        button_layout.addWidget(fullscreen_btn)
        
        reset_btn = QToolButton()
        reset_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        reset_btn.setIconSize(QSize(16, 16))
        reset_btn.setToolTip("Reset zoom & pan")
        reset_btn.setMaximumSize(22, 22)
        reset_btn.setMinimumSize(22, 22)
        reset_btn.setStyleSheet(f"""
            QToolButton {{ background: {SemiconductorTheme.BTN_SECONDARY_BG}; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; border-radius: 3px; }}
            QToolButton:hover {{ background: {SemiconductorTheme.ACCENT_WARNING}; border-color: {SemiconductorTheme.ACCENT_WARNING}; }}
        """)
        reset_btn.clicked.connect(lambda: self.reset_view(orientation))
        button_layout.addWidget(reset_btn)
        
        controls_layout.addLayout(button_layout)
        
        separator = QLabel("|")
        separator.setStyleSheet(f"color: {SemiconductorTheme.BORDER_DEFAULT}; padding: 0px 5px;")
        controls_layout.addWidget(separator)
        
        # Overlays
        overlay_tgv_check = QCheckBox("TGV (C1)")
        overlay_tgv_check.setMaximumHeight(20)
        overlay_tgv_check.setStyleSheet("font-size: 8pt;")
        overlay_tgv_check.stateChanged.connect(lambda state, o=orientation: self.update_plane_view(o, preserve_camera=True))
        controls_layout.addWidget(overlay_tgv_check)
        
        overlay_void_check = QCheckBox("Void (C2)")
        overlay_void_check.setMaximumHeight(20)
        overlay_void_check.setStyleSheet("font-size: 8pt;")
        overlay_void_check.stateChanged.connect(lambda state, o=orientation: self.update_plane_view(o, preserve_camera=True))
        controls_layout.addWidget(overlay_void_check)
        
        # Opacity
        opacity_label = QLabel("α:")
        opacity_label.setStyleSheet("font-size: 8pt;")
        controls_layout.addWidget(opacity_label)
        
        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setMinimum(0)
        opacity_slider.setMaximum(100)
        opacity_slider.setValue(70)
        opacity_slider.setMaximumWidth(60)
        opacity_slider.setMaximumHeight(16)
        opacity_slider.valueChanged.connect(lambda value, o=orientation: self.update_plane_view(o, preserve_camera=True))
        controls_layout.addWidget(opacity_slider)
        
        sep2 = QLabel("|")
        sep2.setStyleSheet(f"color: {SemiconductorTheme.BORDER_DEFAULT}; padding: 0px 3px;")
        controls_layout.addWidget(sep2)
        
        # Brightness
        bright_label = QLabel("☀")
        bright_label.setStyleSheet("font-size: 9pt;")
        controls_layout.addWidget(bright_label)
        brightness_slider = QSlider(Qt.Horizontal)
        brightness_slider.setMinimum(-100)
        brightness_slider.setMaximum(100)
        brightness_slider.setValue(0)
        brightness_slider.setMaximumWidth(55)
        brightness_slider.setMaximumHeight(16)
        brightness_slider.valueChanged.connect(lambda value, o=orientation: self.update_plane_view(o, preserve_camera=True))
        controls_layout.addWidget(brightness_slider)
        
        # Contrast
        contrast_label = QLabel("◐")
        contrast_label.setStyleSheet("font-size: 9pt;")
        controls_layout.addWidget(contrast_label)
        contrast_slider = QSlider(Qt.Horizontal)
        contrast_slider.setMinimum(-100)
        contrast_slider.setMaximum(100)
        contrast_slider.setValue(0)
        contrast_slider.setMaximumWidth(55)
        contrast_slider.setMaximumHeight(16)
        contrast_slider.valueChanged.connect(lambda value, o=orientation: self.update_plane_view(o, preserve_camera=True))
        controls_layout.addWidget(contrast_slider)
        
        # Z-direction toggle
        if orientation == 'axial':
            sep3 = QLabel("|")
            sep3.setStyleSheet(f"color: {SemiconductorTheme.BORDER_DEFAULT}; padding: 0px 3px;")
            controls_layout.addWidget(sep3)
            
            self.reverse_z_btn = QPushButton("↕Z")
            self.reverse_z_btn.setFixedWidth(30)
            self.reverse_z_btn.setMaximumHeight(20)
            self.reverse_z_btn.setCheckable(True)
            self.reverse_z_btn.setStyleSheet(f"""
                QPushButton {{ background: {SemiconductorTheme.BG_LIGHT}; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; border-radius: 3px; font-size: 8pt; font-weight: bold; color: {SemiconductorTheme.TEXT_SECONDARY}; }}
                QPushButton:checked {{ background: {SemiconductorTheme.ACCENT_PRIMARY}; color: {SemiconductorTheme.BG_DARK}; border-color: {SemiconductorTheme.ACCENT_PRIMARY}; }}
            """)
            self.reverse_z_btn.clicked.connect(self.toggle_reverse_z)
            controls_layout.addWidget(self.reverse_z_btn)
            
        controls_layout.addStretch()
        
        # Pixel info
        pixel_label = QLabel("Pixel: --")
        pixel_label.setMinimumWidth(100)
        pixel_label.setStyleSheet(f"font-size: 8pt; color: {SemiconductorTheme.TEXT_SECONDARY};")
        controls_layout.addWidget(pixel_label)
        
        header_widget.setLayout(controls_layout)
        layout.addWidget(header_widget)
        
        # VTK widget
        vtk_widget = QVTKRenderWindowInteractor()
        style = vtk.vtkInteractorStyleImage()
        vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
        vtk_widget.installEventFilter(self)
        
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0, 0, 0)
        vtk_widget.GetRenderWindow().AddRenderer(renderer)
        layout.addWidget(vtk_widget, 1)
        
        # Slice slider
        slice_widget = QWidget()
        slice_widget.setMaximumHeight(25)
        slice_layout = QHBoxLayout()
        slice_layout.setContentsMargins(5, 0, 5, 0)
        slice_layout.setSpacing(5)
        
        slice_text = QLabel("Slice:")
        slice_text.setStyleSheet("font-size: 8pt;")
        slice_layout.addWidget(slice_text)
        
        slice_slider = QSlider(Qt.Horizontal)
        slice_slider.setMinimum(0)
        slice_slider.setMaximum(100)
        slice_slider.setValue(50)
        slice_slider.setMaximumHeight(16)
        slice_slider.valueChanged.connect(lambda value, o=orientation: self.update_slice(o, value))
        slice_layout.addWidget(slice_slider, 1)
        
        slice_label = QLabel("50 / 100")
        slice_label.setStyleSheet("font-size: 8pt; min-width: 50px;")
        slice_slider.valueChanged.connect(lambda value, lbl=slice_label, s=slice_slider: lbl.setText(f"{value} / {s.maximum()}"))
        slice_layout.addWidget(slice_label)
        
        slice_widget.setLayout(slice_layout)
        layout.addWidget(slice_widget)
        
        widget.setLayout(layout)
        
        # Store refs
        setattr(self, f'{orientation}_widget', vtk_widget)
        setattr(self, f'{orientation}_renderer', renderer)
        setattr(self, f'{orientation}_slice_slider', slice_slider)
        setattr(self, f'{orientation}_slice_label', slice_label)
        setattr(self, f'{orientation}_overlay_tgv', overlay_tgv_check)
        setattr(self, f'{orientation}_overlay_void', overlay_void_check)
        setattr(self, f'{orientation}_opacity_slider', opacity_slider)
        setattr(self, f'{orientation}_brightness_slider', brightness_slider)
        setattr(self, f'{orientation}_contrast_slider', contrast_slider)
        setattr(self, f'{orientation}_pixel_label', pixel_label)
        setattr(self, f'{orientation}_container', widget)
        
        return widget
'''

# We need to replace `def create_single_plane_view` block in content
start_idx = content.find('def create_single_plane_view(self, orientation):')
end_idx = content.find('def create_parameter_panel(self):')

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_create_single_plane_view.lstrip() + "\n    " + content[end_idx:]

# Grid setup call changes
content = content.replace(
    'axial_widget = self.create_single_plane_view("axial")',
    'axial_widget = self.create_single_plane_view("XY", "axial")'
)
content = content.replace(
    'sagittal_widget = self.create_single_plane_view("sagittal")',
    'sagittal_widget = self.create_single_plane_view("YZ", "sagittal")'
)
content = content.replace(
    'coronal_widget = self.create_single_plane_view("coronal")',
    'coronal_widget = self.create_single_plane_view("XZ", "coronal")'
)

# Replace the toolbar layout with crosshair tools included
toolbar_code = """        # Actions
        actions_layout = QHBoxLayout()
        
        self.load_dll_btn = QPushButton("Load DLL")
        self.load_dll_btn.clicked.connect(self.load_dll)
        actions_layout.addWidget(self.load_dll_btn)
        
        self.load_config_btn = QPushButton("Load Config")
        self.load_config_btn.clicked.connect(self.load_config)
        actions_layout.addWidget(self.load_config_btn)
        
        self.load_vol_btn = QPushButton("Load Volume")
        self.load_vol_btn.clicked.connect(self.load_volume)
        actions_layout.addWidget(self.load_vol_btn)
        
        # Crosshair toggle
        self.crosshair_check = QCheckBox("Crosshair")
        self.crosshair_check.stateChanged.connect(self.toggle_crosshair)
        actions_layout.addWidget(self.crosshair_check)
        
        # Crosshair Color Pickers
        self.crosshair_color_btn = QPushButton("C. Color")
        self.crosshair_color_btn.setToolTip("Choose Crosshair Color")
        self.crosshair_color_btn.clicked.connect(self.choose_crosshair_color)
        self.crosshair_color_btn.setStyleSheet(f"background-color: {SemiconductorTheme.BG_LIGHT}; color: yellow; font-weight: bold; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; border-radius: 3px;")
        actions_layout.addWidget(self.crosshair_color_btn)
        
        self.ruler_color_btn = QPushButton("R. Color")
        self.ruler_color_btn.setToolTip("Choose Ruler Color")
        self.ruler_color_btn.clicked.connect(self.choose_ruler_color)
        self.ruler_color_btn.setStyleSheet(f"background-color: {SemiconductorTheme.BG_LIGHT}; color: green; font-weight: bold; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; border-radius: 3px;")
        actions_layout.addWidget(self.ruler_color_btn)
        
        # Crosshair position sliders
        for axis in ['x', 'y', 'z']:
            lbl = QLabel(f"{axis.upper()}:")
            lbl.setStyleSheet("font-size: 8pt;")
            actions_layout.addWidget(lbl)
            
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(50)
            slider.setMaximumWidth(60)
            # 0=X (sagittal), 1=Y (coronal), 2=Z (axial)
            idx = 0 if axis=='x' else (1 if axis=='y' else 2)
            slider.valueChanged.connect(lambda value, i=idx: self.on_crosshair_slider_changed(i, value))
            actions_layout.addWidget(slider)
            setattr(self, f'crosshair_slider_{axis}', slider)
            
        actions_layout.addStretch()
        layout.addLayout(actions_layout)"""

# Sub the actions layout inside create_toolbar
content = re.sub(
    r"\s*# Actions\s*actions_layout = QHBoxLayout\(\)[\s\S]*?layout\.addLayout\(actions_layout\)",
    "\n" + toolbar_code,
    content
)

# And inject missing methods: toggle_view_fullscreen, exit_fullscreen, reset_view, eventFilter, update_pixel_value, handle_crosshair_click, on_crosshair_slider_changed, toggle_reverse_z, create_crosshair_actors, add_ruler_overlay, choose_colors, update_slice
missing_methods = """
    def toggle_reverse_z(self):
        if hasattr(self, 'reverse_z_btn'):
            self.reverse_z = self.reverse_z_btn.isChecked()
        if self.volume_data is not None:
            self.current_slices['axial'] = (self.volume_data.shape[0] - 1 - self.current_slices['axial'])
            self.axial_slice_slider.blockSignals(True)
            self.axial_slice_slider.setValue(self.current_slices['axial'])
            self.axial_slice_slider.blockSignals(False)
            self.update_plane_view('axial')

    def toggle_view_fullscreen(self, orientation):
        container = getattr(self, f'{orientation}_container')
        parent = container.parentWidget()
        parent_layout = parent.layout()

        if not hasattr(self, 'fullscreen_widget') or self.fullscreen_widget is None:
            self.fullscreen_widget = container
            self.fullscreen_orientation = orientation
            self.fullscreen_parent = parent
            self.fullscreen_parent_layout = parent_layout

            self.fullscreen_placeholder = QWidget()
            self.fullscreen_parent_layout.replaceWidget(container, self.fullscreen_placeholder)
            self.fullscreen_placeholder.show()

            self.fullscreen_window = QWidget()
            self.fullscreen_window.setWindowTitle(f"Fullscreen - {orientation.capitalize()} View")
            self.fullscreen_window.setWindowFlags(Qt.Window)

            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)

            hint_label = QLabel("Press ESC to exit fullscreen")
            hint_label.setStyleSheet(f"background: {SemiconductorTheme.BG_DARK}; color: {SemiconductorTheme.ACCENT_WARNING}; padding: 5px; font-size: 9pt;")
            layout.addWidget(hint_label)

            container.setParent(self.fullscreen_window)
            layout.addWidget(container)

            self.fullscreen_window.setLayout(layout)
            self.fullscreen_window.showFullScreen()

            self.fullscreen_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self.fullscreen_window)
            self.fullscreen_shortcut.activated.connect(lambda: self.exit_fullscreen())
        else:
            self.exit_fullscreen()

    def exit_fullscreen(self):
        if hasattr(self, 'fullscreen_widget') and self.fullscreen_widget is not None:
            widget = self.fullscreen_widget
            if hasattr(self, 'fullscreen_parent_layout') and hasattr(self, 'fullscreen_placeholder'):
                if self.fullscreen_placeholder is not None:
                    self.fullscreen_parent_layout.replaceWidget(self.fullscreen_placeholder, widget)
                    widget.setParent(self.fullscreen_parent)
                    self.fullscreen_placeholder.deleteLater()
                    self.fullscreen_placeholder = None

            if hasattr(self, 'fullscreen_shortcut') and self.fullscreen_shortcut is not None:
                self.fullscreen_shortcut = None

            if hasattr(self, 'fullscreen_window') and self.fullscreen_window is not None:
                self.fullscreen_window.close()
                self.fullscreen_window = None

            self.fullscreen_widget = None

    def reset_view(self, orientation):
        self.camera_states[orientation] = None
        self.update_plane_view(orientation, preserve_camera=False)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseMove:
            for orientation in ['axial', 'coronal', 'sagittal']:
                widget = getattr(self, f'{orientation}_widget', None)
                if widget and obj == widget and self.volume_data is not None:
                    self.update_pixel_value(orientation, event.pos())
                    if self.crosshair_enabled and (event.buttons() & Qt.LeftButton):
                        self.handle_crosshair_click(orientation, event.pos())
        
        elif event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton and self.crosshair_enabled:
                for orientation in ['axial', 'coronal', 'sagittal']:
                    widget = getattr(self, f'{orientation}_widget', None)
                    if widget and obj == widget and self.volume_data is not None:
                        self.handle_crosshair_click(orientation, event.pos())
        
        return super().eventFilter(obj, event)

    def update_pixel_value(self, orientation, pos):
        try:
            widget = getattr(self, f'{orientation}_widget')
            renderer = getattr(self, f'{orientation}_renderer')
            x, y = pos.x(), pos.y()
            size = widget.GetRenderWindow().GetSize()
            vtk_y = size[1] - y
            
            picker = vtk.vtkWorldPointPicker()
            picker.Pick(x, vtk_y, 0, renderer)
            world_pos = picker.GetPickPosition()
            
            slice_idx = self.current_slices[orientation]
            vol_z, vol_y, vol_x = self.volume_data.shape
            value = None
            
            if orientation == 'axial':
                px, py = int(world_pos[0]), int(world_pos[1])
                if 0 <= px < vol_x and 0 <= py < vol_y:
                    actual_z = (vol_z - 1 - slice_idx) if getattr(self, 'reverse_z', False) else slice_idx
                    value = self.volume_data[actual_z, py, px]
            elif orientation == 'coronal':
                px, pz = int(world_pos[0]), int(world_pos[1])
                if 0 <= px < vol_x and 0 <= pz < vol_z:
                     value = self.volume_data[pz, slice_idx, px]
            else:
                 pz, py = int(world_pos[0]), int(world_pos[1])
                 if 0 <= pz < vol_z and 0 <= py < vol_y:
                     value = self.volume_data[pz, py, slice_idx]

            label = getattr(self, f'{orientation}_pixel_label')
            if value is not None:
                label.setText(f"Pixel: {value}")
            else:
                label.setText("Pixel: --")
        except:
            pass

    def handle_crosshair_click(self, orientation, pos):
        try:
            widget = getattr(self, f'{orientation}_widget')
            renderer = getattr(self, f'{orientation}_renderer')
            
            x, y = pos.x(), pos.y()
            size = widget.GetRenderWindow().GetSize()
            vtk_y = size[1] - y
            
            picker = vtk.vtkWorldPointPicker()
            picker.Pick(x, vtk_y, 0, renderer)
            world_pos = picker.GetPickPosition()
            
            vol_z, vol_y, vol_x = self.volume_data.shape
            
            if orientation == 'axial':
                px, py = int(world_pos[0]), int(world_pos[1])
                if 0 <= px < vol_x and 0 <= py < vol_y:
                    self.crosshair_position[0] = px
                    self.crosshair_position[1] = py
                    self.crosshair_position[2] = self.current_slices['axial'] if not getattr(self, 'reverse_z', False) else (vol_z - 1 - self.current_slices['axial'])
                    self.current_slices['coronal'] = py
                    self.current_slices['sagittal'] = px

            elif orientation == 'coronal':
                px, pz = int(world_pos[0]), int(world_pos[1])
                if 0 <= px < vol_x and 0 <= pz < vol_z:
                    self.crosshair_position[0] = px
                    self.crosshair_position[1] = self.current_slices['coronal']
                    self.crosshair_position[2] = pz
                    self.current_slices['axial'] = pz if not getattr(self, 'reverse_z', False) else (vol_z - 1 - pz)
                    self.current_slices['sagittal'] = px

            else:
                pz, py = int(world_pos[0]), int(world_pos[1])
                if 0 <= py < vol_y and 0 <= pz < vol_z:
                    self.crosshair_position[0] = self.current_slices['sagittal']
                    self.crosshair_position[1] = py
                    self.crosshair_position[2] = pz
                    self.current_slices['axial'] = pz if not getattr(self, 'reverse_z', False) else (vol_z - 1 - pz)
                    self.current_slices['coronal'] = py

            for ori in ['axial', 'coronal', 'sagittal']:
                slider = getattr(self, f'{ori}_slice_slider')
                slider.blockSignals(True)
                slider.setValue(self.current_slices[ori])
                slider.blockSignals(False)
                label = getattr(self, f'{ori}_slice_label')
                label.setText(f"{self.current_slices[ori]} / {slider.maximum()}")
            
            for ori in ['axial', 'coronal', 'sagittal']:
                self.update_plane_view(ori, preserve_camera=True)
            
        except:
            pass
            
    def toggle_crosshair(self, state):
        self.crosshair_enabled = (state == Qt.Checked)
        if self.volume_data is not None:
            for orientation in ['axial', 'coronal', 'sagittal']:
                self.update_plane_view(orientation, preserve_camera=True)

    def on_crosshair_slider_changed(self, axis_idx, value):
        if self.volume_data is None:
            return
        self.crosshair_position[axis_idx] = value
        
        if axis_idx == 0:
            self.sagittal_slice_slider.blockSignals(True)
            self.sagittal_slice_slider.setValue(value)
            self.sagittal_slice_slider.blockSignals(False)
        elif axis_idx == 1:
            self.coronal_slice_slider.blockSignals(True)
            self.coronal_slice_slider.setValue(value)
            self.coronal_slice_slider.blockSignals(False)
        elif axis_idx == 2:
            val = value if not getattr(self, 'reverse_z', False) else (self.volume_data.shape[0] - 1 - value)
            self.axial_slice_slider.blockSignals(True)
            self.axial_slice_slider.setValue(val)
            self.axial_slice_slider.blockSignals(False)
            
        self.current_slices['sagittal'] = self.sagittal_slice_slider.value()
        self.current_slices['coronal'] = self.coronal_slice_slider.value()
        self.current_slices['axial'] = self.axial_slice_slider.value()
        
        for orientation in ['axial', 'coronal', 'sagittal']:
            self.update_plane_view(orientation, preserve_camera=True)

    def choose_crosshair_color(self):
        init_c = QColor(int(self.crosshair_color[0]*255), int(self.crosshair_color[1]*255), int(self.crosshair_color[2]*255))
        color = QColorDialog.getColor(init_c, self, "Crosshair Color")
        if color.isValid():
            self.crosshair_color = [color.redF(), color.greenF(), color.blueF()]
            self.crosshair_color_btn.setStyleSheet(f"background-color: {SemiconductorTheme.BG_LIGHT}; color: {color.name()}; font-weight: bold; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; border-radius: 3px;")
            for ori in ['axial', 'coronal', 'sagittal']:
                self.update_plane_view(ori, preserve_camera=True)
                
    def choose_ruler_color(self):
        init_c = QColor(int(self.ruler_color[0]*255), int(self.ruler_color[1]*255), int(self.ruler_color[2]*255))
        color = QColorDialog.getColor(init_c, self, "Ruler Color")
        if color.isValid():
            self.ruler_color = [color.redF(), color.greenF(), color.blueF()]
            self.ruler_color_btn.setStyleSheet(f"background-color: {SemiconductorTheme.BG_LIGHT}; color: {color.name()}; font-weight: bold; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; border-radius: 3px;")
            for ori in ['axial', 'coronal', 'sagittal']:
                self.update_plane_view(ori, preserve_camera=True)

    def create_crosshair_actors(self, orientation, renderer, slice_shape):
        actors = []
        try:
            h, w = slice_shape
            z_sign = '-Z' if getattr(self, 'reverse_z', False) else '+Z'
            
            if orientation == 'axial':
                cx_w = float(self.crosshair_position[0])
                cy_w = float(self.crosshair_position[1])
                h_label, v_label = '+X', '+Y'
            elif orientation == 'coronal':
                cx_w = float(self.crosshair_position[0])
                cy_w = float(self.crosshair_position[2])
                h_label, v_label = '+X', z_sign
            else:
                cx_w = float(self.crosshair_position[2])
                cy_w = float(self.crosshair_position[1])
                h_label, v_label = z_sign, '+Y'
                
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToWorld()
            coord.SetValue(cx_w, cy_w, 0.5)
            dp = coord.GetComputedDisplayValue(renderer)
            dx = float(dp[0])
            dy = float(dp[1])
            
            win = renderer.GetRenderWindow().GetSize()
            vp = renderer.GetViewport()
            vp_w = (vp[2] - vp[0]) * win[0]
            vp_h = (vp[3] - vp[1]) * win[1]
            ruler_w = 35
            gap = 12
            
            ch_color = self.crosshair_color
            
            def make_line2d(x1, y1, x2, y2, lw=1):
                ls = vtk.vtkLineSource()
                ls.SetPoint1(x1, y1, 0)
                ls.SetPoint2(x2, y2, 0)
                m = vtk.vtkPolyDataMapper2D()
                m.SetInputConnection(ls.GetOutputPort())
                a = vtk.vtkActor2D()
                a.SetMapper(m)
                a.GetProperty().SetColor(*ch_color)
                a.GetProperty().SetLineWidth(lw)
                return a
            
            def make_text2d(text, px, py, size=11):
                t = vtk.vtkTextActor()
                t.SetInput(text)
                t.SetPosition(px, py)
                t.GetTextProperty().SetFontSize(size)
                t.GetTextProperty().SetColor(*ch_color)
                t.GetTextProperty().BoldOn()
                return t
                
            if dx > ruler_w + gap:
                actors.append(make_line2d(ruler_w, dy, dx - gap, dy))
            actors.append(make_line2d(dx + gap, dy, vp_w, dy))
            if dy > gap:
                actors.append(make_line2d(dx, 0, dx, dy - gap))
            if dy < vp_h - gap:
                actors.append(make_line2d(dx, dy + gap, dx, vp_h))
                
            lx = dx + (vp_w - dx) * 0.70
            lx = max(dx + gap + 40, min(lx, vp_w - 30))
            actors.append(make_text2d(h_label, lx, dy + 4))
            
            if orientation == 'coronal':
                ly = dy - (dy) * 0.70
                ly = max(5, min(ly, dy - gap - 10))
                actors.append(make_text2d(v_label, dx + 5, ly))
            else:
                ly = dy + (vp_h - dy) * 0.70
                ly = max(dy + gap + 40, min(ly, vp_h - 20))
                actors.append(make_text2d(v_label, dx + 5, ly))
                
        except:
            pass
        return actors

    def add_ruler_overlay(self, renderer, orientation, slice_shape):
        try:
            h, w = slice_shape
            rc = getattr(self, 'ruler_color', [0,1,0])
            camera = renderer.GetActiveCamera()
            parallel_scale = camera.GetParallelScale()
            win_size = renderer.GetRenderWindow().GetSize()
            viewport = renderer.GetViewport()
            vp_h_px = (viewport[3] - viewport[1]) * win_size[1]
            if vp_h_px < 1: vp_h_px = win_size[1]
            world_per_px = (2.0 * parallel_scale) / vp_h_px if vp_h_px > 0 else 1.0
            mm_per_px = world_per_px * 1.0 # assume spacing 1.0 for simplicity
            if mm_per_px <= 0: mm_per_px = 1.0
            
            for mm_step in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]:
                tick_px = mm_step / mm_per_px
                if tick_px >= 20: break
                
            label_every = 5
            label_mm_step = mm_step * label_every
            label_px_step = label_mm_step / mm_per_px
            while label_px_step < 40:
                label_every *= 2
                label_mm_step = mm_step * label_every
                label_px_step = label_mm_step / mm_per_px
                
            ruler_x = 16
            tick_short = 4
            tick_long = 9
            
            line2d = vtk.vtkLineSource()
            line2d.SetPoint1(ruler_x, 0, 0)
            line2d.SetPoint2(ruler_x, vp_h_px, 0)
            lm = vtk.vtkPolyDataMapper2D()
            lm.SetInputConnection(line2d.GetOutputPort())
            la = vtk.vtkActor2D()
            la.SetMapper(lm)
            la.GetProperty().SetColor(*rc)
            la.GetProperty().SetLineWidth(1)
            renderer.AddActor(la)
            
            mm_val = 0.0
            y_px = 0.0
            tick_count = 0
            max_ticks = int(vp_h_px / max(tick_px, 1)) + 2
            for _ in range(max_ticks):
                if y_px > vp_h_px: break
                is_major = (tick_count % label_every == 0)
                tlen = tick_long if is_major else tick_short
                tick = vtk.vtkLineSource()
                tick.SetPoint1(ruler_x - tlen // 2, y_px, 0)
                tick.SetPoint2(ruler_x + tlen // 2, y_px, 0)
                tm = vtk.vtkPolyDataMapper2D()
                tm.SetInputConnection(tick.GetOutputPort())
                ta = vtk.vtkActor2D()
                ta.SetMapper(tm)
                ta.GetProperty().SetColor(*rc)
                ta.GetProperty().SetLineWidth(1)
                renderer.AddActor(ta)
                
                if is_major:
                    txt = f"{int(round(mm_val))}" if (mm_val >= 1 or mm_val == 0) else f"{mm_val:.1f}"
                    lbl = vtk.vtkTextActor()
                    lbl.SetInput(txt)
                    lbl.GetTextProperty().SetFontSize(9)
                    lbl.GetTextProperty().SetColor(*rc)
                    lbl.GetTextProperty().BoldOn()
                    lbl.SetPosition(ruler_x + tlen // 2 + 3, y_px - 5)
                    renderer.AddActor(lbl)
                mm_val += mm_step
                y_px += tick_px
                tick_count += 1
        except: pass
"""

content += "\n" + missing_methods

# Now we must update `update_plane_view` logic to use opacity_slider and toggle_crosshair
# And we also need to change `def update_plane_view(self, orientation):` to `def update_plane_view(self, orientation, preserve_camera=False):`

update_logic = """
    def update_plane_view(self, orientation, preserve_camera=False):
        if self.volume_data is None:
            return
            
        renderer = getattr(self, f'{orientation}_renderer')
        slice_idx = self.current_slices[orientation]
        
        # Save camera state
        if preserve_camera:
            camera = renderer.GetActiveCamera()
            self.camera_states[orientation] = {
                'position': camera.GetPosition(),
                'focal_point': camera.GetFocalPoint(),
                'view_up': camera.GetViewUp(),
                'parallel_scale': camera.GetParallelScale()
            }
        
        if orientation == 'axial':
            actual_z = (self.volume_data.shape[0] - 1 - slice_idx) if getattr(self, 'reverse_z', False) else slice_idx
            slice_data = self.volume_data[actual_z, :, :]
            tgv_slice = self.tgv_segmentation[actual_z, :, :] if self.tgv_segmentation is not None else None
            void_slice = self.void_segmentation[actual_z, :, :] if self.void_segmentation is not None else None
        elif orientation == 'coronal':
            slice_data = self.volume_data[:, slice_idx, :]
            tgv_slice = self.tgv_segmentation[:, slice_idx, :] if self.tgv_segmentation is not None else None
            void_slice = self.void_segmentation[:, slice_idx, :] if self.void_segmentation is not None else None
        else:
            slice_data = np.flipud(np.rot90(self.volume_data[:, :, slice_idx]))
            tgv_slice = np.flipud(np.rot90(self.tgv_segmentation[:, :, slice_idx])) if self.tgv_segmentation is not None else None
            void_slice = np.flipud(np.rot90(self.void_segmentation[:, :, slice_idx])) if self.void_segmentation is not None else None
            
        h, w = slice_data.shape
        renderer.RemoveAllViewProps()
        
        # Base image
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(w, h, 1)
        vtk_image.SetOrigin(0.0, 0.0, 0.0)
        
        slice_transposed = np.transpose(slice_data, (1, 0))
        flat_data = np.ascontiguousarray(slice_transposed.flatten('F'))
        
        # Assume uint16 or uint8
        if slice_data.dtype == np.uint8:
            vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        else:
            vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
            
        vtk_image.GetPointData().SetScalars(vtk_array)
        image_actor = vtk.vtkImageActor()
        image_actor.GetMapper().SetInputData(vtk_image)
        
        prop = image_actor.GetProperty()
        v_min, v_max = float(slice_data.min()), float(slice_data.max())
        base_window = v_max - v_min if v_max > v_min else 1.0
        base_level = v_min + base_window / 2.0
        
        bright_slider = getattr(self, f'{orientation}_brightness_slider')
        contrast_slider = getattr(self, f'{orientation}_contrast_slider')
        brightness_val = bright_slider.value() if bright_slider else 0
        contrast_val = contrast_slider.value() if contrast_slider else 0
        
        level = base_level + brightness_val * (base_window / 200.0)
        contrast_factor = max(0.2, 1.0 - contrast_val / 100.0)
        window = base_window * contrast_factor
        
        prop.SetColorWindow(window)
        prop.SetColorLevel(level)
        prop.SetInterpolationTypeToNearest()
        renderer.AddActor(image_actor)
        
        # Camera
        if preserve_camera and getattr(self, 'camera_states', {}).get(orientation) is not None:
             state = self.camera_states[orientation]
             camera = renderer.GetActiveCamera()
             camera.SetPosition(state['position'])
             camera.SetFocalPoint(state['focal_point'])
             camera.SetViewUp(state['view_up'])
             camera.SetParallelScale(state['parallel_scale'])
        else:
            camera = renderer.GetActiveCamera()
            camera.ParallelProjectionOn()
            center_x, center_y = w / 2.0, h / 2.0
            camera.SetPosition(center_x, center_y, 1000)
            camera.SetFocalPoint(center_x, center_y, 0)
            camera.SetViewUp(0, 1, 0)
            camera.SetParallelScale(max(h, w) * 0.55)
            renderer.ResetCameraClippingRange()
            
        # Add overlay
        tgv_enabled = getattr(self, f'{orientation}_overlay_tgv').isChecked()
        void_enabled = getattr(self, f'{orientation}_overlay_void').isChecked()
        opacity = getattr(self, f'{orientation}_opacity_slider').value() / 100.0
        
        if (tgv_enabled or void_enabled) and (tgv_slice is not None or void_slice is not None):
            overlay_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            has_overlay = False
            
            if tgv_enabled and tgv_slice is not None:
                mask_tgv = (tgv_slice == 128) | (tgv_slice == 255)
                overlay_rgb[mask_tgv, 1] = 255  # Green
                has_overlay = True
                
            if void_enabled and void_slice is not None:
                mask_void = (void_slice == 128) | (void_slice == 255)
                overlay_rgb[mask_void, 0] = 255
                overlay_rgb[mask_void, 1] = 165
                overlay_rgb[mask_void, 2] = 0
                has_overlay = True
                
            # Add highlight for selected
            if getattr(self, 'selected_highlight_objects', []) and getattr(self, 'labeled_class1_data', None) is not None:
                for cls_num, obj_id in self.selected_highlight_objects:
                    if cls_num == 1:
                        if orientation == 'axial':
                            actual_z_hl = (self.labeled_class1_data.shape[0] - 1 - slice_idx) if getattr(self, 'reverse_z', False) else slice_idx
                            mask_highlight = (self.labeled_class1_data[actual_z_hl, :, :] == obj_id)
                        elif orientation == 'coronal':
                            mask_highlight = (self.labeled_class1_data[:, slice_idx, :] == obj_id)
                        else:
                            mask_highlight = np.flipud(np.rot90(self.labeled_class1_data[:, :, slice_idx])) == obj_id
                        overlay_rgb[mask_highlight] = [0, 255, 255]
                        has_overlay = True
                        
            if has_overlay:
                overlay_transposed = np.transpose(overlay_rgb, (1, 0, 2))
                flat_rgb = np.ascontiguousarray(overlay_transposed.reshape(-1, 3, order='F'))
                vtk_overlay = vtk.vtkImageData()
                vtk_overlay.SetDimensions(w, h, 1)
                vtk_overlay.SetOrigin(0.0, 0.0, 0.0)
                vtk_rgb = numpy_support.numpy_to_vtk(flat_rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
                vtk_rgb.SetNumberOfComponents(3)
                vtk_overlay.GetPointData().SetScalars(vtk_rgb)
                
                overlay_actor = vtk.vtkImageActor()
                overlay_actor.GetMapper().SetInputData(vtk_overlay)
                overlay_actor.SetOpacity(opacity)
                renderer.AddActor(overlay_actor)
        
        # Draw Crosshair
        if getattr(self, 'crosshair_enabled', False):
            actors = self.create_crosshair_actors(orientation, renderer, (h, w))
            for a in actors: renderer.AddActor(a)
            self.add_ruler_overlay(renderer, orientation, (h, w))
            
        widget = getattr(self, f'{orientation}_widget', None)
        if widget:
            widget.GetRenderWindow().Render()

"""

# Regex substitute the old update_plane_view entirely
content = re.sub(
    r"\s+def update_plane_view\(self, orientation\):.*?(?=\s+def run_measurement\(self\):|\s+def get_result_info)",
    update_logic + "\n",
    content,
    flags=re.DOTALL
)

# And missing methods `update_slice(self, orientation, value)`? Wait, it needs to update current_slices and re-render.
# Oh, it was `update_slice` inside `create_single_plane_view` lambda.
missing_update_slice = """
    def update_slice(self, orientation, value):
        self.current_slices[orientation] = value
        if self.volume_data is not None:
            if orientation == 'axial':
                self.crosshair_position[2] = value if not getattr(self, 'reverse_z', False) else (self.volume_data.shape[0] - 1 - value)
            elif orientation == 'coronal':
                self.crosshair_position[1] = value
            elif orientation == 'sagittal':
                self.crosshair_position[0] = value
                
            for ori in ['axial', 'coronal', 'sagittal']:
                self.update_plane_view(ori, preserve_camera=True)
"""
content += missing_update_slice

# Add ndimage import if needed
if "from scipy import ndimage" not in content:
    content = content.replace("from vtk.util import numpy_support", "from vtk.util import numpy_support\nfrom scipy import ndimage")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied.")
