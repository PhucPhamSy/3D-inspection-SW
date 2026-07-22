import glob
import os

import numpy as np
import vtk
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import *
from scipy import ndimage
from skimage import io, measure
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from inno3d.core.styles import SemiconductorTheme
from inno3d.core.view_support import LoadSegmentationThread, RenderSegmentationThread


class InspectionResultTab(QWidget):
    """Tab cho Inspection Result - Segmentation Mask"""
    
    def __init__(self):
        super().__init__()
        self.segmentation_data = None
        self.class1_data = None  # ThÃªm
        self.class2_data = None  # ThÃªm        
        self.labeled_data = None
        self.spacing = [1.0, 1.0, 1.0]
        self.downsample_factor = 1
        
        # Colors: BG tá»‘i, Class1 vÃ ng gold, Class2 Ä‘á»
        self.colors = {
            0:   [0.10, 0.10, 0.12],   # BG xÃ¡m ráº¥t tá»‘i
            128: [1.00, 0.84, 0.00],   # Class 1 - vÃ ng gold
            255: [1.00, 0.10, 0.10]    # Class 2 - Ä‘á»
        }
        
        self.actors = {}
        self.clip_planes = {'x': None, 'y': None, 'z': None}
        self.clip_enabled = {'x': False, 'y': False, 'z': False}
        
        self.load_thread = None
        self.render_thread = None
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Top controls
        top_layout = QHBoxLayout()
        
        # Thay tháº¿ nÃºt load cÅ© báº±ng 2 nÃºt má»›i
        self.load_class1_btn = QPushButton("Load Class 1 Mask")
        self.load_class1_btn.clicked.connect(lambda: self.load_class_mask(1))
        top_layout.addWidget(self.load_class1_btn)

        self.load_class2_btn = QPushButton("Load Class 2 Mask")
        self.load_class2_btn.clicked.connect(lambda: self.load_class_mask(2))
        top_layout.addWidget(self.load_class2_btn)

        self.load_sep_layer_btn = QPushButton("Load from sep_layer/")
        self.load_sep_layer_btn.setStyleSheet(
            f"background-color: {SemiconductorTheme.ACCENT_PRIMARY}; "
            f"color: {SemiconductorTheme.BG_DARK}; font-weight: bold; padding: 4px 10px; border-radius: 4px;"
        )
        self.load_sep_layer_btn.clicked.connect(self.load_from_sep_layer)
        top_layout.addWidget(self.load_sep_layer_btn)

        self.load_ml_output_btn = QPushButton("Load Multi-Layer Output")
        self.load_ml_output_btn.setToolTip("Load bump/void results from multi-layer segmentation output folder")
        self.load_ml_output_btn.setStyleSheet(
            f"background-color: #2196F3; "
            f"color: white; font-weight: bold; padding: 4px 10px; border-radius: 4px;"
        )
        self.load_ml_output_btn.clicked.connect(self.load_multi_layer_output)
        top_layout.addWidget(self.load_ml_output_btn)
        
        self.clear_masks_btn = QPushButton("Clear")
        self.clear_masks_btn.clicked.connect(self.clear_masks)
        top_layout.addWidget(self.clear_masks_btn)        

        # Label hiá»ƒn thá»‹ tráº¡ng thÃ¡i
        self.class_status_label = QLabel("No masks loaded")
        self.class_status_label.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY};")
        top_layout.addWidget(self.class_status_label)
        
        top_layout.addWidget(QLabel("Downsample:"))
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setMinimum(1)
        self.downsample_spin.setMaximum(8)
        self.downsample_spin.setValue(1)
        top_layout.addWidget(self.downsample_spin)
        
        top_layout.addWidget(QLabel("Decimate:"))
        self.decimate_spin = QDoubleSpinBox()
        self.decimate_spin.setMinimum(0.0)
        self.decimate_spin.setMaximum(0.9)
        self.decimate_spin.setValue(0.5)
        self.decimate_spin.setSingleStep(0.1)
        top_layout.addWidget(self.decimate_spin)
        
        top_layout.addWidget(QLabel("Smooth:"))
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setMinimum(0)
        self.smooth_spin.setMaximum(100)
        self.smooth_spin.setValue(10)
        self.smooth_spin.setToolTip("Smoothing iterations: 0=None, 10=Default, 50=High")
        top_layout.addWidget(self.smooth_spin)
        
        self.rerender_btn = QPushButton("Re-render")
        self.rerender_btn.clicked.connect(self.render_segmentation)
        self.rerender_btn.setEnabled(False)
        top_layout.addWidget(self.rerender_btn)
        
        self.color_bg_btn = QPushButton("BG")
        self.color_bg_btn.clicked.connect(lambda: self.change_color(0))
        top_layout.addWidget(self.color_bg_btn)
        
        self.color_c1_btn = QPushButton("C1")
        self.color_c1_btn.clicked.connect(lambda: self.change_color(128))
        top_layout.addWidget(self.color_c1_btn)
        
        self.color_c2_btn = QPushButton("C2")
        self.color_c2_btn.clicked.connect(lambda: self.change_color(255))
        top_layout.addWidget(self.color_c2_btn)
        
        self.show_bg_check = QCheckBox("BG")
        self.show_bg_check.setChecked(True)
        self.show_bg_check.stateChanged.connect(lambda: self.toggle_visibility(0))
        top_layout.addWidget(self.show_bg_check)
        
        self.show_c1_check = QCheckBox("C1")
        self.show_c1_check.setChecked(True)
        self.show_c1_check.stateChanged.connect(lambda: self.toggle_visibility(128))
        top_layout.addWidget(self.show_c1_check)
        
        self.show_c2_check = QCheckBox("C2")
        self.show_c2_check.setChecked(True)
        self.show_c2_check.stateChanged.connect(lambda: self.toggle_visibility(255))
        top_layout.addWidget(self.show_c2_check)
        
        self.show_c2_check.setChecked(True)
        self.show_c2_check.stateChanged.connect(lambda: self.toggle_visibility(255))
        top_layout.addWidget(self.show_c2_check)
        
        top_layout.addStretch()
        
        self.toolbar = QWidget()
        self.toolbar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        tb_layout = QVBoxLayout(self.toolbar)
        tb_layout.setContentsMargins(0, 0, 0, 0)
        tb_layout.addLayout(top_layout)
        
        # --- LIGHTING CONTROLS (New Row) ---
        light_layout = QHBoxLayout()
        light_layout.setContentsMargins(0, 0, 0, 0)
        
        # Opacity
        light_layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setFixedWidth(80)
        self.opacity_slider.valueChanged.connect(self.update_lighting_properties)
        light_layout.addWidget(self.opacity_slider)
        
        # Ambient
        light_layout.addWidget(QLabel("Amb:"))
        self.ambient_slider = QSlider(Qt.Horizontal)
        self.ambient_slider.setRange(0, 100)
        self.ambient_slider.setValue(20) # 0.2
        self.ambient_slider.setFixedWidth(80)
        self.ambient_slider.valueChanged.connect(self.update_lighting_properties)
        light_layout.addWidget(self.ambient_slider)

        # Diffuse
        light_layout.addWidget(QLabel("Diff:"))
        self.diffuse_slider = QSlider(Qt.Horizontal)
        self.diffuse_slider.setRange(0, 100)
        self.diffuse_slider.setValue(80) # 0.8
        self.diffuse_slider.setFixedWidth(80)
        self.diffuse_slider.valueChanged.connect(self.update_lighting_properties)
        light_layout.addWidget(self.diffuse_slider)

        # Specular
        light_layout.addWidget(QLabel("Spec:"))
        self.specular_slider = QSlider(Qt.Horizontal)
        self.specular_slider.setRange(0, 100)
        self.specular_slider.setValue(30) # 0.3
        self.specular_slider.setFixedWidth(80)
        self.specular_slider.valueChanged.connect(self.update_lighting_properties)
        light_layout.addWidget(self.specular_slider)

        # Specular Power
        light_layout.addWidget(QLabel("Pow:"))
        self.spec_power_slider = QSlider(Qt.Horizontal)
        self.spec_power_slider.setRange(0, 100)
        self.spec_power_slider.setValue(30) # 30.0
        self.spec_power_slider.setFixedWidth(80)
        self.spec_power_slider.valueChanged.connect(self.update_lighting_properties)
        light_layout.addWidget(self.spec_power_slider)
        
        light_layout.addStretch()
        
        # Background Color Button
        self.bg_color_btn = QPushButton("BG Color")
        self.bg_color_btn.clicked.connect(self.choose_background_color)
        light_layout.addWidget(self.bg_color_btn)
        
        tb_layout.addLayout(light_layout)
        layout.addWidget(self.toolbar)

        
        # Main content
        content_layout = QHBoxLayout()
        
        # VTK Widget Container - wrap in proper frame
        vtk_container = QWidget()
        vtk_container.setStyleSheet(f"background: #000; border: 1px solid {SemiconductorTheme.BORDER_GLOW};")
        vtk_layout = QVBoxLayout(vtk_container)
        vtk_layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = QVTKRenderWindowInteractor()
        vtk_layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.05, 0.08)  # Deep dark bg
        self.renderer.GetActiveCamera().ParallelProjectionOn() # Orthographic Projection
        
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(1, 1, 1)
        light.SetFocalPoint(0, 0, 0)
        light.SetIntensity(0.8)
        self.renderer.AddLight(light)
        
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        content_layout.addWidget(vtk_container, 3)
        
        # Right panel
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)
        
        self.info_label = QLabel("Waiting for Mask Data...")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #777; font-style: italic;")
        right_panel.addWidget(self.info_label)
        
        # Clipping
        clip_group = QGroupBox("CLIPPING PLANES")
        clip_layout = QVBoxLayout()
        
        for axis in ['x', 'y', 'z']:
            axis_layout = QHBoxLayout()
            checkbox = QCheckBox(f"{axis.upper()}-Axis")
            checkbox.stateChanged.connect(
                lambda state, a=axis: self.toggle_clipping(a, state)
            )
            axis_layout.addWidget(checkbox)
            
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(50)
            slider.valueChanged.connect(
                lambda value, a=axis: self.update_clipping(a, value)
            )
            slider.setEnabled(False)
            axis_layout.addWidget(slider)
            
            label = QLabel("50%")
            label.setFixedWidth(35)
            axis_layout.addWidget(label)
            
            clip_layout.addLayout(axis_layout)
            setattr(self, f'clip_{axis}_check', checkbox)
            setattr(self, f'clip_{axis}_slider', slider)
            setattr(self, f'clip_{axis}_label', label)
        
        clip_group.setLayout(clip_layout)
        right_panel.addWidget(clip_group)
        
        # Statistics
        stats_group = QGroupBox("STATS ANALYSIS")
        stats_layout = QVBoxLayout()
        
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels([
            "ID", "C1", "C2", "Ratio"
        ])
        self.stats_table.setStyleSheet(f"QHeaderView::section {{ background: {SemiconductorTheme.BG_DARK}; color: {SemiconductorTheme.TEXT_SECONDARY}; }}")
        stats_layout.addWidget(self.stats_table)
        
        stats_btn_row = QHBoxLayout()
        self.calc_stats_btn = QPushButton("RUN ANALYSIS")
        self.calc_stats_btn.setProperty("class", "primary")
        self.calc_stats_btn.clicked.connect(self.calculate_statistics)
        self.calc_stats_btn.setEnabled(False)
        stats_btn_row.addWidget(self.calc_stats_btn)

        self.load_batch_csv_btn = QPushButton("Load Batch CSV")
        self.load_batch_csv_btn.setToolTip("Load CSV results from multiple packages")
        self.load_batch_csv_btn.setStyleSheet(
            f"background-color: #FF9800; color: white; font-weight: bold; padding: 4px 10px; border-radius: 4px;"
        )
        self.load_batch_csv_btn.clicked.connect(self.load_batch_csv)
        stats_btn_row.addWidget(self.load_batch_csv_btn)

        self.export_batch_btn = QPushButton("Export Combined")
        self.export_batch_btn.setToolTip("Export combined batch results to CSV")
        self.export_batch_btn.clicked.connect(self.export_batch_csv)
        self.export_batch_btn.setEnabled(False)
        stats_btn_row.addWidget(self.export_batch_btn)
        
        stats_layout.addLayout(stats_btn_row)

        # Batch info label
        self.batch_info_label = QLabel("")
        self.batch_info_label.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        stats_layout.addWidget(self.batch_info_label)

        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)
        
        right_panel.addStretch()
        content_layout.addLayout(right_panel, 1)
        layout.addLayout(content_layout, 1)  # Added stretch factor 1 here
        
        self.setLayout(layout)
        
        for axis in ['x', 'y', 'z']:
            slider = getattr(self, f'clip_{axis}_slider')
            label = getattr(self, f'clip_{axis}_label')
            slider.valueChanged.connect(lambda value, lbl=label: lbl.setText(f"{value}%"))
        

    # ==================== SEP_LAYER FOLDER LOADING ====================

    def load_from_sep_layer(self):
        """Browse a sep_layer/ folder and load files as Class 1 / Class 2 masks."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select sep_layer/ folder or its parent", ""
        )
        if not folder:
            return

        sep_path = os.path.join(folder, "sep_layer")
        if os.path.isdir(sep_path):
            target = sep_path
        elif os.path.basename(folder).lower() == "sep_layer":
            target = folder
        else:
            target = folder

        tif_files = sorted(
            glob.glob(os.path.join(target, "*.tif")) +
            glob.glob(os.path.join(target, "*.tiff"))
        )

        if not tif_files:
            QMessageBox.warning(self, "No TIFF Files",
                "No .tif/.tiff files found in:\n" + target)
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Assign Layer Files to Classes")
        dialog.setMinimumWidth(550)
        dlg_layout = QVBoxLayout(dialog)

        info_lbl = QLabel()
        info_lbl.setText(
            "<b>" + str(len(tif_files)) + " file(s)</b> found in:<br><i>" + target + "</i><br>"
            "Assign each file to Class 1 (Bump) or Class 2 (Void):"
        )
        dlg_layout.addWidget(info_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        dlg_layout.addWidget(scroll)

        combos = []
        for fpath in tif_files:
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 2, 0, 2)
            fname = os.path.basename(fpath)
            row_l.addWidget(QLabel(fname), 1)
            cb = QComboBox()
            cb.addItems(["Class 1 (Bump)", "Class 2 (Void)", "Skip"])
            if "void" in fname.lower():
                cb.setCurrentIndex(1)
            row_l.addWidget(cb)
            scroll_layout.addWidget(row_w)
            combos.append((fpath, cb))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        dlg_layout.addWidget(btn_box)

        if dialog.exec_() != QDialog.Accepted:
            return

        class1_files = [fp for fp, cb in combos if cb.currentIndex() == 0]
        class2_files = [fp for fp, cb in combos if cb.currentIndex() == 1]

        if not class1_files and not class2_files:
            QMessageBox.information(self, "Nothing to load", "All files were skipped.")
            return

        downsample = self.downsample_spin.value()

        progress = QProgressDialog("Stacking layer files...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        try:
            if class1_files:
                progress.setLabelText("Loading " + str(len(class1_files)) + " Class 1 file(s)...")
                progress.setValue(10)
                self.class1_data = self._stack_tif_files(class1_files, downsample)

            if class2_files:
                progress.setLabelText("Loading " + str(len(class2_files)) + " Class 2 file(s)...")
                progress.setValue(55)
                self.class2_data = self._stack_tif_files(class2_files, downsample)

            progress.setValue(90)
            progress.setLabelText("Merging and rendering...")

            self.downsample_factor = downsample

            status_parts = []
            if self.class1_data is not None:
                status_parts.append("C1: " + str(self.class1_data.shape))
            if self.class2_data is not None:
                status_parts.append("C2: " + str(self.class2_data.shape))
            self.class_status_label.setText(" | ".join(status_parts))
            self.class_status_label.setStyleSheet("color: " + SemiconductorTheme.ACCENT_SUCCESS + ";")

            self.merge_class_masks()
            self.rerender_btn.setEnabled(True)
            self.calc_stats_btn.setEnabled(True)
            self.render_segmentation()
        except Exception as e:
            QMessageBox.critical(self, "Error", "Failed to load sep_layer files:\n" + str(e))
        finally:
            progress.close()

    def _stack_tif_files(self, file_paths, downsample=1):
        """Read and stack multiple TIFF files along Z axis into shape (Z, Y, X)."""
        import tifffile
        stacks = []
        for fp in file_paths:
            arr = tifffile.imread(fp)
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            stacks.append(arr)
        combined = np.concatenate(stacks, axis=0)
        if downsample > 1:
            combined = combined[::downsample, ::downsample, ::downsample]
        return combined

    def load_multi_layer_output(self):
        """Load bump/void results from multi-layer segmentation output folder.
        
        Expected structure:
            output_dir/
                layer_name_1/
                    bump_output.tif (or *bump*.tif)
                    void_output.tif (or *void*.tif)
                layer_name_2/
                    ...
        """
        folder = QFileDialog.getExistingDirectory(
            self, "Select Multi-Layer Segmentation Output Folder", ""
        )
        if not folder:
            return

        # Find subfolders containing TIFF files
        subfolders = sorted([
            d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
        ])

        if not subfolders:
            QMessageBox.warning(self, "No Subfolders",
                f"No layer subfolders found in:\n{folder}\n\n"
                "Expected: output_dir/layer_name_1/, layer_name_2/, ...")
            return

        downsample = self.downsample_spin.value()

        progress = QProgressDialog("Loading multi-layer results...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        try:
            bump_stacks = []
            void_stacks = []
            loaded_layers = 0

            for idx, sub in enumerate(subfolders):
                sub_path = os.path.join(folder, sub)
                tifs = glob.glob(os.path.join(sub_path, "*.tif")) + glob.glob(os.path.join(sub_path, "*.tiff"))
                if not tifs:
                    continue

                progress.setLabelText(f"[{idx+1}/{len(subfolders)}] Loading {sub}...")
                progress.setValue(int(idx / len(subfolders) * 80))
                QApplication.processEvents()

                for fp in tifs:
                    fname = os.path.basename(fp).lower()
                    arr = io.imread(fp)
                    if arr.ndim == 2:
                        arr = arr[np.newaxis]
                    if downsample > 1:
                        arr = arr[::downsample, ::downsample, ::downsample]

                    if "bump" in fname:
                        bump_stacks.append(arr)
                    elif "void" in fname:
                        void_stacks.append(arr)

                loaded_layers += 1

            if not bump_stacks and not void_stacks:
                progress.close()
                QMessageBox.warning(self, "No Results",
                    "No bump/void TIFF files found in subfolders.\n"
                    "Expected filenames containing 'bump' or 'void'.")
                return

            # Pad helper for dimension mismatches
            def _pad_and_concat(stack_list):
                if not stack_list:
                    return None
                max_y = max(a.shape[1] for a in stack_list)
                max_x = max(a.shape[2] for a in stack_list)
                padded = []
                for a in stack_list:
                    if a.shape[1] == max_y and a.shape[2] == max_x:
                        padded.append(a)
                    else:
                        p = np.zeros((a.shape[0], max_y, max_x), dtype=a.dtype)
                        p[:, :a.shape[1], :a.shape[2]] = a
                        padded.append(p)
                return np.concatenate(padded, axis=0)

            progress.setLabelText("Stacking arrays...")
            progress.setValue(85)
            QApplication.processEvents()

            if bump_stacks:
                self.class1_data = _pad_and_concat(bump_stacks)
            if void_stacks:
                self.class2_data = _pad_and_concat(void_stacks)

            self.downsample_factor = downsample

            status_parts = []
            if self.class1_data is not None:
                status_parts.append(f"C1(Bump): {self.class1_data.shape}")
            if self.class2_data is not None:
                status_parts.append(f"C2(Void): {self.class2_data.shape}")
            status_parts.append(f"{loaded_layers} layers")
            self.class_status_label.setText(" | ".join(status_parts))
            self.class_status_label.setStyleSheet(f"color: {SemiconductorTheme.ACCENT_SUCCESS};")

            self.merge_class_masks()
            self.rerender_btn.setEnabled(True)
            self.calc_stats_btn.setEnabled(True)
            self.render_segmentation()

            progress.setValue(100)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load multi-layer output:\n{str(e)}")
        finally:
            progress.close()

    def load_segmentation(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select First TIFF File", "", "TIFF Files (*.tif *.tiff)"
        )
        
        if file_path:
            downsample = self.downsample_spin.value()
            
            self.progress = QProgressDialog("Loading segmentation...", "Cancel", 0, 100, self)
            self.progress.setWindowModality(Qt.WindowModal)
            self.progress.setMinimumDuration(0)
            self.progress.setValue(0)
            
            self.load_btn.setEnabled(False)
            
            self.load_thread = LoadSegmentationThread(file_path, downsample)
            self.load_thread.progress.connect(self.on_load_progress)
            self.load_thread.finished.connect(self.on_load_finished)
            self.load_thread.start()
    
    def on_load_progress(self, value, message):
        self.progress.setValue(value)
        self.progress.setLabelText(message)
    
    def on_load_finished(self, data, error):
        self.progress.close()
        self.load_btn.setEnabled(True)
        
        if error:
            QMessageBox.critical(self, "Error", f"Failed to load: {error}")
            return
        
        self.segmentation_data = data
        self.downsample_factor = self.downsample_spin.value()
        
        unique_values = np.unique(data)
        counts = {val: np.sum(data == val) for val in unique_values}
        
        info_text = f"Shape (Z,Y,X): {data.shape}\n"
        info_text += f"Downsample: {self.downsample_factor}x\n"
        info_text += f"Memory: {data.nbytes / 1024**2:.1f} MB\n"
        info_text += f"Unique values: {unique_values}\n"
        for val, count in counts.items():
            info_text += f"  Value {val}: {count} voxels\n"
        
        self.info_label.setText(info_text)
        
        self.rerender_btn.setEnabled(True)
        self.calc_stats_btn.setEnabled(True)
        
        self.render_segmentation()
    
    def render_segmentation(self):
        if self.segmentation_data is None:
            return
        
        self.progress = QProgressDialog("Rendering 3D mesh...", "Cancel", 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setValue(0)
        
        self.load_class1_btn.setEnabled(False)  # Thay vÃ¬ self.load_btn
        self.load_class2_btn.setEnabled(False)  # ThÃªm dÃ²ng nÃ y
        self.rerender_btn.setEnabled(False)
        self.calc_stats_btn.setEnabled(False)
        
        z, y, x = self.segmentation_data.shape
        self.clip_x_slider.setMaximum(x - 1)
        self.clip_x_slider.setValue(x // 2)
        self.clip_y_slider.setMaximum(y - 1)
        self.clip_y_slider.setValue(y // 2)
        self.clip_z_slider.setMaximum(z - 1)
        self.clip_z_slider.setValue(z // 2)
        
        spacing = [s * self.downsample_factor for s in self.spacing]
        decimate = self.decimate_spin.value()
        smooth_iter = self.smooth_spin.value()
        
        self.render_thread = RenderSegmentationThread(
            self.segmentation_data,
            self.colors,
            spacing,
            decimate,
            smooth_iter
        )
        self.render_thread.progress.connect(self.on_render_progress)
        self.render_thread.finished.connect(self.on_render_finished)
        self.render_thread.start()
    
    def on_render_progress(self, value, message):
        self.progress.setValue(value)
        self.progress.setLabelText(message)
    
    def on_render_finished(self, actors, error):
        self.progress.close()
        self.load_class1_btn.setEnabled(True)  # Thay vÃ¬ self.load_btn
        self.load_class2_btn.setEnabled(True)  # ThÃªm dÃ²ng nÃ y
        self.rerender_btn.setEnabled(True)
        self.calc_stats_btn.setEnabled(True)
        
        if error:
            QMessageBox.critical(self, "Error", f"Failed to render: {error}")
            return
        
        for actor in self.actors.values():
            self.renderer.RemoveActor(actor)
        
        self.actors = actors
        for class_val, actor in actors.items():
            self.renderer.AddActor(actor)
            
            if class_val == 0:
                actor.SetVisibility(self.show_bg_check.isChecked())
            elif class_val == 128:
                actor.SetVisibility(self.show_c1_check.isChecked())
            elif class_val == 255:
                actor.SetVisibility(self.show_c2_check.isChecked())
        
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
    
    def toggle_visibility(self, class_value):
        if class_value in self.actors:
            if class_value == 0:
                visible = self.show_bg_check.isChecked()
            elif class_value == 128:
                visible = self.show_c1_check.isChecked()
            else:
                visible = self.show_c2_check.isChecked()
            
            self.actors[class_value].SetVisibility(visible)
            self.vtk_widget.GetRenderWindow().Render()
    
    def change_color(self, class_value):
        color = QColorDialog.getColor()
        if color.isValid():
            self.colors[class_value] = [color.redF(), color.greenF(), color.blueF()]
            
            if class_value in self.actors:
                self.actors[class_value].GetProperty().SetColor(self.colors[class_value])
                self.vtk_widget.GetRenderWindow().Render()

    def update_lighting_properties(self):
        """Update lighting properties for all actors"""
        opacity = self.opacity_slider.value() / 100.0
        ambient = self.ambient_slider.value() / 100.0
        diffuse = self.diffuse_slider.value() / 100.0
        specular = self.specular_slider.value() / 100.0
        specular_power = float(self.spec_power_slider.value())
        
        for class_val, actor in self.actors.items():
            prop = actor.GetProperty()
            
            # Keep BG transparency low relative to master opacity
            if class_val == 0:
                prop.SetOpacity(0.15 * opacity)
            else:
                prop.SetOpacity(opacity)
            
            prop.SetAmbient(ambient)
            prop.SetDiffuse(diffuse)
            prop.SetSpecular(specular)
            prop.SetSpecularPower(specular_power)
            
        self.vtk_widget.GetRenderWindow().Render()

    def choose_background_color(self):
        """Choose background color for 3D view"""
        color = QColorDialog.getColor()
        if color.isValid():
            r, g, b = color.redF(), color.greenF(), color.blueF()
            self.renderer.SetBackground(r, g, b)
            self.vtk_widget.GetRenderWindow().Render()
    
    def toggle_clipping(self, axis, state):
        self.clip_enabled[axis] = (state == Qt.Checked)
        slider = getattr(self, f'clip_{axis}_slider')
        slider.setEnabled(self.clip_enabled[axis])
        
        if self.clip_enabled[axis]:
            self.update_clipping(axis, slider.value())
        else:
            self.remove_clipping(axis)
    
    def update_clipping(self, axis, value):
        if not self.clip_enabled[axis] or self.segmentation_data is None:
            return
        
        plane = vtk.vtkPlane()
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map[axis]
        
        normal = [0, 0, 0]
        normal[axis_idx] = 1
        
        spacing = [s * self.downsample_factor for s in self.spacing]
        origin = [0, 0, 0]
        origin[axis_idx] = value * spacing[axis_idx]
        
        plane.SetNormal(normal)
        plane.SetOrigin(origin)
        self.clip_planes[axis] = plane
        
        for actor in self.actors.values():
            mapper = actor.GetMapper()
            mapper.RemoveAllClippingPlanes()
        
        for ax in ['x', 'y', 'z']:
            if self.clip_enabled[ax] and self.clip_planes[ax]:
                for actor in self.actors.values():
                    mapper = actor.GetMapper()
                    mapper.AddClippingPlane(self.clip_planes[ax])
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def remove_clipping(self, axis):
        for actor in self.actors.values():
            mapper = actor.GetMapper()
            mapper.RemoveAllClippingPlanes()
        
        for ax in ['x', 'y', 'z']:
            if ax != axis and self.clip_enabled[ax] and self.clip_planes[ax]:
                for actor in self.actors.values():
                    mapper = actor.GetMapper()
                    mapper.AddClippingPlane(self.clip_planes[ax])
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def calculate_statistics(self):
        if self.segmentation_data is None:
            return
        
        progress = QProgressDialog("Calculating statistics...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(10)
        
        class1_mask = (self.segmentation_data == 128).astype(np.uint8)
        labeled_class1, num_instances = ndimage.label(class1_mask)
        
        progress.setValue(30)
        
        class2_mask = (self.segmentation_data == 255).astype(np.uint8)
        
        # Use configured voxel sizes from SegmentationTab if available
        vx, vy, vz = 1.0, 1.0, 1.0
        z_stretched_4x = False
        main_window = self.window()
        if hasattr(main_window, 'segmentation_tab') and hasattr(main_window.segmentation_tab, 'param_widgets'):
            sw = main_window.segmentation_tab.param_widgets
            if 'voxel_size_x' in sw:
                vx = sw['voxel_size_x'].value()
                vy = sw['voxel_size_y'].value()
                vz = sw['voxel_size_z'].value()
            if sw.get('z_stretched_4x') and sw['z_stretched_4x'].isChecked():
                z_stretched_4x = True
        
        # If Z-Stretched 4x mode is enabled, divide effective Z voxel size by 4
        if z_stretched_4x:
            vz = vz / 4.0
            print(f"[MEASUREMENT TAB] Z-Stretched 4x mode: effective vz = {vz} µm")
        
        voxel_volume = vx * vy * vz
        
        stats = []
        
        for instance_id in range(1, num_instances + 1):
            progress.setValue(30 + int(70 * instance_id / num_instances))
            
            instance_mask = (labeled_class1 == instance_id)
            class1_volume = np.sum(instance_mask) * voxel_volume
            
            class2_in_instance = class2_mask & instance_mask
            class2_volume = np.sum(class2_in_instance) * voxel_volume
            
            ratio = class2_volume / (class1_volume + class2_volume)
            
            stats.append({
                'instance': instance_id,
                'class1_vol': class1_volume,
                'class2_vol': class2_volume,
                'ratio': ratio
            })
        
        self.stats_table.setRowCount(len(stats))
        
        for row, stat in enumerate(stats):
            self.stats_table.setItem(row, 0, QTableWidgetItem(str(stat['instance'])))
            self.stats_table.setItem(row, 1, QTableWidgetItem(f"{stat['class1_vol']:.2f}"))
            self.stats_table.setItem(row, 2, QTableWidgetItem(f"{stat['class2_vol']:.2f}"))
            
            if stat['ratio']:
                self.stats_table.setItem(row, 3, QTableWidgetItem(f"{stat['ratio']*100:.3f}%"))
            else:
                self.stats_table.setItem(row, 3, QTableWidgetItem("N/A"))
        
        progress.setValue(100)
        progress.close()

    def clear_masks(self):
        """Clear all loaded masks"""
        self.class1_data = None
        self.class2_data = None
        self.segmentation_data = None
        
        # Remove actors
        for actor in self.actors.values():
            self.renderer.RemoveActor(actor)
        self.actors = {}
        
        self.vtk_widget.GetRenderWindow().Render()
        
        # Reset UI state
        self.class_status_label.setText("No masks loaded")
        self.class_status_label.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY};")
        self.info_label.setText("Waiting for Mask Data...")
        self.rerender_btn.setEnabled(False)
        self.calc_stats_btn.setEnabled(False)
        self.stats_table.setRowCount(0)

    def load_class_mask(self, class_num):
        """Load mask cho class 1 hoáº·c 2"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select Class {class_num} Binary Mask", "", "TIFF Files (*.tif *.tiff)"
        )
        
        if file_path:
            downsample = self.downsample_spin.value()
            
            self.progress = QProgressDialog(f"Loading Class {class_num} mask...", "Cancel", 0, 100, self)
            self.progress.setWindowModality(Qt.WindowModal)
            self.progress.setMinimumDuration(0)
            self.progress.setValue(0)
            
            if class_num == 1:
                self.load_class1_btn.setEnabled(False)
            else:
                self.load_class2_btn.setEnabled(False)
            
            # DÃ¹ng thread cÃ³ sáºµn nhÆ°ng tag class number
            self.load_thread = LoadSegmentationThread(file_path, downsample)
            self.load_thread.class_num = class_num  # Tag Ä‘á»ƒ biáº¿t Ä‘ang load class nÃ o
            self.load_thread.progress.connect(self.on_load_progress)
            self.load_thread.finished.connect(lambda data, error: self.on_class_load_finished(data, error, class_num))
            self.load_thread.start()
    
    def on_class_load_finished(self, data, error, class_num):
        """Handle khi load xong 1 class mask"""
        self.progress.close()
        
        if class_num == 1:
            self.load_class1_btn.setEnabled(True)
        else:
            self.load_class2_btn.setEnabled(True)
        
        if error:
            QMessageBox.critical(self, "Error", f"Failed to load Class {class_num}: {error}")
            return
        
        # LÆ°u data theo class
        if class_num == 1:
            self.class1_data = data
        else:
            self.class2_data = data
        
        # Update status
        status_parts = []
        if self.class1_data is not None:
            status_parts.append(f"C1: {self.class1_data.shape}")
        if self.class2_data is not None:
            status_parts.append(f"C2: {self.class2_data.shape}")
        
        if status_parts:
            self.class_status_label.setText(" | ".join(status_parts))
            self.class_status_label.setStyleSheet(f"color: {SemiconductorTheme.ACCENT_SUCCESS};")
        
        # Náº¿u cÃ³ Ã­t nháº¥t 1 class, merge vÃ  render
        if self.class1_data is not None or self.class2_data is not None:
            self.merge_class_masks()
            self.rerender_btn.setEnabled(True)
            self.calc_stats_btn.setEnabled(True)
            self.render_segmentation()
    
    def merge_class_masks(self):
        """Merge cÃ¡c class masks thÃ nh segmentation_data"""
        # Láº¥y shape tá»« class cÃ³ sáºµn
        if self.class1_data is not None:
            shape = self.class1_data.shape
        elif self.class2_data is not None:
            shape = self.class2_data.shape
        else:
            return
        
        # Khá»Ÿi táº¡o segmentation data vá»›i background (0)
        self.segmentation_data = np.zeros(shape, dtype=np.uint8)
        
        # Add Class 1 (value = 128) 
        if self.class1_data is not None:
            # Binary mask: 255 = class, 0 = background
            mask_c1 = self.class1_data > 127  # True where white (255)
            self.segmentation_data[mask_c1] = 128
        
        # Add Class 2 (value = 255) - override Class 1 náº¿u overlap
        if self.class2_data is not None:
            mask_c2 = self.class2_data > 127
            self.segmentation_data[mask_c2] = 255
        
        # Info text
        unique_values = np.unique(self.segmentation_data)
        counts = {val: np.sum(self.segmentation_data == val) for val in unique_values}
        
        info_text = f"Merged Shape (Z,Y,X): {shape}\n"
        info_text += f"Downsample: {self.downsample_factor}x\n"
        info_text += f"Memory: {self.segmentation_data.nbytes / 1024**2:.1f} MB\n"
        info_text += f"Classes loaded:\n"
        
        if self.class1_data is not None:
            c1_pixels = np.sum(self.class1_data > 127)
            info_text += f"  Class 1: {c1_pixels} voxels\n"
        
        if self.class2_data is not None:
            c2_pixels = np.sum(self.class2_data > 127)
            info_text += f"  Class 2: {c2_pixels} voxels\n"
        
        info_text += f"Merged values: {unique_values}\n"
        for val, count in counts.items():
            label = "BG" if val == 0 else f"C{1 if val == 128 else 2}"
            info_text += f"  {label} ({val}): {count} voxels\n"
        
        self.info_label.setText(info_text)

    # ==================== BATCH CSV LOADING ====================

    def load_batch_csv(self):
        """Load CSV measurement results from a batch of packages.
        
        Expected structure:
            batch_root/
                PKg-...-1/out/measurement_results/*.csv
                PKg-...-2/out/measurement_results/*.csv
                ...
        """
        folder = QFileDialog.getExistingDirectory(
            self, "Select Batch Root Folder (contains PKg-* subfolders)", ""
        )
        if not folder:
            return

        import csv

        # Find package subfolders
        pkg_dirs = sorted([
            d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
        ])

        if not pkg_dirs:
            QMessageBox.warning(self, "No Packages", f"No subfolders found in:\n{folder}")
            return

        all_rows = []
        pkg_count = 0
        layer_count = 0

        progress = QProgressDialog("Loading batch CSV results...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        for idx, pkg_name in enumerate(pkg_dirs):
            progress.setLabelText(f"[{idx+1}/{len(pkg_dirs)}] {pkg_name}")
            progress.setValue(int(idx / len(pkg_dirs) * 95))
            QApplication.processEvents()

            # Try multiple possible paths
            mr_path = os.path.join(folder, pkg_name, "out", "measurement_results")
            if not os.path.isdir(mr_path):
                mr_path = os.path.join(folder, pkg_name, "measurement_results")
            if not os.path.isdir(mr_path):
                continue

            # Read all CSV files in this package's measurement_results
            csv_files = sorted(glob.glob(os.path.join(mr_path, "*.csv")))
            if not csv_files:
                continue

            pkg_count += 1

            # Prefer per-layer CSVs (skip summary)
            per_layer_csvs = [f for f in csv_files if "summary" not in os.path.basename(f).lower()]
            if not per_layer_csvs:
                per_layer_csvs = csv_files  # Fallback to all

            for csv_path in per_layer_csvs:
                csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
                try:
                    with open(csv_path, 'r', newline='') as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        if header is None:
                            continue
                        for row_data in reader:
                            all_rows.append([pkg_name, csv_basename] + row_data)
                    layer_count += 1
                except Exception as e:
                    print(f"Warning: could not read {csv_path}: {e}")

        progress.close()

        if not all_rows:
            QMessageBox.warning(self, "No Data", "No CSV data found in any package.")
            return

        # Store for export
        self._batch_csv_data = all_rows

        # Determine header from first CSV
        sample_mr = None
        for pkg_name in pkg_dirs:
            mr_path = os.path.join(folder, pkg_name, "out", "measurement_results")
            if not os.path.isdir(mr_path):
                mr_path = os.path.join(folder, pkg_name, "measurement_results")
            csv_files = glob.glob(os.path.join(mr_path, "*.csv")) if os.path.isdir(mr_path) else []
            if csv_files:
                try:
                    with open(csv_files[0], 'r') as f:
                        sample_mr = next(csv.reader(f), None)
                except:
                    pass
                break

        # Build table header: Package, File, then CSV columns
        if sample_mr:
            table_header = ["Package", "File"] + sample_mr
        else:
            table_header = ["Package", "File"] + [f"Col{i}" for i in range(len(all_rows[0]) - 2)]

        self._batch_csv_header = table_header

        # Populate stats table
        self.stats_table.setSortingEnabled(False)
        self.stats_table.blockSignals(True)
        self.stats_table.setColumnCount(len(table_header))
        self.stats_table.setHorizontalHeaderLabels(table_header)
        self.stats_table.setRowCount(len(all_rows))

        for row_idx, row_data in enumerate(all_rows):
            for col_idx, cell in enumerate(row_data):
                if col_idx < len(table_header):
                    item = QTableWidgetItem(str(cell).strip())
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    # Color Judgment column
                    if col_idx < len(table_header) and table_header[col_idx] == "Judgment":
                        if cell.strip() == "NG":
                            item.setForeground(QColor(255, 80, 80))
                            item.setBackground(QColor(180, 40, 40, 60))
                        elif cell.strip() == "OK":
                            item.setForeground(QColor(80, 255, 100))
                            item.setBackground(QColor(40, 180, 60, 40))
                    self.stats_table.setItem(row_idx, col_idx, item)

        self.stats_table.blockSignals(False)
        self.stats_table.setSortingEnabled(True)
        self.stats_table.resizeColumnsToContents()

        self.batch_info_label.setText(
            f"Batch loaded: {pkg_count} packages, {layer_count} layer CSVs, {len(all_rows)} total objects"
        )
        self.export_batch_btn.setEnabled(True)

        QMessageBox.information(self, "Batch Load Complete",
            f"Loaded {len(all_rows)} objects from {pkg_count} packages.\n"
            f"Layer CSVs read: {layer_count}")

    def export_batch_csv(self):
        """Export the combined batch CSV data to a single file."""
        if not hasattr(self, '_batch_csv_data') or not self._batch_csv_data:
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Combined Batch CSV", "batch_combined_results.csv",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if not output_path:
            return

        import csv
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                if hasattr(self, '_batch_csv_header'):
                    writer.writerow(self._batch_csv_header)
                for row in self._batch_csv_data:
                    writer.writerow(row)
            QMessageBox.information(self, "Export Success", 
                f"Combined batch CSV exported to:\n{output_path}\n"
                f"{len(self._batch_csv_data)} rows written.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{str(e)}")
