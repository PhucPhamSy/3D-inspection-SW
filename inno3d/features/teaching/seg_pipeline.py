# inno3d/features/teaching/seg_pipeline.py
# -----------------------------------------------------------------------
# SegmentationPipelineMixin — extracted from inno3d/tabs/teaching.py (4.3)
#
# Pipeline: load_dll_folder, load_config_file, load_volume,
# run_inspection, run_enhancement, run_measurement, run_object_analysis,
# export_stats_csv, set_paths_and_run (Online API), delete_objects,
# run_distance_transform, run_boundary_analysis, run_false_alarm_remover.
# -----------------------------------------------------------------------

import csv
import ctypes
import glob
import os
import re
import traceback
from pathlib import Path
from pathlib import Path as _Path
from typing import Optional

import numpy as np
import vtk
from scipy import ndimage
from skimage import io, measure
from vtk.util import numpy_support

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint, QItemSelection, QItemSelectionModel
from PyQt5.QtGui import QColor, QFont, QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QProgressDialog,
    QRadioButton, QShortcut, QTableWidgetItem, QVBoxLayout, QWidget,
)

from inno3d.core import bumpvoid
from inno3d.core.resources import config_dir, list_config_files
from inno3d.core.styles import SemiconductorTheme
from inno3d.core.view_support import LoadVolumeThread, RawImportDialog
from inno3d.features.shared.mes_mapping import (
    MesPickDebounce,
    centroid_nav_xyz as _shared_centroid_nav_xyz,
    highlights_from_stats_indices as _shared_highlights_from_stats_indices,
    label_at_volume_xyz as _shared_label_at_volume_xyz,
    resolve_stat_label as _shared_resolve_stat_label,
    stats_index_for_label as _shared_stats_index_for_label,
)
from inno3d.features.shared.mes_mpr_highlight import (
    blend_mes_selection_highlight as _shared_blend_mes_hl,
    build_mes_selection_highlight_rgb as _shared_build_mes_hl_rgb,
    labeled_slice_for_orientation as _shared_labeled_slice,
)
from inno3d.features.teaching.workers import (
    SegmentationInspectionThread, EnhancementThread,
    DistanceTransformThread, BoundaryAnalysisThread,
    _ExportLayerVolumesThread, NoScrollSpinBox,
)
from inno3d.infra.textio import read_text_auto


class SegmentationPipelineMixin:
    """Mixin: SegmentationTab pipeline + IO + online API methods.

    Extracted from inno3d/tabs/teaching.py Phase 4.3.
    """

    def register_config_recipe_combo(self, combo: QComboBox):
        """Track a recipe dropdown (toolbar and/or sidebar) and wire user pick."""
        if combo is None:
            return
        combos = getattr(self, "_config_recipe_combos", None)
        if combos is None:
            self._config_recipe_combos = []
            combos = self._config_recipe_combos
        if combo not in combos:
            combos.append(combo)
            combo.setToolTip(
                "Inspection recipes from the app config/ folder.\n"
                "Browse… for a recipe outside that folder."
            )
            combo.activated[int].connect(self._on_config_recipe_activated)
        # Primary reference used by older code paths
        self.config_recipe_combo = combo
        # Keep legacy attribute name pointing at the live combo when possible
        self.config_path_input = combo

    def refresh_config_recipe_list(self, select_path=None):
        """Rebuild recipe dropdown(s) from config/ next to the app (post-build)."""
        combos = list(getattr(self, "_config_recipe_combos", []) or [])
        primary = getattr(self, "config_recipe_combo", None)
        if primary is not None and primary not in combos:
            combos.append(primary)
        if not combos:
            return

        if select_path is None:
            select_path = getattr(self, "config_path", None) or ""
        select_abs = ""
        if select_path:
            try:
                select_abs = os.path.normcase(os.path.abspath(select_path))
            except OSError:
                select_abs = ""

        recipes = list_config_files()
        for combo in combos:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("— Select recipe —", "")
            seen = set()
            for p in recipes:
                try:
                    full = str(p.resolve())
                    key = os.path.normcase(full)
                except OSError:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                combo.addItem(p.name, full)

            # External path (Browse) not under config/
            if select_abs and os.path.isfile(select_path):
                try:
                    full = os.path.abspath(select_path)
                    key = os.path.normcase(full)
                except OSError:
                    full, key = select_path, select_abs
                if key not in seen:
                    combo.addItem(f"{os.path.basename(full)} (external)", full)
                    seen.add(key)

            idx = 0
            if select_abs:
                for i in range(combo.count()):
                    data = combo.itemData(i) or ""
                    if not data:
                        continue
                    try:
                        if os.path.normcase(os.path.abspath(str(data))) == select_abs:
                            idx = i
                            break
                    except OSError:
                        continue
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def sync_config_path_ui(self, path=None):
        """Sync recipe combo selection after load/save/clear."""
        if path is None:
            path = getattr(self, "config_path", None) or ""
        self.refresh_config_recipe_list(select_path=path or None)
        # Legacy QLineEdit fallback (should not remain after combo migration)
        w = getattr(self, "config_path_input", None)
        if w is not None and not isinstance(w, QComboBox) and hasattr(w, "setText"):
            w.setText(path or "")

    def _on_config_recipe_activated(self, index: int):
        """User picked a recipe from the dropdown."""
        combo = self.sender()
        if not isinstance(combo, QComboBox):
            combo = getattr(self, "config_recipe_combo", None)
        if combo is None or index < 0:
            return
        path = combo.itemData(index) or ""
        if not path:
            return
        try:
            cur = os.path.normcase(os.path.abspath(self.config_path)) if self.config_path else ""
            nxt = os.path.normcase(os.path.abspath(str(path)))
            if cur and cur == nxt:
                return
        except OSError:
            pass
        self.load_config_from_path(str(path))

    def load_dll_folder(self):
        """Browse and load a native DLL package folder (V2 / V3 / custom)."""
        start = ""
        if getattr(self, "dll_path", None):
            start = self.dll_path
        else:
            try:
                from inno3d.infra.paths import default_dll_dir, project_root

                start = default_dll_dir() or str(project_root().parent)
            except Exception:
                start = ""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select DLL Folder (V2 / V3 — must contain BumpVoidSeg.dll)",
            start,
        )
        if folder:
            try:
                self.apply_dll_folder(folder, persist=True, force=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load DLL: {str(e)}")

    def _sync_dll_path_displays(self, folder: str) -> None:
        """Keep Teaching toolbar + sidebar DLL path fields in sync."""
        text = folder or ""
        for attr in ("dll_path_input", "_toolbar_dll_path_input"):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.setText(text)
                except Exception:
                    pass

    def apply_dll_folder(self, folder, persist: bool = True, force: bool = True) -> str:
        """Load SEG (+ companion MES/B2B/ENH) from ``folder`` and bind for Online.

        Returns the loaded SEG version string when available.
        """
        import os

        folder = os.path.abspath(folder)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"DLL folder not found: {folder}")

        bumpvoid.load_dll(folder, force=force)
        self.dll_path = folder
        self._sync_dll_path_displays(folder)

        # Companion modules from the same package (Online uses Teaching folder)
        try:
            from inno3d.core import bumpvoid_mes

            bumpvoid_mes.load_dll(folder, force=True)
        except Exception as e:
            print(f"[DLL] MES not loaded from {folder}: {e}")
        try:
            from inno3d.core import bumpvoid_b2b

            bumpvoid_b2b.load_dll(folder, force=True)
        except Exception as e:
            print(f"[DLL] B2B not loaded from {folder}: {e}")
        try:
            from inno3d.core import enhanced_volume

            enhanced_volume.load_dll(folder)
        except Exception as e:
            print(f"[DLL] ENH not loaded from {folder}: {e}")

        if persist:
            try:
                from inno3d.app.settings import set_dll_dir

                set_dll_dir(folder)
            except Exception as e:
                print(f"[DLL] Failed to persist DLL dir: {e}")

        ver = bumpvoid.get_version() or "?"
        if hasattr(self, "info_label") and self.info_label is not None:
            self.info_label.setText(f"DLL: {folder}  ·  SEG v{ver}")
        self.check_ready_state()
        print(f"[DLL] Active package → {folder} (SEG v{ver})")
        return ver

    def load_config_file(self, file_path=None):
        """Load configuration file"""
        if not file_path:
            start_dir = ""
            try:
                start_dir = str(config_dir())
            except Exception:
                start_dir = ""
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Config File",
                start_dir,
                "Config Files (*.txt);;All Files (*.*)",
            )
        if file_path:
            try:
                self.config = bumpvoid.load_config(file_path)
                self.config_path = file_path
                self.sync_config_path_ui(file_path)
                self.populate_parameters_from_config()
                
                # Apply pending input path if volume was loaded before config
                if self._pending_input_path:
                    self.config.inputPath = self._pending_input_path.encode('utf-8')
                    print(f"Applied pending inputPath: {self._pending_input_path}")
                    self.populate_parameters_from_config()
                
                # Manually parse voxel / MES / layer text — UTF-8 first (not locale cp949)
                try:
                    content = read_text_auto(file_path)
                    mode_match = re.search(r'INPUT_MODE\s*=\s*([a-zA-Z0-9_]+)', content)
                    if mode_match:
                        mode_val = mode_match.group(1).strip()
                        if mode_val == 'test_1layer':
                            self.mode_test1_radio.setChecked(True)
                        elif mode_val == 'single':
                            self.mode_single_radio.setChecked(True)
                        elif mode_val == 'multi_layer':
                            self.mode_multi_radio.setChecked(True)

                    vx_match = re.search(r'VOXEL_SIZE_X\s*=\s*([\d.]+)', content)
                    vy_match = re.search(r'VOXEL_SIZE_Y\s*=\s*([\d.]+)', content)
                    vz_match = re.search(r'VOXEL_SIZE_Z\s*=\s*([\d.]+)', content)

                    bumpmin_match = re.search(r'BUMP_MINIMUM_SIZE\s*=\s*([0-9.xX]+)', content)
                    bumpmax_match = re.search(r'BUMP_MAXIMUM_SIZE\s*=\s*([0-9.xX]+)', content)
                    voidmin_match = re.search(r'VOID_MINIMUM_SIZE\s*=\s*([0-9.xX]+)', content)
                    voidmax_match = re.search(r'VOID_MAXIMUM_SIZE\s*=\s*([0-9.xX]+)', content)

                    if vx_match:
                        self.param_widgets['voxel_size_x'].setValue(float(vx_match.group(1)))
                    if vy_match:
                        self.param_widgets['voxel_size_y'].setValue(float(vy_match.group(1)))
                    if vz_match:
                        self.param_widgets['voxel_size_z'].setValue(float(vz_match.group(1)))

                    if bumpmin_match:
                        self.param_widgets['bump_minimum_size'].setText(bumpmin_match.group(1))
                    if bumpmax_match:
                        self.param_widgets['bump_maximum_size'].setText(bumpmax_match.group(1))
                    if voidmin_match:
                        self.param_widgets['void_minimum_size'].setText(voidmin_match.group(1))
                    if voidmax_match:
                        self.param_widgets['void_maximum_size'].setText(voidmax_match.group(1))

                    prof_match = re.search(
                        r'ENABLE_DLL_PROFILING\s*=\s*(true|false)', content, re.IGNORECASE
                    )
                    if prof_match:
                        self._apply_profiling_flag_from_config_text(content)

                    z4x_match = re.search(r'Z_STRETCHED_4X\s*=\s*(true|false)', content, re.IGNORECASE)
                    if z4x_match and 'z_stretched_4x' in self.param_widgets:
                        self.param_widgets['z_stretched_4x'].setChecked(z4x_match.group(1).lower() == 'true')

                    ng_match = re.search(r'NG_THRESHOLD\s*=\s*([\d.]+)', content)
                    if ng_match and hasattr(self, 'ng_threshold_spin'):
                        self.ng_threshold_spin.setValue(float(ng_match.group(1)))

                    cc3d_conn_match = re.search(r'CC3D_CONNECTIVITY\s*=\s*(\d+)', content)
                    if cc3d_conn_match and 'cc3d_connectivity' in self.param_widgets:
                        conn_val = int(cc3d_conn_match.group(1))
                        conn_map = {6: 0, 18: 1, 26: 2}
                        self.param_widgets['cc3d_connectivity'].setCurrentIndex(conn_map.get(conn_val, 2))

                    cc3d_minvox_match = re.search(r'CC3D_MIN_VOXELS\s*=\s*(\d+)', content)
                    if cc3d_minvox_match and 'cc3d_min_voxels' in self.param_widgets:
                        self.param_widgets['cc3d_min_voxels'].setValue(int(cc3d_minvox_match.group(1)))

                    cc3d_gpu_match = re.search(r'CC3D_USE_GPU\s*=\s*(true|false)', content, re.IGNORECASE)
                    if cc3d_gpu_match and 'cc3d_use_gpu' in self.param_widgets:
                        self.param_widgets['cc3d_use_gpu'].setChecked(cc3d_gpu_match.group(1).lower() == 'true')

                    self._apply_enhancement_from_config_text(content)
                except Exception as e:
                    print(f"Note: Could not parse Measurement Parameters from config: {e}")

                # Parse Layer Definitions from config if present
                try:
                    content = read_text_auto(file_path)
                    num_layers_match = re.search(r'NUM_LAYERS\s*=\s*(\d+)', content)
                    if num_layers_match:
                        num_layers = int(num_layers_match.group(1))
                        loaded_layers = []
                        for i in range(num_layers):
                            layer_match = re.search(rf'LAYER_{i}\s*=\s*(.+)', content)
                            if layer_match:
                                parts = layer_match.group(1).strip().split(',')
                                if len(parts) >= 3:
                                    name = parts[0].strip()
                                    z_start = int(parts[1].strip())
                                    z_end = int(parts[2].strip())
                                    selected = (
                                        parts[3].strip().lower() == 'true'
                                        if len(parts) >= 4
                                        else True
                                    )
                                    loaded_layers.append({
                                        'id': i,
                                        'name': name,
                                        'z_start': z_start,
                                        'z_end': z_end,
                                        'selected': selected,
                                    })

                        if loaded_layers:
                            self.layer_definitions = loaded_layers
                            self.layers_active = True
                            self._populate_layer_table()
                            self._populate_layer_filter()
                            print(f"Loaded {len(loaded_layers)} layer definitions from config")
                except Exception as e:
                    print(f"Note: Could not parse Layer Definitions from config: {e}")

                self.info_label.setText("Config loaded successfully")
                self.check_ready_state()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load config: {str(e)}")
    
    def browse_output_path(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_path_input.setText(folder)
            # Update config if loaded
            if self.config:
                self.config.outputDir = folder.encode('utf-8')
                
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self._handle_volume_load(file_path)

    def load_volume(self, from_folder=False):
        """Load 16-bit volume data.

        Opens a unified dialog that lets the user select either a single
        3D file or a folder of stack images.
        """
        dlg = QFileDialog(self, "Open File or Folder")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("Image Files (*.tif *.tiff *.raw *.bin)")
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.ShowDirsOnly, False)
        from PyQt5.QtWidgets import QTreeView, QListView, QAbstractItemView
        for view in dlg.findChildren((QTreeView, QListView)):
            if isinstance(view, (QTreeView, QListView)):
                view.setSelectionMode(QAbstractItemView.SingleSelection)

        _orig_accept = dlg.accept

        def _custom_accept():
            selected = dlg.selectedFiles()
            if selected and os.path.isdir(selected[0]):
                dlg.done(QFileDialog.Accepted)
                return
            _orig_accept()

        dlg.accept = _custom_accept

        if dlg.exec_() == QFileDialog.Accepted:
            selected = dlg.selectedFiles()
            if selected:
                self._handle_volume_load(selected[0])
            
    def _handle_volume_load(self, file_path):
        from pathlib import Path
        path = Path(file_path)
        raw_shape = None
        raw_dtype = None
        raw_offset = 0
        
        if path.suffix.lower() in ['.raw', '.bin']:
            from inno3d.core.view_support import RawImportDialog
            dlg = RawImportDialog(path.name, self)
            if dlg.exec_():
                raw_shape, raw_dtype, raw_offset = dlg.get_data()
            else:
                return # user cancelled
                
        from inno3d.core.view_support import LoadVolumeThread
        self.pending_load_path = file_path
        self.load_thread = LoadVolumeThread(file_path, downsample_factor=1, raw_shape=raw_shape, raw_dtype=raw_dtype, raw_offset=raw_offset)
        
        self.progress = QProgressDialog("Loading volume...", "Cancel", 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.show()
        
        self.load_thread.progress.connect(lambda v, m: (self.progress.setValue(v), self.progress.setLabelText(m)))
        self.load_thread.finished.connect(self.on_volume_loaded)
        self.load_thread.start()

    def on_volume_loaded(self, data, error):
        if hasattr(self, 'progress') and self.progress:
            self.progress.close()
            
        if error:
            error_msg = f"Failed to load volume: {error}"
            print(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Error", error_msg)
            return
            
        file_path = getattr(self, 'pending_load_path', '')
        try:
            # Clear previous segmentations and stats before setting new data
            self.clear_masks()
            
            self.volume_data = data
            self._3d_needs_initial_camera = True  # Reset camera for new data
            z, y, x = data.shape
            print(f"Final volume shape: {z}x{y}x{x}")
            
            # Update sliders
            self.current_slices = {'axial': z//2, 'coronal': y//2, 'sagittal': x//2}
            
            self.axial_slice_slider.setMaximum(z - 1)
            self.axial_slice_slider.setValue(z // 2)
            self.axial_slice_label.setText(f"{z//2} / {z-1}")
            
            self.coronal_slice_slider.setMaximum(y - 1)
            self.coronal_slice_slider.setValue(y // 2)
            self.coronal_slice_label.setText(f"{y//2} / {y-1}")
            
            self.sagittal_slice_slider.setMaximum(x - 1)
            self.sagittal_slice_slider.setValue(x // 2)
            self.sagittal_slice_label.setText(f"{x//2} / {x-1}")
            
            # Update input path in UI
            if file_path:
                self.input_path_input.setText(file_path)
                
                # Update config with input path (if config exists)
                if self.config:
                    self.config.inputPath = file_path.encode('utf-8')
                    print(f"Updated config.inputPath: {file_path}")
                else:
                    print("Warning: No config loaded yet, inputPath will be set when config is loaded")
                    # Store path to apply when config is loaded
                    self._pending_input_path = file_path
            
            # Initialize ROI Z-range to cover the full volume by default
            if hasattr(self, 'roi_z_start_spin') and hasattr(self, 'roi_z_end_spin'):
                self.roi_z_start_spin.setRange(0, z - 1)
                self.roi_z_start_spin.setValue(0)
                self.roi_z_end_spin.setRange(0, z)
                self.roi_z_end_spin.setValue(z)
            
            # Calculate window/level for proper 16-bit display
            import numpy as np
            if data.dtype == np.uint16:
                # Use percentile-based window/level like Multi-planar tab
                data_min = np.percentile(data, 1)
                data_max = np.percentile(data, 99)
                window = data_max - data_min
                level = (data_max + data_min) / 2
                print(f"Auto window/level: window={window:.1f}, level={level:.1f}")
                
                # Apply to all orientations
                for orientation in ['axial', 'coronal', 'sagittal']:
                    self.window_level[orientation] = (window, level)
            else:
                # For 8-bit, use full range
                for orientation in ['axial', 'coronal', 'sagittal']:
                    self.window_level[orientation] = (255, 127.5)
            
            # Ensure VTK interactors + wheel/crosshair observers are live
            # (also called from MainWindow.showEvent; safe to re-run)
            try:
                self.init_vtk_widgets()
            except Exception as e:
                print(f"[Teaching] init_vtk_widgets on load: {e}")

            # Render views
            print("Rendering planes...")
            for orientation in ['axial', 'coronal', 'sagittal']:
                try:
                    print(f"  Rendering {orientation}...")
                    self.update_plane_view(orientation)
                    print(f"  {orientation} rendered successfully")
                except Exception as e:
                    print(f"  ERROR rendering {orientation}: {e}")
                    import traceback
                    traceback.print_exc()

            # Center crosshair for navigation parity with Viewer
            try:
                self.crosshair_position = [x // 2, y // 2, z // 2]
                if getattr(self, 'crosshair_enabled', False):
                    self.updatePoint(x // 2, y // 2, z // 2)
            except Exception:
                pass

            # ── Phase 4a: Adaptive performance tuning for Teaching tab ───
            data_gb = data.nbytes / (1024 ** 3)
            if data_gb >= 4.0:
                throttle_label = '100ms' if data_gb >= 8 else '33ms'
                quality_hint = 'Draft' if data_gb >= 8 else 'Standard'
                # Teaching 3D quality presets
                if hasattr(self, '_3d_quality_presets') and hasattr(self, '_current_3d_quality'):
                    if quality_hint in self._3d_quality_presets:
                        self._current_3d_quality = quality_hint
                        if hasattr(self, '_3d_quality_combo'):
                            idx = self._3d_quality_combo.findText(quality_hint)
                            if idx >= 0:
                                self._3d_quality_combo.blockSignals(True)
                                self._3d_quality_combo.setCurrentIndex(idx)
                                self._3d_quality_combo.blockSignals(False)
                print(f"[PERF] Teaching volume {data_gb:.1f} GB → 3D crosshair "
                      f"debounce={throttle_label}, quality hint={quality_hint}")

            self.info_label.setText(f"Volume loaded: {z}x{y}x{x}")
            print("Volume loading complete!")
            
            # Update ROI Z range spinboxes to match volume depth
            if hasattr(self, 'roi_z_start_spin'):
                self.roi_z_start_spin.setRange(0, z - 1)
                self.roi_z_start_spin.setValue(0)
            if hasattr(self, 'roi_z_end_spin'):
                self.roi_z_end_spin.setRange(0, z)
                self.roi_z_end_spin.setValue(z)
            
            self.check_ready_state()
            
        except Exception as e:
            error_msg = f"Failed to load volume: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", error_msg)

    def populate_parameters_from_config(self):
        """Populate UI widgets from loaded config"""
        if not self.config:
            return
            
        # Block signals to avoid triggering parameter changed
        for widget in self.param_widgets.values():
            widget.blockSignals(True)
        
        # Bump parameters
        self.param_widgets['bumpThresholdWeight'].setValue(self.config.bumpThresholdWeight)
        self.param_widgets['bumpCleanRadiusX'].setValue(self.config.bumpCleanOpenX)
        self.param_widgets['bumpCleanRadiusY'].setValue(self.config.bumpCleanOpenY)
        self.param_widgets['bumpCleanRadiusZ'].setValue(self.config.bumpCleanOpenZ)
        self.param_widgets['bumpFillHoleRadiusX'].setValue(self.config.bumpBgOpenX)
        self.param_widgets['bumpFillHoleRadiusY'].setValue(self.config.bumpBgOpenY)
        self.param_widgets['bumpFillHoleRadiusZ'].setValue(self.config.bumpBgOpenZ)
        if 'bumpFillHoleCloseX' in self.param_widgets:
            self.param_widgets['bumpFillHoleCloseX'].setValue(
                getattr(self.config, 'bumpFillHoleCloseX', 5.0)
            )
            self.param_widgets['bumpFillHoleCloseY'].setValue(
                getattr(self.config, 'bumpFillHoleCloseY', 5.0)
            )
            self.param_widgets['bumpFillHoleCloseZ'].setValue(
                getattr(self.config, 'bumpFillHoleCloseZ', 5.0)
            )
        
        # Void parameters
        self.param_widgets['voidThresholdWeight'].setValue(self.config.voidThresholdWeight)
        self.param_widgets['openEmVoidX'].setValue(self.config.openEmVoidX)
        self.param_widgets['openEmVoidY'].setValue(self.config.openEmVoidY)
        self.param_widgets['openEmVoidZ'].setValue(self.config.openEmVoidZ)
        self.param_widgets['closeResidueX'].setValue(self.config.closeResidueX)
        self.param_widgets['closeResidueY'].setValue(self.config.closeResidueY)
        self.param_widgets['closeResidueZ'].setValue(self.config.closeResidueZ)
        self.param_widgets['openSmallDotsX'].setValue(self.config.openSmallDotsX)
        self.param_widgets['openSmallDotsY'].setValue(self.config.openSmallDotsY)
        self.param_widgets['openSmallDotsZ'].setValue(self.config.openSmallDotsZ)
        self.param_widgets['erodeBumpX'].setValue(self.config.erodeBumpX)
        self.param_widgets['erodeBumpY'].setValue(self.config.erodeBumpY)
        self.param_widgets['erodeBumpZ'].setValue(self.config.erodeBumpZ)
        
        # Blank slices
        self.param_widgets['blankStart1'].setValue(self.config.blankStart1)
        self.param_widgets['blankEnd1'].setValue(self.config.blankEnd1)
        self.param_widgets['blankStart2'].setValue(self.config.blankStart2)
        self.param_widgets['blankEnd2'].setValue(self.config.blankEnd2)
        
        # Options
        self.param_widgets['saveBumpIntermediate'].setChecked(bool(self.config.saveBumpIntermediate))
        self.param_widgets['saveVoidIntermediate'].setChecked(bool(self.config.saveVoidIntermediate))
        self.param_widgets['showResult'].setChecked(bool(self.config.showResult))
        if 'enableDllProfiling' in self.param_widgets:
            enabled = bool(getattr(self, '_enable_dll_profiling', False))
            self.param_widgets['enableDllProfiling'].setChecked(enabled)
        
        # Unblock signals
        for widget in self.param_widgets.values():
            widget.blockSignals(False)
            
    def update_config_from_parameters(self):
        """Update config object from UI widgets"""
        if not self.config:
            return
            
        # Bump parameters
        self.config.bumpThresholdWeight = self.param_widgets['bumpThresholdWeight'].value()
        self.config.bumpCleanOpenX = self.param_widgets['bumpCleanRadiusX'].value()
        self.config.bumpCleanOpenY = self.param_widgets['bumpCleanRadiusY'].value()
        self.config.bumpCleanOpenZ = self.param_widgets['bumpCleanRadiusZ'].value()
        self.config.bumpBgOpenX = self.param_widgets['bumpFillHoleRadiusX'].value()
        self.config.bumpBgOpenY = self.param_widgets['bumpFillHoleRadiusY'].value()
        self.config.bumpBgOpenZ = self.param_widgets['bumpFillHoleRadiusZ'].value()
        if 'bumpFillHoleCloseX' in self.param_widgets and hasattr(
            self.config, 'bumpFillHoleCloseX'
        ):
            self.config.bumpFillHoleCloseX = self.param_widgets['bumpFillHoleCloseX'].value()
            self.config.bumpFillHoleCloseY = self.param_widgets['bumpFillHoleCloseY'].value()
            self.config.bumpFillHoleCloseZ = self.param_widgets['bumpFillHoleCloseZ'].value()
        
        # Void parameters
        self.config.voidThresholdWeight = self.param_widgets['voidThresholdWeight'].value()
        self.config.openEmVoidX = self.param_widgets['openEmVoidX'].value()
        self.config.openEmVoidY = self.param_widgets['openEmVoidY'].value()
        self.config.openEmVoidZ = self.param_widgets['openEmVoidZ'].value()
        self.config.closeResidueX = self.param_widgets['closeResidueX'].value()
        self.config.closeResidueY = self.param_widgets['closeResidueY'].value()
        self.config.closeResidueZ = self.param_widgets['closeResidueZ'].value()
        self.config.openSmallDotsX = self.param_widgets['openSmallDotsX'].value()
        self.config.openSmallDotsY = self.param_widgets['openSmallDotsY'].value()
        self.config.openSmallDotsZ = self.param_widgets['openSmallDotsZ'].value()
        self.config.erodeBumpX = self.param_widgets['erodeBumpX'].value()
        self.config.erodeBumpY = self.param_widgets['erodeBumpY'].value()
        self.config.erodeBumpZ = self.param_widgets['erodeBumpZ'].value()
        
        # Blank slices — invalid ranges wipe the whole mask; force off
        self.config.blankStart1 = self.param_widgets['blankStart1'].value()
        self.config.blankEnd1 = self.param_widgets['blankEnd1'].value()
        self.config.blankStart2 = self.param_widgets['blankStart2'].value()
        self.config.blankEnd2 = self.param_widgets['blankEnd2'].value()
        if self.config.blankEnd1 <= self.config.blankStart1:
            self.config.blankStart1 = 0
            self.config.blankEnd1 = 0
        if self.config.blankEnd2 <= self.config.blankStart2:
            self.config.blankStart2 = 0
            self.config.blankEnd2 = 0
        
        # Options
        self.config.saveBumpIntermediate = int(self.param_widgets['saveBumpIntermediate'].isChecked())
        self.config.saveVoidIntermediate = int(self.param_widgets['saveVoidIntermediate'].isChecked())
        self.config.showResult = int(self.param_widgets['showResult'].isChecked())
        if 'enableDllProfiling' in self.param_widgets:
            self._enable_dll_profiling = bool(
                self.param_widgets['enableDllProfiling'].isChecked()
            )

        try:
            print(
                f"[SEG cfg] dll={bumpvoid.get_dll_dir()} v={bumpvoid.get_version()} | "
                f"bg=({self.config.bumpBgOpenX},{self.config.bumpBgOpenY},{self.config.bumpBgOpenZ}) "
                f"clean=({self.config.bumpCleanOpenX},{self.config.bumpCleanOpenY},{self.config.bumpCleanOpenZ}) "
                f"fillClose=({getattr(self.config,'bumpFillHoleCloseX',0)},"
                f"{getattr(self.config,'bumpFillHoleCloseY',0)},"
                f"{getattr(self.config,'bumpFillHoleCloseZ',0)}) "
                f"thW={self.config.bumpThresholdWeight} "
                f"blank=({self.config.blankStart1}-{self.config.blankEnd1},"
                f"{self.config.blankStart2}-{self.config.blankEnd2})"
            )
        except Exception:
            pass

    def _apply_profiling_flag_from_config_text(self, content: str) -> None:
        """Sync ENABLE_DLL_PROFILING from config text → checkbox + internal flag."""
        prof_match = re.search(
            r"ENABLE_DLL_PROFILING\s*=\s*(true|false)", content, re.IGNORECASE
        )
        if not prof_match:
            return
        on = prof_match.group(1).lower() == "true"
        self._enable_dll_profiling = on
        if "enableDllProfiling" in getattr(self, "param_widgets", {}):
            self.param_widgets["enableDllProfiling"].setChecked(on)

    def _profiling_enabled_from_config_file(self) -> Optional[bool]:
        """Read ENABLE_DLL_PROFILING directly from loaded config path."""
        path = getattr(self, "config_path", None)
        if not path or not os.path.isfile(path):
            return None
        try:
            content = read_text_auto(path)
            m = re.search(r"ENABLE_DLL_PROFILING\s*=\s*(true|false)", content, re.IGNORECASE)
            if m:
                return m.group(1).lower() == "true"
        except Exception:
            pass
        return None

    def _is_dll_profiling_enabled(self) -> bool:
        w = getattr(self, "param_widgets", None) or {}
        if "enableDllProfiling" in w:
            if w["enableDllProfiling"].isChecked():
                return True
            # Checkbox unchecked — still honor config file when explicitly true
            cfg_on = self._profiling_enabled_from_config_file()
            if cfg_on is True:
                return True
            return False
        cfg_on = self._profiling_enabled_from_config_file()
        if cfg_on is not None:
            return cfg_on
        return bool(getattr(self, "_enable_dll_profiling", False))

    def _apply_dll_profiling_flag(self) -> None:
        """Push Params checkbox to all loaded ISP DLLs (safe if export missing)."""
        import logging
        enabled = self._is_dll_profiling_enabled()
        self._enable_dll_profiling = enabled
        applied = []
        try:
            if bumpvoid.set_profiling(enabled):
                applied.append("SEG")
        except Exception:
            pass
        try:
            from inno3d.core import bumpvoid_mes
            if bumpvoid_mes.set_profiling(enabled):
                applied.append("MES")
        except Exception:
            pass
        try:
            from inno3d.core import bumpvoid_b2b
            if bumpvoid_b2b.set_profiling(enabled):
                applied.append("B2B")
        except Exception:
            pass
        try:
            from inno3d.core import enhanced_volume
            if enhanced_volume.set_profiling(enabled):
                applied.append("ENH")
        except Exception:
            pass
        log = logging.getLogger("DLL_PROF")
        if enabled:
            msg = (
                f"[DLL Profiling] ENABLED → {', '.join(applied) or '(no DLL export yet; rebuild ISP DLLs)'}"
            )
        else:
            msg = "[DLL Profiling] disabled"
        log.info(msg)
        print(msg)

    def _profiling_gpu_device_id(self) -> int:
        spin = getattr(self, "enh_gpuid_spin", None)
        if spin is not None:
            try:
                return int(spin.value())
            except Exception:
                pass
        return 0

    def _profiling_voxel_spacing(self):
        """Best-effort (Z, Y, X) spacing in µm for profiling header."""
        for attr in ("custom_spacing", "voxel_spacing", "spacing"):
            sp = getattr(self, attr, None)
            if sp is not None:
                try:
                    vals = [float(x) for x in sp[:3]]
                    if len(vals) == 3:
                        return vals
                except Exception:
                    pass
        widgets = getattr(self, "param_widgets", None) or {}
        keys = [
            ("voxel_size_z", "voxel_size_y", "voxel_size_x"),
            ("VoxelSizeZ", "VoxelSizeY", "VoxelSizeX"),
        ]
        for kz, ky, kx in keys:
            wz, wy, wx = widgets.get(kz), widgets.get(ky), widgets.get(kx)
            if wz is not None and wy is not None and wx is not None:
                try:
                    return [float(wz.value()), float(wy.value()), float(wx.value())]
                except Exception:
                    pass
        return None

    def _get_profiling_report_builder(self):
        from inno3d.core.dll_profiling import ProfilingReportBuilder
        if not hasattr(self, "_profiling_report_builder"):
            self._profiling_report_builder = ProfilingReportBuilder()
        return self._profiling_report_builder

    def _reset_profiling_report(self, header_lines=None) -> None:
        from inno3d.core.dll_profiling import build_profiling_context_lines

        builder = self._get_profiling_report_builder()
        builder.reset()
        if not header_lines and not self._is_dll_profiling_enabled():
            return
        extra = list(header_lines or [])
        if not extra:
            if getattr(self, "config_path", None):
                extra.append(f"Config: {self.config_path}")
            out = getattr(self, "output_path_input", None)
            if out is not None:
                p = out.text().strip()
                if p:
                    extra.append(f"Output: {p}")
            inp = getattr(self, "input_path_input", None)
            if inp is not None:
                p = inp.text().strip()
                if p:
                    extra.append(f"Input: {p}")
        gpu_id = self._profiling_gpu_device_id()
        ctx = build_profiling_context_lines(
            volume=getattr(self, "volume_data", None),
            spacing=self._profiling_voxel_spacing(),
            gpu_device_id=gpu_id,
            extra_lines=extra or None,
        )
        builder.set_context(ctx, gpu_device_id=gpu_id)

    def _record_profiling_section(self, module: str, info, note=None) -> str:
        from inno3d.core.dll_profiling import format_timing_banner
        banner = format_timing_banner(module, info)
        if not hasattr(self, "_last_dll_timing_reports"):
            self._last_dll_timing_reports = {}
        if info and info.get("report"):
            self._last_dll_timing_reports[module] = info["report"]
        else:
            self._last_dll_timing_reports[module] = banner
        if self._is_dll_profiling_enabled():
            self._get_profiling_report_builder().add_section(module, banner, note=note)
        return banner

    def _flush_profiling_report(self, output_dir=None, footer_lines=None):
        if not self._is_dll_profiling_enabled():
            return None
        if not output_dir:
            out = getattr(self, "output_path_input", None)
            if out is not None:
                output_dir = out.text().strip()
        if not output_dir:
            return None
        builder = self._get_profiling_report_builder()
        if footer_lines:
            builder.add_footer(footer_lines)
        builder.set_runtime_snapshot()
        path = builder.write(output_dir)
        if path:
            import logging
            msg = f"[DLL Profiling] Report saved → {path}"
            logging.getLogger("DLL_PROF").info(msg)
            print(msg)
        return path

    def _log_dll_timing(self, module: str, getter, note=None, flush_dir=None) -> None:
        import logging
        log = logging.getLogger("DLL_PROF")
        if not self._is_dll_profiling_enabled():
            log.info("[%s] Profiling checkbox OFF — skip timing report", module)
            return
        try:
            info = getter()
            banner = self._record_profiling_section(module, info, note=note)
            for line in banner.splitlines():
                log.info(line)
            print(banner)
            if not info:
                log.warning(
                    "[%s] Profiling ON but GetLastTiming empty "
                    "(DLL may be old, or process did not run yet)",
                    module,
                )
            if flush_dir:
                self._flush_profiling_report(flush_dir)
        except Exception as e:
            log.error("[%s] Profiling read failed: %s", module, e)
            print(f"[{module}] Profiling read failed: {e}")

    def on_parameter_changed(self):
        """Handle parameter change - trigger debounced preview"""
        if hasattr(self, 'warning_label'):
             self.warning_label.show()
             
        self.update_config_from_parameters()

        # Keep MPR scale bars in sync when voxel sizes change
        if self.volume_data is not None:
            z, y, x = self.volume_data.shape
            shapes = {
                "axial": (y, x),
                "coronal": (z, x),
                "sagittal": (z, y),
            }
            for o, shape in shapes.items():
                renderer = getattr(self, f"{o}_renderer", None)
                widget = getattr(self, f"{o}_widget", None)
                if renderer is not None and widget is not None:
                    self.add_ruler_overlay(renderer, o, shape)
                    widget.GetRenderWindow().Render()
        
        if hasattr(self, 'realtime_check') and self.realtime_check.isChecked() and self.volume_data is not None:
            # Debounce: restart timer (500ms delay)
            if hasattr(self, 'preview_timer'):
                self.preview_timer.stop()
                self.preview_timer.start(500)
            
    def run_preview_segmentation(self):
        """Run segmentation preview on visible slices"""
        if not self.config or self.volume_data is None:
            return
            
        # Cancel any ongoing preview
        if self.preview_thread and self.preview_thread.isRunning():
            self.preview_thread.cancel()
            self.preview_thread.wait()
        
        # Note: Full preview implementation would require DLL support for slice-based processing
        # For now, we'll just update the info label
        self.info_label.setText("Preview: Parameters updated (full preview requires inspection run)")
        
    def run_inspection(self):
        """Run full inspection (Bump + Void), optionally preceded by ONNX volume enhancement"""
        if not self.config:
            QMessageBox.warning(self, "Warning", "Please load config first")
            return

        # Check output path
        output_path = self.output_path_input.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Warning", "Please specify output path")
            return

        self.update_config_from_parameters()
        self._apply_dll_profiling_flag()
        self._reset_profiling_report()

        # If enhancement is enabled, run the preprocessing first
        if self.enhancement_enabled:
            self._run_enhancement_pipeline_step()
        else:
            self._run_segmentation_inspection_only()

    def _run_enhancement_pipeline_step(self):
        """Pipeline step 1: Run volume enhancement on GPU/TensorRT."""
        model_path = self.enh_model_input.text().strip()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Warning", "Please select a valid ONNX model file.")
            return

        if not self.dll_path:
            QMessageBox.warning(self, "Warning", "Please load the DLL folder first.")
            return

        input_path = self.input_path_input.text().strip()
        if not input_path:
            QMessageBox.warning(self, "Warning", "Please load a volume first.")
            return

        output_path = self.output_path_input.text().strip()

        # Determine input directory containing the raw TIFF slices
        self._enhancement_temp_dir = None
        if os.path.isdir(input_path):
            input_dir = input_path
        else:
            # It is a multi-page TIFF file. Extract slices to a temp directory
            import tempfile
            import tifffile
            self._enhancement_temp_dir = tempfile.mkdtemp(prefix="inno3d_enh_input_")
            input_dir = self._enhancement_temp_dir
            print(f"Extracting slices from memory to temp dir: {input_dir}")
            try:
                num_slices = self.volume_data.shape[0]
                for idx in range(num_slices):
                    slice_file = os.path.join(input_dir, f"{idx:04d}.tif")
                    tifffile.imwrite(slice_file, self.volume_data[idx])
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to extract slices for enhancement:\n{e}")
                return

        enhanced_output = os.path.join(output_path, "enhanced_volume")
        os.makedirs(enhanced_output, exist_ok=True)

        progress = QProgressDialog("Pipeline Step 1/2: Running Volume Enhancement...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        start_slice, end_slice = self.get_enhancement_slice_range()

        self.enhancement_thread = EnhancementThread(
            dll_dir=self.dll_path,
            model_path=model_path,
            trt_cache_path=self.enh_trt_input.text().strip(),
            use_gpu=self.enh_gpu_check.isChecked(),
            gpu_device_id=self.enh_gpuid_spin.value(),
            input_dir=input_dir,
            output_dir=enhanced_output,
            start_slice=start_slice,
            end_slice=end_slice
        )
        self._apply_dll_profiling_flag()
        self.enhancement_thread.progress.connect(
            lambda v, m: (progress.setValue(v), progress.setLabelText(f"Enhancing: {m}"))
        )
        self.enhancement_thread.finished.connect(
            lambda ok, err: self._on_pipeline_enhancement_finished(ok, err, progress, enhanced_output)
        )
        self.enhancement_thread.start()

    def _on_pipeline_enhancement_finished(self, success, error, progress, enhanced_output):
        """Callback when preprocessing step finishes."""
        if progress:
            progress.close()

        try:
            from inno3d.core import enhanced_volume
            out_dir = self.output_path_input.text().strip()
            self._log_dll_timing("ENH", enhanced_volume.get_last_timing, flush_dir=out_dir)
        except Exception:
            pass

        # Clean up temp input directory if any
        if hasattr(self, '_enhancement_temp_dir') and self._enhancement_temp_dir:
            try:
                import shutil
                shutil.rmtree(self._enhancement_temp_dir)
                print(f"Cleaned up enhancement temp input dir: {self._enhancement_temp_dir}")
            except Exception as e:
                print(f"Error cleaning up temp dir: {e}")
            self._enhancement_temp_dir = None

        if not success:
            QMessageBox.critical(self, "Pipeline Error", f"Volume enhancement failed, pipeline stopped:\n{error}")
            return

        try:
            import glob
            import tifffile
            enhanced_files = sorted(glob.glob(os.path.join(enhanced_output, "*.tif")) + 
                                    glob.glob(os.path.join(enhanced_output, "*.tiff")))
            if not enhanced_files:
                raise FileNotFoundError("No enhanced TIFF slices found in output folder.")

            # Backup original state
            self._original_volume_data = self.volume_data
            self._original_input_path = self.config.inputPath

            # Load enhanced slices into memory
            print(f"Loading {len(enhanced_files)} enhanced slices into memory...")
            enhanced_slices = [tifffile.imread(f) for f in enhanced_files]
            self.volume_data = np.stack(enhanced_slices)
            print(f"Enhanced volume loaded successfully. Shape: {self.volume_data.shape}")

            # Point the DLL config input path to the enhanced folder's first file
            self.config.inputPath = enhanced_files[0].encode('utf-8')

            # Proceed to Step 2: Segmentation
            self._run_segmentation_inspection_only()

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Pipeline Error", 
                                 f"Failed to load enhanced volume for segmentation:\n{str(e)}\n{traceback.format_exc()}")

    def _run_segmentation_inspection_only(self):
        """Pipeline step 2: Run segmentation inspection using the current volume data and config."""
        # ---- Multi-layer input mode ----
        if self.input_mode == 'multi_layer':
            ml_files = self._get_ordered_multi_layer_files()
            if not ml_files:
                QMessageBox.warning(self, "Warning",
                    "No layer files loaded. Please browse a sep_layer/ folder first.")
                return

            output_path = self.output_path_input.text().strip()
            self.update_config_from_parameters()
            self.config.outputDir = output_path.encode('utf-8')

            progress = QProgressDialog("Pipeline Step 2/2: Running Multi-Layer inspection...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()

            self.inspection_thread = SegmentationInspectionThread(
                self.config, 'both',
                layers=None, volume_data=None,
                multi_layer_files=ml_files
            )
            self.inspection_thread.progress.connect(lambda v, m: (progress.setValue(v), progress.setLabelText(m)))
            self.inspection_thread.finished.connect(lambda result, error: self.on_inspection_finished(result, error, progress))
            self.inspection_thread.start()
            return

        # ---- Test Volume (1 Layer) mode: entire volume, no splitting ----
        if self.input_mode == 'test_1layer':
            if self.volume_data is None:
                QMessageBox.warning(self, "Warning", "Please load volume and config first")
                return
            output_path = self.output_path_input.text().strip()
            self.layers_active = False
            self.update_config_from_parameters()
            self.config.outputDir = output_path.encode('utf-8')
            progress = QProgressDialog("Pipeline Step 2/2: Running Test Volume (1 Layer) inspection...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            # No layers, no splitting — run DLL on the full volume
            self.inspection_thread = SegmentationInspectionThread(self.config, 'both', layers=None, volume_data=self.volume_data)
            self.inspection_thread.progress.connect(lambda v, m: (progress.setValue(v), progress.setLabelText(m)))
            self.inspection_thread.finished.connect(lambda result, error: self.on_inspection_finished(result, error, progress))
            self.inspection_thread.start()
            return

        # ---- Single Volume (Multi-Layer) mode ----
        if self.volume_data is None:
            QMessageBox.warning(self, "Warning", "Please load volume and config first")
            return
        
        # Check output path
        output_path = self.output_path_input.text().strip()
            
        # Check layers
        selected_layers = None
        if self.layers_active:
            selected_layers = [l for l in self.layer_definitions if l['selected']]
            if not selected_layers:
                QMessageBox.warning(self, "Warning", "No layers selected for inspection")
                return
                
        # Update config from UI
        self.update_config_from_parameters()
        self.config.outputDir = output_path.encode('utf-8')
        
        # Create progress dialog
        progress = QProgressDialog("Pipeline Step 2/2: Running Bump + Void inspection...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        
        # Start inspection thread (always run both), passing selected layers and volume data
        self.inspection_thread = SegmentationInspectionThread(self.config, 'both', selected_layers, volume_data=self.volume_data)
        self.inspection_thread.progress.connect(lambda v, m: (progress.setValue(v), progress.setLabelText(m)))
        self.inspection_thread.finished.connect(lambda result, error: self.on_inspection_finished(result, error, progress))
        self.inspection_thread.start()
        
    def on_inspection_finished(self, result, error, progress):
        """Handle inspection completion"""
        if progress:
            progress.close()
        
        # Restore original volume data and config input path if we ran the enhancement pipeline
        if hasattr(self, '_original_volume_data') and self._original_volume_data is not None:
            self.volume_data = self._original_volume_data
            self._original_volume_data = None
        if hasattr(self, '_original_input_path') and self._original_input_path is not None:
            self.config.inputPath = self._original_input_path
            self._original_input_path = None

        if error:
            QMessageBox.critical(self, "Error", f"Inspection failed:\n{error}")
            return
            
        # result can be a single BumpVoidResult or a list of BumpVoidResult
        results_list = result if isinstance(result, list) else [result]
        valid_results = [r for r in results_list if r and getattr(r, 'success', False)]
        
        if not valid_results:
            QMessageBox.critical(self, "Error", "Inspection failed or returned no results.")
            return

        try:
            if getattr(self, 'input_mode', 'single') == 'multi_layer':
                total_time = sum([getattr(res, 'totalTime', 0.0) for res in valid_results])

                # Per-layer cache: {layer_name: {'input': ndarray, 'bump': ndarray, 'void': ndarray}}
                self._ml_layer_cache = {}
                # Combined arrays for Measurement
                bump_stack = []
                void_stack = []
                input_stack = []
                self.layer_assignment_map = {}
                self._ml_layer_z_starts = {}  # {layer_name: z_start_in_combined}
                z_cursor = 0

                ml_info_map = {mf['name']: mf['path'] for mf in self._get_ordered_multi_layer_files()}

                for res in valid_results:
                    layer_name = getattr(res, 'layer_name', None) or 'unknown'
                    b_data = v_data = i_data = None

                    if res.bumpOutputPath:
                        try:
                            b_data = io.imread(res.bumpOutputPath.decode('utf-8'))
                        except Exception as e:
                            print(f"Warning: could not read bump output for {layer_name}: {e}")
                    if res.voidOutputPath:
                        try:
                            v_data = io.imread(res.voidOutputPath.decode('utf-8'))
                        except Exception as e:
                            print(f"Warning: could not read void output for {layer_name}: {e}")
                    input_path = ml_info_map.get(layer_name)
                    if input_path:
                        try:
                            i_data = io.imread(input_path)
                        except Exception as e:
                            print(f"Warning: could not read input for {layer_name}: {e}")

                    # Ensure 3D
                    for arr_name in ['b_data', 'v_data', 'i_data']:
                        arr = locals()[arr_name]
                        if arr is not None and arr.ndim == 2:
                            locals()[arr_name] = arr[np.newaxis]
                    # Re-read after possible mutation
                    if b_data is not None and b_data.ndim == 2: b_data = b_data[np.newaxis]
                    if v_data is not None and v_data.ndim == 2: v_data = v_data[np.newaxis]
                    if i_data is not None and i_data.ndim == 2: i_data = i_data[np.newaxis]

                    # Store in cache
                    self._ml_layer_cache[layer_name] = {
                        'input': i_data, 'bump': b_data, 'void': v_data
                    }

                    layer_depth = 0
                    if b_data is not None:
                        layer_depth = b_data.shape[0]
                        bump_stack.append(b_data)
                    if v_data is not None:
                        if layer_depth == 0: layer_depth = v_data.shape[0]
                        void_stack.append(v_data)
                    if i_data is not None:
                        if layer_depth == 0: layer_depth = i_data.shape[0]
                        input_stack.append(i_data)

                    self._ml_layer_z_starts[layer_name] = z_cursor
                    for z in range(z_cursor, z_cursor + layer_depth):
                        self.layer_assignment_map[z] = layer_name
                    z_cursor += layer_depth

                # Pad helper for mismatched Y/X
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

                # Default: show first layer
                first_layer = list(self._ml_layer_cache.keys())[0] if self._ml_layer_cache else None
                if first_layer:
                    self._switch_to_ml_layer(first_layer)

                # Highlight first item in list
                if self.multi_layer_list.count() > 0:
                    self.multi_layer_list.setCurrentRow(0)

                self._populate_layer_filter()

                self.info_label.setText(f"Multi-Layer complete: {total_time:.2f}s, {len(valid_results)} layers, Z={z_cursor}")

                # Enable Measurement and FAR buttons
                if hasattr(self, 'measure_btn'):
                    self.measure_btn.setEnabled(len(valid_results) > 0)
                if hasattr(self, 'far_btn'):
                    self.far_btn.setEnabled(True)
                if hasattr(self, 'dt_btn'):
                    self.dt_btn.setEnabled(len(valid_results) > 0)

                QMessageBox.information(self, "Success", 
                    f"Multi-layer Inspection completed!\n\n"
                    f"Layers: {len(valid_results)}, Total Z: {z_cursor}\n"
                    f"Time: {total_time:.2f}s\n\n"
                    f"Click a layer in the list to view it.\n"
                    f"Click 'MEASUREMENT' when ready.")
                return


            combined_bump = np.zeros_like(self.volume_data, dtype=np.uint8) if valid_results else None
            combined_void = np.zeros_like(self.volume_data, dtype=np.uint8) if valid_results else None
            
            # Map layer assignments for stats table (same helper as Online sync)
            if self.layers_active and getattr(self, "layer_definitions", None):
                self._rebuild_layer_assignment_map([
                    {
                        "name": l["name"],
                        "z_start": int(l["z_start"]),
                        "z_end": int(l["z_end"]),
                    }
                    for l in self.layer_definitions
                    if l.get("selected", True)
                ])
            else:
                self._rebuild_layer_assignment_map()

            total_time = 0.0
            
            for res in valid_results:
                total_time += res.totalTime
                
                layer_z_start, layer_z_end = 0, self.volume_data.shape[0]
                if hasattr(res, 'layer_name') and getattr(res, 'layer_name') and self.layers_active:
                    for l in self.layer_definitions:
                        if l['name'] == res.layer_name:
                            layer_z_start = l['z_start']
                            layer_z_end = l['z_end']
                            break

                if res.bumpOutputPath:
                    bump_path = res.bumpOutputPath.decode('utf-8')
                    b_data = io.imread(bump_path)
                    if b_data.ndim == 2:
                        b_data = b_data[np.newaxis]
                    if combined_bump is not None:
                        if b_data.shape[1:] == combined_bump.shape[1:]:
                            z_len = min(b_data.shape[0], layer_z_end - layer_z_start)
                            combined_bump[layer_z_start:layer_z_start+z_len] = np.maximum(
                                combined_bump[layer_z_start:layer_z_start+z_len], b_data[:z_len]
                            )
                        elif hasattr(self, 'last_applied_roi_raw') and self.last_applied_roi_raw is not None:
                            roi = self.last_applied_roi_raw
                            z_coords = sorted([roi['z_start'], roi['z_end']])
                            y_coords = sorted([roi['y1'], roi['y2']])
                            x_coords = sorted([roi['x1'], roi['x2']])
                            z1 = int(max(0, z_coords[0]))
                            z2 = int(min(combined_bump.shape[0], z_coords[1]))
                            y1 = int(max(0, y_coords[0]))
                            y2 = int(min(combined_bump.shape[1], y_coords[1]))
                            x1 = int(max(0, x_coords[0]))
                            x2 = int(min(combined_bump.shape[2], x_coords[1]))
                            
                            target_z = z2 - z1
                            target_y = y2 - y1
                            target_x = x2 - x1
                            b_cropped = b_data[:target_z, :target_y, :target_x]
                            
                            dst_slice = (slice(z1, z1 + b_cropped.shape[0]),
                                         slice(y1, y1 + b_cropped.shape[1]),
                                         slice(x1, x1 + b_cropped.shape[2]))
                            combined_bump[dst_slice] = np.maximum(
                                combined_bump[dst_slice], b_cropped
                            )
                    
                if res.voidOutputPath:
                    void_path = res.voidOutputPath.decode('utf-8')
                    v_data = io.imread(void_path)
                    if v_data.ndim == 2:
                        v_data = v_data[np.newaxis]
                    if combined_void is not None:
                        if v_data.shape[1:] == combined_void.shape[1:]:
                            z_len = min(v_data.shape[0], layer_z_end - layer_z_start)
                            combined_void[layer_z_start:layer_z_start+z_len] = np.maximum(
                                combined_void[layer_z_start:layer_z_start+z_len], v_data[:z_len]
                            )
                        elif hasattr(self, 'last_applied_roi_raw') and self.last_applied_roi_raw is not None:
                            roi = self.last_applied_roi_raw
                            z_coords = sorted([roi['z_start'], roi['z_end']])
                            y_coords = sorted([roi['y1'], roi['y2']])
                            x_coords = sorted([roi['x1'], roi['x2']])
                            z1 = int(max(0, z_coords[0]))
                            z2 = int(min(combined_void.shape[0], z_coords[1]))
                            y1 = int(max(0, y_coords[0]))
                            y2 = int(min(combined_void.shape[1], y_coords[1]))
                            x1 = int(max(0, x_coords[0]))
                            x2 = int(min(combined_void.shape[2], x_coords[1]))
                            
                            target_z = z2 - z1
                            target_y = y2 - y1
                            target_x = x2 - x1
                            v_cropped = v_data[:target_z, :target_y, :target_x]
                            
                            dst_slice = (slice(z1, z1 + v_cropped.shape[0]),
                                         slice(y1, y1 + v_cropped.shape[1]),
                                         slice(x1, x1 + v_cropped.shape[2]))
                            combined_void[dst_slice] = np.maximum(
                                combined_void[dst_slice], v_cropped
                            )
            
            self.bump_segmentation = combined_bump
            self.void_segmentation = combined_void
            
            self._populate_layer_filter()
            
            # Update views
            for orientation in ['axial', 'coronal', 'sagittal']:
                self.update_plane_view(orientation)
            
            self.info_label.setText(f"Inspection complete: {total_time:.2f}s across {len(valid_results)} layers")
            
            # Enable Measurement and FAR buttons but do not emit inspection_done directly
            if hasattr(self, 'measure_btn'):
                self.measure_btn.setEnabled(True)
            if hasattr(self, 'far_btn'):
                self.far_btn.setEnabled(True)
            if hasattr(self, 'dt_btn'):
                self.dt_btn.setEnabled(True)
            
            if hasattr(self, 'inspect_btn') and hasattr(self, 'dll_path'):
                # Notify user they can click Measurement next
                out_dir = self.output_path_input.text().strip()
                seg_note = "last layer only" if len(valid_results) > 1 else None
                if len(valid_results) == 1:
                    res = valid_results[0]
                    msg = (
                        f"Inspection completed successfully! Please click 'MES' to proceed.\n"
                        f"Bump Time: {res.bumpTime:.2f}s\n"
                        f"Void Time: {res.voidTime:.2f}s\n"
                        f"Total Time: {res.totalTime:.2f}s"
                    )
                    self._log_dll_timing(
                        "SEG", bumpvoid.get_last_timing, note=seg_note, flush_dir=out_dir
                    )
                    rep = (getattr(self, "_last_dll_timing_reports", {}) or {}).get("SEG")
                    if rep:
                        msg += "\n\n" + rep
                    QMessageBox.information(self, "Success", msg)
                else:
                    msg = (
                        f"Multi-layer Inspection completed successfully! Please click 'MES' to proceed.\n"
                        f"Total Time: {total_time:.2f}s"
                    )
                    self._log_dll_timing(
                        "SEG", bumpvoid.get_last_timing, note=seg_note, flush_dir=out_dir
                    )
                    rep = (getattr(self, "_last_dll_timing_reports", {}) or {}).get("SEG")
                    if rep:
                        msg += "\n\n(last layer)\n" + rep
                    QMessageBox.information(self, "Success", msg)
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load results: {str(e)}")
            
    def run_measurement(self):
        """Proceed to measurement stage using current masks"""
        # We don't overwrite bump_segmentation anymore; measurement iterates over _ml_layer_cache
        if self.bump_segmentation is None and self.void_segmentation is None:
            QMessageBox.warning(self, "Warning", "Please run inspection first to generate masks before measuring.")
            return

        self._apply_dll_profiling_flag()
        self.run_object_analysis()
        try:
            from inno3d.core import bumpvoid_mes
            out_dir = self.output_path_input.text().strip()
            self._log_dll_timing("MES", bumpvoid_mes.get_last_timing, flush_dir=out_dir)
        except Exception:
            pass
        # MES tab index = 4 (Layers=0, ROI=1, Params=2, Enhance=3, MES=4, B2B=5, 3D=6)
        self._set_teaching_tab(4)

    # ------------------------------------------------------------------
    # Layer name helpers (parity with Online MES Layer column)
    # ------------------------------------------------------------------
    def _rebuild_layer_assignment_map(self, layer_bands=None):
        """Build ``layer_assignment_map[z] → layer_name`` from definitions or bands.

        Used after Online FOV (runtime-remapped Z) and before Teaching MES so
        Layer column is not all ``N/A``.
        """
        bands = list(layer_bands or [])
        if not bands:
            defs = getattr(self, "layer_definitions", None) or []
            for L in defs:
                if L is None:
                    continue
                if not L.get("selected", True):
                    continue
                try:
                    zs = int(L.get("z_start", 0))
                    ze = int(L.get("z_end", 0))
                except (TypeError, ValueError):
                    continue
                if ze > zs:
                    bands.append({
                        "name": str(L.get("name") or f"Layer_{len(bands)+1}"),
                        "z_start": zs,
                        "z_end": ze,
                    })
        self.layer_assignment_map = {}
        self._layer_bands_cache = bands
        for L in bands:
            name = str(L.get("name") or "Layer")
            try:
                zs = int(L["z_start"])
                ze = int(L["z_end"])
            except (KeyError, TypeError, ValueError):
                continue
            for z in range(zs, ze):
                self.layer_assignment_map[int(z)] = name
        return bands

    def _layer_name_for_global_z(self, global_z):
        """Resolve layer name for absolute volume Z (Online-style band match)."""
        try:
            cz = float(global_z)
        except (TypeError, ValueError):
            cz = 0.0
        z_i = int(round(cz))

        m = getattr(self, "layer_assignment_map", None) or {}
        if z_i in m:
            return m[z_i]
        # Fuzzy: nearby keys (blank-slice / off-by-one)
        for dz in (1, -1, 2, -2, 3, -3):
            if (z_i + dz) in m:
                return m[z_i + dz]

        bands = getattr(self, "_layer_bands_cache", None) or []
        if not bands:
            bands = self._rebuild_layer_assignment_map()
        for L in bands:
            try:
                if L["z_start"] <= cz < L["z_end"]:
                    return str(L["name"])
            except (KeyError, TypeError):
                continue
        if bands:
            best = min(
                bands,
                key=lambda L: abs(cz - 0.5 * (float(L["z_start"]) + float(L["z_end"]))),
            )
            return str(best.get("name") or "N/A")
        return "N/A"

    def run_object_analysis(self):
        """Run skimage-based per-object measurement on current masks."""
        if self.bump_segmentation is None:
            return
            
        progress = QProgressDialog("Running measurement analysis...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        try:
            self.object_stats = []
            # Ensure Layer map exists (Online → Teaching often left it empty)
            self._rebuild_layer_assignment_map()
            
            # Use configured voxel sizes
            vx = self.param_widgets['voxel_size_x'].value()
            vy = self.param_widgets['voxel_size_y'].value()
            vz = self.param_widgets['voxel_size_z'].value()
            
            # If Z-Stretched 4x mode is enabled, divide effective Z voxel size by 4
            if self.param_widgets.get('z_stretched_4x') and self.param_widgets['z_stretched_4x'].isChecked():
                vz = vz / 4.0
                print(f"[MEASUREMENT] Z-Stretched 4x mode: effective vz = {vz} um")
            
            voxel_volume = vx * vy * vz

            # Determine cc3d connectivity setting
            connectivity = 26
            conn_options = [6, 18, 26]
            if 'cc3d_connectivity' in self.param_widgets:
                conn_idx = self.param_widgets['cc3d_connectivity'].currentIndex()
                connectivity = conn_options[conn_idx]
            
            # Determine min_voxels for CC3D filtering
            min_voxels = 0
            if 'cc3d_min_voxels' in self.param_widgets:
                min_voxels = self.param_widgets['cc3d_min_voxels'].value()
            
            # Determine CC3D backend: GPU (DLL) → Python cc3d → skimage
            use_gpu_cc3d = False
            use_cc3d = False
            
            # Try GPU CC3D first (if enabled in UI and DLL loaded)
            if self.param_widgets.get('cc3d_use_gpu') and self.param_widgets['cc3d_use_gpu'].isChecked():
                try:
                    if bumpvoid.has_cc3d_gpu():
                        use_gpu_cc3d = True
                        print(f"[MEASUREMENT] Using GPU CC3D (DLL CUDA), connectivity={connectivity}, min_voxels={min_voxels}")
                except Exception:
                    pass
            
            # Fallback to Python cc3d
            if not use_gpu_cc3d:
                try:
                    import cc3d
                    use_cc3d = True
                    print(f"[MEASUREMENT] Using Python cc3d, connectivity={connectivity}")
                except ImportError:
                    use_cc3d = False
                    print("cc3d not found, falling back to skimage.measure.label (may cause OOM on large volumes)")

            # =====================================================================
            # MULTI-LAYER MODE: analyze each layer independently
            # =====================================================================
            if getattr(self, 'input_mode', 'single') == 'multi_layer':
                ml_cache = getattr(self, '_ml_layer_cache', {})
                ml_z_starts = getattr(self, '_ml_layer_z_starts', {})
                
                if not ml_cache:
                    progress.close()
                    QMessageBox.warning(self, "Warning", "No layer data found. Please run inspection first.")
                    return
                
                # We don't build combined highlighted data anymore; each layer will have its own labeled data
                self.labeled_class1_data = np.zeros(self.bump_segmentation.shape, dtype=np.uint32)
                global_label_offset = 0  # Ensure unique labels across layers
                
                # Update ml_layer_cache to hold labeled_class1_data
                layer_names = list(ml_cache.keys())
                total_layers = len(layer_names)
                row_counter = 0
                
                progress.setLabelText(f"Analyzing {total_layers} layers independently...")
                progress.setValue(10)
                QApplication.processEvents()
                
                for layer_idx, layer_name in enumerate(layer_names):
                    layer_data = ml_cache[layer_name]
                    layer_bump = layer_data.get('bump')
                    layer_void = layer_data.get('void')
                    
                    if layer_bump is None:
                        continue
                    
                    layer_pct_start = 10 + int(80 * layer_idx / total_layers)
                    layer_pct_end = 10 + int(80 * (layer_idx + 1) / total_layers)
                    progress.setLabelText(f"[{layer_idx+1}/{total_layers}] Analyzing: {layer_name}")
                    progress.setValue(layer_pct_start)
                    QApplication.processEvents()
                    
                    # Label this layer's bump data independently
                    layer_binary = (layer_bump > 0).astype(np.uint8)
                    layer_labeled = None
                    if use_gpu_cc3d:
                        try:
                            layer_binary_255 = (layer_binary * 255).astype(np.uint8)
                            layer_labeled, _ = bumpvoid.cc3d_gpu(layer_binary_255, connectivity=connectivity, min_voxels=min_voxels)
                        except Exception as e:
                            print(f"[MEASUREMENT] GPU CC3D failed ({e}), falling back to CPU...")
                            layer_labeled = None
                            
                    if layer_labeled is None:
                        if use_cc3d:
                            layer_labeled = cc3d.connected_components(layer_binary, connectivity=connectivity)
                        else:
                            layer_labeled = measure.label(layer_binary)
                    
                    layer_props = measure.regionprops(layer_labeled)
                    
                    # Place labeled data into the current layer's cache
                    z_start_global = ml_z_starts.get(layer_name, 0)
                    layer_depth = layer_bump.shape[0]
                    # Offset labels to be globally unique
                    labeled_offset = layer_labeled.copy()
                    labeled_offset[labeled_offset > 0] += global_label_offset
                    # Store to cache instead of massive array
                    self._ml_layer_cache[layer_name]['labeled_class1_data'] = labeled_offset
                    
                    if layer_name == self.current_display_layer:
                        self.labeled_class1_data = labeled_offset
                    
                    # Compute void volumes for this layer
                    c2_volumes_map = {}
                    if layer_void is not None:
                        c2_mask = (layer_void > 0)
                        labels = [p.label for p in layer_props]
                        if labels:
                            c2_sums = ndimage.sum(c2_mask, layer_labeled, index=labels)
                            c2_volumes_map = {labels[i]: float(c2_sums[i]) * voxel_volume for i in range(len(labels))}
                    
                    # Collect stats - Z coordinates are naturally relative to this layer
                    for prop in layer_props:
                        c1_volume = prop.area * voxel_volume
                        c2_volume = c2_volumes_map.get(prop.label, 0.0)
                        ratio = c2_volume / (c1_volume + c2_volume) if (c1_volume + c2_volume) > 0 else 0.0
                        bbox = prop.bbox
                        centroid = prop.centroid
                        
                        soh = (bbox[3] - bbox[0]) * vz  # Solder height in um
                        row_counter += 1
                        
                        self.object_stats.append({
                            'row_id': row_counter,
                            'label': prop.label + global_label_offset,  # Globally unique label
                            'c1_volume': c1_volume,
                            'c2_volume': c2_volume,
                            'ratio': ratio,
                            'soh': soh,
                            # Z coordinates are relative to this layer's volume
                            'z_min': bbox[0], 'y_min': bbox[1], 'x_min': bbox[2],
                            'z_max': bbox[3], 'y_max': bbox[4], 'x_max': bbox[5],
                            'centroid_z': centroid[0],
                            'centroid_z_global': centroid[0] + z_start_global,  # For slider navigation
                            'centroid_y': centroid[1], 'centroid_x': centroid[2],
                            'layer_name': layer_name,
                        })
                    
                    global_label_offset += max((p.label for p in layer_props), default=0)
                
                # No start_slice offset needed for multi-layer (each layer is self-contained)
                self._measurement_start_slice = 0
            
            # =====================================================================
            # SINGLE MODE: original behavior
            # =====================================================================
            else:
                progress.setLabelText("Labeling Bump objects...")
                progress.setValue(10)
                QApplication.processEvents()
                
                # Determine measurement range from blank slices config
                start_slice = 0
                end_slice = self.bump_segmentation.shape[0]
                
                if self.config:
                    start_slice = max(0, self.config.blankEnd1)
                    end_slice = min(self.bump_segmentation.shape[0], self.config.blankStart2)
                    if end_slice <= start_slice:
                        end_slice = self.bump_segmentation.shape[0]
                        
                # Further restrict by active layers if applicable to save memory!
                if hasattr(self, 'layers_active') and self.layers_active:
                    active_layers = [l for l in self.layer_definitions if l['selected']]
                    if active_layers:
                        layer_z_min = min(l['z_start'] for l in active_layers)
                        layer_z_max = max(l['z_end'] for l in active_layers)
                        start_slice = max(start_slice, layer_z_min)
                        end_slice = min(end_slice, layer_z_max)

                # Save start_slice for navigation (slider offset)
                self._measurement_start_slice = start_slice
                
                bump_subset = self.bump_segmentation[start_slice:end_slice]
                bump_binary = (bump_subset > 0).astype(np.uint8)
                
                labeled_subset = None
                if use_gpu_cc3d:
                    try:
                        bump_binary_255 = (bump_binary * 255).astype(np.uint8)
                        labeled_subset, _ = bumpvoid.cc3d_gpu(bump_binary_255, connectivity=connectivity, min_voxels=min_voxels)
                    except Exception as e:
                        print(f"[MEASUREMENT] GPU CC3D failed ({e}), falling back to CPU...")
                        labeled_subset = None
                        
                if labeled_subset is None:
                    if use_cc3d:
                        labeled_subset = cc3d.connected_components(bump_binary, connectivity=connectivity)
                    else:
                        labeled_subset = measure.label(bump_binary)
                    
                c1_props = measure.regionprops(labeled_subset)
                
                # Keep labeled data in same shape as original volume to avoid breaking UI views
                self.labeled_class1_data = np.zeros(self.bump_segmentation.shape, dtype=np.uint32)
                self.labeled_class1_data[start_slice:end_slice] = labeled_subset
                
                if self.void_segmentation is not None:
                    pass # No need to label void, saves memory
                
                progress.setValue(30)
                QApplication.processEvents()
                
                void_subset = self.void_segmentation[start_slice:end_slice] if self.void_segmentation is not None else None
                c2_mask_all = (void_subset > 0) if void_subset is not None else None
                total = len(c1_props)
                
                # --- Vectorized calculation for C2 volumes (MUCH faster) ---
                c2_volumes_map = {}
                if c2_mask_all is not None:
                    progress.setLabelText("Calculating void volumes (optimized)...")
                    labels = [p.label for p in c1_props]
                    c2_sums = ndimage.sum(c2_mask_all, labeled_subset, index=labels)
                    c2_volumes_map = {labels[i]: float(c2_sums[i]) * voxel_volume for i in range(len(labels))}

                for idx, prop in enumerate(c1_props):
                    progress.setValue(30 + int(60 * idx / max(total, 1)))
                    if idx % 50 == 0:
                        QApplication.processEvents()
                    
                    c1_volume = prop.area * voxel_volume
                    c2_volume = c2_volumes_map.get(prop.label, 0.0)
                    
                    ratio = c2_volume / (c1_volume + c2_volume) if (c1_volume + c2_volume) > 0 else 0.0
                    bbox = prop.bbox
                    centroid = prop.centroid
                    
                    # Subset-local Z → absolute volume Z (for Layer bands + navigation)
                    cz_local = float(centroid[0])
                    cz_global = cz_local + float(start_slice)
                    layer_name = self._layer_name_for_global_z(cz_global)

                    soh = (bbox[3] - bbox[0]) * vz  # Solder height in um
                    self.object_stats.append({
                        'row_id': idx + 1,
                        'label': prop.label,
                        'c1_volume': c1_volume,
                        'c2_volume': c2_volume,
                        'ratio': ratio,
                        'soh': soh,
                        # Store absolute Z in bbox/centroid so Index/MPR match Online
                        'z_min': bbox[0] + start_slice,
                        'y_min': bbox[1],
                        'x_min': bbox[2],
                        'z_max': bbox[3] + start_slice,
                        'y_max': bbox[4],
                        'x_max': bbox[5],
                        'centroid_z': cz_global,
                        'centroid_z_global': cz_global,
                        'centroid_y': centroid[1],
                        'centroid_x': centroid[2],
                        'layer_name': layer_name,
                    })
                # Full-volume absolute Z stored above → no extra offset on navigate
                self._measurement_start_slice = 0
            
            progress.setLabelText(
                f"Populating table ({len(self.object_stats)} objects)..."
            )
            progress.setValue(92)
            QApplication.processEvents()

            # Online-style (1,1)=top-left per-layer grid (fixes messy Bump ID)
            try:
                from inno3d.core.bumpvoid_mes import reindex_grid_top_left
                self.object_stats = reindex_grid_top_left(
                    self.object_stats, voxel_x=vx, voxel_y=vy, per_layer=True
                )
            except Exception as _ix_err:
                print(f"[TEACHING MES] reindex_grid_top_left failed: {_ix_err}")
            
            # Sorting and grid indexing handled inside _refresh_stats_table
            self._refresh_stats_table()
            
            total_c1 = sum(s['c1_volume'] for s in self.object_stats)
            total_c2 = sum(s['c2_volume'] for s in self.object_stats)
            self.stats_info_label.setText(
                f"{len(self.object_stats)} objects | "
                f"Total Bump: {total_c1:,.0f} \u03bcm\u00b3 | Total Void: {total_c2:,.0f} \u03bcm\u00b3"
            )
            self.export_csv_btn.setEnabled(True)
            if hasattr(self, 'bnd_btn'):
                self.bnd_btn.setEnabled(True)
            if hasattr(self, 'far_btn'):
                self.far_btn.setEnabled(True)
            
            progress.setValue(100)
            progress.close()
            
        except Exception as e:
            try:
                progress.close()
            except Exception:
                pass
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Analysis Error", f"Error during analysis:\n{str(e)}")

    def export_stats_csv(self, output_path=None):
        if not self.object_stats:
            return
        
        is_auto_export = isinstance(output_path, str) and bool(output_path)
        
        # Determine output folder
        if not is_auto_export:
            folder = self.output_path_input.text().strip()
            if not os.path.isdir(folder):
                folder = QFileDialog.getExistingDirectory(self, "Select Output Folder for CSV Files")
            if not folder:
                return
        else:
            folder = os.path.dirname(output_path)

        import csv
        ng_threshold = self.ng_threshold_spin.value() / 100.0 if hasattr(self, 'ng_threshold_spin') else 0.05

        csv_header = [
            '#', 'Layer', 'Grid(R,C)', 'B. H', 'B. V', 'V. V', 
            'Ratio (Void/(TGV+Void))', 'Judgment', 'Gap X (um)', 'Gap Y (um)',
            'Z_min', 'Z_max', 'Y_min', 'Y_max', 'X_min', 'X_max',
            'Centroid_Z', 'Centroid_Y', 'Centroid_X'
        ]

        def _make_csv_row(stat, idx):
            ratio_str = f"{stat['ratio']*100:.3f}%" if stat['ratio'] != float('inf') else "N/A"
            is_ng = stat['ratio'] >= ng_threshold if stat['ratio'] != float('inf') else True
            judgment_str = "NG" if is_ng else "OK"
            layer_name = self._resolve_stat_layer_name(stat) or "N/A"
            grid_rc = f"({stat.get('grid_row', '?')},{stat.get('grid_col', '?')})"
            soh_str = f"{stat.get('soh', 0):.3f}"
            return [
                idx,
                layer_name,
                grid_rc,
                soh_str,
                f"{stat['c1_volume']:,.3f}",
                f"{stat['c2_volume']:,.3f}",
                ratio_str,
                judgment_str,
                f"{stat.get('pitch_x', 0):.3f}", f"{stat.get('pitch_y', 0):.3f}",
                str(stat['z_min']), str(stat['z_max']),
                str(stat['y_min']), str(stat['y_max']),
                str(stat['x_min']), str(stat['x_max']),
                f"{stat['centroid_z']:.3f}", f"{stat['centroid_y']:.3f}", f"{stat['centroid_x']:.3f}",
            ]

        try:
            results_dir = os.path.join(folder, "measurement_results")
            os.makedirs(results_dir, exist_ok=True)

            # Group stats by layer
            layer_groups = {}
            for stat in self.object_stats:
                layer_name = self._resolve_stat_layer_name(stat) or "N/A"
                if layer_name not in layer_groups:
                    layer_groups[layer_name] = []
                layer_groups[layer_name].append(stat)

            exported_files = []

            # Write per-layer CSVs
            for layer_name, layer_stats in sorted(layer_groups.items(), key=lambda x: (x[0] == "N/A", x[0])):
                safe_name = layer_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                layer_csv_path = os.path.join(results_dir, f"{safe_name}.csv")
                with open(layer_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_header)
                    for idx, stat in enumerate(layer_stats, 1):
                        writer.writerow(_make_csv_row(stat, idx))
                exported_files.append(layer_csv_path)

            # Write combined summary CSV
            summary_path = os.path.join(results_dir, "all_layers_summary.csv")
            with open(summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)
                for idx, stat in enumerate(self.object_stats, 1):
                    writer.writerow(_make_csv_row(stat, idx))
            exported_files.append(summary_path)

            if not is_auto_export:
                self.stats_info_label.setText(f"CSV exported: {len(exported_files)} files  {results_dir}")
                QMessageBox.information(self, "Export Successful", 
                    f"Exported {len(layer_groups)} per-layer CSVs + 1 summary CSV to:\n{results_dir}")
            return summary_path
        except Exception as e:
            if not is_auto_export:
                QMessageBox.critical(self, "Export Error", f"Failed to export CSV:\n{str(e)}")
            return None

    def on_object_selection_changed(self):
        """MES table → MPR highlight + navigate + 3D cyan surfaces (shared)."""
        selected_rows = sorted(
            list(set(index.row() for index in self.object_stats_table.selectedIndexes()))
        )
        _z_offset = int(getattr(self, "_measurement_start_slice", 0) or 0)
        labeled = getattr(self, "labeled_class1_data", None)

        stats_indices = []
        current_item = self.object_stats_table.currentItem()
        target_visual_row = current_item.row() if current_item else -1
        target_stat = None

        for visual_row in selected_rows:
            id_item = self.object_stats_table.item(visual_row, 0)
            if id_item is None:
                continue
            stats_idx = id_item.data(Qt.UserRole)
            if stats_idx is None:
                continue
            try:
                stats_idx = int(stats_idx)
            except (TypeError, ValueError):
                continue
            if stats_idx < 0 or stats_idx >= len(self.object_stats or []):
                continue
            stats_indices.append(stats_idx)
            st = self.object_stats[stats_idx]
            if visual_row == target_visual_row or target_stat is None:
                target_stat = st

        self.selected_highlight_objects = _shared_highlights_from_stats_indices(
            self.object_stats,
            stats_indices,
            labeled=labeled,
            z_offset=_z_offset,
        )

        # Multi-layer: switch display layer for navigate target
        if target_stat is not None and getattr(self, "input_mode", "single") == "multi_layer":
            if hasattr(self, "layer_filter_combo"):
                layer_name = target_stat.get("layer_name")
                if layer_name and layer_name != self.current_display_layer:
                    idx = self.layer_filter_combo.findData(layer_name)
                    if idx >= 0:
                        self.layer_filter_combo.setCurrentIndex(idx)

        if target_stat is not None and self.volume_data is not None:
            xyz = _shared_centroid_nav_xyz(target_stat, z_offset=_z_offset)
            if xyz is not None and hasattr(self, "updatePoint"):
                cx, cy, cz = xyz
                try:
                    self.updatePoint(cx, cy, cz)
                except Exception:
                    # Fallback: sliders only
                    z_max = self.volume_data.shape[0] - 1
                    y_max = self.volume_data.shape[1] - 1
                    x_max = self.volume_data.shape[2] - 1
                    self.axial_slice_slider.setValue(min(max(0, cz), z_max))
                    self.coronal_slice_slider.setValue(min(max(0, cy), y_max))
                    self.sagittal_slice_slider.setValue(min(max(0, cx), x_max))

        if self.volume_data is not None:
            for ori in ["axial", "coronal", "sagittal"]:
                self.update_plane_view(ori, preserve_camera=True)
            if hasattr(self, "_update_mes_3d_highlight"):
                try:
                    self._update_mes_3d_highlight()
                except Exception as e:
                    print(f"[MES] Teaching 3D highlight skipped: {e}")

    def clear_object_selection(self):
        self.object_stats_table.blockSignals(True)
        self.object_stats_table.clearSelection()
        self.object_stats_table.blockSignals(False)
        self.selected_highlight_objects = []
        if hasattr(self, "_clear_mes_3d_highlight"):
            try:
                self._clear_mes_3d_highlight(render=True)
            except Exception:
                pass
        if self.volume_data is not None:
            for ori in ["axial", "coronal", "sagittal"]:
                self.update_plane_view(ori, preserve_camera=True)

    # ── MES pick: MPR → table (shared mapping; Teaching Multi always ON) ──

    def is_mes_multi_select(self):
        """Teaching MES table is always MultiSelection (parity with Viewer Multi ON)."""
        return True

    def _mes_pick_is_duplicate(self, lab):
        deb = getattr(self, "_mes_pick_debounce_obj", None)
        if deb is None:
            deb = MesPickDebounce(0.12)
            self._mes_pick_debounce_obj = deb
        return deb.is_duplicate(lab)

    def _visual_row_for_stats_index(self, stats_idx):
        table = getattr(self, "object_stats_table", None)
        if table is None or stats_idx is None:
            return None
        for row in range(table.rowCount()):
            if table.isRowHidden(row):
                continue
            item = table.item(row, 0)
            if item is None:
                continue
            try:
                if int(item.data(Qt.UserRole)) == int(stats_idx):
                    return row
            except (TypeError, ValueError):
                if row == int(stats_idx):
                    return row
        return None

    def _apply_mes_pick_label(self, lab, xyz=None, source="mpr"):
        """Sticky multi-toggle MES row + MPR/3D highlight (shared label map)."""
        try:
            lab = int(lab)
        except (TypeError, ValueError):
            return False
        if lab <= 0:
            return False
        if self._mes_pick_is_duplicate(lab):
            return True
        if not getattr(self, "object_stats", None):
            return False

        _z_off = int(getattr(self, "_measurement_start_slice", 0) or 0)
        stats_idx = _shared_stats_index_for_label(
            self.object_stats,
            lab,
            labeled=getattr(self, "labeled_class1_data", None),
            z_offset=_z_off,
        )
        if stats_idx is None:
            print(f"[MES pick] Teaching label={lab} not in object_stats source={source}")
            return False

        visual_row = self._visual_row_for_stats_index(stats_idx)
        if visual_row is None:
            return False

        table = self.object_stats_table
        model = table.model()
        sm = table.selectionModel()
        if model is None or sm is None:
            return False
        left = model.index(int(visual_row), 0)
        right = model.index(int(visual_row), max(0, table.columnCount() - 1))
        if not left.isValid():
            return False
        sel = QItemSelection(left, right)

        table.blockSignals(True)
        try:
            # Teaching Multi ON: toggle row (never setCurrentCell — clears multi)
            already = False
            try:
                already = any(
                    int(idx.row()) == int(visual_row) for idx in sm.selectedRows()
                )
            except Exception:
                pass
            sm.select(sel, QItemSelectionModel.Toggle | QItemSelectionModel.Rows)
            sm.setCurrentIndex(left, QItemSelectionModel.NoUpdate)
            try:
                table.scrollTo(left, QAbstractItemView.PositionAtCenter)
            except Exception:
                pass
            action = "removed" if already else "added"
        finally:
            table.blockSignals(False)

        # Rebuild highlights from full selection (same as table click path)
        self.on_object_selection_changed()
        print(
            f"[MES pick] Teaching source={source} {action} "
            f"stats_idx={stats_idx} visual_row={visual_row} label={lab} sticky"
        )
        return True

    def select_mes_object_from_mpr(self, orientation, pos):
        """Ctrl+click MPR → toggle MES row + highlight (Viewer sticky multi)."""
        if self.volume_data is None:
            return False
        if not getattr(self, "object_stats", None):
            return False
        if getattr(self, "labeled_class1_data", None) is None:
            return False
        if not hasattr(self, "_mpr_pick_volume_xyz"):
            return False
        xyz = self._mpr_pick_volume_xyz(orientation, pos)
        if xyz is None:
            return False
        x, y, z = xyz
        lab = _shared_label_at_volume_xyz(self.labeled_class1_data, x, y, z)
        if lab <= 0:
            print(f"[MES pick] Teaching MPR miss at ({x},{y},{z})")
            return False
        return self._apply_mes_pick_label(lab, xyz=(x, y, z), source="mpr")

    def delete_selected_objects(self):
        """Delete user-selected objects from statistics and masks."""
        selected_rows = set()
        for item in self.object_stats_table.selectedItems():
            selected_rows.add(item.row())
            
        if not selected_rows:
            QMessageBox.information(self, "Delete", "Please select at least one row to delete.")
            return

        reply = QMessageBox.question(self, 'Confirm Delete', 
                                    f"Are you sure you want to delete {len(selected_rows)} selected object(s)?\n"
                                    "This will permanently remove them from the current view masks.",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.No:
            return

        # Map visual rows to actual stats indices via UserRole
        stats_indices = []
        for visual_row in selected_rows:
            id_item = self.object_stats_table.item(visual_row, 0)
            if id_item is not None:
                stats_idx = id_item.data(Qt.UserRole)
                if stats_idx is not None:
                    stats_indices.append(stats_idx)
        # Collect labels and remove stats (reverse order to keep indices valid)
        labels_to_remove = []
        for stats_idx in sorted(stats_indices, reverse=True):
            if stats_idx < len(self.object_stats):
                labels_to_remove.append(self.object_stats[stats_idx]['label'])
                self.object_stats.pop(stats_idx)

        # Batch update masks
        if labels_to_remove and self.labeled_class1_data is not None:
             # Find mask of pixels to zero out
             mask = np.isin(self.labeled_class1_data, labels_to_remove)
             
             # Zero out labels
             self.labeled_class1_data[mask] = 0
             
             # Zero out masks in Bump and Void segmentations
             if self.bump_segmentation is not None:
                 self.bump_segmentation[mask] = 0
             if self.void_segmentation is not None:
                 self.void_segmentation[mask] = 0
                 
        # Clear selection as some rows are gone
        self.selected_highlight_objects = []
        self.object_stats_table.clearSelection()
        
        # Refresh the table (includes renumbering)
        self._refresh_stats_table()
        
        # Refresh views
        for orientation in ['axial', 'coronal', 'sagittal']:
            self.update_plane_view(orientation)
            
        # Update summary info
        total_c1 = sum(s['c1_volume'] for s in self.object_stats)
        total_c2 = sum(s['c2_volume'] for s in self.object_stats)
        self.stats_info_label.setText(
            f"{len(self.object_stats)} objects | "
            f"Total Bump: {total_c1:,.0f} \u03bcm\u00b3 | Total Void: {total_c2:,.0f} \u03bcm\u00b3"
        )
                 
    def delete_edge_objects(self):
        """Delete all objects touching the 4 XY edges of the image (incomplete objects)."""
        if not self.object_stats:
            QMessageBox.information(self, "Delete Edge", "No objects to filter. Run measurement first.")
            return
        
        if self.volume_data is None:
            return
        
        _, max_y, max_x = self.volume_data.shape
        max_y -= 1
        max_x -= 1
        
        # Find edge objects: any object whose bounding box touches x=0, x=max, y=0, or y=max
        edge_indices = []
        for idx, stat in enumerate(self.object_stats):
            x_min = stat['x_min']
            x_max = stat['x_max']
            y_min = stat['y_min']
            y_max = stat['y_max']
            
            if x_min <= 0 or x_max >= max_x or y_min <= 0 or y_max >= max_y:
                edge_indices.append(idx)
        
        if not edge_indices:
            QMessageBox.information(self, "Delete Edge", "No objects touching image edges found.")
            return
        
        reply = QMessageBox.question(self, 'Confirm Delete Edge Objects',
            f"Found {len(edge_indices)} object(s) touching image edges.\n"
            f"These are likely incomplete/truncated objects.\n\n"
            f"Delete them from statistics and masks?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.No:
            return
        
        # Collect labels and remove stats (reverse order)
        labels_to_remove = []
        for idx in sorted(edge_indices, reverse=True):
            labels_to_remove.append(self.object_stats[idx]['label'])
            self.object_stats.pop(idx)
        
        # Batch update masks
        if labels_to_remove and self.labeled_class1_data is not None:
            mask = np.isin(self.labeled_class1_data, labels_to_remove)
            self.labeled_class1_data[mask] = 0
            if self.bump_segmentation is not None:
                self.bump_segmentation[mask] = 0
            if self.void_segmentation is not None:
                self.void_segmentation[mask] = 0
        
        # Clear selection
        self.selected_highlight_objects = []
        self.object_stats_table.clearSelection()
        
        # Refresh
        self._refresh_stats_table()
        for orientation in ['axial', 'coronal', 'sagittal']:
            self.update_plane_view(orientation)
        
        # Update summary
        remaining = len(self.object_stats)
        self.stats_info_label.setText(
            f"{remaining} objects (removed {len(labels_to_remove)} edge objects)")
        QMessageBox.information(self, "Delete Edge Complete",
            f"Removed {len(labels_to_remove)} edge objects.\n{remaining} objects remaining.")

    def run_distance_transform(self):
        """Perform 3D Distance Transform on segmentation masks layer-by-layer."""
        # Collect layers to process
        layers_to_process = []  # list of dicts: {'name': str, 'bump': ndarray, 'void': ndarray, 'output_dir': str}
        
        output_base_dir = self.output_path_input.text().strip()
        if not output_base_dir:
            output_base_dir = os.path.dirname(self.config_path) if self.config_path else os.getcwd()
            
        if getattr(self, 'input_mode', 'single') == 'multi_layer':
            ml_cache = getattr(self, '_ml_layer_cache', {})
            if not ml_cache:
                QMessageBox.warning(self, "Warning", "No layer cache found. Please run inspection/segmentation first.")
                return
            for name, cache_item in ml_cache.items():
                layer_output = os.path.join(output_base_dir, name.replace(" ", "_"))
                layers_to_process.append({
                    'name': name,
                    'bump': cache_item.get('bump'),
                    'void': cache_item.get('void'),
                    'output_dir': layer_output
                })
        else:
            # Single or test_1layer mode
            if self.bump_segmentation is None:
                QMessageBox.warning(self, "Warning", "No bump mask found. Please run inspection/segmentation first.")
                return
                
            if hasattr(self, 'layers_active') and self.layers_active:
                selected_layers = [l for l in self.layer_definitions if l['selected']]
                if not selected_layers:
                    QMessageBox.warning(self, "Warning", "No layers selected. Please define and select layers.")
                    return
                for layer in selected_layers:
                    z_start = max(0, int(layer['z_start']))
                    z_end = min(self.bump_segmentation.shape[0], int(layer['z_end']))
                    if z_start >= z_end:
                        continue
                    layer_bump = self.bump_segmentation[z_start:z_end]
                    layer_void = self.void_segmentation[z_start:z_end] if self.void_segmentation is not None else None
                    layer_output = os.path.join(output_base_dir, layer['name'].replace(" ", "_"))
                    layers_to_process.append({
                        'name': layer['name'],
                        'bump': layer_bump,
                        'void': layer_void,
                        'output_dir': layer_output
                    })
            else:
                # Treat entire volume as one layer
                layers_to_process.append({
                    'name': "Full_Volume",
                    'bump': self.bump_segmentation,
                    'void': self.void_segmentation,
                    'output_dir': output_base_dir
                })

        if not any(item['bump'] is not None for item in layers_to_process):
            QMessageBox.warning(self, "Warning", "No segmentation masks available for 3D distance transform.")
            return

        # Get current voxel spacing
        vx = self.param_widgets['voxel_size_x'].value() if 'voxel_size_x' in self.param_widgets else 1.0
        vy = self.param_widgets['voxel_size_y'].value() if 'voxel_size_y' in self.param_widgets else 1.0
        vz = self.param_widgets['voxel_size_z'].value() if 'voxel_size_z' in self.param_widgets else 1.0
        
        # Adjust for Z-stretched if checked
        if self.param_widgets.get('z_stretched_4x') and self.param_widgets['z_stretched_4x'].isChecked():
            vz_display = vz / 4.0
            z_stretch_suffix = " (adjusted 4x)"
        else:
            vz_display = vz
            z_stretch_suffix = ""

        # Create settings dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("3D Distance Transform Settings")
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet(f"""
            QDialog {{ background-color: {SemiconductorTheme.BG_PANEL}; }}
            QLabel {{ color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 9pt; }}
            QCheckBox {{ color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 9pt; }}
            QRadioButton {{ color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 9pt; }}
        """)
        
        layout = QVBoxLayout(dialog)
        
        info = QLabel(f"<b>Perform 3D Distance Transform layer-by-layer</b><br>"
                      f"Found {len(layers_to_process)} layers/volumes to process.<br>"
                      f"Target: Bump mask (inside bump to boundary)")
        layout.addWidget(info)
        layout.addSpacing(10)
        
        # Spacing Group Box
        spacing_group = QGroupBox("Voxel Spacing (Sampling)")
        spacing_group.setStyleSheet(self._group_box_qss())
        sp_layout = QVBoxLayout(spacing_group)
        rb_physical = QRadioButton(f"Use physical spacing:<br>Z: {vz_display:.3f}{z_stretch_suffix}, Y: {vy:.3f}, X: {vx:.3f} \u03bcm")
        rb_physical.setChecked(True)
        rb_uniform = QRadioButton("Use uniform isotropic spacing (1.0, 1.0, 1.0)")
        sp_layout.addWidget(rb_physical)
        sp_layout.addWidget(rb_uniform)
        layout.addWidget(spacing_group)
        layout.addSpacing(15)
        
        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        if dialog.exec() != QDialog.Accepted:
            return
            
        do_bump = True
        do_void = False
        use_physical = rb_physical.isChecked()

        # Setup progress dialog
        self.dt_progress = QProgressDialog("Starting 3D Distance Transform...", "Cancel", 0, 100, self)
        self.dt_progress.setWindowTitle("Distance Transform")
        self.dt_progress.setWindowModality(Qt.WindowModal)
        self.dt_progress.setMinimumDuration(0)
        self.dt_progress.setValue(0)
        
        # Instantiate and start the thread
        spacing = (vz_display, vy, vx)
        self.dt_thread = DistanceTransformThread(
            layers_to_process, spacing, 
            do_bump=do_bump, do_void=do_void, 
            use_physical=use_physical
        )
        
        def handle_progress(val, text):
            self.dt_progress.setValue(val)
            self.dt_progress.setLabelText(text)
            
        def handle_finished(success, message):
            self.dt_progress.close()
            if success:
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.critical(self, "Error", message)
                
        self.dt_thread.progress.connect(handle_progress)
        self.dt_thread.finished.connect(handle_finished)
        
        # Cancel thread cooperatively
        self.dt_progress.canceled.connect(self.dt_thread.cancel)
        
        self.dt_thread.start()

    def run_boundary_analysis(self):
        """3D Boundary Analysis: erode → boundary → surface-to-surface gap per bump."""
        # Check for labeled data (requires measurement)
        if self.labeled_class1_data is None:
            QMessageBox.warning(self, "Warning",
                "No labeled bump data found.\nPlease run MEASUREMENT first.")
            return

        self._apply_dll_profiling_flag()

        # Collect layers to process
        layers_to_process = []
        output_base_dir = self.output_path_input.text().strip()
        if not output_base_dir:
            output_base_dir = os.path.dirname(self.config_path) if self.config_path else os.getcwd()

        if getattr(self, 'input_mode', 'single') == 'multi_layer':
            ml_cache = getattr(self, '_ml_layer_cache', {})
            if not ml_cache:
                QMessageBox.warning(self, "Warning", "No layer cache found. Run inspection + measurement first.")
                return
            for name, cache_item in ml_cache.items():
                bump = cache_item.get('bump')
                labeled = cache_item.get('labeled_class1_data')
                if bump is None or labeled is None:
                    continue
                layer_output = os.path.join(output_base_dir, name.replace(" ", "_"))
                layers_to_process.append({
                    'name': name, 'bump': bump,
                    'labeled': labeled, 'output_dir': layer_output
                })
        else:
            if self.bump_segmentation is None:
                QMessageBox.warning(self, "Warning", "No bump mask found. Run inspection first.")
                return

            input_mode = getattr(self, 'input_mode', 'single')
            if input_mode == 'single' and getattr(self, 'layers_active', False) and self.layer_definitions:
                selected_layers = [l for l in self.layer_definitions if l['selected']]
                if not selected_layers:
                    QMessageBox.warning(self, "Warning", "No layers selected. Please check at least one layer.")
                    return
                
                for layer in selected_layers:
                    z_start = max(0, layer['z_start'])
                    z_end = min(self.bump_segmentation.shape[0], layer['z_end'])
                    if z_end <= z_start:
                        continue
                    
                    name = layer['name']
                    layer_output = os.path.join(output_base_dir, name.replace(" ", "_"))
                    
                    # Create memory views (slices) of the arrays
                    bump_slice = self.bump_segmentation[z_start:z_end]
                    labeled_slice = self.labeled_class1_data[z_start:z_end]
                    
                    layers_to_process.append({
                        'name': name,
                        'bump': bump_slice,
                        'labeled': labeled_slice,
                        'output_dir': layer_output
                    })
            else:
                # test_1layer mode or no layers defined
                layers_to_process.append({
                    'name': "Full_Volume",
                    'bump': self.bump_segmentation,
                    'labeled': self.labeled_class1_data,
                    'output_dir': output_base_dir
                })
        if not layers_to_process:
            QMessageBox.warning(self, "Warning", "No layers with labeled bump data to process.")
            return

        # Count total bumps
        total_bumps = 0
        for item in layers_to_process:
            total_bumps += len(np.unique(item['labeled'])) - 1  # exclude 0

        # Get voxel spacing
        vx = self.param_widgets['voxel_size_x'].value() if 'voxel_size_x' in self.param_widgets else 1.0
        vy = self.param_widgets['voxel_size_y'].value() if 'voxel_size_y' in self.param_widgets else 1.0
        vz = self.param_widgets['voxel_size_z'].value() if 'voxel_size_z' in self.param_widgets else 1.0
        if self.param_widgets.get('z_stretched_4x') and self.param_widgets['z_stretched_4x'].isChecked():
            vz = vz / 4.0

        # Settings dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("3D Boundary Analysis Settings")
        dialog.setMinimumWidth(420)
        dialog.setStyleSheet(f"""
            QDialog {{ background-color: {SemiconductorTheme.BG_PANEL}; }}
            QLabel {{ color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 9pt; }}
            QCheckBox {{ color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 9pt; }}
            QSpinBox {{ background: {SemiconductorTheme.BG_LIGHT}; color: {SemiconductorTheme.TEXT_PRIMARY};
                        border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; border-radius: 3px; padding: 2px 6px; }}
        """)
        dlayout = QVBoxLayout(dialog)

        info = QLabel(
            f"<b>3D Boundary Analysis (Surface-to-Surface Gap)</b><br>"
            f"Layers: {len(layers_to_process)} | ~{total_bumps} bumps<br>"
            f"Spacing: Z={vz:.3f}, Y={vy:.3f}, X={vx:.3f} μm<br><br>"
            f"<i>Algorithm: Erode bump mask → extract boundary →<br>"
            f"KD-tree nearest-neighbor gap between bump surfaces</i>")
        dlayout.addWidget(info)
        dlayout.addSpacing(8)

        # Kernel size
        kern_row = QHBoxLayout()
        kern_row.addWidget(QLabel("Erosion kernel:"))
        kern_spin = NoScrollSpinBox()
        kern_spin.setRange(3, 7)
        kern_spin.setSingleStep(2)
        kern_spin.setValue(3)
        kern_spin.setToolTip("3D erosion kernel size (must be odd). 3×3×3 gives 1-voxel boundary.")
        kern_row.addWidget(kern_spin)
        kern_row.addStretch()
        dlayout.addLayout(kern_row)

        # Iterations
        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Erosion iterations:"))
        iter_spin = NoScrollSpinBox()
        iter_spin.setRange(1, 5)
        iter_spin.setValue(1)
        iter_spin.setToolTip("Number of erosion iterations. Higher = thicker boundary shell.")
        iter_row.addWidget(iter_spin)
        iter_row.addStretch()
        dlayout.addLayout(iter_row)

        # Save boundary TIFF
        save_tif_cb = QCheckBox("Save boundary_mask.tif")
        save_tif_cb.setChecked(True)
        dlayout.addWidget(save_tif_cb)

        dlayout.addSpacing(10)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        dlayout.addWidget(btn_box)

        if dialog.exec() != QDialog.Accepted:
            return

        kernel_size = kern_spin.value()
        iterations = iter_spin.value()
        save_tif = save_tif_cb.isChecked()
        spacing = (vz, vy, vx)

        # Progress dialog
        self.bnd_progress = QProgressDialog("Starting 3D Boundary Analysis...", "Cancel", 0, 100, self)
        self.bnd_progress.setWindowTitle("Boundary Analysis")
        self.bnd_progress.setWindowModality(Qt.WindowModal)
        self.bnd_progress.setMinimumDuration(0)
        self.bnd_progress.setValue(0)

        dll_folder = ""
        if hasattr(self, "dll_path_input") and self.dll_path_input is not None:
            dll_folder = self.dll_path_input.text().strip()
        if not dll_folder:
            try:
                from inno3d.infra.paths import default_dll_dir
                dll_folder = str(default_dll_dir() or "")
            except Exception:
                dll_folder = ""

        self.bnd_thread = BoundaryAnalysisThread(
            layers_to_process, spacing,
            object_stats=self.object_stats,
            kernel_size=kernel_size, iterations=iterations,
            save_boundary_tif=save_tif,
            dll_folder=dll_folder,
        )

        def handle_bnd_progress(val, text):
            self.bnd_progress.setValue(val)
            self.bnd_progress.setLabelText(text)

        def handle_bnd_finished(success, message):
            self.bnd_progress.close()
            try:
                from inno3d.core import bumpvoid_b2b
                out_dir = self.output_path_input.text().strip()
                self._log_dll_timing("B2B", bumpvoid_b2b.get_last_timing, flush_dir=out_dir)
                rep = (getattr(self, "_last_dll_timing_reports", {}) or {}).get("B2B")
                if rep:
                    message = f"{message}\n\n{rep}"
            except Exception:
                pass
            if success:
                # Auto-load boundary results into the B2B table (gap or summary)
                output_base = self.output_path_input.text().strip()
                if not output_base:
                    output_base = os.path.dirname(self.config_path) if self.config_path else os.getcwd()
                search_roots = [output_base]
                for item in layers_to_process:
                    od = item.get("output_dir")
                    if od and od not in search_roots:
                        search_roots.append(od)
                found = self._find_boundary_csv_paths(search_roots)
                if found:
                    # Prefer gap-format CSVs (Viewer/Online style); load first best
                    best = found[0]
                    try:
                        self._load_boundary_csv(best)
                        # If multiple layer gap files, merge remaining rows of same format
                        if len(found) > 1 and "gap" in os.path.basename(best).lower():
                            self._merge_extra_boundary_csvs(found[1:])
                        QMessageBox.information(
                            self,
                            "Success",
                            f"{message}\n\nLoaded B2B table from:\n{os.path.basename(best)}"
                            + (f" (+{len(found)-1} more)" if len(found) > 1 else ""),
                        )
                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            "B2B table",
                            f"{message}\n\nCSV found but failed to load:\n{e}",
                        )
                else:
                    QMessageBox.warning(
                        self,
                        "B2B finished — no table CSV",
                        f"{message}\n\n"
                        "No boundary_gap.csv / boundary_summary.csv found under:\n"
                        f"{output_base}\n\n"
                        "Check layer subfolders or use Load CSV on the B2B tab.",
                    )
            else:
                QMessageBox.critical(self, "Error", message)

        self.bnd_thread.progress.connect(handle_bnd_progress)
        self.bnd_thread.finished.connect(handle_bnd_finished)
        self.bnd_progress.canceled.connect(self.bnd_thread.cancel)
        self.bnd_thread.start()

    def _far_parse_size_um3(self, text):
        """Parse ``XxYxZ`` size field → product (voxels). Empty / bad → 0."""
        try:
            parts = [float(p.strip()) for p in str(text).lower().split("x") if p.strip()]
            if not parts:
                return 0.0
            res = 1.0
            for p in parts:
                res *= p
            return res
        except Exception:
            return 0.0

    def _far_clear_labels_via_stats(self, labeled, stats_list, labels, bump=None, void=None,
                                    clear_void_only=False):
        """Clear labels using each object's bbox (O(removed·bbox) — fast on large FOVs)."""
        if labeled is None or not labels:
            return
        lab_set = set()
        for x in labels:
            try:
                li = int(x)
                if li > 0:
                    lab_set.add(li)
            except (TypeError, ValueError):
                pass
        if not lab_set:
            return
        nz, ny, nx = labeled.shape
        # Map label → first matching stat with bbox
        by_lab = {}
        for st in stats_list or []:
            try:
                lab = int(st.get("label", 0))
            except (TypeError, ValueError):
                continue
            if lab in lab_set and lab not in by_lab:
                by_lab[lab] = st
        n_done = 0
        for lab in lab_set:
            st = by_lab.get(lab)
            if st is not None:
                try:
                    z0 = max(0, int(st.get("z_min", 0)))
                    z1 = min(nz, int(st.get("z_max", nz)) + 1)
                    y0 = max(0, int(st.get("y_min", 0)))
                    y1 = min(ny, int(st.get("y_max", ny)) + 1)
                    x0 = max(0, int(st.get("x_min", 0)))
                    x1 = min(nx, int(st.get("x_max", nx)) + 1)
                except (TypeError, ValueError):
                    z0, z1, y0, y1, x0, x1 = 0, nz, 0, ny, 0, nx
            else:
                z0, z1, y0, y1, x0, x1 = 0, nz, 0, ny, 0, nx
            if z1 <= z0 or y1 <= y0 or x1 <= x0:
                continue
            region = labeled[z0:z1, y0:y1, x0:x1]
            hit = region == lab
            if not np.any(hit):
                continue
            if not clear_void_only:
                region[hit] = 0
                if bump is not None and bump.shape == labeled.shape:
                    bump[z0:z1, y0:y1, x0:x1][hit] = 0
            if void is not None and void.shape == labeled.shape:
                void[z0:z1, y0:y1, x0:x1][hit] = 0
            n_done += 1
            if n_done % 20 == 0:
                QApplication.processEvents()

    def _far_clear_labels_on_volume(self, labeled, labels, bump=None, void=None, clear_void_only=False,
                                   stats_for_bbox=None):
        """Zero voxels of given labels. Prefer bbox path; LUT fallback for leftovers."""
        if labeled is None or not labels:
            return
        # Preferred: bbox from MES stats (seconds → milliseconds on 900³-class volumes)
        if stats_for_bbox:
            self._far_clear_labels_via_stats(
                labeled, stats_for_bbox, labels,
                bump=bump, void=void, clear_void_only=clear_void_only,
            )
            return
        try:
            labs = [int(x) for x in labels if int(x) > 0]
        except (TypeError, ValueError):
            return
        if not labs:
            return
        max_lab = int(np.max(labeled)) if labeled.size else 0
        if max_lab <= 0:
            return
        if max_lab > 5_000_000:
            for lab in set(labs):
                mask = labeled == lab
                if not np.any(mask):
                    continue
                if not clear_void_only:
                    labeled[mask] = 0
                    if bump is not None:
                        bump[mask] = 0
                if void is not None:
                    void[mask] = 0
            return

        lut_keep = np.ones(max_lab + 1, dtype=bool)
        for lab in labs:
            if 0 < lab <= max_lab:
                lut_keep[lab] = False
        nz = int(labeled.shape[0])
        chunk = max(1, min(16, nz))
        for z0 in range(0, nz, chunk):
            z1 = min(nz, z0 + chunk)
            slab = labeled[z0:z1]
            safe = np.minimum(slab, max_lab)
            kill = ~lut_keep[safe]
            if not np.any(kill):
                continue
            if not clear_void_only:
                slab[kill] = 0
                if bump is not None and bump.shape == labeled.shape:
                    bump[z0:z1][kill] = 0
            if void is not None and void.shape == labeled.shape:
                void[z0:z1][kill] = 0
            QApplication.processEvents()

    def run_false_alarm_remover(self):
        """False Alarm Remover: filter Bump/Void by min/max voxel constraints.

        Runs on the main thread but uses a progress dialog + LUT mask clears
        (not ``np.isin`` on full volumes) so the app stays responsive.
        """
        if not self.object_stats:
            QMessageBox.information(
                self, "FAR", "No objects to filter. Please run Measurement first."
            )
            return

        tgv_min = self._far_parse_size_um3(self.param_widgets["bump_minimum_size"].text())
        tgv_max = self._far_parse_size_um3(self.param_widgets["bump_maximum_size"].text())
        void_min = self._far_parse_size_um3(self.param_widgets["void_minimum_size"].text())
        void_max = self._far_parse_size_um3(self.param_widgets["void_maximum_size"].text())
        # Max 0 / unparseable must mean "no upper limit" — else everything is removed
        if tgv_max <= 0:
            tgv_max = float("inf")
        if void_max <= 0:
            void_max = float("inf")

        vx = self.param_widgets["voxel_size_x"].value()
        vy = self.param_widgets["voxel_size_y"].value()
        vz = self.param_widgets["voxel_size_z"].value()
        if self.param_widgets.get("z_stretched_4x") and self.param_widgets["z_stretched_4x"].isChecked():
            vz = vz / 4.0
        voxel_volume = vx * vy * vz
        if voxel_volume <= 0:
            voxel_volume = 1.0

        progress = QProgressDialog("Running FAR…", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(5)
        progress.show()
        QApplication.processEvents()

        try:
            tgv_labels_to_remove = []
            void_labels_to_remove = []
            rows_to_remove = []

            # Keep full pre-FAR list for bbox mask clear (need coords of removed objs)
            stats_snapshot = list(self.object_stats)
            void_stats_for_bbox = []

            for row_idx, stat in enumerate(self.object_stats):
                c1_voxels = float(stat.get("c1_volume", 0) or 0) / voxel_volume
                c2_voxels = float(stat.get("c2_volume", 0) or 0) / voxel_volume
                remove_bump = c1_voxels < tgv_min or c1_voxels > tgv_max
                remove_void = c2_voxels > 0 and (c2_voxels < void_min or c2_voxels > void_max)
                if remove_bump:
                    tgv_labels_to_remove.append(stat["label"])
                    rows_to_remove.append(row_idx)
                elif remove_void:
                    void_labels_to_remove.append(stat["label"])
                    void_stats_for_bbox.append(stat)
                    stat["c2_volume"] = 0.0
                    stat["ratio"] = 0.0

            progress.setLabelText(
                f"FAR: remove {len(tgv_labels_to_remove)} bumps, "
                f"clear void on {len(void_labels_to_remove)}…"
            )
            progress.setValue(25)
            QApplication.processEvents()
            if progress.wasCanceled():
                progress.close()
                return

            # Stats used for bbox before popping removed rows
            removed_bump_stats = [self.object_stats[i] for i in rows_to_remove]
            for row_idx in sorted(rows_to_remove, reverse=True):
                self.object_stats.pop(row_idx)

            # --- Mask edits: bbox-based (volume is often ~900×240×270) ---
            labeled = getattr(self, "labeled_class1_data", None)
            bump = getattr(self, "bump_segmentation", None)
            void = getattr(self, "void_segmentation", None)

            if labeled is not None and tgv_labels_to_remove:
                progress.setLabelText(
                    f"FAR: clearing {len(tgv_labels_to_remove)} bump mask(s)…"
                )
                progress.setValue(40)
                QApplication.processEvents()
                self._far_clear_labels_on_volume(
                    labeled,
                    tgv_labels_to_remove,
                    bump=bump,
                    void=void,
                    clear_void_only=False,
                    stats_for_bbox=removed_bump_stats or stats_snapshot,
                )
            if labeled is not None and void_labels_to_remove:
                progress.setLabelText(
                    f"FAR: clearing void on {len(void_labels_to_remove)}…"
                )
                progress.setValue(55)
                QApplication.processEvents()
                self._far_clear_labels_on_volume(
                    labeled,
                    void_labels_to_remove,
                    bump=None,
                    void=void,
                    clear_void_only=True,
                    stats_for_bbox=void_stats_for_bbox or stats_snapshot,
                )

            # Multi-layer: only clear matching labels in each layer slab (bbox path)
            ml_cache = getattr(self, "_ml_layer_cache", None) or {}
            if ml_cache and (tgv_labels_to_remove or void_labels_to_remove):
                progress.setLabelText("FAR: updating per-layer cache…")
                progress.setValue(65)
                QApplication.processEvents()
                for _lname, entry in ml_cache.items():
                    lab = entry.get("labeled_class1_data")
                    if lab is None:
                        continue
                    if tgv_labels_to_remove:
                        self._far_clear_labels_on_volume(
                            lab,
                            tgv_labels_to_remove,
                            bump=entry.get("bump"),
                            void=entry.get("void"),
                            clear_void_only=False,
                            stats_for_bbox=removed_bump_stats or stats_snapshot,
                        )
                    if void_labels_to_remove:
                        self._far_clear_labels_on_volume(
                            lab,
                            void_labels_to_remove,
                            bump=None,
                            void=entry.get("void"),
                            clear_void_only=True,
                            stats_for_bbox=void_stats_for_bbox or stats_snapshot,
                        )

            n_rem = len(tgv_labels_to_remove)
            n_void = len(void_labels_to_remove)
            n_left = len(self.object_stats)
            print(
                f"[FAR] mask done · remove_bumps={n_rem} clear_voids={n_void} "
                f"remaining={n_left} → table refresh"
            )

            progress.setLabelText(f"FAR: refreshing table ({n_left} rows)…")
            progress.setValue(80)
            QApplication.processEvents()

            self.selected_highlight_objects = []
            if hasattr(self, "object_stats_table") and self.object_stats_table is not None:
                self.object_stats_table.blockSignals(True)
                self.object_stats_table.clearSelection()
                self.object_stats_table.blockSignals(False)
            # Fast path: no processEvents, no ResizeToContents per cell
            self._refresh_stats_table(process_events=False, reindex=False)

            progress.setValue(95)
            progress.setLabelText("FAR: done")
            QApplication.processEvents()
            progress.close()

            # Defer MPR refresh so UI becomes responsive first (views can be heavy)
            def _far_finish_views():
                try:
                    for orientation in ("axial", "coronal", "sagittal"):
                        self.update_plane_view(orientation)
                except Exception as ve:
                    print(f"[FAR] view refresh: {ve}")
                total_c1 = sum(s.get("c1_volume", 0) for s in (self.object_stats or []))
                total_c2 = sum(s.get("c2_volume", 0) for s in (self.object_stats or []))
                if hasattr(self, "stats_info_label"):
                    self.stats_info_label.setText(
                        f"{len(self.object_stats or [])} objects | "
                        f"Total Bump: {total_c1:,.0f} \u03bcm\u00b3 | "
                        f"Total Void: {total_c2:,.0f} \u03bcm\u00b3"
                    )
                if hasattr(self, "bnd_btn"):
                    self.bnd_btn.setEnabled(bool(self.object_stats))
                print(f"[FAR] complete remaining={len(self.object_stats or [])}")
                QMessageBox.information(
                    self,
                    "FAR Complete",
                    f"Removed {n_rem} Bump object(s)\n"
                    f"Cleared void on {n_void} object(s)\n"
                    f"Remaining: {n_left}",
                )

            QTimer.singleShot(0, _far_finish_views)
        except Exception as e:
            try:
                progress.close()
            except Exception:
                pass
            traceback.print_exc()
            QMessageBox.critical(self, "FAR Error", f"FAR failed:\n{e}")

    def _resolve_stat_layer_name(self, stat):
        """Layer column text for one MES row (never silent empty → N/A when map exists)."""
        name = stat.get("layer_name")
        if name not in (None, "", "N/A"):
            return str(name)
        _z_off = int(getattr(self, "_measurement_start_slice", 0) or 0)
        try:
            global_z = float(stat.get("centroid_z_global", stat.get("centroid_z", 0))) + _z_off
        except (TypeError, ValueError):
            global_z = 0.0
        resolved = self._layer_name_for_global_z(global_z)
        # Cache on stat so export/B2B/Index stay consistent
        if resolved and resolved != "N/A":
            stat["layer_name"] = resolved
        return resolved

    def _refresh_stats_table(self, process_events=True, reindex=True):
        """Rebuild MES stats table from ``object_stats``.

        Critical perf note
        ------------------
        Table header was created with ``ResizeToContents``.  If left on during
        ``setItem`` for ~450 rows × 19 cols, Qt remeasures the whole table on
        **every cell** → multi-minute UI freeze ("Not Responding") at FAR /
        MES "Populating table…".  Always switch to Interactive while filling.
        """
        from PyQt5.QtWidgets import QHeaderView

        table = self.object_stats_table
        if table is None:
            return

        hdr = table.horizontalHeader()
        # Save mode, force Interactive during bulk fill (main freeze fix)
        prev_modes = []
        try:
            for c in range(table.columnCount()):
                prev_modes.append(hdr.sectionResizeMode(c))
            hdr.setSectionResizeMode(QHeaderView.Interactive)
        except Exception:
            prev_modes = []

        table.setSortingEnabled(False)
        table.blockSignals(True)
        table.setUpdatesEnabled(False)

        try:
            if self.object_stats:
                if not getattr(self, "layer_assignment_map", None):
                    self._rebuild_layer_assignment_map()
                for stat in self.object_stats:
                    self._resolve_stat_layer_name(stat)

                vx, vy = 1.0, 1.0
                if hasattr(self, "param_widgets"):
                    sw = self.param_widgets
                    if "voxel_size_x" in sw:
                        vx = sw["voxel_size_x"].value()
                        vy = sw["voxel_size_y"].value()

                already_indexed = all(
                    s.get("grid_row") is not None and s.get("grid_col") is not None
                    for s in self.object_stats
                )

                if reindex and not already_indexed:
                    layer_groups = {}
                    for stat in self.object_stats:
                        layer_name = self._resolve_stat_layer_name(stat) or "N/A"
                        layer_groups.setdefault(layer_name, []).append(stat)

                    final_list = []
                    for layer_name in sorted(
                        layer_groups.keys(), key=lambda k: (k == "N/A", k)
                    ):
                        layer_stats = layer_groups[layer_name]
                        if not layer_stats:
                            continue
                        layer_stats.sort(
                            key=lambda s: (s["centroid_y"], s["centroid_x"])
                        )
                        rows = []
                        current_row = [layer_stats[0]]
                        y_prev = layer_stats[0]["centroid_y"]
                        h_mean = (
                            np.mean([s["y_max"] - s["y_min"] for s in layer_stats])
                            if len(layer_stats) > 1
                            else 10
                        )
                        tolerance = h_mean * 0.7
                        for stat in layer_stats[1:]:
                            if abs(stat["centroid_y"] - y_prev) < tolerance:
                                current_row.append(stat)
                            else:
                                rows.append(current_row)
                                current_row = [stat]
                                y_prev = stat["centroid_y"]
                        rows.append(current_row)

                        for r_idx, row_list in enumerate(rows):
                            row_list.sort(key=lambda s: s["centroid_x"])
                            for c_idx, stat in enumerate(row_list):
                                stat["grid_row"] = r_idx + 1
                                stat["grid_col"] = c_idx + 1
                                stat["pitch_x"] = 0.0
                                stat["pitch_y"] = 0.0
                                if c_idx < len(row_list) - 1:
                                    next_stat = row_list[c_idx + 1]
                                    stat["pitch_x"] = max(
                                        0, (next_stat["x_min"] - stat["x_max"])
                                    ) * vx
                                if r_idx < len(rows) - 1:
                                    next_row = rows[r_idx + 1]
                                    w_mean = stat["x_max"] - stat["x_min"]
                                    closest_stat = min(
                                        next_row,
                                        key=lambda s: abs(
                                            s["centroid_x"] - stat["centroid_x"]
                                        ),
                                    )
                                    if (
                                        abs(
                                            closest_stat["centroid_x"]
                                            - stat["centroid_x"]
                                        )
                                        < w_mean * 1.5
                                    ):
                                        stat["pitch_y"] = max(
                                            0,
                                            (closest_stat["y_min"] - stat["y_max"]),
                                        ) * vy
                                final_list.append(stat)

                    for idx, stat in enumerate(final_list):
                        stat["row_id"] = idx + 1
                    self.object_stats = final_list
                else:
                    for idx, stat in enumerate(self.object_stats):
                        stat["row_id"] = idx + 1

            n_stats = len(self.object_stats or [])
            table.clearContents()
            table.setRowCount(n_stats)

            ng_threshold = (
                self.ng_threshold_spin.value() / 100.0
                if hasattr(self, "ng_threshold_spin")
                else 0.05
            )
            ng_count = 0
            ok_count = 0
            font_bold = QFont("Arial", 9, QFont.Bold)
            bg_ng = QColor(180, 40, 40, 60)
            bg_ok = QColor(40, 180, 60, 40)
            fg_ng = QColor(255, 100, 100)
            fg_ok = QColor(100, 255, 120)
            fg_ng2 = QColor(255, 80, 80)
            fg_ok2 = QColor(80, 255, 100)

            for row, stat in enumerate(self.object_stats or []):
                ratio = stat.get("ratio", 0)
                if ratio != float("inf"):
                    ratio_str = f"{ratio * 100:.3f}%"
                    is_ng = ratio >= ng_threshold
                else:
                    ratio_str = "N/A"
                    is_ng = True
                judgment_str = "NG" if is_ng else "OK"
                if is_ng:
                    ng_count += 1
                else:
                    ok_count += 1

                layer_name = stat.get("layer_name") or "N/A"
                if layer_name in ("", "N/A"):
                    layer_name = self._resolve_stat_layer_name(stat) or "N/A"

                texts = (
                    str(stat.get("row_id", row + 1)),
                    str(layer_name),
                    f"({stat.get('grid_row', '?')},{stat.get('grid_col', '?')})",
                    f"{stat.get('soh', 0):.3f}",
                    f"{stat.get('c1_volume', 0):,.3f}",
                    f"{stat.get('c2_volume', 0):,.3f}",
                    ratio_str,
                    judgment_str,
                    f"{stat.get('pitch_x', 0):.3f}",
                    f"{stat.get('pitch_y', 0):.3f}",
                    str(stat.get("z_min", "")),
                    str(stat.get("z_max", "")),
                    str(stat.get("y_min", "")),
                    str(stat.get("y_max", "")),
                    str(stat.get("x_min", "")),
                    str(stat.get("x_max", "")),
                    f"{float(stat.get('centroid_z', 0)):.3f}",
                    f"{float(stat.get('centroid_y', 0)):.3f}",
                    f"{float(stat.get('centroid_x', 0)):.3f}",
                )
                row_bg = bg_ng if is_ng else bg_ok
                for col, text in enumerate(texts):
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setBackground(row_bg)
                    if col == 0:
                        item.setData(Qt.UserRole, row)
                    elif col == 6:
                        item.setFont(font_bold)
                        item.setForeground(fg_ng if is_ng else fg_ok)
                    elif col == 7:
                        item.setFont(font_bold)
                        item.setTextAlignment(Qt.AlignCenter)
                        item.setForeground(fg_ng2 if is_ng else fg_ok2)
                    table.setItem(row, col, item)

            if self.object_stats and hasattr(self, "stats_info_label"):
                total_c1 = sum(s.get("c1_volume", 0) for s in self.object_stats)
                total_c2 = sum(s.get("c2_volume", 0) for s in self.object_stats)
                self.stats_info_label.setText(
                    f"{len(self.object_stats)} objects | "
                    f"OK: {ok_count} | NG: {ng_count} | "
                    f"Total Bump: {total_c1:,.0f} \u03bcm\u00b3 | "
                    f"Total Void: {total_c2:,.0f} \u03bcm\u00b3"
                )
        finally:
            # One-shot column widths (not per-cell ResizeToContents)
            try:
                table.resizeColumnsToContents()
                # Cap very wide numeric columns
                for c in range(table.columnCount()):
                    w = table.columnWidth(c)
                    if w > 140:
                        table.setColumnWidth(c, 140)
                    elif w < 36:
                        table.setColumnWidth(c, 36)
            except Exception:
                pass
            if prev_modes:
                try:
                    for c, mode in enumerate(prev_modes):
                        # Never restore ResizeToContents for bulk tables
                        if mode == QHeaderView.ResizeToContents:
                            hdr.setSectionResizeMode(c, QHeaderView.Interactive)
                        else:
                            hdr.setSectionResizeMode(c, mode)
                except Exception:
                    pass
            else:
                try:
                    hdr.setSectionResizeMode(QHeaderView.Interactive)
                except Exception:
                    pass
            table.setUpdatesEnabled(True)
            table.blockSignals(False)
            table.setSortingEnabled(True)
            if process_events:
                QApplication.processEvents()
            
    def _on_threshold_changed(self, *args):
        """Re-classify and re-color the table when threshold changes without rebuilding it."""
        if not self.object_stats:
            return
            
        # Prevent massive UI lag during update
        self.object_stats_table.setSortingEnabled(False)
        self.object_stats_table.blockSignals(True)
        
        ng_threshold = self.ng_threshold_spin.value() / 100.0 if hasattr(self, 'ng_threshold_spin') else 0.05
        ng_count = 0
        ok_count = 0
        
        for row, stat in enumerate(self.object_stats):
            is_ng = stat['ratio'] >= ng_threshold if stat['ratio'] != float('inf') else True
            judgment_str = "NG" if is_ng else "OK"
            
            if is_ng:
                ng_count += 1
                row_bg = QColor(180, 40, 40, 60)
                col4_fg = QColor(255, 100, 100)
                col5_fg = QColor(255, 80, 80)
            else:
                ok_count += 1
                row_bg = QColor(40, 180, 60, 40)
                col4_fg = QColor(100, 255, 120)
                col5_fg = QColor(80, 255, 100)
                
            # Update background for all columns in this row
            for col in range(18):
                item = self.object_stats_table.item(row, col)
                if item:
                    item.setBackground(row_bg)
                    # Ratio
                    if col == 6:
                        item.setForeground(col4_fg)
                    # Judgment
                    elif col == 7:
                        item.setText(judgment_str)
                        item.setForeground(col5_fg)
                        
        self.object_stats_table.blockSignals(False)
        self.object_stats_table.setSortingEnabled(True)
        
        # Update OK/NG counts label
        total_c1 = sum(s['c1_volume'] for s in self.object_stats)
        total_c2 = sum(s['c2_volume'] for s in self.object_stats)
        self.stats_info_label.setText(
            f"{len(self.object_stats)} objects | "
            f"OK: {ok_count} | NG: {ng_count} | "
            f"Total Bump: {total_c1:,.0f} \u03bcm\u00b3 | Total Void: {total_c2:,.0f} \u03bcm\u00b3"
        )
            
    def save_config(self):
        """Save current configuration to file"""
        if not self.config:
            QMessageBox.warning(self, "Warning", "No configuration loaded")
            return
            
        start_dir = ""
        try:
            start_dir = str(config_dir())
        except Exception:
            start_dir = ""
        if self.config_path:
            start_dir = self.config_path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Config File",
            start_dir,
            "Config Files (*.txt);;All Files (*.*)",
        )
        
        if file_path:
            try:
                self.update_config_from_parameters()

                # Prefer saving under app config/ when user picks that folder
                # Write recipe file (parameters only — I/O paths set by app at runtime)
                with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write("# Bump + VOID DETECTION CONFIG FILE\n")
                    f.write("# Recipe parameters only — INPUT/OUTPUT paths are set by the app at runtime.\n\n")
                    f.write("# === RECIPE META ===\n")
                    f.write(f"INPUT_MODE = {self.input_mode}\n")
                    _test_name = self.config.testName
                    if isinstance(_test_name, bytes):
                        _test_name = _test_name.decode('utf-8', errors='replace').rstrip('\x00')
                    f.write(f"TEST_NAME = {_test_name}\n\n")
                    
                    f.write("# === BUMP DETECTION ===\n")
                    f.write(f"BUMP_THRESHOLD_WEIGHT = {self.config.bumpThresholdWeight}\n")
                    f.write(f"BUMP_CLEAN_OPEN_RADIUS_X = {self.config.bumpCleanOpenX}\n")
                    f.write(f"BUMP_CLEAN_OPEN_RADIUS_Y = {self.config.bumpCleanOpenY}\n")
                    f.write(f"BUMP_CLEAN_OPEN_RADIUS_Z = {self.config.bumpCleanOpenZ}\n")
                    f.write(f"BUMP_BG_OPEN_RADIUS_X = {self.config.bumpBgOpenX}\n")
                    f.write(f"BUMP_BG_OPEN_RADIUS_Y = {self.config.bumpBgOpenY}\n")
                    f.write(f"BUMP_BG_OPEN_RADIUS_Z = {self.config.bumpBgOpenZ}\n")
                    if hasattr(self.config, 'bumpFillHoleCloseX'):
                        f.write(
                            f"BUMP_FILL_HOLE_CLOSE_RADIUS_X = {self.config.bumpFillHoleCloseX}\n"
                        )
                        f.write(
                            f"BUMP_FILL_HOLE_CLOSE_RADIUS_Y = {self.config.bumpFillHoleCloseY}\n"
                        )
                        f.write(
                            f"BUMP_FILL_HOLE_CLOSE_RADIUS_Z = {self.config.bumpFillHoleCloseZ}\n"
                        )
                    f.write("\n")
                    
                    f.write("# === VOID DETECTION ===\n")
                    f.write(f"VOID_THRESHOLD_WEIGHT = {self.config.voidThresholdWeight}\n")
                    f.write(f"VOID_PRE_EM_OPEN_RADIUS_X = {self.config.openEmVoidX}\n")
                    f.write(f"VOID_PRE_EM_OPEN_RADIUS_Y = {self.config.openEmVoidY}\n")
                    f.write(f"VOID_PRE_EM_OPEN_RADIUS_Z = {self.config.openEmVoidZ}\n")
                    f.write(f"VOID_TOPHAT_RADIUS_X = {self.config.closeResidueX}\n")
                    f.write(f"VOID_TOPHAT_RADIUS_Y = {self.config.closeResidueY}\n")
                    f.write(f"VOID_TOPHAT_RADIUS_Z = {self.config.closeResidueZ}\n")
                    f.write(f"VOID_NOISE_REMOVAL_OPEN_RADIUS_X = {self.config.openSmallDotsX}\n")
                    f.write(f"VOID_NOISE_REMOVAL_OPEN_RADIUS_Y = {self.config.openSmallDotsY}\n")
                    f.write(f"VOID_NOISE_REMOVAL_OPEN_RADIUS_Z = {self.config.openSmallDotsZ}\n")
                    f.write(f"TGV_SHRUNK_ERODE_RADIUS_X = {self.config.erodeBumpX}\n")
                    f.write(f"TGV_SHRUNK_ERODE_RADIUS_Y = {self.config.erodeBumpY}\n")
                    f.write(f"TGV_SHRUNK_ERODE_RADIUS_Z = {self.config.erodeBumpZ}\n\n")
                    
                    f.write("# === BLANK SLICES ===\n")
                    f.write(f"TOP_AIR_START_SLICE_NUM = {self.config.blankStart1}\n")
                    f.write(f"TOP_AIR_END_SLICE_NUM = {self.config.blankEnd1}\n")
                    f.write(f"BTM_AIR_START_SLICE_NUM = {self.config.blankStart2}\n")
                    f.write(f"BTM_AIR_END_SLICE_NUM = {self.config.blankEnd2}\n\n")
                    
                    f.write("# === OPTIONS ===\n")
                    f.write(f"SAVE_TGV_DEBUG_IMG = {'true' if self.config.saveBumpIntermediate else 'false'}\n")
                    f.write(f"SAVE_VOID_DEBUG_IMG = {'true' if self.config.saveVoidIntermediate else 'false'}\n")
                    f.write(f"SHOW_FLATTENED_TGV_VOID = {'true' if self.config.showResult else 'false'}\n")
                    prof_on = self._is_dll_profiling_enabled()
                    f.write(f"ENABLE_DLL_PROFILING = {'true' if prof_on else 'false'}\n\n")
                    
                    f.write("# === CC3D Connected Component Labeling ===\n")
                    conn_options = [6, 18, 26]
                    conn_val = 26
                    if 'cc3d_connectivity' in self.param_widgets:
                        conn_val = conn_options[self.param_widgets['cc3d_connectivity'].currentIndex()]
                    f.write(f"CC3D_CONNECTIVITY = {conn_val}\n")
                    
                    min_vox = 0
                    if 'cc3d_min_voxels' in self.param_widgets:
                        min_vox = self.param_widgets['cc3d_min_voxels'].value()
                    f.write(f"CC3D_MIN_VOXELS = {min_vox}\n")
                    
                    use_gpu = True
                    if 'cc3d_use_gpu' in self.param_widgets:
                        use_gpu = self.param_widgets['cc3d_use_gpu'].isChecked()
                    f.write(f"CC3D_USE_GPU = {'true' if use_gpu else 'false'}\n\n")
                    
                    f.write("# === MEASUREMENT ===\n")
                    f.write(f"VOXEL_SIZE_X = {self.param_widgets['voxel_size_x'].value()}\n")
                    f.write(f"VOXEL_SIZE_Y = {self.param_widgets['voxel_size_y'].value()}\n")
                    f.write(f"VOXEL_SIZE_Z = {self.param_widgets['voxel_size_z'].value()}\n")
                    z4x_enabled = self.param_widgets.get('z_stretched_4x') and self.param_widgets['z_stretched_4x'].isChecked()
                    f.write(f"Z_STRETCHED_4X = {'true' if z4x_enabled else 'false'}\n")
                    f.write(f"BUMP_MINIMUM_SIZE = {self.param_widgets['bump_minimum_size'].text()}\n")
                    f.write(f"BUMP_MAXIMUM_SIZE = {self.param_widgets['bump_maximum_size'].text()}\n")
                    f.write(f"VOID_MINIMUM_SIZE = {self.param_widgets['void_minimum_size'].text()}\n")
                    f.write(f"VOID_MAXIMUM_SIZE = {self.param_widgets['void_maximum_size'].text()}\n")
                    f.write(f"NG_THRESHOLD = {self.ng_threshold_spin.value()}\n")

                    # Enhancement (ONNX) — includes Enable flag for the Enhance tab checkbox
                    f.write("\n# ── Enhancement (ONNX Model) ─────────────────\n")
                    enh_enabled = bool(getattr(self, "enhancement_enabled", False))
                    if hasattr(self, "enh_enable_check"):
                        enh_enabled = self.enh_enable_check.isChecked()
                    f.write(f"ENHANCEMENT_ENABLED = {'true' if enh_enabled else 'false'}\n")

                    model_path = getattr(self, "enhancement_model_path", "") or ""
                    if hasattr(self, "enh_model_input"):
                        model_path = self.enh_model_input.text().strip()
                    f.write(f'ENHANCED_MODEL_PATH = "{model_path}"\n')

                    trt_cache = getattr(self, "enhancement_trt_cache", "./TRT_Cache") or "./TRT_Cache"
                    if hasattr(self, "enh_trt_input"):
                        trt_cache = self.enh_trt_input.text().strip() or "./TRT_Cache"
                    f.write(f"ENHANCED_TRT_CACHE_PATH = {trt_cache}\n")

                    use_enh_gpu = bool(getattr(self, "enhancement_use_gpu", True))
                    if hasattr(self, "enh_gpu_check"):
                        use_enh_gpu = self.enh_gpu_check.isChecked()
                    f.write(f"ENHANCED_USE_GPU = {'true' if use_enh_gpu else 'false'}\n")

                    enh_gpu_id = int(getattr(self, "enhancement_gpu_id", 0) or 0)
                    if hasattr(self, "enh_gpuid_spin"):
                        enh_gpu_id = int(self.enh_gpuid_spin.value())
                    f.write(f"ENHANCED_GPU_DEVICE_ID = {enh_gpu_id}\n")

                    # Enhancement mode: full | layers  (UI: Full Volume / Selected Layers Only)
                    enh_mode = "layers" if (
                        hasattr(self, "enh_mode_layers") and self.enh_mode_layers.isChecked()
                    ) else getattr(self, "enhancement_mode", "full")
                    if enh_mode not in ("full", "layers"):
                        enh_mode = "full"
                    f.write("# Enhancement mode: full = all slices; layers = selected LAYER_* ranges only\n")
                    f.write(f"ENHANCEMENT_MODE = {enh_mode}\n")

                    enh_save = bool(getattr(self, "enhancement_save_output", True))
                    if hasattr(self, "enh_save_check"):
                        enh_save = self.enh_save_check.isChecked()
                    f.write(f"ENHANCEMENT_SAVE_OUTPUT = {'true' if enh_save else 'false'}\n")
                    
                    # Write layer definitions if any exist
                    if self.layer_definitions:
                        f.write("\n# === LAYER DEFINITIONS ===\n")
                        f.write(f"# Format: LAYER_N = name,z_start,z_end,selected\n")
                        f.write(f"NUM_LAYERS = {len(self.layer_definitions)}\n")
                        for i, layer in enumerate(self.layer_definitions):
                            selected_str = "true" if layer['selected'] else "false"
                            f.write(f"LAYER_{i} = {layer['name']},{layer['z_start']},{layer['z_end']},{selected_str}\n")
                
                self.config_path = file_path
                self.sync_config_path_ui(file_path)
                QMessageBox.information(self, "Success", f"Config saved to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save config: {str(e)}")
                
    def reset_config(self):
        """Reset configuration to defaults"""
        if self.config:
            bumpvoid._dll.BumpVoid_InitConfig(ctypes.byref(self.config))
            self.populate_parameters_from_config()
            self.info_label.setText("Config reset to defaults")
            
    def save_results(self):
        """Save segmentation results"""
        if self.bump_segmentation is None and self.void_segmentation is None:
            QMessageBox.warning(self, "Warning", "No results to save")
            return
            
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            try:
                if self.bump_segmentation is not None:
                    bump_path = Path(folder) / "bump_result.tif"
                    io.imsave(str(bump_path), self.bump_segmentation)
                    
                if self.void_segmentation is not None:
                    void_path = Path(folder) / "void_result.tif"
                    io.imsave(str(void_path), self.void_segmentation)
                
                QMessageBox.information(self, "Success", f"Results saved to:\n{folder}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")
                
    def check_ready_state(self):
        """Check if ready to run inspection"""
        if getattr(self, 'input_mode', 'test_1layer') == 'multi_layer':
            ready = (self.dll_path is not None and 
                     self.config is not None and 
                     len(self.multi_layer_files) > 0)
        else:
            # Both 'test_1layer' and 'single' require DLL + Config + Volume
            ready = (self.dll_path is not None and 
                     self.config is not None and 
                     self.volume_data is not None)
            
        self.inspect_btn.setEnabled(ready)
        # Keep test-1-layer info panel in sync
        if hasattr(self, 'test1_shape_lbl'):
            self._update_test1_volume_info()
        
    
    def _window_level_to_uint8(self, slice_data, window, level):
        """Map scalar slice through W/L to display uint8 (H,W) — same as 3D Viewer."""
        arr = np.asarray(slice_data, dtype=np.float32)
        half = max(float(window), 1e-6) * 0.5
        lo = float(level) - half
        hi = float(level) + half
        if hi <= lo:
            hi = lo + 1.0
        out = (arr - lo) * (255.0 / (hi - lo))
        return np.clip(out, 0, 255).astype(np.uint8)

    def _compose_teaching_mpr_overlay_rgb(
        self,
        slice_data,
        window,
        level,
        bump_slice,
        void_slice,
        bump_on,
        void_on,
        opacity,
        orientation,
        slice_idx,
    ):
        """Alpha-blend Bump/Void masks onto W/L-mapped CT → RGB uint8 (H,W,3).

        Matches 3D Viewer ``_compose_mpr_overlay_rgb``:
          - background keeps pure CT gray (no black film from a second actor)
          - hard mask edges (no VTK linear interpolation on a separate overlay)
          - Void overwrites Bump on shared voxels
        """
        gray = self._window_level_to_uint8(slice_data, window, level)
        h, w = gray.shape[:2]
        rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        a = float(np.clip(opacity, 0.0, 1.0))

        c1 = getattr(self, "class1_color", None) or [1.0, 1.0, 0.0]
        c2 = getattr(self, "class2_color", None) or [1.0, 0.0, 0.0]
        c1_color = np.array([float(v * 255) for v in c1], dtype=np.float32)
        c2_color = np.array([float(v * 255) for v in c2], dtype=np.float32)

        def _mask_positive(src):
            if src is None:
                return None
            arr = np.asarray(src)
            if arr.ndim > 2:
                arr = arr[..., 0]
            if arr.shape[:2] != (h, w):
                return None
            # Binary 0/255, label 128/255, or any positive foreground
            return arr > 0

        m_bump = None
        m_void = None
        if bump_on and bump_slice is not None:
            m_bump = _mask_positive(bump_slice)
            if m_bump is not None and np.any(m_bump):
                rgb[m_bump] = (1.0 - a) * rgb[m_bump] + a * c1_color

        if void_on and void_slice is not None:
            m_void = _mask_positive(void_slice)
            if m_void is not None and np.any(m_void):
                rgb[m_void] = (1.0 - a) * rgb[m_void] + a * c2_color

        # MES selection highlight — 100% Viewer SoT (cyan fill + white outline only).
        # No dim of non-selected objects (Viewer does not dim on MPR).
        if getattr(self, "selected_highlight_objects", None) and getattr(
            self, "labeled_class1_data", None
        ) is not None:
            try:
                labeled_slice = _shared_labeled_slice(
                    self.labeled_class1_data,
                    orientation,
                    slice_idx,
                    reverse_z=bool(getattr(self, "reverse_z", False)),
                )
                if labeled_slice is not None and labeled_slice.shape == (h, w):
                    hl_rgb = _shared_build_mes_hl_rgb(
                        labeled_slice,
                        self.selected_highlight_objects,
                        getattr(self, "highlight_color", [0.0, 1.0, 1.0]),
                        outline_iterations=3,
                    )
                    # Same effective opacity as Viewer vtkImageActor.SetOpacity(0.7)
                    rgb = _shared_blend_mes_hl(rgb, hl_rgb, opacity=0.7)
            except (IndexError, ValueError, TypeError) as e:
                print(f"[MES HL] Teaching MPR highlight skipped: {e}")

        return np.clip(rgb, 0, 255).astype(np.uint8)

    def update_plane_view(self, orientation, preserve_camera=False):
        """Update one MPR pane — parity with 3D Viewer ``render_slice``.

        Key parity fixes vs old Teaching path:
          - bake CT + mask into one RGB ImageActor (no second actor / black film)
          - nearest-neighbor interpolation (crisp mask pixels, no bilinear blur)
          - viewport-aspect camera fit (not crude max(h,w)*0.55)
          - use cached per-orientation window/level when available
        """
        if self.volume_data is None:
            return

        renderer = getattr(self, f"{orientation}_renderer")
        slice_idx = self.current_slices[orientation]

        # Save camera state before rebuild
        if preserve_camera and hasattr(self, "camera_states") and isinstance(
            self.camera_states, dict
        ):
            camera = renderer.GetActiveCamera()
            self.camera_states[orientation] = {
                "position": camera.GetPosition(),
                "focal_point": camera.GetFocalPoint(),
                "view_up": camera.GetViewUp(),
                "parallel_scale": camera.GetParallelScale(),
            }

        actual_z = slice_idx
        if orientation == "axial":
            if getattr(self, "reverse_z", False):
                actual_z = self.volume_data.shape[0] - 1 - slice_idx
            slice_data = self.volume_data[actual_z, :, :]
            bump_slice = (
                self.bump_segmentation[actual_z, :, :]
                if self.bump_segmentation is not None
                else None
            )
            void_slice = (
                self.void_segmentation[actual_z, :, :]
                if self.void_segmentation is not None
                else None
            )
        elif orientation == "coronal":
            slice_data = np.flipud(self.volume_data[:, slice_idx, :])
            bump_slice = (
                np.flipud(self.bump_segmentation[:, slice_idx, :])
                if self.bump_segmentation is not None
                else None
            )
            void_slice = (
                np.flipud(self.void_segmentation[:, slice_idx, :])
                if self.void_segmentation is not None
                else None
            )
        else:
            slice_data = np.transpose(self.volume_data[:, :, slice_idx])
            bump_slice = (
                np.transpose(self.bump_segmentation[:, :, slice_idx])
                if self.bump_segmentation is not None
                else None
            )
            void_slice = (
                np.transpose(self.void_segmentation[:, :, slice_idx])
                if self.void_segmentation is not None
                else None
            )

        h, w = slice_data.shape
        renderer.RemoveAllViewProps()

        # Window / level — prefer cached W/L (same as Viewer); fallback to slice min/max
        wl = getattr(self, "window_level", None) or {}
        if isinstance(wl, dict) and wl.get(orientation) is not None:
            base_window, base_level = wl[orientation]
            window = float(base_window)
            level = float(base_level)
        else:
            v_min, v_max = float(slice_data.min()), float(slice_data.max())
            window = v_max - v_min if v_max > v_min else 1.0
            level = v_min + window / 2.0

        bump_check = getattr(self, f"{orientation}_overlay_bump", None)
        void_check = getattr(self, f"{orientation}_overlay_void", None)
        opacity_slider = getattr(self, f"{orientation}_opacity_slider", None)
        bump_enabled = bump_check.isChecked() if bump_check else False
        void_enabled = void_check.isChecked() if void_check else False
        opacity = (opacity_slider.value() / 100.0) if opacity_slider else 0.7
        opacity = float(np.clip(opacity, 0.0, 1.0))

        want_overlay = (
            (bump_enabled or void_enabled)
            and opacity > 0.0
            and (bump_slice is not None or void_slice is not None)
        )

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(w, h, 1)
        vtk_image.SetSpacing(1.0, 1.0, 1.0)
        vtk_image.SetOrigin(0.0, 0.0, 0.0)

        image_actor = vtk.vtkImageActor()
        image_actor.GetMapper().SetInputData(vtk_image)
        prop = image_actor.GetProperty()
        # Critical for sharp mask edges when zoomed (Viewer does the same)
        prop.SetInterpolationTypeToNearest()

        if want_overlay:
            display_rgb = self._compose_teaching_mpr_overlay_rgb(
                slice_data,
                window,
                level,
                bump_slice,
                void_slice,
                bump_enabled,
                void_enabled,
                opacity,
                orientation,
                slice_idx,
            )
            overlay_transposed = np.transpose(display_rgb, (1, 0, 2))
            flat_rgb = np.ascontiguousarray(
                overlay_transposed.reshape(-1, 3, order="F")
            )
            vtk_array = numpy_support.numpy_to_vtk(
                flat_rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
            )
            vtk_array.SetNumberOfComponents(3)
            # Identity W/L — pixels already display-mapped (must stay 255/127.5)
            prop.SetColorWindow(255.0)
            prop.SetColorLevel(127.5)
        else:
            slice_transposed = np.transpose(slice_data, (1, 0))
            flat_data = np.ascontiguousarray(slice_transposed.flatten("F"))
            if slice_data.dtype == np.uint8:
                vtk_array = numpy_support.numpy_to_vtk(
                    flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
                )
            elif slice_data.dtype == np.uint16:
                vtk_array = numpy_support.numpy_to_vtk(
                    flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_SHORT
                )
            else:
                vtk_array = numpy_support.numpy_to_vtk(
                    flat_data.astype(np.float32),
                    deep=True,
                    array_type=vtk.VTK_FLOAT,
                )
            prop.SetColorWindow(window)
            prop.SetColorLevel(level)

        vtk_image.GetPointData().SetScalars(vtk_array)
        vtk_image.Modified()
        renderer.AddActor(image_actor)

        # Camera — viewport-aspect fit like 3D Viewer (not max(h,w)*0.55)
        if (
            preserve_camera
            and hasattr(self, "camera_states")
            and isinstance(self.camera_states, dict)
            and self.camera_states.get(orientation) is not None
        ):
            state = self.camera_states[orientation]
            camera = renderer.GetActiveCamera()
            camera.SetPosition(state["position"])
            camera.SetFocalPoint(state["focal_point"])
            camera.SetViewUp(state["view_up"])
            camera.SetParallelScale(state["parallel_scale"])
        else:
            camera = renderer.GetActiveCamera()
            camera.ParallelProjectionOn()
            center_x, center_y = w / 2.0, h / 2.0
            camera.SetPosition(center_x, center_y, 1000)
            camera.SetFocalPoint(center_x, center_y, 0)
            camera.SetViewUp(0, 1, 0)

            widget_for_fit = getattr(self, f"{orientation}_widget", None)
            if widget_for_fit is not None:
                vp_size = widget_for_fit.GetRenderWindow().GetSize()
                vp_w_px = max(vp_size[0], 1)
                vp_h_px = max(vp_size[1], 1)
                vp_aspect = vp_w_px / float(vp_h_px)
                img_aspect = w / max(h, 1)
                if img_aspect > vp_aspect:
                    parallel_scale = (w / vp_aspect) * 0.5
                else:
                    parallel_scale = h * 0.5
            else:
                parallel_scale = max(h, w) * 0.5
            camera.SetParallelScale(parallel_scale)
            renderer.ResetCameraClippingRange()

        # Crosshair (persistent actors — Dragonfly hover/grab like 3D Viewer)
        if getattr(self, "crosshair_enabled", False):
            if hasattr(self, "_persistent_crosshair"):
                self._persistent_crosshair[orientation] = {}
            self.update_2d_crosshair(orientation)

        # ROI overlay on axial (XY)
        if orientation == "axial":
            self.draw_roi_overlay(renderer, h, w)

        # Layer define bars on sagittal (YZ)
        if orientation == "sagittal" and getattr(self, 'layer_define_active', False):
            self._draw_layer_define_bars(renderer, h, w)

        # Object indices if enabled
        if (
            getattr(self, "show_indices_check", None)
            and self.show_indices_check.isChecked()
            and self.object_stats
        ):
            if orientation == "axial":
                self.draw_object_labels(
                    renderer, actual_z, orientation="axial", slice_idx=actual_z
                )
            elif orientation == "coronal":
                self.draw_object_labels(
                    renderer, 0, orientation="coronal", slice_idx=slice_idx
                )
            elif orientation == "sagittal":
                self.draw_object_labels(
                    renderer, 0, orientation="sagittal", slice_idx=slice_idx
                )

        # Scale bar — always on, updates with zoom
        self.add_ruler_overlay(renderer, orientation, (h, w))

        widget = getattr(self, f"{orientation}_widget", None)
        if widget:
            widget.GetRenderWindow().Render()

    def on_slice_changed(self, orientation, value):
        """Handle slice slider change — keep zoom/pan, sync crosshair (3D Viewer)."""
        newX, newY, newZ = list(self.crosshair_position)
        if orientation == 'axial':
            newZ = value
        elif orientation == 'coronal':
            newY = value
        else:
            newX = value
        # updatePoint preserves camera on re-render and lightweight-updates other views
        if self.volume_data is not None:
            self.updatePoint(newX, newY, newZ)
        else:
            self.current_slices[orientation] = value
            label = getattr(self, f'{orientation}_slice_label', None)
            slider = getattr(self, f'{orientation}_slice_slider', None)
            if label and slider:
                label.setText(f"{value} / {slider.maximum()}")
    
    def init_vtk_widgets(self):
        """Initialize VTK interactors, zoom + crosshair observers (3D Viewer parity)."""
        for orientation in ['axial', 'coronal', 'sagittal']:
            widget = getattr(self, f'{orientation}_widget')
            interactor = widget.GetRenderWindow().GetInteractor()
            interactor.Initialize()
            self._setup_mpr_interaction_observers(widget, orientation)

    def _setup_mpr_interaction_observers(self, vtk_widget, orientation):
        """Ruler refresh + Fiji wheel zoom + Dragonfly crosshair (same as 3D Viewer)."""
        if getattr(self, f'_{orientation}_mpr_obs_ready', False):
            return
        interactor = vtk_widget.GetRenderWindow().GetInteractor()

        if not hasattr(self, 'active_styles'):
            self.active_styles = {}
        if not hasattr(self, '_dummy_style'):
            self._dummy_style = vtk.vtkInteractorStyle()

        def on_interaction(caller, event, o=orientation, w=vtk_widget):
            if self.volume_data is None:
                return
            renderer = getattr(self, f'{o}_renderer')
            z, y, x = self.volume_data.shape
            if o == 'axial':
                h, ww = y, x
            elif o == 'coronal':
                h, ww = z, x
            else:
                h, ww = z, y
            self.add_ruler_overlay(renderer, o, (h, ww))
            # Persist camera after pan/zoom via VTK style
            self._save_camera_state(o)
            w.GetRenderWindow().Render()

        def on_left_button_press(caller, event, o=orientation, w=vtk_widget):
            if self.volume_data is None or not self.crosshair_enabled:
                return
            # ROI drawing owns axial LMB when active
            if self.roi_selection_active and o == 'axial':
                return
            x, y = caller.GetEventPosition()
            size = w.GetRenderWindow().GetSize()
            qt_pos = QPoint(int(x), int(size[1] - y))
            ch_hit = self._get_crosshair_hit(qt_pos, o)
            if ch_hit is None:
                return
            if o not in self.active_styles:
                self.active_styles[o] = caller.GetInteractorStyle()
            caller.SetInteractorStyle(self._dummy_style)
            self._crosshair_drag_mode = ch_hit[1]
            self._crosshair_hover = ch_hit
            self._is_dragging_crosshair = True
            self.handle_crosshair_click(o, qt_pos, mode=ch_hit[1])
            self.update_2d_crosshair(o)

        def on_mouse_move(caller, event, o=orientation, w=vtk_widget):
            if self.volume_data is None:
                return
            x, y = caller.GetEventPosition()
            size = w.GetRenderWindow().GetSize()
            qt_pos = QPoint(int(x), int(size[1] - y))
            self.update_pixel_value(o, qt_pos)
            if not self.crosshair_enabled:
                return
            if getattr(self, '_is_dragging_crosshair', False):
                mode = getattr(self, '_crosshair_drag_mode', 'center') or 'center'
                self.handle_crosshair_click(o, qt_pos, mode=mode)
                return
            ch_hit = self._get_crosshair_hit(qt_pos, o)
            if ch_hit != getattr(self, '_crosshair_hover', None):
                self._crosshair_hover = ch_hit
                self.update_2d_crosshair(o)
                if ch_hit:
                    if ch_hit[1] == 'center':
                        w.setCursor(Qt.SizeAllCursor)
                    elif ch_hit[1] == 'h':
                        w.setCursor(Qt.SizeVerCursor)
                    else:
                        w.setCursor(Qt.SizeHorCursor)
                else:
                    w.unsetCursor()

        def on_left_button_release(caller, event, o=orientation):
            if getattr(self, '_is_dragging_crosshair', False):
                self._is_dragging_crosshair = False
                self._crosshair_drag_mode = None
            self._restore_interactor_style(o)

        def on_mouse_wheel_forward(caller, event, o=orientation, w=vtk_widget):
            if self.volume_data is None:
                return
            if caller.GetControlKey():
                self._scroll_change_slice(o, delta=+1)
            else:
                self._fiji_zoom_at_cursor(w, o, zoom_in=True)

        def on_mouse_wheel_backward(caller, event, o=orientation, w=vtk_widget):
            if self.volume_data is None:
                return
            if caller.GetControlKey():
                self._scroll_change_slice(o, delta=-1)
            else:
                self._fiji_zoom_at_cursor(w, o, zoom_in=False)

        interactor.AddObserver("InteractionEvent", on_interaction, 10.0)
        interactor.AddObserver("EndInteractionEvent", on_interaction, 10.0)
        interactor.AddObserver("LeftButtonPressEvent", on_left_button_press, 10.0)
        interactor.AddObserver("MouseMoveEvent", on_mouse_move, 10.0)
        interactor.AddObserver("LeftButtonReleaseEvent", on_left_button_release, 10.0)
        interactor.AddObserver("MouseWheelForwardEvent", on_mouse_wheel_forward, 100.0)
        interactor.AddObserver("MouseWheelBackwardEvent", on_mouse_wheel_backward, 100.0)
        setattr(self, f'_{orientation}_mpr_obs_ready', True)

    def _restore_interactor_style(self, orientation):
        """Restore VTK interactor style after dummy style used during crosshair drag."""
        if not hasattr(self, 'active_styles'):
            return
        saved = self.active_styles.pop(orientation, None)
        if saved is None:
            return
        widget = getattr(self, f'{orientation}_widget', None)
        if widget is None:
            return
        interactor = widget.GetRenderWindow().GetInteractor()
        if interactor.GetInteractorStyle() is getattr(self, '_dummy_style', None):
            interactor.SetInteractorStyle(saved)

    def _save_camera_state(self, orientation):
        renderer = getattr(self, f'{orientation}_renderer', None)
        if renderer is None:
            return
        camera = renderer.GetActiveCamera()
        if not hasattr(self, 'camera_states') or not isinstance(self.camera_states, dict):
            self.camera_states = {}
        self.camera_states[orientation] = {
            'position': camera.GetPosition(),
            'focal_point': camera.GetFocalPoint(),
            'view_up': camera.GetViewUp(),
            'parallel_scale': camera.GetParallelScale(),
        }

    # ── Fiji-style zoom helpers (ported from 3D Viewer) ─────────────────
    def _fiji_zoom_at_cursor(self, vtk_widget, orientation, zoom_in=True, display_pos=None):
        """Zoom 2D parallel camera at cursor — Fiji/ImageJ scroll-wheel behaviour."""
        ZOOM_FACTOR = 1.15

        renderer = getattr(self, f'{orientation}_renderer', None)
        if renderer is None:
            return
        camera = renderer.GetActiveCamera()
        if not camera.GetParallelProjection():
            return

        if display_pos is not None:
            mx, my = display_pos
        else:
            interactor = vtk_widget.GetRenderWindow().GetInteractor()
            mx, my = interactor.GetEventPosition()

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToDisplay()
        coord.SetValue(float(mx), float(my), 0.0)
        world_before = list(coord.GetComputedWorldValue(renderer))

        old_scale = camera.GetParallelScale()
        new_scale = old_scale / ZOOM_FACTOR if zoom_in else old_scale * ZOOM_FACTOR
        new_scale = max(1.0, min(new_scale, 1e6))
        camera.SetParallelScale(new_scale)

        coord.SetValue(float(mx), float(my), 0.0)
        world_after = list(coord.GetComputedWorldValue(renderer))

        pos = list(camera.GetPosition())
        fp = list(camera.GetFocalPoint())
        for i in range(3):
            delta = world_before[i] - world_after[i]
            pos[i] += delta
            fp[i] += delta
        camera.SetPosition(*pos)
        camera.SetFocalPoint(*fp)
        renderer.ResetCameraClippingRange()

        self._save_camera_state(orientation)

        if self.volume_data is not None:
            z, y, x = self.volume_data.shape
            h, w = (y, x) if orientation == 'axial' else (z, x) if orientation == 'coronal' else (z, y)
            self.add_ruler_overlay(renderer, orientation, (h, w))

        vtk_widget.GetRenderWindow().Render()

    def _scroll_change_slice(self, orientation, delta):
        """Ctrl+scroll → change slice (same as 3D Viewer)."""
        if self.volume_data is None:
            return
        vol_z, vol_y, vol_x = self.volume_data.shape
        newX, newY, newZ = self.crosshair_position
        if orientation == 'axial':
            newZ = max(0, min(newZ + delta, vol_z - 1))
        elif orientation == 'coronal':
            newY = max(0, min(newY + delta, vol_y - 1))
        else:
            newX = max(0, min(newX + delta, vol_x - 1))
        self.updatePoint(newX, newY, newZ)

    # ==================== ONLINE MODE PROGRAMMATIC API ====================


    # --- Online / public API (moved from seg_mpr boundary fix) ---
    """Mixin: SegmentationTab MPR slice views, crosshair, eventFilter.

    Extracted from inno3d/tabs/teaching.py Phase 4.3.
    """

    def load_dll_from_path(self, folder):
        """Load DLL folder programmatically — used by startup + Online mode."""
        try:
            self.apply_dll_folder(folder, persist=True, force=True)
            return True
        except Exception as e:
            print(f"[SegmentationTab] Failed to load DLL from {folder}: {e}")
            return False

    def load_config_from_path(self, config_path):
        """Load config file programmatically (no file dialog)  used by Online mode"""
        try:
            self.config = bumpvoid.load_config(config_path)
            self.config_path = config_path
            self.sync_config_path_ui(config_path)
            self.populate_parameters_from_config()
            # Apply pending input path if volume was loaded before config
            if self._pending_input_path:
                self.config.inputPath = self._pending_input_path.encode('utf-8')
                self._pending_input_path = None

            # Manually parse Voxel Sizes and Measurement Parameters from config file
            # UTF-8 first (ship configs use box-drawing / non-ASCII comments)
            try:
                content = read_text_auto(config_path)
                vx_match = re.search(r'VOXEL_SIZE_X\s*=\s*([\d.]+)', content)
                vy_match = re.search(r'VOXEL_SIZE_Y\s*=\s*([\d.]+)', content)
                vz_match = re.search(r'VOXEL_SIZE_Z\s*=\s*([\d.]+)', content)

                bumpmin_match = re.search(r'BUMP_MINIMUM_SIZE\s*=\s*([0-9.xX]+)', content)
                bumpmax_match = re.search(r'BUMP_MAXIMUM_SIZE\s*=\s*([0-9.xX]+)', content)
                voidmin_match = re.search(r'VOID_MINIMUM_SIZE\s*=\s*([0-9.xX]+)', content)
                voidmax_match = re.search(r'VOID_MAXIMUM_SIZE\s*=\s*([0-9.xX]+)', content)

                if vx_match:
                    self.param_widgets['voxel_size_x'].setValue(float(vx_match.group(1)))
                if vy_match:
                    self.param_widgets['voxel_size_y'].setValue(float(vy_match.group(1)))
                if vz_match:
                    self.param_widgets['voxel_size_z'].setValue(float(vz_match.group(1)))

                if bumpmin_match:
                    self.param_widgets['bump_minimum_size'].setText(bumpmin_match.group(1))
                if bumpmax_match:
                    self.param_widgets['bump_maximum_size'].setText(bumpmax_match.group(1))
                if voidmin_match:
                    self.param_widgets['void_minimum_size'].setText(voidmin_match.group(1))
                if voidmax_match:
                    self.param_widgets['void_maximum_size'].setText(voidmax_match.group(1))

                self._apply_profiling_flag_from_config_text(content)

                z4x_match = re.search(r'Z_STRETCHED_4X\s*=\s*(true|false)', content, re.IGNORECASE)
                if z4x_match and 'z_stretched_4x' in self.param_widgets:
                    self.param_widgets['z_stretched_4x'].setChecked(z4x_match.group(1).lower() == 'true')

                ng_match = re.search(r'NG_THRESHOLD\s*=\s*([\d.]+)', content)
                if ng_match and hasattr(self, 'ng_threshold_spin'):
                    self.ng_threshold_spin.setValue(float(ng_match.group(1)))

                cc3d_conn_match = re.search(r'CC3D_CONNECTIVITY\s*=\s*(\d+)', content)
                if cc3d_conn_match and 'cc3d_connectivity' in self.param_widgets:
                    conn_val = int(cc3d_conn_match.group(1))
                    conn_map = {6: 0, 18: 1, 26: 2}
                    self.param_widgets['cc3d_connectivity'].setCurrentIndex(conn_map.get(conn_val, 2))

                cc3d_minvox_match = re.search(r'CC3D_MIN_VOXELS\s*=\s*(\d+)', content)
                if cc3d_minvox_match and 'cc3d_min_voxels' in self.param_widgets:
                    self.param_widgets['cc3d_min_voxels'].setValue(int(cc3d_minvox_match.group(1)))

                cc3d_gpu_match = re.search(r'CC3D_USE_GPU\s*=\s*(true|false)', content, re.IGNORECASE)
                if cc3d_gpu_match and 'cc3d_use_gpu' in self.param_widgets:
                    self.param_widgets['cc3d_use_gpu'].setChecked(cc3d_gpu_match.group(1).lower() == 'true')

                mode_match = re.search(r'INPUT_MODE\s*=\s*([a-zA-Z0-9_]+)', content)
                if mode_match:
                    mode_val = mode_match.group(1).strip()
                    if mode_val == 'test_1layer':
                        self.mode_test1_radio.setChecked(True)
                    elif mode_val == 'single':
                        self.mode_single_radio.setChecked(True)
                    elif mode_val == 'multi_layer':
                        self.mode_multi_radio.setChecked(True)

                self._apply_enhancement_from_config_text(content)
            except Exception as e:
                print(f"Note: Could not parse Measurement Parameters from config: {e}")

            self._apply_dll_profiling_flag()

            # Parse Layer Definitions from config if present
            try:
                config_content = read_text_auto(config_path)
                num_layers_match = re.search(r'NUM_LAYERS\s*=\s*(\d+)', config_content)
                if num_layers_match:
                    num_layers = int(num_layers_match.group(1))
                    loaded_layers = []
                    for i in range(num_layers):
                        layer_match = re.search(rf'LAYER_{i}\s*=\s*(.+)', config_content)
                        if layer_match:
                            parts = layer_match.group(1).strip().split(',')
                            if len(parts) >= 3:
                                name = parts[0].strip()
                                z_start = int(parts[1].strip())
                                z_end = int(parts[2].strip())
                                selected = parts[3].strip().lower() == 'true' if len(parts) >= 4 else True
                                loaded_layers.append({
                                    'id': i, 'name': name, 'z_start': z_start, 'z_end': z_end, 'selected': selected
                                })
                    if loaded_layers:
                        self.layer_definitions = loaded_layers
                        self.layers_active = True
                        self._populate_layer_table()
                        self._populate_layer_filter()
                        print(f"[Online] Loaded {len(loaded_layers)} layer definitions from config")
            except Exception as e:
                print(f"Note: Could not parse Layer Definitions from config: {e}")

            self.info_label.setText("Config loaded (online)")
            self.check_ready_state()
            return True
        except Exception as e:
            print(f"[SegmentationTab] Failed to load config from {config_path}: {e}")
            return False

    def set_volume_data(self, data, file_path=None):
        """Set volume data programmatically  used by Online mode"""
        if data is None: return
        
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
            
        # Clear previous segmentations to prevent shape mismatch indexing errors
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
            
        self.volume_data = data
        z, y, x = data.shape
        
        # Update sliders
        self.current_slices = {'axial': z//2, 'coronal': y//2, 'sagittal': x//2}
        
        self.axial_slice_slider.setMaximum(z - 1)
        self.axial_slice_slider.setValue(z // 2)
        self.axial_slice_label.setText(f"{z//2} / {z-1}")
        
        self.coronal_slice_slider.setMaximum(y - 1)
        self.coronal_slice_slider.setValue(y // 2)
        self.coronal_slice_label.setText(f"{y//2} / {y-1}")
        
        self.sagittal_slice_slider.setMaximum(x - 1)
        self.sagittal_slice_slider.setValue(x // 2)
        self.sagittal_slice_label.setText(f"{x//2} / {x-1}")
        
        if file_path:
            self.input_path_input.setText(file_path)
            if self.config:
                self.config.inputPath = file_path.encode('utf-8')
        
        # Initialize ROI Z-range to cover the full volume by default
        if hasattr(self, 'roi_z_start_spin') and hasattr(self, 'roi_z_end_spin'):
            self.roi_z_start_spin.setRange(0, z - 1)
            self.roi_z_start_spin.setValue(0)
            self.roi_z_end_spin.setRange(0, z)
            self.roi_z_end_spin.setValue(z)
        
        # Auto window/level
        if data.dtype == np.uint16:
            data_min = np.percentile(data, 1)
            data_max = np.percentile(data, 99)
            self.window_level = {o: (data_max-data_min, (data_max+data_min)/2) for o in ['axial', 'coronal', 'sagittal']}
        
        # Refresh views
        for o in ['axial', 'coronal', 'sagittal']:
            self.update_plane_view(o)
        
        self.check_ready_state()

    def set_segmentation_results(
        self,
        bump_data,
        void_data,
        layer_bands=None,
        object_stats=None,
        labeled_class1=None,
        spacing=None,
    ):
        """Set segmentation results programmatically — used by Online mode.

        Args:
            bump_data / void_data: full-volume masks (runtime FOV Z)
            layer_bands: optional ``[{name, z_start, z_end}, ...]`` in **same Z**
                as masks (Online remapped bands). Rebuilds Layer map for MES.
            object_stats: optional MES rows already tagged with ``layer_name``
                (from Online Viewer) so Teaching table matches without re-measure.
            labeled_class1: optional MES label volume from Viewer (B2B surface patches).
            spacing: optional (sx,sy,sz) world spacing from Viewer volume.
        """
        self.bump_segmentation = bump_data
        self.void_segmentation = void_data
        if labeled_class1 is not None:
            self.labeled_class1_data = labeled_class1
        if spacing is not None:
            try:
                self.custom_spacing = [float(spacing[0]), float(spacing[1]), float(spacing[2])]
                self.spacing = list(self.custom_spacing)
            except Exception:
                pass
        # Also pull spacing/labels from Viewer if present
        if hasattr(self, "_sync_3d_spacing_from_viewer"):
            try:
                self._sync_3d_spacing_from_viewer()
            except Exception:
                pass

        # Runtime Z layer bands (after Online crop remap) — critical for Layer column
        if layer_bands:
            self._rebuild_layer_assignment_map(layer_bands)
            # Keep Teaching layer UI in sync with Online FOV Z (selected all)
            try:
                self.layer_definitions = [
                    {
                        "id": i,
                        "name": str(L.get("name") or f"Layer {i+1}"),
                        "z_start": int(L["z_start"]),
                        "z_end": int(L["z_end"]),
                        "selected": True,
                    }
                    for i, L in enumerate(layer_bands)
                    if int(L.get("z_end", 0)) > int(L.get("z_start", 0))
                ]
                self.layers_active = bool(self.layer_definitions)
                if hasattr(self, "_populate_layer_table"):
                    self._populate_layer_table()
                if hasattr(self, "_populate_layer_filter"):
                    self._populate_layer_filter()
            except Exception as e:
                print(f"[TEACHING] sync layer defs from Online: {e}")
        elif not getattr(self, "layer_assignment_map", None):
            self._rebuild_layer_assignment_map()

        if object_stats is not None:
            self.object_stats = list(object_stats)
            self._measurement_start_slice = 0
            try:
                self._refresh_stats_table()
            except Exception as e:
                print(f"[TEACHING] refresh stats after Online sync: {e}")
            if hasattr(self, "bnd_btn"):
                self.bnd_btn.setEnabled(bool(self.object_stats))
            if hasattr(self, "far_btn"):
                self.far_btn.setEnabled(bool(self.object_stats))
        
        # Refresh views to show overlays
        for o in ['axial', 'coronal', 'sagittal']:
            self.update_plane_view(o)
            
        if hasattr(self, 'measure_btn'):
            self.measure_btn.setEnabled(bump_data is not None)
        if hasattr(self, 'dt_btn'):
            self.dt_btn.setEnabled(bump_data is not None)
        if hasattr(self, 'far_btn') and object_stats is None:
            # MES not synced yet — FAR needs object_stats
            pass
        if hasattr(self, 'bnd_btn') and object_stats is None:
            self.bnd_btn.setEnabled(False)  # Needs measurement first

    def set_paths_and_run(self, input_path, output_path, on_finished_callback=None, layers=None, volume_data=None):
        """
        Set input/output paths, load volume, and auto-run inspection.
        Used by Online mode to drive segmentation without user interaction.

        Args:
            input_path: Path to input TIFF file (e.g. slice16.tif) or multipage stack
            output_path: Output directory for DLL results
            on_finished_callback: Optional callback(result, error) called when done
            layers: Optional list of layer definition dictionaries (INPUT_MODE=single)
                    with z_start/z_end already in *runtime volume* coordinates
            volume_data: Optional preloaded ndarray (Z,Y,X). When provided, skips disk
                         reload so Online can reuse the volume already shown in Viewer.
        """
        if not self.config:
            if on_finished_callback:
                on_finished_callback(None, "No config loaded")
            return

        # Set paths in config
        self.config.inputPath = input_path.encode('utf-8')
        self.config.outputDir = output_path.encode('utf-8')
        self.input_path_input.setText(input_path)
        self.output_path_input.setText(output_path)

        # Clear previous segmentations to prevent shape-mismatch errors
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

        # Prefer preloaded volume (Online already loaded it into Viewer)
        try:
            if volume_data is not None:
                data = volume_data
                if data.ndim == 2:
                    data = data[np.newaxis, :, :]
                elif data.ndim == 4:
                    data = data[:, :, :, 0]
            else:
                data = io.imread(input_path)
                if data.ndim == 2:
                    # Single slice — try loading all files in directory
                    directory = Path(input_path).parent
                    all_tiffs = sorted(glob.glob(str(directory / "*.tif*")))
                    volume_tiffs = [f for f in all_tiffs if '_config' not in f]
                    if len(volume_tiffs) > 1:
                        data = np.array([io.imread(f) for f in volume_tiffs])
                    else:
                        data = data[np.newaxis, :, :]
                elif data.ndim == 4:
                    data = data[:, :, :, 0]

            self.volume_data = data
            z, y, x = data.shape[:3]

            # Update sliders
            self.current_slices = {'axial': z // 2, 'coronal': y // 2, 'sagittal': x // 2}
            for orientation, dim in [('axial', z), ('coronal', y), ('sagittal', x)]:
                slider = getattr(self, f'{orientation}_slice_slider')
                label = getattr(self, f'{orientation}_slice_label')
                slider.setMaximum(dim - 1)
                slider.setValue(dim // 2)
                label.setText(f"{dim // 2} / {dim - 1}")

            self.check_ready_state()
        except Exception as e:
            if on_finished_callback:
                on_finished_callback(None, f"Failed to load volume: {e}")
            return

        # Update config from UI parameters
        self.update_config_from_parameters()
        # Online uses this path — must push profiling flag (DLL reload resets it)
        self._apply_dll_profiling_flag()

        # Run inspection — layers=None means full-volume (test_1layer);
        # layers=[...] means single-volume multi-layer split (INPUT_MODE=single)
        self.inspection_thread = SegmentationInspectionThread(
            self.config, 'both', layers, volume_data=self.volume_data
        )

        if on_finished_callback:
            # Wire callback — bypass the normal on_inspection_finished
            self.inspection_thread.finished.connect(on_finished_callback)
        else:
            self.inspection_thread.finished.connect(
                lambda result, error: self.on_inspection_finished(result, error, None)
            )

        self.inspection_thread.start()

    def reset_for_online(self):
        """Reset tab state but keep DLL loaded  ready for next online folder"""
        # Clear data
        self.config = None
        self.config_path = None
        self.volume_data = None
        self.bump_segmentation = None
        self.void_segmentation = None
        self._pending_input_path = None

        # Clear UI fields (except DLL)
        self.sync_config_path_ui("")
        if hasattr(self, "input_path_input") and self.input_path_input is not None:
            self.input_path_input.setText("")
        if hasattr(self, "output_path_input") and self.output_path_input is not None:
            self.output_path_input.setText("")
        self.info_label.setText("Online mode - waiting for next folder")

        # Disable inspect and measure buttons
        self.inspect_btn.setEnabled(False)
        if hasattr(self, 'measure_btn'):
            self.measure_btn.setEnabled(False)
        if hasattr(self, 'dt_btn'):
            self.dt_btn.setEnabled(False)
        if hasattr(self, 'bnd_btn'):
            self.bnd_btn.setEnabled(False)
        

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
        """In-layout fullscreen for one MPR pane (same idea as 3D Viewer grid expand).

        Teaching uses QSplitter 2×2 (not QGridLayout), so we hide siblings and
        collapse the other splitter row — no external window reparent.
        """
        if getattr(self, "_fullscreen_view", None) is not None:
            self.exit_fullscreen()
            return

        container = getattr(self, f"{orientation}_container", None)
        if container is None:
            return
        if orientation not in ("axial", "coronal", "sagittal"):
            return
        if not all(hasattr(self, n) for n in ("main_splitter", "top_splitter", "bottom_splitter")):
            return

        self._fullscreen_view = orientation
        self._fs_state = {
            "main": list(self.main_splitter.sizes()),
            "top": list(self.top_splitter.sizes()),
            "bottom": list(self.bottom_splitter.sizes()),
            "hidden": [],
            "top_vis": self.top_splitter.isVisible(),
            "bottom_vis": self.bottom_splitter.isVisible(),
        }

        def _hide(w):
            if w is not None and w.isVisible():
                w.setVisible(False)
                self._fs_state["hidden"].append(w)

        for ori in ("axial", "coronal", "sagittal"):
            if ori != orientation:
                _hide(getattr(self, f"{ori}_container", None))

        br = getattr(self, "bottom_right_container", None)
        if orientation in ("axial", "sagittal"):
            # Expand top row; hide bottom row (coronal + teaching sidebar quadrant)
            self.bottom_splitter.setVisible(False)
            total = max(sum(self.main_splitter.sizes()), 100)
            self.main_splitter.setSizes([total, 0])
            if orientation == "axial":
                self.top_splitter.setSizes([1, 0])
            else:
                self.top_splitter.setSizes([0, 1])
        else:
            # Coronal: expand bottom row, hide top + bottom-right quadrant
            self.top_splitter.setVisible(False)
            total = max(sum(self.main_splitter.sizes()), 100)
            self.main_splitter.setSizes([0, total])
            _hide(br)
            self.bottom_splitter.setSizes([1, 0])

        # Visual cue on the pane button if present
        fs_btn = getattr(self, f"{orientation}_fullscreen_btn", None)
        if fs_btn is not None:
            fs_btn.setToolTip("Exit fullscreen")

        QTimer.singleShot(80, lambda o=orientation: self._refresh_mpr_after_layout(o))

    def exit_fullscreen(self):
        """Restore splitter layout after in-layout MPR fullscreen."""
        orientation = getattr(self, "_fullscreen_view", None)
        if orientation is None:
            return

        st = getattr(self, "_fs_state", {}) or {}
        for w in st.get("hidden", []):
            try:
                w.setVisible(True)
            except Exception:
                pass

        if hasattr(self, "top_splitter"):
            self.top_splitter.setVisible(bool(st.get("top_vis", True)))
        if hasattr(self, "bottom_splitter"):
            self.bottom_splitter.setVisible(bool(st.get("bottom_vis", True)))

        try:
            if st.get("main") and hasattr(self, "main_splitter"):
                self.main_splitter.setSizes(st["main"])
            if st.get("top") and hasattr(self, "top_splitter"):
                self.top_splitter.setSizes(st["top"])
            if st.get("bottom") and hasattr(self, "bottom_splitter"):
                self.bottom_splitter.setSizes(st["bottom"])
        except Exception:
            pass

        for ori in ("axial", "coronal", "sagittal"):
            w = getattr(self, f"{ori}_container", None)
            if w is not None:
                w.setVisible(True)
            fs_btn = getattr(self, f"{ori}_fullscreen_btn", None)
            if fs_btn is not None:
                fs_btn.setToolTip("Fullscreen view")

        self._fullscreen_view = None
        self._fs_state = None
        QTimer.singleShot(80, lambda o=orientation: self._refresh_mpr_after_layout(o))

    def _refresh_mpr_after_layout(self, orientation):
        """Re-render an MPR pane after layout/fullscreen change."""
        try:
            widget = getattr(self, f"{orientation}_widget", None)
            if widget is not None:
                rw = widget.GetRenderWindow()
                if rw is not None:
                    interactor = rw.GetInteractor()
                    if interactor is not None and not interactor.GetInitialized():
                        interactor.Initialize()
                    try:
                        wsize = widget.size()
                        if wsize.width() > 1 and wsize.height() > 1:
                            rw.SetSize(int(wsize.width()), int(wsize.height()))
                    except Exception:
                        pass
                    rw.Render()
            if self.volume_data is not None:
                # Fit to new viewport when entering fullscreen (Viewer parity)
                if getattr(self, "_fullscreen_view", None) == orientation:
                    self.reset_view(orientation)
                else:
                    self.update_plane_view(orientation, preserve_camera=True)
        except Exception as e:
            print(f"[Teaching] refresh after layout ({orientation}): {e}")

    def reset_view(self, orientation):
        """Fit slice to view and clear saved camera (same intent as Viewer reset)."""
        if not hasattr(self, "camera_states") or not isinstance(self.camera_states, dict):
            self.camera_states = {}
        self.camera_states[orientation] = None
        renderer = getattr(self, f"{orientation}_renderer", None)
        if renderer is not None:
            try:
                cam = renderer.GetActiveCamera()
                cam.SetPosition(0, 0, 1)
                cam.SetFocalPoint(0, 0, 0)
                cam.SetViewUp(0, 1, 0)
                cam.ParallelProjectionOn()
                renderer.ResetCamera()
            except Exception:
                pass
        self.update_plane_view(orientation, preserve_camera=False)
        widget = getattr(self, f"{orientation}_widget", None)
        if widget is not None:
            try:
                widget.GetRenderWindow().Render()
            except Exception:
                pass

