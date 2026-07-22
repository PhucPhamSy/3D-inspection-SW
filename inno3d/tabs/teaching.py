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

class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()

class NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()

# Import styles
from inno3d.core.styles import apply_theme, SemiconductorTheme



class _ExportLayerVolumesThread(QThread):
    """Background thread that saves each selected layer as a multi-page TIFF file."""
    # (layer_index, layer_name)
    progress = pyqtSignal(int, str)
    # (saved_file_list, error_list, out_dir)
    finished = pyqtSignal(object, object, str)

    def __init__(self, volume_data, selected_layers, out_dir):
        super().__init__()
        self._volume = volume_data        # numpy ndarray (Z, Y, X)
        self._layers = selected_layers    # list of layer dicts
        self._out_dir = out_dir
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import tifffile
        import re

        saved = []
        errors = []

        for idx, layer in enumerate(self._layers):
            if self._cancelled:
                break

            name = layer['name']
            z_start = max(0, int(layer['z_start']))
            z_end   = min(self._volume.shape[0], int(layer['z_end']))

            if z_start >= z_end:
                errors.append(f"{name}: invalid z range ({z_start}{z_end}), skipped.")
                continue

            # Sanitise layer name for use as a filename
            safe_name = re.sub(r'[\\/:*?"<>|]', '_', name)
            out_path = os.path.join(self._out_dir, f"{safe_name}.tif")

            self.progress.emit(idx, name)
            try:
                sub_volume = self._volume[z_start:z_end]   # shape (dZ, Y, X)
                tifffile.imwrite(out_path, sub_volume, photometric='minisblack')
                saved.append(out_path)
            except Exception as e:
                errors.append(f"{name}: {str(e)}")

        self.finished.emit(saved, errors, self._out_dir)


class DistanceTransformThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str) # success, message

    def __init__(self, layers_to_process, spacing, do_bump=True, do_void=False, use_physical=True):
        super().__init__()
        self.layers_to_process = layers_to_process
        self.spacing = spacing
        self.do_bump = do_bump
        self.do_void = do_void
        self.use_physical = use_physical
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        try:
            import scipy.ndimage as ndi
            import tifffile
            import os
            import numpy as np

            total_tasks = len(self.layers_to_process)
            completed_tasks = 0
            total_subtasks = total_tasks * (int(self.do_bump) + int(self.do_void))

            for idx, item in enumerate(self.layers_to_process):
                if self.is_cancelled:
                    self.finished.emit(False, "Operation cancelled by user.")
                    return

                name = item['name']
                bump_mask = item['bump']
                void_mask = item['void']
                output_dir = item['output_dir']

                os.makedirs(output_dir, exist_ok=True)

                # Determine sampling
                sampling = self.spacing if self.use_physical else (1.0, 1.0, 1.0)

                # 1. Bump distance transform
                if self.do_bump and bump_mask is not None:
                    if self.is_cancelled:
                        self.finished.emit(False, "Operation cancelled by user.")
                        return
                    self.progress.emit(
                        int((completed_tasks / max(total_subtasks, 1)) * 100),
                        f"Layer '{name}': Computing bump distance transform..."
                    )
                    # # Convert to binary
                    # binary_bump = (bump_mask > 0).astype(np.uint8)
                    # if np.any(binary_bump):
                    # Convert to binary (inverted: background is 1, bump is 0 to measure distance between bumps)
                    binary_bump = (bump_mask == 0).astype(np.uint8)
                    if np.any(bump_mask > 0):
                        dt_bump = ndi.distance_transform_edt(binary_bump, sampling=sampling).astype(np.float32)
                    else:
                        dt_bump = np.zeros_like(binary_bump, dtype=np.float32)

                    # Save output
                    out_path = os.path.join(output_dir, "distance_map_bump.tif")
                    tifffile.imwrite(out_path, dt_bump, photometric='minisblack')
                    completed_tasks += 1

                # 2. Void distance transform
                if self.do_void and void_mask is not None:
                    if self.is_cancelled:
                        self.finished.emit(False, "Operation cancelled by user.")
                        return
                    self.progress.emit(
                        int((completed_tasks / max(total_subtasks, 1)) * 100),
                        f"Layer '{name}': Computing void distance transform..."
                    )
                    binary_void = (void_mask > 0).astype(np.uint8)
                    if np.any(binary_void):
                        dt_void = ndi.distance_transform_edt(binary_void, sampling=sampling).astype(np.float32)
                    else:
                        dt_void = np.zeros_like(binary_void, dtype=np.float32)

                    out_path = os.path.join(output_dir, "distance_map_void.tif")
                    tifffile.imwrite(out_path, dt_void, photometric='minisblack')
                    completed_tasks += 1

            if self.is_cancelled:
                self.finished.emit(False, "Operation cancelled by user.")
                return

            self.progress.emit(100, "Distance transform complete!")
            self.finished.emit(True, f"Successfully computed 3D distance transform for {total_tasks} layers.\nSaved to layer folders.")
        except Exception as e:
            import traceback
            self.finished.emit(False, f"Distance Transform failed: {str(e)}\n{traceback.format_exc()}")


class BoundaryAnalysisThread(QThread):
    """Thread for 3D boundary extraction and per-voxel surface-to-surface gap measurement.
    
    Output CSV matches advisor's format:
    Bump_id | voxel_number | Voxel_distance_X | Voxel_distance_Y | Voxel_distance_Z
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, layers_to_process, spacing, object_stats,
                 kernel_size=3, iterations=1, save_boundary_tif=True, dll_folder=''):
        super().__init__()
        self.layers_to_process = layers_to_process
        # list of dicts: {'name', 'bump', 'labeled', 'output_dir'}
        self.spacing = spacing  # (vz, vy, vx)
        self.object_stats = object_stats  # list of dicts with 'label', 'grid_row', 'grid_col'
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.save_boundary_tif = save_boundary_tif
        self.dll_folder = dll_folder
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        try:
            import scipy.ndimage as ndi
            from scipy.spatial import cKDTree
            import tifffile
            import os
            import time
            
            start_time = time.time()
            import csv

            # Build label → grid_id mapping from object_stats
            label_to_grid = {}
            for stat in self.object_stats:
                lbl = stat.get('label')
                gr = stat.get('grid_row', '?')
                gc = stat.get('grid_col', '?')
                ln = stat.get('layer_name', '')
                label_to_grid[lbl] = f"{gr},{gc}"

            total_layers = len(self.layers_to_process)
            vz, vy, vx = self.spacing

            for layer_idx, item in enumerate(self.layers_to_process):
                if self.is_cancelled:
                    self.finished.emit(False, "Cancelled by user.")
                    return

                name = item['name']
                bump_mask = item['bump']
                labeled_data = item['labeled']
                output_dir = item['output_dir']
                os.makedirs(output_dir, exist_ok=True)

                # ==============================================================================
                # C# / CUDA NATIVE DLL FAST PATH
                # ==============================================================================
                import os
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                # Check user DLL folder first, then fallback to project folder
                gpu_dll_path = os.path.join(self.dll_folder, "BoundaryGPU.dll") if self.dll_folder else ""
                if not os.path.exists(gpu_dll_path):
                    gpu_dll_path = os.path.join(base_dir, "BoundaryGPU", "BoundaryGPU.dll")
                    
                csharp_dll_path = os.path.join(self.dll_folder, "BoundaryAnalysis.dll") if self.dll_folder else ""
                if not os.path.exists(csharp_dll_path):
                    csharp_dll_path = os.path.join(base_dir, "BoundaryDll", "BoundaryAnalysis", "bin", "Release", "net8.0", "win-x64", "publish", "BoundaryAnalysis.dll")
                
                active_dll_path = gpu_dll_path if os.path.exists(gpu_dll_path) else csharp_dll_path
                func_name = "RunBoundaryAnalysisCUDA" if active_dll_path == gpu_dll_path else "RunBoundaryAnalysis"

                if os.path.exists(active_dll_path):
                    import ctypes
                    try:
                        dll_type = "CUDA GPU" if active_dll_path == gpu_dll_path else "C# Native"
                        self.progress.emit(
                            int(layer_idx / max(total_layers, 1) * 100),
                            f"[{layer_idx+1}/{total_layers}] {name}: Running {dll_type} DLL Boundary Analysis..."
                        )
                        
                        boundary_dll = ctypes.CDLL(active_dll_path)
                        run_func = getattr(boundary_dll, func_name)
                        run_func.argtypes = [
                            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint32),
                            ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.c_float, ctypes.c_float, ctypes.c_float,
                            ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)
                        ]
                        run_func.restype = ctypes.c_int
                        
                        bump_c = bump_mask if (isinstance(bump_mask, np.ndarray) and bump_mask.dtype == np.uint8 and bump_mask.flags.c_contiguous) else np.ascontiguousarray(bump_mask, dtype=np.uint8)
                        lbl_c = labeled_data if (isinstance(labeled_data, np.ndarray) and labeled_data.dtype == np.uint32 and labeled_data.flags.c_contiguous) else np.ascontiguousarray(labeled_data, dtype=np.uint32)
                        depth, height, width = bump_c.shape
                        
                        max_lbl = int(np.max(lbl_c)) if lbl_c.size > 0 else 0
                        grid_rows = np.full(max_lbl + 1, -1, dtype=np.int32)
                        grid_cols = np.full(max_lbl + 1, -1, dtype=np.int32)
                        
                        for stat in self.object_stats:
                            lbl_val = stat.get('label', 0)
                            if 0 <= lbl_val <= max_lbl:
                                gr = stat.get('grid_row', '?')
                                gc = stat.get('grid_col', '?')
                                if gr != '?' and gc != '?':
                                    try:
                                        grid_rows[lbl_val] = int(gr)
                                        grid_cols[lbl_val] = int(gc)
                                    except:
                                        pass
                        
                        out_dir_encoded = output_dir.encode('utf-8')
                        out_dir_ptr = ctypes.cast(ctypes.create_string_buffer(out_dir_encoded), ctypes.c_void_p)
                        
                        ptr_bump = bump_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                        ptr_lbl = lbl_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
                        ptr_gr = grid_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
                        ptr_gc = grid_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
                        
                        res = run_func(
                            ptr_bump, ptr_lbl,
                            depth, height, width,
                            float(vz), float(vy), float(vx),
                            out_dir_ptr,
                            ptr_gr, ptr_gc
                        )
                        
                        if res == 1:
                            if self.save_boundary_tif:
                                try:
                                    self.progress.emit(
                                        int(layer_idx / max(total_layers, 1) * 100),
                                        f"[{layer_idx+1}/{total_layers}] {name}: Saving boundary_mask.tif..."
                                    )
                                    binary = (bump_mask > 0)
                                    struct = np.ones((self.kernel_size,) * 3, dtype=bool)
                                    try:
                                        import cupy as cp
                                        import cupyx.scipy.ndimage as cndi
                                        binary_gpu = cp.asarray(binary)
                                        struct_gpu = cp.asarray(struct)
                                        eroded_gpu = cndi.binary_erosion(binary_gpu, structure=struct_gpu, iterations=self.iterations)
                                        b_mask = cp.asnumpy(binary_gpu & ~eroded_gpu)
                                        del binary_gpu, struct_gpu, eroded_gpu
                                        cp.get_default_memory_pool().free_all_blocks()
                                    except Exception:
                                        b_mask = binary & ~ndi.binary_erosion(binary, structure=struct, iterations=self.iterations)
                                    
                                    tif_path = os.path.join(output_dir, "boundary_mask.tif")
                                    tifffile.imwrite(tif_path, (b_mask * 255).astype(np.uint8), compression='zlib')
                                except Exception as tif_err:
                                    print(f"[Warning] Failed to save boundary_mask.tif: {tif_err}")
                            continue # DLL success, skip python implementation for this layer
                        else:
                            print(f"[Warning] {dll_type} DLL returned error code {res}. Falling back to Python implementation.")
                    except Exception as e:
                        print(f"[Warning] {dll_type} DLL Exception: {e}. Falling back to Python implementation.")
                # ==============================================================================

                # 1. 3D Erosion → boundary = original - eroded
                binary = (bump_mask > 0)
                struct = np.ones((self.kernel_size,) * 3, dtype=bool)
                
                try:
                    import cupy as cp
                    import cupyx.scipy.ndimage as cndi
                    
                    self.progress.emit(
                        int(layer_idx / max(total_layers, 1) * 100),
                        f"[{layer_idx+1}/{total_layers}] {name}: Eroding 3D (GPU Accelerated)..."
                    )
                    
                    # Move to GPU
                    binary_gpu = cp.asarray(binary)
                    struct_gpu = cp.asarray(struct)
                    
                    eroded_gpu = cndi.binary_erosion(binary_gpu, structure=struct_gpu, iterations=self.iterations)
                    eroded = cp.asnumpy(eroded_gpu)
                    
                    # Free GPU memory
                    del binary_gpu
                    del struct_gpu
                    del eroded_gpu
                    cp.get_default_memory_pool().free_all_blocks()
                    
                except ImportError:
                    self.progress.emit(
                        int(layer_idx / max(total_layers, 1) * 100),
                        f"[{layer_idx+1}/{total_layers}] {name}: Eroding 3D (CPU)..."
                    )
                    eroded = ndi.binary_erosion(binary, structure=struct, iterations=self.iterations)
                    
                except Exception as e:
                    print(f"GPU Erosion failed: {e}. Falling back to CPU...")
                    self.progress.emit(
                        int(layer_idx / max(total_layers, 1) * 100),
                        f"[{layer_idx+1}/{total_layers}] {name}: Eroding 3D (CPU Fallback)..."
                    )
                    eroded = ndi.binary_erosion(binary, structure=struct, iterations=self.iterations)

                boundary_mask = binary & ~eroded

                if self.is_cancelled:
                    self.finished.emit(False, "Cancelled by user.")
                    return

                # 2. Label boundary voxels with bump IDs (AND operation)
                boundary_labeled = labeled_data * boundary_mask.astype(labeled_data.dtype)

                if self.save_boundary_tif:
                    try:
                        tif_path = os.path.join(output_dir, "boundary_mask.tif")
                        tifffile.imwrite(tif_path, (boundary_mask * 255).astype(np.uint8), compression='zlib')
                    except Exception as tif_err:
                        print(f"[Warning] Failed to save boundary_mask.tif in Python fallback: {tif_err}")

                self.progress.emit(
                    int((layer_idx + 0.3) / max(total_layers, 1) * 100),
                    f"[{layer_idx+1}/{total_layers}] {name}: Extracting boundary voxels..."
                )

                # 3. Extract ALL boundary voxel coordinates in ONE pass (not per-label!)
                unique_labels = np.unique(boundary_labeled)
                unique_labels = unique_labels[unique_labels > 0]

                if len(unique_labels) < 2:
                    csv_path = os.path.join(output_dir, "boundary_voxel_distances.csv")
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Bump_id', 'voxel_number',
                                         'Voxel_distance_X', 'Voxel_distance_Y', 'Voxel_distance_Z'])
                    continue

                # Single-pass: extract ALL boundary voxels at once
                all_boundary_indices = np.argwhere(boundary_labeled > 0)  # (N, 3) [z, y, x]
                all_boundary_labels = boundary_labeled[
                    all_boundary_indices[:, 0],
                    all_boundary_indices[:, 1],
                    all_boundary_indices[:, 2]
                ].astype(np.int64)

                total_boundary = len(all_boundary_indices)
                self.progress.emit(
                    int((layer_idx + 0.35) / max(total_layers, 1) * 100),
                    f"[{layer_idx+1}/{total_layers}] {name}: {total_boundary:,} boundary voxels, building KD-tree..."
                )

                # Build label → index ranges (sorted by label for efficient grouping)
                sort_order = np.argsort(all_boundary_labels)
                all_boundary_indices = all_boundary_indices[sort_order]
                all_boundary_labels = all_boundary_labels[sort_order]

                # Pre-compute per-label slices
                label_start = {}
                label_end = {}
                current_lbl = all_boundary_labels[0]
                current_start = 0
                for i in range(1, total_boundary):
                    if all_boundary_labels[i] != current_lbl:
                        label_start[current_lbl] = current_start
                        label_end[current_lbl] = i
                        current_lbl = all_boundary_labels[i]
                        current_start = i
                label_start[current_lbl] = current_start
                label_end[current_lbl] = total_boundary

                # 4. FAST KD-Tree approach using local neighborhoods!
                # Instead of querying a global tree with k=bump_size+1 (which is extremely slow),
                # we find the nearest ~30 bumps, collect their boundary voxels, and build a small local tree.
                # Then we only need to query k=1 against that small tree!

                # Pre-allocate result arrays
                nearest_other_dist = np.full(total_boundary, np.nan, dtype=np.float64)
                has_diff = np.zeros(total_boundary, dtype=bool)
                dist_x = np.full(total_boundary, np.nan, dtype=np.float64)
                dist_y = np.full(total_boundary, np.nan, dtype=np.float64)
                dist_z = np.full(total_boundary, np.nan, dtype=np.float64)

                # Compute centroid of each bump's boundary
                num_unique = len(unique_labels)
                all_phys = all_boundary_indices.astype(np.float64) * np.array([vz, vy, vx])
                centroids = np.zeros((num_unique, 3), dtype=np.float64)
                for bump_i, lbl in enumerate(unique_labels):
                    s = label_start.get(lbl, 0)
                    e = label_end.get(lbl, 0)
                    if e > s:
                        centroids[bump_i] = np.mean(all_phys[s:e], axis=0)

                # Build centroid KD-Tree to find neighbor bumps
                centroids_tree = cKDTree(centroids)

                for bump_i, lbl in enumerate(unique_labels):
                    if self.is_cancelled:
                        self.finished.emit(False, "Cancelled by user.")
                        return

                    s = label_start.get(lbl, 0)
                    e = label_end.get(lbl, 0)
                    n_vox = e - s
                    if n_vox == 0:
                        continue

                    bid = label_to_grid.get(int(lbl), f"L{lbl}")
                    if bump_i % 50 == 0:
                        self.progress.emit(
                            int((layer_idx + 0.4 + 0.35 * bump_i / num_unique) / max(total_layers, 1) * 100),
                            f"[{layer_idx+1}/{total_layers}] {name}: Bump ({bid}) {bump_i+1}/{num_unique}"
                        )

                    my_phys = all_phys[s:e]
                    my_indices = all_boundary_indices[s:e]

                    # Find nearest 30 bump centroids (excluding itself)
                    K_bumps = min(30, num_unique)
                    if K_bumps > 1:
                        # k=K_bumps+1 because the nearest neighbor is the bump itself
                        _, neighbor_idx = centroids_tree.query(centroids[bump_i], k=K_bumps + 1)
                        # Ensure array even if K_bumps=1
                        if np.isscalar(neighbor_idx):
                            neighbor_idx = np.array([neighbor_idx])
                        
                        # Remove self (usually the first one, but filter by index just in case)
                        neighbor_idx = neighbor_idx[neighbor_idx != bump_i]
                        # Cap at 30 bumps
                        neighbor_idx = neighbor_idx[:30]
                        
                        # Gather boundary voxels from these neighbor bumps
                        neighbor_phys_list = []
                        neighbor_indices_list = []
                        for n_idx in neighbor_idx:
                            n_lbl = unique_labels[n_idx]
                            n_s = label_start.get(n_lbl, 0)
                            n_e = label_end.get(n_lbl, 0)
                            if n_e > n_s:
                                neighbor_phys_list.append(all_phys[n_s:n_e])
                                neighbor_indices_list.append(all_boundary_indices[n_s:n_e])
                        
                        if neighbor_phys_list:
                            neighbor_phys = np.concatenate(neighbor_phys_list, axis=0)
                            neighbor_indices = np.concatenate(neighbor_indices_list, axis=0)
                            
                            # Build local tree of only the neighboring bumps
                            local_tree = cKDTree(neighbor_phys)
                            
                            # Query k=1 against local tree (since current bump is NOT in this tree!)
                            dists_1, idxs_1 = local_tree.query(my_phys, k=1, workers=-1)
                            
                            # Store results
                            has_diff[s:e] = True
                            nearest_other_dist[s:e] = dists_1
                            
                            other_voxels = neighbor_indices[idxs_1]
                            diff_vox = np.abs(my_indices.astype(np.int64) - other_voxels.astype(np.int64))
                            
                            dist_x[s:e] = diff_vox[:, 2] * vx
                            dist_y[s:e] = diff_vox[:, 1] * vy
                            dist_z[s:e] = diff_vox[:, 0] * vz

                self.progress.emit(
                    int((layer_idx + 0.75) / max(total_layers, 1) * 100),
                    f"[{layer_idx+1}/{total_layers}] {name}: Writing CSVs..."
                )

                # 7. Write per-voxel CSV (grouped by bump)
                csv_path = os.path.join(output_dir, "boundary_voxel_distances.csv")
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Bump_id', 'voxel_number',
                                     'Voxel_distance_X', 'Voxel_distance_Y', 'Voxel_distance_Z'])

                    for lbl in unique_labels:
                        if self.is_cancelled:
                            self.finished.emit(False, "Cancelled by user.")
                            return
                        bump_id_str = label_to_grid.get(int(lbl), f"L{lbl}")
                        s = label_start.get(lbl, 0)
                        e = label_end.get(lbl, 0)
                        for vi_offset, gi in enumerate(range(s, e)):
                            if has_diff[gi]:
                                writer.writerow([bump_id_str, vi_offset,
                                                 f"{dist_x[gi]:.3f}", f"{dist_y[gi]:.3f}", f"{dist_z[gi]:.3f}"])
                            else:
                                writer.writerow([bump_id_str, vi_offset, 'NaN', 'NaN', 'NaN'])

                # (heatmap TIFF removed — full volume too large to save/open)

                # 9. Summary CSV with per-bump min distances + location
                summary_path = os.path.join(output_dir, "boundary_summary.csv")
                with open(summary_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Bump_id', 'label', 'boundary_voxels',
                                     'min_gap_X_um', 'min_gap_Y_um', 'min_gap_Z_um',
                                     'min_gap_euclidean_um',
                                     'min_gap_voxel_Z', 'min_gap_voxel_Y', 'min_gap_voxel_X'])

                    for lbl in unique_labels:
                        bump_id_str = label_to_grid.get(int(lbl), f"L{lbl}")
                        s = label_start.get(lbl, 0)
                        e = label_end.get(lbl, 0)
                        n_vox = e - s

                        bump_has_diff = has_diff[s:e]
                        if np.any(bump_has_diff):
                            bump_dx = dist_x[s:e]
                            bump_dy = dist_y[s:e]
                            bump_dz = dist_z[s:e]
                            bump_eucl = nearest_other_dist[s:e]

                            valid = bump_has_diff
                            min_eucl_idx = np.nanargmin(bump_eucl[valid])
                            # Map back to global index
                            valid_indices = np.where(valid)[0]
                            best_local = valid_indices[min_eucl_idx]
                            best_global = s + best_local

                            min_vox = all_boundary_indices[best_global]

                            def _fmt(v):
                                return f"{v:.3f}" if np.isfinite(v) else "NaN"

                            writer.writerow([
                                bump_id_str, int(lbl), n_vox,
                                _fmt(np.nanmin(bump_dx[valid])),
                                _fmt(np.nanmin(bump_dy[valid])),
                                _fmt(np.nanmin(bump_dz[valid])),
                                _fmt(np.nanmin(bump_eucl[valid])),
                                int(min_vox[0]), int(min_vox[1]), int(min_vox[2]),
                            ])
                        else:
                            writer.writerow([bump_id_str, int(lbl), n_vox,
                                             'NaN', 'NaN', 'NaN', 'NaN',
                                             'NaN', 'NaN', 'NaN'])

            if self.is_cancelled:
                self.finished.emit(False, "Cancelled by user.")
                return

            elapsed = time.time() - start_time
            self.progress.emit(100, f"Boundary analysis complete in {elapsed:.2f}s!")
            self.finished.emit(True,
                f"3D Boundary Analysis complete in {elapsed:.2f} seconds.\n"
                f"Outputs per layer folder:\n"
                f"  • boundary_voxel_distances.csv (per-voxel gaps)\n"
                f"  • boundary_summary.csv (per-bump min gaps)")
        except Exception as e:
            import traceback
            self.finished.emit(False, f"Boundary Analysis failed: {str(e)}\n{traceback.format_exc()}")


class SegmentationPreviewThread(QThread):
    """Thread for running segmentation preview on visible slices"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object, object)  # bump_result, void_result, error
    
    def __init__(self, config, volume_data, slice_indices):
        super().__init__()
        self.setAcceptDrops(True)
        self.config = config
        self.volume_data = volume_data
        self.slice_indices = slice_indices  # {'axial': idx, 'coronal': idx, 'sagittal': idx}
        self._is_cancelled = False
        
    def cancel(self):
        self._is_cancelled = True
        
    def run(self):
        try:
            # For preview, we'll process only the visible slices
            # This is a simplified preview - full inspection runs on entire volume
            self.progress.emit(50, "Processing preview...")
            
            # TODO: Implement slice-based preview segmentation
            # For now, return None to indicate preview is not yet implemented
            if self._is_cancelled:
                self.finished.emit(None, None, "Cancelled")
                return
                
            self.progress.emit(100, "Preview complete")
            self.finished.emit(None, None, None)
            
        except Exception as e:
            import traceback
            self.finished.emit(None, None, f"{str(e)}\n{traceback.format_exc()}")


class SegmentationInspectionThread(QThread):
    """Thread for running full Bump/Void inspection, optionally across multiple layers"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object)  # result(s), error
    
    def __init__(self, config, mode='both', layers=None, volume_data=None, multi_layer_files=None):
        super().__init__()
        self.config = config
        self.mode = mode  # 'bump', 'void', 'both'
        self.layers = layers  # Optional list of layer definitions (single-volume mode)
        self.volume_data = volume_data  # Required for multi-layer (to extract sub-volumes)
        self.multi_layer_files = multi_layer_files  # [{'name': str, 'path': str}] for direct-file mode
        
    def run(self):
        try:
            # Create progress callback
            def progress_callback(message, current, total):
                if total > 0:
                    percent = int((current / total) * 100)
                    self.progress.emit(percent, message)
            
            callback = bumpvoid.make_progress_callback(progress_callback)

            # ---- Multi-layer files mode (direct file path per layer) ----
            if self.multi_layer_files:
                import tifffile
                all_results = []
                original_output = self.config.outputDir.decode('utf-8')
                total_layers = len(self.multi_layer_files)

                for idx, lf in enumerate(self.multi_layer_files):
                    layer_name = lf['name']
                    file_path  = lf['path']

                    self.progress.emit(
                        int(idx / total_layers * 100),
                        f"[{idx+1}/{total_layers}] Running: {layer_name}"
                    )

                    # Set up per-layer output subfolder
                    layer_output = os.path.join(original_output, layer_name.replace(" ", "_"))
                    os.makedirs(layer_output, exist_ok=True)

                    self.config.inputPath = file_path.encode('utf-8')
                    self.config.outputDir = layer_output.encode('utf-8')
                    # Sub-volumes already have no blank regions
                    self.config.blankStart1 = 0
                    self.config.blankEnd1 = 0
                    self.config.blankStart2 = 0
                    self.config.blankEnd2 = 0

                    if self.mode == 'bump':
                        res = bumpvoid.process_bump_only(self.config, callback)
                    else:
                        res = bumpvoid.process(self.config, callback)

                    if res is not None:
                        res.layer_name = layer_name
                    all_results.append(res)

                # Restore output dir
                self.config.outputDir = original_output.encode('utf-8')
                self.finished.emit(all_results, None)
                return
            
            # OpenCV imread requires a file path, not a directory.
            original_input = self.config.inputPath
            current_input = original_input.decode('utf-8')
            
            temp_dir = None
            temp_path = None
            
            if not self.layers:
                # Check if we need to write the volume to a temporary multi-page TIFF file
                need_temp_file = False
                if os.path.isdir(current_input):
                    need_temp_file = True
                elif os.path.isfile(current_input) and self.volume_data is not None:
                    if self.volume_data.ndim == 3 and self.volume_data.shape[0] > 1:
                        try:
                            import tifffile
                            with tifffile.TiffFile(current_input) as tf:
                                if len(tf.pages) < self.volume_data.shape[0]:
                                    need_temp_file = True
                        except Exception:
                            need_temp_file = True

                if need_temp_file and self.volume_data is not None:
                    import tempfile
                    import tifffile
                    self.progress.emit(0, "Preparing temporary TIFF stack...")
                    temp_dir = tempfile.mkdtemp(prefix="inno3d_volume_")
                    orig_base = os.path.basename(current_input.rstrip('/\\'))
                    if not orig_base:
                        orig_base = "volume"
                    if orig_base.lower().endswith(('.tif', '.tiff')):
                        temp_filename = orig_base
                    else:
                        temp_filename = orig_base + ".tif"
                    temp_path = os.path.join(temp_dir, temp_filename)
                    print(f"Saving temporary TIFF stack: {temp_path} (shape: {self.volume_data.shape})")
                    tifffile.imwrite(temp_path, self.volume_data)
                    self.config.inputPath = temp_path.encode('utf-8')
                elif os.path.isdir(current_input):
                    # Fallback if volume_data is not loaded (should not happen)
                    import glob
                    images = (glob.glob(os.path.join(current_input, "*.tif")) 
                             + glob.glob(os.path.join(current_input, "*.tiff"))
                             + glob.glob(os.path.join(current_input, "*.png")) 
                             + glob.glob(os.path.join(current_input, "*.bmp")))
                    if images:
                        images.sort()
                        self.config.inputPath = images[0].encode('utf-8')

                # Single run  no layer splitting
                try:
                    if self.mode == 'bump':
                        result = bumpvoid.process_bump_only(self.config, callback)
                    elif self.mode == 'void':
                        result = None  # TODO: Handle void-only case
                    else:  # both
                        result = bumpvoid.process(self.config, callback)
                finally:
                    # Clean up temporary file if created
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                            os.rmdir(temp_dir)
                            print("Cleaned up temporary volume TIFF")
                        except Exception as cleanup_err:
                            print(f"Warning: failed to clean temp files: {cleanup_err}")
                    # Restore original input path in config
                    self.config.inputPath = original_input
                
                self.finished.emit(result, None)
                return
            
            
            # ---- Multi-layer run ----
            # For each selected layer, extract the sub-volume [z_start:z_end],
            # save it as a temporary TIFF, and run the DLL on that file alone.
            # This ensures memory efficiency (only 1 layer loaded in DLL at a time).
            import tempfile
            import tifffile
            
            all_results = []
            original_output = self.config.outputDir.decode('utf-8')
            original_input = self.config.inputPath  # save to restore later
            original_blankStart1 = self.config.blankStart1
            original_blankEnd1 = self.config.blankEnd1
            original_blankStart2 = self.config.blankStart2
            original_blankEnd2 = self.config.blankEnd2
            
            total_layers = len(self.layers)
            
            for idx, layer in enumerate(self.layers):
                layer_name = layer['name']
                z_start = int(layer['z_start'])
                z_end = int(layer['z_end'])
                
                self.progress.emit(0, f"[{idx+1}/{total_layers}] Preparing {layer_name} (Z: {z_start}-{z_end})...")
                
                # 1. Extract sub-volume for this layer
                if self.volume_data is None:
                    raise RuntimeError("volume_data is required for multi-layer processing")
                
                sub_volume = self.volume_data[z_start:z_end]
                print(f"[Layer {layer_name}] Extracted sub-volume shape: {sub_volume.shape}")
                
                # 2. Save sub-volume as temporary multi-page TIFF
                temp_dir = tempfile.mkdtemp(prefix=f"inno3d_{layer_name.replace(' ', '_')}_")
                temp_path = os.path.join(temp_dir, f"{layer_name.replace(' ', '_')}.tif")
                tifffile.imwrite(temp_path, sub_volume)
                print(f"[Layer {layer_name}] Saved temp TIFF: {temp_path}")
                
                # 3. Setup layer-specific config
                layer_output = os.path.join(original_output, layer_name.replace(" ", "_"))
                os.makedirs(layer_output, exist_ok=True)
                
                self.config.inputPath = temp_path.encode('utf-8')
                self.config.outputDir = layer_output.encode('utf-8')
                # Reset blank ranges  the sub-volume IS the layer, no blanking needed
                self.config.blankStart1 = 0
                self.config.blankEnd1 = 0
                self.config.blankStart2 = 0
                self.config.blankEnd2 = 0
                
                # 4. Run DLL on sub-volume
                self.progress.emit(0, f"[{idx+1}/{total_layers}] Running segmentation on {layer_name}...")
                
                if self.mode == 'bump':
                    res = bumpvoid.process_bump_only(self.config, callback)
                elif self.mode == 'both':
                    res = bumpvoid.process(self.config, callback)
                else:
                    res = None
                
                if res is not None:
                    res.layer_name = layer_name
                    
                all_results.append(res)
                
                # 5. Clean up temporary file
                try:
                    os.remove(temp_path)
                    os.rmdir(temp_dir)
                    print(f"[Layer {layer_name}] Cleaned up temp files")
                except Exception as cleanup_err:
                    print(f"[Layer {layer_name}] Warning: failed to clean temp files: {cleanup_err}")
                
            # Restore original config
            self.config.inputPath = original_input
            self.config.outputDir = original_output.encode('utf-8')
            self.config.blankStart1 = original_blankStart1
            self.config.blankEnd1 = original_blankEnd1
            self.config.blankStart2 = original_blankStart2
            self.config.blankEnd2 = original_blankEnd2
            
            self.finished.emit(all_results, None)
            
        except Exception as e:
            import traceback
            self.finished.emit(None, f"{str(e)}\n{traceback.format_exc()}")


class EnhancementThread(QThread):
    """Thread for running Enhanced Volume DLL preprocessing."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)  # success, error_message

    def __init__(self, dll_dir, model_path, trt_cache_path, use_gpu, gpu_device_id,
                 input_dir, output_dir, start_slice=-1, end_slice=-1):
        super().__init__()
        self.dll_dir = dll_dir
        self.model_path = model_path
        self.trt_cache_path = trt_cache_path
        self.use_gpu = use_gpu
        self.gpu_device_id = gpu_device_id
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.start_slice = start_slice
        self.end_slice = end_slice

    def run(self):
        try:
            self.progress.emit(5, "Loading BumpVoid_ISP_ENH...")
            # Try to load DLL (may already be loaded)
            try:
                enhanced_volume.load_dll(self.dll_dir)
            except Exception:
                pass  # Already loaded or will fail at init

            # Check if model session needs initialization
            try:
                already_init = enhanced_volume.is_initialized()
            except Exception:
                already_init = False

            if not already_init:

                self.progress.emit(10, "Initializing ONNX model...")
                try:
                    ok = enhanced_volume.init(
                        self.model_path,
                        trt_cache_path=self.trt_cache_path,
                        use_gpu=self.use_gpu,
                        gpu_device_id=self.gpu_device_id
                    )
                except OSError as e:
                    # CLR / native crash surfaces as Windows Error 0xe0434352
                    self.finished.emit(
                        False,
                        f"Enhancement init crashed: {e}\n"
                        "Check that BumpVoid_ISP_ENH.dll is in the DLL folder, "
                        "and that Microsoft.ML.OnnxRuntime.dll is not blocked "
                        "(right-click → Properties → Unblock)."
                    )
                    return
                if not ok:
                    err = enhanced_volume.get_last_error()
                    self.finished.emit(False, f"Enhancement init failed: {err}")
                    return

                self.progress.emit(20, "Warming up model (CUDA/TensorRT)...")
                try:
                    wu = enhanced_volume.warm_up(512, 512)
                    if not wu:
                        # Non-fatal: continue; process may still work
                        err = enhanced_volume.get_last_error()
                        self.progress.emit(20, f"Warm-up warning: {err}")
                except OSError as e:
                    self.finished.emit(False, f"Enhancement warm-up crashed: {e}")
                    return

            # Create progress callback
            def progress_cb(message, current, total):
                if total > 0:
                    pct = 20 + int((current / total) * 75)
                    self.progress.emit(pct, message)

            cb = enhanced_volume.make_progress_callback(progress_cb)

            self.progress.emit(25, "Enhancing volume slices...")
            if self.start_slice >= 0 and self.end_slice >= 0:
                result = enhanced_volume.process_slice_range(
                    self.input_dir, self.output_dir,
                    self.start_slice, self.end_slice, cb
                )
            else:
                result = enhanced_volume.process_folder(
                    self.input_dir, self.output_dir, cb
                )

            if result < 0:
                err = enhanced_volume.get_last_error()
                self.finished.emit(False, f"Enhancement failed: {err}")
            elif result == 0:
                self.finished.emit(False, "Enhancement: No files processed")
            else:
                self.progress.emit(100, f"Enhanced {result} slices")
                self.finished.emit(True, "")

        except Exception as e:
            import traceback
            self.finished.emit(False, f"{str(e)}\n{traceback.format_exc()}")


class SegmentationTab(QWidget):
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
        
        # Config Path
        config_label = QLabel("Config:")
        layout.addWidget(config_label)
        
        self.config_path_input = QLineEdit()
        self.config_path_input.setReadOnly(True)
        self.config_path_input.setPlaceholderText("Select config_HBM.txt file...")
        layout.addWidget(self.config_path_input, 2)
        
        config_btn = make_browse_btn("settings.svg", "Browse config file", self.load_config_file, QStyle.SP_FileIcon)
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
        self.object_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
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
        renderer.AddActor(overlay_actor)

    def _refresh_all_index_views(self):
        """Refresh all three views when Show Index checkbox changes."""
        for ori in ['axial', 'coronal', 'sagittal']:
            self.update_plane_view(ori, preserve_camera=True)

    def draw_object_labels(self, renderer, actual_z, orientation='axial', slice_idx=0):
        """Draw bump index labels on any of the three projection views.
        
        For HBM data with dense bump arrays (~27 cols × 31 rows), this uses:
        - Compact (R,C) labels at each bump centroid visible in the current slice
        - Row/Column header labels along the edges for quick spatial reference
        
        Args:
            renderer: VTK renderer for the view
            actual_z: Actual Z slice index (used for axial view)
            orientation: 'axial', 'coronal', or 'sagittal'
            slice_idx: Slice index for coronal (Y) or sagittal (X)
        """
        if not self.object_stats:
            return

        # Determine z-offset strategy
        is_multi_layer = getattr(self, 'input_mode', 'single') == 'multi_layer'
        ml_z_starts = getattr(self, '_ml_layer_z_starts', {})
        single_z_offset = getattr(self, '_measurement_start_slice', 0) if not is_multi_layer else 0

        # Collect visible stats for this orientation/slice
        visible_stats = []
        for stat in self.object_stats:
            # Multi-layer filter: only filter when a specific layer is selected
            if is_multi_layer and self.current_display_layer is not None:
                if stat.get('layer_name') != self.current_display_layer:
                    continue

            if orientation == 'axial':
                # Compute the per-stat z offset
                if is_multi_layer and self.current_display_layer is None:
                    # "All Layers" mode: volume is combined stack, use per-layer z_start
                    layer_z_start = ml_z_starts.get(stat.get('layer_name', ''), 0)
                else:
                    # Specific layer or single mode
                    layer_z_start = 0
                local_z = actual_z - single_z_offset - layer_z_start
                if stat['z_min'] <= local_z < stat['z_max']:
                    visible_stats.append((stat, layer_z_start))
            elif orientation == 'coronal':
                # Coronal slices along Y axis
                if stat['y_min'] <= slice_idx < stat['y_max']:
                    if is_multi_layer and self.current_display_layer is None:
                        layer_z_start = ml_z_starts.get(stat.get('layer_name', ''), 0)
                    else:
                        layer_z_start = single_z_offset
                    visible_stats.append((stat, layer_z_start))
            elif orientation == 'sagittal':
                # Sagittal slices along X axis
                if stat['x_min'] <= slice_idx < stat['x_max']:
                    if is_multi_layer and self.current_display_layer is None:
                        layer_z_start = ml_z_starts.get(stat.get('layer_name', ''), 0)
                    else:
                        layer_z_start = single_z_offset
                    visible_stats.append((stat, layer_z_start))

        if not visible_stats:
            return

        # ----- Determine label scale and font sizing based on bump density -----
        n_bumps = len(visible_stats)
        # Adaptive scaling: smaller text for denser arrays
        if n_bumps > 500:
            label_scale = 0.18
            font_size = 14
        elif n_bumps > 200:
            label_scale = 0.22
            font_size = 15
        elif n_bumps > 50:
            label_scale = 0.25
            font_size = 16
        else:
            label_scale = 0.30
            font_size = 18

        # ----- Compute unique grid rows/cols for header labels -----
        stats_only = [s for s, _ in visible_stats]
        has_grid = all('grid_row' in s and 'grid_col' in s for s in stats_only)

        if orientation == 'axial':
            self._draw_axial_labels(renderer, visible_stats, has_grid, label_scale, font_size)
        elif orientation == 'coronal':
            self._draw_coronal_labels(renderer, visible_stats, has_grid, label_scale, font_size, slice_idx)
        elif orientation == 'sagittal':
            self._draw_sagittal_labels(renderer, visible_stats, has_grid, label_scale, font_size, slice_idx)

    def _draw_axial_labels(self, renderer, visible_stats, has_grid, label_scale, font_size):
        """Draw labels on axial (XY) view with per-bump (R,C) and edge headers."""
        if has_grid:
            # --- Per-bump compact labels ---
            for stat, _z_off in visible_stats:
                label_text = f"{stat['grid_row']},{stat['grid_col']}"
                cx = stat['centroid_x']
                cy = stat['centroid_y']
                caption = vtk.vtkTextActor3D()
                caption.SetInput(label_text)
                caption.SetPosition(cx, cy, 0.5)
                caption.SetScale(label_scale, label_scale, label_scale)
                caption.GetTextProperty().SetFontSize(font_size)
                caption.GetTextProperty().SetBold(True)
                caption.GetTextProperty().SetShadow(True)
                caption.GetTextProperty().SetShadowOffset(1, -1)
                caption.GetTextProperty().SetColor(0.2, 1.0, 1.0)  # Bright Cyan
                renderer.AddActor(caption)

            # --- Column headers along the top edge ---
            col_positions = {}  # col_id -> list of centroid_x
            row_positions = {}  # row_id -> list of centroid_y
            for stat, _z_off in visible_stats:
                gc = stat['grid_col']
                gr = stat['grid_row']
                col_positions.setdefault(gc, []).append(stat['centroid_x'])
                row_positions.setdefault(gr, []).append(stat['centroid_y'])

            # Find Y extent for positioning headers
            all_y_min = min(s['y_min'] for s, _ in visible_stats)
            all_x_min = min(s['x_min'] for s, _ in visible_stats)
            header_scale = label_scale * 1.4

            # Column headers at top
            for col_id in sorted(col_positions.keys()):
                avg_x = np.mean(col_positions[col_id])
                hdr = vtk.vtkTextActor3D()
                hdr.SetInput(f"C{col_id}")
                hdr.SetPosition(avg_x, max(0, all_y_min - 8), 0.6)
                hdr.SetScale(header_scale, header_scale, header_scale)
                hdr.GetTextProperty().SetFontSize(font_size + 2)
                hdr.GetTextProperty().SetBold(True)
                hdr.GetTextProperty().SetShadow(True)
                hdr.GetTextProperty().SetShadowOffset(1, -1)
                hdr.GetTextProperty().SetColor(1.0, 0.85, 0.0)  # Gold
                renderer.AddActor(hdr)

            # Row headers at left
            for row_id in sorted(row_positions.keys()):
                avg_y = np.mean(row_positions[row_id])
                hdr = vtk.vtkTextActor3D()
                hdr.SetInput(f"R{row_id}")
                hdr.SetPosition(max(0, all_x_min - 12), avg_y, 0.6)
                hdr.SetScale(header_scale, header_scale, header_scale)
                hdr.GetTextProperty().SetFontSize(font_size + 2)
                hdr.GetTextProperty().SetBold(True)
                hdr.GetTextProperty().SetShadow(True)
                hdr.GetTextProperty().SetShadowOffset(1, -1)
                hdr.GetTextProperty().SetColor(1.0, 0.85, 0.0)  # Gold
                renderer.AddActor(hdr)
        else:
            # Fallback: simple row_id label
            for stat, _z_off in visible_stats:
                label_text = str(stat['row_id'])
                caption = vtk.vtkTextActor3D()
                caption.SetInput(label_text)
                caption.SetPosition(stat['x_max'], stat['y_max'], 0.5)
                caption.SetScale(0.3, 0.3, 0.3)
                caption.GetTextProperty().SetFontSize(18)
                caption.GetTextProperty().SetBold(False)
                caption.GetTextProperty().SetShadow(False)
                caption.GetTextProperty().SetColor(0.2, 1.0, 1.0)
                renderer.AddActor(caption)

    def _draw_coronal_labels(self, renderer, visible_stats, has_grid, label_scale, font_size, slice_y):
        """Draw labels on coronal (XZ) view.
        
        Coronal view displays X (horizontal) vs Z (vertical, flipped).
        Shows column indices along X, with compact labels per visible bump.
        """
        vol_z = self.volume_data.shape[0] if self.volume_data is not None else 1

        if has_grid:
            # Collect column positions for header markers
            col_positions = {}  # col_id -> list of centroid_x
            for stat, z_off in visible_stats:
                gc = stat['grid_col']
                col_positions.setdefault(gc, []).append(stat['centroid_x'])

            # --- Per-bump labels: show column index ---
            # Group by unique (grid_row, grid_col) to avoid duplicate labels at same position
            seen = set()
            for stat, z_off in visible_stats:
                gr, gc = stat['grid_row'], stat['grid_col']
                key = (gr, gc)
                if key in seen:
                    continue
                seen.add(key)

                label_text = f"C{gc}"
                cx = stat['centroid_x']
                # In coronal view, Z is flipped: display_y = (vol_z - 1) - global_z
                global_z = stat['centroid_z'] + z_off
                cz_display = (vol_z - 1) - global_z

                caption = vtk.vtkTextActor3D()
                caption.SetInput(label_text)
                caption.SetPosition(cx, cz_display, 0.5)
                caption.SetScale(label_scale, label_scale, label_scale)
                caption.GetTextProperty().SetFontSize(font_size)
                caption.GetTextProperty().SetBold(True)
                caption.GetTextProperty().SetShadow(True)
                caption.GetTextProperty().SetShadowOffset(1, -1)
                caption.GetTextProperty().SetColor(0.35, 1.0, 0.55)  # Bright Green
                renderer.AddActor(caption)

            # --- Column header labels along top ---
            all_z_display_min = min((vol_z - 1) - (s['z_max'] + z_off) for s, z_off in visible_stats)
            header_scale = label_scale * 1.4

            for col_id in sorted(col_positions.keys()):
                avg_x = np.mean(col_positions[col_id])
                hdr = vtk.vtkTextActor3D()
                hdr.SetInput(f"C{col_id}")
                hdr.SetPosition(avg_x, max(0, all_z_display_min - 6), 0.6)
                hdr.SetScale(header_scale, header_scale, header_scale)
                hdr.GetTextProperty().SetFontSize(font_size + 2)
                hdr.GetTextProperty().SetBold(True)
                hdr.GetTextProperty().SetShadow(True)
                hdr.GetTextProperty().SetShadowOffset(1, -1)
                hdr.GetTextProperty().SetColor(1.0, 0.85, 0.0)  # Gold
                renderer.AddActor(hdr)
        else:
            for stat, z_off in visible_stats:
                label_text = str(stat['row_id'])
                cx = stat['centroid_x']
                global_z = stat['centroid_z'] + z_off
                cz_display = (vol_z - 1) - global_z
                caption = vtk.vtkTextActor3D()
                caption.SetInput(label_text)
                caption.SetPosition(cx, cz_display, 0.5)
                caption.SetScale(0.25, 0.25, 0.25)
                caption.GetTextProperty().SetFontSize(16)
                caption.GetTextProperty().SetBold(False)
                caption.GetTextProperty().SetShadow(True)
                caption.GetTextProperty().SetColor(0.35, 1.0, 0.55)
                renderer.AddActor(caption)

    def _draw_sagittal_labels(self, renderer, visible_stats, has_grid, label_scale, font_size, slice_x):
        """Draw labels on sagittal (YZ) view.
        
        Sagittal view displays Z (horizontal) vs Y (vertical) after transpose.
        Shows row indices along Y, with compact labels per visible bump.
        """
        if has_grid:
            # Collect row positions for header markers
            row_positions = {}  # row_id -> list of centroid_y
            for stat, z_off in visible_stats:
                gr = stat['grid_row']
                row_positions.setdefault(gr, []).append(stat['centroid_y'])

            # --- Per-bump labels: show row index ---
            seen = set()
            for stat, z_off in visible_stats:
                gr, gc = stat['grid_row'], stat['grid_col']
                key = (gr, gc)
                if key in seen:
                    continue
                seen.add(key)

                label_text = f"R{gr}"
                # In sagittal view: horizontal = Z, vertical = Y  (after transpose)
                cz = stat['centroid_z'] + z_off
                cy = stat['centroid_y']

                caption = vtk.vtkTextActor3D()
                caption.SetInput(label_text)
                caption.SetPosition(cz, cy, 0.5)
                caption.SetScale(label_scale, label_scale, label_scale)
                caption.GetTextProperty().SetFontSize(font_size)
                caption.GetTextProperty().SetBold(True)
                caption.GetTextProperty().SetShadow(True)
                caption.GetTextProperty().SetShadowOffset(1, -1)
                caption.GetTextProperty().SetColor(1.0, 0.55, 0.35)  # Bright Orange
                renderer.AddActor(caption)

            # --- Row header labels along left edge ---
            all_z_min = min(s['z_min'] + z_off for s, z_off in visible_stats)
            header_scale = label_scale * 1.4

            for row_id in sorted(row_positions.keys()):
                avg_y = np.mean(row_positions[row_id])
                hdr = vtk.vtkTextActor3D()
                hdr.SetInput(f"R{row_id}")
                hdr.SetPosition(max(0, all_z_min - 5), avg_y, 0.6)
                hdr.SetScale(header_scale, header_scale, header_scale)
                hdr.GetTextProperty().SetFontSize(font_size + 2)
                hdr.GetTextProperty().SetBold(True)
                hdr.GetTextProperty().SetShadow(True)
                hdr.GetTextProperty().SetShadowOffset(1, -1)
                hdr.GetTextProperty().SetColor(1.0, 0.85, 0.0)  # Gold
                renderer.AddActor(hdr)
        else:
            for stat, z_off in visible_stats:
                label_text = str(stat['row_id'])
                cz = stat['centroid_z'] + z_off
                cy = stat['centroid_y']
                caption = vtk.vtkTextActor3D()
                caption.SetInput(label_text)
                caption.SetPosition(cz, cy, 0.5)
                caption.SetScale(0.25, 0.25, 0.25)
                caption.GetTextProperty().SetFontSize(16)
                caption.GetTextProperty().SetBold(False)
                caption.GetTextProperty().SetShadow(True)
                caption.GetTextProperty().SetColor(1.0, 0.55, 0.35)
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
        
        # Load CSV button
        load_btn = QPushButton("📂 Load CSV")
        load_btn.setMaximumHeight(22)
        load_btn.setStyleSheet(f"""
            QPushButton {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 3px; padding: 2px 8px; font-size: 8pt;
            }}
            QPushButton:hover {{ background: {SemiconductorTheme.ACCENT_PRIMARY}; color: {SemiconductorTheme.BG_DARK}; }}
        """)
        load_btn.setToolTip("Load boundary_summary.csv from file")
        load_btn.clicked.connect(self._browse_boundary_csv)
        header.addWidget(load_btn)
        
        # Auto-detect button
        auto_btn = QPushButton("🔍 Auto-detect")
        auto_btn.setMaximumHeight(22)
        auto_btn.setStyleSheet(f"""
            QPushButton {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 3px; padding: 2px 8px; font-size: 8pt;
            }}
            QPushButton:hover {{ background: {SemiconductorTheme.ACCENT_PRIMARY}; color: {SemiconductorTheme.BG_DARK}; }}
        """)
        auto_btn.setToolTip("Auto-detect boundary_summary.csv from output path")
        auto_btn.clicked.connect(self._autodetect_boundary_csv)
        header.addWidget(auto_btn)
        
        self.bnd_info_label = QLabel("No results loaded")
        self._theme_register("_theme_secondary_labels", self.bnd_info_label)
        self.bnd_info_label.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        header.addWidget(self.bnd_info_label)
        layout.addLayout(header)
        
        # Table
        self.boundary_table = QTableWidget()
        self.boundary_table.setColumnCount(9)
        self.boundary_table.setHorizontalHeaderLabels([
            "#", "Bump (R,C)", "Label", "Bnd Voxels",
            "Gap X (µm)", "Gap Y (µm)", "Gap Z (µm)",
            "Euclidean (µm)", "Min Gap Pos"
        ])
        self.boundary_table.horizontalHeader().setStretchLastSection(True)
        self.boundary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.boundary_table.setSelectionMode(QTableWidget.SingleSelection)
        self.boundary_table.setAlternatingRowColors(True)
        self.boundary_table.verticalHeader().setDefaultSectionSize(22)
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

    def _load_boundary_csv(self, csv_path):
        """Load boundary CSV (supports both boundary_summary.csv and boundary_gap.csv formats)."""
        import csv
        self.boundary_results = []
        try:
            with open(csv_path, 'r') as f:
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
        
        # Auto-detect format by checking column names
        first_row = self.boundary_results[0]
        is_gap_format = 'Src_row' in first_row and 'Direction' in first_row
        self._boundary_csv_format = 'gap' if is_gap_format else 'summary'
        
        self.boundary_table.setSortingEnabled(False)
        
        if is_gap_format:
            # boundary_gap.csv format: Src→Dst per direction
            self.boundary_table.setColumnCount(10)
            self.boundary_table.setHorizontalHeaderLabels([
                "#", "Source (R,C)", "Dir", "Dest (R,C)",
                "Gap X (µm)", "Gap Y (µm)", "Gap Z (µm)",
                "Euclidean (µm)", "Src Voxel", "Dst Voxel"
            ])
            self.boundary_table.setRowCount(len(self.boundary_results))
            
            for i, r in enumerate(self.boundary_results):
                eucl_str = r.get('Min_dist_Euclidean_um', 'NaN')
                try:
                    eucl_val = float(eucl_str)
                    is_close = eucl_val < 1.5
                except (ValueError, TypeError):
                    is_close = False
                
                row_bg = QColor(180, 40, 40, 50) if is_close else QColor(40, 180, 60, 30)
                
                src_str = f"R{r.get('Src_row','?')},C{r.get('Src_col','?')}"
                dst_str = f"R{r.get('Dst_row','?')},C{r.get('Dst_col','?')}"
                direction = r.get('Direction', '?')
                src_pos = f"Z={r.get('Src_voxel_Z','')},Y={r.get('Src_voxel_Y','')},X={r.get('Src_voxel_X','')}"
                dst_pos = f"Z={r.get('Dst_voxel_Z','')},Y={r.get('Dst_voxel_Y','')},X={r.get('Dst_voxel_X','')}"
                
                items_data = [
                    str(i + 1),
                    src_str, direction, dst_str,
                    r.get('Min_dist_X_um', 'NaN'),
                    r.get('Min_dist_Y_um', 'NaN'),
                    r.get('Min_dist_Z_um', 'NaN'),
                    eucl_str, src_pos, dst_pos,
                ]
                for col, text in enumerate(items_data):
                    item = QTableWidgetItem(str(text))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setBackground(row_bg)
                    # Map visual row → boundary_results index after sort
                    item.setData(Qt.UserRole, i)
                    if col == 7:  # Euclidean column
                        item.setFont(QFont("Arial", 9, QFont.Bold))
                        if is_close:
                            item.setForeground(QColor(255, 100, 100))
                        else:
                            item.setForeground(QColor(100, 255, 120))
                    if col == 2:  # Direction column
                        dir_colors = {'Up': QColor(100,200,255), 'Down': QColor(100,200,255),
                                      'Left': QColor(255,200,100), 'Right': QColor(255,200,100)}
                        item.setForeground(dir_colors.get(direction, QColor(200,200,200)))
                    self.boundary_table.setItem(i, col, item)
        else:
            # boundary_summary.csv format (original)
            self.boundary_table.setColumnCount(9)
            self.boundary_table.setHorizontalHeaderLabels([
                "#", "Bump (R,C)", "Label", "Bnd Voxels",
                "Gap X (µm)", "Gap Y (µm)", "Gap Z (µm)",
                "Euclidean (µm)", "Min Gap Pos"
            ])
            self.boundary_table.setRowCount(len(self.boundary_results))
            
            for i, r in enumerate(self.boundary_results):
                eucl_str = r.get('min_gap_euclidean_um', 'NaN')
                try:
                    eucl_val = float(eucl_str)
                    is_close = eucl_val < 1.0
                except (ValueError, TypeError):
                    is_close = False
                
                row_bg = QColor(180, 40, 40, 50) if is_close else QColor(40, 180, 60, 30)
                pos_str = f"Z={r.get('min_gap_voxel_Z','')},Y={r.get('min_gap_voxel_Y','')},X={r.get('min_gap_voxel_X','')}"
                
                items_data = [
                    str(i + 1),
                    r.get('Bump_id', ''),
                    r.get('label', ''),
                    r.get('boundary_voxels', ''),
                    r.get('min_gap_X_um', 'NaN'),
                    r.get('min_gap_Y_um', 'NaN'),
                    r.get('min_gap_Z_um', 'NaN'),
                    eucl_str, pos_str,
                ]
                for col, text in enumerate(items_data):
                    item = QTableWidgetItem(str(text))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setBackground(row_bg)
                    item.setData(Qt.UserRole, i)
                    if col == 7:
                        item.setFont(QFont("Arial", 9, QFont.Bold))
                        if is_close:
                            item.setForeground(QColor(255, 100, 100))
                        else:
                            item.setForeground(QColor(100, 255, 120))
                    self.boundary_table.setItem(i, col, item)
        
        self.boundary_table.setSortingEnabled(True)
        self.boundary_table.resizeColumnsToContents()
        fmt_name = "boundary_gap" if is_gap_format else "boundary_summary"
        z_info = f" | Z+{self._boundary_z_offset} ({self._boundary_layer_name})" if self._boundary_z_offset > 0 else ""
        self.bnd_info_label.setText(
            f"{len(self.boundary_results)} rows | {fmt_name} | {os.path.basename(csv_path)}"
            f"{z_info} | click row → 3D gap"
        )
        self.boundary_table.setToolTip(
            "Click a row to map the 3D boundary gap (Src→Dst) on the Teaching 3D Volume view."
        )
        
        # Switch to boundary tab (tab index 5)
        if hasattr(self, '_set_teaching_tab'):
            self._set_teaching_tab(5)

    def _on_boundary_row_clicked(self, row, col):
        """Map selected boundary gap onto the 3D Volume only (not MPR).

        Gap is a true 3D spatial distance — MPR slices make it hard to interpret.
        Auto-enables Teaching 3D view when needed.
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

        # Layer Z offset (local layer coords → global volume coords)
        z_offset = int(getattr(self, '_boundary_z_offset', 0) or 0)
        # Prefer per-row z_start if present (combined multi-layer CSV)
        if r.get("z_start") not in (None, ""):
            try:
                z_offset = int(r.get("z_start"))
            except (TypeError, ValueError):
                pass

        fmt = getattr(self, '_boundary_csv_format', 'summary')

        def _iv(key, default=0):
            try:
                return int(float(r.get(key, default)))
            except (TypeError, ValueError):
                return default

        # Ensure 3D volume panel is on so the gap is visible
        if not getattr(self, '_3d_view_active', False):
            if hasattr(self, '_3d_enable_check'):
                self._3d_enable_check.blockSignals(True)
                self._3d_enable_check.setChecked(True)
                self._3d_enable_check.blockSignals(False)
            self._toggle_teaching_3d(True)

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
                eucl = r.get('Min_dist_Euclidean_um', '?')
                direction = r.get('Direction', '')
                src_rc = f"R{r.get('Src_row','?')}C{r.get('Src_col','?')}"
                dst_rc = f"R{r.get('Dst_row','?')}C{r.get('Dst_col','?')}"
                layer = str(r.get('Layer', '') or getattr(self, '_boundary_layer_name', '') or '')
                label = f"{layer} {src_rc}→{dst_rc} ({direction}) {eucl}µm".strip()

                self._draw_boundary_gap_line(
                    src_voxel=src,
                    dst_voxel=dst,
                    label_text=label,
                )
                if hasattr(self, 'bnd_info_label'):
                    self.bnd_info_label.setText(f"3D gap: {label}")
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
                self.bnd_info_label.setText(f"3D gap: {label}")
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
        
        # Orientation cube (Dragonfly-style, same as viewer)
        cube = vtk.vtkAnnotatedCubeActor()
        cube.SetXPlusFaceText("+X")
        cube.SetXMinusFaceText("-X")
        cube.SetYPlusFaceText("+Y")
        cube.SetYMinusFaceText("-Y")
        cube.SetZPlusFaceText("+Z")
        cube.SetZMinusFaceText("-Z")
        cube.GetCubeProperty().SetColor(0.95, 0.95, 0.95)
        cube.GetTextEdgesProperty().SetColor(0.2, 0.2, 0.2)
        cube.GetTextEdgesProperty().SetLineWidth(1)
        for prop in [
            cube.GetXPlusFaceProperty(), cube.GetXMinusFaceProperty(),
            cube.GetYPlusFaceProperty(), cube.GetYMinusFaceProperty(),
            cube.GetZPlusFaceProperty(), cube.GetZMinusFaceProperty(),
        ]:
            prop.SetColor(0.1, 0.1, 0.1)
        self._3d_axes_widget = vtk.vtkOrientationMarkerWidget()
        self._3d_axes_widget.SetOrientationMarker(cube)
        self._3d_axes_widget.SetInteractor(interactor)
        self._3d_axes_widget.SetViewport(0.85, 0.0, 1.0, 0.15)
        self._3d_axes_widget.EnabledOn()
        self._3d_axes_widget.InteractiveOff()
        
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
            self._render_teaching_3d()

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
        # Remove old seg actor
        if getattr(self, '_3d_seg_actor', None):
            renderer.RemoveActor(self._3d_seg_actor)
            self._3d_seg_actor = None
        
        vol = self.volume_data
        mode = self._3d_mode_combo.currentText() if hasattr(self, '_3d_mode_combo') else "Volume"
        
        # Determine which data to render
        if mode == "Seg Only" and self.bump_segmentation is not None:
            render_data = (self.bump_segmentation > 0).astype(np.uint8) * 255
        else:
            render_data = vol
        
        # Full resolution — same as viewer.py (no downsample)
        z, y, x = render_data.shape
        
        # Use custom spacing if available (same as viewer.py)
        spacing = getattr(self, 'custom_spacing', None) or getattr(self, 'spacing', (1.0, 1.0, 1.0))
        
        # Convert to VTK (using Fortran-order transpose like viewer)
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(x, y, z)
        vtk_data.SetSpacing(spacing[0], spacing[1], spacing[2])
        
        data_fortran = np.transpose(render_data, (2, 1, 0))
        flat = np.ascontiguousarray(data_fortran.flatten('F'))
        if render_data.dtype == np.uint16:
            vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
        elif render_data.dtype == np.uint8:
            vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        else:
            flat = flat.astype(np.float32)
            vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_data.GetPointData().SetScalars(vtk_arr)
        self._3d_vtk_data = vtk_data
        
        # GPU Volume Mapper (same as viewer.py)
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        mapper.SetInputData(vtk_data)
        
        # Quality preset
        quality_name = self._3d_quality_combo.currentText() if hasattr(self, '_3d_quality_combo') else "Draft"
        preset = self._3d_quality_presets.get(quality_name, self._3d_quality_presets["Draft"])
        
        mapper.SetAutoAdjustSampleDistances(1 if preset["auto_adj"] else 0)
        mapper.SetSampleDistance(preset["sample_dist"])
        if hasattr(mapper, 'SetImageSampleDistance'):
            mapper.SetImageSampleDistance(preset["image_sample"])
        if hasattr(mapper, 'SetUseJittering'):
            mapper.SetUseJittering(1 if preset["jitter"] else 0)
        if hasattr(mapper, 'SetMaxMemoryFraction'):
            mapper.SetMaxMemoryFraction(preset.get("max_mem_fraction", 0.80))
        if hasattr(mapper, 'SetBlendModeToComposite'):
            mapper.SetBlendModeToComposite()
        
        # Store mapper reference for LOD interaction updates
        self._3d_volume_mapper = mapper
        
        # NOTE: Clip planes applied AFTER render via _apply_3d_clipping (GPU cropping)
        
        # Volume property (Dragonfly-style Phong shading — same as viewer.py)
        vol_prop = vtk.vtkVolumeProperty()
        vol_prop.ShadeOn()
        vol_prop.SetInterpolationTypeToLinear()
        vol_prop.SetAmbient(0.05)
        vol_prop.SetDiffuse(0.75)
        vol_prop.SetSpecular(0.95)
        vol_prop.SetSpecularPower(80.0)
        
        # Apply viewer-grade transfer function
        self._apply_3d_transfer_function(vol_prop)
        
        volume_actor = vtk.vtkVolume()
        volume_actor.SetMapper(mapper)
        volume_actor.SetProperty(vol_prop)
        
        renderer.AddVolume(volume_actor)
        self._3d_volume_actor = volume_actor
        self._3d_ds = 1  # Full resolution — no downsample
        self._3d_spacing = (spacing[0], spacing[1], spacing[2])
        self._3d_original_spacing = (spacing[0], spacing[1], spacing[2])
        
        # Seg overlay mode: add bump isosurface
        if mode == "Seg Overlay" and self.bump_segmentation is not None:
            seg_z, seg_y, seg_x = self.bump_segmentation.shape
            bump_binary = (self.bump_segmentation > 0).astype(np.uint8) * 255
            
            vtk_bump = vtk.vtkImageData()
            vtk_bump.SetDimensions(seg_x, seg_y, seg_z)
            vtk_bump.SetSpacing(spacing[0], spacing[1], spacing[2])
            bump_fort = np.transpose(bump_binary, (2, 1, 0))
            bump_flat = np.ascontiguousarray(bump_fort.flatten('F'))
            bump_arr = numpy_support.numpy_to_vtk(bump_flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            vtk_bump.GetPointData().SetScalars(bump_arr)
            
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(vtk_bump)
            contour.SetValue(0, 128)
            contour.Update()
            
            seg_mapper = vtk.vtkPolyDataMapper()
            seg_mapper.SetInputConnection(contour.GetOutputPort())
            # Apply same clip planes to seg
            for ax in ("x", "y", "z"):
                if self._3d_clip_enabled.get(ax) and self._3d_clip_planes.get(ax):
                    seg_mapper.AddClippingPlane(self._3d_clip_planes[ax])
            seg_actor = vtk.vtkActor()
            seg_actor.SetMapper(seg_mapper)
            seg_actor.GetProperty().SetColor(1.0, 0.85, 0.0)
            seg_actor.GetProperty().SetOpacity(0.4)
            renderer.AddActor(seg_actor)
            self._3d_seg_actor = seg_actor

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

    def _clear_boundary_actors(self):
        """Remove all boundary gap line actors from the 3D scene."""
        for actor in getattr(self, '_3d_boundary_actors', []):
            self._3d_renderer.RemoveActor(actor)
        self._3d_boundary_actors = []

    def _draw_boundary_gap_line(self, src_voxel, dst_voxel, label_text=""):
        """Draw a 3D line between two voxel positions with endpoint spheres and label.
        
        Args:
            src_voxel: (z, y, x) voxel coords of source bump boundary
            dst_voxel: (z, y, x) voxel coords of destination bump boundary
            label_text: text label to display at midpoint
        """
        if not self._3d_view_active or not hasattr(self, '_3d_renderer'):
            return
        
        # Clear previous boundary actors
        self._clear_boundary_actors()
        
        # Convert voxel coords to 3D world coords (VTK uses X, Y, Z ordering)
        # Use ORIGINAL spacing (not ds-scaled) since CSV voxel coords are in original space
        # world_coord = original_voxel * original_spacing
        spacing = getattr(self, '_3d_original_spacing', (1.0, 1.0, 1.0))
        src_x = src_voxel[2] * spacing[0]
        src_y = src_voxel[1] * spacing[1]
        src_z = src_voxel[0] * spacing[2]
        dst_x = dst_voxel[2] * spacing[0]
        dst_y = dst_voxel[1] * spacing[1]
        dst_z = dst_voxel[0] * spacing[2]
        
        mid_x = (src_x + dst_x) / 2.0
        mid_y = (src_y + dst_y) / 2.0
        mid_z = (src_z + dst_z) / 2.0
        
        # --- 1. Line between src and dst ---
        line_source = vtk.vtkLineSource()
        line_source.SetPoint1(src_x, src_y, src_z)
        line_source.SetPoint2(dst_x, dst_y, dst_z)
        line_source.Update()
        
        # Create a tube filter for thick, visible line
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputConnection(line_source.GetOutputPort())
        tube_filter.SetRadius(1.5)  # Line thickness
        tube_filter.SetNumberOfSides(12)
        tube_filter.Update()
        
        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputConnection(tube_filter.GetOutputPort())
        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(1.0, 0.2, 0.2)  # Red line
        line_actor.GetProperty().SetOpacity(0.95)
        line_actor.GetProperty().SetLighting(False)  # Unlit for visibility
        self._3d_renderer.AddActor(line_actor)
        self._3d_boundary_actors.append(line_actor)
        
        # --- 2. Source endpoint sphere (cyan) ---
        src_sphere = vtk.vtkSphereSource()
        src_sphere.SetCenter(src_x, src_y, src_z)
        src_sphere.SetRadius(3.0)
        src_sphere.SetPhiResolution(16)
        src_sphere.SetThetaResolution(16)
        src_mapper = vtk.vtkPolyDataMapper()
        src_mapper.SetInputConnection(src_sphere.GetOutputPort())
        src_actor = vtk.vtkActor()
        src_actor.SetMapper(src_mapper)
        src_actor.GetProperty().SetColor(0.0, 1.0, 1.0)  # Cyan
        src_actor.GetProperty().SetOpacity(0.9)
        self._3d_renderer.AddActor(src_actor)
        self._3d_boundary_actors.append(src_actor)
        
        # --- 3. Destination endpoint sphere (yellow) ---
        dst_sphere = vtk.vtkSphereSource()
        dst_sphere.SetCenter(dst_x, dst_y, dst_z)
        dst_sphere.SetRadius(3.0)
        dst_sphere.SetPhiResolution(16)
        dst_sphere.SetThetaResolution(16)
        dst_mapper = vtk.vtkPolyDataMapper()
        dst_mapper.SetInputConnection(dst_sphere.GetOutputPort())
        dst_actor = vtk.vtkActor()
        dst_actor.SetMapper(dst_mapper)
        dst_actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow
        dst_actor.GetProperty().SetOpacity(0.9)
        self._3d_renderer.AddActor(dst_actor)
        self._3d_boundary_actors.append(dst_actor)
        
        # --- 4. Text label at midpoint ---
        if label_text:
            text_actor = vtk.vtkBillboardTextActor3D()
            text_actor.SetInput(label_text)
            text_actor.SetPosition(mid_x, mid_y, mid_z + 5.0)  # Offset up slightly
            text_actor.GetTextProperty().SetFontSize(14)
            text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
            text_actor.GetTextProperty().BoldOn()
            text_actor.GetTextProperty().SetBackgroundColor(0.0, 0.0, 0.0)
            text_actor.GetTextProperty().SetBackgroundOpacity(0.6)
            self._3d_renderer.AddActor(text_actor)
            self._3d_boundary_actors.append(text_actor)
        
        # Update header info
        if hasattr(self, '_3d_bnd_info'):
            self._3d_bnd_info.setText(f"Gap: {label_text}")

        # Keep gap geometry under the same clip planes as the volume
        self._apply_3d_clipping(render=True)

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
        if not input_path:
            QMessageBox.warning(self, "Warning", "Please load a volume first.")
            return

        output_path = self.output_path_input.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Warning", "Please specify output path.")
            return

        # Determine input dir (folder of TIFF slices)
        if os.path.isdir(input_path):
            input_dir = input_path
        else:
            input_dir = os.path.dirname(input_path)

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
            lambda ok, err: self._on_enhancement_finished(ok, err, progress)
        )
        self.enhancement_thread.start()

    def _on_enhancement_finished(self, success, error, progress):
        """Handle enhancement completion."""
        if progress:
            progress.close()
        if success:
            self.enh_status_label.setText("Enhancement: Complete ✓")
            self.enh_status_label.setStyleSheet(self._status_label_qss(SemiconductorTheme.ACCENT_SUCCESS))
            QMessageBox.information(self, "Enhancement Complete",
                "Volume enhancement finished successfully.\n"
                "Enhanced images saved to enhanced_volume/ subfolder.")
        else:
            self.enh_status_label.setText("Enhancement: Failed ✗")
            self.enh_status_label.setStyleSheet(self._status_label_qss(SemiconductorTheme.ACCENT_ERROR))
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
    
    def load_dll_folder(self):
        """Load DLL folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select DLL Folder")
        if folder:
            try:
                bumpvoid.load_dll(folder)
                self.dll_path = folder
                self.dll_path_input.setText(folder)
                self.info_label.setText(f"DLL loaded: {bumpvoid.get_version()}")
                self.check_ready_state()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load DLL: {str(e)}")
                
    def load_config_file(self, file_path=None):
        """Load configuration file"""
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Config File", "", "Config Files (*.txt *.cfg);;All Files (*.*)"
            )
        if file_path:
            try:
                self.config = bumpvoid.load_config(file_path)
                self.config_path = file_path
                self.config_path_input.setText(file_path)
                self.populate_parameters_from_config()
                
                # Apply pending input path if volume was loaded before config
                if self._pending_input_path:
                    self.config.inputPath = self._pending_input_path.encode('utf-8')
                    print(f"Applied pending inputPath: {self._pending_input_path}")
                    self.populate_parameters_from_config()
                
                # Manually parse Voxel Sizes from config file if present
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
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
                        
                        if vx_match: self.param_widgets['voxel_size_x'].setValue(float(vx_match.group(1)))
                        if vy_match: self.param_widgets['voxel_size_y'].setValue(float(vy_match.group(1)))
                        if vz_match: self.param_widgets['voxel_size_z'].setValue(float(vz_match.group(1)))
                        
                        if bumpmin_match: self.param_widgets['bump_minimum_size'].setText(bumpmin_match.group(1))
                        if bumpmax_match: self.param_widgets['bump_maximum_size'].setText(bumpmax_match.group(1))
                        if voidmin_match: self.param_widgets['void_minimum_size'].setText(voidmin_match.group(1))
                        if voidmax_match: self.param_widgets['void_maximum_size'].setText(voidmax_match.group(1))
                        
                        # Parse Z-Stretched 4x toggle
                        z4x_match = re.search(r'Z_STRETCHED_4X\s*=\s*(true|false)', content, re.IGNORECASE)
                        if z4x_match and 'z_stretched_4x' in self.param_widgets:
                            self.param_widgets['z_stretched_4x'].setChecked(z4x_match.group(1).lower() == 'true')
                        
                        # Parse NG threshold
                        ng_match = re.search(r'NG_THRESHOLD\s*=\s*([\d.]+)', content)
                        if ng_match and hasattr(self, 'ng_threshold_spin'):
                            self.ng_threshold_spin.setValue(float(ng_match.group(1)))
                        
                        # Parse CC3D parameters
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

                        # Parse Enhancement parameters from config
                        self._apply_enhancement_from_config_text(content)
                except Exception as e:
                    print(f"Note: Could not parse Measurement Parameters from config: {e}")
                
                # Parse Layer Definitions from config if present
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
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
                                    selected = parts[3].strip().lower() == 'true' if len(parts) >= 4 else True
                                    loaded_layers.append({
                                        'id': i, 'name': name, 'z_start': z_start, 'z_end': z_end, 'selected': selected
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
        """Load 16-bit volume data"""
        if from_folder:
            file_path = QFileDialog.getExistingDirectory(
                self, "Select Folder Containing Image Files"
            )
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image Volume", "", "Image Files (*.tif *.tiff *.raw *.bin)"
            )
        
        if file_path:
            self._handle_volume_load(file_path)
            
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
        
        # Blank slices
        self.config.blankStart1 = self.param_widgets['blankStart1'].value()
        self.config.blankEnd1 = self.param_widgets['blankEnd1'].value()
        self.config.blankStart2 = self.param_widgets['blankStart2'].value()
        self.config.blankEnd2 = self.param_widgets['blankEnd2'].value()
        
        # Options
        self.config.saveBumpIntermediate = int(self.param_widgets['saveBumpIntermediate'].isChecked())
        self.config.saveVoidIntermediate = int(self.param_widgets['saveVoidIntermediate'].isChecked())
        self.config.showResult = int(self.param_widgets['showResult'].isChecked())
        
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
            
            # Map layer assignments for stats table
            self.layer_assignment_map = {}  # z-index -> layer name string
            if self.layers_active:
                for layer in [l for l in self.layer_definitions if l['selected']]:
                    for z in range(layer['z_start'], layer['z_end']):
                         self.layer_assignment_map[z] = layer['name']

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
                if len(valid_results) == 1:
                    res = valid_results[0]
                    QMessageBox.information(self, "Success", 
                        f"Inspection completed successfully! Please click 'MES' to proceed.\n"
                        f"Bump Time: {res.bumpTime:.2f}s\n"
                        f"Void Time: {res.voidTime:.2f}s\n"
                        f"Total Time: {res.totalTime:.2f}s")
                else:
                    QMessageBox.information(self, "Success", 
                        f"Multi-layer Inspection completed successfully! Please click 'MES' to proceed.\n"
                        f"Total Time: {total_time:.2f}s")
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load results: {str(e)}")
            
    def run_measurement(self):
        """Proceed to measurement stage using current masks"""
        # We don't overwrite bump_segmentation anymore; measurement iterates over _ml_layer_cache
        if self.bump_segmentation is None and self.void_segmentation is None:
            QMessageBox.warning(self, "Warning", "Please run inspection first to generate masks before measuring.")
            return
            
        self.run_object_analysis()
        # MES tab index = 4 (Layers=0, ROI=1, Params=2, Enhance=3, MES=4, B2B=5, 3D=6)
        self._set_teaching_tab(4)

    def run_object_analysis(self):
        """Run skimage-based per-object measurement on current masks."""
        if self.bump_segmentation is None:
            return
            
        progress = QProgressDialog("Running measurement analysis...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        
        try:
            self.object_stats = []
            
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
                    if idx % 50 == 0: QApplication.processEvents()
                    
                    c1_volume = prop.area * voxel_volume
                    c2_volume = c2_volumes_map.get(prop.label, 0.0)
                    
                    ratio = c2_volume / (c1_volume + c2_volume) if (c1_volume + c2_volume) > 0 else 0.0
                    bbox = prop.bbox
                    centroid = prop.centroid
                    
                    soh = (bbox[3] - bbox[0]) * vz  # Solder height in um
                    self.object_stats.append({
                        'row_id': idx + 1,
                        'label': prop.label,
                        'c1_volume': c1_volume,
                        'c2_volume': c2_volume,
                        'ratio': ratio,
                        'soh': soh,
                        'z_min': bbox[0], 'y_min': bbox[1], 'x_min': bbox[2],
                        'z_max': bbox[3], 'y_max': bbox[4], 'x_max': bbox[5],
                        'centroid_z': centroid[0],
                        'centroid_z_global': centroid[0],  # Same as centroid_z in single mode
                        'centroid_y': centroid[1], 'centroid_x': centroid[2],
                    })
            
            progress.setLabelText("Populating table...")
            progress.setValue(92)
            QApplication.processEvents()
            
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
            
            progress.setValue(100)
            progress.close()
            
        except Exception as e:
            progress.close()
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
            _z_off = getattr(self, '_measurement_start_slice', 0)
            # Use stored layer_name if available (multi-layer mode), otherwise lookup via global Z
            if 'layer_name' in stat:
                layer_name = stat['layer_name']
            elif hasattr(self, 'layer_assignment_map'):
                global_z = int(stat.get('centroid_z_global', stat['centroid_z'])) + _z_off
                layer_name = self.layer_assignment_map.get(global_z, "N/A")
            else:
                layer_name = "N/A"
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
                # Use stored layer_name if available (multi-layer mode), otherwise lookup
                if 'layer_name' in stat:
                    layer_name = stat['layer_name']
                elif hasattr(self, 'layer_assignment_map'):
                    _z_off = getattr(self, '_measurement_start_slice', 0)
                    global_z = int(stat.get('centroid_z_global', stat['centroid_z'])) + _z_off
                    layer_name = self.layer_assignment_map.get(global_z, "N/A")
                else:
                    layer_name = "N/A"
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
        selected_rows = sorted(list(set(index.row() for index in self.object_stats_table.selectedIndexes())))
        
        self.selected_highlight_objects = []
        target_centroid = None
        _z_offset = getattr(self, '_measurement_start_slice', 0)
        
        # Map visual rows to actual stats indices via UserRole
        current_item = self.object_stats_table.currentItem()
        target_visual_row = current_item.row() if current_item else -1
        
        for visual_row in selected_rows:
            # Get the original stats index from the first column's UserRole
            id_item = self.object_stats_table.item(visual_row, 0)
            if id_item is None:
                continue
            stats_idx = id_item.data(Qt.UserRole)
            if stats_idx is None or stats_idx >= len(self.object_stats):
                continue
            stat = self.object_stats[stats_idx]
            self.selected_highlight_objects.append((1, stat['label']))
            if visual_row == target_visual_row:
                if getattr(self, 'input_mode', 'single') == 'multi_layer':
                    if hasattr(self, 'layer_filter_combo'):
                        layer_name = stat.get('layer_name')
                        if layer_name and layer_name != self.current_display_layer:
                            idx = self.layer_filter_combo.findData(layer_name)
                            if idx >= 0:
                                self.layer_filter_combo.setCurrentIndex(idx)
                                
                nav_z = int(stat['centroid_z'])
                target_centroid = (
                    nav_z + _z_offset,
                    int(stat['centroid_y']),
                    int(stat['centroid_x']),
                )
        
        # Fallback to the first selected item
        if target_centroid is None and selected_rows:
            id_item = self.object_stats_table.item(selected_rows[0], 0)
            if id_item is not None:
                stats_idx = id_item.data(Qt.UserRole)
                if stats_idx is not None and stats_idx < len(self.object_stats):
                    stat = self.object_stats[stats_idx]
                    
                    # Ensure we switch to the right layer to visualize this object
                    if getattr(self, 'input_mode', 'single') == 'multi_layer':
                        if hasattr(self, 'layer_filter_combo'):
                            layer_name = stat.get('layer_name')
                            if layer_name and layer_name != self.current_display_layer:
                                idx = self.layer_filter_combo.findData(layer_name)
                                if idx >= 0:
                                    self.layer_filter_combo.setCurrentIndex(idx)
                    
                    nav_z = int(stat['centroid_z'])  # Use local Z
                    target_centroid = (nav_z + _z_offset, int(stat['centroid_y']), int(stat['centroid_x']))

        if target_centroid is not None and self.volume_data is not None:
            cz, cy, cx = target_centroid
            z_max = self.volume_data.shape[0] - 1
            y_max = self.volume_data.shape[1] - 1
            x_max = self.volume_data.shape[2] - 1
            
            self.axial_slice_slider.setValue(min(max(0, cz), z_max))
            self.coronal_slice_slider.setValue(min(max(0, cy), y_max))
            self.sagittal_slice_slider.setValue(min(max(0, cx), x_max))
        
        if self.volume_data is not None:
            for ori in ['axial', 'coronal', 'sagittal']:
                self.update_plane_view(ori)

    def clear_object_selection(self):
        self.object_stats_table.blockSignals(True)
        self.object_stats_table.clearSelection()
        self.object_stats_table.blockSignals(False)
        self.selected_highlight_objects = []
        if self.volume_data is not None:
            for ori in ['axial', 'coronal', 'sagittal']:
                self.update_plane_view(ori)

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

        self.bnd_thread = BoundaryAnalysisThread(
            layers_to_process, spacing,
            object_stats=self.object_stats,
            kernel_size=kernel_size, iterations=iterations,
            save_boundary_tif=save_tif,
            dll_folder=self.dll_path_input.text().strip()
        )

        def handle_bnd_progress(val, text):
            self.bnd_progress.setValue(val)
            self.bnd_progress.setLabelText(text)

        def handle_bnd_finished(success, message):
            self.bnd_progress.close()
            if success:
                QMessageBox.information(self, "Success", message)
                # Auto-load boundary results into the table
                output_base = self.output_path_input.text().strip()
                if not output_base:
                    output_base = os.path.dirname(self.config_path) if self.config_path else os.getcwd()
                # Try finding boundary_summary.csv in output_base or sub-directories
                candidates = [
                    os.path.join(output_base, "boundary_summary.csv"),
                ]
                # Also check per-layer subdirectories
                if os.path.isdir(output_base):
                    for sub in os.listdir(output_base):
                        sub_path = os.path.join(output_base, sub, "boundary_summary.csv")
                        if os.path.isfile(sub_path):
                            candidates.append(sub_path)
                for csv_path in candidates:
                    if os.path.isfile(csv_path):
                        self._load_boundary_csv(csv_path)
                        break
            else:
                QMessageBox.critical(self, "Error", message)

        self.bnd_thread.progress.connect(handle_bnd_progress)
        self.bnd_thread.finished.connect(handle_bnd_finished)
        self.bnd_progress.canceled.connect(self.bnd_thread.cancel)
        self.bnd_thread.start()

    def run_false_alarm_remover(self):
        """False Alarm Remover: filters out TGV/Void based on minimum/maximum voxel constraints."""
        def parse_size(text):
            try:
                parts = [float(p.strip()) for p in str(text).lower().split('x') if p.strip()]
                if not parts: return 0.0
                res = 1.0
                for p in parts: res *= p
                return res
            except:
                return 0.0
                
        tgv_min = parse_size(self.param_widgets['bump_minimum_size'].text())
        tgv_max = parse_size(self.param_widgets['bump_maximum_size'].text())
        void_min = parse_size(self.param_widgets['void_minimum_size'].text())
        void_max = parse_size(self.param_widgets['void_maximum_size'].text())

        if not self.object_stats:
            QMessageBox.information(self, "FAR", "No objects to filter. Please run Measurement first.")
            return

        vx = self.param_widgets['voxel_size_x'].value()
        vy = self.param_widgets['voxel_size_y'].value()
        vz = self.param_widgets['voxel_size_z'].value()
        
        # If Z-Stretched 4x mode is enabled, divide effective Z voxel size by 4
        if self.param_widgets.get('z_stretched_4x') and self.param_widgets['z_stretched_4x'].isChecked():
            vz = vz / 4.0
        
        voxel_volume = vx * vy * vz
        if voxel_volume <= 0: voxel_volume = 1.0

        tgv_labels_to_remove = []
        void_labels_to_remove = []
        rows_to_remove = []

        for row_idx, stat in enumerate(self.object_stats):
            c1_voxels = stat['c1_volume'] / voxel_volume
            c2_voxels = stat['c2_volume'] / voxel_volume
            
            remove_bump = False
            remove_void = False
            
            if c1_voxels < tgv_min or c1_voxels > tgv_max:
                remove_bump = True
            if c2_voxels > 0 and (c2_voxels < void_min or c2_voxels > void_max):
                remove_void = True
                
            if remove_bump:
                tgv_labels_to_remove.append(stat['label'])
                rows_to_remove.append(row_idx)
            elif remove_void:
                void_labels_to_remove.append(stat['label'])
                stat['c2_volume'] = 0.0
                stat['ratio'] = 0.0

        for row_idx in sorted(rows_to_remove, reverse=True):
            self.object_stats.pop(row_idx)

        if self.labeled_class1_data is not None:
            if tgv_labels_to_remove:
                 mask_bump = np.isin(self.labeled_class1_data, tgv_labels_to_remove)
                 self.labeled_class1_data[mask_bump] = 0
                 if self.bump_segmentation is not None:
                     self.bump_segmentation[mask_bump] = 0
                 if self.void_segmentation is not None:
                     self.void_segmentation[mask_bump] = 0
            if void_labels_to_remove:
                 mask_void = np.isin(self.labeled_class1_data, void_labels_to_remove)
                 if self.void_segmentation is not None:
                     self.void_segmentation[mask_void] = 0

        self.selected_highlight_objects = []
        self.object_stats_table.clearSelection()
        self._refresh_stats_table()

        for orientation in ['axial', 'coronal', 'sagittal']:
            self.update_plane_view(orientation)

        total_c1 = sum(s['c1_volume'] for s in self.object_stats)
        total_c2 = sum(s['c2_volume'] for s in self.object_stats)
        self.stats_info_label.setText(
            f"{len(self.object_stats)} objects | "
            f"Total Bump: {total_c1:,.0f} \u03bcm\u00b3 | Total Void: {total_c2:,.0f} \u03bcm\u00b3"
        )
        
        QMessageBox.information(self, "FAR Complete", f"Removed {len(tgv_labels_to_remove)} Bump objects and {len(void_labels_to_remove)} isolated Voids.")
    def _refresh_stats_table(self):
        """Update the table rows based on the current object_stats list, re-indexing them as (Row, Col) per layer."""
        self.object_stats_table.setSortingEnabled(False)
        self.object_stats_table.blockSignals(True)
        
        GRID_ROWS = 5
        GRID_COLS = 5
        
        if self.object_stats:
            vx, vy = 1.0, 1.0
            if hasattr(self, 'param_widgets'):
                sw = self.param_widgets
                if 'voxel_size_x' in sw:
                    vx = sw['voxel_size_x'].value()
                    vy = sw['voxel_size_y'].value()
                    
            # Group objects by layer using layer_assignment_map
            layer_groups = {}
            for stat in self.object_stats:
                layer_name = "N/A"
                if 'layer_name' in stat:
                    layer_name = stat['layer_name']
                elif hasattr(self, 'layer_assignment_map') and self.layer_assignment_map:
                    _z_off = getattr(self, '_measurement_start_slice', 0)
                    global_z = int(stat.get('centroid_z_global', stat['centroid_z'])) + _z_off
                    layer_name = self.layer_assignment_map.get(global_z, "N/A")
                if layer_name not in layer_groups:
                    layer_groups[layer_name] = []
                layer_groups[layer_name].append(stat)
            
            final_list = []
            
            # Process each layer independently
            for layer_name in sorted(layer_groups.keys(), key=lambda k: (k == "N/A", k)):
                layer_stats = layer_groups[layer_name]
                
                if len(layer_stats) == 0:
                    continue
                
                # Sort by centroid_y then centroid_x
                layer_stats.sort(key=lambda s: (s['centroid_y'], s['centroid_x']))
                
                # Group into rows using adaptive tolerance
                rows = []
                current_row = [layer_stats[0]]
                y_prev = layer_stats[0]['centroid_y']
                
                # Use mean height for tolerance
                h_mean = np.mean([s['y_max'] - s['y_min'] for s in layer_stats]) if len(layer_stats) > 1 else 10
                tolerance = h_mean * 0.7
                
                for stat in layer_stats[1:]:
                    if abs(stat['centroid_y'] - y_prev) < tolerance:
                        current_row.append(stat)
                    else:
                        rows.append(current_row)
                        current_row = [stat]
                        y_prev = stat['centroid_y']
                rows.append(current_row)
                
                # Assign (R, C) index within this layer
                for r_idx, row_list in enumerate(rows):
                    row_list.sort(key=lambda s: s['centroid_x'])
                    for c_idx, stat in enumerate(row_list):
                        r_label = r_idx + 1  # 1-based
                        c_label = c_idx + 1  # 1-based
                        # row_id is assigned globally later
                        stat['grid_row'] = r_label
                        stat['grid_col'] = c_label
                        
                        # --- Calculate Pitch ---
                        stat['pitch_x'] = 0.0
                        stat['pitch_y'] = 0.0
                        if c_idx < len(row_list) - 1:
                            next_stat = row_list[c_idx + 1]
                            stat['pitch_x'] = max(0, (next_stat['x_min'] - stat['x_max'])) * vx
                            
                        if r_idx < len(rows) - 1:
                            next_row = rows[r_idx + 1]
                            w_mean = (stat['x_max'] - stat['x_min'])
                            closest_stat = min(next_row, key=lambda s: abs(s['centroid_x'] - stat['centroid_x']))
                            if abs(closest_stat['centroid_x'] - stat['centroid_x']) < w_mean * 1.5:
                                stat['pitch_y'] = max(0, (closest_stat['y_min'] - stat['y_max'])) * vy

                        final_list.append(stat)
            
            # Assign sequential row_id globally across all layers
            for idx, stat in enumerate(final_list):
                stat['row_id'] = idx + 1
            
            self.object_stats = final_list
            
        self.object_stats_table.setRowCount(len(self.object_stats))
        
        # Get threshold value for OK/NG classification
        ng_threshold = self.ng_threshold_spin.value() / 100.0 if hasattr(self, 'ng_threshold_spin') else 0.05
        
        ng_count = 0
        ok_count = 0
        
        for row, stat in enumerate(self.object_stats):
            ratio_str = f"{stat['ratio']*100:.3f}%" if stat['ratio'] != float('inf') else "N/A"
            # Use stored layer_name if available, otherwise lookup
            if 'layer_name' in stat:
                layer_name = stat['layer_name']
            elif hasattr(self, 'layer_assignment_map'):
                _z_off = getattr(self, '_measurement_start_slice', 0)
                global_z = int(stat.get('centroid_z_global', stat['centroid_z'])) + _z_off
                layer_name = self.layer_assignment_map.get(global_z, "N/A")
            else:
                layer_name = "N/A"
            
            # Determine OK/NG judgment
            is_ng = stat['ratio'] >= ng_threshold if stat['ratio'] != float('inf') else True
            judgment_str = "NG" if is_ng else "OK"
            if is_ng:
                ng_count += 1
            else:
                ok_count += 1
            
            soh_str = f"{stat.get('soh', 0):.3f}"
            items = [
                str(stat['row_id']),
                layer_name,
                f"({stat.get('grid_row', '?')},{stat.get('grid_col', '?')})",
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
            
            # Define row background color based on judgment
            if is_ng:
                row_bg = QColor(180, 40, 40, 60)      # Red tint for NG
            else:
                row_bg = QColor(40, 180, 60, 40)       # Green tint for OK
            
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setBackground(row_bg)
                # Store original stats index in first column's UserRole
                if col == 0:
                    item.setData(Qt.UserRole, row)
                
                if col == 6:  # Ratio column
                    item.setFont(QFont("Arial", 9, QFont.Bold))
                    if is_ng:
                        item.setForeground(QColor(255, 100, 100))  # Red for NG
                    else:
                        item.setForeground(QColor(100, 255, 120))  # Green for OK
                elif col == 7:  # Judgment column
                    item.setFont(QFont("Arial", 9, QFont.Bold))
                    item.setTextAlignment(Qt.AlignCenter)
                    if is_ng:
                        item.setForeground(QColor(255, 80, 80))
                    else:
                        item.setForeground(QColor(80, 255, 100))
                
                self.object_stats_table.setItem(row, col, item)
        
        self.object_stats_table.blockSignals(False)
        self.object_stats_table.setSortingEnabled(True)
        
        # Update stats info label with OK/NG counts
        if self.object_stats:
            total_c1 = sum(s['c1_volume'] for s in self.object_stats)
            total_c2 = sum(s['c2_volume'] for s in self.object_stats)
            self.stats_info_label.setText(
                f"{len(self.object_stats)} objects | "
                f"OK: {ok_count} | NG: {ng_count} | "
                f"Total Bump: {total_c1:,.0f} \u03bcm\u00b3 | Total Void: {total_c2:,.0f} \u03bcm\u00b3"
            )
            
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
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Config File", "", "Config Files (*.txt);;All Files (*.*)"
        )
        
        if file_path:
            try:
                self.update_config_from_parameters()
                
                # Write config file
                with open(file_path, 'w') as f:
                    f.write("# Bump + VOID DETECTION CONFIG FILE\n\n")
                    f.write("# === FILE PATHS ===\n")
                    f.write(f"INPUT_PATH = {self.config.inputPath.decode()}\n")
                    f.write(f"INPUT_MODE = {self.input_mode}\n")
                    f.write(f"OUTPUT_DIR = {self.config.outputDir.decode()}\n")
                    f.write(f"TEST_NAME = {self.config.testName.decode()}\n\n")
                    
                    f.write("# === BUMP DETECTION ===\n")
                    f.write(f"BUMP_THRESHOLD_WEIGHT = {self.config.bumpThresholdWeight}\n")
                    f.write(f"BUMP_CLEAN_OPEN_RADIUS_X = {self.config.bumpCleanOpenX}\n")
                    f.write(f"BUMP_CLEAN_OPEN_RADIUS_Y = {self.config.bumpCleanOpenY}\n")
                    f.write(f"BUMP_CLEAN_OPEN_RADIUS_Z = {self.config.bumpCleanOpenZ}\n")
                    f.write(f"BUMP_BG_OPEN_RADIUS_X = {self.config.bumpBgOpenX}\n")
                    f.write(f"BUMP_BG_OPEN_RADIUS_Y = {self.config.bumpBgOpenY}\n")
                    f.write(f"BUMP_BG_OPEN_RADIUS_Z = {self.config.bumpBgOpenZ}\n\n")
                    
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
                    f.write(f"SHOW_FLATTENED_TGV_VOID = {'true' if self.config.showResult else 'false'}\n\n")
                    
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
        
    
    def update_plane_view(self, orientation, preserve_camera=False):
        """Update a single plane view"""
        if self.volume_data is None:
            return
            
        renderer = getattr(self, f'{orientation}_renderer')
        slice_idx = self.current_slices[orientation]
        
        # Save camera state
        if preserve_camera and hasattr(self, 'camera_states') and isinstance(self.camera_states, dict):
            camera = renderer.GetActiveCamera()
            self.camera_states[orientation] = {
                'position': camera.GetPosition(),
                'focal_point': camera.GetFocalPoint(),
                'view_up': camera.GetViewUp(),
                'parallel_scale': camera.GetParallelScale()
            }
        
        if orientation == 'axial':
            actual_z = slice_idx
            if getattr(self, 'reverse_z', False):
                actual_z = self.volume_data.shape[0] - 1 - slice_idx
            
            slice_data = self.volume_data[actual_z, :, :]
            bump_slice = np.copy(self.bump_segmentation[actual_z, :, :]) if self.bump_segmentation is not None else None
            void_slice = np.copy(self.void_segmentation[actual_z, :, :]) if self.void_segmentation is not None else None
                
        elif orientation == 'coronal':
            slice_data = np.flipud(self.volume_data[:, slice_idx, :])
            bump_slice = np.flipud(self.bump_segmentation[:, slice_idx, :]).copy() if self.bump_segmentation is not None else None
            void_slice = np.flipud(self.void_segmentation[:, slice_idx, :]).copy() if self.void_segmentation is not None else None
                
        else:
            slice_data = np.transpose(self.volume_data[:, :, slice_idx])
            bump_slice = np.transpose(self.bump_segmentation[:, :, slice_idx]).copy() if self.bump_segmentation is not None else None
            void_slice = np.transpose(self.void_segmentation[:, :, slice_idx]).copy() if self.void_segmentation is not None else None
            
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
        window = v_max - v_min if v_max > v_min else 1.0
        level = v_min + window / 2.0
        
        prop.SetColorWindow(window)
        prop.SetColorLevel(level)
        prop.SetInterpolationTypeToNearest()
        renderer.AddActor(image_actor)
        
        # Camera
        if preserve_camera and hasattr(self, 'camera_states') and isinstance(self.camera_states, dict) and self.camera_states.get(orientation) is not None:
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
        bump_check = getattr(self, f'{orientation}_overlay_bump', None)
        void_check = getattr(self, f'{orientation}_overlay_void', None)
        opacity_slider = getattr(self, f'{orientation}_opacity_slider', None)
        
        bump_enabled = bump_check.isChecked() if bump_check else False
        void_enabled = void_check.isChecked() if void_check else False
        opacity = opacity_slider.value() / 100.0 if opacity_slider else 0.7
        
        if (bump_enabled or void_enabled) and (bump_slice is not None or void_slice is not None):
            overlay_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            has_overlay = False
            
            if bump_enabled and bump_slice is not None:
                mask_bump = (bump_slice == 128) | (bump_slice == 255)
                overlay_rgb[mask_bump, 0] = int(self.class1_color[0] * 255)
                overlay_rgb[mask_bump, 1] = int(self.class1_color[1] * 255)
                overlay_rgb[mask_bump, 2] = int(self.class1_color[2] * 255)
                has_overlay = True
                
            if void_enabled and void_slice is not None:
                mask_void = (void_slice == 128) | (void_slice == 255)
                overlay_rgb[mask_void, 0] = int(self.class2_color[0] * 255)
                overlay_rgb[mask_void, 1] = int(self.class2_color[1] * 255)
                overlay_rgb[mask_void, 2] = int(self.class2_color[2] * 255)
                has_overlay = True
                
            # Spotlight effect: dim non-selected bumps, keep selected bright
            if getattr(self, 'selected_highlight_objects', []) and getattr(self, 'labeled_class1_data', None) is not None:
                try:
                    # Get the labeled slice for this view orientation
                    if orientation == 'axial':
                        actual_z_hl = (self.labeled_class1_data.shape[0] - 1 - slice_idx) if getattr(self, 'reverse_z', False) else slice_idx
                        if actual_z_hl < self.labeled_class1_data.shape[0]:
                            labeled_slice = self.labeled_class1_data[actual_z_hl, :, :]
                        else:
                            labeled_slice = None
                    elif orientation == 'coronal':
                        if slice_idx < self.labeled_class1_data.shape[1]:
                            labeled_slice = np.flipud(self.labeled_class1_data[:, slice_idx, :])
                        else:
                            labeled_slice = None
                    else:
                        if slice_idx < self.labeled_class1_data.shape[2]:
                            labeled_slice = np.transpose(self.labeled_class1_data[:, :, slice_idx])
                        else:
                            labeled_slice = None
                    
                    if labeled_slice is not None and labeled_slice.shape == overlay_rgb.shape[:2]:
                        # Build combined mask of all selected object labels
                        selected_ids = [obj_id for cls_num, obj_id in self.selected_highlight_objects if cls_num == 1]
                        if selected_ids:
                            # Create mask of selected bump pixels
                            selected_mask = np.isin(labeled_slice, selected_ids)
                            # Create mask of all overlay-colored pixels (any bump or void)
                            any_overlay = np.any(overlay_rgb > 0, axis=2)
                            # Dim non-selected overlay pixels: reduce to 25% brightness
                            dim_mask = any_overlay & ~selected_mask
                            overlay_rgb[dim_mask] = (overlay_rgb[dim_mask] * 0.25).astype(np.uint8)
                            has_overlay = True
                except (IndexError, ValueError):
                    pass  # Shape mismatch between labeled data and current view
                        
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
        
        # Draw Crosshair (persistent actors — Dragonfly hover/grab like 3D Viewer)
        if getattr(self, 'crosshair_enabled', False):
            # RemoveAllViewProps above dropped actors; rebuild/re-add via lightweight path
            if hasattr(self, '_persistent_crosshair'):
                self._persistent_crosshair[orientation] = {}
            self.update_2d_crosshair(orientation)

        # Draw ROI overlay on axial (XY) view
        if orientation == 'axial':
            self.draw_roi_overlay(renderer, h, w)

        # Draw Object Indices if enabled (on all three views)
        if getattr(self, 'show_indices_check', None) and self.show_indices_check.isChecked() and self.object_stats:
            if orientation == 'axial':
                self.draw_object_labels(renderer, actual_z, orientation='axial', slice_idx=actual_z)
            elif orientation == 'coronal':
                self.draw_object_labels(renderer, 0, orientation='coronal', slice_idx=slice_idx)
            elif orientation == 'sagittal':
                self.draw_object_labels(renderer, 0, orientation='sagittal', slice_idx=slice_idx)

        # Scale bar (same style as 3D Viewer) — always on, updates with zoom
        self.add_ruler_overlay(renderer, orientation, (h, w))

        widget = getattr(self, f'{orientation}_widget', None)
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

    def load_dll_from_path(self, folder):
        """Load DLL folder programmatically (no file dialog)  used by Online mode"""
        try:
            bumpvoid.load_dll(folder)
            self.dll_path = folder
            self.dll_path_input.setText(folder)
            self.info_label.setText(f"DLL loaded: {bumpvoid.get_version()}")
            self.check_ready_state()
            return True
        except Exception as e:
            print(f"[SegmentationTab] Failed to load DLL from {folder}: {e}")
            return False

    def load_config_from_path(self, config_path):
        """Load config file programmatically (no file dialog)  used by Online mode"""
        try:
            self.config = bumpvoid.load_config(config_path)
            self.config_path = config_path
            self.config_path_input.setText(config_path)
            self.populate_parameters_from_config()
            # Apply pending input path if volume was loaded before config
            if self._pending_input_path:
                self.config.inputPath = self._pending_input_path.encode('utf-8')
                self._pending_input_path = None

            # Manually parse Voxel Sizes and Measurement Parameters from config file
            try:
                with open(config_path, 'r') as f:
                    content = f.read()
                    vx_match = re.search(r'VOXEL_SIZE_X\s*=\s*([\d.]+)', content)
                    vy_match = re.search(r'VOXEL_SIZE_Y\s*=\s*([\d.]+)', content)
                    vz_match = re.search(r'VOXEL_SIZE_Z\s*=\s*([\d.]+)', content)
                    
                    bumpmin_match = re.search(r'BUMP_MINIMUM_SIZE\s*=\s*([0-9.xX]+)', content)
                    bumpmax_match = re.search(r'BUMP_MAXIMUM_SIZE\s*=\s*([0-9.xX]+)', content)
                    voidmin_match = re.search(r'VOID_MINIMUM_SIZE\s*=\s*([0-9.xX]+)', content)
                    voidmax_match = re.search(r'VOID_MAXIMUM_SIZE\s*=\s*([0-9.xX]+)', content)
                    
                    if vx_match: self.param_widgets['voxel_size_x'].setValue(float(vx_match.group(1)))
                    if vy_match: self.param_widgets['voxel_size_y'].setValue(float(vy_match.group(1)))
                    if vz_match: self.param_widgets['voxel_size_z'].setValue(float(vz_match.group(1)))
                    
                    if bumpmin_match: self.param_widgets['bump_minimum_size'].setText(bumpmin_match.group(1))
                    if bumpmax_match: self.param_widgets['bump_maximum_size'].setText(bumpmax_match.group(1))
                    if voidmin_match: self.param_widgets['void_minimum_size'].setText(voidmin_match.group(1))
                    if voidmax_match: self.param_widgets['void_maximum_size'].setText(voidmax_match.group(1))
                    
                    # Parse Z-Stretched 4x toggle
                    z4x_match = re.search(r'Z_STRETCHED_4X\s*=\s*(true|false)', content, re.IGNORECASE)
                    if z4x_match and 'z_stretched_4x' in self.param_widgets:
                        self.param_widgets['z_stretched_4x'].setChecked(z4x_match.group(1).lower() == 'true')
                    
                    # Parse NG threshold
                    ng_match = re.search(r'NG_THRESHOLD\s*=\s*([\d.]+)', content)
                    if ng_match and hasattr(self, 'ng_threshold_spin'):
                        self.ng_threshold_spin.setValue(float(ng_match.group(1)))
                    
                    # Parse CC3D parameters
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

                    # Parse Mode
                    mode_match = re.search(r'INPUT_MODE\s*=\s*([a-zA-Z0-9_]+)', content)
                    if mode_match:
                        mode_val = mode_match.group(1).strip()
                        if mode_val == 'test_1layer':
                            self.mode_test1_radio.setChecked(True)
                        elif mode_val == 'single':
                            self.mode_single_radio.setChecked(True)
                        elif mode_val == 'multi_layer':
                            self.mode_multi_radio.setChecked(True)

                    # Parse Enhancement parameters from config
                    self._apply_enhancement_from_config_text(content)
            except Exception as e:
                print(f"Note: Could not parse Measurement Parameters from config: {e}")

            # Parse Layer Definitions from config if present
            try:
                with open(config_path, 'r') as f:
                    config_content = f.read()
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

    def set_segmentation_results(self, bump_data, void_data):
        """Set segmentation results programmatically  used by Online mode"""
        self.bump_segmentation = bump_data
        self.void_segmentation = void_data
        
        # Refresh views to show overlays
        for o in ['axial', 'coronal', 'sagittal']:
            self.update_plane_view(o)
            
        if hasattr(self, 'measure_btn'):
            self.measure_btn.setEnabled(bump_data is not None and void_data is not None)
        if hasattr(self, 'dt_btn'):
            self.dt_btn.setEnabled(bump_data is not None)
        if hasattr(self, 'bnd_btn'):
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
        self.config_path_input.setText("")
        self.input_path_input.setText("")
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
        # --- 3D Volume pane (same navigation as 3D Viewer) ---
        is_3d = (
            obj is getattr(self, '_3d_vtk_widget', None)
            or obj is getattr(self, '_3d_vtk_container', None)
        )
        if is_3d and getattr(self, '_3d_view_active', False):
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                if obj is getattr(self, '_3d_vtk_widget', None):
                    try:
                        self._3d_vtk_widget.setFocus(Qt.MouseFocusReason)
                    except Exception:
                        pass
            elif event.type() == QEvent.Wheel:
                if self.volume_data is not None:
                    pixel = event.pixelDelta().y()
                    angle = event.angleDelta().y()
                    if angle == 0:
                        angle = event.angleDelta().x()
                    if pixel == 0 and angle == 0:
                        pixel = event.pixelDelta().x()
                    if pixel != 0 or angle != 0:
                        if pixel != 0:
                            strength = abs(pixel) / 40.0
                            zoom_in = pixel > 0
                        else:
                            strength = abs(angle) / 120.0
                            zoom_in = angle > 0
                        strength = max(0.15, min(strength, 3.0))
                        self._dragonfly_3d_zoom(zoom_in=zoom_in, strength=strength)
                        return True
            # Let other events reach VTK (rotate / pan / right-drag zoom)
            return super().eventFilter(obj, event)

        # --- ROI Selection Mode (priority over crosshair on axial widget) ---
        if self.roi_selection_active and hasattr(self, 'axial_widget') and obj == self.axial_widget and self.volume_data is not None:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.handle_roi_mouse_press(event.pos())
                return True
            elif event.type() == QEvent.MouseMove:
                self.handle_roi_mouse_move(event.pos(), is_drag=(event.buttons() & Qt.LeftButton))
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.handle_roi_mouse_release(event.pos())
                return True

        # Resolve which 2D orientation this widget belongs to
        orientation = None
        for ori in ['axial', 'coronal', 'sagittal']:
            if obj == getattr(self, f'{ori}_widget', None):
                orientation = ori
                break

        if orientation is not None and self.volume_data is not None:
            widget = getattr(self, f'{orientation}_widget')

            if event.type() == QEvent.MouseMove:
                self.update_pixel_value(orientation, event.pos())
                if self.crosshair_enabled and getattr(self, '_is_dragging_crosshair', False) and (event.buttons() & Qt.LeftButton):
                    mode = getattr(self, '_crosshair_drag_mode', 'center') or 'center'
                    self.handle_crosshair_click(orientation, event.pos(), mode=mode)
                    return True
                if self.crosshair_enabled and not (event.buttons() & Qt.LeftButton):
                    ch_hit = self._get_crosshair_hit(event.pos(), orientation)
                    if ch_hit != getattr(self, '_crosshair_hover', None):
                        self._crosshair_hover = ch_hit
                        self.update_2d_crosshair(orientation)
                    if ch_hit:
                        if ch_hit[1] == 'center':
                            widget.setCursor(Qt.SizeAllCursor)
                        elif ch_hit[1] == 'h':
                            widget.setCursor(Qt.SizeVerCursor)
                        else:
                            widget.setCursor(Qt.SizeHorCursor)
                    else:
                        widget.unsetCursor()

            elif event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                if self.crosshair_enabled:
                    ch_hit = self._get_crosshair_hit(event.pos(), orientation)
                    if ch_hit is None:
                        return False  # miss → VTK pan/window-level free
                    self._crosshair_drag_mode = ch_hit[1]
                    self._crosshair_hover = ch_hit
                    self._is_dragging_crosshair = True
                    self.handle_crosshair_click(orientation, event.pos(), mode=ch_hit[1])
                    self.update_2d_crosshair(orientation)
                    return True

            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._restore_interactor_style(orientation)
                if getattr(self, '_is_dragging_crosshair', False):
                    self._is_dragging_crosshair = False
                    self._crosshair_drag_mode = None
                    return True

            elif event.type() == QEvent.Wheel:
                # Fiji-style zoom / Ctrl+scroll slice (works even under dummy style)
                modifiers = event.modifiers()
                angle_delta = event.angleDelta().y()
                if angle_delta == 0:
                    return super().eventFilter(obj, event)
                size = widget.GetRenderWindow().GetSize()
                mx = event.pos().x()
                my = size[1] - event.pos().y()
                if modifiers & Qt.ControlModifier:
                    delta = 1 if angle_delta > 0 else -1
                    self._scroll_change_slice(orientation, delta)
                else:
                    self._fiji_zoom_at_cursor(
                        widget, orientation,
                        zoom_in=(angle_delta > 0),
                        display_pos=(mx, my),
                    )
                return True

            elif event.type() == QEvent.Leave:
                if getattr(self, '_crosshair_hover', None) and getattr(self, '_crosshair_hover')[0] == orientation:
                    self._crosshair_hover = None
                    if self.crosshair_enabled:
                        self.update_2d_crosshair(orientation)
                widget.unsetCursor()

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
            
            coord_str = ""
            if orientation == 'axial':
                px, py = int(world_pos[0]), int(world_pos[1])
                if 0 <= px < vol_x and 0 <= py < vol_y:
                    actual_z = (vol_z - 1 - slice_idx) if getattr(self, 'reverse_z', False) else slice_idx
                    value = self.volume_data[actual_z, py, px]
                    coord_str = f"X:{px} Y:{py} Z:{actual_z}"
            elif orientation == 'coronal':
                px, pz = int(world_pos[0]), int(world_pos[1])
                if 0 <= px < vol_x and 0 <= pz < vol_z:
                     value = self.volume_data[pz, slice_idx, px]
                     coord_str = f"X:{px} Y:{slice_idx} Z:{pz}"
            else:
                 pz, py = int(world_pos[0]), int(world_pos[1])
                 if 0 <= pz < vol_z and 0 <= py < vol_y:
                     value = self.volume_data[pz, py, slice_idx]
                     coord_str = f"X:{slice_idx} Y:{py} Z:{pz}"

            label = getattr(self, f'{orientation}_pixel_label')
            if value is not None:
                label.setText(f"{coord_str} | Pixel: {value}")
            else:
                label.setText("Pixel: --")
        except:
            pass

    def updatePoint(self, newX, newY, newZ):
        """Update crosshair + slices; only full re-render views whose slice changed."""
        if self.volume_data is None:
            return
        vol_z, vol_y, vol_x = self.volume_data.shape
        newX = max(0, min(int(newX), vol_x - 1))
        newY = max(0, min(int(newY), vol_y - 1))
        newZ = max(0, min(int(newZ), vol_z - 1))

        oldX, oldY, oldZ = self.crosshair_position
        self.crosshair_position = [newX, newY, newZ]

        changed = {
            'sagittal': newX != oldX or self.current_slices.get('sagittal') != newX,
            'coronal': newY != oldY or self.current_slices.get('coronal') != newY,
            'axial': newZ != oldZ or self.current_slices.get('axial') != newZ,
        }

        self.current_slices['sagittal'] = newX
        self.current_slices['coronal'] = newY
        self.current_slices['axial'] = newZ

        if hasattr(self, 'sagittal_slice_slider'):
            self.sagittal_slice_slider.blockSignals(True)
            self.sagittal_slice_slider.setValue(newX)
            self.sagittal_slice_slider.blockSignals(False)
        if hasattr(self, 'sagittal_slice_label'):
            self.sagittal_slice_label.setText(f"{newX} / {vol_x - 1}")

        if hasattr(self, 'coronal_slice_slider'):
            self.coronal_slice_slider.blockSignals(True)
            self.coronal_slice_slider.setValue(newY)
            self.coronal_slice_slider.blockSignals(False)
        if hasattr(self, 'coronal_slice_label'):
            self.coronal_slice_label.setText(f"{newY} / {vol_y - 1}")

        if hasattr(self, 'axial_slice_slider'):
            self.axial_slice_slider.blockSignals(True)
            self.axial_slice_slider.setValue(newZ)
            self.axial_slice_slider.blockSignals(False)
        if hasattr(self, 'axial_slice_label'):
            self.axial_slice_label.setText(f"{newZ} / {vol_z - 1}")

        for ori in ['axial', 'coronal', 'sagittal']:
            if changed[ori]:
                self.update_plane_view(ori, preserve_camera=True)
            elif self.crosshair_enabled:
                self.update_2d_crosshair(ori)

    def _crosshair_display_geometry(self, orientation):
        """Project crosshair center into VTK display pixels (axis-aligned)."""
        if self.volume_data is None:
            return None
        renderer = getattr(self, f'{orientation}_renderer', None)
        if renderer is None:
            return None
        vol_z, vol_y, vol_x = self.volume_data.shape
        if orientation == 'axial':
            cx_w = float(self.crosshair_position[0])
            cy_w = float(self.crosshair_position[1])
        elif orientation == 'coronal':
            cx_w = float(self.crosshair_position[0])
            cy_w = float((vol_z - 1) - self.crosshair_position[2])
        else:
            cx_w = float(self.crosshair_position[2])
            cy_w = float(self.crosshair_position[1])
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        coord.SetValue(cx_w, cy_w, 0.5)
        dp = coord.GetComputedDisplayValue(renderer)
        # Axis-aligned: H = horizontal (+X display), V = vertical (+Y display)
        return {
            'dx': float(dp[0]), 'dy': float(dp[1]),
            'cos_h': 1.0, 'sin_h': 0.0,
            'cos_v': 0.0, 'sin_v': 1.0,
            'gap': 12,
        }

    def _get_crosshair_hit(self, pos, orientation):
        """Dragonfly hit-test: center, then H/V axes. Returns (ori, mode) or None."""
        import math
        if not self.crosshair_enabled or self.volume_data is None:
            return None
        geom = self._crosshair_display_geometry(orientation)
        if geom is None:
            return None
        widget = getattr(self, f'{orientation}_widget')
        size = widget.GetRenderWindow().GetSize()
        x = float(pos.x())
        vtk_y = float(size[1] - pos.y())
        dx, dy = geom['dx'], geom['dy']
        gap = geom['gap']
        dist_center = math.hypot(x - dx, vtk_y - dy)
        if dist_center <= 16.0:
            return (orientation, 'center')

        def dist_to_axis(px, py, cos_t, sin_t):
            vx, vy = px - dx, py - dy
            return abs(vx * sin_t - vy * cos_t)

        dh = dist_to_axis(x, vtk_y, geom['cos_h'], geom['sin_h'])
        dv = dist_to_axis(x, vtk_y, geom['cos_v'], geom['sin_v'])
        line_tol = 8.0
        if dist_center > gap:
            if dh <= line_tol and dv <= line_tol:
                return (orientation, 'h' if dh <= dv else 'v')
            if dh <= line_tol:
                return (orientation, 'h')
            if dv <= line_tol:
                return (orientation, 'v')
        return None

    @staticmethod
    def _boost_color(color, factor=1.35):
        return tuple(min(1.0, float(c) * factor) for c in color)

    def update_2d_crosshair(self, orientation):
        """Lightweight crosshair redraw with hover highlight (3D Viewer parity)."""
        if not self.crosshair_enabled or self.volume_data is None:
            return
        if not hasattr(self, '_persistent_crosshair'):
            self._persistent_crosshair = {'axial': {}, 'coronal': {}, 'sagittal': {}}

        renderer = getattr(self, f'{orientation}_renderer')
        widget = getattr(self, f'{orientation}_widget')
        actors_dict = self._persistent_crosshair[orientation]

        if not actors_dict:
            for name in ['h_line1', 'h_line2', 'v_line1', 'v_line2']:
                ls = vtk.vtkLineSource()
                m = vtk.vtkPolyDataMapper2D()
                m.SetInputConnection(ls.GetOutputPort())
                a = vtk.vtkActor2D()
                a.SetMapper(m)
                a.GetProperty().SetLineWidth(1)
                actors_dict[name] = {'source': ls, 'actor': a}
            for name in ['h_pos', 'h_neg', 'v_pos', 'v_neg']:
                t = vtk.vtkTextActor()
                t.GetTextProperty().SetFontSize(11)
                t.GetTextProperty().BoldOn()
                actors_dict[name] = t

        vol_z, vol_y, vol_x = self.volume_data.shape
        if orientation == 'axial':
            cx_w = float(self.crosshair_position[0])
            cy_w = float(self.crosshair_position[1])
            h_pos, h_neg = '+X', '-X'
            v_pos, v_neg = '+Y', '-Y'
            h_color = (1.0, 0.192, 0.192)
            v_color = (0.223, 1.0, 0.078)
        elif orientation == 'coronal':
            cx_w = float(self.crosshair_position[0])
            cy_w = float((vol_z - 1) - self.crosshair_position[2])
            h_pos, h_neg = '+X', '-X'
            v_pos, v_neg = ('+Z', '-Z') if self.reverse_z else ('-Z', '+Z')
            h_color = (1.0, 0.192, 0.192)
            v_color = (0.121, 0.317, 1.0)
        else:
            cx_w = float(self.crosshair_position[2])
            cy_w = float(self.crosshair_position[1])
            h_pos, h_neg = ('-Z', '+Z') if self.reverse_z else ('+Z', '-Z')
            v_pos, v_neg = '+Y', '-Y'
            h_color = (0.121, 0.317, 1.0)
            v_color = (0.223, 1.0, 0.078)

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        coord.SetValue(cx_w, cy_w, 0.5)
        dp = coord.GetComputedDisplayValue(renderer)
        dx, dy = float(dp[0]), float(dp[1])

        win = renderer.GetRenderWindow().GetSize()
        vp = renderer.GetViewport()
        vp_w = (vp[2] - vp[0]) * win[0]
        vp_h = (vp[3] - vp[1]) * win[1]
        gap = 12

        hover = getattr(self, '_crosshair_hover', None)
        drag_mode = getattr(self, '_crosshair_drag_mode', None) if getattr(self, '_is_dragging_crosshair', False) else None
        active_part = None
        if hover and hover[0] == orientation:
            active_part = drag_mode if drag_mode else hover[1]
        h_hot = active_part in ('center', 'h')
        v_hot = active_part in ('center', 'v')
        h_draw = self._boost_color(h_color, 1.45 if h_hot else 1.0)
        v_draw = self._boost_color(v_color, 1.45 if v_hot else 1.0)
        h_lw = 2.5 if h_hot else 1.0
        v_lw = 2.5 if v_hot else 1.0

        for n in ['h_line1', 'h_line2']:
            prop = actors_dict[n]['actor'].GetProperty()
            prop.SetColor(*h_draw)
            prop.SetLineWidth(h_lw)
            prop.SetOpacity(1.0 if h_hot else 0.92)
        for n in ['v_line1', 'v_line2']:
            prop = actors_dict[n]['actor'].GetProperty()
            prop.SetColor(*v_draw)
            prop.SetLineWidth(v_lw)
            prop.SetOpacity(1.0 if v_hot else 0.92)

        # Axis-aligned segments with center gap
        actors_dict['h_line1']['source'].SetPoint1(0, dy, 0)
        actors_dict['h_line1']['source'].SetPoint2(max(0.0, dx - gap), dy, 0)
        actors_dict['h_line2']['source'].SetPoint1(dx + gap, dy, 0)
        actors_dict['h_line2']['source'].SetPoint2(vp_w, dy, 0)
        actors_dict['v_line1']['source'].SetPoint1(dx, 0, 0)
        actors_dict['v_line1']['source'].SetPoint2(dx, max(0.0, dy - gap), 0)
        actors_dict['v_line2']['source'].SetPoint1(dx, dy + gap, 0)
        actors_dict['v_line2']['source'].SetPoint2(dx, vp_h, 0)

        # Edge labels (Teaching / Viewer style)
        for key, text, pos, color in [
            ('h_pos', h_pos, (max(5.0, min(vp_w - 40.0, vp_w - 30.0)), max(5.0, min(vp_h - 18.0, dy + 4))), h_draw),
            ('h_neg', h_neg, (5.0, max(5.0, min(vp_h - 18.0, dy + 4))), h_draw),
            ('v_pos', v_pos, (max(5.0, min(vp_w - 40.0, dx + 5)), max(5.0, min(vp_h - 18.0, vp_h - 20.0))), v_draw),
            ('v_neg', v_neg, (max(5.0, min(vp_w - 40.0, dx + 5)), 5.0), v_draw),
        ]:
            actors_dict[key].SetInput(text)
            actors_dict[key].SetPosition(pos[0], pos[1])
            actors_dict[key].GetTextProperty().SetColor(*color)

        for k, v in actors_dict.items():
            act = v['actor'] if isinstance(v, dict) and 'actor' in v else v
            if not renderer.HasViewProp(act):
                renderer.AddActor(act)

        widget.GetRenderWindow().Render()

    def handle_crosshair_click(self, orientation, pos, mode='center'):
        """Dragonfly grab modes: center / h / v (same as 3D Viewer)."""
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
            newX, newY, newZ = self.crosshair_position

            if orientation == 'axial':
                pickX = int(round(world_pos[0]))
                pickY = int(round(world_pos[1]))
                if mode in ('center', 'v'):
                    newX = pickX
                if mode in ('center', 'h'):
                    newY = pickY
            elif orientation == 'coronal':
                pickX = int(round(world_pos[0]))
                pickZ = (vol_z - 1) - int(round(world_pos[1]))
                if mode in ('center', 'v'):
                    newX = pickX
                if mode in ('center', 'h'):
                    newZ = pickZ
            elif orientation == 'sagittal':
                pickZ = int(round(world_pos[0]))
                pickY = int(round(world_pos[1]))
                if mode in ('center', 'v'):
                    newZ = pickZ
                if mode in ('center', 'h'):
                    newY = pickY

            self.updatePoint(newX, newY, newZ)
        except Exception:
            pass
            
    def toggle_crosshair(self, state):
        self.crosshair_enabled = (state == Qt.Checked)
        self._crosshair_hover = None
        self._crosshair_drag_mode = None
        self._is_dragging_crosshair = False
        if hasattr(self, '_persistent_crosshair'):
            self._persistent_crosshair = {'axial': {}, 'coronal': {}, 'sagittal': {}}
        if self.volume_data is not None:
            for orientation in ['axial', 'coronal', 'sagittal']:
                self.update_plane_view(orientation, preserve_camera=True)

    def on_crosshair_slider_changed(self, axis_idx, value):
        if self.volume_data is None:
            return
            
        newX, newY, newZ = self.crosshair_position
        if axis_idx == 0:
            newX = value
        elif axis_idx == 1:
            newY = value
        elif axis_idx == 2:
            newZ = value
            
        self.updatePoint(newX, newY, newZ)

    def choose_overlay_color(self, class_num):
        if class_num == 1:
            init_c = QColor(int(self.class1_color[0]*255), int(self.class1_color[1]*255), int(self.class1_color[2]*255))
            color = QColorDialog.getColor(init_c, self, "Class 1 Color")
            if color.isValid():
                self.class1_color = [color.redF(), color.greenF(), color.blueF()]
                self.color_c1_btn.setStyleSheet(f"color: {color.name()}; font-weight: bold; padding: 2px;")
                for ori in ['axial', 'coronal', 'sagittal']:
                    self.update_plane_view(ori, preserve_camera=True)
        else:
            init_c = QColor(int(self.class2_color[0]*255), int(self.class2_color[1]*255), int(self.class2_color[2]*255))
            color = QColorDialog.getColor(init_c, self, "Class 2 Color")
            if color.isValid():
                self.class2_color = [color.redF(), color.greenF(), color.blueF()]
                self.color_c2_btn.setStyleSheet(f"color: {color.name()}; font-weight: bold; padding: 2px;")
                for ori in ['axial', 'coronal', 'sagittal']:
                    self.update_plane_view(ori, preserve_camera=True)

    def create_crosshair_actors(self, orientation, renderer, slice_shape):
        actors = []
        try:
            h, w = slice_shape
            vol_z, vol_y, vol_x = self.volume_data.shape
            z_sign = '-Z' if getattr(self, 'reverse_z', False) else '+Z'
            
            # Define labels for 4 ends: [Right(+H), Left(-H), Top(+V), Bottom(-V)]
            if orientation == 'axial':
                labels = ['+X', '-X', '+Y', '-Y']
                h_color = (1.0, 0.192, 0.192)  # X: Red
                v_color = (0.223, 1.0, 0.078)  # Y: Green
                cx_w = float(self.crosshair_position[0])
                cy_w = float(self.crosshair_position[1])
            elif orientation == 'coronal':
                labels = ['+X', '-X', '-Z', '+Z'] # Top 0 is -Z, Bottom Max is +Z
                h_color = (1.0, 0.192, 0.192)  # X: Red
                v_color = (0.121, 0.317, 1.0)  # Z: Blue
                cx_w = float(self.crosshair_position[0])
                cy_w = float((vol_z - 1) - self.crosshair_position[2])
            else: # sagittal
                labels = ['+Z', '-Z', '+Y', '-Y']
                h_color = (0.121, 0.317, 1.0)  # Z: Blue
                v_color = (0.223, 1.0, 0.078)  # Y: Green
                cx_w = float(self.crosshair_position[2])
                cy_w = float(self.crosshair_position[1])
                
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
            gap = 12
            
            def make_line2d(x1, y1, x2, y2, color, lw=1):
                ls = vtk.vtkLineSource()
                ls.SetPoint1(x1, y1, 0)
                ls.SetPoint2(x2, y2, 0)
                m = vtk.vtkPolyDataMapper2D()
                m.SetInputConnection(ls.GetOutputPort())
                a = vtk.vtkActor2D()
                a.SetMapper(m)
                a.GetProperty().SetColor(*color)
                a.GetProperty().SetLineWidth(lw)
                return a
            
            def make_text2d(text, px, py, color, size=11):
                t = vtk.vtkTextActor()
                t.SetInput(text)
                t.SetPosition(px, py)
                t.GetTextProperty().SetFontSize(size)
                t.GetTextProperty().SetColor(*color)
                t.GetTextProperty().BoldOn()
                return t
                
            if dx > gap:
                actors.append(make_line2d(0, dy, dx - gap, dy, h_color))
            actors.append(make_line2d(dx + gap, dy, vp_w, dy, h_color))
            if dy > gap:
                actors.append(make_line2d(dx, 0, dx, dy - gap, v_color))
            if dy < vp_h - gap:
                actors.append(make_line2d(dx, dy + gap, dx, vp_h, v_color))
                
            # Add 4 labels: Right, Left, Top, Bottom
            # Right (+H)
            actors.append(make_text2d(labels[0], vp_w - 30, dy + 4, h_color))
            # Left (-H)
            actors.append(make_text2d(labels[1], 5, dy + 4, h_color))
            # Top (+V)
            actors.append(make_text2d(labels[2], dx + 5, vp_h - 20, v_color))
            # Bottom (-V)
            actors.append(make_text2d(labels[3], dx + 5, 5, v_color))
                
        except Exception:
            pass
        return actors

    def _get_mpr_spacing_xyz(self):
        """Voxel spacing (sx, sy, sz) in µm for MPR scale bar — prefer Params UI."""
        try:
            pw = getattr(self, "param_widgets", None) or {}
            if "voxel_size_x" in pw and "voxel_size_y" in pw and "voxel_size_z" in pw:
                return [
                    float(pw["voxel_size_x"].value()),
                    float(pw["voxel_size_y"].value()),
                    float(pw["voxel_size_z"].value()),
                ]
        except Exception:
            pass
        sp = getattr(self, "spacing", [1.0, 1.0, 1.0])
        if sp is not None and len(sp) >= 3:
            return [float(sp[0]), float(sp[1]), float(sp[2])]
        return [1.0, 1.0, 1.0]

    def add_ruler_overlay(self, renderer, orientation, slice_shape):
        """
        Horizontal scale bar at bottom-left — same style/behavior as 3D Viewer.
        Auto-adjusts length from zoom; uses display coordinates (fixed under pan).
        """
        try:
            if not hasattr(self, "ruler_actors") or self.ruler_actors is None:
                self.ruler_actors = {"axial": [], "coronal": [], "sagittal": []}
            for act in self.ruler_actors.get(orientation, []):
                try:
                    renderer.RemoveActor(act)
                except Exception:
                    pass
            self.ruler_actors[orientation] = []

            h, w = slice_shape
            rc = getattr(self, "ruler_color", [0.2, 0.9, 0.2])

            camera = renderer.GetActiveCamera()
            parallel_scale = camera.GetParallelScale()
            win_size = renderer.GetRenderWindow().GetSize()
            viewport = renderer.GetViewport()
            vp_w_px = max((viewport[2] - viewport[0]) * win_size[0], 1)
            vp_h_px = max((viewport[3] - viewport[1]) * win_size[1], 1)
            world_per_px = (2.0 * parallel_scale) / vp_h_px if vp_h_px > 0 else 1.0

            spacing = self._get_mpr_spacing_xyz()
            # Horizontal axis spacing per orientation (matches viewer.py)
            if orientation == "axial":       # XY → horizontal = X
                um_per_world = spacing[0]
            elif orientation == "coronal":   # XZ → horizontal = X
                um_per_world = spacing[0]
            else:                            # YZ (sagittal) → horizontal = Z
                um_per_world = spacing[2]

            um_per_px = world_per_px * um_per_world
            if um_per_px <= 0:
                um_per_px = 1.0

            # Nice bar length (~18% of viewport width)
            target_um = vp_w_px * 0.18 * um_per_px
            nice_vals = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
            bar_um = nice_vals[-1]
            for nv in nice_vals:
                if nv >= target_um * 0.6:
                    bar_um = nv
                    break
            bar_px = max(bar_um / um_per_px, 20)

            mx, my = 18, 16
            cap_h = 8

            def _line(x1, y1, x2, y2, color, width):
                src = vtk.vtkLineSource()
                src.SetPoint1(x1, y1, 0)
                src.SetPoint2(x2, y2, 0)
                m = vtk.vtkPolyDataMapper2D()
                m.SetInputConnection(src.GetOutputPort())
                a = vtk.vtkActor2D()
                a.SetMapper(m)
                a.GetProperty().SetColor(*color)
                a.GetProperty().SetLineWidth(width)
                renderer.AddActor(a)
                self.ruler_actors[orientation].append(a)

            # Shadow for contrast
            sc = (0.0, 0.0, 0.0)
            _line(mx, my, mx + bar_px, my, sc, 6)
            _line(mx, my - cap_h // 2, mx, my + cap_h // 2, sc, 4)
            _line(mx + bar_px, my - cap_h // 2, mx + bar_px, my + cap_h // 2, sc, 4)

            # Main bar + caps
            _line(mx, my, mx + bar_px, my, rc, 3)
            _line(mx, my - cap_h // 2, mx, my + cap_h // 2, rc, 2)
            _line(mx + bar_px, my - cap_h // 2, mx + bar_px, my + cap_h // 2, rc, 2)

            if bar_um >= 1000:
                txt = f"{bar_um / 1000:.0f} mm"
            elif bar_um >= 1:
                txt = f"{int(bar_um)} µm"
            else:
                txt = f"{bar_um:.1f} µm"

            lbl = vtk.vtkTextActor()
            lbl.SetInput(txt)
            tp = lbl.GetTextProperty()
            tp.SetFontSize(10)
            tp.SetColor(*rc)
            tp.BoldOn()
            tp.SetFontFamilyToArial()
            tp.SetShadow(True)
            tp.SetShadowOffset(1, 1)
            lbl.GetPositionCoordinate().SetCoordinateSystemToDisplay()
            lbl.SetPosition(mx + bar_px / 2 - 12, my + cap_h // 2 + 2)
            renderer.AddActor(lbl)
            self.ruler_actors[orientation].append(lbl)

        except Exception as e:
            print("Error in add_ruler_overlay:", e)

    def update_slice(self, orientation, value):
        if self.volume_data is None:
            return
        newX, newY, newZ = self.crosshair_position
        if orientation == 'axial':
            newZ = value
        elif orientation == 'coronal':
            newY = value
        elif orientation == 'sagittal':
            newX = value
        
        self.updatePoint(newX, newY, newZ)

