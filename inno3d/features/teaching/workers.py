# inno3d/features/teaching/workers.py
# -----------------------------------------------------------------------
# Teaching workers — extracted from inno3d/tabs/teaching.py (Phase 4.1)
#
# QThread subclasses for background computation:
#   NoScrollDoubleSpinBox, NoScrollSpinBox (UI helpers)
#   _ExportLayerVolumesThread
#   DistanceTransformThread
#   BoundaryAnalysisThread
#   SegmentationPreviewThread
#   SegmentationInspectionThread
#   EnhancementThread
# -----------------------------------------------------------------------

import csv
import ctypes
import glob
import os
import re
import traceback
from pathlib import Path as _Path
from typing import Optional

import numpy as np

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox, QSpinBox,
)

from inno3d.core import bumpvoid
from inno3d.core import enhanced_volume
from inno3d.core.styles import apply_theme, SemiconductorTheme


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()



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
                # workers.py → teaching → features → inno3d → project root
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                
                # Prefer explicit DLL folder (V2), then project V2/, then legacy paths
                gpu_dll_path = os.path.join(self.dll_folder, "BoundaryGPU.dll") if self.dll_folder else ""
                if not gpu_dll_path or not os.path.exists(gpu_dll_path):
                    gpu_dll_path = os.path.join(project_root, "V2", "BoundaryGPU.dll")
                if not os.path.exists(gpu_dll_path):
                    gpu_dll_path = os.path.join(project_root, "BoundaryGPU", "BoundaryGPU.dll")
                    
                csharp_dll_path = os.path.join(self.dll_folder, "BoundaryAnalysis.dll") if self.dll_folder else ""
                if not csharp_dll_path or not os.path.exists(csharp_dll_path):
                    csharp_dll_path = os.path.join(project_root, "V2", "BoundaryAnalysis.dll")
                if not os.path.exists(csharp_dll_path):
                    csharp_dll_path = os.path.join(project_root, "BoundaryDll", "BoundaryAnalysis", "bin", "Release", "net8.0", "win-x64", "publish", "BoundaryAnalysis.dll")
                
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
                        try:
                            if hasattr(boundary_dll, "B2B_SetProfiling"):
                                from inno3d.core import bumpvoid_b2b as _b2b
                                # Same HMODULE as bumpvoid_b2b when paths match — sync flag
                                if hasattr(_b2b, "_dll") and _b2b._dll is not None and hasattr(_b2b._dll, "B2B_GetProfiling"):
                                    boundary_dll.B2B_SetProfiling(int(_b2b._dll.B2B_GetProfiling()))
                        except Exception:
                            pass
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

                    try:
                        info = bumpvoid.get_last_timing()
                        if info and info.get("enabled"):
                            from inno3d.core.dll_profiling import format_timing_banner
                            print(f"--- {layer_name} ---\n{format_timing_banner('SEG', info)}")
                    except Exception:
                        pass

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
                 input_dir=None, output_dir=None, start_slice=-1, end_slice=-1,
                 volume_u16=None):
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
        # Optional in-memory volume (ZYX uint16) — preferred when DLL >= 0.0.1
        self.volume_u16 = volume_u16
        self.enhanced_volume = None  # filled on buffer-path success
        self.used_buffer_api = False

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

            # Prefer in-memory path (BumpVoid_ISP_ENH >= 0.0.1) — no temp TIFF export
            use_buffer = (
                self.volume_u16 is not None
                and hasattr(enhanced_volume, "has_process_volume_buffer")
                and enhanced_volume.has_process_volume_buffer()
            )
            if use_buffer:
                import numpy as np
                import tifffile

                self.progress.emit(25, "Enhancing volume (in-memory buffer)...")
                src = np.ascontiguousarray(self.volume_u16, dtype=np.uint16)
                dst = np.empty_like(src)
                result = enhanced_volume.process_volume_buffer(
                    src,
                    dst,
                    start_slice=self.start_slice,
                    end_slice=self.end_slice,
                    callback=cb,
                )
                if result < 0:
                    err = enhanced_volume.get_last_error()
                    self.finished.emit(False, f"Enhancement failed: {err}")
                    return
                if result == 0:
                    self.finished.emit(False, "Enhancement: No slices processed")
                    return

                self.enhanced_volume = dst
                self.used_buffer_api = True
                # Optional archive under Results (single multipage, not 2500 temp files)
                if self.output_dir:
                    try:
                        os.makedirs(self.output_dir, exist_ok=True)
                        out_path = os.path.join(self.output_dir, "Enhanced_Volume.tif")
                        tifffile.imwrite(out_path, dst, imagej=True)
                    except Exception as e:
                        self.progress.emit(95, f"Archive write skipped: {e}")
                self.progress.emit(100, f"Enhanced {result} slices (buffer)")
                self.finished.emit(True, "")
                return

            if not self.input_dir or not self.output_dir:
                self.finished.emit(
                    False,
                    "Enhancement: folder path required (DLL has no ProcessVolumeBuffer). "
                    "Deploy BumpVoid_ISP_ENH >= 0.0.1 or provide input_dir.",
                )
                return

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


