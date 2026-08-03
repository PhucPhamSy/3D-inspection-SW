"""MPR render pipeline mixin — Phase 3.3b extract from tabs/viewer.py.

Contains:
  - ``render_slice`` — reslice + VTK actor update for one MPR pane
  - ``add_ruler_overlay`` — physical-unit ruler lines
  - ``reset_view`` — fit camera to slice
  - Progressive MPR helpers (generation ID, preview mip, idle refine)

Layer: features/viewer (imports Qt, VTK — not for domain/infra use).
"""
from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support
from PyQt5.QtCore import Qt, QTimer

from inno3d.core.perf_timing import perf_phase
from inno3d.features.shared.mes_mpr_highlight import (
    build_mes_selection_highlight_rgb,
    labeled_slice_for_orientation,
)


class MprRenderMixin:
    """Mixin providing MPR slice rendering for MultiPlanarView.

    All methods use ``self.*`` attributes owned by MultiPlanarView.
    Do not instantiate directly.
    """

    # ── Progressive MPR (large_volume_engine) ─────────────────────────────

    def _mpr_lve_active(self) -> bool:
        from inno3d.core.volume_store import large_volume_engine_enabled

        return (
            large_volume_engine_enabled(self)
            and getattr(self, "volume_store", None) is not None
        )

    def _mpr_is_interacting(self) -> bool:
        return bool(
            getattr(self, "_mpr_interacting", False)
            or getattr(self, "_is_dragging_crosshair", False)
        )

    def _mpr_preview_level(self) -> int:
        """Coarse mip while dragging; level-0 when idle / refine."""
        if not self._mpr_is_interacting():
            return 0
        # Prefer level 2 if available, else 1, else 0
        store = getattr(self, "volume_store", None)
        if store is None:
            return 0
        for cand in (2, 1):
            if store._level_available(cand) or cand == 1:
                # level 1 always available via stride fallback on memmap/dense
                return cand
        return 0

    def _bump_mpr_generation(self, orientation: str) -> int:
        gens = getattr(self, "_mpr_gen", None)
        if gens is None:
            self._mpr_gen = {"axial": 0, "coronal": 0, "sagittal": 0}
            gens = self._mpr_gen
        gens[orientation] = int(gens.get(orientation, 0)) + 1
        return gens[orientation]

    def _schedule_mpr_idle_refine(self, orientation: str, gen: int) -> None:
        """After idle, re-render level-0 if generation is still current."""
        if not self._mpr_lve_active():
            return
        timers = getattr(self, "_mpr_refine_timers", None)
        if timers is None:
            self._mpr_refine_timers = {}
            timers = self._mpr_refine_timers
        old = timers.get(orientation)
        if old is not None:
            try:
                old.stop()
            except Exception:
                pass

        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(100)  # 80–120 ms idle window

        def _refine():
            if not self._mpr_lve_active():
                return
            if self._mpr_is_interacting():
                # Still interacting — reschedule
                self._schedule_mpr_idle_refine(orientation, gen)
                return
            current = getattr(self, "_mpr_gen", {}).get(orientation, -1)
            if current != gen:
                return  # stale
            # Force level-0 refine
            self._mpr_refine_level = getattr(self, "_mpr_refine_level", {})
            self._mpr_refine_level[orientation] = 0
            self.render_slice(orientation, preserve_camera=True, _mpr_force_level=0)

        timer.timeout.connect(_refine)
        timers[orientation] = timer
        timer.start()

    def mark_mpr_interaction_start(self) -> None:
        self._mpr_interacting = True

    def mark_mpr_interaction_end(self) -> None:
        self._mpr_interacting = False
        if not self._mpr_lve_active():
            return
        for ori in ("axial", "coronal", "sagittal"):
            gen = getattr(self, "_mpr_gen", {}).get(ori, 0)
            self._schedule_mpr_idle_refine(ori, gen)

    def _extract_mpr_slice_from_store(self, orientation, slice_idx, level: int):
        """Level-aware slice matching current render_slice orientation semantics."""
        from inno3d.core.volume_store import upsample_slice_nearest
        from inno3d.services.volume_pyramid import level_downsample_factor

        store = self.volume_store
        z, y, x = store.shape
        factor = level_downsample_factor(level)

        if orientation == "axial":
            actual_z = slice_idx
            if self.reverse_z:
                actual_z = z - 1 - slice_idx
            lvl_idx = actual_z // factor
            target_hw = (y, x)
            # After reverse_z, level-0 index maps into downsampled Z
            slice_data = store.get_slice("axial", lvl_idx, level=level)
            if level > 0:
                slice_data = upsample_slice_nearest(slice_data, target_hw)
            # Masks at level-0 indices; flipud to match store axial / non-LVE path
            seg_slice = (
                np.flipud(self.segmentation_data[actual_z, :, :])
                if self.segmentation_data is not None
                else None
            )
            c1_slice = (
                np.flipud(self.class1_data[actual_z, :, :])
                if self.class1_data is not None
                else None
            )
            c2_slice = (
                np.flipud(self.class2_data[actual_z, :, :])
                if self.class2_data is not None
                else None
            )
            return slice_data, seg_slice, c1_slice, c2_slice

        if orientation == "coronal":
            lvl_idx = slice_idx // factor
            slice_data = store.get_slice("coronal", lvl_idx, level=level)
            if level > 0:
                slice_data = upsample_slice_nearest(slice_data, (z, x))
            seg_slice = np.flipud(self.segmentation_data[:, slice_idx, :]) if self.segmentation_data is not None else None
            c1_slice = np.flipud(self.class1_data[:, slice_idx, :]) if self.class1_data is not None else None
            c2_slice = np.flipud(self.class2_data[:, slice_idx, :]) if self.class2_data is not None else None
            return slice_data, seg_slice, c1_slice, c2_slice

        # sagittal
        lvl_idx = slice_idx // factor
        slice_data = store.get_slice("sagittal", lvl_idx, level=level)
        if level > 0:
            # get_slice already transposed → (Y, Z) wait — sagittal returns transpose of (Z,Y) = (Y,Z)
            # level-0 target is (y, z) after transpose of volume[:,:,x] which is (Z,Y).T => (Y,Z)
            slice_data = upsample_slice_nearest(slice_data, (y, z))
        seg_slice = np.transpose(self.segmentation_data[:, :, slice_idx]) if self.segmentation_data is not None else None
        c1_slice = np.transpose(self.class1_data[:, :, slice_idx]) if self.class1_data is not None else None
        c2_slice = np.transpose(self.class2_data[:, :, slice_idx]) if self.class2_data is not None else None
        return slice_data, seg_slice, c1_slice, c2_slice

    def reset_view(self, orientation):
        self.camera_state[orientation] = None
        self.render_slice(orientation, preserve_camera=False)
    

    def render_slice(self, orientation, preserve_camera=False, _mpr_force_level=None):
        if self.volume_data is None:
            return

        with perf_phase("mpr.render_slice", detail=orientation):
            self._render_slice_impl(orientation, preserve_camera, _mpr_force_level)

    def _render_slice_impl(self, orientation, preserve_camera=False, _mpr_force_level=None):
        if preserve_camera:
            self.save_camera_state(orientation)
        
        renderer = getattr(self, f'{orientation}_renderer')
                
        slice_idx = self.current_slices[orientation]
        gen = self._bump_mpr_generation(orientation) if self._mpr_lve_active() else 0
        
        is_oblique = (abs(self.oblique_angles.get('axial', 0.0)) > 0.5 or 
                      abs(self.oblique_angles.get('coronal', 0.0)) > 0.5 or 
                      abs(self.oblique_angles.get('sagittal', 0.0)) > 0.5)
        
        # In manual align mode, oblique_angles are only for crosshair preview
        # — don't reslice the volume until the user commits
        if getattr(self, '_manual_align_active', False):
            is_oblique = False
        
        if is_oblique:
            slice_data, seg_slice, c1_slice, c2_slice = self._oblique_reslice_3d(orientation, slice_idx)
        elif self._mpr_lve_active():
            if _mpr_force_level is not None:
                level = int(_mpr_force_level)
            elif self._mpr_is_interacting():
                level = self._mpr_preview_level()
            else:
                level = 0
            # Use best available if preferred pyramid level not ready yet
            store = self.volume_store
            level = store.best_available_level(level) if level > 0 else 0
            slice_data, seg_slice, c1_slice, c2_slice = self._extract_mpr_slice_from_store(
                orientation, slice_idx, level
            )
            if level > 0:
                self._schedule_mpr_idle_refine(orientation, gen)
                # One-line breadcrumb so preview/refine is observable in logs
                if getattr(self, "_mpr_last_logged_level", None) != (orientation, level):
                    self._mpr_last_logged_level = (orientation, level)
                    print(f"[MPR] {orientation} preview level={level} (idle → refine L0)")
        else:
            if orientation == 'axial':
                # Axial (XY): X = horizontal, Y = vertical (np.flipud matches Dragonfly/Fiji orientation)
                actual_z = slice_idx
                if self.reverse_z:
                    actual_z = self.volume_data.shape[0] - 1 - slice_idx
                slice_data = np.flipud(self.volume_data[actual_z, :, :])
                seg_slice = np.flipud(self.segmentation_data[actual_z, :, :]) if self.segmentation_data is not None else None
                c1_slice = np.flipud(self.class1_data[actual_z, :, :]) if self.class1_data is not None else None
                c2_slice = np.flipud(self.class2_data[actual_z, :, :]) if self.class2_data is not None else None

            elif orientation == 'coronal':
                # Coronal (XZ): X = horizontal, Z = vertical (Z increases downwards -> Z=0 at top)
                slice_data = np.flipud(self.volume_data[:, slice_idx, :])
                seg_slice = np.flipud(self.segmentation_data[:, slice_idx, :]) if self.segmentation_data is not None else None
                c1_slice = np.flipud(self.class1_data[:, slice_idx, :]) if self.class1_data is not None else None
                c2_slice = np.flipud(self.class2_data[:, slice_idx, :]) if self.class2_data is not None else None

            else:
                # Sagittal (YZ): Z = horizontal, Y = vertical (90 deg CCW of previous)
                slice_data = np.transpose(self.volume_data[:, :, slice_idx])
                seg_slice = np.transpose(self.segmentation_data[:, :, slice_idx]) if self.segmentation_data is not None else None
                c1_slice = np.transpose(self.class1_data[:, :, slice_idx]) if self.class1_data is not None else None
                c2_slice = np.transpose(self.class2_data[:, :, slice_idx]) if self.class2_data is not None else None

        # Stale cancel: if a newer generation started, skip VTK upload
        if self._mpr_lve_active():
            current = getattr(self, "_mpr_gen", {}).get(orientation, gen)
            if current != gen:
                return
            applied = getattr(self, "_mpr_applied_gen", None)
            if applied is None:
                self._mpr_applied_gen = {"axial": 0, "coronal": 0, "sagittal": 0}
                applied = self._mpr_applied_gen
            applied[orientation] = gen
        
        h, w = slice_data.shape
        
        # ── Performance: Reuse VTK image actors instead of recreating ──
        # Cache the base image actor, vtkImageData, and overlay state per orientation.
        # On subsequent calls with same dimensions, we only update the pixel buffer
        # (zero-copy where possible), avoiding expensive VTK object creation.
        if not hasattr(self, '_cached_slice_actors'):
            self._cached_slice_actors = {}
        
        cache = self._cached_slice_actors.get(orientation)
        try:
            _c1_on = bool(getattr(self, f'{orientation}_overlay_c1').isChecked())
            _c2_on = bool(getattr(self, f'{orientation}_overlay_c2').isChecked())
            _op_val = int(getattr(self, f'{orientation}_opacity_slider').value())
        except Exception:
            _c1_on, _c2_on, _op_val = True, True, 70
        has_overlays = (seg_slice is not None or c1_slice is not None or c2_slice is not None)
        _want_rgb = (
            (_c1_on or _c2_on)
            and _op_val > 0
            and has_overlays
        )
        _display_mode = 'rgb_overlay' if _want_rgb else 'grayscale'

        need_full_rebuild = (
            cache is None or 
            cache.get('h') != h or cache.get('w') != w or
            cache.get('dtype') != slice_data.dtype or
            cache.get('display_mode') != _display_mode
        )
        
        # Overlay visibility/opacity/colour must be part of the cache key so
        # C1/C2/opacity/colour toggles always rebuild (not a stale RGB film).
        _seg = getattr(self, "seg_colors", {}) or {}
        _c1_col = tuple(_seg.get(128, [1.0, 1.0, 0.0]))
        _c2_col = tuple(_seg.get(255, [1.0, 0.0, 0.0]))
        overlay_state = (
            has_overlays,
            _c1_on,
            _c2_on,
            _op_val,
            bool(self.selected_highlight_objects) if self.volume_data is not None else False,
            _display_mode,
            _c1_col,
            _c2_col,
        )
        if cache and cache.get('overlay_state') != overlay_state:
            need_full_rebuild = True
        
        if need_full_rebuild:
            # Full rebuild: remove everything and recreate
            renderer.RemoveAllViewProps()
            
            # Invalidate persistent crosshair actors — they were removed from this renderer
            if hasattr(self, '_persistent_crosshair') and orientation in self._persistent_crosshair:
                self._persistent_crosshair[orientation] = {}
            
            vtk_image = vtk.vtkImageData()
            vtk_image.SetDimensions(w, h, 1)
            vtk_image.SetSpacing(1.0, 1.0, 1.0)
            vtk_image.SetOrigin(0.0, 0.0, 0.0)
            
            image_actor = vtk.vtkImageActor()
            image_actor.GetMapper().SetInputData(vtk_image)
            image_actor.GetProperty().SetInterpolationTypeToNearest()
            renderer.AddActor(image_actor)
            
            # Store in cache
            self._cached_slice_actors[orientation] = {
                'h': h, 'w': w, 'dtype': slice_data.dtype,
                'vtk_image': vtk_image,
                'image_actor': image_actor,
                'overlay_state': overlay_state,
                'display_mode': _display_mode,
            }
            cache = self._cached_slice_actors[orientation]
        else:
            # Fast path: reuse existing actors — just need to remove overlays/crosshair/ruler
            # and re-add them after data update
            vtk_image = cache['vtk_image']
            image_actor = cache['image_actor']
            cache['display_mode'] = _display_mode
            cache['overlay_state'] = overlay_state
            
            # Remove only non-base actors (crosshair, ruler, overlay, text labels) — keep image_actor
            actors_to_keep = {image_actor}
            props = renderer.GetViewProps()
            props.InitTraversal()
            props_to_remove = []
            for _ in range(props.GetNumberOfItems()):
                prop = props.GetNextProp()
                if prop and prop not in actors_to_keep:
                    props_to_remove.append(prop)
            for prop in props_to_remove:
                renderer.RemoveViewProp(prop)
            
            # Invalidate persistent crosshair so they get re-added
            if hasattr(self, '_persistent_crosshair') and orientation in self._persistent_crosshair:
                self._persistent_crosshair[orientation] = {}
        
        prop = image_actor.GetProperty()
        
        # Sliders removed, defaulting to 0
        brightness_val = 0
        contrast_val = 0
        
        if self.window_level[orientation] is not None:
            base_window, base_level = self.window_level[orientation]
        else:
            v_min, v_max = float(slice_data.min()), float(slice_data.max())
            base_window = v_max - v_min if v_max > v_min else 1.0
            base_level = v_min + base_window / 2.0
        
        # Brightness shifts the level, contrast scales the window
        level = base_level + brightness_val * (base_window / 200.0)
        # contrast_val: -100 → window*2 (flat), +100 → window*0.2 (sharp)
        contrast_factor = max(0.2, 1.0 - contrast_val / 100.0)
        window = base_window * contrast_factor

        # Resolve overlay toggles early — when any class is on we bake a display RGB
        # image (CT + mask alpha-blend). This avoids VTK ImageActor RGB black-film
        # and broken per-pixel alpha on some OpenGL backends.
        try:
            overlay_c1_enabled = bool(getattr(self, f'{orientation}_overlay_c1').isChecked())
            overlay_c2_enabled = bool(getattr(self, f'{orientation}_overlay_c2').isChecked())
            opacity = float(getattr(self, f'{orientation}_opacity_slider').value()) / 100.0
        except Exception:
            overlay_c1_enabled, overlay_c2_enabled, opacity = False, False, 0.7
        opacity = float(np.clip(opacity, 0.0, 1.0))
        want_overlay = (
            (overlay_c1_enabled or overlay_c2_enabled)
            and opacity > 0.0
            and (c1_slice is not None or c2_slice is not None or seg_slice is not None)
        )

        if want_overlay:
            display_rgb = self._compose_mpr_overlay_rgb(
                slice_data, window, level,
                c1_slice, c2_slice, seg_slice,
                overlay_c1_enabled, overlay_c2_enabled, opacity,
                orientation,
            )
            # display_rgb: (H,W,3) uint8 in same image axes as slice_data
            overlay_transposed = np.transpose(display_rgb, (1, 0, 2))
            flat_rgb = np.ascontiguousarray(overlay_transposed.reshape(-1, 3, order='F'))
            vtk_array = numpy_support.numpy_to_vtk(
                flat_rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
            )
            vtk_array.SetNumberOfComponents(3)
            # Identity W/L — pixels already display-mapped (MUST stay 255/127.5).
            # Data-range W/L (e.g. 15k–51k) on 0–255 RGB blacks out the whole MPR.
            prop.SetColorWindow(255.0)
            prop.SetColorLevel(127.5)
            if cache is not None:
                cache['display_mode'] = 'rgb_overlay'
        else:
            # Grayscale path — keep native dtype + VTK W/L (fast, full dynamic range)
            slice_transposed = np.transpose(slice_data, (1, 0))
            flat_data = np.ascontiguousarray(slice_transposed.flatten('F'))
            if slice_data.dtype == np.uint16:
                vtk_array = numpy_support.numpy_to_vtk(
                    flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_SHORT
                )
            elif slice_data.dtype == np.uint8:
                vtk_array = numpy_support.numpy_to_vtk(
                    flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
                )
            else:
                vtk_array = numpy_support.numpy_to_vtk(
                    flat_data.astype(np.float32), deep=True, array_type=vtk.VTK_FLOAT
                )
            prop.SetColorWindow(window)
            prop.SetColorLevel(level)
            if cache is not None:
                cache['display_mode'] = 'grayscale'

        vtk_image.GetPointData().SetScalars(vtk_array)
        vtk_image.Modified()
        
        # Camera
        if preserve_camera and self.camera_state[orientation] is not None:
            self.restore_camera_state(orientation)
        else:
            camera = renderer.GetActiveCamera()
            camera.ParallelProjectionOn()
            
            center_x = w / 2.0
            center_y = h / 2.0
            
            camera.SetPosition(center_x, center_y, 1000)
            camera.SetFocalPoint(center_x, center_y, 0)
            camera.SetViewUp(0, 1, 0)
            
            # Fit this pane independently (fill viewport). Do not unify scale
            # across MPR planes — shared µm/px makes large XY look too small.
            widget = getattr(self, f'{orientation}_widget')
            vp_size = widget.GetRenderWindow().GetSize()
            vp_w_px = max(vp_size[0], 1)
            vp_h_px = max(vp_size[1], 1)
            vp_aspect = vp_w_px / vp_h_px
            img_aspect = w / max(h, 1)
            
            if img_aspect > vp_aspect:
                # Image is wider than viewport -> fit by width
                parallel_scale = (w / vp_aspect) * 0.5
            else:
                # Image is taller or same -> fit by height
                parallel_scale = h * 0.5
            
            camera.SetParallelScale(parallel_scale)
            
            renderer.ResetCameraClippingRange()
        
        # Segmentation mask is already baked into the base image when want_overlay
        # (see _compose_mpr_overlay_rgb). No second ImageActor — avoids black film.

        # --- Object Highlight Overlay (shared cyan fill + white outline) ---
        if self.selected_highlight_objects and self.volume_data is not None:
            labeled_slice = labeled_slice_for_orientation(
                getattr(self, "labeled_class1_data", None),
                orientation,
                self.current_slices[orientation],
                reverse_z=bool(getattr(self, "reverse_z", False)),
            )
            hl_rgb_only = build_mes_selection_highlight_rgb(
                labeled_slice,
                self.selected_highlight_objects,
                getattr(self, "highlight_color", None),
                outline_iterations=3,
            )
            if hl_rgb_only is not None and hl_rgb_only.shape[:2] == (h, w):
                hl_transposed = np.transpose(hl_rgb_only, (1, 0, 2))
                hl_flat = np.ascontiguousarray(hl_transposed.reshape(-1, 3, order="F"))

                vtk_hl = vtk.vtkImageData()
                vtk_hl.SetDimensions(w, h, 1)
                vtk_hl.SetSpacing(1.0, 1.0, 1.0)
                vtk_hl.SetOrigin(0.0, 0.0, 0.0)

                vtk_hl_arr = numpy_support.numpy_to_vtk(
                    hl_flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
                )
                vtk_hl_arr.SetNumberOfComponents(3)
                vtk_hl.GetPointData().SetScalars(vtk_hl_arr)

                hl_actor = vtk.vtkImageActor()
                hl_actor.GetMapper().SetInputData(vtk_hl)
                hl_actor.SetOpacity(0.7)
                renderer.AddActor(hl_actor)
        
        if self.crosshair_enabled:
            # Use the unified persistent crosshair system to avoid double crosshair
            self.update_2d_crosshair(orientation)
        
        # Always draw scale bar (updates with zoom)
        self.add_ruler_overlay(renderer, orientation, (h, w))

        # Dragonfly clip-box overlay (re-added after render_slice clears non-base actors)
        self._df_clip_update_mpr_overlay(orientation)
        
        # Draw Object Indices if enabled
        if getattr(self, 'show_indices_check', None) and self.show_indices_check.isChecked() and getattr(self, 'object_stats', None):
            slice_idx = self.current_slices[orientation]
            if orientation == 'axial':
                actual_z = (self.volume_data.shape[0] - 1 - slice_idx) if self.reverse_z else slice_idx
                self.draw_object_labels(renderer, actual_z, orientation='axial', slice_idx=actual_z)
            elif orientation == 'coronal':
                self.draw_object_labels(renderer, 0, orientation='coronal', slice_idx=slice_idx)
            elif orientation == 'sagittal':
                self.draw_object_labels(renderer, 0, orientation='sagittal', slice_idx=slice_idx)
        
        widget = getattr(self, f'{orientation}_widget')
        widget.GetRenderWindow().Render()
    

    def add_ruler_overlay(self, renderer, orientation, slice_shape):
        """
        Draw a horizontal scale bar at the bottom-left of the viewport,
        matching the style and behavior of the DEMO app.
        Auto-adjusts length based on current zoom level. Uses display (screen-pixel)
        coordinates so it stays fixed regardless of pan/zoom position.
        """
        try:
            if not hasattr(self, 'ruler_actors'):
                self.ruler_actors = {'axial': [], 'coronal': [], 'sagittal': []}
            for act in self.ruler_actors.get(orientation, []):
                renderer.RemoveActor(act)
            self.ruler_actors[orientation] = []

            h, w = slice_shape
            rc = self.ruler_color  # configurable color
            
            # 1. Compute µm/screen-pixel from camera
            camera = renderer.GetActiveCamera()
            parallel_scale = camera.GetParallelScale()
            win_size = renderer.GetRenderWindow().GetSize()
            viewport = renderer.GetViewport()
            vp_w_px = max((viewport[2] - viewport[0]) * win_size[0], 1)
            vp_h_px = max((viewport[3] - viewport[1]) * win_size[1], 1)
            world_per_px = (2.0 * parallel_scale) / vp_h_px if vp_h_px > 0 else 1.0

            spacing = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing

            # Horizontal axis spacing — correct per orientation
            if orientation == 'axial':       # XY view → horizontal = X
                um_per_world = spacing[0]
            elif orientation == 'coronal':   # XZ view → horizontal = X
                um_per_world = spacing[0]
            else:                            # YZ (sagittal) → horizontal = Z
                um_per_world = spacing[2]

            um_per_px = world_per_px * um_per_world
            if um_per_px <= 0:
                um_per_px = 1.0

            # 2. Choose a "nice" bar length (~18% of viewport width)
            target_um = vp_w_px * 0.18 * um_per_px
            nice_vals = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
            bar_um = nice_vals[-1]
            for nv in nice_vals:
                if nv >= target_um * 0.6:
                    bar_um = nv
                    break
            bar_px = max(bar_um / um_per_px, 20)

            # 3. Layout constants
            mx, my = 18, 16       # margin from bottom-left
            cap_h = 8             # end-cap height

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

            # 4. Draw shadow outline (dark) for contrast on bright images
            sc = (0.0, 0.0, 0.0)
            _line(mx, my, mx + bar_px, my, sc, 6)                            # bar shadow
            _line(mx, my - cap_h // 2, mx, my + cap_h // 2, sc, 4)          # left cap shadow
            _line(mx + bar_px, my - cap_h // 2, mx + bar_px, my + cap_h // 2, sc, 4)  # right cap

            # 5. Draw main bar + end caps (colored)
            _line(mx, my, mx + bar_px, my, rc, 3)                            # main bar
            _line(mx, my - cap_h // 2, mx, my + cap_h // 2, rc, 2)          # left cap
            _line(mx + bar_px, my - cap_h // 2, mx + bar_px, my + cap_h // 2, rc, 2)  # right cap

            # 6. Label with unit
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
            # Center label above bar
            lbl.GetPositionCoordinate().SetCoordinateSystemToDisplay()
            lbl.SetPosition(mx + bar_px / 2 - 12, my + cap_h // 2 + 2)
            renderer.AddActor(lbl)
            self.ruler_actors[orientation].append(lbl)

        except Exception as e:
            print("Error in add_ruler_overlay:", e)
