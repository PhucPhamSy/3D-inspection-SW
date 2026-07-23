# inno3d/features/viewer/clip_box.py
# -----------------------------------------------------------------------
# ClipBoxMixin  --  extracted from inno3d/tabs/viewer.py (Phase 3.5)
#
# Dragonfly-style clip-box: Advanced MPR axis planes, 3D face drag,
# MPR overlay handles, GPU crop sync (_sync_all_clip_planes).
# -----------------------------------------------------------------------

import copy
import math

import vtk
from PyQt5.QtCore import Qt


class ClipBoxMixin:
    """Mixin: Advanced MPR clipping + Dragonfly clip-box.

    Extracted from inno3d/tabs/viewer.py Phase 3.5.
    State variables (_df_clip_bounds, _df_clip_enabled, etc.) are
    initialised in MultiPlanarView.__init__ and accessed via self.*.
    """

    def _build_clip_plane(self, axis, slider_val, flipped=False):
        """Build a vtkPlane for the given axis from a slider value (0..1000).
        
        Dragonfly-style clipping:
        - slider_val 0..1000 maps to 0..100% of the axis extent
        - VTK clips everything where dot(normal, point - origin) < 0
        - Default normal = +1 → keeps everything ABOVE the slider position
        - Flipped normal = -1 → keeps everything BELOW the slider position
        
        This is orientation-agnostic: X, Y, Z always map to VTK world axes
        regardless of volume shape or aspect ratio.
        """
        extent = self._get_vtk_world_extent()
        amin, amax = extent[axis]
        t = max(0.0, min(1.0, slider_val / 1000.0))
        pos = amin + t * (amax - amin)
        
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        normal = [0.0, 0.0, 0.0]
        # +1 = keep positive side, -1 = keep negative side
        normal[axis_idx] = -1.0 if flipped else 1.0
        
        origin = [0.0, 0.0, 0.0]
        origin[axis_idx] = pos
        
        plane = vtk.vtkPlane()
        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)
        return plane

    def _sync_all_clip_planes(self):
        """Apply clipping to all volume mappers using GPU-native cropping.
        
        Uses SetCropping/SetCroppingRegionPlanes for vtkGPUVolumeRayCastMapper
        instead of AddClippingPlane, because VTK's GPU mapper has known
        rendering artifacts with ClippingPlanes from certain viewing angles.
        
        Crop region is the intersection of:
          1) Fullscreen Advanced MPR axis clip planes (if any)
          2) Dragonfly clip-box bounds from the 2×2 right sidebar
        
        Dragonfly behavior: turning CLIP BOX tool OFF only hides the box UI;
        the current crop is KEPT. Reset restores full volume (bounds 0–100%).
        
        Segmentation volume overlays also use GPU cropping when available;
        polydata mappers still receive ClippingPlanes for axis planes.
        """
        extent = self._get_vtk_world_extent()
        
        # Determine crop region from active clip planes
        crop_min = [extent['x'][0], extent['y'][0], extent['z'][0]]
        crop_max = [extent['x'][1], extent['y'][1], extent['z'][1]]
        has_any_clip = False
        
        active_planes = []  # For polydata / legacy clip-plane consumers
        for axis in ['x', 'y', 'z']:
            plane = self.adv_clip_planes.get(axis)
            if plane is None:
                continue
            has_any_clip = True
            active_planes.append(plane)
            
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
            normal = plane.GetNormal()
            origin = plane.GetOrigin()
            pos = origin[axis_idx]
            
            if normal[axis_idx] > 0:
                # Keep positive side (above pos)
                crop_min[axis_idx] = max(crop_min[axis_idx], pos)
            else:
                # Keep negative side (below pos)
                crop_max[axis_idx] = min(crop_max[axis_idx], pos)

        # Dragonfly clip box: apply stored bounds even when the tool UI is off
        # (toggle only shows/hides the interactive box; Reset clears the crop).
        box = self._df_clip_world_bounds()
        if box is not None and self._df_clip_bounds_are_partial():
            has_any_clip = True
            for i, ax in enumerate(['x', 'y', 'z']):
                crop_min[i] = max(crop_min[i], box[ax][0])
                crop_max[i] = min(crop_max[i], box[ax][1])
            # Six half-space planes for polydata consumers
            for ax, idx in (('x', 0), ('y', 1), ('z', 2)):
                lo, hi = box[ax]
                p_lo = vtk.vtkPlane()
                o = [0.0, 0.0, 0.0]
                o[idx] = lo
                n = [0.0, 0.0, 0.0]
                n[idx] = 1.0
                p_lo.SetOrigin(*o)
                p_lo.SetNormal(*n)
                active_planes.append(p_lo)
                p_hi = vtk.vtkPlane()
                o2 = [0.0, 0.0, 0.0]
                o2[idx] = hi
                n2 = [0.0, 0.0, 0.0]
                n2[idx] = -1.0
                p_hi.SetOrigin(*o2)
                p_hi.SetNormal(*n2)
                active_planes.append(p_hi)

        # Ensure non-empty crop region
        for i in range(3):
            if crop_max[i] <= crop_min[i]:
                mid = 0.5 * (extent[['x', 'y', 'z'][i]][0] + extent[['x', 'y', 'z'][i]][1])
                crop_min[i] = mid - 1e-3
                crop_max[i] = mid + 1e-3
        
        # Apply GPU-native cropping to volume + mask volumes
        for actor_name in ['volume_actor', 'c1_actor_3d', 'c2_actor_3d']:
            actor = getattr(self, actor_name, None)
            if actor is None:
                continue
            mapper = actor.GetMapper()
            if mapper is None:
                continue

            if hasattr(mapper, 'SetCropping'):
                if hasattr(mapper, 'RemoveAllClippingPlanes'):
                    mapper.RemoveAllClippingPlanes()
                if has_any_clip:
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
            elif hasattr(mapper, 'RemoveAllClippingPlanes'):
                # Polydata / non-GPU path
                mapper.RemoveAllClippingPlanes()
                for p in active_planes:
                    mapper.AddClippingPlane(p)
                mapper.Modified()

    def on_mpr_show_hide_all(self, show):
        for axis, ctrl in self.adv_mpr_controls.items():
            ctrl['show'].blockSignals(True)
            ctrl['show'].setChecked(show)
            ctrl['show'].blockSignals(False)
            self.update_advanced_mpr_clip(axis)

    def on_mpr_reset_all(self):
        for axis, ctrl in self.adv_mpr_controls.items():
            ctrl['slider'].blockSignals(True)
            ctrl['slider'].setValue(500)
            ctrl['pct_label'].setText("50%")
            ctrl['slider'].blockSignals(False)
            
            ctrl['show'].blockSignals(True)
            ctrl['show'].setChecked(False)
            ctrl['show'].blockSignals(False)
            
            ctrl['clip'].blockSignals(True)
            ctrl['clip'].setChecked(False)
            ctrl['clip'].blockSignals(False)
            
            ctrl['flip'].blockSignals(True)
            ctrl['flip'].setChecked(False)
            ctrl['flip'].blockSignals(False)
            
            self.update_advanced_mpr_clip(axis)

    def update_mpr_thickness(self, val):
        self.update_advanced_mpr_clip('x')
        self.update_advanced_mpr_clip('y')
        self.update_advanced_mpr_clip('z')

    def update_advanced_mpr_clip(self, axis):
        """Dragonfly-style clipping: Show = always clip volume.
        
        - Show checked → clip plane applied, volume is cut at slider position
        - Clip checked → additionally show a colored indicator plane
        - Flip → reverse which half is kept
        
        Coordinate logic (orientation-agnostic):
          1. Read world extent from _get_vtk_world_extent() 
          2. Map slider % → position within [0, max_world]
          3. Build vtkPlane at that position with axis-aligned normal
          4. Apply to ALL mappers via _sync_all_clip_planes()
        
        This works correctly regardless of volume orientation (horizontal,
        vertical, tall, flat) because it operates purely in VTK world space.
        """
        if not hasattr(self, 'view_3d_renderer') or self.volume_data is None:
            return
            
        ctrl = self.adv_mpr_controls[axis]
        enabled = ctrl['show'].isChecked()
        show_plane = ctrl['clip'].isChecked()
        flipped = ctrl['flip'].isChecked()
        slider_val = ctrl['slider'].value()
        
        # Update percentage label
        ctrl['pct_label'].setText(f"{slider_val / 10.0:.0f}%")
        
        # 1. Build clip plane — always active when Show is checked
        if enabled:
            self.adv_clip_planes[axis] = self._build_clip_plane(axis, slider_val, flipped)
        else:
            self.adv_clip_planes[axis] = None
        
        # 2. Sync clip planes to ALL volume mappers
        self._sync_all_clip_planes()
            
        # 3. Optional colored indicator plane at the clip position
        if self.adv_mpr_actors[axis]:
            self.view_3d_renderer.RemoveActor(self.adv_mpr_actors[axis])
            self.adv_mpr_actors[axis] = None
            
        if enabled and show_plane:
            extent = self._get_vtk_world_extent()
            t = max(0.0, min(1.0, slider_val / 1000.0))
            pos = extent[axis][0] + t * (extent[axis][1] - extent[axis][0])
            
            # Build a visible rectangle spanning the other two axes
            other_axes = [a for a in ['x', 'y', 'z'] if a != axis]
            a1_min, a1_max = extent[other_axes[0]]
            a2_min, a2_max = extent[other_axes[1]]
            
            plane_src = vtk.vtkPlaneSource()
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
            o1_idx = {'x': 0, 'y': 1, 'z': 2}[other_axes[0]]
            o2_idx = {'x': 0, 'y': 1, 'z': 2}[other_axes[1]]
            
            # Origin point (base corner of the indicator plane)
            origin_pt = [0.0, 0.0, 0.0]
            origin_pt[axis_idx] = pos
            origin_pt[o1_idx] = a1_min
            origin_pt[o2_idx] = a2_min
            
            # Point1: extend along first other axis
            pt1 = list(origin_pt)
            pt1[o1_idx] = a1_max
            
            # Point2: extend along second other axis
            pt2 = list(origin_pt)
            pt2[o2_idx] = a2_max
            
            plane_src.SetOrigin(*origin_pt)
            plane_src.SetPoint1(*pt1)
            plane_src.SetPoint2(*pt2)
                
            p_mapper = vtk.vtkPolyDataMapper()
            p_mapper.SetInputConnection(plane_src.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(p_mapper)
            
            colors = {'x': (1, 0.3, 0.3), 'y': (0.3, 1, 0.3), 'z': (0.3, 0.5, 1)}
            actor.GetProperty().SetColor(*colors[axis])
            actor.GetProperty().SetOpacity(0.35)
            actor.GetProperty().SetLighting(False)
            
            self.adv_mpr_actors[axis] = actor
            self.view_3d_renderer.AddActor(actor)
            
        self.view_3d_widget.GetRenderWindow().Render()

    # ══════════════════════════════════════════════════════════════════════
    # Dragonfly-style clip box (2×2 right sidebar — linked MPR + 3D)
    # Independent from fullscreen Advanced MPR & Clipping axis planes.
    # ══════════════════════════════════════════════════════════════════════

    def _df_clip_world_bounds(self):
        """Return clip-box world bounds from percentage sliders, or None."""
        if self.volume_data is None:
            return None
        extent = self._get_vtk_world_extent()
        out = {}
        for ax in ('x', 'y', 'z'):
            amin, amax = extent[ax]
            lo_t = max(0.0, min(1.0, self._df_clip_bounds[ax][0] / 1000.0))
            hi_t = max(0.0, min(1.0, self._df_clip_bounds[ax][1] / 1000.0))
            if hi_t <= lo_t:
                hi_t = min(1.0, lo_t + 0.001)
            out[ax] = (amin + lo_t * (amax - amin), amin + hi_t * (amax - amin))
        return out

    def _df_clip_bounds_are_partial(self):
        """True if stored clip bounds are smaller than the full volume."""
        b = getattr(self, '_df_clip_bounds', None)
        if not b:
            return False
        for ax in ('x', 'y', 'z'):
            lo, hi = b.get(ax, [0, 1000])
            if int(lo) > 0 or int(hi) < 1000:
                return True
        return False

    def _df_clip_on_enable_toggled(self, checked=False):
        """Toggle clip-box tool UI (Dragonfly): OFF keeps crop, only hides box."""
        self._df_clip_enabled = bool(
            self.btn_df_clip_enable.isChecked() if hasattr(self, 'btn_df_clip_enable') else checked
        )
        if not self._df_clip_enabled:
            # Stop interaction + hide box graphics; do NOT clear crop bounds
            self._df_clip_3d_hover = None
            self._df_clip_3d_drag = None
            self._df_clip_hover = None
            self._df_clip_drag = None
            style = getattr(self, '_3d_interactor_style', None)
            if style is not None:
                style._df_clip_dragging = False
            w = getattr(self, 'view_3d_widget', None)
            if w is not None:
                try:
                    w.unsetCursor()
                except Exception:
                    pass
            self._df_clip_clear_3d_actors()
            self._df_clip_clear_mpr_actors()
            # Re-apply crop from stored bounds (keeps clipped volume)
            self._sync_all_clip_planes()
            if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
                try:
                    self.view_3d_widget.GetRenderWindow().Render()
                except Exception:
                    pass
            for ori in ('axial', 'coronal', 'sagittal'):
                ww = getattr(self, f'{ori}_widget', None)
                if ww is not None:
                    try:
                        ww.GetRenderWindow().Render()
                    except Exception:
                        pass
            return
        # Tool ON: show box at current bounds and re-sync
        self._df_clip_refresh_all()

    def _df_clip_reset(self):
        """Reset clip to full volume — this is what restores the uncropped volume."""
        self._df_clip_bounds = {'x': [0, 1000], 'y': [0, 1000], 'z': [0, 1000]}
        for ax, ctrls in getattr(self, '_df_clip_bound_ctrls', {}).items():
            for key in ('lo', 'hi'):
                ctrls[key].blockSignals(True)
            ctrls['lo'].setValue(0)
            ctrls['hi'].setValue(1000)
            ctrls['pct'].setText("0–100%")
            for key in ('lo', 'hi'):
                ctrls[key].blockSignals(False)
        # Always re-apply crop (full) even if tool UI is off
        self._sync_all_clip_planes()
        if self._df_clip_enabled:
            self._df_clip_update_3d_visuals()
            for ori in ('axial', 'coronal', 'sagittal'):
                self._df_clip_update_mpr_overlay(ori)
        else:
            self._df_clip_clear_3d_actors()
            self._df_clip_clear_mpr_actors()
        if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
            try:
                self.view_3d_widget.GetRenderWindow().Render()
            except Exception:
                pass
        for ori in ('axial', 'coronal', 'sagittal'):
            ww = getattr(self, f'{ori}_widget', None)
            if ww is not None:
                try:
                    ww.GetRenderWindow().Render()
                except Exception:
                    pass

    def _df_clip_on_bound_slider(self, axis, which, value):
        """Update one end of the clip-box axis (0..1000) and re-apply."""
        lo, hi = self._df_clip_bounds[axis]
        if which == 'lo':
            lo = int(value)
            if lo > hi:
                hi = lo
                ctrls = self._df_clip_bound_ctrls.get(axis)
                if ctrls:
                    ctrls['hi'].blockSignals(True)
                    ctrls['hi'].setValue(hi)
                    ctrls['hi'].blockSignals(False)
        else:
            hi = int(value)
            if hi < lo:
                lo = hi
                ctrls = self._df_clip_bound_ctrls.get(axis)
                if ctrls:
                    ctrls['lo'].blockSignals(True)
                    ctrls['lo'].setValue(lo)
                    ctrls['lo'].blockSignals(False)
        self._df_clip_bounds[axis] = [lo, hi]
        ctrls = self._df_clip_bound_ctrls.get(axis)
        if ctrls:
            ctrls['pct'].setText(f"{lo / 10.0:.0f}–{hi / 10.0:.0f}%")
        # Crop always follows bounds (tool ON/OFF only controls box UI)
        if self._df_clip_enabled:
            self._df_clip_refresh_all()
        else:
            self._sync_all_clip_planes()
            if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
                try:
                    self.view_3d_widget.GetRenderWindow().Render()
                except Exception:
                    pass

    def _df_clip_on_options_changed(self, *_args):
        """Sync checkbox / grid-size options and redraw visuals."""
        if hasattr(self, 'chk_df_clip_keep'):
            self._df_clip_keep_when_hidden = self.chk_df_clip_keep.isChecked()
        if hasattr(self, 'chk_df_clip_grid_on_obj'):
            self._df_clip_display_grid_on_object = self.chk_df_clip_grid_on_obj.isChecked()
        if hasattr(self, 'chk_df_clip_grid_lines'):
            self._df_clip_show_grid = self.chk_df_clip_grid_lines.isChecked()
        if hasattr(self, 'chk_df_clip_borders'):
            self._df_clip_show_borders = self.chk_df_clip_borders.isChecked()
        if hasattr(self, 'chk_df_clip_axes'):
            self._df_clip_show_axes = self.chk_df_clip_axes.isChecked()
        if hasattr(self, 'df_clip_grid_size_spin'):
            self._df_clip_grid_size = float(self.df_clip_grid_size_spin.value())
        # Options only affect visuals (crop already applied when enabled)
        if self._df_clip_enabled or self._df_clip_keep_when_hidden:
            self._df_clip_update_3d_visuals()
            for ori in ('axial', 'coronal', 'sagittal'):
                self._df_clip_update_mpr_overlay(ori)
            if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
                self.view_3d_widget.GetRenderWindow().Render()
            for ori in ('axial', 'coronal', 'sagittal'):
                w = getattr(self, f'{ori}_widget', None)
                if w is not None:
                    try:
                        w.GetRenderWindow().Render()
                    except Exception:
                        pass

    def _df_clip_refresh_all(self):
        """Re-apply crop + rebuild 3D/MPR box visuals."""
        self._sync_all_clip_planes()
        self._df_clip_update_3d_visuals()
        for ori in ('axial', 'coronal', 'sagittal'):
            self._df_clip_update_mpr_overlay(ori)
        if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
            try:
                self.view_3d_widget.GetRenderWindow().Render()
            except Exception:
                pass
        for ori in ('axial', 'coronal', 'sagittal'):
            w = getattr(self, f'{ori}_widget', None)
            if w is not None:
                try:
                    w.GetRenderWindow().Render()
                except Exception:
                    pass

    def _df_clip_clear_3d_actors(self):
        ren = getattr(self, 'view_3d_renderer', None)
        if ren is not None:
            for a in getattr(self, '_df_clip_3d_actors', []):
                try:
                    ren.RemoveActor(a)
                except Exception:
                    pass
        self._df_clip_3d_actors = []
        self._df_clip_3d_face_actors = {}

    def _df_clip_clear_mpr_actors(self, orientation=None):
        oris = [orientation] if orientation else ['axial', 'coronal', 'sagittal']
        for ori in oris:
            ren = getattr(self, f'{ori}_renderer', None)
            actors = self._df_clip_mpr_actors.get(ori, [])
            if ren is not None:
                for a in actors:
                    try:
                        ren.RemoveActor(a)
                    except Exception:
                        pass
            self._df_clip_mpr_actors[ori] = []

    def _df_clip_should_show_visuals(self):
        """Whether the clip-box graphics should be drawn."""
        if self.volume_data is None:
            return False
        if self._df_clip_enabled:
            return True
        # Keep box when volume is "hidden" (volume actor invisible / missing)
        if self._df_clip_keep_when_hidden and getattr(self, 'volume_actor', None) is not None:
            try:
                if not self.volume_actor.GetVisibility():
                    return True
            except Exception:
                pass
        return False

    def _df_clip_add_line_actor_3d(self, p1, p2, color, width=1.5):
        src = vtk.vtkLineSource()
        src.SetPoint1(*p1)
        src.SetPoint2(*p2)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(src.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(width)
        actor.GetProperty().SetLighting(False)
        actor.GetProperty().SetOpacity(0.95)
        return actor

    def _df_clip_add_face_plate_3d(self, origin, p1, p2, color, opacity=0.12):
        """Semi-transparent rectangle for one clip-box face."""
        src = vtk.vtkPlaneSource()
        src.SetOrigin(*origin)
        src.SetPoint1(*p1)
        src.SetPoint2(*p2)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(src.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetLighting(False)
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        actor.GetProperty().BackfaceCullingOff()
        return actor

    # ── 3D viewport face drag (Dragonfly) ───────────────────────────────
    # Primary path: high-priority VTK observers + Qt eventFilter backup.
    # Pick: display-space proximity to face-center handles (reliable),
    #       with geometric ray–face as secondary.

    def _df_clip_3d_qt_to_display(self, qt_pos):
        """Qt widget coords (top-left origin) → VTK display (bottom-left)."""
        w = getattr(self, 'view_3d_widget', None)
        if w is None:
            return None
        try:
            size = w.GetRenderWindow().GetSize()
        except Exception:
            return None
        return float(qt_pos.x()), float(size[1] - qt_pos.y())

    def _df_clip_3d_event_display(self):
        """Current interactor display position, or None."""
        w = getattr(self, 'view_3d_widget', None)
        if w is None:
            return None
        iren = w.GetRenderWindow().GetInteractor()
        if iren is None:
            return None
        return iren.GetEventPosition()

    def _on_df_clip_3d_left_press(self, obj, event):
        """VTK observer: grab clip face before camera Track starts."""
        if not getattr(self, '_df_clip_enabled', False) or self.volume_data is None:
            return
        if self._df_clip_3d_on_left_down():
            style = getattr(self, '_3d_interactor_style', None)
            if style is not None:
                style._df_clip_dragging = True
                style._track_active = False
            # Abort further observers / style for this press when possible
            try:
                if hasattr(obj, 'SetAbortFlag'):
                    obj.SetAbortFlag(1)
            except Exception:
                pass

    def _on_df_clip_3d_mouse_move(self, obj, event):
        """VTK observer: face drag or hover highlight."""
        if getattr(self, '_df_clip_3d_drag', None):
            self._df_clip_3d_on_mouse_move()
            return
        if getattr(self, '_df_clip_enabled', False) and self.volume_data is not None:
            self._df_clip_3d_on_hover()

    def _on_df_clip_3d_left_release(self, obj, event):
        """VTK observer: end face drag."""
        if getattr(self, '_df_clip_3d_drag', None):
            self._df_clip_3d_on_left_up()
            style = getattr(self, '_3d_interactor_style', None)
            if style is not None:
                style._df_clip_dragging = False

    def _df_clip_3d_try_grab_at_qt_pos(self, qt_pos):
        """Qt eventFilter press path. Returns True if face grabbed."""
        disp = self._df_clip_3d_qt_to_display(qt_pos)
        if disp is None:
            return False
        return self._df_clip_3d_begin_drag_at_display(disp[0], disp[1])

    def _df_clip_3d_drag_at_qt_pos(self, qt_pos):
        disp = self._df_clip_3d_qt_to_display(qt_pos)
        if disp is None:
            return
        self._df_clip_3d_update_drag_at_display(disp[0], disp[1])

    def _df_clip_3d_hover_at_qt_pos(self, qt_pos):
        disp = self._df_clip_3d_qt_to_display(qt_pos)
        if disp is None:
            return
        self._df_clip_3d_hover_at_display(disp[0], disp[1])

    def _df_clip_3d_display_ray(self, dx, dy):
        """Camera ray through display pixel → (origin, direction) world."""
        ren = getattr(self, 'view_3d_renderer', None)
        if ren is None:
            return None
        try:
            ren.SetDisplayPoint(float(dx), float(dy), 0.0)
            ren.DisplayToWorld()
            n = ren.GetWorldPoint()
            if abs(n[3]) < 1e-12:
                return None
            near = [n[0] / n[3], n[1] / n[3], n[2] / n[3]]

            ren.SetDisplayPoint(float(dx), float(dy), 1.0)
            ren.DisplayToWorld()
            f = ren.GetWorldPoint()
            if abs(f[3]) < 1e-12:
                return None
            far = [f[0] / f[3], f[1] / f[3], f[2] / f[3]]
        except Exception:
            return None

        d = [far[0] - near[0], far[1] - near[1], far[2] - near[2]]
        ln = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) ** 0.5
        if ln < 1e-15:
            return None
        d = [d[0] / ln, d[1] / ln, d[2] / ln]
        return near, d

    def _df_clip_3d_faces(self):
        """Six clip faces with center points for handle picking."""
        box = self._df_clip_world_bounds()
        if box is None:
            return []
        x0, x1 = box['x']
        y0, y1 = box['y']
        z0, z1 = box['z']
        mx, my, mz = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)
        # (axis, side, normal, plane_point, a0,a1, b0,b1, a_idx, b_idx, center)
        return [
            ('x', 'lo', (-1.0, 0.0, 0.0), (x0, y0, z0), y0, y1, z0, z1, 1, 2, (x0, my, mz)),
            ('x', 'hi', (1.0, 0.0, 0.0), (x1, y0, z0), y0, y1, z0, z1, 1, 2, (x1, my, mz)),
            ('y', 'lo', (0.0, -1.0, 0.0), (x0, y0, z0), x0, x1, z0, z1, 0, 2, (mx, y0, mz)),
            ('y', 'hi', (0.0, 1.0, 0.0), (x0, y1, z0), x0, x1, z0, z1, 0, 2, (mx, y1, mz)),
            ('z', 'lo', (0.0, 0.0, -1.0), (x0, y0, z0), x0, x1, y0, y1, 0, 1, (mx, my, z0)),
            ('z', 'hi', (0.0, 0.0, 1.0), (x0, y0, z1), x0, x1, y0, y1, 0, 1, (mx, my, z1)),
        ]

    def _get_df_clip_3d_face_hit(self, dx, dy, pad=0.0):
        """Pick a clip face under display pixel.

        Prefer display-space hit on face-center handles (works when box is full
        volume / faces edge-on). Fall back to geometric ray–face intersection.
        """
        import math
        ren = getattr(self, 'view_3d_renderer', None)
        if ren is None or self.volume_data is None:
            return None
        faces = self._df_clip_3d_faces()
        if not faces:
            return None

        # 1) Display-space proximity to face-center handles
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        handle_tol = 22.0  # pixels — generous for screenshot-scale views
        best_h = None
        best_hd = handle_tol
        for axis, side, normal, p0, a0, a1, b0, b1, ai, bi, center in faces:
            coord.SetValue(float(center[0]), float(center[1]), float(center[2]))
            try:
                dp = coord.GetComputedDisplayValue(ren)
            except Exception:
                continue
            dist = math.hypot(float(dx) - float(dp[0]), float(dy) - float(dp[1]))
            if dist <= best_hd:
                best_hd = dist
                best_h = {
                    'axis': axis,
                    'side': side,
                    'point': list(center),
                    't': 0.0,
                    'normal': normal,
                    'via': 'handle',
                }
        if best_h is not None:
            return best_h

        # 2) Geometric ray–face (full plate pick)
        ray = self._df_clip_3d_display_ray(dx, dy)
        if ray is None:
            return None
        origin, direction = ray
        best = None
        best_t = 1e30
        eps = max(pad, 1e-4)
        for axis, side, normal, p0, a0, a1, b0, b1, ai, bi, center in faces:
            nd = normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2]
            if abs(nd) < 1e-10:
                continue
            w = [p0[0] - origin[0], p0[1] - origin[1], p0[2] - origin[2]]
            t = (normal[0] * w[0] + normal[1] * w[1] + normal[2] * w[2]) / nd
            if t < 0.0:
                continue
            hit = [
                origin[0] + t * direction[0],
                origin[1] + t * direction[1],
                origin[2] + t * direction[2],
            ]
            a = hit[ai]
            b = hit[bi]
            if (a0 - eps) <= a <= (a1 + eps) and (b0 - eps) <= b <= (b1 + eps):
                if t < best_t:
                    best_t = t
                    best = {
                        'axis': axis,
                        'side': side,
                        'point': hit,
                        't': t,
                        'normal': normal,
                        'via': 'ray',
                    }
        return best

    def _df_clip_3d_hover_at_display(self, dx, dy):
        """Highlight face under display pixel; redraw only on change."""
        if not getattr(self, '_df_clip_enabled', False) or self.volume_data is None:
            return
        if getattr(self, '_df_clip_3d_drag', None):
            return
        box = self._df_clip_world_bounds()
        pad = 0.0
        if box is not None:
            span = max(
                box['x'][1] - box['x'][0],
                box['y'][1] - box['y'][0],
                box['z'][1] - box['z'][0],
                1.0,
            )
            pad = span * 0.03
        hit = self._get_df_clip_3d_face_hit(dx, dy, pad=pad)
        key = (hit['axis'], hit['side']) if hit else None
        prev = self._df_clip_3d_hover
        prev_key = (prev['axis'], prev['side']) if prev else None
        w = getattr(self, 'view_3d_widget', None)
        if key == prev_key:
            if w is not None:
                if hit is not None:
                    w.setCursor(Qt.SizeAllCursor)
                elif prev is not None:
                    w.unsetCursor()
            return
        self._df_clip_3d_hover = (
            {'axis': hit['axis'], 'side': hit['side']} if hit else None
        )
        self._df_clip_update_3d_visuals()
        if w is not None:
            if hit is not None:
                w.setCursor(Qt.SizeAllCursor)
            else:
                w.unsetCursor()
            try:
                w.GetRenderWindow().Render()
            except Exception:
                pass

    def _df_clip_3d_on_hover(self):
        """Hover via VTK interactor event position."""
        pos = self._df_clip_3d_event_display()
        if pos is None:
            return
        self._df_clip_3d_hover_at_display(pos[0], pos[1])

    def _df_clip_3d_begin_drag_at_display(self, dx, dy):
        """Start face drag at display pixel. Returns True if consumed."""
        if not getattr(self, '_df_clip_enabled', False) or self.volume_data is None:
            return False
        # Already dragging — ignore re-entrant begin (style + observer double fire)
        if getattr(self, '_df_clip_3d_drag', None):
            return True
        box = self._df_clip_world_bounds()
        pad = 0.0
        if box is not None:
            span = max(
                box['x'][1] - box['x'][0],
                box['y'][1] - box['y'][0],
                box['z'][1] - box['z'][0],
                1.0,
            )
            pad = span * 0.03
        hit = self._get_df_clip_3d_face_hit(dx, dy, pad=pad)
        if hit is None:
            return False

        ren = self.view_3d_renderer
        cam = ren.GetActiveCamera()
        fn = list(hit['normal'])
        pos_c = cam.GetPosition()
        fp = cam.GetFocalPoint()
        view = [fp[0] - pos_c[0], fp[1] - pos_c[1], fp[2] - pos_c[2]]
        vn = (view[0] ** 2 + view[1] ** 2 + view[2] ** 2) ** 0.5
        if vn < 1e-12:
            view = [0.0, 0.0, -1.0]
        else:
            view = [view[0] / vn, view[1] / vn, view[2] / vn]
        side = [
            view[1] * fn[2] - view[2] * fn[1],
            view[2] * fn[0] - view[0] * fn[2],
            view[0] * fn[1] - view[1] * fn[0],
        ]
        sn = (side[0] ** 2 + side[1] ** 2 + side[2] ** 2) ** 0.5
        if sn < 1e-8:
            up = list(cam.GetViewUp())
            side = [
                up[1] * fn[2] - up[2] * fn[1],
                up[2] * fn[0] - up[0] * fn[2],
                up[0] * fn[1] - up[1] * fn[0],
            ]
            sn = (side[0] ** 2 + side[1] ** 2 + side[2] ** 2) ** 0.5
        if sn < 1e-8:
            return False
        side = [side[0] / sn, side[1] / sn, side[2] / sn]
        dpn = [
            fn[1] * side[2] - fn[2] * side[1],
            fn[2] * side[0] - fn[0] * side[2],
            fn[0] * side[1] - fn[1] * side[0],
        ]
        dpn_n = (dpn[0] ** 2 + dpn[1] ** 2 + dpn[2] ** 2) ** 0.5
        if dpn_n < 1e-8:
            return False
        dpn = [dpn[0] / dpn_n, dpn[1] / dpn_n, dpn[2] / dpn_n]

        import copy
        self._df_clip_3d_drag = {
            'axis': hit['axis'],
            'side': hit['side'],
            'start_bounds': copy.deepcopy(self._df_clip_bounds),
            'start_point': list(hit['point']),
            'plane_point': list(hit['point']),
            'plane_normal': dpn,
            'face_normal': fn,
        }
        self._df_clip_3d_hover = {'axis': hit['axis'], 'side': hit['side']}
        style = getattr(self, '_3d_interactor_style', None)
        if style is not None:
            style._df_clip_dragging = True
            style._track_active = False
        self._df_clip_update_3d_visuals()
        w = getattr(self, 'view_3d_widget', None)
        if w is not None:
            w.setCursor(Qt.SizeAllCursor)
            try:
                w.GetRenderWindow().Render()
            except Exception:
                pass
        return True

    def _df_clip_3d_on_left_down(self):
        """Begin face drag from VTK event position. Returns True if consumed."""
        pos = self._df_clip_3d_event_display()
        if pos is None:
            return False
        return self._df_clip_3d_begin_drag_at_display(pos[0], pos[1])

    def _df_clip_3d_update_drag_at_display(self, dx, dy):
        """Update clip bound while dragging a 3D face (display coords)."""
        drag = getattr(self, '_df_clip_3d_drag', None)
        if not drag or self.volume_data is None:
            return
        ray = self._df_clip_3d_display_ray(dx, dy)
        if ray is None:
            return
        origin, direction = ray
        n = drag['plane_normal']
        p0 = drag['plane_point']
        nd = n[0] * direction[0] + n[1] * direction[1] + n[2] * direction[2]
        if abs(nd) < 1e-10:
            return
        w = [p0[0] - origin[0], p0[1] - origin[1], p0[2] - origin[2]]
        t = (n[0] * w[0] + n[1] * w[1] + n[2] * w[2]) / nd
        hit = [
            origin[0] + t * direction[0],
            origin[1] + t * direction[1],
            origin[2] + t * direction[2],
        ]
        axis = drag['axis']
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        world_pos = hit[axis_idx]

        extent = self._get_vtk_world_extent()
        amin, amax = extent[axis]
        span = max(amax - amin, 1e-12)
        pct = int(round(max(0.0, min(1.0, (world_pos - amin) / span)) * 1000.0))

        bounds = {
            'x': list(drag['start_bounds']['x']),
            'y': list(drag['start_bounds']['y']),
            'z': list(drag['start_bounds']['z']),
        }
        lo0, hi0 = drag['start_bounds'][axis]
        if drag['side'] == 'lo':
            lo = max(0, min(pct, hi0 - 1))
            hi = hi0
        else:
            hi = min(1000, max(pct, lo0 + 1))
            lo = lo0
        bounds[axis] = [lo, hi]
        self._df_clip_bounds = bounds
        self._df_clip_sync_bound_sliders()
        self._sync_all_clip_planes()
        self._df_clip_update_3d_visuals()
        for o in ('axial', 'coronal', 'sagittal'):
            self._df_clip_update_mpr_overlay(o)
            ww = getattr(self, f'{o}_widget', None)
            if ww is not None:
                try:
                    ww.GetRenderWindow().Render()
                except Exception:
                    pass
        if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
            try:
                self.view_3d_widget.GetRenderWindow().Render()
            except Exception:
                pass

    def _df_clip_3d_on_mouse_move(self):
        """Update drag from VTK event position."""
        pos = self._df_clip_3d_event_display()
        if pos is None:
            return
        self._df_clip_3d_update_drag_at_display(pos[0], pos[1])

    def _df_clip_3d_on_left_up(self):
        self._df_clip_3d_drag = None
        style = getattr(self, '_3d_interactor_style', None)
        if style is not None:
            style._df_clip_dragging = False
        self._df_clip_refresh_all()
        # Restore hover cursor if still over a face
        self._df_clip_3d_on_hover()

    def _df_clip_update_3d_visuals(self):
        """Draw Dragonfly-style clip box (borders / face grids / axes) in 3D."""
        self._df_clip_clear_3d_actors()
        ren = getattr(self, 'view_3d_renderer', None)
        if ren is None or not self._df_clip_should_show_visuals():
            return
        show_borders = self._df_clip_show_borders or self._df_clip_enabled
        if not (show_borders or self._df_clip_show_grid
                or self._df_clip_display_grid_on_object or self._df_clip_show_axes):
            return

        box = self._df_clip_world_bounds()
        if box is None:
            return
        x0, x1 = box['x']
        y0, y1 = box['y']
        z0, z1 = box['z']

        # Active / hovered face for highlight
        active_face = None
        if getattr(self, '_df_clip_3d_drag', None):
            active_face = (
                self._df_clip_3d_drag.get('axis'),
                self._df_clip_3d_drag.get('side'),
            )
        elif getattr(self, '_df_clip_3d_hover', None):
            active_face = (
                self._df_clip_3d_hover.get('axis'),
                self._df_clip_3d_hover.get('side'),
            )

        # Face plates only when hovered/dragged (no permanent sphere handles)
        if self._df_clip_enabled and active_face is not None:
            face_defs = [
                # axis, side, origin, p1, p2, color
                ('x', 'lo', (x0, y0, z0), (x0, y1, z0), (x0, y0, z1), (1.0, 0.3, 0.3)),
                ('x', 'hi', (x1, y0, z0), (x1, y1, z0), (x1, y0, z1), (1.0, 0.3, 0.3)),
                ('y', 'lo', (x0, y0, z0), (x1, y0, z0), (x0, y0, z1), (0.3, 1.0, 0.3)),
                ('y', 'hi', (x0, y1, z0), (x1, y1, z0), (x0, y1, z1), (0.3, 1.0, 0.3)),
                ('z', 'lo', (x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (0.35, 0.55, 1.0)),
                ('z', 'hi', (x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (0.35, 0.55, 1.0)),
            ]
            for ax, side, origin, p1, p2, col in face_defs:
                if active_face != (ax, side):
                    continue
                plate = self._df_clip_add_face_plate_3d(origin, p1, p2, col, opacity=0.28)
                ren.AddActor(plate)
                self._df_clip_3d_actors.append(plate)

        # 8 corners
        corners = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        # 12 edges (i,j)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        # Edges belonging to each face (for highlight thickness)
        face_edge_map = {
            ('x', 'lo'): {(3, 0), (0, 4), (4, 7), (7, 3), (0, 3), (4, 0), (7, 4), (3, 7)},
            ('x', 'hi'): {(1, 2), (2, 6), (6, 5), (5, 1), (2, 1), (6, 2), (5, 6), (1, 5)},
            ('y', 'lo'): {(0, 1), (1, 5), (5, 4), (4, 0), (1, 0), (5, 1), (4, 5), (0, 4)},
            ('y', 'hi'): {(3, 2), (2, 6), (6, 7), (7, 3), (2, 3), (6, 2), (7, 6), (3, 7)},
            ('z', 'lo'): {(0, 1), (1, 2), (2, 3), (3, 0), (1, 0), (2, 1), (3, 2), (0, 3)},
            ('z', 'hi'): {(4, 5), (5, 6), (6, 7), (7, 4), (5, 4), (6, 5), (7, 6), (4, 7)},
        }
        border_color = (0.85, 0.9, 1.0)
        if show_borders:
            for i, j in edges:
                ekey = (i, j)
                ekey_r = (j, i)
                thick = 2.0
                col = border_color
                if active_face is not None:
                    em = face_edge_map.get(active_face, set())
                    if ekey in em or ekey_r in em:
                        thick = 3.5
                        col = (1.0, 0.95, 0.4)
                actor = self._df_clip_add_line_actor_3d(corners[i], corners[j], col, thick)
                ren.AddActor(actor)
                self._df_clip_3d_actors.append(actor)

        # Face grids (on the six faces of the box) — density capped for perf
        show_grid = self._df_clip_show_grid or self._df_clip_display_grid_on_object
        if show_grid:
            gs = max(float(self._df_clip_grid_size), 1e-3)
            # Cap ~24 lines per axis per face to keep FPS high on large volumes
            max_lines = 24
            dx = max(x1 - x0, 1e-6)
            dy = max(y1 - y0, 1e-6)
            dz = max(z1 - z0, 1e-6)
            gs_x = max(gs, dx / max_lines)
            gs_y = max(gs, dy / max_lines)
            gs_z = max(gs, dz / max_lines)
            grid_color = (0.55, 0.75, 0.95)

            def _grid_range(a0, a1, step):
                vals = []
                t = a0
                while t <= a1 + 1e-9:
                    vals.append(t)
                    t += step
                if not vals or abs(vals[-1] - a1) > 1e-6:
                    vals.append(a1)
                return vals

            # XY faces (z = z0, z1)
            for zf in (z0, z1):
                for x in _grid_range(x0, x1, gs_x):
                    actor = self._df_clip_add_line_actor_3d((x, y0, zf), (x, y1, zf), grid_color, 1.0)
                    ren.AddActor(actor)
                    self._df_clip_3d_actors.append(actor)
                for y in _grid_range(y0, y1, gs_y):
                    actor = self._df_clip_add_line_actor_3d((x0, y, zf), (x1, y, zf), grid_color, 1.0)
                    ren.AddActor(actor)
                    self._df_clip_3d_actors.append(actor)
            # XZ faces (y = y0, y1)
            for yf in (y0, y1):
                for x in _grid_range(x0, x1, gs_x):
                    actor = self._df_clip_add_line_actor_3d((x, yf, z0), (x, yf, z1), grid_color, 1.0)
                    ren.AddActor(actor)
                    self._df_clip_3d_actors.append(actor)
                for z in _grid_range(z0, z1, gs_z):
                    actor = self._df_clip_add_line_actor_3d((x0, yf, z), (x1, yf, z), grid_color, 1.0)
                    ren.AddActor(actor)
                    self._df_clip_3d_actors.append(actor)
            # YZ faces (x = x0, x1)
            for xf in (x0, x1):
                for y in _grid_range(y0, y1, gs_y):
                    actor = self._df_clip_add_line_actor_3d((xf, y, z0), (xf, y, z1), grid_color, 1.0)
                    ren.AddActor(actor)
                    self._df_clip_3d_actors.append(actor)
                for z in _grid_range(z0, z1, gs_z):
                    actor = self._df_clip_add_line_actor_3d((xf, y0, z), (xf, y1, z), grid_color, 1.0)
                    ren.AddActor(actor)
                    self._df_clip_3d_actors.append(actor)

        # RGB axes at box origin (min corner)
        if self._df_clip_show_axes:
            extent = self._get_vtk_world_extent()
            L = 0.15 * max(
                extent['x'][1] - extent['x'][0],
                extent['y'][1] - extent['y'][0],
                extent['z'][1] - extent['z'][0],
                1.0,
            )
            for p2, col in (
                ((x0 + L, y0, z0), (1.0, 0.25, 0.25)),
                ((x0, y0 + L, z0), (0.25, 1.0, 0.25)),
                ((x0, y0, z0 + L), (0.3, 0.45, 1.0)),
            ):
                actor = self._df_clip_add_line_actor_3d((x0, y0, z0), p2, col, 2.5)
                ren.AddActor(actor)
                self._df_clip_3d_actors.append(actor)

    def _df_clip_plane_rect(self, orientation):
        """Clip-box rectangle in MPR display (voxel) coords + axis mapping.

        Returns dict:
          u0,u1,v0,v1  — display-space rect (always u0<=u1, v0<=v1)
          u_axis, v_axis — 'x'|'y'|'z'
          u_lo_is_min, v_lo_is_min — whether display-lo edge is axis min
          h_color, v_color, u_spacing, v_spacing
        """
        if self.volume_data is None:
            return None
        box = self._df_clip_world_bounds()
        if box is None:
            return None
        spacing = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        vol_z, vol_y, vol_x = self.volume_data.shape

        vx0 = box['x'][0] / max(sx, 1e-12)
        vx1 = box['x'][1] / max(sx, 1e-12)
        vy0 = box['y'][0] / max(sy, 1e-12)
        vy1 = box['y'][1] / max(sy, 1e-12)
        vz0 = box['z'][0] / max(sz, 1e-12)
        vz1 = box['z'][1] / max(sz, 1e-12)

        if orientation == 'axial':
            return {
                'u0': vx0, 'u1': vx1, 'v0': vy0, 'v1': vy1,
                'u_axis': 'x', 'v_axis': 'y',
                'u_lo_is_min': True, 'v_lo_is_min': True,
                'h_color': (1.0, 0.25, 0.25), 'v_color': (0.25, 1.0, 0.2),
                'u_spacing': sx, 'v_spacing': sy,
            }
        if orientation == 'coronal':
            # flipud(z): display_v = (z-1) - vz  →  low display = high z
            dv0 = (vol_z - 1) - vz1  # edge at z_max
            dv1 = (vol_z - 1) - vz0  # edge at z_min
            return {
                'u0': vx0, 'u1': vx1,
                'v0': min(dv0, dv1), 'v1': max(dv0, dv1),
                'u_axis': 'x', 'v_axis': 'z',
                'u_lo_is_min': True, 'v_lo_is_min': False,  # display lo = z max
                'h_color': (1.0, 0.25, 0.25), 'v_color': (0.25, 0.45, 1.0),
                'u_spacing': sx, 'v_spacing': sz,
            }
        # sagittal: display x=z, y=y
        return {
            'u0': vz0, 'u1': vz1, 'v0': vy0, 'v1': vy1,
            'u_axis': 'z', 'v_axis': 'y',
            'u_lo_is_min': True, 'v_lo_is_min': True,
            'h_color': (0.25, 0.45, 1.0), 'v_color': (0.25, 1.0, 0.2),
            'u_spacing': sz, 'v_spacing': sy,
        }

    def _df_clip_pick_display_uv(self, pos, orientation):
        """Map Qt mouse pos → MPR display (u,v) in voxel/image coords."""
        widget = getattr(self, f'{orientation}_widget', None)
        renderer = getattr(self, f'{orientation}_renderer', None)
        if widget is None or renderer is None:
            return None
        size = widget.GetRenderWindow().GetSize()
        x = float(pos.x())
        vtk_y = float(size[1] - pos.y())
        picker = vtk.vtkWorldPointPicker()
        picker.Pick(x, vtk_y, 0, renderer)
        wp = picker.GetPickPosition()
        return float(wp[0]), float(wp[1])

    def _df_clip_display_to_pct(self, orientation, u=None, v=None):
        """Convert display u/v (voxel) to clip % (0..1000) for the mapped axes.

        Returns dict partial: {axis: pct_int, ...}
        """
        rect = self._df_clip_plane_rect(orientation)
        if rect is None or self.volume_data is None:
            return {}
        spacing = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        vol_z = self.volume_data.shape[0]
        extent = self._get_vtk_world_extent()
        out = {}

        def _axis_pct(axis, voxel):
            sp = {'x': sx, 'y': sy, 'z': sz}[axis]
            world = float(voxel) * sp
            amin, amax = extent[axis]
            span = max(amax - amin, 1e-12)
            t = (world - amin) / span
            return int(round(max(0.0, min(1.0, t)) * 1000.0))

        if u is not None:
            u_axis = rect['u_axis']
            # display u always maps monotonically for axial/sagittal/coronal-x
            out[u_axis] = _axis_pct(u_axis, u)
        if v is not None:
            v_axis = rect['v_axis']
            if orientation == 'coronal' and v_axis == 'z':
                # display_v = (z-1) - vz  →  vz = (z-1) - display_v
                vz = (vol_z - 1) - float(v)
                out[v_axis] = _axis_pct(v_axis, vz)
            else:
                out[v_axis] = _axis_pct(v_axis, v)
        return out

    @staticmethod
    def _df_clip_hit_key(hit):
        """Stable compare key for hover redraws."""
        if hit is None:
            return None
        return (hit.get('kind'), hit.get('u_side'), hit.get('v_side'), hit.get('orientation'))

    @staticmethod
    def _df_clip_cursor_for_hit(hit):
        if hit is None:
            return Qt.ArrowCursor
        kind = hit.get('kind')
        if kind == 'move':
            return Qt.SizeAllCursor
        if kind == 'corner':
            # Diagonal: lo/lo & hi/hi → FDiag; lo/hi & hi/lo → BDiag
            us, vs = hit.get('u_side'), hit.get('v_side')
            if (us == 'lo' and vs == 'lo') or (us == 'hi' and vs == 'hi'):
                return Qt.SizeFDiagCursor
            return Qt.SizeBDiagCursor
        if kind == 'edge':
            if hit.get('u_side') is not None:
                return Qt.SizeHorCursor
            return Qt.SizeVerCursor
        return Qt.ArrowCursor

    def _get_df_clip_hit(self, pos, orientation):
        """Hit-test clip-box edges / corners / interior on an MPR pane.

        Returns hit dict or None:
          kind: 'corner'|'edge'|'move'
          u_side: 'lo'|'hi'|None
          v_side: 'lo'|'hi'|None
        """
        import math
        if not getattr(self, '_df_clip_enabled', False) or self.volume_data is None:
            return None
        rect = self._df_clip_plane_rect(orientation)
        if rect is None:
            return None
        ren = getattr(self, f'{orientation}_renderer', None)
        widget = getattr(self, f'{orientation}_widget', None)
        if ren is None or widget is None:
            return None

        # Project rect corners to display pixels
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()

        def _to_disp(wu, wv):
            coord.SetValue(float(wu), float(wv), 0.5)
            d = coord.GetComputedDisplayValue(ren)
            return float(d[0]), float(d[1])

        u0, u1 = rect['u0'], rect['u1']
        v0, v1 = rect['v0'], rect['v1']
        # Four corners in display
        c_ll = _to_disp(u0, v0)
        c_hl = _to_disp(u1, v0)
        c_lh = _to_disp(u0, v1)
        c_hh = _to_disp(u1, v1)

        size = widget.GetRenderWindow().GetSize()
        mx = float(pos.x())
        my = float(size[1] - pos.y())  # VTK display Y

        corner_tol = 12.0
        edge_tol = 9.0

        # Corners first
        corners = [
            ('lo', 'lo', c_ll),
            ('hi', 'lo', c_hl),
            ('lo', 'hi', c_lh),
            ('hi', 'hi', c_hh),
        ]
        for us, vs, (cx, cy) in corners:
            if math.hypot(mx - cx, my - cy) <= corner_tol:
                return {'kind': 'corner', 'u_side': us, 'v_side': vs, 'orientation': orientation}

        def _dist_seg(px, py, ax, ay, bx, by):
            abx, aby = bx - ax, by - ay
            apx, apy = px - ax, py - ay
            ab2 = abx * abx + aby * aby
            if ab2 < 1e-12:
                return math.hypot(apx, apy)
            t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
            qx, qy = ax + t * abx, ay + t * aby
            return math.hypot(px - qx, py - qy)

        edges = [
            ('u', 'lo', c_ll, c_lh),  # left  u=u0
            ('u', 'hi', c_hl, c_hh),  # right u=u1
            ('v', 'lo', c_ll, c_hl),  # bottom v=v0
            ('v', 'hi', c_lh, c_hh),  # top    v=v1
        ]
        best = None
        best_d = edge_tol
        for axis_uv, side, (ax, ay), (bx, by) in edges:
            d = _dist_seg(mx, my, ax, ay, bx, by)
            if d <= best_d:
                best_d = d
                if axis_uv == 'u':
                    best = {'kind': 'edge', 'u_side': side, 'v_side': None, 'orientation': orientation}
                else:
                    best = {'kind': 'edge', 'u_side': None, 'v_side': side, 'orientation': orientation}
        if best is not None:
            return best

        # Center handle only (not full interior) so W/L + crosshair stay usable
        mid_x = 0.25 * (c_ll[0] + c_hl[0] + c_lh[0] + c_hh[0])
        mid_y = 0.25 * (c_ll[1] + c_hl[1] + c_lh[1] + c_hh[1])
        if math.hypot(mx - mid_x, my - mid_y) <= 14.0:
            return {'kind': 'move', 'u_side': None, 'v_side': None, 'orientation': orientation}
        return None

    def _df_clip_begin_drag(self, orientation, pos, hit):
        uv = self._df_clip_pick_display_uv(pos, orientation)
        if uv is None:
            return
        import copy
        self._df_clip_drag = {
            'orientation': orientation,
            'hit': dict(hit),
            'start_bounds': copy.deepcopy(self._df_clip_bounds),
            'start_uv': uv,
        }
        self._df_clip_hover = hit
        # Highlight handles
        self._df_clip_update_mpr_overlay(orientation)
        w = getattr(self, f'{orientation}_widget', None)
        if w is not None:
            w.setCursor(self._df_clip_cursor_for_hit(hit))
            try:
                w.GetRenderWindow().Render()
            except Exception:
                pass

    def _df_clip_handle_drag(self, pos, orientation):
        drag = getattr(self, '_df_clip_drag', None)
        if not drag or self.volume_data is None:
            return
        # Stick to the pane that started the drag
        ori = drag.get('orientation', orientation)
        uv = self._df_clip_pick_display_uv(pos, ori)
        if uv is None:
            return
        hit = drag['hit']
        kind = hit.get('kind')
        start_bounds = drag['start_bounds']
        su, sv = drag['start_uv']
        cu, cv = uv

        # Working copy from start (absolute for edges; delta for move)
        bounds = {
            'x': list(start_bounds['x']),
            'y': list(start_bounds['y']),
            'z': list(start_bounds['z']),
        }
        rect = self._df_clip_plane_rect(ori)
        if rect is None:
            return

        def _set_side(axis, side_is_lo, pct):
            lo, hi = bounds[axis]
            if side_is_lo:
                lo = max(0, min(int(pct), hi - 1))
            else:
                hi = min(1000, max(int(pct), lo + 1))
            bounds[axis] = [lo, hi]

        def _apply_u_side(side, pct):
            # side 'lo'/'hi' is display side; map to axis min/max
            if side is None:
                return
            is_min = rect['u_lo_is_min'] if side == 'lo' else (not rect['u_lo_is_min'])
            _set_side(rect['u_axis'], is_min, pct)

        def _apply_v_side(side, pct):
            if side is None:
                return
            is_min = rect['v_lo_is_min'] if side == 'lo' else (not rect['v_lo_is_min'])
            _set_side(rect['v_axis'], is_min, pct)

        if kind == 'move':
            # Delta in display → delta in % for both plane axes
            pct_start = self._df_clip_display_to_pct(ori, u=su, v=sv)
            pct_now = self._df_clip_display_to_pct(ori, u=cu, v=cv)
            for axis in (rect['u_axis'], rect['v_axis']):
                if axis not in pct_start or axis not in pct_now:
                    continue
                d = int(pct_now[axis] - pct_start[axis])
                lo0, hi0 = start_bounds[axis]
                lo = lo0 + d
                hi = hi0 + d
                # Clamp keeping width
                width = hi0 - lo0
                if lo < 0:
                    lo, hi = 0, width
                if hi > 1000:
                    hi, lo = 1000, 1000 - width
                bounds[axis] = [max(0, lo), min(1000, hi)]
        else:
            # Edge / corner: set absolute positions from current pick
            if hit.get('u_side') is not None:
                pcts = self._df_clip_display_to_pct(ori, u=cu, v=None)
                if rect['u_axis'] in pcts:
                    _apply_u_side(hit['u_side'], pcts[rect['u_axis']])
            if hit.get('v_side') is not None:
                pcts = self._df_clip_display_to_pct(ori, u=None, v=cv)
                if rect['v_axis'] in pcts:
                    _apply_v_side(hit['v_side'], pcts[rect['v_axis']])

        self._df_clip_bounds = bounds
        self._df_clip_sync_bound_sliders()
        # Lightweight live update (no full reslice)
        self._sync_all_clip_planes()
        self._df_clip_update_3d_visuals()
        for o in ('axial', 'coronal', 'sagittal'):
            self._df_clip_update_mpr_overlay(o)
            w = getattr(self, f'{o}_widget', None)
            if w is not None:
                try:
                    w.GetRenderWindow().Render()
                except Exception:
                    pass
        if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
            try:
                self.view_3d_widget.GetRenderWindow().Render()
            except Exception:
                pass

    def _df_clip_end_drag(self):
        self._df_clip_drag = None
        # Final full sync
        self._df_clip_refresh_all()

    def _df_clip_sync_bound_sliders(self):
        """Push current bounds into sidebar sliders without re-entrancy."""
        for ax, ctrls in getattr(self, '_df_clip_bound_ctrls', {}).items():
            lo, hi = self._df_clip_bounds[ax]
            ctrls['lo'].blockSignals(True)
            ctrls['hi'].blockSignals(True)
            ctrls['lo'].setValue(int(lo))
            ctrls['hi'].setValue(int(hi))
            ctrls['pct'].setText(f"{lo / 10.0:.0f}–{hi / 10.0:.0f}%")
            ctrls['lo'].blockSignals(False)
            ctrls['hi'].blockSignals(False)

    def _df_clip_update_mpr_overlay(self, orientation):
        """Draw clip-box rectangle, grid, and interactive handles on one MPR pane.

        Coordinates match render_slice display space (voxel units, not world):
          axial    → (x, y)
          coronal  → (x, flipud z)
          sagittal → (z, y) after transpose
        """
        self._df_clip_clear_mpr_actors(orientation)
        ren = getattr(self, f'{orientation}_renderer', None)
        if ren is None or self.volume_data is None:
            return
        if not self._df_clip_should_show_visuals():
            return

        rect = self._df_clip_plane_rect(orientation)
        if rect is None:
            return

        u0, u1 = rect['u0'], rect['u1']
        v0, v1 = rect['v0'], rect['v1']
        h_color = rect['h_color']
        v_color = rect['v_color']
        u_spacing = rect['u_spacing']
        v_spacing = rect['v_spacing']

        # Always show at least a thin box when clip is enabled (for drag targets)
        show_borders = self._df_clip_show_borders or self._df_clip_enabled
        show_grid = self._df_clip_show_grid or self._df_clip_display_grid_on_object
        if not (show_borders or show_grid):
            return

        def _add_line(x1, y1, x2, y2, color, width=1.5):
            src = vtk.vtkLineSource()
            src.SetPoint1(float(x1), float(y1), 0.5)
            src.SetPoint2(float(x2), float(y2), 0.5)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(src.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(width)
            actor.GetProperty().SetLighting(False)
            ren.AddActor(actor)
            self._df_clip_mpr_actors[orientation].append(actor)

        def _add_handle(cx, cy, color, size=3.5, highlight=False):
            # Small square handle via 4 lines (robust, no extra VTK filters)
            s = size * (1.4 if highlight else 1.0)
            col = tuple(min(1.0, c * 1.25) for c in color) if highlight else color
            w = 2.5 if highlight else 1.8
            _add_line(cx - s, cy - s, cx + s, cy - s, col, w)
            _add_line(cx + s, cy - s, cx + s, cy + s, col, w)
            _add_line(cx + s, cy + s, cx - s, cy + s, col, w)
            _add_line(cx - s, cy + s, cx - s, cy - s, col, w)

        hover = getattr(self, '_df_clip_hover', None)
        hover_on = (
            hover is not None
            and hover.get('orientation') == orientation
        ) or (
            getattr(self, '_df_clip_drag', None) is not None
            and self._df_clip_drag.get('orientation') == orientation
        )
        active = self._df_clip_drag['hit'] if getattr(self, '_df_clip_drag', None) else hover

        border_col = (0.9, 0.95, 1.0)
        if show_borders:
            # rectangle
            _add_line(u0, v0, u1, v0, border_col, 2.0)
            _add_line(u1, v0, u1, v1, border_col, 2.0)
            _add_line(u1, v1, u0, v1, border_col, 2.0)
            _add_line(u0, v1, u0, v0, border_col, 2.0)
            # colored edge accents (Dragonfly axis tint); thicken active edge
