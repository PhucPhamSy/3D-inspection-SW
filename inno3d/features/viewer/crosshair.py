"""Extracted mixin — part of inno3d.features.viewer refactor.

This module contains CrosshairMixin, originally extracted from tabs/viewer.py.
Methods were MOVED (not copied) in Phase 3.1b+3.2b.

Layer: features/viewer (imports Qt, VTK — not for domain/infra use).
"""
from __future__ import annotations

import math

import numpy as np
import vtk
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QColorDialog
from inno3d.core.styles import SemiconductorTheme



class CrosshairMixin:
    """Mixin providing Crosshair behaviour for MultiPlanarView.

    All methods use ``self.*`` attributes owned by MultiPlanarView.
    Do not instantiate directly.
    """
    def _crosshair_display_geometry(self, orientation):
        """Project crosshair center + axis directions into VTK display pixels."""
        import math
        if self.volume_data is None:
            return None
        renderer = getattr(self, f'{orientation}_renderer', None)
        if renderer is None:
            return None

        vol_z, vol_y, vol_x = self.volume_data.shape
        if orientation == 'axial':
            cx_w = float(self.crosshair_position[0])
            cy_w = float(self.crosshair_position[1])
            theta_h = math.radians(self.oblique_angles.get('coronal', 0.0))
            theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
        elif orientation == 'coronal':
            cx_w = float(self.crosshair_position[0])
            cy_w = float((vol_z - 1) - self.crosshair_position[2])
            theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
            theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
        else:
            cx_w = float(self.crosshair_position[2])
            cy_w = float(self.crosshair_position[1])
            theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
            theta_v = math.radians(self.oblique_angles.get('coronal', 0.0)) + math.pi / 2

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        coord.SetValue(cx_w, cy_w, 0.5)
        dp = coord.GetComputedDisplayValue(renderer)
        dx, dy = float(dp[0]), float(dp[1])
        return {
            'dx': dx, 'dy': dy,
            'cos_h': math.cos(theta_h), 'sin_h': math.sin(theta_h),
            'cos_v': math.cos(theta_v), 'sin_v': math.sin(theta_v),
            'gap': 12,
        }

    # ── Adaptive 3D crosshair debounce (Phase 2a) ────────────────────────
    def _volume_data_gb(self):
        """Return volume size in GB (0.0 if no volume)."""
        if self.volume_data is None:
            return 0.0
        return self.volume_data.nbytes / (1024 ** 3)

    def _schedule_3d_crosshair_update(self):
        """Adaptively debounce 3D crosshair render based on volume size.

        - < 4 GB  → immediate (unchanged behaviour, butter-smooth on small data)
        - 4–8 GB  → 33 ms debounce (~30 fps cap — imperceptible)
        - > 8 GB  → 100 ms debounce (~10 fps cap — avoids GPU ray-cast stall)

        Only the *3D pane* render is deferred; 2D MPR planes stay real-time.
        """
        from PyQt5.QtCore import QTimer

        data_gb = self._volume_data_gb()

        if data_gb < 4.0:
            # Small volume: call immediately (no debounce)
            self.update_3d_crosshair()
            return

        delay_ms = 33 if data_gb < 8.0 else 100

        if not hasattr(self, '_3d_ch_debounce_timer'):
            self._3d_ch_debounce_timer = QTimer(self)
            self._3d_ch_debounce_timer.setSingleShot(True)
            self._3d_ch_debounce_timer.timeout.connect(self.update_3d_crosshair)

        # Restart timer on each event (coalesce rapid drags)
        self._3d_ch_debounce_timer.setInterval(delay_ms)
        if self._3d_ch_debounce_timer.isActive():
            self._3d_ch_debounce_timer.stop()
        self._3d_ch_debounce_timer.start()

    def _crosshair_drag_end(self):
        """End drag — force one final sync (sliders + all panes + 3D)."""
        # Flush any pending debounced 3D update immediately
        if getattr(self, '_3d_ch_debounce_timer', None) is not None:
            self._3d_ch_debounce_timer.stop()
        if getattr(self, "_ch_sync_timer", None) is not None:
            self._ch_sync_timer.stop()
        self._ch_sync_pending = False
        if self.volume_data is None:
            return
        x, y, z = self.crosshair_position
        self.updatePoint(x, y, z)
        if hasattr(self, "mark_mpr_interaction_end"):
            self.mark_mpr_interaction_end()


    def _crosshair_drag_update(self, orientation, pos, mode="center"):
        """Move crosshair from a pointer position.

        Always goes through ``updatePoint`` so crosshair ↔ slice sliders ↔
        linked MPR panes stay in lockstep.  Selective re-render inside
        updatePoint already avoids full rebuild when only the hair moves
        on an unchanged slice.
        """
        if hasattr(self, "mark_mpr_interaction_start"):
            self.mark_mpr_interaction_start()
        newX, newY, newZ = self._pick_voxel(orientation, pos, mode=mode)
        self.updatePoint(newX, newY, newZ)


    def _crosshair_hit_radii(self, orientation):
        """Center / line hit radii in display px — scale gently with zoom."""
        renderer = getattr(self, f"{orientation}_renderer", None)
        base_c, base_l = 16.0, 8.0
        if renderer is None or self.volume_data is None:
            return base_c, base_l
        try:
            cam = renderer.GetActiveCamera()
            cur = float(cam.GetParallelScale())
            default = self._mpr_default_parallel_scale(orientation)
            # Zoomed in → slightly larger grab; zoomed out → clamp
            zoom = default / max(cur, 1e-6)
            scale = max(0.85, min(1.55, 0.75 + 0.25 * zoom))
            return base_c * scale, base_l * scale
        except Exception:
            return base_c, base_l


    def _cursor_for_crosshair_hit(self, hit):
        if hit is None:
            return None
        mode = hit[1]
        if mode == "center":
            return Qt.SizeAllCursor
        if mode == "h":
            return Qt.SizeVerCursor  # drag H line → move vertically
        return Qt.SizeHorCursor


    def _flush_crosshair_sync(self):
        """Timer callback kept for compatibility — always full linked sync."""
        self._ch_sync_pending = False
        if self.volume_data is None:
            return
        x, y, z = self.crosshair_position
        self.updatePoint(x, y, z)


    def _pick_voxel(self, orientation, pos, mode="center"):
        """Map display pos → integer voxel (X,Y,Z) with grab-mode constraints."""
        if self.volume_data is None:
            return tuple(self.crosshair_position)
        world = self._pick_world(orientation, pos)
        vol_z, vol_y, vol_x = self.volume_data.shape
        newX, newY, newZ = self.crosshair_position

        if orientation == "axial":
            pickX = int(round(world[0]))
            pickY = int(round(world[1]))
            if mode in ("center", "v", "place"):
                newX = pickX
            if mode in ("center", "h", "place"):
                newY = pickY
        elif orientation == "coronal":
            pickX = int(round(world[0]))
            pickZ = (vol_z - 1) - int(round(world[1]))
            if mode in ("center", "v", "place"):
                newX = pickX
            if mode in ("center", "h", "place"):
                newZ = pickZ
        else:  # sagittal
            pickZ = int(round(world[0]))
            pickY = int(round(world[1]))
            if mode in ("center", "v", "place"):
                newZ = pickZ
            if mode in ("center", "h", "place"):
                newY = pickY

        newX = max(0, min(newX, vol_x - 1))
        newY = max(0, min(newY, vol_y - 1))
        newZ = max(0, min(newZ, vol_z - 1))
        return newX, newY, newZ


    def _pick_world(self, orientation, pos):
        """World coords under Qt pos (widget coords)."""
        widget = getattr(self, f"{orientation}_widget")
        renderer = getattr(self, f"{orientation}_renderer")
        x, y = pos.x(), pos.y()
        size = widget.GetRenderWindow().GetSize()
        vtk_y = size[1] - y
        picker = vtk.vtkWorldPointPicker()
        picker.Pick(float(x), float(vtk_y), 0, renderer)
        return list(picker.GetPickPosition())


    def _sync_slice_sliders_from_crosshair(self, newX, newY, newZ, vol_x, vol_y, vol_z):
        """Push crosshair indices onto the three slice sliders without feedback loops."""
        pairs = (
            ('sagittal', newX, vol_x - 1),
            ('coronal', newY, vol_y - 1),
            ('axial', newZ, vol_z - 1),
        )
        for ori, val, vmax in pairs:
            slider = getattr(self, f'{ori}_slice_slider', None)
            label = getattr(self, f'{ori}_slice_label', None)
            if slider is None:
                continue
            # Keep range correct if volume was reloaded mid-session
            if slider.maximum() != vmax and vmax >= 0:
                slider.blockSignals(True)
                slider.setMaximum(vmax)
                slider.blockSignals(False)
            if slider.value() != val:
                slider.blockSignals(True)
                slider.setValue(val)
                slider.blockSignals(False)
            if label is not None:
                label.setText(f"{val} / {vmax}")


    def choose_crosshair_color(self):
        """Pick crosshair color"""
        init_c = QColor(int(self.crosshair_color[0]*255), int(self.crosshair_color[1]*255), int(self.crosshair_color[2]*255))
        color = QColorDialog.getColor(init_c, self, "Crosshair Color")
        if color.isValid():
            self.crosshair_color = [color.redF(), color.greenF(), color.blueF()]
            self.crosshair_color_btn.setStyleSheet(f"background-color: {SemiconductorTheme.BG_LIGHT}; color: {color.name()}; font-weight: bold; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; border-radius: 3px;")
            for ori in ['axial', 'coronal', 'sagittal']:
                self.render_slice(ori, preserve_camera=True)
    

    def reset_crosshair(self):
        """Reset crosshair position to the center of the volume and angles to 0."""
        if self.volume_data is None:
            return
        vol_z, vol_y, vol_x = self.volume_data.shape
        self.oblique_angles = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
        self.updatePoint(vol_x // 2, vol_y // 2, vol_z // 2)

    # ══════════════════════════════════════════════════════════════════
    # P0 / P1 — MPR navigation (crosshair / zoom / pan)
    # ══════════════════════════════════════════════════════════════════

    def toggle_crosshair(self, state):
        """Enable/disable crosshair"""
        self.crosshair_enabled = (state == Qt.Checked)
        self._crosshair_hover = None
        self._crosshair_drag_mode = None
        self._is_dragging_crosshair = False

        if self.volume_data is not None:
            # Re-render all views
            for orientation in ['axial', 'coronal', 'sagittal']:
                self.render_slice(orientation, preserve_camera=True)
            self.update_3d_crosshair()


    def updatePoint(self, newX, newY, newZ):
        """Single source of truth: crosshair (X,Y,Z) ↔ slice indices ↔ sliders.

        Mapping (volume index order Z,Y,X):
          sagittal slider  ↔ X  ↔ crosshair_position[0]
          coronal  slider  ↔ Y  ↔ crosshair_position[1]
          axial    slider  ↔ Z  ↔ crosshair_position[2]

        Performance (unchanged from pre-P0): only ``render_slice`` when the
        underlying plane index changes; otherwise lightweight crosshair move.
        """
        if self.volume_data is None:
            return
        vol_z, vol_y, vol_x = self.volume_data.shape
        newX = max(0, min(int(round(newX)), vol_x - 1))
        newY = max(0, min(int(round(newY)), vol_y - 1))
        newZ = max(0, min(int(round(newZ)), vol_z - 1))

        oldX, oldY, oldZ = self.crosshair_position
        # No-op if nothing changed (avoids slider/render thrash)
        if (
            newX == oldX and newY == oldY and newZ == oldZ
            and self.current_slices.get('sagittal') == newX
            and self.current_slices.get('coronal') == newY
            and self.current_slices.get('axial') == newZ
        ):
            return

        # Progressive MPR: treat continuous updatePoint as interaction
        if hasattr(self, "mark_mpr_interaction_start") and getattr(
            self, "_is_dragging_crosshair", False
        ):
            self.mark_mpr_interaction_start()

        self.crosshair_position = [newX, newY, newZ]

        # Which displayed planes need new voxel data?
        changed = {
            'sagittal': newX != oldX or self.current_slices.get('sagittal') != newX,
            'coronal': newY != oldY or self.current_slices.get('coronal') != newY,
            'axial': newZ != oldZ or self.current_slices.get('axial') != newZ,
        }

        self.current_slices['sagittal'] = newX
        self.current_slices['coronal'] = newY
        self.current_slices['axial'] = newZ

        # Sync all three sliders + labels (blocked → no re-entrant update_slice)
        self._sync_slice_sliders_from_crosshair(newX, newY, newZ, vol_x, vol_y, vol_z)

        for ori in ['axial', 'coronal', 'sagittal']:
            if changed[ori]:
                self.render_slice(ori, preserve_camera=True)
            elif self.crosshair_enabled:
                self.update_2d_crosshair(ori)

        if self.crosshair_enabled:
            for ori in ['axial', 'coronal', 'sagittal']:
                self._ensure_crosshair_visible(ori)

        self._schedule_3d_crosshair_update()


    def update_2d_crosshair(self, orientation):
        """Lightweight crosshair update that reuses persistent VTK actors.
        
        Instead of calling render_slice (which destroys and rebuilds ALL
        VTK actors including the base image), this method only repositions
        the crosshair line endpoints and re-renders. This is ~10x faster
        and matches v113's smooth crosshair behavior.
        """
        if not self.crosshair_enabled or self.volume_data is None:
            return
        
        # Initialize persistent crosshair actor cache if needed
        if not hasattr(self, '_persistent_crosshair'):
            self._persistent_crosshair = {'axial': {}, 'coronal': {}, 'sagittal': {}}
        
        renderer = getattr(self, f'{orientation}_renderer')
        widget = getattr(self, f'{orientation}_widget')
        actors_dict = self._persistent_crosshair[orientation]
        
        # Initialize actors if not created yet
        if not actors_dict:
            for name in ['h_line1', 'h_line2', 'v_line1', 'v_line2']:
                ls = vtk.vtkLineSource()
                m = vtk.vtkPolyDataMapper2D()
                m.SetInputConnection(ls.GetOutputPort())
                a = vtk.vtkActor2D()
                a.SetMapper(m)
                a.GetProperty().SetLineWidth(1)
                actors_dict[name] = {'source': ls, 'actor': a}

            # 4 edge labels like 3D Teaching: +H / -H / +V / -V
            for name in ['h_pos', 'h_neg', 'v_pos', 'v_neg']:
                t = vtk.vtkTextActor()
                t.GetTextProperty().SetFontSize(11)
                t.GetTextProperty().BoldOn()
                actors_dict[name] = t

        # Remove legacy near-center labels + center handles (clutter object view)
        for legacy in ('center_fill', 'center_ring', 'h_label', 'v_label'):
            if legacy in actors_dict:
                item = actors_dict[legacy]
                act = item.get('actor') if isinstance(item, dict) else item
                if act is not None and renderer.HasViewProp(act):
                    renderer.RemoveActor(act)
                del actors_dict[legacy]

        # Migrate older caches that only had 2 labels
        for name in ['h_pos', 'h_neg', 'v_pos', 'v_neg']:
            if name not in actors_dict:
                t = vtk.vtkTextActor()
                t.GetTextProperty().SetFontSize(11)
                t.GetTextProperty().BoldOn()
                actors_dict[name] = t
        
        vol_z, vol_y, vol_x = self.volume_data.shape
        
        # Axis-based colors + signed labels (match 3D Teaching / create_crosshair_actors)
        # labels: h_pos, h_neg, v_pos, v_neg  →  +H, -H, +V, -V along axis directions
        if orientation == 'axial':
            cx_w = float(self.crosshair_position[0])
            cy_w = float(self.crosshair_position[1])
            h_pos, h_neg = '+X', '-X'
            v_pos, v_neg = '+Y', '-Y'
            h_color = (1.0, 0.192, 0.192)     # Red   = X axis
            v_color = (0.223, 1.0, 0.078)     # Green = Y axis
            h_name, v_name = 'coronal', 'sagittal'
        elif orientation == 'coronal':
            cx_w = float(self.crosshair_position[0])
            cy_w = float((vol_z - 1) - self.crosshair_position[2])
            h_pos, h_neg = '+X', '-X'
            # Display Y is flipped Z: top (higher display Y) is -Z when reverse_z is False
            if self.reverse_z:
                v_pos, v_neg = '+Z', '-Z'
            else:
                v_pos, v_neg = '-Z', '+Z'
            h_color = (1.0, 0.192, 0.192)     # Red  = X axis
            v_color = (0.121, 0.317, 1.0)     # Blue = Z axis
            h_name, v_name = 'axial', 'sagittal'
        else:  # sagittal
            cx_w = float(self.crosshair_position[2])
            cy_w = float(self.crosshair_position[1])
            if self.reverse_z:
                h_pos, h_neg = '-Z', '+Z'
            else:
                h_pos, h_neg = '+Z', '-Z'
            v_pos, v_neg = '+Y', '-Y'
            h_color = (0.121, 0.317, 1.0)     # Blue  = Z axis
            v_color = (0.223, 1.0, 0.078)     # Green = Y axis
            h_name, v_name = 'axial', 'coronal'
        
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

        # Dragonfly hover / drag highlight — only the active pane lights up
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
        
        # Configure colors + line widths (brighten on hover/grab)
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
        
        import math
        if orientation == 'axial':
            theta_h = math.radians(self.oblique_angles.get('coronal', 0.0))
            theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
        elif orientation == 'coronal':
            theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
            theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
        else:  # sagittal
            theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
            theta_v = math.radians(self.oblique_angles.get('coronal', 0.0)) + math.pi / 2
        
        cos_h, sin_h = math.cos(theta_h), math.sin(theta_h)
        cos_v, sin_v = math.cos(theta_v), math.sin(theta_v)
        line_ext = max(vp_w, vp_h) * 2.5
        
        # Update line endpoints (no new objects created)
        actors_dict['h_line1']['source'].SetPoint1(dx - line_ext * cos_h, dy - line_ext * sin_h, 0)
        actors_dict['h_line1']['source'].SetPoint2(dx - gap * cos_h, dy - gap * sin_h, 0)
        actors_dict['h_line2']['source'].SetPoint1(dx + gap * cos_h, dy + gap * sin_h, 0)
        actors_dict['h_line2']['source'].SetPoint2(dx + line_ext * cos_h, dy + line_ext * sin_h, 0)
        
        actors_dict['v_line1']['source'].SetPoint1(dx - line_ext * cos_v, dy - line_ext * sin_v, 0)
        actors_dict['v_line1']['source'].SetPoint2(dx - gap * cos_v, dy - gap * sin_v, 0)
        actors_dict['v_line2']['source'].SetPoint1(dx + gap * cos_v, dy + gap * sin_v, 0)
        actors_dict['v_line2']['source'].SetPoint2(dx + line_ext * cos_v, dy + line_ext * sin_v, 0)

        # Edge labels at viewport borders (Teaching-style: +X/-X, +Y/-Y, +Z/-Z)
        def _border_ends(cos_t, sin_t):
            t_candidates = []
            if abs(cos_t) > 1e-5:
                t = -dx / cos_t
                y = dy + t * sin_t
                if 0 <= y <= vp_h:
                    t_candidates.append((t, 0.0, y))
                t = (vp_w - dx) / cos_t
                y = dy + t * sin_t
                if 0 <= y <= vp_h:
                    t_candidates.append((t, vp_w, y))
            if abs(sin_t) > 1e-5:
                t = -dy / sin_t
                x = dx + t * cos_t
                if 0 <= x <= vp_w:
                    t_candidates.append((t, x, 0.0))
                t = (vp_h - dy) / sin_t
                x = dx + t * cos_t
                if 0 <= x <= vp_w:
                    t_candidates.append((t, x, vp_h))
            unique = []
            seen = set()
            for item in t_candidates:
                key = (round(item[0], 2), round(item[1], 2), round(item[2], 2))
                if key not in seen:
                    seen.add(key)
                    unique.append(item)
            unique.sort(key=lambda it: it[0])
            neg_pt, pos_pt = None, None
            for t, x, y in unique:
                if t < -0.1:
                    neg_pt = (x, y)
                elif t > 0.1:
                    pos_pt = (x, y)
            if neg_pt is None:
                neg_pt = (dx - 200.0 * cos_t, dy - 200.0 * sin_t)
            if pos_pt is None:
                pos_pt = (dx + 200.0 * cos_t, dy + 200.0 * sin_t)
            nx = max(5.0, min(neg_pt[0], vp_w - 40.0))
            ny = max(5.0, min(neg_pt[1], vp_h - 18.0))
            px = max(5.0, min(pos_pt[0], vp_w - 40.0))
            py = max(5.0, min(pos_pt[1], vp_h - 18.0))
            return (nx, ny), (px, py)

        h_neg_pt, h_pos_pt = _border_ends(cos_h, sin_h)
        v_neg_pt, v_pos_pt = _border_ends(cos_v, sin_v)

        # Optional oblique angle hint (same as create_crosshair_actors)
        ang_h = self.oblique_angles.get(h_name, 0.0)
        ang_v = self.oblique_angles.get(v_name, 0.0)
        h_txt_pos = f"{h_pos} ({ang_h:.1f}°)" if abs(ang_h) > 0.5 else h_pos
        h_txt_neg = f"{h_neg} ({ang_h:.1f}°)" if abs(ang_h) > 0.5 else h_neg
        v_txt_pos = f"{v_pos} ({ang_v:.1f}°)" if abs(ang_v) > 0.5 else v_pos
        v_txt_neg = f"{v_neg} ({ang_v:.1f}°)" if abs(ang_v) > 0.5 else v_neg

        for key, text, pos, color in [
            ('h_pos', h_txt_pos, h_pos_pt, h_draw),
            ('h_neg', h_txt_neg, h_neg_pt, h_draw),
            ('v_pos', v_txt_pos, v_pos_pt, v_draw),
            ('v_neg', v_txt_neg, v_neg_pt, v_draw),
        ]:
            actors_dict[key].SetInput(text)
            actors_dict[key].SetPosition(pos[0], pos[1])
            actors_dict[key].GetTextProperty().SetColor(*color)

        # Ensure all core actors are added to renderer
        for k, v in actors_dict.items():
            if k == 'handles': continue
            act = v['actor'] if isinstance(v, dict) and 'actor' in v else v
            if not renderer.HasViewProp(act):
                renderer.AddActor(act)
                
        # Clean up old handle actors
        if 'handles' in actors_dict:
            for act in actors_dict['handles']:
                renderer.RemoveActor(act)
        actors_dict['handles'] = []
        
        # Create and add new handle actors if enabled
        if getattr(self, 'oblique_handles_enabled', False):
            extent = max(vp_w, vp_h) * 0.6
            handle_dist = min(extent * 0.35, 120)
            hover = getattr(self, '_oblique_hover_handle', None)
            dragging = getattr(self, '_oblique_dragging', None)

            if orientation == 'axial':
                h_name, v_name = 'coronal', 'sagittal'
            elif orientation == 'coronal':
                h_name, v_name = 'axial', 'sagittal'
            else:
                h_name, v_name = 'axial', 'coronal'

            def make_circle2d(cx, cy, r, color, fill=False, opacity=1.0, lw=1):
                src = vtk.vtkRegularPolygonSource()
                src.SetNumberOfSides(24)
                src.SetRadius(r)
                src.SetCenter(cx, cy, 0)
                if fill:
                    src.GeneratePolygonOn()
                else:
                    src.GeneratePolygonOff()
                m = vtk.vtkPolyDataMapper2D()
                m.SetInputConnection(src.GetOutputPort())
                a = vtk.vtkActor2D()
                a.SetMapper(m)
                a.GetProperty().SetColor(*color)
                a.GetProperty().SetOpacity(opacity)
                a.GetProperty().SetLineWidth(lw)
                return a

            def make_curved_arrow_2d(cx, cy, radius, start_angle, sweep_deg, color, opacity=1.0, lw=2.0, with_arrowhead=True):
                import math as _m
                n_pts = max(12, int(abs(sweep_deg) / 3))
                pts = vtk.vtkPoints()
                lines = vtk.vtkCellArray()
                for i in range(n_pts + 1):
                    a = _m.radians(start_angle + sweep_deg * i / n_pts)
                    pts.InsertNextPoint(cx + radius * _m.cos(a), cy + radius * _m.sin(a), 0)
                line = vtk.vtkPolyLine()
                line.GetPointIds().SetNumberOfIds(n_pts + 1)
                for i in range(n_pts + 1):
                    line.GetPointIds().SetId(i, i)
                lines.InsertNextCell(line)
                if with_arrowhead:
                    end_a = _m.radians(start_angle + sweep_deg)
                    ex, ey = cx + radius * _m.cos(end_a), cy + radius * _m.sin(end_a)
                    sign = 1 if sweep_deg > 0 else -1
                    tx = -sign * _m.sin(end_a)
                    ty = sign * _m.cos(end_a)
                    arrow_len = max(8.0, radius * 0.35)
                    nx_d = _m.cos(end_a)
                    ny_d = _m.sin(end_a)
                    tip_x = ex + arrow_len * tx
                    tip_y = ey + arrow_len * ty
                    w1_x = ex + arrow_len * 0.35 * nx_d
                    w1_y = ey + arrow_len * 0.35 * ny_d
                    w2_x = ex - arrow_len * 0.35 * nx_d
                    w2_y = ey - arrow_len * 0.35 * ny_d
                    base_id = pts.GetNumberOfPoints()
                    pts.InsertNextPoint(tip_x, tip_y, 0)
                    pts.InsertNextPoint(w1_x, w1_y, 0)
                    pts.InsertNextPoint(w2_x, w2_y, 0)
                    tri_line = vtk.vtkPolyLine()
                    tri_line.GetPointIds().SetNumberOfIds(4)
                    tri_line.GetPointIds().SetId(0, base_id)
                    tri_line.GetPointIds().SetId(1, base_id + 1)
                    tri_line.GetPointIds().SetId(2, base_id + 2)
                    tri_line.GetPointIds().SetId(3, base_id)
                    lines.InsertNextCell(tri_line)
                pd = vtk.vtkPolyData()
                pd.SetPoints(pts)
                pd.SetLines(lines)
                m = vtk.vtkPolyDataMapper2D()
                m.SetInputData(pd)
                a = vtk.vtkActor2D()
                a.SetMapper(m)
                a.GetProperty().SetColor(*color)
                a.GetProperty().SetOpacity(opacity)
                a.GetProperty().SetLineWidth(lw)
                return a

            for cos_val, sin_val, color, which in [
                (cos_h, sin_h, h_color, h_name),
                (-cos_h, -sin_h, h_color, h_name),
                (cos_v, sin_v, v_color, v_name),
                (-cos_v, -sin_v, v_color, v_name),
            ]:
                hx = dx + handle_dist * cos_val
                hy = dy + handle_dist * sin_val
                is_active = (hover == (orientation, which)) or (dragging == (orientation, which))

                base_angle = math.degrees(math.atan2(sin_val, cos_val))
                arc_radius = 16
                arc_sweep = 120
                arc_start = base_angle - arc_sweep / 2 + 90

                if is_active:
                    actors_dict['handles'].append(make_circle2d(hx, hy, 22, color, fill=True, opacity=0.25))
                    actors_dict['handles'].append(make_curved_arrow_2d(hx, hy, arc_radius, arc_start, arc_sweep, color, opacity=1.0, lw=2.5))
                    actors_dict['handles'].append(make_curved_arrow_2d(hx, hy, arc_radius, arc_start + 180, arc_sweep, color, opacity=0.7, lw=2.0))
                else:
                    actors_dict['handles'].append(make_curved_arrow_2d(hx, hy, arc_radius * 0.8, arc_start, arc_sweep * 0.7, color, opacity=0.35, lw=1.5))

            for act in actors_dict['handles']:
                renderer.AddActor(act)

        widget.GetRenderWindow().Render()

    def _get_crosshair_hit(self, pos, orientation):
        """Dragonfly-style hit test: center handle, then H/V axes.

        Hit radii scale gently with zoom (P0). Returns (orientation, mode) or None.
        """
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
        center_r, line_tol = self._crosshair_hit_radii(orientation)

        dist_center = math.hypot(x - dx, vtk_y - dy)
        if dist_center <= center_r:
            return (orientation, 'center')

        def dist_to_axis(px, py, cos_t, sin_t):
            vx, vy = px - dx, py - dy
            return abs(vx * sin_t - vy * cos_t)

        dh = dist_to_axis(x, vtk_y, geom['cos_h'], geom['sin_h'])
        dv = dist_to_axis(x, vtk_y, geom['cos_v'], geom['sin_v'])

        # Prefer nearer axis when both qualify; require outside center gap
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


    def handle_crosshair_click(self, orientation, pos, mode='center'):
        """Handle mouse drag / place for crosshair (Dragonfly grab modes).

        mode:
          - 'center' / 'place': move both in-plane axes
          - 'h': move only the axis normal to the H line
          - 'v': move only the axis normal to the V line

        Always full-sync via updatePoint (crosshair + sliders + linked panes).
        """
        try:
            self._crosshair_drag_update(orientation, pos, mode=mode)
        except Exception:
            pass


    def update_3d_crosshair(self):
        """Draw RGB crosshair axes in the 3D pane.

        Independent of the volume-render toggle: lines are plain polydata and
        must still appear when 3D GPU volume is OFF (Online default).
        Geometry/actors via shared ``Crosshair3DOverlay`` (Teaching parity).
        """
        if not hasattr(self, 'view_3d_renderer') or self.volume_data is None:
            return
        if getattr(self, 'view_3d_widget', None) is None:
            return

        if not hasattr(self, '_crosshair_3d_overlay') or self._crosshair_3d_overlay is None:
            from inno3d.features.shared.crosshair_3d import Crosshair3DOverlay
            self._crosshair_3d_overlay = Crosshair3DOverlay()

        # Keep legacy attribute names for any external introspection
        if self._crosshair_3d_overlay.line_sources is None:
            self._crosshair_3d_overlay._ensure_actors()
        self.crosshair_3d_actors = self._crosshair_3d_overlay.line_sources
        self.crosshair_3d_actor_objs = self._crosshair_3d_overlay.actors

        sp = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        spx, spy, spz = float(sp[0]), float(sp[1]), float(sp[2])
        R = self.get_oblique_R() if hasattr(self, 'get_oblique_R') else None

        visible = self._crosshair_3d_overlay.update(
            self.view_3d_renderer,
            self.view_3d_widget,
            bool(self.crosshair_enabled),
            self.crosshair_position,
            self.volume_data.shape,
            (spx, spy, spz),
            R=R,
            render=False,
        )

        if visible:
            # Volume render OFF never runs render_3d camera setup — frame the
            # volume AABB so the RGB axes are visible without GPU volume.
            if not getattr(self, '_camera_initialized', False):
                try:
                    cam = self.view_3d_renderer.GetActiveCamera()
                    pmode = str(getattr(self, 'projection_mode', 'perspective')).split('#')[0].strip().lower()
                    if pmode == 'perspective':
                        cam.ParallelProjectionOff()
                        cam.SetViewAngle(40.0)
                    else:
                        cam.ParallelProjectionOn()
                    if hasattr(self, 'set_camera_preset'):
                        self.set_camera_preset("Top")
                    self._camera_initialized = True
                except Exception:
                    pass
            else:
                try:
                    self.view_3d_renderer.ResetCameraClippingRange()
                except Exception:
                    pass

        try:
            self.view_3d_widget.GetRenderWindow().Render()
        except Exception:
            pass

    def on_crosshair_slider_changed(self, axis_idx, value):
        """Update crosshair position from toolbar slider and re-render all views.
        
        axis_idx: 0=X (sagittal), 1=Y (coronal), 2=Z (axial)
        """
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
    

    def toggle_reverse_z(self):
        """Toggle Z direction (upward/downward)"""
        self.reverse_z = self.reverse_z_btn.isChecked()
        if self.volume_data is not None:
            # Re-render all views (image reloads with reversed Z mapping)
            for orientation in ['axial', 'coronal', 'sagittal']:
                self.render_slice(orientation, preserve_camera=True)
    

    def create_crosshair_actors(self, orientation, renderer, slice_shape):
        """
        Draw full-viewport crosshair in DISPLAY coordinates so it does not
        follow/zoom with the image. Crosshair center is computed by projecting
        the world crosshair position into display (screen-pixel) space.
        """
        actors = []
        try:
            h, w = slice_shape
            vol_z, vol_y, vol_x = self.volume_data.shape
            
            z_sign = '-Z' if self.reverse_z else '+Z'
            
            # Map world crosshair position to the specific plane orientations 
            # Axial: VTK X = X, VTK Y = Y
            # Coronal: VTK X = X, VTK Y = Z_flipped
            # Sagittal: VTK X = Z, VTK Y = Y
            if orientation == 'axial':
                cx_w = float(self.crosshair_position[0])
                cy_w = float(self.crosshair_position[1])
                h_label, v_label = '+X', '+Y'
                h_color = (1.0, 0.192, 0.192)  # X: Red
                v_color = (0.223, 1.0, 0.078)  # Y: Green
            elif orientation == 'coronal':
                cx_w = float(self.crosshair_position[0])
                cy_w = float((vol_z - 1) - self.crosshair_position[2])
                h_label, v_label = '+X', z_sign
                h_color = (1.0, 0.192, 0.192)  # X: Red
                v_color = (0.121, 0.317, 1.0)  # Z: Blue
            else:  # sagittal
                cx_w = float(self.crosshair_position[2])
                cy_w = float(self.crosshair_position[1])
                h_label, v_label = z_sign, '+Y'
                h_color = (0.121, 0.317, 1.0)  # Z: Blue
                v_color = (0.223, 1.0, 0.078)  # Y: Green
            
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToWorld()
            coord.SetValue(cx_w, cy_w, 0.5)
            dp = coord.GetComputedDisplayValue(renderer)
            dx = float(dp[0])   # display X (pixels from left)
            dy = float(dp[1])   # display Y (pixels from bottom)
            
            win = renderer.GetRenderWindow().GetSize()
            vp = renderer.GetViewport()
            vp_w = (vp[2] - vp[0]) * win[0]
            vp_h = (vp[3] - vp[1]) * win[1]
            gap = 12       # fixed-pixel gap at crosshair center
            
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
            
            def make_circle2d(cx, cy, r, color, fill=False, opacity=1.0, lw=1):
                src = vtk.vtkRegularPolygonSource()
                src.SetNumberOfSides(24)
                src.SetRadius(r)
                src.SetCenter(cx, cy, 0)
                if fill:
                    src.GeneratePolygonOn()
                else:
                    src.GeneratePolygonOff()
                m = vtk.vtkPolyDataMapper2D()
                m.SetInputConnection(src.GetOutputPort())
                a = vtk.vtkActor2D()
                a.SetMapper(m)
                a.GetProperty().SetColor(*color)
                a.GetProperty().SetOpacity(opacity)
                a.GetProperty().SetLineWidth(lw)
                return a
            
            import math
            # Define labels, colors, and angles for each view orientation
            if orientation == 'axial':
                h_name, v_name = 'coronal', 'sagittal'
                h_color = (1.0, 0.192, 0.192)  # X: Red
                v_color = (0.223, 1.0, 0.078)  # Y: Green
                
                h_label_pos, h_label_neg = '+X', '-X'
                v_label_pos, v_label_neg = '-Y', '+Y'
                
                theta_h = math.radians(self.oblique_angles.get('coronal', 0.0))
                theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
                
            elif orientation == 'coronal':
                h_name, v_name = 'axial', 'sagittal'
                h_color = (1.0, 0.192, 0.192)  # X: Red
                v_color = (0.121, 0.317, 1.0)  # Z: Blue
                
                h_label_pos, h_label_neg = '+X', '-X'
                v_label_pos = '+Z' if self.reverse_z else '-Z'
                v_label_neg = '-Z' if self.reverse_z else '+Z'
                
                theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
                theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
                
            else:  # sagittal
                h_name, v_name = 'axial', 'coronal'
                h_color = (0.121, 0.317, 1.0)  # Z: Blue
                v_color = (0.223, 1.0, 0.078)  # Y: Green
                
                h_label_pos = '-Z' if self.reverse_z else '+Z'
                h_label_neg = '+Z' if self.reverse_z else '-Z'
                v_label_pos = '-Y'
                v_label_neg = '+Y'
                
                theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
                theta_v = math.radians(self.oblique_angles.get('coronal', 0.0)) + math.pi / 2

            # Compute line directions
            cos_h, sin_h = math.cos(theta_h), math.sin(theta_h)
            cos_v, sin_v = math.cos(theta_v), math.sin(theta_v)
            extent = max(vp_w, vp_h) * 0.6
            line_extent = max(vp_w, vp_h) * 2.5

            # Draw horizontal-ish line
            actors.append(make_line2d(dx - line_extent * cos_h, dy - line_extent * sin_h, dx - gap * cos_h, dy - gap * sin_h, h_color))
            actors.append(make_line2d(dx + gap * cos_h, dy + gap * sin_h, dx + line_extent * cos_h, dy + line_extent * sin_h, h_color))

            # Draw vertical-ish line
            actors.append(make_line2d(dx - line_extent * cos_v, dy - line_extent * sin_v, dx - gap * cos_v, dy - gap * sin_v, v_color))
            actors.append(make_line2d(dx + gap * cos_v, dy + gap * sin_v, dx + line_extent * cos_v, dy + line_extent * sin_v, v_color))

            # Rotation handles — Dragonfly-style curved arrows
            # Only draw when oblique handles are enabled
            if getattr(self, 'oblique_handles_enabled', False):
                handle_dist = min(extent * 0.35, 120)
                hover = getattr(self, '_oblique_hover_handle', None)
                dragging = getattr(self, '_oblique_dragging', None)

                def make_curved_arrow_2d(cx, cy, radius, start_angle, sweep_deg, color, opacity=1.0, lw=2.0, with_arrowhead=True):
                    """Create a curved arc with an arrowhead at the end (Dragonfly rotation handle)."""
                    import math as _m
                    n_pts = max(12, int(abs(sweep_deg) / 3))
                    pts = vtk.vtkPoints()
                    lines = vtk.vtkCellArray()
                    for i in range(n_pts + 1):
                        a = _m.radians(start_angle + sweep_deg * i / n_pts)
                        pts.InsertNextPoint(cx + radius * _m.cos(a), cy + radius * _m.sin(a), 0)
                    line = vtk.vtkPolyLine()
                    line.GetPointIds().SetNumberOfIds(n_pts + 1)
                    for i in range(n_pts + 1):
                        line.GetPointIds().SetId(i, i)
                    lines.InsertNextCell(line)
                    # Arrowhead triangle at the end of the arc
                    if with_arrowhead:
                        end_a = _m.radians(start_angle + sweep_deg)
                        ex, ey = cx + radius * _m.cos(end_a), cy + radius * _m.sin(end_a)
                        # Tangent direction at end (perpendicular to radius)
                        sign = 1 if sweep_deg > 0 else -1
                        tx = -sign * _m.sin(end_a)
                        ty = sign * _m.cos(end_a)
                        arrow_len = max(8, radius * 0.35)
                        # Normal (outward from center)
                        nx_d = _m.cos(end_a)
                        ny_d = _m.sin(end_a)
                        # Triangle points: tip along tangent, wings along normal
                        tip_x = ex + arrow_len * tx
                        tip_y = ey + arrow_len * ty
                        w1_x = ex + arrow_len * 0.35 * nx_d
                        w1_y = ey + arrow_len * 0.35 * ny_d
                        w2_x = ex - arrow_len * 0.35 * nx_d
                        w2_y = ey - arrow_len * 0.35 * ny_d
                        base_id = pts.GetNumberOfPoints()
                        pts.InsertNextPoint(tip_x, tip_y, 0)
                        pts.InsertNextPoint(w1_x, w1_y, 0)
                        pts.InsertNextPoint(w2_x, w2_y, 0)
                        tri_line = vtk.vtkPolyLine()
                        tri_line.GetPointIds().SetNumberOfIds(4)
                        tri_line.GetPointIds().SetId(0, base_id)
                        tri_line.GetPointIds().SetId(1, base_id + 1)
                        tri_line.GetPointIds().SetId(2, base_id + 2)
                        tri_line.GetPointIds().SetId(3, base_id)
                        lines.InsertNextCell(tri_line)
                    pd = vtk.vtkPolyData()
                    pd.SetPoints(pts)
                    pd.SetLines(lines)
                    m = vtk.vtkPolyDataMapper2D()
                    m.SetInputData(pd)
                    a = vtk.vtkActor2D()
                    a.SetMapper(m)
                    a.GetProperty().SetColor(*color)
                    a.GetProperty().SetOpacity(opacity)
                    a.GetProperty().SetLineWidth(lw)
                    return a

                for cos_val, sin_val, color, which in [
                    (cos_h, sin_h, h_color, h_name),
                    (-cos_h, -sin_h, h_color, h_name),
                    (cos_v, sin_v, v_color, v_name),
                    (-cos_v, -sin_v, v_color, v_name),
                ]:
                    hx = dx + handle_dist * cos_val
                    hy = dy + handle_dist * sin_val
                    is_active = (hover == (orientation, which)) or (dragging == (orientation, which))

                    # Base angle of this handle position relative to crosshair center
                    base_angle = math.degrees(math.atan2(sin_val, cos_val))
                    arc_radius = 16
                    arc_sweep = 120  # degrees of arc
                    arc_start = base_angle - arc_sweep / 2 + 90  # perpendicular to radial direction

                    if is_active:
                        # Glow background circle
                        actors.append(make_circle2d(hx, hy, 22, color, fill=True, opacity=0.25))
                        # Bright curved arrow
                        actors.append(make_curved_arrow_2d(hx, hy, arc_radius, arc_start, arc_sweep, color, opacity=1.0, lw=2.5))
                        # Counter-direction arrow for bidirectional hint
                        actors.append(make_curved_arrow_2d(hx, hy, arc_radius, arc_start + 180, arc_sweep, color, opacity=0.7, lw=2.0))
                    else:
                        # Dim curved arrow — subtle but visible
                        actors.append(make_curved_arrow_2d(hx, hy, arc_radius * 0.8, arc_start, arc_sweep * 0.7, color, opacity=0.35, lw=1.5))

            # Angle labels — axis letter + oblique angle
            ang_h = self.oblique_angles.get(h_name, 0.0)
            ang_v = self.oblique_angles.get(v_name, 0.0)
            
            h_txt_pos = f"{h_label_pos} ({ang_h:.1f}°)" if abs(ang_h) > 0.5 else h_label_pos
            h_txt_neg = f"{h_label_neg} ({ang_h:.1f}°)" if abs(ang_h) > 0.5 else h_label_neg
            v_txt_pos = f"{v_label_pos} ({ang_v:.1f}°)" if abs(ang_v) > 0.5 else v_label_pos
            v_txt_neg = f"{v_label_neg} ({ang_v:.1f}°)" if abs(ang_v) > 0.5 else v_label_neg

            # Calculate boundary intersections for rotated/oblique lines
            def get_border_intersection(dx, dy, cos_val, sin_val, vp_w, vp_h):
                t_candidates = []
                if abs(cos_val) > 1e-5:
                    # Left border (x = 0)
                    t = -dx / cos_val
                    y = dy + t * sin_val
                    if 0 <= y <= vp_h:
                        t_candidates.append((t, 0.0, y))
                    # Right border (x = vp_w)
                    t = (vp_w - dx) / cos_val
                    y = dy + t * sin_val
                    if 0 <= y <= vp_h:
                        t_candidates.append((t, vp_w, y))
                if abs(sin_val) > 1e-5:
                    # Bottom border (y = 0)
                    t = -dy / sin_val
                    x = dx + t * cos_val
                    if 0 <= x <= vp_w:
                        t_candidates.append((t, x, 0.0))
                    # Top border (y = vp_h)
                    t = (vp_h - dy) / sin_val
                    x = dx + t * cos_val
                    if 0 <= x <= vp_w:
                        t_candidates.append((t, x, vp_h))
                
                unique = []
                seen = set()
                for item in t_candidates:
                    rounded = (round(item[0], 2), round(item[1], 2), round(item[2], 2))
                    if rounded not in seen:
                        seen.add(rounded)
                        unique.append(item)
                unique.sort(key=lambda item: item[0])
                
                neg_pt, pos_pt = None, None
                for t, x, y in unique:
                    if t < -0.1:
                        neg_pt = (x, y)
                    elif t > 0.1:
                        pos_pt = (x, y)
                        
                if neg_pt is None:
                    neg_pt = (dx - 200.0 * cos_val, dy - 200.0 * sin_val)
                if pos_pt is None:
                    pos_pt = (dx + 200.0 * cos_val, dy + 200.0 * sin_val)
                    
                nx = max(5.0, min(neg_pt[0], vp_w - 60.0))
                ny = max(5.0, min(neg_pt[1], vp_h - 20.0))
                px = max(5.0, min(pos_pt[0], vp_w - 60.0))
                py = max(5.0, min(pos_pt[1], vp_h - 20.0))
                return (nx, ny), (px, py)

            h_neg_pt, h_pos_pt = get_border_intersection(dx, dy, cos_h, sin_h, vp_w, vp_h)
            v_neg_pt, v_pos_pt = get_border_intersection(dx, dy, cos_v, sin_v, vp_w, vp_h)

            actors.append(make_text2d(h_txt_neg, h_neg_pt[0], h_neg_pt[1], h_color))
            actors.append(make_text2d(h_txt_pos, h_pos_pt[0], h_pos_pt[1], h_color))
            actors.append(make_text2d(v_txt_neg, v_neg_pt[0], v_neg_pt[1], v_color))
            actors.append(make_text2d(v_txt_pos, v_pos_pt[0], v_pos_pt[1], v_color))
            
        except Exception:
            pass
        return actors
