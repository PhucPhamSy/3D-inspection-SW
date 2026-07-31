# inno3d/features/teaching/seg_mpr.py
# -----------------------------------------------------------------------
# SegmentationMPRMixin — extracted from inno3d/tabs/teaching.py (Phase 4.3)
#
# MPR slice rendering, crosshair, updatePoint, eventFilter,
# VTK observers, zoom/pan, rulers, overlay labels.
# -----------------------------------------------------------------------

import math
import traceback
from pathlib import Path as _Path

import numpy as np
import vtk
from vtk.util import numpy_support

from PyQt5.QtCore import Qt, QEvent, QPoint, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QCursor, QIcon
from PyQt5.QtWidgets import QColorDialog, QWidget
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from inno3d.core.styles import SemiconductorTheme
from inno3d.core.view_support import (
    apply_dragonfly_volume_zoom,
    push_camera_outside_aabb,
    volume_world_aabb,
)


class SegmentationMPRMixin:
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

        # --- Layer Define Mode (priority over crosshair on sagittal widget) ---
        if (getattr(self, 'layer_define_active', False)
                and hasattr(self, 'sagittal_widget')
                and obj == self.sagittal_widget
                and self.volume_data is not None):
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._handle_layer_define_mouse_press(event.pos())
                return True
            elif event.type() == QEvent.MouseMove:
                self._handle_layer_define_mouse_move(
                    event.pos(), is_drag=bool(event.buttons() & Qt.LeftButton))
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._handle_layer_define_mouse_release(event.pos())
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
                # Ctrl+LMB → sticky multi MES pick (shared with Viewer Multi ON)
                if event.modifiers() & Qt.ControlModifier:
                    if hasattr(self, "select_mes_object_from_mpr"):
                        try:
                            self.select_mes_object_from_mpr(orientation, event.pos())
                        except Exception as e:
                            print(f"[MES pick] Teaching Ctrl+click failed: {e}")
                        return True

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
                px = int(world_pos[0])
                # Un-flip: coronal rendering uses np.flipud so VTK Y is inverted Z
                pz = (vol_z - 1) - int(world_pos[1])
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

    # ── Adaptive 3D crosshair debounce (mirrors Viewer Phase 2a) ─────
    def _schedule_3d_crosshair_update_teaching(self):
        """Adaptively debounce 3D crosshair render for Teaching tab.

        - < 4 GB  → immediate (no change from current behaviour)
        - 4–8 GB  → 33 ms debounce
        - > 8 GB  → 100 ms debounce
        """
        from PyQt5.QtCore import QTimer

        data_gb = (self.volume_data.nbytes / (1024 ** 3)) if self.volume_data is not None else 0.0

        if data_gb < 4.0:
            self.update_3d_crosshair()
            return

        delay_ms = 33 if data_gb < 8.0 else 100

        if not hasattr(self, '_3d_ch_debounce_timer_t'):
            self._3d_ch_debounce_timer_t = QTimer(self)
            self._3d_ch_debounce_timer_t.setSingleShot(True)
            self._3d_ch_debounce_timer_t.timeout.connect(self.update_3d_crosshair)

        self._3d_ch_debounce_timer_t.setInterval(delay_ms)
        if self._3d_ch_debounce_timer_t.isActive():
            self._3d_ch_debounce_timer_t.stop()
        self._3d_ch_debounce_timer_t.start()

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

        # RGB axes on Teaching 3D volume (Viewer parity)
        # ── Adaptive debounce: skip expensive 3D render for large volumes ──
        self._schedule_3d_crosshair_update_teaching()

    def update_3d_crosshair(self, render=True):
        """Draw RGB crosshair axes in the Teaching 3D Volume pane.

        Mirrors Viewer ``CrosshairMixin.update_3d_crosshair`` via shared
        ``Crosshair3DOverlay``. No-op when 3D view is off or volume missing.
        Teaching MPR is axis-aligned (no oblique R).
        """
        if not getattr(self, "_3d_view_active", False):
            return
        ren = getattr(self, "_3d_renderer", None)
        widget = getattr(self, "_3d_vtk_widget", None)
        if ren is None or widget is None or self.volume_data is None:
            return

        if not hasattr(self, "_crosshair_3d_overlay") or self._crosshair_3d_overlay is None:
            from inno3d.features.shared.crosshair_3d import Crosshair3DOverlay
            self._crosshair_3d_overlay = Crosshair3DOverlay()

        # Prefer live 3D world spacing (synced from Viewer when available)
        if hasattr(self, "_get_3d_world_spacing"):
            try:
                sp = self._get_3d_world_spacing()
            except Exception:
                sp = None
        else:
            sp = None
        if sp is None:
            sp = getattr(self, "custom_spacing", None) or getattr(
                self, "spacing", (1.0, 1.0, 1.0)
            )
        spx, spy, spz = float(sp[0]), float(sp[1]), float(sp[2])

        pos = getattr(self, "crosshair_position", None)
        enabled = bool(getattr(self, "crosshair_enabled", False))
        try:
            self._crosshair_3d_overlay.update(
                ren,
                widget,
                enabled,
                pos,
                self.volume_data.shape,
                (spx, spy, spz),
                R=None,  # Teaching: axis-aligned
                render=bool(render),
            )
        except Exception as e:
            print(f"[TEACHING 3D] crosshair update skipped: {e}")

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

    def _mpr_pick_volume_xyz(self, orientation, pos):
        """Map Qt pos on Teaching MPR pane → volume voxel (X, Y, Z).

        Matches crosshair place axes; fixed axis from current slice/crosshair.
        Used by Ctrl+click MES pick (shared label path).
        """
        if self.volume_data is None:
            return None
        try:
            widget = getattr(self, f"{orientation}_widget")
            renderer = getattr(self, f"{orientation}_renderer")
            x, y = pos.x(), pos.y()
            size = widget.GetRenderWindow().GetSize()
            vtk_y = size[1] - y
            picker = vtk.vtkWorldPointPicker()
            picker.Pick(x, vtk_y, 0, renderer)
            world_pos = picker.GetPickPosition()

            vol_z, vol_y, vol_x = self.volume_data.shape
            ch = list(getattr(self, "crosshair_position", [0, 0, 0]) or [0, 0, 0])
            slices = getattr(self, "current_slices", None) or {}

            if orientation == "axial":
                px = int(round(world_pos[0]))
                py = int(round(world_pos[1]))
                pz = int(slices.get("axial", ch[2] if len(ch) > 2 else 0))
                if getattr(self, "reverse_z", False):
                    pz = vol_z - 1 - pz
            elif orientation == "coronal":
                px = int(round(world_pos[0]))
                pz = (vol_z - 1) - int(round(world_pos[1]))
                py = int(slices.get("coronal", ch[1] if len(ch) > 1 else 0))
            else:  # sagittal
                pz = int(round(world_pos[0]))
                py = int(round(world_pos[1]))
                px = int(slices.get("sagittal", ch[0] if len(ch) > 0 else 0))

            px = max(0, min(int(px), vol_x - 1))
            py = max(0, min(int(py), vol_y - 1))
            pz = max(0, min(int(pz), vol_z - 1))
            return px, py, pz
        except Exception as e:
            print(f"[MES pick] Teaching _mpr_pick_volume_xyz failed: {e}")
            return None

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
            # Show/hide RGB axes on Teaching 3D volume
            self.update_3d_crosshair()

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

