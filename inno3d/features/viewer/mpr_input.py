"""MPR input handling mixin — Phase 3.3a extract from tabs/viewer.py.

Contains the two main input dispatch methods:
  - ``setup_vtk_observers`` — wires VTK interactor callbacks (LMB, move, release,
    scroll) for each MPR pane.
  - ``eventFilter`` — Qt-level event filter for all panes (3D + MPR), handling
    mouse, wheel, keyboard, double-click events.

Layer: features/viewer (imports Qt, VTK — not for domain/infra use).
"""
from __future__ import annotations

import vtk
from PyQt5.QtCore import Qt, QEvent, QPoint
from PyQt5.QtWidgets import QApplication


class MprInputMixin:
    """Mixin providing mouse/keyboard input dispatch for MultiPlanarView.

    All methods use ``self.*`` attributes owned by MultiPlanarView.
    Do not instantiate directly.
    """

    def setup_vtk_observers(self, vtk_widget, orientation):
        interactor = vtk_widget.GetRenderWindow().GetInteractor()
        
        if not hasattr(self, 'active_styles'):
            self.active_styles = {}
        if not hasattr(self, '_dummy_style'):
            self._dummy_style = vtk.vtkInteractorStyle()
            
        def on_left_button_press(caller, event):
            if self.volume_data is None:
                return

            x, y = caller.GetEventPosition()
            size = vtk_widget.GetRenderWindow().GetSize()
            qt_y = size[1] - y
            qt_pos = QPoint(x, qt_y)

            # 1) Dragonfly clip-box handles (priority over crosshair when CLIP BOX on)
            if getattr(self, '_df_clip_enabled', False):
                clip_hit = self._get_df_clip_hit(qt_pos, orientation)
                if clip_hit is not None:
                    if orientation not in self.active_styles:
                        self.active_styles[orientation] = caller.GetInteractorStyle()
                    caller.SetInteractorStyle(self._dummy_style)
                    self._df_clip_begin_drag(orientation, qt_pos, clip_hit)
                    return

            if not self.crosshair_enabled and not getattr(self, 'rotate_mode', False):
                return

            if not self.crosshair_enabled:
                # rotate_mode only — block default VTK interaction
                if orientation not in self.active_styles:
                    self.active_styles[orientation] = caller.GetInteractorStyle()
                caller.SetInteractorStyle(self._dummy_style)
                self.last_mouse_pos = qt_pos
                return

            # Dragonfly: only grab when near center / axis (or oblique handle)
            if getattr(self, '_oblique_hover_handle', None) and getattr(self, '_oblique_hover_handle')[0] == orientation:
                if orientation not in self.active_styles:
                    self.active_styles[orientation] = caller.GetInteractorStyle()
                caller.SetInteractorStyle(self._dummy_style)
                self._oblique_dragging = self._oblique_hover_handle
                return

            ch_hit = self._get_crosshair_hit(qt_pos, orientation)
            if orientation not in self.active_styles:
                self.active_styles[orientation] = caller.GetInteractorStyle()
            caller.SetInteractorStyle(self._dummy_style)
            if ch_hit is None:
                # Click-to-place on empty space
                self._crosshair_drag_mode = 'place'
                self._crosshair_hover = (orientation, 'center')
                self._is_dragging_crosshair = True
                self.handle_crosshair_click(orientation, qt_pos, mode='place')
                return

            self._crosshair_drag_mode = ch_hit[1]
            self._crosshair_hover = ch_hit
            self._is_dragging_crosshair = True
            self.handle_crosshair_click(orientation, qt_pos, mode=ch_hit[1])
            self.update_2d_crosshair(orientation)

        def on_mouse_move(caller, event):
            if self.volume_data is None:
                return

            x, y = caller.GetEventPosition()
            size = vtk_widget.GetRenderWindow().GetSize()
            qt_y = size[1] - y
            qt_pos = QPoint(x, qt_y)

            self.update_pixel_value(orientation, qt_pos)

            # Active clip-box drag
            if getattr(self, '_df_clip_drag', None):
                self._df_clip_handle_drag(qt_pos, orientation)
                return

            # Hover: clip handles first when CLIP BOX is on
            if getattr(self, '_df_clip_enabled', False) and not (getattr(self, '_is_dragging_crosshair', False)
                    or getattr(self, '_oblique_dragging', None)):
                clip_hit = self._get_df_clip_hit(qt_pos, orientation)
                prev = getattr(self, '_df_clip_hover', None)
                if self._df_clip_hit_key(clip_hit) != self._df_clip_hit_key(prev):
                    self._df_clip_hover = clip_hit
                    self._df_clip_update_mpr_overlay(orientation)
                    vtk_widget.GetRenderWindow().Render()
                if clip_hit is not None:
                    vtk_widget.setCursor(self._df_clip_cursor_for_hit(clip_hit))
                    return
                elif prev is not None:
                    self._df_clip_hover = None
                    self._df_clip_update_mpr_overlay(orientation)
                    vtk_widget.GetRenderWindow().Render()

            if not self.crosshair_enabled:
                return

            if getattr(self, '_oblique_dragging', None):
                self._handle_oblique_rotate(qt_pos, orientation)
                return

            if getattr(self, '_is_dragging_crosshair', False):
                mode = getattr(self, '_crosshair_drag_mode', 'center') or 'center'
                self.handle_crosshair_click(
                    orientation, qt_pos, mode=mode
                )
                return

            # Hover priority: oblique handles first, then crosshair center/lines
            obl_hit = self._get_oblique_hit(qt_pos, orientation)
            if obl_hit != getattr(self, '_oblique_hover_handle', None):
                self._oblique_hover_handle = obl_hit
                if obl_hit:
                    self._crosshair_hover = None
                    self.render_slice(orientation, preserve_camera=True)
                    vtk_widget.setCursor(Qt.SizeAllCursor)
                    return
                self.render_slice(orientation, preserve_camera=True)

            if obl_hit:
                vtk_widget.setCursor(Qt.SizeAllCursor)
                return

            ch_hit = self._get_crosshair_hit(qt_pos, orientation)
            if ch_hit != getattr(self, '_crosshair_hover', None):
                self._crosshair_hover = ch_hit
                self.update_2d_crosshair(orientation)
            cur = self._cursor_for_crosshair_hit(ch_hit)
            if cur is not None:
                vtk_widget.setCursor(cur)
            else:
                vtk_widget.setCursor(Qt.OpenHandCursor)

        def on_left_button_release(caller, event):
            if getattr(self, '_df_clip_drag', None):
                self._df_clip_end_drag()
                x, y = caller.GetEventPosition()
                size = vtk_widget.GetRenderWindow().GetSize()
                qt_y = size[1] - y
                qt_pos = QPoint(x, qt_y)
                hit = self._get_df_clip_hit(qt_pos, orientation) if getattr(self, '_df_clip_enabled', False) else None
                self._df_clip_hover = hit
                if hit is not None:
                    vtk_widget.setCursor(self._df_clip_cursor_for_hit(hit))
                else:
                    vtk_widget.unsetCursor()
                self._restore_interactor_style(orientation)
                return

            if getattr(self, '_is_dragging_crosshair', False):
                self._is_dragging_crosshair = False
                self._crosshair_drag_mode = None
                self._crosshair_drag_end()

            if getattr(self, '_oblique_dragging', None):
                self._oblique_dragging = None

                # Full-quality render of all views on release
                self._oblique_drag_finish()

                x, y = caller.GetEventPosition()
                size = vtk_widget.GetRenderWindow().GetSize()
                qt_y = size[1] - y
                qt_pos = QPoint(x, qt_y)

                hit = self._get_oblique_hit(qt_pos, orientation)
                self._oblique_hover_handle = hit
                if hit:
                    vtk_widget.setCursor(Qt.SizeAllCursor)
                else:
                    vtk_widget.unsetCursor()

            # Restore the active style (centralized helper — idempotent)
            self._restore_interactor_style(orientation)

        interactor.AddObserver("LeftButtonPressEvent", on_left_button_press, 10.0)
        interactor.AddObserver("MouseMoveEvent", on_mouse_move, 10.0)
        interactor.AddObserver("LeftButtonReleaseEvent", on_left_button_release, 10.0)

        # ── Fiji-style scroll wheel zoom at cursor position ──────────────
        # These observers fire at high priority (100.0) so they intercept
        # the wheel event BEFORE the interactor style (including dummy style
        # during crosshair mode) can consume or block it.
        def on_mouse_wheel_forward(caller, event):
            if self.volume_data is None:
                return
            # Qt eventFilter is primary; this is a backup path
            if caller.GetControlKey():
                self._scroll_change_slice(orientation, delta=+1)
            else:
                self._fiji_zoom_at_cursor(
                    vtk_widget, orientation, zoom_in=True, strength=1.0
                )
            caller.SetEventInformation(
                caller.GetEventPosition()[0], caller.GetEventPosition()[1],
                caller.GetControlKey(), caller.GetShiftKey())

        def on_mouse_wheel_backward(caller, event):
            if self.volume_data is None:
                return
            if caller.GetControlKey():
                self._scroll_change_slice(orientation, delta=-1)
            else:
                self._fiji_zoom_at_cursor(
                    vtk_widget, orientation, zoom_in=False, strength=1.0
                )
            caller.SetEventInformation(
                caller.GetEventPosition()[0], caller.GetEventPosition()[1],
                caller.GetControlKey(), caller.GetShiftKey())

        interactor.AddObserver("MouseWheelForwardEvent", on_mouse_wheel_forward, 100.0)
        interactor.AddObserver("MouseWheelBackwardEvent", on_mouse_wheel_backward, 100.0)

        def on_interaction(caller, event):
            if self.volume_data is None:
                return
            renderer = getattr(self, f'{orientation}_renderer')
            z, y, x = self.volume_data.shape
            h, w = (y, x) if orientation == 'axial' else (z, x) if orientation == 'coronal' else (z, y)
            self.add_ruler_overlay(renderer, orientation, (h, w))
            vtk_widget.GetRenderWindow().Render()

        interactor.AddObserver("InteractionEvent", on_interaction, 10.0)
        interactor.AddObserver("EndInteractionEvent", on_interaction, 10.0)

    # ── Fiji-style zoom helpers ──────────────────────────────────────────

    def eventFilter(self, obj, event):
        # 3D pane / container: focus ring + authoritative scroll-zoom
        is_3d = (
            obj is getattr(self, 'view_3d_widget', None)
            or obj is getattr(self, '_vtk_3d_container', None)
        )
        if is_3d:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._set_active_grid_pane('view_3d')
                if obj is getattr(self, 'view_3d_widget', None):
                    try:
                        self.view_3d_widget.setFocus(Qt.MouseFocusReason)
                    except Exception:
                        pass
                # Qt backup path for clip-face grab (if VTK style misses the press)
                if getattr(self, '_df_clip_enabled', False) and self.volume_data is not None:
                    if self._df_clip_3d_try_grab_at_qt_pos(event.pos()):
                        return True
            elif event.type() == QEvent.MouseMove:
                if getattr(self, '_df_clip_3d_drag', None) and (event.buttons() & Qt.LeftButton):
                    self._df_clip_3d_drag_at_qt_pos(event.pos())
                    return True
                if (
                    getattr(self, '_df_clip_enabled', False)
                    and self.volume_data is not None
                    and not (event.buttons() & Qt.LeftButton)
                ):
                    self._df_clip_3d_hover_at_qt_pos(event.pos())
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if getattr(self, '_df_clip_3d_drag', None):
                    self._df_clip_3d_on_left_up()
                    style = getattr(self, '_3d_interactor_style', None)
                    if style is not None:
                        style._df_clip_dragging = False
                    return True
            elif event.type() == QEvent.Wheel:
                if self.volume_data is None:
                    return super().eventFilter(obj, event)
                # Prefer pixelDelta (trackpad / high-res) for continuous feel;
                # fall back to angleDelta (classic mouse, usually ±120 per notch).
                pixel = event.pixelDelta().y()
                angle = event.angleDelta().y()
                if angle == 0:
                    angle = event.angleDelta().x()
                if pixel == 0 and angle == 0:
                    pixel = event.pixelDelta().x()
                if pixel == 0 and angle == 0:
                    return super().eventFilter(obj, event)

                if pixel != 0:
                    # ~40 px ≈ one mild notch; sign = zoom direction
                    strength = abs(pixel) / 40.0
                    zoom_in = pixel > 0
                else:
                    # One wheel notch (120°) → strength 1.0
                    strength = abs(angle) / 120.0
                    zoom_in = angle > 0

                # Clamp so a single event never jumps wildly
                strength = max(0.15, min(strength, 3.0))
                # 3D: dolly about orbit pivot only (no zoom-to-cursor pan).
                # Cursor-based shift was re-enabled briefly and broke Track/pan feel.
                self._dragonfly_3d_zoom(zoom_in=zoom_in, strength=strength, display_xy=None)
                return True
            return super().eventFilter(obj, event)


        # Handle mouse events for pixel value and crosshair
        if obj in [self.axial_widget, self.coronal_widget, self.sagittal_widget]:
            orientation = None
            for ori in ['axial', 'coronal', 'sagittal']:
                if obj == getattr(self, f'{ori}_widget'):
                    orientation = ori
                    break

            if event.type() == QEvent.MouseMove:
                if self.volume_data is not None:
                    self.update_pixel_value(orientation, event.pos())
                    # MPR pan (middle or Shift+LMB)
                    if getattr(self, "_mpr_pan", None) and (
                        (event.buttons() & Qt.MiddleButton)
                        or ((event.buttons() & Qt.LeftButton) and (event.modifiers() & Qt.ShiftModifier))
                        or ((event.buttons() & Qt.LeftButton) and getattr(self, "_mpr_pan", None))
                    ):
                        self._mpr_pan_move(orientation, event.pos())
                        return True
                    # Clip-box drag (Dragonfly handles on MPR)
                    if getattr(self, '_df_clip_drag', None) and (event.buttons() & Qt.LeftButton):
                        self._df_clip_handle_drag(event.pos(), orientation)
                        return True
                    # Oblique rotation drag
                    if getattr(self, '_oblique_dragging', None) and self.crosshair_enabled and (event.buttons() & Qt.LeftButton):
                        self._handle_oblique_rotate(event.pos(), orientation)
                        return True
                    # Drag to rotate camera roll
                    elif getattr(self, 'rotate_mode', False) and (event.buttons() & Qt.LeftButton):
                        self.handle_rotation_drag(orientation, event.pos())
                    # Drag crosshair only while actively grabbing center/axis
                    elif self.crosshair_enabled and getattr(self, '_is_dragging_crosshair', False) and (event.buttons() & Qt.LeftButton):
                        mode = getattr(self, '_crosshair_drag_mode', 'center') or 'center'
                        self.handle_crosshair_click(
                            orientation, event.pos(), mode=mode
                        )
                        return True

                    # Hover: clip box → oblique → crosshair → open-hand (pan available)
                    if not (event.buttons() & (Qt.LeftButton | Qt.MiddleButton)):
                        widget = getattr(self, f'{orientation}_widget')
                        if getattr(self, '_df_clip_enabled', False):
                            clip_hit = self._get_df_clip_hit(event.pos(), orientation)
                            prev = getattr(self, '_df_clip_hover', None)
                            if self._df_clip_hit_key(clip_hit) != self._df_clip_hit_key(prev):
                                self._df_clip_hover = clip_hit
                                self._df_clip_update_mpr_overlay(orientation)
                                widget.GetRenderWindow().Render()
                            if clip_hit is not None:
                                widget.setCursor(self._df_clip_cursor_for_hit(clip_hit))
                                return False
                            elif prev is not None:
                                self._df_clip_hover = None
                                self._df_clip_update_mpr_overlay(orientation)
                                widget.GetRenderWindow().Render()
                        if self.crosshair_enabled:
                            obl_hit = self._get_oblique_hit(event.pos(), orientation)
                            if obl_hit != getattr(self, '_oblique_hover_handle', None):
                                self._oblique_hover_handle = obl_hit
                                if obl_hit:
                                    self._crosshair_hover = None
                                self.render_slice(orientation, preserve_camera=True)
                            if obl_hit:
                                widget.setCursor(Qt.SizeAllCursor)
                            else:
                                ch_hit = self._get_crosshair_hit(event.pos(), orientation)
                                if ch_hit != getattr(self, '_crosshair_hover', None):
                                    self._crosshair_hover = ch_hit
                                    self.update_2d_crosshair(orientation)
                                cur = self._cursor_for_crosshair_hit(ch_hit)
                                if cur is not None:
                                    widget.setCursor(cur)
                                else:
                                    # Empty space: pan available (middle / shift+lmb)
                                    widget.setCursor(Qt.OpenHandCursor)
                        else:
                            widget.setCursor(Qt.OpenHandCursor)
            
            elif event.type() == QEvent.MouseButtonPress:
                # Middle button → pan
                if event.button() == Qt.MiddleButton and self.volume_data is not None:
                    self._set_active_grid_pane(orientation)
                    self._mpr_pan_begin(orientation, event.pos())
                    return True
                if event.button() == Qt.LeftButton:
                    # Dragonfly-style: focus ring on the pane being interacted with
                    self._set_active_grid_pane(orientation)
                    if self.volume_data is not None:
                        # Shift+LMB → pan (unified with middle)
                        if event.modifiers() & Qt.ShiftModifier:
                            self._mpr_pan_begin(orientation, event.pos())
                            return True
                        # Clip-box handles first
                        if getattr(self, '_df_clip_enabled', False):
                            clip_hit = self._get_df_clip_hit(event.pos(), orientation)
                            if clip_hit is not None:
                                self._df_clip_begin_drag(orientation, event.pos(), clip_hit)
                                return True
                        if getattr(self, 'rotate_mode', False):
                            self.last_mouse_pos = event.pos()
                        elif self.crosshair_enabled:
                            if getattr(self, '_oblique_hover_handle', None) and getattr(self, '_oblique_hover_handle')[0] == orientation:
                                self._oblique_dragging = self._oblique_hover_handle
                                return True
                            ch_hit = self._get_crosshair_hit(event.pos(), orientation)
                            if ch_hit is None:
                                # P0: click-to-place crosshair on empty space
                                self._crosshair_drag_mode = 'place'
                                self._crosshair_hover = (orientation, 'center')
                                self._is_dragging_crosshair = True
                                self.handle_crosshair_click(
                                    orientation, event.pos(), mode='place'
                                )
                                return True
                            self._crosshair_drag_mode = ch_hit[1]
                            self._crosshair_hover = ch_hit
                            self._is_dragging_crosshair = True
                            self.handle_crosshair_click(
                                orientation, event.pos(), mode=ch_hit[1]
                            )
                            self.update_2d_crosshair(orientation)
                            return True

            elif event.type() == QEvent.MouseButtonDblClick:
                if event.button() == Qt.LeftButton and self.volume_data is not None:
                    # P1: double-click → fit pane
                    self._set_active_grid_pane(orientation)
                    self.reset_view(orientation)
                    return True
            
            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.MiddleButton:
                    if getattr(self, "_mpr_pan", None):
                        self._mpr_pan_end(orientation)
                        return True
                if event.button() == Qt.LeftButton:
                    self.last_mouse_pos = None

                    if getattr(self, "_mpr_pan", None):
                        self._mpr_pan_end(orientation)
                        self._restore_interactor_style(orientation)
                        return True

                    # ── Safety: always restore interactor style if it was ──
                    # swapped to dummy during press.  This prevents the style
                    # from getting stuck when eventFilter consumed the press
                    # but VTK observer also ran (or vice-versa).
                    self._restore_interactor_style(orientation)

                    if getattr(self, '_df_clip_drag', None):
                        self._df_clip_end_drag()
                        widget = getattr(self, f'{orientation}_widget')
                        hit = self._get_df_clip_hit(event.pos(), orientation) if getattr(self, '_df_clip_enabled', False) else None
                        self._df_clip_hover = hit
                        if hit is not None:
                            widget.setCursor(self._df_clip_cursor_for_hit(hit))
                        else:
                            widget.unsetCursor()
                        return True

                    if getattr(self, '_is_dragging_crosshair', False):
                        self._is_dragging_crosshair = False
                        self._crosshair_drag_mode = None
                        self._crosshair_drag_end()
                        return True
                if getattr(self, '_oblique_dragging', None):
                    self._oblique_dragging = None

                    self._restore_interactor_style(orientation)

                    hit = self._get_oblique_hit(event.pos(), orientation)
                    self._oblique_hover_handle = hit
                    self.render_slice(orientation, preserve_camera=True)
                    widget = getattr(self, f'{orientation}_widget')
                    if hit:
                        widget.setCursor(Qt.SizeAllCursor)
                    else:
                        widget.unsetCursor()
                    return True
            
            elif event.type() == QEvent.Wheel:
                # ── Fiji-style zoom via Qt wheel event ──────────────────
                # This is the primary handler for scroll-wheel on the 2D views.
                # It fires even when the VTK interactor style is swapped to
                # dummy (crosshair mode), so zoom always works.
                if self.volume_data is not None:
                    modifiers = event.modifiers()
                    angle_delta = event.angleDelta().y()
                    pixel_delta = event.pixelDelta().y() if hasattr(event, "pixelDelta") else 0
                    if angle_delta == 0 and pixel_delta == 0:
                        return super().eventFilter(obj, event)

                    if modifiers & Qt.ControlModifier:
                        # Ctrl+scroll → change slice (step by wheel notches)
                        if pixel_delta != 0:
                            delta = 1 if pixel_delta > 0 else -1
                        else:
                            delta = 1 if angle_delta > 0 else -1
                        self._scroll_change_slice(orientation, delta)
                    else:
                        vtk_widget = getattr(self, f'{orientation}_widget')
                        if pixel_delta != 0:
                            strength = abs(pixel_delta) / 40.0
                            zoom_in = pixel_delta > 0
                        else:
                            strength = abs(angle_delta) / 120.0
                            zoom_in = angle_delta > 0
                        strength = max(0.15, min(strength, 3.0))
                        self._fiji_zoom_at_cursor(
                            vtk_widget, orientation, zoom_in=zoom_in, strength=strength
                        )
                    return True  # consume the event

            elif event.type() == QEvent.Leave:
                widget = getattr(self, f'{orientation}_widget')
                if getattr(self, '_df_clip_hover', None) is not None:
                    self._df_clip_hover = None
                    if getattr(self, '_df_clip_enabled', False):
                        self._df_clip_update_mpr_overlay(orientation)
                        try:
                            widget.GetRenderWindow().Render()
                        except Exception:
                            pass
                if getattr(self, '_oblique_hover_handle', None) and getattr(self, '_oblique_hover_handle')[0] == orientation:
                    self._oblique_hover_handle = None
                    self.render_slice(orientation, preserve_camera=True)
                if getattr(self, '_crosshair_hover', None) and getattr(self, '_crosshair_hover')[0] == orientation:
                    self._crosshair_hover = None
                    if self.crosshair_enabled:
                        self.update_2d_crosshair(orientation)
                widget.unsetCursor()
        
        return super().eventFilter(obj, event)
