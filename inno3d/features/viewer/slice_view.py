# inno3d/features/viewer/slice_view.py
# -----------------------------------------------------------------------
# SliceViewMixin  --  extracted from inno3d/tabs/viewer.py (Phase 3.7)
#
# create_slice_view, toggle_view_fullscreen, exit_fullscreen,
# set_projection, oblique reslice/drag, overlay compositing,
# label drawing, choose_overlay_color, choose_ruler_color.
# -----------------------------------------------------------------------

import math
from pathlib import Path as _Path

import numpy as np
import vtk

from PyQt5.QtCore import Qt, QSize, QPoint, QTimer
from PyQt5.QtGui import QColor, QIcon, QImage, QPixmap, QPainter, QPen, QCursor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QSizePolicy,
    QColorDialog, QFileDialog, QGraphicsDropShadowEffect,
)
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from inno3d.core.styles import SemiconductorTheme
from inno3d.core.view_support import (
    SliceControlOverlay,
    SliceBarWheelFilter,
    StepOneSliceSlider,
)


class SliceViewMixin:
    """Mixin: 2D slice view creation, fullscreen, oblique, overlays, labels.

    Extracted from inno3d/tabs/viewer.py Phase 3.7.
    """

    def _plane_cell_stylesheet(self, active=False):
        """Dragonfly-style pane chrome: seamless black fill, optional focus ring."""
        border = (
            SemiconductorTheme.ACCENT_WARNING if active
            else "transparent"
        )
        # 1px transparent border reserves space so activating a pane does not reflow
        return (
            f"QWidget#PlaneCell {{"
            f"  background-color: #000000;"
            f"  border: 1px solid {border};"
            f"}}"
        )

    def _set_active_grid_pane(self, orientation):
        """Highlight the active MPR/3D pane with a thin focus ring (Dragonfly-like)."""
        if not hasattr(self, '_grid_cells'):
            return
        self._active_grid_pane = orientation
        for key, cell in self._grid_cells.items():
            widget = cell[0]
            is_active = (key == orientation)
            widget.setProperty("active", "true" if is_active else "false")
            widget.setStyleSheet(self._plane_cell_stylesheet(active=is_active))

    def create_slice_view(self, title, orientation):
        widget = QWidget()
        widget.setObjectName("PlaneCell")
        widget.setStyleSheet(self._plane_cell_stylesheet(active=False))
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        widget.setMinimumSize(0, 0)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # --- COMPACT HEADER (Title & Pixel Info) — soft overlay bar, no hard seam ---
        header_widget = QWidget()
        header_widget.setMaximumHeight(26)
        header_widget.setStyleSheet(
            "background: rgba(0, 0, 0, 0.45); border: none;"
        )
        
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(8, 0, 8, 0)
        
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setStyleSheet(f"font-size: 9pt; color: {SemiconductorTheme.ACCENT_PRIMARY};")
        header_layout.addWidget(title_label)
        
        # Apply shadow effect for readability
        title_shadow = QGraphicsDropShadowEffect()
        title_shadow.setBlurRadius(3)
        title_shadow.setOffset(1, 1)
        title_shadow.setColor(QColor(0, 0, 0, 220))
        title_label.setGraphicsEffect(title_shadow)

        header_layout.addStretch()
        
        pixel_label = QLabel("Pixel: --")
        pixel_label.setStyleSheet(f"font-size: 8pt; color: white;")
        header_layout.addWidget(pixel_label)
        
        pixel_shadow = QGraphicsDropShadowEffect()
        pixel_shadow.setBlurRadius(3)
        pixel_shadow.setOffset(1, 1)
        pixel_shadow.setColor(QColor(0, 0, 0, 220))
        pixel_label.setGraphicsEffect(pixel_shadow)
        
        layout.addWidget(header_widget)
        
        # VTK container — pure black so hairline grid dividers read cleanly
        vtk_container = QWidget()
        vtk_container.setStyleSheet("background-color: #000000; border: none;")
        vtk_grid = QGridLayout(vtk_container)
        vtk_grid.setContentsMargins(0, 0, 0, 0)
        vtk_grid.setSpacing(0)
        
        vtk_widget = QVTKRenderWindowInteractor()
        vtk_grid.addWidget(vtk_widget, 0, 0)
        
        style = vtk.vtkInteractorStyleImage()
        rw = vtk_widget.GetRenderWindow()
        # Need alpha bit-planes so RGBA segmentation overlays composite correctly
        # (otherwise mask RGBA may look solid yellow / ignore per-pixel alpha).
        try:
            rw.SetAlphaBitPlanes(1)
            rw.SetMultiSamples(0)
        except Exception:
            pass
        rw.GetInteractor().SetInteractorStyle(style)
        # Default: pixel-probe cross (VTK Image style otherwise shows open-hand)
        try:
            vtk_widget.setCursor(Qt.CrossCursor)
        except Exception:
            pass
        self.setup_vtk_observers(vtk_widget, orientation)
        vtk_widget.installEventFilter(self)
        
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0, 0, 0)
        try:
            renderer.SetUseDepthPeeling(0)
        except Exception:
            pass
        rw.AddRenderer(renderer)
        
        # --- VERTICAL OVERLAY ---
        on_fs = lambda checked=False, o=orientation: self.toggle_view_fullscreen(o)
        on_reset = lambda checked=False, o=orientation: self.reset_view(o)
        on_rev_z = self.toggle_reverse_z if orientation == 'axial' else None
        
        overlay = SliceControlOverlay(vtk_widget, orientation, on_fs, on_reset, on_rev_z)
        setattr(self, f'{orientation}_overlay_group', overlay)
        QTimer.singleShot(100, overlay.update_position) # Delay to ensure widget is laid out
        
        layout.addWidget(vtk_container, 1)
        
        # Bottom Slider — soft bar so panes stay visually linked
        bottom_widget = QWidget()
        bottom_widget.setMaximumHeight(28)
        bottom_widget.setStyleSheet(
            "background: rgba(0, 0, 0, 0.45); border: none;"
        )
        
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(8, 0, 8, 0)
        
        slice_text = QLabel("SLICE:")
        slice_text.setStyleSheet(f"font-size: 8pt; color: {SemiconductorTheme.ACCENT_PRIMARY}; font-weight: bold;")
        bottom_layout.addWidget(slice_text)
        
        slice_text_shadow = QGraphicsDropShadowEffect()
        slice_text_shadow.setBlurRadius(3)
        slice_text_shadow.setOffset(1, 1)
        slice_text_shadow.setColor(QColor(0, 0, 0, 220))
        slice_text.setGraphicsEffect(slice_text_shadow)
        
        slice_slider = StepOneSliceSlider(Qt.Horizontal)
        slice_slider.setMinimum(0)
        slice_slider.setMaximum(100)
        slice_slider.setValue(50)
        slice_slider.setMaximumHeight(16)
        slice_slider.setToolTip("Drag or scroll (step 1) to change slice")
        slice_slider.valueChanged.connect(lambda value, o=orientation: self.update_slice(o, value))
        bottom_layout.addWidget(slice_slider, 1)
        
        slice_label = QLabel("50 / 100")
        slice_label.setStyleSheet(f"font-size: 8pt; color: white; min-width: 60px;")
        slice_slider.valueChanged.connect(lambda value, lbl=slice_label, s=slice_slider: lbl.setText(f"{value} / {s.maximum()}"))
        bottom_layout.addWidget(slice_label)
        
        slice_label_shadow = QGraphicsDropShadowEffect()
        slice_label_shadow.setBlurRadius(3)
        slice_label_shadow.setOffset(1, 1)
        slice_label_shadow.setColor(QColor(0, 0, 0, 220))
        slice_label.setGraphicsEffect(slice_label_shadow)

        # Wheel on entire slice bar (label + groove + value) → step 1
        wheel_filter = SliceBarWheelFilter(slice_slider, bottom_widget)
        bottom_widget.installEventFilter(wheel_filter)
        slice_slider.installEventFilter(wheel_filter)
        slice_text.installEventFilter(wheel_filter)
        slice_label.installEventFilter(wheel_filter)
        # Keep filter alive with the bar
        bottom_widget._slice_wheel_filter = wheel_filter
        
        layout.addWidget(bottom_widget)
        widget.setLayout(layout)
        
        # Store refs
        setattr(self, f'{orientation}_widget', vtk_widget)
        setattr(self, f'{orientation}_renderer', renderer)
        setattr(self, f'{orientation}_slice_slider', slice_slider)
        setattr(self, f'{orientation}_slice_label', slice_label)
        setattr(self, f'{orientation}_overlay_c1', overlay.c1_btn)
        setattr(self, f'{orientation}_overlay_c2', overlay.c2_btn)
        setattr(self, f'{orientation}_opacity_slider', overlay.opacity_slider)
        if orientation == 'axial' and hasattr(overlay, 'z_btn'):
            self.reverse_z_btn = overlay.z_btn
        
        # Use toggled/valueChanged with default-arg capture so each MPR pane
        # always refreshes the correct orientation (and checkable clicked(bool)
        # never confuses the slot).
        overlay.c1_btn.toggled.connect(
            lambda _checked=False, o=orientation: self.update_overlay_visibility(o)
        )
        overlay.c2_btn.toggled.connect(
            lambda _checked=False, o=orientation: self.update_overlay_visibility(o)
        )
        overlay.opacity_slider.valueChanged.connect(
            lambda _val=0, o=orientation: self.update_overlay_visibility(o)
        )
        
        setattr(self, f'{orientation}_pixel_label', pixel_label)
        setattr(self, f'{orientation}_container', widget)
        
        return widget

    def toggle_view_fullscreen(self, orientation):
        """Toggle in-grid fullscreen: expand one view to fill the current layout grid."""
        if self._fullscreen_view is not None:
            # Already fullscreen — exit
            self.exit_fullscreen()
            return

        if orientation not in getattr(self, "_layout_visible_keys", set(self._grid_cells.keys())):
            # Pane not in current layout — nothing to maximize
            return

        # ── Enter fullscreen ──
        self._fullscreen_view = orientation
        self._set_active_grid_pane(orientation)

        # Hide the other views in the current layout
        for key, cell in self._grid_cells.items():
            widget = cell[0]
            if key != orientation:
                widget.setVisible(False)

        # Make the target view span the full current grid
        cell = self._grid_cells[orientation]
        target_widget = cell[0]
        nrows = max(1, int(getattr(self, "_grid_nrows", 2)))
        ncols = max(1, int(getattr(self, "_grid_ncols", 2)))
        self._grid_layout.removeWidget(target_widget)
        self._grid_layout.addWidget(target_widget, 0, 0, nrows, ncols)

        # For 3D: show the advanced sidebar (reserve width so dense rows
        # like MPR Show/Clip/Flip are not clipped when the v-scrollbar appears)
        if orientation == 'view_3d':
            self.advanced_3d_controls.setVisible(True)
            self.advanced_3d_controls.setMinimumWidth(360)
            self.advanced_3d_controls.setMaximumWidth(420)
            self.advanced_3d_controls.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Keep overlay strip identical to non-fullscreen (same maximize icon,
        # tooltip, and vertical button column). Only re-anchor after grid span.
        overlay = getattr(self, f'{orientation}_overlay_group', None)
        if not overlay:
            overlay = getattr(self, 'view_3d_overlay_group', None)
        if overlay:
            overlay.set_fullscreen(True)
            QTimer.singleShot(50, overlay.update_position)

        # After layout settles (sidebar + grid span), reframe 3D so Track pivot
        # and arcball match the new viewport — same feel as non-fullscreen.
        if orientation == 'view_3d':
            def _post_fullscreen_3d_sync():
                try:
                    if hasattr(self, '_reframe_3d_after_viewport_change'):
                        self._reframe_3d_after_viewport_change()
                    else:
                        self._sync_3d_orbit_pivot()
                        ren = getattr(self, 'view_3d_renderer', None)
                        widget = getattr(self, 'view_3d_widget', None)
                        if ren and widget:
                            ren.ResetCameraClippingRange()
                            widget.GetRenderWindow().Render()
                except Exception:
                    pass
            # Two passes: Qt may still be applying sidebar min-width on first tick
            QTimer.singleShot(50, _post_fullscreen_3d_sync)
            QTimer.singleShot(180, _post_fullscreen_3d_sync)
        else:
            # MPR panes: reset camera to fit new viewport
            QTimer.singleShot(80, lambda ori=orientation: self.reset_view(ori))

    def exit_fullscreen(self):
        """Restore the current view layout from in-grid fullscreen."""
        if self._fullscreen_view is None:
            return

        orientation = self._fullscreen_view

        # Remove the spanning widget and place it back at its layout position
        cell = self._grid_cells.get(orientation)
        if cell is None:
            self._fullscreen_view = None
            return
        target_widget, orig_row, orig_col = cell[0], cell[1], cell[2]
        rowspan = cell[3] if len(cell) > 3 else 1
        colspan = cell[4] if len(cell) > 4 else 1
        self._grid_layout.removeWidget(target_widget)
        self._grid_layout.addWidget(target_widget, orig_row, orig_col, rowspan, colspan)

        # Show only panes that belong to the active layout preset
        visible = getattr(self, "_layout_visible_keys", set(self._grid_cells.keys()))
        for key, cell in self._grid_cells.items():
            widget = cell[0]
            widget.setVisible(key in visible)

        # For 3D: hide + collapse advanced sidebar so 2×2 cells stay even
        if orientation == 'view_3d':
            self.advanced_3d_controls.setVisible(False)
            self.advanced_3d_controls.setMinimumWidth(0)
            self.advanced_3d_controls.setMaximumWidth(0)
            self.advanced_3d_controls.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

        # Same strip look as enter — no icon/tooltip swap
        overlay = getattr(self, f'{orientation}_overlay_group', None)
        if not overlay:
            overlay = getattr(self, 'view_3d_overlay_group', None)
        if overlay:
            overlay.set_fullscreen(False)
            QTimer.singleShot(50, overlay.update_position)

        was_3d = orientation == 'view_3d'
        self._fullscreen_view = None
        QTimer.singleShot(40, self._refresh_layout_viewports)
        # Exit FS also changes aspect — reframe orbit pivot like enter
        if was_3d and hasattr(self, '_reframe_3d_after_viewport_change'):
            QTimer.singleShot(80, self._reframe_3d_after_viewport_change)
            QTimer.singleShot(200, self._reframe_3d_after_viewport_change)

    def set_projection(self, mode):
        self.projection_mode = str(mode).split('#')[0].strip().lower()
        if not hasattr(self, 'view_3d_renderer') or self.view_3d_renderer is None: 
            return
        
        cam = self.view_3d_renderer.GetActiveCamera()
        if self.projection_mode == "perspective":
            if hasattr(self, 'btn_perspective'):
                self.btn_perspective.setChecked(True)
            if hasattr(self, 'btn_ortho'):
                self.btn_ortho.setChecked(False)
            cam.ParallelProjectionOff()
            cam.SetViewAngle(40.0)
            self.view_3d_renderer.ResetCameraClippingRange()
        else:
            if hasattr(self, 'btn_ortho'):
                self.btn_ortho.setChecked(True)
            if hasattr(self, 'btn_perspective'):
                self.btn_perspective.setChecked(False)
            cam.ParallelProjectionOn()
            
        if hasattr(self, 'view_3d_widget') and self.view_3d_widget is not None:
            self.view_3d_widget.GetRenderWindow().Render()

    def update_pixel_value(self, orientation, pos):
        """Delegate to VolumeIOMixin implementation (full X/Y/Z + intensity).

        Kept for MRO safety if mixin order changes; real logic lives in
        ``VolumeIOMixin.update_pixel_value``.
        """
        # Prefer the VolumeIOMixin method if available via MRO (usual case is
        # this body is never reached). Fallback: same XYZ format as Teaching.
        try:
            if self.volume_data is None:
                return
            widget = getattr(self, f"{orientation}_widget")
            renderer = getattr(self, f"{orientation}_renderer")
            label = getattr(self, f"{orientation}_pixel_label", None)
            if label is None:
                return

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

            if orientation == "axial":
                px = int(round(world_pos[0]))
                py = int(round(world_pos[1]))
                if 0 <= px < vol_x and 0 <= py < vol_y:
                    actual_z = (
                        (vol_z - 1 - slice_idx)
                        if getattr(self, "reverse_z", False)
                        else slice_idx
                    )
                    value = self.volume_data[actual_z, py, px]
                    coord_str = f"X:{px} Y:{py} Z:{actual_z}"
            elif orientation == "coronal":
                px = int(round(world_pos[0]))
                pz = int(round(world_pos[1]))
                if 0 <= px < vol_x and 0 <= pz < vol_z:
                    value = self.volume_data[pz, slice_idx, px]
                    coord_str = f"X:{px} Y:{slice_idx} Z:{pz}"
            else:
                pz = int(round(world_pos[0]))
                py = int(round(world_pos[1]))
                if 0 <= pz < vol_z and 0 <= py < vol_y:
                    value = self.volume_data[pz, py, slice_idx]
                    coord_str = f"X:{slice_idx} Y:{py} Z:{pz}"

            if value is not None:
                label.setText(f"{coord_str} | Pixel: {value}")
            else:
                label.setText("Pixel: --")
        except Exception:
            pass
    
    def _get_oblique_hit(self, pos, orientation):
        import math
        if self.volume_data is None:
            return None
        if not getattr(self, 'oblique_handles_enabled', False):
            return None
        widget = getattr(self, f'{orientation}_widget')
        renderer = getattr(self, f'{orientation}_renderer')
        x, y = pos.x(), pos.y()
        size = widget.GetRenderWindow().GetSize()
        vtk_y = size[1] - y

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
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
            
        coord.SetValue(cx_w, cy_w, 0.5)
        dp = coord.GetComputedDisplayValue(renderer)
        dx, dy = float(dp[0]), float(dp[1])

        vp = renderer.GetViewport()
        win = renderer.GetRenderWindow().GetSize()
        vp_w = (vp[2] - vp[0]) * win[0]
        vp_h = (vp[3] - vp[1]) * win[1]

        if orientation == 'axial':
            theta_h = math.radians(self.oblique_angles.get('coronal', 0.0))
            theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
            h_name, v_name = 'coronal', 'sagittal'
        elif orientation == 'coronal':
            theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
            theta_v = math.radians(self.oblique_angles.get('sagittal', 0.0)) + math.pi / 2
            h_name, v_name = 'axial', 'sagittal'
        else: # sagittal
            theta_h = math.radians(self.oblique_angles.get('axial', 0.0))
            theta_v = math.radians(self.oblique_angles.get('coronal', 0.0)) + math.pi / 2
            h_name, v_name = 'axial', 'coronal'

        extent = max(vp_w, vp_h) * 0.6
        handle_dist = min(extent * 0.35, 120)
        hit_dist = 35  # px tolerance

        # Check horizontal handles
        cos_h, sin_h = math.cos(theta_h), math.sin(theta_h)
        for sign in [1, -1]:
            ex = dx + sign * handle_dist * cos_h
            ey = dy + sign * handle_dist * sin_h
            if math.hypot(x - ex, vtk_y - ey) < hit_dist:
                return (orientation, h_name)

        # Check vertical handles
        cos_v, sin_v = math.cos(theta_v), math.sin(theta_v)
        for sign in [1, -1]:
            ex = dx + sign * handle_dist * cos_v
            ey = dy + sign * handle_dist * sin_v
            if math.hypot(x - ex, vtk_y - ey) < hit_dist:
                return (orientation, v_name)

        return None

    def _handle_oblique_rotate(self, pos, orientation):
        import math
        if self.volume_data is None:
            return
        widget = getattr(self, f'{orientation}_widget')
        renderer = getattr(self, f'{orientation}_renderer')
        x, y = pos.x(), pos.y()
        size = widget.GetRenderWindow().GetSize()
        vtk_y = size[1] - y

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
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
            
        coord.SetValue(cx_w, cy_w, 0.5)
        dp = coord.GetComputedDisplayValue(renderer)
        dx, dy = float(dp[0]), float(dp[1])

        angle = math.degrees(math.atan2(vtk_y - dy, x - dx))

        dragging = getattr(self, '_oblique_dragging', None)
        if dragging:
            drag_ori, axis = dragging
            if drag_ori == 'axial':
                h_name, v_name = 'coronal', 'sagittal'
            elif drag_ori == 'coronal':
                h_name, v_name = 'axial', 'sagittal'
            else: # sagittal
                h_name, v_name = 'axial', 'coronal'
                
            if axis == h_name:
                self.oblique_angles[axis] = angle
            else:
                self.oblique_angles[axis] = angle - 90.0

        # Update manual align angle labels immediately (lightweight)
        if getattr(self, '_manual_align_active', False):
            self._update_manual_angle_labels()

        # ── Throttled rendering during drag ──
        # Instead of rendering all 3 views on every mouse move (very slow),
        # we store which orientation needs refresh and use a debounce timer.
        self._oblique_drag_orientation = orientation
        
        if not hasattr(self, '_oblique_throttle_timer'):
            self._oblique_throttle_timer = QTimer(self)
            self._oblique_throttle_timer.setSingleShot(True)
            self._oblique_throttle_timer.timeout.connect(self._oblique_throttle_render)
        
        # Only schedule a new render if one is not already pending
        if not self._oblique_throttle_timer.isActive():
            self._oblique_throttle_timer.start(60)  # max ~16 fps during drag
    
    def _oblique_throttle_render(self):
        """Render callback for throttled oblique drag.
        
        Synchronises ALL views during drag for real-time feedback:
        - Manual align mode: lightweight crosshair overlay refresh on all 3 views.
        - Normal oblique mode: fast reslice on the active view, lightweight
          crosshair refresh on the other two, plus 3D crosshair update.
        """
        ori = getattr(self, '_oblique_drag_orientation', None)
        if ori is None:
            return
        
        if getattr(self, '_manual_align_active', False):
            # Manual align mode: only redraw crosshair lines (ultra-lightweight)
            for o in ['axial', 'coronal', 'sagittal']:
                self._refresh_crosshair_overlay(o)
        else:
            # Normal oblique mode: fast reslice on active view
            self._oblique_drag_fast = True
            self.render_slice(ori, preserve_camera=True)
            self._oblique_drag_fast = False
            
            # Lightweight crosshair overlay refresh on the other two views
            for o in ['axial', 'coronal', 'sagittal']:
                if o != ori:
                    self._refresh_crosshair_overlay(o)
            
            # Update 3D crosshair lines to follow the rotation
            self.update_3d_crosshair()
    
    def _oblique_drag_finish(self):
        """Called on mouse release to do a full-quality render of all views."""
        # Cancel any pending throttled render
        if hasattr(self, '_oblique_throttle_timer') and self._oblique_throttle_timer.isActive():
            self._oblique_throttle_timer.stop()
        
        self._oblique_drag_orientation = None
        self._oblique_drag_fast = False
        
        if getattr(self, '_manual_align_active', False):
            # In manual align mode: just refresh crosshair overlays (no reslicing)
            for ori in ['axial', 'coronal', 'sagittal']:
                self._refresh_crosshair_overlay(ori)
        else:
            # Normal oblique mode: full quality render of ALL orientations
            for ori in ['axial', 'coronal', 'sagittal']:
                self.render_slice(ori, preserve_camera=True)
            self.update_3d_crosshair()
    
    def _refresh_crosshair_overlay(self, orientation):
        """Redraw ONLY the crosshair overlay actors without re-reslicing the volume.
        
        This is extremely lightweight — only repositions 2D line/circle actors
        based on current oblique_angles. Used during manual align drag for
        instant visual feedback without any volume computation.
        """
        if self.volume_data is None:
            return
        if not self.crosshair_enabled:
            return
        
        # Use the unified persistent crosshair system
        self.update_2d_crosshair(orientation)


    def get_oblique_R(self):
        import math
        # axial: yaw (around Z)
        tz = math.radians(self.oblique_angles.get('axial', 0.0))
        cz, sz = math.cos(tz), math.sin(tz)
        Rz = np.array([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ])
        
        # coronal: pitch (around Y)
        ty = math.radians(self.oblique_angles.get('coronal', 0.0))
        cy, sy = math.cos(ty), math.sin(ty)
        Ry = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ])
        
        # sagittal: roll (around X)
        tx = math.radians(self.oblique_angles.get('sagittal', 0.0))
        cx, sx = math.cos(tx), math.sin(tx)
        Rx = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ])
        
        # Combined rotation matrix R
        return np.dot(Rz, np.dot(Ry, Rx))

    def _oblique_reslice_3d(self, orientation, slice_idx):
        from scipy.ndimage import map_coordinates
        import math
        
        vol = self.volume_data
        vol_z, vol_y, vol_x = vol.shape
        
        cx = float(self.crosshair_position[0])
        cy = float(self.crosshair_position[1])
        cz = float(self.crosshair_position[2])
        
        R = self.get_oblique_R()
        
        def _reslice_vol_3d(data, order=1):
            if data is None:
                return None
            bg_val = float(data.min())
            
            if orientation == 'axial':
                yy, xx = np.mgrid[0:vol_y, 0:vol_x].astype(np.float64)
                x_coords = cx + (xx - cx) * R[0, 0] + (yy - cy) * R[0, 1]
                y_coords = cy + (xx - cx) * R[1, 0] + (yy - cy) * R[1, 1]
                z_coords = cz + (xx - cx) * R[2, 0] + (yy - cy) * R[2, 1]
                
                result = map_coordinates(data, [z_coords, y_coords, x_coords],
                                         order=order, mode='constant', cval=bg_val)
                return result
                
            elif orientation == 'coronal':
                zz, xx = np.mgrid[0:vol_z, 0:vol_x].astype(np.float64)
                x_coords = cx + (xx - cx) * R[0, 0] + (zz - cz) * R[0, 2]
                y_coords = cy + (xx - cx) * R[1, 0] + (zz - cz) * R[1, 2]
                z_coords = cz + (xx - cx) * R[2, 0] + (zz - cz) * R[2, 2]
                
                result = map_coordinates(data, [z_coords, y_coords, x_coords],
                                         order=order, mode='constant', cval=bg_val)
                return np.flipud(result)
                
            else: # sagittal
                yy, zz = np.mgrid[0:vol_y, 0:vol_z].astype(np.float64)
                x_coords = cx + (zz - cz) * R[0, 2] + (yy - cy) * R[0, 1]
                y_coords = cy + (zz - cz) * R[1, 2] + (yy - cy) * R[1, 1]
                z_coords = cz + (zz - cz) * R[2, 2] + (yy - cy) * R[2, 1]
                
                result = map_coordinates(data, [z_coords, y_coords, x_coords],
                                         order=order, mode='constant', cval=bg_val)
                return result

        # During fast drag: use order=0 (nearest-neighbor) for speed, skip masks
        is_fast = getattr(self, '_oblique_drag_fast', False)
        vol_order = 0 if is_fast else 1
        
        slice_data = _reslice_vol_3d(vol, order=vol_order)
        
        if is_fast:
            # Skip mask reslicing during drag for performance
            seg_slice = None
            c1_slice = None
            c2_slice = None
        else:
            seg_slice = _reslice_vol_3d(self.segmentation_data, order=0) if self.segmentation_data is not None else None
            c1_slice = _reslice_vol_3d(self.class1_data, order=0) if self.class1_data is not None else None
            c2_slice = _reslice_vol_3d(self.class2_data, order=0) if self.class2_data is not None else None
        
        return slice_data, seg_slice, c1_slice, c2_slice

