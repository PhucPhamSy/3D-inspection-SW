# inno3d/features/teaching/layer_define.py
# -----------------------------------------------------------------------
# LayerDefineMixin — Interactive layer Z-boundary definition on YZ view
#
# When activated, vertical bars appear on the sagittal (YZ) view
# representing z_start / z_end for each layer. Users can drag bars
# to reposition them, with real-time table sync.
# -----------------------------------------------------------------------

import math
import numpy as np
import vtk
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import QMessageBox

from inno3d.core.styles import SemiconductorTheme

# 8-colour palette for layer bars — high-contrast, dark-theme friendly
LAYER_COLORS = [
    (0.13, 0.68, 0.82),   # cyan
    (0.93, 0.65, 0.15),   # orange
    (0.78, 0.34, 0.82),   # magenta
    (0.42, 0.82, 0.28),   # lime
    (0.92, 0.40, 0.55),   # pink
    (0.95, 0.85, 0.25),   # gold
    (0.30, 0.70, 0.95),   # sky blue
    (0.85, 0.50, 0.30),   # burnt orange
]


class LayerDefineMixin:
    """Interactive layer Z-boundary placement on the sagittal (YZ) MPR view.

    State attributes (initialised in SegmentationTab.__init__):
        layer_define_active      bool   – True while mode is engaged
        _layer_define_dragging   bool   – True while a bar is being dragged
        _layer_define_drag_idx   tuple  – (layer_index, 'start'|'end') or None
        _layer_define_hover_idx  tuple  – currently hovered bar or None
    """

    # ────────────────── toggle entry / exit ──────────────────

    def _toggle_layer_define_mode(self, checked=None):
        """Toggle interactive layer-define mode on the sagittal view."""
        if checked is None:
            checked = not getattr(self, 'layer_define_active', False)

        if checked:
            # Guard: need volume
            if self.volume_data is None:
                QMessageBox.warning(self, "Define Layers",
                                    "Please load a volume first.")
                if hasattr(self, 'layer_manual_define_btn'):
                    self.layer_manual_define_btn.setChecked(False)
                return

            # Guard: need layers (auto-gen if empty)
            if not self.layer_definitions:
                self.generate_layers()
                if not self.layer_definitions:
                    QMessageBox.warning(self, "Define Layers",
                                        "Could not generate layers. "
                                        "Set the number of layers first.")
                    if hasattr(self, 'layer_manual_define_btn'):
                        self.layer_manual_define_btn.setChecked(False)
                    return

            self.layer_define_active = True
            self._layer_define_dragging = False
            self._layer_define_drag_idx = None
            self._layer_define_hover_idx = None

            # Visual feedback on the button
            if hasattr(self, 'layer_manual_define_btn'):
                self.layer_manual_define_btn.setChecked(True)
                self.layer_manual_define_btn.setText("\u2705 Done (Exit)")
                self.layer_manual_define_btn.setStyleSheet(
                    f"QPushButton {{"
                    f"  background-color: {SemiconductorTheme.ACCENT_SUCCESS};"
                    f"  color: {SemiconductorTheme.BG_DARK};"
                    f"  font-weight: bold; font-size: 8pt;"
                    f"  padding: 4px 10px; border-radius: 4px;"
                    f"}}"
                    f"QPushButton:hover {{ opacity: 0.85; }}"
                )

            # Show instruction
            if hasattr(self, '_layer_define_hint_lbl'):
                self._layer_define_hint_lbl.setVisible(True)

        else:
            self.layer_define_active = False
            self._layer_define_dragging = False
            self._layer_define_drag_idx = None
            self._layer_define_hover_idx = None

            if hasattr(self, 'layer_manual_define_btn'):
                self.layer_manual_define_btn.setChecked(False)
                self.layer_manual_define_btn.setText("\u270b Manually Define")
                self.layer_manual_define_btn.setStyleSheet(
                    f"QPushButton {{"
                    f"  background-color: {SemiconductorTheme.BTN_SECONDARY_BG};"
                    f"  color: {SemiconductorTheme.TEXT_PRIMARY};"
                    f"  font-weight: bold; font-size: 8pt;"
                    f"  padding: 4px 10px; border-radius: 4px;"
                    f"  border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};"
                    f"}}"
                    f"QPushButton:hover {{ background-color: {SemiconductorTheme.BG_LIGHT}; }}"
                )

            if hasattr(self, '_layer_define_hint_lbl'):
                self._layer_define_hint_lbl.setVisible(False)

        # Refresh the sagittal view to draw / clear bars
        if self.volume_data is not None:
            self.update_plane_view('sagittal', preserve_camera=True)

    # ────────────────── coordinate helpers ──────────────────

    def _layer_bar_world_x_for_z(self, z_val):
        """Map volume Z index -> VTK world X on sagittal view.

        Sagittal rendering: slice_data = transpose(volume[:, :, x])
        produces (Y, Z).  h=Y, w=Z.  VTK X axis = Z.
        So world_x == z_val directly.
        """
        return float(z_val)

    def _sagittal_pick_z(self, qt_pos):
        """Pick the Z value (horizontal axis) on the sagittal VTK view.

        Returns float Z or None.
        """
        try:
            widget = getattr(self, 'sagittal_widget')
            renderer = getattr(self, 'sagittal_renderer')
            x, y = qt_pos.x(), qt_pos.y()
            size = widget.GetRenderWindow().GetSize()
            vtk_y = size[1] - y

            picker = vtk.vtkWorldPointPicker()
            picker.Pick(x, vtk_y, 0, renderer)
            world_pos = picker.GetPickPosition()
            return world_pos[0]  # X axis = Z on sagittal
        except Exception:
            return None

    def _layer_define_grab_margin(self):
        """Adaptive grab margin: smaller when zoomed in, larger when zoomed out.

        Based on the camera parallel scale -- proportional to how many world
        units are visible vertically.  We want ~8 screen-pixels of grab zone.
        """
        try:
            renderer = getattr(self, 'sagittal_renderer')
            camera = renderer.GetActiveCamera()
            ps = camera.GetParallelScale()  # half-height in world units
            size = self.sagittal_widget.GetRenderWindow().GetSize()
            vp_h = max(size[1], 1)
            # world-units per screen-pixel
            world_per_px = (2.0 * ps) / vp_h
            return max(1.5, world_per_px * 8.0)
        except Exception:
            return 5.0

    # ────────────────── hit testing ──────────────────

    def _hit_test_layer_bar(self, world_z):
        """Find the closest layer bar to the cursor's Z position.

        Returns (layer_index, 'start'|'end') or None.
        """
        if not self.layer_definitions:
            return None

        margin = self._layer_define_grab_margin()
        best = None
        best_dist = margin

        for i, ld in enumerate(self.layer_definitions):
            z_s = ld['z_start']
            z_e = ld['z_end']

            d_start = abs(world_z - z_s)
            d_end = abs(world_z - z_e)

            if d_start < best_dist:
                best_dist = d_start
                best = (i, 'start')
            if d_end < best_dist:
                best_dist = d_end
                best = (i, 'end')

        return best

    # ────────────────── mouse handlers ──────────────────

    def _handle_layer_define_mouse_press(self, pos):
        """Start dragging a layer bar."""
        world_z = self._sagittal_pick_z(pos)
        if world_z is None:
            return

        hit = self._hit_test_layer_bar(world_z)
        if hit is not None:
            self._layer_define_dragging = True
            self._layer_define_drag_idx = hit
            self.sagittal_widget.setCursor(Qt.SizeHorCursor)

    def _handle_layer_define_mouse_move(self, pos, is_drag=False):
        """Handle hover highlight or drag-update."""
        world_z = self._sagittal_pick_z(pos)
        if world_z is None:
            return

        if is_drag and self._layer_define_dragging and self._layer_define_drag_idx is not None:
            layer_idx, bound = self._layer_define_drag_idx
            ld = self.layer_definitions[layer_idx]

            vol_z = self.volume_data.shape[0]
            new_z = int(round(max(0, min(world_z, vol_z - 1))))

            # Constraint: start < end within the same layer (minimum gap = 2)
            min_gap = 2
            if bound == 'start':
                new_z = min(new_z, ld['z_end'] - min_gap)
                new_z = max(0, new_z)
                ld['z_start'] = new_z
            else:
                new_z = max(new_z, ld['z_start'] + min_gap)
                new_z = min(new_z, vol_z - 1)
                ld['z_end'] = new_z

            # Live-update table + flash the changed cell
            self._populate_layer_table()
            col = 2 if bound == 'start' else 3
            if hasattr(self, '_flash_layer_cell'):
                self._flash_layer_cell(layer_idx, col)

            # Refresh sagittal view
            self.update_plane_view('sagittal', preserve_camera=True)

        else:
            # Hover — highlight nearest bar
            hit = self._hit_test_layer_bar(world_z)
            if hit != self._layer_define_hover_idx:
                self._layer_define_hover_idx = hit
                if hit:
                    self.sagittal_widget.setCursor(Qt.SizeHorCursor)
                else:
                    self.sagittal_widget.setCursor(Qt.ArrowCursor)
                self.update_plane_view('sagittal', preserve_camera=True)

    def _handle_layer_define_mouse_release(self, pos):
        """Commit the bar position."""
        if self._layer_define_dragging:
            self._layer_define_dragging = False
            self._layer_define_drag_idx = None
            self.sagittal_widget.setCursor(Qt.ArrowCursor)

            # Final refresh + table sync
            self._populate_layer_table()
            self.update_plane_view('sagittal', preserve_camera=True)

    # ────────────────── VTK overlay rendering ──────────────────

    def _draw_layer_define_bars(self, renderer, h, w):
        """Draw vertical bars + translucent bands on the sagittal (YZ) view.

        Called from update_plane_view when layer_define_active is True.
        h = Y-dimension (vertical), w = Z-dimension (horizontal) in VTK.

        Uses vtkActor for world-coordinate lines. Line width is set
        in screen pixels so bars remain visible at any zoom level.
        """
        if not self.layer_definitions:
            return

        try:
            widget = getattr(self, 'sagittal_widget')
            ren_win = widget.GetRenderWindow()
            vp_size = ren_win.GetSize()
            vp_w = max(vp_size[0], 1)
            vp_h = max(vp_size[1], 1)
        except Exception:
            vp_w, vp_h = 800, 600

        camera = renderer.GetActiveCamera()
        cam_fp = camera.GetFocalPoint()
        cam_ps = camera.GetParallelScale()   # half-height in world units

        # Camera center in world coords
        cam_cx = cam_fp[0]
        cam_cy = cam_fp[1]

        # Visible world range
        aspect = vp_w / float(vp_h)
        world_half_w = cam_ps * aspect
        world_x_min = cam_cx - world_half_w
        world_x_max = cam_cx + world_half_w

        hover_idx = self._layer_define_hover_idx
        drag_idx = self._layer_define_drag_idx if self._layer_define_dragging else None

        for i, ld in enumerate(self.layer_definitions):
            color = LAYER_COLORS[i % len(LAYER_COLORS)]

            for bound in ('start', 'end'):
                z_val = ld['z_start'] if bound == 'start' else ld['z_end']
                world_x = self._layer_bar_world_x_for_z(z_val)

                # Skip if not in visible range (don't draw off-screen bars)
                if world_x < world_x_min - 20 or world_x > world_x_max + 20:
                    continue

                is_hovered = (hover_idx == (i, bound))
                is_dragged = (drag_idx == (i, bound))

                # Determine style
                if is_dragged:
                    line_width = 3.0
                    line_color = (1.0, 1.0, 1.0)  # white while dragging
                    line_opacity = 1.0
                elif is_hovered:
                    line_width = 2.5
                    line_color = (min(1.0, color[0] + 0.3),
                                  min(1.0, color[1] + 0.3),
                                  min(1.0, color[2] + 0.3))
                    line_opacity = 1.0
                else:
                    line_width = 1.5
                    line_color = color
                    line_opacity = 0.85

                # Draw vertical line using vtkLineSource + vtkActor (world coords)
                # Line spans the full Y (vertical) extent of the image
                line_src = vtk.vtkLineSource()
                line_src.SetPoint1(world_x, 0, 0.3)
                line_src.SetPoint2(world_x, h, 0.3)
                line_src.Update()

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(line_src.GetOutputPort())

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                prop = actor.GetProperty()
                prop.SetColor(*line_color)
                prop.SetLineWidth(line_width)
                prop.SetOpacity(line_opacity)

                # Dashed line for 'end' bars
                if bound == 'end':
                    prop.SetLineStipplePattern(0xF0F0)  # dashed
                    prop.SetLineStippleRepeatFactor(1)

                renderer.AddActor(actor)

                # -- Label at the top of the bar --
                label_text = f"L{i+1} {'S' if bound == 'start' else 'E'}"

                text_actor = vtk.vtkTextActor()
                text_actor.SetInput(label_text)
                text_actor.GetTextProperty().SetFontSize(11)
                text_actor.GetTextProperty().SetColor(*line_color)
                text_actor.GetTextProperty().SetBold(True)
                text_actor.GetTextProperty().SetFontFamilyToCourier()
                text_actor.GetTextProperty().SetJustificationToCentered()
                text_actor.GetTextProperty().SetVerticalJustificationToTop()
                # Slight background for readability
                text_actor.GetTextProperty().SetBackgroundColor(0.05, 0.08, 0.12)
                text_actor.GetTextProperty().SetBackgroundOpacity(0.7)

                # Convert world coords to display (viewport) coords for text placement
                coord = vtk.vtkCoordinate()
                coord.SetCoordinateSystemToWorld()
                coord.SetValue(world_x, h - 2, 0.3)
                display_pos = coord.GetComputedDisplayValue(renderer)

                text_actor.SetDisplayPosition(int(display_pos[0]), int(display_pos[1]))
                renderer.AddActor2D(text_actor)

            # -- Translucent band between z_start and z_end --
            z_s = ld['z_start']
            z_e = ld['z_end']
            x_start = self._layer_bar_world_x_for_z(z_s)
            x_end = self._layer_bar_world_x_for_z(z_e)

            # Only draw band if partially visible
            if x_end < world_x_min - 20 or x_start > world_x_max + 20:
                continue

            # Create a thin quad for the band
            points = vtk.vtkPoints()
            points.InsertNextPoint(x_start, 0, 0.25)
            points.InsertNextPoint(x_end,   0, 0.25)
            points.InsertNextPoint(x_end,   h, 0.25)
            points.InsertNextPoint(x_start, h, 0.25)

            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, 0)
            quad.GetPointIds().SetId(1, 1)
            quad.GetPointIds().SetId(2, 2)
            quad.GetPointIds().SetId(3, 3)

            cells = vtk.vtkCellArray()
            cells.InsertNextCell(quad)

            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            poly.SetPolys(cells)

            band_mapper = vtk.vtkPolyDataMapper()
            band_mapper.SetInputData(poly)

            band_actor = vtk.vtkActor()
            band_actor.SetMapper(band_mapper)
            band_prop = band_actor.GetProperty()
            band_prop.SetColor(*color)
            band_prop.SetOpacity(0.10)  # very subtle tint
            band_prop.LightingOff()

            renderer.AddActor(band_actor)

            # -- Layer name label in the center of the band --
            band_center_x = (x_start + x_end) / 2.0
            band_center_y = h * 0.5

            # Only show center label if band is wide enough to be visible
            band_width_world = x_end - x_start
            # Check if at least ~30 pixels wide on screen
            world_per_px = (2.0 * cam_ps * aspect) / vp_w
            band_width_px = band_width_world / max(world_per_px, 0.001)

            if band_width_px > 30:
                center_text = vtk.vtkTextActor()
                center_text.SetInput(ld.get('name', f'Layer {i+1}'))
                center_text.GetTextProperty().SetFontSize(12)
                center_text.GetTextProperty().SetColor(*color)
                center_text.GetTextProperty().SetBold(False)
                center_text.GetTextProperty().SetFontFamilyToCourier()
                center_text.GetTextProperty().SetJustificationToCentered()
                center_text.GetTextProperty().SetVerticalJustificationToCentered()
                center_text.GetTextProperty().SetBackgroundColor(0.05, 0.08, 0.12)
                center_text.GetTextProperty().SetBackgroundOpacity(0.6)

                coord2 = vtk.vtkCoordinate()
                coord2.SetCoordinateSystemToWorld()
                coord2.SetValue(band_center_x, band_center_y, 0.3)
                dp2 = coord2.GetComputedDisplayValue(renderer)
                center_text.SetDisplayPosition(int(dp2[0]), int(dp2[1]))
                renderer.AddActor2D(center_text)
