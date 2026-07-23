# inno3d/features/viewer/window_level.py
# ─────────────────────────────────────────────────────────────────────────────
# WindowLevelMixin — extracted from inno3d/tabs/viewer.py (Phase 3.5)
#
# Contains all Window/Level (3D TF range + MPR W/L) methods that bridge the
# TF histogram widget ↔ volume_actor ↔ MPR slice rendering.
#
# MRO target (after Phase 3.5):
#   MultiPlanarView(WindowLevelMixin, ClipBoxMixin, Volume3dMixin,
#                   MprInputMixin, MprRenderMixin, MprNavMixin, CrosshairMixin, QWidget)
# ─────────────────────────────────────────────────────────────────────────────

import vtk
from PyQt5.QtCore import Qt

from inno3d.core.tf_widgets import (
    OPACITY_SHAPES,
    make_opacity_shape_icon,
    make_color_map_icon,
)


class WindowLevelMixin:
    """Mixin: Window/Level callbacks for 3D TF + MPR planes.

    Extracted from ``inno3d/tabs/viewer.py`` Phase 3.5.
    """

    # ── 3D TF range bar drag callback ────────────────────────────────────────

    def _on_tf_range_changed(self, norm_min, norm_max):
        """Called when user drags yellow Min/Max bars on the TF widget (0..1)."""
        data_min = getattr(self, '_volume_data_min', float(self.view_3d_min_spin.minimum()))
        data_max = getattr(self, '_volume_data_max', float(self.view_3d_max_spin.maximum()))
        rng = data_max - data_min
        if rng <= 0:
            rng = 1.0
        new_min = int(round(data_min + norm_min * rng))
        new_max = int(round(data_min + norm_max * rng))
        self.view_3d_min_spin.blockSignals(True)
        self.view_3d_max_spin.blockSignals(True)
        self.view_3d_min_spin.setValue(new_min)
        self.view_3d_max_spin.setValue(new_max)
        self.view_3d_min_spin.blockSignals(False)
        self.view_3d_max_spin.blockSignals(False)
        self._update_wl_readout()
        self.update_transfer_function()

    def _on_wl_spin_changed(self):
        """Called when user edits Min/Max spinboxes."""
        v_min = self.view_3d_min_spin.value()
        v_max = self.view_3d_max_spin.value()
        data_min = getattr(self, '_volume_data_min', float(self.view_3d_min_spin.minimum()))
        data_max = getattr(self, '_volume_data_max', float(self.view_3d_max_spin.maximum()))
        rng = data_max - data_min
        if rng > 0 and hasattr(self, 'tf_widget'):
            self.tf_widget.set_range((v_min - data_min) / rng,
                                     (v_max - data_min) / rng)
        self._update_wl_readout()
        self.update_transfer_function()

    # ── MPR Window/Level (right sidebar, linked to all 3 planes) ─────────────

    def _on_mpr_wl_hist_changed(self, v_min, v_max):
        """Histogram Min/Max markers dragged → update spins + apply to MPR."""
        if not hasattr(self, 'mpr_wl_min_spin'):
            return
        self.mpr_wl_min_spin.blockSignals(True)
        self.mpr_wl_max_spin.blockSignals(True)
        self.mpr_wl_min_spin.setValue(int(v_min))
        self.mpr_wl_max_spin.setValue(int(v_max))
        self.mpr_wl_min_spin.blockSignals(False)
        self.mpr_wl_max_spin.blockSignals(False)
        self._apply_mpr_window_level(int(v_min), int(v_max))

    def _on_mpr_wl_spin_changed(self):
        """Min/Max spinboxes edited → sync histogram + apply to MPR."""
        if not hasattr(self, 'mpr_wl_min_spin'):
            return
        v_min = self.mpr_wl_min_spin.value()
        v_max = self.mpr_wl_max_spin.value()
        if v_max <= v_min:
            v_max = v_min + 1
            self.mpr_wl_max_spin.blockSignals(True)
            self.mpr_wl_max_spin.setValue(v_max)
            self.mpr_wl_max_spin.blockSignals(False)
        if hasattr(self, 'mpr_hist_widget'):
            self.mpr_hist_widget.blockSignals(True)
            self.mpr_hist_widget.set_range(v_min, v_max)
            self.mpr_hist_widget.blockSignals(False)
        self._apply_mpr_window_level(v_min, v_max)

    def _update_mpr_wl_readout(self, v_min=None, v_max=None):
        if not hasattr(self, 'mpr_wl_info_label'):
            return
        if v_min is None:
            v_min = self.mpr_wl_min_spin.value() if hasattr(self, 'mpr_wl_min_spin') else 0
        if v_max is None:
            v_max = self.mpr_wl_max_spin.value() if hasattr(self, 'mpr_wl_max_spin') else 65535
        w = max(1, int(v_max) - int(v_min))
        l = (float(v_max) + float(v_min)) / 2.0
        self.mpr_wl_info_label.setText(f"W: {w}   L: {l:.1f}")

    def _mpr_needs_baked_overlay(self, orientation=None):
        """True when MPR displays CT+mask as pre-baked RGB (W/L cannot be a property tweak)."""
        oris = (orientation,) if orientation else ('axial', 'coronal', 'sagittal')
        has_mask = (
            getattr(self, 'class1_data', None) is not None
            or getattr(self, 'class2_data', None) is not None
            or getattr(self, 'segmentation_data', None) is not None
        )
        if not has_mask:
            return False
        for o in oris:
            try:
                c1 = bool(getattr(self, f'{o}_overlay_c1').isChecked())
                c2 = bool(getattr(self, f'{o}_overlay_c2').isChecked())
                op = int(getattr(self, f'{o}_opacity_slider').value())
            except Exception:
                continue
            if (c1 or c2) and op > 0:
                return True
        return False

    def _apply_mpr_window_level(self, v_min, v_max):
        """Push Window/Level to all three MPR planes (linked).

        Fast path: only update vtkImageProperty on cached *grayscale* actors.
        When C1/C2 overlay is baked into RGB pixels, must re-render slices.
        """
        v_min = int(v_min)
        v_max = int(v_max)
        if v_max <= v_min:
            v_max = v_min + 1
        window = float(v_max - v_min)
        level = (float(v_max) + float(v_min)) / 2.0

        for orientation in ('axial', 'coronal', 'sagittal'):
            self.window_level[orientation] = (window, level)

        self._update_mpr_wl_readout(v_min, v_max)

        cache = getattr(self, '_cached_slice_actors', None) or {}

        # Baked RGB overlay mode → full re-compose with new W/L
        if self._mpr_needs_baked_overlay() and self.volume_data is not None:
            for orientation in ('axial', 'coronal', 'sagittal'):
                self.render_slice(orientation, preserve_camera=True)
            return

        # Also re-render if any cached actor is still tagged as rgb_overlay
        if any(
            (cache.get(o) or {}).get('display_mode') == 'rgb_overlay'
            for o in ('axial', 'coronal', 'sagittal')
        ) and self.volume_data is not None:
            for orientation in ('axial', 'coronal', 'sagittal'):
                self.render_slice(orientation, preserve_camera=True)
            return

        any_fast = False
        for orientation in ('axial', 'coronal', 'sagittal'):
            entry = cache.get(orientation)
            if not entry or 'image_actor' not in entry:
                continue
            if entry.get('display_mode') == 'rgb_overlay':
                continue
            try:
                prop = entry['image_actor'].GetProperty()
                prop.SetColorWindow(window)
                prop.SetColorLevel(level)
                widget = getattr(self, f'{orientation}_widget', None)
                if widget is not None:
                    rw = widget.GetRenderWindow()
                    if rw is not None:
                        rw.Render()
                any_fast = True
            except Exception:
                pass

        # Actors not cached yet → full render
        if not any_fast and self.volume_data is not None:
            for orientation in ('axial', 'coronal', 'sagittal'):
                try:
                    self.render_slice(orientation, preserve_camera=True)
                except Exception:
                    pass

    def _sync_mpr_wl_ui(self, v_min, v_max, data_min=None, data_max=None):
        """Update MPR W/L controls without re-emitting apply (used on volume load)."""
        v_min, v_max = int(v_min), int(v_max)
        if data_min is None:
            data_min = getattr(self, '_volume_data_min', v_min)
        if data_max is None:
            data_max = getattr(self, '_volume_data_max', v_max)
        imin, imax = int(data_min), int(data_max)
        if imax <= imin:
            imax = imin + 1

        if hasattr(self, 'mpr_wl_min_spin'):
            self.mpr_wl_min_spin.blockSignals(True)
            self.mpr_wl_max_spin.blockSignals(True)
            self.mpr_wl_min_spin.setRange(imin, imax)
            self.mpr_wl_max_spin.setRange(imin, imax)
            self.mpr_wl_min_spin.setValue(max(imin, min(v_min, imax)))
            self.mpr_wl_max_spin.setValue(max(imin, min(v_max, imax)))
            self.mpr_wl_min_spin.blockSignals(False)
            self.mpr_wl_max_spin.blockSignals(False)

        if hasattr(self, 'mpr_hist_widget'):
            self.mpr_hist_widget.blockSignals(True)
            self.mpr_hist_widget.set_data_range(imin, imax)
            self.mpr_hist_widget.set_range(v_min, v_max)
            self.mpr_hist_widget.blockSignals(False)

        self._update_mpr_wl_readout(v_min, v_max)

    def _reset_mpr_wl_to_data_range(self):
        """Reset MPR W/L selected range to full volume data range."""
        dmin = int(getattr(self, '_volume_data_min', 0))
        dmax = int(getattr(self, '_volume_data_max', 65535))
        if dmax <= dmin:
            dmax = dmin + 1
        if hasattr(self, 'mpr_wl_min_spin'):
            self.mpr_wl_min_spin.blockSignals(True)
            self.mpr_wl_max_spin.blockSignals(True)
            self.mpr_wl_min_spin.setValue(dmin)
            self.mpr_wl_max_spin.setValue(dmax)
            self.mpr_wl_min_spin.blockSignals(False)
            self.mpr_wl_max_spin.blockSignals(False)
        if hasattr(self, 'mpr_hist_widget'):
            self.mpr_hist_widget.blockSignals(True)
            self.mpr_hist_widget.set_range(dmin, dmax)
            self.mpr_hist_widget.blockSignals(False)
        self._apply_mpr_window_level(dmin, dmax)

    def _update_wl_readout(self):
        """Update the W/L readout label."""
        v_min = self.view_3d_min_spin.value()
        v_max = self.view_3d_max_spin.value()
        w = v_max - v_min
        l = (v_max + v_min) / 2.0
        if hasattr(self, '_wl_info_label'):
            self._wl_info_label.setText(f"W: {w}   L: {l:.1f}")

    def _set_data_range_ui(self, data_min, data_max):
        """Sync Plotted range / Data range widgets after volume load."""
        self._volume_data_min = float(data_min)
        self._volume_data_max = float(data_max)
        if hasattr(self, '_data_range_min_spin'):
            self._data_range_min_spin.blockSignals(True)
            self._data_range_max_spin.blockSignals(True)
            self._data_range_min_spin.setValue(self._volume_data_min)
            self._data_range_max_spin.setValue(self._volume_data_max)
            self._data_range_min_spin.blockSignals(False)
            self._data_range_max_spin.blockSignals(False)
        if hasattr(self, '_data_extent_label'):
            self._data_extent_label.setText(
                f"{self._volume_data_min:.2f}  \u2014  {self._volume_data_max:.2f}"
            )
        # Spin box hard limits track full data extent
        imin, imax = int(data_min), int(data_max)
        if imax <= imin:
            imax = imin + 1
        self.view_3d_min_spin.setRange(imin, imax)
        self.view_3d_max_spin.setRange(imin, imax)

    def _reset_wl_to_data_range(self):
        """Reset selected range to full data range (Dragonfly Reset)."""
        dmin = int(getattr(self, '_volume_data_min', 0))
        dmax = int(getattr(self, '_volume_data_max', 65535))
        self.view_3d_min_spin.blockSignals(True)
        self.view_3d_max_spin.blockSignals(True)
        self.view_3d_min_spin.setValue(dmin)
        self.view_3d_max_spin.setValue(dmax)
        self.view_3d_min_spin.blockSignals(False)
        self.view_3d_max_spin.blockSignals(False)
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_range(0.0, 1.0)
        self._update_wl_readout()
        self.update_transfer_function()

    def _set_color_invert(self, invert):
        self._color_invert = bool(invert)
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_color_invert(self._color_invert)
        # Refresh icon selected state
        if hasattr(self, 'btn_cmap_normal'):
            self.btn_cmap_normal.setIcon(make_color_map_icon("normal", selected=not invert))
            self.btn_cmap_invert.setIcon(make_color_map_icon("invert", selected=invert))
        self.update_transfer_function()

    def _set_opacity_shape(self, shape):
        self._opacity_shape = shape if shape in OPACITY_SHAPES else "linear"
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_opacity_shape(self._opacity_shape)
        # Highlight selected icon
        if hasattr(self, '_opacity_shape_btns'):
            for s, btn in self._opacity_shape_btns.items():
                btn.setIcon(make_opacity_shape_icon(s, selected=(s == self._opacity_shape)))
                btn.setChecked(s == self._opacity_shape)
        self.update_transfer_function()

    def _on_wl_opacity_changed(self, val):
        scale = val / 100.0
        if hasattr(self, 'wl_opacity_value_lbl'):
            self.wl_opacity_value_lbl.setText(f"{scale:.2f}")
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_opacity_scale(scale)
        # Keep lighting-panel opacity in sync if present
        if hasattr(self, 'view_3d_vol_opacity_slider'):
            self.view_3d_vol_opacity_slider.blockSignals(True)
            self.view_3d_vol_opacity_slider.setValue(val)
            self.view_3d_vol_opacity_slider.blockSignals(False)
            if hasattr(self, 'view_3d_vol_opacity_label'):
                self.view_3d_vol_opacity_label.setText(f"{scale:.2f}")
        self.update_transfer_function()

    def _on_wl_gamma_changed(self, val):
        gamma = val / 100.0
        if hasattr(self, 'wl_gamma_spin'):
            self.wl_gamma_spin.blockSignals(True)
            self.wl_gamma_spin.setValue(gamma)
            self.wl_gamma_spin.blockSignals(False)
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_opacity_gamma(gamma)
        self.update_transfer_function()

    def _on_wl_gamma_spin_changed(self, gamma):
        if hasattr(self, 'wl_gamma_slider'):
            self.wl_gamma_slider.blockSignals(True)
            self.wl_gamma_slider.setValue(int(round(gamma * 100)))
            self.wl_gamma_slider.blockSignals(False)
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_opacity_gamma(gamma)
        self.update_transfer_function()
