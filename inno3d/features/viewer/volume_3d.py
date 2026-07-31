# inno3d/features/viewer/volume_3d.py
# ─────────────────────────────────────────────────────────────────────────────
# Volume3dMixin — extracted from inno3d/tabs/viewer.py (Phase 3.4)
#
# Contains all 3D volume rendering, lighting, transfer function, orientation
# marker, LOD system, and 3D UI creation methods.
#
# MOVE not COPY:  after this extract the same methods are DELETED from viewer.py.
# MRO target:
#   MultiPlanarView(Volume3dMixin, MprInputMixin, MprRenderMixin,
#                   MprNavMixin, CrosshairMixin, QWidget)
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QColor, QIcon, QPixmap, QPainter
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QToolButton, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QSizePolicy, QColorDialog, QGroupBox,
    QButtonGroup, QScrollArea, QFrame, QAbstractSpinBox,
    QGraphicsDropShadowEffect,
)
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from inno3d.core.styles import SemiconductorTheme
from inno3d.core.tf_widgets import (
    ColorGradientWidget,
    TransferFunctionWidget,
    OPACITY_SHAPES,
    OPACITY_SHAPE_TIPS,
    make_opacity_shape_icon,
    make_color_map_icon,
    make_icon_toolbutton,
    sample_opacity_curve,
)
from inno3d.core.view_support import (
    Dragonfly3DInteractorStyle,
    SliceControlOverlay,
    apply_dragonfly_volume_zoom,
    push_camera_outside_aabb,
    volume_world_aabb,
)


# ── Module-level helper (mirrors viewer._ui_icon; avoids circular import) ────

def _ui_icon_3d(name, widget=None, fallback=None):
    """Resolve an icon from assets/icons/ relative to the package root."""
    icon_path = Path(__file__).resolve().parents[3] / "assets" / "icons" / name
    if icon_path.exists():
        return QIcon(str(icon_path))
    if widget is not None and fallback is not None:
        return widget.style().standardIcon(fallback)
    return QIcon()


# ─────────────────────────────────────────────────────────────────────────────
class Volume3dMixin:
    """Mixin: 3-D volume rendering, transfer function, orientation marker, LOD.

    Extracted from ``inno3d/tabs/viewer.py`` Phase 3.4.
    All methods use ``self.*`` attributes set by MultiPlanarView.__init__ /
    create_3d_view; cross-mixin calls (clip planes, MES highlights, …) resolve
    via the normal MRO of the concrete MultiPlanarView class.
    """

    # ── Dragonfly-style LUT Presets ──────────────────────────────────────────

    _TF_PRESETS = {
        "Default 3D": {
            "opacity": [(0.0, 0.0), (0.2, 0.0), (0.5, 0.3), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 0)), (0.3, (100, 100, 90)), (0.6, (200, 200, 180)),
                (1.0, (255, 255, 240)),
            ],
        },
        "Discrete 24": {
            "opacity": [(0.0, 0.0), (1.0, 1.0)],
            "colors": [
                (0.0, (255, 0, 0)), (0.16, (255, 255, 0)), (0.33, (0, 255, 0)),
                (0.5, (0, 255, 255)), (0.66, (0, 0, 255)), (0.83, (255, 0, 255)),
                (1.0, (255, 0, 0)),
            ],
        },
        "Discrete 64": {
            "opacity": [(0.0, 0.0), (1.0, 1.0)],
            "colors": [
                (0.0, (255, 0, 0)), (0.2, (255, 255, 0)), (0.4, (0, 255, 0)),
                (0.6, (0, 255, 255)), (0.8, (0, 0, 255)), (1.0, (255, 0, 255)),
            ],
        },
        "Electronics": {
            "opacity": [(0.0, 0.0), (0.2, 0.0), (0.5, 0.4), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 0)), (0.3, (80, 40, 20)), (0.6, (180, 100, 50)),
                (0.8, (230, 180, 120)), (1.0, (255, 255, 255)),
            ],
        },
        "Entomology": {
            "opacity": [(0.0, 0.0), (0.15, 0.0), (0.5, 0.5), (1.0, 0.95)],
            "colors": [
                (0.0, (0, 0, 0)), (0.3, (60, 30, 0)), (0.6, (180, 100, 20)),
                (0.8, (230, 160, 50)), (1.0, (255, 240, 200)),
            ],
        },
        "Fire": {
            "opacity": [(0.0, 0.0), (0.2, 0.05), (0.5, 0.4), (0.8, 0.8), (1.0, 1.0)],
            "colors": [
                (0.0, (0, 0, 0)), (0.33, (255, 0, 0)), (0.66, (255, 255, 0)),
                (1.0, (255, 255, 255)),
            ],
        },
        "Grayscale Solid": {
            "opacity": [(0.0, 0.0), (0.1, 0.1), (0.5, 0.5), (1.0, 1.0)],
            "colors": [
                (0.0, (64, 64, 64)), (0.5, (160, 160, 160)), (1.0, (255, 255, 255)),
            ],
        },
        "Grayscale Solid2": {
            "opacity": [(0.0, 0.0), (0.1, 0.2), (0.5, 0.6), (1.0, 1.0)],
            "colors": [
                (0.0, (128, 128, 128)), (0.5, (192, 192, 192)), (1.0, (255, 255, 255)),
            ],
        },
        "Green": {
            "opacity": [(0.0, 0.0), (0.2, 0.0), (0.5, 0.3), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 0)), (0.4, (0, 128, 0)), (0.7, (0, 200, 50)),
                (1.0, (200, 255, 200)),
            ],
        },
        "Grayscale": {
            "opacity": [(0.0, 0.0), (0.15, 0.0), (0.3, 0.08), (0.5, 0.25), (0.75, 0.6), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 0)), (0.25, (64, 64, 64)), (0.5, (128, 128, 128)),
                (0.75, (192, 192, 192)), (1.0, (255, 255, 255)),
            ],
        },
        "Steel": {
            "opacity": [(0.0, 0.0), (0.12, 0.0), (0.3, 0.06), (0.5, 0.22), (0.75, 0.6), (1.0, 0.92)],
            "colors": [
                (0.0, (0, 0, 0)), (0.15, (10, 18, 35)), (0.35, (40, 70, 110)),
                (0.55, (100, 140, 180)), (0.75, (170, 195, 215)),
                (0.9, (215, 225, 240)), (1.0, (240, 245, 255)),
            ],
        },
        "Steel2": {
            "opacity": [(0.0, 0.0), (0.1, 0.0), (0.25, 0.04), (0.45, 0.2), (0.7, 0.55), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 5)), (0.2, (25, 35, 55)), (0.4, (65, 90, 125)),
                (0.6, (120, 150, 180)), (0.8, (185, 200, 220)),
                (1.0, (230, 238, 248)),
            ],
        },
        "Temperature": {
            "opacity": [(0.0, 0.0), (0.1, 0.0), (0.25, 0.05), (0.5, 0.25), (0.75, 0.6), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 0)), (0.15, (10, 0, 40)), (0.3, (60, 0, 120)),
                (0.45, (180, 0, 80)), (0.6, (230, 60, 10)), (0.75, (250, 170, 0)),
                (0.9, (255, 240, 100)), (1.0, (255, 255, 220)),
            ],
        },
        "Thermal": {
            "opacity": [(0.0, 0.0), (0.1, 0.0), (0.25, 0.04), (0.5, 0.2), (0.75, 0.55), (1.0, 0.88)],
            "colors": [
                (0.0, (0, 0, 0)), (0.2, (20, 0, 80)), (0.35, (80, 0, 160)),
                (0.5, (200, 30, 30)), (0.65, (240, 120, 0)),
                (0.8, (255, 220, 50)), (1.0, (255, 255, 200)),
            ],
        },
        "Vegetal": {
            "opacity": [(0.0, 0.0), (0.12, 0.0), (0.3, 0.06), (0.5, 0.2), (0.75, 0.55), (1.0, 0.85)],
            "colors": [
                (0.0, (0, 0, 0)), (0.15, (5, 20, 5)), (0.3, (15, 65, 15)),
                (0.5, (40, 140, 30)), (0.7, (100, 200, 60)),
                (0.85, (180, 235, 120)), (1.0, (230, 255, 200)),
            ],
        },
        "Vegetal2": {
            "opacity": [(0.0, 0.0), (0.1, 0.0), (0.25, 0.05), (0.5, 0.22), (0.75, 0.58), (1.0, 0.88)],
            "colors": [
                (0.0, (0, 0, 0)), (0.2, (10, 30, 8)), (0.4, (30, 100, 25)),
                (0.6, (80, 170, 50)), (0.8, (160, 220, 100)),
                (1.0, (220, 250, 180)),
            ],
        },
        "Warm Metal": {
            "opacity": [(0.0, 0.0), (0.1, 0.0), (0.25, 0.03), (0.4, 0.12), (0.6, 0.35), (0.8, 0.65), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 0)), (0.15, (20, 5, 0)), (0.3, (80, 25, 0)),
                (0.45, (160, 60, 5)), (0.6, (220, 120, 15)),
                (0.75, (245, 190, 50)), (0.9, (255, 230, 120)),
                (1.0, (255, 250, 210)),
            ],
        },
        "White": {
            "opacity": [(0.0, 0.0), (0.15, 0.0), (0.3, 0.1), (0.5, 0.35), (0.75, 0.7), (1.0, 1.0)],
            "colors": [
                (0.0, (0, 0, 0)), (0.3, (80, 80, 80)), (0.5, (160, 160, 160)),
                (0.7, (220, 220, 220)), (1.0, (255, 255, 255)),
            ],
        },
        "Wood4": {
            "opacity": [(0.0, 0.0), (0.1, 0.0), (0.25, 0.04), (0.45, 0.18), (0.65, 0.45), (0.85, 0.75), (1.0, 0.92)],
            "colors": [
                (0.0, (0, 0, 0)), (0.15, (25, 12, 5)), (0.3, (80, 45, 15)),
                (0.5, (160, 100, 35)), (0.65, (200, 145, 60)),
                (0.8, (230, 190, 100)), (0.95, (250, 225, 160)),
                (1.0, (255, 245, 210)),
            ],
        },
        "Yellow": {
            "opacity": [(0.0, 0.0), (0.12, 0.0), (0.3, 0.06), (0.5, 0.22), (0.75, 0.58), (1.0, 0.9)],
            "colors": [
                (0.0, (0, 0, 0)), (0.2, (30, 25, 0)), (0.4, (120, 100, 0)),
                (0.6, (200, 180, 10)), (0.8, (240, 225, 60)),
                (1.0, (255, 255, 150)),
            ],
        },
    }

    # ── Group A: 3D Volume Enable/Disable ────────────────────────────────────

    def is_3d_volume_render_enabled(self) -> bool:
        return bool(getattr(self, "_3d_volume_render_enabled", True))

    def _on_3d_render_btn_toggled(self, checked):
        """User toggled the 3D strip button under C2."""
        if getattr(self, "_online_viewer_mode", False):
            # Remember explicit choice during Online session
            self._3d_render_force_on_online = bool(checked)
        self.set_3d_volume_render_enabled(bool(checked), sync_button=False)

    def set_3d_volume_render_enabled(self, enabled, sync_button=True):
        """Enable/disable GPU 3D volume + C1/C2 3D overlays (keeps MPR free).

        When OFF: tear down volume actors (VRAM) and show a short placeholder.
        When ON: full ``render_3d()`` if volume is loaded.
        """
        enabled = bool(enabled)
        prev = bool(getattr(self, "_3d_volume_render_enabled", True))
        self._3d_volume_render_enabled = enabled

        btn = None
        og = getattr(self, "view_3d_overlay_group", None)
        if og is not None:
            btn = getattr(og, "render_3d_btn", None)
        if sync_button and btn is not None:
            btn.blockSignals(True)
            btn.setChecked(enabled)
            btn.blockSignals(False)

        # C1/C2 3D overlays only make sense when volume render is ON
        if og is not None:
            for name in ("c1_btn", "c2_btn", "opacity_slider"):
                w = getattr(og, name, None)
                if w is not None:
                    w.setEnabled(enabled)

        if enabled == prev and enabled and getattr(self, "volume_actor", None) is not None:
            return

        if not enabled:
            self._teardown_3d_volume_actors()
            self._show_3d_disabled_placeholder()
            # Crosshair is independent of GPU volume — restore after teardown
            if getattr(self, "crosshair_enabled", False) and hasattr(self, "update_3d_crosshair"):
                try:
                    self.update_3d_crosshair()
                except Exception:
                    pass
            return

        self._hide_3d_disabled_placeholder()
        if self.volume_data is not None:
            # Force camera reset when re-enabling — the camera was set by
            # crosshair-only code while volume was OFF (no AABB context).
            self._camera_initialized = False
            self.render_3d()
            if getattr(self, "crosshair_enabled", False) and hasattr(self, "update_3d_crosshair"):
                try:
                    self.update_3d_crosshair()
                except Exception:
                    pass

    def _teardown_3d_volume_actors(self):
        """Remove GPU volumes / MES surfaces to free VRAM when 3D is OFF."""
        ren = getattr(self, "view_3d_renderer", None)
        if ren is None:
            return
        if getattr(self, "volume_actor", None) is not None:
            try:
                ren.RemoveVolume(self.volume_actor)
            except Exception:
                pass
            self.volume_actor = None
        from inno3d.features.viewer.seg_mask_3d import remove_mask_overlay_from_renderer
        for name in ("c1_actor_3d", "c2_actor_3d"):
            act = getattr(self, name, None)
            if act is not None:
                remove_mask_overlay_from_renderer(ren, act)
                setattr(self, name, None)
        self._clear_mes_3d_highlight(render=False)
        # Keep B2B actors? Clear them when 3D off (no volume to attach to)
        self._clear_b2b_gap_actors()
        self._volume_mapper = None
        widget = getattr(self, "view_3d_widget", None)
        if widget is not None:
            try:
                ren.ResetCameraClippingRange()
                widget.GetRenderWindow().Render()
            except Exception:
                pass

    def _show_3d_disabled_placeholder(self):
        """Dark 3D pane message when volume rendering is OFF."""
        ren = getattr(self, "view_3d_renderer", None)
        widget = getattr(self, "view_3d_widget", None)
        if ren is None:
            return
        self._hide_3d_disabled_placeholder()
        try:
            text = vtk.vtkTextActor()
            text.SetInput(
                "3D OFF\n"
                "Toggle  3D  under C2 to render volume\n"
                "(B2B gap / MES pick auto-enable when needed)"
            )
            tp = text.GetTextProperty()
            tp.SetFontSize(16)
            tp.SetBold(True)
            tp.SetColor(0.55, 0.72, 0.85)
            tp.SetJustificationToCentered()
            tp.SetVerticalJustificationToCentered()
            tp.SetBackgroundColor(0.04, 0.06, 0.10)
            tp.SetBackgroundOpacity(0.55)
            # Center in viewport
            try:
                text.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
                text.SetPosition(0.5, 0.52)
            except Exception:
                text.SetDisplayPosition(40, 120)
            ren.AddActor2D(text)
            self._3d_disabled_text_actor = text
            if widget is not None:
                widget.GetRenderWindow().Render()
        except Exception:
            self._3d_disabled_text_actor = None

    def _hide_3d_disabled_placeholder(self):
        ren = getattr(self, "view_3d_renderer", None)
        act = getattr(self, "_3d_disabled_text_actor", None)
        if ren is not None and act is not None:
            try:
                ren.RemoveActor(act)
            except Exception:
                try:
                    ren.RemoveActor2D(act)
                except Exception:
                    pass
        self._3d_disabled_text_actor = None

    def ensure_3d_volume_render_for_overlay(self, reason=""):
        """Turn 3D ON if needed (B2B gap / MES surface) then rebuild volume once.

        Returns True when 3D is ready for overlay actors.
        """
        if self.volume_data is None:
            return False
        if self.is_3d_volume_render_enabled() and getattr(self, "volume_actor", None) is not None:
            return True
        if getattr(self, "_online_viewer_mode", False):
            self._3d_render_force_on_online = True
        self.set_3d_volume_render_enabled(True, sync_button=True)
        return (
            self.is_3d_volume_render_enabled()
            and getattr(self, "volume_actor", None) is not None
        )

    # ── Group B: 3D View UI Creation ─────────────────────────────────────────

    def create_3d_view(self):
        widget = QWidget()
        widget.setObjectName("PlaneCell")
        widget.setStyleSheet(self._plane_cell_stylesheet(active=False))
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        widget.setMinimumSize(0, 0)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- COMPACT HEADER (Matches 2D Views) ---
        header_widget = QWidget()
        header_widget.setMaximumHeight(26)
        header_widget.setStyleSheet(
            "background: rgba(0, 0, 0, 0.45); border: none;"
        )

        controls_layout = QHBoxLayout(header_widget)
        controls_layout.setContentsMargins(8, 0, 8, 0)

        # Title
        title_label = QLabel("<b>3D Volume</b>")
        title_label.setStyleSheet(f"font-size: 9pt; color: {SemiconductorTheme.ACCENT_PRIMARY};")
        controls_layout.addWidget(title_label)

        # Apply shadow effect for readability (matches 2D)
        title_shadow = QGraphicsDropShadowEffect()
        title_shadow.setBlurRadius(3)
        title_shadow.setOffset(1, 1)
        title_shadow.setColor(QColor(0, 0, 0, 220))
        title_label.setGraphicsEffect(title_shadow)

        controls_layout.addStretch()

        # Pixel info (empty for 3D)
        pixel_label = QLabel("")
        pixel_label.setStyleSheet("font-size: 8pt; color: white;")
        controls_layout.addWidget(pixel_label)

        self.controls_3d_container = header_widget
        layout.addWidget(self.controls_3d_container)

        # --- MAIN CONTENT DIVIDER ---
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(content_layout)
        self._3d_content_layout = content_layout  # keep ref for exit_fullscreen

        # --- LEFT: VTK WIDGET (75%) ---
        vtk_container = QWidget()
        vtk_container.setObjectName("VTK3DContainer")
        vtk_container.setStyleSheet("#VTK3DContainer { background-color: #000000; border: none; }")
        vtk_layout = QVBoxLayout(vtk_container)
        vtk_layout.setContentsMargins(0, 0, 0, 0)
        vtk_layout.setSpacing(0)

        vtk_widget = QVTKRenderWindowInteractor()
        renderer = vtk.vtkRenderer()

        # --- 1. Dragonfly-style Gradient Background ---
        try:
            # Try native radial gradient (VTK 9.3+)
            renderer.GradientBackgroundOn()
            renderer.SetGradientMode(renderer.VTK_GRADIENT_RADIAL_FARTHEST_CORNER)
            renderer.SetBackground(0.85, 0.85, 0.85)  # Center: Very Bright
            renderer.SetBackground2(0.15, 0.15, 0.15)  # Edges: Dark Gray
        except AttributeError:
            # Fallback to Textured Background (VTK < 9.3)
            try:
                dim = 256
                xx, yy = np.meshgrid(np.arange(dim), np.arange(dim))
                center = dim / 2.0
                r = np.sqrt((xx - center)**2 + (yy - center)**2)
                r_norm = r / (dim / 2.0)
                factor = np.exp(-1.8 * (r_norm ** 1.2))

                c_color = np.array([240, 240, 240])
                e_color = np.array([30, 30, 30])

                img_array = (c_color[np.newaxis, np.newaxis, :] * factor[:, :, np.newaxis] +
                             e_color[np.newaxis, np.newaxis, :] * (1.0 - factor[:, :, np.newaxis]))
                img_array = img_array.astype(np.uint8)

                image_data = vtk.vtkImageData()
                image_data.SetDimensions(dim, dim, 1)
                vtk_array = numpy_support.numpy_to_vtk(
                    img_array.reshape(-1, 3), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
                )
                image_data.GetPointData().SetScalars(vtk_array)

                bg_texture = vtk.vtkTexture()
                bg_texture.SetInputData(image_data)

                renderer.SetBackgroundTexture(bg_texture)
                renderer.SetTexturedBackground(True)
                self.bg_texture = bg_texture
            except Exception:
                renderer.GradientBackgroundOn()
                renderer.SetBackground(0.15, 0.15, 0.15)
                renderer.SetBackground2(0.85, 0.85, 0.85)

        vtk_widget.GetRenderWindow().AddRenderer(renderer)

        # --- 2. Soft three-point lighting (Dragonfly-like) ---
        self.light_kit = vtk.vtkLightKit()
        self.light_kit.AddLightsToRenderer(renderer)
        self.light_kit.SetKeyLightIntensity(0.72)
        self.light_kit.SetKeyToFillRatio(1.8)
        self.light_kit.SetKeyLightWarmth(0.55)
        try:
            self.light_kit.SetKeyLightElevation(45.0)
            self.light_kit.SetKeyLightAzimuth(10.0)
            self.light_kit.SetFillLightWarmth(0.45)
            self.light_kit.SetHeadLightWarmth(0.5)
            self.light_kit.SetKeyToHeadRatio(2.2)
            self.light_kit.SetKeyToBackRatio(3.5)
        except Exception:
            pass

        # --- RTX 5090 Performance Optimizations ---
        render_window = vtk_widget.GetRenderWindow()
        render_window.SetDesiredUpdateRate(60.0)
        if hasattr(render_window, 'SetStillUpdateRate'):
            render_window.SetStillUpdateRate(0.001)
        if hasattr(render_window, 'SetMultiSamples'):
            render_window.SetMultiSamples(4)
        renderer.SetUseDepthPeeling(1)
        renderer.SetMaximumNumberOfPeels(4)
        renderer.SetOcclusionRatio(0.1)

        # Dragonfly-style 3D navigation
        style = Dragonfly3DInteractorStyle()
        style._host = self
        interactor = vtk_widget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(style)
        self._3d_interactor_style = style

        # LOD Interaction Observers
        def _on_interaction_start(caller, event):
            self._start_lod_interaction()

        def _on_interaction_end(caller, event):
            self._end_lod_interaction()

        interactor.AddObserver('StartInteractionEvent', _on_interaction_start)
        interactor.AddObserver('EndInteractionEvent', _on_interaction_end)

        # --- 3. Orientation Marker (shared interactive cube) ---
        from inno3d.features.shared.orientation_cube import build_orientation_marker

        self._ori_hover_face = None
        marker, ori_state = build_orientation_marker()
        self._ori_cube_state = ori_state
        self._axes_cube_actor = ori_state.get("cube")
        self._ori_face_highlights = ori_state.get("face_highlights") or {}
        self._ori_face_text_props = ori_state.get("face_text_props") or {}

        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(marker)
        self.axes_widget.SetInteractor(interactor)
        self.axes_widget.SetViewport(0.82, 0.0, 1.0, 0.18)
        self.axes_widget.EnabledOn()
        self.axes_widget.InteractiveOff()

        interactor.AddObserver(
            "LeftButtonPressEvent", self._on_orientation_cube_click, 10.0
        )
        interactor.AddObserver(
            "MouseMoveEvent", self._on_orientation_cube_hover, 5.0
        )

        # Dragonfly clip-box face interaction
        interactor.AddObserver(
            "LeftButtonPressEvent", self._on_df_clip_3d_left_press, 25.0
        )
        interactor.AddObserver(
            "MouseMoveEvent", self._on_df_clip_3d_mouse_move, 25.0
        )
        interactor.AddObserver(
            "LeftButtonReleaseEvent", self._on_df_clip_3d_left_release, 25.0
        )
        self._df_clip_3d_face_actors = {}

        vtk_layout.addWidget(vtk_widget)

        # --- VERTICAL OVERLAY (Matches 2D Views) ---
        on_fs = lambda checked=False: self.toggle_view_fullscreen('view_3d')
        on_reset = lambda checked=False: self.reset_3d_view()

        self.view_3d_overlay_group = SliceControlOverlay(vtk_widget, '3d', on_fs, on_reset)
        # Master 3D render switch. Default ON offline; Online auto-OFF.
        self._3d_volume_render_enabled = True
        self._3d_render_force_on_online = False
        self._3d_disabled_text_actor = None
        if hasattr(self.view_3d_overlay_group, 'c1_btn'):
            self.view_3d_overlay_group.c1_btn.toggled.connect(
                lambda _c=False: self.update_overlay_visibility('3d')
            )
        if hasattr(self.view_3d_overlay_group, 'c2_btn'):
            self.view_3d_overlay_group.c2_btn.toggled.connect(
                lambda _c=False: self.update_overlay_visibility('3d')
            )
        if hasattr(self.view_3d_overlay_group, 'opacity_slider'):
            self.view_3d_overlay_group.opacity_slider.valueChanged.connect(
                lambda _v=0: self.update_overlay_visibility('3d')
            )
        if getattr(self.view_3d_overlay_group, "render_3d_btn", None) is not None:
            self.view_3d_overlay_group.render_3d_btn.toggled.connect(
                self._on_3d_render_btn_toggled
            )

        QTimer.singleShot(100, self.view_3d_overlay_group.update_position)
        content_layout.addWidget(vtk_container, 1)

        self.view_3d_widget = vtk_widget
        self.view_3d_renderer = renderer
        self._vtk_3d_container = vtk_container
        vtk_widget.installEventFilter(self)
        vtk_container.installEventFilter(self)
        vtk_widget.setMouseTracking(True)
        vtk_container.setMouseTracking(True)
        vtk_widget.setFocusPolicy(Qt.StrongFocus)
        vtk_container.setFocusPolicy(Qt.StrongFocus)

        # --- RIGHT: ADVANCED CONTROLS ---
        self.advanced_3d_controls = QWidget()

        adv_main_layout = QVBoxLayout()
        adv_main_layout.setContentsMargins(10, 10, 10, 10)
        adv_main_layout.setSpacing(15)

        controls_panel = QVBoxLayout()
        controls_panel.setSpacing(10)

        # --- Rendering Quality Group ---
        qual_group = QGroupBox("Rendering Quality")
        q_lay = QVBoxLayout()

        color_row = QHBoxLayout()
        self.color_btn = QPushButton("Volume Color")
        self.color_btn.clicked.connect(self.choose_volume_color)
        color_row.addWidget(self.color_btn)
        self.view_3d_bg_color_btn = QPushButton("Background Color")
        self.view_3d_bg_color_btn.clicked.connect(self.choose_3d_bg_color)
        color_row.addWidget(self.view_3d_bg_color_btn)
        q_lay.addLayout(color_row)

        self._quality_presets = {
            "Draft":    {"sample_dist": 2.0,  "image_sample": 1.0, "interact_image_sample": 3.0, "interact_sample_dist": 4.0, "auto_adj": True,  "jitter": False, "max_mem_fraction": 0.75},
            "Standard": {"sample_dist": 1.0,  "image_sample": 1.0, "interact_image_sample": 2.0, "interact_sample_dist": 3.0, "auto_adj": True,  "jitter": True,  "max_mem_fraction": 0.80},
            "High":     {"sample_dist": 0.5,  "image_sample": 1.0, "interact_image_sample": 1.5, "interact_sample_dist": 2.0, "auto_adj": True,  "jitter": True,  "max_mem_fraction": 0.85},
            "Ultra":    {"sample_dist": 0.25, "image_sample": 0.75, "interact_image_sample": 1.25, "interact_sample_dist": 1.5, "auto_adj": True,  "jitter": True,  "max_mem_fraction": 0.90},
        }
        self._current_quality = "High"

        self._lighting_look_presets = {
            "Inno3D": {
                "shade": True, "ambient": 40, "diffuse": 58, "specular": 18, "power": 16,
                "edge": 22, "opacity": 100,
                "key_intensity": 0.72, "key_fill": 1.8, "key_warmth": 0.55,
                "vol_shadows": False,
            },
            "Inspection": {
                "shade": True, "ambient": 48, "diffuse": 45, "specular": 8, "power": 10,
                "edge": 15, "opacity": 100,
                "key_intensity": 0.65, "key_fill": 1.5, "key_warmth": 0.5,
                "vol_shadows": False,
            },
            "Plastic": {
                "shade": True, "ambient": 25, "diffuse": 85, "specular": 80, "power": 50,
                "edge": 0, "opacity": 100,
                "key_intensity": 0.85, "key_fill": 2.5, "key_warmth": 0.6,
                "vol_shadows": True,
            },
            "Flat": {
                "shade": False, "ambient": 100, "diffuse": 0, "specular": 0, "power": 1,
                "edge": 0, "opacity": 100,
                "key_intensity": 0.7, "key_fill": 1.5, "key_warmth": 0.5,
                "vol_shadows": False,
            },
        }
        # Alias: older sessions / docs may still say "Dragonfly"
        self._lighting_look_presets["Dragonfly"] = self._lighting_look_presets["Inno3D"]
        self._current_lighting_look = "Inno3D"

        # LOD refinement timer
        self._lod_refine_timer = QTimer(self)
        self._lod_refine_timer.setSingleShot(True)
        self._lod_refine_timer.setInterval(300)
        self._lod_refine_timer.timeout.connect(self._refine_after_interaction)
        self._is_interacting_3d = False

        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Draft", "Standard", "High", "Ultra"])
        self.quality_combo.setCurrentText("High")
        self.quality_combo.setToolTip(
            "Draft: Fast preview, lower detail (good for very large volumes)\n"
            "Standard: Balanced quality and performance\n"
            "High: High quality (recommended)\n"
            "Ultra: Dragonfly-like still quality (denser rays + mild supersample)"
        )
        self.quality_combo.currentTextChanged.connect(self._on_quality_preset_changed)
        quality_row.addWidget(self.quality_combo)
        q_lay.addLayout(quality_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(["Composite", "MIP", "MinIP", "Average", "Additive"])
        self.render_mode_combo.setToolTip(
            "Composite: Standard volume rendering\n"
            "MIP: Maximum Intensity Projection\n"
            "MinIP: Minimum Intensity Projection\n"
            "Average: Average Intensity\n"
            "Additive: Additive (X-Ray like)"
        )
        self.render_mode_combo.currentTextChanged.connect(self.on_render_mode_changed)
        mode_row.addWidget(self.render_mode_combo)
        q_lay.addLayout(mode_row)

        downsample_row = QHBoxLayout()
        downsample_row.addWidget(QLabel("Downsample:"))
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 8)
        self.downsample_spin.setValue(2)
        downsample_row.addWidget(self.downsample_spin)
        q_lay.addLayout(downsample_row)

        qual_group.setLayout(q_lay)
        controls_panel.addWidget(qual_group)

        # ═══════════════════════════════════════════════════════════════
        # Window Leveling — Dragonfly-style panel
        # ═══════════════════════════════════════════════════════════════
        hist_group = QGroupBox("Window Leveling")
        hist_lay = QVBoxLayout()
        hist_lay.setContentsMargins(6, 10, 6, 6)
        hist_lay.setSpacing(6)

        mode_row2 = QHBoxLayout()
        mode_row2.setSpacing(4)
        self._tf_mode_group = QButtonGroup(self)
        self.btn_wl_mode = QPushButton("W/L")
        self.btn_wl_mode.setCheckable(True)
        self.btn_wl_mode.setChecked(True)
        self.btn_wl_mode.setToolTip(
            "Window Leveling (Dragonfly): Min/Max bars + opacity curve shapes."
        )
        self.btn_tf_mode = QPushButton("TF")
        self.btn_tf_mode.setCheckable(True)
        self.btn_tf_mode.setToolTip(
            "Transfer Function: free opacity spline (click add / drag / right-click delete)."
        )
        for b in (self.btn_wl_mode, self.btn_tf_mode):
            b.setFixedHeight(22)
            b.setStyleSheet(
                "QPushButton { font-size: 8pt; padding: 2px 10px; }"
                "QPushButton:checked { background: #3a6ea5; color: white; "
                "border: 1px solid #5a9ed5; }"
            )
            mode_row2.addWidget(b)
        self._tf_mode_group.addButton(self.btn_wl_mode, 0)
        self._tf_mode_group.addButton(self.btn_tf_mode, 1)
        self._tf_mode_group.setExclusive(True)
        mode_row2.addStretch(1)
        self.tf_log_y_check = QCheckBox("Log Y")
        self.tf_log_y_check.setStyleSheet("color: #a0a0a0; font-size: 8pt;")
        mode_row2.addWidget(self.tf_log_y_check)
        hist_lay.addLayout(mode_row2)

        self.tf_widget = TransferFunctionWidget()
        self.tf_widget.set_mode(TransferFunctionWidget.MODE_WL)
        self.tf_widget.setMinimumHeight(130)
        self.tf_log_y_check.toggled.connect(self.tf_widget.set_log_y)
        self.tf_widget.opacityChanged.connect(self.on_opacity_spline_changed)
        self.tf_widget.rangeChanged.connect(self._on_tf_range_changed)
        self.tf_widget.modeChanged.connect(self._on_tf_mode_changed)
        self.btn_wl_mode.clicked.connect(
            lambda: self.tf_widget.set_mode(TransferFunctionWidget.MODE_WL)
        )
        self.btn_tf_mode.clicked.connect(
            lambda: self.tf_widget.set_mode(TransferFunctionWidget.MODE_TF)
        )
        hist_lay.addWidget(self.tf_widget)

        self.color_grad_widget = ColorGradientWidget()
        self.color_grad_widget.colorChanged.connect(self.on_color_gradient_changed)
        self.color_grad_widget.colorChanged.connect(
            lambda _: (
                self.tf_preset_combo.blockSignals(True),
                self.tf_preset_combo.setCurrentIndex(0),
                self.tf_preset_combo.blockSignals(False),
            )
        )
        hist_lay.addWidget(self.color_grad_widget)

        lut_row = QHBoxLayout()
        lut_lbl = QLabel("Lookup table (LUT)")
        lut_lbl.setStyleSheet("color: #c0c0c0; font-size: 8pt; font-weight: bold;")
        lut_row.addWidget(lut_lbl)
        hist_lay.addLayout(lut_row)

        self.tf_preset_combo = QComboBox()
        self.tf_preset_combo.setIconSize(QSize(72, 14))
        lut_names = [
            "Custom", "Default 3D", "Discrete 24", "Discrete 64", "Electronics",
            "Entomology", "Fire", "Grayscale", "Grayscale Solid", "Grayscale Solid2",
            "Green", "Steel", "Steel2", "Temperature", "Thermal", "Vegetal",
            "Vegetal2", "Warm Metal", "White", "Wood4", "Yellow",
        ]
        for name in lut_names:
            icon = self._make_lut_icon(name, 72, 14)
            self.tf_preset_combo.addItem(icon, name)
        self.tf_preset_combo.setCurrentText("Grayscale")
        self.tf_preset_combo.setToolTip("Dragonfly-style Lookup Table preset")
        self.tf_preset_combo.currentTextChanged.connect(self.apply_tf_preset)
        hist_lay.addWidget(self.tf_preset_combo)

        sel_lbl = QLabel("Selected range")
        sel_lbl.setStyleSheet("color: #c0c0c0; font-size: 8pt; font-weight: bold;")
        hist_lay.addWidget(sel_lbl)

        wl_row = QHBoxLayout()
        self.view_3d_min_spin = QSpinBox()
        self.view_3d_min_spin.setRange(0, 65535)
        self.view_3d_min_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.view_3d_min_spin.setAlignment(Qt.AlignCenter)
        self.view_3d_min_spin.valueChanged.connect(self._on_wl_spin_changed)
        wl_row.addWidget(self.view_3d_min_spin)
        self.view_3d_max_spin = QSpinBox()
        self.view_3d_max_spin.setRange(0, 65535)
        self.view_3d_max_spin.setValue(65535)
        self.view_3d_max_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.view_3d_max_spin.setAlignment(Qt.AlignCenter)
        self.view_3d_max_spin.valueChanged.connect(self._on_wl_spin_changed)
        wl_row.addWidget(self.view_3d_max_spin)
        hist_lay.addLayout(wl_row)

        self._wl_info_label = QLabel("W: 65535   L: 32767.5")
        self._wl_info_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 7pt;"
        )
        self._wl_info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hist_lay.addWidget(self._wl_info_label)

        plot_lbl = QLabel("Plotted range / Data range")
        plot_lbl.setStyleSheet("color: #c0c0c0; font-size: 8pt; font-weight: bold;")
        hist_lay.addWidget(plot_lbl)

        data_row = QHBoxLayout()
        # 1px vertical slack so hover border is not clipped by tight layout
        data_row.setContentsMargins(0, 1, 0, 2)
        self._data_range_min_spin = QDoubleSpinBox()
        self._data_range_min_spin.setDecimals(2)
        self._data_range_min_spin.setRange(-1e9, 1e9)
        self._data_range_min_spin.setReadOnly(True)
        self._data_range_min_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._data_range_min_spin.setAlignment(Qt.AlignCenter)
        data_row.addWidget(self._data_range_min_spin)

        self.btn_reset_wl_range = QPushButton("Reset")
        # AlignSidebarMixin helper (same MultiPlanarView instance)
        from inno3d.features.viewer.align_sidebar import _style_compact_reset_button
        _style_compact_reset_button(self.btn_reset_wl_range)
        self.btn_reset_wl_range.setToolTip("Reset selected range to full data range")
        self.btn_reset_wl_range.clicked.connect(self._reset_wl_to_data_range)
        data_row.addWidget(self.btn_reset_wl_range)

        self._data_range_max_spin = QDoubleSpinBox()
        self._data_range_max_spin.setDecimals(2)
        self._data_range_max_spin.setRange(-1e9, 1e9)
        self._data_range_max_spin.setReadOnly(True)
        self._data_range_max_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._data_range_max_spin.setAlignment(Qt.AlignCenter)
        data_row.addWidget(self._data_range_max_spin)
        hist_lay.addLayout(data_row)

        self._data_extent_label = QLabel("—")
        self._data_extent_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 7pt;"
        )
        self._data_extent_label.setAlignment(Qt.AlignHCenter)
        hist_lay.addWidget(self._data_extent_label)

        cm_lbl = QLabel("Color mapping")
        cm_lbl.setStyleSheet("color: #c0c0c0; font-size: 8pt; font-weight: bold;")
        hist_lay.addWidget(cm_lbl)

        cm_row = QHBoxLayout()
        cm_row.setSpacing(4)
        self._color_map_group = QButtonGroup(self)
        self._color_map_group.setExclusive(True)
        self.btn_cmap_normal = make_icon_toolbutton(
            make_color_map_icon("normal", selected=True),
            "Normal color mapping (low → high)",
            self,
        )
        self.btn_cmap_invert = make_icon_toolbutton(
            make_color_map_icon("invert"),
            "Invert color mapping (high → low)",
            self,
        )
        self.btn_cmap_normal.setChecked(True)
        self._color_map_group.addButton(self.btn_cmap_normal, 0)
        self._color_map_group.addButton(self.btn_cmap_invert, 1)
        self.btn_cmap_normal.clicked.connect(lambda: self._set_color_invert(False))
        self.btn_cmap_invert.clicked.connect(lambda: self._set_color_invert(True))
        cm_row.addWidget(self.btn_cmap_normal)
        cm_row.addWidget(self.btn_cmap_invert)
        cm_row.addStretch(1)
        hist_lay.addLayout(cm_row)

        om_lbl = QLabel("Opacity mapping")
        om_lbl.setStyleSheet("color: #c0c0c0; font-size: 8pt; font-weight: bold;")
        hist_lay.addWidget(om_lbl)

        op_map_row = QHBoxLayout()
        op_map_row.addWidget(QLabel("Opacity:"))
        self.wl_opacity_slider = QSlider(Qt.Horizontal)
        self.wl_opacity_slider.setRange(0, 100)
        self.wl_opacity_slider.setValue(100)
        self.wl_opacity_slider.setToolTip("Global opacity / solidity scale (Dragonfly-style)")
        self.wl_opacity_slider.valueChanged.connect(self._on_wl_opacity_changed)
        op_map_row.addWidget(self.wl_opacity_slider, 1)
        self.wl_opacity_value_lbl = QLabel("1.00")
        self.wl_opacity_value_lbl.setFixedWidth(32)
        self.wl_opacity_value_lbl.setStyleSheet("font-size: 8pt; color: #aaa;")
        op_map_row.addWidget(self.wl_opacity_value_lbl)
        hist_lay.addLayout(op_map_row)

        shape_row = QHBoxLayout()
        shape_row.setSpacing(3)
        self._opacity_shape_group = QButtonGroup(self)
        self._opacity_shape_group.setExclusive(True)
        self._opacity_shape_btns = {}
        for i, shape in enumerate(OPACITY_SHAPES):
            btn = make_icon_toolbutton(
                make_opacity_shape_icon(shape, selected=(shape == "linear")),
                OPACITY_SHAPE_TIPS.get(shape, shape),
                self,
            )
            if shape == "linear":
                btn.setChecked(True)
            self._opacity_shape_group.addButton(btn, i)
            self._opacity_shape_btns[shape] = btn
            btn.clicked.connect(lambda checked=False, s=shape: self._set_opacity_shape(s))
            shape_row.addWidget(btn)
        shape_row.addStretch(1)
        hist_lay.addLayout(shape_row)

        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Gamma:"))
        self.wl_gamma_slider = QSlider(Qt.Horizontal)
        self.wl_gamma_slider.setRange(20, 300)
        self.wl_gamma_slider.setValue(100)
        self.wl_gamma_slider.setToolTip("Gamma warp on opacity curve (1.00 = linear response)")
        self.wl_gamma_slider.valueChanged.connect(self._on_wl_gamma_changed)
        gamma_row.addWidget(self.wl_gamma_slider, 1)
        self.wl_gamma_spin = QDoubleSpinBox()
        self.wl_gamma_spin.setRange(0.20, 3.00)
        self.wl_gamma_spin.setSingleStep(0.05)
        self.wl_gamma_spin.setDecimals(2)
        self.wl_gamma_spin.setValue(1.00)
        self.wl_gamma_spin.setFixedWidth(58)
        self.wl_gamma_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.wl_gamma_spin.valueChanged.connect(self._on_wl_gamma_spin_changed)
        gamma_row.addWidget(self.wl_gamma_spin)
        hist_lay.addLayout(gamma_row)

        hist_group.setLayout(hist_lay)
        controls_panel.addWidget(hist_group)

        self._volume_data_min = 0.0
        self._volume_data_max = 65535.0
        self._color_invert = False
        self._opacity_shape = "linear"

        # --- B. View & Camera Presets ---
        cam_group = QGroupBox("View & Camera Presets")
        cam_lay = QVBoxLayout()

        proj_lay = QHBoxLayout()
        self.btn_ortho = QPushButton("Orthographic")
        self.btn_ortho.setCheckable(True)
        self.btn_ortho.clicked.connect(lambda: self.set_projection("orthographic"))
        proj_lay.addWidget(self.btn_ortho)

        self.btn_perspective = QPushButton("Perspective")
        self.btn_perspective.setCheckable(True)
        self.btn_perspective.setChecked(True)
        self.btn_perspective.clicked.connect(lambda: self.set_projection("perspective"))
        proj_lay.addWidget(self.btn_perspective)
        cam_lay.addLayout(proj_lay)

        grid_lay = QGridLayout()
        grid_lay.setSpacing(5)
        btn_front = QPushButton("Front")
        btn_front.clicked.connect(lambda: self.set_camera_preset("Front"))
        btn_back = QPushButton("Back")
        btn_back.clicked.connect(lambda: self.set_camera_preset("Back"))
        btn_left = QPushButton("Left")
        btn_left.clicked.connect(lambda: self.set_camera_preset("Left"))
        btn_right = QPushButton("Right")
        btn_right.clicked.connect(lambda: self.set_camera_preset("Right"))
        btn_top = QPushButton("Top")
        btn_top.clicked.connect(lambda: self.set_camera_preset("Top"))
        btn_bottom = QPushButton("Bottom")
        btn_bottom.clicked.connect(lambda: self.set_camera_preset("Bottom"))

        grid_lay.addWidget(btn_front, 0, 0)
        grid_lay.addWidget(btn_back, 0, 1)
        grid_lay.addWidget(btn_left, 1, 0)
        grid_lay.addWidget(btn_right, 1, 1)
        grid_lay.addWidget(btn_top, 2, 0)
        grid_lay.addWidget(btn_bottom, 2, 1)
        cam_lay.addLayout(grid_lay)

        utils_lay = QHBoxLayout()
        btn_reset = QPushButton("Reset View")
        btn_reset.clicked.connect(self.reset_3d_view)

        self.btn_autorotate = QPushButton("Auto-Rotate")
        self.btn_autorotate.setCheckable(True)
        self.btn_autorotate.toggled.connect(self.toggle_auto_rotate)

        utils_lay.addWidget(btn_reset)
        utils_lay.addWidget(self.btn_autorotate)
        cam_lay.addLayout(utils_lay)

        cam_group.setLayout(cam_lay)
        controls_panel.addWidget(cam_group)

        # --- C. Advanced MPR & Clipping ---
        mpr_clip_group = QGroupBox("Advanced MPR & Clipping")
        mc_lay = QVBoxLayout()
        mc_lay.setSpacing(8)

        self.bbox_check = QCheckBox("Show Bounding Box")
        self.bbox_check.stateChanged.connect(self.toggle_bounding_box)
        mc_lay.addWidget(self.bbox_check)

        top_mc_lay = QHBoxLayout()
        top_mc_lay.setSpacing(4)
        btn_show_all = QPushButton("Show All")
        btn_show_all.clicked.connect(lambda: self.on_mpr_show_hide_all(True))
        btn_hide_all = QPushButton("Hide All")
        btn_hide_all.clicked.connect(lambda: self.on_mpr_show_hide_all(False))
        btn_reset_all = QPushButton("Reset")
        btn_reset_all.clicked.connect(self.on_mpr_reset_all)
        top_mc_lay.addWidget(btn_show_all, 1)
        top_mc_lay.addWidget(btn_hide_all, 1)
        top_mc_lay.addWidget(btn_reset_all, 1)
        mc_lay.addLayout(top_mc_lay)

        thick_lay = QHBoxLayout()
        thick_lay.addWidget(QLabel("MIP Thickness:"))
        self.mpr_thickness_slider = QSlider(Qt.Horizontal)
        self.mpr_thickness_slider.setRange(0, 50)
        self.mpr_thickness_slider.setValue(0)
        self.mpr_thickness_slider.valueChanged.connect(self.update_mpr_thickness)
        thick_lay.addWidget(self.mpr_thickness_slider)

        self.thick_lbl = QLabel("0 px")
        self.thick_lbl.setMinimumWidth(35)
        self.mpr_thickness_slider.valueChanged.connect(lambda v: self.thick_lbl.setText(f"{v} px"))
        thick_lay.addWidget(self.thick_lbl)
        mc_lay.addLayout(thick_lay)

        self.adv_mpr_controls = {}
        axes_info = [('x', 'Sag (X)', '#FF4444'), ('y', 'Cor (Y)', '#44FF44'), ('z', 'Axi (Z)', '#4444FF')]

        for axis, label, color in axes_info:
            row_lay = QHBoxLayout()
            row_lay.setSpacing(4)
            row_lay.setContentsMargins(0, 0, 0, 0)

            lbl = QLabel(label)
            lbl.setFixedWidth(48)
            lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 8pt;")
            row_lay.addWidget(lbl)

            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 1000)
            sl.setValue(500)
            sl.setMinimumWidth(60)
            sl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            sl.setStyleSheet(f"""
                QSlider::groove:horizontal {{ height: 4px; background: #555; }}
                QSlider::handle:horizontal {{ background: {color}; width: 12px; margin: -4px 0; border-radius: 6px; }}
            """)
            sl.valueChanged.connect(lambda val, a=axis: self.update_advanced_mpr_clip(a))
            row_lay.addWidget(sl, 1)

            pct_lbl = QLabel("50%")
            pct_lbl.setFixedWidth(32)
            pct_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pct_lbl.setStyleSheet(f"color: {color}; font-size: 8pt;")
            row_lay.addWidget(pct_lbl)
            mc_lay.addLayout(row_lay)

            toggles_lay = QHBoxLayout()
            toggles_lay.setSpacing(6)
            toggles_lay.setContentsMargins(48, 0, 0, 4)

            chk_show = QCheckBox("Show")
            chk_show.setToolTip("Enable clipping along this axis")
            chk_show.stateChanged.connect(lambda state, a=axis: self.update_advanced_mpr_clip(a))
            toggles_lay.addWidget(chk_show)

            chk_clip = QCheckBox("Clip")
            chk_clip.setToolTip("Show colored indicator plane at clip position")
            chk_clip.stateChanged.connect(lambda state, a=axis: self.update_advanced_mpr_clip(a))
            toggles_lay.addWidget(chk_clip)

            btn_flip = QPushButton()
            btn_flip.setCheckable(True)
            btn_flip.setProperty("class", "icon-button")
            btn_flip.setIcon(_ui_icon_3d("arrows-up-down.svg", btn_flip))
            btn_flip.setIconSize(QSize(14, 14))
            btn_flip.setFixedSize(28, 26)
            btn_flip.setCursor(Qt.PointingHandCursor)
            btn_flip.setToolTip("Flip clip direction (keep other half)")
            btn_flip.clicked.connect(lambda checked, a=axis: self.update_advanced_mpr_clip(a))
            toggles_lay.addWidget(btn_flip)
            toggles_lay.addStretch()
            mc_lay.addLayout(toggles_lay)

            self.adv_mpr_controls[axis] = {
                'slider': sl,
                'pct_label': pct_lbl,
                'show': chk_show,
                'clip': chk_clip,
                'flip': btn_flip
            }

        mpr_clip_group.setLayout(mc_lay)
        controls_panel.addWidget(mpr_clip_group)

        # --- D. Spacing Control Group ---
        spacing_group = QGroupBox("3D Spacing (X, Y, Z)")
        sp_lay = QVBoxLayout()
        sp_lay.setContentsMargins(5, 10, 5, 5)
        sp_lay.setSpacing(5)

        sp_grid = QGridLayout()

        self.spacing_x_spin = QDoubleSpinBox()
        self.spacing_x_spin.setRange(0.0001, 100.0)
        self.spacing_x_spin.setValue(self.spacing[0])
        self.spacing_x_spin.setDecimals(4)
        self.spacing_x_spin.setSingleStep(0.1)
        self.spacing_x_spin.setToolTip("X Spacing")

        self.spacing_y_spin = QDoubleSpinBox()
        self.spacing_y_spin.setRange(0.0001, 100.0)
        self.spacing_y_spin.setValue(self.spacing[1])
        self.spacing_y_spin.setDecimals(4)
        self.spacing_y_spin.setSingleStep(0.1)
        self.spacing_y_spin.setToolTip("Y Spacing")

        self.spacing_z_spin = QDoubleSpinBox()
        self.spacing_z_spin.setRange(0.0001, 100.0)
        self.spacing_z_spin.setValue(self.spacing[2])
        self.spacing_z_spin.setDecimals(4)
        self.spacing_z_spin.setSingleStep(0.1)
        self.spacing_z_spin.setToolTip("Z Spacing")

        sp_grid.addWidget(QLabel("X:"), 0, 0)
        sp_grid.addWidget(self.spacing_x_spin, 0, 1)
        sp_grid.addWidget(QLabel("Y:"), 0, 2)
        sp_grid.addWidget(self.spacing_y_spin, 0, 3)
        sp_grid.addWidget(QLabel("Z:"), 0, 4)
        sp_grid.addWidget(self.spacing_z_spin, 0, 5)
        sp_lay.addLayout(sp_grid)

        self.btn_apply_spacing = QPushButton("Apply Spacing")
        self.btn_apply_spacing.clicked.connect(self.apply_custom_spacing)
        sp_lay.addWidget(self.btn_apply_spacing)

        spacing_group.setLayout(sp_lay)
        controls_panel.addWidget(spacing_group)

        adv_main_layout.addLayout(controls_panel)

        # --- Lighting Controls Group (Dragonfly-like defaults) ---
        light_group = QGroupBox("Lighting Properties")
        light_layout = QFormLayout()
        light_layout.setContentsMargins(5, 5, 5, 5)

        look_row = QHBoxLayout()
        self.lighting_look_combo = QComboBox()
        self.lighting_look_combo.addItems(["Inno3D", "Inspection", "Plastic", "Flat"])
        self.lighting_look_combo.setCurrentText("Inno3D")
        self.lighting_look_combo.setToolTip(
            "Inno3D: soft metallic (recommended — no plastic highlight caps)\n"
            "Inspection: flatter light for gap / surface metrology\n"
            "Plastic: old glossy demo look (strong specular pop)\n"
            "Flat: shading off (density TF only)"
        )
        look_row.addWidget(self.lighting_look_combo, 1)
        self.btn_apply_lighting_look = QPushButton("Apply Look")
        self.btn_apply_lighting_look.setToolTip(
            "Apply the selected lighting look (Phong + edge + LightKit)"
        )
        self.btn_apply_lighting_look.clicked.connect(self._on_apply_lighting_look_clicked)
        look_row.addWidget(self.btn_apply_lighting_look)
        light_layout.addRow("Look:", look_row)

        self.shade_check = QCheckBox("Enable Shading")
        self.shade_check.setToolTip(
            "Phong shading (Inno3D uses soft shade, not harsh specular)"
        )
        self.shade_check.setChecked(True)
        self.shade_check.stateChanged.connect(self.on_shade_toggled)
        light_layout.addRow("", self.shade_check)

        self.gradient_opacity_slider = QSlider(Qt.Horizontal)
        self.gradient_opacity_slider.setRange(0, 100)
        self.gradient_opacity_slider.setValue(22)
        self.gradient_opacity_slider.setToolTip(
            "Gradient opacity — soft surface edges (Dragonfly-like).\n"
            "0 = off · 15–30 recommended · high values look noisy"
        )
        self.gradient_opacity_slider.valueChanged.connect(self.on_gradient_opacity_changed)
        light_layout.addRow("Edge:", self.gradient_opacity_slider)

        self.view_3d_vol_opacity_slider = QSlider(Qt.Horizontal)
        self.view_3d_vol_opacity_slider.setRange(0, 100)
        self.view_3d_vol_opacity_slider.setValue(100)
        self.view_3d_vol_opacity_slider.valueChanged.connect(self.update_3d_lighting_properties)
        self.view_3d_vol_opacity_label = QLabel("1.00")
        self.view_3d_vol_opacity_label.setFixedWidth(30)
        self.view_3d_vol_opacity_slider.valueChanged.connect(
            lambda val: self.view_3d_vol_opacity_label.setText(f"{val/100:.2f}")
        )
        op_row = QHBoxLayout()
        op_row.addWidget(self.view_3d_vol_opacity_slider)
        op_row.addWidget(self.view_3d_vol_opacity_label)
        light_layout.addRow("Opacity:", op_row)

        self.view_3d_ambient_slider = QSlider(Qt.Horizontal)
        self.view_3d_ambient_slider.setRange(0, 100)
        self.view_3d_ambient_slider.setValue(40)
        self.view_3d_ambient_slider.valueChanged.connect(self.update_3d_lighting_properties)
        self.view_3d_ambient_label = QLabel("0.40")
        self.view_3d_ambient_label.setFixedWidth(30)
        self.view_3d_ambient_slider.valueChanged.connect(
            lambda val: self.view_3d_ambient_label.setText(f"{val/100:.2f}")
        )
        amb_row = QHBoxLayout()
        amb_row.addWidget(self.view_3d_ambient_slider)
        amb_row.addWidget(self.view_3d_ambient_label)
        light_layout.addRow("Ambient:", amb_row)

        self.view_3d_diffuse_slider = QSlider(Qt.Horizontal)
        self.view_3d_diffuse_slider.setRange(0, 100)
        self.view_3d_diffuse_slider.setValue(58)
        self.view_3d_diffuse_slider.valueChanged.connect(self.update_3d_lighting_properties)
        self.view_3d_diffuse_label = QLabel("0.58")
        self.view_3d_diffuse_label.setFixedWidth(30)
        self.view_3d_diffuse_slider.valueChanged.connect(
            lambda val: self.view_3d_diffuse_label.setText(f"{val/100:.2f}")
        )
        diff_row = QHBoxLayout()
        diff_row.addWidget(self.view_3d_diffuse_slider)
        diff_row.addWidget(self.view_3d_diffuse_label)
        light_layout.addRow("Diffuse:", diff_row)

        self.view_3d_specular_slider = QSlider(Qt.Horizontal)
        self.view_3d_specular_slider.setRange(0, 100)
        self.view_3d_specular_slider.setValue(18)
        self.view_3d_specular_slider.valueChanged.connect(self.update_3d_lighting_properties)
        self.view_3d_specular_label = QLabel("0.18")
        self.view_3d_specular_label.setFixedWidth(30)
        self.view_3d_specular_slider.valueChanged.connect(
            lambda val: self.view_3d_specular_label.setText(f"{val/100:.2f}")
        )
        spec_row = QHBoxLayout()
        spec_row.addWidget(self.view_3d_specular_slider)
        spec_row.addWidget(self.view_3d_specular_label)
        light_layout.addRow("Specular:", spec_row)

        self.view_3d_spec_power_slider = QSlider(Qt.Horizontal)
        self.view_3d_spec_power_slider.setRange(1, 128)
        self.view_3d_spec_power_slider.setValue(16)
        self.view_3d_spec_power_slider.valueChanged.connect(self.update_3d_lighting_properties)
        self.view_3d_spec_power_label = QLabel("16")
        self.view_3d_spec_power_label.setFixedWidth(30)
        self.view_3d_spec_power_slider.valueChanged.connect(
            lambda val: self.view_3d_spec_power_label.setText(str(val))
        )
        pow_row = QHBoxLayout()
        pow_row.addWidget(self.view_3d_spec_power_slider)
        pow_row.addWidget(self.view_3d_spec_power_label)
        light_layout.addRow("Power:", pow_row)

        light_group.setLayout(light_layout)
        adv_main_layout.addWidget(light_group)

        adv_main_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(360)
        scroll_content = QWidget()
        scroll_content.setMinimumWidth(340)
        scroll_content.setLayout(adv_main_layout)
        scroll.setWidget(scroll_content)

        sidebar_layout = QVBoxLayout(self.advanced_3d_controls)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.addWidget(scroll)

        self.advanced_3d_controls.setMinimumWidth(0)
        self.advanced_3d_controls.setMaximumWidth(0)
        self.advanced_3d_controls.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.advanced_3d_controls.setVisible(False)
        content_layout.addWidget(self.advanced_3d_controls, 0)

        widget.setLayout(layout)
        return widget

    # ── Group C: Render Mode / Gradient / Shade ───────────────────────────────

    def on_render_mode_changed(self, mode_text):
        """Switch volume rendering blend mode."""
        if not hasattr(self, 'volume_actor') or self.volume_actor is None:
            return
        mapper = self.volume_actor.GetMapper()

        mode_map = {
            "Composite":  0,
            "MIP":        1,
            "MinIP":      2,
            "Average":    3,
            "Additive":   4,
        }
        blend = mode_map.get(mode_text, 0)
        mapper.SetBlendMode(blend)
        self.view_3d_widget.GetRenderWindow().Render()
        print(f">>> Render mode: {mode_text} (blend={blend})")

    def on_gradient_opacity_changed(self, value):
        """Apply gradient-based opacity modulation for edge enhancement."""
        if not hasattr(self, 'volume_actor') or self.volume_actor is None:
            return
        prop = self.volume_actor.GetProperty()
        scale = value / 100.0

        if value == 0:
            prop.DisableGradientOpacityOn()
        else:
            prop.DisableGradientOpacityOff()
            grad = vtk.vtkPiecewiseFunction()
            grad.AddPoint(0, 0.0)
            grad.AddPoint(40 * (1.0 - scale * 0.5), 0.05 * scale)
            grad.AddPoint(120 * (1.0 - scale * 0.4), 0.35 * scale)
            grad.AddPoint(280, 0.75 * scale)
            grad.AddPoint(1000, 1.0)
            prop.SetGradientOpacity(grad)

        self.view_3d_widget.GetRenderWindow().Render()

    def on_shade_toggled(self, state):
        """Enable/disable Phong shading."""
        if not hasattr(self, 'volume_actor') or self.volume_actor is None:
            return
        prop = self.volume_actor.GetProperty()
        if state == Qt.Checked:
            prop.ShadeOn()
        else:
            prop.ShadeOff()
        self.view_3d_widget.GetRenderWindow().Render()

    def _on_apply_lighting_look_clicked(self):
        name = "Inno3D"
        if hasattr(self, "lighting_look_combo"):
            name = self.lighting_look_combo.currentText()
        self.apply_lighting_look(name)

    def apply_lighting_look(self, name="Inno3D", render=True):
        """Apply a named lighting look (Inno3D / Inspection / Plastic / Flat).

        Inno3D target: soft silver form light, mild edge OP, no plastic specular caps.
        """
        presets = getattr(self, "_lighting_look_presets", None) or {}
        # Map legacy name if any caller still passes "Dragonfly"
        if name == "Dragonfly":
            name = "Inno3D"
        p = presets.get(name) or presets.get("Inno3D")
        if not p:
            return

        self._current_lighting_look = name
        self._vol_shadows_enabled = bool(p.get("vol_shadows", False))

        def _set_slider(slider, label, value, is_float=True):
            if slider is None:
                return
            slider.blockSignals(True)
            slider.setValue(int(value))
            slider.blockSignals(False)
            if label is not None:
                if is_float:
                    label.setText(f"{int(value) / 100.0:.2f}")
                else:
                    label.setText(str(int(value)))

        if hasattr(self, "shade_check"):
            self.shade_check.blockSignals(True)
            self.shade_check.setChecked(bool(p.get("shade", True)))
            self.shade_check.blockSignals(False)

        _set_slider(
            getattr(self, "view_3d_ambient_slider", None),
            getattr(self, "view_3d_ambient_label", None),
            p.get("ambient", 40),
        )
        _set_slider(
            getattr(self, "view_3d_diffuse_slider", None),
            getattr(self, "view_3d_diffuse_label", None),
            p.get("diffuse", 58),
        )
        _set_slider(
            getattr(self, "view_3d_specular_slider", None),
            getattr(self, "view_3d_specular_label", None),
            p.get("specular", 18),
        )
        _set_slider(
            getattr(self, "view_3d_spec_power_slider", None),
            getattr(self, "view_3d_spec_power_label", None),
            p.get("power", 16),
            is_float=False,
        )
        _set_slider(
            getattr(self, "view_3d_vol_opacity_slider", None),
            getattr(self, "view_3d_vol_opacity_label", None),
            p.get("opacity", 100),
        )
        if hasattr(self, "gradient_opacity_slider"):
            self.gradient_opacity_slider.blockSignals(True)
            self.gradient_opacity_slider.setValue(int(p.get("edge", 22)))
            self.gradient_opacity_slider.blockSignals(False)

        lk = getattr(self, "light_kit", None)
        if lk is not None:
            try:
                lk.SetKeyLightIntensity(float(p.get("key_intensity", 0.72)))
                lk.SetKeyToFillRatio(float(p.get("key_fill", 1.8)))
                lk.SetKeyLightWarmth(float(p.get("key_warmth", 0.55)))
            except Exception:
                pass

        mapper = getattr(self, "_volume_mapper", None)
        if mapper is not None and hasattr(mapper, "SetShadows"):
            try:
                mapper.SetShadows(1 if self._vol_shadows_enabled else 0)
            except Exception:
                pass

        if hasattr(self, "lighting_look_combo"):
            self.lighting_look_combo.blockSignals(True)
            idx = self.lighting_look_combo.findText(name)
            if idx >= 0:
                self.lighting_look_combo.setCurrentIndex(idx)
            self.lighting_look_combo.blockSignals(False)

        if getattr(self, "volume_actor", None) is not None:
            prop = self.volume_actor.GetProperty()
            if p.get("shade", True):
                prop.ShadeOn()
            else:
                prop.ShadeOff()
            prop.SetAmbient(float(p.get("ambient", 40)) / 100.0)
            prop.SetDiffuse(float(p.get("diffuse", 58)) / 100.0)
            prop.SetSpecular(float(p.get("specular", 18)) / 100.0)
            prop.SetSpecularPower(float(p.get("power", 16)))
            self.apply_transfer_function(prop)
            if hasattr(self, "on_gradient_opacity_changed"):
                self.on_gradient_opacity_changed(
                    self.gradient_opacity_slider.value()
                    if hasattr(self, "gradient_opacity_slider")
                    else 0
                )
            if render and hasattr(self, "view_3d_widget") and self.view_3d_widget:
                try:
                    self.view_3d_widget.GetRenderWindow().Render()
                except Exception:
                    pass
        print(
            f"[3D LOOK] {name}: amb={p.get('ambient')} dif={p.get('diffuse')} "
            f"spec={p.get('specular')} pow={p.get('power')} edge={p.get('edge')} "
            f"shadows={p.get('vol_shadows')}"
        )

    # ── Group D: TF Presets + Update ─────────────────────────────────────────

    def apply_tf_preset(self, preset_name):
        """Apply a named LUT / transfer function preset (Dragonfly-style)."""
        if preset_name == "Custom" or preset_name not in self._TF_PRESETS:
            return

        preset = self._TF_PRESETS[preset_name]

        self.color_grad_widget.color_stops = [
            (pos, QColor(r, g, b)) for pos, (r, g, b) in preset["colors"]
        ]
        self.color_grad_widget.update()

        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_lut_stops(self.color_grad_widget.color_stops)

        if hasattr(self, 'volume_actor') and self.volume_actor is not None:
            self.update_transfer_function()

    def _make_lut_icon(self, preset_name, w=60, h=16):
        """Generate a QIcon with a gradient pixmap for the given LUT preset."""
        pixmap = QPixmap(w, h)
        preset = self._TF_PRESETS.get(preset_name)
        if preset is None:
            pixmap.fill(QColor(50, 50, 50))
            return QIcon(pixmap)
        painter = QPainter(pixmap)
        from PyQt5.QtGui import QLinearGradient
        grad = QLinearGradient(0, 0, w, 0)
        for pos, (r, g, b) in preset["colors"]:
            grad.setColorAt(pos, QColor(r, g, b))
        painter.fillRect(0, 0, w, h, grad)
        painter.end()
        return QIcon(pixmap)

    def update_transfer_function(self):
        """Lightweight TF update without rebuilding the volume actor."""
        if self.volume_actor:
            self.apply_transfer_function(self.volume_actor.GetProperty())
            self.view_3d_widget.GetRenderWindow().Render()

    def apply_custom_spacing(self):
        """Apply custom user-specified spacing values to the 3D volume rendering."""
        from PyQt5.QtWidgets import QMessageBox
        if self.volume_data is None:
            QMessageBox.warning(self, "Warning", "No volume data loaded.")
            return

        x_sp = self.spacing_x_spin.value()
        y_sp = self.spacing_y_spin.value()
        z_sp = self.spacing_z_spin.value()

        self.custom_spacing = [x_sp, y_sp, z_sp]
        print(f">>> Applying custom spacing: X={x_sp}, Y={y_sp}, Z={z_sp}")
        if self.volume_data is not None:
            self.render_3d()

    # ── Group E: TF Mode / Opacity / Color callbacks ──────────────────────────

    def _on_tf_mode_changed(self, mode):
        """Keep mode buttons in sync; enable W/L-only controls only in W/L mode."""
        is_wl = mode == TransferFunctionWidget.MODE_WL
        if hasattr(self, 'btn_wl_mode') and hasattr(self, 'btn_tf_mode'):
            self.btn_wl_mode.blockSignals(True)
            self.btn_tf_mode.blockSignals(True)
            self.btn_wl_mode.setChecked(is_wl)
            self.btn_tf_mode.setChecked(not is_wl)
            self.btn_wl_mode.blockSignals(False)
            self.btn_tf_mode.blockSignals(False)
        if hasattr(self, '_opacity_shape_btns'):
            for btn in self._opacity_shape_btns.values():
                btn.setEnabled(is_wl)
        for name in ('wl_gamma_slider', 'wl_gamma_spin', 'wl_opacity_slider'):
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(True)
        if hasattr(self, 'wl_gamma_slider'):
            self.wl_gamma_slider.setEnabled(is_wl)
            self.wl_gamma_spin.setEnabled(is_wl)
        self.update_transfer_function()

    def on_opacity_spline_changed(self, points):
        if not hasattr(self, 'volume_actor') or self.volume_actor is None:
            return
        self.update_transfer_function()

    def on_color_gradient_changed(self, stops):
        if not hasattr(self, 'volume_actor') or self.volume_actor is None:
            return
        prop = self.volume_actor.GetProperty()
        v_min = float(self.view_3d_min_spin.value())
        v_max = float(self.view_3d_max_spin.value())
        if v_max <= v_min:
            v_max = v_min + 1.0

        color = vtk.vtkColorTransferFunction()
        for pos, qcolor in stops:
            intensity = v_min + pos * (v_max - v_min)
            color.AddRGBPoint(intensity, qcolor.redF(), qcolor.greenF(), qcolor.blueF())

        prop.SetColor(color)
        self.view_3d_widget.GetRenderWindow().Render()

    @staticmethod
    def _theme_primary_rgb():
        from inno3d.features.shared.orientation_cube import theme_primary_rgb
        return theme_primary_rgb()

    @staticmethod
    def _theme_primary_hover_rgb():
        from inno3d.features.shared.orientation_cube import theme_primary_hover_rgb
        return theme_primary_hover_rgb()

    # ── Group F: Orientation Marker (shared orientation_cube module) ─────────

    def _build_orientation_marker(self):
        """vtkAssembly via shared builder (kept for any external callers)."""
        from inno3d.features.shared.orientation_cube import build_orientation_marker

        assembly, state = build_orientation_marker()
        self._ori_cube_state = state
        self._axes_cube_actor = state.get("cube")
        self._ori_face_highlights = state.get("face_highlights") or {}
        self._ori_face_text_props = state.get("face_text_props") or {}
        return assembly

    def _set_orientation_face_hover(self, face):
        """Highlight one cube face in app primary cyan (or clear if face is None)."""
        from inno3d.features.shared.orientation_cube import set_orientation_face_hover

        state = getattr(self, "_ori_cube_state", None)
        if state is None:
            state = {
                "cube": getattr(self, "_axes_cube_actor", None),
                "face_highlights": getattr(self, "_ori_face_highlights", {}) or {},
                "face_text_props": getattr(self, "_ori_face_text_props", {}) or {},
                "hover_face": getattr(self, "_ori_hover_face", None),
            }
            self._ori_cube_state = state
        if set_orientation_face_hover(state, face):
            self._ori_hover_face = face
            if hasattr(self, "view_3d_widget") and self.view_3d_widget is not None:
                self.view_3d_widget.GetRenderWindow().Render()

    def _on_orientation_cube_hover(self, obj, event):
        """Mouse-move: glow face under cursor (Viewer SoT)."""
        if not hasattr(self, "axes_widget") or self.axes_widget is None:
            return
        if not hasattr(self, "view_3d_widget") or self.view_3d_widget is None:
            return
        iren = self.view_3d_widget.GetRenderWindow().GetInteractor()
        if iren is None:
            return
        x, y = iren.GetEventPosition()
        face = self._pick_orientation_cube_face(x, y)

        over_marker = self._is_over_orientation_viewport(x, y)
        try:
            if face is not None:
                self.view_3d_widget.setCursor(Qt.PointingHandCursor)
            elif over_marker:
                self.view_3d_widget.setCursor(Qt.ArrowCursor)
            elif getattr(self, "_ori_hover_face", None) is not None:
                self.view_3d_widget.unsetCursor()
        except Exception:
            pass

        self._set_orientation_face_hover(face)

    def _is_over_orientation_viewport(self, display_x, display_y):
        from inno3d.features.shared.orientation_cube import is_over_orientation_viewport

        if not hasattr(self, "axes_widget") or self.axes_widget is None:
            return False
        ren_win = self.view_3d_widget.GetRenderWindow()
        size = ren_win.GetSize()
        if not size or size[0] <= 0 or size[1] <= 0:
            return False
        return is_over_orientation_viewport(
            display_x, display_y, size[0], size[1], self.axes_widget.GetViewport()
        )

    def _on_orientation_cube_click(self, obj, event):
        """Left-click on the ±X/±Y/±Z cube → orient volume camera to that face."""
        if not hasattr(self, 'axes_widget') or self.axes_widget is None:
            return
        if not hasattr(self, 'view_3d_widget') or self.view_3d_widget is None:
            return
        iren = self.view_3d_widget.GetRenderWindow().GetInteractor()
        if iren is None:
            return
        x, y = iren.GetEventPosition()
        face = self._pick_orientation_cube_face(x, y)
        if face is None:
            return
        try:
            iren.SetAbortFlag(1)
        except Exception:
            pass
        self._set_orientation_face_hover(face)
        self._orient_camera_to_face(face)

    def _pick_orientation_cube_face(self, display_x, display_y):
        """Ray-pick the orientation marker cube. Returns face label or None."""
        from inno3d.features.shared.orientation_cube import pick_orientation_cube_face

        ren_win = self.view_3d_widget.GetRenderWindow()
        size = ren_win.GetSize()
        if not size or size[0] <= 0 or size[1] <= 0:
            return None
        return pick_orientation_cube_face(
            display_x,
            display_y,
            size[0],
            size[1],
            self.axes_widget.GetViewport(),
            self.view_3d_renderer.GetActiveCamera(),
        )

    @staticmethod
    def _ray_hit_unit_cube_face(origin, direction):
        from inno3d.features.shared.orientation_cube import ray_hit_unit_cube_face
        return ray_hit_unit_cube_face(origin, direction)

    def _orient_camera_to_face(self, face):
        """Snap 3D camera so the given cube face points toward the viewer."""
        from inno3d.features.shared.orientation_cube import FACE_TO_PRESET

        preset = FACE_TO_PRESET.get(face)
        if preset is None:
            return
        if self.volume_data is None:
            self._orient_camera_to_face_no_volume(face)
            return
        self.set_camera_preset(preset)

    def _orient_camera_to_face_no_volume(self, face):
        """Fallback orient when no volume is loaded yet."""
        from inno3d.features.shared.orientation_cube import apply_camera_face_preset

        if not hasattr(self, 'view_3d_renderer') or self.view_3d_renderer is None:
            return
        cam = self.view_3d_renderer.GetActiveCamera()
        if not apply_camera_face_preset(cam, face, center=(0.0, 0.0, 0.0), radius=500.0):
            return
        self.view_3d_renderer.ResetCameraClippingRange()
        self.view_3d_widget.GetRenderWindow().Render()

    def set_camera_preset(self, preset):
        if not hasattr(self, 'view_3d_renderer') or self.volume_data is None:
            return
        cam = self.view_3d_renderer.GetActiveCamera()
        cx, cy, cz = self._volume_world_center()
        z, y, x = self.volume_data.shape
        sp = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        half = max((x - 1) * sp[0], (y - 1) * sp[1], (z - 1) * sp[2]) * 0.5
        r = max(half, 1.0) * 2.5

        cam.SetFocalPoint(cx, cy, cz)
        if preset == "Front":
            cam.SetPosition(cx, cy - r, cz)
            cam.SetViewUp(0, 0, 1)
        elif preset == "Back":
            cam.SetPosition(cx, cy + r, cz)
            cam.SetViewUp(0, 0, 1)
        elif preset == "Right":
            cam.SetPosition(cx + r, cy, cz)
            cam.SetViewUp(0, 0, 1)
        elif preset == "Left":
            cam.SetPosition(cx - r, cy, cz)
            cam.SetViewUp(0, 0, 1)
        elif preset == "Top":
            cam.SetPosition(cx, cy, cz + r)
            cam.SetViewUp(0, 1, 0)
        elif preset == "Bottom":
            cam.SetPosition(cx, cy, cz - r)
            cam.SetViewUp(0, -1, 0)

        self._sync_3d_orbit_pivot()
        self.view_3d_renderer.ResetCameraClippingRange()
        self.view_3d_widget.GetRenderWindow().Render()

    def toggle_auto_rotate(self, checked):
        if checked:
            self.auto_rotate_timer.start(30)
        else:
            self.auto_rotate_timer.stop()

    def auto_rotate_step(self):
        if hasattr(self, 'view_3d_renderer') and self.view_3d_renderer:
            cam = self.view_3d_renderer.GetActiveCamera()
            cam.Azimuth(0.5)
            self.view_3d_widget.GetRenderWindow().Render()

    # ── Group G: Bounding Box ────────────────────────────────────────────────

    def toggle_bounding_box(self, state):
        if not hasattr(self, 'view_3d_renderer') or self.volume_data is None:
            return

        if state == Qt.Checked:
            if self.bbox_actor is None:
                outline = vtk.vtkOutlineFilter()
                z, y, x = self.volume_data.shape
                img = vtk.vtkImageData()
                img.SetDimensions(x, y, z)
                img.SetSpacing(self.spacing[0], self.spacing[1], self.spacing[2])
                outline.SetInputData(img)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(outline.GetOutputPort())
                self.bbox_actor = vtk.vtkActor()
                self.bbox_actor.SetMapper(mapper)
                self.bbox_actor.GetProperty().SetColor(1, 1, 1)
            self.view_3d_renderer.AddActor(self.bbox_actor)
        else:
            if self.bbox_actor:
                self.view_3d_renderer.RemoveActor(self.bbox_actor)
        self.view_3d_widget.GetRenderWindow().Render()

    def _get_vtk_world_extent(self):
        """Return world-space axis ranges as ``{'x': (min, max), ...}``.

        Origin is always 0 (VTK image data origin). Max is (dim-1)*spacing.
        Clip-box / Advanced MPR always index by axis name and [0]/[1].
        """
        if self.volume_data is None:
            return {'x': (0.0, 1.0), 'y': (0.0, 1.0), 'z': (0.0, 1.0)}
        z, y, x = self.volume_data.shape
        sp = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        return {
            'x': (0.0, float((x - 1) * sp[0])),
            'y': (0.0, float((y - 1) * sp[1])),
            'z': (0.0, float((z - 1) * sp[2])),
        }

    # ── Group H: render_3d + apply_transfer_function ──────────────────────────

    def render_3d(self):
        if self.volume_data is None:
            return

        # Master switch (Online default OFF for large volumes)
        if not self.is_3d_volume_render_enabled():
            self._teardown_3d_volume_actors()
            self._show_3d_disabled_placeholder()
            # Keep 3D crosshair when volume GPU is off (lines do not need VRAM volume)
            if getattr(self, "crosshair_enabled", False) and hasattr(self, "update_3d_crosshair"):
                try:
                    self.update_3d_crosshair()
                except Exception:
                    pass
            return

        self._hide_3d_disabled_placeholder()

        # NEW: GPU Driver/Vendor Information Check
        win = self.view_3d_widget.GetRenderWindow()
        if hasattr(win, 'ReportCapabilities'):
            caps = win.ReportCapabilities()
            if "NVIDIA" in caps.upper() or "RTX" in caps.upper():
                print(">>> 3D Viewer: High-Performance GPU Detected (NVIDIA/RTX)")
            elif "INTEL" in caps.upper():
                print(">>> 3D Viewer WARNING: Only Integrated Graphics Detected (INTEL)")
                print("Please check Windows Graphics Settings to force NVIDIA GPU.")
            else:
                print(f">>> 3D Viewer Vendor Info: {caps[:100]}...")

        if self.volume_actor:
            self.view_3d_renderer.RemoveVolume(self.volume_actor)
        from inno3d.features.viewer.seg_mask_3d import remove_mask_overlay_from_renderer

        if getattr(self, 'c1_actor_3d', None):
            remove_mask_overlay_from_renderer(self.view_3d_renderer, self.c1_actor_3d)
            self.c1_actor_3d = None
        if getattr(self, 'c2_actor_3d', None):
            remove_mask_overlay_from_renderer(self.view_3d_renderer, self.c2_actor_3d)
            self.c2_actor_3d = None

        z, y, x = self.volume_data.shape

        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(x, y, z)
        spacing = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        vtk_data.SetSpacing(spacing[0], spacing[1], spacing[2])

        data_fortran = np.transpose(self.volume_data, (2, 1, 0))
        flat_data = np.ascontiguousarray(data_fortran.flatten('F'))

        if self.volume_data.dtype == np.uint16:
            vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
        elif self.volume_data.dtype == np.uint8:
            vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        else:
            flat_data = flat_data.astype(np.float32)
            vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_FLOAT)

        vtk_data.GetPointData().SetScalars(vtk_array)

        # Setup histogram widgets with volume data
        if hasattr(self, 'tf_widget'):
            self.tf_widget.set_histogram(self.volume_data)
        if hasattr(self, 'tf_widget') and hasattr(self, 'color_grad_widget'):
            self.tf_widget.set_lut_stops(self.color_grad_widget.color_stops)
        if hasattr(self, 'mpr_hist_widget'):
            dmin = getattr(self, '_volume_data_min', None)
            dmax = getattr(self, '_volume_data_max', None)
            self.mpr_hist_widget.set_histogram(
                self.volume_data, data_min=dmin, data_max=dmax
            )
            if dmin is not None and dmax is not None:
                self.mpr_hist_widget.set_range(int(dmin), int(dmax))

        # Quality rendering parameters from selected preset
        quality_name = getattr(self, '_current_quality', 'High')
        preset = self._quality_presets.get(quality_name) if hasattr(self, '_quality_presets') else None
        if preset is None:
            preset = {"sample_dist": 0.5, "image_sample": 1.0, "auto_adj": True, "jitter": True, "max_mem_fraction": 0.85}

        sample_distance = preset["sample_dist"]
        image_sample_dist = preset["image_sample"]
        use_auto_adjust = preset["auto_adj"]
        use_jitter = preset["jitter"]
        max_mem_frac = preset.get("max_mem_fraction", 0.85)

        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_data)
        volume_mapper.SetAutoAdjustSampleDistances(1 if use_auto_adjust else 0)
        volume_mapper.SetSampleDistance(sample_distance)
        if hasattr(volume_mapper, 'SetImageSampleDistance'):
            volume_mapper.SetImageSampleDistance(image_sample_dist)
        if hasattr(volume_mapper, 'SetUseJittering'):
            volume_mapper.SetUseJittering(1 if use_jitter else 0)
        if hasattr(volume_mapper, 'SetMaxMemoryFraction'):
            volume_mapper.SetMaxMemoryFraction(max_mem_frac)
        if hasattr(volume_mapper, 'SetMaxMemoryInBytes'):
            volume_mapper.SetMaxMemoryInBytes(int(28 * 1024 * 1024 * 1024))

        if hasattr(self, 'render_mode_combo'):
            mode_text = self.render_mode_combo.currentText()
            mode_map = {
                "Composite": 0, "MIP": 1, "MinIP": 2, "Average": 3, "Additive": 4,
            }
            volume_mapper.SetBlendMode(mode_map.get(mode_text, 0))
        elif hasattr(volume_mapper, 'SetBlendModeToComposite'):
            volume_mapper.SetBlendModeToComposite()

        if hasattr(volume_mapper, 'SetShadows'):
            use_sh = bool(getattr(self, "_vol_shadows_enabled", False))
            volume_mapper.SetShadows(1 if use_sh else 0)

        self._volume_mapper = volume_mapper

        volume_property = vtk.vtkVolumeProperty()
        if hasattr(self, 'shade_check') and self.shade_check.isChecked():
            volume_property.ShadeOn()
        else:
            volume_property.ShadeOff()

        volume_property.SetInterpolationTypeToLinear()
        try:
            sp = self.custom_spacing if getattr(self, "custom_spacing", None) is not None else self.spacing
            soud = float(min(sp[0], sp[1], sp[2]))
            if soud > 0:
                volume_property.SetScalarOpacityUnitDistance(soud)
        except Exception:
            pass

        self.apply_transfer_function(volume_property)

        if hasattr(self, 'gradient_opacity_slider') and self.gradient_opacity_slider.value() > 0:
            scale = self.gradient_opacity_slider.value() / 100.0
            volume_property.DisableGradientOpacityOff()
            grad = vtk.vtkPiecewiseFunction()
            grad.AddPoint(0, 0.0)
            grad.AddPoint(40 * (1.0 - scale * 0.5), 0.05 * scale)
            grad.AddPoint(120 * (1.0 - scale * 0.4), 0.35 * scale)
            grad.AddPoint(280, 0.75 * scale)
            grad.AddPoint(1000, 1.0)
            volume_property.SetGradientOpacity(grad)
        else:
            try:
                volume_property.DisableGradientOpacityOn()
            except Exception:
                pass

        amb = 0.40
        dif = 0.58
        spe = 0.18
        spow = 16.0
        if hasattr(self, 'view_3d_ambient_slider'):
            amb = self.view_3d_ambient_slider.value() / 100.0
            dif = self.view_3d_diffuse_slider.value() / 100.0
            spe = self.view_3d_specular_slider.value() / 100.0
            spow = float(self.view_3d_spec_power_slider.value())
        volume_property.SetAmbient(amb)
        volume_property.SetDiffuse(dif)
        volume_property.SetSpecular(spe)
        volume_property.SetSpecularPower(spow)

        self.volume_actor = vtk.vtkVolume()
        self.volume_actor.SetMapper(volume_mapper)
        self.volume_actor.SetProperty(volume_property)

        self.view_3d_renderer.AddVolume(self.volume_actor)

        # --- Segmentation overlays: medical glass shells (surface mesh) ---
        # Lighter than multi-volume ray-cast; CT remains readable under translucent mask.
        from inno3d.features.viewer.seg_mask_3d import (
            create_mask_surface_actor,
            add_mask_overlay_to_renderer,
        )

        sp = (
            self.custom_spacing
            if getattr(self, "custom_spacing", None) is not None
            else self.spacing
        )
        # Prefer user opacity slider when present (else glass defaults)
        try:
            og = getattr(self, "view_3d_overlay_group", None)
            slider_op = (
                float(og.opacity_slider.value()) / 100.0
                if og is not None and hasattr(og, "opacity_slider")
                else None
            )
        except Exception:
            slider_op = None

        if getattr(self, "class1_data", None) is not None:
            c1_color = self.seg_colors.get(128, [0.0, 1.0, 0.0])
            c1_op = float(slider_op) if slider_op is not None else 0.35
            self.c1_actor_3d = create_mask_surface_actor(
                self.class1_data, sp, c1_color, base_opacity=c1_op, smooth=True
            )
            if self.c1_actor_3d is not None:
                add_mask_overlay_to_renderer(self.view_3d_renderer, self.c1_actor_3d)

        if getattr(self, "class2_data", None) is not None:
            c2_color = self.seg_colors.get(255, [1.0, 0.0, 0.0])
            # Void slightly more opaque than bump so it reads inside shells
            if slider_op is not None:
                c2_op = min(1.0, float(slider_op) * 1.15)
            else:
                c2_op = 0.48
            self.c2_actor_3d = create_mask_surface_actor(
                self.class2_data, sp, c2_color, base_opacity=c2_op, smooth=True
            )
            if self.c2_actor_3d is not None:
                add_mask_overlay_to_renderer(self.view_3d_renderer, self.c2_actor_3d)

        # Sync clip planes to ALL actors (volume + c1 + c2) in one pass
        self._sync_all_clip_planes()
        self._df_clip_update_3d_visuals()

        if not getattr(self, '_camera_initialized', False):
            self.view_3d_renderer.ResetCamera()
            cam = self.view_3d_renderer.GetActiveCamera()
            pmode = str(getattr(self, 'projection_mode', 'perspective')).split('#')[0].strip().lower()
            if pmode == 'perspective':
                cam.ParallelProjectionOff()
                cam.SetViewAngle(40.0)
            else:
                cam.ParallelProjectionOn()

            self.set_camera_preset("Top")
            self._camera_initialized = True
        else:
            cam = self.view_3d_renderer.GetActiveCamera()
            pmode = str(getattr(self, 'projection_mode', 'perspective')).split('#')[0].strip().lower()
            if pmode == 'perspective':
                cam.ParallelProjectionOff()
            else:
                cam.ParallelProjectionOn()

        self._sync_3d_orbit_pivot()
        self._mes_3d_hl_cache_key = None
        if getattr(self, "selected_highlight_objects", None):
            self._update_mes_3d_highlight()
        else:
            self._clear_mes_3d_highlight(render=True)

    def apply_transfer_function(self, volume_property):
        """Apply opacity + colour TF for the current W/L or TF mode (Dragonfly-style)."""
        v_min = float(self.view_3d_min_spin.value())
        v_max = float(self.view_3d_max_spin.value())

        if v_max <= v_min:
            v_max = v_min + 1.0

        opacity_scale = 1.0
        if hasattr(self, 'wl_opacity_slider'):
            opacity_scale *= self.wl_opacity_slider.value() / 100.0
        elif hasattr(self, 'view_3d_vol_opacity_slider'):
            opacity_scale *= self.view_3d_vol_opacity_slider.value() / 100.0

        tw = getattr(self, 'tf_widget', None)
        if tw is not None and tw.is_wl_mode():
            shape = tw.opacity_shape()
            gamma = tw.opacity_gamma()
            eff = sample_opacity_curve(shape, gamma=gamma, n=33, opacity_scale=opacity_scale)
        elif tw is not None:
            eff = [(float(x), float(y) * opacity_scale) for x, y in tw.get_points()]
        else:
            shape = getattr(self, '_opacity_shape', 'linear')
            gamma = self.wl_gamma_spin.value() if hasattr(self, 'wl_gamma_spin') else 1.0
            eff = sample_opacity_curve(shape, gamma=gamma, n=33, opacity_scale=opacity_scale)

        cg_stops = (
            getattr(self.color_grad_widget, 'color_stops', None)
            if hasattr(self, 'color_grad_widget') else None
        )
        invert = bool(getattr(self, '_color_invert', False))
        if tw is not None:
            invert = bool(tw.color_invert())

        if eff and len(eff) > 0 and cg_stops and len(cg_stops) > 0:
            opacity = vtk.vtkPiecewiseFunction()
            opacity.AddPoint(v_min - 1.0, 0.0)
            for nx, ny in eff:
                intensity = v_min + float(nx) * (v_max - v_min)
                opacity.AddPoint(intensity, float(ny))
            opacity.AddPoint(v_max + 1.0, float(eff[-1][1]))

            color = vtk.vtkColorTransferFunction()
            stops = list(cg_stops)
            if invert:
                stops = [(1.0 - pos, c) for pos, c in stops]
                stops.sort(key=lambda s: s[0])
            for pos, qcolor in stops:
                intensity = v_min + float(pos) * (v_max - v_min)
                color.AddRGBPoint(
                    intensity, qcolor.redF(), qcolor.greenF(), qcolor.blueF()
                )

            volume_property.SetScalarOpacity(opacity)
            volume_property.SetColor(color)
            return

        # Fallback silver/grayscale
        mid = v_min + 0.55 * (v_max - v_min)
        r, g, b = self.volume_color
        dark = (r * 0.22, g * 0.22, b * 0.22)
        mid_c = (r * 0.72, g * 0.72, b * 0.70)
        hi = (min(1.0, r * 1.08), min(1.0, g * 1.08), min(1.0, b * 1.06))

        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(v_min, 0.0, 0.0, 0.0)
        color_func.AddRGBPoint(v_min + 0.22 * (v_max - v_min), *dark)
        color_func.AddRGBPoint(mid, *mid_c)
        color_func.AddRGBPoint(v_min + 0.88 * (v_max - v_min), r, g, b)
        color_func.AddRGBPoint(v_max, *hi)

        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(v_min, 0.0)
        opacity_func.AddPoint(v_min + 0.28 * (v_max - v_min), 0.0)
        opacity_func.AddPoint(mid, 0.22 * opacity_scale)
        opacity_func.AddPoint(v_min + 0.85 * (v_max - v_min), 0.72 * opacity_scale)
        opacity_func.AddPoint(v_max, 0.92 * opacity_scale)

        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)

    # ── Group I: LOD (Level-of-Detail) Interaction System ────────────────────

    def _start_lod_interaction(self):
        """Called when user starts rotating/zooming the 3D view."""
        self._is_interacting_3d = True
        if hasattr(self, '_lod_refine_timer'):
            self._lod_refine_timer.stop()

        mapper = getattr(self, '_volume_mapper', None)
        if mapper is None:
            return

        quality_name = getattr(self, '_current_quality', 'High')
        preset = self._quality_presets.get(quality_name, {})

        interact_sample = preset.get('interact_sample_dist', 2.0)
        interact_image = preset.get('interact_image_sample', 2.0)

        mapper.SetSampleDistance(interact_sample)
        if hasattr(mapper, 'SetImageSampleDistance'):
            mapper.SetImageSampleDistance(interact_image)
        mapper.SetAutoAdjustSampleDistances(1)

    def _end_lod_interaction(self):
        """Called when user stops rotating/zooming."""
        self._is_interacting_3d = False
        if hasattr(self, '_lod_refine_timer'):
            self._lod_refine_timer.start()

    def _refine_after_interaction(self):
        """Restore full-quality rendering after interaction stops."""
        if self._is_interacting_3d:
            return

        mapper = getattr(self, '_volume_mapper', None)
        if mapper is None:
            return

        quality_name = getattr(self, '_current_quality', 'High')
        preset = self._quality_presets.get(quality_name, {})

        mapper.SetSampleDistance(preset.get('sample_dist', 0.5))
        if hasattr(mapper, 'SetImageSampleDistance'):
            mapper.SetImageSampleDistance(preset.get('image_sample', 1.0))
        mapper.SetAutoAdjustSampleDistances(1 if preset.get('auto_adj', True) else 0)

        self._sync_all_clip_planes()

        if hasattr(self, 'view_3d_widget') and self.view_3d_widget:
            self.view_3d_widget.GetRenderWindow().Render()

    def _on_quality_preset_changed(self, quality_name):
        """Handle quality preset combo box change."""
        self._current_quality = quality_name

        mapper = getattr(self, '_volume_mapper', None)
        if mapper is None:
            return

        preset = self._quality_presets.get(quality_name, {})

        mapper.SetSampleDistance(preset.get('sample_dist', 0.5))
        if hasattr(mapper, 'SetImageSampleDistance'):
            mapper.SetImageSampleDistance(preset.get('image_sample', 1.0))
        mapper.SetAutoAdjustSampleDistances(1 if preset.get('auto_adj', True) else 0)
        if hasattr(mapper, 'SetUseJittering'):
            mapper.SetUseJittering(1 if preset.get('jitter', True) else 0)
        if hasattr(mapper, 'SetMaxMemoryFraction'):
            mapper.SetMaxMemoryFraction(preset.get('max_mem_fraction', 0.85))

        if hasattr(self, 'view_3d_widget') and self.view_3d_widget:
            self.view_3d_widget.GetRenderWindow().Render()

        print(f">>> 3D Quality: {quality_name} | SampleDist={preset.get('sample_dist'):.2f} | "
              f"ImageSample={preset.get('image_sample'):.1f} | "
              f"InteractDist={preset.get('interact_sample_dist'):.1f} | "
              f"MaxMem={preset.get('max_mem_fraction', 0.85)*100:.0f}%")

    # ── Group J: 3D Color Pickers ────────────────────────────────────────────

    def choose_volume_color(self):
        """Choose color for volume rendering."""
        parent = getattr(self, 'fullscreen_window', None) or self
        init_c = QColor(
            int(self.volume_color[0] * 255),
            int(self.volume_color[1] * 255),
            int(self.volume_color[2] * 255),
        )
        color = QColorDialog.getColor(init_c, parent, "Volume Color")
        if color.isValid():
            self.volume_color = [color.redF(), color.greenF(), color.blueF()]

            if hasattr(self, 'color_grad_widget'):
                dark = QColor(
                    int(color.red() * 0.35),
                    int(color.green() * 0.35),
                    int(color.blue() * 0.35),
                )
                mid_c = QColor(
                    int(color.red() * 0.85),
                    int(color.green() * 0.85),
                    int(color.blue() * 0.85),
                )
                highlight_c = QColor(
                    min(255, int(color.red() * 1.25)),
                    min(255, int(color.green() * 1.25 + 12)),
                    min(255, int(color.blue() * 1.25 + 63)),
                )
                self.color_grad_widget.color_stops = [
                    (0.0, QColor(0, 0, 0)),
                    (0.3, dark),
                    (0.6, mid_c),
                    (0.95, color),
                    (1.0, highlight_c),
                ]
                self.color_grad_widget.update()

            if self.volume_actor:
                self.update_transfer_function()

    def update_3d_lighting_properties(self):
        """Update lighting properties for 3D volume actor."""
        if not self.volume_actor:
            return
        prop = self.volume_actor.GetProperty()
        if hasattr(self, "shade_check") and self.shade_check.isChecked():
            prop.ShadeOn()
        else:
            prop.ShadeOff()
        prop.SetAmbient(self.view_3d_ambient_slider.value() / 100.0)
        prop.SetDiffuse(self.view_3d_diffuse_slider.value() / 100.0)
        prop.SetSpecular(self.view_3d_specular_slider.value() / 100.0)
        prop.SetSpecularPower(float(self.view_3d_spec_power_slider.value()))
        self.apply_transfer_function(prop)
        self.view_3d_widget.GetRenderWindow().Render()

    def choose_3d_bg_color(self):
        """Choose background color for 3D volume viewport with a Dragonfly-style gradient."""
        parent = getattr(self, 'fullscreen_window', None) or self
        bg2 = self.view_3d_renderer.GetBackground2()
        init_c = QColor(int(bg2[0] * 255), int(bg2[1] * 255), int(bg2[2] * 255))
        color = QColorDialog.getColor(init_c, parent, "Choose Viewport Top/Center Color")
        if color.isValid():
            r, g, b = color.redF(), color.greenF(), color.blueF()

            self.view_3d_renderer.SetTexturedBackground(False)
            self.view_3d_renderer.GradientBackgroundOn()
            try:
                self.view_3d_renderer.SetGradientMode(
                    self.view_3d_renderer.VTK_GRADIENT_RADIAL_FARTHEST_CORNER
                )
                self.view_3d_renderer.SetBackground(r, g, b)
                self.view_3d_renderer.SetBackground2(r * 0.15, g * 0.15, b * 0.15)
            except AttributeError:
                self.view_3d_renderer.SetBackground2(r, g, b)
                self.view_3d_renderer.SetBackground(r * 0.15, g * 0.15, b * 0.15)

            self.view_3d_widget.GetRenderWindow().Render()
