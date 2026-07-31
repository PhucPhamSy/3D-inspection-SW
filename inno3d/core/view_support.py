import glob
from pathlib import Path

import numpy as np
import vtk
from PyQt5.QtCore import QEvent, QObject, QSize, QThread, QTimer, Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QCursor, QIcon, QPainter
from PyQt5.QtWidgets import (
    QBoxLayout, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QSlider, QStyle, QToolButton, QVBoxLayout, QGraphicsOpacityEffect,
    QDoubleSpinBox, QWidget,
)
from skimage import io
from vtk.util import numpy_support

from inno3d.core.styles import SemiconductorTheme


def _ui_icon(name, widget=None, fallback=None):
    icon_path = Path(__file__).resolve().parents[2] / "assets" / "icons" / name
    if icon_path.exists():
        return QIcon(str(icon_path))
    if widget is not None and fallback is not None:
        return widget.style().standardIcon(fallback)
    return QIcon()


class StepOneSliceSlider(QSlider):
    """Slice index slider: hover + mouse wheel moves value by exactly 1.

    Used by 3D Viewer and 3D Teaching MPR slice bars.
    """

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setSingleStep(1)
        self.setPageStep(1)
        # StrongFocus + WheelFocus-like: accept wheel without prior click
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_Hover, True)
        self.setMouseTracking(True)

    @staticmethod
    def _wheel_delta(event):
        """Normalize wheel delta across mouse / trackpad / high-res devices."""
        dy = event.angleDelta().y()
        if dy == 0:
            dy = event.angleDelta().x()
        if dy == 0:
            dy = event.pixelDelta().y()
        return dy

    def wheelEvent(self, event):
        dy = self._wheel_delta(event)
        if dy == 0:
            event.ignore()
            return
        self.setValue(self.value() + (1 if dy > 0 else -1))
        event.accept()

    def event(self, event):
        # Ensure Wheel is handled even when Qt would otherwise route it elsewhere
        if event.type() == QEvent.Wheel:
            self.wheelEvent(event)
            return event.isAccepted()
        return super().event(event)


class SliceBarWheelFilter(QObject):
    """Forward mouse-wheel on the whole slice bar (label + groove) to the slider.

    Install on the bottom slice bar container so hover anywhere on the bar
    (not only the thin handle) steps the slice by 1.
    """

    def __init__(self, slider, parent=None):
        super().__init__(parent)
        self._slider = slider

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and self._slider is not None:
            dy = StepOneSliceSlider._wheel_delta(event)
            if dy == 0:
                return False
            self._slider.setValue(self._slider.value() + (1 if dy > 0 else -1))
            return True  # consume — do not bubble to VTK / parent
        return False


def volume_world_aabb(shape_zyx, spacing_xyz, origin_xyz=(0.0, 0.0, 0.0)):
    """World-space AABB of a VTK image volume (origin + spacing * index).

    Parameters
    ----------
    shape_zyx : (Z, Y, X) array shape
    spacing_xyz : (sx, sy, sz)
    origin_xyz : (ox, oy, oz) image origin in world coords
    """
    z, y, x = [int(v) for v in shape_zyx]
    sx, sy, sz = [float(v) for v in spacing_xyz]
    ox, oy, oz = [float(v) for v in origin_xyz]
    return (
        ox,
        ox + max(x - 1, 0) * sx,
        oy,
        oy + max(y - 1, 0) * sy,
        oz,
        oz + max(z - 1, 0) * sz,
    )


def _point_inside_aabb(px, py, pz, bounds, pad=0.0):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (
        (xmin - pad) < px < (xmax + pad)
        and (ymin - pad) < py < (ymax + pad)
        and (zmin - pad) < pz < (zmax + pad)
    )


def _finite3(v):
    """True if v is a 3-vector of finite floats."""
    try:
        return (
            len(v) >= 3
            and v[0] == v[0] and v[1] == v[1] and v[2] == v[2]
            and abs(v[0]) < 1e30 and abs(v[1]) < 1e30 and abs(v[2]) < 1e30
        )
    except Exception:
        return False


def push_camera_outside_aabb(cam, bounds, pad_frac=0.05):
    """Optional helper: push eye outside volume AABB (not used for deep zoom).

    Kept for callers that want a hard outside clamp (e.g. reset framing).
    """
    if cam is None or bounds is None:
        return
    try:
        xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in bounds]
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        hx = max(0.5 * (xmax - xmin), 1e-9)
        hy = max(0.5 * (ymax - ymin), 1e-9)
        hz = max(0.5 * (zmax - zmin), 1e-9)
        pad = max(max(hx, hy, hz) * float(pad_frac), 1e-4)

        px, py, pz = cam.GetPosition()
        if not _finite3((px, py, pz)):
            return
        if not _point_inside_aabb(px, py, pz, bounds, pad=pad * 0.25):
            return

        vx, vy, vz = px - cx, py - cy, pz - cz
        d = (vx * vx + vy * vy + vz * vz) ** 0.5
        if d < 1e-12:
            fp = cam.GetFocalPoint()
            vx, vy, vz = px - fp[0], py - fp[1], pz - fp[2]
            d = (vx * vx + vy * vy + vz * vz) ** 0.5
        if d < 1e-12:
            try:
                vpn = cam.GetViewPlaneNormal()
                vx, vy, vz, d = float(vpn[0]), float(vpn[1]), float(vpn[2]), 1.0
            except Exception:
                vx, vy, vz, d = 0.0, 0.0, 1.0, 1.0
        inv = 1.0 / d
        vx, vy, vz = vx * inv, vy * inv, vz * inv

        tx = (hx + pad) / max(abs(vx), 1e-12)
        ty = (hy + pad) / max(abs(vy), 1e-12)
        tz = (hz + pad) / max(abs(vz), 1e-12)
        t_exit = min(tx, ty, tz)
        if not (t_exit == t_exit) or t_exit < 1e-6:
            return
        cam.SetPosition(cx + vx * t_exit, cy + vy * t_exit, cz + vz * t_exit)
    except Exception:
        return


def _safe_reset_clipping(renderer, cam, volume_radius=100.0):
    """Clipping for unlimited zoom-in (camera may fly through the volume)."""
    if renderer is None or cam is None:
        return
    try:
        R = max(float(volume_radius), 1.0)
        pos = cam.GetPosition()
        fp = cam.GetFocalPoint()
        dist = (
            (pos[0] - fp[0]) ** 2
            + (pos[1] - fp[1]) ** 2
            + (pos[2] - fp[2]) ** 2
        ) ** 0.5
        if not (dist == dist) or dist < 1e-12:
            dist = max(R * 0.01, 1e-3)

        # Very small near + large far so fly-through does not hard-stop
        near = max(min(dist * 0.0005, R * 0.0005), 1e-6)
        far = max(dist * 200.0, R * 80.0, 100.0, near * 100.0)
        cam.SetClippingRange(near, far)
    except Exception:
        try:
            R = max(float(volume_radius), 1.0)
            cam.SetClippingRange(1e-6, max(R * 100.0, 1e6))
        except Exception:
            pass


def _world_at_display(renderer, display_xy):
    """World coords under a display pixel (safe; returns None on failure)."""
    if renderer is None or display_xy is None:
        return None
    try:
        mx = float(display_xy[0])
        my = float(display_xy[1])
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToDisplay()
        coord.SetValue(mx, my, 0.0)
        return list(coord.GetComputedWorldValue(renderer))
    except Exception:
        return None


def apply_dragonfly_volume_zoom(
    renderer,
    *,
    zoom_in=True,
    strength=1.0,
    volume_center=(0.0, 0.0, 0.0),
    volume_radius=100.0,
    volume_bounds=None,
    display_xy=None,
):
    """Unlimited 3D volume zoom (scroll / right-drag) — same depth as pre-fix.

    Behaviour:
      • Far: classic dolly toward focal point (FP fixed).
      • Close / through block: **free-flight** — camera *and* focal point
        translate together so zoom continues until the volume leaves the
        view (can disappear completely), not stopped at the AABB shell.
      • Only a tiny numerical floor prevents a true zero-distance singularity.
      • Soft zoom-out cap so the volume does not vanish as a speck far away.
      • Optional ``display_xy``: zoom-to-cursor (keep world under cursor fixed).

    ``strength`` ≈ 1.0 is one wheel notch (~8% scale change).
    """
    if renderer is None:
        return

    try:
        strength = max(0.05, min(float(strength), 4.0))
        cam = renderer.GetActiveCamera()
        if cam is None:
            return
        R = max(float(volume_radius), 1.0)
        cx, cy, cz = [float(v) for v in volume_center]

        # Capture world under cursor BEFORE camera change (zoom-to-cursor)
        world_before = _world_at_display(renderer, display_xy) if display_xy is not None else None

        base = 1.08
        factor = base ** strength

        if cam.GetParallelProjection():
            s = float(cam.GetParallelScale())
            if not (s == s) or s <= 0:
                s = R
            # Near-zero floor so ortho can zoom until nothing is visible
            s_min = 1e-12
            s_max = R * 20.0
            s = (s / factor) if zoom_in else (s * factor)
            cam.SetParallelScale(min(max(s, s_min), s_max))
        else:
            pos = list(cam.GetPosition())
            fp = list(cam.GetFocalPoint())
            if not _finite3(pos) or not _finite3(fp):
                cam.SetFocalPoint(cx, cy, cz)
                cam.SetPosition(cx, cy - R * 2.5, cz)
                pos = list(cam.GetPosition())
                fp = list(cam.GetFocalPoint())

            vx, vy, vz = fp[0] - pos[0], fp[1] - pos[1], fp[2] - pos[2]
            dist = (vx * vx + vy * vy + vz * vz) ** 0.5
            if dist < 1e-15:
                # Degenerate: keep flying along last view plane normal
                try:
                    vpn = cam.GetViewPlaneNormal()
                    # VPN points from FP toward camera → fly opposite (into view)
                    vx, vy, vz = -float(vpn[0]), -float(vpn[1]), -float(vpn[2])
                    n = (vx * vx + vy * vy + vz * vz) ** 0.5
                    if n < 1e-15:
                        vx, vy, vz = 0.0, -1.0, 0.0
                        n = 1.0
                    vx, vy, vz = vx / n, vy / n, vz / n
                except Exception:
                    vx, vy, vz = 0.0, -1.0, 0.0
                dist = max(R * 0.01, 1e-3)
            else:
                inv = 1.0 / dist
                vx, vy, vz = vx * inv, vy * inv, vz * inv

            max_d = max(R * 40.0, 10.0)
            # Continuous step (same spirit as pre-fix free-flight)
            frac = 1.0 - 1.0 / factor  # ~0.074 for strength 1 @ base 1.08
            step = max(dist * frac, R * 0.002 * strength)

            if zoom_in:
                # Far: dolly (FP fixed). Close / would overshoot: free-flight so
                # zoom never stalls and can pass through until volume is gone.
                if dist < R * 0.35 or step >= dist * 0.9:
                    pos = [
                        pos[0] + vx * step,
                        pos[1] + vy * step,
                        pos[2] + vz * step,
                    ]
                    fp = [
                        fp[0] + vx * step,
                        fp[1] + vy * step,
                        fp[2] + vz * step,
                    ]
                    if not (_finite3(pos) and _finite3(fp)):
                        return
                    cam.SetPosition(pos[0], pos[1], pos[2])
                    cam.SetFocalPoint(fp[0], fp[1], fp[2])
                else:
                    new_dist = dist / factor
                    # Tiny floor only if dolly would hit singularity this step
                    if new_dist < 1e-9:
                        pos = [
                            pos[0] + vx * step,
                            pos[1] + vy * step,
                            pos[2] + vz * step,
                        ]
                        fp = [
                            fp[0] + vx * step,
                            fp[1] + vy * step,
                            fp[2] + vz * step,
                        ]
                        if not (_finite3(pos) and _finite3(fp)):
                            return
                        cam.SetPosition(pos[0], pos[1], pos[2])
                        cam.SetFocalPoint(fp[0], fp[1], fp[2])
                    else:
                        nx = fp[0] - vx * new_dist
                        ny = fp[1] - vy * new_dist
                        nz = fp[2] - vz * new_dist
                        if not _finite3((nx, ny, nz)):
                            return
                        cam.SetPosition(nx, ny, nz)
            else:
                # Zoom out: pull back along view (keep FP); soft max distance
                new_dist = min(dist * factor, max_d)
                new_dist = max(new_dist, dist + R * 0.001 * strength)
                new_dist = min(new_dist, max_d)
                nx = fp[0] - vx * new_dist
                ny = fp[1] - vy * new_dist
                nz = fp[2] - vz * new_dist
                if not _finite3((nx, ny, nz)):
                    return
                cam.SetPosition(nx, ny, nz)

        # Zoom-to-cursor: shift camera so world_before stays under the same pixel
        if world_before is not None and display_xy is not None:
            try:
                world_after = _world_at_display(renderer, display_xy)
                if world_after is not None and _finite3(world_before) and _finite3(world_after):
                    dx = world_before[0] - world_after[0]
                    dy = world_before[1] - world_after[1]
                    dz = world_before[2] - world_after[2]
                    if abs(dx) + abs(dy) + abs(dz) > 1e-12:
                        pos = list(cam.GetPosition())
                        fp = list(cam.GetFocalPoint())
                        cam.SetPosition(pos[0] + dx, pos[1] + dy, pos[2] + dz)
                        cam.SetFocalPoint(fp[0] + dx, fp[1] + dy, fp[2] + dz)
            except Exception:
                pass

        _safe_reset_clipping(renderer, cam, volume_radius=R)
    except Exception:
        # Never let zoom exceptions propagate into Qt event loop / hard-crash path
        return


class Dragonfly3DInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """3D volume navigation — Dragonfly-like Track + pan + unlimited zoom.

    Controls:
      • Left-drag   — **Track**: free object tumble (screen-space arcball)
                      around the orbit pivot (default = volume center)
      • Middle-drag — pan
      • Right-drag  — zoom (dolly → free-flight when close)
      • Scroll      — same zoom path as right-drag

    Track model (matches ORS Dragonfly “rotate objects freely”):
      • Pivot fixed in world (volume center, or later user pivot tool)
      • Focal point stays on the pivot while tracking (object spins in place)
      • Rotation = virtual trackball / arcball in screen space (not gimbal
        az/el that re-samples axes every mouse move)
      • Left-down silently looks at the pivot without a visual jump
    """

    def __init__(self):
        super().__init__()
        # VTK default-like; arcball scales independently
        self.SetMotionFactor(10.0)
        if hasattr(self, "SetMouseWheelMotionFactor"):
            self.SetMouseWheelMotionFactor(0.9)

        self._pivot = [0.0, 0.0, 0.0]
        self._has_pivot = False
        self._volume_radius = 100.0
        self._volume_bounds = None
        self._wheel_zoom_factor = 1.12

        # Arcball state (filled on left-down / each move)
        self._track_active = False
        self._track_last_sphere = None  # unit vector on virtual sphere
        self._track_ball_center = (0.0, 0.0)  # display (x, y)
        self._track_ball_radius = 1.0

    # ── public API ──────────────────────────────────────────────────────
    def SetOrbitPivot(self, x, y, z, radius=None, bounds=None):
        """World-space Track/orbit center (default = volume center)."""
        self._pivot = [float(x), float(y), float(z)]
        self._has_pivot = True
        if radius is not None and float(radius) > 1e-6:
            self._volume_radius = float(radius)
        if bounds is not None and len(bounds) >= 6:
            self._volume_bounds = tuple(float(v) for v in bounds[:6])

    def GetOrbitPivot(self):
        return tuple(self._pivot)

    def SetVolumeBounds(self, bounds):
        """World AABB of the volume (optional; used by zoom helpers)."""
        if bounds is None:
            self._volume_bounds = None
        else:
            self._volume_bounds = tuple(float(v) for v in bounds[:6])

    # ── distance helpers ────────────────────────────────────────────────
    def _max_camera_distance(self):
        return max(self._volume_radius * 40.0, 10.0)

    def _camera_distance(self, cam):
        pos = cam.GetPosition()
        fp = cam.GetFocalPoint()
        return (
            (pos[0] - fp[0]) ** 2
            + (pos[1] - fp[1]) ** 2
            + (pos[2] - fp[2]) ** 2
        ) ** 0.5

    def _view_dir_and_dist(self, cam):
        pos = cam.GetPosition()
        fp = cam.GetFocalPoint()
        dx, dy, dz = fp[0] - pos[0], fp[1] - pos[1], fp[2] - pos[2]
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist < 1e-15:
            return (0.0, 0.0, -1.0), 0.0
        return (dx / dist, dy / dist, dz / dist), dist

    # ── helpers ─────────────────────────────────────────────────────────
    def _renderer(self):
        ren = self.GetCurrentRenderer()
        if ren is not None:
            return ren
        iren = self.GetInteractor()
        if iren is None:
            return None
        x, y = iren.GetEventPosition()
        self.FindPokedRenderer(int(x), int(y))
        return self.GetCurrentRenderer()

    def _track_pivot(self):
        if self._has_pivot:
            return list(self._pivot)
        ren = self._renderer()
        if ren is None:
            return [0.0, 0.0, 0.0]
        return list(ren.GetActiveCamera().GetFocalPoint())

    @staticmethod
    def _norm3(v):
        n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5
        if n < 1e-15:
            return [0.0, 0.0, 0.0], 0.0
        return [v[0] / n, v[1] / n, v[2] / n], n

    @staticmethod
    def _cross3(a, b):
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    @staticmethod
    def _dot3(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _camera_basis(self, cam):
        """Orthonormal camera basis: right, up, back (toward camera / out of screen).

        look  = unit(FP - Position)  — into the scene
        right = unit(look × up)
        up    = unit(right × look)   — re-orthogonalized
        back  = -look                — toward viewer (arcball +Z)
        """
        pos = cam.GetPosition()
        fp = cam.GetFocalPoint()
        look = [fp[0] - pos[0], fp[1] - pos[1], fp[2] - pos[2]]
        look, ln = self._norm3(look)
        if ln < 1e-15:
            look = [0.0, 0.0, -1.0]
        up = list(cam.GetViewUp())
        up, un = self._norm3(up)
        if un < 1e-15:
            up = [0.0, 1.0, 0.0]
        right = self._cross3(look, up)
        right, rn = self._norm3(right)
        if rn < 1e-15:
            right = [1.0, 0.0, 0.0]
        # Re-orthogonalize up
        up = self._cross3(right, look)
        up, _ = self._norm3(up)
        back = [-look[0], -look[1], -look[2]]
        return right, up, back

    def _look_at_pivot_no_jump(self, ren, cam, pivot):
        """Make FP = pivot without changing the rendered image.

        Shifts camera and focal point by the same world delta (pure translation).
        Used once at Track start so subsequent arcball orbits the true pivot.
        """
        fp = list(cam.GetFocalPoint())
        pos = list(cam.GetPosition())
        dx = pivot[0] - fp[0]
        dy = pivot[1] - fp[1]
        dz = pivot[2] - fp[2]
        if abs(dx) + abs(dy) + abs(dz) < 1e-12:
            return
        cam.SetFocalPoint(pivot[0], pivot[1], pivot[2])
        cam.SetPosition(pos[0] + dx, pos[1] + dy, pos[2] + dz)
        _safe_reset_clipping(ren, cam, volume_radius=self._volume_radius)

    def _project_to_sphere(self, display_x, display_y):
        """Shoemake-style map of display point → unit vector on virtual sphere.

        Sphere is centered on the viewport; radius is chosen so pixel drag
        sensitivity stays similar in small pane vs fullscreen (see
        ``_begin_track_gesture``).
        """
        import math
        cx, cy = self._track_ball_center
        r = max(self._track_ball_radius, 1.0)
        # Normalized device in ball space; Y flipped so up is +Y on screen
        px = (float(display_x) - cx) / r
        py = (float(display_y) - cy) / r
        d2 = px * px + py * py
        if d2 > 1.0:
            n = math.sqrt(d2)
            return [px / n, py / n, 0.0]
        return [px, py, math.sqrt(max(0.0, 1.0 - d2))]

    def _viewport_display_metrics(self, ren):
        """Display origin + size for the active renderer (robust after resize).

        Fullscreen / sidebar layout changes can leave ``ren.GetSize()`` stale
        for a frame; fall back to interactor / render-window size so the
        arcball center matches what the user sees.
        """
        iren = self.GetInteractor()
        w = h = 0.0
        ox = oy = 0.0
        try:
            size = ren.GetSize()
            w, h = float(size[0]), float(size[1])
            origin = ren.GetOrigin() if hasattr(ren, "GetOrigin") else (0, 0)
            ox, oy = float(origin[0]), float(origin[1])
        except Exception:
            pass
        if w < 2.0 or h < 2.0:
            try:
                if iren is not None:
                    wsz = iren.GetSize()
                    w, h = float(wsz[0]), float(wsz[1])
                    ox, oy = 0.0, 0.0
            except Exception:
                pass
        if w < 2.0 or h < 2.0:
            try:
                if iren is not None:
                    rw = iren.GetRenderWindow()
                    if rw is not None:
                        wsz = rw.GetSize()
                        w, h = float(wsz[0]), float(wsz[1])
                        ox, oy = 0.0, 0.0
            except Exception:
                pass
        if w < 2.0 or h < 2.0:
            w, h = 800.0, 600.0
            ox, oy = 0.0, 0.0
        return ox, oy, w, h

    def _begin_track_gesture(self):
        """Prepare arcball + pivot look-at when left button goes down."""
        ren = self._renderer()
        iren = self.GetInteractor()
        if ren is None or iren is None:
            self._track_active = False
            return
        cam = ren.GetActiveCamera()
        if cam is None:
            self._track_active = False
            return

        # Host may have just left/entered fullscreen — refresh pivot + size
        host = getattr(self, "_host", None)
        if host is not None:
            try:
                if hasattr(host, "_force_3d_render_window_size"):
                    host._force_3d_render_window_size()
                if hasattr(host, "_sync_3d_orbit_pivot"):
                    host._sync_3d_orbit_pivot()
            except Exception:
                pass

        pivot = self._track_pivot()
        # Object-centric: always orbit looking at the pivot (no image jump)
        self._look_at_pivot_no_jump(ren, cam, pivot)

        ox, oy, w, h = self._viewport_display_metrics(ren)

        # Ball centered in this renderer. Use a *clamped* radius so fullscreen
        # does not make Track feel sluggish (pure 0.5·min(w,h) scales poorly).
        # ~half short side, but keep pixel→angle similar to a mid-size pane.
        short = min(w, h)
        self._track_ball_center = (ox + 0.5 * w, oy + 0.5 * h)
        self._track_ball_radius = max(140.0, min(0.5 * short, 280.0))
        # Store viewport short side for angle normalization in Rotate()
        self._track_view_short = short

        x, y = iren.GetEventPosition()
        self._track_last_sphere = self._project_to_sphere(x, y)
        self._track_active = True

    def Rotate(self):
        """Dragonfly Track: screen-space arcball free rotate around pivot.

        Mouse motion is mapped to a rotation on a virtual sphere (Shoemake
        arcball). The rotation axis is converted from camera space → world
        and applied about the orbit pivot so the volume tumbles freely under
        the cursor (object-centric, FP locked to pivot).
        """
        import math

        ren = self._renderer()
        iren = self.GetInteractor()
        if ren is None or iren is None:
            return
        cam = ren.GetActiveCamera()
        if cam is None:
            return

        if not self._track_active or self._track_last_sphere is None:
            # Safety: gesture prep missed (e.g. style swap mid-drag)
            self._begin_track_gesture()
            if not self._track_active:
                return

        x, y = iren.GetEventPosition()
        cur = self._project_to_sphere(x, y)
        prev = self._track_last_sphere

        # Rotation axis ∝ prev × cur, angle = atan2(|cross|, dot)
        axis_c = self._cross3(prev, cur)
        axis_c, axis_n = self._norm3(axis_c)
        dot = max(-1.0, min(1.0, self._dot3(prev, cur)))
        if axis_n < 1e-12 or abs(dot - 1.0) < 1e-12:
            self._track_last_sphere = cur
            return

        # Angle on the sphere; MotionFactor scales feel (10 ≈ 1:1 arcball).
        # Normalize by ball radius vs a reference so fullscreen (large r) is not
        # sluggish and tiny panes are not hypersensitive — same px ≈ same °.
        angle_rad = math.acos(dot)
        motion = float(self.GetMotionFactor()) if hasattr(self, "GetMotionFactor") else 10.0
        r_ball = max(float(getattr(self, "_track_ball_radius", 200.0)), 1.0)
        r_ref = 200.0
        angle_deg = math.degrees(angle_rad) * (motion / 10.0) * (r_ball / r_ref)
        if abs(angle_deg) < 1e-6:
            self._track_last_sphere = cur
            return

        # Camera basis: map sphere axes (x=right, y=up, z=toward viewer) → world
        right, up, back = self._camera_basis(cam)
        axis_w = [
            axis_c[0] * right[0] + axis_c[1] * up[0] + axis_c[2] * back[0],
            axis_c[0] * right[1] + axis_c[1] * up[1] + axis_c[2] * back[1],
            axis_c[0] * right[2] + axis_c[1] * up[2] + axis_c[2] * back[2],
        ]
        axis_w, an = self._norm3(axis_w)
        if an < 1e-12:
            self._track_last_sphere = cur
            return

        pivot = self._track_pivot()
        pos = list(cam.GetPosition())
        vup = list(cam.GetViewUp())

        # Rotate camera *position* around pivot (object appears to spin)
        # Opposite sign so dragging right rotates the object right (Dragonfly)
        ang = -angle_deg
        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Translate(-pivot[0], -pivot[1], -pivot[2])
        t.RotateWXYZ(ang, axis_w[0], axis_w[1], axis_w[2])
        t.Translate(pivot[0], pivot[1], pivot[2])

        new_pos = list(t.TransformPoint(pos[0], pos[1], pos[2]))
        # View-up: rotate as vector (same R, no translation)
        t_vec = vtk.vtkTransform()
        t_vec.RotateWXYZ(ang, axis_w[0], axis_w[1], axis_w[2])
        new_up = list(t_vec.TransformVector(vup[0], vup[1], vup[2]))

        if not (_finite3(new_pos) and _finite3(new_up)):
            self._track_last_sphere = cur
            return

        # Lock look-at to pivot (object-centric Track)
        cam.SetPosition(new_pos[0], new_pos[1], new_pos[2])
        cam.SetFocalPoint(pivot[0], pivot[1], pivot[2])

        # Orthogonalize view-up to look direction
        look = [pivot[0] - new_pos[0], pivot[1] - new_pos[1], pivot[2] - new_pos[2]]
        look_u, _ = self._norm3(look)
        up_u, un = self._norm3(new_up)
        if un < 1e-12:
            up_u = [0.0, 0.0, 1.0]
        d = self._dot3(up_u, look_u)
        up_u = [up_u[0] - d * look_u[0], up_u[1] - d * look_u[1], up_u[2] - d * look_u[2]]
        up_u, un = self._norm3(up_u)
        if un < 1e-12:
            up_u = [0.0, 0.0, 1.0] if abs(look_u[2]) < 0.9 else [0.0, 1.0, 0.0]
        cam.SetViewUp(up_u[0], up_u[1], up_u[2])

        _safe_reset_clipping(ren, cam, volume_radius=self._volume_radius)
        self._track_last_sphere = cur
        iren.Render()

    def _display_to_world_on_focal_plane(self, ren, cam, dx, dy):
        """World coords of display pixel at the depth of the focal point."""
        fp = cam.GetFocalPoint()
        ren.SetWorldPoint(fp[0], fp[1], fp[2], 1.0)
        ren.WorldToDisplay()
        z = ren.GetDisplayPoint()[2]

        ren.SetDisplayPoint(float(dx), float(dy), z)
        ren.DisplayToWorld()
        w = ren.GetWorldPoint()
        if abs(w[3]) < 1e-12:
            return None
        return [w[0] / w[3], w[1] / w[3], w[2] / w[3]]

    def _apply_zoom(self, zoom_in=True, strength=1.0):
        """Shared zoom path (host override or built-in Dragonfly zoom).

        Host path never uses zoom-to-cursor pan (breaks Track pivot feel on
        volume rendering). Built-in path also keeps display_xy=None.
        """
        iren = self.GetInteractor()
        host = getattr(self, '_host', None)
        if host is not None and hasattr(host, '_dragonfly_3d_zoom'):
            host._dragonfly_3d_zoom(
                zoom_in=zoom_in, strength=strength, display_xy=None
            )
            return
        ren = self._renderer()
        if ren is None:
            return
        apply_dragonfly_volume_zoom(
            ren,
            zoom_in=zoom_in,
            strength=strength,
            volume_center=tuple(self._pivot),
            volume_radius=self._volume_radius,
            volume_bounds=self._volume_bounds,
            display_xy=None,
        )
        if iren:
            iren.Render()

    def Dolly(self, factor):
        """Right-drag zoom — continuous, proportional to drag (smooth)."""
        import math
        factor = max(float(factor), 1e-6)
        zoom_in = factor > 1.0
        strength = abs(math.log(factor)) / math.log(1.06)
        strength = max(0.08, min(strength, 2.5))
        self._apply_zoom(zoom_in=zoom_in, strength=strength)

    def _zoom_to_cursor(self, zoom_in=True):
        """VTK wheel backup — mild strength (Qt path is preferred)."""
        self._apply_zoom(zoom_in=zoom_in, strength=1.0)

    # ── VTK overrides ───────────────────────────────────────────────────
    def OnLeftButtonDown(self):
        # Already grabbed by high-priority observer — do not start Track
        if getattr(self, '_df_clip_dragging', False):
            self._track_active = False
            return
        # Dragonfly clip-box face grab (host) — steals LMB from Track rotate
        host = getattr(self, '_host', None)
        if (
            host is not None
            and getattr(host, '_df_clip_enabled', False)
            and hasattr(host, '_df_clip_3d_on_left_down')
        ):
            try:
                if host._df_clip_3d_on_left_down():
                    self._df_clip_dragging = True
                    self._track_active = False
                    return
            except Exception:
                pass
        self._df_clip_dragging = False
        # Start Track gesture (arcball + look-at pivot) before VTK rotate state
        self._begin_track_gesture()
        super().OnLeftButtonDown()

    def OnMouseMove(self):
        host = getattr(self, '_host', None)
        if getattr(self, '_df_clip_dragging', False) and host is not None:
            if hasattr(host, '_df_clip_3d_on_mouse_move'):
                try:
                    host._df_clip_3d_on_mouse_move()
                except Exception:
                    pass
            return
        # Hover highlight on clip faces when not interacting
        if (
            host is not None
            and getattr(host, '_df_clip_enabled', False)
            and hasattr(host, '_df_clip_3d_on_hover')
            and not getattr(self, '_track_active', False)
        ):
            try:
                # Only when no button drag state from base style
                st = self.GetState() if hasattr(self, 'GetState') else 0
                if st == 0:  # VTKIS_NONE
                    host._df_clip_3d_on_hover()
            except Exception:
                pass
        super().OnMouseMove()

    def OnLeftButtonUp(self):
        if getattr(self, '_df_clip_dragging', False):
            host = getattr(self, '_host', None)
            if host is not None and hasattr(host, '_df_clip_3d_on_left_up'):
                try:
                    host._df_clip_3d_on_left_up()
                except Exception:
                    pass
            self._df_clip_dragging = False
            self._track_active = False
            self._track_last_sphere = None
            return
        super().OnLeftButtonUp()
        self._track_active = False
        self._track_last_sphere = None
        self._ensure_camera_sane(do_render=False)

    def OnMiddleButtonUp(self):
        super().OnMiddleButtonUp()
        self._ensure_camera_sane(do_render=False)

    def OnRightButtonUp(self):
        super().OnRightButtonUp()
        self._ensure_camera_sane(do_render=False)

    def _ensure_camera_sane(self, do_render=False):
        """Only fix non-finite camera — do not clamp zoom depth."""
        ren = self._renderer()
        if ren is None:
            return
        try:
            cam = ren.GetActiveCamera()
            if cam is None:
                return
            pos = cam.GetPosition()
            fp = cam.GetFocalPoint()
            if not _finite3(pos) or not _finite3(fp):
                cx, cy, cz = self._pivot
                R = max(self._volume_radius, 1.0)
                cam.SetFocalPoint(cx, cy, cz)
                cam.SetPosition(cx, cy - R * 2.5, cz)
            _safe_reset_clipping(ren, cam, volume_radius=self._volume_radius)
            if do_render:
                iren = self.GetInteractor()
                if iren:
                    iren.Render()
        except Exception:
            return

    def OnMouseWheelForward(self):
        self._zoom_to_cursor(zoom_in=True)

    def OnMouseWheelBackward(self):
        self._zoom_to_cursor(zoom_in=False)


class LoadSegmentationThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object)
    
    def __init__(self, file_path, downsample_factor=1):
        super().__init__()
        self.file_path = file_path
        self.downsample_factor = downsample_factor
        
    def run(self):
        try:
            self.progress.emit(10, "Loading TIFF files...")
            directory = Path(self.file_path).parent
            data = io.imread(self.file_path)
            
            if data.ndim == 2:
                self.progress.emit(20, "Loading image stack...")
                all_files = sorted(glob.glob(str(directory / "*.tif*")))
                if len(all_files) > 1:
                    data = np.array([io.imread(f) for f in all_files])
                else:
                    data = data[np.newaxis, :, :]
            
            self.progress.emit(50, f"Loaded shape (Z,Y,X): {data.shape}")
            
            if self.downsample_factor > 1:
                self.progress.emit(60, f"Downsampling...")
                data = data[::self.downsample_factor, 
                           ::self.downsample_factor, 
                           ::self.downsample_factor]
            
            self.progress.emit(100, "Complete!")
            self.finished.emit(data, None)
            
        except Exception as e:
            import traceback
            self.finished.emit(None, f"{str(e)}\n{traceback.format_exc()}")


class RenderSegmentationThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object)
    
    def __init__(self, segmentation_data, colors, spacing, decimate_factor=0.5, smooth_iterations=10):
        super().__init__()
        self.segmentation_data = segmentation_data
        self.colors = colors
        self.spacing = spacing
        self.decimate_factor = decimate_factor
        self.smooth_iterations = smooth_iterations
        
    def run(self):
        try:
            actors = {}
            class_values = [0, 128, 255]
            
            for idx, class_value in enumerate(class_values):
                progress = int(33 * idx)
                self.progress.emit(progress, f"Processing class {class_value}...")
                
                mask = (self.segmentation_data == class_value).astype(np.uint8)
                if np.any(mask):
                    actor = self.create_volume_actor(
                        mask, self.colors[class_value],
                        self.spacing, self.decimate_factor, self.smooth_iterations
                    )
                    actors[class_value] = actor
            
            self.progress.emit(100, "Rendering complete!")
            self.finished.emit(actors, None)
            
        except Exception as e:
            import traceback
            self.finished.emit(None, f"{str(e)}\n{traceback.format_exc()}")
    

    def create_volume_actor(self, mask, color, spacing, decimate_factor, smooth_iterations):
        z, y, x = mask.shape
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(x, y, z)
        vtk_data.SetSpacing(spacing[0], spacing[1], spacing[2])
        
        mask_fortran = np.transpose(mask, (2, 1, 0))
        flat_mask = np.ascontiguousarray(mask_fortran.flatten('F'))
        
        vtk_array = numpy_support.numpy_to_vtk(flat_mask, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_data.GetPointData().SetScalars(vtk_array)
        
        surface = vtk.vtkMarchingCubes()
        surface.SetInputData(vtk_data)
        surface.SetValue(0, 0.5)
        surface.Update()
        
        if decimate_factor > 0 and decimate_factor < 1:
            decimate = vtk.vtkDecimatePro()
            decimate.SetInputConnection(surface.GetOutputPort())
            decimate.SetTargetReduction(decimate_factor)
            decimate.PreserveTopologyOn()
            decimate.Update()
            output_port = decimate.GetOutputPort()
        else:
            output_port = surface.GetOutputPort()
        
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(output_port)
        smoother.SetNumberOfIterations(smooth_iterations)
        smoother.Update()
        

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(smoother.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.SplittingOff()
        normals.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())  
        mapper.ScalarVisibilityOff()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        prop = actor.GetProperty()
        prop.SetColor(color)
        

        prop.SetRepresentationToSurface()
        prop.BackfaceCullingOff()
        prop.ShadingOn()

        if np.allclose(color, self.colors.get(0, [0, 0, 0])):  # BG
            prop.SetOpacity(0.15)       
            prop.SetAmbient(0.3)
            prop.SetDiffuse(0.5)
            prop.SetSpecular(0.0)
        else:
        
            prop.SetOpacity(1.0)        
            prop.SetAmbient(0.8)      
            prop.SetDiffuse(0.85)       
            prop.SetSpecular(0.3)       
            prop.SetSpecularPower(30.0) 
        
        return actor


from PyQt5.QtWidgets import QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QDialogButtonBox, QCheckBox

class RawImportDialog(QDialog):
    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Import RAW: {filename}")
        layout = QFormLayout(self)
        self.spin_width = QSpinBox()
        self.spin_width.setRange(1, 10000)
        self.spin_width.setValue(2000)
        
        self.spin_height = QSpinBox()
        self.spin_height.setRange(1, 10000)
        self.spin_height.setValue(2000)
        
        self.spin_depth = QSpinBox()
        self.spin_depth.setRange(1, 20000)
        self.spin_depth.setValue(2000)
        
        self.spin_offset = QSpinBox()
        self.spin_offset.setRange(0, 1000000000)
        self.spin_offset.setValue(0)
        
        self.chk_little_endian = QCheckBox("Little Endian byte order")
        self.chk_little_endian.setChecked(True)
        
        self.combo_dtype = QComboBox()
        self.combo_dtype.addItems([
            "uint8 (Unsigned 8-bit)", 
            "uint16 (Unsigned 16-bit)", 
            "uint32 (Unsigned 32-bit)", 
            "int8 (Signed 8-bit)", 
            "int16 (Signed 16-bit)", 
            "int32 (Signed 32-bit)", 
            "float32", 
            "float64"
        ])
        self.combo_dtype.setCurrentIndex(1)  # Default to uint16 for HBM semiconductor data
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        
        layout.addRow("Width (X):", self.spin_width)
        layout.addRow("Height (Y):", self.spin_height)
        layout.addRow("Depth (Z slices):", self.spin_depth)
        layout.addRow("Data Type:", self.combo_dtype)
        layout.addRow("Header Offset (bytes):", self.spin_offset)
        layout.addRow("", self.chk_little_endian)
        layout.addRow(btn_box)

    def get_data(self):
        w = self.spin_width.value()
        h = self.spin_height.value()
        d = self.spin_depth.value()
        dt_str = self.combo_dtype.currentText().split()[0]
        
        prefix = '<' if self.chk_little_endian.isChecked() else '>'
        endian_map = {
            'uint8': 'u1', 'int8': 'i1',
            'uint16': 'u2', 'int16': 'i2',
            'uint32': 'u4', 'int32': 'i4',
            'float32': 'f4', 'float64': 'f8'
        }
        
        if dt_str in endian_map and endian_map[dt_str][-1] != '1':
            dt = np.dtype(prefix + endian_map[dt_str])
        else:
            dt = np.dtype(dt_str)
            
        offset = self.spin_offset.value()
        return (d, h, w), dt, offset

class LoadVolumeThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object)
    
    def __init__(
        self,
        file_path,
        downsample_factor=1,
        raw_shape=None,
        raw_dtype=None,
        raw_offset=0,
        use_large_volume_engine=False,
    ):
        super().__init__()
        self.file_path = file_path
        self.downsample_factor = downsample_factor
        self.raw_shape = raw_shape
        self.raw_dtype = raw_dtype
        self.raw_offset = raw_offset
        self.use_large_volume_engine = bool(use_large_volume_engine)
        
    def run(self):
        try:
            import time as _time
            _t_start = _time.perf_counter()
            path = Path(self.file_path)
            
            if path.suffix.lower() in ['.raw', '.bin']:
                self.progress.emit(5, f"Preparing RAW file {path.name}...")
                if not self.raw_shape or not self.raw_dtype:
                    raise ValueError("Raw shape and dtype must be provided for .raw files")
                
                dtype = np.dtype(self.raw_dtype)
                expected_size = self.raw_shape[0] * self.raw_shape[1] * self.raw_shape[2]
                
                # Check file size with offset
                import os
                file_size = os.path.getsize(str(path))
                required_bytes = expected_size * dtype.itemsize + self.raw_offset
                
                if file_size < required_bytes:
                    raise ValueError(f"RAW file is too small! Expected {required_bytes} bytes for this shape and offset, but file is {file_size} bytes.")
                
                total_bytes = expected_size * dtype.itemsize
                import time
                t0 = time.perf_counter()

                # large_volume_engine: open VolumeStore (memmap) — no 15 GB RAM copy
                if self.use_large_volume_engine:
                    from inno3d.core.volume_store import (
                        large_volume_byte_threshold,
                        open_raw_volume_store,
                    )
                    if total_bytes >= large_volume_byte_threshold() or self.use_large_volume_engine:
                        self.progress.emit(
                            10,
                            f"Opening RAW via VolumeStore ({total_bytes / (1024**3):.1f} GB, no full copy)...",
                        )
                        store = open_raw_volume_store(
                            path,
                            tuple(self.raw_shape),
                            dtype,
                            offset=self.raw_offset,
                            start_pyramid=True,
                        )
                        elapsed = time.perf_counter() - t0
                        self.progress.emit(
                            100,
                            f"VolumeStore ready {store.shape} in {elapsed:.1f}s (out-of-core)",
                        )
                        self.finished.emit(store, None)
                        return
                
                # Memory-mapped I/O: near-instant "loading" on 128GB RAM systems.
                # The OS maps the file directly into virtual memory — no sequential 
                # read required. Pages are loaded on-demand by the OS page cache.
                self.progress.emit(10, f"Memory-mapping RAW ({total_bytes / (1024**3):.1f} GB)...")
                
                mmap_data = np.memmap(
                    str(path), dtype=dtype, mode='r',
                    offset=self.raw_offset, shape=self.raw_shape
                )
                
                self.progress.emit(30, "Copying to RAM for fast access...")
                
                # Copy from mmap to contiguous RAM — this forces all pages to load
                # and gives us a writable array. On 128GB DDR5 this takes ~2-4s for 16GB.
                data = np.array(mmap_data, dtype=dtype, copy=True)
                del mmap_data  # Release mmap
                
                elapsed = time.perf_counter() - t0
                
                if not data.dtype.isnative:
                    data = data.byteswap().newbyteorder()
                
                self.progress.emit(85, f"Loaded {data.shape} in {elapsed:.1f}s")
                
            else:
                self.progress.emit(5, "Scanning files...")
                
                if path.is_dir():
                    all_files = sorted(glob.glob(str(path / "*.tif*")))
                    if not all_files:
                        raise ValueError(f"No .tif files found in {self.file_path}")
                    
                    n = len(all_files)
                    self.progress.emit(8, f"Found {n} files, reading first...")
                    
                    import time
                    t0 = time.perf_counter()
                    
                    # Use tifffile for faster reads (no overhead of skimage plugin system)
                    try:
                        import tifffile
                        _read_fn = tifffile.imread
                    except ImportError:
                        _read_fn = io.imread
                    
                    first = _read_fn(all_files[0])
                    first_sq = np.squeeze(first)

                    if first_sq.ndim == 3:
                        # Files inside directory are already 3D volume stacks (e.g. 900x236x269)
                        if n == 1:
                            data = first_sq
                        else:
                            self.progress.emit(10, f"Loading {n} 3D volume files...")
                            from concurrent.futures import ThreadPoolExecutor, as_completed
                            import os as _os
                            num_workers = min(16, max(4, _os.cpu_count() or 8))

                            def _load_stack(idx):
                                return idx, np.squeeze(_read_fn(all_files[idx]))

                            results = {0: first_sq}
                            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                                futures = {executor.submit(_load_stack, i): i for i in range(1, n)}
                                for future in as_completed(futures):
                                    idx, stk = future.result()
                                    results[idx] = stk
                            stacks = [results[i] for i in range(n)]
                            data = np.concatenate(stacks, axis=0)
                    else:
                        # Single 2D slices
                        h, w = first_sq.shape[:2]
                        total_gb = n * h * w * first.itemsize / (1024**3)
                        self.progress.emit(10, f"Allocating {total_gb:.1f} GB...")
                        
                        data = np.empty((n, h, w), dtype=first.dtype)
                        data[0] = first_sq
                        
                        # Parallel loading: use 16 workers to saturate NVMe + Xeon W7 cores.
                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        import os as _os
                        
                        num_workers = min(16, max(4, _os.cpu_count() or 8))
                        progress_interval = max(1, n // 100)  # Update progress every 1%
                        
                        def _load_slice(idx):
                            return idx, _read_fn(all_files[idx])
                        
                        loaded_count = 1  # first slice already loaded
                        with ThreadPoolExecutor(max_workers=num_workers) as executor:
                            futures = {executor.submit(_load_slice, i): i for i in range(1, n)}
                            for future in as_completed(futures):
                                idx, slc = future.result()
                                slc_sq = np.squeeze(slc)
                                if slc_sq.ndim == 3 and slc_sq.shape[1:] == (h, w):
                                    slc_sq = slc_sq[0]
                                if slc_sq.shape == (h, w):
                                    data[idx] = slc_sq
                                else:
                                    print(f"[LoadVolumeThread] Warning: slice {idx} shape {slc_sq.shape} != ({h}, {w})")
                                loaded_count += 1
                                if loaded_count % progress_interval == 0 or loaded_count == n:
                                    elapsed = time.perf_counter() - t0
                                    speed = (loaded_count * h * w * first.itemsize) / (1024**3) / max(elapsed, 0.01)
                                    prog = 10 + int(75 * loaded_count / n)
                                    self.progress.emit(prog, f"Slice {loaded_count}/{n} | {speed:.1f} GB/s | {num_workers} threads")
                else:
                    directory = path.parent
                    
                    import os, time
                    file_size = os.path.getsize(str(path))
                    file_size_mb = file_size / (1024 * 1024)
                    t0 = time.perf_counter()
                    
                    self.progress.emit(10, f"Loading {path.name} ({file_size_mb:.0f} MB)...")
                    
                    # Try tifffile for multi-threaded decompression (massively faster on Xeon)
                    try:
                        import tifffile
                        # Use all available CPU threads for decompression + I/O
                        import os as _os
                        num_threads = min(_os.cpu_count() or 8, 16)
                        io_threads = min(4, num_threads)
                        self.progress.emit(15, f"Reading TIFF ({num_threads} decode + {io_threads} I/O threads)...")
                        data = tifffile.imread(str(path), maxworkers=num_threads, ioworkers=io_threads)
                    except (ImportError, Exception):
                        # Fallback to skimage
                        self.progress.emit(12, f"Reading TIFF (single-thread)...")
                        data = io.imread(self.file_path)
                    
                    elapsed = time.perf_counter() - t0
                    self.progress.emit(60, f"Loaded in {elapsed:.1f}s, processing...")
                    
                    if data.ndim == 2:
                        self.progress.emit(62, "Checking for multi-file stack...")
                        all_files = sorted(glob.glob(str(directory / "*.tif*")))
                        if len(all_files) > 1:
                            from concurrent.futures import ThreadPoolExecutor, as_completed
                            
                            n = len(all_files)
                            h, w = data.shape
                            new_data = np.zeros((n, h, w), dtype=data.dtype)
                            idx_found = 0
                            
                            num_workers = min(8, max(2, n // 10))
                            batch_size = max(1, n // 50)
                            
                            def _load_slc(filepath):
                                return io.imread(filepath)
                            
                            # Parallel load of individual TIFF files
                            loaded = 0
                            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                                futures = {executor.submit(_load_slc, f): i for i, f in enumerate(all_files)}
                                for future in as_completed(futures):
                                    i = futures[future]
                                    slc = future.result()
                                    if slc.shape == (h, w):
                                        new_data[i] = slc
                                        idx_found += 1
                                    loaded += 1
                                    if loaded % batch_size == 0 or loaded == n:
                                        prog = 62 + int(23 * loaded / n)
                                        self.progress.emit(prog, f"Loading slice {loaded}/{n} ({num_workers} workers)...")
                            data = new_data[:max(idx_found, 1)]
                        else:
                            data = data[np.newaxis, :, :]

            _t_loaded = _time.perf_counter()
            total_gb = data.nbytes / (1024**3)
            self.progress.emit(88, f"Loaded {data.shape} ({total_gb:.1f} GB) in {_t_loaded - _t_start:.1f}s")
            
            if self.downsample_factor > 1:
                self.progress.emit(90, f"Downsampling by {self.downsample_factor}x...")
                data = data[::self.downsample_factor, 
                           ::self.downsample_factor, 
                           ::self.downsample_factor]
                self.progress.emit(95, f"Downsampled to {data.shape}")
            
            _t_total = _time.perf_counter() - _t_start
            self.progress.emit(100, f"Complete in {_t_total:.1f}s")
            self.finished.emit(data, None)
            
        except Exception as e:
            import traceback
            self.finished.emit(None, f"{str(e)}\n{traceback.format_exc()}")


# ==================== MULTI-PLANAR VIEW WITH CROSSHAIR ====================
class SliceControlOverlay(QFrame):
    """Floating button strip for a VTK slice/3D viewport.

    Uses a *tool* window owned by the parent so it can paint above the native
    VTK HWND, but deliberately does **not** use WindowStaysOnTopHint (that
    leaked buttons on top of other apps like VS Code when the main window
    lost focus). Visibility is strictly tied to the host main window.
    """
    def __init__(self, parent, orientation, on_fullscreen, on_reset, on_reverse_z=None):
        super().__init__(parent)
        self.orientation = orientation
        self.on_fullscreen = on_fullscreen
        self.on_reset = on_reset
        self.on_reverse_z = on_reverse_z

        # Tool window: floats above its parent *window*, not above the entire desktop.
        # WindowDoesNotAcceptFocus avoids stealing focus from the main app.
        self.setWindowFlags(
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        self.setMouseTracking(True)
        # Debounced path only for resize/activate (not for window drag — that must be instant)
        self._reposition_timer = QTimer(self)
        self._reposition_timer.setSingleShot(True)
        self._reposition_timer.timeout.connect(self.update_position)
        # High-frequency follow while user drags the title bar (Windows can coalesce Move)
        self._drag_follow_timer = QTimer(self)
        self._drag_follow_timer.setInterval(8)  # ~120 Hz
        self._drag_follow_timer.timeout.connect(self._sync_move_only)
        self._drag_follow_stop_timer = QTimer(self)
        self._drag_follow_stop_timer.setSingleShot(True)
        self._drag_follow_stop_timer.setInterval(80)
        self._drag_follow_stop_timer.timeout.connect(self._stop_drag_follow)
        self._main_win_filtered = False
        self._app_filtered = False
        self._overlay_wh = None  # last applied (w, h) — avoid setFixedSize on every Move
        self._last_global_pos = None
        if self.parent():
            self.parent().installEventFilter(self)
        self.init_ui()
        # Start hidden; update_position() will show only when host is visible+active
        self.hide()
        
    def init_ui(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        
        # Root layout — horizontal: slider left, button column right
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        
        # ─── Left: Opacity slider (hidden by default, pops on hover) ───
        self.slider_bar = QFrame()
        self.slider_bar.setObjectName("SliderBar")
        self.slider_bar.setFixedWidth(26)
        slider_layout = QVBoxLayout(self.slider_bar)
        slider_layout.setContentsMargins(2, 6, 2, 6)
        self.opacity_slider = QSlider(Qt.Vertical)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(70)
        self.opacity_slider.setToolTip("Overlay Opacity")
        slider_layout.addWidget(self.opacity_slider)
        root.addWidget(self.slider_bar)
        
        # ─── Right: Button row ───
        self.btn_row = QWidget()
        self.btn_row.setAttribute(Qt.WA_TranslucentBackground, True)
        is_axial = (self.orientation == 'axial')
        
        btn_layout = QBoxLayout(QBoxLayout.TopToBottom, self.btn_row)
        self.btn_layout = btn_layout
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)
        # Prevent stretch from squashing fixed-size buttons into each other
        btn_layout.setSizeConstraint(QBoxLayout.SetFixedSize)
        
        BTN_SZ = 30
        
        # Fullscreen button
        self.fs_btn = QToolButton()
        self.fs_btn.setIcon(_ui_icon("maximize.svg", self, QStyle.SP_TitleBarMaxButton))
        self.fs_btn.setIconSize(QSize(14, 14))
        self.fs_btn.setToolTip("Fullscreen")
        self.fs_btn.setFixedSize(BTN_SZ, BTN_SZ)
        self.fs_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.fs_btn.clicked.connect(self.on_fullscreen)
        btn_layout.addWidget(self.fs_btn, 0, Qt.AlignHCenter)
        
        # Reset button
        self.reset_btn = QToolButton()
        self.reset_btn.setIcon(_ui_icon("refresh.svg", self, QStyle.SP_BrowserReload))
        self.reset_btn.setIconSize(QSize(14, 14))
        self.reset_btn.setToolTip("Reset Zoom/Pan")
        self.reset_btn.setFixedSize(BTN_SZ, BTN_SZ)
        self.reset_btn.clicked.connect(self.on_reset)
        btn_layout.addWidget(self.reset_btn, 0, Qt.AlignHCenter)
        
        # C1 / C2 buttons
        self.c1_btn = QToolButton()
        self.c1_btn.setText("C1")
        self.c1_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.c1_btn.setCheckable(True)
        self.c1_btn.setChecked(True)
        self.c1_btn.setFixedSize(BTN_SZ, BTN_SZ)
        self.c1_btn.setToolTip("Toggle Class 1 Overlay")
        self.c1_btn.installEventFilter(self)
        btn_layout.addWidget(self.c1_btn, 0, Qt.AlignHCenter)
        
        self.c2_btn = QToolButton()
        self.c2_btn.setText("C2")
        self.c2_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.c2_btn.setCheckable(True)
        self.c2_btn.setChecked(True)
        self.c2_btn.setFixedSize(BTN_SZ, BTN_SZ)
        self.c2_btn.setToolTip("Toggle Class 2 Overlay")
        self.c2_btn.installEventFilter(self)
        btn_layout.addWidget(self.c2_btn, 0, Qt.AlignHCenter)

        # 3D volume render master switch (only on 3D pane — under C2)
        # Online mode often leaves this OFF for large volumes; user can re-enable
        # for B2B gap / MES surface review.
        self.render_3d_btn = None
        if self.orientation == "3d":
            self.render_3d_btn = QToolButton()
            self.render_3d_btn.setText("3D")
            self.render_3d_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self.render_3d_btn.setCheckable(True)
            self.render_3d_btn.setChecked(True)
            self.render_3d_btn.setFixedSize(BTN_SZ, BTN_SZ)
            self.render_3d_btn.setToolTip(
                "Toggle 3D volume rendering\n"
                "OFF: skip GPU volume + C1/C2 3D overlays (faster Online)\n"
                "ON: full 3D volume, segmentation overlays, B2B gap, MES pick"
            )
            btn_layout.addWidget(self.render_3d_btn, 0, Qt.AlignHCenter)
            
        # Reverse-Z button (axial only)
        if self.orientation == 'axial' and self.on_reverse_z:
            self.z_btn = QToolButton()
            self.z_btn.setIcon(_ui_icon("arrows-left-right.svg", self))
            self.z_btn.setIconSize(QSize(14, 14))
            self.z_btn.setCheckable(True)
            self.z_btn.setFixedSize(BTN_SZ, BTN_SZ)
            self.z_btn.setToolTip("Reverse Z Direction")
            self.z_btn.clicked.connect(self.on_reverse_z)
            btn_layout.addWidget(self.z_btn, 0, Qt.AlignHCenter)

        # Fixed-size strip — no stretch that would compress 30×30 buttons
        self.btn_row.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        root.addWidget(self.btn_row, 0, Qt.AlignTop)
        
        self._slider_opacity = QGraphicsOpacityEffect(self.slider_bar)
        self._slider_opacity.setOpacity(0.0)
        self.slider_bar.setGraphicsEffect(self._slider_opacity)
        self.slider_bar.hide()
        self._slider_visible = False
        
        self.refresh_theme()

    def refresh_theme(self):
        is_light = SemiconductorTheme.is_light()
        
        # Container: fully transparent — no bounding box
        self.setStyleSheet("background: transparent; border: none;")
        self.btn_row.setStyleSheet("background: transparent; border: none;")
        
        # ── Individual button style: glassmorphism pills ──
        btn_bg     = "rgba(230,240,250,0.88)" if is_light else "rgba(30,35,50,0.78)"
        btn_hover  = "rgba(200,225,248,0.95)" if is_light else "rgba(45,55,75,0.88)"
        btn_border = "rgba(180,200,220,0.5)"  if is_light else "rgba(60,75,95,0.55)"
        btn_text   = "#334" if is_light else "#cdd8e6"
        
        checked_bg     = "rgba(10,143,183,0.18)" if is_light else "rgba(36,183,216,0.20)"
        checked_border = "rgba(10,143,183,0.75)" if is_light else "rgba(36,183,216,0.70)"
        checked_text   = SemiconductorTheme.ACCENT_PRIMARY
        accent         = SemiconductorTheme.ACCENT_PRIMARY
        
        common_qss = f"""
            QToolButton {{
                background: {btn_bg};
                border: 1px solid {btn_border};
                border-radius: 6px;
                color: {btn_text};
                font-size: 8.5pt;
                font-weight: 700;
                padding: 0px;
            }}
            QToolButton:hover {{
                background: {btn_hover};
                border: 1px solid {accent};
            }}
            QToolButton:checked {{
                background: {checked_bg};
                border: 1px solid {checked_border};
                border-bottom: 2px solid {accent};
                color: {checked_text};
            }}
        """
        
        # Apply to all buttons
        buttons = [self.fs_btn, self.reset_btn, self.c1_btn, self.c2_btn]
        if getattr(self, "render_3d_btn", None) is not None:
            buttons.append(self.render_3d_btn)
        if hasattr(self, 'z_btn'):
            buttons.append(self.z_btn)
            
        for btn in buttons:
            btn.setStyleSheet(common_qss)
            
        # ── Slider bar style ──
        slider_bg    = "rgba(235,242,250,0.92)" if is_light else "rgba(25,30,45,0.85)"
        slider_bdr   = "rgba(180,200,220,0.5)"  if is_light else "rgba(55,65,85,0.6)"
        groove_bg    = "rgba(134,157,181,0.4)"   if is_light else "rgba(50,60,80,0.6)"
        
        self.slider_bar.setStyleSheet(f"""
            QFrame#SliderBar {{
                background: {slider_bg};
                border: 1px solid {slider_bdr};
                border-radius: 6px;
            }}
            QSlider::groove:vertical {{
                background: {groove_bg};
                width: 4px;
                border-radius: 2px;
            }}
            QSlider::handle:vertical {{
                background: {accent};
                height: 12px;
                width: 12px;
                margin: 0 -4px;
                border-radius: 6px;
            }}
            QSlider::add-page:vertical {{
                background: {accent};
                border-radius: 2px;
            }}
        """)

    def set_fullscreen(self, is_fs):
        """Re-anchor the tool strip after enter/exit in-grid fullscreen.

        Keep the same look as the non-fullscreen pane: vertical button column,
        maximize icon, and "Fullscreen" tooltip (toggle still exits when
        already maximized). Horizontal strip + minimize icon were removed so
        FS state no longer looks different from the normal strip.
        """
        # Always keep TopToBottom — matches non-fullscreen appearance
        if self.btn_layout.direction() != QBoxLayout.TopToBottom:
            self.btn_layout.setDirection(QBoxLayout.TopToBottom)
            self._overlay_wh = None
            self.btn_row.adjustSize()
            self.adjustSize()
        self.update_position()

    def _host_is_live(self):
        """True when MPR tool strip should be visible.

        Must NOT require ``isActiveWindow()`` — Online progress dialogs, log
        clicks, wafer-map clicks, and sidebar focus all deactivate the main
        window briefly and used to hide C1/C2 forever until the user clicked
        an MPR pane again.
        """
        container = self.parent()
        if container is None:
            return False
        main_win = container.window()
        if main_win is None:
            return False
        if not main_win.isVisible():
            return False
        if main_win.isMinimized():
            return False
        if not container.isVisible():
            return False
        # Zero-size / not laid out yet
        if container.width() < 8 or container.height() < 8:
            return False
        # Only hide when the *process* lost focus (alt-tab to VS Code, etc.)
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None and app.activeWindow() is None:
                # No Qt window of this app has focus → user left the app
                return False
        except Exception:
            pass
        return True

    def _button_count(self):
        """How many tool buttons are in the strip (drives overlay height/width)."""
        n = 0
        for name in ("fs_btn", "reset_btn", "c1_btn", "c2_btn", "render_3d_btn", "z_btn"):
            btn = getattr(self, name, None)
            if btn is not None:
                n += 1
        return max(n, 1)

    def _desired_size(self):
        """Overlay size from button count + optional opacity slider.

        Root layout is always HBox: [slider_bar | btn_row] with a vertical
        button column (same in fullscreen and normal). Showing the slider
        grows width leftward so 30×30 button hit targets stay intact.
        """
        slider_visible = hasattr(self, "slider_bar") and self.slider_bar.isVisible()
        n = self._button_count()
        btn_sz = 30
        gap = 4
        # n buttons + (n-1) gaps between them
        strip = n * btn_sz + max(0, n - 1) * gap
        # Small outer pad so last button isn't clipped by setFixedSize
        pad = 2
        strip_ext = strip + pad
        slider_w = 26
        root_gap = 4  # matches root.setSpacing(4)

        # Vertical button column; slider docks on the left when open
        if slider_visible:
            w = slider_w + root_gap + btn_sz
            h = strip_ext
        else:
            w = btn_sz
            h = strip_ext
        return w, h

    def _sync_move_only(self):
        """Ultra-light reposition for title-bar drag — no show/raise/setFixedSize.

        Called on every main-window Move (and ~120 Hz while drag follow is active)
        so the tool strip tracks the app without visible lag.
        """
        container = self.parent()
        if container is None or not self.isVisible():
            return
        wh = self._overlay_wh or (self.width(), self.height())
        w, h = wh
        if w < 1 or h < 1:
            return
        global_pos = container.mapToGlobal(QPoint(container.width() - w - 10, 10))
        if global_pos != self._last_global_pos:
            self.move(global_pos)
            self._last_global_pos = global_pos

    def _start_drag_follow(self):
        """Keep syncing during title-bar drag until Move events stop."""
        if not self.isVisible():
            return
        if not self._drag_follow_timer.isActive():
            self._drag_follow_timer.start()
        # Reset idle stop — when Moves stop for 80ms, end follow loop
        self._drag_follow_stop_timer.start()

    def _stop_drag_follow(self):
        self._drag_follow_timer.stop()
        # One final snap after drag ends
        if self.isVisible():
            self._sync_move_only()

    def update_position(self):
        """Full layout + visibility + position (use on show/resize/activate)."""
        container = self.parent()
        if container is None:
            self.hide()
            return

        main_win = container.window()

        # Install filters once: main window + app (for alt-tab / focus loss)
        if main_win and not getattr(self, "_main_win_filtered", False):
            main_win.installEventFilter(self)
            self._main_win_filtered = True
        if not getattr(self, "_app_filtered", False):
            try:
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                if app is not None:
                    app.installEventFilter(self)
                    self._app_filtered = True
            except Exception:
                pass

        if not self._host_is_live():
            self.hide()
            return

        w, h = self._desired_size()
        if self._overlay_wh != (w, h):
            self.setFixedSize(w, h)
            self._overlay_wh = (w, h)

        # Anchor top-right of the VTK host so growing for the opacity slider
        # expands left/down and keeps button global positions stable.
        global_pos = container.mapToGlobal(QPoint(container.width() - w - 10, 10))
        if global_pos != self._last_global_pos:
            self.move(global_pos)
            self._last_global_pos = global_pos

        # show without activating so we don't steal focus from VTK/main
        if not self.isVisible():
            self.setVisible(True)
        # Always raise above the native VTK HWND after layout changes
        self.raise_()

    def paintEvent(self, event):
        """Nearly-invisible fill so the full rect receives mouse events.

        With WA_TranslucentBackground, fully transparent pixels pass through to
        the VTK HWND underneath. Gaps between glass buttons then steal hover
        from the tool strip (enter/leave thrash, clicks miss). Alpha=1 is
        invisible but creates a proper hit target for the whole strip.
        """
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 1))
        painter.end()
        super().paintEvent(event)

    def _show_slider(self):
        """Fade in the opacity slider and expand its width."""
        if not self._slider_visible:
            self._slider_visible = True
            self.slider_bar.show()
            self._slider_opacity.setOpacity(1.0)
            self.update_position()

    def _hide_slider(self):
        """Fade out the opacity slider and collapse its width."""
        if self._slider_visible:
            self._slider_visible = False
            self._slider_opacity.setOpacity(0.0)
            self.slider_bar.hide()
            self.update_position()

    def _cursor_over_overlay(self):
        """True if global cursor is still over this tool strip (or its children)."""
        try:
            pos = self.mapFromGlobal(QCursor.pos())
            if self.rect().contains(pos):
                return True
            # Child tool windows / popups are rare; still cover child widgets
            w = self.childAt(pos)
            return w is not None
        except Exception:
            return False

    def _check_and_hide_slider(self):
        """Hide slider if mouse has left the overlay entirely."""
        if not self._cursor_over_overlay():
            self._hide_slider()

    def enterEvent(self, event):
        # Keep strip raised above VTK while interacting
        self.raise_()
        self._show_slider()
        super().enterEvent(event)

    def leaveEvent(self, event):
        # Delay hide so moving between buttons / into the slider does not
        # collapse the strip mid-click (especially in horizontal fullscreen).
        QTimer.singleShot(400, self._check_and_hide_slider)
        super().leaveEvent(event)

    def eventFilter(self, obj, event):
        et = event.type()

        # App lost focus entirely (user clicked VS Code, desktop, etc.)
        if et == QEvent.ApplicationDeactivate:
            self._stop_drag_follow()
            self.hide()
            return False
        if et == QEvent.ApplicationActivate:
            # Bring tool strips back after progress dialog / alt-tab return
            self._reposition_timer.start(16)
            return False

        container = self.parent()
        if container is not None:
            main_win = container.window()

            if obj == main_win:
                if et == QEvent.WindowActivate:
                    self._reposition_timer.start(16)
                    return False
                # Do NOT hide on WindowDeactivate — modal Online progress,
                # QMessageBox, and sidebar focus all fire this and used to
                # wipe MPR buttons until the user clicked a VTK pane again.
                if et in (QEvent.Hide, QEvent.Close):
                    self._stop_drag_follow()
                    self.hide()
                    return False
                if et == QEvent.WindowDeactivate:
                    # Keep strips visible while a child dialog of this app is open
                    self._stop_drag_follow()
                    return False
                if et == QEvent.WindowStateChange:
                    if main_win.isMinimized() or not main_win.isVisible():
                        self._stop_drag_follow()
                        self.hide()
                    else:
                        self._reposition_timer.start(16)
                    return False
                # Title-bar drag: Move must be instant — no debounce timer
                if et == QEvent.Move:
                    if self.isVisible():
                        self._sync_move_only()
                        self._start_drag_follow()
                    elif self._host_is_live():
                        self.update_position()
                    return False
                if et == QEvent.Resize:
                    self._stop_drag_follow()
                    if self._host_is_live():
                        self.update_position()
                    else:
                        self.hide()
                    return False

            if obj == container:
                if et in (QEvent.Hide, QEvent.Close):
                    self._stop_drag_follow()
                    self.hide()
                    return False
                if et == QEvent.Move:
                    if self.isVisible():
                        self._sync_move_only()
                        self._start_drag_follow()
                    return False
                if et == QEvent.Resize:
                    self._stop_drag_follow()
                    if self._host_is_live():
                        self.update_position()
                    else:
                        self.hide()
                    return False
                if et == QEvent.Show:
                    self._reposition_timer.start(16)
                    return False

        # Keep opacity slider open while hovering C1/C2 (and while over strip)
        monitored = []
        if hasattr(self, "c1_btn"):
            monitored.append(self.c1_btn)
        if hasattr(self, "c2_btn"):
            monitored.append(self.c2_btn)

        if obj in monitored:
            if et == QEvent.Enter:
                self.raise_()
                self._show_slider()
            elif et == QEvent.Leave:
                QTimer.singleShot(400, self._check_and_hide_slider)
        return super().eventFilter(obj, event)
