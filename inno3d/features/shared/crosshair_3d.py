# inno3d/features/shared/crosshair_3d.py
# -----------------------------------------------------------------------
# Host-agnostic RGB crosshair axes on a 3D Volume renderer (Viewer SoT).
# Used by MultiPlanarView (view_3d_*) and SegmentationTab (_3d_*).
# Host supplies renderer, widget, index position, shape, spacing, R.
# -----------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import vtk

# Viewer SoT colors (sagittal / coronal / axial)
AXIS_COLORS: Dict[str, Tuple[float, float, float]] = {
    "x": (1.0, 0.192, 0.192),   # red — sagittal (X)
    "y": (0.223, 1.0, 0.078),   # green — coronal (Y)
    "z": (0.121, 0.317, 1.0),   # blue — axial (Z)
}


def crosshair_axis_endpoints(
    position_xyz: Sequence[float],
    shape_zyx: Sequence[int],
    spacing_xyz: Sequence[float],
    R: Optional[np.ndarray] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """World-space endpoints for the three crosshair axes.

    Parameters
    ----------
    position_xyz : (cx, cy, cz) volume indices (same as crosshair_position)
    shape_zyx : volume.shape as (Z, Y, X)
    spacing_xyz : (sx, sy, sz) world spacing
    R : optional 3×3 rotation (columns = axis directions). Identity if None.

    Returns
    -------
    dict with keys 'x','y','z' → (p1, p2) each shape (3,) float arrays
    """
    z, y, x = int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2])
    cx, cy, cz = float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2])
    spx, spy, spz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])

    pc = np.array([cx * spx, cy * spy, cz * spz], dtype=float)

    if R is None:
        R = np.eye(3, dtype=float)
    else:
        R = np.asarray(R, dtype=float).reshape(3, 3)

    rx, ry, rz = R[:, 0], R[:, 1], R[:, 2]

    lx = max((x - 1) * spx, 1e-6)
    ly = max((y - 1) * spy, 1e-6)
    lz = max((z - 1) * spz, 1e-6)
    max_len = max(lx, ly, lz) * 2.0

    return {
        "x": (pc - max_len * rx, pc + max_len * rx),
        "y": (pc - max_len * ry, pc + max_len * ry),
        "z": (pc - max_len * rz, pc + max_len * rz),
    }


class Crosshair3DOverlay:
    """Persistent RGB line actors for crosshair on a 3D volume renderer."""

    def __init__(self) -> None:
        self.line_sources: Optional[Dict[str, Any]] = None
        self.actors: list = []

    def _ensure_actors(self) -> None:
        if self.line_sources is not None:
            return
        self.line_sources = {
            "x": vtk.vtkLineSource(),
            "y": vtk.vtkLineSource(),
            "z": vtk.vtkLineSource(),
        }
        self.actors = []
        for axis in ("x", "y", "z"):
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(self.line_sources[axis].GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*AXIS_COLORS[axis])
            actor.GetProperty().SetLineWidth(1)
            actor.SetPickable(False)
            self.actors.append(actor)

    def clear(self, ren=None, widget=None, *, render: bool = False) -> None:
        """Remove actors from renderer (keep sources for reuse)."""
        if ren is not None:
            for actor in self.actors:
                try:
                    ren.RemoveActor(actor)
                except Exception:
                    pass
        if render and widget is not None and ren is not None:
            try:
                widget.GetRenderWindow().Render()
            except Exception:
                pass

    def update(
        self,
        ren,
        widget,
        enabled: bool,
        position_xyz: Optional[Sequence[float]],
        shape_zyx: Optional[Sequence[int]],
        spacing_xyz: Sequence[float],
        R: Optional[np.ndarray] = None,
        *,
        render: bool = True,
        line_width: float = 1.0,
    ) -> bool:
        """Show/hide and reposition 3D crosshair axes.

        Returns True if axes are visible, False if cleared/hidden.
        """
        if ren is None:
            return False

        self._ensure_actors()

        if not enabled or position_xyz is None or shape_zyx is None:
            self.clear(ren, widget, render=render)
            return False

        for actor in self.actors:
            try:
                if not ren.HasViewProp(actor):
                    ren.AddActor(actor)
                actor.GetProperty().SetLineWidth(float(line_width))
            except Exception:
                pass

        ends = crosshair_axis_endpoints(position_xyz, shape_zyx, spacing_xyz, R=R)
        for axis in ("x", "y", "z"):
            p1, p2 = ends[axis]
            src = self.line_sources[axis]
            src.SetPoint1(float(p1[0]), float(p1[1]), float(p1[2]))
            src.SetPoint2(float(p2[0]), float(p2[1]), float(p2[2]))
            try:
                src.Modified()
            except Exception:
                pass

        if render and widget is not None:
            try:
                ren.ResetCameraClippingRange()
            except Exception:
                pass
            try:
                widget.GetRenderWindow().Render()
            except Exception:
                pass
        return True
