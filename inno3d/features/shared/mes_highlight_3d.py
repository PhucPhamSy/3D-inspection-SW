# inno3d/features/shared/mes_highlight_3d.py
# -----------------------------------------------------------------------
# Host-agnostic MES selection highlight on 3D Volume (Viewer SoT).
# Host supplies renderer, labeled mask, object_stats, spacing, hooks.
# -----------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import vtk
from vtk.util import numpy_support

from inno3d.features.shared.mes_mapping import (
    first_centroid_world,
    label_set_from_highlights,
    mes_bbox_for_labels,
)


class MESHighlightOverlay:
    """Cyan surface stack for selected MES bumps on a 3D renderer."""

    def __init__(self):
        self.actors: List[Any] = []
        self.cache_key = None

    def clear(
        self,
        ren=None,
        widget=None,
        *,
        render: bool = True,
        on_restore_context: Optional[Callable[[], None]] = None,
    ) -> None:
        if ren is not None:
            for actor in self.actors:
                try:
                    ren.RemoveActor(actor)
                except Exception:
                    pass
        self.actors = []
        self.cache_key = None
        if on_restore_context is not None:
            try:
                on_restore_context()
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
        labeled,
        object_stats: Optional[Sequence[Dict[str, Any]]],
        selected_highlight_objects: Optional[Sequence[Sequence[Any]]],
        spacing: Tuple[float, float, float],
        *,
        z_offset: int = 0,
        highlight_color: Optional[Sequence[float]] = None,
        on_dim_context: Optional[Callable[[], None]] = None,
        on_restore_context: Optional[Callable[[], None]] = None,
    ) -> bool:
        """Rebuild cyan pick surfaces for the current highlight set.

        Returns True if actors are shown (or cache hit), False if cleared.
        """
        if ren is None or labeled is None:
            self.clear(ren, widget, render=True, on_restore_context=on_restore_context)
            return False

        label_set = label_set_from_highlights(selected_highlight_objects)
        if not label_set:
            self.clear(ren, widget, render=True, on_restore_context=on_restore_context)
            return False

        bbox = mes_bbox_for_labels(object_stats, label_set, z_offset=z_offset)
        if bbox is None:
            self.clear(ren, widget, render=True, on_restore_context=on_restore_context)
            return False

        z0, z1, y0, y1, x0, x1 = bbox
        Z, Y, X = labeled.shape
        pad = 2
        z0 = max(0, z0 - pad)
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        z1 = min(Z - 1, z1 + pad)
        y1 = min(Y - 1, y1 + pad)
        x1 = min(X - 1, x1 + pad)
        if z1 < z0 or y1 < y0 or x1 < x0:
            self.clear(ren, widget, render=True, on_restore_context=on_restore_context)
            return False

        cache_key = (frozenset(label_set), z0, z1, y0, y1, x0, x1, "mes_hl_v2")
        if cache_key == self.cache_key and self.actors:
            if on_dim_context is not None:
                try:
                    on_dim_context()
                except Exception:
                    pass
            if widget is not None:
                try:
                    widget.GetRenderWindow().Render()
                except Exception:
                    pass
            return True

        crop = labeled[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
        if len(label_set) == 1:
            lab0 = next(iter(label_set))
            mask = crop == lab0
        else:
            mask = np.isin(crop, list(label_set))
        if not np.any(mask):
            self.clear(ren, widget, render=True, on_restore_context=on_restore_context)
            return False

        mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))
        # Clear previous without undim flicker
        self.clear(ren, widget, render=False, on_restore_context=None)

        sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        vtk_img = vtk.vtkImageData()
        dz, dy, dx = mask_u8.shape
        vtk_img.SetDimensions(dx, dy, dz)
        vtk_img.SetSpacing(sx, sy, sz)
        vtk_img.SetOrigin(float(x0) * sx, float(y0) * sy, float(z0) * sz)
        flat = np.ascontiguousarray(np.transpose(mask_u8, (2, 1, 0)).ravel(order="F"))
        vtk_arr = numpy_support.numpy_to_vtk(
            flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
        )
        vtk_arr.SetNumberOfComponents(1)
        vtk_img.GetPointData().SetScalars(vtk_arr)

        try:
            contour = vtk.vtkFlyingEdges3D()
        except Exception:
            contour = vtk.vtkMarchingCubes()
        contour.SetInputData(vtk_img)
        contour.SetValue(0, 0.5)
        contour.ComputeNormalsOn()
        try:
            contour.ComputeScalarsOff()
        except Exception:
            pass

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.SetNumberOfIterations(8)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetPassBand(0.1)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        poly = smoother.GetOutput()
        if poly is None or int(poly.GetNumberOfPoints()) < 3:
            self.clear(ren, widget, render=True, on_restore_context=on_restore_context)
            return False
        poly_owned = vtk.vtkPolyData()
        poly_owned.DeepCopy(poly)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_owned)
        mapper.ScalarVisibilityOff()
        try:
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
        except Exception:
            pass

        hl = list(highlight_color or [0.0, 1.0, 1.0])
        if sum(hl) < 1.2:
            hl = [0.15, 0.95, 1.0]
        hr, hg, hb = float(hl[0]), float(hl[1]), float(hl[2])

        actors: List[Any] = []

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(hr, hg, hb)
        prop.SetOpacity(0.78)
        prop.SetSpecular(0.55)
        prop.SetSpecularPower(28.0)
        prop.SetAmbient(0.55)
        prop.SetDiffuse(0.65)
        prop.EdgeVisibilityOn()
        prop.SetEdgeColor(1.0, 1.0, 1.0)
        prop.SetLineWidth(2.0)
        try:
            prop.SetInterpolationToPhong()
        except Exception:
            pass
        ren.AddActor(actor)
        actors.append(actor)

        try:
            wire_mapper = vtk.vtkPolyDataMapper()
            wire_mapper.SetInputData(poly_owned)
            wire_mapper.ScalarVisibilityOff()
            wire = vtk.vtkActor()
            wire.SetMapper(wire_mapper)
            wp = wire.GetProperty()
            wp.SetRepresentationToWireframe()
            wp.SetColor(hr, hg, hb)
            wp.SetOpacity(1.0)
            wp.SetLineWidth(1.8)
            wp.SetLighting(False)
            wp.SetAmbient(1.0)
            ren.AddActor(wire)
            actors.append(wire)
        except Exception:
            pass

        try:
            outline = vtk.vtkOutlineFilter()
            outline.SetInputData(poly_owned)
            outline.Update()
            om = vtk.vtkPolyDataMapper()
            om.SetInputConnection(outline.GetOutputPort())
            oa = vtk.vtkActor()
            oa.SetMapper(om)
            op = oa.GetProperty()
            op.SetColor(1.0, 1.0, 0.25)
            op.SetOpacity(0.95)
            op.SetLineWidth(2.5)
            op.SetLighting(False)
            ren.AddActor(oa)
            actors.append(oa)
        except Exception:
            pass

        # Centroid beacon
        try:
            ctr = first_centroid_world(
                object_stats, label_set, (sx, sy, sz), z_offset=z_offset
            )
            if ctr is not None:
                cx, cy, cz = ctr
                min_sp = max(1e-6, min(sx, sy, sz))
                for scale, alpha, col in (
                    (6.0, 0.22, (hr, hg, hb)),
                    (3.2, 0.95, (1.0, 1.0, 0.35)),
                ):
                    sph = vtk.vtkSphereSource()
                    sph.SetCenter(cx, cy, cz)
                    sph.SetRadius(max(1.2, min_sp * scale))
                    sph.SetPhiResolution(16)
                    sph.SetThetaResolution(16)
                    sm = vtk.vtkPolyDataMapper()
                    sm.SetInputConnection(sph.GetOutputPort())
                    sa = vtk.vtkActor()
                    sa.SetMapper(sm)
                    sa.GetProperty().SetColor(float(col[0]), float(col[1]), float(col[2]))
                    sa.GetProperty().SetOpacity(float(alpha))
                    sa.GetProperty().SetLighting(False)
                    ren.AddActor(sa)
                    actors.append(sa)
        except Exception:
            pass

        self.actors = actors
        self.cache_key = cache_key

        if on_dim_context is not None:
            try:
                on_dim_context()
            except Exception:
                pass

        try:
            ren.ResetCameraClippingRange()
        except Exception:
            pass
        if widget is not None:
            try:
                widget.GetRenderWindow().Render()
            except Exception:
                pass
        return True
