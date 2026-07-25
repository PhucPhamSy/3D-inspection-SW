# Shared 3D segmentation mask overlay helpers — Viewer + Teaching.
"""Medical-style glass shell overlay for GPU volume rendering.

Replaces multi-volume solid mask ray-casting with an isosurface mesh
(FlyingEdges / MarchingCubes): lighter per-frame, CT still readable.
"""
from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support


def _normalize_mask_u8(mask_data):
    """Return contiguous uint8 mask (0 / positive)."""
    m = np.ascontiguousarray(mask_data)
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8)
    else:
        # Keep binary-ish; threshold later at 0.5
        if m.max() > 1:
            m = (m > 0).astype(np.uint8)
    return m


def _bbox_nonzero(m, pad=1):
    """Axis-aligned bbox of positive voxels with pad; None if empty.

    Uses axis projections (cheap) instead of ``np.argwhere`` on full FOV.
    """
    # m: (Z,Y,X)
    z_any = np.any(m > 0, axis=(1, 2))
    if not np.any(z_any):
        return None
    y_any = np.any(m > 0, axis=(0, 2))
    x_any = np.any(m > 0, axis=(0, 1))
    z_idx = np.flatnonzero(z_any)
    y_idx = np.flatnonzero(y_any)
    x_idx = np.flatnonzero(x_any)
    vz, vy, vx = m.shape
    z0 = max(0, int(z_idx[0]) - pad)
    y0 = max(0, int(y_idx[0]) - pad)
    x0 = max(0, int(x_idx[0]) - pad)
    z1 = min(vz, int(z_idx[-1]) + pad + 1)
    y1 = min(vy, int(y_idx[-1]) + pad + 1)
    x1 = min(vx, int(x_idx[-1]) + pad + 1)
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return None
    return z0, z1, y0, y1, x0, x1


def create_mask_surface_actor(
    mask_data,
    spacing,
    color_val,
    base_opacity=0.35,
    smooth=True,
    edges=False,
):
    """Build a translucent isosurface shell for a binary/class mask.

    Parameters
    ----------
    mask_data : ndarray (Z,Y,X)
        Non-zero = foreground.
    spacing : (sx, sy, sz)
        Must match the grey volume actor spacing (world units).
    color_val : (r,g,b) floats 0..1
    base_opacity : float
        Glass-shell opacity (medical overlay default ~0.3–0.45).
    smooth : bool
        Light WindowedSinc smoothing for softer medical look.
    edges : bool
        Edge lines (good for single-object highlight; busy for full FOV).

    Returns
    -------
    vtk.vtkActor or None
    """
    if mask_data is None:
        return None
    try:
        m = _normalize_mask_u8(mask_data)
    except Exception:
        return None
    if m.ndim != 3 or not np.any(m):
        return None

    s0, s1, s2 = float(spacing[0]), float(spacing[1]), float(spacing[2])
    bbox = _bbox_nonzero(m, pad=1)
    if bbox is None:
        return None
    z0, z1, y0, y1, x0, x1 = bbox
    crop = np.ascontiguousarray(m[z0:z1, y0:y1, x0:x1])
    if not np.any(crop):
        return None

    # VTK image: dims (X,Y,Z), origin at crop corner in world coords
    dz, dy, dx = crop.shape
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(int(dx), int(dy), int(dz))
    vtk_img.SetSpacing(s0, s1, s2)
    vtk_img.SetOrigin(float(x0) * s0, float(y0) * s1, float(z0) * s2)

    flat = np.ascontiguousarray(np.transpose(crop, (2, 1, 0)).ravel(order="F"))
    vtk_arr = numpy_support.numpy_to_vtk(
        flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_arr.SetNumberOfComponents(1)
    vtk_img.GetPointData().SetScalars(vtk_arr)

    # Isosurface at mid-binary
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

    poly = None
    if smooth:
        try:
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputConnection(contour.GetOutputPort())
            smoother.SetNumberOfIterations(10)
            smoother.BoundarySmoothingOff()
            smoother.FeatureEdgeSmoothingOff()
            smoother.SetPassBand(0.1)
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.Update()
            poly = smoother.GetOutput()
        except Exception:
            poly = None
    if poly is None:
        contour.Update()
        poly = contour.GetOutput()
    if poly is None or int(poly.GetNumberOfPoints()) < 3:
        return None

    # Own a stable copy — filter output buffers can be reused
    poly_owned = vtk.vtkPolyData()
    poly_owned.DeepCopy(poly)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_owned)
    mapper.ScalarVisibilityOff()
    try:
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
    except Exception:
        pass

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    cr, cg, cb = float(color_val[0]), float(color_val[1]), float(color_val[2])
    prop.SetColor(cr, cg, cb)
    op = float(np.clip(base_opacity, 0.0, 1.0))
    prop.SetOpacity(op)
    # Glass / medical shell lighting
    prop.SetAmbient(0.38)
    prop.SetDiffuse(0.62)
    prop.SetSpecular(0.28)
    prop.SetSpecularPower(22.0)
    try:
        prop.SetInterpolationToPhong()
    except Exception:
        pass
    # See interior voids through shell
    try:
        prop.BackfaceCullingOff()
        prop.FrontfaceCullingOff()
    except Exception:
        pass
    if edges:
        prop.EdgeVisibilityOn()
        prop.SetEdgeColor(
            min(1.0, cr * 0.35 + 0.65),
            min(1.0, cg * 0.35 + 0.65),
            min(1.0, cb * 0.35 + 0.65),
        )
        prop.SetLineWidth(1.0)
    else:
        prop.EdgeVisibilityOff()

    # Tag for opacity/color helpers (surface vs legacy volume)
    try:
        actor._inno3d_mask_surface = True
    except Exception:
        pass
    return actor


def create_mask_volume_actor(mask_data, spacing, color_val, base_opacity=0.4):
    """Backward-compatible entry — now returns a medical surface shell actor.

    Callers must ``AddActor`` (not ``AddVolume``). Prefer
    ``add_mask_overlay_to_renderer`` / ``create_mask_surface_actor``.
    """
    return create_mask_surface_actor(
        mask_data,
        spacing,
        color_val,
        base_opacity=base_opacity,
        smooth=True,
        edges=False,
    )


def is_mask_surface_actor(actor):
    if actor is None:
        return False
    if getattr(actor, "_inno3d_mask_surface", False):
        return True
    # Heuristic: polydata mapper / vtkProperty with SetOpacity, no TF
    try:
        prop = actor.GetProperty()
        if prop is not None and hasattr(prop, "SetOpacity"):
            if not hasattr(prop, "GetScalarOpacity"):
                return True
            try:
                so = prop.GetScalarOpacity()
                if so is None:
                    return True
            except Exception:
                return True
    except Exception:
        pass
    return False


def set_mask_overlay_opacity(actor, opacity):
    """Set overlay opacity for surface shell or legacy volume actor."""
    if actor is None:
        return
    op = float(np.clip(opacity, 0.0, 1.0))
    prop = actor.GetProperty()
    if prop is None:
        return
    if is_mask_surface_actor(actor):
        prop.SetOpacity(op)
        prop.Modified()
        return
    # Legacy volume TF path
    try:
        so = prop.GetScalarOpacity()
        if so is not None and hasattr(so, "RemoveAllPoints"):
            so.RemoveAllPoints()
            so.AddPoint(0, 0.0)
            so.AddPoint(1, op)
            so.AddPoint(255, op)
            prop.Modified()
            return
    except Exception:
        pass
    if hasattr(prop, "SetOpacity"):
        prop.SetOpacity(op)
        prop.Modified()


def set_mask_overlay_color(actor, color_val):
    """Push RGB colour into surface or legacy volume mask actor."""
    if actor is None:
        return
    r, g, b = float(color_val[0]), float(color_val[1]), float(color_val[2])
    prop = actor.GetProperty()
    if prop is None:
        return
    if is_mask_surface_actor(actor):
        prop.SetColor(r, g, b)
        prop.Modified()
        return
    try:
        color_tf = None
        try:
            color_tf = prop.GetRGBTransferFunction()
        except Exception:
            color_tf = None
        if color_tf is None:
            try:
                color_tf = prop.GetColor()
            except Exception:
                color_tf = None
        if color_tf is not None and hasattr(color_tf, "RemoveAllPoints"):
            color_tf.RemoveAllPoints()
            color_tf.AddRGBPoint(0, 0, 0, 0)
            color_tf.AddRGBPoint(1, r, g, b)
            color_tf.AddRGBPoint(255, r, g, b)
            color_tf.Modified()
        else:
            color_tf = vtk.vtkColorTransferFunction()
            color_tf.AddRGBPoint(0, 0, 0, 0)
            color_tf.AddRGBPoint(1, r, g, b)
            color_tf.AddRGBPoint(255, r, g, b)
            prop.SetColor(color_tf)
        prop.Modified()
    except Exception:
        if hasattr(prop, "SetColor"):
            try:
                prop.SetColor(r, g, b)
            except Exception:
                pass


def add_mask_overlay_to_renderer(renderer, actor):
    """Add mask actor to renderer (surface → AddActor, volume → AddVolume)."""
    if renderer is None or actor is None:
        return
    if is_mask_surface_actor(actor):
        renderer.AddActor(actor)
    else:
        try:
            renderer.AddVolume(actor)
        except Exception:
            renderer.AddActor(actor)


def remove_mask_overlay_from_renderer(renderer, actor):
    """Remove mask actor whether surface or volume."""
    if renderer is None or actor is None:
        return
    try:
        renderer.RemoveActor(actor)
    except Exception:
        pass
    try:
        renderer.RemoveVolume(actor)
    except Exception:
        pass
