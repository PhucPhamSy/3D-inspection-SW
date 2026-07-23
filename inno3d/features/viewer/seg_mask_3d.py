# Shared 3D segmentation mask volume helpers — source of truth = Viewer render_3d.
"""Pixel/voxel mask overlay for GPU volume rendering.

Copied from ``Volume3dMixin.render_3d`` ``create_mask_volume`` so Teaching and
Viewer produce the same look.
"""
from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support


def create_mask_volume_actor(mask_data, spacing, color_val, base_opacity=0.4):
    """Build a ``vtkVolume`` for a binary/class mask (Viewer-identical).

    Parameters
    ----------
    mask_data : ndarray (Z,Y,X) uint8-like
        Non-zero = mask. Viewer passes ``class1_data`` / ``class2_data`` as-is.
    spacing : (sx, sy, sz)
        **Must** match the grey volume actor spacing (index * spacing world).
    color_val : (r,g,b) floats 0..1
    base_opacity : float
    """
    if mask_data is None:
        return None
    vz, vy, vx = mask_data.shape
    s0, s1, s2 = float(spacing[0]), float(spacing[1]), float(spacing[2])

    # Normalize to 0/255 for TF points at 1 and 255
    m = np.ascontiguousarray(mask_data)
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * np.uint8(255)
    else:
        # keep existing levels; ensure binary-ish masks are 0/255
        if m.max() == 1:
            m = m * np.uint8(255)

    vtk_mask = vtk.vtkImageData()
    vtk_mask.SetDimensions(int(vx), int(vy), int(vz))
    vtk_mask.SetSpacing(s0, s1, s2)
    vtk_mask.SetOrigin(0.0, 0.0, 0.0)

    mask_fortran = np.transpose(m, (2, 1, 0))
    flat_mask = np.ascontiguousarray(mask_fortran.flatten("F"))
    vtk_m_array = numpy_support.numpy_to_vtk(
        flat_mask, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_mask.GetPointData().SetScalars(vtk_m_array)

    m_mapper = vtk.vtkGPUVolumeRayCastMapper()
    m_mapper.SetInputData(vtk_mask)
    m_mapper.SetBlendModeToComposite()
    # Match Viewer default quality-ish sampling for solid mask blocks
    try:
        m_mapper.SetAutoAdjustSampleDistances(0)
        min_sp = max(1e-6, min(s0, s1, s2))
        m_mapper.SetSampleDistance(max(0.35, min_sp * 0.5))
        if hasattr(m_mapper, "SetImageSampleDistance"):
            m_mapper.SetImageSampleDistance(1.0)
        if hasattr(m_mapper, "SetUseJittering"):
            m_mapper.SetUseJittering(0)
    except Exception:
        pass

    m_prop = vtk.vtkVolumeProperty()
    color_tf = vtk.vtkColorTransferFunction()
    cr, cg, cb = float(color_val[0]), float(color_val[1]), float(color_val[2])
    color_tf.AddRGBPoint(0, 0, 0, 0)
    color_tf.AddRGBPoint(1, cr, cg, cb)
    color_tf.AddRGBPoint(255, cr, cg, cb)
    m_prop.SetColor(color_tf)

    opacity_tf = vtk.vtkPiecewiseFunction()
    opacity_tf.AddPoint(0, 0.0)
    opacity_tf.AddPoint(1, float(base_opacity))
    opacity_tf.AddPoint(255, float(base_opacity))
    m_prop.SetScalarOpacity(opacity_tf)

    # Viewer: ShadeOn + Ambient/Diffuse — solid filled voxels (not soft cloud)
    m_prop.ShadeOn()
    m_prop.SetAmbient(0.3)
    m_prop.SetDiffuse(0.7)
    # Nearest keeps hard voxel edges (ô vuông pixel)
    try:
        m_prop.SetInterpolationTypeToNearest()
    except Exception:
        m_prop.SetInterpolationTypeToLinear()
    try:
        min_sp = max(1e-6, min(s0, s1, s2))
        m_prop.SetScalarOpacityUnitDistance(min_sp)
    except Exception:
        pass

    m_actor = vtk.vtkVolume()
    m_actor.SetMapper(m_mapper)
    m_actor.SetProperty(m_prop)
    return m_actor
