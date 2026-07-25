# inno3d/features/shared/orientation_cube.py
# -----------------------------------------------------------------------
# Interactive 3D orientation cube (Viewer SoT): AnnotatedCube + face hover
# glow + ray-pick click → camera face presets.
# Used by Viewer volume_3d and Teaching 3D volume.
# -----------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import vtk

from inno3d.core.styles import SemiconductorTheme

FACE_TO_PRESET = {
    "+X": "Right",
    "-X": "Left",
    "+Y": "Back",
    "-Y": "Front",
    "+Z": "Top",
    "-Z": "Bottom",
}


def theme_primary_rgb() -> Tuple[float, float, float]:
    h = SemiconductorTheme.PRIMARY_DEFAULT.lstrip("#")
    try:
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except Exception:
        return (0.133, 0.682, 0.820)


def theme_primary_hover_rgb() -> Tuple[float, float, float]:
    h = SemiconductorTheme.PRIMARY_HOVER.lstrip("#")
    try:
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except Exception:
        return (0.235, 0.769, 0.910)


def build_orientation_marker() -> Tuple[Any, Dict[str, Any]]:
    """vtkAssembly: AnnotatedCube + 6 thin face plates for hover glow.

    Returns (assembly, state) where state holds:
      cube, face_highlights, face_text_props, hover_face
    """
    assembly = vtk.vtkAssembly()

    cube = vtk.vtkAnnotatedCubeActor()
    cube.SetXPlusFaceText("+X")
    cube.SetXMinusFaceText("-X")
    cube.SetYPlusFaceText("+Y")
    cube.SetYMinusFaceText("-Y")
    cube.SetZPlusFaceText("+Z")
    cube.SetZMinusFaceText("-Z")
    cube.GetCubeProperty().SetColor(0.93, 0.94, 0.95)
    cube.GetTextEdgesProperty().SetColor(0.15, 0.18, 0.22)
    cube.GetTextEdgesProperty().SetLineWidth(1)
    cube.SetFaceTextScale(0.35)

    face_prop_getters = {
        "+X": cube.GetXPlusFaceProperty,
        "-X": cube.GetXMinusFaceProperty,
        "+Y": cube.GetYPlusFaceProperty,
        "-Y": cube.GetYMinusFaceProperty,
        "+Z": cube.GetZPlusFaceProperty,
        "-Z": cube.GetZMinusFaceProperty,
    }
    face_text_props = {}
    for name, getter in face_prop_getters.items():
        prop = getter()
        prop.SetColor(0.12, 0.14, 0.16)
        prop.SetDiffuse(0.8)
        prop.SetAmbient(0.4)
        face_text_props[name] = prop

    assembly.AddPart(cube)

    t = 0.02
    e = 0.50
    o = 0.51
    face_bounds = {
        "+X": (o - t, o + t, -e, e, -e, e),
        "-X": (-o - t, -o + t, -e, e, -e, e),
        "+Y": (-e, e, o - t, o + t, -e, e),
        "-Y": (-e, e, -o - t, -o + t, -e, e),
        "+Z": (-e, e, -e, e, o - t, o + t),
        "-Z": (-e, e, -e, e, -o - t, -o + t),
    }
    face_highlights = {}
    pr, pg, pb = theme_primary_rgb()
    for name, bounds in face_bounds.items():
        src = vtk.vtkCubeSource()
        src.SetBounds(*bounds)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(src.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(pr, pg, pb)
        prop.SetOpacity(0.0)
        prop.SetAmbient(0.85)
        prop.SetDiffuse(0.5)
        prop.SetSpecular(0.25)
        prop.LightingOn()
        assembly.AddPart(actor)
        face_highlights[name] = actor

    state = {
        "cube": cube,
        "face_highlights": face_highlights,
        "face_text_props": face_text_props,
        "hover_face": None,
    }
    return assembly, state


def set_orientation_face_hover(state: Dict[str, Any], face: Optional[str]) -> bool:
    """Update face glow. Returns True if state changed (needs Render)."""
    if face == state.get("hover_face"):
        return False
    state["hover_face"] = face

    pr, pg, pb = theme_primary_rgb()
    phr, phg, phb = theme_primary_hover_rgb()
    default_text = (0.12, 0.14, 0.16)
    hover_text = (0.04, 0.08, 0.12)

    for name, actor in (state.get("face_highlights") or {}).items():
        prop = actor.GetProperty()
        if name == face:
            prop.SetColor(phr, phg, phb)
            prop.SetOpacity(0.72)
        else:
            prop.SetColor(pr, pg, pb)
            prop.SetOpacity(0.0)

    for name, tprop in (state.get("face_text_props") or {}).items():
        if name == face:
            tprop.SetColor(*hover_text)
            tprop.SetAmbient(1.0)
            tprop.SetDiffuse(0.2)
        else:
            tprop.SetColor(*default_text)
            tprop.SetAmbient(0.4)
            tprop.SetDiffuse(0.8)

    cube = state.get("cube")
    if cube is not None:
        if face:
            cube.GetCubeProperty().SetColor(
                0.75 * 0.93 + 0.25 * pr,
                0.75 * 0.94 + 0.25 * pg,
                0.75 * 0.95 + 0.25 * pb,
            )
        else:
            cube.GetCubeProperty().SetColor(0.93, 0.94, 0.95)
    return True


def ray_hit_unit_cube_face(origin, direction) -> Optional[str]:
    """Nearest front-face hit of ray vs cube [-1,1]^3. Returns '+X'… or None."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    nrm = np.linalg.norm(direction)
    if nrm < 1e-12:
        return None
    direction = direction / nrm

    planes = [
        (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), "+X"),
        (np.array([-1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), "-X"),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), "+Y"),
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, -1.0, 0.0]), "-Y"),
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), "+Z"),
        (np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, -1.0]), "-Z"),
    ]
    best_t = None
    best_face = None
    eps = 1e-6
    for n, p0, name in planes:
        denom = float(np.dot(n, direction))
        if denom >= -1e-12:
            continue
        t = float(np.dot(n, p0 - origin) / denom)
        if t < eps:
            continue
        hit = origin + t * direction
        if abs(hit[0]) <= 1.02 and abs(hit[1]) <= 1.02 and abs(hit[2]) <= 1.02:
            if best_t is None or t < best_t:
                best_t = t
                best_face = name
    return best_face


def is_over_orientation_viewport(
    display_x: float,
    display_y: float,
    win_w: float,
    win_h: float,
    viewport,
) -> bool:
    if not win_w or not win_h or win_w <= 0 or win_h <= 0:
        return False
    nx, ny = display_x / win_w, display_y / win_h
    return viewport[0] <= nx <= viewport[2] and viewport[1] <= ny <= viewport[3]


def pick_orientation_cube_face(
    display_x: float,
    display_y: float,
    win_w: float,
    win_h: float,
    viewport,
    camera,
) -> Optional[str]:
    """Ray-pick orientation marker face from display coords + main 3D camera."""
    if not win_w or not win_h or win_w <= 0 or win_h <= 0 or camera is None:
        return None
    w, h = float(win_w), float(win_h)
    nx = display_x / w
    ny = display_y / h
    if not (viewport[0] <= nx <= viewport[2] and viewport[1] <= ny <= viewport[3]):
        return None

    u = (nx - viewport[0]) / max(1e-9, (viewport[2] - viewport[0]))
    v = (ny - viewport[1]) / max(1e-9, (viewport[3] - viewport[1]))
    sx = 2.0 * u - 1.0
    sy = 2.0 * v - 1.0

    pos = np.array(camera.GetPosition(), dtype=float)
    fp = np.array(camera.GetFocalPoint(), dtype=float)
    vpn = pos - fp
    nrm = np.linalg.norm(vpn)
    if nrm < 1e-12:
        return None
    vpn = vpn / nrm
    vup = np.array(camera.GetViewUp(), dtype=float)
    nrm = np.linalg.norm(vup)
    if nrm < 1e-12:
        return None
    vup = vup / nrm
    vright = np.cross(vup, vpn)
    nrm = np.linalg.norm(vright)
    if nrm < 1e-12:
        return None
    vright = vright / nrm
    vup = np.cross(vpn, vright)

    scale = 1.35
    dist = 6.0
    ray_origin = sx * scale * vright + sy * scale * vup + dist * vpn
    ray_dir = -vpn
    return ray_hit_unit_cube_face(ray_origin, ray_dir)


def apply_camera_face_preset(
    camera,
    face: str,
    *,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 500.0,
) -> bool:
    """Snap camera so cube face points toward the viewer. Returns True if applied."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    r = max(float(radius), 1.0)
    dirs = {
        "+X": ((cx + r, cy, cz), (0, 0, 1)),
        "-X": ((cx - r, cy, cz), (0, 0, 1)),
        "+Y": ((cx, cy + r, cz), (0, 0, 1)),
        "-Y": ((cx, cy - r, cz), (0, 0, 1)),
        "+Z": ((cx, cy, cz + r), (0, 1, 0)),
        "-Z": ((cx, cy, cz - r), (0, -1, 0)),
    }
    if face not in dirs or camera is None:
        return False
    pos, up = dirs[face]
    camera.SetFocalPoint(cx, cy, cz)
    camera.SetPosition(*pos)
    camera.SetViewUp(*up)
    return True
