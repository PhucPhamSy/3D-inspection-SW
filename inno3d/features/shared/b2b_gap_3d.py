# inno3d/features/shared/b2b_gap_3d.py
# -----------------------------------------------------------------------
# Host-agnostic B2B gap → 3D Volume visualization (Viewer SoT).
#
# Both MultiPlanarView (stats_panel) and SegmentationTab (seg_ui) call this
# module so gap geometry / colors / caption stay in lock-step.
# Host supplies renderer, spacing, labeled mask, and optional hooks
# (context dim, camera frame, orbit pivot).
# -----------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import vtk
from vtk.util import numpy_support

# Colors (Viewer SoT — amber DST contrasts yellow C1 shells)
COL_SRC = (0.05, 0.95, 1.0)   # cyan
COL_DST = (1.0, 0.55, 0.08)   # amber
RGB_SRC = (10, 240, 255)
RGB_DST = (255, 150, 20)


def world_spacing(
    custom_spacing: Optional[Sequence[float]] = None,
    spacing: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float]:
    """Return (sx, sy, sz) for VTK world = index * spacing, origin 0."""
    sp = custom_spacing if custom_spacing is not None else spacing
    if sp is None:
        return (1.0, 1.0, 1.0)
    try:
        return (float(sp[0]), float(sp[1]), float(sp[2]))
    except Exception:
        return (1.0, 1.0, 1.0)


def layer_z_offset(
    row_dict: Optional[Dict[str, Any]],
    *,
    layer_bands: Optional[Sequence[Dict[str, Any]]] = None,
    fallback: int = 0,
) -> int:
    """Local B2B layer Z → full-volume Z.

    Priority:
      1) row keys z_start / Z_start / layer_z_start
      2) match Layer name against ``layer_bands`` (Online bands)
      3) ``fallback`` (Teaching folder-detected offset)
    """
    if not row_dict:
        return int(fallback or 0)
    for key in ("z_start", "Z_start", "layer_z_start"):
        if key in row_dict and row_dict[key] not in (None, ""):
            try:
                return int(row_dict[key])
            except (TypeError, ValueError):
                pass
    ln = str(row_dict.get("Layer", "") or "")
    if layer_bands and ln:
        key = ln.replace(" ", "_").lower()
        for L in layer_bands:
            name = str(L.get("name", ""))
            if name == ln or name.replace(" ", "_").lower() == key:
                try:
                    return int(L.get("z_start", 0) or 0)
                except (TypeError, ValueError):
                    return int(fallback or 0)
    return int(fallback or 0)


def format_gap_label(
    layer,
    eucl,
    src_rc,
    dst_rc,
    direction,
    src_zyx=None,
    dst_zyx=None,
) -> str:
    """Compact 3-line callout — gap value + explicit SRC / DST identity."""
    layer_s = (str(layer).strip() if layer is not None else "") or ""
    try:
        eucl_s = f"{float(eucl):.2f} µm"
    except (TypeError, ValueError):
        eucl_s = f"{eucl} µm" if eucl not in (None, "") else "— µm"
    dir_s = str(direction or "").strip()
    line1 = eucl_s if not dir_s else f"{eucl_s}  ·  {dir_s}"
    if layer_s:
        line1 = f"{layer_s}  ·  {line1}"
    line2 = f"SRC  {src_rc}"
    line3 = f"DST  {dst_rc}"
    return f"{line1}\n{line2}\n{line3}"


def world_to_normalized_viewport(ren, wx, wy, wz):
    """Project world point → normalized viewport [0..1]² (origin bottom-left)."""
    if ren is None:
        return None
    try:
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        coord.SetValue(float(wx), float(wy), float(wz))
        dx, dy = coord.GetComputedDisplayValue(ren)
        origin = ren.GetOrigin()
        size = ren.GetSize()
        w = float(size[0]) if size and size[0] else 0.0
        h = float(size[1]) if size and size[1] else 0.0
        if w <= 1.0 or h <= 1.0:
            return None
        nx = (float(dx) - float(origin[0])) / w
        ny = (float(dy) - float(origin[1])) / h
        return (nx, ny)
    except Exception:
        return None


def caption_viewport_pos(ren, mid_x, mid_y, mid_z, box_w=0.30, box_h=0.08):
    """Place callout box near the gap in screen space (short leader)."""
    proj = world_to_normalized_viewport(ren, mid_x, mid_y, mid_z)
    if proj is None:
        return (0.55, 0.50)
    nx, ny = proj
    if nx < -0.15 or nx > 1.15 or ny < -0.15 or ny > 1.15:
        return (0.55, 0.50)
    cap_x = nx + 0.06
    cap_y = ny + 0.04
    if cap_x + box_w > 0.98:
        cap_x = nx - box_w - 0.04
    if cap_y + box_h > 0.96:
        cap_y = ny - box_h - 0.04
    cap_x = max(0.02, min(cap_x, 0.98 - box_w))
    cap_y = max(0.02, min(cap_y, 0.96 - box_h))
    return (cap_x, cap_y)


class B2BGapOverlay:
    """Stateful B2B gap actor stack on a VTK 3D renderer.

    Hosts keep one instance (e.g. ``self._b2b_gap_overlay``) and call
    :meth:`clear` / :meth:`draw`. Does not know about Qt tables.
    """

    def __init__(self):
        self.actors: List[Any] = []
        self.active_stats_idx: Optional[int] = None

    # ── lifecycle ─────────────────────────────────────────────────────

    def clear(
        self,
        ren=None,
        widget=None,
        *,
        render: bool = True,
        on_restore_context: Optional[Callable[[], None]] = None,
    ) -> None:
        """Remove all gap actors from the renderer."""
        if ren is not None:
            for actor in self.actors:
                try:
                    ren.RemoveActor(actor)
                except Exception:
                    pass
                try:
                    ren.RemoveActor2D(actor)
                except Exception:
                    pass
        self.actors = []
        self.active_stats_idx = None
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

    def _add(self, ren, actor, is_2d: bool = False) -> None:
        if is_2d:
            ren.AddActor2D(actor)
        else:
            ren.AddActor(actor)
        self.actors.append(actor)

    # ── geometry helpers ──────────────────────────────────────────────

    def add_surface_voxel_marker(
        self, ren, z, y, x, sx, sy, sz, color, label_tag: str = ""
    ) -> None:
        """1-voxel cell (fill + wire) + bead + optional SRC/DST tag."""
        x0, x1 = (float(x) - 0.5) * sx, (float(x) + 0.5) * sx
        y0, y1 = (float(y) - 0.5) * sy, (float(y) + 0.5) * sy
        z0, z1 = (float(z) - 0.5) * sz, (float(z) + 0.5) * sz
        cx, cy, cz = float(x) * sx, float(y) * sy, float(z) * sz
        min_sp = max(1e-6, min(sx, sy, sz))

        cube = vtk.vtkCubeSource()
        cube.SetBounds(x0, x1, y0, y1, z0, z1)
        cube.Update()

        fill_m = vtk.vtkPolyDataMapper()
        fill_m.SetInputConnection(cube.GetOutputPort())
        fill_a = vtk.vtkActor()
        fill_a.SetMapper(fill_m)
        fill_a.GetProperty().SetColor(*color)
        fill_a.GetProperty().SetOpacity(0.55)
        fill_a.GetProperty().SetLighting(False)
        fill_a.GetProperty().SetAmbient(1.0)
        fill_a.GetProperty().SetDiffuse(0.0)
        self._add(ren, fill_a)

        wire_m = vtk.vtkPolyDataMapper()
        wire_m.SetInputConnection(cube.GetOutputPort())
        wire_a = vtk.vtkActor()
        wire_a.SetMapper(wire_m)
        wire_a.GetProperty().SetRepresentationToWireframe()
        wire_a.GetProperty().SetColor(1.0, 1.0, 1.0)
        wire_a.GetProperty().SetLineWidth(2.0)
        wire_a.GetProperty().SetOpacity(1.0)
        wire_a.GetProperty().SetLighting(False)
        wire_a.GetProperty().SetAmbient(1.0)
        self._add(ren, wire_a)

        bead_r = max(0.35, min_sp * 0.35)
        sph = vtk.vtkSphereSource()
        sph.SetCenter(cx, cy, cz)
        sph.SetRadius(bead_r)
        sph.SetPhiResolution(14)
        sph.SetThetaResolution(14)
        sm = vtk.vtkPolyDataMapper()
        sm.SetInputConnection(sph.GetOutputPort())
        sa = vtk.vtkActor()
        sa.SetMapper(sm)
        sa.GetProperty().SetColor(*color)
        sa.GetProperty().SetOpacity(1.0)
        sa.GetProperty().SetLighting(False)
        sa.GetProperty().SetAmbient(1.0)
        self._add(ren, sa)

        if label_tag:
            try:
                tag = vtk.vtkBillboardTextActor3D()
                tag.SetInput(str(label_tag))
                tag.SetPosition(cx, cy, cz + max(sz, min_sp) * 2.2)
                try:
                    tag.SetScale(max(min_sp * 1.4, 1.1))
                except Exception:
                    pass
                tp = tag.GetTextProperty()
                tp.SetFontSize(18)
                tp.SetColor(*color)
                tp.BoldOn()
                tp.ShadowOn()
                tp.SetBackgroundColor(0.02, 0.04, 0.08)
                tp.SetBackgroundOpacity(0.88)
                tp.SetJustificationToCentered()
                tp.SetVerticalJustificationToBottom()
                self._add(ren, tag)
            except Exception:
                pass

    def add_local_object_surface(
        self,
        ren,
        z,
        y,
        x,
        sx,
        sy,
        sz,
        color,
        *,
        labeled=None,
        class1_binary=None,
        pad: int = 14,
    ) -> None:
        """Local isosurface of the object owning this surface voxel."""
        if labeled is None and class1_binary is None:
            return
        try:
            if labeled is not None:
                Z, Y, X = labeled.shape
            else:
                Z, Y, X = class1_binary.shape
            z, y, x = int(z), int(y), int(x)
            if not (0 <= z < Z and 0 <= y < Y and 0 <= x < X):
                return

            lab = int(labeled[z, y, x]) if labeled is not None else 0
            z0, z1 = max(0, z - pad), min(Z, z + pad + 1)
            y0, y1 = max(0, y - pad), min(Y, y + pad + 1)
            x0, x1 = max(0, x - pad), min(X, x + pad + 1)

            if labeled is not None and lab > 0:
                crop = labeled[z0:z1, y0:y1, x0:x1]
                mask = crop == lab
            else:
                src = class1_binary if class1_binary is not None else labeled
                crop = src[z0:z1, y0:y1, x0:x1]
                mask = crop > 0
            if not np.any(mask):
                return

            mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))
            vtk_img = vtk.vtkImageData()
            dz, dy, dx = mask_u8.shape
            vtk_img.SetDimensions(dx, dy, dz)
            vtk_img.SetSpacing(float(sx), float(sy), float(sz))
            vtk_img.SetOrigin(float(x0) * sx, float(y0) * sy, float(z0) * sz)
            flat = np.ascontiguousarray(
                np.transpose(mask_u8, (2, 1, 0)).ravel(order="F")
            )
            vtk_arr = numpy_support.numpy_to_vtk(
                flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
            )
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
            contour.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(contour.GetOutputPort())
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(*color)
            prop.SetOpacity(0.72)
            prop.SetAmbient(0.55)
            prop.SetDiffuse(0.65)
            prop.SetSpecular(0.35)
            prop.EdgeVisibilityOn()
            prop.SetEdgeColor(1.0, 1.0, 1.0)
            prop.SetLineWidth(1.5)
            self._add(ren, actor)
        except Exception as e:
            print(f"[B2B] local surface patch skipped: {e}")

    def frame_gap_camera(
        self,
        ren,
        mid_x,
        mid_y,
        mid_z,
        gap_len,
        min_sp,
        *,
        volume_world_radius: Optional[float] = None,
        sync_orbit_pivot: Optional[Callable[[], None]] = None,
    ) -> None:
        """Soft-frame camera on the gap (keep view direction, re-center + zoom in)."""
        if ren is None:
            return
        cam = ren.GetActiveCamera()
        if cam is None:
            return
        try:
            pos = list(cam.GetPosition())
            fp = list(cam.GetFocalPoint())
            dx, dy, dz = pos[0] - fp[0], pos[1] - fp[1], pos[2] - fp[2]
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist < 1e-9:
                dx, dy, dz, dist = 0.0, 0.0, 1.0, 100.0
            inv = 1.0 / dist
            dx, dy, dz = dx * inv, dy * inv, dz * inv

            R = float(volume_world_radius) if volume_world_radius is not None else 100.0
            target = max(gap_len * 12.0, min_sp * 40.0, 25.0)
            target = min(target, max(R * 0.55, target))
            new_dist = min(dist, target) if dist > target * 1.15 else dist

            cam.SetFocalPoint(float(mid_x), float(mid_y), float(mid_z))
            cam.SetPosition(
                float(mid_x) + dx * new_dist,
                float(mid_y) + dy * new_dist,
                float(mid_z) + dz * new_dist,
            )
            ren.ResetCameraClippingRange()
            if sync_orbit_pivot is not None:
                try:
                    sync_orbit_pivot()
                except Exception:
                    pass
        except Exception as e:
            print(f"[B2B] frame camera skipped: {e}")

    # ── main draw ─────────────────────────────────────────────────────

    def draw(
        self,
        ren,
        widget,
        src_voxel: Sequence[float],
        dst_voxel: Sequence[float],
        spacing: Tuple[float, float, float],
        label_text: str = "",
        *,
        labeled=None,
        class1_binary=None,
        on_dim_context: Optional[Callable[[], None]] = None,
        volume_world_radius: Optional[float] = None,
        sync_orbit_pivot: Optional[Callable[[], None]] = None,
        frame_camera: bool = True,
        stats_idx: Optional[int] = None,
    ) -> bool:
        """Draw Src→Dst gap on 3D Volume (Viewer SoT design).

        Args:
            src_voxel / dst_voxel: (z, y, x) in **global volume** voxel indices
            spacing: (sx, sy, sz)
            labeled: labeled_class1 volume (optional, for local surfaces)
            class1_binary: binary bump mask fallback (class1_data / bump_segmentation)
            on_dim_context: host hook to dim global C1/C2 shells
            frame_camera: soft-frame camera on the gap (Viewer default True)

        Returns:
            True if actors were added.
        """
        if ren is None:
            print("[B2B] 3D gap skipped: no renderer")
            return False

        # Clear previous gap actors without intermediate undim
        self.clear(ren, widget, render=False, on_restore_context=None)

        sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        src_x = float(src_voxel[2]) * sx
        src_y = float(src_voxel[1]) * sy
        src_z = float(src_voxel[0]) * sz
        dst_x = float(dst_voxel[2]) * sx
        dst_y = float(dst_voxel[1]) * sy
        dst_z = float(dst_voxel[0]) * sz
        mid_x = 0.5 * (src_x + dst_x)
        mid_y = 0.5 * (src_y + dst_y)
        mid_z = 0.5 * (src_z + dst_z)

        dx = dst_x - src_x
        dy = dst_y - src_y
        dz = dst_z - src_z
        gap_len = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if gap_len < 1e-9:
            gap_len = 1e-9
            dx, dy, dz = gap_len, 0.0, 0.0
        ux, uy, uz = dx / gap_len, dy / gap_len, dz / gap_len

        min_sp = max(1e-6, min(sx, sy, sz))
        tube_r = max(0.55, min_sp * 1.35)
        arrow_h = max(min_sp * 3.5, tube_r * 4.5)
        arrow_r = max(tube_r * 2.0, min_sp * 1.8)

        src_z_i, src_y_i, src_x_i = (
            int(src_voxel[0]), int(src_voxel[1]), int(src_voxel[2])
        )
        dst_z_i, dst_y_i, dst_x_i = (
            int(dst_voxel[0]), int(dst_voxel[1]), int(dst_voxel[2])
        )

        if on_dim_context is not None:
            try:
                on_dim_context()
            except Exception:
                pass

        # 0) Local object surfaces
        self.add_local_object_surface(
            ren, src_z_i, src_y_i, src_x_i, sx, sy, sz, COL_SRC,
            labeled=labeled, class1_binary=class1_binary, pad=14,
        )
        self.add_local_object_surface(
            ren, dst_z_i, dst_y_i, dst_x_i, sx, sy, sz, COL_DST,
            labeled=labeled, class1_binary=class1_binary, pad=14,
        )

        # 1) Gradient tube cyan → amber
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(src_x, src_y, src_z)
        pts.InsertNextPoint(dst_x, dst_y, dst_z)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        colors.InsertNextTuple3(*RGB_SRC)
        colors.InsertNextTuple3(*RGB_DST)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetLines(lines)
        poly.GetPointData().SetScalars(colors)

        tube = vtk.vtkTubeFilter()
        tube.SetInputData(poly)
        tube.SetRadius(tube_r)
        tube.SetNumberOfSides(20)
        tube.CappingOn()
        tube.SetVaryRadiusToVaryRadiusOff()
        tube.Update()

        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputConnection(tube.GetOutputPort())
        line_mapper.SetScalarModeToUsePointData()
        line_mapper.ScalarVisibilityOn()
        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetOpacity(1.0)
        line_actor.GetProperty().SetLighting(False)
        line_actor.GetProperty().SetAmbient(1.0)
        line_actor.GetProperty().SetDiffuse(0.0)
        self._add(ren, line_actor)

        # 2) Surface voxel markers
        self.add_surface_voxel_marker(
            ren, src_z_i, src_y_i, src_x_i, sx, sy, sz, COL_SRC, label_tag="SRC",
        )
        self.add_surface_voxel_marker(
            ren, dst_z_i, dst_y_i, dst_x_i, sx, sy, sz, COL_DST, label_tag="DST",
        )

        # 3) Arrow at DST
        try:
            cone = vtk.vtkConeSource()
            cone.SetRadius(arrow_r)
            cone.SetHeight(arrow_h)
            cone.SetResolution(20)
            cone.SetDirection(ux, uy, uz)
            back = min(arrow_h * 0.55, gap_len * 0.35)
            cone.SetCenter(
                dst_x - ux * back,
                dst_y - uy * back,
                dst_z - uz * back,
            )
            cone.Update()
            cone_mapper = vtk.vtkPolyDataMapper()
            cone_mapper.SetInputConnection(cone.GetOutputPort())
            cone_actor = vtk.vtkActor()
            cone_actor.SetMapper(cone_mapper)
            cone_actor.GetProperty().SetColor(1.0, 0.2, 0.15)
            cone_actor.GetProperty().SetOpacity(1.0)
            cone_actor.GetProperty().SetLighting(False)
            cone_actor.GetProperty().SetAmbient(1.0)
            cone_actor.GetProperty().SetDiffuse(0.0)
            self._add(ren, cone_actor)
        except Exception as _arr_e:
            print(f"[B2B] DST arrow skipped: {_arr_e}")

        # 4) Screen-space caption
        if label_text:
            try:
                box_w, box_h = 0.36, 0.14
                cap_x, cap_y = caption_viewport_pos(
                    ren, mid_x, mid_y, mid_z, box_w=box_w, box_h=box_h
                )
                caption = vtk.vtkCaptionActor2D()
                caption.SetCaption(str(label_text))
                caption.SetAttachmentPoint(mid_x, mid_y, mid_z)
                caption.BorderOn()
                caption.LeaderOn()
                try:
                    caption.ThreeDimensionalLeaderOff()
                except Exception:
                    pass
                try:
                    caption.SetPadding(8)
                except Exception:
                    pass
                caption.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                caption.SetPosition(float(cap_x), float(cap_y))
                caption.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
                caption.SetWidth(box_w)
                caption.SetHeight(box_h)
                cprop = caption.GetCaptionTextProperty()
                cprop.SetFontSize(15)
                cprop.SetBold(1)
                cprop.SetColor(1.0, 0.98, 0.92)
                cprop.SetBackgroundColor(0.05, 0.08, 0.14)
                cprop.SetBackgroundOpacity(0.92)
                cprop.ShadowOn()
                cprop.SetJustificationToLeft()
                cprop.SetVerticalJustificationToCentered()
                caption.GetProperty().SetColor(0.15, 0.85, 1.0)
                caption.GetProperty().SetLineWidth(2.0)
                try:
                    caption.GetAttachmentPointCoordinate().SetCoordinateSystemToWorld()
                except Exception:
                    pass
                self._add(ren, caption, is_2d=True)
            except Exception as _cap_e:
                print(f"[B2B] screen caption failed: {_cap_e}")
                # Fallback billboard (Teaching path used this)
                try:
                    off = max(tube_r * 6.0, min_sp * 12.0)
                    text_actor = vtk.vtkBillboardTextActor3D()
                    text_actor.SetInput(str(label_text))
                    text_actor.SetPosition(mid_x + off, mid_y, mid_z + off * 0.35)
                    tp = text_actor.GetTextProperty()
                    tp.SetFontSize(14)
                    tp.SetColor(1.0, 1.0, 1.0)
                    tp.BoldOn()
                    tp.SetBackgroundColor(0.05, 0.08, 0.12)
                    tp.SetBackgroundOpacity(0.85)
                    self._add(ren, text_actor)
                except Exception as _fb_e:
                    print(f"[B2B] label fallback failed: {_fb_e}")

        if frame_camera:
            self.frame_gap_camera(
                ren, mid_x, mid_y, mid_z, gap_len, min_sp,
                volume_world_radius=volume_world_radius,
                sync_orbit_pivot=sync_orbit_pivot,
            )

        if widget is not None:
            try:
                ren.ResetCameraClippingRange()
                widget.GetRenderWindow().Render()
            except Exception:
                pass

        if stats_idx is not None:
            try:
                self.active_stats_idx = int(stats_idx)
            except (TypeError, ValueError):
                self.active_stats_idx = None

        print(
            f"[B2B] 3D gap drawn: {label_text}  "
            f"src={tuple(src_voxel)} dst={tuple(dst_voxel)} "
            f"spacing=({sx:.3f},{sy:.3f},{sz:.3f}) len={gap_len:.3f}"
        )
        return True
