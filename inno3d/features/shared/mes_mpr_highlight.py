# inno3d/features/shared/mes_mpr_highlight.py
# -----------------------------------------------------------------------
# Shared MPR selection highlight (Viewer SoT): cyan fill + white outline.
# Used by Viewer mpr_render and Teaching compose overlay.
# -----------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


def selected_c1_labels(
    selected_highlight_objects: Optional[Sequence[Sequence[Any]]],
) -> list:
    """Positive class-1 labels from ``selected_highlight_objects``."""
    out = []
    for item in selected_highlight_objects or []:
        try:
            cls_num, lab = item[0], int(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if int(cls_num) == 1 and lab > 0:
            out.append(lab)
    return out


def labeled_slice_for_orientation(
    labeled,
    orientation: str,
    slice_idx: int,
    *,
    reverse_z: bool = False,
) -> Optional[np.ndarray]:
    """Extract 2D labeled slice matching MPR pane orientation.

    Returns array shape (H, W) in the same layout as the rendered CT slice
    (axial / coronal flipud / sagittal transpose).
    """
    if labeled is None:
        return None
    try:
        slice_idx = int(slice_idx)
    except (TypeError, ValueError):
        return None
    Z, Y, X = labeled.shape
    if orientation == "axial":
        actual = (Z - 1 - slice_idx) if reverse_z else slice_idx
        if not (0 <= actual < Z):
            return None
        return labeled[actual, :, :]
    if orientation == "coronal":
        if not (0 <= slice_idx < Y):
            return None
        return np.flipud(labeled[:, slice_idx, :])
    # sagittal
    if not (0 <= slice_idx < X):
        return None
    return np.transpose(labeled[:, :, slice_idx])


def build_mes_selection_highlight_rgb(
    labeled_slice: Optional[np.ndarray],
    selected_highlight_objects: Optional[Sequence[Sequence[Any]]],
    highlight_color: Optional[Sequence[float]] = None,
    *,
    outline_iterations: int = 3,
) -> Optional[np.ndarray]:
    """Build Viewer-style selection highlight RGB (H,W,3) uint8.

    · Interior: semi-saturated ``highlight_color`` (default cyan)
    · Outline: bright white ring (binary_dilation − mask)

    Returns None if nothing to draw.
    """
    if labeled_slice is None:
        return None
    labels = selected_c1_labels(selected_highlight_objects)
    if not labels:
        return None

    labeled_slice = np.asarray(labeled_slice)
    if labeled_slice.ndim != 2:
        return None
    h, w = labeled_slice.shape

    hl = list(highlight_color or [0.0, 1.0, 1.0])
    if sum(float(c) for c in hl[:3]) < 1.2:
        hl = [0.15, 0.95, 1.0]
    hl_r = int(max(0, min(255, round(float(hl[0]) * 255))))
    hl_g = int(max(0, min(255, round(float(hl[1]) * 255))))
    hl_b = int(max(0, min(255, round(float(hl[2]) * 255))))

    # Pre-fill channels; only write where selected (viewer max-composites multi)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    has = False
    iters = max(1, int(outline_iterations))

    for lab in labels:
        obj_mask = labeled_slice == int(lab)
        if not np.any(obj_mask):
            continue
        has = True
        dilated = ndimage.binary_dilation(obj_mask, iterations=iters)
        outline = dilated & ~obj_mask

        # Interior cyan (overwrite / max so multi-select stacks cleanly)
        out[obj_mask, 0] = np.maximum(out[obj_mask, 0], hl_r)
        out[obj_mask, 1] = np.maximum(out[obj_mask, 1], hl_g)
        out[obj_mask, 2] = np.maximum(out[obj_mask, 2], hl_b)

        # Bright white border
        out[outline, 0] = 255
        out[outline, 1] = 255
        out[outline, 2] = 255

    return out if has else None


def blend_mes_selection_highlight(
    rgb: np.ndarray,
    highlight_rgb: Optional[np.ndarray],
    *,
    opacity: float = 0.7,
) -> np.ndarray:
    """Alpha-blend selection highlight onto an RGB image (Teaching single-actor).

    Matches Viewer ``vtkImageActor.SetOpacity(0.7)`` over the base slice.
    """
    if highlight_rgb is None or rgb is None:
        return rgb
    base = np.asarray(rgb)
    hl = np.asarray(highlight_rgb)
    if base.shape[:2] != hl.shape[:2] or hl.ndim != 3 or hl.shape[2] < 3:
        return base
    a = float(np.clip(opacity, 0.0, 1.0))
    if a <= 0:
        return base
    mask = np.any(hl > 0, axis=2)
    if not np.any(mask):
        return base
    out = base.astype(np.float32, copy=True)
    hl_f = hl[..., :3].astype(np.float32)
    out[mask] = (1.0 - a) * out[mask] + a * hl_f[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def dim_non_selected_foreground(
    rgb: np.ndarray,
    gray: np.ndarray,
    labeled_slice: Optional[np.ndarray],
    selected_highlight_objects: Optional[Sequence[Sequence[Any]]],
    foreground_mask: Optional[np.ndarray] = None,
    *,
    dim_to_gray: float = 0.75,
) -> np.ndarray:
    """Optional spotlight: pull non-selected mask pixels toward CT gray.

    Kept for Teaching as a *secondary* cue under cyan outline; Viewer MPR
    does not use this (cyan overlay alone).
    """
    if labeled_slice is None or rgb is None:
        return rgb
    labels = selected_c1_labels(selected_highlight_objects)
    if not labels:
        return rgb
    labeled_slice = np.asarray(labeled_slice)
    h, w = labeled_slice.shape[:2]
    if rgb.shape[:2] != (h, w):
        return rgb
    selected_mask = np.isin(labeled_slice, labels)
    if foreground_mask is None:
        fg = labeled_slice > 0
    else:
        fg = np.asarray(foreground_mask, dtype=bool)
        if fg.shape[:2] != (h, w):
            fg = labeled_slice > 0
    dim_mask = fg & ~selected_mask
    if not np.any(dim_mask):
        return rgb
    g = float(np.clip(dim_to_gray, 0.0, 1.0))
    keep = 1.0 - g
    out = rgb.astype(np.float32, copy=True)
    gray_f = np.asarray(gray, dtype=np.float32)
    if gray_f.ndim == 2:
        for ch in range(3):
            out[dim_mask, ch] = keep * out[dim_mask, ch] + g * gray_f[dim_mask]
    return np.clip(out, 0, 255).astype(np.uint8 if rgb.dtype == np.uint8 else np.float32)
