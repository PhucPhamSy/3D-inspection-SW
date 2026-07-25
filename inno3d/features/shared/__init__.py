# inno3d.features.shared — host-agnostic mapping / 3D overlay helpers
# Used by both 3D Viewer and 3D Teaching to avoid dual implementations.
from inno3d.features.shared.b2b_gap_3d import (
    B2BGapOverlay,
    format_gap_label,
    layer_z_offset,
    world_spacing,
)
from inno3d.features.shared.crosshair_3d import (
    AXIS_COLORS,
    Crosshair3DOverlay,
    crosshair_axis_endpoints,
)
from inno3d.features.shared.mes_highlight_3d import MESHighlightOverlay
from inno3d.features.shared.mes_mapping import (
    MesPickDebounce,
    centroid_nav_xyz,
    highlights_from_stats_indices,
    label_at_stat_centroid,
    label_at_volume_xyz,
    label_set_from_highlights,
    mes_bbox_for_labels,
    resolve_stat_label,
    stats_index_for_label,
)
from inno3d.features.shared.mes_mpr_highlight import (
    blend_mes_selection_highlight,
    build_mes_selection_highlight_rgb,
    labeled_slice_for_orientation,
)

__all__ = [
    "AXIS_COLORS",
    "B2BGapOverlay",
    "Crosshair3DOverlay",
    "MESHighlightOverlay",
    "MesPickDebounce",
    "blend_mes_selection_highlight",
    "build_mes_selection_highlight_rgb",
    "centroid_nav_xyz",
    "crosshair_axis_endpoints",
    "format_gap_label",
    "highlights_from_stats_indices",
    "label_at_stat_centroid",
    "label_at_volume_xyz",
    "label_set_from_highlights",
    "labeled_slice_for_orientation",
    "layer_z_offset",
    "mes_bbox_for_labels",
    "resolve_stat_label",
    "stats_index_for_label",
    "world_spacing",
]
