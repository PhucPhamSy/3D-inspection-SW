"""Scan viewer.py for methods that ALSO exist in the mixin files.
Prints their exact line ranges so the agent can delete them."""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "inno3d" / "tabs" / "viewer.py"

# Methods in each mixin (from extract_viewer_mixins.py)
NAV_METHODS = {
    "_restore_interactor_style", "_fiji_zoom_at_cursor", "_scroll_change_slice",
    "_ensure_crosshair_visible", "_toggle_link_mpr_nav", "_active_mpr_orientation",
    "_shortcut_fit_active_pane", "_shortcut_one_to_one_active_pane",
    "_shortcut_reset_active_pane", "_mpr_image_size", "_mpr_default_parallel_scale",
    "_mpr_zoom_limits", "_set_mpr_cursor", "_mpr_pan_begin", "_mpr_pan_move",
    "_mpr_pan_by_pixels", "_mpr_pan_end", "_set_mpr_one_to_one",
    "_zoom_parallel_about_point",
}
CROSSHAIR_METHODS = {
    "reset_crosshair", "_crosshair_hit_radii", "_pick_world", "_pick_voxel",
    "_flush_crosshair_sync", "_crosshair_drag_update", "_crosshair_drag_end",
    "_cursor_for_crosshair_hit", "_crosshair_display_geometry", "toggle_crosshair",
    "updatePoint", "_sync_slice_sliders_from_crosshair", "update_2d_crosshair",
    "choose_crosshair_color",
}
ALL_DUPE = NAV_METHODS | CROSSHAIR_METHODS

METHOD_RE = re.compile(r"^    def (\w+)\(")

lines = VIEWER.read_text(encoding="utf-8").splitlines(keepends=True)
in_class = False
dupes = []
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("class MultiPlanarView"):
        in_class = True
    if not in_class:
        i += 1
        continue
    m = METHOD_RE.match(line)
    if m:
        name = m.group(1)
        start = i  # 0-indexed
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if (nxt.startswith("    def ") or nxt.startswith("class ")) and j > i + 1:
                break
            j += 1
        if name in ALL_DUPE:
            mixin = "MprNavMixin" if name in NAV_METHODS else "CrosshairMixin"
            dupes.append((name, start + 1, j, mixin))  # 1-indexed start
        i = j
    else:
        i += 1

print(f"Found {len(dupes)} duplicate methods in viewer.py:")
total_lines = 0
for name, start, end, mixin in sorted(dupes, key=lambda x: x[1]):
    n = end - start + 1
    total_lines += n
    print(f"  {mixin:20s}  {name:40s}  lines {start:5d}–{end:5d}  ({n} lines)")
print(f"\nTotal lines to remove: {total_lines}")
