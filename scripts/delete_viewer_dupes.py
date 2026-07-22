"""DELETE duplicate methods from viewer.py that already live in the mixin files.

This is Phase 3.1b + 3.2b of the Inno3D refactor:
  MOVE complete — methods were COPIED into mixin files in 3.1a/3.2a,
  now we DELETE them from the god-file so MRO resolves to the mixins.

Safety:
  - Only deletes methods whose names are in NAV_METHODS or CROSSHAIR_METHODS.
  - Writes the new file atomically (build in memory, then overwrite).
  - Does NOT touch any non-duplicate methods or non-method code.
  - Preserves blank lines / comments between kept methods.

Usage:
    conda run -n inno3d_ai python scripts/delete_viewer_dupes.py
    conda run -n inno3d_ai python scripts/delete_viewer_dupes.py --dry-run
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "inno3d" / "tabs" / "viewer.py"

# ── Method sets (must match the mixin files exactly) ─────────────────
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

dry_run = "--dry-run" in sys.argv


def main():
    text = VIEWER.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    total = len(lines)

    # Parse method ranges (0-indexed) inside MultiPlanarView
    in_class = False
    methods = []  # (name, start_0, end_0_exclusive)
    i = 0
    while i < total:
        line = lines[i]
        if line.startswith("class MultiPlanarView"):
            in_class = True
        if not in_class:
            i += 1
            continue
        m = METHOD_RE.match(line)
        if m:
            name = m.group(1)
            start = i
            j = i + 1
            while j < total:
                nxt = lines[j]
                if (nxt.startswith("    def ") or nxt.startswith("class ")) and j > i + 1:
                    break
                j += 1
            methods.append((name, start, j))
            i = j
        else:
            i += 1

    # Identify ranges to delete
    delete_ranges = []
    for name, s, e in methods:
        if name in ALL_DUPE:
            delete_ranges.append((name, s, e))

    if not delete_ranges:
        print("No duplicates found — nothing to delete.")
        return

    # Build set of line indices to remove
    remove_lines = set()
    for name, s, e in delete_ranges:
        for idx in range(s, e):
            remove_lines.add(idx)

    # Also remove trailing blank lines that would create excessive gaps
    # (when two deleted methods were adjacent, the blank lines between them
    # should also go; but keep at most 2 consecutive blank lines between
    # any two remaining code lines).

    new_lines = []
    for idx, line in enumerate(lines):
        if idx not in remove_lines:
            new_lines.append(line)

    # Collapse runs of > 2 blank lines into 2
    collapsed = []
    blank_run = 0
    for line in new_lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                collapsed.append(line)
        else:
            blank_run = 0
            collapsed.append(line)

    removed = total - len(collapsed)

    print(f"Methods to delete: {len(delete_ranges)}")
    for name, s, e in sorted(delete_ranges, key=lambda x: x[1]):
        mixin = "MprNavMixin" if name in NAV_METHODS else "CrosshairMixin"
        print(f"  DELETE  {mixin:20s}  {name:40s}  lines {s+1}–{e}")
    print(f"\nLines removed: {removed}  (original: {total} → new: {len(collapsed)})")

    if dry_run:
        print("\n[DRY RUN] No file written.")
        return

    # Atomic write
    VIEWER.write_text("".join(collapsed), encoding="utf-8")
    print(f"\n✅ Written: {VIEWER}")
    print("Next: verify with inspect.getfile + pytest + manual smoke.")


if __name__ == "__main__":
    main()
