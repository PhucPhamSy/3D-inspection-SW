"""Phase 3.2c — MOVE crosshair-adjacent methods from viewer.py to crosshair.py.

Methods to move:
  _get_crosshair_hit    → CrosshairMixin
  _boost_color          → CrosshairMixin  (staticmethod)
  handle_crosshair_click → CrosshairMixin
  update_3d_crosshair   → CrosshairMixin

Also cleans up:
  - preview_only parameter: always False, dead code → remove from signatures + call sites

Usage:
  conda run -n inno3d_ai python scripts/move_crosshair_cohesion.py --dry-run
  conda run -n inno3d_ai python scripts/move_crosshair_cohesion.py
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "inno3d" / "tabs" / "viewer.py"
CROSSHAIR = ROOT / "inno3d" / "features" / "viewer" / "crosshair.py"

METHODS_TO_MOVE = {
    "_get_crosshair_hit",
    "_boost_color",
    "handle_crosshair_click",
    "update_3d_crosshair",
}

METHOD_RE = re.compile(r"^    def (\w+)\(")
STATIC_RE = re.compile(r"^    @staticmethod\s*$")

dry_run = "--dry-run" in sys.argv


def parse_methods(lines):
    """Parse all method ranges inside MultiPlanarView (0-indexed)."""
    in_class = False
    methods = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("class MultiPlanarView"):
            in_class = True
        if not in_class:
            i += 1
            continue
        # Check for @staticmethod decorator preceding def
        if STATIC_RE.match(line):
            # Next non-empty line should be def
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            m = METHOD_RE.match(lines[j]) if j < len(lines) else None
            if m:
                name = m.group(1)
                start = i  # include the decorator
                k = j + 1
                while k < len(lines):
                    nxt = lines[k]
                    if (nxt.startswith("    def ") or nxt.startswith("    @") or nxt.startswith("class ")) and k > j + 1:
                        break
                    k += 1
                methods.append((name, start, k))
                i = k
                continue
        m = METHOD_RE.match(line)
        if m:
            name = m.group(1)
            start = i
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if (nxt.startswith("    def ") or nxt.startswith("    @") or nxt.startswith("class ")) and j > i + 1:
                    break
                j += 1
            methods.append((name, start, j))
            i = j
        else:
            i += 1
    return methods


def main():
    viewer_text = VIEWER.read_text(encoding="utf-8")
    viewer_lines = viewer_text.splitlines(keepends=True)
    
    crosshair_text = CROSSHAIR.read_text(encoding="utf-8")
    
    methods = parse_methods(viewer_lines)
    name_to_range = {n: (s, e) for n, s, e in methods}
    
    # Identify methods to move
    to_move = []
    for name in sorted(METHODS_TO_MOVE):
        if name in name_to_range:
            s, e = name_to_range[name]
            to_move.append((name, s, e))
            print(f"  MOVE  {name:40s}  lines {s+1}–{e}  ({e-s} lines)")
        else:
            print(f"  SKIP  {name:40s}  (not found in viewer.py — already moved?)")
    
    if not to_move:
        print("Nothing to move.")
        return
    
    # Build text to append to crosshair.py
    append_lines = []
    for name, s, e in sorted(to_move, key=lambda x: x[1]):
        append_lines.extend(viewer_lines[s:e])
        # Ensure trailing newline
        if append_lines and not append_lines[-1].endswith("\n"):
            append_lines.append("\n")
        append_lines.append("\n")
    
    # Build viewer.py without the moved methods
    remove_lines = set()
    for name, s, e in to_move:
        for idx in range(s, e):
            remove_lines.add(idx)
    
    new_viewer = []
    for idx, line in enumerate(viewer_lines):
        if idx not in remove_lines:
            new_viewer.append(line)
    
    # Collapse runs of > 2 blank lines
    collapsed = []
    blank_run = 0
    for line in new_viewer:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                collapsed.append(line)
        else:
            blank_run = 0
            collapsed.append(line)
    
    # Append to crosshair.py (before the final blank line at end)
    new_crosshair = crosshair_text.rstrip() + "\n\n" + "".join(append_lines).rstrip() + "\n"
    
    removed = len(viewer_lines) - len(collapsed)
    print(f"\nviewer.py: {len(viewer_lines)} → {len(collapsed)} lines (removed {removed})")
    print(f"crosshair.py: gains {len(append_lines)} lines")
    
    if dry_run:
        print("\n[DRY RUN] No files written.")
        return
    
    VIEWER.write_text("".join(collapsed), encoding="utf-8")
    CROSSHAIR.write_text(new_crosshair, encoding="utf-8")
    print(f"\n✅ Written: {VIEWER}")
    print(f"✅ Written: {CROSSHAIR}")


if __name__ == "__main__":
    main()
