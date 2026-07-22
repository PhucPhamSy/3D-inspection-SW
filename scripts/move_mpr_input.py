"""Phase 3.3a — MOVE eventFilter + setup_vtk_observers into MprInputMixin.

Methods to move:
  setup_vtk_observers  → MprInputMixin  (VTK interactor observers for MPR panes)
  eventFilter          → MprInputMixin  (Qt event dispatch for all panes)

Usage:
  conda run -n inno3d_ai python scripts/move_mpr_input.py --dry-run
  conda run -n inno3d_ai python scripts/move_mpr_input.py
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "inno3d" / "tabs" / "viewer.py"
OUTPUT = ROOT / "inno3d" / "features" / "viewer" / "mpr_input.py"

METHODS_TO_MOVE = {
    "setup_vtk_observers",
    "eventFilter",
}

METHOD_RE = re.compile(r"^    def (\w+)\(")

dry_run = "--dry-run" in sys.argv


MIXIN_HEADER = '''"""MPR input handling mixin — Phase 3.3a extract from tabs/viewer.py.

Contains the two main input dispatch methods:
  - ``setup_vtk_observers`` — wires VTK interactor callbacks (LMB, move, release,
    scroll) for each MPR pane.
  - ``eventFilter`` — Qt-level event filter for all panes (3D + MPR), handling
    mouse, wheel, keyboard, double-click events.

Layer: features/viewer (imports Qt, VTK — not for domain/infra use).
"""
from __future__ import annotations

import vtk
from PyQt5.QtCore import Qt, QEvent, QPoint
from PyQt5.QtWidgets import QApplication


class MprInputMixin:
    """Mixin providing mouse/keyboard input dispatch for MultiPlanarView.

    All methods use ``self.*`` attributes owned by MultiPlanarView.
    Do not instantiate directly.
    """

'''


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
    
    methods = parse_methods(viewer_lines)
    name_to_range = {n: (s, e) for n, s, e in methods}
    
    to_move = []
    for name in sorted(METHODS_TO_MOVE):
        if name in name_to_range:
            s, e = name_to_range[name]
            to_move.append((name, s, e))
            print(f"  MOVE  {name:30s}  lines {s+1}–{e}  ({e-s} lines)")
        else:
            print(f"  SKIP  {name:30s}  (not found)")
    
    if not to_move:
        print("Nothing to move.")
        return
    
    # Build mixin file
    body_lines = []
    for name, s, e in sorted(to_move, key=lambda x: x[1]):
        body_lines.extend(viewer_lines[s:e])
        if body_lines and not body_lines[-1].endswith("\n"):
            body_lines.append("\n")
        body_lines.append("\n")
    
    mixin_text = MIXIN_HEADER + "".join(body_lines).rstrip() + "\n"
    
    # Remove from viewer.py
    remove_lines = set()
    for name, s, e in to_move:
        for idx in range(s, e):
            remove_lines.add(idx)
    
    new_viewer = []
    for idx, line in enumerate(viewer_lines):
        if idx not in remove_lines:
            new_viewer.append(line)
    
    # Collapse >2 blank lines
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
    
    removed = len(viewer_lines) - len(collapsed)
    total_moved = sum(e - s for _, s, e in to_move)
    print(f"\nviewer.py: {len(viewer_lines)} → {len(collapsed)} lines (removed {removed})")
    print(f"mpr_input.py: new file with {total_moved} method lines + header")
    
    if dry_run:
        print("\n[DRY RUN] No files written.")
        return
    
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(mixin_text, encoding="utf-8")
    VIEWER.write_text("".join(collapsed), encoding="utf-8")
    print(f"\n✅ Written: {OUTPUT}")
    print(f"✅ Written: {VIEWER}")


if __name__ == "__main__":
    main()
