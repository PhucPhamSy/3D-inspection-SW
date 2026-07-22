"""Phase 3.3b — MOVE render methods into MprRenderMixin + more crosshair into CrosshairMixin.

Batch 1 → MprRenderMixin (mpr_render.py):
  render_slice, add_ruler_overlay, reset_view

Batch 2 → CrosshairMixin (crosshair.py):
  create_crosshair_actors, on_crosshair_slider_changed, toggle_reverse_z

Usage:
  conda run -n inno3d_ai python scripts/move_mpr_render.py --dry-run
  conda run -n inno3d_ai python scripts/move_mpr_render.py
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "inno3d" / "tabs" / "viewer.py"
RENDER_OUT = ROOT / "inno3d" / "features" / "viewer" / "mpr_render.py"
CROSSHAIR_OUT = ROOT / "inno3d" / "features" / "viewer" / "crosshair.py"

RENDER_METHODS = {"render_slice", "add_ruler_overlay", "reset_view"}
CROSSHAIR_METHODS = {"create_crosshair_actors", "on_crosshair_slider_changed", "toggle_reverse_z"}

METHOD_RE = re.compile(r"^    def (\w+)\(")
STATIC_RE = re.compile(r"^    @staticmethod\s*$")

dry_run = "--dry-run" in sys.argv

RENDER_HEADER = '''"""MPR render pipeline mixin — Phase 3.3b extract from tabs/viewer.py.

Contains:
  - ``render_slice`` — reslice + VTK actor update for one MPR pane
  - ``add_ruler_overlay`` — physical-unit ruler lines
  - ``reset_view`` — fit camera to slice

Layer: features/viewer (imports Qt, VTK — not for domain/infra use).
"""
from __future__ import annotations

import numpy as np
import vtk
from PyQt5.QtCore import Qt


class MprRenderMixin:
    """Mixin providing MPR slice rendering for MultiPlanarView.

    All methods use ``self.*`` attributes owned by MultiPlanarView.
    Do not instantiate directly.
    """

'''


def parse_methods(lines):
    in_class = False
    methods = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("class MultiPlanarView"):
            in_class = True
        if not in_class:
            i += 1; continue
        if STATIC_RE.match(line):
            j = i + 1
            while j < len(lines) and lines[j].strip() == "": j += 1
            m = METHOD_RE.match(lines[j]) if j < len(lines) else None
            if m:
                name = m.group(1)
                k = j + 1
                while k < len(lines):
                    if (lines[k].startswith("    def ") or lines[k].startswith("    @") or lines[k].startswith("class ")) and k > j + 1:
                        break
                    k += 1
                methods.append((name, i, k))
                i = k; continue
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


def extract(methods, targets, viewer_lines):
    to_move = []
    for name in sorted(targets):
        found = [(n, s, e) for n, s, e in methods if n == name]
        if found:
            # Take the LAST occurrence (in case of duplicates, the one lower in file is usually the real one)
            n, s, e = found[-1]
            to_move.append((n, s, e))
            print(f"  MOVE  {n:35s}  lines {s+1}–{e}  ({e-s} lines)")
        else:
            print(f"  SKIP  {n:35s}  (not found)")
    return to_move


def main():
    viewer_text = VIEWER.read_text(encoding="utf-8")
    viewer_lines = viewer_text.splitlines(keepends=True)
    methods = parse_methods(viewer_lines)

    print("=== Batch 1: MprRenderMixin ===")
    render_moves = extract(methods, RENDER_METHODS, viewer_lines)

    print("\n=== Batch 2: CrosshairMixin (append) ===")
    ch_moves = extract(methods, CROSSHAIR_METHODS, viewer_lines)

    all_moves = render_moves + ch_moves
    if not all_moves:
        print("Nothing to move.")
        return

    # Build render mixin file
    render_body = []
    for name, s, e in sorted(render_moves, key=lambda x: x[1]):
        render_body.extend(viewer_lines[s:e])
        if render_body and not render_body[-1].endswith("\n"):
            render_body.append("\n")
        render_body.append("\n")
    render_text = RENDER_HEADER + "".join(render_body).rstrip() + "\n"

    # Append to crosshair.py
    ch_body = []
    for name, s, e in sorted(ch_moves, key=lambda x: x[1]):
        ch_body.extend(viewer_lines[s:e])
        if ch_body and not ch_body[-1].endswith("\n"):
            ch_body.append("\n")
        ch_body.append("\n")
    crosshair_text = CROSSHAIR_OUT.read_text(encoding="utf-8")
    new_crosshair = crosshair_text.rstrip() + "\n\n" + "".join(ch_body).rstrip() + "\n"

    # Remove from viewer.py
    remove_lines = set()
    for name, s, e in all_moves:
        for idx in range(s, e):
            remove_lines.add(idx)

    new_viewer = [l for i, l in enumerate(viewer_lines) if i not in remove_lines]
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
    r_total = sum(e - s for _, s, e in render_moves)
    c_total = sum(e - s for _, s, e in ch_moves)
    print(f"\nviewer.py: {len(viewer_lines)} → {len(collapsed)} lines (removed {removed})")
    print(f"mpr_render.py: NEW file with {r_total} method lines")
    print(f"crosshair.py: gains {c_total} lines")

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return

    RENDER_OUT.parent.mkdir(parents=True, exist_ok=True)
    RENDER_OUT.write_text(render_text, encoding="utf-8")
    CROSSHAIR_OUT.write_text(new_crosshair, encoding="utf-8")
    VIEWER.write_text("".join(collapsed), encoding="utf-8")
    print(f"\n✅ Written: {RENDER_OUT}")
    print(f"✅ Written: {CROSSHAIR_OUT}")
    print(f"✅ Written: {VIEWER}")


if __name__ == "__main__":
    main()
