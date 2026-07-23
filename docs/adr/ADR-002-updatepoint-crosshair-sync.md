# ADR-002: updatePoint — Single Source of Truth for Crosshair ↔ Slicer Sync

**Status:** Accepted  
**Date:** 2026-07-22  
**Deciders:** Inno3D team  
**Related:** Phase 3 (viewer extract), `features/viewer/crosshair.py`, `features/viewer/mpr_input.py`

---

## Context

The MPR viewer has three orthogonal slice panes (axial / coronal / sagittal) and a crosshair overlay. Both the crosshair position and the slice index sliders must stay in sync. Early code had multiple paths that could update one without the other, causing "ghost crosshair" desync bugs.

## Decision

**All crosshair + slider sync must pass through `updatePoint`.**

### Contract

```python
# crosshair_position is always volume-index coordinates
crosshair_position = [X, Y, Z]

# Slice indices map directly:
current_slices['sagittal'] = X   # left-right axis
current_slices['coronal']  = Y   # front-back axis
current_slices['axial']    = Z   # top-bottom axis
```

### `updatePoint(x, y, z)` responsibilities

1. Set `self.crosshair_position = [x, y, z]`
2. Update `self.current_slices` dict
3. Move all Qt sliders (without re-triggering `updatePoint` via `blockSignals`)
4. Re-render all slice panes via `render_slice`
5. Move the crosshair actors (hair lines) in all VTK renderers

### Rules (enforced)

- **No "preview path"** that updates hair position without updating sliders.
- **No direct `current_slices` mutation** outside `updatePoint` (except init).
- Slider `valueChanged` → `updatePoint`. Crosshair drag → `updatePoint`.
- VTK observer click → converts screen → volume index → `updatePoint`.

## Consequences

- Single debug point for crosshair bugs.
- All UI input paths (slider, VTK click, keyboard) converge on one function.
- Teaching MPR stack is a **parallel** implementation (not shared mixin) but follows the same contract — `seg_ui.py` has its own `updatePoint` equivalent.

---

## Alternative considered

**Qt signal chaining:** Ruled out — signal loops required `blockSignals` gymnastics; `updatePoint` is simpler and debuggable with a single breakpoint.
