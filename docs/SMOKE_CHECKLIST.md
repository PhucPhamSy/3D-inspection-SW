# Inno3D HBM — Smoke Checklist

> **Purpose:** Quick manual regression gate after any refactor phase.  
> Run before merging a PR that touches Viewer / Teaching / Online.  
> Mark each item ✅ pass / ❌ fail / ⚠️ partial.

---

## How to run

```bash
conda activate inno3d_ai
python main.py
```

Packaged build:
```
dist\Inno3D\Inno3D.exe
```

---

## 8.1 Dev smoke (`python main.py`)

### Startup

| # | Check | Result |
|---|-------|--------|
| S1 | App starts — splash shown, no crash | |
| S2 | Theme loads (dark / light per `app_config.ini`) | |
| S3 | All tabs visible (Viewer, Teaching, Online, Batch Review, Analysis, AI, Help) | |
| S4 | No import error in console / `Inno3D_Logs/` | |

---

### Viewer tab

| # | Check | Result |
|---|-------|--------|
| V1 | Load multipage TIFF → volume rendered in all 3 MPR panes + 3D | |
| V2 | Scroll wheel → zoom at cursor (2D panes) | |
| V3 | Ctrl + Scroll → step through slices | |
| V4 | Middle-click drag (or Shift+LMB) → pan | |
| V5 | Crosshair ON: click in pane → crosshair appears | |
| V6 | Drag crosshair → X/Y/Z sliders update in sync | |
| V7 | Move slider → crosshair hair moves in sync | |
| V8 | Double-click pane → fit view | |
| V9 | 3D pane: LMB rotate, middle pan, scroll zoom (pivot-based, no display_xy pan glitch) | |
| V10 | Window/Level slider changes brightness correctly | |

> **Crosshair contract:** `crosshair_position = [X, Y, Z]` maps to  
> `sagittal=X, coronal=Y, axial=Z`.  All moves must go through `updatePoint`.

---

### Teaching tab

| # | Check | Result |
|---|-------|--------|
| T1 | "Load DLL Folder" → select `V2/` → DLL loaded (no error) | |
| T2 | `BumpVoidSeg.dll` preferred; `BumpVoidDLL.dll` fallback | |
| T3 | Load config file → params populate | |
| T4 | Run SEG on small volume → mask returned, stats shown | |
| T5 | Enhancement toggle loads ONNX model (if GPU available) | |
| T6 | Recipe config `.txt` loads relative path (no hard-coded `E:\`) | |

---

### Online mode

| # | Check | Result |
|---|-------|--------|
| O1 | Online toggle ON → server starts on port 8000 | |
| O2 | Send one FOV path packet (520 or 528 bytes) | |
| O3 | Progress bar shows stages: Load → Enhance? → Segment → Measure → Done | |
| O4 | Wafer map updates with result | |
| O5 | DB record written to `Inno3D_Data/` | |
| O6 | Toggle OFF → server stops cleanly | |

---

### Batch Review / Analysis

| # | Check | Result |
|---|-------|--------|
| B1 | Batch Review tab opens, DB rows load | |
| B2 | Click row → volume loads in viewer | |
| A1 | Analysis tab opens without error | |
| A2 | Charts/stats render for existing DB records | |

---

## 8.2 Packaged smoke (`dist\Inno3D\Inno3D.exe`)

| # | Check | Result |
|---|-------|--------|
| P1 | Machine has NO `E:\semiconductor` path | |
| P2 | `DLL Folder` auto-detects `...\Inno3D\V2` | |
| P3 | SEG loads from shipped `V2/BumpVoidSeg.dll` | |
| P4 | Online receive FOV completes full pipeline | |
| P5 | Logs written to `Inno3D_Logs\` next to exe | |
| P6 | Zip (`dist/Inno3D`) includes `V2/`, `config/`, `app_config.ini` | |

---

## pytest gate (always run)

```bash
conda activate inno3d_ai
pytest tests/unit -q
# Expected: ≥ 68 passed, 0 failed
```

---

## Phase-specific gates

### After Phase 3.1b + 3.2b (viewer de-dupe — **Opus task**)

```bash
conda activate inno3d_ai
python -c "
from inno3d.tabs.viewer import MultiPlanarView
import inspect
print(inspect.getfile(MultiPlanarView.updatePoint))
print(inspect.getfile(MultiPlanarView._fiji_zoom_at_cursor))
"
# Must print:
#   ...features/viewer/crosshair.py
#   ...features/viewer/mpr_nav.py
```

Then run: **V1–V10** (all Viewer rows above).

### After Phase 4 (teaching split — Opus)

Run: **T1–T6** + **O1–O6** (Teaching internals unchanged from Online's perspective).

### After Phase 5 (online service — Opus)

Run: **O1–O6** with real FOV packet. Verify packet size 520/528 still accepted.

### After Phase 8 (packaging — Sonnet)

Run: **P1–P6** on a clean machine without source tree.

---

## Known-good baseline (2026-07-22)

- Unit tests: **68 passed** (`tests/unit/`)
- Packaged: `dist/Inno3D/` — exe + `V2/` + `config/` + `app_config.ini` ✅
- SEG: `BumpVoidSeg.dll` preferred, `BumpVoidDLL.dll` fallback ✅
- Online progress: stage-only bar, non-blocking ✅

---

*Last updated: 2026-07-22 | Maintainer: refactor agent (Sonnet/Opus per phase)*
