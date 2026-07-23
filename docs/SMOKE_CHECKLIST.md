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

## Known-good baseline (2026-07-23 — post Phase 4–8 partial)

- Unit tests: **82 passed** (`tests/unit/` — added `test_native_loader.py`) ✅
- **Startup smoke (S1–S4):** ✅ App boots, all tabs visible, no import crash (post P1b slim)
- **Online smoke (O1–O6):** ✅ Full FOV pipeline verified:
  - Server starts port 8000 ✅
  - Packet 520 bytes received (Counter #13) ✅
  - 12 layers segmented (BumpVoidSeg.dll v0.0.0) ✅
  - MES: 457 objects, 1.3s ✅
  - B2B: 1530 gaps / 12 layers (BoundaryGPU.dll) ✅
  - DB catalogued (run_id `run_9c4933b9b612`) ✅
  - 3D viewer load after pipeline ✅
- **main.py:** 154 LOC (−93%) ✅
- **tabs/analysis.py:** 30 LOC (−99%) ✅
- **tabs/batch_review.py:** 36 LOC (−98%) ✅
- **tabs/teaching.py:** 156 LOC (−99%) ✅
- **modes/online.py:** 54 LOC (−98%) ✅

---

### After Phase 6 (main.py slim — P1b)

```bash
conda activate inno3d_ai
python -c "
import ast
src = open('main.py').read()
classes = [c.name for c in ast.walk(ast.parse(src)) if isinstance(c, type(ast.parse('class X: pass').body[0]))]
print('Classes in main.py (should be empty):', classes)
n = src.count(chr(10))
print(f'main.py: {n} lines (target < 200)')
"
# Expected: Classes in main.py: [] | lines: 154
```

Then run: **S1–S4** (full startup).

### After Phase 7 (analysis/batch slim — P1)

```bash
conda activate inno3d_ai
python -c "
from inno3d.tabs.analysis import AnalysisTab
from inno3d.features.analysis.tab_ui import AnalysisTab as FeatAT
print('Same class:', AnalysisTab is FeatAT)
from inno3d.tabs.batch_review import BatchReviewTab
from inno3d.features.batch_review.tab import BatchReviewTab as FeatBR
print('Same class:', BatchReviewTab is FeatBR)
"
# Expected: Same class: True (both lines)
```

### After Phase 8 (packaging — Sonnet)

Run: **P1–P6** on a clean machine without source tree.

```bash
# Verify VERSION.txt written next to exe after build_final.bat:
type dist\Inno3D\VERSION.txt
# Expected: 1.2.0
```

---

*Last updated: 2026-07-23 | Maintainer: refactor agent (Sonnet/Opus per phase)*
