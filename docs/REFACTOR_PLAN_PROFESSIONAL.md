# Inno3D HBM — Professional Source Refactor Plan

**Status:** In progress — Viewer Phase 3 through **3.3b ✅** · Phase 0 ✅ · Phase 1 ✅ · human smoke ✅  
**Product:** Inno3D Inspection — HBM 3D void/bump analysis (Python + PyQt5 + VTK + native ISP DLLs)  
**Workspace:** `HBM_frontend_backend_v12`  
**Audience:** Refactor agent (Claude Opus) + human reviewer  
**Created:** 2026-07-22  
**Last updated:** 2026-07-22 (re-audit source; 3.2c–3.3b landed)  
**Agents:** Claude **Opus** (hard / high-risk), Claude **Sonnet** (easy / mechanical)  
**Env:** `conda activate inno3d_ai` (Python 3.10+)  
**Related context:** packaging for outsource (`build_final.bat` + `V2/`), Online progress UX, MPR nav (crosshair/zoom/pan), C# port plans  

---

## 0. Snapshot — repo health (re-audit)

Use this section as the **source of truth for “where we are”** before starting the next agent session.  
**Verify command:** `conda activate inno3d_ai` then `pytest tests/unit -q`.

### 0.1 Verified OK (automated + human)

| Item | Status |
|------|--------|
| Unit tests | **68 passed** (`inno3d_ai`) |
| Human smoke | **✅ User confirmed OK** |
| `VERSION` / `inno3d.__version__` | **1.2.0** |
| Phase 0 artifacts | `requirements.txt`, `pyproject.toml`, `docs/SMOKE_CHECKLIST.md` |
| Phase 1 paths | relative `ENHANCED_MODEL_PATH` + runtime resolve (logged done) |
| Method **duplicates** viewer ↔ feature mixins | **none** |
| Mixin imports | **top of** `tabs/viewer.py` (lines 43–46) |
| `preview_only` dead path | **removed** (no True/False force left) |
| Packaged layout | `dist/Inno3D/` + `V2/` + config |

### 0.2 Viewer extract map (current — inspect.getfile)

MRO: `MultiPlanarView` → `MprInputMixin` → `MprRenderMixin` → `MprNavMixin` → `CrosshairMixin` → `QWidget`

| Symbol | Module (owner) | LOC file (approx) |
|--------|----------------|-------------------|
| `updatePoint`, hit/click, `update_3d_crosshair`, actors | `features/viewer/crosshair.py` | ~1,100 |
| zoom/pan/slice scroll helpers | `features/viewer/mpr_nav.py` | ~500 |
| `eventFilter`, `setup_vtk_observers` | `features/viewer/mpr_input.py` | ~550 |
| `render_slice`, ruler, `reset_view` | `features/viewer/mpr_render.py` | ~490 |
| **Still on** `tabs/viewer.py` | UI shell, 3D volume, WL, clip, stats/tables, align, `update_slice`, `save_camera_state`, … | **~12,827 lines / ~277 methods** |

**Progress on viewer size:** ~15k → **~12.8k** (nav+crosshair+input+render extracted). Still far from &lt;2.5k façade target.

### 0.3 Remaining debt (honest feedback)

| ID | Issue | Status | Agent |
|----|--------|--------|-------|
| F1 | Crosshair cohesion | **✅ fixed in 3.2c** | — |
| F2 | `preview_only` dead API | **✅ fixed** | — |
| F3 | Mid-file imports | **✅ fixed** (top imports) | — |
| F3b | Mojibake / encoding in comments | ⚠️ low priority if only intentional UTF-8 | Sonnet optional |
| F4 | God-file remainder (3D, WL, clip, FrozenTable, stats, align) | 🔴 open (~12.8k still on façade) | **Opus 3.4–3.6** |
| F4b | `save_camera_state` / `update_slice` still on façade (OK short-term) | 🟡 minor | Opus optional with 3.3b/3.4 |
| F5 | `requirements.txt` | **✅ improved** (scikit-image, tifffile, pyqt5, …) | — |
| F6 | Layout A (`pipeline/`, `domain/`, `store/`) | ⚪ not started — **wait** | later |
| F7 | `docs/V2_PACKAGE.md`, `docs/OUTSOURCE_PACKAGE.md` | **✅ present** | — |
| F8 | `features/viewer/__init__.py` still empty (no public exports) | 🟡 minor | Sonnet |
| F9 | Teaching / Online / AI god-files untouched | 🔴 open | Opus later phases |

### 0.4 Extract rule (enforced)

**MOVE, never COPY.** Dual definitions = failed PR.  
After MOVE, keep **related symbols** in the same epic (do not strand hit-test / input next to façade).

### 0.5 Package layout vs target A

Stakeholder target (scale layout):

```text
inno3d/
  app/        # shell
  viewer/     # MPR + volume (final name)
  pipeline/   # SEG/MES/B2B/enhance + jobs
  domain/     # pure models
  store/      # DB
  ai/
  online/
  analysis/
```

**Current tree is intermediate** (`features/`, `core/`, `modes/`, `tabs/`). Do **not** big-bang rename to A until viewer extract is complete and de-duplicated. Mapping:

| Target A | Interim (now) | Final migration |
|----------|---------------|-----------------|
| `app/` | `app/` | keep |
| `viewer/` | `features/viewer/` + `tabs/viewer.py` | rename after split |
| `pipeline/` | `core/bumpvoid*` | Phase 2 native/pipeline |
| `domain/` | bits of `wafer_context`, models | Phase 0+/domain |
| `store/` | `core/inspection_db.py` | rename later |
| `online/` | `modes/online.py` | Phase 5 |
| `analysis/` | `tabs/analysis.py` | Phase 7 |
| `ai/` | `tabs/ai.py` + `services/` | Phase 7 |

---

## 1. Goals

Make the codebase **production-grade, maintainable, and portable** without changing product behavior unless explicitly listed as an intentional fix.

| Goal | Definition of done |
|------|-------------------|
| **Structure** | Clear layers: app shell → features/tabs → domain services → native/IO → UI kit |
| **Size** | No single module > ~2,000 lines; god-files split by responsibility |
| **Portability** | Zero hard-coded `E:\semiconductor\...` on frozen runtime paths; `V2/` next to exe |
| **Testability** | Domain pure logic unit-tested; critical UI flows smoke-testable |
| **Packaging** | One reproducible build script; documented ship layout for outsource/C# teams |
| **DX** | Consistent logging, errors, typing on public APIs, predictable config |

### Non-goals (do not do in this refactor)

- Full rewrite to PySide6 / Qt6 (optional later phase)
- Porting core algorithms into pure Python (native DLLs stay)
- Redesigning every UI pixel
- Completing C# rewrite (only keep Python API contracts stable for C# consumers)
- Deleting working features without migration

---

## 2. Current state (inventory)

### 2.1 Size hotspots (approx. lines)

| Module | ~LOC | Problem |
|--------|------|---------|
| `inno3d/tabs/viewer.py` | 14,200 | God object: MPR, 3D, WL, clip, oblique, stats, nav, overlays |
| `inno3d/tabs/teaching.py` | 10,300 | SEG UI + threads + layers + enhance + online hooks |
| `inno3d/tabs/ai.py` | 6,400 | AI tab + training + services mixed |
| `inno3d/tabs/analysis.py` | 3,000 | SOH/analysis + DB queries mixed |
| `inno3d/modes/online.py` | 2,750 | TCP + pipeline + UI mixin + DB |
| `main.py` | 2,000 | App shell + splash + wiring + some UI |

### 2.2 Architectural smells

1. **God tabs** — Viewer/Teaching own too many concerns; hard to review, test, or port to C#.
2. **Path chaos** — Historical absolute paths; partially fixed via `resources.app_runtime_dir()` / `default_dll_dir()` — must be completed and enforced.
3. **Mixin mega-window** — `OnlineModeMixin` on MainWindow blurs Online vs shell.
4. **Dual event paths** — Qt `eventFilter` + VTK observers both handle MPR input (race / double-fire risk).
5. **Native load surface** — Multiple wrappers (`bumpvoid`, `_mes`, `_b2b`, `enhanced_volume`) with duplicated `add_dll_directory` / PATH logic.
6. **Config drift** — `app_config.ini`, recipe `config_*.txt`, hard-coded defaults, ENHANCED_MODEL_PATH absolute.
7. **Logging** — stdout redirected to file only; structured levels / UI correlation incomplete.
8. **Build** — `build_final.bat` improved (copies `V2/`) but no version stamp, smoke test, or slim package profile.
9. **Dead / parallel trees** — `unnecessary/`, mixed docs, C# plans at repo root.
10. **No automated tests** for domain (wafer path parse, layer remap, DLL path resolution, DB schema).

### 2.3 What already works (preserve)

- Online FOV pipeline: receive path → crop → load → optional enhance → SEG → MES → B2B → DB → wafer map
- Teaching SEG recipe lock for Online
- Multi-pane MPR + 3D volume (Dragonfly-inspired)
- Inspection SQLite catalog (`inspection_db.py`)
- Portable runtime helpers in `inno3d/core/resources.py` (expand, do not delete)

---

## 3. Target architecture

```
HBM_frontend_backend_v12/
├── main.py                      # thin entry only
├── pyproject.toml / requirements.txt
├── build_final.bat
├── app_config.ini
├── config/                      # recipes (no machine-absolute paths in defaults)
├── V2/                          # native runtime (not imported as Python package)
├── assets/
├── docs/
│   └── REFACTOR_PLAN_PROFESSIONAL.md   # this file
├── inno3d/
│   ├── app/                     # application shell
│   │   ├── bootstrap.py         # logging, excepthook, high-DPI
│   │   ├── main_window.py
│   │   ├── splash.py
│   │   └── settings.py          # app_config.ini + env overrides
│   ├── domain/                  # pure logic, no Qt if possible
│   │   ├── geometry.py          # voxel/world mapping helpers
│   │   ├── wafer_path.py        # host path parse (from wafer_context)
│   │   ├── layers.py            # layer remap / recipe
│   │   └── models.py            # dataclasses / enums
│   ├── native/                  # all ctypes / DLL loaders
│   │   ├── loader.py            # shared add_dll_directory / PATH
│   │   ├── seg.py               # BumpVoidSeg / legacy DLL
│   │   ├── mes.py
│   │   ├── b2b.py
│   │   └── enhance.py
│   ├── services/                # use-cases (online pipeline, inspection catalog)
│   │   ├── online_pipeline.py
│   │   ├── inspection_catalog.py
│   │   └── ai/                  # existing services moved here
│   ├── ui/                      # shared widgets, theme, progress
│   │   ├── theme.py
│   │   ├── progress.py          # OnlinePipelineProgress
│   │   └── icons.py
│   ├── features/                # feature modules (one public tab each)
│   │   ├── viewer/              # split from viewer.py
│   │   │   ├── tab.py
│   │   │   ├── mpr_nav.py
│   │   │   ├── mpr_render.py
│   │   │   ├── volume_3d.py
│   │   │   ├── crosshair.py
│   │   │   ├── window_level.py
│   │   │   ├── clip_box.py
│   │   │   └── stats_panel.py
│   │   ├── teaching/
│   │   │   ├── tab.py
│   │   │   ├── seg_worker.py
│   │   │   ├── enhance_worker.py
│   │   │   └── recipe.py
│   │   ├── online/
│   │   │   ├── server.py        # TCP protocol
│   │   │   ├── controller.py    # orchestration (not a Qt mixin mega-file)
│   │   │   └── ui_log.py
│   │   ├── analysis/
│   │   ├── batch_review/
│   │   ├── ai/
│   │   └── help/
│   └── infra/
│       ├── paths.py             # app_runtime_dir, default_dll_dir (from resources)
│       ├── db.py                # inspection_db
│       └── logging_setup.py
└── tests/
    ├── unit/
    └── smoke/
```

### Layer rules

| Layer | May import | Must not import |
|-------|------------|-----------------|
| `domain` | stdlib, numpy (sparingly) | Qt, VTK, ctypes loaders |
| `native` | domain, infra.paths | Qt tabs |
| `services` | domain, native, infra | concrete Tab widgets |
| `features/*` | services, ui, domain | other features’ private modules (prefer signals/ports) |
| `app` | features, infra, ui | native details |

---

## 4. Principles for the refactor agent

1. **Behavior-preserving first** — golden path Online + Teaching SEG + Viewer load must pass after each PR.
2. **Strangler fig** — extract modules; keep re-exports so old imports work during transition.
3. **MOVE, do not COPY** — after extract, **delete** the same methods from the god-file. Dual definitions are a failed extract. **Also import every free name** used by moved methods into the target module (`numpy_support`, `ndimage`, Qt types, …) — Python looks up names in the **defining** module, not the host class.
4. **One PR = one concern** — reviewable; avoid “move everything” commits.
5. **No drive-by renames** of public DLL symbols / packet layout / DB schema.
6. **Absolute paths** allowed only as **dev fallback** behind `if not frozen` (document).
7. **Prefer composition** over another mixin on MainWindow.
8. **After each phase:** run app manually smoke + any unit tests added; update §10 Progress log.
9. **Do not** expand `unnecessary/` or leave dead duplicates of moved code.
10. **Do not** rename to package layout A until the current god-file for that area is actually thin (< ~2.5k façade).
---

## 5. Phased plan (execute in order)

### Phase 0 — Baseline & safety net (1 PR)

**Why:** Safe refactor without regression blindness.

**Status:** ⚠️ **Partial** — unit tests green; missing VERSION / requirements / SMOKE_CHECKLIST.

**Tasks:**

1. Document golden smoke checklist (below §8) in `docs/SMOKE_CHECKLIST.md`.
2. Add `tests/unit/` with pure tests (no GPU required): ✅ largely done
   - `wafer_context.parse_host_volume_path` (or extracted `domain/wafer_path.py`)
   - `resources.default_dll_dir` / `app_runtime_dir` with monkeypatched `sys.frozen` / `sys.executable`
   - Layer Z remap helper if extractable from Online/Teaching
3. Pin `requirements.txt` (or export from current env) with versions used for build. ❌
4. Add `VERSION` file or `__version__` in `inno3d/__init__.py`. ❌
5. Ensure `.gitignore` covers `dist/`, `build/`, `Inno3D_Logs/`, `*.pyc`, large local DBs if needed.

**Exit:** `pytest tests/unit -q` green; smoke checklist filled once on dev machine; VERSION + requirements present.

---

### Phase 1 — Paths, config, logging (1–2 PRs)

**Why:** Outsource build failures + hard-coded paths.

**Status:** ⚠️ **Partial** — `infra/paths`, `logging_setup`, `app/settings` exist; recipe absolute paths / full audit may remain.

**Tasks:**

1. Expand `inno3d/core/resources.py` → `inno3d/infra/paths.py` (keep shim re-export). ✅
2. Audit and remove/guard all remaining absolute paths in runtime code:
   ```
   rg "E:\\\\|E:/" --glob "*.py"
   ```
   - Runtime: must use `app_runtime_dir() / "V2"` etc.
   - Dev-only fallbacks OK under `not frozen`.
3. Config system:
   - `app/settings.py` reads `app_config.ini` next to exe ✅ (theme path)
   - Optional sections: `[DLL] dir=`, `[ONLINE] port=`, `[APPEARANCE] theme=`
4. Recipe defaults: relative `config/config_HBM_c2848_M.txt`; rewrite absolute `ENHANCED_MODEL_PATH` to `V2/restormer_....onnx` when present.
5. Central logging in `infra/logging_setup.py`:
   - file + optional console
   - do not swallow stdout in a way that breaks CLI/pytest
   - structured prefix: `[ONLINE]`, `[SEG]`, `[NAV]`

**Exit:** Frozen or simulated frozen resolves `V2` next to “exe”; no absolute path required for DLL default on clean machine.

---

### Phase 2 — Native DLL façade (1–2 PRs)

**Why:** Duplicated load logic; confusing SEG vs legacy names; packaging clarity.

**Tasks:**

1. `native/loader.py`:
   - `add_search_dirs(dll_dir)`
   - CUDA path discovery (shared)
2. Unify wrappers under `native/`:
   - `seg.py` ← `bumpvoid.py` (BumpVoidSeg + BumpVoidDLL fallback — already intended)
   - `mes.py`, `b2b.py`, `enhance.py`
3. Public re-exports from old modules for one release cycle.
4. Error messages: never leak algorithm internals to Online UI progress; keep detail in logs only.
5. Document required `V2` file list in `docs/V2_PACKAGE.md` (minimal vs full CUDA/TRT).

**Exit:** Teaching “Load DLL folder” works against app-local `V2/`; unit test for name fallback.

---

### Phase 3 — Split `viewer.py` (3–5 PRs) ⭐ largest

**Why:** 14k LOC blocks all professional work.

**Status:** 🟢 **3.1 → 3.3b DONE** (nav, crosshair+cohesion, input, render). `viewer.py` ~**12.8k** left. **NEXT = 3.4 volume_3d** (or 3.5 WL/clip).

**Suggested extraction order (keep `MultiPlanarView` as façade initially):**

| PR | Extract to | Contents | Status |
|----|------------|----------|--------|
| 3.1a/b | `mpr_nav.py` | Zoom Fiji, pan, wheel, shortcuts | ✅ |
| 3.2a/b/c | `crosshair.py` | updatePoint, hit, click, 2D/3D hair, actors | ✅ |
| 3.3a | `mpr_input.py` | `eventFilter` + `setup_vtk_observers` | ✅ |
| 3.3b | `mpr_render.py` | `render_slice`, ruler, `reset_view` | ✅ |
| **3.4** | **`volume_3d.py`** | Volume actor, Dragonfly interactor host, lighting, TF glue | ❌ **NEXT Opus** |
| 3.5 | `window_level.py` + `clip_box.py` | WL widgets, Dragonfly clip | not started |
| 3.6 | `stats_panel.py` + tables | FrozenTable / Excel filters / object stats | not started |
| 3.7 | rename → `inno3d/viewer` (layout A) | Only after façade thin | later |

**Acceptance so far:** pytest 68 + human smoke ✅ after early phases; re-smoke after each 3.4+ extract.

**3.4 acceptance (next):**

```bash
conda activate inno3d_ai
# volume helpers resolve outside tabs/viewer.py
python -c "from inno3d.tabs.viewer import MultiPlanarView; import inspect; print(inspect.getfile(MultiPlanarView.render_3d))"
# expect features/viewer/volume_3d.py (or agreed module)
pytest tests/unit -q
# human: load volume, 3D track/pan/zoom, lighting still works
```

**Rules:**

- `updatePoint` remains **single source of truth** for crosshair ↔ sliders.
- Input already consolidated in `mpr_input.py` — avoid re-duplicating observers.
- Do not reintroduce 3D zoom-to-cursor pan.
- MOVE not COPY; prove `inspect.getfile` for every moved public method.
- Prefer extracting **volume_3d** before stats tables (risk: 3D regressions).

**Exit (phase 3 overall):** `tabs/viewer.py` ≪ 5k then ≪ 2.5k façade; volume/WL/stats out; smoke Viewer OK.

---

### Phase 4 — Split `teaching.py` (2–3 PRs)

**Why:** SEG workers + UI + recipe + online apply mixed.

| PR | Extract |
|----|---------|
| 4.1 | Workers: `SegmentationInspectionThread`, `EnhancementThread`, layer prep |
| 4.2 | Recipe model: INPUT_MODE, layers, ROI apply-for-online |
| 4.3 | UI panels: params, enhance, stats hooks |

Keep stable API:

- `set_paths_and_run(...)`
- `set_volume_data` / `set_segmentation_results`
- `load_dll_from_path` / `load_config_from_path`
- Online “Apply for Online Mode” signal/payload

**Exit:** Teaching tab façade < ~2.5k; workers importable without full tab.

---

### Phase 5 — Online mode as service + thin controller (2 PRs)

**Why:** Mixin on MainWindow is hard to test and reason about.

1. Extract TCP protocol constants + `OnlineServerThread` → `features/online/server.py`.
2. Extract FOV pipeline orchestration → `services/online_pipeline.py` (callable steps, progress stage enums).
3. Replace fat mixin with:
   - `OnlineController` (QObject) owned by MainWindow
   - UI: stage progress (`OnlinePipelineProgress`), log panel hooks
4. Progress UI stays stage-only (Load / Enhance / Segment / Measure) — no algorithm text.
5. Keep packet layout 520/528 and port 8000 stable.

**Exit:** Online ON → receive FOV → complete pipeline without importing Teaching internals except narrow ports.

---

### Phase 6 — App shell & `main.py` thin entry (1 PR)

1. Move splash, MainWindow class, theme load → `inno3d/app/`.
2. `main.py` becomes:
   ```python
   from inno3d.app.bootstrap import run
   if __name__ == "__main__":
       run()
   ```
3. Lazy-import heavy tabs after splash (already partially done — keep).

**Exit:** `main.py` < 80 lines.

---

### Phase 7 — Analysis / Batch / AI hygiene (1–2 PRs)

- Analysis: separate DB queries from plot widgets.
- Batch review: keep DB-first design; extract row models.
- AI: keep `services/` pure; widgets only in `features/ai/` or `widgets/`.
- Remove or quarantine dead code under `unnecessary/` from packaging.

---

### Phase 8 — Packaging & outsource ship (1 PR)

1. `build_final.bat`:
   - version stamp into dist
   - copy `V2`, `config`, `app_config.ini`, create `Inno3D_Data`
   - optional `PACKAGE_SLIM=1` (core DLLs only, skip TRT bulk) for UI-only outsource
2. `docs/OUTSOURCE_PACKAGE.md`: zip entire `dist/Inno3D`, point DLL Folder to `V2`, GPU/driver notes.
3. Post-build smoke script (optional): launch exe with `--help` or env `INNO3D_SMOKE=1` exit 0 after QTimer.

**Exit:** Clean machine (no `E:\semiconductor`) can run SEG with shipped `V2`.

---

### Phase 9 — Polish (optional, last)

- Type hints on public APIs (`from __future__ import annotations`)
- `ruff` / `black` config; format only touched files per PR if full format too noisy
- Architecture decision records (ADR) for Online protocol, updatePoint mapping, packaging
- C# port note: list stable ports/interfaces Python exposes (for `teaching_tab_csharp_port_plan.md` alignment)

---

## 6. Cross-cutting conventions (apply during all phases)

### 6.1 Crosshair ↔ slicer contract (do not break)

```
crosshair_position = [X, Y, Z]   # volume indices
current_slices['sagittal'] = X
current_slices['coronal']  = Y
current_slices['axial']    = Z
```

Any crosshair move or slider move must go through **one** function (`updatePoint` or successor). No “preview path” that updates only hair without sliders.

### 6.2 MPR input (target)

| Gesture | Action |
|---------|--------|
| Scroll | Zoom at cursor (2D) |
| Ctrl+Scroll | Slice step |
| Middle / Shift+LMB | Pan |
| LMB | Crosshair place/drag |
| Double-click | Fit pane |

3D: Track (LMB) + pan (middle) + zoom about pivot (**no** unstable zoom-to-cursor pan).

### 6.3 Online progress stages

`load` → `enhance?` → `segment` → `measure` → `done`  
User-visible strings only; details in log.

### 6.4 Packaging layout

```
dist/Inno3D/
  Inno3D.exe
  _internal/
  V2/
  config/
  app_config.ini
  Inno3D_Data/
  Inno3D_Logs/   # created at runtime
```

---

## 7. PR / agent execution guide (for Claude Opus)

### Per-PR checklist

- [ ] Scope limited to phase subsection
- [ ] Re-exports for moved symbols
- [ ] No new hard-coded absolute runtime paths
- [ ] Smoke: Viewer load volume + Teaching load DLL/config + Online toggle (if touched)
- [ ] Unit tests for pure logic touched
- [ ] Update this plan’s **Progress log** (§10)

### Suggested agent prompt skeleton

```text
You are refactoring Inno3D HBM Python app per docs/REFACTOR_PLAN_PROFESSIONAL.md.
Execute ONLY Phase N.N. Preserve behavior. Keep public APIs used by Online/Teaching.
After changes, list files moved, re-exports added, and smoke steps the human should run.
Do not reformat entire repo. Do not touch V2 binary files.
```

### Risk controls

| Risk | Mitigation |
|------|------------|
| Break Online | Run TCP sample path after Phase 5; keep packet sizes |
| Break SEG | Compare mask outputs on fixture volume before/after Teaching split |
| VTK event regressions | Manual MPR nav checklist after Phase 3 |
| Import cycles | features → services → native/domain only |
| PyInstaller miss | Re-test frozen build after path/native phases |

---

## 8. Smoke checklist (manual)

### 8.1 Dev (`python main.py`)

1. App starts, theme loads, no crash.
2. Viewer: load multipage TIFF; scroll zoom; pan; crosshair on; drag hair → sliders move; move slider → hair moves.
3. Teaching: load `V2` + config; SEG on small volume.
4. Online: toggle ON; send one FOV path; progress stages; results + map update.
5. Batch Review / Analysis open without error.

### 8.2 Packaged (`dist/Inno3D/Inno3D.exe`)

1. Machine without project source paths.
2. DLL Folder auto = `...\Inno3D\V2`.
3. SEG load succeeds (`BumpVoidSeg.dll` or legacy).
4. Zip size includes `V2` (GB-scale if full).

---

## 9. Success metrics

| Metric | Target |
|--------|--------|
| Largest `.py` file | < 2,500 LOC (façade OK if thin) |
| `main.py` | < 100 LOC |
| Absolute path in frozen default DLL | 0 |
| Unit tests | ≥ 15 meaningful cases |
| Online/Teaching/Viewer golden smoke | Pass |
| Outsource can run SEG from zip | Documented + verified once |

---

## 10. Progress log

| Date | Phase | Notes | Agent |
|------|-------|-------|-------|
| 2026-07-22 | Plan authored | Baseline inventory; packaging/nav lessons captured | Planning |
| 2026-07-22 | 0 partial | `tests/unit` 68 passed; missing VERSION, requirements, SMOKE_CHECKLIST | Prior agent |
| 2026-07-22 | 1 partial | `infra/paths`, `logging_setup`, `app/settings`; main.py wired | Prior agent |
| 2026-07-22 | 3.1a / 3.2a | Created `MprNavMixin` / `CrosshairMixin` + MRO; **methods still duplicated in viewer.py** | Prior agent |
| 2026-07-22 | Health check | Confirmed dead mixin copies; dist/V2 package OK; plan refreshed | Grok review |
| 2026-07-22 | Plan | §12 Sonnet vs Opus assignment matrix | Grok |
| 2026-07-22 | **0 ✅ DONE** | `VERSION` (1.2.0), `requirements.txt` (inno3d_ai env pinned), `docs/SMOKE_CHECKLIST.md`, `conftest.py`, `pyproject.toml`; `.gitignore` updated (BoundaryDll/GPU, pytest cache, .log); `inno3d/__init__.py` reads VERSION file; **pytest 68 passed** | **Sonnet** |
| 2026-07-22 | **3.1b+3.2b ✅ DONE** | DELETED 33 duplicate methods (19 nav + 14 crosshair) from `tabs/viewer.py`. `inspect.getfile` verified mixins. Zero AST duplicates. **pytest 68 passed** | **Opus** |
| 2026-07-22 | Human smoke | User confirmed **smoke test OK** (Viewer checklist) | Human |
| 2026-07-22 | Review | Feedback F1–F7 written into plan | Grok |
| 2026-07-22 | **Bugfix** | Mixin define-scope imports; 3D fullscreen pivot sync | **Opus** |
| 2026-07-22 | **3.2c ✅ DONE** | hit/click/3d hair → `crosshair.py`; `preview_only` stripped | **Opus** |
| 2026-07-22 | **1 ✅ DONE** | Relative `ENHANCED_MODEL_PATH` + `app_runtime_dir` resolve | **Opus** |
| 2026-07-22 | **Bugfix** | Teaching relative enhance path | **Opus** |
| 2026-07-22 | **3.3a ✅ DONE** | `eventFilter` + `setup_vtk_observers` → `mpr_input.py` | **Opus** |
| 2026-07-22 | **3.3b ✅ DONE** | `render_slice`/ruler/`reset_view` → `mpr_render.py`; more hair → crosshair | **Opus** |
| 2026-07-22 | **F3+F5+F7 ✅ DONE** | imports top; requirements; V2/OUTSOURCE docs | **Sonnet** |
| 2026-07-22 | **Re-audit** | pytest 68; MRO Input→Render→Nav→Crosshair; no dups; viewer ~12827; next 3.4 | Grok |
| 2026-07-22 | **Bugfix** | Load volume / Online crash: `NameError: numpy_support is not defined` in `mpr_render.render_slice` (extract forgot `from vtk.util import numpy_support` + `scipy.ndimage`). Fixed imports in `features/viewer/mpr_render.py`. | Grok |
| | **NEXT (Opus)** | **3.4 volume_3d** — when extracting, **import every symbol used in moved code** (define-module scope) | **Opus** |
| | **NEXT (Sonnet)** | Optional F8 / Phase 9 only — do not block 3.4 | **Sonnet** |
| | | | |

---

## 11. Appendix — known recent fixes (do not regress)

1. **Packaging:** `build_final.bat` copies `V2/` and `config/` next to exe; portable `default_dll_dir()`.
2. **SEG names:** prefer `BumpVoidSeg.dll`, fallback `BumpVoidDLL.dll`.
3. **Online progress:** stage-only UI; smooth bar; non-blocking dialog.
4. **3D zoom:** pivot-based dolly; **do not** re-enable display_xy pan on volume render.
5. **Crosshair ↔ slicer:** always via `updatePoint` full sync (no desynced preview).
6. **Extract rule:** MOVE not COPY — dual method definitions = failed PR.
7. **Extract imports:** methods use **define-module** globals — every name (`numpy_support`, `ndimage`, `vtk`, …) must be imported in the **target** module, not only in `tabs/viewer.py`.

---

## 12. Agent assignment — Opus (hard) vs Sonnet (easy)

### 12.1 How to choose

| Prefer **Opus** when… | Prefer **Sonnet** when… |
|----------------------|-------------------------|
| Touches `viewer.py` / `teaching.py` / `online.py` core behavior | Adds files without rewriting gods |
| MOVE extract / **cohesion finish (3.2c, 3.3a)** | Docs, VERSION, requirements freeze, checklists |
| Crosshair ↔ slider / VTK+Qt event paths | Path audit grep + mechanical rewrites behind tests |
| Online TCP + pipeline orchestration | Re-exports / shims / `__init__.py` scaffolding |
| Split that can regress SEG or MPR | Unit tests for pure functions already stable |
| Judgment on “what stays on façade” | Hygiene: imports to top, mojibake, pin deps from `inno3d_ai` |

**Always run agents with:** `conda activate inno3d_ai` then `pytest` / inspect.

**Parallelism:** Sonnet may run **Phase 0 finish** or **docs/package docs** while Opus does **3.1b/3.2b**, as long as they do not both edit `tabs/viewer.py`.

**Never give Sonnet alone:** first-time extract of a new 2k+ slice from `viewer.py` without Opus review; Online pipeline redesign; “rename whole monorepo to layout A” in one shot.

---

### 12.2 Full phase matrix

| Phase | Difficulty | **Agent** | Why |
|-------|------------|-----------|-----|
| **0 finish** (VERSION, `requirements.txt`, `SMOKE_CHECKLIST.md`) | 🟢 Easy | **Sonnet** | Mechanical files; low risk |
| **0 extra** (more unit tests for pure helpers) | 🟢 Easy | **Sonnet** | Tests only; no UI |
| **1 finish** (recipe `ENHANCED_MODEL_PATH`, path audit, settings sections) | 🟡 Medium–easy | **Sonnet** (default) / Opus if many call sites break | Mostly grep + portable paths; use tests |
| **2** Native DLL façade (`pipeline/` or `native/`) | 🔴 Hard | **Opus** | ctypes, CUDA search, freeze packaging, SEG fallback |
| **3.1b + 3.2b** Delete duplicate methods in `viewer.py` | 🔴 Hard | **Opus** | Easy to break MRO / leave stubs / desync; high value |
| **3.3** `mpr_render` extract (MOVE) | 🔴 Hard | **Opus** | VTK render path; Online/Teaching call into it |
| **3.4** `volume_3d` extract (MOVE) | 🔴 Hard | **Opus** | Volume + lighting + TF glue |
| **3.5** WL + clip_box extract | 🟠 Hard–medium | **Opus** | Coupled UI + VTK |
| **3.6** stats_panel extract | 🟡 Medium | **Sonnet** *after* Opus façade is stable | More UI isolation if ports clear |
| **3.7** Rename `features/viewer` → `inno3d/viewer` (layout A) | 🟡 Medium | **Sonnet** *after* thin façade | Import rewrites + re-exports; mechanical if plan lists paths |
| **4** Split `teaching.py` | 🔴 Hard | **Opus** | SEG workers + Online hooks |
| **5** Online → controller/service | 🔴 Hard | **Opus** | Protocol + pipeline + progress |
| **6** Thin `main.py` / app shell | 🟡 Medium | **Sonnet** *if* only moves already-isolated code; else **Opus** | Import graph + splash order |
| **7** Analysis / Batch / AI hygiene | 🟡 Medium | **Sonnet** for AI services tidy; **Opus** if Analysis SOH logic moves | Prefer Sonnet for file moves + tests |
| **8** Packaging docs + `V2_PACKAGE.md` + slim profile notes | 🟢 Easy | **Sonnet** | Docs + bat comments; no algorithm |
| **9** Polish (ruff config, ADRs, type hints on public APIs) | 🟢 Easy–medium | **Sonnet** | Incremental; no behavior change |

---

### 12.3 Recommended queue (can run two agents)

```text
DONE
  ├─ Opus   → 3.1b/3.2b MOVE, 3.2c cohesion, 3.3a mpr_input, 3.3b mpr_render
  ├─ Opus   → Phase 1 path/recipe fix
  ├─ Sonnet → Phase 0 (VERSION, requirements, SMOKE_CHECKLIST)
  └─ Human  → smoke OK

NOW
  ├─ Opus   → 3.4 volume_3d (hard)
  └─ Sonnet → mojibake + V2_PACKAGE.md + OUTSOURCE_PACKAGE.md (easy)

THEN
  ├─ Opus   → 3.5 WL/clip, 3.6 stats/tables (or Sonnet 3.6 if ports clear)
  └─ Opus   → Phase 2 native/pipeline when viewer façade &lt; ~8k

LATER
  ├─ Opus   → Phase 4 teaching, Phase 5 online
  └─ Sonnet → 3.7 layout A rename, Phase 6 thin main, Phase 9 polish
```

---

### 12.4 Prompts (copy-paste)

#### A) Opus — next (hard) — 3.4 volume_3d

```text
You are Claude Opus. Env: conda activate inno3d_ai.
Read docs/REFACTOR_PLAN_PROFESSIONAL.md §0, Phase 3.4, §11, §12.

Execute ONLY Phase 3.4 (HARD — MOVE volume/3D stack out of tabs/viewer.py):
1. Create features/viewer/volume_3d.py with Volume3dMixin (or agreed name)
2. MOVE render_3d / volume actor setup / lighting / TF glue / Dragonfly style host hooks
   (only methods that clearly belong to 3D volume — do not move MPR render again)
3. Wire MRO on MultiPlanarView; DELETE originals from viewer.py (MOVE not COPY)
4. Do NOT touch teaching/online; do NOT rename package to layout A
5. Preserve: 3D Track/pan/zoom about pivot (NO display_xy pan), LOD if present
6. **CRITICAL:** in volume_3d.py import ALL names used by moved code (vtk, numpy_support, np, Qt, themes, …). Regression 2026-07-22: mpr_render crash NameError numpy_support.

Verify:
  conda activate inno3d_ai
  pytest tests/unit -q
  python -c "from inno3d.tabs.viewer import MultiPlanarView; import inspect; print(inspect.getfile(MultiPlanarView.render_3d))"
  # must be features/viewer/volume_3d.py (or listed module)
  # smoke-load: call path must not NameError on first render_slice / render_3d

Update §10. List human smoke: load volume, 3D rotate/pan/zoom, opacity/lighting.
```

#### B) Sonnet — optional (easy)

```text
You are Claude Sonnet. Env: conda activate inno3d_ai.
Read docs/REFACTOR_PLAN_PROFESSIONAL.md §0.3 F8, Phase 9.

V2_PACKAGE + OUTSOURCE docs already exist. Optional only:
1. features/viewer/__init__.py — document/export mixins + MRO
2. Phase 9 polish (ruff config / light typing) — no behavior change
3. Do NOT extract volume_3d (Opus owns 3.4)

pytest tests/unit -q; update §10 if you ship anything.
```

#### D) Opus — later hard tracks (pick one per session)

```text
You are Claude Opus. Read docs/REFACTOR_PLAN_PROFESSIONAL.md.

Execute ONLY [Phase 2 | Phase 3.3 mpr_render | Phase 4 teaching | Phase 5 online].
MOVE not COPY. One concern. Preserve §11 contracts.
pytest + list smoke steps. Update §10.
```

---

### 12.5 Human gate (you)

- **Baseline smoke:** ✅ confirmed OK after 3.1b/3.2b (2026-07-22).
- After **every later Opus** session that touches Viewer/Teaching/Online: re-run relevant rows of `docs/SMOKE_CHECKLIST.md` (especially crosshair ↔ sliders after 3.2c / 3.3a).
- Sonnet sessions: `conda activate inno3d_ai` → `pytest tests/unit -q` unless they touched `build_final.bat`.

---

*End of plan. Save location: `docs/REFACTOR_PLAN_PROFESSIONAL.md`*
