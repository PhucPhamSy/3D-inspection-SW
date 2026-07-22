# Inno3D HBM — Professional Source Refactor Plan

**Status:** Ready for execution (Claude Opus / multi-PR stack)  
**Product:** Inno3D Inspection — HBM 3D void/bump analysis (Python + PyQt5 + VTK + native ISP DLLs)  
**Workspace:** `HBM_frontend_backend_v12`  
**Audience:** Refactor agent (Claude Opus) + human reviewer  
**Created:** 2026-07-22  
**Related context:** packaging for outsource (`build_final.bat` + `V2/`), Online progress UX, MPR nav (crosshair/zoom/pan), C# port plans  

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
3. **One PR = one concern** — reviewable; avoid “move everything” commits.
4. **No drive-by renames** of public DLL symbols / packet layout / DB schema.
5. **Absolute paths** allowed only as **dev fallback** behind `if not frozen` (document).
6. **Prefer composition** over another mixin on MainWindow.
7. **After each phase:** run app manually smoke + any unit tests added.
8. **Do not** expand `unnecessary/` or leave dead duplicates of moved code.

---

## 5. Phased plan (execute in order)

### Phase 0 — Baseline & safety net (1 PR)

**Why:** Safe refactor without regression blindness.

**Tasks:**

1. Document golden smoke checklist (below §8) in `docs/SMOKE_CHECKLIST.md`.
2. Add `tests/unit/` with pure tests (no GPU required):
   - `wafer_context.parse_host_volume_path` (or extracted `domain/wafer_path.py`)
   - `resources.default_dll_dir` / `app_runtime_dir` with monkeypatched `sys.frozen` / `sys.executable`
   - Layer Z remap helper if extractable from Online/Teaching
3. Pin `requirements.txt` (or export from current env) with versions used for build.
4. Add `VERSION` file or `__version__` in `inno3d/__init__.py`.
5. Ensure `.gitignore` covers `dist/`, `build/`, `Inno3D_Logs/`, `*.pyc`, large local DBs if needed.

**Exit:** `pytest tests/unit -q` green; smoke checklist filled once on dev machine.

---

### Phase 1 — Paths, config, logging (1–2 PRs)

**Why:** Outsource build failures + hard-coded paths.

**Tasks:**

1. Expand `inno3d/core/resources.py` → `inno3d/infra/paths.py` (keep shim re-export).
2. Audit and remove/guard all remaining absolute paths in runtime code:
   ```
   rg "E:\\\\|E:/" --glob "*.py"
   ```
   - Runtime: must use `app_runtime_dir() / "V2"` etc.
   - Dev-only fallbacks OK under `not frozen`.
3. Config system:
   - `app/settings.py` reads `app_config.ini` next to exe
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

**Suggested extraction order (keep `MultiPlanarView` as façade initially):**

| PR | Extract to | Contents |
|----|------------|----------|
| 3.1 | `features/viewer/mpr_nav.py` | Zoom Fiji, pan, wheel, shortcuts, link MPR, hit radii |
| 3.2 | `features/viewer/crosshair.py` | Hit test, drag, updatePoint sync, 2D/3D hair |
| 3.3 | `features/viewer/mpr_render.py` | `render_slice`, camera state, ruler |
| 3.4 | `features/viewer/volume_3d.py` | Volume actor, Dragonfly style host, lighting, TF glue |
| 3.5 | `features/viewer/window_level.py` + `clip_box.py` | WL widgets, Dragonfly clip |
| 3.6 | `features/viewer/stats_panel.py` | Object stats / B2B tables if tightly coupled |

**Rules:**

- `updatePoint` remains **single source of truth** for crosshair ↔ sliders (`X/Y/Z` mapping documented).
- Prefer **one** input path long-term (Qt eventFilter **or** VTK observers); document interim dual-path.
- Do not reintroduce 3D zoom-to-cursor pan (breaks Track pivot on volume render).
- Public methods used by Online/Teaching (`set_volume_data`, `update_all_views`, `clear_all_views`, overlay APIs) stay on façade or explicit ports.

**Exit:** `viewer/tab.py` < ~2.5k LOC façade; nav/crosshair/render in separate files; smoke Viewer OK.

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
| | | | |

---

## 11. Appendix — known recent fixes (do not regress)

1. **Packaging:** `build_final.bat` copies `V2/` and `config/` next to exe; portable `default_dll_dir()`.
2. **SEG names:** prefer `BumpVoidSeg.dll`, fallback `BumpVoidDLL.dll`.
3. **Online progress:** stage-only UI; smooth bar; non-blocking dialog.
4. **3D zoom:** pivot-based dolly; **do not** re-enable display_xy pan on volume render.
5. **Crosshair ↔ slicer:** always via `updatePoint` full sync (no desynced preview).

---

## 12. Appendix — recommended first three PRs for Opus

If capacity is limited, execute only:

1. **Phase 0** — tests + version + checklist  
2. **Phase 1** — paths/config/logging  
3. **Phase 3.1 + 3.2** — extract `mpr_nav` + `crosshair` from `viewer.py`  

Then reassess LOC and Online stability before Teaching/Online splits.

---

*End of plan. Save location: `docs/REFACTOR_PLAN_PROFESSIONAL.md`*
