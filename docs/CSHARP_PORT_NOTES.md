# C# / outsource port notes (out of Python refactor scope)

**Status:** Reference only — **not** part of the active Python refactor in `REFACTOR_PLAN_PROFESSIONAL.md`.  
**Last updated:** 2026-07-23  

This document parks C# / .NET port topics so they do not drive Python extract work.

## Already documented in-repo

| Topic | Where |
|-------|--------|
| Online TCP port 8000, packets 520/528 | `docs/adr/ADR-001-online-protocol.md` |
| Crosshair / `updatePoint` contract | `docs/adr/ADR-002-updatepoint-crosshair-sync.md` |
| Ship layout (`dist/Inno3D`, `V2/`) | `docs/adr/ADR-003-packaging-layout.md`, `docs/OUTSOURCE_PACKAGE.md` |
| Native SEG config ABI | `inno3d/core/bumpvoid.py` (`BumpVoidConfig` and load helpers) |
| Thin native façades | `inno3d/native/{seg,mes,b2b,enhance}.py` → re-export core |

## Stable Python ports C# teams may call later

1. **Online FOV path** — receive host path → crop → load → enhance? → SEG → MES → B2B → DB.  
2. **Teaching “Apply for Online”** — ROI + recipe payload via `apply_online_roi_signal` / `set_paths_and_run`.  
3. **Result artifacts** — MES object CSV, B2B `boundary_gap*.csv` / `boundary_summary.csv`, inspection DB schema.  
4. **Wafer path parse** — `inno3d.core.wafer_context` / future `domain/wafer_path.py`.

## Explicitly out of active Python refactor

- Full rewrite of algorithms into C#.  
- Migrating Teaching / Viewer UI to WinForms/WPF.  
- Changing DLL ABI for “easier C# interop” without a separate product decision.  
- Expanding work under historical `teaching_tab_csharp_port_plan.md` unless product schedules a port epic.

## When to resume

Only when product schedules a **C# port epic**. Until then: keep Python APIs stable; do not refactor Python solely to ease a future port.
