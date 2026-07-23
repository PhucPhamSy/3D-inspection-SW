# ADR-003: Packaging Layout — dist/Inno3D + V2 + config

**Status:** Accepted  
**Date:** 2026-07-22  
**Deciders:** Inno3D team  
**Related:** Phase 8 (packaging), `build_final.bat`, `docs/V2_PACKAGE.md`, `docs/OUTSOURCE_PACKAGE.md`

---

## Context

PyInstaller bundles the Python runtime and all `.py` files into `dist/Inno3D/_internal/`. However:
- Native DLLs (`BumpVoidSeg.dll`, OpenCV, CUDA runtimes) **cannot be bundled** by PyInstaller — they must live next to the exe.
- Recipe config files must be editable by the outsource / end-user station without recompiling.
- `app_config.ini` stores station-specific settings (theme path, port, DLL dir override).

## Decision

### Ship layout

```
dist/Inno3D/
  Inno3D.exe              ← PyInstaller entry point
  _internal/              ← Python runtime + all .py wheels (do not modify)
  VERSION.txt             ← "<version>\nBuild: <date>" (written by build_final.bat)
  app_config.ini          ← Station config (copied from project root)
  V2/                     ← Native DLLs (copied by robocopy from project V2/)
    BumpVoidSeg.dll
    BumpVoidMes.dll
    BoundaryGPU.dll
    BumpVoid_ISP_ENH.dll
    opencv_world4110.dll
    cuda*.dll / cublas*.dll  (full or slim profile)
  config/                 ← Recipe .txt files (copied from project config/)
    config_HBM_c2848_M.txt
    ...
  Inno3D_Data/            ← Created at runtime; holds inspection.db + results
  Inno3D_Logs/            ← Created at runtime; holds app + pipeline logs
```

### PACKAGE_SLIM=1 (outsource UI-only profile)

When `set PACKAGE_SLIM=1 && build_final.bat`:
- Only copies core DLLs: SEG + MES + B2B + ENH + OpenCV + CUDA-core (cublas/cudart/curand).
- **Skips**: TensorRT bulk (`nvinfer*.dll`, `nvcudnn*.dll`), onnxruntime-gpu bulk.
- Use when shipping to outsource teams who only need SEG/MES/B2B without ONNX enhancement.
- Enhancement (`BumpVoid_ISP_ENH.dll`) still copied for future use, but ONNX GPU providers will fall back to CPU.

### DLL resolution at runtime

```
Priority (inno3d.infra.paths.default_dll_dir()):
  1. User override in app_config.ini [DLL] dir=
  2. <exe_dir>/V2/           (frozen)
  3. <project_root>/V2/      (dev mode)
```

### Path contract

- **Frozen runtime:** use `app_runtime_dir()` / `default_dll_dir()` from `inno3d.infra.paths`.
- **Dev fallback:** `if not getattr(sys, 'frozen', False): use project_root() / 'V2'`
- **Hard rule:** zero absolute paths like `E:\\semiconductor\\...` in production code paths.

## Consequences

- Outsource team receives the full `dist/Inno3D/` folder zipped — no source code included.
- Station config (`app_config.ini`) can be edited without rebuilding.
- `V2/` folder is the single source for all native binaries; never split across multiple locations.
- Build must be re-run after any native DLL update to refresh `dist/Inno3D/V2/`.

---

## Alternative considered

**Bundle DLLs with PyInstaller `--add-binary`:** Ruled out — CUDA + OpenCV DLLs are too large and must be loaded via `os.add_dll_directory()` not Python import machinery.  
**Install as Windows service:** Out of scope for current product phase.
