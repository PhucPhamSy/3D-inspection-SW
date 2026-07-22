# Batch Review + Inspection Database

## What it is

After Online mode finishes a FOV, results still live under:

```text
.../Chip_x_y/FOV_Pn/Results/
```

In addition, Inno3D now **catalogues** each FOV into a local SQLite DB so you can browse history, yield, MES objects, and artifact paths without scanning the whole tree.

## Database location

| Mode | Path |
|------|------|
| Dev | `HBM_frontend_backend_v12/Inno3D_Data/inspection.db` |
| Frozen exe | `<folder containing Inno3D.exe>/Inno3D_Data/inspection.db` |

Code: `inno3d/core/inspection_db.py`

### Tables

- `wafers` — lot/wafer/recipe keys  
- `chips` — die col/row + final_bin  
- `fov_runs` — one row per Online FOV completion (KPI, paths, timings)  
- `mes_objects` — object stats for that run  
- `artifacts` — mask/csv/enhanced paths on disk  
- `timeline` — step log (enhance / seg / mes / b2b)

**Heavy files are not stored in SQLite** — only paths + metadata.

## Batch Review tab

Nav: **BATCH REVIEW** (between 3D ANALYSIS and HELP).

Layout (from mockup):

1. **Left** — lot/wafer filters, KPIs, wafer list  
2. **Center**
   - **Wafer map** — die OK/NG  
   - **FOV map** — P1–P9 on selected chip  
   - **MPR replay** — XY / YZ / XZ / MIP + bump(green)/void(red) overlay (loads `Input`/`Enhanced_Volume` + `online_combined_*` from Results)  
   - FOV run table + MES object table  
3. **Right** — FOV-point yield bars, void-ratio histogram, run timeline  

### Buttons

| Button | Action |
|--------|--------|
| Refresh | Reload DB |
| Seed demo | Insert synthetic FOV runs if DB empty |
| DB folder | Open `Inno3D_Data` |
| Open Results folder | Explorer on `results_dir` |
| Open MES CSV | Open `object_statistics.csv` if present |
| Export wafer FOV list | CSV of filtered runs |

## Online write path

`online.py` → `_record_online_run_to_db` after SEG/MES/B2B complete.

Online log line:

```text
[DB] FOV catalogued · run_id=... · OK/NG · N objects
```

## Build note

`build_final.bat` creates `dist/Inno3D/Inno3D_Data/` and copies `app_config.ini` next to the exe when present. The DB file is created on first Online complete or first Batch Review open / seed.

## Next steps (not in this drop)

- Deep-link load volume + masks into MPR automatically  
- Postgres multi-station  
- Lot PDF report  
- MES/B2B column order from `app_config.ini` (Phase 2 tables)
