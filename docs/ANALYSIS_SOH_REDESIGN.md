# 3D Analysis — SOH Analytics Workbench (DB-first redesign)

**Status:** P0 implemented (DB-first Overview) · concept + mockup retained  
**Mockup:** [`docs/mockups/analysis_soh_tab.html`](mockups/analysis_soh_tab.html)  
**Code:** `inno3d/tabs/analysis.py`, `inno3d/core/soh_data.py`, `inno3d/core/inspection_db.py` (SOH APIs)  
**Related:** [`BATCH_REVIEW.md`](BATCH_REVIEW.md)

### Implemented (P0)

- Scope browser: Lot / Wafer filters + Chip→FOV tree with checkboxes  
- Load wafer / load checked FOVs from `inspection.db`  
- Each FOV run → `SampleData` (advanced modes reuse analysis set)  
- Overview: KPI cards, SOH by Layer bar, SOH by FOV P1–P9 heatmap, layer summary table  
- Sidebar: Refresh DB · Load wafer · Load checked · CSV lab fallback  
- SQL helpers: `list_mes_joined`, `aggregate_soh_by_*`, `soh_kpis`, `list_chips_for_wafer`

---

## 1. Problem statement

### Today (`analysis.py`)

| Aspect | Behavior |
|--------|----------|
| Data source | Folder CSV (`object_statistics.csv` / per-layer measurement CSVs) |
| Unit of work | Ambiguous “Sample” = folder name |
| Hierarchy | None (no Lot / Wafer / Chip / FOV) |
| Views | SOH Summary, Cross-σ, Per-Bump σ, 3D-SIM, LOO-VD |
| Sidebar | Import Sample / Batch / Export / Recalculate |

The statistical engines (σ, %Tol, Moran/SII, LOO) are valuable. The **binding to production data is wrong**: Online already catalogues every FOV into SQLite (`Inno3D_Data/inspection.db`), and Batch Review already browses that DB. Analysis still asks the operator to re-import CSVs.

### Target

> **Batch Review** answers: *did this FOV / chip pass?*  
> **3D Analysis** answers: *how does SOH / void / variance change by FOV, chip, layer, bump grid — and what drives σ?*

Same database. Complementary jobs.

---

## 2. Goals & non-goals

### Goals

1. **DB-first** load of MES objects + FOV run metadata (default path).
2. Explicit **analysis scope**: Lot → Wafer → Chip → FOV → Layer → Bump grid.
3. First-class cuts: **SOH by FOV**, **by Chip**, **by Layer**, **by Bump**.
4. Preserve advanced modes (Cross-σ, Per-Bump σ, 3D-SIM, LOO-VD) under clearer labels and a defined cohort (“analysis set”).
5. Visual parity with Batch Review (theme, wafer/FOV maps) but **metrology-colored** maps (SOH / NG%), not only pass/fail bins.
6. CSV import remains as **advanced / lab** fallback, mapped into the same in-memory model.

### Non-goals (this redesign)

- Storing heavy volumes in SQLite (paths only — unchanged).
- Replacing Batch Review MPR replay.
- Multi-station Postgres (future; same query API shape).

---

## 3. Information architecture

### 3.1 Hierarchy

```text
Date / Lot-FOUP / Recipe
  └─ Wafer
       └─ Chip (col, row)
            └─ FOV run (P1–P9)          ← fov_runs
                 └─ mes_objects rows    ← layer, grid, soh, ratio, volumes
```

### 3.2 Compare units (replace “sample folder”)

| Unit | Question | Typical cohort |
|------|----------|----------------|
| FOV index | Position effect on die? | Same chip, P1…P9 |
| Chip | Die-to-die SOH? | Same wafer |
| Layer | Stack trend? | Scope set, Layer 1…N |
| Bump grid (R,C) | Site-specific σ? | Multi-run, fixed layer |
| FOV re-run / multi-wafer | Gage / process? | Explicit multi-select set |

### 3.3 Analysis set

Operator multi-selects FOV runs (or chips/wafers) into an **analysis set**. All Compare / Bump-σ / LOO use that set. Overview defaults to current selection or whole set.

---

## 4. UI layout

Three columns (mirrors Batch Review width language):

```text
┌────────────────┬──────────────────────────────┬────────────────────┐
│ LEFT ~280px    │ CENTER flex                  │ RIGHT ~360px       │
│ Scope browser  │ Metric bar + studio tabs     │ Selection insight  │
│ Analysis set   │ Maps / charts / tables       │ Hist + top NG      │
└────────────────┴──────────────────────────────┴────────────────────┘
```

### Left — Scope browser

- Source toggle: **Database** (default) | CSV…
- Filters: Date/Lot, Wafer, Recipe, Judgment
- KPI strip: #FOV, #Objects, Yield, Mean SOH
- Lazy tree: Lot → Wafer → Chip → FOV (checkbox for set)
- Actions: Refresh DB, Clear set, Export…

### Center — Analysis studio

**Sticky top bar**

- Breadcrumb scope
- Metric: SOH | Void% | NG rate | Bump vol  
- Aggregate: Mean / Median / P95  
- NG threshold + %Tol Excellent / Acceptable  

**Segmented tabs** (not five equal left-panel mode buttons)

| Tab | Role |
|-----|------|
| **Overview** | KPI cards, SOH by Layer, FOV 3×3 heatmap, summary table |
| **Maps** | Wafer SOH map + chip FOV map + optional bump grid |
| **Compare** | Cross-sample / cross-run σ (%Tol) |
| **Bump** | Per-bump σ across set |
| **Spatial** | 3D-SIM (SII, Moran, propagation) |
| **Root cause** | LOO-VD impact ranking |
| **Table** | Flat MES query (power user) |

### Right — Insight panel

- Focus card (wafer / chip / FOV / layer)
- SOH + void histograms for current filter
- Top NG bumps table
- Deep links: Batch Review, Results folder, Viewer

### App sidebar (main window)

Replace primary Import Sample/Batch with:

- Refresh DB  
- Add selection to set / Clear set  
- Export report  
- Recalculate thresholds  
- Advanced: CSV import  

---

## 5. Data model (in-app)

```python
@dataclass
class AnalysisScope:
    date_folder: str = ""
    lot_foup_id: str = ""

    wafer_keys: list[str] = field(default_factory=list)
    chip_keys: list[str] = field(default_factory=list)
    fov_indices: list[int] = field(default_factory=list)  # 1..9
    layers: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)      # resolved

@dataclass
class MesRecord:
    run_id: str
    wafer_key: str
    chip_col: int
    chip_row: int
    fov_index: int
    layer_name: str
    grid_row: int
    grid_col: int
    soh: float
    ratio: float
    c1_volume: float
    c2_volume: float
    judgment: str
    # optional centroids / bbox for spatial

@dataclass
class AggregateStats:
    key: str          # "Layer 4" | "FOV_P1" | "Chip_10_12"
    n: int
    n_ng: int
    ng_rate: float
    mean_soh: float
    std_soh: float
    min_soh: float
    max_soh: float
    mean_ratio: float
    max_ratio: float
```

CSV path: `parse_measurement_csv` → list of `MesRecord` (synthetic `run_id` / chip tags).

Legacy `SampleData` / `LayerStats` become adapters or are retired after migration.

---

## 6. Database API additions

File: `inno3d/core/inspection_db.py`

| Method | Purpose |
|--------|---------|
| `query_mes(wafer_key, chip_col, chip_row, fov_index, layer, judgment, limit)` | Filtered MES rows + join run meta |
| `aggregate_soh_by_layer(scope)` | GROUP BY layer_name |
| `aggregate_soh_by_fov(scope)` | GROUP BY fov_index |
| `aggregate_soh_by_chip(scope)` | GROUP BY chip_col, chip_row |
| `chip_soh_summary(wafer_key)` | Mean SOH / NG% per die for map |
| `bump_metric_series(layer, grid_r, grid_c, run_ids)` | Per-bump cross-run values |

Prefer SQL aggregation for Overview/Maps; fetch raw rows only for Spatial / LOO / Bump detail.

Existing: `list_lots`, `list_wafers`, `list_fov_runs`, `get_run`, `wafer_chip_bins`, `ratio_histogram`, `global_kpis`.

---

## 7. View mapping (old → new)

| Current | New home | Default scope |
|---------|----------|---------------|
| Import sample/batch | Left source + sidebar advanced | DB |
| Sample tree | Scope tree + analysis set | — |
| SOH Summary | **Overview** | Set / selection |
| Intra-sample layer chart | Overview · SOH by Layer | Focus chip/FOV |
| Inter-sample layer chart | Overview / Compare multi-wafer | Set |
| Cross-σ | **Compare** | Multi run_id, same layers |
| Per-Bump σ | **Bump** | Multi run, fixed layer |
| 3D-SIM | **Spatial** | Single FOV run preferred |
| LOO-VD | **Root cause** | One layer + cohort |
| Export CSV | Multi-section export by scope | — |

---

## 8. Color & metric conventions

| Signal | Encoding |
|--------|----------|
| SOH continuous | Cyan → amber → red scale (low→high deviation from set mean, or absolute) |
| NG rate | Green / amber / red vs NG threshold |
| %Tol | Excellent / Acceptable / Poor (existing spin boxes) |
| Pass/fail bin | Batch Review only (Analysis maps use metrology scales) |
| FOV 3×3 | Same P1–P9 layout as Online / Batch Review |

---

## 9. Implementation phases

| Phase | Deliverable | Exit criteria |
|-------|-------------|-----------------|
| **P0** | DB scope tree, load MES, Overview (by Layer + by FOV) + summary table | Open tab, select WAFER05, see real SOH bars without CSV |
| **P1** | Maps (wafer SOH, FOV 3×3), KPI cards, export by scope | Click die → FOV metrics update |
| **P2** | Compare + Bump on `run_ids` | Cross-σ uses analysis set |
| **P3** | Spatial + LOO adapters on `MesRecord` | 3D-SIM/LOO work without CSV folder |
| **P4** | CSV fallback, deep-link Batch Review/Viewer, polish | Feature parity + docs |

---

## 10. Open questions

1. **Spatial on multi-FOV merge?** Default single FOV; optional “stitch” later.  
2. **Chip SOH color scale:** absolute µm vs z-score within wafer?  
3. **Re-inspect same FOV:** treat as multiple runs in set for gage σ?  
4. **Report format:** CSV only vs multi-sheet Excel / PDF.

---

## 11. Mockup

Open in browser:

```text
docs/mockups/analysis_soh_tab.html
```

Interactive (lightweight):

- Left tree expand / analysis-set checkboxes  
- Center tab switcher (Overview · Maps · Compare · Bump · Spatial · Root cause · Table)  
- Metric / FOV heatmap tooltips (static demo numbers aligned with demo DB shape)

Demo numbers approximate `inspection.db` seed (LOT240701_F01, WAFER05–07, 12 layers, P1–P9 FOVs).
