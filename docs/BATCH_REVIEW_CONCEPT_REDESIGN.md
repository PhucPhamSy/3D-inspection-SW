# Batch Review — Design / Concept Redesign

**Status:** Concept review (UI/IA only — no implementation in this doc)  
**Audience:** PE, operator, product, engineering  
**Related:** [`BATCH_REVIEW.md`](BATCH_REVIEW.md), [`ANALYSIS_SOH_REDESIGN.md`](ANALYSIS_SOH_REDESIGN.md)  
**Reference data tree:** `Environment_Test/26_07_25/LOT240101_F09/...`

---

## 0. One-line product thesis

| Tab (current name) | New role (proposed) | One question |
|--------------------|---------------------|--------------|
| **Batch Review** | **Line Pulse** — mass-production status + FOV drill-down | *What is running / done / failing right now?* |
| **3D Analysis** | **Metrology Lab** — SOH / σ / root-cause cohort | *Why is yield or SOH shifting?* |

Same DB (`inspection.db`). **No overlapping job.**  
Line Pulse = **live + judgment + navigation**.  
Metrology Lab = **offline/deep stats + comparison set**.

---

## 1. Naming: drop “Batch Review”

### Why the name fails

- “Batch” sounds like offline CSV import, not wafer fab flow.  
- “Review” is vague (QA photo review? code review?).  
- Operator cannot map it to **Lot / FOUP / Wafer / Chip / FOV**.

### Name candidates

| Candidate | Feel | Pros | Cons |
|-----------|------|------|------|
| **LINE PULSE** ⭐ recommended | Live fab dashboard | Short, unique, “real-time” | Needs 1 subtitle first time |
| **WAFER PULSE** | Live per-wafer | Clear unit | Weak for multi-lot |
| **RUN MONITOR** | Online sibling | Technical | Cold, tool-like |
| **PRODUCTION MAP** | Spatial | Map-first mental model | Misses KPI/live |
| **LOT DASHBOARD** | Fab language | Familiar | Passive if multi-date |

**Recommendation**

```text
Nav label:     LINE PULSE
Page title:    LINE PULSE  ·  Lot / FOUP / Wafer status
Subtitle:      Live yield · FOV progress · open Viewer on demand
```

Keep internal module name `batch_review` for a while (rename code later). Product language changes first.

---

## 2. Hierarchy language (fix the parser, not only the UI)

### Disk / Online convention

```text
DateFolder / LotId_FOUP / WaferId / Chip_col_row / FOV_Pn / [Input.tif | Results/]
```

Example:

```text
26_07_25 / LOT240101_F09 / WAFER01 / Chip_19_11 / FOV_P6
              │        └── FOUP = 09   (_F## suffix)
              └── Lot  = LOT240101
```

**Rule:** `LOT{yymmdd?}_{F##}` or generic `{LOT}_{F##}` → always split:

| Field | Parse from `LOT240701_F01` | Display |
|-------|----------------------------|---------|
| Lot ID | `LOT240701` | `LOT240701` |
| FOUP | `F01` → `01` | `FOUP 01` |
| Raw key | full string | tooltip only |

### Production Browser — current pain

Today:

```text
Date / Lot   [ LOT240701_F01  ▼ ]     ← FOUP buried in string
Wafer        [ WAFER01        ▼ ]
```

List row (unclear):

```text
WAFER01 · 94 FOV · 9.6%
```

### Proposed browser (simple + fab-native)

```text
┌─ PRODUCTION BROWSER ─────────────────────┐
│  Date          [ 2026-07-25        ▼ ]   │  ← date folder / run day
│  Lot           [ LOT240701         ▼ ]   │
│  FOUP          [ 01                ▼ ]   │  ← first-class, not suffix
│  Wafer         [ WAFER01           ▼ ]   │
│  Judgment      [ All / OK / NG …   ▼ ]   │
│  Search        [ chip / FOV …        ]   │
│                                          │
│  ┌ FOV done  ┐ ┌ Yield    ┐             │
│  │  94 / 120 │ │  90.4%   │             │  ← progress, not cryptic
│  └───────────┘ └──────────┘             │
│  ┌ NG FOV    ┐ ┌ Bumps NG ┐             │
│  │    9      │ │   412    │             │
│  └───────────┘ └──────────┘             │
│                                          │
│  Wafer queue                              │
│  ● WAFER01   94/120 FOV   Yield 90.4%  │
│    FOUP 01 · Recipe HBM_… · live ●     │
│  ○ WAFER09   0/120 FOV    —            │
└──────────────────────────────────────────┘
```

**Date** may come from:

1. Parent folder name (`26_07_25` → display `2026-07-25` if parseable), or  
2. `MIN(started_at)` / run day from DB.

If lot+FOUP still stored as one key in DB, **split only for display + filter**; keep `lot_foup_id` as storage key until schema migration.

Breadcrumb (always full path):

```text
LINE PULSE  ›  2026-07-25  ›  LOT240701  ›  FOUP 01  ›  WAFER01  ›  Chip 19,11  ›  FOV P6
```

---

## 3. KPI wording — no more mystery strings

### 3.1 Wafer list row

| Current | Problem | Proposed |
|---------|---------|----------|
| `WAFER01 · 94 FOV · 9.6%` | 94 of what? 9.6% yield or progress? | See below |

**Proposed primary line + secondary line**

```text
WAFER01                          live ●
94 / 120 FOV done · Yield 90.4% · NG 9
FOUP 01 · Recipe config_HBM · updated 14:22
```

Rules:

- **Progress:** `done / planned` when planned known; else `done FOV` only.  
- **Yield:** always labeled `Yield` = OK FOV / finished FOV (define formula in tooltip).  
- **Never** put bare `9.6%` without the word Yield or Progress.  
- Color: yield green ≥ target, amber between, red below; progress bar optional thin under row.

Tooltip on yield:

```text
Yield (FOV) = OK FOV / finished FOV
Target: 95%  ·  NG FOV count against finished only
```

### 3.2 “Objects (MES)” — rename or kill as top KPI

**What it actually is today**

Rows in `mes_objects`: each **measured bump (or void-bearing object)** on a FOV after MES — layer, height, volume, void ratio, judgment, gaps…

Operators do **not** think “MES objects”. They think **bumps** / **sites**.

| Current label | User hears | Better label |
|---------------|------------|--------------|
| Objects (MES) | ERP? database rows? | **Bumps measured** or **Sites** |
| MES OBJECTS table | jargon | **Bump results · this FOV** |
| Open MES CSV | jargon | **Open bump stats CSV** |

**Top KPI proposal (Line Pulse)**

| Card | Meaning |
|------|---------|
| **FOV done** | Finished FOV count (and `/ planned` if known) |
| **Yield** | % OK FOV among finished |
| **NG FOV** | Count of NG FOV |
| **NG bumps** | Count of NG sites across scope (optional) |

Drop “Objects (MES)” from the 2×2 hero KPI. Put bump count in secondary strip or table column:

```text
FOV runs table columns:
Chip | FOV | Status | Bumps | NG bumps | Time | Finished
```

Table section title:

```text
BUMP RESULTS · selected FOV
(columns: Layer, Bump ID, Height, Bump vol, Void vol, Void%, Judge, Gap X/Y)
```

First-time helper (one line under table):

```text
Each row = one bump site measured on this FOV (from inspection MES step).
```

---

## 4. Maps: one canvas, drill (not two side-by-side maps)

### Current waste

```text
[ Wafer map ] [ Chip FOV P1–P9 ] [ MPR Replay … ]
     ~equal width              huge, weak zoom
```

Wafer + FOV maps fight for space; FOV map only useful **after** a die is selected.

### Concept: **Map Stage** with levels (Google-Maps mental model)

```text
Level 0 — WAFER          Level 1 — CHIP (zoom)        Level 2 — FOV focus
┌─────────────────┐      ┌─────────────────┐          Selection only:
│  die grid OK/NG │  →   │  same die full  │          highlight P-point
│  click die      │      │  + P1–P9 overlay│          open Viewer / table
└─────────────────┘      └─────────────────┘
```

**Interaction**

| Action | Result |
|--------|--------|
| Click die | Select chip; auto zoom to Level 1 (FOV overlay on that die) |
| Click empty wafer / Back | Return Level 0 |
| Click P1–P9 | Select FOV run → filter table + detail panel |
| Mouse wheel / + − | Zoom scale (wafer overview ↔ die cluster) |
| Double-click FOV | Open Full Viewer (primary deep path) |

**Visual on Level 1**

- Selected die enlarged (or full canvas = one die context).  
- P1–P9 as colored cells: OK / NG / Pending / Running.  
- Legend always visible.  
- Chip coordinate label: `Chip 19,11`.

**Space win**

- One map panel (~same width as old wafer+FOV combined).  
- Freed column → KPI strip, run table, or live ticker (not a weak mini-MPR).

```text
CENTER (after map merge)
┌──────────────────────────────────────────────┐
│  MAP STAGE  [Wafer ▸ Chip]   + −  legend     │
│  ┌────────────────────────────────────────┐  │
│  │         unified zoomable map           │  │
│  └────────────────────────────────────────┘  │
│  FOV RUNS (filtered by map selection)        │
│  BUMP RESULTS (when FOV selected)            │
└──────────────────────────────────────────────┘
```

Parity with Online CONTEXT maps stays (same bin colors / P1–P9 layout); only **layout & zoom IA** changes.

---

## 5. Kill (or demote) inline FOV Replay MPR

### Why it feels wrong

1. **Open full Viewer** already loads volume + mask + bump table + B2B + click-to-jump (Online parity).  
2. Inline MPR has **no real zoom / pan / measurement** → fails on large / anisotropic volumes.  
3. Loading 900-slice enhanced volume into a side panel is **RAM + time** for a “quick glance” that PE still verifies in Viewer.  
4. Two MPR UIs = two bugs / two themes / operator confusion.

### Design options

| Option | Idea | Verdict |
|--------|------|---------|
| **A. Remove inline MPR** ⭐ | Map + tables + **Open Viewer** only | Simplest, clearest |
| **B. Thumbnail strip** | Static mid-slice + mask preview (JPEG cache) | Nice “proof of run” without volume I/O |
| **C. Deferred mini-MPR** | Collapsed drawer “Quick slice” on demand | Only if thumbnail not enough |
| **D. Keep full MPR** | Current | Reject for Line Pulse |

**Recommended: A + B**

```text
When FOV selected:
┌─ FOV SNAPSHOT ─────────────────────────────┐
│  [XY mid-slice thumb]  [mask overlay thumb]│  ← pre-rendered or lazy 1-slice
│  Status OK · 12 layers · 0.8s enhance …    │
│  [ Open full Viewer ]  [ Results folder ]  │
└────────────────────────────────────────────┘
```

**Open full Viewer** becomes the **only** 3D path:

- Full zoom / window-level / crosshair / MES↔MPR link.  
- Shape extremes handled once, in Viewer (already designed for that).

If snapshot missing (no Results yet):

```text
No snapshot · FOV pending or Results incomplete
```

Do **not** pretend to stream full volume in the dashboard tab.

---

## 6. Overlap with 3D Analysis — hard split

**3D Analysis deep method (cross chip/wafer/FOUP/lot at same FOV+bump index):**  
see [`ANALYSIS_CROSS_HOMOLOGY_PATENT.md`](ANALYSIS_CROSS_HOMOLOGY_PATENT.md) — Line Pulse never does SHK join or nested variance.

### What overlaps today (honest)

| Capability | Batch Review now | 3D Analysis now | Keep where? |
|------------|------------------|-----------------|-------------|
| Lot / wafer filter | ✓ | ✓ (scope) | Both, different depth |
| Yield / NG KPI | ✓ FOV yield | ✓ SOH yield-ish | **Line Pulse** owns production yield |
| Wafer / FOV maps | ✓ bin colors | ✓ metrology colors (plan) | **Both**, different metric |
| MES / bump table | ✓ per FOV | ✓ aggregate table | Pulse = 1 FOV; Analysis = set |
| Hist void ratio | ✓ | ✓ | **Analysis** owns distributions |
| Yield by P1–P9 bar | ✓ | FOV heatmap | **Line Pulse** simple bar OK; Analysis deeper |
| Cross-σ / LOO / Moran | — | ✓ | **Analysis only** |
| Live / refresh while Online | weak | no | **Line Pulse only** |
| MPR replay | ✓ | — | **Remove** → Viewer |
| Multi-FOV analysis set | — | ✓ | **Analysis only** |

### Golden rule

> **If the number updates while the tool is inspecting, it belongs in Line Pulse.**  
> **If the number needs a multi-FOV cohort or σ / root-cause, it belongs in 3D Analysis.**

### “Who answers what”

```text
Operator on line
  “Wafer 01 stuck? which FOV NG?”     → LINE PULSE
  “Open that NG FOV and look”         → LINE PULSE → Viewer

Process engineer next day
  “Is SOH dropping on Layer 8?”       → 3D ANALYSIS
  “P6 systematically higher void%?”   → 3D ANALYSIS
  “Which bump sites drive σ?”         → 3D ANALYSIS
```

### Right panel of Line Pulse (after cleanup)

**Keep production-native only:**

- Progress (FOV done / planned)  
- Yield / NG FOV  
- Yield by FOV point P1–P9 (simple bars)  
- Run timeline for selected FOV (enhance → seg → mes → b2b)  
- Live event ticker (optional)

**Move or drop:**

| Widget | Action |
|--------|--------|
| Void ratio distribution histogram | Move to **3D Analysis** (or keep 1-line summary: mean void% only) |
| Deep SOH stats | Analysis |
| Multi-sample compare | Analysis |

---

## 7. Real-time mass production — Line Pulse is the live tab

### Goal

One screen for **shift lead / operator** during Online:

- What lot/FOUP/wafer is active  
- How far along (FOV progress)  
- Instant yield / NG rate  
- Where NG clusters on the map  
- Jump to Viewer without leaving the mental “line” context  

**3D Analysis stays post-process / deep dive** (can refresh on demand; not the live wall).

### Live model (simple)

```text
Online pipeline finishes FOV
        │
        ▼
  write inspection.db  (fov_runs + mes_objects + artifacts)
        │
        ├──► LINE PULSE  auto-refresh (2–5 s poll or signal)
        └──► 3D ANALYSIS  no auto storm; manual “Load scope”
```

Prefer **Qt signal** from Online controller → Line Pulse (`run_catalogued`) over blind polling; poll as fallback when tab visible.

### Live UI elements (unique, still simple)

```text
┌─ LIVE STRIP (always on when Online active) ─────────────┐
│  ● LIVE  LOT240701 · FOUP 01 · WAFER01                  │
│  FOV 94/120  ████████░░  78%                            │
│  Yield 90.4%   NG FOV 9   Last: Chip 19,11 P6 OK  0.2s │
└─────────────────────────────────────────────────────────┘
```

Map cells update as runs complete (Pending → OK/NG).  
Optional **sound / flash** only on NG FOV (configurable).

### Mass-production KPI definitions (lock these)

| KPI | Formula | Notes |
|-----|---------|-------|
| **FOV progress** | finished / planned | Planned from recipe × chips × FOV points if known |
| **FOV yield** | OK FOV / finished FOV | Primary line yield |
| **NG rate** | NG FOV / finished FOV | = 100% − yield (if only OK/NG) |
| **Bump NG rate** | NG bumps / bumps measured | Secondary; needs MES |
| **Tact** | median total_s of recent FOVs | Optional ops metric |

Show **units and formula in tooltip** always — no bare percentages.

### What Line Pulse is **not**

- Not a second Viewer  
- Not SOH lab  
- Not lot genealogy ERP  
- Not multi-day statistical process control (that can be Analysis / export)

---

## 8. Target information architecture (Line Pulse)

```text
┌──────────────┬────────────────────────────┬────────────────┐
│ LEFT ~260px  │ CENTER flex                │ RIGHT ~280px   │
│ Browser      │ Map Stage (zoom Wafer↔Chip)│ Live + summary │
│ Lot·FOUP·Wfr │ FOV run table              │ P1–P9 yield    │
│ KPIs clear   │ Bump results (1 FOV)       │ Timeline       │
│ Wafer queue   │ Snapshot + Open Viewer     │ Event ticker   │
└──────────────┴────────────────────────────┴────────────────┘
```

**Primary actions**

| Button | Role |
|--------|------|
| Refresh / Live toggle | Manual or auto |
| Open full Viewer | Only 3D path |
| Open Results folder | Disk |
| Export FOV list | Shift report CSV |
| (Hide) Seed demo | Dev only / advanced |

---

## 9. Answers mapped to your 8 points

| # | Issue | Decision |
|---|--------|----------|
| **1** | Date/Lot missing FOUP | Split **Lot** + **FOUP** filters; parse `_F##`; breadcrumb shows all |
| **2** | `WAFER · 94 FOV · 9.6%` unclear | Two-line row: `done/planned`, labeled **Yield**, recipe, live state |
| **3** | Objects (MES) unclear | Rename to **Bumps / Bump results**; remove as hero KPI; short helper text |
| **4** | Two maps waste space | **One Map Stage** + zoom Wafer → Chip FOV overlay |
| **5** | FOV Replay vs Viewer | **Remove full MPR**; optional 1-slice snapshot; Viewer = deep 3D |
| **6** | Name Batch Review | Product rename → **LINE PULSE** (or WAFER PULSE) |
| **7** | Overlap with Analysis | Pulse = live judgment + map + 1-FOV table; Analysis = cohort SOH/σ |
| **8** | Real-time mass production | **Line Pulse** owns live yield/NG/progress; Analysis stays deep dive |

---

## 10. Unique concept hook (memorable)

**“Pulse, then Probe.”**

1. **Pulse (this tab)** — feel the line: progress, yield, where it hurts on the wafer.  
2. **Probe (Viewer)** — open one FOV and see volume truth.  
3. **Prove (3D Analysis)** — multi-FOV numbers that explain process drift.

Three verbs, three surfaces. No third MPR in the middle.

---

## 11. Phased delivery (concept only)

| Phase | Scope | Value |
|-------|--------|-------|
| **P0** | Rename labels (Lot/FOUP, KPI, Bumps); breadcrumb; clarify wafer list strings | Clarity without layout rewrite |
| **P1** | Remove inline MPR; snapshot + Open Viewer; free layout | Less confusion, better performance |
| **P2** | Unified Map Stage zoom | Space + better drill |
| **P3** | Live strip + Online signal refresh | True mass-production monitor |
| **P4** | Nav rename LINE PULSE; move void hist ownership; docs/smoke | Product identity |

---

## 12. Open questions (for PE / fab)

1. Is planned FOV count always `chips_in_recipe × 9`, or variable P-points?  
2. Yield primary unit: **FOV** or **chip final bin**? (Recommend FOV for Online line; chip bin secondary.)  
3. FOUP always `_F##` on lot string, or separate folder level later?  
4. Should Line Pulse support **multi-tool** later (station id), or single box forever?  
5. Snapshot generation: on Online complete (write PNG) vs on-demand first open?

---

## 13. Non-goals (this redesign)

- Storing volumes in SQLite  
- Replacing 3D Viewer  
- Full SPC charts / Cpk in Line Pulse  
- Re-implementing SOH Cross-σ / LOO here  

---

*End of concept. Next step when approved: P0 label/copy PR only, then P1 MPR removal + snapshot sketch.*
